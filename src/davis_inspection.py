"""Load and inspect DAVIS data without aligning or copying its streams."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

TIMESTAMP_UNIT = "microseconds"
EVENT_FIELDS = ("t", "x", "y", "p")
VALID_POLARITIES = (0, 1)
DEFAULT_RECORDING = "shapes_6dof"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "DAVIS"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "dataset_inspection"
VALIDATION_CHUNK_SIZE = 1_000_000


@dataclass(frozen=True)
class DavisConfig:
    """Input and output paths for the DAVIS dataset-inspection stage."""

    data_dir: Path = DEFAULT_DATA_DIR
    recording: str = DEFAULT_RECORDING
    output_dir: Path = DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class DavisRecording:
    """References to Tonic-loaded arrays; no stream data is copied here."""

    events: np.ndarray
    imu: Mapping[str, Any] | None
    aps: Mapping[str, Any]
    target: Mapping[str, Any] | None


def load_davis(config: DavisConfig = DavisConfig()) -> DavisRecording:
    """Load one DAVIS recording with the repository's existing Tonic loader."""
    try:
        from tonic.datasets import DAVISDATA
    except ImportError as exc:
        raise RuntimeError(
            "Loading DAVIS requires Tonic. Use the repository's configured "
            "Python environment and install requirements.txt if needed."
        ) from exc

    dataset = DAVISDATA(
        save_to=str(config.data_dir),
        recording=config.recording,
    )
    (events, imu, aps), target = dataset[0]
    if events is None or aps is None:
        raise ValueError("DAVIS recording did not provide events and APS images")
    return DavisRecording(events=events, imu=imu, aps=aps, target=target)


def _check_timestamp_array(
    timestamps: Any, name: str, chunk_size: int = VALIDATION_CHUNK_SIZE
) -> tuple[int, int, int]:
    """Validate Tonic's integer-microsecond timestamp arrays in bounded chunks."""
    values = np.asarray(timestamps)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} timestamps must be a non-empty 1-D array")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} timestamps must use integer {TIMESTAMP_UNIT}")

    for start in range(0, values.size, chunk_size):
        stop = min(start + chunk_size, values.size)
        block = values[start:stop]
        if np.any(block < 0):
            raise ValueError(f"{name} timestamps must be non-negative")
        if np.any(block[1:] < block[:-1]):
            raise ValueError(f"{name} timestamps are not monotonically ordered")
        if start and values[start] < values[start - 1]:
            raise ValueError(f"{name} timestamps are not monotonically ordered")

    return int(values.size), int(values[0]), int(values[-1])


def _stream_metadata(
    name: str, timestamps: Any, sample_count: int | None = None
) -> dict[str, Any]:
    count, start, end = _check_timestamp_array(timestamps, name)
    if sample_count is not None and sample_count != count:
        raise ValueError(
            f"{name} sample count ({sample_count}) does not match "
            f"timestamp count ({count})"
        )
    return {
        "available": True,
        "sample_count": count,
        "timestamp_unit": TIMESTAMP_UNIT,
        "timestamp_origin": "Tonic-zeroed independently for this stream",
        "start": start,
        "end": end,
        "duration": end - start,
        "monotonic": True,
    }


def inspect_davis(
    recording: DavisRecording, config: DavisConfig = DavisConfig()
) -> dict[str, Any]:
    """Validate loaded DAVIS streams and return small, JSON-serializable metadata."""
    events = recording.events
    if not isinstance(events, np.ndarray) or events.dtype.names is None:
        raise ValueError("Events must be a structured NumPy array")
    missing_fields = [field for field in EVENT_FIELDS if field not in events.dtype.names]
    if missing_fields:
        raise ValueError(f"Event array is missing fields: {', '.join(missing_fields)}")
    if events.size == 0:
        raise ValueError("Event array is empty")

    aps = recording.aps
    if not isinstance(aps, Mapping):
        raise ValueError("APS data must be a mapping containing frames and timestamps")
    frames = np.asarray(aps.get("frames"))
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError("APS frames must have shape (frame_count, height, width)")
    frame_count, height, width = (int(value) for value in frames.shape)
    if height <= 0 or width <= 0:
        raise ValueError("APS image resolution must be positive")

    event_stream = _stream_metadata("events", events["t"], sample_count=int(events.size))
    frame_stream = _stream_metadata(
        "APS frames", aps.get("ts"), sample_count=frame_count
    )

    observed_polarities: set[int] = set()
    for start in range(0, events.size, VALIDATION_CHUNK_SIZE):
        stop = min(start + VALIDATION_CHUNK_SIZE, events.size)
        block = events[start:stop]
        if np.any(block["x"] >= width) or np.any(block["y"] >= height):
            raise ValueError("Event coordinates exceed the APS sensor dimensions")
        if np.any(block["x"] < 0) or np.any(block["y"] < 0):
            raise ValueError("Event coordinates must be non-negative")
        polarities = np.unique(block["p"])
        if not np.isin(polarities, VALID_POLARITIES).all():
            raise ValueError("Event polarity values must be 0 or 1")
        observed_polarities.update(int(polarity) for polarity in polarities)

    streams: dict[str, Any] = {
        "events": event_stream,
        "aps_frames": {
            **frame_stream,
            "frame_count": frame_count,
            "resolution_px": {"width": width, "height": height},
        },
    }

    if recording.imu is None:
        streams["imu"] = {"available": False}
    else:
        imu = recording.imu
        if not isinstance(imu, Mapping) or "ts" not in imu:
            raise ValueError("IMU data must contain a timestamp array named 'ts'")
        imu_metadata = _stream_metadata("IMU", imu["ts"])
        for field in ("acc", "angV"):
            if field in imu and len(imu[field]) != imu_metadata["sample_count"]:
                raise ValueError(f"IMU {field} count does not match its timestamps")
        streams["imu"] = imu_metadata

    if recording.target is None:
        streams["ground_truth"] = {"available": False}
    else:
        target = recording.target
        if not isinstance(target, Mapping) or "ts" not in target:
            raise ValueError(
                "Ground-truth data must contain a timestamp array named 'ts'"
            )
        target_metadata = _stream_metadata("ground truth", target["ts"])
        if "point" in target and len(target["point"]) != target_metadata["sample_count"]:
            raise ValueError("Ground-truth point count does not match its timestamps")
        streams["ground_truth"] = target_metadata

    return {
        "stage": "1A",
        "recording": config.recording,
        "dataset_loader": "tonic.datasets.DAVISDATA",
        "dataset_directory": str(config.data_dir),
        "event_fields": list(EVENT_FIELDS),
        "event_polarities_observed": sorted(observed_polarities),
        "timestamp_unit": TIMESTAMP_UNIT,
        "timestamp_interpretation": (
            "Tonic converts source seconds to integer microseconds and resets "
            "each stream's first timestamp independently to zero. Reported "
            "ranges are stream-relative and are not synchronized."
        ),
        "synchronization_performed": False,
        "streams": streams,
    }


def plot_timestamp_ranges(metadata: Mapping[str, Any], output_path: Path) -> Path:
    """Save stream-relative ranges on one numeric axis, without aligning streams."""
    import matplotlib.pyplot as plt

    streams = metadata["streams"]
    labels = {
        "events": "Events",
        "aps_frames": "APS frames",
        "imu": "IMU",
        "ground_truth": "Ground truth",
    }
    plotted = [
        (labels[name], stream)
        for name, stream in streams.items()
        if stream.get("available")
    ]
    if not plotted:
        raise ValueError("No timestamped streams are available to plot")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(3.2, 0.75 * len(plotted))))
    for row, (label, stream) in enumerate(plotted):
        ax.hlines(
            y=row,
            xmin=stream["start"],
            xmax=stream["end"],
            linewidth=8,
            color="tab:blue",
        )
        ax.plot(stream["start"], row, "|", color="tab:green", markersize=12)
        ax.plot(stream["end"], row, "|", color="tab:red", markersize=12)
    ax.set_yticks(range(len(plotted)), [label for label, _ in plotted])
    ax.set_xlabel(f"Tonic stream-relative timestamp ({TIMESTAMP_UNIT})")
    ax.set_title("DAVIS timestamp ranges (stream origins are independent)")
    ax.grid(axis="x", alpha=0.3)
    ax.text(
        0.5,
        -0.24,
        "Each stream is zero-based independently by Tonic; this plot does not "
        "show cross-stream synchronization.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_metadata(metadata: Mapping[str, Any], output_path: Path) -> Path:
    """Persist validated dataset metadata without copying the underlying data."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "data_dir": metadata["dataset_directory"],
            "recording": metadata["recording"],
            "timestamp_unit": TIMESTAMP_UNIT,
            "validation_chunk_size": VALIDATION_CHUNK_SIZE,
        },
        **metadata,
    }
    output_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return output_path


def print_report(metadata: Mapping[str, Any]) -> None:
    streams = metadata["streams"]
    frames = streams["aps_frames"]
    print(f"DAVIS recording: {metadata['recording']}")
    print(f"Events: {streams['events']['sample_count']:,}")
    print(
        "Image resolution: "
        f"{frames['resolution_px']['width']} x {frames['resolution_px']['height']} px"
    )
    print(f"APS frames: {frames['frame_count']:,}")
    for name in ("events", "aps_frames", "imu", "ground_truth"):
        stream = streams[name]
        if stream["available"]:
            print(
                f"{name}: [{stream['start']:,}, {stream['end']:,}] "
                f"{stream['timestamp_unit']}"
            )
        else:
            print(f"{name}: unavailable")
    print(f"Timestamp interpretation: {metadata['timestamp_interpretation']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load and inspect a DAVIS recording without synchronizing streams."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Tonic DAVIS dataset directory.",
    )
    parser.add_argument(
        "--recording",
        default=DEFAULT_RECORDING,
        help="DAVISDATA recording name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the metadata JSON and timestamp-range plot.",
    )
    args = parser.parse_args()
    config = DavisConfig(
        data_dir=args.data_dir,
        recording=args.recording,
        output_dir=args.output_dir,
    )

    recording = load_davis(config)
    metadata = inspect_davis(recording, config)
    print_report(metadata)

    metadata_path = save_metadata(
        metadata, config.output_dir / f"{config.recording}_metadata.json"
    )
    plot_path = plot_timestamp_ranges(
        metadata, config.output_dir / f"{config.recording}_timestamp_ranges.png"
    )
    print(f"Metadata: {metadata_path}")
    print(f"Timestamp plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
