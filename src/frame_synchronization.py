"""Index DAVIS sensor streams into APS-to-APS half-open time windows."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .davis_inspection import (
    DEFAULT_RECORDING,
    DavisConfig,
    DavisRecording,
    _check_timestamp_array,
    load_davis,
)

TIMESTAMP_UNIT = "microseconds"


@dataclass(frozen=True)
class IndexedSamples:
    """A lazily sliced view into a loaded timestamped stream."""

    source: Mapping[str, Any]
    start_index: int
    end_index: int
    timestamp_offset_us: int

    @property
    def timestamps(self) -> np.ndarray:
        """Return original Tonic stream-local timestamps without modifying them."""
        return self.source["ts"][self.start_index : self.end_index]

    @property
    def common_timestamps_us(self) -> np.ndarray:
        """Return timestamps mapped onto the common microsecond timeline."""
        return self.timestamps + self.timestamp_offset_us

    def __getitem__(self, field: str) -> Any:
        value = self.source[field]
        if field == "ts":
            return value[self.start_index : self.end_index]
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return value[self.start_index : self.end_index]
        return value

    def __len__(self) -> int:
        return self.end_index - self.start_index


@dataclass(frozen=True)
class FrameWindow:
    """One consecutive APS-frame interval and its lazy sensor-stream slices."""

    index: int
    frame_k: np.ndarray
    frame_k_plus_1: np.ndarray
    frame_k_timestamp_us: int
    frame_k_plus_1_timestamp_us: int
    frame_k_common_time_us: int
    frame_k_plus_1_common_time_us: int
    event_source: np.ndarray
    event_start_index: int
    event_end_index: int
    event_timestamp_offset_us: int
    imu_k: IndexedSamples | None
    groundtruth_k: IndexedSamples | None

    @property
    def events_k(self) -> np.ndarray:
        """View the original structured event array for [frame k, frame k+1)."""
        return self.event_source[self.event_start_index : self.event_end_index]

    @property
    def event_index_range(self) -> tuple[int, int]:
        return self.event_start_index, self.event_end_index

    @property
    def interval_us(self) -> tuple[int, int]:
        return self.frame_k_common_time_us, self.frame_k_plus_1_common_time_us


@dataclass(frozen=True)
class FrameWindows(Sequence[FrameWindow]):
    """Lazy consecutive APS intervals backed by the original loaded arrays."""

    recording: DavisRecording
    aps_common_timestamps_us: np.ndarray
    timestamp_offsets_us: Mapping[str, int]
    common_origin_s: float

    def __len__(self) -> int:
        return max(0, len(self.aps_common_timestamps_us) - 1)

    def __iter__(self) -> Iterator[FrameWindow]:
        for index in range(len(self)):
            yield self.get_window(index)

    def __getitem__(self, index: int | slice) -> FrameWindow | list[FrameWindow]:
        if isinstance(index, slice):
            return [self.get_window(i) for i in range(*index.indices(len(self)))]
        return self.get_window(index)

    def get_window(self, index: int) -> FrameWindow:
        """Retrieve one window by index without copying event or IMU samples."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Frame window index {index} is out of range")

        aps = self.recording.aps
        t0 = int(self.aps_common_timestamps_us[index])
        t1 = int(self.aps_common_timestamps_us[index + 1])
        event_offset = self.timestamp_offsets_us["events"]
        event_times = self.recording.events["t"]
        event_start = int(np.searchsorted(event_times, t0 - event_offset, side="left"))
        event_end = int(np.searchsorted(event_times, t1 - event_offset, side="left"))

        return FrameWindow(
            index=index,
            frame_k=aps["frames"][index],
            frame_k_plus_1=aps["frames"][index + 1],
            frame_k_timestamp_us=int(aps["ts"][index]),
            frame_k_plus_1_timestamp_us=int(aps["ts"][index + 1]),
            frame_k_common_time_us=t0,
            frame_k_plus_1_common_time_us=t1,
            event_source=self.recording.events,
            event_start_index=event_start,
            event_end_index=event_end,
            event_timestamp_offset_us=event_offset,
            imu_k=_make_stream_slice(
                self.recording.imu,
                self.timestamp_offsets_us.get("imu"),
                t0,
                t1,
            ),
            groundtruth_k=_make_stream_slice(
                self.recording.target,
                self.timestamp_offsets_us.get("ground_truth"),
                t0,
                t1,
            ),
        )


def load_timestamp_origins(
    recording: DavisRecording,
    config: DavisConfig = DavisConfig(),
    manifest_path: Path | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Load raw ROS-bag stream origins from the dataset-preparation manifest.

    The manifest's source times are seconds. Tonic's arrays are integer
    microseconds relative to each stream's first sample. Counts and durations
    are checked before the origins are used to put those arrays on one timeline.
    """
    bag_path = config.data_dir / "DAVISDATA" / f"{config.recording}.bag"
    if manifest_path is None:
        manifest_path = config.data_dir.parent / "metadata" / "dataset_manifest.json"
    manifest_path = Path(manifest_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    davis = document.get("davis")
    if not isinstance(davis, Mapping):
        raise ValueError(f"Manifest has no DAVIS metadata: {manifest_path}")

    project_root = config.data_dir.parent.parent
    manifest_bag = davis.get("bag")
    if not manifest_bag or (project_root / manifest_bag).resolve() != bag_path.resolve():
        raise ValueError("Timestamp manifest refers to a different DAVIS bag")
    if not bag_path.is_file():
        raise FileNotFoundError(f"DAVIS bag not found: {bag_path}")
    if davis.get("size_bytes") != bag_path.stat().st_size:
        raise ValueError(
            "DAVIS bag size differs from the timestamp manifest; refresh it with "
            "`python -m scripts.prepare_datasets` before synchronizing."
        )

    raw_streams = davis.get("streams")
    if not isinstance(raw_streams, Mapping):
        raise ValueError("Timestamp manifest has no DAVIS stream metadata")
    tonic_streams: dict[str, tuple[Any, int]] = {
        "events": (recording.events["t"], len(recording.events)),
        "aps_frames": (recording.aps["ts"], len(recording.aps["frames"])),
    }
    if recording.imu is not None:
        tonic_streams["imu"] = (recording.imu["ts"], len(recording.imu["ts"]))
    if recording.target is not None:
        tonic_streams["ground_truth"] = (
            recording.target["ts"],
            len(recording.target["ts"]),
        )

    origins_s: dict[str, float] = {}
    for name, (timestamps, sample_count) in tonic_streams.items():
        raw = raw_streams.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Timestamp manifest is missing the {name} stream")
        if raw.get("samples") != sample_count:
            raise ValueError(
                f"{name} sample count differs from the timestamp manifest"
            )
        local_count, local_start, local_end = _check_timestamp_array(
            timestamps, name
        )
        if local_count != sample_count or local_start != 0:
            raise ValueError(
                f"{name} Tonic timestamps do not match the expected zero-based stream"
            )

        raw_start_s = raw.get("timestamp_start_s")
        raw_end_s = raw.get("timestamp_end_s")
        if not isinstance(raw_start_s, (int, float)) or not isinstance(
            raw_end_s, (int, float)
        ):
            raise ValueError(f"{name} manifest timestamps must be in source seconds")
        if raw_end_s < raw_start_s:
            raise ValueError(f"{name} manifest time range is invalid")
        manifest_duration_us = (raw_end_s - raw_start_s) * 1_000_000
        if abs(manifest_duration_us - local_end) > 2:
            raise ValueError(
                f"{name} duration differs from the timestamp manifest by more "
                "than 2 microseconds"
            )
        origins_s[name] = float(raw_start_s)

    return origins_s, {
        "manifest_path": str(manifest_path),
        "manifest_generated_at_utc": document.get("generated_at_utc"),
        "manifest_timestamp_origin_s": davis.get("timestamp_origin_s"),
        "bag_path": str(bag_path),
        "bag_size_bytes": bag_path.stat().st_size,
    }


def _origin_offsets_us(
    stream_start_timestamps_s: Mapping[str, float],
) -> tuple[dict[str, int], float]:
    if "events" not in stream_start_timestamps_s or "aps_frames" not in stream_start_timestamps_s:
        raise ValueError("Raw start timestamps for events and APS frames are required")
    origins = {
        name: Decimal(str(value))
        for name, value in stream_start_timestamps_s.items()
        if value is not None
    }
    if not origins or any(not value.is_finite() for value in origins.values()):
        raise ValueError("Stream start timestamps must be finite source seconds")
    common_origin_s = min(origins.values())
    offsets_us = {
        name: int(
            ((origin - common_origin_s) * Decimal(1_000_000)).to_integral_value(
                rounding=ROUND_HALF_UP
            )
        )
        for name, origin in origins.items()
    }
    return offsets_us, float(common_origin_s)


def _make_stream_slice(
    source: Mapping[str, Any] | None,
    offset_us: int | None,
    interval_start_us: int,
    interval_end_us: int,
) -> IndexedSamples | None:
    if source is None or offset_us is None:
        return None
    timestamps = source["ts"]
    start = int(
        np.searchsorted(timestamps, interval_start_us - offset_us, side="left")
    )
    end = int(np.searchsorted(timestamps, interval_end_us - offset_us, side="left"))
    return IndexedSamples(source, start, end, offset_us)


def make_frame_windows(
    recording: DavisRecording,
    stream_start_timestamps_s: Mapping[str, float],
) -> FrameWindows:
    """Index consecutive APS intervals and attach in-interval streams.

    `stream_start_timestamps_s` must contain raw ROS-bag timestamp origins in
    seconds for events and APS frames, and for each available IMU/ground-truth
    stream. Returned data retains Tonic's original stream-local microsecond
    timestamps; common-time coordinates are separate fields computed from
    explicit raw-origin offsets rounded to the Tonic microsecond resolution.
    """
    offsets_us, common_origin_s = _origin_offsets_us(stream_start_timestamps_s)
    required_offsets = {"events", "aps_frames"}
    available_offsets = {"events", "aps_frames"}
    if recording.imu is not None:
        available_offsets.add("imu")
    if recording.target is not None:
        available_offsets.add("ground_truth")
    missing = available_offsets - offsets_us.keys()
    missing_required = required_offsets - offsets_us.keys()
    if missing or missing_required:
        raise ValueError(
            "Raw stream origins must be provided for all available streams; "
            f"missing={sorted(missing | missing_required)}"
        )

    event_times = recording.events["t"]
    _, event_start, _ = _check_timestamp_array(event_times, "events")
    if event_start != 0:
        raise ValueError("Tonic event timestamps must start at their stream origin (0)")
    aps_times = recording.aps["ts"]
    _, aps_start, _ = _check_timestamp_array(
        aps_times, "APS frames", chunk_size=max(1, len(aps_times))
    )
    if aps_start != 0:
        raise ValueError("Tonic APS timestamps must start at their stream origin (0)")
    aps_common = aps_times + offsets_us["aps_frames"]
    if len(aps_common) > 1 and np.any(aps_common[1:] < aps_common[:-1]):
        raise ValueError("APS timestamps are not ordered on the common timeline")

    for name, source in (("imu", recording.imu), ("ground truth", recording.target)):
        if source is not None:
            _, local_start, _ = _check_timestamp_array(source["ts"], name)
            if local_start != 0:
                raise ValueError(f"Tonic {name} timestamps must start at their stream origin (0)")

    windows = FrameWindows(recording, aps_common, offsets_us, common_origin_s)
    validate_frame_windows(windows)
    return windows


def validate_frame_windows(windows: FrameWindows) -> dict[str, Any]:
    """Verify interval membership and exact, non-overlapping event partitioning."""
    event_times = windows.recording.events["t"]
    event_offset = windows.timestamp_offsets_us["events"]
    intervals = len(windows)
    if intervals == 0:
        raise ValueError("At least two APS frames are required to form an interval")

    frame_start = int(windows.aps_common_timestamps_us[0])
    frame_end = int(windows.aps_common_timestamps_us[-1])
    covered_start = int(
        np.searchsorted(event_times, frame_start - event_offset, side="left")
    )
    covered_end = int(
        np.searchsorted(event_times, frame_end - event_offset, side="left")
    )
    expected_events = covered_end - covered_start
    actual_events = 0
    empty_event_windows = 0
    synchronized_sample_counts = {"imu": 0, "ground_truth": 0}
    stream_index_ranges: dict[str, tuple[int, int] | None] = {
        "imu": None,
        "ground_truth": None,
    }
    if windows.get_window(0).event_start_index != covered_start:
        raise ValueError("First frame interval does not start at the first in-span event")
    if windows.get_window(intervals - 1).event_end_index != covered_end:
        raise ValueError("Last frame interval does not end after the last in-span event")
    for index in range(intervals):
        window = windows.get_window(index)
        t0, t1 = window.interval_us
        if t0 > t1:
            raise ValueError(f"Frame interval {index} has decreasing timestamps")
        local_times = window.events_k["t"]
        if len(local_times):
            common_times = local_times + event_offset
            if np.any(common_times < t0) or np.any(common_times >= t1):
                raise ValueError(f"Event outside half-open frame interval {index}")
        else:
            empty_event_windows += 1
        if window.event_start_index < covered_start or window.event_end_index > covered_end:
            raise ValueError(f"Frame interval {index} extends outside the APS span")
        if index and window.event_start_index != windows.get_window(index - 1).event_end_index:
            raise ValueError("Adjacent frame intervals overlap or lose event indices")
        actual_events += len(window.events_k)
        for name, samples in (
            ("imu", window.imu_k),
            ("ground_truth", window.groundtruth_k),
        ):
            if samples is None:
                continue
            common_times = samples.common_timestamps_us
            if len(common_times) and (
                np.any(common_times < t0) or np.any(common_times >= t1)
            ):
                raise ValueError(f"{name} sample outside half-open frame interval {index}")
            previous_range = stream_index_ranges[name]
            current_range = (samples.start_index, samples.end_index)
            if previous_range is not None and current_range[0] != previous_range[1]:
                raise ValueError(
                    f"Adjacent frame intervals overlap or lose {name} samples"
                )
            stream_index_ranges[name] = (
                (current_range[0], current_range[1])
                if previous_range is None
                else (previous_range[0], current_range[1])
            )
            synchronized_sample_counts[name] += len(samples)
    if actual_events != expected_events:
        raise ValueError(
            f"Event partition lost or duplicated samples: expected {expected_events}, "
            f"found {actual_events}"
        )

    first_time = int(windows.aps_common_timestamps_us[0])
    last_time = int(windows.aps_common_timestamps_us[-1])
    for name, source, offset in (
        ("imu", windows.recording.imu, windows.timestamp_offsets_us.get("imu")),
        (
            "ground_truth",
            windows.recording.target,
            windows.timestamp_offsets_us.get("ground_truth"),
        ),
    ):
        if source is None or offset is None:
            continue
        expected_start = int(
            np.searchsorted(source["ts"], first_time - offset, side="left")
        )
        expected_end = int(
            np.searchsorted(source["ts"], last_time - offset, side="left")
        )
        if stream_index_ranges[name] != (expected_start, expected_end):
            raise ValueError(f"{name} samples were lost or duplicated in the APS span")

    return {
        "passed": True,
        "checks": {
            "half_open_event_membership": True,
            "adjacent_event_indices_contiguous": True,
            "no_event_overlap": True,
            "no_event_lost_within_aps_span": True,
            "imu_and_groundtruth_half_open_membership": True,
            "imu_and_groundtruth_partitioned_without_loss": True,
            "empty_windows_supported": True,
            "first_and_last_aps_frames_used_as_interval_boundaries": True,
        },
        "frame_interval_count": intervals,
        "events_within_aps_span": expected_events,
        "empty_event_windows": empty_event_windows,
        "events_before_first_aps_frame": covered_start,
        "events_at_or_after_last_aps_frame": len(event_times) - covered_end,
        "imu_samples_within_aps_span": synchronized_sample_counts["imu"],
        "groundtruth_samples_within_aps_span": synchronized_sample_counts[
            "ground_truth"
        ],
        "timestamp_unit": TIMESTAMP_UNIT,
    }


def plot_frame_intervals(
    windows: FrameWindows,
    output_path: Path,
    interval_count: int = 5,
) -> Path:
    """Plot event, IMU, and ground-truth timestamps for sampled APS intervals."""
    import matplotlib.pyplot as plt

    if interval_count <= 0:
        raise ValueError("interval_count must be positive")
    if len(windows) == 0:
        raise ValueError("At least two APS frames are required to plot intervals")
    indices = np.unique(
        np.linspace(0, len(windows) - 1, min(interval_count, len(windows)), dtype=int)
    )
    fig, axes = plt.subplots(
        len(indices), 1, figsize=(12, max(3.0, 2.3 * len(indices))), squeeze=False
    )
    lane_y = {"events": 0, "imu": 1, "ground truth": 2, "APS": 3}

    for ax, index in zip(axes[:, 0], indices, strict=True):
        window = windows.get_window(int(index))
        start_us, end_us = window.interval_us
        duration_us = end_us - start_us
        event_times = window.events_k["t"] + window.event_timestamp_offset_us
        ax.scatter(
            (event_times - start_us) / 1000,
            np.full(len(event_times), lane_y["events"]),
            s=1,
            alpha=0.35,
            color="tab:blue",
            rasterized=True,
        )
        for samples, y, color in (
            (window.imu_k, lane_y["imu"], "tab:orange"),
            (window.groundtruth_k, lane_y["ground truth"], "tab:green"),
        ):
            if samples is not None and len(samples):
                ax.scatter(
                    (samples.common_timestamps_us - start_us) / 1000,
                    np.full(len(samples), y),
                    s=9,
                    marker="|",
                    color=color,
                )
        ax.scatter(
            [0, duration_us / 1000],
            [lane_y["APS"], lane_y["APS"]],
            s=28,
            marker="|",
            color="tab:red",
        )
        ax.set_yticks(list(lane_y.values()), list(lane_y.keys()))
        ax.set_xlim(0, max(duration_us / 1000, 1e-9))
        ax.set_title(
            f"APS frames {index}–{index + 1}: "
            f"[{start_us}, {end_us}) us, events={len(window.events_k)}"
        )
        ax.set_xlabel("Time from first APS frame (ms)")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        "DAVIS frame intervals on a common timeline from raw bag timestamp origins",
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_synchronization_report(
    windows: FrameWindows,
    validation: Mapping[str, Any],
    manifest_info: Mapping[str, Any],
    output_path: Path,
    recording_name: str = DEFAULT_RECORDING,
) -> Path:
    """Write a concise human-readable synchronization report."""
    recording = windows.recording
    aps_times = recording.aps["ts"]
    offsets = windows.timestamp_offsets_us
    text = f"""# DAVIS frame-window synchronization report

- Recording: `{recording_name}`
- Timestamp values in the loaded Tonic arrays: integer **microseconds**, preserved unchanged.
- Raw ROS-bag stream origins: **seconds** from `{manifest_info['manifest_path']}`.
- Common comparison axis: microseconds relative to the earliest raw stream start
  ({windows.common_origin_s:.7f} s in the source clock); per-stream origin
  differences are explicitly rounded to the Tonic microsecond resolution.
- APS frame count: {len(aps_times):,}; consecutive half-open frame intervals:
  {len(windows):,}.
- Origin offsets on the common axis (microseconds): {dict(offsets)}.
- Events assigned within the first-to-last APS interval span:
  {validation['events_within_aps_span']:,}.
- Empty event intervals: {validation['empty_event_windows']:,}.
- Events before the first APS frame: {validation['events_before_first_aps_frame']:,}.
- Events at or after the final APS frame: {validation['events_at_or_after_last_aps_frame']:,}.
- IMU samples assigned inside the APS span:
  {validation['imu_samples_within_aps_span']:,}.
- Ground-truth samples assigned inside the APS span:
  {validation['groundtruth_samples_within_aps_span']:,}.
- Event membership uses `[t_k, t_(k+1))`; samples remain index slices into the
  original arrays and are not copied per interval.
- IMU and ground truth are selected with the same half-open interval on the
  common timeline. Their returned sample timestamps remain stream-local Tonic
  microseconds; `common_timestamps_us` exposes mapped times separately.
- Validation passed: `{validation['passed']}`.
- Synchronization method: add measured per-stream source-start offsets from the
  raw ROS-bag manifest. No interpolation, resampling, clock correction, or
  calibration is performed.
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create and validate consecutive DAVIS APS-frame time windows."
    )
    parser.add_argument("--data-dir", type=Path, default=DavisConfig().data_dir)
    parser.add_argument("--recording", default=DEFAULT_RECORDING)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "dataset_synchronization",
    )
    args = parser.parse_args()

    config = DavisConfig(data_dir=args.data_dir, recording=args.recording)
    recording = load_davis(config)
    origins, manifest_info = load_timestamp_origins(recording, config)
    windows = make_frame_windows(recording, origins)
    validation = validate_frame_windows(windows)
    plot_path = plot_frame_intervals(
        windows, args.output_dir / f"{args.recording}_frame_intervals.png"
    )
    report_path = write_synchronization_report(
        windows,
        validation,
        manifest_info,
        args.output_dir / f"{args.recording}_synchronization_report.md",
        recording_name=args.recording,
    )
    print(f"Frame intervals: {len(windows):,}")
    print(f"Events assigned inside APS span: {validation['events_within_aps_span']:,}")
    print(f"Events before first APS frame: {validation['events_before_first_aps_frame']:,}")
    print(
        "Events at/after last APS frame: "
        f"{validation['events_at_or_after_last_aps_frame']:,}"
    )
    print(f"Validation passed: {validation['passed']}")
    print(f"Report: {report_path}")
    print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
