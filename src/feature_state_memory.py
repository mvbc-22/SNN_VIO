"""Fixed-capacity software model of persistent APS feature-state memory."""

from __future__ import annotations

import argparse
import json
import operator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .aps_features import (
    ShiTomasiConfig,
    extract_shi_tomasi_features,
    get_aps_frame,
)
from .davis_inspection import DavisConfig, load_davis
from .frame_synchronization import (
    load_timestamp_origins,
    make_frame_windows,
)

FEATURE_STATE_DTYPE = np.dtype(
    [
        ("id", np.int32),
        ("x", np.float32),
        ("y", np.float32),
        ("vx", np.float32),
        ("vy", np.float32),
        ("confidence", np.float32),
        ("timestamp", np.int64),
        ("valid", np.uint8),
    ]
)
INVALID_ID = np.int32(-1)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "feature_state_memory"
)


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _timestamp(value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("timestamp must be an integer number of microseconds")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            "timestamp must be an integer number of microseconds"
        ) from exc
    if not np.iinfo(np.int64).min <= result <= np.iinfo(np.int64).max:
        raise ValueError("timestamp is outside the int64 range")
    return result


def _confidence(value: float) -> float:
    result = _finite_float(value, "confidence")
    if not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be within [0, 1]")
    return result


class FeatureStateMemory:
    """Fixed-size array-backed feature records with deterministic slot reuse.

    Coordinates are pixels, velocities are pixels per microsecond, timestamps
    are preserved integer microseconds, and confidence is in [0, 1].
    """

    def __init__(self, slot_count: int) -> None:
        if isinstance(slot_count, (bool, np.bool_)) or not isinstance(
            slot_count, (int, np.integer)
        ):
            raise TypeError("slot_count must be a positive integer")
        if slot_count <= 0:
            raise ValueError("slot_count must be a positive integer")
        self._states = np.zeros(slot_count, dtype=FEATURE_STATE_DTYPE)
        self._states["id"] = INVALID_ID
        self._next_id = 0

    @property
    def slot_count(self) -> int:
        return len(self._states)

    @property
    def storage(self) -> np.ndarray:
        """Read-only fixed-capacity view of all state slots, including invalid ones."""
        view = self._states.view()
        view.flags.writeable = False
        return view

    def allocate(
        self,
        x: float,
        y: float,
        timestamp: int,
        *,
        vx: float = 0.0,
        vy: float = 0.0,
        confidence: float = 1.0,
    ) -> int:
        """Allocate the first invalid slot and return its slot index."""
        free_slots = np.flatnonzero(self._states["valid"] == 0)
        if not len(free_slots):
            raise BufferError("FeatureStateMemory is full")
        if self._next_id > np.iinfo(np.int32).max:
            raise OverflowError("FeatureState id space exhausted")

        values = (
            _finite_float(x, "x"),
            _finite_float(y, "y"),
            _finite_float(vx, "vx"),
            _finite_float(vy, "vy"),
            _confidence(confidence),
            _timestamp(timestamp),
        )
        slot = int(free_slots[0])
        self._states[slot] = (
            self._next_id,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            1,
        )
        self._next_id += 1
        return slot

    def update(
        self,
        slot: int,
        *,
        x: float,
        y: float,
        timestamp: int,
        vx: float | None = None,
        vy: float | None = None,
        confidence: float | None = None,
    ) -> None:
        """Replace position/time and optionally velocity/confidence in a valid slot."""
        index = self._slot_index(slot)
        state = self._states[index]
        if not state["valid"]:
            raise ValueError(f"FeatureState slot {index} is invalid")
        next_timestamp = _timestamp(timestamp)
        if next_timestamp < int(state["timestamp"]):
            raise ValueError("timestamp must not move backward")

        next_x = _finite_float(x, "x")
        next_y = _finite_float(y, "y")
        next_vx = float(state["vx"]) if vx is None else _finite_float(vx, "vx")
        next_vy = float(state["vy"]) if vy is None else _finite_float(vy, "vy")
        next_confidence = (
            float(state["confidence"])
            if confidence is None
            else _confidence(confidence)
        )
        state["x"] = next_x
        state["y"] = next_y
        state["vx"] = next_vx
        state["vy"] = next_vy
        state["confidence"] = next_confidence
        state["timestamp"] = next_timestamp

    def invalidate(self, slot: int) -> None:
        """Reset a valid or already-invalid slot to the documented invalid record."""
        index = self._slot_index(slot)
        self._states[index] = (INVALID_ID, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    def predict(self, slot: int, timestamp: int) -> tuple[float, float]:
        """Predict coordinates without mutating state; timestamp must not precede it."""
        index = self._slot_index(slot)
        state = self._states[index]
        if not state["valid"]:
            raise ValueError(f"FeatureState slot {index} is invalid")
        target_timestamp = _timestamp(timestamp)
        delta_t = target_timestamp - int(state["timestamp"])
        if delta_t < 0:
            raise ValueError("prediction timestamp must not precede state timestamp")
        return (
            float(state["x"]) + float(state["vx"]) * delta_t,
            float(state["y"]) + float(state["vy"]) * delta_t,
        )

    def get_active(self) -> np.ndarray:
        """Return a compact copy of valid records in slot order."""
        return self._states[self._states["valid"] != 0].copy()

    def _slot_index(self, slot: int) -> int:
        if isinstance(slot, (bool, np.bool_)) or not isinstance(
            slot, (int, np.integer)
        ):
            raise TypeError("slot must be an integer")
        index = int(slot)
        if index < 0 or index >= self.slot_count:
            raise IndexError(f"slot must be within [0, {self.slot_count - 1}]")
        return index


def initialize_from_aps_features(
    features: np.ndarray,
    timestamp_us: int,
    slot_count: int,
) -> FeatureStateMemory:
    """Load valid Shi-Tomasi records with zero velocity and full initial confidence."""
    if features.dtype.names is None or not {
        "x",
        "y",
        "valid",
    }.issubset(features.dtype.names):
        raise TypeError("features must contain x, y, and valid fields")
    valid_features = features[features["valid"] != 0]
    if len(valid_features) > slot_count:
        raise BufferError(
            f"{len(valid_features)} valid detections exceed {slot_count} state slots"
        )
    memory = FeatureStateMemory(slot_count)
    for feature in valid_features:
        memory.allocate(
            float(feature["x"]),
            float(feature["y"]),
            timestamp_us,
            vx=0.0,
            vy=0.0,
            confidence=1.0,
        )
    return memory


def run_initialization(
    frame: np.ndarray,
    timestamp_us: int,
    feature_count: int = 500,
    slot_count: int = 500,
    quality_level: float = 0.01,
    min_feature_distance_px: float = 5.0,
    block_size_px: int = 3,
    bucket_columns: int = 8,
    bucket_rows: int = 6,
) -> tuple[np.ndarray, FeatureStateMemory]:
    """Detect Shi-Tomasi corners in one APS frame and initialize fixed state slots."""
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    if feature_count > slot_count:
        raise ValueError("feature_count cannot exceed slot_count")
    result = extract_shi_tomasi_features(
        frame,
        ShiTomasiConfig(
            max_features=feature_count,
            quality_level=quality_level,
            min_feature_distance=min_feature_distance_px,
            block_size=block_size_px,
            bucket_columns=bucket_columns,
            bucket_rows=bucket_rows,
        ),
        frame_timestamp_us=timestamp_us,
    )
    return result.features, initialize_from_aps_features(
        result.features, timestamp_us, slot_count
    )


def save_initialization_outputs(
    frame: np.ndarray,
    features: np.ndarray,
    memory: FeatureStateMemory,
    frame_index: int,
    output_dir: Path,
    configuration: dict[str, Any],
) -> dict[str, Path]:
    """Persist numeric states, metadata, a visualization, and a short report."""
    if features.dtype != np.dtype(
        [
            ("id", np.int32),
            ("x", np.float32),
            ("y", np.float32),
            ("quality", np.float32),
            ("valid", np.uint8),
        ]
    ):
        raise TypeError("features must use the Stage 2A feature record dtype")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states = memory.storage

    state_path = output_dir / "davis_feature_states.npz"
    np.savez_compressed(
        state_path,
        states=states,
        active_states=memory.get_active(),
        features=features,
    )
    config_path = output_dir / "davis_feature_state_configuration.json"
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "frame_index": int(frame_index),
        "slot_count": memory.slot_count,
        "active_state_count": len(memory.get_active()),
        "state_dtype": [
            {"name": name, "dtype": str(FEATURE_STATE_DTYPE.fields[name][0])}
            for name in FEATURE_STATE_DTYPE.names
        ],
        "timestamp_unit": "microseconds (original Stage 1 Tonic units)",
        "velocity_unit": "pixels per microsecond",
        "invalid_state": {
            "id": -1,
            "x": 0,
            "y": 0,
            "vx": 0,
            "vy": 0,
            "confidence": 0,
            "timestamp": 0,
            "valid": 0,
        },
        "confidence_initialization": (
            "1.0 for APS-detected valid features; Stage 2A detector response is "
            "preserved separately in the features array and is not directly "
            "normalized into confidence."
        ),
    }
    config_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    active = memory.get_active()
    figure_path = output_dir / "davis_feature_state_initialization.png"
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
    ax.scatter(
        active["x"],
        active["y"],
        s=18,
        c="tab:red",
        marker="+",
        linewidths=0.8,
    )
    ax.set_title(
        f"APS frame {frame_index}: {len(active)} initialized FeatureState records "
        f"({memory.slot_count} fixed slots)"
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    report_path = output_dir / "davis_feature_state_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Stage 3 — Persistent FeatureState memory initialization",
                "",
                f"- APS frame index: {frame_index}",
                f"- Requested Shi-Tomasi features: {configuration['feature_count']}",
                f"- Detected features: {len(features)}",
                f"- Fixed state slots: {memory.slot_count}",
                f"- Active initialized states: {len(active)}",
                f"- State timestamp: {configuration['timestamp_us']} microseconds",
                "- State velocity: initialized to 0 pixels per microsecond.",
                "- Confidence: initialized to 1.0 for each valid APS detection.",
                "- Unused slots: `id=-1`, numeric fields zero, `valid=0`.",
                "- No image patches, event association, or event processing are included.",
                "",
                "The NPZ contains the full fixed-capacity `states` array, an `active_states` "
                "compact view, and the originating Stage 2A detections. All state fields "
                "use fixed-width numeric dtypes suitable for a BRAM-oriented layout.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"states": state_path, "configuration": config_path, "visualization": figure_path, "report": report_path}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize fixed-capacity FeatureState memory from a DAVIS APS frame."
    )
    parser.add_argument("--recording", default="shapes_6dof")
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--feature-count", type=int, default=500)
    parser.add_argument("--slots", type=int, default=500)
    parser.add_argument("--quality-level", type=float, default=0.01)
    parser.add_argument("--min-feature-distance", type=float, default=5.0)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--bucket-columns", type=int, default=8)
    parser.add_argument("--bucket-rows", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.frame_index is not None and args.frame_index < 0:
        parser.error("--frame-index must be non-negative")
    if args.feature_count <= 0 or args.slots <= 0:
        parser.error("--feature-count and --slots must be positive")
    if args.feature_count > args.slots:
        parser.error("--feature-count cannot exceed --slots")

    dataset_config = DavisConfig(recording=args.recording)
    recording = load_davis(dataset_config)
    origins_s, _ = load_timestamp_origins(recording, dataset_config)
    windows = make_frame_windows(recording, origins_s)
    frame_count = len(windows) + 1
    frame_index = (
        frame_count // 2
        if args.frame_index is None
        else args.frame_index
    )
    if frame_index >= frame_count:
        parser.error(f"--frame-index must be within [0, {frame_count - 1}]")
    frame, timestamp_us = get_aps_frame(windows, frame_index)
    features, memory = run_initialization(
        frame,
        timestamp_us,
        feature_count=args.feature_count,
        slot_count=args.slots,
        quality_level=args.quality_level,
        min_feature_distance_px=args.min_feature_distance,
        block_size_px=args.block_size,
        bucket_columns=args.bucket_columns,
        bucket_rows=args.bucket_rows,
    )
    configuration = {
        "recording": args.recording,
        "frame_index": frame_index,
        "timestamp_us": timestamp_us,
        "feature_count": args.feature_count,
        "slot_count": args.slots,
        "quality_level": args.quality_level,
        "min_feature_distance_px": args.min_feature_distance,
        "block_size_px": args.block_size,
        "bucket_grid": [args.bucket_columns, args.bucket_rows],
        "detector": "Shi-Tomasi",
    }
    outputs = save_initialization_outputs(
        frame,
        features,
        memory,
        frame_index,
        args.output_dir,
        configuration,
    )
    print(
        f"APS frame {frame_index} at {timestamp_us} us: "
        f"{len(features)} detections initialized in {len(memory.get_active())}/"
        f"{memory.slot_count} fixed state slots"
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
