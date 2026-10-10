"""Brute-force Stage 4A event-to-feature association reference."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from .aps_features import ShiTomasiConfig, extract_shi_tomasi_features
from .davis_inspection import DavisConfig, load_davis
from .feature_state_memory import FeatureStateMemory, initialize_from_aps_features
from .frame_synchronization import (
    FrameWindow,
    load_timestamp_origins,
    make_frame_windows,
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "event_association"
)
ASSOCIATION_DTYPE = np.dtype(
    [
        ("event_index", np.int64),
        ("t_event", np.int64),
        ("t_common_us", np.int64),
        ("x_event", np.int64),
        ("y_event", np.int64),
        ("p", np.int64),
        ("feature_id", np.int32),
        ("feature_slot", np.int32),
        ("distance_squared", np.float64),
        ("candidate_count", np.uint32),
        ("associated", np.uint8),
    ]
)
EVENTS_PER_FEATURE_DTYPE = np.dtype(
    [("feature_id", np.int32), ("slot", np.int32), ("event_count", np.int64)]
)


@dataclass(frozen=True)
class AssociationResult:
    """Ordered event decisions and quantitative diagnostic summaries."""

    assignments: np.ndarray
    events_per_feature: np.ndarray
    diagnostics: Mapping[str, int | float]


@dataclass(frozen=True)
class SingleEventAssociation:
    """Result of applying the Stage 4A gates to one event and current memory."""

    feature_id: int
    feature_slot: int
    distance_squared: float
    candidate_count: int


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _int64_field_values(values: np.ndarray, name: str) -> np.ndarray:
    if np.issubdtype(values.dtype, np.unsignedinteger):
        if values.size and int(np.max(values)) > np.iinfo(np.int64).max:
            raise OverflowError(f"event field {name!r} is outside int64 range")
    elif values.size and (
        int(np.min(values)) < np.iinfo(np.int64).min
        or int(np.max(values)) > np.iinfo(np.int64).max
    ):
        raise OverflowError(f"event field {name!r} is outside int64 range")
    return values.astype(np.int64, copy=False)


def _required_fields(events: np.ndarray) -> None:
    required = {"x", "y", "t", "p"}
    if events.dtype.names is None or not required.issubset(events.dtype.names):
        raise TypeError("events must be structured and contain x, y, t, and p fields")
    for field in required:
        if not np.issubdtype(events.dtype.fields[field][0], np.integer):
            raise TypeError(f"event field {field!r} must have an integer dtype")


def _select_candidate(
    active: np.ndarray,
    event_time: int,
    event_x: int,
    event_y: int,
    radius_squared: float,
    time_window: float,
) -> tuple[int, int, float | None, int]:
    candidate_count = 0
    distance_calculations = 0
    best_distance_squared: float | None = None
    best_active_index = -1
    best_feature_id: int | None = None
    for active_index, state in enumerate(active):
        delta_t = event_time - int(state["timestamp"])
        if abs(delta_t) >= time_window:
            continue
        distance_calculations += 1
        predicted_x = float(state["x"]) + float(state["vx"]) * delta_t
        predicted_y = float(state["y"]) + float(state["vy"]) * delta_t
        dx = event_x - predicted_x
        dy = event_y - predicted_y
        distance_squared = dx * dx + dy * dy
        if distance_squared < radius_squared:
            candidate_count += 1
            feature_id = int(state["id"])
            if (
                best_distance_squared is None
                or distance_squared < best_distance_squared
                or (
                    distance_squared == best_distance_squared
                    and feature_id < best_feature_id
                )
            ):
                best_distance_squared = distance_squared
                best_active_index = active_index
                best_feature_id = feature_id
    return (
        candidate_count,
        best_active_index,
        best_distance_squared,
        distance_calculations,
    )


def associate_one_event_brute_force(
    event_time_us: int,
    event_x: int,
    event_y: int,
    memory: FeatureStateMemory,
    radius_px: float,
    time_window_us: float,
) -> SingleEventAssociation:
    """Apply the Stage 4A policy to one event against the current memory."""
    radius = _positive_finite(radius_px, "radius_px")
    time_window = _positive_finite(time_window_us, "time_window_us")
    if not np.isfinite(radius * radius):
        raise ValueError("radius_px is too large to square safely")
    values = (event_time_us, event_x, event_y)
    labels = ("event_time_us", "event_x", "event_y")
    for value, label in zip(values, labels):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{label} must be an integer")
    event_time, x, y = (int(value) for value in values)
    if not np.iinfo(np.int64).min <= event_time <= np.iinfo(np.int64).max:
        raise OverflowError("event_time_us is outside int64 range")
    active = memory.get_active()
    count, active_index, distance_squared, _ = _select_candidate(
        active, event_time, x, y, radius * radius, time_window
    )
    if active_index < 0:
        return SingleEventAssociation(-1, -1, float("nan"), count)
    feature_id = int(active[active_index]["id"])
    valid_slots = np.flatnonzero(memory.storage["valid"] != 0)
    slot = int(valid_slots[active_index])
    return SingleEventAssociation(feature_id, slot, float(distance_squared), count)


def associate_events_brute_force(
    events: np.ndarray,
    memory: FeatureStateMemory,
    radius_px: float,
    time_window_us: float,
    *,
    event_timestamp_offset_us: int = 0,
) -> AssociationResult:
    """Associate chronologically using strict temporal/spatial gates and no sqrt.

    When multiple features match, the event is marked ambiguous and assigned to
    the feature with minimum squared distance, breaking exact ties by feature ID.
    Each event is assigned to at most one feature. Unmatched events are rejected.
    Polarity is copied to the output but is not used by the gates.
    """
    radius = _positive_finite(radius_px, "radius_px")
    time_window = _positive_finite(time_window_us, "time_window_us")
    if not np.isfinite(radius * radius):
        raise ValueError("radius_px is too large to square safely")
    _required_fields(events)
    if isinstance(event_timestamp_offset_us, (bool, np.bool_)) or not isinstance(
        event_timestamp_offset_us, (int, np.integer)
    ):
        raise TypeError("event_timestamp_offset_us must be an integer")

    active = memory.get_active()
    if len(active) and (
        not np.isfinite(active["x"]).all()
        or not np.isfinite(active["y"]).all()
        or not np.isfinite(active["vx"]).all()
        or not np.isfinite(active["vy"]).all()
    ):
        raise ValueError("active FeatureState positions and velocities must be finite")

    event_times = _int64_field_values(events["t"], "t")
    event_xs = _int64_field_values(events["x"], "x")
    event_ys = _int64_field_values(events["y"], "y")
    polarities = _int64_field_values(events["p"], "p")
    if len(events):
        offset = int(event_timestamp_offset_us)
        if offset > 0 and int(event_times.max()) > np.iinfo(np.int64).max - offset:
            raise OverflowError("common event timestamp is outside int64 range")
        if offset < 0 and int(event_times.min()) < np.iinfo(np.int64).min - offset:
            raise OverflowError("common event timestamp is outside int64 range")
        common_times = event_times + offset
        # Stable ordering keeps source order for equal-timestamp events.
        processing_order = np.argsort(common_times, kind="stable")
    else:
        common_times = np.empty(0, dtype=np.int64)
        processing_order = np.empty(0, dtype=np.int64)

    assignments = np.zeros(len(events), dtype=ASSOCIATION_DTYPE)
    event_counts = np.zeros(len(active), dtype=np.int64)
    state_slots = np.flatnonzero(memory.storage["valid"] != 0)
    slots_by_id = {
        int(memory.storage["id"][slot]): int(slot) for slot in state_slots
    }
    radius_squared = radius * radius
    associated_count = 0
    ambiguous_count = 0
    distance_calculations_total = 0
    candidate_features_total = 0
    started_ns = time.perf_counter_ns()

    for output_index, event_index in enumerate(processing_order):
        event = events[event_index]
        event_time = int(common_times[event_index])
        event_x = int(event_xs[event_index])
        event_y = int(event_ys[event_index])
        polarity = int(polarities[event_index])

        (
            candidate_count,
            best_active_index,
            best_distance_squared,
            distance_calculations,
        ) = _select_candidate(
            active,
            event_time,
            event_x,
            event_y,
            radius_squared,
            time_window,
        )
        distance_calculations_total += distance_calculations
        candidate_features_total += len(active)

        is_associated = candidate_count > 0
        if candidate_count > 1:
            ambiguous_count += 1
        feature_slot = -1
        feature_id = -1
        selected_distance_squared = np.nan
        if is_associated:
            state = active[best_active_index]
            feature_slot = slots_by_id[int(state["id"])]
            feature_id = int(state["id"])
            selected_distance_squared = float(best_distance_squared)
            event_counts[best_active_index] += 1
            associated_count += 1

        assignments[output_index] = (
            int(event_index),
            int(event["t"]),
            event_time,
            event_x,
            event_y,
            polarity,
            feature_id,
            feature_slot,
            selected_distance_squared,
            candidate_count,
            int(is_associated),
        )

    per_feature = np.zeros(len(active), dtype=EVENTS_PER_FEATURE_DTYPE)
    if len(active):
        for index, state in enumerate(active):
            per_feature[index] = (
                int(state["id"]),
                slots_by_id[int(state["id"])],
                int(event_counts[index]),
            )

    processed = len(events)
    elapsed_seconds = (time.perf_counter_ns() - started_ns) / 1e9
    diagnostics: dict[str, int | float] = {
        "events_processed": processed,
        "events_associated": associated_count,
        "association_ratio": float(associated_count / processed) if processed else 0.0,
        "events_rejected": processed - associated_count,
        "events_ambiguous": ambiguous_count,
        "active_features": len(active),
        "candidate_features_total": candidate_features_total,
        "candidate_features_per_event": (
            candidate_features_total / processed if processed else 0.0
        ),
        "distance_calculations_total": distance_calculations_total,
        "distance_calculations_per_event": (
            distance_calculations_total / processed if processed else 0.0
        ),
        "total_runtime_seconds": elapsed_seconds,
        "events_processed_per_second": (
            processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
    }
    return AssociationResult(assignments, per_feature, diagnostics)


def save_association_outputs(
    frame: np.ndarray,
    states: np.ndarray,
    result: AssociationResult,
    frame_window: FrameWindow,
    radius_px: float,
    time_window_us: float,
    output_dir: Path,
    maximum_plot_associations: int = 300,
) -> dict[str, Path]:
    """Write decisions, diagnostics, report, and a frame/event/link visualization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "davis_event_associations.npz"
    np.savez_compressed(
        records_path,
        assignments=result.assignments,
        events_per_feature=result.events_per_feature,
    )
    diagnostics_path = output_dir / "davis_event_association_diagnostics.json"
    diagnostics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame_window_index": frame_window.index,
        "interval_common_time_us": list(frame_window.interval_us),
        "event_count_in_interval": len(frame_window.events_k),
        "event_timestamp_offset_us": frame_window.event_timestamp_offset_us,
        "radius_px": float(radius_px),
        "time_window_us": float(time_window_us),
        "timestamp_interpretation": (
            "Original event and APS timestamps are integer microseconds. Event "
            "timestamps are shifted by the Stage 1 stream-origin offset only for "
            "common-axis association; original event t is retained in decisions."
        ),
        "association_policy": (
            "Stable chronological processing (source order for equal times); "
            "strict d2 < R2 and abs(dt) < T; no match is rejected; multiple "
            "candidates are counted ambiguous and assigned once to minimum d2, "
            "then lower feature ID. Polarity is preserved but not used."
        ),
        **dict(result.diagnostics),
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )

    csv_path = output_dir / "davis_event_associations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(ASSOCIATION_DTYPE.names)
        for assignment in result.assignments:
            writer.writerow(tuple(assignment[name] for name in ASSOCIATION_DTYPE.names))

    figure_path = output_dir / "davis_event_association_visualization.png"
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
    active_states = states[states["valid"] != 0]
    if len(active_states):
        ax.scatter(
            active_states["x"],
            active_states["y"],
            s=18,
            facecolors="none",
            edgecolors="lime",
            linewidths=0.7,
            label="APS feature states",
        )

    assignments = result.assignments
    if len(assignments):
        positive = assignments["p"] > 0
        for polarity_value, color, marker in ((False, "tab:blue", "."), (True, "tab:orange", "x")):
            selected = positive == polarity_value
            ax.scatter(
                assignments["x_event"][selected],
                assignments["y_event"][selected],
                s=5,
                c=color,
                marker=marker,
                alpha=0.55,
                label=f"events p={'positive' if polarity_value else 'non-positive'}",
            )
        linked = assignments[assignments["associated"] != 0]
        if len(linked):
            if len(linked) > maximum_plot_associations:
                selected_indices = np.linspace(
                    0, len(linked) - 1, maximum_plot_associations, dtype=np.int64
                )
                linked = linked[selected_indices]
            slot_indices = linked["feature_slot"].astype(np.int64)
            ax.plot(
                np.stack(
                    (
                        linked["x_event"],
                        states["x"][slot_indices],
                    )
                ),
                np.stack(
                    (
                        linked["y_event"],
                        states["y"][slot_indices],
                    )
                ),
                color="magenta",
                linewidth=0.45,
                alpha=0.45,
            )
    ax.set_title(
        f"Frame interval {frame_window.index}: "
        f"{result.diagnostics['events_associated']}/"
        f"{result.diagnostics['events_processed']} events associated "
        f"(R={radius_px:g}px, T={time_window_us:g}us)"
    )
    ax.set_axis_off()
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    report_path = output_dir / "davis_event_association_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Stage 4A — Brute-force event-to-feature association",
                "",
                f"- Frame interval: {frame_window.index}, "
                f"[{frame_window.interval_us[0]}, {frame_window.interval_us[1]}) us",
                f"- Active feature states: {result.diagnostics['active_features']}",
                f"- Events processed: {result.diagnostics['events_processed']}",
                f"- Events associated: {result.diagnostics['events_associated']}",
                f"- Association ratio: {result.diagnostics['association_ratio']:.6f}",
                f"- Events rejected: {result.diagnostics['events_rejected']}",
                f"- Ambiguous events: {result.diagnostics['events_ambiguous']}",
                f"- Spatial threshold R: {radius_px:g} px (strict squared-distance gate)",
                f"- Temporal threshold T: {time_window_us:g} us (strict absolute-time gate)",
                "- Chronological policy: stable timestamp sort, preserving source order for ties.",
                "- No match: rejected; feature ID and slot are -1.",
                "- Multiple matches: count as ambiguous; select minimum squared distance, then lower feature ID.",
                "- One event can be assigned to at most one feature; a feature can receive multiple events.",
                "- Polarity is retained for every event, visualized, and not used for association.",
                "- Feature states are not updated by this stage.",
                "- Distance uses squared arithmetic only; no square root is computed.",
                "",
                "Events per feature (including zero-count active features) are included in the NPZ archive.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "assignments": records_path,
        "diagnostics": diagnostics_path,
        "csv": csv_path,
        "visualization": figure_path,
        "report": report_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run brute-force event-to-feature association on one DAVIS APS interval."
    )
    parser.add_argument("--recording", default="shapes_6dof")
    parser.add_argument("--interval-index", type=int, default=678)
    parser.add_argument("--feature-count", type=int, default=500)
    parser.add_argument("--slots", type=int, default=500)
    parser.add_argument("--radius-px", type=float, default=8.0)
    parser.add_argument("--time-window-us", type=float, default=50000.0)
    parser.add_argument("--quality-level", type=float, default=0.01)
    parser.add_argument("--min-feature-distance", type=float, default=5.0)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--bucket-columns", type=int, default=8)
    parser.add_argument("--bucket-rows", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.interval_index < 0:
        parser.error("--interval-index must be non-negative")
    if args.feature_count <= 0 or args.slots <= 0:
        parser.error("--feature-count and --slots must be positive")
    if args.feature_count > args.slots:
        parser.error("--feature-count cannot exceed --slots")
    try:
        _positive_finite(args.radius_px, "radius_px")
        _positive_finite(args.time_window_us, "time_window_us")
    except ValueError as exc:
        parser.error(str(exc))

    dataset_config = DavisConfig(recording=args.recording)
    recording = load_davis(dataset_config)
    origins_s, _ = load_timestamp_origins(recording, dataset_config)
    windows = make_frame_windows(recording, origins_s)
    if args.interval_index >= len(windows):
        parser.error(f"--interval-index must be within [0, {len(windows) - 1}]")
    frame_window = windows.get_window(args.interval_index)
    features = extract_shi_tomasi_features(
        frame_window.frame_k,
        ShiTomasiConfig(
            max_features=args.feature_count,
            quality_level=args.quality_level,
            min_feature_distance=args.min_feature_distance,
            block_size=args.block_size,
            bucket_columns=args.bucket_columns,
            bucket_rows=args.bucket_rows,
        ),
        frame_index=args.interval_index,
        frame_timestamp_us=frame_window.frame_k_common_time_us,
    ).features
    memory = initialize_from_aps_features(
        features, frame_window.frame_k_common_time_us, args.slots
    )
    result = associate_events_brute_force(
        frame_window.events_k,
        memory,
        args.radius_px,
        args.time_window_us,
        event_timestamp_offset_us=frame_window.event_timestamp_offset_us,
    )
    outputs = save_association_outputs(
        frame_window.frame_k,
        memory.storage,
        result,
        frame_window,
        args.radius_px,
        args.time_window_us,
        args.output_dir,
    )
    print(json.dumps(dict(result.diagnostics), indent=2))
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
