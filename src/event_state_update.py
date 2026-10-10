"""Stage 4B chronological asynchronous alpha-beta FeatureState updates."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from .aps_features import ShiTomasiConfig, extract_shi_tomasi_features
from .davis_inspection import DavisConfig, load_davis
from .event_association import (
    EVENTS_PER_FEATURE_DTYPE,
    _int64_field_values,
    _required_fields,
    _select_candidate,
)
from .feature_state_memory import (
    FEATURE_STATE_DTYPE,
    FeatureStateMemory,
    initialize_from_aps_features,
)
from .frame_synchronization import (
    FrameWindow,
    load_timestamp_origins,
    make_frame_windows,
)
from .spatial_grid import SpatialGridConfig, SpatialGridIndex

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "event_state_update"
)
EVOLUTION_DTYPE = np.dtype(
    [
        ("event_index", np.int64),
        ("t_event", np.int64),
        ("t_common_us", np.int64),
        ("x_event", np.int64),
        ("y_event", np.int64),
        ("p", np.int64),
        ("feature_id", np.int32),
        ("feature_slot", np.int32),
        ("candidate_count", np.uint32),
        ("updated", np.uint8),
        ("status", "U24"),
        ("delta_t_us", np.int64),
        ("predicted_x", np.float64),
        ("predicted_y", np.float64),
        ("residual_x", np.float64),
        ("residual_y", np.float64),
        ("updated_x", np.float64),
        ("updated_y", np.float64),
        ("updated_vx", np.float64),
        ("updated_vy", np.float64),
        ("position_update_magnitude", np.float64),
        ("velocity_update_magnitude", np.float64),
        ("confidence_before", np.float32),
        ("confidence_after", np.float32),
        ("update_latency_us", np.float64),
    ]
)


def _unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return value


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _int64_offset(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("event_timestamp_offset_us must be an integer")
    result = int(value)
    if not np.iinfo(np.int64).min <= result <= np.iinfo(np.int64).max:
        raise OverflowError("event_timestamp_offset_us is outside int64 range")
    return result


def run_asynchronous_updates(
    events: np.ndarray,
    memory: FeatureStateMemory,
    radius_px: float,
    time_window_us: float,
    *,
    alpha: float = 0.5,
    beta: float = 0.1,
    minimum_delta_t_us: float = 1.0,
    confidence_penalty: float = 0.05,
    event_timestamp_offset_us: int = 0,
    spatial_grid: SpatialGridConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Associate and update each event online in stable chronological order.

    Matched events use alpha-beta position and velocity corrections. At zero
    elapsed time the position update proceeds, but velocity remains unchanged;
    for positive intervals the velocity denominator is at least
    ``minimum_delta_t_us``. Confidence is multiplied by
    ``1 - penalty * min(d2/R2, 1)``. Invalid slots are never candidates.
    Numerically unrepresentable updates are rejected without partially mutating
    the state. This function does not use Kalman filtering or image patches.
    """
    radius = _positive_finite(radius_px, "radius_px")
    time_window = _positive_finite(time_window_us, "time_window_us")
    if not np.isfinite(radius * radius):
        raise ValueError("radius_px is too large to square safely")
    alpha_value = _unit_interval(alpha, "alpha")
    beta_value = _unit_interval(beta, "beta")
    min_delta = _positive_finite(minimum_delta_t_us, "minimum_delta_t_us")
    confidence_decay = _unit_interval(confidence_penalty, "confidence_penalty")
    offset = _int64_offset(event_timestamp_offset_us)

    _required_fields(events)
    timestamps = _int64_field_values(events["t"], "t")
    event_xs = _int64_field_values(events["x"], "x")
    event_ys = _int64_field_values(events["y"], "y")
    polarities = _int64_field_values(events["p"], "p")
    if len(events):
        if offset > 0 and int(timestamps.max()) > np.iinfo(np.int64).max - offset:
            raise OverflowError("common event timestamp is outside int64 range")
        if offset < 0 and int(timestamps.min()) < np.iinfo(np.int64).min - offset:
            raise OverflowError("common event timestamp is outside int64 range")
        common_times = timestamps + offset
        processing_order = np.argsort(common_times, kind="stable")
    else:
        common_times = np.empty(0, dtype=np.int64)
        processing_order = np.empty(0, dtype=np.int64)

    rows = np.zeros(len(events), dtype=EVOLUTION_DTYPE)
    for field_name in EVOLUTION_DTYPE.names:
        field_dtype = EVOLUTION_DTYPE.fields[field_name][0]
        if np.issubdtype(field_dtype, np.floating):
            rows[field_name] = np.nan
    event_counts = np.zeros(memory.slot_count, dtype=np.int64)
    initial_active_count = len(memory.get_active())
    radius_squared = radius * radius
    float32_limit = float(np.finfo(np.float32).max)
    grid_index = (
        SpatialGridIndex(memory.get_active(), spatial_grid, radius, time_window)
        if spatial_grid is not None
        else None
    )
    ambiguous_events = 0
    successful_updates = 0
    rejected_events = 0
    candidate_features_total = 0
    distance_calculations_total = 0

    for row_index, event_index in enumerate(processing_order):
        event_started_ns = time.perf_counter_ns()
        t_event = int(common_times[event_index])
        event_x = int(event_xs[event_index])
        event_y = int(event_ys[event_index])
        polarity = int(polarities[event_index])
        active = memory.get_active()
        if len(active) and (
            not np.isfinite(active["x"]).all()
            or not np.isfinite(active["y"]).all()
            or not np.isfinite(active["vx"]).all()
            or not np.isfinite(active["vy"]).all()
        ):
            raise ValueError("active FeatureState positions and velocities must be finite")
        if spatial_grid is None:
            (
                candidate_count,
                active_index,
                distance_squared,
                distance_calculations,
            ) = _select_candidate(
                active,
                t_event,
                event_x,
                event_y,
                radius_squared,
                time_window,
            )
            candidate_features = len(active)
        else:
            (
                candidate_count,
                active_index,
                distance_squared,
                candidate_features,
                distance_calculations,
            ) = grid_index.select(
                active,
                t_event,
                event_x,
                event_y,
                radius_squared,
                time_window,
            )
        candidate_features_total += candidate_features
        distance_calculations_total += distance_calculations
        if candidate_count > 1:
            ambiguous_events += 1
        row = rows[row_index]
        row["event_index"] = int(event_index)
        row["t_event"] = int(timestamps[event_index])
        row["t_common_us"] = t_event
        row["x_event"] = event_x
        row["y_event"] = event_y
        row["p"] = polarity
        row["candidate_count"] = candidate_count

        if active_index < 0:
            row["feature_id"] = -1
            row["feature_slot"] = -1
            row["status"] = "unmatched"
            row["update_latency_us"] = (
                time.perf_counter_ns() - event_started_ns
            ) / 1000.0
            rejected_events += 1
            continue

        state = active[active_index]
        feature_id = int(state["id"])
        slot = int(np.flatnonzero(memory.storage["id"] == feature_id)[0])
        state_before = memory.storage[slot].copy()
        delta_t = t_event - int(state_before["timestamp"])
        if delta_t < 0:
            raise RuntimeError("chronological processing produced a backward state update")

        predicted_x = float(state_before["x"]) + float(state_before["vx"]) * delta_t
        predicted_y = float(state_before["y"]) + float(state_before["vy"]) * delta_t
        residual_x = event_x - predicted_x
        residual_y = event_y - predicted_y
        position_correction_x = alpha_value * residual_x
        position_correction_y = alpha_value * residual_y
        updated_x = predicted_x + position_correction_x
        updated_y = predicted_y + position_correction_y

        if delta_t == 0:
            velocity_correction_x = 0.0
            velocity_correction_y = 0.0
            updated_vx = float(state_before["vx"])
            updated_vy = float(state_before["vy"])
        else:
            velocity_denominator = max(float(delta_t), min_delta)
            velocity_correction_x = beta_value / velocity_denominator * residual_x
            velocity_correction_y = beta_value / velocity_denominator * residual_y
            updated_vx = float(state_before["vx"]) + velocity_correction_x
            updated_vy = float(state_before["vy"]) + velocity_correction_y

        residual_ratio = min(1.0, float(distance_squared) / radius_squared)
        confidence_before = float(state_before["confidence"])
        confidence_after = confidence_before * (
            1.0 - confidence_decay * residual_ratio
        )
        proposed = (
            updated_x,
            updated_y,
            updated_vx,
            updated_vy,
            confidence_after,
        )
        if not all(np.isfinite(value) and abs(value) <= float32_limit for value in proposed):
            row["feature_id"] = feature_id
            row["feature_slot"] = slot
            row["delta_t_us"] = delta_t
            row["predicted_x"] = predicted_x
            row["predicted_y"] = predicted_y
            row["residual_x"] = residual_x
            row["residual_y"] = residual_y
            row["status"] = "numerical_rejection"
            row["update_latency_us"] = (
                time.perf_counter_ns() - event_started_ns
            ) / 1000.0
            rejected_events += 1
            continue

        memory.update(
            slot,
            x=updated_x,
            y=updated_y,
            timestamp=t_event,
            vx=updated_vx,
            vy=updated_vy,
            confidence=confidence_after,
        )
        if grid_index is not None:
            grid_index.update(active_index, memory.storage[slot])
        row["feature_id"] = feature_id
        row["feature_slot"] = slot
        row["updated"] = 1
        row["status"] = "updated"
        row["delta_t_us"] = delta_t
        row["predicted_x"] = predicted_x
        row["predicted_y"] = predicted_y
        row["residual_x"] = residual_x
        row["residual_y"] = residual_y
        row["updated_x"] = updated_x
        row["updated_y"] = updated_y
        row["updated_vx"] = updated_vx
        row["updated_vy"] = updated_vy
        row["position_update_magnitude"] = float(
            np.hypot(position_correction_x, position_correction_y)
        )
        row["velocity_update_magnitude"] = float(
            np.hypot(velocity_correction_x, velocity_correction_y)
        )
        row["confidence_before"] = confidence_before
        row["confidence_after"] = confidence_after
        row["update_latency_us"] = (
            time.perf_counter_ns() - event_started_ns
        ) / 1000.0
        event_counts[slot] += 1
        successful_updates += 1

    per_feature = np.zeros(memory.slot_count, dtype=EVENTS_PER_FEATURE_DTYPE)
    storage = memory.storage
    for slot, state in enumerate(storage):
        if state["valid"]:
            per_feature[slot] = (int(state["id"]), slot, int(event_counts[slot]))
        else:
            per_feature[slot] = (-1, slot, 0)

    processed = len(events)
    diagnostics: dict[str, int | float] = {
        "events_processed": processed,
        "successful_updates": successful_updates,
        "events_rejected": rejected_events,
        "events_ambiguous": ambiguous_events,
        "association_ratio": (
            float(successful_updates / processed) if processed else 0.0
        ),
        "active_features_initial": initial_active_count,
        "active_features_final": int(np.count_nonzero(storage["valid"])),
        "candidate_features_total": candidate_features_total,
        "candidate_features_per_event": (
            candidate_features_total / processed if processed else 0.0
        ),
        "distance_calculations_total": distance_calculations_total,
        "distance_calculations_per_event": (
            distance_calculations_total / processed if processed else 0.0
        ),
        "features_with_updates": int(np.count_nonzero(event_counts)),
        "mean_events_per_updated_feature": (
            float(np.mean(event_counts[event_counts > 0]))
            if np.any(event_counts > 0)
            else 0.0
        ),
        "max_events_per_feature": int(np.max(event_counts, initial=0)),
        "mean_position_update_magnitude": (
            float(np.mean(rows["position_update_magnitude"][rows["updated"] != 0]))
            if successful_updates
            else 0.0
        ),
        "max_position_update_magnitude": (
            float(np.max(rows["position_update_magnitude"][rows["updated"] != 0]))
            if successful_updates
            else 0.0
        ),
        "mean_velocity_update_magnitude": (
            float(np.mean(rows["velocity_update_magnitude"][rows["updated"] != 0]))
            if successful_updates
            else 0.0
        ),
        "max_velocity_update_magnitude": (
            float(np.max(rows["velocity_update_magnitude"][rows["updated"] != 0]))
            if successful_updates
            else 0.0
        ),
    }
    return rows, per_feature, diagnostics


def save_update_outputs(
    first_frame: np.ndarray,
    next_frame: np.ndarray,
    initial_states: np.ndarray,
    final_states: np.ndarray,
    evolution: np.ndarray,
    events_per_feature: np.ndarray,
    diagnostics: Mapping[str, int | float],
    frame_window: FrameWindow,
    configuration: Mapping[str, Any],
    output_dir: Path,
    trajectory_count: int = 20,
) -> dict[str, Path]:
    """Save state evolution and produce event/prediction/update trajectory plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "davis_feature_state_evolution.npz"
    np.savez_compressed(
        archive_path,
        evolution=evolution,
        events_per_feature=events_per_feature,
        initial_states=initial_states,
        final_states=final_states,
    )

    diagnostics_path = output_dir / "davis_feature_state_update_diagnostics.json"
    report_data = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": dict(configuration),
        "interval_index": frame_window.index,
        "interval_common_time_us": list(frame_window.interval_us),
        "timestamp_unit": "microseconds",
        "velocity_unit": "pixels per microsecond",
        "confidence_rule": (
            "confidence_after = confidence_before * "
            "(1 - confidence_penalty * min(distance_squared / R_squared, 1))"
        ),
        "zero_delta_t_rule": (
            "Apply the alpha position correction; keep velocity unchanged to "
            "avoid division by zero."
        ),
        "small_delta_t_rule": (
            "For positive dt, divide the beta residual correction by "
            "max(dt, minimum_delta_t_us)."
        ),
        "valid_state_rule": (
            "Only valid slots participate in association; inactive slots remain "
            "inactive and are not modified."
        ),
        "numerical_rejection_rule": (
            "Reject an associated update atomically if any proposed float state "
            "is non-finite or outside float32 range."
        ),
        **dict(diagnostics),
    }
    diagnostics_path.write_text(
        json.dumps(report_data, indent=2) + "\n", encoding="utf-8"
    )

    csv_path = output_dir / "davis_feature_state_evolution.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(EVOLUTION_DTYPE.names)
        for row in evolution:
            writer.writerow(tuple(row[name] for name in EVOLUTION_DTYPE.names))

    initial_by_id = {
        int(state["id"]): state for state in initial_states if state["valid"]
    }
    final_by_id = {
        int(state["id"]): state for state in final_states if state["valid"]
    }
    updates = evolution[evolution["updated"] != 0]
    counts_by_id = {
        int(row["feature_id"]): int(row["event_count"])
        for row in events_per_feature
        if int(row["feature_id"]) >= 0
    }
    selected_ids = [
        feature_id
        for feature_id, _ in sorted(
            counts_by_id.items(), key=lambda item: (-item[1], item[0])
        )[:trajectory_count]
        if counts_by_id[feature_id] > 0
    ]
    selected_updates = updates[np.isin(updates["feature_id"], selected_ids)]

    figure_path = output_dir / "davis_feature_state_trajectories.png"
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, frame, title in (
        (axes[0], first_frame, f"APS frame {frame_window.index}"),
        (axes[1], next_frame, f"APS frame {frame_window.index + 1}"),
    ):
        ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.set_axis_off()

    initial_active = initial_states[initial_states["valid"] != 0]
    if len(initial_active):
        axes[0].scatter(
            initial_active["x"],
            initial_active["y"],
            s=14,
            facecolors="none",
            edgecolors="lime",
            linewidths=0.7,
            label="initial APS feature",
        )
        axes[1].scatter(
            initial_active["x"],
            initial_active["y"],
            s=12,
            facecolors="none",
            edgecolors="lime",
            linewidths=0.6,
            label="initial APS feature",
        )

    if len(evolution):
        positive = evolution["p"] > 0
        for positive_polarity, color in ((False, "tab:blue"), (True, "tab:orange")):
            selection = positive == positive_polarity
            axes[0].scatter(
                evolution["x_event"][selection],
                evolution["y_event"][selection],
                s=4,
                c=color,
                alpha=0.35,
                marker=".",
                label=f"event p={'positive' if positive_polarity else 'non-positive'}",
            )

    if len(selected_updates):
        axes[0].scatter(
            selected_updates["predicted_x"],
            selected_updates["predicted_y"],
            marker="^",
            s=25,
            facecolors="none",
            edgecolors="cyan",
            linewidths=0.8,
            label="predicted location",
        )
        axes[0].scatter(
            selected_updates["updated_x"],
            selected_updates["updated_y"],
            marker="s",
            s=18,
            c="magenta",
            alpha=0.75,
            label="updated location",
        )
        for feature_id in selected_ids:
            track = selected_updates[selected_updates["feature_id"] == feature_id]
            initial = initial_by_id.get(feature_id)
            if initial is None or not len(track):
                continue
            x_path = np.concatenate(([float(initial["x"])], track["updated_x"]))
            y_path = np.concatenate(([float(initial["y"])], track["updated_y"]))
            axes[0].plot(x_path, y_path, color="white", linewidth=0.7, alpha=0.75)
            final = final_by_id.get(feature_id)
            if final is not None:
                axes[1].plot(
                    [float(initial["x"]), float(final["x"])],
                    [float(initial["y"]), float(final["y"])],
                    color="magenta",
                    linewidth=1.0,
                    alpha=0.8,
                )
                axes[1].scatter(
                    [float(final["x"])],
                    [float(final["y"])],
                    marker="s",
                    s=24,
                    c="magenta",
                    label="final updated location" if feature_id == selected_ids[0] else None,
                )

    axes[0].legend(loc="upper right", fontsize=7)
    axes[1].legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"Stage 4B state evolution; cyan triangles = pre-update predictions, "
        f"magenta squares = updated positions; {len(selected_ids)} trajectories"
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    report_path = output_dir / "davis_feature_state_update_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Stage 4B — Asynchronous FeatureState update",
                "",
                f"- APS interval: {frame_window.index}, "
                f"[{frame_window.interval_us[0]}, {frame_window.interval_us[1]}) μs",
                f"- Events processed: {diagnostics['events_processed']}",
                f"- Successful updates: {diagnostics['successful_updates']}",
                f"- Rejected events: {diagnostics['events_rejected']}",
                f"- Ambiguous events: {diagnostics['events_ambiguous']}",
                f"- Initial active features: {diagnostics['active_features_initial']}",
                f"- Final active features: {diagnostics['active_features_final']}",
                f"- Features with at least one update: {diagnostics['features_with_updates']}",
                f"- Mean / maximum events per updated feature: "
                f"{diagnostics['mean_events_per_updated_feature']:.3f} / "
                f"{diagnostics['max_events_per_feature']}",
                f"- Mean position update magnitude: {diagnostics['mean_position_update_magnitude']:.6g} px",
                f"- Maximum position update magnitude: {diagnostics['max_position_update_magnitude']:.6g} px",
                f"- Mean velocity update magnitude: {diagnostics['mean_velocity_update_magnitude']:.6g} px/μs",
                f"- Maximum velocity update magnitude: {diagnostics['max_velocity_update_magnitude']:.6g} px/μs",
                f"- Alpha / beta: {configuration['alpha']} / {configuration['beta']}",
                f"- Confidence penalty: {configuration['confidence_penalty']}",
                "- Zero Δt: position correction is applied; velocity is unchanged.",
                "- Positive small Δt: velocity denominator is clamped to the configured minimum.",
                "- Only valid states are considered; invalid slots remain unchanged.",
                "- Numerically unrepresentable updates are rejected atomically.",
                "- Feature states contain no image patches; no Kalman filter is used.",
                "",
                "Event pixels, pre-update predictions, and post-update positions are separately marked in the trajectory visualization.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "evolution": archive_path,
        "diagnostics": diagnostics_path,
        "csv": csv_path,
        "visualization": figure_path,
        "report": report_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run asynchronous alpha-beta feature-state updates on one DAVIS interval."
    )
    parser.add_argument("--recording", default="shapes_6dof")
    parser.add_argument("--interval-index", type=int, default=678)
    parser.add_argument("--feature-count", type=int, default=500)
    parser.add_argument("--slots", type=int, default=500)
    parser.add_argument("--radius-px", type=float, default=8.0)
    parser.add_argument("--time-window-us", type=float, default=50000.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--minimum-delta-t-us", type=float, default=1.0)
    parser.add_argument("--confidence-penalty", type=float, default=0.05)
    parser.add_argument("--quality-level", type=float, default=0.01)
    parser.add_argument("--min-feature-distance", type=float, default=5.0)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--bucket-columns", type=int, default=8)
    parser.add_argument("--bucket-rows", type=int, default=6)
    parser.add_argument("--trajectory-count", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.interval_index < 0:
        parser.error("--interval-index must be non-negative")
    if args.feature_count <= 0 or args.slots <= 0:
        parser.error("--feature-count and --slots must be positive")
    if args.feature_count > args.slots:
        parser.error("--feature-count cannot exceed --slots")
    if args.trajectory_count <= 0:
        parser.error("--trajectory-count must be positive")

    dataset_config = DavisConfig(recording=args.recording)
    recording = load_davis(dataset_config)
    origins_s, _ = load_timestamp_origins(recording, dataset_config)
    windows = make_frame_windows(recording, origins_s)
    if args.interval_index >= len(windows):
        parser.error(f"--interval-index must be within [0, {len(windows) - 1}]")
    window = windows.get_window(args.interval_index)
    features = extract_shi_tomasi_features(
        window.frame_k,
        ShiTomasiConfig(
            max_features=args.feature_count,
            quality_level=args.quality_level,
            min_feature_distance=args.min_feature_distance,
            block_size=args.block_size,
            bucket_columns=args.bucket_columns,
            bucket_rows=args.bucket_rows,
        ),
        frame_index=args.interval_index,
        frame_timestamp_us=window.frame_k_common_time_us,
    ).features
    memory = initialize_from_aps_features(
        features, window.frame_k_common_time_us, args.slots
    )
    initial_states = memory.storage.copy()
    evolution, events_per_feature, diagnostics = run_asynchronous_updates(
        window.events_k,
        memory,
        args.radius_px,
        args.time_window_us,
        alpha=args.alpha,
        beta=args.beta,
        minimum_delta_t_us=args.minimum_delta_t_us,
        confidence_penalty=args.confidence_penalty,
        event_timestamp_offset_us=window.event_timestamp_offset_us,
    )
    configuration = {
        "recording": args.recording,
        "interval_index": args.interval_index,
        "feature_count_requested": args.feature_count,
        "slot_count": args.slots,
        "radius_px": args.radius_px,
        "time_window_us": args.time_window_us,
        "alpha": args.alpha,
        "beta": args.beta,
        "minimum_delta_t_us": args.minimum_delta_t_us,
        "confidence_penalty": args.confidence_penalty,
        "quality_level": args.quality_level,
        "minimum_feature_distance_px": args.min_feature_distance,
        "block_size_px": args.block_size,
        "bucket_grid": [args.bucket_columns, args.bucket_rows],
    }
    outputs = save_update_outputs(
        window.frame_k,
        window.frame_k_plus_1,
        initial_states,
        memory.storage,
        evolution,
        events_per_feature,
        diagnostics,
        window,
        configuration,
        args.output_dir,
        trajectory_count=args.trajectory_count,
    )
    print(json.dumps(diagnostics, indent=2))
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
