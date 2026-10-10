"""Stage 8 exact spatial-grid association and brute-force comparison."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .aps_features import ShiTomasiConfig, extract_shi_tomasi_features
from .davis_inspection import DavisConfig, load_davis
from .event_association import (
    ASSOCIATION_DTYPE,
    EVENTS_PER_FEATURE_DTYPE,
    AssociationResult,
    _int64_field_values,
    _positive_finite,
    _required_fields,
    associate_events_brute_force,
)
from .feature_state_memory import FeatureStateMemory, initialize_from_aps_features
from .frame_synchronization import load_timestamp_origins, make_frame_windows
from .spatial_grid import SpatialGridConfig, SpatialGridIndex

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "grid_event_association"
)


def associate_events_grid(
    events: np.ndarray,
    memory: FeatureStateMemory,
    radius_px: float,
    time_window_us: float,
    grid: SpatialGridConfig,
    *,
    event_timestamp_offset_us: int = 0,
) -> AssociationResult:
    """Associate events in chronological order using exact grid candidate pruning."""
    started_ns = time.perf_counter_ns()
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
    if len(active) and any(
        not np.isfinite(active[field]).all()
        for field in ("x", "y", "vx", "vy")
    ):
        raise ValueError("active FeatureState positions and velocities must be finite")

    timestamps = _int64_field_values(events["t"], "t")
    event_xs = _int64_field_values(events["x"], "x")
    event_ys = _int64_field_values(events["y"], "y")
    polarities = _int64_field_values(events["p"], "p")
    offset = int(event_timestamp_offset_us)
    if len(events):
        if offset > 0 and int(timestamps.max()) > np.iinfo(np.int64).max - offset:
            raise OverflowError("common event timestamp is outside int64 range")
        if offset < 0 and int(timestamps.min()) < np.iinfo(np.int64).min - offset:
            raise OverflowError("common event timestamp is outside int64 range")
        common_times = timestamps + offset
        order = np.argsort(common_times, kind="stable")
    else:
        common_times = np.empty(0, dtype=np.int64)
        order = np.empty(0, dtype=np.int64)

    slots_by_id = {
        int(memory.storage["id"][slot]): int(slot)
        for slot in np.flatnonzero(memory.storage["valid"] != 0)
    }
    event_counts = np.zeros(len(active), dtype=np.int64)
    assignments = np.zeros(len(events), dtype=ASSOCIATION_DTYPE)
    radius_squared = radius * radius
    associated_count = 0
    ambiguous_count = 0
    distance_calculations = 0
    candidate_features_total = 0
    grid_index = SpatialGridIndex(active, grid, radius, time_window)
    for output_index, event_index in enumerate(order):
        (
            count,
            best_index,
            best_distance,
            candidate_features,
            distance_checks,
        ) = grid_index.select(
            active,
            int(common_times[event_index]),
            int(event_xs[event_index]),
            int(event_ys[event_index]),
            radius_squared,
            time_window,
        )
        distance_calculations += distance_checks
        candidate_features_total += candidate_features
        is_associated = count > 0
        if count > 1:
            ambiguous_count += 1
        feature_id = -1
        feature_slot = -1
        distance = np.nan
        if is_associated:
            state = active[best_index]
            feature_id = int(state["id"])
            feature_slot = slots_by_id[feature_id]
            distance = float(best_distance)
            event_counts[best_index] += 1
            associated_count += 1

        event = events[event_index]
        assignments[output_index] = (
            int(event_index),
            int(event["t"]),
            int(common_times[event_index]),
            int(event_xs[event_index]),
            int(event_ys[event_index]),
            int(polarities[event_index]),
            feature_id,
            feature_slot,
            distance,
            count,
            int(is_associated),
        )

    per_feature = np.zeros(len(active), dtype=EVENTS_PER_FEATURE_DTYPE)
    for index, state in enumerate(active):
        feature_id = int(state["id"])
        per_feature[index] = (
            feature_id,
            slots_by_id[feature_id],
            int(event_counts[index]),
        )

    processed = len(events)
    elapsed_seconds = (time.perf_counter_ns() - started_ns) / 1e9
    diagnostics: dict[str, int | float] = {
        "events_processed": processed,
        "events_associated": associated_count,
        "association_ratio": associated_count / processed if processed else 0.0,
        "events_rejected": processed - associated_count,
        "events_ambiguous": ambiguous_count,
        "active_features": len(active),
        "candidate_features_total": candidate_features_total,
        "candidate_features_per_event": (
            candidate_features_total / processed if processed else 0.0
        ),
        "distance_calculations_total": distance_calculations,
        "distance_calculations_per_event": (
            distance_calculations / processed if processed else 0.0
        ),
        "total_runtime_seconds": elapsed_seconds,
        "events_processed_per_second": (
            processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
    }
    return AssociationResult(assignments, per_feature, diagnostics)


def compare_association_results(
    brute: AssociationResult,
    grid: AssociationResult,
    *,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compare decisions exactly and selected squared distances within tolerance."""
    for name in ("absolute_tolerance", "relative_tolerance"):
        value = float(locals()[name])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if brute.assignments.dtype != grid.assignments.dtype:
        raise ValueError("association assignment dtypes differ")
    if len(brute.assignments) != len(grid.assignments):
        raise ValueError("association result lengths differ")

    exact_fields = (
        "event_index",
        "t_event",
        "t_common_us",
        "x_event",
        "y_event",
        "p",
        "feature_id",
        "feature_slot",
        "candidate_count",
        "associated",
    )
    differences: dict[str, int] = {
        name: int(np.count_nonzero(brute.assignments[name] != grid.assignments[name]))
        for name in exact_fields
    }
    distance_equal = np.isclose(
        brute.assignments["distance_squared"],
        grid.assignments["distance_squared"],
        atol=absolute_tolerance,
        rtol=relative_tolerance,
        equal_nan=True,
    )
    differences["distance_squared"] = int(np.count_nonzero(~distance_equal))
    per_feature_equal = (
        brute.events_per_feature.dtype == grid.events_per_feature.dtype
        and brute.events_per_feature.shape == grid.events_per_feature.shape
        and np.array_equal(brute.events_per_feature, grid.events_per_feature)
    )
    differences["events_per_feature"] = 0 if per_feature_equal else 1
    return {
        "equivalent": not any(differences.values()),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "differences": differences,
    }


def _save_comparison(
    brute: AssociationResult,
    grid_result: AssociationResult,
    comparison: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "davis_grid_association_comparison.json"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "comparison": comparison,
        "performance": {
            "brute_force": dict(brute.diagnostics),
            "grid": dict(grid_result.diagnostics),
        },
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    csv_path = output_dir / "davis_grid_association_comparison.csv"
    metrics = [
        ("Candidate features/event", brute.diagnostics["active_features"], grid_result.diagnostics["candidate_features_per_event"]),
        ("Distance calculations/event", brute.diagnostics["distance_calculations_per_event"], grid_result.diagnostics["distance_calculations_per_event"]),
        ("Events processed/sec", brute.diagnostics["events_processed_per_second"], grid_result.diagnostics["events_processed_per_second"]),
        ("Total runtime (s)", brute.diagnostics["total_runtime_seconds"], grid_result.diagnostics["total_runtime_seconds"]),
        ("Association differences", 0, sum(comparison["differences"].values())),
        ("Rejected events", brute.diagnostics["events_rejected"], grid_result.diagnostics["events_rejected"]),
        ("Ambiguous associations", brute.diagnostics["events_ambiguous"], grid_result.diagnostics["events_ambiguous"]),
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Metric", "Brute Force", "Grid"))
        for row in metrics:
            writer.writerow(row)

    md_path = output_dir / "davis_grid_association_comparison.md"
    md_path.write_text(
        "\n".join(
            [
                "# Stage 8 — Spatial grid association",
                "",
                f"- Result parity: {'PASS' if comparison['equivalent'] else 'FAIL'}",
                f"- Exact decision fields: {comparison['differences']}",
                f"- Distance tolerance: abs={comparison['absolute_tolerance']}, "
                f"rel={comparison['relative_tolerance']}",
                f"- Grid cells: {config['cell_width']} x {config['cell_height']} px",
                "- The persistent grid indexes stored feature positions. It searches "
                "cells within a conservative velocity-expanded neighborhood, then "
                "applies the exact per-feature prediction, temporal gate, and distance gate.",
                "- Both paths use strict d2 < R2 and abs(dt) < T, and choose minimum "
                "squared distance with lower feature ID as the tie-break.",
                "- No square root is used; the strict squared-distance gate is "
                "unchanged.",
                "",
                "| Metric | Brute Force | Grid |",
                "|--------|-------------|------|",
                f"| Candidate features/event | {brute.diagnostics['active_features']:.3f} | {grid_result.diagnostics['candidate_features_per_event']:.3f} |",
                f"| Distance calculations/event | {brute.diagnostics['distance_calculations_per_event']:.3f} | {grid_result.diagnostics['distance_calculations_per_event']:.3f} |",
                f"| Events processed/sec | {brute.diagnostics.get('events_processed_per_second', 0):.3f} | {grid_result.diagnostics['events_processed_per_second']:.3f} |",
                f"| Total runtime (s) | {brute.diagnostics.get('total_runtime_seconds', 0):.6f} | {grid_result.diagnostics['total_runtime_seconds']:.6f} |",
                f"| Association differences | 0 | {sum(comparison['differences'].values())} |",
                f"| Rejected events | {brute.diagnostics['events_rejected']} | {grid_result.diagnostics['events_rejected']} |",
                f"| Ambiguous associations | {brute.diagnostics['events_ambiguous']} | {grid_result.diagnostics['events_ambiguous']} |",
                "",
                "The brute-force implementation remains the correctness reference. "
                "Performance depends on grid dimensions and feature motion. "
                "Runtime figures are software measurements, not hardware estimates.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"json": json_path, "csv": csv_path, "report": md_path}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare exact grid and brute-force DAVIS event association."
    )
    parser.add_argument("--recording", default="shapes_6dof")
    parser.add_argument("--interval-index", type=int, default=678)
    parser.add_argument("--feature-count", type=int, default=500)
    parser.add_argument("--slots", type=int, default=500)
    parser.add_argument("--radius-px", type=float, default=8.0)
    parser.add_argument("--time-window-us", type=float, default=50000.0)
    parser.add_argument("--cell-width", type=int, default=16)
    parser.add_argument("--cell-height", type=int, default=16)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-9)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.interval_index < 0 or args.feature_count <= 0 or args.slots < args.feature_count:
        parser.error("interval must be non-negative; slots must cover positive feature-count")

    dataset_config = DavisConfig(recording=args.recording)
    recording = load_davis(dataset_config)
    origins_s, _ = load_timestamp_origins(recording, dataset_config)
    windows = make_frame_windows(recording, origins_s)
    if args.interval_index >= len(windows):
        parser.error(f"--interval-index must be within [0, {len(windows) - 1}]")
    window = windows.get_window(args.interval_index)
    height, width = window.frame_k.shape[:2]
    grid = SpatialGridConfig(width, height, args.cell_width, args.cell_height)
    features = extract_shi_tomasi_features(
        window.frame_k,
        ShiTomasiConfig(max_features=args.feature_count),
        frame_index=args.interval_index,
        frame_timestamp_us=window.frame_k_common_time_us,
    ).features
    memory = initialize_from_aps_features(features, window.frame_k_common_time_us, args.slots)

    brute_started = time.perf_counter_ns()
    brute = associate_events_brute_force(
        window.events_k,
        memory,
        args.radius_px,
        args.time_window_us,
        event_timestamp_offset_us=window.event_timestamp_offset_us,
    )
    brute_elapsed = (time.perf_counter_ns() - brute_started) / 1e9
    brute_diagnostics = dict(brute.diagnostics)
    brute_diagnostics.update(
        total_runtime_seconds=brute_elapsed,
        events_processed_per_second=(
            len(window.events_k) / brute_elapsed if brute_elapsed > 0 else 0.0
        ),
    )
    brute = AssociationResult(brute.assignments, brute.events_per_feature, brute_diagnostics)
    grid_started = time.perf_counter_ns()
    grid_result = associate_events_grid(
        window.events_k,
        memory,
        args.radius_px,
        args.time_window_us,
        grid,
        event_timestamp_offset_us=window.event_timestamp_offset_us,
    )
    grid_elapsed = (time.perf_counter_ns() - grid_started) / 1e9
    grid_diagnostics = dict(grid_result.diagnostics)
    grid_diagnostics.update(
        total_runtime_seconds=grid_elapsed,
        events_processed_per_second=(
            len(window.events_k) / grid_elapsed if grid_elapsed > 0 else 0.0
        ),
    )
    grid_result = AssociationResult(
        grid_result.assignments, grid_result.events_per_feature, grid_diagnostics
    )
    comparison = compare_association_results(
        brute,
        grid_result,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    config = {
        "recording": args.recording,
        "interval_index": args.interval_index,
        "events": len(window.events_k),
        "active_features": len(memory.get_active()),
        "radius_px": args.radius_px,
        "time_window_us": args.time_window_us,
        "sensor_width": width,
        "sensor_height": height,
        "cell_width": args.cell_width,
        "cell_height": args.cell_height,
        "timestamp_unit": "microseconds",
    }
    outputs = _save_comparison(brute, grid_result, comparison, config, args.output_dir)
    print(json.dumps({"comparison": comparison, "brute_force": brute.diagnostics, "grid": grid_result.diagnostics}, indent=2))
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0 if comparison["equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
