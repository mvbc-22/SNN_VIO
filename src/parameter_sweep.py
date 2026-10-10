"""Stage 9 one-factor-at-a-time sweep of tracker operating parameters."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from .davis_inspection import DavisConfig, load_davis
from .feature_state_memory import FEATURE_STATE_DTYPE
from .frame_synchronization import load_timestamp_origins, make_frame_windows
from .three_way_comparison import (
    ComparisonConfig,
    SYSTEM_LABELS,
    SYSTEMS,
    compare_three_systems,
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "parameter_sweep"
)

SWEEP_AXES: tuple[tuple[str, tuple[int | float, ...]], ...] = (
    ("feature_count", (100, 300, 500, 800)),
    ("grid_cell_size_px", (8, 16, 24, 32)),
    ("association_radius_px", (4.0, 8.0, 12.0, 16.0)),
    ("association_time_window_us", (10_000.0, 25_000.0, 50_000.0, 100_000.0)),
    ("alpha", (0.25, 0.5, 0.75)),
    ("beta", (0.025, 0.1, 0.25)),
)

CSV_FIELDS = (
    "configuration_id",
    "sweep_parameter",
    "sweep_value",
    "system",
    "interval_start",
    "interval_count",
    "feature_count_requested",
    "initial_features_detected",
    "grid_cell_size_px",
    "association_radius_px",
    "association_time_window_us",
    "alpha",
    "beta",
    "epe_px",
    "track_lifetime_mean_us",
    "mean_survival_fraction",
    "final_survival_fraction",
    "failure_rate",
    "events_per_useful_update",
    "events_per_successful_state_update",
    "candidate_features_per_event",
    "distance_calculations_per_event",
    "processing_time_ms",
    "update_latency_mean_us",
    "state_updates",
    "events_processed",
    "state_memory_bytes",
    "logical_grid_cells",
    "logical_grid_index_bytes",
)


def _csv_values(text: str, conversion: type) -> tuple[Any, ...]:
    try:
        values = tuple(conversion(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid comma-separated values: {text}") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def build_sweep_configs(
    base: ComparisonConfig,
    axes: Iterable[tuple[str, tuple[int | float, ...]]] = SWEEP_AXES,
) -> list[tuple[str, str, int | float, ComparisonConfig]]:
    """Create a reproducible one-factor-at-a-time set including its baseline."""
    axes = tuple(axes)
    base.validate()
    configs = [("cfg_000", "baseline", "baseline", replace(base, use_spatial_grid=True))]
    seen = {tuple(getattr(base, name) for name, _ in axes)}
    index = 1
    for name, values in axes:
        for value in values:
            changes: dict[str, Any] = {name: value}
            if name == "feature_count":
                changes["slot_count"] = value
            candidate = replace(base, **changes, use_spatial_grid=True)
            candidate.validate()
            key = tuple(getattr(candidate, axis_name) for axis_name, _ in axes)
            if key in seen:
                continue
            seen.add(key)
            configs.append((f"cfg_{index:03d}", name, value, candidate))
            index += 1
    return configs


def _configuration_rows(
    configs: list[tuple[str, str, int | float, ComparisonConfig]],
    windows: list[Any],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    height, width = windows[0].frame_k.shape[:2]
    for config_id, sweep_parameter, sweep_value, config in configs:
        result = compare_three_systems(windows, config)
        if not all(result.checks.values()):
            failed = [key for key, value in result.checks.items() if not value]
            raise RuntimeError(
                f"{config_id} failed Stage 7 checks: {', '.join(failed)}"
            )
        detected_count = len(result.initial_features)
        cells_x = math.ceil(width / config.grid_cell_size_px)
        cells_y = math.ceil(height / config.grid_cell_size_px)
        logical_grid_cells = cells_x * cells_y
        state_memory_bytes = config.slot_count * FEATURE_STATE_DTYPE.itemsize
        logical_grid_index_bytes = (
            logical_grid_cells * 4 + config.slot_count * 4
        )
        for system in SYSTEMS:
            diagnostics = result.diagnostics[system]
            survival = result.survival[system]
            initial = max(detected_count, 1)
            event_count = int(diagnostics["events_processed"])
            async_updates = int(diagnostics["asynchronous_state_updates"])
            # These memory figures describe a compact logical layout only; they
            # do not estimate CPython's dict/list/object overhead.
            output_rows.append(
                {
                    "configuration_id": config_id,
                    "sweep_parameter": sweep_parameter,
                    "sweep_value": sweep_value,
                    "system": system,
                    "interval_start": config.interval_start,
                    "interval_count": config.interval_count,
                    "feature_count_requested": config.feature_count,
                    "initial_features_detected": detected_count,
                    "grid_cell_size_px": config.grid_cell_size_px,
                    "association_radius_px": config.association_radius_px,
                    "association_time_window_us": config.association_time_window_us,
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "epe_px": diagnostics["epe_px"],
                    "track_lifetime_mean_us": diagnostics[
                        "track_lifetime_mean_us"
                    ],
                    "mean_survival_fraction": (
                        float(np.mean(survival / initial)) if len(survival) else 0.0
                    ),
                    "final_survival_fraction": (
                        float(survival[-1] / initial) if len(survival) else 0.0
                    ),
                    "failure_rate": diagnostics["failure_rate"],
                    "events_per_useful_update": diagnostics[
                        "events_per_useful_update"
                    ],
                    "events_per_successful_state_update": (
                        event_count / async_updates
                        if event_count and async_updates
                        else None
                    ),
                    "candidate_features_per_event": diagnostics[
                        "candidate_features_per_event"
                    ],
                    "distance_calculations_per_event": diagnostics[
                        "distance_calculations_per_event"
                    ],
                    "processing_time_ms": diagnostics["processing_time_ms"],
                    "update_latency_mean_us": diagnostics["update_latency_mean_us"],
                    "state_updates": diagnostics["state_updates"],
                    "events_processed": event_count,
                    "state_memory_bytes": state_memory_bytes,
                    "logical_grid_cells": logical_grid_cells,
                    "logical_grid_index_bytes": logical_grid_index_bytes,
                }
            )
    return output_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _plot_tradeoffs(
    rows: list[dict[str, Any]], output_dir: Path
) -> dict[str, Path]:
    parameters = (
        ("feature_count", "feature_count_requested", "Requested active features"),
        ("grid_cell_size_px", "grid_cell_size_px", "Grid cell size (px)"),
        ("association_radius_px", "association_radius_px", "Spatial radius R (px)"),
        ("association_time_window_us", "association_time_window_us", "Temporal threshold T (μs)"),
        ("alpha", "alpha", "Alpha"),
        ("beta", "beta", "Beta"),
    )
    metrics = (
        ("epe_px", "EPE (px)"),
        ("track_lifetime_mean_us", "Track lifetime (μs)"),
        ("mean_survival_fraction", "Mean feature survival fraction"),
        ("failure_rate", "Failure rate"),
        ("events_per_successful_state_update", "Events / successful state update"),
        ("candidate_features_per_event", "Candidate features / event"),
        ("processing_time_ms", "Processing time (ms)"),
    )
    paths: dict[str, Path] = {}
    for parameter, parameter_field, parameter_label in parameters:
        subset = [
            row for row in rows
            if row["sweep_parameter"] in ("baseline", parameter)
        ]
        figures, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes_flat = axes.ravel()
        for axis, (metric, metric_label) in zip(axes_flat, metrics):
            for system in ("event_driven", "hybrid"):
                system_rows = [
                    row for row in subset
                    if row["system"] == system
                    and row[metric] is not None
                    and np.isfinite(row[metric])
                ]
                values_by_parameter: dict[float, list[float]] = {}
                for row in system_rows:
                    x = float(row[parameter_field])
                    values_by_parameter.setdefault(x, []).append(float(row[metric]))
                x_values = sorted(values_by_parameter)
                y_values = [
                    float(np.mean(values_by_parameter[value])) for value in x_values
                ]
                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    label=SYSTEM_LABELS[system],
                )
            axis.set_xlabel(parameter_label)
            axis.set_ylabel(metric_label)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        axes_flat[-1].axis("off")
        figures.suptitle(f"Stage 9 tradeoffs by {parameter_label}")
        figures.tight_layout()
        path = output_dir / f"tradeoff_{parameter}.png"
        figures.savefig(path, dpi=150)
        plt.close(figures)
        paths[parameter] = path
    return paths


def _write_report(
    rows: list[dict[str, Any]],
    configs: list[tuple[str, str, int | float, ComparisonConfig]],
    output_dir: Path,
) -> Path:
    event_rows = [row for row in rows if row["system"] == "event_driven"]
    baseline = next(row for row in event_rows if row["sweep_parameter"] == "baseline")
    feature_variants = [
        row for row in event_rows if row["sweep_parameter"] == "feature_count"
    ]
    time_variants = [
        row for row in event_rows
        if row["sweep_parameter"] == "association_time_window_us"
    ]
    cell_variants = [
        row for row in event_rows if row["sweep_parameter"] == "grid_cell_size_px"
    ]
    parameter_findings: list[str] = []
    for parameter, field, label in (
        ("feature_count", "feature_count_requested", "Feature count"),
        ("grid_cell_size_px", "grid_cell_size_px", "Grid-cell size"),
        ("association_radius_px", "association_radius_px", "Radius R"),
        ("association_time_window_us", "association_time_window_us", "Threshold T"),
        ("alpha", "alpha", "Alpha"),
        ("beta", "beta", "Beta"),
    ):
        candidates = [row for row in event_rows if row["sweep_parameter"] == parameter]
        accuracy = [
            row for row in candidates
            if row["epe_px"] is not None and np.isfinite(row["epe_px"])
        ]
        if accuracy:
            best = min(accuracy, key=lambda row: row["epe_px"])
            span = max(row["epe_px"] for row in accuracy) - min(
                row["epe_px"] for row in accuracy
            )
            parameter_findings.append(
                f"- **{label} (accuracy):** measured EPE span "
                f"{span:.3f} px across tested values; lowest EPE was "
                f"{best[field]} ({best['epe_px']:.3f} px)."
            )
        else:
            parameter_findings.append(
                f"- **{label} (accuracy):** no finite EPE matches in tested values."
            )
    pareto_metrics = (
        ("epe_px", False),
        ("track_lifetime_mean_us", True),
        ("mean_survival_fraction", True),
        ("failure_rate", False),
        ("events_per_successful_state_update", False),
        ("candidate_features_per_event", False),
        ("processing_time_ms", False),
        ("state_memory_bytes", False),
        ("logical_grid_index_bytes", False),
    )
    finite_rows = [
        row for row in event_rows
        if all(
            row[name] is not None and np.isfinite(row[name])
            for name, _ in pareto_metrics
        )
    ]
    pareto_rows: list[dict[str, Any]] = []
    for candidate in finite_rows:
        dominated = False
        for other in finite_rows:
            if other is candidate:
                continue
            no_worse = all(
                other[name] >= candidate[name] if maximize
                else other[name] <= candidate[name]
                for name, maximize in pareto_metrics
            )
            strictly_better = any(
                other[name] > candidate[name] if maximize
                else other[name] < candidate[name]
                for name, maximize in pareto_metrics
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto_rows.append(candidate)
    pareto_rows.sort(key=lambda row: row["configuration_id"])
    path = output_dir / "stage9_parameter_sweep_report.md"
    lines = [
        "# Stage 9 — Systematic parameter sweep",
        "",
        f"- Controlled configurations: {len(configs)} (baseline plus one-factor-at-a-time variants).",
        f"- APS segment: intervals {int(baseline['interval_start'])} through "
        f"{int(baseline['interval_start']) + int(baseline['interval_count']) - 1}.",
        "- Every variant changes one parameter from the same baseline; this identifies "
        "local sensitivity, not parameter interactions or a global optimum.",
        "- The event-driven and hybrid trackers use the exact Stage 8 grid association. "
        "Frame-only is retained as a reference and does not use event/grid parameters.",
        "",
        "## Baseline",
        "",
        f"- EPE: {baseline['epe_px']}",
        f"- Mean track lifetime: {baseline['track_lifetime_mean_us']:.1f} μs",
        f"- Mean survival: {baseline['mean_survival_fraction']:.4f}",
        f"- Failure rate: {baseline['failure_rate']:.4f}",
        f"- Events / successful update: {baseline['events_per_successful_state_update']}",
        f"- Candidate features / event: {baseline['candidate_features_per_event']:.3f}",
        f"- Processing time: {baseline['processing_time_ms']:.2f} ms",
        "",
        "## Accuracy sensitivity",
        "",
        *parameter_findings,
        "",
        "## Resource and implementation sensitivity",
        "",
        f"- **Latency-sensitive (measured):** feature-count variants changed event-driven "
        f"processing time from {min(row['processing_time_ms'] for row in feature_variants):.0f} "
        f"to {max(row['processing_time_ms'] for row in feature_variants):.0f} ms; "
        f"T variants changed it from {min(row['processing_time_ms'] for row in time_variants):.0f} "
        f"to {max(row['processing_time_ms'] for row in time_variants):.0f} ms. "
        f"Changing cell size changed time from {min(row['processing_time_ms'] for row in cell_variants):.0f} "
        f"to {max(row['processing_time_ms'] for row in cell_variants):.0f} ms, but did not "
        "materially prune candidate checks in this run.",
        f"- **Memory-sensitive:** feature count controls fixed FeatureState storage "
        f"({int(baseline['state_memory_bytes'])} bytes at baseline using the current NumPy state dtype). "
        "Grid-cell size changes logical cell count and grid index bookkeeping.",
        "- **Hardware-oriented:** feature count/slots, integer cell indices, fixed grid "
        "dimensions, bounded R/T, and alpha/beta represented as quantized coefficients are "
        "natural fixed-width parameters. The present Python implementation does not establish "
        "fixed-point precision or FPGA timing.",
        "- **Grid selectivity limitation:** online velocity estimates and the conservative "
        "velocity-expanded grid neighborhood yielded approximately the full active feature "
        "set per event (about "
        f"{baseline['candidate_features_per_event']:.2f}/{baseline['initial_features_detected']} "
        "at baseline). The grid was correct but not selective in this five-interval sweep. "
        "Cell-size variation changed logical cell count, not measured candidate pruning; "
        "improving a selective, exact motion-aware index is a follow-up before claiming "
        "grid latency benefits for online tracking.",
        "",
        "## Multi-metric operating-point candidates",
        "",
        "A configuration is listed below if no other tested configuration is "
        "simultaneously no worse in EPE, lifetime, survival, failure rate, "
        "events/update, candidate count, processing time, and compact memory proxies. "
        "These are Pareto candidates; the sweep does not force a single weighted-score winner.",
        "",
        "| ID | Features | Cell px | R px | T μs | α | β | EPE px | Lifetime μs | Survival | Failure | ms | Candidates/event |",
        "|----|----------|---------|------|------|---|---|--------|-------------|----------|---------|----|------------------|",
        *[
            f"| {row['configuration_id']} | {int(row['feature_count_requested'])} | "
            f"{int(row['grid_cell_size_px'])} | {row['association_radius_px']:g} | "
            f"{row['association_time_window_us']:g} | {row['alpha']:g} | "
            f"{row['beta']:g} | {row['epe_px']:.3f} | "
            f"{row['track_lifetime_mean_us']:.0f} | "
            f"{row['mean_survival_fraction']:.3f} | "
            f"{row['failure_rate']:.3f} | {row['processing_time_ms']:.1f} | "
            f"{row['candidate_features_per_event']:.2f} |"
            for row in pareto_rows[:20]
        ],
        "",
        "Do not treat the frontier as a final selection: choose among these only "
        "after repeating on additional segments/sequences and checking intermediate "
        "state behavior. The algorithm remains unfrozen.",
        "",
        "## Limitations",
        "",
        "APS detections are the endpoint reference, not ground-truth feature identities. "
        "EPE only includes successful one-to-one matches; failure rate and survival must be "
        "read alongside it. The sweep is one-factor-at-a-time, limited to this sequence and "
        "segment, and uses software runtime. It does not establish statistical significance "
        "or freeze the algorithm.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_sweep(
    recording_name: str,
    base: ComparisonConfig,
    axes: Iterable[tuple[str, tuple[int | float, ...]]] = SWEEP_AXES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    configs = build_sweep_configs(base, axes)
    dataset_config = DavisConfig(recording=recording_name)
    recording = load_davis(dataset_config)
    origins, _ = load_timestamp_origins(recording, dataset_config)
    all_windows = make_frame_windows(recording, origins)
    end = base.interval_start + base.interval_count
    if end > len(all_windows):
        raise ValueError(f"segment end interval must be at most {len(all_windows) - 1}")
    windows = [
        all_windows.get_window(index)
        for index in range(base.interval_start, end)
    ]
    rows = _configuration_rows(configs, windows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "stage9_parameter_sweep.csv"
    _write_csv(csv_path, rows)
    config_path = output_dir / "stage9_sweep_configurations.json"
    config_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "design": "one-factor-at-a-time",
                "base_configuration": asdict(base),
                "configurations": [
                    {
                        "configuration_id": config_id,
                        "sweep_parameter": parameter,
                        "sweep_value": value,
                        "configuration": asdict(config),
                    }
                    for config_id, parameter, value, config in configs
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plots = _plot_tradeoffs(rows, output_dir)
    report_path = _write_report(rows, configs, output_dir)
    return {
        "csv": csv_path,
        "configurations": config_path,
        "report": report_path,
        **{f"plot_{name}": path for name, path in plots.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a reproducible one-factor-at-a-time tracker parameter sweep."
    )
    parser.add_argument("--recording", default="shapes_6dof")
    parser.add_argument("--interval-start", type=int, default=678)
    parser.add_argument("--interval-count", type=int, default=5)
    parser.add_argument("--reference-match-radius-px", type=float, default=15.0)
    parser.add_argument("--feature-counts", default="100,300,500,800")
    parser.add_argument("--grid-cell-sizes", default="8,16,24,32")
    parser.add_argument("--radii-px", default="4,8,12,16")
    parser.add_argument("--time-windows-us", default="10000,25000,50000,100000")
    parser.add_argument("--alphas", default="0.25,0.5,0.75")
    parser.add_argument("--betas", default="0.025,0.1,0.25")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    axes = (
        ("feature_count", _csv_values(args.feature_counts, int)),
        ("grid_cell_size_px", _csv_values(args.grid_cell_sizes, int)),
        ("association_radius_px", _csv_values(args.radii_px, float)),
        ("association_time_window_us", _csv_values(args.time_windows_us, float)),
        ("alpha", _csv_values(args.alphas, float)),
        ("beta", _csv_values(args.betas, float)),
    )
    base = ComparisonConfig(
        interval_start=args.interval_start,
        interval_count=args.interval_count,
        feature_count=500,
        slot_count=500,
        reference_match_radius_px=args.reference_match_radius_px,
        use_spatial_grid=True,
    )
    try:
        base.validate()
        outputs = run_sweep(args.recording, base, axes, args.output_dir)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"configurations: {len(build_sweep_configs(base, axes))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
