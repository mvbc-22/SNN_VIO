"""Stage 6A Shi-Tomasi plus KLT frame-only tracking baseline."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .aps_features import ShiTomasiConfig, extract_shi_tomasi_features
from .asynchronous_feature_tracker import (
    LIFETIME_DTYPE,
    MATCH_DTYPE,
    PREDICTED_DTYPE,
    match_predictions_to_references,
)
from .davis_inspection import DavisConfig, load_davis
from .frame_synchronization import FrameWindow, load_timestamp_origins, make_frame_windows

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "frame_only_baseline"
)
KLT_DTYPE = np.dtype(
    [
        ("feature_id", np.int32),
        ("initial_x", np.float32),
        ("initial_y", np.float32),
        ("predicted_x", np.float32),
        ("predicted_y", np.float32),
        ("status", np.uint8),
        ("error", np.float32),
    ]
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


@dataclass(frozen=True)
class FrameBaselineConfig:
    feature_count: int = 500
    quality_level: float = 0.01
    min_feature_distance_px: float = 5.0
    block_size_px: int = 3
    bucket_columns: int = 8
    bucket_rows: int = 6
    reference_match_radius_px: float = 15.0
    klt_window_px: int = 21
    klt_max_pyramid_level: int = 3
    klt_max_iterations: int = 30
    klt_epsilon: float = 0.01

    def validate(self) -> None:
        if self.feature_count <= 0:
            raise ValueError("feature_count must be positive")
        if not np.isfinite(self.quality_level) or not 0 < self.quality_level <= 1:
            raise ValueError("quality_level must be finite and within (0, 1]")
        if (
            not np.isfinite(self.min_feature_distance_px)
            or self.min_feature_distance_px < 0
        ):
            raise ValueError("min_feature_distance_px must be finite and non-negative")
        if self.block_size_px < 2:
            raise ValueError("block_size_px must be at least 2")
        if self.bucket_columns <= 0 or self.bucket_rows <= 0:
            raise ValueError("bucket dimensions must be positive")
        if (
            not np.isfinite(self.reference_match_radius_px)
            or self.reference_match_radius_px <= 0
        ):
            raise ValueError("reference_match_radius_px must be finite and positive")
        if self.klt_window_px < 3 or self.klt_window_px % 2 == 0:
            raise ValueError("klt_window_px must be an odd integer of at least 3")
        if self.klt_max_pyramid_level < 0:
            raise ValueError("klt_max_pyramid_level must be non-negative")
        if self.klt_max_iterations <= 0:
            raise ValueError("klt_max_iterations must be positive")
        if not np.isfinite(self.klt_epsilon) or self.klt_epsilon <= 0:
            raise ValueError("klt_epsilon must be finite and positive")


@dataclass(frozen=True)
class FrameBaselineResult:
    initial_features: np.ndarray
    klt_tracks: np.ndarray
    reference_features: np.ndarray
    predicted_features: np.ndarray
    matches: np.ndarray
    lifetimes: np.ndarray
    diagnostics: dict[str, Any]
    checks: dict[str, bool]


def run_frame_only_baseline(
    window: FrameWindow,
    config: FrameBaselineConfig = FrameBaselineConfig(),
) -> FrameBaselineResult:
    """Run Shi-Tomasi and pyramidal KLT without accessing the event stream."""
    config.validate()
    started_ns = time.perf_counter_ns()
    first = extract_shi_tomasi_features(
        window.frame_k,
        ShiTomasiConfig(
            max_features=config.feature_count,
            quality_level=config.quality_level,
            min_feature_distance=config.min_feature_distance_px,
            block_size=config.block_size_px,
            bucket_columns=config.bucket_columns,
            bucket_rows=config.bucket_rows,
        ),
        frame_index=window.index,
        frame_timestamp_us=window.frame_k_common_time_us,
    ).features

    klt_started_ns = time.perf_counter_ns()
    if len(first):
        initial_points = np.column_stack((first["x"], first["y"])).astype(
            np.float32
        ).reshape(-1, 1, 2)
        tracked_points, status, error = cv2.calcOpticalFlowPyrLK(
            window.frame_k,
            window.frame_k_plus_1,
            initial_points,
            None,
            winSize=(config.klt_window_px, config.klt_window_px),
            maxLevel=config.klt_max_pyramid_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                config.klt_max_iterations,
                config.klt_epsilon,
            ),
        )
        if tracked_points is None or status is None:
            raise RuntimeError("OpenCV KLT returned no tracked point/status arrays")
        if tracked_points.shape != initial_points.shape or status.size != len(first):
            raise RuntimeError("OpenCV KLT returned arrays with unexpected dimensions")
        status_values = status.reshape(-1).astype(bool)
        tracked_xy = tracked_points.reshape(-1, 2)
        height, width = window.frame_k_plus_1.shape[:2]
        status_values &= np.isfinite(tracked_xy).all(axis=1)
        status_values &= (
            (tracked_xy[:, 0] >= 0)
            & (tracked_xy[:, 0] < width)
            & (tracked_xy[:, 1] >= 0)
            & (tracked_xy[:, 1] < height)
        )
        error_values = (
            np.full(len(first), np.nan, dtype=np.float32)
            if error is None
            else error.reshape(-1).astype(np.float32)
        )
        tracks = np.zeros(len(first), dtype=KLT_DTYPE)
        tracks["feature_id"] = first["id"]
        tracks["initial_x"] = first["x"]
        tracks["initial_y"] = first["y"]
        tracks["predicted_x"] = np.nan
        tracks["predicted_y"] = np.nan
        tracks["status"] = status_values.astype(np.uint8)
        tracks["error"] = error_values
        tracks["predicted_x"][status_values] = tracked_xy[status_values, 0]
        tracks["predicted_y"][status_values] = tracked_xy[status_values, 1]
    else:
        tracks = np.empty(0, dtype=KLT_DTYPE)
    klt_time_ms = (time.perf_counter_ns() - klt_started_ns) / 1_000_000

    next_features = extract_shi_tomasi_features(
        window.frame_k_plus_1,
        ShiTomasiConfig(
            max_features=config.feature_count,
            quality_level=config.quality_level,
            min_feature_distance=config.min_feature_distance_px,
            block_size=config.block_size_px,
            bucket_columns=config.bucket_columns,
            bucket_rows=config.bucket_rows,
        ),
        frame_index=window.index + 1,
        frame_timestamp_us=window.frame_k_plus_1_common_time_us,
    ).features

    successful_tracks = tracks[tracks["status"] != 0]
    predicted = np.zeros(len(successful_tracks), dtype=PREDICTED_DTYPE)
    for index, track in enumerate(successful_tracks):
        predicted[index] = (
            int(track["feature_id"]),
            int(track["feature_id"]),
            float(track["predicted_x"]),
            float(track["predicted_y"]),
            0.0,
            0.0,
            1.0,
            window.frame_k_plus_1_common_time_us,
        )
    matches = match_predictions_to_references(
        predicted,
        next_features,
        config.reference_match_radius_px,
    )

    interval_us = (
        window.frame_k_plus_1_common_time_us - window.frame_k_common_time_us
    )
    lifetimes = np.zeros(len(tracks), dtype=LIFETIME_DTYPE)
    for index, track in enumerate(tracks):
        successful = bool(track["status"])
        lifetimes[index] = (
            int(track["feature_id"]),
            int(track["feature_id"]),
            interval_us if successful else 0,
            interval_us / 1_000_000 if successful else 0.0,
        )

    successful_matches = matches["matched"] != 0
    tracked_count = int(np.count_nonzero(tracks["status"]))
    initial_count = len(first)
    matched_count = int(np.count_nonzero(successful_matches))
    errors = matches["pixel_error"][successful_matches]
    processing_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    diagnostics: dict[str, Any] = {
        "method": "frame_only_shi_tomasi_pyramidal_lk",
        "initial_features": initial_count,
        "tracked_features": tracked_count,
        "klt_successful_features": tracked_count,
        "klt_failed_features": initial_count - tracked_count,
        "klt_failure_rate": (
            float((initial_count - tracked_count) / initial_count)
            if initial_count
            else 0.0
        ),
        "successfully_predicted_features": matched_count,
        "reference_matched_features": matched_count,
        "reference_features_detected": len(next_features),
        "epe_px": float(np.mean(errors)) if len(errors) else None,
        "epe_median_px": float(np.median(errors)) if len(errors) else None,
        "track_lifetime_mean_us": (
            float(np.mean(lifetimes["lifetime_us"])) if initial_count
            else 0.0
        ),
        "track_lifetime_max_us": int(np.max(lifetimes["lifetime_us"], initial=0)),
        "failure_rate": (
            float((initial_count - matched_count) / initial_count)
            if initial_count
            else 0.0
        ),
        "processing_time_ms": processing_ms,
        "latency_klt_call_ms": klt_time_ms,
        "latency_per_initial_feature_us": (
            float(klt_time_ms * 1000 / initial_count) if initial_count else 0.0
        ),
        "latency_definition": (
            "Wall-clock time around the single cv2.calcOpticalFlowPyrLK call; "
            "per-feature value is amortized KLT-call time, not hardware latency."
        ),
        "processing_time_definition": (
            "In-memory time for frame-k detection, KLT, frame-k+1 reference "
            "detection, and shared endpoint matching; excludes dataset loading "
            "and plotting."
        ),
        "reference_match_policy": (
            "Same greedy one-to-one nearest reference-feature matching and "
            "configured distance gate as Stage 5."
        ),
        "epe_definition": (
            "Mean endpoint Euclidean pixel error over successful KLT predictions "
            "matched to frame-k+1 Shi-Tomasi references."
        ),
        "failure_rate_definition": (
            "Unmatched initial feature tracks after endpoint reference matching "
            "divided by the common initial Shi-Tomasi feature count; comparable "
            "to Stage 5. KLT-only failures are reported separately."
        ),
        "klt_failure_rate_definition": (
            "KLT status failures or out-of-frame/non-finite points divided by "
            "initial Shi-Tomasi feature count."
        ),
        "track_lifetime_definition": (
            "For this one interval, a KLT-successful point receives the full APS "
            "interval lifetime and a KLT failure receives zero; the reported mean "
            "uses all initial features. This is interval-censored."
        ),
        "common_metrics": {
            "initial_features": initial_count,
            "tracked_features": tracked_count,
            "successfully_predicted_features": matched_count,
            "reference_matched_features": matched_count,
            "epe_px": float(np.mean(errors)) if len(errors) else None,
            "track_lifetime_mean_us": (
                float(np.mean(lifetimes["lifetime_us"])) if initial_count else 0.0
            ),
            "failure_rate": (
                float((initial_count - matched_count) / initial_count)
                if initial_count
                else 0.0
            ),
            "processing_time_ms": processing_ms,
            "latency_ms": klt_time_ms,
        },
    }
    checks = {
        "initialization": (
            len(first) <= config.feature_count
            and len(np.unique(first["id"])) == len(first)
        ),
        "klt_output": (
            len(tracks) == initial_count
            and np.all(tracks["status"] <= 1)
            and np.isfinite(tracks["predicted_x"][tracks["status"] != 0]).all()
            and np.isfinite(tracks["predicted_y"][tracks["status"] != 0]).all()
        ),
        "reference_matching": (
            len(matches) == tracked_count
            and len(np.unique(matches["reference_index"][matches["matched"] != 0]))
            == matched_count
        ),
        "evaluation": (
            matched_count <= min(tracked_count, len(next_features))
            and np.isfinite(errors).all()
        ),
    }
    return FrameBaselineResult(
        initial_features=first,
        klt_tracks=tracks,
        reference_features=next_features,
        predicted_features=predicted,
        matches=matches,
        lifetimes=lifetimes,
        diagnostics=diagnostics,
        checks=checks,
    )


def _save_visualizations(
    window: FrameWindow,
    result: FrameBaselineResult,
    output_dir: Path,
) -> dict[str, Path]:
    output_paths: dict[str, Path] = {}

    def save(key: str, filename: str, fig: plt.Figure) -> None:
        path = output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        output_paths[key] = path

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(window.frame_k, cmap="gray", vmin=0, vmax=255)
    ax.scatter(
        result.initial_features["x"],
        result.initial_features["y"],
        c="lime",
        marker="+",
        s=18,
    )
    ax.set_title(
        f"Frame k={window.index}: {len(result.initial_features)} initial Shi-Tomasi features"
    )
    ax.set_axis_off()
    save("01_frame_k_initial_features", "01_frame_k_initial_features.png", fig)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(window.frame_k, cmap="gray", vmin=0, vmax=255)
    ax.text(
        0.5,
        0.5,
        "No event data used in the frame-only baseline",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=15,
        bbox={"facecolor": "black", "alpha": 0.7, "pad": 10},
    )
    ax.set_title("Event-stream panel — intentionally unused")
    ax.set_axis_off()
    save("02_event_stream", "02_event_stream.png", fig)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(window.frame_k, cmap="gray", vmin=0, vmax=255)
    for track in result.klt_tracks[result.klt_tracks["status"] != 0]:
        ax.plot(
            [track["initial_x"], track["predicted_x"]],
            [track["initial_y"], track["predicted_y"]],
            color="cyan",
            linewidth=0.6,
            alpha=0.65,
        )
    ax.scatter(
        result.klt_tracks["initial_x"],
        result.klt_tracks["initial_y"],
        s=10,
        c="lime",
        marker="+",
        label="initial features",
    )
    successful = result.klt_tracks[result.klt_tracks["status"] != 0]
    ax.scatter(
        successful["predicted_x"],
        successful["predicted_y"],
        s=12,
        c="magenta",
        marker="s",
        label="KLT endpoints",
    )
    ax.set_title(f"KLT frame-to-frame tracks: {len(successful)} / {len(result.klt_tracks)}")
    ax.set_axis_off()
    ax.legend(loc="upper right")
    save("03_event_feature_trajectories", "03_klt_feature_trajectories.png", fig)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(window.frame_k_plus_1, cmap="gray", vmin=0, vmax=255)
    if len(result.predicted_features):
        ax.scatter(
            result.predicted_features["x"],
            result.predicted_features["y"],
            s=18,
            facecolors="none",
            edgecolors="cyan",
            marker="^",
        )
    ax.set_title("KLT tracked positions at frame k+1")
    ax.set_axis_off()
    save(
        "04_predicted_positions_frame_k_plus_1",
        "04_klt_positions_frame_k_plus_1.png",
        fig,
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(window.frame_k_plus_1, cmap="gray", vmin=0, vmax=255)
    ax.scatter(
        result.reference_features["x"],
        result.reference_features["y"],
        c="yellow",
        marker="+",
        s=18,
    )
    ax.set_title("Reference Shi-Tomasi features at frame k+1")
    ax.set_axis_off()
    save(
        "05_reference_features_frame_k_plus_1",
        "05_reference_features_frame_k_plus_1.png",
        fig,
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(window.frame_k_plus_1, cmap="gray", vmin=0, vmax=255)
    matched = result.matches[result.matches["matched"] != 0]
    if len(matched):
        ax.scatter(
            matched["predicted_x"],
            matched["predicted_y"],
            c="cyan",
            marker="^",
            s=20,
            label="KLT position",
        )
        ax.scatter(
            matched["reference_x"],
            matched["reference_y"],
            c="yellow",
            marker="+",
            s=24,
            label="reference",
        )
        for row in matched:
            ax.plot(
                [row["predicted_x"], row["reference_x"]],
                [row["predicted_y"], row["reference_y"]],
                c="magenta",
                linewidth=0.7,
            )
    ax.set_title(
        f"Frame-only evaluation: {len(matched)} matches; "
        + (
            f"EPE={result.diagnostics['epe_px']:.3f}px"
            if result.diagnostics["epe_px"] is not None
            else "EPE unavailable"
        )
    )
    ax.set_axis_off()
    if len(matched):
        ax.legend(loc="upper right")
    save(
        "06_predicted_vs_reference_error",
        "06_predicted_vs_reference_error.png",
        fig,
    )
    return output_paths


def save_baseline(
    window: FrameWindow,
    result: FrameBaselineResult,
    config: FrameBaselineConfig,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "davis_frame_only_baseline_results.npz"
    np.savez_compressed(
        result_path,
        initial_features=result.initial_features,
        klt_tracks=result.klt_tracks,
        reference_features=result.reference_features,
        predicted_features=result.predicted_features,
        matches=result.matches,
        lifetimes=result.lifetimes,
    )
    comparison = {
        "schema": "snn-vio-tracker-comparison-v1",
        "stage": "6A",
        "method": "frame_only_shi_tomasi_pyramidal_lk",
        "uses_events": False,
        "interval_index": window.index,
        "frame_timestamps_common_us": list(window.interval_us),
        "timestamp_unit": "microseconds",
        "configuration": asdict(config),
        "metrics": dict(result.diagnostics),
        "common_metrics": dict(result.diagnostics["common_metrics"]),
        "checks": dict(result.checks),
        "definitions": {
            "tracked_features": "KLT status-positive, finite, in-image endpoints.",
            "EPE": result.diagnostics["epe_definition"],
            "failure_rate": result.diagnostics["failure_rate_definition"],
            "track_lifetime": result.diagnostics["track_lifetime_definition"],
            "latency": result.diagnostics["latency_definition"],
            "common_latency_ms": (
                "Stage 6A cv2.calcOpticalFlowPyrLK call time; excludes detection, "
                "dataset I/O and plotting."
            ),
        },
        "arrays_file": result_path.name,
    }
    comparison_path = output_dir / "davis_frame_only_comparison_ready.json"
    comparison_path.write_text(
        json.dumps(_json_safe(comparison), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    visualization_paths = _save_visualizations(window, result, output_dir)
    report_path = output_dir / "davis_frame_only_baseline_report.md"
    epe = result.diagnostics["epe_px"]
    report_path.write_text(
        "\n".join(
            [
                "# Stage 6A — Conventional frame-only KLT baseline",
                "",
                f"- APS pair: {window.index} → {window.index + 1}",
                f"- Common-axis interval: [{window.interval_us[0]}, {window.interval_us[1]}) μs",
                f"- Initial Shi-Tomasi features: {result.diagnostics['initial_features']}",
                f"- Tracked features: {result.diagnostics['tracked_features']}",
                f"- Reference-matched features: {result.diagnostics['reference_matched_features']}",
                f"- EPE: {epe:.6g} px" if epe is not None else "- EPE: unavailable (no matches)",
                f"- Mean track lifetime: {result.diagnostics['track_lifetime_mean_us']:.0f} μs",
                f"- Failure rate: {result.diagnostics['failure_rate']:.6g}",
                f"- KLT-only failure rate: {result.diagnostics['klt_failure_rate']:.6g}",
                f"- Processing time: {result.diagnostics['processing_time_ms']:.3f} ms",
                f"- KLT-call latency: {result.diagnostics['latency_klt_call_ms']:.3f} ms",
                f"- Amortized latency per initial feature: {result.diagnostics['latency_per_initial_feature_us']:.3f} μs",
                "- Event data: not accessed or used.",
                "",
                "## Baseline checks",
                "",
                "| Check | Result |",
                "|---|---|",
                *[
                    f"| {name.replace('_', ' ').title()} | {'PASS' if passed else 'FAIL'} |"
                    for name, passed in result.checks.items()
                ],
                "",
                "The endpoint evaluation reuses Stage 5's greedy one-to-one reference-feature matching with the same configured radius. Track lifetime is interval-censored: successful KLT tracks are assigned the full pair duration and failures zero lifetime.",
                "",
                "The `davis_frame_only_comparison_ready.json` structure exposes method, event-use flag, configuration, metrics, check results and the numeric-array archive for direct comparison with Stage 5.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "results": result_path,
        "comparison_ready": comparison_path,
        "report": report_path,
        **visualization_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run frame-only Shi-Tomasi plus KLT baseline on DAVIS APS frames."
    )
    parser.add_argument("--recording", default="shapes_6dof")
    parser.add_argument("--interval-index", type=int, default=678)
    parser.add_argument("--feature-count", type=int, default=500)
    parser.add_argument("--reference-match-radius-px", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.interval_index < 0:
        parser.error("--interval-index must be non-negative")
    config = FrameBaselineConfig(
        feature_count=args.feature_count,
        reference_match_radius_px=args.reference_match_radius_px,
    )
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))

    dataset_config = DavisConfig(recording=args.recording)
    recording = load_davis(dataset_config)
    origins_s, _ = load_timestamp_origins(recording, dataset_config)
    windows = make_frame_windows(recording, origins_s)
    if args.interval_index >= len(windows):
        parser.error(f"--interval-index must be within [0, {len(windows) - 1}]")
    window = windows.get_window(args.interval_index)
    result = run_frame_only_baseline(window, config)
    outputs = save_baseline(window, result, config, args.output_dir)
    print(json.dumps(result.diagnostics, indent=2, allow_nan=False))
    print("Checks:")
    for name, passed in result.checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0 if all(result.checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
