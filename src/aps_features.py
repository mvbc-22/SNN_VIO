"""Spatially bucketed Shi-Tomasi APS feature extraction for Stage 2A."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .davis_inspection import DavisConfig, load_davis
from .frame_synchronization import (
    FrameWindows,
    load_timestamp_origins,
    make_frame_windows,
)

REQUESTED_FEATURE_COUNTS = (50, 100, 200, 300, 500, 1000)
FEATURE_DTYPE = np.dtype(
    [
        ("id", np.int32),
        ("x", np.float32),
        ("y", np.float32),
        ("quality", np.float32),
        ("valid", np.uint8),
    ]
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "aps_feature_extraction"
)


@dataclass(frozen=True)
class ShiTomasiConfig:
    """OpenCV parameters plus the image grid used for spatial feature bucketing."""

    max_features: int
    quality_level: float = 0.01
    min_feature_distance: float = 5.0
    block_size: int = 3
    bucket_columns: int = 8
    bucket_rows: int = 6

    def validate(self) -> None:
        if self.max_features <= 0:
            raise ValueError("max_features must be positive")
        if not np.isfinite(self.quality_level) or not 0 < self.quality_level <= 1:
            raise ValueError("quality_level must be in (0, 1]")
        if not np.isfinite(self.min_feature_distance) or self.min_feature_distance < 0:
            raise ValueError("min_feature_distance must be non-negative")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if self.bucket_columns <= 0 or self.bucket_rows <= 0:
            raise ValueError("bucket grid dimensions must be positive")


@dataclass(frozen=True)
class FeatureResult:
    """Compact feature records and measurements for one APS image."""

    frame_index: int
    frame_timestamp_us: int
    requested_count: int
    features: np.ndarray
    metrics: Mapping[str, Any]


def get_aps_frame(windows: FrameWindows, frame_index: int) -> tuple[np.ndarray, int]:
    """Fetch an APS frame through the synchronized frame-pair sequence."""
    if frame_index < 0 or frame_index > len(windows):
        raise IndexError(
            f"APS frame index {frame_index} is out of range [0, {len(windows)}]"
        )
    if frame_index == len(windows):
        final_window = windows.get_window(len(windows) - 1)
        return final_window.frame_k_plus_1, final_window.frame_k_plus_1_timestamp_us
    window = windows.get_window(frame_index)
    return window.frame_k, window.frame_k_timestamp_us


def _gray_u8(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim != 2:
        raise ValueError("Shi-Tomasi input must be a single-channel APS frame")
    if image.dtype != np.uint8:
        raise ValueError("APS frame must be uint8; implicit intensity conversion is disabled")
    if image.shape[0] < 3 or image.shape[1] < 3:
        raise ValueError("APS frame must be at least 3 by 3 pixels")
    return image


def _detect_bucket_candidates(
    gray: np.ndarray, config: ShiTomasiConfig
) -> tuple[list[list[tuple[float, float, float]]], np.ndarray]:
    height, width = gray.shape
    response = cv2.cornerMinEigenVal(gray, blockSize=config.block_size)
    buckets: list[list[tuple[float, float, float]]] = [
        [] for _ in range(config.bucket_columns * config.bucket_rows)
    ]
    x_edges = np.linspace(0, width, config.bucket_columns + 1, dtype=np.int32)
    y_edges = np.linspace(0, height, config.bucket_rows + 1, dtype=np.int32)

    for row in range(config.bucket_rows):
        y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
        for column in range(config.bucket_columns):
            x0, x1 = int(x_edges[column]), int(x_edges[column + 1])
            roi = gray[y0:y1, x0:x1]
            bucket_index = row * config.bucket_columns + column
            if roi.shape[0] < config.block_size or roi.shape[1] < config.block_size:
                continue
            corners = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=config.max_features,
                qualityLevel=config.quality_level,
                minDistance=config.min_feature_distance,
                blockSize=config.block_size,
                useHarrisDetector=False,
            )
            if corners is None:
                continue
            xy = corners.reshape(-1, 2)
            candidates = [
                (
                    float(x0 + point[0]),
                    float(y0 + point[1]),
                    float(response[
                        min(height - 1, max(0, int(round(y0 + point[1])))),
                        min(width - 1, max(0, int(round(x0 + point[0])))),
                    ]),
                )
                for point in xy
            ]
            candidates.sort(key=lambda item: (-item[2], item[1], item[0]))
            buckets[bucket_index] = candidates
    return buckets, response


def _min_distance_allows(
    x: float,
    y: float,
    min_distance: float,
    spatial_hash: dict[tuple[int, int], list[tuple[float, float]]],
) -> bool:
    if min_distance <= 0:
        return True
    cell_size = min_distance
    cell_x, cell_y = int(x // cell_size), int(y // cell_size)
    distance_squared = min_distance * min_distance
    for grid_y in range(cell_y - 1, cell_y + 2):
        for grid_x in range(cell_x - 1, cell_x + 2):
            for other_x, other_y in spatial_hash.get((grid_x, grid_y), ()):
                if (x - other_x) ** 2 + (y - other_y) ** 2 < distance_squared:
                    return False
    return True


def extract_shi_tomasi_features(
    frame: np.ndarray,
    config: ShiTomasiConfig,
    frame_index: int = 0,
    frame_timestamp_us: int = 0,
    first_feature_id: int = 0,
) -> FeatureResult:
    """Extract up to `max_features`, selecting candidates round-robin by image bucket."""
    config.validate()
    gray = _gray_u8(frame)
    buckets, _ = _detect_bucket_candidates(gray, config)
    return _select_candidates(
        gray,
        config,
        buckets,
        frame_index,
        frame_timestamp_us,
        first_feature_id,
    )


def default_frame_indices(frame_count: int, sample_count: int = 5) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    return np.unique(
        np.linspace(0, frame_count - 1, min(sample_count, frame_count), dtype=int)
    ).tolist()


def run_feature_count_experiment(
    windows: FrameWindows,
    frame_indices: Sequence[int] | None = None,
    requested_counts: Sequence[int] = REQUESTED_FEATURE_COUNTS,
    quality_level: float = 0.01,
    min_feature_distance: float = 5.0,
    block_size: int = 3,
    bucket_columns: int = 8,
    bucket_rows: int = 6,
) -> tuple[list[FeatureResult], dict[str, Any]]:
    """Compare target counts over selected synchronized APS frames."""
    frame_count = len(windows) + 1
    if frame_indices is None:
        frame_indices = default_frame_indices(frame_count)
    frame_indices = [int(index) for index in frame_indices]
    if not frame_indices:
        raise ValueError("At least one APS frame must be selected")
    if any(index < 0 or index >= frame_count for index in frame_indices):
        raise IndexError(f"Frame indices must be within [0, {frame_count - 1}]")
    if len(set(frame_indices)) != len(frame_indices):
        raise ValueError("Frame indices must be unique")
    counts = [int(count) for count in requested_counts]
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("Requested feature counts must be positive")
    if len(set(counts)) != len(counts):
        raise ValueError("Requested feature counts must be unique")

    results: list[FeatureResult] = []
    next_feature_id = 0
    for frame_index in frame_indices:
        frame, timestamp_us = get_aps_frame(windows, frame_index)
        max_count = max(counts)
        candidate_config = ShiTomasiConfig(
            max_features=max_count,
            quality_level=quality_level,
            min_feature_distance=min_feature_distance,
            block_size=block_size,
            bucket_columns=bucket_columns,
            bucket_rows=bucket_rows,
        )
        bucket_candidates, _ = _detect_bucket_candidates(
            _gray_u8(frame), candidate_config
        )
        for requested in counts:
            config = ShiTomasiConfig(
                max_features=requested,
                quality_level=quality_level,
                min_feature_distance=min_feature_distance,
                block_size=block_size,
                bucket_columns=bucket_columns,
                bucket_rows=bucket_rows,
            )
            result = _select_candidates(
                frame,
                config,
                bucket_candidates,
                frame_index,
                timestamp_us,
                next_feature_id,
            )
            results.append(result)
            next_feature_id += len(result.features)
    return results, {
        "requested_counts": counts,
        "frame_indices": frame_indices,
        "timestamp_unit": "microseconds",
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "quality_level": quality_level,
        "min_feature_distance_px": min_feature_distance,
        "block_size_px": block_size,
        "bucket_grid": [bucket_columns, bucket_rows],
        "feature_dtype": [
            {"name": name, "dtype": str(FEATURE_DTYPE.fields[name][0])}
            for name in FEATURE_DTYPE.names
        ],
        "feature_id_semantics": (
            "Unique monotonically assigned record IDs within this experiment run; "
            "IDs do not imply feature identity across frames."
        ),
    }


def _select_candidates(
    frame: np.ndarray,
    config: ShiTomasiConfig,
    buckets: list[list[tuple[float, float, float]]],
    frame_index: int,
    frame_timestamp_us: int,
    first_feature_id: int,
) -> FeatureResult:
    """Select a requested prefix from the shared per-frame bucket candidates."""
    # Candidate detection is done once at the largest requested cap. Reuse the
    # sorted candidates for each requested count while enforcing the configured
    # global minimum distance across bucket boundaries.
    config.validate()
    gray = _gray_u8(frame)
    height, width = gray.shape
    bucket_counts = np.zeros((config.bucket_rows, config.bucket_columns), dtype=np.int32)
    candidate_capacity = min(config.max_features, sum(map(len, buckets)))
    records = np.zeros(candidate_capacity, dtype=FEATURE_DTYPE)
    spatial_hash: dict[tuple[int, int], list[tuple[float, float]]] = {}
    selected = 0
    cursors = [0] * len(buckets)
    while selected < config.max_features:
        any_added = False
        for index, candidates in enumerate(buckets):
            while cursors[index] < len(candidates):
                x, y, quality = candidates[cursors[index]]
                cursors[index] += 1
                if not _min_distance_allows(
                    x, y, config.min_feature_distance, spatial_hash
                ):
                    continue
                records[selected] = (
                    first_feature_id + selected,
                    x,
                    y,
                    quality,
                    1,
                )
                selected += 1
                bucket_y = min(config.bucket_rows - 1, int(y * config.bucket_rows / height))
                bucket_x = min(config.bucket_columns - 1, int(x * config.bucket_columns / width))
                bucket_counts[bucket_y, bucket_x] += 1
                if config.min_feature_distance > 0:
                    cell = (
                        int(x // config.min_feature_distance),
                        int(y // config.min_feature_distance),
                    )
                    spatial_hash.setdefault(cell, []).append((x, y))
                any_added = True
                break
            if selected >= config.max_features:
                break
        if not any_added:
            break
    records = records[:selected]
    return FeatureResult(
        frame_index=int(frame_index),
        frame_timestamp_us=int(frame_timestamp_us),
        requested_count=config.max_features,
        features=records,
        metrics=_feature_metrics(records, bucket_counts, config),
    )


def _feature_metrics(
    records: np.ndarray, bucket_counts: np.ndarray, config: ShiTomasiConfig
) -> dict[str, Any]:
    selected = len(records)
    occupied = int(np.count_nonzero(bucket_counts))
    qualities = records["quality"]
    probabilities = bucket_counts.ravel()
    probabilities = probabilities[probabilities > 0].astype(np.float64)
    if probabilities.size:
        probabilities /= probabilities.sum()
        entropy = float(
            -np.sum(probabilities * np.log(probabilities))
            / np.log(config.bucket_columns * config.bucket_rows)
        )
    else:
        entropy = 0.0
    return {
        "requested_count": config.max_features,
        "detected_count": int(selected),
        "count_shortfall": int(config.max_features - selected),
        "occupied_buckets": occupied,
        "total_buckets": int(config.bucket_columns * config.bucket_rows),
        "bucket_coverage_percent": float(
            100.0 * occupied / (config.bucket_columns * config.bucket_rows)
        ),
        "bucket_count_std": float(np.std(bucket_counts)),
        "normalized_bucket_entropy": entropy,
        "max_bucket_occupancy": int(np.max(bucket_counts, initial=0)),
        "mean_quality": float(np.mean(qualities)) if selected else 0.0,
        "median_quality": float(np.median(qualities)) if selected else 0.0,
        "min_quality": float(np.min(qualities)) if selected else 0.0,
        "max_quality": float(np.max(qualities)) if selected else 0.0,
        "bucket_counts": bucket_counts.tolist(),
    }


def _summary_rows(results: Sequence[FeatureResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        row = {
            "frame_index": result.frame_index,
            "frame_timestamp_us": result.frame_timestamp_us,
            **{
                key: value
                for key, value in result.metrics.items()
                if key != "bucket_counts"
            },
        }
        rows.append(row)
    return rows


def save_experiment_outputs(
    results: Sequence[FeatureResult],
    configuration: Mapping[str, Any],
    output_dir: Path,
    frame_lookup: Mapping[int, np.ndarray],
) -> dict[str, Path]:
    """Save numeric feature records, per-frame CSV, plots, and experiment report."""
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("Cannot save an experiment with no feature results")
    if not configuration.get("requested_counts") or not configuration.get("frame_indices"):
        raise ValueError("Experiment configuration must include counts and frames")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "davis_shapes_6dof"
    rows = _summary_rows(results)
    feature_rows = []
    for result in results:
        stored = np.empty(
            len(result.features),
            dtype=np.dtype(
                [("frame_index", np.int32), ("requested_count", np.int32)]
                + list(FEATURE_DTYPE.descr)
            ),
        )
        stored["frame_index"] = result.frame_index
        stored["requested_count"] = result.requested_count
        for field in FEATURE_DTYPE.names:
            stored[field] = result.features[field]
        feature_rows.append(stored)
    all_features = (
        np.concatenate(feature_rows)
        if feature_rows
        else np.empty(0, dtype=[("frame_index", np.int32), ("requested_count", np.int32)] + list(FEATURE_DTYPE.descr))
    )
    features_path = output_dir / f"{stem}_features.npz"
    np.savez_compressed(
        features_path,
        features=all_features,
        requested_counts=np.asarray(configuration["requested_counts"], dtype=np.int32),
        frame_indices=np.asarray(configuration["frame_indices"], dtype=np.int32),
    )

    csv_path = output_dir / f"{stem}_measurements.csv"
    csv_fields = [key for key in rows[0] if key != "bucket_counts"] if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = output_dir / f"{stem}_configuration.json"
    metadata_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "algorithm": "Shi-Tomasi (OpenCV goodFeaturesToTrack, Harris disabled)",
                "configuration": dict(configuration),
                "measurement_count": len(rows),
                "feature_record_count": int(len(all_features)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    frame_for_examples = int(
        configuration["frame_indices"][len(configuration["frame_indices"]) // 2]
    )
    frame_result_by_count = {
        result.requested_count: result
        for result in results
        if result.frame_index == frame_for_examples
    }
    requested_counts = list(configuration["requested_counts"])
    panel_columns = min(3, len(requested_counts))
    panel_rows = math.ceil(len(requested_counts) / panel_columns)
    fig, axes = plt.subplots(
        panel_rows,
        panel_columns,
        figsize=(5 * panel_columns, 4.5 * panel_rows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, requested in zip(axes.flat, requested_counts):
        example = frame_result_by_count[requested]
        if example.frame_index not in frame_lookup:
            raise ValueError(f"Missing APS frame {example.frame_index} for overlay plot")
        ax.imshow(frame_lookup[example.frame_index], cmap="gray", vmin=0, vmax=255)
        ax.scatter(
            example.features["x"],
            example.features["y"],
            s=13,
            c="tab:red",
            marker="+",
            linewidths=0.8,
        )
        ax.set_title(
            f"requested {requested}, detected {len(example.features)}\n"
            f"buckets {example.metrics['occupied_buckets']}/"
            f"{example.metrics['total_buckets']}"
        )
        ax.set_axis_off()
    for ax in axes.flat[len(requested_counts) :]:
        ax.set_axis_off()
    overlay_path = output_dir / f"{stem}_feature_overlays.png"
    fig.savefig(overlay_path, dpi=160)
    plt.close(fig)

    bucket_path = output_dir / f"{stem}_spatial_buckets.png"
    fig, axes = plt.subplots(
        panel_rows,
        panel_columns,
        figsize=(4 * panel_columns, 3.5 * panel_rows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, requested in zip(axes.flat, requested_counts):
        example = frame_result_by_count[requested]
        grid = np.asarray(example.metrics["bucket_counts"])
        image = ax.imshow(grid, cmap="viridis", vmin=0)
        ax.set_title(f"requested {requested}; {len(example.features)} detected")
        ax.set_xlabel("image bucket x")
        ax.set_ylabel("image bucket y")
        fig.colorbar(image, ax=ax, shrink=0.75, label="features")
    for ax in axes.flat[len(requested_counts) :]:
        ax.set_axis_off()
    bucket_plot_path = output_dir / f"{stem}_spatial_buckets.png"
    fig.savefig(bucket_plot_path, dpi=160)
    plt.close(fig)

    aggregate: dict[int, dict[str, float]] = {}
    for requested in requested_counts:
        subset = [r for r in results if r.requested_count == requested]
        aggregate[requested] = {
            "detected_mean": float(np.mean([r.metrics["detected_count"] for r in subset])),
            "coverage_mean": float(
                np.mean([r.metrics["bucket_coverage_percent"] for r in subset])
            ),
            "quality_median_mean": float(
                np.mean([r.metrics["median_quality"] for r in subset])
            ),
            "entropy_mean": float(
                np.mean([r.metrics["normalized_bucket_entropy"] for r in subset])
            ),
        }
    summary_path = output_dir / f"{stem}_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    x = np.asarray(requested_counts)
    axes[0].plot(
        x,
        [aggregate[count]["detected_mean"] for count in requested_counts],
        "o-",
        label="mean detected",
    )
    axes[0].plot(x, x, "--", color="gray", label="requested")
    axes[0].set_xlabel("requested feature cap")
    axes[0].set_ylabel("feature count")
    axes[0].legend()
    axes[1].plot(
        x,
        [aggregate[count]["coverage_mean"] for count in requested_counts],
        "o-",
    )
    axes[1].set_xlabel("requested feature cap")
    axes[1].set_ylabel("mean occupied buckets (%)")
    axes[1].set_ylim(0, 100)
    axes[2].plot(
        x,
        [aggregate[count]["quality_median_mean"] for count in requested_counts],
        "o-",
    )
    axes[2].set_xlabel("requested feature cap")
    axes[2].set_ylabel("mean per-frame median min-eigen quality")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(summary_path, dpi=160)
    plt.close(fig)

    eligible = [
        count
        for count in requested_counts
        if aggregate[count]["detected_mean"] >= 0.8 * count
        and aggregate[count]["coverage_mean"] >= 50.0
    ]
    recommendation = max(eligible) if eligible else min(
        requested_counts,
        key=lambda count: (
            -aggregate[count]["coverage_mean"],
            -aggregate[count]["detected_mean"],
        ),
    )
    report_path = output_dir / f"{stem}_experiment_report.md"
    report_path.write_text(
        _make_report(configuration, rows, aggregate, recommendation),
        encoding="utf-8",
    )
    return {
        "features": features_path,
        "measurements": csv_path,
        "configuration": metadata_path,
        "overlays": overlay_path,
        "spatial_buckets": bucket_plot_path,
        "comparison": summary_path,
        "report": report_path,
    }


def _make_report(
    configuration: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[int, Mapping[str, float]],
    recommendation: int,
) -> str:
    counts = configuration["requested_counts"]
    lines = [
        "# Stage 2A — APS Shi–Tomasi feature extraction",
        "",
        f"- Algorithm: OpenCV Shi–Tomasi `goodFeaturesToTrack`; Harris mode disabled.",
        f"- Selected APS frame indices: `{configuration['frame_indices']}`.",
        f"- Bucket grid: {configuration['bucket_grid'][0]} × {configuration['bucket_grid'][1]}.",
        "- Feature record: `id:int32, x:float32, y:float32, quality:float32, valid:uint8`.",
        "- IDs are unique within this experiment output. They do not establish identity across frames.",
        "- No event association, velocity, KLT, or tracking is performed.",
        "",
        "| Requested cap | Mean detected | Mean bucket coverage | Mean normalized entropy | Mean median quality |",
        "|---:|---:|---:|---:|---:|",
    ]
    for count in counts:
        result = aggregate[int(count)]
        lines.append(
            f"| {count} | {result['detected_mean']:.1f} | "
            f"{result['coverage_mean']:.1f}% | "
            f"{result['entropy_mean']:.3f} | "
            f"{result['quality_median_mean']:.6g} |"
        )
    lines += [
        "",
        f"**Recommended count for the next experiment: {recommendation}.**",
        "Selection rule: choose the highest requested cap with mean detected count "
        "at least 80% of the cap and mean occupancy of at least 50% of the spatial "
        "buckets. If no cap qualifies, choose the cap with the highest mean bucket "
        "coverage, then highest mean detected count. This is a development heuristic, "
        "not a tracking-performance result.",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| Quality level | {configuration['quality_level']} |",
        f"| Minimum feature distance | {configuration['min_feature_distance_px']} px |",
        f"| Block/window size | {configuration['block_size_px']} px |",
        f"| Bucket grid | {configuration['bucket_grid'][0]} × {configuration['bucket_grid'][1]} |",
        "",
        "Feature quality is OpenCV's minimum-eigenvalue corner response, sampled "
        "at each detected point. It is an image-response score, not a calibrated "
        "confidence probability.",
        "",
        "Plots: `davis_shapes_6dof_feature_overlays.png`, "
        "`davis_shapes_6dof_spatial_buckets.png`, and "
        "`davis_shapes_6dof_comparison.png`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare spatially bucketed Shi-Tomasi detection on DAVIS APS frames."
    )
    parser.add_argument("--recording", default="shapes_6dof")
    parser.add_argument("--frame-samples", type=int, default=5)
    parser.add_argument("--quality-level", type=float, default=0.01)
    parser.add_argument("--min-feature-distance", type=float, default=5.0)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--bucket-columns", type=int, default=8)
    parser.add_argument("--bucket-rows", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    config = DavisConfig(recording=args.recording)
    recording = load_davis(config)
    origins_s, _ = load_timestamp_origins(recording, config)
    windows = make_frame_windows(recording, origins_s)
    indices = default_frame_indices(len(windows) + 1, args.frame_samples)
    results, experiment_config = run_feature_count_experiment(
        windows,
        frame_indices=indices,
        requested_counts=REQUESTED_FEATURE_COUNTS,
        quality_level=args.quality_level,
        min_feature_distance=args.min_feature_distance,
        block_size=args.block_size,
        bucket_columns=args.bucket_columns,
        bucket_rows=args.bucket_rows,
    )
    frame_lookup = {index: get_aps_frame(windows, index)[0] for index in indices}
    outputs = save_experiment_outputs(
        results, experiment_config, args.output_dir, frame_lookup
    )
    print(f"Frames sampled: {indices}")
    print("Requested counts:", REQUESTED_FEATURE_COUNTS)
    for count in REQUESTED_FEATURE_COUNTS:
        subset = [r for r in results if r.requested_count == count]
        print(
            f"{count:4d}: detected mean="
            f"{np.mean([r.metrics['detected_count'] for r in subset]):.1f}, "
            f"bucket coverage mean="
            f"{np.mean([r.metrics['bucket_coverage_percent'] for r in subset]):.1f}%, "
            f"median quality mean="
            f"{np.mean([r.metrics['median_quality'] for r in subset]):.6g}"
        )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
