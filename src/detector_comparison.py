"""Stage 2B comparison of Shi-Tomasi and FAST on synchronized APS pairs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .aps_features import (
    FEATURE_DTYPE,
    REQUESTED_FEATURE_COUNTS,
    ShiTomasiConfig,
    _detect_bucket_candidates,
    _feature_metrics,
    _gray_u8,
    _select_candidates,
    default_frame_indices,
    get_aps_frame,
)
from .davis_inspection import DavisConfig, load_davis
from .frame_synchronization import (
    FrameWindows,
    load_timestamp_origins,
    make_frame_windows,
)

DETECTORS = ("shi_tomasi", "fast")
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "detector_comparison"
)
COMPARISON_DTYPE = np.dtype(
    [
        ("detector", "U16"),
        ("frame_index", np.int32),
        ("requested_count", np.int32),
        *FEATURE_DTYPE.descr,
    ]
)


@dataclass(frozen=True)
class DetectorConfig:
    quality_level: float = 0.01
    min_feature_distance_px: float = 5.0
    block_size_px: int = 3
    bucket_columns: int = 8
    bucket_rows: int = 6
    fast_threshold: int = 20
    fast_type: int = cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
    fast_nonmax_suppression: bool = True
    repeatability_radius_px: float = 3.0
    timing_repeats: int = 3

    def validate(self) -> None:
        if not 0 < self.quality_level <= 1:
            raise ValueError("quality_level must be in (0, 1]")
        if self.min_feature_distance_px < 0:
            raise ValueError("min_feature_distance_px must be non-negative")
        if self.block_size_px < 2:
            raise ValueError("block_size_px must be at least 2")
        if self.bucket_columns <= 0 or self.bucket_rows <= 0:
            raise ValueError("bucket grid dimensions must be positive")
        if not 0 <= self.fast_threshold <= 255:
            raise ValueError("fast_threshold must be within [0, 255]")
        if self.fast_type not in (
            cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,
            cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,
            cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,
        ):
            raise ValueError("fast_type must be a supported OpenCV FAST pattern")
        if self.repeatability_radius_px < 0:
            raise ValueError("repeatability_radius_px must be non-negative")
        if self.timing_repeats <= 0:
            raise ValueError("timing_repeats must be positive")


def default_consecutive_pair_indices(
    frame_count: int, pair_count: int = 5
) -> list[int]:
    """Choose evenly spaced pair starts; each selected pair is k and k+1."""
    if frame_count < 2:
        raise ValueError("At least two APS frames are required")
    return default_frame_indices(frame_count - 1, sample_count=pair_count)


def _fast_bucket_candidates(
    image: np.ndarray, max_candidates_per_bucket: int, config: DetectorConfig
) -> list[list[tuple[float, float, float]]]:
    gray = _gray_u8(image)
    height, width = gray.shape
    edges_x = np.linspace(0, width, config.bucket_columns + 1, dtype=np.int32)
    edges_y = np.linspace(0, height, config.bucket_rows + 1, dtype=np.int32)
    detector = cv2.FastFeatureDetector_create(
        threshold=config.fast_threshold,
        nonmaxSuppression=config.fast_nonmax_suppression,
        type=config.fast_type,
    )
    buckets: list[list[tuple[float, float, float]]] = [
        [] for _ in range(config.bucket_columns * config.bucket_rows)
    ]
    for row in range(config.bucket_rows):
        y0, y1 = int(edges_y[row]), int(edges_y[row + 1])
        for column in range(config.bucket_columns):
            x0, x1 = int(edges_x[column]), int(edges_x[column + 1])
            roi = gray[y0:y1, x0:x1]
            if roi.shape[0] < 7 or roi.shape[1] < 7:
                continue
            keypoints = detector.detect(roi, None)
            bucket_index = row * config.bucket_columns + column
            candidates = [
                (float(kp.pt[0] + x0), float(kp.pt[1] + y0), float(kp.response))
                for kp in keypoints
            ]
            candidates.sort(key=lambda point: (-point[2], point[1], point[0]))
            buckets[bucket_index] = candidates[:max_candidates_per_bucket]
    return buckets


def _get_candidates(
    image: np.ndarray,
    detector: str,
    max_candidates: int,
    config: DetectorConfig,
) -> list[list[tuple[float, float, float]]]:
    gray = _gray_u8(image)
    if detector == "shi_tomasi":
        shi_config = ShiTomasiConfig(
            max_features=max_candidates,
            quality_level=config.quality_level,
            min_feature_distance=config.min_feature_distance_px,
            block_size=config.block_size_px,
            bucket_columns=config.bucket_columns,
            bucket_rows=config.bucket_rows,
        )
        buckets, _ = _detect_bucket_candidates(gray, shi_config)
        return buckets
    if detector == "fast":
        return _fast_bucket_candidates(gray, max_candidates, config)
    raise ValueError(f"Unsupported detector: {detector}")


def _extract_from_candidates(
    image: np.ndarray,
    candidates: list[list[tuple[float, float, float]]],
    detector: str,
    requested_count: int,
    frame_index: int,
    frame_timestamp_us: int,
    first_feature_id: int,
    config: DetectorConfig,
):
    selection_config = ShiTomasiConfig(
        max_features=requested_count,
        quality_level=config.quality_level,
        min_feature_distance=config.min_feature_distance_px,
        block_size=config.block_size_px,
        bucket_columns=config.bucket_columns,
        bucket_rows=config.bucket_rows,
    )
    result = _select_candidates(
        image,
        selection_config,
        candidates,
        frame_index,
        frame_timestamp_us,
        first_feature_id,
    )
    metrics = dict(result.metrics)
    metrics["response_definition"] = (
        "Shi-Tomasi minimum eigenvalue" if detector == "shi_tomasi"
        else "OpenCV FAST keypoint response"
    )
    return result.features, metrics


def mutual_nearest_repeatability(
    first: np.ndarray, second: np.ndarray, radius_px: float = 3.0
) -> dict[str, float | int]:
    """Count mutual nearest coordinate pairs within radius (not optical flow)."""
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    points_a = np.asarray(first, dtype=np.float64).reshape(-1, 2)
    points_b = np.asarray(second, dtype=np.float64).reshape(-1, 2)
    denominator = min(len(points_a), len(points_b))
    if denominator == 0:
        return {
            "matched_count": 0,
            "denominator": denominator,
            "repeatability": 0.0,
            "mean_match_distance_px": 0.0,
        }
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    distance_ab, index_ab = tree_b.query(points_a, k=1)
    _, index_ba = tree_a.query(points_b, k=1)
    mutual = np.arange(len(points_a)) == index_ba[index_ab]
    accepted = mutual & (distance_ab <= radius_px)
    distances = distance_ab[accepted]
    matched = int(np.count_nonzero(accepted))
    return {
        "matched_count": matched,
        "denominator": int(denominator),
        "repeatability": float(matched / denominator),
        "mean_match_distance_px": float(np.mean(distances)) if matched else 0.0,
    }


def _sampled_pair_indices(
    windows: FrameWindows, pair_indices: Sequence[int] | None, pair_count: int
) -> list[int]:
    if pair_indices is None:
        pair_indices = default_consecutive_pair_indices(len(windows) + 1, pair_count)
    indices = [int(index) for index in pair_indices]
    if not indices:
        raise ValueError("At least one consecutive APS frame pair is required")
    if any(index < 0 or index >= len(windows) for index in indices):
        raise IndexError(f"Pair indices must be within [0, {len(windows) - 1}]")
    if len(set(indices)) != len(indices):
        raise ValueError("Pair indices must be unique")
    return indices


def compare_detectors(
    windows: FrameWindows,
    pair_indices: Sequence[int] | None = None,
    pair_count: int = 5,
    requested_counts: Sequence[int] = REQUESTED_FEATURE_COUNTS,
    config: DetectorConfig = DetectorConfig(),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[int, np.ndarray]]:
    """Run both detectors on identical consecutive APS pairs and measure outputs."""
    config.validate()
    pair_indices = _sampled_pair_indices(windows, pair_indices, pair_count)
    counts = [int(value) for value in requested_counts]
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("requested_counts must contain positive values")
    if len(set(counts)) != len(counts):
        raise ValueError("requested_counts must be unique")

    frame_indices = sorted({index for pair in pair_indices for index in (pair, pair + 1)})
    frames = {index: get_aps_frame(windows, index) for index in frame_indices}
    feature_rows: list[dict[str, Any]] = []
    repeatability_rows: list[dict[str, Any]] = []
    next_feature_id = 0
    max_count = max(counts)

    for frame_index in frame_indices:
        frame, frame_timestamp = frames[frame_index]
        gray = _gray_u8(frame)
        for detector in DETECTORS:
            _get_candidates(gray, detector, max_count, config)
            candidate_times = []
            candidates = None
            for _ in range(config.timing_repeats):
                start = time.perf_counter()
                candidates = _get_candidates(gray, detector, max_count, config)
                candidate_times.append((time.perf_counter() - start) * 1000)
            detection_ms = float(np.median(candidate_times))
            assert candidates is not None

            for count in counts:
                selection_times = []
                features = None
                metrics = None
                for _ in range(config.timing_repeats):
                    start = time.perf_counter()
                    features, metrics = _extract_from_candidates(
                        gray,
                        candidates,
                        detector,
                        count,
                        frame_index,
                        frame_timestamp,
                        next_feature_id,
                        config,
                    )
                    selection_times.append((time.perf_counter() - start) * 1000)
                assert features is not None and metrics is not None
                selection_ms = float(np.median(selection_times))
                feature_rows.append(
                    {
                        "detector": detector,
                        "frame_index": frame_index,
                        "frame_timestamp_us": frame_timestamp,
                        "requested_count": count,
                        "features": features,
                        "metrics": metrics,
                        "detection_time_ms": detection_ms,
                        "selection_time_ms": selection_ms,
                        "total_time_ms": detection_ms + selection_ms,
                    }
                )
                next_feature_id += len(features)

    result_lookup = {
        (row["detector"], row["frame_index"], row["requested_count"]): row
        for row in feature_rows
    }
    for pair_index in pair_indices:
        for detector in DETECTORS:
            for count in counts:
                first = result_lookup[(detector, pair_index, count)]["features"]
                second = result_lookup[(detector, pair_index + 1, count)]["features"]
                repeatability = mutual_nearest_repeatability(
                    np.column_stack((first["x"], first["y"])),
                    np.column_stack((second["x"], second["y"])),
                    radius_px=config.repeatability_radius_px,
                )
                repeatability_rows.append(
                    {
                        "detector": detector,
                        "frame_index": pair_index,
                        "next_frame_index": pair_index + 1,
                        "requested_count": count,
                        "first_detected": len(first),
                        "next_detected": len(second),
                        **repeatability,
                        "radius_px": config.repeatability_radius_px,
                    }
                )

    parameters = {
        "algorithm": "Stage 2B APS detector comparison",
        "timestamp_unit": "microseconds",
        "frame_indices": frame_indices,
        "consecutive_pair_indices": pair_indices,
        "requested_counts": counts,
        "spatial_bucket_grid": [config.bucket_columns, config.bucket_rows],
        "minimum_feature_distance_px": config.min_feature_distance_px,
        "shi_tomasi": {
            "quality_level": config.quality_level,
            "block_size_px": config.block_size_px,
            "quality_response": "cornerMinEigenVal minimum eigenvalue",
        },
        "fast": {
            "threshold": config.fast_threshold,
            "pattern": "TYPE_9_16",
            "nonmax_suppression": config.fast_nonmax_suppression,
            "quality_response": "OpenCV KeyPoint.response",
        },
        "repeatability": {
            "method": "mutual nearest feature-coordinate matches",
            "radius_px": config.repeatability_radius_px,
            "warning": (
                "No geometric motion compensation or KLT is applied; raw-coordinate "
                "repeatability is sensitive to scene/camera motion."
            ),
        },
        "timing": {
            "repeats": config.timing_repeats,
            "statistic": "median wall-clock milliseconds",
            "includes": "candidate detection plus per-cap spatial selection",
            "excludes": "dataset load, image decode, synchronization, and plotting",
        },
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "feature_record_dtype": [
            {"name": name, "dtype": str(FEATURE_DTYPE.fields[name][0])}
            for name in FEATURE_DTYPE.names
        ],
    }
    return feature_rows, repeatability_rows, parameters, {
        index: value[0] for index, value in frames.items()
    }


def _aggregate(
    feature_rows: Sequence[Mapping[str, Any]],
    repeatability_rows: Sequence[Mapping[str, Any]],
    requested_counts: Sequence[int],
) -> dict[str, dict[int, dict[str, float]]]:
    summary: dict[str, dict[int, dict[str, float]]] = {}
    for detector in DETECTORS:
        summary[detector] = {}
        for count in requested_counts:
            rows = [
                row for row in feature_rows
                if row["detector"] == detector and row["requested_count"] == count
            ]
            repeats = [
                row for row in repeatability_rows
                if row["detector"] == detector and row["requested_count"] == count
            ]
            summary[detector][int(count)] = {
                "mean_detected": float(np.mean([len(row["features"]) for row in rows])),
                "mean_bucket_coverage_percent": float(
                    np.mean([row["metrics"]["bucket_coverage_percent"] for row in rows])
                ),
                "mean_normalized_bucket_entropy": float(
                    np.mean([row["metrics"]["normalized_bucket_entropy"] for row in rows])
                ),
                "mean_native_response": float(
                    np.mean([row["metrics"]["mean_quality"] for row in rows])
                ),
                "mean_median_native_response": float(
                    np.mean([row["metrics"]["median_quality"] for row in rows])
                ),
                "mean_detection_time_ms": float(
                    np.mean([row["detection_time_ms"] for row in rows])
                ),
                "mean_selection_time_ms": float(
                    np.mean([row["selection_time_ms"] for row in rows])
                ),
                "mean_total_time_ms": float(
                    np.mean([row["total_time_ms"] for row in rows])
                ),
                "mean_repeatability_percent": 100.0 * float(
                    np.mean([row["repeatability"] for row in repeats])
                ),
            }
    return summary


def _recommend_detector(
    summary: Mapping[str, Mapping[int, Mapping[str, float]]],
    operating_count: int = 500,
) -> tuple[str, str]:
    shi = summary["shi_tomasi"][operating_count]
    fast = summary["fast"][operating_count]
    eligible = {
        detector: values
        for detector, values in (("shi_tomasi", shi), ("fast", fast))
        if values["mean_detected"] >= 0.8 * operating_count
        and values["mean_bucket_coverage_percent"] >= 50
    }
    if not eligible:
        detector = max(
            ("shi_tomasi", "fast"),
            key=lambda name: (
                summary[name][operating_count]["mean_repeatability_percent"],
                summary[name][operating_count]["mean_bucket_coverage_percent"],
            ),
        )
        return detector, "Neither detector met count/coverage gates; selected by repeatability, then coverage."
    if len(eligible) == 1:
        detector = next(iter(eligible))
        return detector, "Only this detector met the count attainment and spatial coverage gates."

    repeatability_difference = (
        eligible["shi_tomasi"]["mean_repeatability_percent"]
        - eligible["fast"]["mean_repeatability_percent"]
    )
    if abs(repeatability_difference) > 2.0:
        detector = "shi_tomasi" if repeatability_difference > 0 else "fast"
        return detector, "Selected for higher raw-coordinate repeatability (>2 percentage-point difference)."
    coverage_difference = (
        eligible["shi_tomasi"]["mean_bucket_coverage_percent"]
        - eligible["fast"]["mean_bucket_coverage_percent"]
    )
    if abs(coverage_difference) > 2.0:
        detector = "shi_tomasi" if coverage_difference > 0 else "fast"
        return detector, "Repeatability was within 2 points; selected for higher bucket coverage."
    detector = min(
        eligible,
        key=lambda name: eligible[name]["mean_total_time_ms"],
    )
    return detector, "Repeatability and coverage were within 2 points; selected for lower measured runtime."


def save_comparison_outputs(
    feature_rows: Sequence[Mapping[str, Any]],
    repeatability_rows: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
    frame_lookup: Mapping[int, np.ndarray],
    output_dir: Path,
    operating_count: int = 500,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    if not feature_rows or not repeatability_rows:
        raise ValueError("Comparison results are empty")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = [int(count) for count in parameters["requested_counts"]]
    pairs = parameters["consecutive_pair_indices"]
    example_frame = int(pairs[len(pairs) // 2])
    if example_frame not in frame_lookup:
        raise ValueError(f"Missing representative APS frame {example_frame}")
    summary = _aggregate(feature_rows, repeatability_rows, counts)

    record_rows = []
    for row in feature_rows:
        for feature in row["features"]:
            record_rows.append(
                (
                    row["detector"],
                    row["frame_index"],
                    row["requested_count"],
                    int(feature["id"]),
                    float(feature["x"]),
                    float(feature["y"]),
                    float(feature["quality"]),
                    int(feature["valid"]),
                )
            )
    archive_dtype = np.dtype(
        [
            ("detector", "U16"),
            ("frame_index", np.int32),
            ("requested_count", np.int32),
            *FEATURE_DTYPE.descr,
        ]
    )
    archive = np.asarray(record_rows, dtype=archive_dtype)
    archive_path = output_dir / "davis_detector_comparison_features.npz"
    np.savez_compressed(
        archive_path,
        features=archive,
        frame_indices=np.asarray(parameters["frame_indices"], dtype=np.int32),
        pair_indices=np.asarray(parameters["consecutive_pair_indices"], dtype=np.int32),
        requested_counts=np.asarray(counts, dtype=np.int32),
    )

    measurement_path = output_dir / "davis_detector_comparison_measurements.csv"
    feature_fields = (
        "detector",
        "frame_index",
        "frame_timestamp_us",
        "requested_count",
        "detected_count",
        "count_shortfall",
        "occupied_buckets",
        "total_buckets",
        "bucket_coverage_percent",
        "normalized_bucket_entropy",
        "mean_quality",
        "median_quality",
        "min_quality",
        "max_quality",
        "response_definition",
        "detection_time_ms",
        "selection_time_ms",
        "total_time_ms",
    )
    with measurement_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=feature_fields)
        writer.writeheader()
        for row in feature_rows:
            writer.writerow(
                {
                    "detector": row["detector"],
                    "frame_index": row["frame_index"],
                    "frame_timestamp_us": row["frame_timestamp_us"],
                    "requested_count": row["requested_count"],
                    "detected_count": len(row["features"]),
                    **{
                        key: row["metrics"][key]
                        for key in feature_fields
                        if key in row["metrics"]
                    },
                    "response_definition": row["metrics"]["response_definition"],
                    "detection_time_ms": f"{row['detection_time_ms']:.6f}",
                    "selection_time_ms": f"{row['selection_time_ms']:.6f}",
                    "total_time_ms": f"{row['total_time_ms']:.6f}",
                }
            )

    repeatability_path = output_dir / "davis_detector_comparison_repeatability.csv"
    repeat_fields = (
        "detector",
        "frame_index",
        "next_frame_index",
        "requested_count",
        "first_detected",
        "next_detected",
        "matched_count",
        "denominator",
        "repeatability",
        "mean_match_distance_px",
        "radius_px",
    )
    with repeatability_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=repeat_fields)
        writer.writeheader()
        writer.writerows(repeatability_rows)

    config_path = output_dir / "davis_detector_comparison_configuration.json"
    config_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "parameters": dict(parameters),
                "summary": summary,
                "recommendation": _recommend_detector(summary, operating_count),
                "archive_record_count": len(archive),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    columns = min(3, len(counts))
    rows = int(np.ceil(len(counts) / columns))
    fig, axes = plt.subplots(
        2, len(counts), figsize=(4.2 * len(counts), 7.2), squeeze=False
    )
    for col, count in enumerate(counts):
        for row_index, detector in enumerate(DETECTORS):
            data = next(
                row
                for row in feature_rows
                if row["detector"] == detector
                and row["frame_index"] == example_frame
                and row["requested_count"] == count
            )
            ax = axes[row_index, col]
            ax.imshow(frame_lookup[example_frame], cmap="gray", vmin=0, vmax=255)
            ax.scatter(
                data["features"]["x"],
                data["features"]["y"],
                s=12,
                marker="+",
                linewidths=0.8,
                color="tab:red",
            )
            ax.set_title(
                f"{detector.replace('_', ' ')} | {count} req / "
                f"{len(data['features'])} det"
            )
            ax.set_axis_off()
    fig.suptitle(
        f"Same APS frame {example_frame} and parameters; visual comparison only",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    overlay_path = output_dir / "davis_detector_comparison_overlays.png"
    fig.savefig(overlay_path, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(counts), figsize=(3.4 * len(counts), 6))
    for col, count in enumerate(counts):
        for row_index, detector in enumerate(DETECTORS):
            row = next(
                item for item in feature_rows
                if item["detector"] == detector
                and item["frame_index"] == example_frame
                and item["requested_count"] == count
            )
            image = axes[row_index, col].imshow(
                row["metrics"]["bucket_counts"], cmap="viridis", vmin=0
            )
            axes[row_index, col].set_title(
                f"{detector.replace('_', ' ')}; cap {count}"
            )
            axes[row_index, col].set_xticks([])
            axes[row_index, col].set_yticks([])
            fig.colorbar(image, ax=axes[row_index, col], shrink=0.75)
    fig.tight_layout()
    spatial_path = output_dir / "davis_detector_comparison_spatial.png"
    fig.savefig(spatial_path, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for detector in DETECTORS:
        values = summary[detector]
        axes[0].plot(
            counts,
            [values[count]["mean_repeatability_percent"] for count in counts],
            "o-",
            label=detector.replace("_", " "),
        )
        axes[1].plot(
            counts,
            [values[count]["mean_total_time_ms"] for count in counts],
            "o-",
            label=detector.replace("_", " "),
        )
        axes[2].plot(
            counts,
            [values[count]["mean_bucket_coverage_percent"] for count in counts],
            "o-",
            label=detector.replace("_", " "),
        )
    axes[0].set_ylabel("raw-coordinate repeatability (%)")
    axes[1].set_ylabel("median detection + selection time (ms)")
    axes[2].set_ylabel("occupied spatial buckets (%)")
    for ax in axes:
        ax.set_xlabel("requested feature cap")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    metrics_path = output_dir / "davis_detector_comparison_metrics.png"
    fig.savefig(metrics_path, dpi=160)
    plt.close(fig)

    report_path = output_dir / "davis_detector_comparison_report.md"
    report_path.write_text(
        _make_report(parameters, summary, operating_count), encoding="utf-8"
    )
    return {
        "features": archive_path,
        "measurements": measurement_path,
        "repeatability": repeatability_path,
        "configuration": config_path,
        "overlays": overlay_path,
        "spatial_distribution": spatial_path,
        "metrics_plot": metrics_path,
        "report": report_path,
    }


def _make_report(
    parameters: Mapping[str, Any],
    summary: Mapping[str, Mapping[int, Mapping[str, float]]],
    operating_count: int,
) -> str:
    recommendation, rationale = _recommend_detector(summary, operating_count)
    lines = [
        "# Stage 2B — Shi-Tomasi vs FAST on DAVIS APS",
        "",
        f"- Consecutive frame-pair starts: `{parameters['consecutive_pair_indices']}`.",
        f"- Requested feature caps: `{parameters['requested_counts']}`.",
        f"- Spatial bucket grid: {parameters['spatial_bucket_grid'][0]} × "
        f"{parameters['spatial_bucket_grid'][1]}; minimum spacing: "
        f"{parameters['minimum_feature_distance_px']} px.",
        f"- Shi-Tomasi: quality level {parameters['shi_tomasi']['quality_level']}, "
        f"block size {parameters['shi_tomasi']['block_size_px']} px.",
        f"- FAST: threshold {parameters['fast']['threshold']}, "
        f"{parameters['fast']['pattern']}, nonmax suppression "
        f"{parameters['fast']['nonmax_suppression']}.",
        f"- Repeatability: mutual nearest neighbors within "
        f"{parameters['repeatability']['radius_px']} px on unwarped consecutive images.",
        f"- Runtime: median over {parameters['timing']['repeats']} repeats; "
        f"{parameters['timing']['includes']}.",
        "",
        "## Quantitative results",
        "",
        "| Detector | Cap | Mean detected | Mean bucket coverage | Mean repeatability | Mean runtime (ms) | Mean median native response |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for count in parameters["requested_counts"]:
        for detector in DETECTORS:
            item = summary[detector][int(count)]
            lines.append(
                f"| {detector.replace('_', ' ')} | {count} | "
                f"{item['mean_detected']:.1f} | "
                f"{item['mean_bucket_coverage_percent']:.1f}% | "
                f"{item['mean_repeatability_percent']:.1f}% | "
                f"{item['mean_total_time_ms']:.3f} | "
                f"{item['mean_median_native_response']:.6g} |"
            )
    chosen = summary[recommendation][operating_count]
    lines.extend(
        [
            "",
            f"## Recommendation: {recommendation.replace('_', ' ')}",
            "",
            f"At the {operating_count}-feature operating point, {recommendation.replace('_', ' ')} "
            f"detected {chosen['mean_detected']:.1f} features on average, achieved "
            f"{chosen['mean_bucket_coverage_percent']:.1f}% mean bucket coverage, "
            f"{chosen['mean_repeatability_percent']:.1f}% raw-coordinate repeatability, "
            f"and took {chosen['mean_total_time_ms']:.3f} ms per frame on average. "
            f"{rationale}",
            "",
            "Native detector-response numbers are **not directly comparable**: "
            "Shi-Tomasi reports a minimum-eigenvalue score, while FAST reports "
            "OpenCV's FAST keypoint response.",
            "",
            "Raw-coordinate repeatability is a detector-stability proxy, not "
            "tracking accuracy. No geometric motion compensation, optical flow, "
            "or KLT is applied, so camera/object motion can lower both scores.",
            "",
            "Visual overlays and bucket maps are illustrative only; the recommendation "
            "uses the measurements above, not visual preference.",
            "",
            "No event association, velocity, or event-driven tracking is implemented.",
            "",
            "Artifacts: `davis_detector_comparison_overlays.png`, "
            "`davis_detector_comparison_spatial.png`, "
            "`davis_detector_comparison_metrics.png`, "
            "`davis_detector_comparison_measurements.csv`, and "
            "`davis_detector_comparison_repeatability.csv`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Shi-Tomasi and FAST on identical consecutive DAVIS APS pairs."
    )
    parser.add_argument("--recording", default="shapes_6dof")
    parser.add_argument("--pair-samples", type=int, default=5)
    parser.add_argument("--repeatability-radius", type=float, default=3.0)
    parser.add_argument("--fast-threshold", type=int, default=20)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.pair_samples <= 0:
        parser.error("--pair-samples must be positive")

    config = DavisConfig(recording=args.recording)
    recording = load_davis(config)
    origins_s, _ = load_timestamp_origins(recording, config)
    windows = make_frame_windows(recording, origins_s)
    detector_config = DetectorConfig(
        repeatability_radius_px=args.repeatability_radius,
        fast_threshold=args.fast_threshold,
        timing_repeats=args.timing_repeats,
    )
    feature_rows, repeatability_rows, parameters, frame_lookup = compare_detectors(
        windows,
        pair_count=args.pair_samples,
        config=detector_config,
    )
    outputs = save_comparison_outputs(
        feature_rows,
        repeatability_rows,
        parameters,
        frame_lookup,
        args.output_dir,
    )
    summary = _aggregate(feature_rows, repeatability_rows, parameters["requested_counts"])
    for count in parameters["requested_counts"]:
        print(f"Feature cap {count}")
        for detector in DETECTORS:
            values = summary[detector][count]
            print(
                f"  {detector}: detected={values['mean_detected']:.1f}, "
                f"coverage={values['mean_bucket_coverage_percent']:.1f}%, "
                f"repeatability={values['mean_repeatability_percent']:.1f}%, "
                f"time={values['mean_total_time_ms']:.3f} ms, "
                f"median native response={values['mean_median_native_response']:.6g}"
            )
    recommendation = _recommend_detector(summary)
    print(f"Recommendation at 500 features: {recommendation[0]} ({recommendation[1]})")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
