"""Stage 7 comparison of frame-only, event-driven, and hybrid tracking."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .aps_features import ShiTomasiConfig, extract_shi_tomasi_features
from .asynchronous_feature_tracker import (
    PREDICTED_DTYPE,
    match_predictions_to_references,
)
from .davis_inspection import DavisConfig, load_davis
from .event_state_update import run_asynchronous_updates
from .feature_state_memory import FeatureStateMemory, initialize_from_aps_features
from .frame_synchronization import FrameWindow, load_timestamp_origins, make_frame_windows
from .spatial_grid import SpatialGridConfig

SYSTEMS = ("frame_only", "event_driven", "hybrid")
SYSTEM_LABELS = {
    "frame_only": "Frame-only",
    "event_driven": "Event-driven",
    "hybrid": "Hybrid",
}
COMMON_METRICS = (
    "epe_px",
    "track_lifetime_mean_us",
    "features_successfully_tracked_mean",
    "failure_rate",
    "processing_time_ms",
    "update_latency_mean_us",
    "events_per_useful_update",
    "state_updates",
    "events_processed",
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "three_way_comparison"
)
FRAME_ONLY = np.int8(0)
EVENT_DRIVEN = np.int8(1)
HYBRID = np.int8(2)


@dataclass(frozen=True)
class ComparisonConfig:
    interval_start: int = 678
    interval_count: int = 5
    feature_count: int = 500
    slot_count: int = 500
    quality_level: float = 0.01
    min_feature_distance_px: float = 5.0
    block_size_px: int = 3
    bucket_columns: int = 8
    bucket_rows: int = 6
    association_radius_px: float = 8.0
    association_time_window_us: float = 50_000.0
    grid_cell_size_px: int = 16
    use_spatial_grid: bool = False
    alpha: float = 0.5
    beta: float = 0.1
    minimum_delta_t_us: float = 1.0
    confidence_penalty: float = 0.05
    reference_match_radius_px: float = 15.0
    klt_window_px: int = 21
    klt_max_pyramid_level: int = 3
    klt_max_iterations: int = 30
    klt_epsilon: float = 0.01

    def validate(self) -> None:
        if self.interval_start < 0 or self.interval_count <= 0:
            raise ValueError("interval_start must be non-negative and interval_count positive")
        if self.feature_count <= 0 or self.slot_count < self.feature_count:
            raise ValueError("slot_count must be at least the positive feature_count")
        if not np.isfinite(self.quality_level) or not 0 < self.quality_level <= 1:
            raise ValueError("quality_level must be finite and within (0, 1]")
        if not np.isfinite(self.min_feature_distance_px) or self.min_feature_distance_px < 0:
            raise ValueError("min_feature_distance_px must be finite and non-negative")
        if self.block_size_px < 2 or self.bucket_columns <= 0 or self.bucket_rows <= 0:
            raise ValueError("block size and bucket dimensions must be positive")
        if (
            isinstance(self.grid_cell_size_px, bool)
            or not isinstance(self.grid_cell_size_px, (int, np.integer))
            or self.grid_cell_size_px <= 0
        ):
            raise ValueError("grid_cell_size_px must be a positive integer")
        for value, name in (
            (self.association_radius_px, "association_radius_px"),
            (self.association_time_window_us, "association_time_window_us"),
            (self.minimum_delta_t_us, "minimum_delta_t_us"),
            (self.reference_match_radius_px, "reference_match_radius_px"),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.alpha <= 1 or not 0 <= self.beta <= 1:
            raise ValueError("alpha and beta must be within [0, 1]")
        if not 0 <= self.confidence_penalty <= 1:
            raise ValueError("confidence_penalty must be within [0, 1]")
        if self.klt_window_px < 3 or self.klt_window_px % 2 == 0:
            raise ValueError("klt_window_px must be odd and at least 3")
        if self.klt_max_pyramid_level < 0 or self.klt_max_iterations <= 0:
            raise ValueError("KLT pyramid level and iteration count are invalid")
        if not np.isfinite(self.klt_epsilon) or self.klt_epsilon <= 0:
            raise ValueError("klt_epsilon must be finite and positive")


@dataclass(frozen=True)
class ComparisonResult:
    pair_rows: np.ndarray
    lifetimes: np.ndarray
    epe_samples: Mapping[str, np.ndarray]
    survival: Mapping[str, np.ndarray]
    initial_features: np.ndarray
    reference_features: tuple[np.ndarray, ...]
    diagnostics: Mapping[str, Mapping[str, Any]]
    checks: Mapping[str, bool]


PAIR_DTYPE = np.dtype(
    [
        ("system", "U16"),
        ("interval_index", np.int32),
        ("frame_k_common_us", np.int64),
        ("frame_k_plus_1_common_us", np.int64),
        ("initial_features", np.int32),
        ("features_tracked", np.int32),
        ("features_successfully_tracked", np.int32),
        ("reference_matches", np.int32),
        ("epe_px", np.float64),
        ("failure_rate", np.float64),
        ("processing_time_ms", np.float64),
        ("update_latency_mean_us", np.float64),
        ("events_processed", np.int64),
        ("state_updates", np.int64),
        ("events_per_useful_update", np.float64),
    ]
)
LIFETIME_DTYPE = np.dtype(
    [
        ("system", "U16"),
        ("feature_id", np.int32),
        ("lifetime_intervals", np.int32),
        ("lifetime_us", np.int64),
    ]
)


def _detect(frame: np.ndarray, config: ComparisonConfig, frame_index: int, timestamp: int) -> np.ndarray:
    return extract_shi_tomasi_features(
        frame,
        ShiTomasiConfig(
            max_features=config.feature_count,
            quality_level=config.quality_level,
            min_feature_distance=config.min_feature_distance_px,
            block_size=config.block_size_px,
            bucket_columns=config.bucket_columns,
            bucket_rows=config.bucket_rows,
        ),
        frame_index=frame_index,
        frame_timestamp_us=timestamp,
    ).features


def _predictions_from_points(
    feature_ids: np.ndarray,
    slots: np.ndarray,
    xy: np.ndarray,
    timestamp_us: int,
) -> np.ndarray:
    predictions = np.zeros(len(feature_ids), dtype=PREDICTED_DTYPE)
    for index, (feature_id, slot, point) in enumerate(zip(feature_ids, slots, xy)):
        predictions[index] = (
            int(feature_id),
            int(slot),
            float(point[0]),
            float(point[1]),
            0.0,
            0.0,
            1.0,
            timestamp_us,
        )
    return predictions


def _evaluation(
    predictions: np.ndarray,
    references: np.ndarray,
    match_radius_px: float,
) -> tuple[np.ndarray, dict[int, float]]:
    matches = match_predictions_to_references(
        predictions, references, match_radius_px
    )
    errors = {
        int(row["feature_id"]): float(row["pixel_error"])
        for row in matches
        if row["matched"]
    }
    return matches, errors


def compare_three_systems(
    windows: Sequence[FrameWindow],
    config: ComparisonConfig = ComparisonConfig(),
) -> ComparisonResult:
    """Run all systems on identical contiguous APS frames and feature records."""
    config.validate()
    if len(windows) != config.interval_count:
        raise ValueError("windows length must equal configured interval_count")
    if not windows:
        raise ValueError("at least one APS interval is required")
    for previous, following in zip(windows, windows[1:]):
        if (
            previous.index + 1 != following.index
            or previous.frame_k_plus_1_common_time_us
            != following.frame_k_common_time_us
        ):
            raise ValueError("windows must be contiguous consecutive APS intervals")

    first_window = windows[0]
    pair_count = len(windows)
    detection_started = time.perf_counter_ns()
    initial_features = _detect(
        first_window.frame_k,
        config,
        first_window.index,
        first_window.frame_k_common_time_us,
    )
    initial_detection_ms = (time.perf_counter_ns() - detection_started) / 1_000_000
    if len(initial_features) > config.slot_count:
        raise BufferError("initial detections exceed configured FeatureState slots")
    reference_detection_ms: list[float] = []
    references_list: list[np.ndarray] = []
    for window in windows:
        detection_started = time.perf_counter_ns()
        references_list.append(_detect(
            window.frame_k_plus_1,
            config,
            window.index + 1,
            window.frame_k_plus_1_common_time_us,
        ))
        reference_detection_ms.append(
            (time.perf_counter_ns() - detection_started) / 1_000_000
        )
    references = tuple(references_list)
    shared_initial_xy = np.column_stack(
        (initial_features["x"], initial_features["y"])
    ).astype(np.float32)
    initial_ids = initial_features["id"].astype(np.int32, copy=True)
    initial_slots = np.arange(len(initial_features), dtype=np.int32)
    interval_durations = np.asarray(
        [
            window.frame_k_plus_1_common_time_us - window.frame_k_common_time_us
            for window in windows
        ],
        dtype=np.int64,
    )

    rows: list[tuple[Any, ...]] = []
    epe_samples: dict[str, list[float]] = {system: [] for system in SYSTEMS}
    survival: dict[str, list[int]] = {system: [] for system in SYSTEMS}
    lifetime_steps: dict[str, np.ndarray] = {
        system: np.zeros(len(initial_features), dtype=np.int32)
        for system in SYSTEMS
    }
    ever_failed = {
        system: np.zeros(len(initial_features), dtype=bool) for system in SYSTEMS
    }

    total_events = {system: 0 for system in SYSTEMS}
    total_state_updates = {system: 0 for system in SYSTEMS}
    total_async_updates = {system: 0 for system in SYSTEMS}
    total_candidates = {system: 0 for system in SYSTEMS}
    total_distance_calculations = {system: 0 for system in SYSTEMS}
    update_latencies: dict[str, list[float]] = {system: [] for system in SYSTEMS}
    tracked_sum = {system: 0 for system in SYSTEMS}
    useful_sum = {system: 0 for system in SYSTEMS}
    initial_count = len(initial_features)
    identity_to_index = {int(feature_id): i for i, feature_id in enumerate(initial_ids)}

    # System A: KLT carries the exact common initial Shi-Tomasi points across frames.
    frame_points = shared_initial_xy.reshape(-1, 1, 2).copy()
    klt_initial_points = frame_points.reshape(-1, 2).copy()
    frame_valid = np.ones(initial_count, dtype=bool)
    for interval_offset, window in enumerate(windows):
        pair_started = time.perf_counter_ns()
        operation_started = time.perf_counter_ns()
        klt_latency_us = 0.0
        indices = np.flatnonzero(frame_valid)
        if len(indices):
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                window.frame_k,
                window.frame_k_plus_1,
                frame_points[indices],
                None,
                winSize=(config.klt_window_px, config.klt_window_px),
                maxLevel=config.klt_max_pyramid_level,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    config.klt_max_iterations,
                    config.klt_epsilon,
                ),
            )
            klt_latency_us = (
                time.perf_counter_ns() - operation_started
            ) / 1000.0
            if next_points is None or status is None:
                raise RuntimeError("KLT returned no point or status output")
            xy = next_points.reshape(-1, 2)
            height, width = window.frame_k_plus_1.shape[:2]
            local_valid = status.reshape(-1).astype(bool)
            local_valid &= np.isfinite(xy).all(axis=1)
            local_valid &= (
                (xy[:, 0] >= 0)
                & (xy[:, 0] < width)
                & (xy[:, 1] >= 0)
                & (xy[:, 1] < height)
            )
            frame_points[indices[local_valid], 0, :] = xy[local_valid]
            frame_valid[indices[~local_valid]] = False
        predictions = _predictions_from_points(
            initial_ids[frame_valid],
            initial_slots[frame_valid],
            frame_points[frame_valid, 0, :],
            window.frame_k_plus_1_common_time_us,
        )
        matches, errors = _evaluation(
            predictions, references[interval_offset], config.reference_match_radius_px
        )
        matched_ids = set(errors)
        for feature_id, index in identity_to_index.items():
            if not ever_failed["frame_only"][index]:
                if feature_id in matched_ids:
                    lifetime_steps["frame_only"][index] += 1
                else:
                    ever_failed["frame_only"][index] = True
        count_matches = len(matched_ids)
        epe_samples["frame_only"].extend(errors.values())
        survival["frame_only"].append(
            int(np.count_nonzero(~ever_failed["frame_only"]))
        )
        update_latencies["frame_only"].append(klt_latency_us)
        tracked = int(np.count_nonzero(frame_valid))
        tracked_sum["frame_only"] += tracked
        useful_sum["frame_only"] += count_matches
        rows.append(
            (
                "frame_only",
                window.index,
                window.frame_k_common_time_us,
                window.frame_k_plus_1_common_time_us,
                initial_count,
                tracked,
                count_matches,
                count_matches,
                float(np.mean(list(errors.values()))) if errors else np.nan,
                1.0 - count_matches / initial_count if initial_count else 0.0,
                (time.perf_counter_ns() - pair_started) / 1_000_000,
                klt_latency_us,
                0,
                tracked,
                np.nan,
            )
        )

    # Systems B/C have identical starting states and consume the same raw event windows.
    memories = {}
    initial_state_locations_match = True
    for system in ("event_driven", "hybrid"):
        memory = initialize_from_aps_features(
            initial_features,
            first_window.frame_k_common_time_us,
            config.slot_count,
        )
        memories[system] = memory
        active = memory.get_active()
        initial_state_locations_match &= np.array_equal(
            np.column_stack((active["x"], active["y"])),
            shared_initial_xy,
        )

    for interval_offset, window in enumerate(windows):
        for system in ("event_driven", "hybrid"):
            memory = memories[system]
            pair_started = time.perf_counter_ns()
            events = window.events_k
            image_height, image_width = window.frame_k.shape[:2]
            evolution, events_per_feature, update_metrics = run_asynchronous_updates(
                events,
                memory,
                config.association_radius_px,
                config.association_time_window_us,
                alpha=config.alpha,
                beta=config.beta,
                minimum_delta_t_us=config.minimum_delta_t_us,
                confidence_penalty=config.confidence_penalty,
                event_timestamp_offset_us=window.event_timestamp_offset_us,
                spatial_grid=(
                    SpatialGridConfig(
                        image_width,
                        image_height,
                        config.grid_cell_size_px,
                        config.grid_cell_size_px,
                    )
                    if config.use_spatial_grid
                    else None
                ),
            )
            final_states = memory.storage.copy()
            valid_slots = np.flatnonzero(final_states["valid"] != 0)
            predictions = np.zeros(len(valid_slots), dtype=PREDICTED_DTYPE)
            for prediction_index, slot in enumerate(valid_slots):
                state = final_states[slot]
                dt = (
                    window.frame_k_plus_1_common_time_us
                    - int(state["timestamp"])
                )
                if dt < 0:
                    raise ValueError("state timestamp is later than APS evaluation time")
                predictions[prediction_index] = (
                    int(state["id"]),
                    int(slot),
                    float(state["x"]) + float(state["vx"]) * dt,
                    float(state["y"]) + float(state["vy"]) * dt,
                    float(state["vx"]),
                    float(state["vy"]),
                    float(state["confidence"]),
                    window.frame_k_plus_1_common_time_us,
                )
            matches, errors = _evaluation(
                predictions,
                references[interval_offset],
                config.reference_match_radius_px,
            )
            matched_rows = matches[matches["matched"] != 0]
            matched_ids = set(errors)
            for feature_id, index in identity_to_index.items():
                if not ever_failed[system][index]:
                    if feature_id in matched_ids:
                        lifetime_steps[system][index] += 1
                    else:
                        ever_failed[system][index] = True

            if system == "hybrid":
                # The chosen frame correction: matched tracks snap to reference corners.
                corrections = 0
                for match in matched_rows:
                    slot = int(match["slot"])
                    memory.update(
                        slot,
                        x=float(match["reference_x"]),
                        y=float(match["reference_y"]),
                        timestamp=window.frame_k_plus_1_common_time_us,
                    )
                    corrections += 1
            else:
                corrections = 0

            epe_samples[system].extend(errors.values())
            survival[system].append(
                int(np.count_nonzero(~ever_failed[system]))
            )
            events_count = len(events)
            update_count = int(update_metrics["successful_updates"])
            total_events[system] += events_count
            total_async_updates[system] += update_count
            total_candidates[system] += int(
                update_metrics["candidate_features_total"]
            )
            total_distance_calculations[system] += int(
                update_metrics["distance_calculations_total"]
            )
            total_state_updates[system] += update_count + corrections
            tracked = len(valid_slots)
            tracked_sum[system] += tracked
            useful_sum[system] += len(matched_ids)
            update_latencies[system].extend(
                evolution["update_latency_us"].astype(np.float64).tolist()
            )
            pair_elapsed_ms = (time.perf_counter_ns() - pair_started) / 1_000_000
            rows.append(
                (
                    system,
                    window.index,
                    window.frame_k_common_time_us,
                    window.frame_k_plus_1_common_time_us,
                    initial_count,
                    tracked,
                    len(matched_ids),
                    len(matched_rows),
                    float(np.mean(list(errors.values()))) if errors else np.nan,
                    1.0 - len(matched_ids) / initial_count if initial_count else 0.0,
                    pair_elapsed_ms,
                    float(np.mean(evolution["update_latency_us"]))
                    if len(evolution)
                    else 0.0,
                    events_count,
                    update_count + corrections,
                    events_count / update_count if update_count else np.inf,
                )
            )

    pair_rows = np.asarray(rows, dtype=PAIR_DTYPE)
    for row in pair_rows:
        interval_offset = int(row["interval_index"]) - config.interval_start
        row["processing_time_ms"] += reference_detection_ms[interval_offset]
        if interval_offset == 0:
            row["processing_time_ms"] += initial_detection_ms
    lifetime_rows: list[tuple[str, int, int, int]] = []
    for system in SYSTEMS:
        for index, feature_id in enumerate(initial_ids):
            lifetime_intervals = int(lifetime_steps[system][index])
            lifetime_us = int(np.sum(interval_durations[:lifetime_intervals]))
            lifetime_rows.append(
                (system, int(feature_id), lifetime_intervals, lifetime_us)
            )
    lifetimes = np.asarray(lifetime_rows, dtype=LIFETIME_DTYPE)

    active_track_lifetime = {
        system: float(np.mean(lifetimes["lifetime_us"][lifetimes["system"] == system]))
        if initial_count
        else 0.0
        for system in SYSTEMS
    }
    useful_updates = {
        "frame_only": tracked_sum["frame_only"],
        "event_driven": useful_sum["event_driven"],
        "hybrid": useful_sum["hybrid"],
    }
    per_system_diagnostics: dict[str, dict[str, Any]] = {}
    for system in SYSTEMS:
        subset = pair_rows[pair_rows["system"] == system]
        all_errors = np.asarray(epe_samples[system], dtype=np.float64)
        per_system_diagnostics[system] = {
            "epe_px": float(np.mean(all_errors)) if len(all_errors) else None,
            "track_lifetime_mean_us": active_track_lifetime[system],
            "features_successfully_tracked_mean": float(
                np.mean(subset["features_successfully_tracked"])
            ) if len(subset) else 0.0,
            "features_successfully_tracked_total": useful_sum[system],
            "failure_rate": (
                float(1.0 - useful_sum[system] / (initial_count * pair_count))
                if initial_count and pair_count
                else 0.0
            ),
            "processing_time_ms": float(
                np.sum(subset["processing_time_ms"])
            ),
            "update_latency_mean_us": (
                float(np.mean(update_latencies[system]))
                if update_latencies[system]
                else None
            ),
            "events_processed": total_events[system],
            "candidate_features_per_event": (
                float(total_candidates[system] / total_events[system])
                if total_events[system]
                else 0.0
            ),
            "distance_calculations_per_event": (
                float(total_distance_calculations[system] / total_events[system])
                if total_events[system]
                else 0.0
            ),
            "state_updates": (
                total_state_updates[system]
                if system != "frame_only"
                else tracked_sum[system]
            ),
            "asynchronous_state_updates": total_async_updates[system],
            "aps_corrections": (
                total_state_updates[system] - total_async_updates[system]
                if system == "hybrid"
                else 0
            ),
            "events_per_useful_update": (
                float(total_events[system] / useful_updates[system])
                if useful_updates[system] and system != "frame_only"
                else None
            ),
            "interval_count": pair_count,
            "initial_features": initial_count,
            "track_lifetime_definition": (
                "Number of consecutive APS intervals with a successful one-to-one "
                "reference match, then interval duration summed; this is a "
                "reference-match survival proxy, not identity ground truth."
            ),
            "processing_time_definition": (
                "Sum of method time plus the same Shi-Tomasi initialization and "
                "per-interval reference-detection work for every system; includes "
                "endpoint matching but excludes dataset loading and plotting."
            ),
            "latency_definition": (
                "Frame-only: per-pair KLT-call latency. Event systems: per-event "
                "association/update-loop latency (all events, including rejects)."
            ),
            "state_updates_definition": (
                "Frame-only: successful KLT point updates. Event-driven/hybrid: "
                "successful alpha-beta FeatureState updates."
            ),
            "useful_update_definition": (
                "Frame-only: successful KLT coordinate updates; event systems: "
                "one-to-one endpoint matches per interval."
            ),
            "events_per_useful_update_definition": (
                "Total events / endpoint-matched feature-intervals for event systems; "
                "not applicable to frame-only, which processes no events."
            ),
            "event_data_used": system != "frame_only",
        }

    checks = {
        "identical_initial_locations": (
            np.array_equal(
                shared_initial_xy,
                np.column_stack(
                    (initial_features["x"], initial_features["y"])
                ).astype(np.float32),
            )
            and initial_state_locations_match
            and np.array_equal(klt_initial_points, shared_initial_xy)
        ),
        "identical_frame_intervals": (
            len(pair_rows) == pair_count * len(SYSTEMS)
            and all(
                np.array_equal(
                    pair_rows[pair_rows["system"] == system]["interval_index"],
                    np.asarray([window.index for window in windows], dtype=np.int32),
                )
                for system in SYSTEMS
            )
        ),
        "no_event_use_by_frame_only": (
            int(np.sum(pair_rows[pair_rows["system"] == "frame_only"]["events_processed"]))
            == 0
            and total_events["frame_only"] == 0
        ),
        "frame_only_klt_output": (
            all(np.isfinite(values) for values in epe_samples["frame_only"])
            if epe_samples["frame_only"]
            else True
        ),
        "event_processing": all(
            int(total_events[system])
            == sum(len(window.events_k) for window in windows)
            for system in ("event_driven", "hybrid")
        ),
        "event_state_updates": all(
            0 <= int(diagnostics["asynchronous_state_updates"])
            <= int(diagnostics["events_processed"])
            for system, diagnostics in per_system_diagnostics.items()
            if system in ("event_driven", "hybrid")
        ),
        "matching_bounded": all(
            row["reference_matches"] <= min(row["features_tracked"], initial_count)
            for row in pair_rows
        ),
        "finite_active_event_states": all(
            np.isfinite(memory.storage["x"][memory.storage["valid"] != 0]).all()
            and np.isfinite(memory.storage["y"][memory.storage["valid"] != 0]).all()
            and np.isfinite(memory.storage["vx"][memory.storage["valid"] != 0]).all()
            and np.isfinite(memory.storage["vy"][memory.storage["valid"] != 0]).all()
            for memory in memories.values()
        ),
    }
    return ComparisonResult(
        pair_rows=pair_rows,
        lifetimes=lifetimes,
        epe_samples={key: np.asarray(value, dtype=np.float64) for key, value in epe_samples.items()},
        survival={key: np.asarray(value, dtype=np.int32) for key, value in survival.items()},
        initial_features=initial_features,
        reference_features=references,
        diagnostics=per_system_diagnostics,
        checks=checks,
    )


def _comparison_table(diagnostics: Mapping[str, Mapping[str, Any]]) -> str:
    metrics = (
        ("EPE (px)", "epe_px", 3),
        ("Track lifetime mean (μs)", "track_lifetime_mean_us", 1),
        ("Successfully tracked / interval", "features_successfully_tracked_mean", 1),
        ("Failure rate", "failure_rate", 4),
        ("Processing time total (ms)", "processing_time_ms", 2),
        ("Update latency mean (μs)", "update_latency_mean_us", 2),
        ("Events / useful update", "events_per_useful_update", 3),
        ("State updates", "state_updates", 0),
        ("Events processed", "events_processed", 0),
    )
    lines = [
        "| Metric | Frame-only | Event-driven | Hybrid |",
        "|--------|------------|--------------|--------|",
    ]
    for label, key, precision in metrics:
        values = []
        for system in SYSTEMS:
            value = diagnostics[system].get(key)
            if value is None or not np.isfinite(value):
                values.append("N/A")
            elif key == "failure_rate":
                values.append(f"{100.0 * value:.{precision}f}%")
            else:
                values.append(f"{value:.{precision}f}")
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
    return "\n".join(lines)


def save_comparison(
    windows: Sequence[FrameWindow],
    config: ComparisonConfig,
    result: ComparisonResult,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "davis_three_way_results.npz"
    arrays: dict[str, np.ndarray] = {
        "pair_rows": result.pair_rows,
        "lifetimes": result.lifetimes,
        "initial_features": result.initial_features,
    }
    for system in SYSTEMS:
        arrays[f"epe_{system}"] = result.epe_samples[system]
        arrays[f"survival_{system}"] = result.survival[system]
        arrays[f"references_{system}"] = np.concatenate(
            [
                result.reference_features[index]
                for index in range(len(result.reference_features))
            ]
        )
    np.savez_compressed(archive_path, **arrays)

    table = _comparison_table(result.diagnostics)
    summary_path = output_dir / "davis_three_way_comparison.json"
    summary = {
        "schema": "snn-vio-three-way-comparison-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": asdict(config),
        "systems": {
            system: {
                "label": SYSTEM_LABELS[system],
                "metrics": result.diagnostics[system],
            }
            for system in SYSTEMS
        },
        "checks": dict(result.checks),
        "table_markdown": table,
        "results_archive": archive_path.name,
        "metric_definitions": {
            "EPE": "Mean Euclidean endpoint error over one-to-one matches to next-frame Shi-Tomasi detections within the same radius.",
            "track_lifetime": "Consecutive APS interval boundaries with a successful match, summed in interval duration; proxy, not identity ground truth.",
            "features_successfully_tracked": "Mean one-to-one reference-matched features per APS interval.",
            "failure_rate": "1 - total successful matches / (initial feature count * interval count).",
            "processing_time": "Summed per-interval method time plus identical Shi-Tomasi initialization/reference-detection cost, excluding data load and plotting.",
            "update_latency": "Frame-only reports per-pair KLT call; event systems report per-event association/update-loop time, including rejects.",
            "events_per_useful_update": "Event systems: total event samples / successful endpoint matched feature-intervals; frame-only: N/A.",
            "state_updates": "Frame-only: successful KLT coordinate updates; event systems: successful alpha-beta state updates.",
            "track_segment": "A single contiguous APS segment; features start once in frame k and are not replenished.",
            "hybrid_correction": "After evaluation, each one-to-one matched state snaps x/y to its reference corner and timestamp to APS time; velocity/confidence are preserved; unmatched states are unchanged.",
            "event_driven_evaluation": "APS references are used only by the evaluator and do not correct or alter event-driven state.",
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    csv_path = output_dir / "davis_three_way_per_interval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(PAIR_DTYPE.names)
        for row in result.pair_rows:
            writer.writerow(
                [
                    "N/A" if isinstance(row[name], (float, np.floating)) and not np.isfinite(row[name])
                    else row[name]
                    for name in PAIR_DTYPE.names
                ]
            )

    lifetime_csv = output_dir / "davis_three_way_track_lifetimes.csv"
    with lifetime_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(LIFETIME_DTYPE.names)
        writer.writerows(tuple(row[name] for name in LIFETIME_DTYPE.names) for row in result.lifetimes)

    def chart(name: str, build) -> Path:
        path = output_dir / name
        fig, ax = plt.subplots(figsize=(8, 5))
        build(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path

    colors = ("tab:blue", "tab:orange", "tab:green")
    chart_paths: dict[str, Path] = {}

    def epe_chart(ax):
        values = [result.epe_samples[system] for system in SYSTEMS]
        ax.boxplot(values, tick_labels=[SYSTEM_LABELS[key] for key in SYSTEMS], showfliers=False)
        ax.set_ylabel("matched endpoint error (pixels)")
        ax.set_title("EPE distribution across tracked feature-intervals")
        ax.grid(axis="y", alpha=0.25)

    chart_paths["epe_distribution"] = chart("01_epe_distribution.png", epe_chart)

    def lifetime_chart(ax):
        data = [
            result.lifetimes["lifetime_us"][result.lifetimes["system"] == system]
            for system in SYSTEMS
        ]
        ax.boxplot(data, tick_labels=[SYSTEM_LABELS[key] for key in SYSTEMS], showfliers=False)
        ax.set_ylabel("consecutive useful lifetime (μs)")
        ax.set_title("Track lifetime proxy")
        ax.grid(axis="y", alpha=0.25)

    chart_paths["track_lifetime"] = chart("02_track_lifetime.png", lifetime_chart)

    def survival_chart(ax):
        x = np.arange(1, config.interval_count + 1)
        for system, color in zip(SYSTEMS, colors):
            ax.plot(x, result.survival[system], "o-", color=color, label=SYSTEM_LABELS[system])
        ax.set_xlabel("APS interval")
        ax.set_ylabel("features still useful (consecutive matches)")
        ax.set_title("Feature survival")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend()

    chart_paths["feature_survival"] = chart("03_feature_survival.png", survival_chart)

    def processing_chart(ax):
        values = [result.diagnostics[system]["processing_time_ms"] for system in SYSTEMS]
        ax.bar([SYSTEM_LABELS[key] for key in SYSTEMS], values, color=colors)
        ax.set_ylabel("summed in-memory time (ms)")
        ax.set_title("Processing time per contiguous segment")
        ax.grid(axis="y", alpha=0.25)

    chart_paths["processing_time"] = chart("04_processing_time.png", processing_chart)

    def latency_chart(ax):
        values = [
            result.pair_rows["update_latency_mean_us"][
                result.pair_rows["system"] == system
            ]
            for system in SYSTEMS
        ]
        ax.boxplot(values, tick_labels=[SYSTEM_LABELS[key] for key in SYSTEMS], showfliers=False)
        ax.set_ylabel("operation latency (μs)")
        ax.set_title("Latency (operations differ by system; see definitions)")
        ax.grid(axis="y", alpha=0.25)

    chart_paths["update_latency"] = chart("05_update_latency.png", latency_chart)

    def events_update_chart(ax):
        values = [
            result.diagnostics[system]["events_per_useful_update"]
            for system in SYSTEMS
        ]
        plotted = [0.0 if value is None else value for value in values]
        bars = ax.bar(
            [SYSTEM_LABELS[key] for key in SYSTEMS], plotted, color=colors
        )
        for bar, value in zip(bars, values):
            if value is None:
                ax.text(bar.get_x() + bar.get_width() / 2, 0, "N/A", ha="center", va="bottom")
        ax.set_ylabel("events / successful endpoint match")
        ax.set_title("Events per useful update")
        ax.grid(axis="y", alpha=0.25)

    chart_paths["events_per_update"] = chart("06_events_per_update.png", events_update_chart)

    report_path = output_dir / "davis_three_way_comparison_report.md"
    lines = [
        "# Stage 7 — Three-way feature tracking comparison",
        "",
        f"- Dataset segment: APS intervals {config.interval_start} through "
        f"{config.interval_start + config.interval_count - 1}.",
        f"- Initial features: {len(result.initial_features)} shared Shi-Tomasi locations.",
        f"- Frame interval indices: `{[window.index for window in windows]}`.",
        f"- Reference match radius: {config.reference_match_radius_px} px.",
        "",
        table,
        "",
        "## Measurement interpretation",
        "",
        "EPE uses successful one-to-one greedy matches to next-frame Shi-Tomasi detections only; unmatched tracks contribute to failure rate, not EPE. Feature survival counts consecutive intervals with a successful endpoint match. Mean lifetime is measured from the first frame until the first unmatched interval and is interval-censored; it is a useful-track proxy, not identity ground truth.",
        "",
        "Frame-only and event systems have different primitive operations. Frame-only latency is one KLT call per APS interval; event-driven and hybrid latency is association/update-loop wall time per event (including rejected events). Processing time is summed in-memory algorithm plus endpoint evaluation time, but excludes data loading and plots. Do not interpret these latency values as identical hardware operations.",
        "",
        "Hybrid APS correction snaps each matched event-updated state to its matched next-frame Shi-Tomasi reference, advances its timestamp, and preserves velocity/confidence. Unmatched states are unchanged. Event-driven APS detections are strictly evaluation-only and do not alter state.",
        "",
        "## Correctness checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    lines.extend(
        f"| {name.replace('_', ' ').title()} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in result.checks.items()
    )
    lines.extend(
        [
            "",
            "## Scientific scope",
            "",
            "This segment comparison can describe the measured operating point only. It does not establish general superiority, cross-sequence performance, motion-compensated accuracy, or statistical significance. The event-driven hypothesis is evaluated by whether its states survive/match at APS boundaries and how accuracy, update counts, and runtime compare; between-frame utility is not directly ground-truthed by APS-only endpoints.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "results": archive_path,
        "summary": summary_path,
        "per_interval": csv_path,
        "track_lifetimes": lifetime_csv,
        "report": report_path,
        **chart_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare frame-only, event-driven, and hybrid trackers on identical DAVIS intervals."
    )
    parser.add_argument("--recording", default="shapes_6dof")
    parser.add_argument("--interval-start", type=int, default=678)
    parser.add_argument("--interval-count", type=int, default=5)
    parser.add_argument("--feature-count", type=int, default=500)
    parser.add_argument("--slots", type=int, default=500)
    parser.add_argument("--reference-match-radius-px", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    config = ComparisonConfig(
        interval_start=args.interval_start,
        interval_count=args.interval_count,
        feature_count=args.feature_count,
        slot_count=args.slots,
        reference_match_radius_px=args.reference_match_radius_px,
    )
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    dataset_config = DavisConfig(recording=args.recording)
    recording = load_davis(dataset_config)
    origins, _ = load_timestamp_origins(recording, dataset_config)
    windows = make_frame_windows(recording, origins)
    end = config.interval_start + config.interval_count
    if end > len(windows):
        parser.error(f"segment end interval must be at most {len(windows) - 1}")
    selected = [windows.get_window(index) for index in range(config.interval_start, end)]
    result = compare_three_systems(selected, config)
    outputs = save_comparison(selected, config, result, args.output_dir)
    print(_comparison_table(result.diagnostics))
    print("\nChecks:")
    for name, passed in result.checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0 if all(result.checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
