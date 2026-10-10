"""Stage 5 end-to-end asynchronous APS/event feature tracking experiment."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from .aps_features import (
    ShiTomasiConfig,
    extract_shi_tomasi_features,
)
from .davis_inspection import DavisConfig, load_davis
from .event_state_update import run_asynchronous_updates
from .feature_state_memory import (
    FeatureStateMemory,
    initialize_from_aps_features,
)
from .frame_synchronization import (
    FrameWindow,
    load_timestamp_origins,
    make_frame_windows,
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "asynchronous_feature_tracker"
)
PREDICTED_DTYPE = np.dtype(
    [
        ("id", np.int32),
        ("slot", np.int32),
        ("x", np.float64),
        ("y", np.float64),
        ("vx", np.float32),
        ("vy", np.float32),
        ("confidence", np.float32),
        ("timestamp", np.int64),
    ]
)
MATCH_DTYPE = np.dtype(
    [
        ("feature_id", np.int32),
        ("slot", np.int32),
        ("reference_index", np.int32),
        ("predicted_x", np.float64),
        ("predicted_y", np.float64),
        ("reference_x", np.float32),
        ("reference_y", np.float32),
        ("pixel_error", np.float64),
        ("matched", np.uint8),
    ]
)
LIFETIME_DTYPE = np.dtype(
    [
        ("feature_id", np.int32),
        ("slot", np.int32),
        ("lifetime_us", np.int64),
        ("lifetime_seconds", np.float64),
    ]
)


@dataclass(frozen=True)
class TrackerConfig:
    feature_count: int = 500
    slot_count: int = 500
    quality_level: float = 0.01
    min_feature_distance_px: float = 5.0
    block_size_px: int = 3
    bucket_columns: int = 8
    bucket_rows: int = 6
    radius_px: float = 8.0
    time_window_us: float = 50_000.0
    alpha: float = 0.5
    beta: float = 0.1
    minimum_delta_t_us: float = 1.0
    confidence_penalty: float = 0.05
    reference_match_radius_px: float = 15.0
    trajectory_count: int = 20

    def validate(self) -> None:
        if self.feature_count <= 0 or self.slot_count <= 0:
            raise ValueError("feature_count and slot_count must be positive")
        if self.feature_count > self.slot_count:
            raise ValueError("feature_count cannot exceed slot_count")
        if not 0 < self.quality_level <= 1:
            raise ValueError("quality_level must be within (0, 1]")
        if (
            not np.isfinite(self.min_feature_distance_px)
            or self.min_feature_distance_px < 0
        ):
            raise ValueError("min_feature_distance_px must be non-negative")
        if self.block_size_px < 2:
            raise ValueError("block_size_px must be at least 2")
        if self.bucket_columns <= 0 or self.bucket_rows <= 0:
            raise ValueError("bucket grid dimensions must be positive")
        if not np.isfinite(self.radius_px) or self.radius_px <= 0:
            raise ValueError("radius_px must be finite and positive")
        if not np.isfinite(self.time_window_us) or self.time_window_us <= 0:
            raise ValueError("time_window_us must be finite and positive")
        if not 0 <= self.alpha <= 1 or not 0 <= self.beta <= 1:
            raise ValueError("alpha and beta must be within [0, 1]")
        if (
            not np.isfinite(self.minimum_delta_t_us)
            or self.minimum_delta_t_us <= 0
        ):
            raise ValueError("minimum_delta_t_us must be finite and positive")
        if not 0 <= self.confidence_penalty <= 1:
            raise ValueError("confidence_penalty must be within [0, 1]")
        if (
            not np.isfinite(self.reference_match_radius_px)
            or self.reference_match_radius_px <= 0
        ):
            raise ValueError("reference_match_radius_px must be finite and positive")
        if self.trajectory_count <= 0:
            raise ValueError("trajectory_count must be positive")


@dataclass(frozen=True)
class FramePairResult:
    """All reproducible intermediate and end-point results for a frame pair."""

    initial_features: np.ndarray
    reference_features: np.ndarray
    initial_states: np.ndarray
    final_states: np.ndarray
    event_evolution: np.ndarray
    events_per_feature: np.ndarray
    predicted_features: np.ndarray
    matches: np.ndarray
    lifetimes: np.ndarray
    diagnostics: Mapping[str, Any]
    stage_pass: Mapping[str, bool]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def predict_states_at(
    states: np.ndarray, target_timestamp_us: int
) -> np.ndarray:
    """Predict valid states at the requested timestamp without mutating them."""
    if states.dtype.names is None or not {
        "id", "x", "y", "vx", "vy", "confidence", "timestamp", "valid"
    }.issubset(states.dtype.names):
        raise TypeError("states do not contain the FeatureState fields")
    target = int(target_timestamp_us)
    predicted = np.zeros(np.count_nonzero(states["valid"]), dtype=PREDICTED_DTYPE)
    output_index = 0
    for slot, state in enumerate(states):
        if not state["valid"]:
            continue
        delta_t = target - int(state["timestamp"])
        if delta_t < 0:
            raise ValueError("prediction timestamp precedes a valid state timestamp")
        x = float(state["x"]) + float(state["vx"]) * delta_t
        y = float(state["y"]) + float(state["vy"]) * delta_t
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(f"feature {int(state['id'])} prediction is non-finite")
        predicted[output_index] = (
            int(state["id"]),
            slot,
            x,
            y,
            float(state["vx"]),
            float(state["vy"]),
            float(state["confidence"]),
            target,
        )
        output_index += 1
    return predicted


def match_predictions_to_references(
    predicted: np.ndarray,
    references: np.ndarray,
    maximum_distance_px: float,
) -> np.ndarray:
    """Greedy global one-to-one nearest matches, ordered by distance then IDs."""
    radius = float(maximum_distance_px)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("maximum_distance_px must be finite and positive")
    required = {"id", "x", "y"}
    if predicted.dtype.names is None or not required.issubset(predicted.dtype.names):
        raise TypeError("predicted records must contain id, x, and y")
    if references.dtype.names is None or not {"x", "y"}.issubset(
        references.dtype.names
    ):
        raise TypeError("reference records must contain x and y")
    matches = np.zeros(len(predicted), dtype=MATCH_DTYPE)
    matches["reference_index"] = -1
    matches["pixel_error"] = np.nan
    if not len(predicted):
        return matches

    candidates: list[tuple[float, int, int, int]] = []
    radius_squared = radius * radius
    for predicted_index, prediction in enumerate(predicted):
        for reference_index, reference in enumerate(references):
            dx = float(prediction["x"]) - float(reference["x"])
            dy = float(prediction["y"]) - float(reference["y"])
            distance_squared = dx * dx + dy * dy
            if distance_squared <= radius_squared:
                candidates.append(
                    (
                        distance_squared,
                        int(prediction["id"]),
                        predicted_index,
                        reference_index,
                    )
                )
    candidates.sort()
    used_predictions: set[int] = set()
    used_references: set[int] = set()
    for distance_squared, _, predicted_index, reference_index in candidates:
        if predicted_index in used_predictions or reference_index in used_references:
            continue
        prediction = predicted[predicted_index]
        reference = references[reference_index]
        matches[predicted_index] = (
            int(prediction["id"]),
            int(prediction["slot"]),
            reference_index,
            float(prediction["x"]),
            float(prediction["y"]),
            float(reference["x"]),
            float(reference["y"]),
            float(np.sqrt(distance_squared)),
            1,
        )
        used_predictions.add(predicted_index)
        used_references.add(reference_index)

    for index, prediction in enumerate(predicted):
        if index not in used_predictions:
            matches[index]["feature_id"] = int(prediction["id"])
            matches[index]["slot"] = int(prediction["slot"])
            matches[index]["predicted_x"] = float(prediction["x"])
            matches[index]["predicted_y"] = float(prediction["y"])
    return matches


def evaluate_frame_pair(
    window: FrameWindow,
    config: TrackerConfig = TrackerConfig(),
) -> FramePairResult:
    """Run detection, state initialization, online updates, prediction, and evaluation."""
    config.validate()
    total_started = time.perf_counter_ns()
    first_features = extract_shi_tomasi_features(
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
    memory = initialize_from_aps_features(
        first_features,
        window.frame_k_common_time_us,
        config.slot_count,
    )
    initial_states = memory.storage.copy()

    event_started = time.perf_counter_ns()
    evolution, events_per_feature, update_diagnostics = run_asynchronous_updates(
        window.events_k,
        memory,
        config.radius_px,
        config.time_window_us,
        alpha=config.alpha,
        beta=config.beta,
        minimum_delta_t_us=config.minimum_delta_t_us,
        confidence_penalty=config.confidence_penalty,
        event_timestamp_offset_us=window.event_timestamp_offset_us,
    )
    event_processing_ms = (time.perf_counter_ns() - event_started) / 1_000_000

    final_states = memory.storage.copy()
    predicted = predict_states_at(final_states, window.frame_k_plus_1_common_time_us)
    reference_features = extract_shi_tomasi_features(
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
    matches = match_predictions_to_references(
        predicted,
        reference_features,
        config.reference_match_radius_px,
    )

    lifetime_us = (
        window.frame_k_plus_1_common_time_us - window.frame_k_common_time_us
    )
    lifetime_rows = [
        (int(state["id"]), int(slot), lifetime_us, lifetime_us / 1_000_000.0)
        for slot, state in enumerate(initial_states)
        if state["valid"]
    ]
    lifetimes = np.asarray(lifetime_rows, dtype=LIFETIME_DTYPE)
    successful = matches["matched"] != 0
    errors = matches["pixel_error"][successful]
    prediction_count = len(predicted)
    success_count = int(np.count_nonzero(successful))
    total_elapsed_ms = (time.perf_counter_ns() - total_started) / 1_000_000
    event_latencies = evolution["update_latency_us"]
    diagnostics: dict[str, Any] = {
        **dict(update_diagnostics),
        "aps_frame_k_features_detected": len(first_features),
        "aps_frame_k_plus_1_reference_features_detected": len(reference_features),
        "active_features_at_frame_k_plus_1": len(final_states[final_states["valid"] != 0]),
        "successfully_predicted_features": success_count,
        "predicted_features_evaluated": prediction_count,
        "epe_px": float(np.mean(errors)) if len(errors) else None,
        "epe_median_px": float(np.median(errors)) if len(errors) else None,
        "failure_rate": (
            float((prediction_count - success_count) / prediction_count)
            if prediction_count
            else 0.0
        ),
        "track_lifetime_mean_us": (
            float(np.mean(lifetimes["lifetime_us"])) if len(lifetimes) else 0.0
        ),
        "track_lifetime_max_us": (
            int(np.max(lifetimes["lifetime_us"], initial=0))
        ),
        "events_per_successful_feature_update": (
            float(len(window.events_k) / update_diagnostics["successful_updates"])
            if update_diagnostics["successful_updates"]
            else float("inf")
        ),
        "processing_time_ms": total_elapsed_ms,
        "event_processing_time_ms": event_processing_ms,
        "update_latency_mean_us": (
            float(np.mean(event_latencies)) if len(event_latencies) else 0.0
        ),
        "update_latency_median_us": (
            float(np.median(event_latencies)) if len(event_latencies) else 0.0
        ),
        "update_latency_p95_us": (
            float(np.percentile(event_latencies, 95)) if len(event_latencies) else 0.0
        ),
        "association_policy": (
            "Stage 4A brute-force strict gates; nearest squared-distance candidate, "
            "then lower feature ID; online re-evaluation after every successful update."
        ),
        "reference_match_policy": (
            "Greedy global one-to-one nearest pairs within the configured pixel "
            "radius, sorted by distance squared, predicted feature ID, then indices."
        ),
        "epe_definition": (
            "Mean Euclidean pixel error among successful one-to-one predicted-to-"
            "reference Shi-Tomasi matches only."
        ),
        "failure_rate_definition": (
            "Unmatched valid predicted features divided by all valid predicted features."
        ),
        "track_lifetime_definition": (
            "For this single frame pair, every initially valid track is considered "
            "observable over the frame-k to frame-k+1 interval; per-feature lifetime "
            "is that interval duration."
        ),
        "update_latency_definition": (
            "Per-event wall-clock time inside the online association/update loop, "
            "including rejected events; reported mean, median and p95."
        ),
        "stage_checks": {},
    }
    diagnostics["common_metrics"] = {
        "initial_features": len(first_features),
        "tracked_features": len(final_states[final_states["valid"] != 0]),
        "successfully_predicted_features": success_count,
        "reference_matched_features": success_count,
        "epe_px": diagnostics["epe_px"],
        "track_lifetime_mean_us": diagnostics["track_lifetime_mean_us"],
        "failure_rate": diagnostics["failure_rate"],
        "processing_time_ms": total_elapsed_ms,
        "latency_ms": event_processing_ms,
    }
    stage_pass = _validate_pipeline(
        window,
        first_features,
        initial_states,
        evolution,
        events_per_feature,
        final_states,
        predicted,
        reference_features,
        matches,
        diagnostics,
    )
    diagnostics["stage_checks"] = dict(stage_pass)
    return FramePairResult(
        initial_features=first_features,
        reference_features=reference_features,
        initial_states=initial_states,
        final_states=final_states,
        event_evolution=evolution,
        events_per_feature=events_per_feature,
        predicted_features=predicted,
        matches=matches,
        lifetimes=lifetimes,
        diagnostics=diagnostics,
        stage_pass=stage_pass,
    )


def _validate_pipeline(
    window: FrameWindow,
    initial_features: np.ndarray,
    initial_states: np.ndarray,
    evolution: np.ndarray,
    events_per_feature: np.ndarray,
    final_states: np.ndarray,
    predicted: np.ndarray,
    reference_features: np.ndarray,
    matches: np.ndarray,
    diagnostics: Mapping[str, Any],
) -> dict[str, bool]:
    event_count = len(window.events_k)
    event_processing = (
        len(evolution) == event_count
        and sorted(evolution["event_index"].tolist()) == list(range(event_count))
        and np.all(np.diff(evolution["t_common_us"]) >= 0)
        and np.all(evolution["t_common_us"] >= window.interval_us[0])
        and np.all(evolution["t_common_us"] < window.interval_us[1])
    )
    association = (
        np.all(evolution["updated"] <= 1)
        and np.all(
            np.isin(evolution["status"], ("updated", "unmatched", "numerical_rejection"))
        )
        and np.all(
            (evolution["feature_slot"] >= 0)
            == (evolution["status"] != "unmatched")
        )
        and all(
            int(initial_states[int(row["feature_slot"])]["id"])
            == int(row["feature_id"])
            for row in evolution
            if int(row["feature_slot"]) >= 0
        )
        and int(diagnostics["successful_updates"])
        == int(np.count_nonzero(evolution["updated"]))
        and int(diagnostics["events_rejected"])
        == int(np.count_nonzero(evolution["updated"] == 0))
        and int(np.sum(events_per_feature["event_count"]))
        == int(diagnostics["successful_updates"])
    )
    state_update = (
        int(np.count_nonzero(final_states["valid"]))
        == int(np.count_nonzero(initial_states["valid"]))
        and np.all(final_states["valid"] <= 1)
        and np.all(final_states["timestamp"][final_states["valid"] != 0]
                   >= window.frame_k_common_time_us)
        and np.all(
            final_states["timestamp"][final_states["valid"] != 0]
            <= window.frame_k_plus_1_common_time_us
        )
        and len(events_per_feature) == len(final_states)
    )
    prediction = (
        len(predicted) == int(np.count_nonzero(final_states["valid"]))
        and np.all(predicted["timestamp"] == window.frame_k_plus_1_common_time_us)
        and np.isfinite(predicted["x"]).all()
        and np.isfinite(predicted["y"]).all()
        and all(
            np.isclose(
                float(row["x"]),
                float(final_states[int(row["slot"])]["x"])
                + float(final_states[int(row["slot"])]["vx"])
                * (
                    window.frame_k_plus_1_common_time_us
                    - int(final_states[int(row["slot"])]["timestamp"])
                ),
                rtol=1e-12,
                atol=1e-9,
            )
            and np.isclose(
                float(row["y"]),
                float(final_states[int(row["slot"])]["y"])
                + float(final_states[int(row["slot"])]["vy"])
                * (
                    window.frame_k_plus_1_common_time_us
                    - int(final_states[int(row["slot"])]["timestamp"])
                ),
                rtol=1e-12,
                atol=1e-9,
            )
            for row in predicted
        )
    )
    frame_evaluation = (
        len(matches) == len(predicted)
        and int(np.count_nonzero(matches["matched"]))
        <= min(len(predicted), len(reference_features))
        and np.isfinite(matches["pixel_error"][matches["matched"] != 0]).all()
        and len(
            np.unique(matches["reference_index"][matches["matched"] != 0])
        )
        == int(np.count_nonzero(matches["matched"]))
        and all(
            int(row["reference_index"]) < len(reference_features)
            and np.isclose(
                float(row["pixel_error"]),
                float(
                    np.hypot(
                        float(row["predicted_x"]) - float(row["reference_x"]),
                        float(row["predicted_y"]) - float(row["reference_y"]),
                    )
                ),
                rtol=1e-12,
                atol=1e-9,
            )
            for row in matches
            if row["matched"]
        )
    )
    return {
        "event_processing": bool(event_processing),
        "association": bool(association),
        "state_update": bool(state_update),
        "prediction": bool(prediction),
        "frame_to_frame_evaluation": bool(frame_evaluation),
    }


def _save_visualizations(
    window: FrameWindow,
    result: FramePairResult,
    output_dir: Path,
    trajectory_count: int,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    def save(name: str, figure: plt.Figure) -> None:
        path = output_dir / name
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths[name.removesuffix(".png")] = path

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(window.frame_k, cmap="gray", vmin=0, vmax=255)
    ax.scatter(
        result.initial_features["x"],
        result.initial_features["y"],
        s=18,
        c="lime",
        marker="+",
        linewidths=0.8,
    )
    ax.set_title(
        f"Frame k={window.index}: {len(result.initial_features)} initial Shi–Tomasi features"
    )
    ax.set_axis_off()
    save("01_frame_k_initial_features.png", fig)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(window.frame_k, cmap="gray", vmin=0, vmax=255)
    events = result.event_evolution
    if len(events):
        for polarity, color, label in (
            (False, "tab:blue", "p=0"),
            (True, "tab:orange", "p=1"),
        ):
            select = (events["p"] != 0) == polarity
            ax.scatter(
                events["x_event"][select],
                events["y_event"][select],
                s=4,
                c=color,
                marker=".",
                alpha=0.45,
                label=label,
            )
    ax.set_title(
        f"Events in [{window.interval_us[0]}, {window.interval_us[1]}) μs "
        f"({len(events)} events)"
    )
    ax.set_axis_off()
    if len(events):
        ax.legend(loc="upper right")
    save("02_event_stream.png", fig)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(window.frame_k, cmap="gray", vmin=0, vmax=255)
    active_updates = result.event_evolution[result.event_evolution["updated"] != 0]
    feature_counts = {
        int(row["feature_id"]): int(row["event_count"])
        for row in result.events_per_feature
        if int(row["feature_id"]) >= 0 and int(row["event_count"]) > 0
    }
    featured = [
        feature_id for feature_id, _ in sorted(
            feature_counts.items(), key=lambda item: (-item[1], item[0])
        )[:trajectory_count]
    ]
    for feature_id in featured:
        track = active_updates[active_updates["feature_id"] == feature_id]
        initial = result.initial_states[result.initial_states["id"] == feature_id]
        if not len(track) or not len(initial):
            continue
        xpoints = np.concatenate(([float(initial[0]["x"])], track["updated_x"]))
        ypoints = np.concatenate(([float(initial[0]["y"])], track["updated_y"]))
        ax.plot(xpoints, ypoints, linewidth=1, alpha=0.85)
        ax.scatter(xpoints[0], ypoints[0], c="lime", marker="+", s=25)
        ax.scatter(track["x_event"], track["y_event"], s=9, c="orange", marker=".")
        ax.scatter(track["predicted_x"], track["predicted_y"], s=16, c="cyan", marker="^")
        ax.scatter(track["updated_x"], track["updated_y"], s=14, c="magenta", marker="s")
    ax.set_title(
        f"Top {len(featured)} updated tracks: events (orange), predictions (cyan), "
        "updates (magenta)"
    )
    ax.set_axis_off()
    save("03_event_feature_trajectories.png", fig)

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
            label="predicted at t(k+1)",
        )
    ax.set_title("Predicted feature positions at frame k+1")
    ax.set_axis_off()
    if len(result.predicted_features):
        ax.legend(loc="upper right")
    save("04_predicted_positions_frame_k_plus_1.png", fig)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(window.frame_k_plus_1, cmap="gray", vmin=0, vmax=255)
    if len(result.reference_features):
        ax.scatter(
            result.reference_features["x"],
            result.reference_features["y"],
            s=18,
            c="yellow",
            marker="+",
            linewidths=0.8,
            label="reference Shi–Tomasi features",
        )
    ax.set_title("Reference Shi–Tomasi features at frame k+1")
    ax.set_axis_off()
    if len(result.reference_features):
        ax.legend(loc="upper right")
    save("05_reference_features_frame_k_plus_1.png", fig)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(window.frame_k_plus_1, cmap="gray", vmin=0, vmax=255)
    matched = result.matches[result.matches["matched"] != 0]
    if len(matched):
        ax.scatter(
            matched["predicted_x"],
            matched["predicted_y"],
            c="cyan",
            marker="^",
            s=22,
            label="prediction",
        )
        ax.scatter(
            matched["reference_x"],
            matched["reference_y"],
            c="yellow",
            marker="+",
            s=25,
            label="reference",
        )
        for row in matched:
            ax.plot(
                [row["predicted_x"], row["reference_x"]],
                [row["predicted_y"], row["reference_y"]],
                color="magenta",
                linewidth=0.7,
                alpha=0.65,
            )
    ax.set_title(
        f"One-to-one frame evaluation: {len(matched)} matches; "
        + (
            f"EPE={result.diagnostics['epe_px']:.3f} px"
            if result.diagnostics["epe_px"] is not None
            else "EPE unavailable"
        )
    )
    ax.set_axis_off()
    if len(matched):
        ax.legend(loc="upper right")
    save("06_predicted_vs_reference_error.png", fig)
    return paths


def save_experiment(
    window: FrameWindow,
    result: FramePairResult,
    config: TrackerConfig,
    output_dir: Path,
) -> dict[str, Path]:
    """Persist all numeric pipeline outputs, config, diagnostics, report and plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_path = output_dir / "davis_asynchronous_tracker_results.npz"
    np.savez_compressed(
        numeric_path,
        initial_features=result.initial_features,
        reference_features=result.reference_features,
        initial_states=result.initial_states,
        final_states=result.final_states,
        event_evolution=result.event_evolution,
        events_per_feature=result.events_per_feature,
        predicted_features=result.predicted_features,
        matches=result.matches,
        lifetimes=result.lifetimes,
    )
    parameters_path = output_dir / "davis_asynchronous_tracker_parameters.json"
    parameters = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "5 complete asynchronous feature tracker",
        "configuration": asdict(config),
        "interval_index": window.index,
        "frame_timestamps_original_us": [
            window.frame_k_timestamp_us,
            window.frame_k_plus_1_timestamp_us,
        ],
        "frame_timestamps_common_us": list(window.interval_us),
        "event_timestamp_offset_us": window.event_timestamp_offset_us,
        "timestamp_unit": "microseconds",
        "velocity_unit": "pixels per microsecond",
        "metrics_and_policies": _json_safe(result.diagnostics),
        "stage_pass": dict(result.stage_pass),
        "expected_pipeline": [
            "APS frame k -> Shi-Tomasi -> FeatureStateMemory initialization",
            "half-open [t_k, t_(k+1)) events -> stable chronological processing",
            "Stage 4A brute-force association -> Stage 4B online alpha-beta update",
            "predict valid states at t_(k+1)",
            "Shi-Tomasi reference detections in frame k+1",
            "greedy one-to-one reference match -> endpoint pixel errors",
        ],
    }
    parameters_path.write_text(
        json.dumps(_json_safe(parameters), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    comparison = {
        "schema": "snn-vio-tracker-comparison-v1",
        "stage": "5",
        "method": "asynchronous_event_feature_tracker",
        "uses_events": True,
        "interval_index": window.index,
        "frame_timestamps_common_us": list(window.interval_us),
        "timestamp_unit": "microseconds",
        "configuration": asdict(config),
        "metrics": _json_safe(result.diagnostics),
        "common_metrics": _json_safe(result.diagnostics["common_metrics"]),
        "checks": dict(result.stage_pass),
        "definitions": {
            "tracked_features": (
                "Valid predicted FeatureState records at frame-k+1; endpoint "
                "reference match is separately reported."
            ),
            "EPE": result.diagnostics["epe_definition"],
            "failure_rate": result.diagnostics["failure_rate_definition"],
            "track_lifetime": result.diagnostics["track_lifetime_definition"],
            "latency": result.diagnostics["update_latency_definition"],
            "common_latency_ms": (
                "Stage 5 event-loop wall-clock time; excludes dataset I/O and plots."
            ),
        },
        "arrays_file": "davis_asynchronous_tracker_results.npz",
    }
    comparison_path = output_dir / "davis_asynchronous_comparison_ready.json"
    comparison_path.write_text(
        json.dumps(_json_safe(comparison), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    diagnostics_path = output_dir / "davis_asynchronous_tracker_diagnostics.csv"
    scalar_rows = [
        {"metric": key, "value": json.dumps(_json_safe(value), allow_nan=False)}
        for key, value in result.diagnostics.items()
        if key != "stage_checks"
        and isinstance(value, (int, float, str, bool))
    ]
    with diagnostics_path.open("w", newline="", encoding="utf-8") as stream:
        import csv

        writer = csv.DictWriter(stream, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerows(scalar_rows)

    visualization_paths = _save_visualizations(
        window, result, output_dir, config.trajectory_count
    )
    report_path = output_dir / "davis_asynchronous_tracker_report.md"
    report_lines = [
        "# Stage 5 — Complete asynchronous feature tracker",
        "",
        f"- APS pair: {window.index} → {window.index + 1}",
        f"- Common-axis interval: [{window.interval_us[0]}, {window.interval_us[1]}) μs",
        f"- Initial features: {result.diagnostics['aps_frame_k_features_detected']}",
        f"- Events processed: {result.diagnostics['events_processed']}",
        f"- Successful asynchronous updates: {result.diagnostics['successful_updates']}",
        f"- Events rejected: {result.diagnostics['events_rejected']}",
        f"- Active features at t(k+1): {result.diagnostics['active_features_at_frame_k_plus_1']}",
        f"- Successfully predicted/reference-matched features: {result.diagnostics['successfully_predicted_features']}",
        (
            f"- Endpoint EPE (matched features only): "
            f"{result.diagnostics['epe_px']:.6g} px"
            if result.diagnostics["epe_px"] is not None
            else "- Endpoint EPE (matched features only): unavailable (no matches)"
        ),
        f"- Failure rate: {result.diagnostics['failure_rate']:.6g}",
        f"- Mean track lifetime: {result.diagnostics['track_lifetime_mean_us']:.0f} μs",
        f"- Events per successful feature update: {result.diagnostics['events_per_successful_feature_update']:.6g}",
        f"- Total processing time: {result.diagnostics['processing_time_ms']:.3f} ms",
        f"- Event-loop time: {result.diagnostics['event_processing_time_ms']:.3f} ms",
        f"- Per-event update-loop latency mean / median / p95: "
        f"{result.diagnostics['update_latency_mean_us']:.3f} / "
        f"{result.diagnostics['update_latency_median_us']:.3f} / "
        f"{result.diagnostics['update_latency_p95_us']:.3f} μs",
        "",
        "## Stage checks",
        "",
        "| Component | Result |",
        "|---|---|",
    ]
    check_labels = {
        "event_processing": "Event processing",
        "association": "Association",
        "state_update": "State update",
        "prediction": "Prediction",
        "frame_to_frame_evaluation": "Frame-to-frame evaluation",
    }
    for key, label in check_labels.items():
        report_lines.append(
            f"| {label} | {'PASS' if result.stage_pass[key] else 'FAIL'} |"
        )
    report_lines.extend(
        [
            "",
            "EPE is the mean Euclidean pixel error over successful, one-to-one greedy matches from predicted states to frame-k+1 Shi-Tomasi detections within the configured match radius. Unmatched predicted states are counted as failures, not included in EPE.",
            "",
            "Track lifetime is interval-censored: because this experiment evaluates one frame pair and does not invalidate tracks, every initialized valid feature is counted for the full APS interval. Events per successful update is total processed events divided by successful state updates.",
            "",
            "This is an algorithm-validation baseline, not a tracking-quality conclusion. No KLT, VIO, SNN, FPGA, HLS, RTL, SLAM, grid hashing, or Kalman filtering is included.",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "results": numeric_path,
        "parameters": parameters_path,
        "comparison_ready": comparison_path,
        "diagnostics": diagnostics_path,
        "report": report_path,
        **visualization_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the complete Stage 5 asynchronous feature tracker on a DAVIS APS pair."
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
    parser.add_argument("--reference-match-radius-px", type=float, default=15.0)
    parser.add_argument("--trajectory-count", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.interval_index < 0:
        parser.error("--interval-index must be non-negative")
    config = TrackerConfig(
        feature_count=args.feature_count,
        slot_count=args.slots,
        radius_px=args.radius_px,
        time_window_us=args.time_window_us,
        alpha=args.alpha,
        beta=args.beta,
        minimum_delta_t_us=args.minimum_delta_t_us,
        confidence_penalty=args.confidence_penalty,
        reference_match_radius_px=args.reference_match_radius_px,
        trajectory_count=args.trajectory_count,
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
    result = evaluate_frame_pair(window, config)
    outputs = save_experiment(window, result, config, args.output_dir)
    print(json.dumps(dict(result.diagnostics), indent=2, allow_nan=True))
    print("Stage checks:")
    for name, passed in result.stage_pass.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0 if all(result.stage_pass.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
