import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.asynchronous_feature_tracker import (
    MATCH_DTYPE,
    TrackerConfig,
    evaluate_frame_pair,
    match_predictions_to_references,
    predict_states_at,
    save_experiment,
)
from src.frame_synchronization import FrameWindow


EVENT_DTYPE = np.dtype(
    [("t", np.int64), ("x", np.int64), ("y", np.int64), ("p", np.int8)]
)
REFERENCE_DTYPE = np.dtype(
    [("id", np.int32), ("x", np.float32), ("y", np.float32), ("quality", np.float32), ("valid", np.uint8)]
)
PREDICTION_DTYPE = np.dtype(
    [("id", np.int32), ("slot", np.int32), ("x", np.float64), ("y", np.float64)]
)


def checkerboard(width=128, height=96, square=8, shift=0):
    y, x = np.indices((height, width))
    image = (((((x - shift) // square) + y // square) % 2) * 255).astype(np.uint8)
    return image


def make_window():
    frame0 = checkerboard()
    frame1 = checkerboard(shift=1)
    # Sample a reproducible subset of corners from the checkerboard.
    events = np.array(
        [
            (110, 8, 8, 1),
            (120, 24, 8, 0),
            (130, 40, 24, 1),
            (140, 70, 60, 0),
            (150, 115, 85, 1),
            (160, 0, 0, 0),
        ],
        dtype=EVENT_DTYPE,
    )
    return FrameWindow(
        index=0,
        frame_k=frame0,
        frame_k_plus_1=frame1,
        frame_k_timestamp_us=100,
        frame_k_plus_1_timestamp_us=200,
        frame_k_common_time_us=100,
        frame_k_plus_1_common_time_us=200,
        event_source=events,
        event_start_index=0,
        event_end_index=len(events),
        event_timestamp_offset_us=0,
        imu_k=None,
        groundtruth_k=None,
    )


class AsynchronousFeatureTrackerTests(unittest.TestCase):
    def test_predict_states_at_endpoint_uses_explicit_dt_without_mutation(self):
        memory_window = make_window()
        result = evaluate_frame_pair(
            memory_window,
            TrackerConfig(
                feature_count=40,
                slot_count=45,
                radius_px=12,
                time_window_us=200,
                reference_match_radius_px=30,
                trajectory_count=5,
            ),
        )
        predicted = predict_states_at(result.final_states, 200)
        valid = result.final_states[result.final_states["valid"] != 0]
        self.assertEqual(len(predicted), len(valid))
        self.assertTrue(np.all(predicted["timestamp"] == 200))
        for row in predicted:
            state = result.final_states[int(row["slot"])]
            delta = 200 - int(state["timestamp"])
            self.assertAlmostEqual(
                row["x"], float(state["x"]) + float(state["vx"]) * delta
            )

    def test_reference_matching_is_one_to_one_and_deterministic(self):
        predicted = np.array(
            [(7, 2, 0.0, 0.0), (3, 1, 1.0, 0.0)], dtype=PREDICTION_DTYPE
        )
        references = np.array(
            [(0, 0.4, 0, 1, 1), (1, 5, 5, 1, 1)], dtype=REFERENCE_DTYPE
        )
        matches = match_predictions_to_references(predicted, references, 2)
        self.assertEqual(matches.dtype, MATCH_DTYPE)
        self.assertEqual(matches["matched"].tolist(), [1, 0])
        self.assertEqual(matches["reference_index"].tolist(), [0, -1])
        self.assertAlmostEqual(float(matches[0]["pixel_error"]), 0.4, places=6)

    def test_end_to_end_stage_checks_metrics_and_initialization(self):
        window = make_window()
        config = TrackerConfig(
            feature_count=40,
            slot_count=45,
            radius_px=12,
            time_window_us=200,
            alpha=0.5,
            beta=0.1,
            minimum_delta_t_us=1,
            confidence_penalty=0.05,
            reference_match_radius_px=30,
            trajectory_count=5,
        )
        result = evaluate_frame_pair(window, config)
        self.assertEqual(
            set(result.stage_pass),
            {
                "event_processing",
                "association",
                "state_update",
                "prediction",
                "frame_to_frame_evaluation",
            },
        )
        self.assertTrue(all(result.stage_pass.values()))
        self.assertGreater(len(result.initial_features), 0)
        self.assertEqual(len(result.initial_states), config.slot_count)
        self.assertEqual(len(result.event_evolution), len(window.events_k))
        self.assertEqual(len(result.predicted_features), len(result.final_states[result.final_states["valid"] != 0]))
        self.assertEqual(len(result.matches), len(result.predicted_features))
        self.assertGreaterEqual(result.diagnostics["successful_updates"], 0)
        self.assertGreaterEqual(result.diagnostics["processing_time_ms"], 0)
        self.assertGreaterEqual(result.diagnostics["update_latency_p95_us"], 0)
        self.assertEqual(
            result.diagnostics["events_per_successful_feature_update"],
            len(window.events_k) / result.diagnostics["successful_updates"]
            if result.diagnostics["successful_updates"]
            else float("inf"),
        )

    def test_all_six_requested_visualizations_and_parameters_are_saved(self):
        window = make_window()
        config = TrackerConfig(
            feature_count=30,
            slot_count=32,
            radius_px=12,
            time_window_us=200,
            reference_match_radius_px=30,
            trajectory_count=3,
        )
        result = evaluate_frame_pair(window, config)
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = save_experiment(window, result, config, Path(temp_dir))
            visualization_names = {
                "01_frame_k_initial_features",
                "02_event_stream",
                "03_event_feature_trajectories",
                "04_predicted_positions_frame_k_plus_1",
                "05_reference_features_frame_k_plus_1",
                "06_predicted_vs_reference_error",
            }
            self.assertTrue(visualization_names.issubset(outputs))
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            archive = np.load(outputs["results"], allow_pickle=False)
            self.assertIn("event_evolution", archive.files)
            self.assertIn("predicted_features", archive.files)
            import json

            parameters = json.loads(outputs["parameters"].read_text())
            self.assertEqual(parameters["configuration"]["alpha"], config.alpha)
            self.assertEqual(parameters["interval_index"], 0)
            report = outputs["report"].read_text()
            self.assertIn("| Event processing | PASS |", report)
            self.assertIn("| Frame-to-frame evaluation | PASS |", report)

    def test_configuration_validation(self):
        for config in (
            TrackerConfig(feature_count=10, slot_count=9),
            TrackerConfig(alpha=float("nan")),
            TrackerConfig(beta=2),
            TrackerConfig(min_feature_distance_px=float("nan")),
        ):
            with self.assertRaises(ValueError):
                config.validate()


if __name__ == "__main__":
    unittest.main()
