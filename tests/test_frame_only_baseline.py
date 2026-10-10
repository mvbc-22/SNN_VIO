import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.frame_only_baseline import (
    FrameBaselineConfig,
    run_frame_only_baseline,
    save_baseline,
)
from src.frame_synchronization import FrameWindow


EVENT_DTYPE = np.dtype(
    [("t", np.int64), ("x", np.int64), ("y", np.int64), ("p", np.int8)]
)


def make_frame(shift=0):
    y, x = np.indices((160, 200))
    image = (((((x - shift) // 10) + y // 10) % 2) * 255).astype(np.uint8)
    return image


def make_window():
    first = make_frame()
    second = make_frame(2)
    events = np.array([(10, 1, 1, 1)], dtype=EVENT_DTYPE)
    return FrameWindow(
        index=12,
        frame_k=first,
        frame_k_plus_1=second,
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


class FrameOnlyBaselineTests(unittest.TestCase):
    def test_runs_without_accessing_event_data(self):
        window = make_window()
        no_events = SimpleNamespace(
            index=window.index,
            frame_k=window.frame_k,
            frame_k_plus_1=window.frame_k_plus_1,
            frame_k_common_time_us=window.frame_k_common_time_us,
            frame_k_plus_1_common_time_us=window.frame_k_plus_1_common_time_us,
        )
        result = run_frame_only_baseline(
            no_events,
            FrameBaselineConfig(
                feature_count=80,
                reference_match_radius_px=30,
            ),
        )
        self.assertEqual(len(result.initial_features), 80)
        self.assertEqual(len(result.klt_tracks), len(result.initial_features))
        self.assertGreater(result.diagnostics["tracked_features"], 0)
        self.assertNotIn("events_processed", result.diagnostics)
        self.assertTrue(all(result.checks.values()))

    def test_same_start_detector_count_and_reference_feature_count(self):
        config = FrameBaselineConfig(
            feature_count=60,
            quality_level=0.01,
            min_feature_distance_px=5,
            block_size_px=3,
            bucket_columns=8,
            bucket_rows=6,
        )
        result = run_frame_only_baseline(make_window(), config)
        self.assertEqual(len(result.initial_features), 60)
        self.assertEqual(len(result.reference_features), 60)
        np.testing.assert_array_equal(
            result.klt_tracks["initial_x"], result.initial_features["x"]
        )
        np.testing.assert_array_equal(
            result.klt_tracks["initial_y"], result.initial_features["y"]
        )
        self.assertEqual(
            result.diagnostics["tracked_features"],
            int(np.count_nonzero(result.klt_tracks["status"])),
        )
        self.assertAlmostEqual(
            result.diagnostics["klt_failure_rate"],
            1 - result.diagnostics["tracked_features"] / len(result.initial_features),
        )
        self.assertAlmostEqual(
            result.diagnostics["failure_rate"],
            1
            - result.diagnostics["reference_matched_features"]
            / len(result.initial_features),
        )

    def test_comparison_ready_artifacts_and_shared_visualizations(self):
        window = make_window()
        config = FrameBaselineConfig(feature_count=40, reference_match_radius_px=30)
        result = run_frame_only_baseline(window, config)
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = save_baseline(window, result, config, Path(temp_dir))
            expected = {
                "01_frame_k_initial_features",
                "02_event_stream",
                "03_event_feature_trajectories",
                "04_predicted_positions_frame_k_plus_1",
                "05_reference_features_frame_k_plus_1",
                "06_predicted_vs_reference_error",
                "comparison_ready",
            }
            self.assertTrue(expected.issubset(outputs))
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            import json

            comparison = json.loads(outputs["comparison_ready"].read_text())
            self.assertEqual(comparison["schema"], "snn-vio-tracker-comparison-v1")
            self.assertFalse(comparison["uses_events"])
            self.assertIn("epe_px", comparison["metrics"])
            self.assertIn("failure_rate", comparison["metrics"])
            self.assertEqual(
                set(comparison["common_metrics"]),
                {
                    "initial_features",
                    "tracked_features",
                    "successfully_predicted_features",
                    "reference_matched_features",
                    "epe_px",
                    "track_lifetime_mean_us",
                    "failure_rate",
                    "processing_time_ms",
                    "latency_ms",
                },
            )
            report = outputs["report"].read_text()
            self.assertIn("Event data: not accessed or used", report)

    def test_configuration_rejects_unsupported_values(self):
        for config in (
            FrameBaselineConfig(feature_count=0),
            FrameBaselineConfig(klt_window_px=20),
            FrameBaselineConfig(klt_max_pyramid_level=-1),
            FrameBaselineConfig(reference_match_radius_px=float("inf")),
        ):
            with self.assertRaises(ValueError):
                config.validate()


if __name__ == "__main__":
    unittest.main()
