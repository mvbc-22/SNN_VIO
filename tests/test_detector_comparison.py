import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.detector_comparison import (
    DetectorConfig,
    _recommend_detector,
    compare_detectors,
    default_consecutive_pair_indices,
    mutual_nearest_repeatability,
    save_comparison_outputs,
)
from src.davis_inspection import DavisRecording
from src.frame_synchronization import make_frame_windows


def make_checkerboard(width: int = 120, height: int = 90, square: int = 9) -> np.ndarray:
    y, x = np.indices((height, width))
    return (((x // square + y // square) % 2) * 255).astype(np.uint8)


def make_texture(width: int = 120, height: int = 90) -> np.ndarray:
    rng = np.random.default_rng(2026)
    image = rng.integers(0, 256, (height, width), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def make_windows():
    texture = make_texture()
    frames = np.stack([np.roll(texture, shift, axis=1) for shift in range(4)])
    events = np.array(
        [(0, 0, 0, 0), (100, 1, 1, 1), (200, 2, 2, 0)],
        dtype=[("t", "i8"), ("x", "i8"), ("y", "i8"), ("p", "i8")],
    )
    recording = DavisRecording(
        events=events,
        imu=None,
        aps={"ts": np.array([0, 10, 20, 30]), "frames": frames},
        target=None,
    )
    return make_frame_windows(recording, {"events": 100.0, "aps_frames": 100.0})


class DetectorComparisonTests(unittest.TestCase):
    def test_default_pair_selection_uses_consecutive_frame_starts(self) -> None:
        indices = default_consecutive_pair_indices(1356, 5)
        self.assertEqual(indices, [0, 338, 677, 1015, 1354])
        self.assertTrue(all(index + 1 < 1356 for index in indices))

    def test_mutual_nearest_repeatability_and_empty_features(self) -> None:
        points = np.array([[0, 0], [10, 10], [20, 5]], dtype=np.float32)
        exact = mutual_nearest_repeatability(points, points, radius_px=1)
        self.assertEqual(exact["matched_count"], 3)
        self.assertEqual(exact["repeatability"], 1.0)
        shifted = mutual_nearest_repeatability(points, points + 2, radius_px=3)
        self.assertEqual(shifted["repeatability"], 1.0)
        empty = mutual_nearest_repeatability(np.empty((0, 2)), points)
        self.assertEqual(empty["denominator"], 0)
        self.assertEqual(empty["repeatability"], 0.0)

    def test_both_detectors_use_same_consecutive_aps_pairs(self) -> None:
        windows = make_windows()
        rows, repeatability, parameters, frames = compare_detectors(
            windows,
            pair_indices=[0, 2],
            requested_counts=[10, 20],
            config=DetectorConfig(
                bucket_columns=4,
                bucket_rows=3,
                timing_repeats=1,
                repeatability_radius_px=3.0,
            ),
        )
        self.assertEqual(parameters["consecutive_pair_indices"], [0, 2])
        self.assertEqual(parameters["frame_indices"], [0, 1, 2, 3])
        self.assertEqual(set(frames), {0, 1, 2, 3})
        self.assertEqual(
            {(row["detector"], row["frame_index"], row["requested_count"]) for row in rows},
            {
                (detector, frame, count)
                for detector in ("shi_tomasi", "fast")
                for frame in (0, 1, 2, 3)
                for count in (10, 20)
            },
        )
        self.assertEqual(len(repeatability), 8)
        self.assertTrue(all(row["total_time_ms"] >= 0 for row in rows))
        self.assertTrue(all(row["metrics"]["bucket_coverage_percent"] >= 0 for row in rows))
        self.assertTrue(
            all(
                row["metrics"]["response_definition"]
                == ("Shi-Tomasi minimum eigenvalue" if row["detector"] == "shi_tomasi"
                    else "OpenCV FAST keypoint response")
                for row in rows
            )
        )

    def test_recommendation_uses_measurements_and_gates(self) -> None:
        summary = {
            detector: {
                500: {
                    "mean_detected": 500.0,
                    "mean_bucket_coverage_percent": 80.0,
                    "mean_repeatability_percent": 70.0 if detector == "fast" else 60.0,
                    "mean_total_time_ms": 1.0 if detector == "fast" else 4.0,
                }
            }
            for detector in ("shi_tomasi", "fast")
        }
        self.assertEqual(_recommend_detector(summary, 500)[0], "fast")

    def test_comparison_outputs_keep_numeric_feature_records(self) -> None:
        windows = make_windows()
        rows, repeatability, parameters, frames = compare_detectors(
            windows,
            pair_indices=[0, 2],
            requested_counts=[10, 20],
            config=DetectorConfig(
                bucket_columns=4,
                bucket_rows=3,
                timing_repeats=1,
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_paths = save_comparison_outputs(
                rows,
                repeatability,
                parameters,
                frames,
                Path(temp_dir),
                operating_count=20,
            )
            archive = np.load(output_paths["features"], allow_pickle=False)
            features = archive["features"]
            self.assertFalse(features.dtype.hasobject)
            self.assertEqual(len(np.unique(features["id"])), len(features))
            self.assertEqual(set(features["detector"].tolist()), {"fast", "shi_tomasi"})
            self.assertTrue(all(path.is_file() for path in output_paths.values()))
            report = output_paths["report"].read_text()
            self.assertIn("raw-coordinate repeatability", report)
            self.assertIn("not directly comparable", report)

    def test_configuration_rejects_invalid_fast_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "fast_threshold"):
            DetectorConfig(fast_threshold=256).validate()


if __name__ == "__main__":
    unittest.main()
