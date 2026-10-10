import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.davis_inspection import DavisRecording
from src.aps_features import (
    FEATURE_DTYPE,
    REQUESTED_FEATURE_COUNTS,
    ShiTomasiConfig,
    default_frame_indices,
    extract_shi_tomasi_features,
    get_aps_frame,
    run_feature_count_experiment,
    save_experiment_outputs,
)
from src.frame_synchronization import make_frame_windows


def make_recording() -> DavisRecording:
    events = np.array(
        [(0, 0, 0, 0), (10, 1, 1, 1), (20, 2, 1, 0), (30, 3, 2, 1)],
        dtype=[("t", "i8"), ("x", "i8"), ("y", "i8"), ("p", "i8")],
    )
    aps = {
        "ts": np.array([0, 10, 20, 30], dtype=np.int64),
        "frames": np.zeros((4, 3, 4), dtype=np.uint8),
    }
    imu = {"ts": np.array([0, 10, 20, 30], dtype=np.int64)}
    target = {"ts": np.array([0, 10, 20, 30], dtype=np.int64)}
    return DavisRecording(events=events, imu=imu, aps=aps, target=target)


def checkerboard(width: int = 240, height: int = 180, square: int = 12) -> np.ndarray:
    y, x = np.indices((height, width))
    return (((x // square + y // square) % 2) * 255).astype(np.uint8)


class APSShiTomasiTests(unittest.TestCase):
    def test_output_record_layout_and_unique_ids(self) -> None:
        result = extract_shi_tomasi_features(
            checkerboard(),
            ShiTomasiConfig(max_features=100),
            frame_index=7,
            frame_timestamp_us=1234,
            first_feature_id=50,
        )
        self.assertEqual(result.features.dtype, FEATURE_DTYPE)
        self.assertLessEqual(len(result.features), 100)
        self.assertTrue(np.all(result.features["valid"] == 1))
        self.assertEqual(result.features["id"].tolist(), list(range(50, 50 + len(result.features))))
        self.assertEqual(result.frame_index, 7)
        self.assertEqual(result.frame_timestamp_us, 1234)

    def test_features_are_spatially_distributed(self) -> None:
        min_distance = 5.0
        result = extract_shi_tomasi_features(
            checkerboard(),
            ShiTomasiConfig(
                max_features=100,
                bucket_columns=8,
                bucket_rows=6,
                min_feature_distance=min_distance,
            ),
        )
        self.assertGreater(result.metrics["occupied_buckets"], 10)
        self.assertGreater(result.metrics["bucket_coverage_percent"], 20)
        self.assertGreater(result.metrics["normalized_bucket_entropy"], 0.5)
        points = np.column_stack((result.features["x"], result.features["y"]))
        if len(points) > 1:
            distances = np.sqrt(
                np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
            )
            np.fill_diagonal(distances, np.inf)
            self.assertGreaterEqual(float(np.min(distances)), min_distance)

    def test_constant_image_returns_valid_empty_feature_array(self) -> None:
        result = extract_shi_tomasi_features(
            np.zeros((60, 80), dtype=np.uint8),
            ShiTomasiConfig(max_features=50),
        )
        self.assertEqual(len(result.features), 0)
        self.assertEqual(result.features.dtype, FEATURE_DTYPE)
        self.assertEqual(result.metrics["detected_count"], 0)
        self.assertEqual(result.metrics["median_quality"], 0.0)

    def test_invalid_configuration_and_frame_types_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "quality_level"):
            ShiTomasiConfig(max_features=10, quality_level=0).validate()
        with self.assertRaisesRegex(ValueError, "block_size"):
            ShiTomasiConfig(max_features=10, block_size=1).validate()
        with self.assertRaisesRegex(ValueError, "uint8"):
            extract_shi_tomasi_features(
                np.zeros((20, 20), dtype=np.float32),
                ShiTomasiConfig(max_features=10),
            )

    def test_default_frames_include_both_recording_ends(self) -> None:
        self.assertEqual(default_frame_indices(1356), [0, 338, 677, 1016, 1355])
        recording = make_recording()
        windows = make_frame_windows(
            recording,
            {
                "events": 100.0,
                "aps_frames": 100.000005,
                "imu": 100.000004,
                "ground_truth": 100.000008,
            },
        )
        frame, timestamp = get_aps_frame(windows, len(windows))
        np.testing.assert_array_equal(frame, recording.aps["frames"][-1])
        self.assertEqual(timestamp, recording.aps["ts"][-1])

    def test_count_experiment_uses_selected_frame_pair_images(self) -> None:
        recording = make_recording()
        frames = np.stack(
            [checkerboard(120, 90, 9), checkerboard(120, 90, 10),
             checkerboard(120, 90, 11), checkerboard(120, 90, 12)]
        )
        recording = type(recording)(
            events=recording.events,
            imu=recording.imu,
            aps={"ts": recording.aps["ts"], "frames": frames},
            target=recording.target,
        )
        windows = make_frame_windows(
            recording,
            {
                "events": 100.0,
                "aps_frames": 100.000005,
                "imu": 100.000004,
                "ground_truth": 100.000008,
            },
        )
        results, config = run_feature_count_experiment(
            windows,
            frame_indices=[0, 3],
            requested_counts=[10, 25],
            bucket_columns=4,
            bucket_rows=3,
        )
        self.assertEqual([r.frame_index for r in results], [0, 0, 3, 3])
        self.assertEqual([r.requested_count for r in results], [10, 25, 10, 25])
        self.assertTrue(all(r.features.dtype == FEATURE_DTYPE for r in results))
        self.assertTrue(config["feature_id_semantics"].startswith("Unique"))
        np.testing.assert_array_equal(
            results[0].features[["x", "y"]],
            results[1].features[["x", "y"]][:10],
        )

    def test_outputs_are_non_object_arrays_and_include_plots(self) -> None:
        frame = checkerboard()
        frame_lookup = {0: frame}
        results = [
            extract_shi_tomasi_features(
                frame,
                ShiTomasiConfig(max_features=count),
                frame_index=0,
                first_feature_id=sum(REQUESTED_FEATURE_COUNTS[:i]),
            )
            for i, count in enumerate(REQUESTED_FEATURE_COUNTS)
        ]
        configuration = {
            "requested_counts": list(REQUESTED_FEATURE_COUNTS),
            "frame_indices": [0],
            "quality_level": 0.01,
            "min_feature_distance_px": 5.0,
            "block_size_px": 3,
            "bucket_grid": [8, 6],
            "feature_dtype": [],
            "feature_id_semantics": "test",
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            outputs = save_experiment_outputs(
                results, configuration, Path(temporary_dir), frame_lookup
            )
            self.assertEqual(set(outputs), {
                "features",
                "measurements",
                "configuration",
                "overlays",
                "spatial_buckets",
                "comparison",
                "report",
            })
            archive = np.load(outputs["features"], allow_pickle=False)
            stored = archive["features"]
            self.assertFalse(stored.dtype.hasobject)
            self.assertEqual(len(np.unique(stored["id"])), len(stored))
            self.assertTrue(np.isfinite(stored["quality"]).all())
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            report = outputs["report"].read_text()
            self.assertIn("Recommended count for the next experiment", report)


if __name__ == "__main__":
    unittest.main()
