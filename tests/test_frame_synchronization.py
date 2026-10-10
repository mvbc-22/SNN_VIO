import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.davis_inspection import DavisConfig, DavisRecording
from src.frame_synchronization import (
    load_timestamp_origins,
    make_frame_windows,
    validate_frame_windows,
)


def make_recording() -> DavisRecording:
    events = np.array(
        [
            (0, 0, 0, 0),
            (5, 0, 0, 1),
            (10, 1, 0, 0),
            (10, 1, 1, 1),
            (15, 2, 0, 1),
            (15, 2, 1, 0),
            (20, 2, 1, 0),
            (30, 3, 2, 1),
            (40, 0, 2, 0),
        ],
        dtype=[("t", "i8"), ("x", "i8"), ("y", "i8"), ("p", "i8")],
    )
    aps = {
        "ts": np.array([0, 10, 20, 30], dtype=np.int64),
        "frames": np.arange(4 * 3 * 4).reshape((4, 3, 4)),
    }
    imu = {
        "ts": np.array([0, 5, 10, 15, 20, 25, 30, 35], dtype=np.int64),
        "acc": np.zeros((8, 3)),
        "angV": np.zeros((8, 3)),
    }
    target = {
        "ts": np.array([0, 10, 20, 30, 40], dtype=np.int64),
        "point": np.zeros((5, 3)),
    }
    return DavisRecording(events=events, imu=imu, aps=aps, target=target)


class FrameSynchronizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recording = make_recording()
        self.origins = {
            "events": 100.0,
            "aps_frames": 100.000005,
            "imu": 100.000004,
            "ground_truth": 100.000008,
        }
        self.windows = make_frame_windows(self.recording, self.origins)

    def test_consecutive_half_open_event_intervals(self) -> None:
        first = self.windows.get_window(0)
        second = self.windows.get_window(1)
        self.assertEqual(first.interval_us, (5, 15))
        self.assertEqual(first.events_k["t"].tolist(), [5, 10, 10])
        self.assertEqual(second.events_k["t"].tolist(), [15, 15, 20])
        self.assertEqual(first.frame_k_timestamp_us, 0)
        self.assertEqual(first.frame_k_common_time_us, 5)
        self.assertEqual(first.frame_k_plus_1_timestamp_us, 10)
        self.assertEqual(first.frame_k_plus_1_common_time_us, 15)

    def test_boundary_events_are_assigned_once(self) -> None:
        # An event exactly at 15 is excluded from [5, 15) and included by the
        # following interval [15, 25).
        first = self.windows.get_window(0)
        second = self.windows.get_window(1)
        self.assertTrue(np.all(first.events_k["t"] < 15))
        self.assertEqual(second.events_k["t"][0], 15)
        self.assertEqual(
            first.event_end_index, second.event_start_index
        )

    def test_no_events_lost_or_duplicated_within_aps_span(self) -> None:
        result = validate_frame_windows(self.windows)
        self.assertTrue(result["passed"])
        self.assertEqual(result["events_within_aps_span"], 7)
        self.assertEqual(result["events_before_first_aps_frame"], 1)
        self.assertEqual(result["events_at_or_after_last_aps_frame"], 1)
        self.assertTrue(result["checks"]["no_event_overlap"])
        self.assertTrue(result["checks"]["no_event_lost_within_aps_span"])
        self.assertTrue(result["checks"]["imu_and_groundtruth_partitioned_without_loss"])

    def test_imu_and_groundtruth_are_sliced_on_the_common_interval(self) -> None:
        window = self.windows.get_window(0)
        self.assertEqual(window.imu_k.timestamps.tolist(), [5, 10])
        self.assertEqual(window.imu_k.common_timestamps_us.tolist(), [9, 14])
        self.assertEqual(window.groundtruth_k.timestamps.tolist(), [0])
        self.assertEqual(window.groundtruth_k.common_timestamps_us.tolist(), [8])

    def test_stream_data_remains_lazy_views(self) -> None:
        window = self.windows.get_window(0)
        self.assertTrue(np.shares_memory(window.events_k, self.recording.events))
        self.assertTrue(
            np.shares_memory(window.imu_k["acc"], self.recording.imu["acc"])
        )
        self.assertTrue(
            np.shares_memory(
                window.groundtruth_k["point"], self.recording.target["point"]
            )
        )

    def test_empty_event_windows_are_supported(self) -> None:
        events = self.recording.events[[0, 1, 8]].copy()
        recording = DavisRecording(
            events=events,
            imu=self.recording.imu,
            aps=self.recording.aps,
            target=self.recording.target,
        )
        windows = make_frame_windows(recording, self.origins)
        middle = windows.get_window(1)
        self.assertEqual(len(middle.events_k), 0)
        self.assertEqual(middle.event_index_range[0], middle.event_index_range[1])
        self.assertTrue(validate_frame_windows(windows)["passed"])

    def test_first_and_last_frames_define_full_window_span(self) -> None:
        self.assertEqual(len(self.windows), len(self.recording.aps["ts"]) - 1)
        np.testing.assert_array_equal(
            self.windows[0].frame_k,
            self.recording.aps["frames"][0],
        )
        np.testing.assert_array_equal(
            self.windows[-1].frame_k_plus_1,
            self.recording.aps["frames"][-1],
        )
        with self.assertRaises(IndexError):
            self.windows.get_window(len(self.windows))

    def test_missing_origin_for_available_stream_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            make_frame_windows(self.recording, {"events": 100.0, "aps_frames": 100.0})

    def test_manifest_origins_are_checked_against_loaded_data(self) -> None:
        recording = self.recording
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            data_dir = root / "data" / "DAVIS"
            bag_path = data_dir / "DAVISDATA" / "sample.bag"
            bag_path.parent.mkdir(parents=True)
            bag_path.write_bytes(b"sample rosbag")
            streams = {
                "events": (recording.events["t"], 100.0),
                "aps_frames": (recording.aps["ts"], 100.000005),
                "imu": (recording.imu["ts"], 100.000004),
                "ground_truth": (recording.target["ts"], 100.000008),
            }
            metadata = {}
            for name, (timestamps, origin) in streams.items():
                metadata[name] = {
                    "samples": len(timestamps),
                    "timestamp_start_s": origin,
                    "timestamp_end_s": origin + int(timestamps[-1]) / 1_000_000,
                }
            manifest_path = root / "data" / "metadata" / "dataset_manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "test",
                        "davis": {
                            "bag": "data/DAVIS/DAVISDATA/sample.bag",
                            "size_bytes": bag_path.stat().st_size,
                            "streams": metadata,
                        },
                    }
                )
            )
            config = DavisConfig(data_dir=data_dir, recording="sample")
            origins, details = load_timestamp_origins(
                recording, config, manifest_path
            )
            self.assertEqual(origins, {name: value[1] for name, value in streams.items()})
            self.assertEqual(details["bag_size_bytes"], bag_path.stat().st_size)

            document = json.loads(manifest_path.read_text())
            document["davis"]["streams"]["events"]["samples"] += 1
            manifest_path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "sample count differs"):
                load_timestamp_origins(recording, config, manifest_path)


if __name__ == "__main__":
    unittest.main()
