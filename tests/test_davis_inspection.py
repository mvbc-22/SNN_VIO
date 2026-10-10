import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.davis_inspection import (
    DavisConfig,
    DavisRecording,
    inspect_davis,
    plot_timestamp_ranges,
    save_metadata,
)


def make_recording() -> DavisRecording:
    events = np.array(
        [(0, 0, 0, 0), (10, 3, 2, 1), (25, 1, 1, 0)],
        dtype=[("t", "i8"), ("x", "i8"), ("y", "i8"), ("p", "i8")],
    )
    aps = {"ts": np.array([0, 20], dtype=np.int64), "frames": np.zeros((2, 3, 4))}
    imu = {
        "ts": np.array([0, 12, 24], dtype=np.int64),
        "acc": np.zeros((3, 3)),
        "angV": np.zeros((3, 3)),
    }
    target = {
        "ts": np.array([0, 25], dtype=np.int64),
        "point": np.zeros((2, 3)),
    }
    return DavisRecording(events=events, imu=imu, aps=aps, target=target)


class DavisInspectionTests(unittest.TestCase):
    def test_valid_recording_metadata_and_units(self) -> None:
        recording = make_recording()
        metadata = inspect_davis(recording, DavisConfig(recording="synthetic"))
        streams = metadata["streams"]

        self.assertEqual(streams["events"]["sample_count"], 3)
        self.assertEqual(streams["events"]["start"], 0)
        self.assertEqual(streams["events"]["end"], 25)
        self.assertEqual(streams["aps_frames"]["frame_count"], 2)
        self.assertEqual(
            streams["aps_frames"]["resolution_px"], {"width": 4, "height": 3}
        )
        self.assertEqual(streams["aps_frames"]["end"], 20)
        self.assertEqual(streams["imu"]["end"], 24)
        self.assertEqual(streams["ground_truth"]["end"], 25)
        self.assertEqual(metadata["timestamp_unit"], "microseconds")
        self.assertFalse(metadata["synchronization_performed"])
        self.assertEqual(metadata["event_polarities_observed"], [0, 1])
        for stream in streams.values():
            if stream["available"]:
                self.assertEqual(stream["timestamp_unit"], "microseconds")

    def test_missing_event_field_is_rejected(self) -> None:
        recording = make_recording()
        events = np.zeros(2, dtype=[("t", "i8"), ("x", "i8"), ("p", "i8")])
        with self.assertRaisesRegex(ValueError, "missing fields"):
            inspect_davis(
                DavisRecording(events, recording.imu, recording.aps, recording.target)
            )

    def test_non_monotonic_stream_timestamps_are_rejected(self) -> None:
        recording = make_recording()
        events = recording.events.copy()
        events["t"] = [0, 20, 10]
        with self.assertRaisesRegex(ValueError, "monotonically ordered"):
            inspect_davis(
                DavisRecording(events, recording.imu, recording.aps, recording.target)
            )

    def test_event_coordinates_must_fit_sensor(self) -> None:
        recording = make_recording()
        events = recording.events.copy()
        events["x"][1] = 4
        with self.assertRaisesRegex(ValueError, "sensor dimensions"):
            inspect_davis(
                DavisRecording(events, recording.imu, recording.aps, recording.target)
            )

    def test_event_polarities_must_be_binary(self) -> None:
        recording = make_recording()
        events = recording.events.copy()
        events["p"][1] = 2
        with self.assertRaisesRegex(ValueError, "polarity"):
            inspect_davis(
                DavisRecording(events, recording.imu, recording.aps, recording.target)
            )

    def test_frame_timestamps_must_match_frame_count_and_be_ordered(self) -> None:
        recording = make_recording()
        aps = {"ts": np.array([10, 0]), "frames": recording.aps["frames"]}
        with self.assertRaisesRegex(ValueError, "monotonically ordered"):
            inspect_davis(
                DavisRecording(recording.events, recording.imu, aps, recording.target)
            )

        aps = {"ts": np.array([0]), "frames": recording.aps["frames"]}
        with self.assertRaisesRegex(ValueError, "does not match timestamp count"):
            inspect_davis(
                DavisRecording(recording.events, recording.imu, aps, recording.target)
            )

    def test_timestamp_units_must_be_integer_microseconds(self) -> None:
        recording = make_recording()
        imu = dict(recording.imu)
        imu["ts"] = np.array([0.0, 12.0, 24.0])
        with self.assertRaisesRegex(ValueError, "integer microseconds"):
            inspect_davis(
                DavisRecording(recording.events, imu, recording.aps, recording.target)
            )

    def test_optional_imu_and_ground_truth_are_reported(self) -> None:
        recording = make_recording()
        metadata = inspect_davis(
            DavisRecording(recording.events, None, recording.aps, None)
        )
        self.assertFalse(metadata["streams"]["imu"]["available"])
        self.assertFalse(metadata["streams"]["ground_truth"]["available"])

    def test_metadata_and_timestamp_plot_are_saved(self) -> None:
        metadata = inspect_davis(make_recording())
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            metadata_path = save_metadata(
                metadata, output_dir / "metadata.json"
            )
            plot_path = plot_timestamp_ranges(metadata, output_dir / "ranges.png")
            self.assertTrue(metadata_path.is_file())
            self.assertTrue(plot_path.is_file())
            self.assertGreater(plot_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
