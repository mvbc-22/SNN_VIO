import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.frame_synchronization import FrameWindow
from src.three_way_comparison import (
    ComparisonConfig,
    compare_three_systems,
    save_comparison,
)


EVENT_DTYPE = np.dtype(
    [("t", np.int64), ("x", np.int64), ("y", np.int64), ("p", np.int8)]
)


def frame(shift=0):
    y, x = np.indices((96, 128))
    image = (((((x - shift) // 8) + y // 8) % 2) * 255).astype(np.uint8)
    return image


def windows(count=2):
    images = [frame(index) for index in range(count + 1)]
    times = [100 + index * 100 for index in range(count + 1)]
    output = []
    for index in range(count):
        events = np.array(
            [
                (times[index] + 10, 8, 8, 1),
                (times[index] + 20, 24, 8, 0),
                (times[index] + 30, 40, 24, 1),
            ],
            dtype=EVENT_DTYPE,
        )
        output.append(
            FrameWindow(
                index=index,
                frame_k=images[index],
                frame_k_plus_1=images[index + 1],
                frame_k_timestamp_us=times[index],
                frame_k_plus_1_timestamp_us=times[index + 1],
                frame_k_common_time_us=times[index],
                frame_k_plus_1_common_time_us=times[index + 1],
                event_source=events,
                event_start_index=0,
                event_end_index=len(events),
                event_timestamp_offset_us=0,
                imu_k=None,
                groundtruth_k=None,
            )
        )
    return output


class ThreeWayComparisonTests(unittest.TestCase):
    def config(self, **kwargs):
        values = {
            "interval_start": 0,
            "interval_count": 2,
            "feature_count": 30,
            "slot_count": 30,
            "association_radius_px": 20,
            "association_time_window_us": 1000,
            "reference_match_radius_px": 25,
        }
        values.update(kwargs)
        return ComparisonConfig(**values)

    def test_same_initial_feature_records_and_contiguous_frames(self):
        result = compare_three_systems(windows(), self.config())
        self.assertTrue(result.checks["identical_initial_locations"])
        self.assertTrue(result.checks["identical_frame_intervals"])
        self.assertEqual(len(result.initial_features), 30)
        for system in ("frame_only", "event_driven", "hybrid"):
            self.assertEqual(len(result.survival[system]), 2)
            self.assertEqual(len(result.epe_samples[system]) <= 60, True)
        self.assertEqual(len(result.pair_rows), 6)

    def test_frame_only_does_not_process_events_and_events_systems_do(self):
        result = compare_three_systems(windows(), self.config())
        frame_rows = result.pair_rows[result.pair_rows["system"] == "frame_only"]
        event_rows = result.pair_rows[result.pair_rows["system"] == "event_driven"]
        hybrid_rows = result.pair_rows[result.pair_rows["system"] == "hybrid"]
        self.assertEqual(int(frame_rows["events_processed"].sum()), 0)
        self.assertGreater(int(event_rows["events_processed"].sum()), 0)
        self.assertEqual(
            int(event_rows["events_processed"].sum()),
            int(hybrid_rows["events_processed"].sum()),
        )
        self.assertEqual(
            int(event_rows["events_processed"].sum()),
            sum(len(window.events_k) for window in windows()),
        )

    def test_hybrid_counts_frame_corrections_in_state_updates(self):
        result = compare_three_systems(windows(), self.config())
        event_updates = int(
            result.pair_rows[result.pair_rows["system"] == "event_driven"][
                "state_updates"
            ].sum()
        )
        hybrid_updates = int(
            result.pair_rows[result.pair_rows["system"] == "hybrid"][
                "state_updates"
            ].sum()
        )
        self.assertGreaterEqual(hybrid_updates, event_updates)

    def test_contiguity_is_required(self):
        pair_windows = windows()
        broken = list(pair_windows)
        broken[1] = FrameWindow(
            **{
                **pair_windows[1].__dict__,
                "frame_k_common_time_us": pair_windows[1].frame_k_common_time_us + 1,
            }
        )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            compare_three_systems(broken, self.config())

    def test_comparison_outputs_share_metric_table_and_six_plots(self):
        pair_windows = windows()
        config = self.config()
        result = compare_three_systems(pair_windows, config)
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = save_comparison(pair_windows, config, result, Path(temp_dir))
            expected = {
                "epe_distribution",
                "track_lifetime",
                "feature_survival",
                "processing_time",
                "update_latency",
                "events_per_update",
            }
            self.assertTrue(expected.issubset(outputs))
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            summary = json.loads(outputs["summary"].read_text())
            self.assertEqual(summary["schema"], "snn-vio-three-way-comparison-v1")
            self.assertEqual(
                set(summary["systems"]),
                {"frame_only", "event_driven", "hybrid"},
            )
            self.assertIn(
                "| Metric | Frame-only | Event-driven | Hybrid |",
                summary["table_markdown"],
            )
            self.assertFalse(summary["systems"]["frame_only"]["metrics"]["event_data_used"])

    def test_configuration_rejects_invalid_parameters(self):
        for config in (
            self.config(interval_count=0),
            self.config(slot_count=10),
            self.config(alpha=float("nan")),
            self.config(association_radius_px=0),
        ):
            with self.assertRaises(ValueError):
                config.validate()


if __name__ == "__main__":
    unittest.main()
