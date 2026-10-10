import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.event_association import (
    ASSOCIATION_DTYPE,
    EVENTS_PER_FEATURE_DTYPE,
    associate_events_brute_force,
    save_association_outputs,
)
from src.feature_state_memory import FEATURE_STATE_DTYPE, FeatureStateMemory
from src.frame_synchronization import FrameWindow


EVENT_DTYPE = np.dtype(
    [("t", np.int64), ("x", np.int64), ("y", np.int64), ("p", np.int8)]
)


def make_events(rows):
    return np.array(rows, dtype=EVENT_DTYPE)


def make_memory(states):
    memory = FeatureStateMemory(len(states))
    for state in states:
        memory.allocate(
            state[0],
            state[1],
            state[2],
            vx=state[3],
            vy=state[4],
            confidence=state[5] if len(state) > 5 else 1.0,
        )
    return memory


class BruteForceEventAssociationTests(unittest.TestCase):
    def test_associates_by_prediction_without_mutating_state(self) -> None:
        memory = make_memory([(10, 20, 100, 0.5, -0.25)])
        before = memory.storage.copy()
        events = make_events([(108, 12, 18, 1), (109, 12, 19, 0)])
        result = associate_events_brute_force(
            events, memory, radius_px=3.0, time_window_us=20
        )
        self.assertEqual(result.assignments["associated"].tolist(), [1, 1])
        self.assertEqual(result.assignments["feature_id"].tolist(), [0, 0])
        self.assertEqual(result.assignments["distance_squared"].tolist(), [4.0, 7.8125])
        np.testing.assert_array_equal(memory.storage, before)

    def test_strict_spatial_and_temporal_thresholds_reject_equal_boundary(self) -> None:
        memory = make_memory([(0, 0, 100, 0, 0)])
        events = make_events(
            [
                (100, 3, 0, 1),  # d2 == R2
                (110, 0, 0, 0),  # abs(dt) == T
            ]
        )
        result = associate_events_brute_force(
            events, memory, radius_px=3, time_window_us=10
        )
        self.assertEqual(result.assignments["associated"].tolist(), [0, 0])
        self.assertEqual(result.diagnostics["events_rejected"], 2)

    def test_no_match_is_rejected_and_feature_counts_include_zeroes(self) -> None:
        memory = make_memory([(0, 0, 0, 0, 0), (100, 100, 0, 0, 0)])
        result = associate_events_brute_force(
            make_events([(1, 50, 50, 1)]), memory, 2, 10
        )
        self.assertEqual(result.assignments["feature_id"].tolist(), [-1])
        self.assertEqual(result.assignments["feature_slot"].tolist(), [-1])
        self.assertEqual(result.events_per_feature.dtype, EVENTS_PER_FEATURE_DTYPE)
        self.assertEqual(result.events_per_feature["event_count"].tolist(), [0, 0])
        self.assertEqual(result.diagnostics["events_rejected"], 1)

    def test_multiple_matches_count_ambiguous_and_select_nearest(self) -> None:
        memory = make_memory([(0, 0, 0, 0, 0), (2, 0, 0, 0, 0)])
        result = associate_events_brute_force(
            make_events([(1, 1, 0, 1)]), memory, radius_px=3, time_window_us=5
        )
        assignment = result.assignments[0]
        self.assertEqual(int(assignment["candidate_count"]), 2)
        self.assertEqual(int(assignment["associated"]), 1)
        self.assertEqual(int(assignment["feature_id"]), 0)
        self.assertEqual(result.diagnostics["events_ambiguous"], 1)
        self.assertEqual(result.diagnostics["events_associated"], 1)

    def test_exact_distance_tie_breaks_by_lower_feature_id(self) -> None:
        memory = make_memory([(0, 0, 0, 0, 0), (2, 0, 0, 0, 0)])
        result = associate_events_brute_force(
            make_events([(0, 1, 0, 0)]), memory, radius_px=3, time_window_us=5
        )
        self.assertEqual(int(result.assignments[0]["feature_id"]), 0)

    def test_one_event_is_assigned_once_and_one_feature_can_receive_many(self) -> None:
        memory = make_memory([(5, 5, 0, 0, 0)])
        events = make_events([(1, 5, 5, 0), (2, 6, 5, 1), (3, 4, 5, 0)])
        result = associate_events_brute_force(events, memory, 3, 10)
        self.assertEqual(result.diagnostics["events_associated"], 3)
        self.assertEqual(result.events_per_feature["event_count"].tolist(), [3])
        self.assertTrue(np.all(result.assignments["feature_id"] == 0))

    def test_events_are_stably_processed_chronologically_and_polarity_is_preserved(self) -> None:
        memory = make_memory([(0, 0, 10, 0, 0)])
        events = make_events([(12, 0, 0, 1), (10, 0, 0, 0), (10, 0, 0, 1)])
        result = associate_events_brute_force(
            events, memory, 1, 10, event_timestamp_offset_us=5
        )
        self.assertEqual(result.assignments["event_index"].tolist(), [1, 2, 0])
        self.assertEqual(result.assignments["t_event"].tolist(), [10, 10, 12])
        self.assertEqual(result.assignments["t_common_us"].tolist(), [15, 15, 17])
        self.assertEqual(result.assignments["p"].tolist(), [0, 1, 1])

    def test_diagnostics_ratio_and_empty_event_stream(self) -> None:
        memory = make_memory([(0, 0, 0, 0, 0)])
        events = make_events([(0, 0, 0, 0), (1, 20, 20, 1)])
        result = associate_events_brute_force(events, memory, 2, 10)
        self.assertEqual(result.diagnostics["events_processed"], 2)
        self.assertEqual(result.diagnostics["events_associated"], 1)
        self.assertEqual(result.diagnostics["association_ratio"], 0.5)
        self.assertEqual(result.diagnostics["events_rejected"], 1)
        self.assertEqual(result.diagnostics["active_features"], 1)
        empty = associate_events_brute_force(
            np.empty(0, dtype=EVENT_DTYPE), memory, 2, 10
        )
        self.assertEqual(empty.assignments.dtype, ASSOCIATION_DTYPE)
        self.assertEqual(empty.diagnostics["association_ratio"], 0.0)
        self.assertEqual(empty.events_per_feature["event_count"].tolist(), [0])

    def test_noncontiguous_active_slots_keep_correct_slot_indices(self) -> None:
        memory = make_memory([(10, 10, 0, 0, 0), (20, 20, 0, 0, 0), (30, 30, 0, 0, 0)])
        memory.invalidate(1)
        result = associate_events_brute_force(
            make_events([(1, 30, 30, 1)]), memory, 2, 5
        )
        self.assertEqual(int(result.assignments[0]["feature_slot"]), 2)
        self.assertEqual(result.events_per_feature["slot"].tolist(), [0, 2])

    def test_validates_thresholds_and_event_schema(self) -> None:
        memory = make_memory([(0, 0, 0, 0, 0)])
        for radius, time_window in ((0, 1), (1, 0), (float("nan"), 1), (1, float("inf"))):
            with self.assertRaises(ValueError):
                associate_events_brute_force(
                    np.empty(0, dtype=EVENT_DTYPE), memory, radius, time_window
                )
        with self.assertRaises(TypeError):
            associate_events_brute_force(
                np.zeros(1, dtype=[("t", "i8"), ("x", "i8")]),
                memory,
                1,
                1,
            )
        bad_types = np.zeros(
            1, dtype=[("t", "f8"), ("x", "i8"), ("y", "i8"), ("p", "i8")]
        )
        with self.assertRaises(TypeError):
            associate_events_brute_force(bad_types, memory, 1, 1)

    def test_visualization_and_files_cover_frame_events_and_associations(self) -> None:
        memory = make_memory([(10, 10, 100, 0, 0)])
        events = make_events([(101, 10, 10, 1), (102, 30, 30, 0)])
        result = associate_events_brute_force(events, memory, 5, 10)
        frames = np.zeros((20, 20), dtype=np.uint8)
        window = FrameWindow(
            index=0,
            frame_k=frames,
            frame_k_plus_1=frames,
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
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = save_association_outputs(
                frames,
                memory.storage,
                result,
                window,
                5,
                10,
                Path(temp_dir),
            )
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            archive = np.load(outputs["assignments"], allow_pickle=False)
            self.assertEqual(archive["assignments"].dtype, ASSOCIATION_DTYPE)
            self.assertEqual(archive["assignments"]["p"].tolist(), [1, 0])
            self.assertIn("minimum squared distance", outputs["report"].read_text())


if __name__ == "__main__":
    unittest.main()
