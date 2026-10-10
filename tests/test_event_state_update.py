import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.event_association import EVENTS_PER_FEATURE_DTYPE
from src.event_state_update import (
    EVOLUTION_DTYPE,
    run_asynchronous_updates,
    save_update_outputs,
)
from src.feature_state_memory import FeatureStateMemory
from src.frame_synchronization import FrameWindow


EVENT_DTYPE = np.dtype(
    [("t", np.int64), ("x", np.int64), ("y", np.int64), ("p", np.int8)]
)


def memory_with_state(x=10, y=20, timestamp=100, vx=0.1, vy=-0.2, confidence=1.0):
    memory = FeatureStateMemory(2)
    memory.allocate(x, y, timestamp, vx=vx, vy=vy, confidence=confidence)
    return memory


class AsynchronousFeatureStateUpdateTests(unittest.TestCase):
    def test_alpha_beta_update_matches_equations_and_updates_timestamp(self) -> None:
        memory = memory_with_state()
        events = np.array([(110, 13, 14, 1)], dtype=EVENT_DTYPE)
        evolution, per_feature, diagnostics = run_asynchronous_updates(
            events,
            memory,
            radius_px=10,
            time_window_us=100,
            alpha=0.5,
            beta=1.0,
            confidence_penalty=0.2,
        )
        state = memory.storage[0]
        self.assertEqual(float(state["x"]), 12.0)
        self.assertEqual(float(state["y"]), 16.0)
        self.assertAlmostEqual(float(state["vx"]), 0.3, places=6)
        self.assertAlmostEqual(float(state["vy"]), -0.6, places=6)
        self.assertEqual(int(state["timestamp"]), 110)
        self.assertAlmostEqual(float(state["confidence"]), 0.96, places=6)
        self.assertEqual(evolution.dtype, EVOLUTION_DTYPE)
        self.assertAlmostEqual(evolution[0]["predicted_x"], 11.0)
        self.assertAlmostEqual(evolution[0]["predicted_y"], 18.0)
        self.assertAlmostEqual(evolution[0]["residual_x"], 2.0)
        self.assertAlmostEqual(evolution[0]["residual_y"], -4.0)
        self.assertAlmostEqual(evolution[0]["position_update_magnitude"], np.hypot(1, -2))
        self.assertAlmostEqual(evolution[0]["velocity_update_magnitude"], np.hypot(0.2, -0.4))
        self.assertEqual(per_feature.dtype, EVENTS_PER_FEATURE_DTYPE)
        self.assertEqual(per_feature[0]["event_count"], 1)
        self.assertEqual(diagnostics["successful_updates"], 1)
        self.assertEqual(diagnostics["events_rejected"], 0)

    def test_zero_delta_t_updates_position_but_preserves_velocity(self) -> None:
        memory = memory_with_state(timestamp=100, vx=0.75, vy=-0.25)
        events = np.array([(100, 12, 18, 0)], dtype=EVENT_DTYPE)
        evolution, _, _ = run_asynchronous_updates(
            events,
            memory,
            radius_px=5,
            time_window_us=10,
            alpha=0.5,
            beta=1.0,
        )
        state = memory.storage[0]
        self.assertEqual(float(state["x"]), 11.0)
        self.assertEqual(float(state["y"]), 19.0)
        self.assertEqual(float(state["vx"]), 0.75)
        self.assertEqual(float(state["vy"]), -0.25)
        self.assertEqual(int(state["timestamp"]), 100)
        self.assertEqual(int(evolution[0]["delta_t_us"]), 0)
        self.assertEqual(float(evolution[0]["velocity_update_magnitude"]), 0.0)

    def test_small_positive_delta_t_uses_configured_denominator_floor(self) -> None:
        memory = memory_with_state(x=0, y=0, timestamp=0, vx=0, vy=0)
        events = np.array([(1, 1, 0, 1)], dtype=EVENT_DTYPE)
        evolution, _, _ = run_asynchronous_updates(
            events,
            memory,
            radius_px=3,
            time_window_us=10,
            alpha=0,
            beta=1,
            minimum_delta_t_us=10,
        )
        self.assertEqual(float(memory.storage[0]["x"]), 0.0)
        self.assertAlmostEqual(float(memory.storage[0]["vx"]), 0.1, places=7)
        self.assertAlmostEqual(float(evolution[0]["velocity_update_magnitude"]), 0.1)

    def test_online_stable_chronology_reassociates_against_updated_state(self) -> None:
        memory = memory_with_state(x=0, y=0, timestamp=0, vx=0, vy=0)
        events = np.array(
            [(2, 2, 0, 1), (1, 1, 0, 0), (1, 2, 0, 1)], dtype=EVENT_DTYPE
        )
        evolution, per_feature, diagnostics = run_asynchronous_updates(
            events,
            memory,
            radius_px=3,
            time_window_us=10,
            alpha=1,
            beta=0,
        )
        self.assertEqual(evolution["event_index"].tolist(), [1, 2, 0])
        self.assertEqual(evolution["t_event"].tolist(), [1, 1, 2])
        self.assertEqual(evolution["status"].tolist(), ["updated"] * 3)
        self.assertEqual(float(memory.storage[0]["x"]), 2.0)
        self.assertEqual(int(memory.storage[0]["timestamp"]), 2)
        self.assertEqual(int(per_feature[0]["event_count"]), 3)
        self.assertEqual(diagnostics["successful_updates"], 3)

    def test_unmatched_events_rejected_and_inactive_slots_unchanged(self) -> None:
        memory = memory_with_state(x=0, y=0, timestamp=0, vx=0, vy=0)
        memory.allocate(100, 100, 0)
        memory.invalidate(1)
        before_inactive = memory.storage[1].copy()
        events = np.array([(1, 100, 100, 1)], dtype=EVENT_DTYPE)
        evolution, per_feature, diagnostics = run_asynchronous_updates(
            events, memory, radius_px=2, time_window_us=10
        )
        self.assertEqual(evolution[0]["status"], "unmatched")
        self.assertEqual(int(evolution[0]["feature_slot"]), -1)
        self.assertEqual(diagnostics["events_rejected"], 1)
        self.assertEqual(int(per_feature[0]["event_count"]), 0)
        np.testing.assert_array_equal(memory.storage[1], before_inactive)

    def test_multiple_candidate_assignment_is_deterministic_and_counted(self) -> None:
        memory = FeatureStateMemory(2)
        memory.allocate(0, 0, 0, vx=0, vy=0)
        memory.allocate(2, 0, 0, vx=0, vy=0)
        events = np.array([(1, 1, 0, 1)], dtype=EVENT_DTYPE)
        evolution, _, diagnostics = run_asynchronous_updates(
            events, memory, radius_px=3, time_window_us=5, alpha=0, beta=0
        )
        self.assertEqual(int(evolution[0]["candidate_count"]), 2)
        self.assertEqual(int(evolution[0]["feature_id"]), 0)
        self.assertEqual(diagnostics["events_ambiguous"], 1)
        self.assertEqual(diagnostics["successful_updates"], 1)

    def test_confidence_penalty_is_configurable_and_bounded(self) -> None:
        memory = memory_with_state(x=0, y=0, timestamp=0, vx=0, vy=0, confidence=0.8)
        events = np.array([(1, 1, 0, 0)], dtype=EVENT_DTYPE)
        run_asynchronous_updates(
            events,
            memory,
            radius_px=2,
            time_window_us=5,
            alpha=0,
            beta=0,
            confidence_penalty=0.5,
        )
        # d2 / R2 = 1/4, so c' = 0.8 * (1 - 0.5 * 0.25).
        self.assertAlmostEqual(float(memory.storage[0]["confidence"]), 0.7, places=6)

    def test_empty_input_and_configuration_validation(self) -> None:
        memory = memory_with_state()
        empty = np.empty(0, dtype=EVENT_DTYPE)
        evolution, per_feature, diagnostics = run_asynchronous_updates(
            empty, memory, radius_px=5, time_window_us=10
        )
        self.assertEqual(len(evolution), 0)
        self.assertEqual(per_feature[0]["event_count"], 0)
        self.assertEqual(diagnostics["association_ratio"], 0.0)
        for options in (
            {"alpha": -0.1},
            {"beta": 1.1},
            {"minimum_delta_t_us": 0},
            {"confidence_penalty": float("nan")},
        ):
            with self.assertRaises(ValueError):
                run_asynchronous_updates(
                    empty, memory, radius_px=5, time_window_us=10, **options
                )

    def test_visualization_and_persistent_outputs(self) -> None:
        memory = memory_with_state(x=3, y=3, timestamp=10, vx=0, vy=0)
        initial_states = memory.storage.copy()
        events = np.array([(11, 4, 3, 1), (12, 5, 3, 0)], dtype=EVENT_DTYPE)
        evolution, per_feature, diagnostics = run_asynchronous_updates(
            events, memory, radius_px=4, time_window_us=10, alpha=0.5, beta=0.1
        )
        frame = np.zeros((12, 12), dtype=np.uint8)
        window = FrameWindow(
            index=0,
            frame_k=frame,
            frame_k_plus_1=frame,
            frame_k_timestamp_us=10,
            frame_k_plus_1_timestamp_us=20,
            frame_k_common_time_us=10,
            frame_k_plus_1_common_time_us=20,
            event_source=events,
            event_start_index=0,
            event_end_index=len(events),
            event_timestamp_offset_us=0,
            imu_k=None,
            groundtruth_k=None,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = save_update_outputs(
                frame,
                frame,
                initial_states,
                memory.storage,
                evolution,
                per_feature,
                diagnostics,
                window,
                {
                    "alpha": 0.5,
                    "beta": 0.1,
                    "confidence_penalty": 0.05,
                },
                Path(temp_dir),
            )
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            archive = np.load(outputs["evolution"], allow_pickle=False)
            self.assertEqual(archive["evolution"].dtype, EVOLUTION_DTYPE)
            report = outputs["report"].read_text()
            self.assertIn("pre-update predictions", report)
            self.assertIn("post-update positions", report)


if __name__ == "__main__":
    unittest.main()
