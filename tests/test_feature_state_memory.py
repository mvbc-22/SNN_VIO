import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.aps_features import FEATURE_DTYPE
from src.feature_state_memory import (
    FEATURE_STATE_DTYPE,
    FeatureStateMemory,
    initialize_from_aps_features,
    run_initialization,
    save_initialization_outputs,
)


class FeatureStateMemoryTests(unittest.TestCase):
    def test_fixed_storage_and_invalid_slot_initialization(self) -> None:
        memory = FeatureStateMemory(3)
        self.assertEqual(memory.storage.shape, (3,))
        self.assertEqual(memory.storage.dtype, FEATURE_STATE_DTYPE)
        np.testing.assert_array_equal(memory.storage["id"], [-1, -1, -1])
        np.testing.assert_array_equal(memory.storage["valid"], [0, 0, 0])
        for field in ("x", "y", "vx", "vy", "confidence", "timestamp"):
            np.testing.assert_array_equal(memory.storage[field], [0, 0, 0])
        with self.assertRaises(ValueError):
            memory.storage["valid"][0] = 1

    def test_allocate_uses_first_free_slot_and_monotonic_ids(self) -> None:
        memory = FeatureStateMemory(2)
        first_slot = memory.allocate(4.5, 6.25, 123, confidence=0.8)
        second_slot = memory.allocate(8, 9, 124)
        self.assertEqual((first_slot, second_slot), (0, 1))
        self.assertEqual(memory.storage["id"].tolist(), [0, 1])
        self.assertEqual(memory.storage["valid"].tolist(), [1, 1])
        with self.assertRaises(BufferError):
            memory.allocate(1, 2, 125)
        memory.invalidate(first_slot)
        reused_slot = memory.allocate(2, 3, 126)
        self.assertEqual(reused_slot, first_slot)
        self.assertEqual(int(memory.storage["id"][reused_slot]), 2)

    def test_allocation_validates_types_ranges_and_finite_values(self) -> None:
        for slot_count in (0, -1):
            with self.assertRaises(ValueError):
                FeatureStateMemory(slot_count)
        memory = FeatureStateMemory(1)
        for confidence in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                memory.allocate(0, 0, 0, confidence=confidence)
        with self.assertRaises(ValueError):
            memory.allocate(float("inf"), 0, 0)
        with self.assertRaises(TypeError):
            memory.allocate(0, 0, 1.5)

    def test_update_changes_fields_and_preserves_omitted_values(self) -> None:
        memory = FeatureStateMemory(1)
        slot = memory.allocate(1, 2, 10, vx=0.5, vy=-0.25, confidence=0.4)
        original_id = int(memory.storage["id"][slot])
        memory.update(slot, x=4, y=5, timestamp=20, confidence=0.75)
        state = memory.storage[slot]
        self.assertEqual(int(state["id"]), original_id)
        self.assertEqual((float(state["x"]), float(state["y"])), (4.0, 5.0))
        self.assertEqual((float(state["vx"]), float(state["vy"])), (0.5, -0.25))
        self.assertEqual(float(state["confidence"]), 0.75)
        self.assertEqual(int(state["timestamp"]), 20)

    def test_update_rejects_invalid_slot_backward_time_and_bad_confidence(self) -> None:
        memory = FeatureStateMemory(2)
        with self.assertRaises(ValueError):
            memory.update(0, x=1, y=2, timestamp=1)
        with self.assertRaises(IndexError):
            memory.update(2, x=1, y=2, timestamp=1)
        slot = memory.allocate(1, 2, 10)
        with self.assertRaises(ValueError):
            memory.update(slot, x=1, y=2, timestamp=9)
        with self.assertRaises(ValueError):
            memory.update(slot, x=1, y=2, timestamp=10, confidence=2)
        with self.assertRaises(IndexError):
            memory.update(2, x=1, y=2, timestamp=10)

    def test_invalidate_resets_full_record_to_invalid_representation(self) -> None:
        memory = FeatureStateMemory(1)
        slot = memory.allocate(3, 4, 500, vx=1, vy=2, confidence=0.9)
        memory.invalidate(slot)
        state = memory.storage[slot]
        self.assertEqual(int(state["id"]), -1)
        self.assertEqual(int(state["valid"]), 0)
        for field in ("x", "y", "vx", "vy", "confidence", "timestamp"):
            self.assertEqual(float(state[field]), 0.0)
        memory.invalidate(slot)

    def test_prediction_uses_explicit_microsecond_delta_without_mutation(self) -> None:
        memory = FeatureStateMemory(1)
        slot = memory.allocate(
            10, 20, 100, vx=0.25, vy=-0.5, confidence=1.0
        )
        before = memory.storage[slot].copy()
        self.assertEqual(memory.predict(slot, 108), (12.0, 16.0))
        self.assertEqual(memory.predict(slot, 100), (10.0, 20.0))
        np.testing.assert_array_equal(memory.storage[slot], before)
        with self.assertRaises(ValueError):
            memory.predict(slot, 99)
        memory.invalidate(slot)
        with self.assertRaises(ValueError):
            memory.predict(slot, 108)

    def test_get_active_is_compact_ordered_copy(self) -> None:
        memory = FeatureStateMemory(3)
        memory.allocate(1, 2, 10)
        middle = memory.allocate(3, 4, 20)
        memory.allocate(5, 6, 30)
        memory.invalidate(middle)
        active = memory.get_active()
        self.assertEqual(active.dtype, FEATURE_STATE_DTYPE)
        self.assertEqual(active["id"].tolist(), [0, 2])
        active["x"][0] = 99
        self.assertEqual(float(memory.storage[0]["x"]), 1.0)

    def test_aps_initialization_sets_defined_defaults_and_respects_capacity(self) -> None:
        features = np.zeros(3, dtype=FEATURE_DTYPE)
        features["id"] = [10, 11, 12]
        features["x"] = [1.5, 2.5, 3.5]
        features["y"] = [4.5, 5.5, 6.5]
        features["valid"] = [1, 0, 1]
        memory = initialize_from_aps_features(features, 1234, 2)
        active = memory.get_active()
        self.assertEqual(len(active), 2)
        self.assertEqual(active["id"].tolist(), [0, 1])
        np.testing.assert_array_equal(active["x"], [1.5, 3.5])
        np.testing.assert_array_equal(active["vx"], [0, 0])
        np.testing.assert_array_equal(active["vy"], [0, 0])
        np.testing.assert_array_equal(active["confidence"], [1, 1])
        np.testing.assert_array_equal(active["timestamp"], [1234, 1234])
        with self.assertRaises(BufferError):
            initialize_from_aps_features(features, 1234, 1)
        with self.assertRaises(TypeError):
            initialize_from_aps_features(np.zeros(2, dtype=np.float32), 0, 2)

    def test_end_to_end_shi_tomasi_initialization_and_visualization(self) -> None:
        y, x = np.indices((96, 128))
        frame = (((x // 8 + y // 8) % 2) * 255).astype(np.uint8)
        features, memory = run_initialization(
            frame,
            timestamp_us=777,
            feature_count=20,
            slot_count=24,
            bucket_columns=4,
            bucket_rows=3,
        )
        self.assertEqual(features.dtype, FEATURE_DTYPE)
        self.assertGreater(len(features), 0)
        self.assertEqual(len(memory.get_active()), len(features))
        np.testing.assert_array_equal(
            memory.get_active()["timestamp"], np.full(len(features), 777)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = save_initialization_outputs(
                frame,
                features,
                memory,
                frame_index=4,
                output_dir=Path(temp_dir),
                configuration={"feature_count": 20, "timestamp_us": 777},
            )
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            archive = np.load(outputs["states"], allow_pickle=False)
            self.assertEqual(archive["states"].shape, (24,))
            self.assertEqual(archive["states"].dtype, FEATURE_STATE_DTYPE)
            self.assertEqual(len(archive["active_states"]), len(features))
            self.assertIn("No image patches", outputs["report"].read_text())


if __name__ == "__main__":
    unittest.main()
