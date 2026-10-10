import unittest

import numpy as np

from src.event_association import (
    AssociationResult,
    associate_events_brute_force,
)
from src.event_state_update import run_asynchronous_updates
from src.feature_state_memory import FeatureStateMemory
from src.grid_event_association import (
    associate_events_grid,
    compare_association_results,
)
from src.spatial_grid import SpatialGridConfig


EVENT_DTYPE = np.dtype(
    [("t", np.int64), ("x", np.int64), ("y", np.int64), ("p", np.int8)]
)


def make_memory(states, slots=None):
    memory = FeatureStateMemory(slots or len(states))
    for state in states:
        memory.allocate(
            state[0],
            state[1],
            state[2],
            vx=state[3],
            vy=state[4],
        )
    return memory


class GridEventAssociationTests(unittest.TestCase):
    def setUp(self):
        self.grid = SpatialGridConfig(240, 180, cell_width=16, cell_height=12)

    def test_grid_matches_brute_force_with_cross_cell_motion_and_ties(self):
        memory = make_memory(
            [
                (15, 15, 0, 2.0, 0.0),
                (40, 10, 0, -0.5, 0.5),
                (80, 60, 10, 0.0, 0.0),
                (100, 100, 0, 0.0, 0.0),
            ]
        )
        events = np.array(
            [
                (10, 35, 15, 1),  # cross-cell prediction and exact distance tie
                (10, 30, 25, 0),
                (20, 80, 60, 1),
                (10, 180, 150, 0),
            ],
            dtype=EVENT_DTYPE,
        )
        brute = associate_events_brute_force(events, memory, 5, 20)
        spatial = associate_events_grid(events, memory, 5, 20, self.grid)
        comparison = compare_association_results(brute, spatial)
        self.assertTrue(comparison["equivalent"], comparison)
        self.assertEqual(
            spatial.assignments["feature_id"].tolist(),
            brute.assignments["feature_id"].tolist(),
        )
        self.assertEqual(spatial.diagnostics["events_ambiguous"], 1)
        self.assertLessEqual(
            spatial.diagnostics["distance_calculations_total"],
            brute.diagnostics["distance_calculations_total"],
        )

    def test_randomized_events_preserve_all_association_fields(self):
        rng = np.random.default_rng(20261010)
        states = [
            (
                float(x),
                float(y),
                int(timestamp),
                float(vx),
                float(vy),
            )
            for x, y, timestamp, vx, vy in zip(
                rng.uniform(0, 240, 80),
                rng.uniform(0, 180, 80),
                rng.integers(0, 1000, 80),
                rng.uniform(-0.2, 0.2, 80),
                rng.uniform(-0.2, 0.2, 80),
            )
        ]
        memory = make_memory(states)
        events = np.empty(500, dtype=EVENT_DTYPE)
        events["t"] = rng.integers(0, 1200, len(events))
        events["x"] = rng.integers(0, 240, len(events))
        events["y"] = rng.integers(0, 180, len(events))
        events["p"] = rng.integers(0, 2, len(events))
        brute = associate_events_brute_force(events, memory, 9.5, 400)
        spatial = associate_events_grid(events, memory, 9.5, 400, self.grid)
        comparison = compare_association_results(brute, spatial)
        self.assertTrue(comparison["equivalent"], comparison)

    def test_grid_and_brute_updates_remain_equivalent(self):
        initial_states = [
            (10, 10, 100, 0.1, 0.0),
            (50, 40, 100, -0.1, 0.05),
            (100, 80, 100, 0.0, 0.0),
            (15, 15, 100, 0.0, 0.0),
        ]
        events = np.array(
            [
                (101, 10, 10, 1),
                (104, 11, 10, 0),
                (102, 50, 40, 1),
                (110, 101, 80, 0),
                (105, 11, 10, 1),
                (101, 20, 15, 0),
                (102, 20, 15, 1),
            ],
            dtype=EVENT_DTYPE,
        )
        brute_memory = make_memory(initial_states, slots=4)
        grid_memory = make_memory(initial_states, slots=4)
        brute_rows, brute_per_feature, _ = run_asynchronous_updates(
            events,
            brute_memory,
            radius_px=6,
            time_window_us=50,
            alpha=0.5,
            beta=0.1,
        )
        grid_rows, grid_per_feature, grid_diagnostics = run_asynchronous_updates(
            events,
            grid_memory,
            radius_px=6,
            time_window_us=50,
            alpha=0.5,
            beta=0.1,
            spatial_grid=self.grid,
        )
        for field in brute_rows.dtype.names:
            if field == "update_latency_us":
                continue
            np.testing.assert_array_equal(brute_rows[field], grid_rows[field])
        np.testing.assert_array_equal(brute_memory.storage, grid_memory.storage)
        np.testing.assert_array_equal(brute_per_feature, grid_per_feature)
        self.assertLessEqual(
            grid_diagnostics["distance_calculations_total"],
            len(events) * len(initial_states),
        )

    def test_configuration_and_comparison_tolerance_validation(self):
        with self.assertRaises(ValueError):
            SpatialGridConfig(0, 180)
        empty = AssociationResult(
            assignments=np.empty(0, dtype=[]),
            events_per_feature=np.empty(0, dtype=[]),
            diagnostics={},
        )
        with self.assertRaises(ValueError):
            compare_association_results(
                empty, empty, absolute_tolerance=float("nan")
            )


if __name__ == "__main__":
    unittest.main()
