import unittest

from src.parameter_sweep import build_sweep_configs
from src.three_way_comparison import ComparisonConfig


class ParameterSweepTests(unittest.TestCase):
    def test_sweep_is_one_factor_at_a_time_and_includes_grid_baseline(self):
        base = ComparisonConfig(
            interval_start=0,
            interval_count=1,
            feature_count=20,
            slot_count=20,
        )
        axes = (
            ("feature_count", (10, 20, 30)),
            ("grid_cell_size_px", (8, 16)),
            ("association_radius_px", (4.0, 8.0)),
            ("association_time_window_us", (10.0, 50_000.0)),
            ("alpha", (0.25, 0.5)),
            ("beta", (0.05, 0.1)),
        )
        configs = build_sweep_configs(base, axes)
        baseline = configs[0]
        self.assertEqual(baseline[1], "baseline")
        self.assertTrue(baseline[3].use_spatial_grid)
        for _, parameter, value, config in configs[1:]:
            differences = [
                name
                for name, _ in axes
                if getattr(config, name) != getattr(base, name)
            ]
            self.assertEqual(differences, [parameter])
            self.assertEqual(getattr(config, parameter), value)
            self.assertTrue(config.use_spatial_grid)
            self.assertEqual(config.slot_count, config.feature_count)

    def test_duplicate_baseline_axis_values_are_not_repeated(self):
        base = ComparisonConfig(
            interval_start=0,
            interval_count=1,
            feature_count=20,
            slot_count=20,
        )
        configs = build_sweep_configs(
            base,
            (("feature_count", (20, 30, 30)),),
        )
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[1][2], 30)


if __name__ == "__main__":
    unittest.main()
