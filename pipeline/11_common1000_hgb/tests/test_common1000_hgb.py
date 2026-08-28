from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "run_common1000_hgb.py"
SPEC = importlib.util.spec_from_file_location("common1000_hgb", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Common1000HGBTests(unittest.TestCase):
    def test_deterministic_time_positions_are_unique_and_cover_ends(self) -> None:
        positions = MODULE.deterministic_time_positions(100_000, 10_000)
        self.assertEqual(len(positions), 10_000)
        self.assertEqual(len(np.unique(positions)), 10_000)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 99_999)

    def test_small_site_uses_all_training_rows(self) -> None:
        np.testing.assert_array_equal(MODULE.deterministic_time_positions(700, 10_000), np.arange(700))

    def test_feature_leakage_screen_rejects_target_and_events(self) -> None:
        config = {
            "target_scaled": "00400",
            "target_native": "00400_native",
            "forbidden_feature_markers": ["00400", "ph", "target", "label", "event", "residual", "prediction", "future", "test", "validation"],
        }
        MODULE.assert_safe_features(["00010", "00095", "00300", "63680", "sin_hour"], config)
        with self.assertRaises(ValueError):
            MODULE.assert_safe_features(["00400"], config)
        with self.assertRaises(ValueError):
            MODULE.assert_safe_features(["in_ph_event"], config)

    def test_native_metric_accumulator(self) -> None:
        observed = np.array([6.0, 7.0, 8.0], dtype=np.float32)
        predicted = np.array([6.5, 7.0, 7.5], dtype=np.float32)
        acc = MODULE.empty_accumulator()
        MODULE.update_accumulator(acc, observed, predicted)
        row = MODULE.accumulator_row(acc)
        self.assertEqual(row["n_rows"], 3)
        self.assertAlmostEqual(row["mae"], 1.0 / 3.0)
        self.assertAlmostEqual(row["rmse"], np.sqrt(0.5 / 3.0))
        self.assertAlmostEqual(row["r2"], 0.75)

    def test_smoke_quotas_sum_to_requested_total(self) -> None:
        counts = pd.Series({"EA": 703, "N": 56, "USGS": 250})
        quotas = MODULE.largest_remainder_quotas(counts, 50)
        self.assertEqual(sum(quotas.values()), 50)
        self.assertTrue(all(value >= 1 for value in quotas.values()))


if __name__ == "__main__":
    unittest.main()
