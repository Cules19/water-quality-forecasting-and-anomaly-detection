from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_common1000_site_analysis.py"
SPEC = importlib.util.spec_from_file_location("common1000_site_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SiteAnalysisTests(unittest.TestCase):
    def test_flow_descriptors(self) -> None:
        daily = pd.Series([1.0, 2.0, 4.0], index=pd.date_range("2024-01-01", periods=3, freq="D"))
        result = MODULE.flow_descriptors(daily)
        self.assertAlmostEqual(result["mean_flow"], 7 / 3)
        self.assertAlmostEqual(result["richards_baker_flashiness"], 3 / 7)
        self.assertAlmostEqual(result["q50_median_flow"], 2.0)

    def test_flashiness_does_not_bridge_gaps(self) -> None:
        daily = pd.Series([1.0, 10.0], index=pd.to_datetime(["2024-01-01", "2024-01-03"]))
        result = MODULE.flow_descriptors(daily)
        self.assertEqual(result["flow_consecutive_pairs"], 0)
        self.assertEqual(result["richards_baker_flashiness"], 0.0)

    def test_exact_global_top_mask(self) -> None:
        mask, threshold = MODULE.exact_global_top_mask(np.arange(100, dtype=float), 0.05)
        self.assertEqual(int(mask.sum()), 5)
        self.assertEqual(threshold, 95.0)

    def test_site_id_normalisation(self) -> None:
        self.assertEqual(MODULE.normalise_site_id("123.0"), "00000123")
        self.assertEqual(MODULE.normalise_site_id("e0001a"), "E0001A")


if __name__ == "__main__":
    unittest.main()
