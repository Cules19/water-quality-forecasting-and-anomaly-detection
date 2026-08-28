from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_common1000_gru_fullmeta_nonevent.py"
SPEC = importlib.util.spec_from_file_location("common1000_gru_nonevent", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NoneventTests(unittest.TestCase):
    def test_overall_metrics(self) -> None:
        frame = pd.DataFrame({"site_id": ["A", "A"], "observed_ph": [7.0, 8.0], "predicted_ph": [7.0, 7.0]})
        result = MODULE.overall_metrics(frame, "M", "test")
        self.assertAlmostEqual(result["rmse"], np.sqrt(0.5))
        self.assertEqual(result["n_sites"], 1)

    def test_tail_selects_requested_count(self) -> None:
        frame = pd.DataFrame({"abs_residual": np.arange(100), "in_ph_event": [False] * 95 + [True] * 5})
        result = MODULE.global_tail_metrics(frame, "M", [0.05])[0]
        self.assertEqual(result["n_selected"], 5)
        self.assertEqual(result["event_enrichment"], 20.0)


if __name__ == "__main__":
    unittest.main()
