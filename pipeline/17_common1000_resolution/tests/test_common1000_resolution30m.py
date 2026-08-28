from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_common1000_resolution30m.py"
SPEC = importlib.util.spec_from_file_location("common1000_resolution30m", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ResolutionTests(unittest.TestCase):
    def test_valid_48_step_half_hour_window(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=49, freq="30min"),
                "split": ["train"] * 49,
                "in_ph_event": [False] * 49,
                "ph_event_direction": ["none"] * 49,
                "event_id": [-1] * 49,
            }
        )
        result = MODULE.valid_windows(frame, 48, 30)
        self.assertEqual(result["local_end"].tolist(), [47, 48])

    def test_window_does_not_cross_gap(self) -> None:
        timestamp = list(pd.date_range("2024-01-01", periods=30, freq="30min")) + list(pd.date_range("2024-01-02", periods=30, freq="30min"))
        frame = pd.DataFrame({"timestamp": timestamp, "split": "train", "in_ph_event": False, "ph_event_direction": "none", "event_id": -1})
        self.assertTrue(MODULE.valid_windows(frame, 48, 30).empty)


if __name__ == "__main__":
    unittest.main()
