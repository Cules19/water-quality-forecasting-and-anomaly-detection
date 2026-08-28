from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "run_common1000_sequence_models.py"
SPEC = importlib.util.spec_from_file_location("common1000_sequence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SequenceTests(unittest.TestCase):
    def test_valid_windows_require_24_consecutive_same_split_hours(self) -> None:
        timestamps = pd.date_range("2020-01-01", periods=30, freq="h")
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "split": ["train"] * 25 + ["val"] * 5,
                "in_ph_event": False,
                "ph_event_direction": "none",
                "event_id": -1,
            }
        )
        windows = MODULE.valid_window_index(frame, 24)
        self.assertEqual(windows["local_end"].tolist(), [23, 24])

    def test_gap_breaks_sequence(self) -> None:
        timestamps = list(pd.date_range("2020-01-01", periods=24, freq="h"))
        timestamps[12] = timestamps[11] + pd.Timedelta(hours=2)
        for index in range(13, 24):
            timestamps[index] = timestamps[index - 1] + pd.Timedelta(hours=1)
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "split": "train",
                "in_ph_event": False,
                "ph_event_direction": "none",
                "event_id": -1,
            }
        )
        self.assertTrue(MODULE.valid_window_index(frame, 24).empty)

    def test_deterministic_positions(self) -> None:
        positions = MODULE.deterministic_positions(100_000, 5_000)
        self.assertEqual(len(positions), 5_000)
        self.assertEqual(len(np.unique(positions)), 5_000)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 99_999)

    def test_event_direction(self) -> None:
        self.assertEqual(MODULE.event_direction_from_flags(True, False, False), "low")
        self.assertEqual(MODULE.event_direction_from_flags(False, True, False), "high")
        self.assertEqual(MODULE.event_direction_from_flags(True, True, False), "mixed")
        self.assertEqual(MODULE.event_direction_from_flags(False, False, False), "none")

    def test_metric_values(self) -> None:
        observed = np.array([6.0, 7.0, 8.0], dtype=np.float32)
        predicted = np.array([6.5, 7.0, 7.5], dtype=np.float32)
        result = MODULE.metric_values(observed, predicted)
        self.assertAlmostEqual(result["rmse"], np.sqrt(0.5 / 3.0))
        self.assertAlmostEqual(result["mae"], 1.0 / 3.0)
        self.assertAlmostEqual(result["r2"], 0.75)

    def test_window_array_validation_uses_nanoseconds(self) -> None:
        timestamps = np.arange(30, dtype=np.int64) * 3600 * 1_000_000_000
        site_code = np.zeros(30, dtype=np.int16)
        arrays = {
            "timestamp_ns": timestamps,
            "site_code": site_code,
            "train_ends": np.array([23, 24], dtype=np.int64),
            "val_ends": np.array([25], dtype=np.int64),
            "test_ends": np.array([29], dtype=np.int64),
        }
        self.assertEqual(MODULE.validate_window_arrays(arrays, 24), [])


if __name__ == "__main__":
    unittest.main()
