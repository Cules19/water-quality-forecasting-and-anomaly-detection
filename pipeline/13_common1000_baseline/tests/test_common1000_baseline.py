from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "run_common1000_baselines.py"
SPEC = importlib.util.spec_from_file_location("common1000_baseline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


FEATURES = [
    "00010",
    "00095",
    "00300",
    "63680",
    "hour",
    "month",
    "day_of_year",
    "sin_hour",
    "cos_hour",
    "sin_day",
    "cos_day",
]


class BaselineTests(unittest.TestCase):
    def arrays(self) -> dict[str, np.ndarray]:
        features = np.zeros((50, len(FEATURES)), dtype=np.float32)
        features[:, FEATURES.index("hour")] = np.arange(50) % 24
        features[:, FEATURES.index("month")] = 1
        return {
            "features": features,
            "target": np.linspace(0.2, 0.8, 50, dtype=np.float32),
            "timestamp_ns": np.arange(50, dtype=np.int64) * MODULE.HOUR_NS,
            "site_code": np.r_[np.zeros(25, dtype=np.int16), np.ones(25, dtype=np.int16)],
            "event_direction": np.zeros(50, dtype=np.int8),
            "event_id": np.full(50, -1, dtype=np.int64),
            "train_sample_ends": np.array([23, 24, 48, 49], dtype=np.int64),
            "val_ends": np.array([23, 48], dtype=np.int64),
            "test_ends": np.array([24, 49], dtype=np.int64),
        }

    def test_training_only_site_mean(self) -> None:
        arrays = self.arrays()
        stats = MODULE.fit_simple_statistics(arrays, {"n_sites": 2}, FEATURES)
        expected_0 = arrays["target"][[23, 24]].mean()
        expected_1 = arrays["target"][[48, 49]].mean()
        self.assertAlmostEqual(stats["site_mean"][0], expected_0)
        self.assertAlmostEqual(stats["site_mean"][1], expected_1)

    def test_site_without_training_uses_global_fallback(self) -> None:
        arrays = self.arrays()
        stats = MODULE.fit_simple_statistics(arrays, {"n_sites": 3}, FEATURES)
        self.assertFalse(stats["site_has_sampled_training"][2])
        self.assertAlmostEqual(stats["site_mean"][2], stats["global_mean"])

    def test_seasonal_falls_back_to_site_mean(self) -> None:
        arrays = self.arrays()
        stats = MODULE.fit_simple_statistics(arrays, {"n_sites": 2}, FEATURES)
        arrays["features"][10, FEATURES.index("month")] = 12
        prediction = MODULE.predict_seasonal(np.array([10]), arrays, stats, FEATURES)
        self.assertAlmostEqual(prediction[0], stats["site_mean"][0])

    def test_persistence_is_exactly_one_hour_and_same_site(self) -> None:
        arrays = self.arrays()
        self.assertEqual(MODULE.validate_persistence(np.array([24, 49]), arrays), [])
        prediction = MODULE.predict_persistence(np.array([24, 49]), arrays)
        np.testing.assert_allclose(prediction, arrays["target"][[23, 48]])
        arrays["timestamp_ns"][24] += 1
        self.assertTrue(MODULE.validate_persistence(np.array([24]), arrays))

    def test_window_validation_uses_nanoseconds(self) -> None:
        arrays = self.arrays()
        self.assertEqual(MODULE.validate_windows(arrays), [])
        arrays["timestamp_ns"][0] += 1
        self.assertTrue(MODULE.validate_windows(arrays))

    def test_metric_values(self) -> None:
        observed = np.array([6.0, 7.0, 8.0])
        predicted = np.array([6.5, 7.0, 7.5])
        result = MODULE.metric_values(observed, predicted)
        self.assertAlmostEqual(result["rmse"], np.sqrt(0.5 / 3.0))
        self.assertAlmostEqual(result["mae"], 1.0 / 3.0)
        self.assertAlmostEqual(result["r2"], 0.75)

    def test_tail_masks_are_direction_specific(self) -> None:
        residual = np.arange(-50, 50, dtype=float)
        _, masks = MODULE.tail_masks(residual, 0.1)
        self.assertTrue(np.all(residual[masks["negative"]] <= 0))
        self.assertTrue(np.all(residual[masks["positive"]] >= 0))
        self.assertEqual(int(masks["negative"].sum()), 10)
        self.assertEqual(int(masks["positive"].sum()), 10)

    def test_feature_leakage_check_exempts_only_persistence(self) -> None:
        config = {"forbidden_predictors": ["00400", "event_id"]}
        payload = {
            "models": {
                "B2_persistence_1h": {"inputs": ["previous-hour 00400"]},
                "safe": {"inputs": ["00010", "month"]},
            }
        }
        self.assertEqual(MODULE.validate_feature_manifest(payload, config), [])
        payload["models"]["safe"]["inputs"].append("00400")
        self.assertTrue(MODULE.validate_feature_manifest(payload, config))


if __name__ == "__main__":
    unittest.main()
