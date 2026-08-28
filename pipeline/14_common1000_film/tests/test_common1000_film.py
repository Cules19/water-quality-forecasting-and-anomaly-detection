from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "run_common1000_film_scaling.py"
SPEC = importlib.util.spec_from_file_location("common1000_film", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FilmTests(unittest.TestCase):
    def test_film_initialises_as_identity(self) -> None:
        model = MODULE.ConditionedSequenceRegressor(
            recurrent_type="gru",
            conditioning="film",
            n_dynamic=3,
            hidden_size=4,
            layers=1,
            n_sites=2,
            site_embedding_size=2,
            metadata_size=5,
            film_hidden=6,
            mlp_hidden=4,
        )
        gamma, beta = model.film_parameters(torch.randn(7, 5))
        torch.testing.assert_close(gamma, torch.ones_like(gamma))
        torch.testing.assert_close(beta, torch.zeros_like(beta))

    def test_film_modulates_every_output_state_shape(self) -> None:
        model = MODULE.ConditionedSequenceRegressor(
            recurrent_type="lstm",
            conditioning="film",
            n_dynamic=3,
            hidden_size=4,
            layers=1,
            n_sites=2,
            site_embedding_size=2,
            metadata_size=5,
            film_hidden=6,
            mlp_hidden=4,
        )
        result = model(torch.randn(7, 24, 3), torch.zeros(7, dtype=torch.long), torch.randn(7, 5))
        self.assertEqual(tuple(result.shape), (7,))

    def test_subset_ends_is_nested_by_site_code(self) -> None:
        arrays = {"site_code": np.array([0, 1, 2, 3], dtype=np.int16)}
        ends = np.arange(4, dtype=np.int64)
        np.testing.assert_array_equal(MODULE.subset_ends(ends, arrays, 2), [0, 1])

    def test_window_validation_uses_nanoseconds(self) -> None:
        arrays = {
            "site_code": np.zeros(30, dtype=np.int16),
            "timestamp_ns": np.arange(30, dtype=np.int64) * MODULE.HOUR_NS,
        }
        self.assertEqual(MODULE.validate_windows(np.array([23, 29]), arrays, 1), [])
        arrays["timestamp_ns"][0] += 1
        self.assertTrue(MODULE.validate_windows(np.array([23]), arrays, 1))

    def test_metric_values(self) -> None:
        result = MODULE.metric_values(np.array([6.0, 7.0, 8.0]), np.array([6.5, 7.0, 7.5]))
        self.assertAlmostEqual(result["rmse"], np.sqrt(0.5 / 3.0))
        self.assertAlmostEqual(result["r2"], 0.75)

    def test_identical_architecture_has_identical_seeded_initial_state(self) -> None:
        config = {
            "dynamic_features": ["a", "b", "c"],
            "hidden_size": 4,
            "recurrent_layers": 1,
            "film_hidden_size": 6,
            "mlp_hidden_size": 4,
        }
        spec_a = {"recurrent_type": "gru", "conditioning": "film"}
        spec_b = {"recurrent_type": "gru", "conditioning": "film"}
        MODULE.set_seed(42)
        a = MODULE.create_model(spec_a, config, 10, 5, torch.device("cpu"))
        MODULE.set_seed(42)
        b = MODULE.create_model(spec_b, config, 10, 5, torch.device("cpu"))
        self.assertEqual(MODULE.model_state_hash(a), MODULE.model_state_hash(b))


if __name__ == "__main__":
    unittest.main()
