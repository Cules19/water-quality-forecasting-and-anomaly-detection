from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "prepare_leave_sites_out.py"
SPEC = importlib.util.spec_from_file_location("leave_sites_out", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_causal_fill_never_uses_future() -> None:
    values = pd.Series([1.0, np.nan, np.nan, 9.0])
    filled = MODULE.causal_fill(values, 1)
    assert filled.iloc[1] == 1.0
    assert np.isnan(filled.iloc[2])
    assert filled.iloc[3] == 9.0


def test_benson_krause_freshwater_saturation_matches_reference_values() -> None:
    saturation = MODULE.do_saturation_mg_l(np.array([0.0, 10.0, 20.0, 30.0]))
    expected = np.array([14.6208337, 11.28794737, 9.09242604, 7.55879605])
    np.testing.assert_allclose(saturation, expected, rtol=1e-7, atol=1e-7)


def test_ea_percent_saturation_conversion_uses_matched_temperature() -> None:
    converted = MODULE.convert_ea_do_percent_to_mg_l(
        pd.Series([100.0, 50.0, 75.0]),
        pd.Series([20.0, 10.0, np.nan]),
    )
    assert np.isclose(converted.iloc[0], 9.09242604)
    assert np.isclose(converted.iloc[1], 11.28794737 / 2.0)
    assert np.isnan(converted.iloc[2])


def test_site_split_is_disjoint_and_deterministic() -> None:
    sites = pd.DataFrame(
        {
            "site_id": [f"A{i:02d}" for i in range(10)] + [f"B{i:02d}" for i in range(10)],
            "network": ["A"] * 10 + ["B"] * 10,
        }
    )
    fractions = {"train": 0.7, "validation": 0.15, "test": 0.15}
    first = MODULE.assign_site_splits(sites, fractions, 42)
    second = MODULE.assign_site_splits(sites, fractions, 42)
    pd.testing.assert_frame_equal(first, second)
    groups = {name: set(first.loc[first.split.eq(name), "site_id"]) for name in MODULE.SPLITS}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]


def test_valid_window_target_uses_requested_horizon() -> None:
    n = 40
    frame = pd.DataFrame(
        {
            "target_00400_native": np.arange(n, dtype=float),
            "x_a_missing": np.zeros(n, dtype=np.int8),
            "x_b_missing": np.zeros(n, dtype=np.int8),
        }
    )
    index = MODULE.valid_window_targets(frame, "00400", ["a", "b"], 24, 3, 0.5)
    assert index[0] == 26
    assert index[-1] == 39
