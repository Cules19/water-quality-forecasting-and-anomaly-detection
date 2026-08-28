from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "run_lowcost_conformal.py"
SPEC = importlib.util.spec_from_file_location("lowcost_conformal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_finite_sample_radius_uses_higher_order_statistic() -> None:
    scores = np.arange(1, 11, dtype=float)
    assert MODULE.finite_sample_radius(scores, 0.8) == 9.0
    assert MODULE.finite_sample_radius(scores, 0.95) == 10.0


def test_site_hash_is_stable_and_target_specific() -> None:
    assert MODULE.site_hash(42, "pH", "A") == MODULE.site_hash(42, "pH", "A")
    assert MODULE.site_hash(42, "pH", "A") != MODULE.site_hash(42, "DO", "A")
