from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "summarize_lowcost_deployment.py"
SPEC = importlib.util.spec_from_file_location("deployment_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_earliest_threshold_uses_first_tested_duration() -> None:
    metrics = pd.DataFrame(
        {
            "site_id": ["A"] * 4 + ["B"] * 4,
            "strategy": ["film_affine"] * 8,
            "warmup_order": [0, 1, 2, 6] * 2,
            "warmup_label": [
                "2_weeks",
                "1_month",
                "2_months",
                "12_months",
            ]
            * 2,
            "mae": [0.3, 0.19, 0.1, 0.08, 0.4, 0.3, 0.25, 0.21],
            "within_tolerance_fraction": [
                0.4,
                0.6,
                0.8,
                0.9,
                0.2,
                0.3,
                0.4,
                0.45,
            ],
        }
    )
    result = MODULE.earliest_site_thresholds(metrics, "pH", 0.2).set_index(
        "site_id"
    )
    assert result.loc["A", "earliest_mae_threshold_warmup"] == "1_month"
    assert result.loc["B", "earliest_mae_threshold_warmup"] == "not_achieved"
