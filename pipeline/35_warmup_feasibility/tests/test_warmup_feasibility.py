from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "audit_warmup_feasibility.py"
SPEC = importlib.util.spec_from_file_location("warmup_feasibility", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_calendar_offsets_are_exact() -> None:
    start = pd.Timestamp("2020-02-29 00:00:00")
    assert start + MODULE.date_offset({"days": 14}) == pd.Timestamp(
        "2020-03-14 00:00:00"
    )
    assert start + MODULE.date_offset({"months": 12}) == pd.Timestamp(
        "2021-02-28 00:00:00"
    )


def test_adaptation_and_evaluation_keys_do_not_overlap() -> None:
    timestamps = pd.date_range("2020-01-01", periods=24 * 400, freq="h")
    config = {
        "warmup_periods": [
            {"label": "2_weeks", "days": 14},
            {"label": "12_months", "months": 12},
        ],
        "fixed_evaluation_after": {"months": 12},
        "minimum_adaptation_targets": 168,
        "minimum_adaptation_target_coverage": 0.2,
        "minimum_evaluation_targets": 168,
        "immediate_evaluation_days": 30,
    }
    rows, summary = MODULE.build_site_period_rows(
        "00000001", "validation", timestamps, config
    )
    assert all(row["adaptation_fixed_overlap_targets"] == 0 for row in rows)
    assert all(row["adaptation_immediate_overlap_targets"] == 0 for row in rows)
    assert rows[0]["adaptation_targets"] == 14 * 24
    assert rows[1]["fixed_curve_eligible"]
    assert summary["full_fixed_curve_eligible"]


def test_fixed_evaluation_digest_is_constant_across_warmups() -> None:
    timestamps = pd.date_range("2021-01-01", periods=24 * 500, freq="h")
    config = {
        "warmup_periods": [
            {"label": "2_weeks", "days": 14},
            {"label": "1_month", "months": 1},
            {"label": "12_months", "months": 12},
        ],
        "fixed_evaluation_after": {"months": 12},
        "minimum_adaptation_targets": 24,
        "minimum_adaptation_target_coverage": 0.2,
        "minimum_evaluation_targets": 24,
        "immediate_evaluation_days": 30,
    }
    rows, _ = MODULE.build_site_period_rows(
        "00000002", "test", timestamps, config
    )
    assert len({row["fixed_evaluation_key_sha256"] for row in rows}) == 1
    assert [
        row["adaptation_targets"] for row in rows
    ] == sorted(row["adaptation_targets"] for row in rows)
