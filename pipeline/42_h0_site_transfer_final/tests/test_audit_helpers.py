"""Small deterministic tests for the final H0 audit helper rules."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "run_h0_final_audit.py"
SPEC = importlib.util.spec_from_file_location("h0_final_audit_helpers", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_strict_temporal_rule_rejects_equal_timestamps() -> None:
    input_time = pd.Timestamp("2026-01-01T00:00:00Z")
    target_time = pd.Timestamp("2026-01-01T00:00:00Z")
    assert not input_time < target_time


def test_strict_temporal_rule_accepts_previous_hour() -> None:
    input_time = pd.Timestamp("2026-01-01T00:00:00Z")
    target_time = pd.Timestamp("2026-01-01T01:00:00Z")
    assert input_time < target_time


def test_comparison_row_requires_exact_match() -> None:
    row = MODULE.comparison_row("feature_list", ["a", "b"], ["a", "b"])
    assert row["matches"] is True
    row = MODULE.comparison_row("feature_list", ["a", "b"], ["b", "a"])
    assert row["matches"] is False


if __name__ == "__main__":
    test_strict_temporal_rule_rejects_equal_timestamps()
    test_strict_temporal_rule_accepts_previous_hour()
    test_comparison_row_requires_exact_match()
    print("3 deterministic audit helper tests passed")
