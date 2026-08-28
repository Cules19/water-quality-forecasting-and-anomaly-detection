from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "run_error_onset_analysis.py"
SPEC = importlib.util.spec_from_file_location("error_onset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def synthetic_frame() -> pd.DataFrame:
    timestamp = pd.date_range("2025-01-01", periods=120, freq="1h")
    event = np.zeros(120, dtype=bool)
    event[60:66] = True
    residual = np.full(120, 0.1)
    residual[54:57] = 2.0
    return pd.DataFrame(
        {
            "site_id": "A",
            "timestamp": timestamp,
            "observed_native": np.linspace(6.5, 8.5, 120),
            "predicted_native": np.linspace(6.7, 8.3, 120),
            "in_event": event,
            "event_direction": np.where(event, "low", "none"),
            "residual": residual,
            "abs_residual": np.abs(residual),
        }
    )


def test_lead_time_detects_six_hour_warning() -> None:
    lead = MODULE.lead_time_rows(
        synthetic_frame(),
        threshold=1.0,
        maximum_lead_hours=48,
        minimum_consecutive_hours=3,
    )
    assert len(lead) == 1
    assert lead.iloc[0]["alerted"]
    assert lead.iloc[0]["lead_hours"] == 6.0


def test_lead_time_ignores_early_isolated_residual() -> None:
    frame = synthetic_frame()
    frame.loc[frame.index[20], "abs_residual"] = 2.0
    lead = MODULE.lead_time_rows(
        frame,
        threshold=1.0,
        maximum_lead_hours=48,
        minimum_consecutive_hours=3,
    )
    assert lead.iloc[0]["lead_hours"] == 6.0
    assert lead.iloc[0]["selected_episode_hours"] == 3


def test_lead_time_selects_alert_episode_nearest_event() -> None:
    frame = synthetic_frame()
    frame.loc[frame.index[30:33], "abs_residual"] = 2.0
    lead = MODULE.lead_time_rows(
        frame,
        threshold=1.0,
        maximum_lead_hours=48,
        minimum_consecutive_hours=3,
    )
    assert lead.iloc[0]["lead_hours"] == 6.0


def test_positive_shift_moves_event_earlier() -> None:
    frame = synthetic_frame()
    shifted = MODULE.shifted_event_mask(frame, lead_hours=6)
    assert shifted[54]
    assert not shifted[60]


def test_compression_slope_is_below_one() -> None:
    observed = pd.Series(np.linspace(0.0, 10.0, 100))
    predicted = 2.0 + 0.5 * observed
    metric = MODULE.compression_metrics(observed, predicted, tail_fraction=0.1)
    assert abs(metric["overall_slope_predicted_on_observed"] - 0.5) < 1e-12
    assert abs(metric["predicted_to_observed_p05_p95_range_ratio"] - 0.5) < 1e-12


def test_threshold_uses_validation_non_event_rows_only() -> None:
    frame = synthetic_frame()
    frame.loc[frame["in_event"], "abs_residual"] = 100.0
    threshold = MODULE.validation_threshold(frame, tail_fraction=0.05)
    assert threshold["fit_split"] == "validation"
    assert not threshold["test_labels_used_for_threshold"]
    assert threshold["threshold"] < 100.0
