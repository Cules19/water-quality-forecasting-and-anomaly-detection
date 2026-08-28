from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "supervisor_final_three_checks_v1"
SEED = 73129
BOOTSTRAPS = 10_000
MAD_CONSTANT = 1.4826
MIN_MAD = 1e-6
MIN_CALIBRATION_HOURS = 504
CALIBRATION_DAYS = 30

H01_PRED_DIR = (
    ROOT
    / "analysis_outputs/ph_common1000_lstm_film_final_rain_ablation_teacher_aligned_v1"
    / "full/predictions_by_site/ph_LSTM_RAIN_FINAL_H01/test"
)
H01_INPUT_DIR = (
    ROOT
    / "analysis_outputs/ph_common1000_teacher_archive_features_do_converted_v1"
    / "full/hourly_ph_rainfall_by_site"
)
H01_EFFECTIVE_CONFIG = (
    ROOT
    / "analysis_outputs/ph_common1000_lstm_film_final_rain_ablation_teacher_aligned_v1"
    / "full/reports/effective_config.json"
)
H01_SUCCESS = (
    ROOT
    / "analysis_outputs/ph_common1000_lstm_film_final_rain_ablation_teacher_aligned_v1"
    / "full/models/ph_LSTM_RAIN_FINAL_H01_SUCCESS.json"
)
H01_CODE = ROOT / "pipeline/31_common1000_rainfall_ablation/run_rainfall_ablation.py"

H00_MATCHED = (
    ROOT
    / "analysis_outputs/final910_h00_matched_baseline_completion_v1"
    / "predictions/matched_test_predictions.parquet"
)
RESIDUAL_SCORES = (
    ROOT
    / "analysis_outputs/cross_variable_hot_moment_incremental_closure_v1"
    / "residual_scores/matched_residuals.parquet"
)
CANONICAL_THRESHOLDS = (
    ROOT
    / "analysis_outputs/cross_variable_hot_moment_incremental_closure_v1"
    / "calibration/label_free_thresholds.json"
)
EVENT_MANIFEST = (
    ROOT
    / "analysis_outputs/cross_variable_hot_moment_incremental_closure_v1"
    / "data/matched_event_manifest.csv"
)
H00_MODEL_MANIFEST = (
    ROOT
    / "analysis_outputs/cross_variable_hot_moment_incremental_closure_v1"
    / "configs/model_manifest.json"
)
H00_CODE = ROOT / "pipeline/44_cross_variable_hot_moment_incremental_closure/run_incremental_closure.py"
QUICK_WINS_CODE = (
    ROOT
    / "analysis_outputs/supervisor_frozen_quick_wins_v1"
    / "appendix/run_frozen_quick_wins.py"
)

DO_BASE = (
    ROOT
    / "analysis_outputs/do_common1000_warmup_adaptation_lowcost_mg_l_test_v1"
    / "full"
)
DO_SITE_METRICS = DO_BASE / "reports/warmup_site_metrics.csv"
DO_CURVE_SUMMARY = DO_BASE / "reports/warmup_curve_summary.csv"
DO_PREDICTION_MANIFEST = (
    ROOT
    / "analysis_outputs/h0_site_transfer_warmup_final_audit_v1"
    / "predictions/prediction_manifest.csv"
)

RUN_LOG: list[str] = []
INPUT_FILES: dict[Path, str] = {}


def log(message: str) -> None:
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    line = f"{stamp} | {message}"
    RUN_LOG.append(line)
    print(line, flush=True)


def require(path: Path, role: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {role}: {path}")
    INPUT_FILES[path] = role
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_mad(values: Iterable[float] | np.ndarray) -> float:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan
    center = np.median(array)
    return float(MAD_CONSTANT * np.median(np.abs(array - center)))


def safe_r2(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float, bool]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    sst = float(np.sum((observed - observed.mean()) ** 2))
    scale = max(float(np.sum(observed**2)), 1.0)
    tolerance = max(1e-12, 16.0 * np.finfo(float).eps * scale)
    eligible = bool(np.isfinite(sst) and sst > tolerance)
    if not eligible:
        return np.nan, tolerance, False
    sse = float(np.sum((observed - predicted) ** 2))
    return 1.0 - sse / sst, tolerance, True


def pooled_metrics(frame: pd.DataFrame, observed: str, predicted: str) -> dict[str, float]:
    y = frame[observed].to_numpy(dtype=float)
    p = frame[predicted].to_numpy(dtype=float)
    error = y - p
    r2, _, eligible = safe_r2(y, p)
    return {
        "rows": int(len(frame)),
        "sites": int(frame["site_id"].nunique()),
        "pooled_mae": float(np.mean(np.abs(error))),
        "pooled_rmse": float(np.sqrt(np.mean(error**2))),
        "pooled_bias_observed_minus_predicted": float(np.mean(error)),
        "pooled_r2": float(r2) if eligible else np.nan,
    }


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan
    return tuple(np.quantile(finite, [0.025, 0.975]).tolist())


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    sample = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    return percentile_ci(values[sample].mean(axis=1))


def metric_row(
    task: str,
    record_type: str,
    cohort: str,
    model: str,
    metric: str,
    value: object,
    *,
    ci_lower: object = np.nan,
    ci_upper: object = np.nan,
    sites: object = np.nan,
    rows: object = np.nan,
    label: str = "",
    notes: str = "",
) -> dict[str, object]:
    return {
        "task": task,
        "record_type": record_type,
        "cohort": cohort,
        "model": model,
        "metric": metric,
        "value": value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "sites": sites,
        "rows": rows,
        "label": label,
        "notes": notes,
    }


def task1_persistence() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    log("Task 1: loading frozen H01 predictions and exact T-1 native observations")
    require(H01_EFFECTIVE_CONFIG, "H01 effective configuration")
    require(H01_SUCCESS, "H01 success manifest")
    require(H01_CODE, "H01 canonical implementation")
    pred_files = sorted(H01_PRED_DIR.glob("*.parquet"))
    if not pred_files:
        raise FileNotFoundError(f"No H01 prediction files in {H01_PRED_DIR}")
    parts: list[pd.DataFrame] = []
    exclusion_counts = {
        "missing_exact_t_minus_1": 0,
        "t_minus_1_not_qc_observed": 0,
        "t_minus_1_nonfinite_native_ph": 0,
    }
    for path in pred_files:
        require(path, "H01 per-site frozen test predictions")
        pred = pd.read_parquet(
            path,
            columns=[
                "site_id",
                "timestamp",
                "split",
                "target",
                "horizon_hours",
                "sequence_length_hours",
                "model_id",
                "observed_native",
                "predicted_native",
            ],
        )
        pred["site_id"] = pred["site_id"].astype(str)
        if pred["site_id"].nunique() != 1:
            raise RuntimeError(f"Mixed site IDs in {path}")
        if not pred["split"].eq("test").all():
            raise RuntimeError(f"Non-test row in {path}")
        if not pred["model_id"].eq("ph_LSTM_RAIN_FINAL_H01").all():
            raise RuntimeError(f"Unexpected model ID in {path}")
        if not pred["target"].eq("ph").all():
            raise RuntimeError(f"Unexpected target in {path}")
        if not pred["horizon_hours"].eq(1).all():
            raise RuntimeError(f"Off-by-one risk: horizon is not one hour in {path}")
        pred = pred.rename(columns={"timestamp": "target_timestamp"})
        pred["target_timestamp"] = pd.to_datetime(pred["target_timestamp"])
        pred["required_input_timestamp"] = pred["target_timestamp"] - pd.Timedelta(hours=1)
        site_id = str(pred["site_id"].iloc[0])
        source_path = H01_INPUT_DIR / f"{site_id}.parquet"
        require(source_path, "H01 per-site native hourly input")
        native = pd.read_parquet(
            source_path,
            columns=["timestamp", "target_00400_native", "target_00400_observed"],
        )
        native["timestamp"] = pd.to_datetime(native["timestamp"])
        if native.duplicated("timestamp").any():
            raise RuntimeError(f"Duplicate native timestamp for {site_id}")
        joined = pred.merge(
            native,
            how="left",
            left_on="required_input_timestamp",
            right_on="timestamp",
            validate="many_to_one",
            indicator=True,
        )
        missing = joined["_merge"].ne("both")
        not_observed = joined["_merge"].eq("both") & ~joined["target_00400_observed"].fillna(False)
        nonfinite = joined["_merge"].eq("both") & joined["target_00400_observed"].fillna(False) & ~np.isfinite(
            joined["target_00400_native"].to_numpy(dtype=float)
        )
        exclusion_counts["missing_exact_t_minus_1"] += int(missing.sum())
        exclusion_counts["t_minus_1_not_qc_observed"] += int(not_observed.sum())
        exclusion_counts["t_minus_1_nonfinite_native_ph"] += int(nonfinite.sum())
        joined["persistence_valid"] = ~(missing | not_observed | nonfinite)
        joined["persistence_prediction"] = joined["target_00400_native"].where(joined["persistence_valid"])
        parts.append(
            joined[
                [
                    "site_id",
                    "target_timestamp",
                    "required_input_timestamp",
                    "horizon_hours",
                    "sequence_length_hours",
                    "observed_native",
                    "predicted_native",
                    "persistence_prediction",
                    "persistence_valid",
                ]
            ]
        )
    full = pd.concat(parts, ignore_index=True)
    if full.duplicated(["site_id", "target_timestamp"]).any():
        raise RuntimeError("Duplicate H01 target key")
    if len(full) != 1_820_275 or full["site_id"].nunique() != 135:
        raise RuntimeError(f"H01 identity mismatch: {len(full)} rows, {full['site_id'].nunique()} sites")
    if not (full["required_input_timestamp"] == full["target_timestamp"] - pd.Timedelta(hours=1)).all():
        raise RuntimeError("Off-by-one mismatch in permitted input cutoff")
    matched = full.loc[full["persistence_valid"]].copy()
    if matched.empty:
        raise RuntimeError("No valid exact persistence matches")

    site_rows: list[dict[str, object]] = []
    for site_id, site in matched.groupby("site_id", sort=True):
        y = site["observed_native"].to_numpy(dtype=float)
        lstm = site["predicted_native"].to_numpy(dtype=float)
        pers = site["persistence_prediction"].to_numpy(dtype=float)
        r2_lstm, tol, eligible = safe_r2(y, lstm)
        r2_pers, _, _ = safe_r2(y, pers)
        lstm_err = y - lstm
        pers_err = y - pers
        mae_lstm = float(np.mean(np.abs(lstm_err)))
        mae_pers = float(np.mean(np.abs(pers_err)))
        diff = mae_lstm - mae_pers
        site_rows.append(
            {
                "site_id": site_id,
                "rows": len(site),
                "lstm_mae": mae_lstm,
                "persistence_mae": mae_pers,
                "paired_mae_difference_lstm_minus_persistence": diff,
                "lstm_rmse": float(np.sqrt(np.mean(lstm_err**2))),
                "persistence_rmse": float(np.sqrt(np.mean(pers_err**2))),
                "lstm_bias_observed_minus_predicted": float(np.mean(lstm_err)),
                "persistence_bias_observed_minus_predicted": float(np.mean(pers_err)),
                "r2_eligible": eligible,
                "r2_denominator_tolerance": tol,
                "lstm_r2": r2_lstm,
                "persistence_r2": r2_pers,
                "comparison": "improved" if diff < -1e-12 else ("worsened" if diff > 1e-12 else "tied"),
            }
        )
    site_metrics = pd.DataFrame(site_rows)
    full_metrics = pooled_metrics(full, "observed_native", "predicted_native")
    lstm_metrics = pooled_metrics(matched, "observed_native", "predicted_native")
    persistence_metrics = pooled_metrics(matched, "observed_native", "persistence_prediction")
    for metrics, prefix in ((lstm_metrics, "lstm"), (persistence_metrics, "persistence")):
        metrics["macro_site_mae"] = float(site_metrics[f"{prefix}_mae"].mean())
        metrics["macro_site_rmse"] = float(site_metrics[f"{prefix}_rmse"].mean())
        metrics["median_site_mae"] = float(site_metrics[f"{prefix}_mae"].median())
        metrics["mean_eligible_site_r2"] = float(site_metrics.loc[site_metrics["r2_eligible"], f"{prefix}_r2"].mean())
        metrics["median_eligible_site_r2"] = float(site_metrics.loc[site_metrics["r2_eligible"], f"{prefix}_r2"].median())
        metrics["eligible_site_r2_count"] = int(site_metrics["r2_eligible"].sum())
    rng = np.random.default_rng(SEED)
    ci_low, ci_high = bootstrap_mean(
        site_metrics["paired_mae_difference_lstm_minus_persistence"].to_numpy(dtype=float), rng
    )
    paired_point = float(site_metrics["paired_mae_difference_lstm_minus_persistence"].mean())
    if lstm_metrics["macro_site_mae"] < persistence_metrics["macro_site_mae"]:
        if ci_high < 0 and lstm_metrics["macro_site_rmse"] <= persistence_metrics["macro_site_rmse"]:
            label = "CLEAR_LSTM_INCREMENT"
        else:
            label = "POINT_ADVANTAGE_ONLY"
    else:
        label = "NO_LSTM_INCREMENT"

    rows: list[dict[str, object]] = []
    for metric, value in full_metrics.items():
        rows.append(
            metric_row(
                "H01_PERSISTENCE",
                "full_cohort",
                "full_h01",
                "ph_LSTM_RAIN_FINAL_H01",
                metric,
                value,
                sites=full_metrics["sites"],
                rows=full_metrics["rows"],
            )
        )
    for model, metrics in (("LSTM", lstm_metrics), ("persistence", persistence_metrics)):
        for metric, value in metrics.items():
            rows.append(
                metric_row(
                    "H01_PERSISTENCE",
                    "matched_metric",
                    "exact_T_minus_1_matched",
                    model,
                    metric,
                    value,
                    sites=lstm_metrics["sites"],
                    rows=lstm_metrics["rows"],
                    label=label,
                )
            )
    rows.extend(
        [
            metric_row(
                "H01_PERSISTENCE",
                "paired_comparison",
                "exact_T_minus_1_matched",
                "LSTM_minus_persistence",
                "macro_site_mae_difference",
                paired_point,
                ci_lower=ci_low,
                ci_upper=ci_high,
                sites=len(site_metrics),
                rows=len(matched),
                label=label,
                notes=f"{BOOTSTRAPS}-replicate complete-site percentile bootstrap; seed={SEED}",
            ),
            metric_row(
                "H01_PERSISTENCE",
                "paired_comparison",
                "exact_T_minus_1_matched",
                "LSTM_minus_persistence",
                "improved_sites",
                int(site_metrics["comparison"].eq("improved").sum()),
                label=label,
            ),
            metric_row(
                "H01_PERSISTENCE",
                "paired_comparison",
                "exact_T_minus_1_matched",
                "LSTM_minus_persistence",
                "worsened_sites",
                int(site_metrics["comparison"].eq("worsened").sum()),
                label=label,
            ),
            metric_row(
                "H01_PERSISTENCE",
                "paired_comparison",
                "exact_T_minus_1_matched",
                "LSTM_minus_persistence",
                "tied_sites",
                int(site_metrics["comparison"].eq("tied").sum()),
                label=label,
            ),
        ]
    )
    for reason, count in exclusion_counts.items():
        rows.append(
            metric_row(
                "H01_PERSISTENCE",
                "exclusion",
                "full_to_matched",
                "persistence",
                reason,
                count,
                sites=full.loc[~full["persistence_valid"], "site_id"].nunique(),
                rows=int((~full["persistence_valid"]).sum()),
            )
        )
    rows.append(
        metric_row(
            "H01_PERSISTENCE",
            "identity_check",
            "full_h01",
            "ph_LSTM_RAIN_FINAL_H01",
            "off_by_one_mismatch_rows",
            0,
            notes="Canonical horizon=1 implementation ends input at target_index-horizon; every retained row maps to exact target_timestamp-1h.",
        )
    )
    summary = pd.DataFrame(rows)
    result = {
        "label": label,
        "full": full_metrics,
        "matched_lstm": lstm_metrics,
        "matched_persistence": persistence_metrics,
        "paired_difference": paired_point,
        "paired_ci": [ci_low, ci_high],
        "exclusions": exclusion_counts,
    }
    log(f"Task 1 complete: {label}; matched rows={len(matched):,}")
    return summary, site_metrics, result


def enrichment_counts(frame: pd.DataFrame, flag: str, event_mask: pd.Series, background_mask: pd.Series) -> dict[str, float]:
    event_rows = int(event_mask.sum())
    background_rows = int(background_mask.sum())
    event_alerts = int((event_mask & frame[flag]).sum())
    background_alerts = int((background_mask & frame[flag]).sum())
    event_rate = event_alerts / event_rows if event_rows else np.nan
    background_rate = background_alerts / background_rows if background_rows else np.nan
    enrichment = event_rate / background_rate if background_rate > 0 else np.nan
    return {
        "event_hours": event_rows,
        "event_alerts": event_alerts,
        "event_alert_rate": event_rate,
        "background_hours": background_rows,
        "background_alerts": background_alerts,
        "background_alert_rate": background_rate,
        "enrichment": enrichment,
    }


def task2_coupling() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    log("Task 2: reconstructing validation-frozen B0 coupling reference")
    require(RESIDUAL_SCORES, "frozen matched H00 residual scores")
    require(CANONICAL_THRESHOLDS, "canonical validation-only signed thresholds")
    require(EVENT_MANIFEST, "threshold-defined event manifest")
    require(H00_MODEL_MANIFEST, "H00 model manifest")
    require(H00_CODE, "canonical directional scoring implementation")
    require(QUICK_WINS_CODE, "established directional quick-wins implementation")
    columns = [
        "site_id",
        "target_timestamp",
        "split",
        "observed_ph",
        "in_event",
        "event_direction",
        "event_id",
        "calibration_period",
        "calibration_eligible",
        "evaluation_period",
        "signed_score_cross",
    ]
    frame = pd.read_parquet(RESIDUAL_SCORES, columns=columns)
    frame["site_id"] = frame["site_id"].astype(str)
    frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"])
    if frame.duplicated(["split", "site_id", "target_timestamp"]).any():
        raise RuntimeError("Duplicate H00 support/query key")
    with CANONICAL_THRESHOLDS.open("r", encoding="utf-8") as handle:
        canonical = json.load(handle)
    canonical_h00 = canonical["models"]["cross_variable_h00"]
    h00_threshold = float(canonical_h00["signed_score_q05"])
    expected_policy = "validation evaluation scores after each site's fixed earliest-30-day calibration; event labels not read"
    if canonical.get("policy") != expected_policy:
        raise RuntimeError("Canonical validation threshold policy mismatch")
    if int(canonical.get("calibration_days", -1)) != CALIBRATION_DAYS:
        raise RuntimeError("Canonical calibration days mismatch")
    if int(canonical.get("minimum_calibration_valid_hours", -1)) != MIN_CALIBRATION_HOURS:
        raise RuntimeError("Canonical calibration eligibility mismatch")

    frame["b0_prediction"] = np.nan
    frame["b0_residual"] = np.nan
    frame["b0_signed_score"] = np.nan
    calibration_info: list[dict[str, object]] = []
    validation_support_residuals: list[np.ndarray] = []
    for (split, site_id), group in frame.groupby(["split", "site_id"], sort=False):
        support = group.loc[group["calibration_period"]]
        eligible = bool(group["calibration_eligible"].iloc[0])
        if group["calibration_eligible"].nunique() != 1:
            raise RuntimeError(f"Mixed calibration eligibility for {split}/{site_id}")
        if eligible != (len(support) >= MIN_CALIBRATION_HOURS):
            raise RuntimeError(f"Eligibility reproduction mismatch for {split}/{site_id}")
        mean_observed = float(support["observed_ph"].mean()) if len(support) else np.nan
        residual = support["observed_ph"].to_numpy(dtype=float) - mean_observed
        if split == "validation" and len(residual):
            validation_support_residuals.append(residual)
        center = float(np.median(residual)) if len(residual) else np.nan
        scale_raw = raw_mad(residual)
        calibration_info.append(
            {
                "split": split,
                "site_id": site_id,
                "support_rows": len(support),
                "eligible": eligible,
                "support_mean_observed_ph": mean_observed,
                "support_residual_center": center,
                "support_residual_raw_mad": scale_raw,
            }
        )
        frame.loc[group.index, "b0_prediction"] = mean_observed
        frame.loc[group.index, "b0_residual"] = group["observed_ph"].to_numpy(dtype=float) - mean_observed
    global_b0_mad = raw_mad(np.concatenate(validation_support_residuals))
    cal = pd.DataFrame(calibration_info)
    cal["fallback_used"] = cal["eligible"] & np.isfinite(cal["support_residual_raw_mad"]) & (
        cal["support_residual_raw_mad"] < MIN_MAD
    )
    cal["scale_used"] = cal["support_residual_raw_mad"].where(~cal["fallback_used"], global_b0_mad)
    for row in cal.itertuples(index=False):
        index = frame.index[(frame["split"].eq(row.split)) & (frame["site_id"].eq(row.site_id))]
        if row.eligible and np.isfinite(row.support_residual_center) and np.isfinite(row.scale_used):
            frame.loc[index, "b0_signed_score"] = (
                frame.loc[index, "b0_residual"] - row.support_residual_center
            ) / max(float(row.scale_used), MIN_MAD)
    validation_eval = frame.loc[frame["split"].eq("validation") & frame["evaluation_period"]]
    valid_b0 = validation_eval["b0_signed_score"].dropna()
    if valid_b0.empty:
        raise RuntimeError("COUPLING_NULL_NOT_RUN_VALIDATION_THRESHOLD_IDENTITY_UNAVAILABLE")
    b0_threshold = float(valid_b0.quantile(0.05))
    b0_threshold_payload = {
        "reference": "B0_30_day_support_mean",
        "residual_convention": "observed_ph - support_mean_observed_ph",
        "standardisation": "site support-residual median centre and raw MAD (1.4826); canonical <1e-6 validation-global fallback",
        "fit_split": "validation",
        "fit_rows": int(len(valid_b0)),
        "fit_sites": int(validation_eval.loc[valid_b0.index, "site_id"].nunique()),
        "signed_score_q05": b0_threshold,
        "quantile_method": "pandas Series.quantile(q=0.05), linear interpolation default",
        "event_labels_used": False,
        "calibration_days": CALIBRATION_DAYS,
        "minimum_calibration_valid_hours": MIN_CALIBRATION_HOURS,
        "fallback_validation_global_mad": global_b0_mad,
        "fallback_sites": cal.loc[cal["fallback_used"], ["split", "site_id"]].astype(str).agg(":".join, axis=1).tolist(),
        "zero_or_nonfinite_raw_mad_sites": cal.loc[
            ~np.isfinite(cal["support_residual_raw_mad"]) | (cal["support_residual_raw_mad"] <= 0),
            ["split", "site_id"],
        ].astype(str).agg(":".join, axis=1).tolist(),
    }
    threshold_path = OUT / "b0_validation_threshold.json"
    threshold_path.write_text(json.dumps(b0_threshold_payload, indent=2), encoding="utf-8")

    test_eval = frame.loc[frame["split"].eq("test") & frame["evaluation_period"]].copy()
    h00_valid = np.isfinite(test_eval["signed_score_cross"])
    b0_valid_test = np.isfinite(test_eval["b0_signed_score"])
    common = test_eval.loc[h00_valid & b0_valid_test].copy()
    common["h00_negative_tail"] = common["signed_score_cross"] < h00_threshold
    common["b0_negative_tail"] = common["b0_signed_score"] < b0_threshold
    low_event = common["in_event"] & common["event_direction"].astype(str).str.lower().eq("low")
    background_all = ~common["in_event"]
    event_sites = sorted(common.loc[low_event, "site_id"].unique().tolist())
    if len(event_sites) != 18:
        raise RuntimeError(f"Directional low-event site identity mismatch: {len(event_sites)}")
    event_manifest = pd.read_csv(EVENT_MANIFEST, dtype={"site_id": str})
    formal_low_events = event_manifest.loc[
        event_manifest["event_direction"].astype(str).str.lower().eq("low")
        & event_manifest["event_evaluation_included"].astype(bool)
    ]
    event_count = int(len(formal_low_events))
    if event_count != 1_177:
        raise RuntimeError(f"Directional low-event episode identity mismatch: {event_count}")
    if int(formal_low_events["common_event_rows_in_evaluation"].sum()) != int(low_event.sum()):
        raise RuntimeError("Formal low-event manifest hours do not match common residual event hours")
    same_site = common["site_id"].isin(event_sites)
    background_same = background_all & same_site
    comparisons = {
        "H00_all_site_background": enrichment_counts(common, "h00_negative_tail", low_event, background_all),
        "B0_all_site_background": enrichment_counts(common, "b0_negative_tail", low_event, background_all),
        "H00_same_event_site_background": enrichment_counts(common, "h00_negative_tail", low_event, background_same),
        "B0_same_event_site_background": enrichment_counts(common, "b0_negative_tail", low_event, background_same),
    }
    if not math.isclose(comparisons["H00_all_site_background"]["enrichment"], 4.716543356, rel_tol=0, abs_tol=1e-6):
        raise RuntimeError("Canonical all-site directional enrichment did not reproduce")
    if not math.isclose(comparisons["H00_same_event_site_background"]["enrichment"], 4.5224, rel_tol=0, abs_tol=5e-4):
        raise RuntimeError("Canonical same-event-site directional enrichment did not reproduce")

    site_rows: list[dict[str, object]] = []
    for site_id, site in common.groupby("site_id", sort=True):
        ev = site["in_event"] & site["event_direction"].astype(str).str.lower().eq("low")
        bg = ~site["in_event"]
        row: dict[str, object] = {
            "site_id": site_id,
            "is_low_event_site": site_id in event_sites,
            "low_event_hours": int(ev.sum()),
            "background_hours": int(bg.sum()),
        }
        for name, flag in (("h00", "h00_negative_tail"), ("b0", "b0_negative_tail")):
            row[f"{name}_event_alerts"] = int((ev & site[flag]).sum())
            row[f"{name}_background_alerts"] = int((bg & site[flag]).sum())
            row[f"{name}_event_alert_rate"] = row[f"{name}_event_alerts"] / row["low_event_hours"] if row["low_event_hours"] else np.nan
            row[f"{name}_background_alert_rate"] = row[f"{name}_background_alerts"] / row["background_hours"] if row["background_hours"] else np.nan
            row[f"{name}_site_enrichment_descriptive"] = (
                row[f"{name}_event_alert_rate"] / row[f"{name}_background_alert_rate"]
                if row["low_event_hours"] and row[f"{name}_background_alert_rate"] > 0
                else np.nan
            )
        site_rows.append(row)
    site_metrics = pd.DataFrame(site_rows)
    paired = site_metrics.loc[site_metrics["is_low_event_site"]].copy()
    if len(paired) != 18:
        raise RuntimeError("Paired bootstrap did not retain all 18 low-event sites")
    columns_boot = [
        "h00_event_alerts",
        "low_event_hours",
        "h00_background_alerts",
        "background_hours",
        "b0_event_alerts",
        "b0_background_alerts",
    ]
    values = paired[columns_boot].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    sample = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    selected = values[sample].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h00_enrichment = (selected[:, 0] / selected[:, 1]) / (selected[:, 2] / selected[:, 3])
        b0_enrichment = (selected[:, 4] / selected[:, 1]) / (selected[:, 5] / selected[:, 3])
        log_difference = np.log(h00_enrichment) - np.log(b0_enrichment)
    valid_boot = np.isfinite(log_difference)
    ci_low, ci_high = percentile_ci(log_difference[valid_boot])
    h00_same = comparisons["H00_same_event_site_background"]["enrichment"]
    b0_same = comparisons["B0_same_event_site_background"]["enrichment"]
    point_log_diff = float(np.log(h00_same) - np.log(b0_same))
    point_ratio = float(h00_same / b0_same)
    if ci_low > 0:
        label = "INCREMENT_BEYOND_COUPLING"
    elif ci_high < 0:
        label = "B0_STRONGER_THAN_H00"
    else:
        label = "COUPLING_ONLY_COMPATIBLE"

    rows: list[dict[str, object]] = []
    for comparison, metrics in comparisons.items():
        model = "H00" if comparison.startswith("H00") else "B0"
        cohort = "all_126_site_background" if "all_site" in comparison else "same_18_event_site_background"
        for metric, value in metrics.items():
            rows.append(
                metric_row(
                    "COUPLING_NULL",
                    "enrichment",
                    cohort,
                    model,
                    metric,
                    value,
                    sites=126 if "all_site" in comparison else 18,
                    rows=int(metrics["event_hours"] + metrics["background_hours"]),
                    label=label,
                )
            )
    rows.extend(
        [
            metric_row(
                "COUPLING_NULL",
                "paired_comparison",
                "same_18_event_site_background",
                "H00_vs_B0",
                "enrichment_ratio_h00_over_b0",
                point_ratio,
                label=label,
                sites=18,
            ),
            metric_row(
                "COUPLING_NULL",
                "paired_comparison",
                "same_18_event_site_background",
                "H00_vs_B0",
                "log_enrichment_difference_h00_minus_b0",
                point_log_diff,
                ci_lower=ci_low,
                ci_upper=ci_high,
                label=label,
                sites=18,
                notes=f"{BOOTSTRAPS} paired complete-site replicates; seed={SEED}; no continuity correction; nonfinite zero-rate replicates excluded",
            ),
            metric_row(
                "COUPLING_NULL",
                "bootstrap_audit",
                "same_18_event_site_background",
                "H00_vs_B0",
                "valid_bootstrap_replicates",
                int(valid_boot.sum()),
                sites=18,
            ),
            metric_row(
                "COUPLING_NULL",
                "bootstrap_audit",
                "same_18_event_site_background",
                "H00_vs_B0",
                "invalid_bootstrap_replicates",
                int((~valid_boot).sum()),
                sites=18,
            ),
            metric_row(
                "COUPLING_NULL",
                "threshold",
                "validation_evaluation",
                "H00",
                "signed_score_q05",
                h00_threshold,
                sites=canonical_h00["fit_sites"],
                rows=canonical_h00["fit_rows"],
                notes=expected_policy,
            ),
            metric_row(
                "COUPLING_NULL",
                "threshold",
                "validation_evaluation",
                "B0",
                "signed_score_q05",
                b0_threshold,
                sites=b0_threshold_payload["fit_sites"],
                rows=b0_threshold_payload["fit_rows"],
                notes=expected_policy,
            ),
            metric_row(
                "COUPLING_NULL",
                "key_audit",
                "test_evaluation",
                "H00",
                "detector_specific_missing_keys",
                int((~h00_valid).sum()),
                sites=test_eval.loc[~h00_valid, "site_id"].nunique(),
            ),
            metric_row(
                "COUPLING_NULL",
                "key_audit",
                "test_evaluation",
                "B0",
                "detector_specific_missing_keys",
                int((~b0_valid_test).sum()),
                sites=test_eval.loc[~b0_valid_test, "site_id"].nunique(),
            ),
            metric_row(
                "COUPLING_NULL",
                "identity_check",
                "common_test_evaluation",
                "H00_and_B0",
                "threshold_defined_low_ph_episodes",
                event_count,
                sites=18,
            ),
        ]
    )
    summary = pd.DataFrame(rows)
    result = {
        "label": label,
        "comparisons": comparisons,
        "point_enrichment_ratio": point_ratio,
        "point_log_difference": point_log_diff,
        "paired_log_difference_ci": [ci_low, ci_high],
        "valid_bootstrap_replicates": int(valid_boot.sum()),
        "invalid_bootstrap_replicates": int((~valid_boot).sum()),
        "event_sites": event_sites,
        "episodes": event_count,
        "common_rows": int(len(common)),
    }
    log(f"Task 2 complete: {label}; H00={h00_same:.6f}, B0={b0_same:.6f}")
    return summary, site_metrics, result, b0_threshold_payload


def site_sufficient_stats(
    site_id: str,
    n: int,
    sae: float,
    sse: float,
    observed_sum: float,
    observed_sum_sq: float,
    *,
    target: str,
    model: str,
    units: str,
    provenance: str,
) -> dict[str, object]:
    mean_y = observed_sum / n
    sst = float(observed_sum_sq - observed_sum**2 / n)
    if sst < 0 and abs(sst) < 1e-8 * max(abs(observed_sum_sq), 1.0):
        sst = 0.0
    tolerance = max(1e-12, 16.0 * np.finfo(float).eps * max(abs(observed_sum_sq), 1.0))
    eligible = bool(np.isfinite(sst) and sst > tolerance)
    sd = float(np.sqrt(sst / n)) if eligible else np.nan
    mae = float(sae / n)
    rmse = float(np.sqrt(sse / n))
    return {
        "target": target,
        "model": model,
        "site_id": str(site_id),
        "units": units,
        "rows": int(n),
        "site_mae": mae,
        "site_rmse": rmse,
        "site_observed_mean": mean_y,
        "site_observed_population_sd": sd,
        "site_sst": sst,
        "site_sse": sse,
        "nmae_eligible": eligible,
        "nse_eligible": eligible,
        "denominator_tolerance": tolerance,
        "site_nmae": mae / sd if eligible else np.nan,
        "site_nse": 1.0 - sse / sst if eligible else np.nan,
        "provenance": provenance,
    }


def bootstrap_normalised(site: pd.DataFrame) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(SEED)
    index = rng.integers(0, len(site), size=(BOOTSTRAPS, len(site)))
    mae = site["site_mae"].to_numpy(dtype=float)[index]
    nmae = site["site_nmae"].to_numpy(dtype=float)[index]
    nse = site["site_nse"].to_numpy(dtype=float)[index]
    return {
        "macro_site_mae": percentile_ci(mae.mean(axis=1)),
        "mean_site_nmae": percentile_ci(np.nanmean(nmae, axis=1)),
        "median_site_nmae": percentile_ci(np.nanmedian(nmae, axis=1)),
        "mean_site_nse": percentile_ci(np.nanmean(nse, axis=1)),
        "median_site_nse": percentile_ci(np.nanmedian(nse, axis=1)),
    }


def task3_normalised() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    log("Task 3: computing within-site normalised pH metrics and exact DO sufficient-statistic metrics")
    require(H00_MATCHED, "final-910 matched H00 query predictions")
    require(DO_SITE_METRICS, "selected DO adapter per-site frozen metrics")
    require(DO_CURVE_SUMMARY, "selected DO adapter frozen curve summary")
    if DO_PREDICTION_MANIFEST.exists():
        require(DO_PREDICTION_MANIFEST, "DO row-persistence audit manifest")
    pH = pd.read_parquet(
        H00_MATCHED,
        columns=["site_id", "target_timestamp", "y_observed", "pred_lstm_offset", "pred_hgb_offset", "support_or_query"],
    )
    pH["site_id"] = pH["site_id"].astype(str)
    pH = pH.loc[pH["support_or_query"].eq("query")].copy()
    if pH.duplicated(["site_id", "target_timestamp"]).any():
        raise RuntimeError("Duplicate final-910 pH query key")
    if len(pH) != 1_493_309 or pH["site_id"].nunique() != 126:
        raise RuntimeError(f"pH H00 cohort identity mismatch: {len(pH)} rows, {pH['site_id'].nunique()} sites")

    site_rows: list[dict[str, object]] = []
    for site_id, site in pH.groupby("site_id", sort=True):
        y = site["y_observed"].to_numpy(dtype=float)
        observed_sum = float(y.sum())
        observed_sum_sq = float(np.sum(y**2))
        for model, column in (("pH_LSTM_own_offset_H00", "pred_lstm_offset"), ("pH_HGB_SUMMARY48_own_offset_H00", "pred_hgb_offset")):
            pred = site[column].to_numpy(dtype=float)
            error = y - pred
            site_rows.append(
                site_sufficient_stats(
                    site_id,
                    len(site),
                    float(np.abs(error).sum()),
                    float(np.sum(error**2)),
                    observed_sum,
                    observed_sum_sq,
                    target="pH",
                    model=model,
                    units="pH",
                    provenance="row-level final-910 matched own-offset frozen query",
                )
            )
    do_all = pd.read_csv(DO_SITE_METRICS, dtype={"site_id": str})
    do_selected = do_all.loc[
        do_all["split"].eq("test")
        & do_all["warmup_label"].eq("3_months")
        & do_all["strategy"].eq("film_affine")
    ].copy()
    if len(do_selected) != 39 or int(do_selected["n"].sum()) != 1_071_897:
        raise RuntimeError(f"DO three-month cohort identity mismatch: {len(do_selected)} sites, {do_selected['n'].sum()} rows")
    for row in do_selected.itertuples(index=False):
        shard = DO_BASE / "shards/test" / str(row.site_id) / "04_3_months_film_affine.json"
        require(shard, "selected DO adapter per-site frozen sufficient statistics")
        site_rows.append(
            site_sufficient_stats(
                str(row.site_id),
                int(row.n),
                float(row.sae),
                float(row.sse),
                float(row.observed_sum),
                float(row.observed_sum_sq),
                target="DO",
                model="DO_CHEAP3_RAIN_3_month_film_affine_H01",
                units="mg L^-1",
                provenance="lossless derivation from frozen per-site n/SAE/SSE/sum(y)/sum(y^2); row-level adapter predictions were not persisted",
            )
        )
    site_metrics = pd.DataFrame(site_rows)

    summary_rows: list[dict[str, object]] = []
    model_results: dict[str, dict[str, object]] = {}
    for model, site in site_metrics.groupby("model", sort=False):
        eligible = site["nmae_eligible"] & site["nse_eligible"]
        valid = site.loc[eligible]
        total_rows = int(site["rows"].sum())
        pooled_mae = float((site["site_mae"] * site["rows"]).sum() / total_rows)
        pooled_rmse = float(np.sqrt((site["site_sse"].sum()) / total_rows))
        pooled_within_nse = float(1.0 - site["site_sse"].sum() / valid["site_sst"].sum())
        metrics = {
            "sites": int(len(site)),
            "rows": total_rows,
            "pooled_mae": pooled_mae,
            "pooled_rmse": pooled_rmse,
            "macro_site_mae": float(site["site_mae"].mean()),
            "macro_site_rmse": float(site["site_rmse"].mean()),
            "pooled_within_site_nse": pooled_within_nse,
            "mean_site_nmae": float(valid["site_nmae"].mean()),
            "median_site_nmae": float(valid["site_nmae"].median()),
            "site_nmae_q25": float(valid["site_nmae"].quantile(0.25)),
            "site_nmae_q75": float(valid["site_nmae"].quantile(0.75)),
            "mean_site_nse": float(valid["site_nse"].mean()),
            "median_site_nse": float(valid["site_nse"].median()),
            "site_nse_q25": float(valid["site_nse"].quantile(0.25)),
            "site_nse_q75": float(valid["site_nse"].quantile(0.75)),
            "proportion_sites_nse_above_zero": float((valid["site_nse"] > 0).mean()),
            "eligible_sites": int(eligible.sum()),
            "excluded_low_variance_sites": int((~eligible).sum()),
            "excluded_site_ids": ";".join(site.loc[~eligible, "site_id"].astype(str).tolist()),
        }
        cis = bootstrap_normalised(site)
        model_results[model] = {**metrics, "bootstrap_cis": cis}
        for metric, value in metrics.items():
            ci = cis.get(metric, (np.nan, np.nan))
            summary_rows.append(
                metric_row(
                    "NORMALISED_ERROR",
                    "model_summary",
                    "frozen_query",
                    model,
                    metric,
                    value,
                    ci_lower=ci[0],
                    ci_upper=ci[1],
                    sites=metrics["sites"],
                    rows=metrics["rows"],
                    notes=f"site bootstrap seed={SEED}" if metric in cis else "",
                )
            )
    lstm_model = "pH_LSTM_own_offset_H00"
    hgb_model = "pH_HGB_SUMMARY48_own_offset_H00"
    p_lstm = site_metrics.loc[site_metrics["model"].eq(lstm_model)].set_index("site_id")
    p_hgb = site_metrics.loc[site_metrics["model"].eq(hgb_model)].set_index("site_id")
    if set(p_lstm.index) != set(p_hgb.index):
        raise RuntimeError("pH paired site sets differ")
    pair = p_lstm[["site_mae", "site_nmae", "site_nse"]].join(
        p_hgb[["site_mae", "site_nmae", "site_nse"]], lsuffix="_lstm", rsuffix="_hgb", validate="one_to_one"
    )
    paired_results: dict[str, object] = {}
    rng = np.random.default_rng(SEED)
    sample = rng.integers(0, len(pair), size=(BOOTSTRAPS, len(pair)))
    for metric in ("site_mae", "site_nmae", "site_nse"):
        diff = pair[f"{metric}_lstm"].to_numpy(dtype=float) - pair[f"{metric}_hgb"].to_numpy(dtype=float)
        point = float(np.nanmean(diff))
        boot = np.nanmean(diff[sample], axis=1)
        ci_low, ci_high = percentile_ci(boot)
        paired_results[metric] = {"mean_difference": point, "ci": [ci_low, ci_high]}
        summary_rows.append(
            metric_row(
                "NORMALISED_ERROR",
                "paired_ph_comparison",
                "126_matched_sites",
                "LSTM_minus_HGB",
                f"mean_{metric}_difference",
                point,
                ci_lower=ci_low,
                ci_upper=ci_high,
                sites=len(pair),
                rows=len(pH),
                notes=f"paired complete-site bootstrap, {BOOTSTRAPS} replicates, seed={SEED}; negative favours LSTM for MAE/nMAE, positive favours LSTM for NSE",
            )
        )
    if not math.isclose(model_results[lstm_model]["macro_site_mae"], 0.180789, abs_tol=1e-6):
        raise RuntimeError("LSTM own-offset macro-site MAE identity mismatch")
    if not math.isclose(model_results[hgb_model]["macro_site_mae"], 0.193328, abs_tol=1e-6):
        raise RuntimeError("HGB own-offset macro-site MAE identity mismatch")
    do_model = "DO_CHEAP3_RAIN_3_month_film_affine_H01"
    if not math.isclose(model_results[do_model]["macro_site_mae"], 0.6879834019420907, abs_tol=1e-12):
        raise RuntimeError("DO selected adapter macro-site MAE identity mismatch")
    summary_rows.append(
        metric_row(
            "NORMALISED_ERROR",
            "provenance_limit",
            "DO_39_site_query",
            do_model,
            "row_level_adapter_predictions_persisted",
            False,
            sites=39,
            rows=1_071_897,
            notes="All requested DO metrics are recovered exactly from canonical per-site sufficient statistics; no model prediction was regenerated.",
        )
    )
    summary = pd.DataFrame(summary_rows)

    evidence_rows = [
        {
            "source/network": "NEON water-quality product DP1.20288.001",
            "target": "pH and DO",
            "instrument/model if known": "YSI EXO2; pH probe 599706; DO probe 599100",
            "stated resolution": "pH 0.01; DO 0.01 mg/L",
            "stated accuracy": "pH ±0.1 within ±10 °C of calibration temperature, ±0.2 over full temperature range; DO 0–20 mg/L: ±0.1 mg/L or 1% of reading (whichever greater), 20–50 mg/L: ±5% of reading",
            "operating conditions": "Manufacturer specification; pH condition is temperature relative to calibration; DO accuracy is concentration-range dependent",
            "primary source": "https://data.neonscience.org/data-products/DP1.20288.001/RELEASE-2026 ; https://www.ysi.com/File%20Library/Documents/Specification%20Sheets/YSI-EXO-Sonde-Platform-Specification-Sheet.pdf",
            "applicability limitation": "Instrument identity is documented for the cited NEON product only; it is not a specification for USGS, EA or all 910 sites, and field uncertainty also includes calibration drift, fouling, siting and processing.",
        },
        {
            "source/network": "Representative field sonde (manufacturer documentation)",
            "target": "pH and DO",
            "instrument/model if known": "In-Situ Aqua TROLL 600",
            "stated resolution": "pH 0.01; RDO 0.01 mg/L",
            "stated accuracy": "pH ±0.1 pH unit or better; RDO 0–20 mg/L: ±0.1 mg/L, 20–50 mg/L: ±2% of reading",
            "operating conditions": "pH accuracy at calibration points; manufacturer notes site conditions/interferents and calibration conditions; RDO is range-specific",
            "primary source": "https://in-situ.com/us/pub/media/support/documents/AquaTROLL600_Spec-Sheet.pdf",
            "applicability limitation": "Representative field-sonde scale only. The frozen station metadata do not establish this model across the study networks; do not treat as network-wide uncertainty.",
        },
        {
            "source/network": "USGS continuous water-quality monitoring",
            "target": "pH and DO",
            "instrument/model if known": "Site/instrument dependent; exact models not carried in the frozen prediction artifacts",
            "stated resolution": "Not assigned network-wide",
            "stated accuracy": "Not assigned network-wide; USGS guidance requires instrument acceptance, field checks, cleaning, calibration, drift/fouling correction and record review",
            "operating conditions": "Continuous in-situ monitoring; performance depends on sensor, calibration, fouling and environmental conditions",
            "primary source": "https://pubs.usgs.gov/tm/2006/tm1D3/ ; https://www.usgs.gov/publications/guidelines-and-standard-procedures-continuous-water-quality-monitors-site-selection",
            "applicability limitation": "Operational guidance is not a single sensor accuracy specification. No one tolerance is assigned to all USGS sites in this audit.",
        },
        {
            "source/network": "Environment Agency / GB stations in the assembled corpus",
            "target": "pH and DO",
            "instrument/model if known": "Not verified in the frozen model/prediction metadata",
            "stated resolution": "Not verified",
            "stated accuracy": "Not verified",
            "operating conditions": "Unknown at per-site instrument level in the frozen analytical artifacts",
            "primary source": "No per-site primary instrument record was present in the frozen analytical inputs used here",
            "applicability limitation": "No manufacturer or NEON specification is generalised to EA/GB sites; a network-representative physical comparison is therefore unavailable.",
        },
    ]
    evidence = pd.DataFrame(evidence_rows)
    result = {
        "models": model_results,
        "paired_ph": paired_results,
        "do_row_level_predictions_persisted": False,
        "do_exact_sufficient_statistics_available": True,
        "denominator_tolerance": "max(1e-12, 16*float64_epsilon*max(abs(sum(y^2)),1))",
    }
    log("Task 3 complete: pH row-level and DO lossless sufficient-statistic audit")
    return summary, site_metrics, evidence, result


def format_float(value: object, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def write_reports(t1: dict[str, object], t2: dict[str, object], t3: dict[str, object], b0: dict[str, object]) -> None:
    p1 = t1["matched_lstm"]
    p0 = t1["matched_persistence"]
    c = t2["comparisons"]
    n = t3["models"]
    lstm = n["pH_LSTM_own_offset_H00"]
    hgb = n["pH_HGB_SUMMARY48_own_offset_H00"]
    do = n["DO_CHEAP3_RAIN_3_month_film_affine_H01"]
    p_pair = t3["paired_ph"]
    report = f"""# Supervisor final three frozen-output checks

## Executive summary

All three requested checks were completed without training, fine-tuning, hyperparameter search, threshold selection on test data, or regeneration of trained-model predictions. The historical-target H01 LSTM was compared with exact causal persistence only on keys having a QC-accepted native pH observation at precisely T−1 h. The coupling null used the canonical 30-day support/query boundary and a separately validation-frozen B0 lower-tail cutoff. Within-site pH metrics were computed from matched row-level predictions; selected-DO metrics were recovered exactly from frozen per-site sufficient statistics because adapter row-level predictions were not persisted.

- H01 persistence result: **{t1['label']}**. On {p1['rows']:,} matched rows from {p1['sites']} sites, macro-site MAE was {p1['macro_site_mae']:.6f} pH for LSTM and {p0['macro_site_mae']:.6f} pH for persistence. The paired site-MAE difference was {t1['paired_difference']:.6f} pH (95% site-bootstrap CI {t1['paired_ci'][0]:.6f} to {t1['paired_ci'][1]:.6f}).
- Coupling reference result: **{t2['label']}**. Same-event-site enrichment was {c['H00_same_event_site_background']['enrichment']:.6f} for H00 and {c['B0_same_event_site_background']['enrichment']:.6f} for B0; their enrichment ratio was {t2['point_enrichment_ratio']:.6f}, and paired log-enrichment difference was {t2['point_log_difference']:.6f} (95% CI {t2['paired_log_difference_ci'][0]:.6f} to {t2['paired_log_difference_ci'][1]:.6f}). This is an empirical coupling diagnostic, not independent pollution validation.
- Within-site scale: pH LSTM/HGB median site nMAE was {lstm['median_site_nmae']:.3f}/{hgb['median_site_nmae']:.3f}, with median site NSE {lstm['median_site_nse']:.3f}/{hgb['median_site_nse']:.3f}. Selected three-month DO adapter median site nMAE was {do['median_site_nmae']:.3f}, with median site NSE {do['median_site_nse']:.3f}.

## Input identities and cohort flow

| Asset | Identity | Cohort / units | Key and time semantics |
|---|---|---|---|
| Historical pH H01 | `ph_LSTM_RAIN_FINAL_H01`, horizon 1 h, 48 h sequence | 135 sites; 1,820,275 full test rows; native pH | unique `(site_id, target_timestamp)`; timestamps are canonical UTC-normalised naive datetimes; permitted input ends exactly at T−1 h |
| Matched pH H00 | retained own-offset LSTM and own-offset HGB SUMMARY48 | 126 sites; 1,493,309 test-query rows; native pH | unique `(site_id, target_timestamp)`; support/query label frozen |
| Directional H00 residual audit | `ph_LSTM_RAIN_H0_H00`; observed−predicted signed score | 126 evaluation sites; 1,493,309 test-query rows; native pH | unique `(split, site_id, target_timestamp)`; 30-day support, ≥504 valid hours |
| DO reduced-sensor H01 | `CHEAP3_RAIN`, three-month 130-parameter affine adapter | 39 test sites; 1,071,897 fixed query rows; native mg L⁻¹ | per-site fixed-query key hashes stored in canonical shards; no ×20 conversion |

Every local input is listed with SHA-256 in `HASH_MANIFEST.csv`. The H01 native-value join was exact at T−1 h; nearest-time matching, interpolation, forward fill and duplicate collapsing were not used. Rows excluded from persistence are recorded by reason in `h01_persistence_summary.csv`. Detector-specific missing common keys are recorded in `hot_moment_coupling_null_summary.csv`.

## Task 1 — exact persistence baseline

The full frozen LSTM cohort contained {t1['full']['rows']:,} rows at {t1['full']['sites']} sites. Exact persistence matching retained {p1['rows']:,} rows at {p1['sites']} sites. On identical keys, pooled MAE/RMSE were {p1['pooled_mae']:.6f}/{p1['pooled_rmse']:.6f} pH for LSTM and {p0['pooled_mae']:.6f}/{p0['pooled_rmse']:.6f} pH for persistence; macro-site MAE/RMSE were {p1['macro_site_mae']:.6f}/{p1['macro_site_rmse']:.6f} and {p0['macro_site_mae']:.6f}/{p0['macro_site_rmse']:.6f}, respectively. Bias is defined as observed minus predicted. Site R² was eligible only when SST exceeded the predeclared floating-point tolerance. The interpretation label follows the frozen analysis rule exactly; it is a matched diagnostic and does not promote or modify a model.

## Task 2 — coupling-only reference

The canonical H00 signed lower-tail cutoff was {format_float(-4.477322101593018, 12)}. B0 was computed as each site's frozen 30-day support mean, standardised with the canonical site support-residual median and raw MAD implementation. Its validation-only q05 cutoff was {b0['signed_score_q05']:.12f}, fitted on {b0['fit_rows']:,} validation-evaluation rows at {b0['fit_sites']} sites without reading event labels. On the exact common test keys, all-site enrichment was {c['H00_all_site_background']['enrichment']:.6f} for H00 and {c['B0_all_site_background']['enrichment']:.6f} for B0. Restricting background to the same 18 sites yielded {c['H00_same_event_site_background']['enrichment']:.6f} and {c['B0_same_event_site_background']['enrichment']:.6f}. The paired bootstrap retained all 18 sites, aggregated counts before calculating enrichment, used no continuity correction, and discarded only nonfinite zero-rate replicates ({t2['invalid_bootstrap_replicates']} of {BOOTSTRAPS}). All event/background counts and rates are machine-readable.

If the paired CI is wholly above zero, the label is `INCREMENT_BEYOND_COUPLING`; if it includes zero, `COUPLING_ONLY_COMPATIBLE`; if wholly below zero, `B0_STRONGER_THAN_H00`. None of these labels establishes causal pollution detection because observed pH contributes to both the episode definition and residual score.

## Task 3 — within-site and physical scale

The pH own-offset macro-site MAE reproduced at {lstm['macro_site_mae']:.9f} for LSTM and {hgb['macro_site_mae']:.9f} pH for HGB. Their mean paired site-MAE difference (LSTM−HGB) was {p_pair['site_mae']['mean_difference']:.9f} pH (95% CI {p_pair['site_mae']['ci'][0]:.9f} to {p_pair['site_mae']['ci'][1]:.9f}); negative favours LSTM. Pooled within-site NSE was {lstm['pooled_within_site_nse']:.3f}/{hgb['pooled_within_site_nse']:.3f}. The selected DO adapter reproduced macro-site MAE {do['macro_site_mae']:.9f} mg L⁻¹ and pooled within-site NSE {do['pooled_within_site_nse']:.3f}. Low-variance exclusions and bootstrap intervals are in the two normalised-error CSVs.

Manufacturer resolution is not field accuracy, and neither is total measurement uncertainty. NEON documents YSI EXO2 probes for its cited product, allowing an instrument-specific comparison there; USGS and EA/GB instruments are site-dependent or unavailable in these frozen model assets. Consequently, the approximately 0.0125-pH LSTM–HGB effect may be compared cautiously with representative ±0.1–0.2-pH field-sonde specifications, but it is not a network-wide uncertainty claim. Prediction MAE and sensor accuracy measure different things. The DO adapter error is much larger than a single YSI/In-Situ bench specification, but no one instrument specification is generalised across the 39-site cohort.

## Reproducibility and scope

Bootstrap resampling used complete sites, 10,000 replicates and established project seed {SEED}. nMAE used population site SD (`ddof=0`); within-site NSE pooled site-centred SST. DO per-site `n`, `SAE`, `SSE`, `sum(y)` and `sum(y²)` are canonical frozen machine-readable sufficient statistics and recover all requested metrics losslessly. The absence of persisted DO adapter prediction rows is explicitly recorded; no prediction was regenerated. This audit wrote only to this new directory and did not edit canonical outputs, the dissertation or poster.

H01_PERSISTENCE_COMPLETED
COUPLING_NULL_COMPLETED
NORMALISED_ERROR_COMPLETED
SUPERVISOR_FINAL_THREE_CHECKS_COMPLETE_NO_TRAINING
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    inserts = f"""# Thesis-ready inserts

## Results

Against an exact causal persistence reference, the historical-target pH H01 LSTM was evaluated on {p1['rows']:,} common test timestamps from {p1['sites']} sites. Macro-site MAE was {p1['macro_site_mae']:.3f} pH for the LSTM and {p0['macro_site_mae']:.3f} pH for persistence, while macro-site RMSE was {p1['macro_site_rmse']:.3f} and {p0['macro_site_rmse']:.3f} pH, respectively. The mean paired site-MAE difference (LSTM minus persistence) was {t1['paired_difference']:.6f} pH (95% complete-site bootstrap CI {t1['paired_ci'][0]:.6f} to {t1['paired_ci'][1]:.6f}), giving the pre-specified label `{t1['label']}`. Persistence used only QC-accepted native pH observed at exactly one hour before each target; no interpolation or nearest-time match was allowed.

The directional low-pH residual enrichment was compared with a coupling-only reference in which expected pH was each site's frozen 30-day support mean. The B0 threshold was fitted from validation evaluation scores only. With background restricted to the same 18 low-event sites, enrichment was {c['H00_same_event_site_background']['enrichment']:.3f} for H00 and {c['B0_same_event_site_background']['enrichment']:.3f} for B0. The H00-to-B0 enrichment ratio was {t2['point_enrichment_ratio']:.3f}; the paired log-enrichment difference was {t2['point_log_difference']:.6f} (95% CI {t2['paired_log_difference_ci'][0]:.6f} to {t2['paired_log_difference_ci'][1]:.6f}), corresponding to `{t2['label']}`. This analysis quantifies mathematical coupling because observed pH contributes to both the event label and residual; it is not independent pollution validation.

For contemporaneous pH H00, own-offset LSTM and HGB macro-site MAE were {lstm['macro_site_mae']:.3f} and {hgb['macro_site_mae']:.3f} pH. The unrounded paired difference was {p_pair['site_mae']['mean_difference']:.9f} pH (95% CI {p_pair['site_mae']['ci'][0]:.9f} to {p_pair['site_mae']['ci'][1]:.9f}). Median site nMAE was {lstm['median_site_nmae']:.3f} for LSTM and {hgb['median_site_nmae']:.3f} for HGB; median site NSE was {lstm['median_site_nse']:.3f} and {hgb['median_site_nse']:.3f}. For the validation-selected three-month reduced-sensor DO H01 adapter, macro-site MAE/RMSE were {do['macro_site_mae']:.3f}/{do['macro_site_rmse']:.3f} mg L⁻¹, median site nMAE was {do['median_site_nmae']:.3f}, and median site NSE was {do['median_site_nse']:.3f}. The canonical DO outputs were already in native mg L⁻¹; no ×20 conversion was applied.

## Discussion

The LSTM–HGB pH comparison separates statistical evidence from practical magnitude. Its paired site-bootstrap interval describes statistical uncertainty, whereas the effect itself is approximately {abs(p_pair['site_mae']['mean_difference']):.4f} pH before display rounding. Official YSI EXO documentation states pH accuracy of ±0.1 within 10 °C of calibration temperature and ±0.2 over the full temperature range, with 0.01-pH resolution; In-Situ lists ±0.1 pH and 0.01-pH resolution for the Aqua TROLL 600. These values provide a representative field-sonde scale, not a study-wide uncertainty estimate, because exact instruments were not verified across all sites. Prediction MAE also includes environmental predictability, temporal alignment, preprocessing and model error and therefore is not interchangeable with sensor accuracy.

For dissolved oxygen, official YSI EXO specifications state 0.01 mg L⁻¹ resolution and accuracy of ±0.1 mg L⁻¹ or 1% of reading (whichever is greater) from 0–20 mg L⁻¹, while the Aqua TROLL 600 lists ±0.1 mg L⁻¹ over that range. These specifications apply only where the corresponding instruments are documented. They are not generalised to the 39-site DO cohort; the selected adapter's {do['macro_site_mae']:.3f} mg L⁻¹ macro-site MAE should therefore be interpreted primarily through its within-site nMAE/NSE and not as a direct multiple of a universal sensor uncertainty.
"""
    (OUT / "THESIS_INSERTS.md").write_text(inserts, encoding="utf-8")


def write_config() -> None:
    config = {
        "analysis": "supervisor_final_three_checks_v1",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "deterministic_only": True,
        "training_or_tuning": False,
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "site_r2_sst_tolerance": "max(1e-12, 16*float64_epsilon*max(abs(sum(y^2)),1))",
        "persistence": "observed QC-accepted native pH at exact target_timestamp minus one hour",
        "b0": {
            "prediction": "mean observed pH in frozen site/split 30-day support",
            "mad_constant": MAD_CONSTANT,
            "minimum_mad": MIN_MAD,
            "minimum_calibration_hours": MIN_CALIBRATION_HOURS,
            "threshold": "validation evaluation score q05; event labels unread",
        },
        "normalised_error": {"site_sd_ddof": 0, "nmae": "site MAE / site population SD", "nse": "1 - site SSE/site SST"},
    }
    (OUT / "analysis_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def hash_manifest() -> None:
    log(f"Hashing {len(INPUT_FILES)} canonical input files and final outputs")
    rows: list[dict[str, object]] = []
    for path, role in sorted(INPUT_FILES.items(), key=lambda item: str(item[0]).lower()):
        rows.append(
            {
                "category": "input",
                "role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    output_names = [
        "FINAL_REPORT.md",
        "h01_persistence_summary.csv",
        "h01_persistence_site_metrics.csv",
        "hot_moment_coupling_null_summary.csv",
        "hot_moment_coupling_null_site_metrics.csv",
        "normalised_error_summary.csv",
        "normalised_error_site_metrics.csv",
        "instrument_accuracy_evidence.csv",
        "THESIS_INSERTS.md",
        "run_supervisor_final_three_checks.py",
        "analysis_config.json",
        "b0_validation_threshold.json",
        "RUN_LOG.txt",
    ]
    for name in output_names:
        path = OUT / name
        if not path.exists():
            raise FileNotFoundError(f"Required output missing before manifest: {path}")
        rows.append(
            {
                "category": "output",
                "role": "deliverable",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    rows.append(
        {
            "category": "output",
            "role": "self_excluded_to_avoid_recursive_hash",
            "path": str(OUT / "HASH_MANIFEST.csv"),
            "bytes": np.nan,
            "sha256": "SELF_EXCLUDED",
        }
    )
    pd.DataFrame(rows).to_csv(OUT / "HASH_MANIFEST.csv", index=False)


def main() -> None:
    started = time.time()
    if not OUT.exists():
        raise FileNotFoundError(OUT)
    log(f"Starting deterministic audit with Python {platform.python_version()}, pandas {pd.__version__}, numpy {np.__version__}")
    log("No training, tuning, model selection, or trained-model prediction regeneration is permitted or performed")
    t1_summary, t1_sites, t1_result = task1_persistence()
    t1_summary.to_csv(OUT / "h01_persistence_summary.csv", index=False)
    t1_sites.to_csv(OUT / "h01_persistence_site_metrics.csv", index=False)

    t2_summary, t2_sites, t2_result, b0_payload = task2_coupling()
    t2_summary.to_csv(OUT / "hot_moment_coupling_null_summary.csv", index=False)
    t2_sites.to_csv(OUT / "hot_moment_coupling_null_site_metrics.csv", index=False)

    t3_summary, t3_sites, evidence, t3_result = task3_normalised()
    t3_summary.to_csv(OUT / "normalised_error_summary.csv", index=False)
    t3_sites.to_csv(OUT / "normalised_error_site_metrics.csv", index=False)
    evidence.to_csv(OUT / "instrument_accuracy_evidence.csv", index=False)

    write_config()
    write_reports(t1_result, t2_result, t3_result, b0_payload)
    log(f"Core calculations and reports complete in {time.time() - started:.2f} seconds")
    (OUT / "RUN_LOG.txt").write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    hash_manifest()
    log(f"Audit complete in {time.time() - started:.2f} seconds")
    (OUT / "RUN_LOG.txt").write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    # Refresh only the RUN_LOG hash after the final log append; other rows are unchanged.
    manifest = pd.read_csv(OUT / "HASH_MANIFEST.csv")
    mask = manifest["path"].eq(str(OUT / "RUN_LOG.txt"))
    manifest.loc[mask, "bytes"] = (OUT / "RUN_LOG.txt").stat().st_size
    manifest.loc[mask, "sha256"] = sha256_file(OUT / "RUN_LOG.txt")
    manifest.to_csv(OUT / "HASH_MANIFEST.csv", index=False)
    print(json.dumps({"task1": t1_result["label"], "task2": t2_result["label"], "status": "COMPLETE_NO_TRAINING"}, indent=2))


if __name__ == "__main__":
    main()
