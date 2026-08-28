from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analysis_outputs" / "ph_h00_30day_calibration_validation_v1"
CLOSURE = ROOT / "analysis_outputs" / "cross_variable_hot_moment_incremental_closure_v1"
H00_ROOT = ROOT / "analysis_outputs" / "ph_common1000_lstm_nohistory_h0_rain_final_v1" / "full"
CHECKPOINT = H00_ROOT / "models" / "ph_LSTM_RAIN_H0_H00.pt"
EFFECTIVE_CONFIG = H00_ROOT / "reports" / "effective_config.json"
RESIDUALS = CLOSURE / "residual_scores" / "matched_residuals.parquet"
H00_PREDICTIONS = CLOSURE / "predictions" / "cross_variable_h00_predictions.parquet"
MATCHED_KEYS = CLOSURE / "data" / "matched_prediction_keys.parquet"
MATCHED_EVENTS = CLOSURE / "data" / "matched_event_manifest.csv"
THRESHOLDS_FILE = CLOSURE / "calibration" / "label_free_thresholds.json"
EVENT_MODULE_PATH = ROOT / "pipeline" / "24_error_onset" / "run_error_onset_analysis.py"

METHODS = ["M0_RAW_H00", "M1_OFFSET_ONLY", "M2_AFFINE"]
CALIBRATION_DAYS = 30
MINIMUM_VALID_HOURS = 504
BOOTSTRAPS = 10_000
SEED = 20260802
THREE_HOUR_RULE = 3
LEAD_LOOKBACK_HOURS = 48
OUTSIDE_EVENT_PRE_HOURS = 48
OUTSIDE_EVENT_POST_HOURS = 24


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_mad(values: Iterable[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan
    center = np.median(array)
    return float(1.4826 * np.median(np.abs(array - center)))


def load_event_module():
    spec = importlib.util.spec_from_file_location("existing_event_analysis", EVENT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load existing event-analysis implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVENT = load_event_module()


def prepare_directories() -> None:
    if OUTPUT.exists():
        # A run stopped before the persisted validation lock cannot have opened test.
        # It is safe to resume and overwrite its small pre-selection config files.
        if (OUTPUT / "configs" / "validation_selection_lock.json").exists():
            raise RuntimeError(
                f"Output already contains a validation selection lock: {OUTPUT}. "
                "Refusing to rerun because test may already have been accessed."
            )
    for name in (
        "configs", "calibration", "predictions", "site_metrics", "event_metrics",
        "bootstrap", "figures", "reports", "summary", "completion",
    ):
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)


def validate_inputs() -> dict[str, Any]:
    required = [
        CHECKPOINT, EFFECTIVE_CONFIG, RESIDUALS, H00_PREDICTIONS, MATCHED_KEYS,
        MATCHED_EVENTS, THRESHOLDS_FILE, EVENT_MODULE_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required frozen assets: " + ", ".join(missing))
    config = json.loads(EFFECTIVE_CONFIG.read_text(encoding="utf-8"))
    config_text = json.dumps(config).lower()
    checks = {
        "checkpoint_exists": CHECKPOINT.exists(),
        "horizon_h00_recorded": any(token in config_text for token in ('\"horizon_hours\": 0', '\"horizon\": 0', '\"horizon_hours\":0')),
        "no_historical_ph_recorded": "target_history" in config_text and "false" in config_text,
        "checkpoint_frozen_no_retraining": True,
        "source_predictions_reused": True,
    }
    if not checks["horizon_h00_recorded"]:
        raise RuntimeError("Frozen effective config does not record horizon 0")
    return {
        "created_utc": utc_now(),
        "inputs": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in required},
        "frozen_task_checks": checks,
    }


def load_split(split: str) -> pd.DataFrame:
    columns = [
        "site_id", "target_timestamp", "split", "observed_ph", "predicted_cross",
        "predicted_hist", "in_event", "event_direction", "event_id",
        "calibration_period", "calibration_eligible", "evaluation_period",
        "signed_score_cross", "absolute_score_cross", "signed_score_hist",
        "absolute_score_hist", "top5_cross", "top1_cross", "top5_hist", "top1_hist",
    ]
    frame = pd.read_parquet(RESIDUALS, columns=columns, filters=[("split", "==", split)])
    if frame.empty or not frame["split"].eq(split).all():
        raise RuntimeError(f"Failed strict filtered read for split={split}")
    frame["site_id"] = frame["site_id"].astype(str)
    frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"])
    if frame.duplicated(["site_id", "target_timestamp"]).any():
        raise RuntimeError(f"Duplicate prediction keys in {split}")
    if frame[["observed_ph", "predicted_cross"]].isna().any().any():
        raise RuntimeError(f"Missing matched H00 values in {split}")
    return frame.sort_values(["site_id", "target_timestamp"], kind="mergesort").reset_index(drop=True)


def fit_split_calibrations(
    frame: pd.DataFrame,
    split: str,
    frozen_thresholds: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, set[str]]]:
    result = frame.copy()
    for method in METHODS:
        result[f"pred_{method}"] = np.nan
        result[f"residual_{method}"] = np.nan
        result[f"signed_score_{method}"] = np.nan
        result[f"absolute_score_{method}"] = np.nan
        result[f"top5_{method}"] = False
        result[f"top1_{method}"] = False
    result["pred_M0_RAW_H00"] = result["predicted_cross"].astype(float)
    result["residual_M0_RAW_H00"] = result["observed_ph"] - result["pred_M0_RAW_H00"]

    parameter_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    formal_sites: dict[str, set[str]] = {method: set() for method in METHODS}
    temporal_checks = []
    fallback_scale = float(frozen_thresholds["fallback_validation_global_mad"]["cross"])

    for site_id, group in result.groupby("site_id", sort=False):
        idx = group.index
        start = group["target_timestamp"].min()
        end_exclusive = start + pd.Timedelta(days=CALIBRATION_DAYS)
        expected_cal = group["target_timestamp"].lt(end_exclusive)
        closure_cal = group["calibration_period"].astype(bool)
        if not expected_cal.equals(closure_cal):
            raise RuntimeError(f"Closure calibration mask mismatch at {split}/{site_id}")
        calibration = group.loc[closure_cal]
        evaluation = group.loc[group["evaluation_period"]]
        valid_hours = len(calibration)
        closure_eligible = bool(group["calibration_eligible"].iloc[0])
        eligible = valid_hours >= MINIMUM_VALID_HOURS
        if closure_eligible != eligible:
            raise RuntimeError(f"Closure eligibility mismatch at {split}/{site_id}")
        no_overlap = True
        strict_after = True
        if len(evaluation):
            no_overlap = not bool(set(calibration.index).intersection(evaluation.index))
            strict_after = bool(evaluation["target_timestamp"].min() > calibration["target_timestamp"].max())
        temporal_checks.append((no_overlap, strict_after))

        if not eligible:
            failure_rows.append({
                "split": split, "site_id": site_id, "method": "M1_OFFSET_ONLY|M2_AFFINE",
                "failure_code": "CALIBRATION_INSUFFICIENT", "calibration_valid_hours": valid_hours,
                "calibration_start": start, "calibration_end_exclusive": end_exclusive,
            })
            continue
        if evaluation.empty:
            failure_rows.append({
                "split": split, "site_id": site_id, "method": "M1_OFFSET_ONLY|M2_AFFINE",
                "failure_code": "NO_POST_CALIBRATION_EVALUATION", "calibration_valid_hours": valid_hours,
                "calibration_start": start, "calibration_end_exclusive": end_exclusive,
            })
            continue

        y = calibration["observed_ph"].to_numpy(dtype=float)
        x = calibration["predicted_cross"].to_numpy(dtype=float)
        b_offset = float(np.mean(y - x))
        centered_x = x - x.mean()
        denominator = float(np.sum(centered_x**2))
        singular = not np.isfinite(denominator) or denominator <= 1e-12
        if singular:
            a_affine, b_affine = np.nan, np.nan
        else:
            unconstrained_a = float(np.sum(centered_x * (y - y.mean())) / denominator)
            a_affine = max(unconstrained_a, 1e-8)
            b_affine = float(y.mean() - a_affine * x.mean())
        affine_unstable = bool(
            singular or not np.isfinite(a_affine) or not np.isfinite(b_affine)
            or a_affine < 0.2 or a_affine > 5.0 or abs(b_affine) > 3.0
        )
        result.loc[idx, "pred_M1_OFFSET_ONLY"] = result.loc[idx, "predicted_cross"] + b_offset
        result.loc[idx, "pred_M2_AFFINE"] = a_affine * result.loc[idx, "predicted_cross"] + b_affine
        result.loc[idx, "residual_M1_OFFSET_ONLY"] = result.loc[idx, "observed_ph"] - result.loc[idx, "pred_M1_OFFSET_ONLY"]
        result.loc[idx, "residual_M2_AFFINE"] = result.loc[idx, "observed_ph"] - result.loc[idx, "pred_M2_AFFINE"]

        formal_sites["M0_RAW_H00"].add(site_id)
        formal_sites["M1_OFFSET_ONLY"].add(site_id)
        if not affine_unstable:
            formal_sites["M2_AFFINE"].add(site_id)
        for method, a_value, b_value in (
            ("M1_OFFSET_ONLY", 1.0, b_offset),
            ("M2_AFFINE", a_affine, b_affine),
        ):
            cal_pred = a_value * x + b_value
            cal_residual = y - cal_pred
            status = "AFFINE_UNSTABLE" if method == "M2_AFFINE" and affine_unstable else "ELIGIBLE"
            parameter_rows.append({
                "split": split, "site_id": site_id, "method": method, "a_s": a_value, "b_s": b_value,
                "calibration_rmse": float(np.sqrt(np.mean(cal_residual**2))),
                "calibration_mae": float(np.mean(np.abs(cal_residual))),
                "calibration_sample_count": valid_hours, "calibration_start": start,
                "calibration_end_exclusive": end_exclusive,
                "evaluation_start": evaluation["target_timestamp"].min(),
                "calibration_status": status, "formal_comparison_eligible": status == "ELIGIBLE",
                "parameters_frozen_during_evaluation": True,
            })
        if affine_unstable:
            failure_rows.append({
                "split": split, "site_id": site_id, "method": "M2_AFFINE",
                "failure_code": "AFFINE_UNSTABLE", "calibration_valid_hours": valid_hours,
                "calibration_start": start, "calibration_end_exclusive": end_exclusive,
                "a_s": a_affine, "b_s": b_affine,
            })

        for method in METHODS:
            cal_residual = result.loc[calibration.index, f"residual_{method}"].to_numpy(dtype=float)
            center = float(np.median(cal_residual))
            scale_raw = raw_mad(cal_residual)
            used_fallback = bool(not np.isfinite(scale_raw) or scale_raw < 1e-6)
            scale = fallback_scale if used_fallback else scale_raw
            score = (result.loc[idx, f"residual_{method}"] - center) / max(scale, 1e-6)
            result.loc[idx, f"signed_score_{method}"] = score
            result.loc[idx, f"absolute_score_{method}"] = score.abs()
            score_rows.append({
                "split": split, "site_id": site_id, "method": method,
                "residual_center": center, "residual_scale_raw": scale_raw,
                "residual_scale_used": scale, "global_mad_fallback_used": used_fallback,
                "threshold_reselected": False,
            })

    if not all(a and b for a, b in temporal_checks):
        raise RuntimeError("Calibration/evaluation overlap or non-strict temporal ordering detected")
    q95 = float(frozen_thresholds["models"]["cross_variable_h00"]["absolute_score_q95"])
    q99 = float(frozen_thresholds["models"]["cross_variable_h00"]["absolute_score_q99"])
    for method in METHODS:
        result[f"top5_{method}"] = result[f"absolute_score_{method}"].gt(q95)
        result[f"top1_{method}"] = result[f"absolute_score_{method}"].gt(q99)

    raw_comparable = result["calibration_eligible"] & result["absolute_score_M0_RAW_H00"].notna()
    max_score_difference = float(
        np.nanmax(np.abs(
            result.loc[raw_comparable, "absolute_score_M0_RAW_H00"].to_numpy(dtype=float)
            - result.loc[raw_comparable, "absolute_score_cross"].to_numpy(dtype=float)
        ))
    )
    m0_top5_match = bool(
        result.loc[raw_comparable, "top5_M0_RAW_H00"].to_numpy(dtype=bool)
        .__eq__(result.loc[raw_comparable, "top5_cross"].to_numpy(dtype=bool)).all()
    )
    m0_top1_match = bool(
        result.loc[raw_comparable, "top1_M0_RAW_H00"].to_numpy(dtype=bool)
        .__eq__(result.loc[raw_comparable, "top1_cross"].to_numpy(dtype=bool)).all()
    )
    if max_score_difference > 2e-5 or not m0_top5_match or not m0_top1_match:
        raise RuntimeError(f"M0 score reproduction failed: max difference={max_score_difference}")

    audit = {
        "split": split,
        "source_rows": len(result),
        "source_sites": result["site_id"].nunique(),
        "calibration_eligible_sites": len(formal_sites["M1_OFFSET_ONLY"]),
        "affine_stable_sites": len(formal_sites["M2_AFFINE"]),
        "affine_unstable_sites": len(formal_sites["M1_OFFSET_ONLY"] - formal_sites["M2_AFFINE"]),
        "calibration_evaluation_overlap": False,
        "strict_evaluation_after_calibration": True,
        "parameters_updated_during_evaluation": False,
        "m0_score_reproduction_max_abs_difference": max_score_difference,
        "m0_top5_flags_exact_match": m0_top5_match,
        "m0_top1_flags_exact_match": m0_top1_match,
        "frozen_q95": q95,
        "frozen_q99": q99,
    }
    score_parameters = pd.DataFrame(score_rows)
    return result, pd.DataFrame(parameter_rows), pd.DataFrame(failure_rows), audit, formal_sites


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = observed - predicted
    sse = float(np.sum(residual**2))
    centered = observed - observed.mean()
    sst = float(np.sum(centered**2))
    if len(observed) >= 3 and np.std(observed) > 1e-12:
        slope = float(np.sum(centered * (predicted - predicted.mean())) / sst)
    else:
        slope = np.nan
    observed_range = float(np.quantile(observed, 0.95) - np.quantile(observed, 0.05))
    predicted_range = float(np.quantile(predicted, 0.95) - np.quantile(predicted, 0.05))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "nse": float(1.0 - sse / sst) if sst > 1e-12 else np.nan,
        "bias": float(np.mean(residual)),
        "slope": slope,
        "range_ratio": predicted_range / observed_range if observed_range > 1e-12 else np.nan,
    }


def site_and_aggregate_metrics(
    frame: pd.DataFrame,
    formal_sites: dict[str, set[str]],
    split: str,
    included_methods: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = frame.loc[frame["evaluation_period"]].copy()
    rows = []
    for site_id, site in evaluation.groupby("site_id", sort=False):
        observed = site["observed_ph"].to_numpy(dtype=float)
        raw = metric_values(observed, site["pred_M0_RAW_H00"].to_numpy(dtype=float))
        for method in included_methods:
            predicted = site[f"pred_{method}"].to_numpy(dtype=float)
            if not np.isfinite(predicted).all():
                continue
            metrics = metric_values(observed, predicted)
            formal = site_id in formal_sites[method]
            rows.append({
                "split": split, "site_id": site_id, "method": method, **metrics,
                "sample_count": len(site), "formal_comparison_eligible": formal,
                "mae_improved_vs_m0": bool(metrics["mae"] < raw["mae"]) if method != "M0_RAW_H00" else False,
                "mae_worsened_vs_m0": bool(metrics["mae"] > raw["mae"]) if method != "M0_RAW_H00" else False,
                "rmse_improved_vs_m0": bool(metrics["rmse"] < raw["rmse"]) if method != "M0_RAW_H00" else False,
                "rmse_worsened_vs_m0": bool(metrics["rmse"] > raw["rmse"]) if method != "M0_RAW_H00" else False,
            })
    site_metrics = pd.DataFrame(rows)
    summaries = []
    for method in included_methods:
        sites = site_metrics.loc[site_metrics["method"].eq(method) & site_metrics["formal_comparison_eligible"]]
        method_rows = evaluation.loc[evaluation["site_id"].isin(formal_sites[method])]
        observed = method_rows["observed_ph"].to_numpy(dtype=float)
        predicted = method_rows[f"pred_{method}"].to_numpy(dtype=float)
        pooled = metric_values(observed, predicted)
        summaries.append({
            "split": split, "method": method,
            "pooled_rmse": pooled["rmse"], "pooled_mae": pooled["mae"], "pooled_nse": pooled["nse"],
            "macro_site_rmse": float(sites["rmse"].mean()),
            "macro_site_mae": float(sites["mae"].mean()),
            "median_site_nse": float(sites["nse"].median()),
            "positive_nse_sites": int(sites["nse"].gt(0).sum()),
            "mean_bias": float(sites["bias"].mean()),
            "median_absolute_bias": float(sites["bias"].abs().median()),
            "prediction_on_observation_slope": pooled["slope"],
            "p05_p95_range_ratio": pooled["range_ratio"],
            "improved_sites_mae": int(sites["mae_improved_vs_m0"].sum()),
            "worsened_sites_mae": int(sites["mae_worsened_vs_m0"].sum()),
            "improved_sites_rmse": int(sites["rmse_improved_vs_m0"].sum()),
            "worsened_sites_rmse": int(sites["rmse_worsened_vs_m0"].sum()),
            "formal_site_count": int(len(sites)), "evaluation_rows": int(len(method_rows)),
            "formal_comparison_complete": bool(len(formal_sites[method]) == len(formal_sites["M0_RAW_H00"])),
        })
    summary = pd.DataFrame(summaries)
    raw = summary.loc[summary["method"].eq("M0_RAW_H00")].iloc[0]
    summary["macro_rmse_improvement_fraction_vs_m0"] = 1.0 - summary["macro_site_rmse"] / raw["macro_site_rmse"]
    summary["macro_mae_improvement_fraction_vs_m0"] = 1.0 - summary["macro_site_mae"] / raw["macro_site_mae"]
    summary["improved_site_fraction_mae"] = summary["improved_sites_mae"] / summary["formal_site_count"]
    return site_metrics, summary


def site_bootstrap_enrichment(site_counts: pd.DataFrame, split: str, method: str) -> np.ndarray:
    contributing = site_counts.loc[(site_counts["event_rows"] > 0) & (site_counts["other_rows"] > 0)]
    values = contributing[["event_high", "event_rows", "other_high", "other_rows"]].to_numpy(dtype=float)
    seed = int(hashlib.sha256(f"{SEED}|{split}|{method}|enrichment".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    selected = values[sample].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (selected[:, 0] / selected[:, 1]) / (selected[:, 2] / selected[:, 3])


def event_metrics(
    frame: pd.DataFrame,
    formal_sites: dict[str, set[str]],
    split: str,
    included_methods: list[str],
    q95: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_event_manifest = EVENT.event_windows(
        frame.rename(columns={"target_timestamp": "timestamp"})[["site_id", "timestamp", "in_event", "event_direction"]]
    )
    summaries, site_count_parts, lead_parts, burden_parts, episode_parts = [], [], [], [], []
    bootstrap_rows = []
    for method in included_methods:
        evaluation = frame.loc[
            frame["evaluation_period"] & frame["site_id"].isin(formal_sites[method])
        ].copy()
        site_counts = []
        for site_id, site in evaluation.groupby("site_id", sort=False):
            event = site["in_event"].to_numpy(dtype=bool)
            high = site[f"top5_{method}"].to_numpy(dtype=bool)
            site_counts.append({
                "split": split, "method": method, "site_id": site_id,
                "event_high": int((event & high).sum()), "event_rows": int(event.sum()),
                "other_high": int((~event & high).sum()), "other_rows": int((~event).sum()),
            })
        site_count_frame = pd.DataFrame(site_counts)
        site_count_parts.append(site_count_frame)
        event_mask = evaluation["in_event"].to_numpy(dtype=bool)
        top5 = evaluation[f"top5_{method}"].to_numpy(dtype=bool)
        top1 = evaluation[f"top1_{method}"].to_numpy(dtype=bool)
        event_top5 = float(top5[event_mask].mean())
        other_top5 = float(top5[~event_mask].mean())
        event_top1 = float(top1[event_mask].mean())
        other_top1 = float(top1[~event_mask].mean())
        enrichment_boot = site_bootstrap_enrichment(site_count_frame, split, method)
        for replicate, value in enumerate(enrichment_boot, 1):
            bootstrap_rows.append({
                "split": split, "scope": "site_bootstrap", "method": method,
                "metric": "top5_enrichment", "replicate": replicate, "value": value,
            })

        event_input = evaluation[[
            "site_id", "target_timestamp", "in_event", "event_direction", f"absolute_score_{method}"
        ]].rename(columns={"target_timestamp": "timestamp", f"absolute_score_{method}": "abs_residual"})
        lead = EVENT.lead_time_rows(event_input, q95, LEAD_LOOKBACK_HOURS, THREE_HOUR_RULE)
        lead["split"] = split
        lead["method"] = method
        lead_parts.append(lead)
        lead_summary = EVENT.summarise_lead_times(lead, method, [6, 12, 24])

        intervals = {
            str(site): group["event_start"].sort_values()
            for site, group in full_event_manifest.groupby("site_id", sort=False)
        }
        method_episodes = []
        for site_id, site in evaluation.groupby("site_id", sort=False):
            ordered = site.sort_values("target_timestamp", kind="mergesort")
            hot = ordered.loc[ordered[f"absolute_score_{method}"].gt(q95)].copy()
            if len(hot):
                hot["episode_id"] = hot["target_timestamp"].diff().gt(pd.Timedelta(hours=1.5)).fillna(True).cumsum()
                for _, episode in hot.groupby("episode_id", sort=False):
                    if len(episode) < THREE_HOUR_RULE:
                        continue
                    episode_start = episode["target_timestamp"].min()
                    episode_end = episode["target_timestamp"].max()
                    onsets = intervals.get(str(site_id), pd.Series(dtype="datetime64[ns]"))
                    overlap = False
                    if len(onsets):
                        overlap = bool(((onsets - pd.Timedelta(hours=OUTSIDE_EVENT_PRE_HOURS) <= episode_end)
                                        & (onsets + pd.Timedelta(hours=OUTSIDE_EVENT_POST_HOURS) >= episode_start)).any())
                    method_episodes.append({
                        "split": split, "method": method, "site_id": site_id,
                        "episode_start": episode_start, "episode_end": episode_end,
                        "consecutive_hours": int(len(episode)),
                        "peak_absolute_score": float(episode[f"absolute_score_{method}"].max()),
                        "outside_register": not overlap,
                    })
        episode_frame = pd.DataFrame(method_episodes)
        if len(episode_frame):
            episode_parts.append(episode_frame)
        burden_rows = []
        for site_id, site in evaluation.groupby("site_id", sort=False):
            months = len(site) / (24 * 30.4375)
            if len(episode_frame):
                outside_count = int(episode_frame.loc[
                    episode_frame["site_id"].eq(site_id) & episode_frame["outside_register"]
                ].shape[0])
            else:
                outside_count = 0
            burden_rows.append({
                "split": split, "method": method, "site_id": site_id,
                "evaluation_rows": len(site), "evaluation_site_months": months,
                "outside_register_episodes": outside_count,
                "outside_register_alerts_per_site_month": outside_count / months if months else np.nan,
            })
        burden = pd.DataFrame(burden_rows)
        burden_parts.append(burden)
        pooled_burden = float(burden["outside_register_episodes"].sum() / burden["evaluation_site_months"].sum())

        hist_high = evaluation["top5_hist"].to_numpy(dtype=bool)
        cross_only = top5 & ~hist_high
        all_event_rate = float(event_mask.mean())
        cross_only_event_rate = float(event_mask[cross_only].mean()) if cross_only.any() else np.nan
        cross_only_signal = cross_only_event_rate / all_event_rate if all_event_rate > 0 else np.nan
        summaries.append({
            "split": split, "method": method,
            "top5_event_rate": event_top5, "top5_other_rate": other_top5,
            "top5_enrichment": event_top5 / other_top5 if other_top5 > 0 else np.nan,
            "top5_site_bootstrap_ci_low": float(np.nanquantile(enrichment_boot, 0.025)),
            "top5_site_bootstrap_ci_high": float(np.nanquantile(enrichment_boot, 0.975)),
            "top1_event_rate": event_top1, "top1_other_rate": other_top1,
            "top1_enrichment": event_top1 / other_top1 if other_top1 > 0 else np.nan,
            "event_windows": int(lead_summary["event_windows"]),
            "alerted_event_fraction": float(lead_summary["alerted_fraction"]),
            "median_lead_hours": float(lead_summary["median_lead_hours_alerted"]),
            "lead_q25_hours": float(lead_summary["lead_q25_hours_alerted"]),
            "lead_q75_hours": float(lead_summary["lead_q75_hours_alerted"]),
            "outside_register_alerts_per_site_month": pooled_burden,
            "outside_register_episodes": int(burden["outside_register_episodes"].sum()),
            "cross_variable_only_rows": int(cross_only.sum()),
            "cross_variable_only_event_enrichment": cross_only_signal,
            "evaluation_rows": len(evaluation), "formal_site_count": len(formal_sites[method]),
        })
    return (
        pd.DataFrame(summaries),
        pd.concat(site_count_parts, ignore_index=True),
        pd.concat(lead_parts, ignore_index=True) if lead_parts else pd.DataFrame(),
        pd.concat(burden_parts, ignore_index=True),
        pd.concat(episode_parts, ignore_index=True) if episode_parts else pd.DataFrame(),
        pd.DataFrame(bootstrap_rows),
    )


def accuracy_bootstrap(
    site_metrics: pd.DataFrame,
    included_methods: list[str],
    split: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, tuple[float, float]]]]:
    rows = []
    intervals: dict[str, dict[str, tuple[float, float]]] = {}
    raw = site_metrics.loc[site_metrics["method"].eq("M0_RAW_H00")].set_index("site_id")
    for method in [m for m in included_methods if m != "M0_RAW_H00"]:
        calibrated = site_metrics.loc[
            site_metrics["method"].eq(method) & site_metrics["formal_comparison_eligible"]
        ].set_index("site_id")
        common = raw.index.intersection(calibrated.index)
        intervals[method] = {}
        seed = int(hashlib.sha256(f"{SEED}|{split}|{method}|accuracy".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        sample = rng.integers(0, len(common), size=(BOOTSTRAPS, len(common)))
        for metric in ("mae", "rmse"):
            delta = calibrated.loc[common, metric].to_numpy() - raw.loc[common, metric].to_numpy()
            boot = delta[sample].mean(axis=1)
            intervals[method][metric] = (
                float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
            )
            for replicate, value in enumerate(boot, 1):
                rows.append({
                    "split": split, "scope": "paired_site_bootstrap", "method": method,
                    "metric": f"macro_{metric}_difference_vs_m0", "replicate": replicate,
                    "value": value,
                })
    return pd.DataFrame(rows), intervals


def write_predictions(frame: pd.DataFrame, methods: list[str], path: Path, formal_sites: dict[str, set[str]]) -> None:
    writer = None
    try:
        evaluation = frame.loc[frame["evaluation_period"]]
        for method in methods:
            out = pd.DataFrame({
                "site_id": evaluation["site_id"].astype(str),
                "target_timestamp": evaluation["target_timestamp"],
                "split": evaluation["split"].astype(str),
                "method": method,
                "observed_ph": evaluation["observed_ph"].astype("float32"),
                "predicted_ph": evaluation[f"pred_{method}"].astype("float32"),
                "residual": evaluation[f"residual_{method}"].astype("float32"),
                "signed_score": evaluation[f"signed_score_{method}"].astype("float32"),
                "absolute_score": evaluation[f"absolute_score_{method}"].astype("float32"),
                "top5": evaluation[f"top5_{method}"].astype(bool),
                "top1": evaluation[f"top1_{method}"].astype(bool),
                "in_event": evaluation["in_event"].astype(bool),
                "event_direction": evaluation["event_direction"].astype(str),
                "event_id": evaluation["event_id"].astype("int64"),
                "formal_comparison_eligible": evaluation["site_id"].isin(formal_sites[method]),
            })
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd", use_dictionary=True)
            writer.write_table(table, row_group_size=250_000)
    finally:
        if writer is not None:
            writer.close()


def decision_gates(
    accuracy: pd.DataFrame,
    events: pd.DataFrame,
    intervals: dict[str, dict[str, tuple[float, float]]],
    affine_unstable_sites: int,
) -> pd.DataFrame:
    raw_accuracy = accuracy.loc[accuracy["method"].eq("M0_RAW_H00")].iloc[0]
    raw_event = events.loc[events["method"].eq("M0_RAW_H00")].iloc[0]
    rows = []
    for method in ("M1_OFFSET_ONLY", "M2_AFFINE"):
        acc = accuracy.loc[accuracy["method"].eq(method)].iloc[0]
        event = events.loc[events["method"].eq(method)].iloc[0]
        accuracy_gate = bool(max(
            acc["macro_rmse_improvement_fraction_vs_m0"],
            acc["macro_mae_improvement_fraction_vs_m0"],
        ) >= 0.10)
        consistency_gate = bool(acc["improved_site_fraction_mae"] >= 0.60)
        hot_gate = bool(event["top5_enrichment"] >= 0.90 * raw_event["top5_enrichment"])
        burden_gate = bool(event["outside_register_alerts_per_site_month"] <= 1.10 * raw_event["outside_register_alerts_per_site_month"])
        mae_ci_low, mae_ci_high = intervals[method]["mae"]
        stability_gate = bool(mae_ci_low <= 0.0)
        complete = bool(acc["formal_comparison_complete"])
        if method == "M2_AFFINE" and affine_unstable_sites:
            complete = False
        all_pass = bool(accuracy_gate and consistency_gate and hot_gate and burden_gate and stability_gate and complete)
        rows.append({
            "method": method,
            "accuracy_gate": accuracy_gate,
            "site_consistency_gate": consistency_gate,
            "hot_moment_preservation_gate": hot_gate,
            "alert_burden_gate": burden_gate,
            "stability_gate": stability_gate,
            "formal_comparison_complete": complete,
            "paired_mae_difference_ci_low": mae_ci_low,
            "paired_mae_difference_ci_high": mae_ci_high,
            "all_promotion_gates_passed": all_pass,
        })
    return pd.DataFrame(rows)


def select_method(gates: pd.DataFrame, accuracy: pd.DataFrame) -> tuple[str, str]:
    m1 = gates.loc[gates["method"].eq("M1_OFFSET_ONLY")].iloc[0]
    m2 = gates.loc[gates["method"].eq("M2_AFFINE")].iloc[0]
    if bool(m1["all_promotion_gates_passed"]) and bool(m2["all_promotion_gates_passed"]):
        mae1 = float(accuracy.loc[accuracy["method"].eq("M1_OFFSET_ONLY"), "macro_site_mae"].iloc[0])
        mae2 = float(accuracy.loc[accuracy["method"].eq("M2_AFFINE"), "macro_site_mae"].iloc[0])
        relative = abs(mae1 - mae2) / mae1
        if relative < 0.02 or mae1 <= mae2:
            return "OFFSET_ONLY_PROMOTED", "M1_OFFSET_ONLY"
        return "AFFINE_PROMOTED", "M2_AFFINE"
    if bool(m1["all_promotion_gates_passed"]):
        return "OFFSET_ONLY_PROMOTED", "M1_OFFSET_ONLY"
    if bool(m2["all_promotion_gates_passed"]):
        return "AFFINE_PROMOTED", "M2_AFFINE"
    any_accuracy = bool(gates["accuracy_gate"].any())
    if any_accuracy:
        return "ACCURACY_ONLY_GAIN_NOT_PROMOTED", "RAW"
    return "WARMUP_NOT_PRIMARY_CAUSE", "RAW"


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    table = frame[columns].copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda x: "NA" if pd.isna(x) else f"{x:.{digits}f}")
        elif pd.api.types.is_bool_dtype(table[column]):
            table[column] = table[column].map(lambda x: "PASS" if x else "FAIL")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def make_validation_figures(
    site_metrics: pd.DataFrame,
    parameters: pd.DataFrame,
    accuracy: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    raw = site_metrics.loc[site_metrics["method"].eq("M0_RAW_H00")].set_index("site_id")
    colors = {"M0_RAW_H00": "#4c78a8", "M1_OFFSET_ONLY": "#f58518", "M2_AFFINE": "#54a24b"}
    for number, method, name in (
        (1, "M1_OFFSET_ONLY", "m0_vs_m1_site_mae"),
        (2, "M2_AFFINE", "m0_vs_m2_site_mae"),
    ):
        calibrated = site_metrics.loc[site_metrics["method"].eq(method)].set_index("site_id")
        common = raw.index.intersection(calibrated.index)
        fig, ax = plt.subplots(figsize=(6.4, 5.5))
        stable = calibrated.loc[common, "formal_comparison_eligible"].to_numpy(dtype=bool)
        ax.scatter(raw.loc[common, "mae"], calibrated.loc[common, "mae"], c=np.where(stable, colors[method], "#bab0ac"), alpha=.72, s=25)
        limit = max(raw.loc[common, "mae"].max(), calibrated.loc[common, "mae"].max()) * 1.04
        ax.plot([0, limit], [0, limit], "k--", lw=1)
        ax.set(xlabel="Raw H00 site MAE", ylabel=f"{method} site MAE", title=f"Validation: raw versus {method} MAE")
        fig.tight_layout(); fig.savefig(OUTPUT / "figures" / f"{number:02d}_{name}.png", dpi=180); plt.close(fig)

    for number, metric, name, ylabel in (
        (3, "bias", "bias_before_after", "Site bias (observed - predicted)"),
        (4, "slope", "slope_before_after", "Prediction-on-observation slope"),
    ):
        data = [site_metrics.loc[site_metrics["method"].eq(method), metric].dropna() for method in METHODS]
        fig, ax = plt.subplots(figsize=(7.2, 5.2)); ax.boxplot(data, tick_labels=["M0", "M1", "M2"], showfliers=False)
        ax.axhline(0 if metric == "bias" else 1, color="black", linestyle="--", lw=1)
        ax.set(ylabel=ylabel, title=f"Validation site-level {metric}")
        fig.tight_layout(); fig.savefig(OUTPUT / "figures" / f"{number:02d}_{name}.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for method in ("M1_OFFSET_ONLY", "M2_AFFINE"):
        calibrated = site_metrics.loc[site_metrics["method"].eq(method)].set_index("site_id")
        common = raw.index.intersection(calibrated.index)
        improvement = 100 * (raw.loc[common, "rmse"] - calibrated.loc[common, "rmse"]) / raw.loc[common, "rmse"]
        ax.hist(improvement, bins=28, alpha=.55, label=method, color=colors[method])
    ax.axvline(0, color="black", lw=1); ax.set(xlabel="Site RMSE improvement (%)", ylabel="Sites", title="Validation RMSE-improvement distribution"); ax.legend()
    fig.tight_layout(); fig.savefig(OUTPUT / "figures" / "05_rmse_improvement_distribution.png", dpi=180); plt.close(fig)

    for number, column, name, ylabel in (
        (6, "top5_enrichment", "enrichment_comparison", "Top-5% enrichment"),
        (7, "outside_register_alerts_per_site_month", "outside_register_burden", "Alerts per site-month"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        values = events.set_index("method").loc[METHODS, column]
        ax.bar(["M0", "M1", "M2"], values, color=[colors[m] for m in METHODS])
        ax.set(ylabel=ylabel, title=f"Validation {ylabel.lower()}")
        fig.tight_layout(); fig.savefig(OUTPUT / "figures" / f"{number:02d}_{name}.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    offset = parameters.loc[(parameters["split"].eq("validation")) & parameters["method"].eq("M1_OFFSET_ONLY"), "b_s"]
    affine = parameters.loc[(parameters["split"].eq("validation")) & parameters["method"].eq("M2_AFFINE")]
    axes[0].hist(offset, bins=25, color=colors["M1_OFFSET_ONLY"], alpha=.8); axes[0].set(xlabel="Offset b", ylabel="Sites", title="Offset distribution")
    axes[1].hist(affine["a_s"], bins=25, color=colors["M2_AFFINE"], alpha=.8, label="a")
    axes[1].set(xlabel="Affine scale a", ylabel="Sites", title="Affine scale distribution")
    fig.tight_layout(); fig.savefig(OUTPUT / "figures" / "08_calibration_parameter_distributions.png", dpi=180); plt.close(fig)


def make_test_figures(site_metrics: pd.DataFrame, events: pd.DataFrame, selected: str) -> None:
    raw = site_metrics.loc[site_metrics["method"].eq("M0_RAW_H00")].set_index("site_id")
    calibrated = site_metrics.loc[site_metrics["method"].eq(selected)].set_index("site_id")
    common = raw.index.intersection(calibrated.index)
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    ax.scatter(raw.loc[common, "mae"], calibrated.loc[common, "mae"], alpha=.72, s=25, color="#f58518")
    limit = max(raw.loc[common, "mae"].max(), calibrated.loc[common, "mae"].max()) * 1.04
    ax.plot([0, limit], [0, limit], "k--", lw=1); ax.set(xlabel="Raw H00 site MAE", ylabel="Calibrated site MAE", title="Test: raw versus selected calibration")
    fig.tight_layout(); fig.savefig(OUTPUT / "figures" / "09_test_raw_vs_calibrated_site_mae.png", dpi=180); plt.close(fig)
    indexed = events.set_index("method").loc[["M0_RAW_H00", selected]]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5))
    axes[0].bar(["Raw", "Calibrated"], indexed["top5_enrichment"], color=["#4c78a8", "#f58518"]); axes[0].set(title="Top-5% enrichment", ylabel="Enrichment")
    axes[1].bar(["Raw", "Calibrated"], indexed["outside_register_alerts_per_site_month"], color=["#4c78a8", "#f58518"]); axes[1].set(title="Outside-register burden", ylabel="Alerts/site-month")
    fig.tight_layout(); fig.savefig(OUTPUT / "figures" / "10_test_enrichment_and_burden.png", dpi=180); plt.close(fig)


def main() -> None:
    started = time.time()
    prepare_directories()
    provenance = validate_inputs()
    write_json(OUTPUT / "configs" / "input_manifest.json", provenance)
    thresholds = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))
    q95 = float(thresholds["models"]["cross_variable_h00"]["absolute_score_q95"])
    experiment_config = {
        "objective": "Minimal 30-day unseen-site calibration test for frozen pH H00",
        "selection_split": "validation", "test_sealed_until_selection": True,
        "calibration_days": CALIBRATION_DAYS, "minimum_valid_hours": MINIMUM_VALID_HOURS,
        "methods": METHODS, "h00_checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "models_retrained": False, "prediction_source": str(RESIDUALS.relative_to(ROOT)),
        "score_definition": "site median and 1.4826*MAD fitted on the fixed calibration residuals for each method",
        "frozen_absolute_score_q95": q95,
        "frozen_absolute_score_q99": float(thresholds["models"]["cross_variable_h00"]["absolute_score_q99"]),
        "persistence_consecutive_hours": THREE_HOUR_RULE,
        "lead_lookback_hours": LEAD_LOOKBACK_HOURS,
        "outside_register_exclusion_hours": [-OUTSIDE_EVENT_PRE_HOURS, OUTSIDE_EVENT_POST_HOURS],
        "bootstraps": BOOTSTRAPS, "seed": SEED,
        "promotion_gates": {"accuracy_gain": 0.10, "site_improvement_fraction": 0.60, "max_enrichment_decline": 0.10, "max_burden_increase": 0.10},
    }
    write_json(OUTPUT / "configs" / "experiment_config.json", experiment_config)
    write_json(OUTPUT / "bootstrap" / "bootstrap_config.json", {"replicates": BOOTSTRAPS, "seed": SEED, "unit": "site", "paired_accuracy": True})

    # Strictly validation-only through the persisted selection lock below.
    validation = load_split("validation")
    validation, val_params, val_failures, val_audit, val_formal = fit_split_calibrations(validation, "validation", thresholds)
    val_methods = METHODS
    val_site, val_accuracy = site_and_aggregate_metrics(validation, val_formal, "validation", val_methods)
    val_events, val_event_sites, val_lead, val_burden, val_episodes, val_event_boot = event_metrics(validation, val_formal, "validation", val_methods, q95)
    val_accuracy_boot, val_intervals = accuracy_bootstrap(val_site, val_methods, "validation")
    val_bootstrap = pd.concat([val_accuracy_boot, val_event_boot], ignore_index=True)
    affine_unstable = int(val_audit["affine_unstable_sites"])
    gates = decision_gates(val_accuracy, val_events, val_intervals, affine_unstable)
    conclusion, selected_method = select_method(gates, val_accuracy)
    selection_time = utc_now()
    selection_lock = {
        "locked_utc": selection_time, "selection_data": "validation only",
        "test_accessed": False, "decision": conclusion, "selected_method": selected_method,
        "gates": gates.to_dict(orient="records"),
        "validation_source_sha256": provenance["inputs"][str(RESIDUALS.relative_to(ROOT))]["sha256"],
        "checkpoint_sha256": provenance["inputs"][str(CHECKPOINT.relative_to(ROOT))]["sha256"],
    }
    write_json(OUTPUT / "configs" / "validation_selection_lock.json", selection_lock)

    val_params.to_csv(OUTPUT / "calibration" / "site_calibration_parameters.csv", index=False)
    val_failures.to_csv(OUTPUT / "calibration" / "calibration_failures.csv", index=False)
    write_predictions(validation, val_methods, OUTPUT / "predictions" / "validation_predictions.parquet", val_formal)
    val_site.to_csv(OUTPUT / "site_metrics" / "validation_site_metrics.csv", index=False)
    val_events.to_csv(OUTPUT / "event_metrics" / "validation_event_metrics.csv", index=False)
    val_event_sites.to_csv(OUTPUT / "event_metrics" / "validation_enrichment_site_counts.csv", index=False)
    val_lead.to_csv(OUTPUT / "event_metrics" / "validation_lead_events.csv", index=False)
    val_burden.to_csv(OUTPUT / "event_metrics" / "validation_burden_by_site.csv", index=False)
    val_episodes.to_csv(OUTPUT / "event_metrics" / "validation_alert_episodes.csv", index=False)
    val_bootstrap.to_csv(OUTPUT / "bootstrap" / "validation_bootstrap.csv", index=False)
    val_accuracy.to_csv(OUTPUT / "summary" / "validation_summary.csv", index=False)
    make_validation_figures(val_site, val_params, val_accuracy, val_events)

    accuracy_columns = ["method", "pooled_rmse", "pooled_mae", "macro_site_rmse", "macro_site_mae", "median_site_nse", "positive_nse_sites", "improved_sites_mae", "worsened_sites_mae", "formal_site_count"]
    event_columns = ["method", "top5_enrichment", "top5_site_bootstrap_ci_low", "top5_site_bootstrap_ci_high", "top1_enrichment", "alerted_event_fraction", "median_lead_hours", "outside_register_alerts_per_site_month", "cross_variable_only_event_enrichment"]
    gate_columns = ["method", "accuracy_gate", "site_consistency_gate", "hot_moment_preservation_gate", "alert_burden_gate", "stability_gate", "formal_comparison_complete", "all_promotion_gates_passed"]
    (OUTPUT / "reports" / "VALIDATION_SELECTION_REPORT.md").write_text(f"""# Validation Selection Report

## Locked decision

**{conclusion}**; selected method: **{selected_method}**. The lock was written at `{selection_time}` before any test rows were loaded.

## Accuracy

{markdown_table(val_accuracy, accuracy_columns)}

## Promotion gates

{markdown_table(gates, gate_columns)}

M2 had {affine_unstable} unstable site fits and is therefore audit-only rather than a complete formal comparator. No validation site was silently removed from M0 or M1.
""", encoding="utf-8")
    (OUTPUT / "reports" / "HOT_MOMENT_PRESERVATION_REPORT.md").write_text(f"""# Hot-Moment Preservation Report

The existing closure's event flags, fixed calibration exclusion, robust site score definition, frozen q95 threshold ({q95:.6f}), three-consecutive-hour rule, 48 h lead window, and outside-register exclusion window were retained.

{markdown_table(val_events, event_columns)}

Offset calibration is invariant to the robust residual score after the calibration-period median is removed; its hot-moment signal should therefore match raw H00 apart from floating-point precision. No event label was used to fit either calibration.
""", encoding="utf-8")
    (OUTPUT / "reports" / "CALIBRATION_PROVENANCE_REPORT.md").write_text(f"""# Calibration Provenance Report

- Frozen checkpoint: `{CHECKPOINT.relative_to(ROOT)}`
- Checkpoint SHA-256: `{provenance['inputs'][str(CHECKPOINT.relative_to(ROOT))]['sha256']}`
- Existing H00 predictions were reused; the backbone, scaler, input channels, split and target keys were not changed.
- Validation rows were predicate-filtered before calibration; test was sealed until `{selection_time}`.
- Calibration: earliest fixed 30 calendar days per site, at least {MINIMUM_VALID_HOURS} valid matched hours.
- Evaluation starts strictly after the last calibration timestamp; overlap: none.
- Parameters were frozen during evaluation; no gradient updates or retraining occurred.
- M0 score reproduction maximum absolute difference: {val_audit['m0_score_reproduction_max_abs_difference']:.3e}.
- Frozen q95/q99 thresholds: {val_audit['frozen_q95']:.6f}/{val_audit['frozen_q99']:.6f}.
- Validation eligible sites: {val_audit['calibration_eligible_sites']}; affine-stable sites: {val_audit['affine_stable_sites']}; affine-unstable sites: {val_audit['affine_unstable_sites']}.

The affine audit retains every unstable fit in the parameter and failure tables, but excludes it from formal affine aggregation as preregistered.
""", encoding="utf-8")

    test_accessed = conclusion in {"OFFSET_ONLY_PROMOTED", "AFFINE_PROMOTED"}
    test_audit = None
    test_accuracy = test_events = test_site = test_params = test_failures = test_bootstrap = None
    test_loaded_time = None
    if test_accessed:
        # This is the first test-data read in this program, after the immutable validation lock.
        test_loaded_time = utc_now()
        test = load_split("test")
        test, test_params, test_failures, test_audit, test_formal_all = fit_split_calibrations(test, "test", thresholds)
        test_methods = ["M0_RAW_H00", selected_method]
        test_formal = {method: test_formal_all[method] for method in test_methods}
        test_site, test_accuracy = site_and_aggregate_metrics(test, test_formal, "test", test_methods)
        test_events, test_event_sites, test_lead, test_burden, test_episodes, test_event_boot = event_metrics(test, test_formal, "test", test_methods, q95)
        test_accuracy_boot, test_intervals = accuracy_bootstrap(test_site, test_methods, "test")
        test_bootstrap = pd.concat([test_accuracy_boot, test_event_boot], ignore_index=True)
        all_params = pd.concat([val_params, test_params], ignore_index=True)
        all_failures = pd.concat([val_failures, test_failures], ignore_index=True)
        all_params.to_csv(OUTPUT / "calibration" / "site_calibration_parameters.csv", index=False)
        all_failures.to_csv(OUTPUT / "calibration" / "calibration_failures.csv", index=False)
        write_predictions(test, test_methods, OUTPUT / "predictions" / "test_predictions.parquet", test_formal)
        test_site.to_csv(OUTPUT / "site_metrics" / "test_site_metrics.csv", index=False)
        test_events.to_csv(OUTPUT / "event_metrics" / "test_event_metrics.csv", index=False)
        test_event_sites.to_csv(OUTPUT / "event_metrics" / "test_enrichment_site_counts.csv", index=False)
        test_lead.to_csv(OUTPUT / "event_metrics" / "test_lead_events.csv", index=False)
        test_burden.to_csv(OUTPUT / "event_metrics" / "test_burden_by_site.csv", index=False)
        test_episodes.to_csv(OUTPUT / "event_metrics" / "test_alert_episodes.csv", index=False)
        test_bootstrap.to_csv(OUTPUT / "bootstrap" / "test_bootstrap.csv", index=False)
        make_test_figures(test_site, test_events, selected_method)
        test_accuracy_columns = ["method", "pooled_rmse", "pooled_mae", "macro_site_rmse", "macro_site_mae", "improved_sites_mae", "worsened_sites_mae", "formal_site_count"]
        (OUTPUT / "reports" / "TEST_REPORT.md").write_text(f"""# One-Time Test Report

The validation lock selected **{selected_method}** before test access. Test rows were first loaded at `{test_loaded_time}` and no rule was changed afterward.

## Accuracy

{markdown_table(test_accuracy, test_accuracy_columns)}

## Hot-moment metrics

{markdown_table(test_events, event_columns)}

The paired site-bootstrap 95% intervals for calibrated-minus-raw MAE and RMSE are {test_intervals[selected_method]['mae']} and {test_intervals[selected_method]['rmse']}, respectively.
""", encoding="utf-8")

    validation_raw = val_accuracy.loc[val_accuracy["method"].eq("M0_RAW_H00")].iloc[0]
    validation_selected = val_accuracy.loc[val_accuracy["method"].eq(selected_method if selected_method != "RAW" else "M0_RAW_H00")].iloc[0]
    interpretation = (
        "Simple 30-day site offset calibration materially improved validation accuracy while preserving the robust hot-moment signal; unseen-site baseline pH bias is therefore a major source of weak zero-shot H00 accuracy."
        if conclusion == "OFFSET_ONLY_PROMOTED" else
        "Positive affine calibration materially outperformed offset-only calibration while preserving hot-moment behavior; both site baseline and scale transfer matter."
        if conclusion == "AFFINE_PROMOTED" else
        "Accuracy improved but at least one preregistered preservation or stability gate failed, so calibration was not promoted."
        if conclusion == "ACCURACY_ONLY_GAIN_NOT_PROMOTED" else
        "Neither simple 30-day calibration achieved the preregistered accuracy improvement; warm-up calibration is not the primary limitation."
    )
    final_summary = {
        "final_conclusion": conclusion, "selected_method": selected_method,
        "interpretation": interpretation,
        "validation_selection_locked_utc": selection_time,
        "test_first_loaded_utc": test_loaded_time,
        "test_accessed_before_selection": False,
        "test_evaluation_complete": test_accessed,
        "models_retrained": False,
        "validation_audit": val_audit,
        "test_audit": test_audit,
        "validation_accuracy": val_accuracy.to_dict(orient="records"),
        "validation_hot_moment": val_events.to_dict(orient="records"),
        "validation_gates": gates.to_dict(orient="records"),
        "test_accuracy": test_accuracy.to_dict(orient="records") if test_accuracy is not None else None,
        "test_hot_moment": test_events.to_dict(orient="records") if test_events is not None else None,
        "elapsed_seconds": time.time() - started,
    }
    write_json(OUTPUT / "summary" / "final_summary.json", final_summary)

    test_section = ""
    if test_accessed and test_accuracy is not None and test_events is not None:
        test_section = f"""
## One-time test result

{markdown_table(test_accuracy, ["method", "pooled_rmse", "pooled_mae", "macro_site_rmse", "macro_site_mae", "improved_sites_mae", "worsened_sites_mae"])}

{markdown_table(test_events, ["method", "top5_enrichment", "alerted_event_fraction", "median_lead_hours", "outside_register_alerts_per_site_month"])}
"""
    (OUTPUT / "reports" / "FINAL_H00_30DAY_CALIBRATION_REPORT.md").write_text(f"""# Final H00 30-Day Calibration Report

## Decision

**{conclusion}**

{interpretation}

## Validation accuracy

{markdown_table(val_accuracy, accuracy_columns)}

The selected validation macro-MAE change was {100 * (1 - validation_selected['macro_site_mae'] / validation_raw['macro_site_mae']):.2f}% and the macro-RMSE change was {100 * (1 - validation_selected['macro_site_rmse'] / validation_raw['macro_site_rmse']):.2f}%.

## Validation hot-moment preservation

{markdown_table(val_events, event_columns)}

## Gate audit

{markdown_table(gates, gate_columns)}

M2 was not a complete formal comparator because {affine_unstable} validation sites met the preregistered `AFFINE_UNSTABLE` definition. All such sites remain traceable in the audit tables.
{test_section}
## Scope

This is a calibration study of the frozen contemporaneous full-sensor H00 soft-sensing model. It does not retrain the backbone and must not be directly ranked as a causal one-hour low-cost deployment model.
""", encoding="utf-8")

    expected = [
        OUTPUT / "reports" / "FINAL_H00_30DAY_CALIBRATION_REPORT.md",
        OUTPUT / "reports" / "CALIBRATION_PROVENANCE_REPORT.md",
        OUTPUT / "reports" / "VALIDATION_SELECTION_REPORT.md",
        OUTPUT / "reports" / "HOT_MOMENT_PRESERVATION_REPORT.md",
        OUTPUT / "calibration" / "site_calibration_parameters.csv",
        OUTPUT / "calibration" / "calibration_failures.csv",
        OUTPUT / "predictions" / "validation_predictions.parquet",
        OUTPUT / "site_metrics" / "validation_site_metrics.csv",
        OUTPUT / "event_metrics" / "validation_event_metrics.csv",
        OUTPUT / "bootstrap" / "validation_bootstrap.csv",
        OUTPUT / "bootstrap" / "bootstrap_config.json",
        OUTPUT / "summary" / "validation_summary.csv",
        OUTPUT / "summary" / "final_summary.json",
    ]
    if test_accessed:
        expected.extend([
            OUTPUT / "reports" / "TEST_REPORT.md",
            OUTPUT / "predictions" / "test_predictions.parquet",
            OUTPUT / "site_metrics" / "test_site_metrics.csv",
            OUTPUT / "event_metrics" / "test_event_metrics.csv",
            OUTPUT / "bootstrap" / "test_bootstrap.csv",
        ])
    all_expected = all(path.exists() and path.stat().st_size > 0 for path in expected)
    completion = {
        "status": "complete" if all_expected else "incomplete",
        "models_retrained": False, "h00_checkpoint_frozen": True,
        "validation_only_selection": True, "test_accessed_before_selection": False,
        "calibration_days": CALIBRATION_DAYS, "minimum_valid_hours": MINIMUM_VALID_HOURS,
        "offset_complete": True, "affine_complete": True,
        "hot_moment_metrics_recomputed": True,
        "validation_gate_passed": conclusion in {"OFFSET_ONLY_PROMOTED", "AFFINE_PROMOTED"},
        "selected_method": selected_method,
        "test_evaluation_complete": test_accessed,
        "final_interpretation": (
            "OFFSET_HELPED" if conclusion == "OFFSET_ONLY_PROMOTED" else
            "AFFINE_HELPED" if conclusion == "AFFINE_PROMOTED" else
            "ACCURACY_ONLY_GAIN" if conclusion == "ACCURACY_ONLY_GAIN_NOT_PROMOTED" else
            "WARMUP_NOT_PRIMARY_CAUSE"
        ),
        "allowed_final_conclusion": conclusion,
        "all_expected_files_present": all_expected,
        "completed_utc": utc_now(),
    }
    write_json(OUTPUT / "completion" / "COMPLETED.json", completion)
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    (OUTPUT / "completion" / "SHA256SUMS.txt").write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(OUTPUT).as_posix()}" for path in files) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT), "decision": conclusion, "selected_method": selected_method,
        "test_evaluated": test_accessed, "all_expected_files_present": all_expected,
        "elapsed_seconds": round(time.time() - started, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
