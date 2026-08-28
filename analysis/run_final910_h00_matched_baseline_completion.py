from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "final910_h00_matched_baseline_completion_v1"

H00 = ROOT / "analysis_outputs" / "ph_common1000_lstm_nohistory_h0_rain_final_v1" / "full"
CONFIG = H00 / "reports" / "effective_config.json"
PREPROCESSING = H00 / "reports" / "train_only_preprocessing.json"
CHECKPOINT = H00 / "models" / "ph_LSTM_RAIN_H0_H00.pt"
SUCCESS = H00 / "models" / "ph_LSTM_RAIN_H0_H00_SUCCESS.json"
PRED_ROOT = H00 / "predictions_by_site" / "ph_LSTM_RAIN_H0_H00"

COHORT = (
    ROOT
    / "analysis_outputs"
    / "ph_common1000_teacher_archive_features_do_converted_v1"
    / "full"
    / "reports"
    / "primary_archive_experiment_cohort.csv"
)
HOURLY_ROOT = (
    ROOT
    / "analysis_outputs"
    / "ph_common1000_teacher_archive_features_do_converted_v1"
    / "full"
    / "hourly_ph_rainfall_by_site"
)
RESIDUALS = (
    ROOT
    / "analysis_outputs"
    / "cross_variable_hot_moment_incremental_closure_v1"
    / "residual_scores"
    / "matched_residuals.parquet"
)
CAL_ROOT = ROOT / "analysis_outputs" / "ph_h00_30day_calibration_validation_v1"
CAL_PRED = {
    "validation": CAL_ROOT / "predictions" / "validation_predictions.parquet",
    "test": CAL_ROOT / "predictions" / "test_predictions.parquet",
}
CAL_PARAMS = CAL_ROOT / "calibration" / "site_calibration_parameters.csv"
CAL_SUMMARY = CAL_ROOT / "summary" / "final_summary.json"
WINDOW_CODE = ROOT / "pipeline" / "31_common1000_rainfall_ablation" / "run_rainfall_ablation.py"
CAL_CODE = ROOT / "pipeline" / "45_ph_h00_30day_calibration" / "run_h00_30day_calibration.py"
DISCOVERY_CONFIG = ROOT / "config" / "common1000_baseline_v1.json"

PROTOCOL = OUT / "protocol" / "FROZEN_BASELINE_COMPLETION_PROTOCOL.md"
FEATURE_SPEC = OUT / "protocol" / "baseline_feature_spec.json"
INPUT_MANIFEST = OUT / "audit" / "input_manifest.csv"
INPUT_HASHES = OUT / "audit" / "input_hashes.csv"
SPLIT_AUDIT = OUT / "audit" / "split_and_key_audit.md"
FEATURE_COLUMNS = OUT / "audit" / "baseline_feature_columns.csv"
TRAIN_PREPROCESSING = OUT / "audit" / "train_only_preprocessing.csv"
MODEL_MANIFEST = OUT / "models" / "baseline_model_manifest.csv"
VALIDATION_LOCK = OUT / "audit" / "validation_freeze_hashes.csv"
TEST_LOCK = OUT / "audit" / "TEST_EXECUTION_LOCK.json"

CAL_DAYS = 30
MIN_SUPPORT = 504
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260811
WINDOW = 48

MODEL_FEATURES = ["x_00010", "x_00095", "x_00300", "x_63680", "rain_00045"]
SUMMARY_STATS = ["latest", "mean", "std", "min", "max", "slope", "observed_fraction"]
CALENDAR_FEATURES = ["sin_hour", "cos_hour", "sin_day", "cos_day"]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def key_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    ordered = frame[["site_id", "target_timestamp"]].sort_values(
        ["site_id", "target_timestamp"], kind="mergesort"
    )
    for site_id, group in ordered.groupby("site_id", sort=False):
        digest.update(str(site_id).encode("utf-8"))
        values = pd.to_datetime(group["target_timestamp"]).to_numpy(dtype="datetime64[ns]").view("int64")
        digest.update(values.tobytes())
    return digest.hexdigest()


def metric_values(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    residual = y - pred
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "r2": float(1.0 - sse / sst) if sst > 1e-12 else np.nan,
    }


def canonical_query(split: str) -> pd.DataFrame:
    columns = [
        "site_id", "target_timestamp", "method", "observed_ph", "predicted_ph",
        "formal_comparison_eligible",
    ]
    raw = pd.read_parquet(CAL_PRED[split], columns=columns, filters=[("method", "==", "M0_RAW_H00")])
    off = pd.read_parquet(CAL_PRED[split], columns=columns, filters=[("method", "==", "M1_OFFSET_ONLY")])
    for frame in (raw, off):
        frame["site_id"] = frame["site_id"].astype(str)
        frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"])
        if not frame["formal_comparison_eligible"].all():
            raise RuntimeError(f"Non-formal rows found in canonical {split} predictions")
        if frame.duplicated(["site_id", "target_timestamp"]).any():
            raise RuntimeError(f"Duplicate canonical prediction keys in {split}")
    merged = raw.merge(
        off,
        on=["site_id", "target_timestamp"],
        how="inner",
        validate="one_to_one",
        suffixes=("_raw", "_offset"),
    )
    if len(merged) != len(raw) or len(merged) != len(off):
        raise RuntimeError(f"Raw/offset key mismatch in {split}")
    if not np.array_equal(
        merged["observed_ph_raw"].to_numpy(dtype=np.float32),
        merged["observed_ph_offset"].to_numpy(dtype=np.float32),
    ):
        raise RuntimeError(f"Observed pH differs between raw/offset rows in {split}")
    return merged.rename(
        columns={
            "observed_ph_raw": "y_observed",
            "predicted_ph_raw": "pred_lstm_raw",
            "predicted_ph_offset": "pred_lstm_offset",
        }
    )[["site_id", "target_timestamp", "y_observed", "pred_lstm_raw", "pred_lstm_offset"]]


def canonical_support_query(split: str) -> pd.DataFrame:
    columns = [
        "site_id", "target_timestamp", "split", "observed_ph", "predicted_cross",
        "calibration_period", "calibration_eligible", "evaluation_period",
    ]
    frame = pd.read_parquet(RESIDUALS, columns=columns, filters=[("split", "==", split)])
    frame["site_id"] = frame["site_id"].astype(str)
    frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"])
    if frame.duplicated(["site_id", "target_timestamp"]).any():
        raise RuntimeError(f"Duplicate support/query source keys in {split}")
    # `calibration_eligible` means support count is adequate, but a rare site
    # may still have no later query rows.  The formal cohort is therefore the
    # exact set present in the saved M0/M1 query prediction artifact.
    formal_site_frame = pd.read_parquet(
        CAL_PRED[split], columns=["site_id"], filters=[("method", "==", "M0_RAW_H00")]
    )
    formal_sites = set(formal_site_frame["site_id"].astype(str).unique())
    frame = frame.loc[
        frame["calibration_eligible"].astype(bool) & frame["site_id"].isin(formal_sites)
    ].copy()
    phase_count = frame[["calibration_period", "evaluation_period"]].astype(int).sum(axis=1)
    if not phase_count.eq(1).all():
        raise RuntimeError(f"Support/query phase is not exclusive in {split}")
    frame["support_or_query"] = np.where(frame["calibration_period"], "support", "query")
    for site_id, group in frame.groupby("site_id", sort=False):
        support = group.loc[group["support_or_query"].eq("support")]
        query = group.loc[group["support_or_query"].eq("query")]
        if len(support) < MIN_SUPPORT or query.empty:
            raise RuntimeError(f"Invalid support/query eligibility at {split}/{site_id}")
        if query["target_timestamp"].min() <= support["target_timestamp"].max():
            raise RuntimeError(f"Support/query temporal overlap at {split}/{site_id}")
    return frame.rename(
        columns={"observed_ph": "y_observed", "predicted_cross": "pred_lstm_raw"}
    )[["site_id", "target_timestamp", "y_observed", "pred_lstm_raw", "support_or_query"]].sort_values(
        ["site_id", "target_timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def canonical_metrics(query: pd.DataFrame, pred_column: str) -> dict[str, float]:
    site_rows = []
    for site_id, group in query.groupby("site_id", sort=False):
        site_rows.append({"site_id": site_id, **metric_values(group["y_observed"], group[pred_column])})
    sites = pd.DataFrame(site_rows)
    pooled = metric_values(query["y_observed"], query[pred_column])
    return {
        "rows": len(query),
        "sites": query["site_id"].nunique(),
        "pooled_mae": pooled["mae"],
        "pooled_rmse": pooled["rmse"],
        "macro_site_mae": float(sites["mae"].mean()),
        "macro_site_rmse": float(sites["rmse"].mean()),
    }


def verify_original_predictions(split: str, query: pd.DataFrame) -> dict[str, Any]:
    max_pred = 0.0
    max_y = 0.0
    rows = 0
    for site_id, keys in query.groupby("site_id", sort=False):
        source = pd.read_parquet(
            PRED_ROOT / split / f"{site_id}.parquet",
            columns=["timestamp", "observed_native", "predicted_native"],
        )
        source["timestamp"] = pd.to_datetime(source["timestamp"])
        merged = keys.merge(
            source,
            left_on="target_timestamp",
            right_on="timestamp",
            how="left",
            validate="one_to_one",
        )
        if merged[["observed_native", "predicted_native"]].isna().any().any():
            raise RuntimeError(f"Original H00 prediction key missing at {split}/{site_id}")
        max_pred = max(max_pred, float(np.max(np.abs(merged["pred_lstm_raw"] - merged["predicted_native"]))))
        max_y = max(max_y, float(np.max(np.abs(merged["y_observed"] - merged["observed_native"]))))
        rows += len(merged)
    if max_pred > 2e-6 or max_y > 2e-6:
        raise RuntimeError(f"Original H00 prediction identity failed in {split}: pred={max_pred}, y={max_y}")
    return {"rows_checked": rows, "maximum_prediction_difference": max_pred, "maximum_observed_difference": max_y}


def required_paths() -> list[tuple[str, Path]]:
    paths = [
        ("final_910_register_and_split", COHORT),
        ("retained_h00_effective_config", CONFIG),
        ("retained_h00_train_preprocessing", PREPROCESSING),
        ("retained_h00_checkpoint", CHECKPOINT),
        ("retained_h00_success_manifest", SUCCESS),
        ("support_query_source", RESIDUALS),
        ("saved_validation_calibration_predictions", CAL_PRED["validation"]),
        ("saved_test_calibration_predictions", CAL_PRED["test"]),
        ("saved_calibration_parameters", CAL_PARAMS),
        ("saved_calibration_summary", CAL_SUMMARY),
        ("exact_h00_window_implementation", WINDOW_CODE),
        ("retained_calibration_implementation", CAL_CODE),
        ("incompatible_discovery_baseline_config", DISCOVERY_CONFIG),
    ]
    register = pd.read_csv(COHORT, dtype={"site_id": str})
    for row in register.itertuples(index=False):
        paths.append((f"final910_hourly_input_{row.split}", Path(row.output_file)))
    for split in ("validation", "test"):
        for path in sorted((PRED_ROOT / split).glob("*.parquet")):
            paths.append((f"saved_h00_prediction_{split}", path))
    return paths


def preflight() -> None:
    if TEST_LOCK.exists():
        raise RuntimeError("Test lock already exists; refusing to repeat the task")
    print("[preflight] validating authoritative identities", flush=True)
    for _, path in required_paths()[:14]:
        if not path.exists():
            raise FileNotFoundError(path)
    config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    success = json.loads(SUCCESS.read_text(encoding="utf-8-sig"))
    register = pd.read_csv(COHORT, dtype={"site_id": str})
    counts = register.groupby("split")["site_id"].nunique().to_dict()
    if counts != {"test": 135, "train": 638, "validation": 137} or len(register) != 910:
        raise RuntimeError(f"Final split identity mismatch: rows={len(register)}, counts={counts}")
    if (
        int(config["horizon_hours"]) != 0
        or int(config["sequence_length_hours"]) != 48
        or bool(config.get("include_target_history", True))
        or config["models"][0]["model_id"] != "LSTM_RAIN_H0"
        or config["models"][0]["feature_names"] != MODEL_FEATURES
    ):
        raise RuntimeError("Retained H00 configuration identity is ambiguous or unexpected")
    if success.get("run_id") not in (None, "ph_LSTM_RAIN_H0_H00") and success.get("model_id") not in (None, "LSTM_RAIN_H0"):
        raise RuntimeError("Checkpoint success identity does not match retained H00")

    feature_order = []
    for name in MODEL_FEATURES:
        feature_order.extend([f"{name}_standardized", f"{name}_missing"])
    feature_order.extend(["sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"])

    audits: dict[str, Any] = {}
    split_lines = []
    for split, expected_sites in (("validation", 128), ("test", 126)):
        print(f"[preflight] reproducing saved {split} H00 cohort", flush=True)
        query = canonical_query(split)
        support_query = canonical_support_query(split)
        query_source = support_query.loc[support_query["support_or_query"].eq("query")]
        matched = query.merge(
            query_source,
            on=["site_id", "target_timestamp"],
            how="inner",
            validate="one_to_one",
            suffixes=("_saved", "_source"),
        )
        if len(matched) != len(query) or len(query) != len(query_source):
            raise RuntimeError(f"Canonical support/query and saved prediction keys differ in {split}")
        if query["site_id"].nunique() != expected_sites:
            raise RuntimeError(f"Eligible {split} sites={query['site_id'].nunique()}, expected {expected_sites}")
        if np.max(np.abs(matched["pred_lstm_raw_saved"] - matched["pred_lstm_raw_source"])) > 2e-6:
            raise RuntimeError(f"Saved/source LSTM values differ in {split}")
        raw_metrics = canonical_metrics(query, "pred_lstm_raw")
        offset_metrics = canonical_metrics(query, "pred_lstm_offset")
        if split == "test":
            expected = {
                "raw_mae": 0.2421,
                "raw_rmse": 0.2929,
                "offset_mae": 0.1808,
                "offset_rmse": 0.2374,
            }
            observed = {
                "raw_mae": raw_metrics["macro_site_mae"],
                "raw_rmse": raw_metrics["macro_site_rmse"],
                "offset_mae": offset_metrics["macro_site_mae"],
                "offset_rmse": offset_metrics["macro_site_rmse"],
            }
            if any(abs(observed[k] - expected[k]) > 0.001 for k in expected):
                raise RuntimeError(f"Saved H00 metric reproduction outside tolerance: {observed}")
        original_check = verify_original_predictions(split, query)
        audits[split] = {
            "eligible_sites": expected_sites,
            "support_rows": int(support_query["support_or_query"].eq("support").sum()),
            "query_rows": int(support_query["support_or_query"].eq("query").sum()),
            "support_key_sha256": key_digest(support_query.loc[support_query["support_or_query"].eq("support")]),
            "query_key_sha256": key_digest(query),
            "raw_metrics": raw_metrics,
            "offset_metrics": offset_metrics,
            "original_prediction_check": original_check,
        }
        split_lines.append(
            f"| {split} | {expected_sites} | {audits[split]['support_rows']:,} | {audits[split]['query_rows']:,} | "
            f"{raw_metrics['macro_site_mae']:.9f} | {raw_metrics['macro_site_rmse']:.9f} | "
            f"{offset_metrics['macro_site_mae']:.9f} | {offset_metrics['macro_site_rmse']:.9f} |"
        )

    split_text = f"""# Final-910 H00 split and key audit

Generated UTC: {now_utc()}
Decision: authoritative identities are unique and reproducible; mandatory stop conditions did not trigger.

## Final register and retained model

- Register: `{COHORT}`
- Site split: 638 train / 137 validation / 135 test; 910 unique sites.
- Checkpoint: `{CHECKPOINT}`
- Checkpoint SHA-256: `{sha256(CHECKPOINT)}`
- Model identity: `ph_LSTM_RAIN_H0_H00`; pH H00; 48-hour window ending at target time; no current or historical pH input.
- Target unit: native pH units.
- Residual convention: `observed pH - predicted pH`.
- Exact recurrent input order: `{feature_order}`.
- Window eligibility: finite target; target index has 48-hour history; combined missing fraction across the four non-rain sensor channels <=0.5; precipitation feature available at all 48 input hours.

## Saved prediction and support/query reproduction

| split | eligible sites | support rows | query rows | raw macro MAE | raw macro RMSE | offset macro MAE | offset macro RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
{os.linesep.join(split_lines)}

The key is exactly `(site_id, target_timestamp)`. Keys are unique. Support and query rows are disjoint and strictly ordered within site. Saved calibration predictions match the original retained per-site H00 predictions to the recorded float tolerance. No nearest-time join, filling, duplicate collapse or cohort expansion was used.

## Key digests

```json
{json.dumps(json_ready(audits), indent=2, ensure_ascii=False)}
```
"""
    SPLIT_AUDIT.write_text(split_text, encoding="utf-8")

    protocol_text = f"""# Frozen final-910 H00 matched-baseline completion protocol

Frozen UTC: {now_utc()}
Status at freeze: **no new baseline test prediction has been generated**.

## Scope and chronology

This is a post-hoc completion of missing matched baselines after the retained H00 test result already existed. It is not an originally preregistered experiment. The retained `ph_LSTM_RAIN_H0_H00` checkpoint and its raw/30-day-offset conclusions remain frozen and will not be retrained or altered. Early 1,009-site metrics will not be reused as final-910 evidence.

## Frozen cohort

The primary comparison uses the exact eligible post-support query keys reconstructed in `{SPLIT_AUDIT}`: 128 validation sites and 126 test sites. The same 30-day support rows are used only to define own-offset calibration. Validation/test sites are excluded from every fitted parameter and preprocessing estimate.

## Frozen baselines

1. `B_GLOBAL_TRAIN_MEAN`: one constant from all 9,729,447 eligible H00 targets on 638 training sites.
2. `B_UTC_MONTH_HOUR_MEAN`: UTC month x hour means from the same training targets, with global-mean fallback.
3. `RIDGE_SUMMARY48_H00`: SUMMARY48 features; alpha=1.0; intercept enabled; no search.
4. `HGB_SUMMARY48_H00`: squared error; learning_rate=0.05; max_iter=200; max_leaf_nodes=31; min_samples_leaf=50; l2_regularization=1.0; early_stopping=false; random_state=20260811.
5. `B0_SUPPORT_MEAN_30D`: per-site support pH mean, calibrated track only.

The old discovery Ridge/HGB configuration is incompatible because it used a different feature table and a 1,009-site discovery cohort. Therefore the protocol fallback configurations above are frozen.

## Frozen SUMMARY48 representation

For each of `x_00010`, `x_00095`, `x_00300`, `x_63680`, and log1p-nonnegative `rain_00045`, over the 48-hour window ending at target time t: latest available, mean, population standard deviation, minimum, maximum, ordinary least-squares time slope over observed hours, and observed fraction. Add target-time `sin_hour`, `cos_hour`, `sin_day`, and `cos_day`, which already exist in the retained H00 inputs. Exclude pH, pH lags, target-derived rolling values, events, site identity, coordinates, country/source, static metadata and hydrology.

Ridge uses training-only medians, continuous-column standardisation, and one missing indicator per retained summary column. HGB consumes the identical retained raw summary columns and native missing values. All-missing columns, if any, are dropped using training evidence only and recorded.

## Frozen evaluation

- Validation verifies the pipeline and does not select a configuration.
- No completed baseline will be removed because it performs strongly.
- Test baseline inference is one-shot after validation artifacts, configurations and models are hashed.
- Raw predictions receive no target-derived adjustment.
- Each calibrated model receives only its own support-period mean residual.
- `B_GLOBAL_TRAIN_MEAN + own offset` must equal `B0_SUPPORT_MEAN_30D` numerically and is not duplicated in the main table.
- Primary metric: post-offset macro-site MAE. Safety metric: post-offset macro-site RMSE.
- Uncertainty: 10,000 fixed-seed complete-site bootstrap replicates; hourly rows are never bootstrapped.
- No residual alert, conformal, event, hot-moment, architecture, feature or threshold experiment is authorised.
"""
    PROTOCOL.write_text(protocol_text, encoding="utf-8")

    print("[preflight] hashing authoritative inputs", flush=True)
    manifest_rows = []
    for role, path in required_paths():
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        manifest_rows.append({
            "role": role,
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        })
    write_csv(INPUT_MANIFEST, pd.DataFrame(manifest_rows))
    hash_rows = []
    for row in manifest_rows:
        path = Path(row["path"])
        hash_rows.append({**row, "sha256": sha256(path), "note": "authoritative input"})
    for role, path in [
        ("frozen_protocol", PROTOCOL),
        ("input_manifest", INPUT_MANIFEST),
        ("split_and_key_audit", SPLIT_AUDIT),
    ]:
        stat = path.stat()
        hash_rows.append({
            "role": role,
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": sha256(path),
            "note": "pre-test frozen audit artifact",
        })
    write_csv(INPUT_HASHES, pd.DataFrame(hash_rows))
    print(f"[preflight] complete: {len(hash_rows)} hashes; no stop condition triggered", flush=True)


def transform_values(values: np.ndarray, transform: str) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    if transform == "identity":
        return result
    if transform == "log1p_nonnegative":
        result[result < 0] = np.nan
        return np.log1p(result)
    raise ValueError(transform)


def common_indices(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    target = pd.to_numeric(frame["target_00400_native"], errors="coerce").to_numpy(dtype=np.float64)
    indices = np.flatnonzero(np.isfinite(target))
    indices = indices[indices - WINDOW + 1 >= 0]
    if not len(indices):
        return indices
    missing = np.column_stack(
        [frame[f"x_{code}_missing"].to_numpy(dtype=np.int8) for code in config["base_dynamic_codes"]]
    )
    hourly_missing = missing.sum(axis=1).astype(np.int64)
    prefix = np.concatenate(([0], np.cumsum(hourly_missing, dtype=np.int64)))
    start = indices - WINDOW + 1
    missing_total = prefix[indices + 1] - prefix[start]
    valid = missing_total / float(WINDOW * len(config["base_dynamic_codes"])) <= float(
        config["maximum_dynamic_missing_fraction_per_window"]
    )
    available = frame["precip_feature_available"].fillna(False).to_numpy(dtype=bool)
    avail_prefix = np.concatenate(([0], np.cumsum(available, dtype=np.int64)))
    valid &= (avail_prefix[indices + 1] - avail_prefix[start]) == WINDOW
    return indices[valid]


def rolling_sum(values: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(end - WINDOW, 0)
    return cumulative[end] - cumulative[start]


def channel_summary(values: np.ndarray, selected: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    y = np.where(finite, values, 0.0)
    x = np.arange(len(values), dtype=np.float64)
    count = rolling_sum(finite.astype(np.float64))
    sum_y = rolling_sum(y)
    sum_y2 = rolling_sum(y * y)
    sum_x = rolling_sum(np.where(finite, x, 0.0))
    sum_x2 = rolling_sum(np.where(finite, x * x, 0.0))
    sum_xy = rolling_sum(np.where(finite, x * y, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sum_y / count
        variance = np.maximum(sum_y2 / count - mean * mean, 0.0)
        std = np.sqrt(variance)
        denominator = sum_x2 - sum_x * sum_x / count
        slope = (sum_xy - sum_x * sum_y / count) / denominator
    mean[count == 0] = np.nan
    std[count == 0] = np.nan
    slope[(count < 2) | (denominator <= 1e-12)] = np.nan
    series = pd.Series(values)
    latest = series.ffill(limit=WINDOW - 1).to_numpy(dtype=np.float64)
    minimum = series.rolling(WINDOW, min_periods=1).min().to_numpy(dtype=np.float64)
    maximum = series.rolling(WINDOW, min_periods=1).max().to_numpy(dtype=np.float64)
    fraction = count / float(WINDOW)
    return np.column_stack(
        [latest[selected], mean[selected], std[selected], minimum[selected], maximum[selected], slope[selected], fraction[selected]]
    ).astype(np.float32)


def summary_feature_names() -> list[str]:
    names = [f"{channel}__{stat}" for channel in MODEL_FEATURES for stat in SUMMARY_STATS]
    names.extend(CALENDAR_FEATURES)
    return names


def site_summary_features(frame: pd.DataFrame, selected: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    parts = []
    definitions = config["feature_definitions"]
    for name in MODEL_FEATURES:
        definition = definitions[name]
        raw = pd.to_numeric(frame[definition["source_column"]], errors="coerce").to_numpy(dtype=np.float64)
        parts.append(channel_summary(transform_values(raw, definition["transform"]), selected))
    parts.append(frame.loc[selected, CALENDAR_FEATURES].to_numpy(dtype=np.float32))
    return np.column_stack(parts).astype(np.float32)


def load_hourly(site_id: str, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "timestamp", "target_00400_native", "target_00400_observed", "precip_feature_available",
        *[f"x_{code}_missing" for code in config["base_dynamic_codes"]],
        *CALENDAR_FEATURES,
    ]
    for name in MODEL_FEATURES:
        source = config["feature_definitions"][name]["source_column"]
        if source not in columns:
            columns.append(source)
    frame = pd.read_parquet(HOURLY_ROOT / f"{site_id}.parquet", columns=columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def features_for_keys(keys: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    output = np.empty((len(keys), len(summary_feature_names())), dtype=np.float32)
    for number, (site_id, group) in enumerate(keys.groupby("site_id", sort=False), 1):
        frame = load_hourly(site_id, config)
        indexer = pd.Index(frame["timestamp"]).get_indexer(pd.DatetimeIndex(group["target_timestamp"]))
        if (indexer < 0).any():
            raise RuntimeError(f"Exact target keys absent from hourly input for {site_id}")
        valid = common_indices(frame, config)
        if not np.isin(indexer, valid).all():
            raise RuntimeError(f"Selected support/query key violates frozen H00 eligibility for {site_id}")
        output[group.index.to_numpy()] = site_summary_features(frame, indexer, config)
        if number % 25 == 0:
            print(f"[features] {number} sites", flush=True)
    return output


def ridge_transform(x: np.ndarray, prep: dict[str, np.ndarray]) -> np.ndarray:
    imputed = np.where(np.isfinite(x), x, prep["medians"])
    continuous = (imputed - prep["means"]) / prep["scales"]
    indicators = (~np.isfinite(x)).astype(np.float64)
    return np.column_stack([continuous, indicators])


def ridge_predict(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    z = ridge_transform(x.astype(np.float64), model)
    return model["intercept"] + z @ model["coefficients"]


def build_prediction_frame(split: str, config: dict[str, Any], models: dict[str, Any]) -> pd.DataFrame:
    keys = canonical_support_query(split)
    keys = keys.reset_index(drop=True)
    print(f"[{split}] building SUMMARY48 for {len(keys):,} support/query rows", flush=True)
    x = features_for_keys(keys, config)
    kept = np.asarray(models["kept_indices"], dtype=int)
    x = x[:, kept]
    timestamps = pd.to_datetime(keys["target_timestamp"])
    month_hour_key = timestamps.dt.month.astype(str).str.zfill(2) + "_" + timestamps.dt.hour.astype(str).str.zfill(2)
    month_map = models["month_hour_map"]
    keys["pred_global_raw"] = float(models["global_mean"])
    keys["pred_month_hour_raw"] = month_hour_key.map(month_map).fillna(models["global_mean"]).to_numpy(dtype=float)
    keys["pred_ridge_raw"] = ridge_predict(x, models["ridge"])
    if models.get("hgb") is not None:
        keys["pred_hgb_raw"] = models["hgb"].predict(x)
    else:
        keys["pred_hgb_raw"] = np.nan
    for name in ["global", "month_hour", "ridge", "hgb", "lstm"]:
        raw_col = f"pred_{name}_raw"
        offset_col = f"offset_{name}"
        if name != "hgb" or models.get("hgb") is not None:
            support = keys.loc[keys["support_or_query"].eq("support"), ["site_id", "y_observed", raw_col]].copy()
            support["residual"] = (
                support["y_observed"].to_numpy(dtype=np.float64)
                - support[raw_col].to_numpy(dtype=np.float64)
            )
            offset_map = support.groupby("site_id", sort=False)["residual"].mean()
            keys[offset_col] = keys["site_id"].map(offset_map)
            keys[f"pred_{name}_offset"] = keys[raw_col] + keys[offset_col]
        else:
            keys[offset_col] = np.nan
            keys[f"pred_{name}_offset"] = np.nan
    support_values = keys.loc[keys["support_or_query"].eq("support"), ["site_id", "y_observed"]].copy()
    support_values["y_observed"] = support_values["y_observed"].astype(np.float64)
    support_means = support_values.groupby("site_id")["y_observed"].mean()
    keys["pred_b0_support_mean_30d"] = keys["site_id"].map(support_means)
    equality = np.max(np.abs(keys["pred_global_offset"] - keys["pred_b0_support_mean_30d"]))
    if equality > 1e-10:
        raise RuntimeError(f"Global+offset != B0 support mean: {equality}")
    if keys.filter(regex=r"^pred_(global|month_hour|ridge|lstm)_(raw|offset)$").isna().any().any():
        raise RuntimeError(f"Non-finite required baseline prediction in {split}")
    return keys


def load_models() -> dict[str, Any]:
    config = json.loads((OUT / "models" / "frozen_baseline_config.json").read_text(encoding="utf-8"))
    month = pd.read_csv(OUT / "models" / "month_hour_means.csv")
    models: dict[str, Any] = {
        "global_mean": config["global_train_mean"],
        "kept_indices": config["kept_feature_indices"],
        "month_hour_map": dict(zip(month["month_hour_key"], month["mean_ph"])),
        "ridge": joblib.load(OUT / "models" / "ridge_summary48_h00.joblib"),
        "hgb": None,
    }
    hgb_path = OUT / "models" / "hgb_summary48_h00.joblib"
    if hgb_path.exists():
        models["hgb"] = joblib.load(hgb_path)
    return models


def method_columns(hgb_available: bool) -> dict[str, list[tuple[str, str]]]:
    raw = [
        ("B_GLOBAL_TRAIN_MEAN", "pred_global_raw"),
        ("B_UTC_MONTH_HOUR_MEAN", "pred_month_hour_raw"),
        ("RIDGE_SUMMARY48_H00", "pred_ridge_raw"),
    ]
    calibrated = [
        ("B0_SUPPORT_MEAN_30D", "pred_b0_support_mean_30d"),
        ("B_UTC_MONTH_HOUR_MEAN_OFFSET", "pred_month_hour_offset"),
        ("RIDGE_SUMMARY48_H00_OFFSET", "pred_ridge_offset"),
    ]
    if hgb_available:
        raw.append(("HGB_SUMMARY48_H00", "pred_hgb_raw"))
        calibrated.append(("HGB_SUMMARY48_H00_OFFSET", "pred_hgb_offset"))
    raw.append(("LSTM_H00_RAW", "pred_lstm_raw"))
    calibrated.append(("LSTM_H00_OFFSET", "pred_lstm_offset"))
    return {"raw_zero_calibration": raw, "own_offset_calibrated": calibrated}


def basic_metric_tables(frame: pd.DataFrame, split: str, hgb_available: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    query = frame.loc[frame["support_or_query"].eq("query")].copy()
    site_rows = []
    aggregate_rows = []
    for track, methods in method_columns(hgb_available).items():
        for method, column in methods:
            if not np.isfinite(query[column]).all():
                raise RuntimeError(f"Non-finite {split} prediction for {method}")
            per_site = []
            for site_id, group in query.groupby("site_id", sort=False):
                values = metric_values(group["y_observed"], group[column])
                row = {
                    "split": split,
                    "track": track,
                    "method": method,
                    "site_id": site_id,
                    "rows": len(group),
                    **values,
                }
                site_rows.append(row)
                per_site.append(row)
            sites = pd.DataFrame(per_site)
            pooled = metric_values(query["y_observed"], query[column])
            aggregate_rows.append({
                "split": split,
                "track": track,
                "method": method,
                "rows": len(query),
                "sites": len(sites),
                "pooled_mae": pooled["mae"],
                "pooled_rmse": pooled["rmse"],
                "pooled_bias_observed_minus_predicted": pooled["bias"],
                "pooled_r2": pooled["r2"],
                "macro_site_mae": float(sites["mae"].mean()),
                "macro_site_rmse": float(sites["rmse"].mean()),
                "macro_site_bias_observed_minus_predicted": float(sites["bias"].mean()),
                "macro_site_r2_eligible_mean": float(sites["r2"].dropna().mean()),
                "macro_site_r2_eligible_sites": int(sites["r2"].notna().sum()),
                "median_site_mae": float(sites["mae"].median()),
                "lstm_improves_sites": np.nan,
                "lstm_improves_percent": np.nan,
                "lstm_worsens_sites": np.nan,
                "lstm_worsens_percent": np.nan,
                "sites_with_gt2pct_lstm_harm": np.nan,
                "paired_interpretation": "",
            })
    return pd.DataFrame(site_rows), pd.DataFrame(aggregate_rows)


def validation() -> None:
    if TEST_LOCK.exists():
        raise RuntimeError("Test lock exists; validation cannot be rerun")
    if VALIDATION_LOCK.exists():
        raise RuntimeError("Validation freeze already exists; refusing to alter frozen baselines")
    for path in (PROTOCOL, INPUT_MANIFEST, INPUT_HASHES, SPLIT_AUDIT):
        if not path.exists():
            raise RuntimeError(f"Preflight artifact missing: {path}")
    config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    official_preprocessing = json.loads(PREPROCESSING.read_text(encoding="utf-8-sig"))
    expected_rows = int(official_preprocessing["target"]["fit_common_target_rows"])
    names = summary_feature_names()
    register = pd.read_csv(COHORT, dtype={"site_id": str})
    train = register.loc[register["split"].eq("train")].copy()
    if len(train) != 638:
        raise RuntimeError("Training site count changed")

    x_path = OUT / "working" / "train_summary48.npy"
    y_path = OUT / "working" / "train_target.npy"
    x_mem = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(expected_rows, len(names)))
    y_mem = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.float32, shape=(expected_rows,))
    month_sums = np.zeros((12, 24), dtype=np.float64)
    month_counts = np.zeros((12, 24), dtype=np.int64)
    cursor = 0
    print(f"[validation] building training SUMMARY48: {expected_rows:,} rows", flush=True)
    for number, row in enumerate(train.itertuples(index=False), 1):
        site_id = str(row.site_id)
        frame = load_hourly(site_id, config)
        indices = common_indices(frame, config)
        count = len(indices)
        end = cursor + count
        if end > expected_rows:
            raise RuntimeError("Training feature row count exceeds frozen target count")
        x_mem[cursor:end] = site_summary_features(frame, indices, config)
        target = pd.to_numeric(frame.loc[indices, "target_00400_native"], errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(target).all():
            raise RuntimeError(f"Non-finite training target at {site_id}")
        y_mem[cursor:end] = target
        stamps = pd.DatetimeIndex(frame.loc[indices, "timestamp"])
        codes = (stamps.month.to_numpy() - 1) * 24 + stamps.hour.to_numpy()
        month_sums.flat[:] += np.bincount(codes, weights=target, minlength=12 * 24)
        month_counts.flat[:] += np.bincount(codes, minlength=12 * 24)
        cursor = end
        if number % 25 == 0:
            print(f"[validation] training features: {number}/638 sites; {cursor:,} rows", flush=True)
    x_mem.flush()
    y_mem.flush()
    if cursor != expected_rows:
        raise RuntimeError(f"Training row identity mismatch: {cursor:,} != {expected_rows:,}")
    global_mean = float(np.mean(y_mem, dtype=np.float64))
    if abs(global_mean - float(official_preprocessing["target"]["mean"])) > 1e-9:
        raise RuntimeError("Training target mean does not reproduce official H00 preprocessing")

    print("[validation] fitting train-only medians and scaling", flush=True)
    medians = np.empty(len(names), dtype=np.float64)
    means = np.empty(len(names), dtype=np.float64)
    scales = np.empty(len(names), dtype=np.float64)
    missing_counts = np.empty(len(names), dtype=np.int64)
    kept = []
    preprocessing_rows = []
    for j, name in enumerate(names):
        column = np.asarray(x_mem[:, j], dtype=np.float64)
        finite = np.isfinite(column)
        missing = int((~finite).sum())
        missing_counts[j] = missing
        if finite.sum() == 0:
            medians[j] = np.nan
            means[j] = np.nan
            scales[j] = np.nan
            dropped = True
        else:
            medians[j] = float(np.median(column[finite]))
            total = float(column[finite].sum()) + missing * medians[j]
            means[j] = total / expected_rows
            square = float(np.square(column[finite] - means[j]).sum()) + missing * (medians[j] - means[j]) ** 2
            scales[j] = math.sqrt(max(square / expected_rows, 0.0))
            if scales[j] < 1e-12:
                scales[j] = 1.0
            dropped = False
            kept.append(j)
        preprocessing_rows.append({
            "feature": name,
            "training_rows": expected_rows,
            "nonmissing_rows": int(finite.sum()),
            "missing_rows": missing,
            "training_median": medians[j],
            "post_imputation_mean": means[j],
            "post_imputation_population_std": scales[j],
            "all_missing_dropped": dropped,
            "ridge_imputation": "training median",
            "ridge_missing_indicator": not dropped,
            "hgb_missing_rule": "native NaN",
        })
    kept = np.asarray(kept, dtype=int)
    if not len(kept):
        raise RuntimeError("All SUMMARY48 columns are missing")
    write_csv(TRAIN_PREPROCESSING, pd.DataFrame(preprocessing_rows))

    feature_rows = []
    for j, name in enumerate(names):
        feature_rows.append({
            "source_order": j,
            "feature": name,
            "retained": j in set(kept.tolist()),
            "ridge_continuous_column": f"{name}__standardized" if j in kept else "",
            "ridge_missing_indicator": f"{name}__missing" if j in kept else "",
            "hgb_column": name if j in kept else "",
            "uses_target_or_event": False,
        })
    write_csv(FEATURE_COLUMNS, pd.DataFrame(feature_rows))

    feature_spec = {
        "frozen_utc": now_utc(),
        "source": str(HOURLY_ROOT),
        "window_hours": WINDOW,
        "window_end": "target timestamp t inclusive",
        "horizon_hours": 0,
        "target_history": False,
        "channels_in_retained_h00_order": MODEL_FEATURES,
        "channel_sources_and_transforms": {
            name: config["feature_definitions"][name] for name in MODEL_FEATURES
        },
        "summary_statistics_in_order": SUMMARY_STATS,
        "standard_deviation": "population ddof=0 over observed hours",
        "slope": "ordinary least squares of transformed value on integer hour position using observed hours only",
        "observed_fraction_denominator": 48,
        "calendar_features_at_target": CALENDAR_FEATURES,
        "raw_feature_columns": names,
        "retained_feature_columns": [names[j] for j in kept],
        "dropped_all_missing_columns": [names[j] for j in range(len(names)) if j not in set(kept.tolist())],
        "ridge": {
            "implementation": "closed-form Ridge normal equations on streamed sufficient statistics",
            "alpha": 1.0,
            "fit_intercept": True,
            "intercept_penalized": False,
            "continuous_preprocessing": "training median then training population standardisation",
            "missing_indicators": "one raw 0/1 indicator per retained feature",
            "hyperparameter_search": False,
        },
        "hgb": {
            "implementation": "sklearn.ensemble.HistGradientBoostingRegressor",
            "loss": "squared_error",
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0,
            "early_stopping": False,
            "random_state": 20260811,
            "native_missing_values": True,
            "hyperparameter_search": False,
        },
        "forbidden": [
            "current_or_historical_pH", "target_derived_rolling_values", "event_labels", "site_id",
            "coordinates", "country", "source", "static_metadata", "hydrology", "Flow", "Stage",
        ],
    }
    write_json(FEATURE_SPEC, feature_spec)

    month_rows = []
    for month in range(1, 13):
        for hour in range(24):
            count = int(month_counts[month - 1, hour])
            mean = float(month_sums[month - 1, hour] / count) if count else global_mean
            month_rows.append({
                "month": month,
                "utc_hour": hour,
                "month_hour_key": f"{month:02d}_{hour:02d}",
                "training_target_count": count,
                "mean_ph": mean,
                "fallback_to_global": count == 0,
            })
    month_path = OUT / "models" / "month_hour_means.csv"
    write_csv(month_path, pd.DataFrame(month_rows))

    kept_x = x_mem[:, kept]
    kept_medians = medians[kept]
    kept_means = means[kept]
    kept_scales = scales[kept]
    p = len(kept)
    xtx = np.zeros((2 * p, 2 * p), dtype=np.float64)
    xty = np.zeros(2 * p, dtype=np.float64)
    sum_z = np.zeros(2 * p, dtype=np.float64)
    sum_y = 0.0
    chunk = 100_000
    print(f"[validation] fitting Ridge alpha=1.0 with {2*p} coefficients", flush=True)
    prep = {"medians": kept_medians, "means": kept_means, "scales": kept_scales}
    for start in range(0, expected_rows, chunk):
        end = min(start + chunk, expected_rows)
        z = ridge_transform(np.asarray(kept_x[start:end], dtype=np.float64), prep)
        y = np.asarray(y_mem[start:end], dtype=np.float64)
        xtx += z.T @ z
        xty += z.T @ y
        sum_z += z.sum(axis=0)
        sum_y += float(y.sum())
    augmented = np.empty((2 * p + 1, 2 * p + 1), dtype=np.float64)
    augmented[0, 0] = expected_rows
    augmented[0, 1:] = sum_z
    augmented[1:, 0] = sum_z
    augmented[1:, 1:] = xtx + np.eye(2 * p, dtype=np.float64)
    rhs = np.concatenate(([sum_y], xty))
    solution = np.linalg.solve(augmented, rhs)
    ridge_model = {
        **prep,
        "intercept": float(solution[0]),
        "coefficients": solution[1:],
        "feature_names": [names[j] for j in kept],
        "design_columns": [f"{names[j]}__standardized" for j in kept]
        + [f"{names[j]}__missing" for j in kept],
        "alpha": 1.0,
        "solver": "numpy.linalg.solve on exact streamed normal equations",
        "fit_intercept": True,
        "intercept_penalized": False,
        "training_rows": expected_rows,
        "training_sites": 638,
        "numpy_version": np.__version__,
    }
    ridge_path = OUT / "models" / "ridge_summary48_h00.joblib"
    joblib.dump(ridge_model, ridge_path, compress=3)

    hgb_path = OUT / "models" / "hgb_summary48_h00.joblib"
    hgb_status = "COMPLETED"
    hgb_error = ""
    print("[validation] fitting frozen HGB (no early stopping, no search)", flush=True)
    hgb = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=20260811,
        verbose=1,
    )
    try:
        hgb.fit(kept_x, y_mem)
        joblib.dump(hgb, hgb_path, compress=3)
    except (MemoryError, OSError) as exc:
        hgb_status = "HGB_NOT_RUN_RESOURCE_LIMIT"
        hgb_error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        write_json(OUT / "audit" / "hgb_resource_failure.json", {
            "status": hgb_status,
            "error": hgb_error,
            "training_rows": expected_rows,
            "training_features": p,
            "timestamp_utc": now_utc(),
        })
        hgb = None

    frozen_config = {
        "frozen_utc": now_utc(),
        "post_hoc_completion": True,
        "training_sites": 638,
        "eligible_training_rows": expected_rows,
        "global_train_mean": global_mean,
        "kept_feature_indices": kept.tolist(),
        "kept_feature_names": [names[j] for j in kept],
        "ridge_alpha": 1.0,
        "ridge_coefficient_count": 2 * p,
        "hgb_parameters": feature_spec["hgb"],
        "hgb_status": hgb_status,
        "hgb_error": hgb_error,
        "validation_selects_configuration": False,
        "test_baseline_inference_performed": False,
    }
    frozen_config_path = OUT / "models" / "frozen_baseline_config.json"
    write_json(frozen_config_path, frozen_config)

    model_rows = [
        {
            "model": "B_GLOBAL_TRAIN_MEAN", "status": "COMPLETED", "training_sites": 638,
            "training_rows": expected_rows, "configuration": "one training-target constant",
            "artifact": str(frozen_config_path), "artifact_sha256": sha256(frozen_config_path),
        },
        {
            "model": "B_UTC_MONTH_HOUR_MEAN", "status": "COMPLETED", "training_sites": 638,
            "training_rows": expected_rows, "configuration": "UTC month x hour; global fallback",
            "artifact": str(month_path), "artifact_sha256": sha256(month_path),
        },
        {
            "model": "RIDGE_SUMMARY48_H00", "status": "COMPLETED", "training_sites": 638,
            "training_rows": expected_rows, "configuration": f"alpha=1.0; intercept; {2*p} coefficients",
            "artifact": str(ridge_path), "artifact_sha256": sha256(ridge_path),
        },
        {
            "model": "HGB_SUMMARY48_H00", "status": hgb_status, "training_sites": 638,
            "training_rows": expected_rows, "configuration": "frozen protocol fallback; no search",
            "artifact": str(hgb_path) if hgb_path.exists() else "",
            "artifact_sha256": sha256(hgb_path) if hgb_path.exists() else "",
        },
        {
            "model": "B0_SUPPORT_MEAN_30D", "status": "DEFERRED_TO_CALIBRATED_COHORT", "training_sites": 0,
            "training_rows": 0, "configuration": "per-site exact 30-day support mean; query only",
            "artifact": str(PROTOCOL), "artifact_sha256": sha256(PROTOCOL),
        },
    ]
    write_csv(MODEL_MANIFEST, pd.DataFrame(model_rows))

    models = load_models()
    validation_frame = build_prediction_frame("validation", config, models)
    validation_query = validation_frame.loc[validation_frame["support_or_query"].eq("query")]
    canonical = canonical_query("validation")
    check = validation_query.merge(
        canonical,
        on=["site_id", "target_timestamp"],
        how="inner",
        validate="one_to_one",
        suffixes=("_new", "_saved"),
    )
    if len(check) != len(validation_query) or len(check) != len(canonical):
        raise RuntimeError("Validation LSTM key reproduction failed")
    max_raw = float(np.max(np.abs(check["pred_lstm_raw_new"] - check["pred_lstm_raw_saved"])))
    max_offset = float(np.max(np.abs(check["pred_lstm_offset_new"] - check["pred_lstm_offset_saved"])))
    if max_raw > 2e-6 or max_offset > 2e-6:
        raise RuntimeError(f"Validation LSTM prediction reproduction failed: raw={max_raw}, offset={max_offset}")
    validation_path = OUT / "predictions" / "matched_validation_predictions.parquet"
    write_parquet(validation_path, validation_frame)
    site_metrics, aggregate = basic_metric_tables(validation_frame, "validation", hgb is not None)
    validation_site_path = OUT / "metrics" / "validation_site_metrics.csv"
    validation_aggregate_path = OUT / "metrics" / "validation_aggregate_metrics.csv"
    write_csv(validation_site_path, site_metrics)
    write_csv(validation_aggregate_path, aggregate)
    cohort_rows = []
    for site_id, group in validation_query.groupby("site_id", sort=False):
        cohort_rows.append({
            "site_id": site_id,
            "query_rows": len(group),
            "first_query": group["target_timestamp"].min(),
            "last_query": group["target_timestamp"].max(),
            "key_sha256": key_digest(group),
        })
    validation_cohort = OUT / "audit" / "validation_cohort_key_manifest.csv"
    write_csv(validation_cohort, pd.DataFrame(cohort_rows))
    statement = OUT / "audit" / "VALIDATION_FREEZE_STATEMENT.md"
    statement.write_text(
        "# Validation freeze statement\n\n"
        f"Frozen UTC: {now_utc()}\n\n"
        "All baseline definitions, feature columns, preprocessing parameters and fitted artifacts are frozen. "
        "Validation was used only to verify generalisation and pipeline integrity. No configuration or feature "
        "was selected from validation performance, and no completed baseline will be removed because it performs strongly. "
        "No new baseline test prediction existed when this statement was written.\n",
        encoding="utf-8",
    )
    freeze_paths = [
        PROTOCOL, FEATURE_SPEC, FEATURE_COLUMNS, TRAIN_PREPROCESSING, frozen_config_path, month_path,
        ridge_path, MODEL_MANIFEST, validation_path, validation_site_path, validation_aggregate_path,
        validation_cohort, statement,
    ]
    if hgb_path.exists():
        freeze_paths.append(hgb_path)
    freeze_rows = []
    for path in freeze_paths:
        freeze_rows.append({
            "path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size,
            "frozen_before_test": True,
        })
    write_csv(VALIDATION_LOCK, pd.DataFrame(freeze_rows))
    del x_mem, y_mem, kept_x
    for path in (x_path, y_path):
        if path.exists():
            path.unlink()
    print(
        f"[validation] frozen: validation rows={len(validation_query):,}, sites={validation_query['site_id'].nunique()}, "
        f"HGB={hgb_status}",
        flush=True,
    )


def verify_validation_lock() -> None:
    rows = pd.read_csv(VALIDATION_LOCK)
    problems = []
    for row in rows.itertuples(index=False):
        path = Path(row.path)
        if not path.exists() or sha256(path) != row.sha256:
            problems.append(str(path))
    if problems:
        raise RuntimeError("Frozen validation artifact changed: " + "; ".join(problems))


def bootstrap_mean(values: np.ndarray, seed: int, replicates: int = BOOTSTRAPS) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(replicates, dtype=np.float64)
    batch = 500
    cursor = 0
    while cursor < replicates:
        take = min(batch, replicates - cursor)
        indices = rng.integers(0, n, size=(take, n))
        means[cursor:cursor + take] = values[indices].mean(axis=1)
        cursor += take
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def full_test_metrics(
    frame: pd.DataFrame,
    hgb_available: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    site_metrics, aggregate = basic_metric_tables(frame, "test", hgb_available)
    paired_rows = []
    bootstrap_rows = []
    for track, methods in method_columns(hgb_available).items():
        lstm_method = "LSTM_H00_RAW" if track == "raw_zero_calibration" else "LSTM_H00_OFFSET"
        lstm = site_metrics.loc[
            site_metrics["track"].eq(track) & site_metrics["method"].eq(lstm_method),
            ["site_id", "mae", "rmse"],
        ].rename(columns={"mae": "lstm_mae", "rmse": "lstm_rmse"})
        for method, _ in methods:
            method_sites = site_metrics.loc[
                site_metrics["track"].eq(track) & site_metrics["method"].eq(method),
                ["site_id", "mae", "rmse"],
            ]
            seed = int(hashlib.sha256(f"{BOOTSTRAP_SEED}|{track}|{method}|macro".encode()).hexdigest()[:8], 16)
            estimate, low, high = bootstrap_mean(method_sites["mae"].to_numpy(), seed)
            bootstrap_rows.append({
                "summary_kind": "method_macro_site_mae", "track": track, "comparison": method,
                "estimate": estimate, "ci_low": low, "ci_high": high, "sites": len(method_sites),
                "replicates": BOOTSTRAPS, "seed": seed, "sampling_unit": "complete site",
                "interpretation": "not_applicable",
            })
            if method == lstm_method:
                continue
            paired = method_sites.rename(columns={"mae": "baseline_mae", "rmse": "baseline_rmse"}).merge(
                lstm, on="site_id", how="inner", validate="one_to_one"
            )
            paired["delta_mae_lstm_minus_baseline"] = paired["lstm_mae"] - paired["baseline_mae"]
            paired["relative_lstm_harm_fraction"] = paired["delta_mae_lstm_minus_baseline"] / paired["baseline_mae"]
            paired["lstm_improves"] = paired["delta_mae_lstm_minus_baseline"] < 0
            paired["lstm_worsens"] = paired["delta_mae_lstm_minus_baseline"] > 0
            paired["lstm_harm_gt_2pct"] = paired["relative_lstm_harm_fraction"] > 0.02
            paired["split"] = "test"
            paired["track"] = track
            paired["baseline"] = method
            paired_rows.extend(paired.to_dict("records"))
            diff = paired["delta_mae_lstm_minus_baseline"].to_numpy()
            pair_seed = int(hashlib.sha256(f"{BOOTSTRAP_SEED}|{track}|{method}|paired".encode()).hexdigest()[:8], 16)
            mean_diff, ci_low, ci_high = bootstrap_mean(diff, pair_seed)
            baseline_agg = aggregate.loc[(aggregate["track"].eq(track)) & (aggregate["method"].eq(method))].iloc[0]
            lstm_agg = aggregate.loc[(aggregate["track"].eq(track)) & (aggregate["method"].eq(lstm_method))].iloc[0]
            if lstm_agg["macro_site_mae"] >= baseline_agg["macro_site_mae"]:
                interpretation = "NO_LSTM_ADVANTAGE"
            elif ci_high < 0 and lstm_agg["macro_site_rmse"] <= baseline_agg["macro_site_rmse"]:
                interpretation = "CLEAR_LSTM_OUTPERFORMANCE"
            else:
                interpretation = "POINT_ADVANTAGE_ONLY"
            bootstrap_rows.append({
                "summary_kind": "paired_lstm_minus_baseline_mae", "track": track, "comparison": method,
                "estimate": mean_diff, "ci_low": ci_low, "ci_high": ci_high, "sites": len(paired),
                "replicates": BOOTSTRAPS, "seed": pair_seed, "sampling_unit": "complete site",
                "interpretation": interpretation,
            })
            mask = (aggregate["track"].eq(track)) & (aggregate["method"].eq(method))
            aggregate.loc[mask, "lstm_improves_sites"] = int(paired["lstm_improves"].sum())
            aggregate.loc[mask, "lstm_improves_percent"] = 100.0 * float(paired["lstm_improves"].mean())
            aggregate.loc[mask, "lstm_worsens_sites"] = int(paired["lstm_worsens"].sum())
            aggregate.loc[mask, "lstm_worsens_percent"] = 100.0 * float(paired["lstm_worsens"].mean())
            aggregate.loc[mask, "sites_with_gt2pct_lstm_harm"] = int(paired["lstm_harm_gt_2pct"].sum())
            aggregate.loc[mask, "paired_interpretation"] = interpretation
    paired_frame = pd.DataFrame(paired_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    return site_metrics, aggregate, paired_frame, bootstrap_frame


def make_figure(
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    bootstrap: pd.DataFrame,
    hgb_available: bool,
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 6, "axes.labelsize": 7,
        "axes.titlesize": 7, "xtick.labelsize": 5.5, "ytick.labelsize": 6,
        "legend.fontsize": 5.5, "axes.linewidth": 0.6, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    labels = ["Global", "UTC\nmonth×hour", "Ridge", "HGB", "LSTM", "B0\nsupport mean"]
    raw_methods = ["B_GLOBAL_TRAIN_MEAN", "B_UTC_MONTH_HOUR_MEAN", "RIDGE_SUMMARY48_H00", "HGB_SUMMARY48_H00", "LSTM_H00_RAW", None]
    cal_methods = [None, "B_UTC_MONTH_HOUR_MEAN_OFFSET", "RIDGE_SUMMARY48_H00_OFFSET", "HGB_SUMMARY48_H00_OFFSET", "LSTM_H00_OFFSET", "B0_SUPPORT_MEAN_30D"]
    if not hgb_available:
        raw_methods[3] = None
        cal_methods[3] = None
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    colors = {"raw": "#0072B2", "cal": "#D55E00"}
    for offset, methods, track, color, legend in [
        (-width / 2, raw_methods, "raw_zero_calibration", colors["raw"], "Raw / zero calibration"),
        (width / 2, cal_methods, "own_offset_calibrated", colors["cal"], "Own-offset calibrated"),
    ]:
        vals, lows, highs = [], [], []
        for method in methods:
            if method is None:
                vals.append(np.nan); lows.append(np.nan); highs.append(np.nan); continue
            row = aggregate.loc[(aggregate["track"].eq(track)) & (aggregate["method"].eq(method))].iloc[0]
            boot = bootstrap.loc[
                bootstrap["summary_kind"].eq("method_macro_site_mae")
                & bootstrap["track"].eq(track)
                & bootstrap["comparison"].eq(method)
            ].iloc[0]
            vals.append(row["macro_site_mae"])
            lows.append(row["macro_site_mae"] - boot["ci_low"])
            highs.append(boot["ci_high"] - row["macro_site_mae"])
        vals_a = np.asarray(vals, dtype=float)
        axes[0].bar(x + offset, vals_a, width=width, color=color, edgecolor="black", linewidth=0.35, label=legend)
        valid = np.isfinite(vals_a)
        axes[0].errorbar(x[valid] + offset, vals_a[valid], yerr=np.vstack([np.asarray(lows)[valid], np.asarray(highs)[valid]]), fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5, capthick=0.6)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Macro-site MAE (pH)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.35)

    comparisons = ["B0_SUPPORT_MEAN_30D", "B_UTC_MONTH_HOUR_MEAN_OFFSET", "RIDGE_SUMMARY48_H00_OFFSET"]
    comparison_labels = ["B0", "UTC month×hour", "Ridge"]
    if hgb_available:
        comparisons.append("HGB_SUMMARY48_H00_OFFSET")
        comparison_labels.append("HGB")
    arrays = [
        paired.loc[(paired["track"].eq("own_offset_calibrated")) & (paired["baseline"].eq(method)), "delta_mae_lstm_minus_baseline"].to_numpy()
        for method in comparisons
    ]
    violin = axes[1].violinplot(arrays, positions=np.arange(1, len(arrays) + 1), showmeans=False, showmedians=True, widths=0.75)
    palette = ["#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
    for body, color in zip(violin["bodies"], palette):
        body.set_facecolor(color); body.set_edgecolor("black"); body.set_linewidth(0.4); body.set_alpha(0.85)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        violin[key].set_color("black"); violin[key].set_linewidth(0.6)
    axes[1].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[1].set_xticks(np.arange(1, len(arrays) + 1), comparison_labels)
    axes[1].set_ylabel("Site MAE difference: LSTM − baseline (pH)")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.35)
    axes[0].text(-0.12, 1.03, "a", transform=axes[0].transAxes, fontsize=8, fontweight="bold")
    axes[1].text(-0.12, 1.03, "b", transform=axes[1].transAxes, fontsize=8, fontweight="bold")
    for spine in ("top", "right"):
        axes[0].spines[spine].set_visible(False)
        axes[1].spines[spine].set_visible(False)
    base = OUT / "figures" / "final910_h00_matched_baselines"
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for row in frame[columns].itertuples(index=False, name=None):
        cells = []
        for column, value in zip(columns, row):
            if pd.isna(value):
                cells.append("")
            elif column in formats and isinstance(value, (float, np.floating)):
                cells.append(format(value, formats[column]))
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows])


def report_text(
    aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
    hgb_available: bool,
    test_frame: pd.DataFrame,
) -> str:
    raw = aggregate.loc[aggregate["track"].eq("raw_zero_calibration")].copy()
    calibrated = aggregate.loc[aggregate["track"].eq("own_offset_calibrated")].copy()
    pair = bootstrap.loc[bootstrap["summary_kind"].eq("paired_lstm_minus_baseline_mae")].copy()
    status = "FINAL910_H00_MATCHED_BASELINE_COMPLETION_COMPLETE" if hgb_available else "FINAL910_H00_MATCHED_BASELINE_COMPLETION_PARTIAL"
    cols = ["method", "sites", "rows", "macro_site_mae", "macro_site_rmse", "pooled_mae", "pooled_rmse", "median_site_mae", "pooled_r2"]
    formats = {name: ".6f" for name in cols if name not in {"method", "sites", "rows"}}
    pair_cols = ["track", "comparison", "estimate", "ci_low", "ci_high", "sites", "interpretation"]
    pair_formats = {"estimate": ".6f", "ci_low": ".6f", "ci_high": ".6f"}
    raw_lstm = raw.loc[raw["method"].eq("LSTM_H00_RAW")].iloc[0]
    cal_lstm = calibrated.loc[calibrated["method"].eq("LSTM_H00_OFFSET")].iloc[0]
    support_rows = int(test_frame["support_or_query"].eq("support").sum())
    query_rows = int(test_frame["support_or_query"].eq("query").sum())
    return f"""# Final-910 matched pH H00 baseline completion

Completion date: {now_utc()}
Scope: final 910-site register; retained `ph_LSTM_RAIN_H0_H00`; point-prediction accuracy only.

## Decision and chronology

This is a **post-hoc completion of missing matched baselines** because the retained H00 test result already existed. The protocol, SUMMARY48 representation, preprocessing, model configurations, fitted artifacts and validation results were frozen and hashed before any new baseline test prediction. Early 1,009-site metrics were not reused as final-910 evidence.

Persistence was excluded because it consumes historical pH and violates H00. Test-site means and site x month x hour means were excluded because they require target history from an unseen site. No model was removed because of strong performance. The retained LSTM checkpoint and prior H00 conclusions were not changed.

## Authoritative identity and matched cohort

- Final split: 638 training / 137 validation / 135 test sites.
- Primary comparison: 126 calibration-eligible test sites, {support_rows:,} exact support rows and {query_rows:,} exact later query rows.
- Unique key: `(site_id, target_timestamp)`; no nearest-time join, filling, duplicate collapse or cohort expansion.
- H00 input boundary: 48 hours ending at time t; non-pH sensors only; current/historical pH forbidden.
- Residual/bias convention: observed pH minus predicted pH.
- Saved LSTM reproduction on the primary query cohort: raw macro-site MAE {raw_lstm['macro_site_mae']:.9f}, raw macro-site RMSE {raw_lstm['macro_site_rmse']:.9f}; offset macro-site MAE {cal_lstm['macro_site_mae']:.9f}, offset macro-site RMSE {cal_lstm['macro_site_rmse']:.9f}.

## Frozen baseline implementation

`B_GLOBAL_TRAIN_MEAN` and UTC month x hour means used all 9,729,447 eligible training targets and no validation/test target. `RIDGE_SUMMARY48_H00` used alpha 1.0, an unpenalised intercept, 39 SUMMARY48/calendar features plus one missing indicator per feature, and no hyperparameter search. `HGB_SUMMARY48_H00` used the protocol fallback configuration with 200 iterations, native missing values, early stopping disabled and no search. The discovery configuration was rejected as incompatible with the new feature table.

Every model-specific offset is the mean support residual for that same model and site. The global-mean own-offset prediction equalled `B0_SUPPORT_MEAN_30D` numerically and is not duplicated below.

## Raw / zero-calibration results

{markdown_table(raw, cols, formats)}

## Own-offset calibrated results

Primary metric: macro-site MAE. Safety metric: macro-site RMSE.

{markdown_table(calibrated, cols, formats)}

## Paired complete-site uncertainty and interpretation

Differences are `LSTM site MAE − baseline site MAE`; negative favours LSTM. Intervals are fixed-seed 10,000-replicate bootstraps over complete sites, never hourly rows.

{markdown_table(pair, pair_cols, pair_formats)}

`CLEAR_LSTM_OUTPERFORMANCE` requires lower LSTM macro-site MAE, an entirely negative paired 95% interval and no worsening of macro-site RMSE. `POINT_ADVANTAGE_ONLY` means the LSTM point MAE is lower but that full rule is not met. `NO_LSTM_ADVANTAGE` means LSTM macro-site MAE is equal or higher.

Site-level metrics, LSTM improve/worsen counts and the >2% LSTM-harm count are in `metrics/site_metrics.csv`, `metrics/aggregate_metrics.csv` and `metrics/paired_site_differences.csv`. The editable PDF/SVG and 600-dpi PNG figure report all completed models without axis truncation.

## Integrity statements

- Validation reported generalisation but selected no configuration.
- Training-only medians/scales and all fitted model parameters were frozen before the one-shot test run.
- Support/query overlap is zero; query targets entered metrics only.
- Each offset used support outcomes from the same model and site; the LSTM offset was never reused for a baseline.
- No residual alert, event enrichment, conformal, hot-moment, feature rescue, threshold or additional architecture experiment was run.
- All completed baselines are reported, including any that outperform the LSTM.

{status}
"""


def write_output_hashes() -> None:
    target = OUT / "output_hashes.csv"
    rows = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p != target):
        stat = path.stat()
        rows.append({
            "path": str(path), "sha256": sha256(path), "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "note": "final-910 matched-baseline completion output",
        })
    rows.append({
        "path": str(target), "sha256": "", "size_bytes": "", "mtime_utc": "",
        "note": "SELF_EXCLUDED_TO_AVOID_CIRCULAR_HASH",
    })
    write_csv(target, pd.DataFrame(rows))


def test_once() -> None:
    if TEST_LOCK.exists():
        raise RuntimeError("One-shot test lock already exists; refusing a rescue or repeated run")
    if not VALIDATION_LOCK.exists():
        raise RuntimeError("Validation freeze is absent")
    verify_validation_lock()
    write_json(TEST_LOCK, {
        "created_utc": now_utc(),
        "purpose": "one-shot final-910 H00 matched-baseline test execution",
        "validation_freeze_sha256": sha256(VALIDATION_LOCK),
        "protocol_sha256": sha256(PROTOCOL),
        "repeated_or_rescue_run_permitted": False,
    })
    print("[test] one-shot lock written; constructing exact test-key features", flush=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    models = load_models()
    hgb_available = models.get("hgb") is not None
    frame = build_prediction_frame("test", config, models)
    query = frame.loc[frame["support_or_query"].eq("query")]
    canonical = canonical_query("test")
    check = query.merge(
        canonical,
        on=["site_id", "target_timestamp"],
        how="inner",
        validate="one_to_one",
        suffixes=("_new", "_saved"),
    )
    if len(check) != len(query) or len(check) != len(canonical):
        raise RuntimeError("One-shot test keys do not reproduce the frozen cohort")
    raw_difference = float(np.max(np.abs(check["pred_lstm_raw_new"] - check["pred_lstm_raw_saved"])))
    offset_difference = float(np.max(np.abs(check["pred_lstm_offset_new"] - check["pred_lstm_offset_saved"])))
    if raw_difference > 2e-6 or offset_difference > 2e-6:
        raise RuntimeError(f"Saved LSTM reproduction failed in one-shot test: raw={raw_difference}, offset={offset_difference}")
    prediction_path = OUT / "predictions" / "matched_test_predictions.parquet"
    write_parquet(prediction_path, frame)
    print("[test] computing complete-site metrics and 10,000-replicate bootstraps", flush=True)
    site_metrics, aggregate, paired, bootstrap = full_test_metrics(frame, hgb_available)
    write_csv(OUT / "metrics" / "aggregate_metrics.csv", aggregate)
    write_csv(OUT / "metrics" / "site_metrics.csv", site_metrics)
    write_csv(OUT / "metrics" / "paired_site_differences.csv", paired)
    write_csv(OUT / "metrics" / "site_bootstrap_summary.csv", bootstrap)
    make_figure(aggregate, paired, bootstrap, hgb_available)
    report = report_text(aggregate, bootstrap, hgb_available, frame)
    report_path = OUT / "FINAL_910_H00_MATCHED_BASELINE_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    write_output_hashes()
    terminal = report_path.read_text(encoding="utf-8").splitlines()[-1]
    expected = "FINAL910_H00_MATCHED_BASELINE_COMPLETION_COMPLETE" if hgb_available else "FINAL910_H00_MATCHED_BASELINE_COMPLETION_PARTIAL"
    if terminal != expected:
        raise RuntimeError("Final report terminal status is invalid")
    print(f"[test] complete: {expected}", flush=True)


def verify_outputs() -> None:
    required = [
        OUT / "FINAL_910_H00_MATCHED_BASELINE_REPORT.md", PROTOCOL, FEATURE_SPEC, INPUT_MANIFEST,
        INPUT_HASHES, SPLIT_AUDIT, FEATURE_COLUMNS, TRAIN_PREPROCESSING, MODEL_MANIFEST,
        OUT / "predictions" / "matched_test_predictions.parquet",
        OUT / "metrics" / "aggregate_metrics.csv", OUT / "metrics" / "site_metrics.csv",
        OUT / "metrics" / "paired_site_differences.csv", OUT / "metrics" / "site_bootstrap_summary.csv",
        OUT / "output_hashes.csv", OUT / "figures" / "final910_h00_matched_baselines.pdf",
        OUT / "figures" / "final910_h00_matched_baselines.svg",
        OUT / "figures" / "final910_h00_matched_baselines.png",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing required output: " + "; ".join(missing))
    hashes = pd.read_csv(OUT / "output_hashes.csv").fillna("")
    mismatches = []
    for row in hashes.itertuples(index=False):
        if row.note == "SELF_EXCLUDED_TO_AVOID_CIRCULAR_HASH":
            continue
        path = Path(row.path)
        if not path.exists() or sha256(path) != row.sha256:
            mismatches.append(str(path))
    if mismatches:
        raise RuntimeError("Output hash mismatch: " + "; ".join(mismatches))
    pred = pq.ParquetFile(OUT / "predictions" / "matched_test_predictions.parquet")
    required_columns = {
        "site_id", "target_timestamp", "y_observed", "support_or_query", "pred_global_raw",
        "pred_month_hour_raw", "pred_ridge_raw", "pred_lstm_raw", "offset_global",
        "offset_month_hour", "offset_ridge", "offset_lstm", "pred_b0_support_mean_30d",
        "pred_month_hour_offset", "pred_ridge_offset", "pred_lstm_offset",
    }
    if not required_columns.issubset(set(pred.schema_arrow.names)):
        raise RuntimeError("Prediction table is missing required columns")
    print(f"[verify] all outputs and hashes valid; prediction rows={pred.metadata.num_rows:,}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "validation", "test", "verify"])
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight()
    elif args.stage == "validation":
        validation()
    elif args.stage == "test":
        test_once()
    else:
        verify_outputs()


if __name__ == "__main__":
    main()
