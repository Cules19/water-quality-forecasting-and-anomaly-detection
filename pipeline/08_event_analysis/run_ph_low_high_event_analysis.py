from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import seaborn as sns


warnings.filterwarnings("ignore", category=FutureWarning)


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
OUT_DIR = ROOT / "analysis_outputs/ph_low_high_event_analysis_v1"
FIG_DIR = OUT_DIR / "figures"
LOG_PATH = OUT_DIR / "ph_low_high_event_analysis_log.txt"


TOP_FRACTIONS = [0.01, 0.05, 0.10]
EVENT_CLASSES = ["normal", "low_pH_event", "high_pH_event", "mixed_or_unknown"]
METRIC_CLASSES = ["overall", "normal", "low_pH_event", "high_pH_event", "all_pH_event", "mixed_or_unknown"]
REPORT_MODEL_ORDER = [
    "M0_reference_nonevent",
    "M1_no_metadata",
    "M2_with_metadata",
    "HGB50_V2",
    "M200_A_no_metadata",
    "M200_B_with_metadata",
    "HGB200_V2",
    "G100_A",
    "G100_B",
    "GRU100_V2",
    "L100_B",
]


UNIFIED_SCHEMA = pa.schema(
    [
        ("model_id", pa.string()),
        ("model_type", pa.string()),
        ("site_count", pa.int16()),
        ("metadata_set", pa.string()),
        ("site_id", pa.string()),
        ("timestamp", pa.timestamp("ns")),
        ("y_true", pa.float32()),
        ("y_pred", pa.float32()),
        ("residual_y_true_minus_pred", pa.float32()),
        ("prediction_error_y_pred_minus_true", pa.float32()),
        ("abs_residual", pa.float32()),
        ("in_ph_event", pa.int8()),
        ("ph_event_label", pa.string()),
        ("event_state", pa.int8()),
        ("event_class", pa.string()),
        ("is_event", pa.bool_()),
        ("is_low_pH_event", pa.bool_()),
        ("is_high_pH_event", pa.bool_()),
    ]
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_type: str
    site_count: int
    metadata_set: str
    prediction_file: str
    include: bool = True
    row_filter_col: str | None = None
    row_filter_value: str | None = None
    main_role: str = ""
    expected: bool = True
    exclusion_reason: str = ""


MODEL_SPECS = [
    ModelSpec(
        "M0_reference_nonevent",
        "HGB",
        50,
        "non_event_train_no_metadata",
        "analysis_outputs/ph_nonevent_training_full50_v3/nonevent_predictions_test.parquet",
        row_filter_col="model",
        row_filter_value="M4_hgb_nonevent_train",
        main_role="normal-behaviour residual reference",
    ),
    ModelSpec(
        "M3_ridge_nonevent_train",
        "Ridge",
        50,
        "non_event_train_no_metadata",
        "analysis_outputs/ph_nonevent_training_full50_v3/nonevent_predictions_test.parquet",
        include=False,
        row_filter_col="model",
        row_filter_value="M3_ridge_nonevent_train",
        main_role="non-core sensitivity model",
        expected=False,
        exclusion_reason="Ridge non-core sensitivity model; retained in inventory but excluded from the core low/high comparison.",
    ),
    ModelSpec(
        "M1_no_metadata",
        "HGB",
        50,
        "no_metadata",
        "analysis_outputs/ph_hgb_full50_metadata_v1/prediction_test_M1_no_metadata.parquet",
        main_role="full-data HGB baseline",
    ),
    ModelSpec(
        "M2_with_metadata",
        "HGB",
        50,
        "full_metadata",
        "analysis_outputs/ph_hgb_full50_metadata_v1/prediction_test_M2_with_metadata.parquet",
        main_role="metadata-aware HGB prediction model",
    ),
    ModelSpec(
        "HGB50_V2",
        "HGB",
        50,
        "compact_hydrological_v2_metadata",
        "analysis_outputs/ph_v2_metadata_rerun_v1/hgb50_v2_predictions_test.parquet",
        main_role="V2 interpretable metadata model",
    ),
    ModelSpec(
        "M200_A_no_metadata",
        "HGB",
        200,
        "no_metadata",
        "analysis_outputs/ph_hgb_processed200_metadata_v1/prediction_test_M200_A_no_metadata.parquet",
        main_role="full-data HGB baseline",
    ),
    ModelSpec(
        "M200_B_with_metadata",
        "HGB",
        200,
        "full_metadata",
        "analysis_outputs/ph_hgb_processed200_metadata_v1/prediction_test_M200_B_with_metadata.parquet",
        main_role="metadata-aware HGB prediction model",
    ),
    ModelSpec(
        "HGB200_V2",
        "HGB",
        200,
        "compact_hydrological_v2_metadata",
        "analysis_outputs/ph_v2_metadata_rerun_v1/hgb200_v2_predictions_test.parquet",
        main_role="V2 interpretable metadata model",
    ),
    ModelSpec(
        "G100_A",
        "GRU",
        100,
        "no_metadata",
        "analysis_outputs/ph_sequence100_gru_lstm_metadata_v1/prediction_test_G100_A_gru_no_metadata.parquet",
        main_role="temporal GRU model",
    ),
    ModelSpec(
        "G100_B",
        "GRU",
        100,
        "full_metadata",
        "analysis_outputs/ph_sequence100_gru_lstm_metadata_v1/prediction_test_G100_B_gru_with_metadata.parquet",
        main_role="temporal GRU model",
    ),
    ModelSpec(
        "GRU100_V2",
        "GRU",
        100,
        "compact_hydrological_v2_metadata",
        "analysis_outputs/ph_v2_metadata_rerun_v1/gru100_v2_predictions_test.parquet",
        main_role="temporal GRU model",
    ),
    ModelSpec(
        "L100_B",
        "LSTM",
        100,
        "full_metadata",
        "analysis_outputs/ph_sequence100_lstm_metadata_v1/prediction_test_L100_B_lstm_with_metadata.parquet",
        main_role="LSTM sensitivity model",
    ),
    ModelSpec(
        "LSTM100_V2",
        "LSTM",
        100,
        "compact_hydrological_v2_metadata",
        "analysis_outputs/ph_v2_metadata_rerun_v1/optional_lstm100_v2_predictions_test.parquet",
        include=False,
        expected=False,
        main_role="optional LSTM sensitivity model",
        exclusion_reason="Placeholder skipped file from V2 run; no y_true/y_pred columns were generated.",
    ),
]


def log(message: str) -> None:
    print(message, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")


def parquet_info(path: Path) -> tuple[int, list[str], dict[str, str]]:
    pf = pq.ParquetFile(path)
    columns = pf.schema_arrow.names
    dtypes = {field.name: str(field.type) for field in pf.schema_arrow}
    return pf.metadata.num_rows, columns, dtypes


def first_present(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def required_columns(columns: list[str], spec: ModelSpec) -> dict[str, str | None]:
    return {
        "site_id": first_present(columns, ["site_id", "site", "site_no"]),
        "timestamp": first_present(columns, ["timestamp", "target_timestamp", "datetime"]),
        "y_true": first_present(columns, ["00400", "y_true", "observed", "observed_pH"]),
        "y_pred": first_present(columns, ["y_pred", "prediction", "predicted", "predicted_pH"]),
        "residual": first_present(columns, ["residual", "residual_y_true_minus_pred"]),
        "abs_residual": first_present(columns, ["abs_residual"]),
        "in_ph_event": first_present(columns, ["in_ph_event"]),
        "ph_event_label": first_present(columns, ["ph_event_label"]),
        "event_state": first_present(columns, ["event_state"]),
        "row_filter": spec.row_filter_col if spec.row_filter_col in columns else None,
    }


def count_filtered_rows(path: Path, cols: list[str], spec: ModelSpec, total_rows: int) -> int:
    if not spec.row_filter_col or spec.row_filter_col not in cols:
        return total_rows
    series = pd.read_parquet(path, columns=[spec.row_filter_col])[spec.row_filter_col]
    return int((series.astype(str) == str(spec.row_filter_value)).sum())


def inventory_prediction_files() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    included: list[dict[str, Any]] = []
    inventory_cols = [
        "model_id",
        "model_type",
        "site_count",
        "metadata_set",
        "prediction_file",
        "n_rows_or_windows",
        "columns",
        "has_site_id",
        "has_timestamp",
        "has_y_true",
        "has_y_pred",
        "has_event_labels",
        "include_in_analysis",
        "exclusion_reason",
    ]
    included_cols = [
        "model_id",
        "model_type",
        "site_count",
        "metadata_set",
        "prediction_file",
        "n_rows_or_windows",
        "main_role",
    ]
    missing_cols = ["model_id", "prediction_file", "reason"]

    for spec in MODEL_SPECS:
        path = ROOT / spec.prediction_file
        if not path.exists():
            reason = "Prediction file missing."
            rows.append(
                {
                    "model_id": spec.model_id,
                    "model_type": spec.model_type,
                    "site_count": spec.site_count,
                    "metadata_set": spec.metadata_set,
                    "prediction_file": spec.prediction_file,
                    "n_rows_or_windows": 0,
                    "columns": "",
                    "has_site_id": False,
                    "has_timestamp": False,
                    "has_y_true": False,
                    "has_y_pred": False,
                    "has_event_labels": False,
                    "include_in_analysis": False,
                    "exclusion_reason": reason,
                }
            )
            if spec.expected:
                missing.append(
                    {
                        "model_id": spec.model_id,
                        "prediction_file": spec.prediction_file,
                        "reason": reason,
                    }
                )
            continue

        total_rows, columns, _ = parquet_info(path)
        mapping = required_columns(columns, spec)
        n_rows = count_filtered_rows(path, columns, spec, total_rows)
        has_required = all(mapping[key] is not None for key in ["site_id", "timestamp", "y_true", "y_pred"])
        has_event_labels = any(mapping[key] is not None for key in ["in_ph_event", "ph_event_label", "event_state"])
        include = bool(spec.include and has_required and has_event_labels and n_rows > 0)
        reasons = []
        if spec.exclusion_reason:
            reasons.append(spec.exclusion_reason)
        if not has_required:
            reasons.append("Missing one or more required prediction columns: site_id/timestamp/y_true/y_pred.")
        if not has_event_labels:
            reasons.append("Missing regulatory event label columns.")
        if n_rows == 0:
            reasons.append("No rows after model filter.")
        if not spec.include and not reasons:
            reasons.append("Marked as non-core model.")
        reason = " ".join(reasons)

        rows.append(
            {
                "model_id": spec.model_id,
                "model_type": spec.model_type,
                "site_count": spec.site_count,
                "metadata_set": spec.metadata_set,
                "prediction_file": spec.prediction_file,
                "n_rows_or_windows": n_rows,
                "columns": "|".join(columns),
                "has_site_id": mapping["site_id"] is not None,
                "has_timestamp": mapping["timestamp"] is not None,
                "has_y_true": mapping["y_true"] is not None,
                "has_y_pred": mapping["y_pred"] is not None,
                "has_event_labels": has_event_labels,
                "include_in_analysis": include,
                "exclusion_reason": "" if include else reason,
            }
        )
        if include:
            included.append(
                {
                    "model_id": spec.model_id,
                    "model_type": spec.model_type,
                    "site_count": spec.site_count,
                    "metadata_set": spec.metadata_set,
                    "prediction_file": spec.prediction_file,
                    "n_rows_or_windows": n_rows,
                    "main_role": spec.main_role,
                }
            )
        elif spec.expected:
            missing.append(
                {
                    "model_id": spec.model_id,
                    "prediction_file": spec.prediction_file,
                    "reason": reason or "Prediction file not included.",
                }
            )

    inventory = pd.DataFrame(rows, columns=inventory_cols)
    included_df = pd.DataFrame(included, columns=included_cols)
    missing_df = pd.DataFrame(missing, columns=missing_cols)
    inventory.to_csv(OUT_DIR / "unified_prediction_file_inventory.csv", index=False)
    included_df.to_csv(OUT_DIR / "models_included.csv", index=False)
    missing_df.to_csv(OUT_DIR / "models_missing_prediction_files.csv", index=False)
    return inventory, included_df, missing_df


def normalise_event_labels(raw: pd.Series, in_event: pd.Series | None = None) -> tuple[pd.Series, pd.Series]:
    label = raw.astype("string").fillna("").str.strip().str.lower()
    result = pd.Series("mixed_or_unknown", index=raw.index, dtype="string")
    event_state = pd.Series(3, index=raw.index, dtype="int8")

    normal_mask = label.isin(["", "none", "normal", "no_event", "not_event", "nan"])
    low_mask = label.str.contains("low", regex=False)
    high_mask = label.str.contains("high", regex=False)

    if in_event is not None:
        in_event_num = pd.to_numeric(in_event, errors="coerce").fillna(0).astype("int8")
        normal_mask = normal_mask | (in_event_num == 0)

    result.loc[normal_mask] = "none"
    event_state.loc[normal_mask] = 0
    result.loc[low_mask] = "pH_exceedance_low"
    event_state.loc[low_mask] = 1
    result.loc[high_mask] = "pH_exceedance_high"
    event_state.loc[high_mask] = 2

    return result, event_state


def event_class_from_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    if "in_ph_event_raw" in df:
        in_event = pd.to_numeric(df["in_ph_event_raw"], errors="coerce").fillna(0).astype("int8")
    else:
        in_event = pd.Series(0, index=df.index, dtype="int8")

    if "ph_event_label_raw" in df:
        label, label_state = normalise_event_labels(df["ph_event_label_raw"], in_event)
    else:
        label = pd.Series("none", index=df.index, dtype="string")
        label_state = pd.Series(0, index=df.index, dtype="int8")

    if "event_state_raw" in df:
        event_state = pd.to_numeric(df["event_state_raw"], errors="coerce")
        event_state = event_state.where(event_state.isin([0, 1, 2, 3]), label_state).fillna(label_state).astype("int8")
    else:
        event_state = label_state

    event_class = pd.Series("mixed_or_unknown", index=df.index, dtype="string")
    event_class.loc[event_state == 0] = "normal"
    event_class.loc[event_state == 1] = "low_pH_event"
    event_class.loc[event_state == 2] = "high_pH_event"
    event_class.loc[event_state == 3] = "mixed_or_unknown"

    label = label.astype("string")
    label.loc[event_state == 0] = "none"
    label.loc[event_state == 1] = "pH_exceedance_low"
    label.loc[event_state == 2] = "pH_exceedance_high"
    label.loc[event_state == 3] = "mixed_or_unknown"

    in_event = (event_state != 0).astype("int8")
    return label, event_state, event_class


def load_standardised_predictions(spec: ModelSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ROOT / spec.prediction_file
    total_rows, columns, dtypes = parquet_info(path)
    mapping = required_columns(columns, spec)
    read_cols = {
        mapping["site_id"],
        mapping["timestamp"],
        mapping["y_true"],
        mapping["y_pred"],
        mapping["residual"],
        mapping["abs_residual"],
        mapping["in_ph_event"],
        mapping["ph_event_label"],
        mapping["event_state"],
        mapping["row_filter"],
    }
    read_cols = [col for col in read_cols if col is not None]
    raw = pd.read_parquet(path, columns=read_cols)
    if spec.row_filter_col:
        raw = raw.loc[raw[spec.row_filter_col].astype(str) == str(spec.row_filter_value)].copy()

    rename_map = {
        mapping["site_id"]: "site_id",
        mapping["timestamp"]: "timestamp",
        mapping["y_true"]: "y_true",
        mapping["y_pred"]: "y_pred",
    }
    if mapping["in_ph_event"]:
        rename_map[mapping["in_ph_event"]] = "in_ph_event_raw"
    if mapping["ph_event_label"]:
        rename_map[mapping["ph_event_label"]] = "ph_event_label_raw"
    if mapping["event_state"]:
        rename_map[mapping["event_state"]] = "event_state_raw"
    raw = raw.rename(columns=rename_map)

    df = pd.DataFrame(index=raw.index)
    df["model_id"] = spec.model_id
    df["model_type"] = spec.model_type
    df["site_count"] = np.int16(spec.site_count)
    df["metadata_set"] = spec.metadata_set
    df["site_id"] = raw["site_id"].astype("string")
    df["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    df["y_true"] = pd.to_numeric(raw["y_true"], errors="coerce").astype("float32")
    df["y_pred"] = pd.to_numeric(raw["y_pred"], errors="coerce").astype("float32")
    residual = (df["y_true"].astype("float64") - df["y_pred"].astype("float64")).astype("float32")
    df["residual_y_true_minus_pred"] = residual
    df["prediction_error_y_pred_minus_true"] = (-residual).astype("float32")
    df["abs_residual"] = residual.abs().astype("float32")

    event_input = raw[[col for col in ["in_ph_event_raw", "ph_event_label_raw", "event_state_raw"] if col in raw]].copy()
    label, state, cls = event_class_from_columns(event_input)
    df["in_ph_event"] = (state != 0).astype("int8")
    df["ph_event_label"] = label.astype("string")
    df["event_state"] = state.astype("int8")
    df["event_class"] = cls.astype("string")
    df["is_event"] = df["event_state"] != 0
    df["is_low_pH_event"] = df["event_state"] == 1
    df["is_high_pH_event"] = df["event_state"] == 2

    df = df.dropna(subset=["site_id", "timestamp", "y_true", "y_pred"]).reset_index(drop=True)
    df = df[[field.name for field in UNIFIED_SCHEMA]]

    audit = {
        "model_id": spec.model_id,
        "prediction_file": spec.prediction_file,
        "raw_n_rows": total_rows,
        "unified_n_rows": len(df),
        "raw_columns": "|".join(columns),
        "site_id_col": mapping["site_id"],
        "timestamp_col": mapping["timestamp"],
        "y_true_col": mapping["y_true"],
        "y_pred_col": mapping["y_pred"],
        "raw_residual_col": mapping["residual"],
        "raw_abs_residual_col": mapping["abs_residual"],
        "in_ph_event_col": mapping["in_ph_event"],
        "ph_event_label_col": mapping["ph_event_label"],
        "event_state_col": mapping["event_state"],
        "n_sites": int(df["site_id"].nunique()),
        "n_normal": int((df["event_class"] == "normal").sum()),
        "n_low_pH_event": int((df["event_class"] == "low_pH_event").sum()),
        "n_high_pH_event": int((df["event_class"] == "high_pH_event").sum()),
        "n_mixed_or_unknown": int((df["event_class"] == "mixed_or_unknown").sum()),
        "raw_dtypes_json": json.dumps(dtypes, sort_keys=True),
    }
    return df, audit


def as_arrow_table(df: pd.DataFrame) -> pa.Table:
    table = pa.Table.from_pandas(df, schema=UNIFIED_SCHEMA, preserve_index=False)
    return table.cast(UNIFIED_SCHEMA, safe=False)


def r2_score_manual(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, str]:
    n = len(y_true)
    if n < 3:
        return math.nan, "too_few_samples"
    yt = y_true.to_numpy(dtype="float64")
    yp = y_pred.to_numpy(dtype="float64")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 1e-12:
        return math.nan, "near_zero_target_variance"
    warning = ""
    if n < 30:
        warning = "low_sample_count"
    return 1.0 - ss_res / ss_tot, warning


def metric_for_subset(df: pd.DataFrame, mask: pd.Series, event_class: str) -> dict[str, Any]:
    sub = df.loc[mask, ["site_id", "y_true", "y_pred", "prediction_error_y_pred_minus_true", "abs_residual"]]
    n = len(sub)
    if n == 0:
        return {
            "event_class": event_class,
            "mixed_included_in_all_pH_event": event_class == "all_pH_event",
            "n": 0,
            "n_sites": 0,
            "rmse": math.nan,
            "mae": math.nan,
            "r2": math.nan,
            "bias": math.nan,
            "median_error": math.nan,
            "mean_abs_residual": math.nan,
            "median_abs_residual": math.nan,
            "p90_abs_residual": math.nan,
            "p95_abs_residual": math.nan,
            "r2_warning": "no_rows",
        }

    err = sub["prediction_error_y_pred_minus_true"].astype("float64")
    abs_resid = sub["abs_residual"].astype("float64")
    r2, warning = r2_score_manual(sub["y_true"], sub["y_pred"])
    return {
        "event_class": event_class,
        "mixed_included_in_all_pH_event": event_class == "all_pH_event",
        "n": n,
        "n_sites": int(sub["site_id"].nunique()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(abs_resid.mean()),
        "r2": float(r2) if not math.isnan(r2) else math.nan,
        "bias": float(err.mean()),
        "median_error": float(err.median()),
        "mean_abs_residual": float(abs_resid.mean()),
        "median_abs_residual": float(abs_resid.median()),
        "p90_abs_residual": float(abs_resid.quantile(0.90)),
        "p95_abs_residual": float(abs_resid.quantile(0.95)),
        "r2_warning": warning,
    }


def compute_metrics(df: pd.DataFrame, spec: ModelSpec) -> list[dict[str, Any]]:
    masks = {
        "overall": pd.Series(True, index=df.index),
        "normal": df["event_class"] == "normal",
        "low_pH_event": df["event_class"] == "low_pH_event",
        "high_pH_event": df["event_class"] == "high_pH_event",
        "mixed_or_unknown": df["event_class"] == "mixed_or_unknown",
        "all_pH_event": df["event_class"].isin(["low_pH_event", "high_pH_event", "mixed_or_unknown"]),
    }
    rows = []
    for event_class in METRIC_CLASSES:
        row = metric_for_subset(df, masks[event_class], event_class)
        row.update(model_fields(spec))
        rows.append(row)
    return rows


def compute_distribution(df: pd.DataFrame, spec: ModelSpec) -> list[dict[str, Any]]:
    rows = []
    masks = {
        "normal": df["event_class"] == "normal",
        "low_pH_event": df["event_class"] == "low_pH_event",
        "high_pH_event": df["event_class"] == "high_pH_event",
        "mixed_or_unknown": df["event_class"] == "mixed_or_unknown",
        "all_pH_event": df["event_class"].isin(["low_pH_event", "high_pH_event", "mixed_or_unknown"]),
    }
    for event_class, mask in masks.items():
        sub = df.loc[mask, ["residual_y_true_minus_pred", "abs_residual"]]
        n = len(sub)
        base = {
            **model_fields(spec),
            "event_class": event_class,
            "n": n,
        }
        if n == 0:
            base.update(
                {
                    "mean_residual_y_true_minus_pred": math.nan,
                    "median_residual_y_true_minus_pred": math.nan,
                    "std_residual_y_true_minus_pred": math.nan,
                    "mean_abs_residual": math.nan,
                    "median_abs_residual": math.nan,
                    "p75_abs_residual": math.nan,
                    "p90_abs_residual": math.nan,
                    "p95_abs_residual": math.nan,
                    "p99_abs_residual": math.nan,
                }
            )
        else:
            residual = sub["residual_y_true_minus_pred"].astype("float64")
            abs_res = sub["abs_residual"].astype("float64")
            base.update(
                {
                    "mean_residual_y_true_minus_pred": float(residual.mean()),
                    "median_residual_y_true_minus_pred": float(residual.median()),
                    "std_residual_y_true_minus_pred": float(residual.std(ddof=1)) if n > 1 else math.nan,
                    "mean_abs_residual": float(abs_res.mean()),
                    "median_abs_residual": float(abs_res.median()),
                    "p75_abs_residual": float(abs_res.quantile(0.75)),
                    "p90_abs_residual": float(abs_res.quantile(0.90)),
                    "p95_abs_residual": float(abs_res.quantile(0.95)),
                    "p99_abs_residual": float(abs_res.quantile(0.99)),
                }
            )
        rows.append(base)
    return rows


def compute_enrichment_for_score(
    df: pd.DataFrame,
    spec: ModelSpec,
    score_col: str,
    threshold_type: str,
) -> list[dict[str, Any]]:
    rows = []
    valid = df[score_col].replace([np.inf, -np.inf], np.nan).notna()
    n_test = int(valid.sum())
    group_masks = {
        "all_pH_event": df["event_class"].isin(["low_pH_event", "high_pH_event", "mixed_or_unknown"]),
        "low_pH_event": df["event_class"] == "low_pH_event",
        "high_pH_event": df["event_class"] == "high_pH_event",
    }
    for top_fraction in TOP_FRACTIONS:
        if n_test == 0:
            top_mask = pd.Series(False, index=df.index)
            threshold = math.nan
        else:
            threshold = float(df.loc[valid, score_col].quantile(1 - top_fraction))
            top_mask = valid & (df[score_col] >= threshold)
        n_top = int(top_mask.sum())
        for event_group, group_mask in group_masks.items():
            group_mask = group_mask & valid
            n_group = int(group_mask.sum())
            n_top_group = int((top_mask & group_mask).sum())
            base_group_rate = n_group / n_test if n_test else math.nan
            top_group_rate = n_top_group / n_top if n_top else math.nan
            enrichment = top_group_rate / base_group_rate if base_group_rate and not math.isnan(base_group_rate) else math.nan
            rows.append(
                {
                    **model_fields(spec),
                    "residual_threshold_type": threshold_type,
                    "event_group": event_group,
                    "top_fraction": top_fraction,
                    "threshold": threshold,
                    "n_test": n_test,
                    "n_group": n_group,
                    "base_group_rate": base_group_rate,
                    "n_top": n_top,
                    "n_top_group": n_top_group,
                    "top_group_rate": top_group_rate,
                    "enrichment": enrichment,
                }
            )
    return rows


def compute_enrichment(df: pd.DataFrame, spec: ModelSpec) -> list[dict[str, Any]]:
    rows = compute_enrichment_for_score(df, spec, "abs_residual", "global_abs_residual")
    try:
        site_median = df.groupby("site_id", observed=True)["abs_residual"].transform("median")
        site_median = site_median.mask(site_median <= 0)
        df = df.copy()
        df["site_normalised_abs_residual"] = df["abs_residual"] / site_median
        rows.extend(
            compute_enrichment_for_score(
                df,
                spec,
                "site_normalised_abs_residual",
                "site_normalised_abs_residual",
            )
        )
    except Exception as exc:
        log(f"Site-normalised enrichment skipped for {spec.model_id}: {exc}")
    return rows


def infer_gap_threshold(df: pd.DataFrame, spec: ModelSpec) -> pd.Timedelta:
    if spec.model_type in {"GRU", "LSTM"}:
        return pd.Timedelta(hours=3)
    try:
        sample = df[["site_id", "timestamp"]].head(min(len(df), 500_000)).copy()
        sample = sample.sort_values(["site_id", "timestamp"])
        diffs = sample.groupby("site_id", observed=True)["timestamp"].diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        if len(diffs) == 0:
            return pd.Timedelta(minutes=30)
        median_diff = diffs.median()
        threshold = median_diff * 3
        threshold = max(threshold, pd.Timedelta(minutes=30))
        threshold = min(threshold, pd.Timedelta(hours=3))
        return threshold
    except Exception:
        return pd.Timedelta(minutes=30)


def base_event_windows(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    event_rows = df.loc[df["event_class"].isin(["low_pH_event", "high_pH_event", "mixed_or_unknown"])].copy()
    if event_rows.empty:
        return pd.DataFrame()
    gap = infer_gap_threshold(df, spec)
    event_rows = event_rows.sort_values(["site_id", "event_class", "timestamp"]).reset_index(drop=False)
    prev_site = event_rows["site_id"].shift()
    prev_class = event_rows["event_class"].shift()
    prev_time = event_rows["timestamp"].shift()
    new_window = (
        (event_rows["site_id"] != prev_site)
        | (event_rows["event_class"] != prev_class)
        | ((event_rows["timestamp"] - prev_time) > gap)
    )
    new_window = new_window.fillna(True)
    event_rows["event_window_id"] = np.cumsum(np.asarray(new_window, dtype=bool)).astype("int64")
    event_rows["gap_threshold_minutes"] = gap.total_seconds() / 60
    return event_rows


def compute_event_hit_rate(df: pd.DataFrame, spec: ModelSpec) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    rows = []
    event_rows = base_event_windows(df, spec)
    if event_rows.empty:
        for group in ["all_pH_event", "low_pH_event", "high_pH_event"]:
            for top_fraction in TOP_FRACTIONS:
                rows.append(
                    {
                        **model_fields(spec),
                        "event_group": group,
                        "top_fraction": top_fraction,
                        "n_event_windows": 0,
                        "n_hit_event_windows": 0,
                        "event_level_hit_rate": math.nan,
                        "method": "approx_contiguous_rows_by_site_label",
                        "gap_threshold_minutes": math.nan,
                    }
                )
        return rows, event_rows

    window_summary = (
        event_rows.groupby("event_window_id", observed=True)
        .agg(
            site_id=("site_id", "first"),
            event_class=("event_class", "first"),
            event_start=("timestamp", "min"),
            event_end=("timestamp", "max"),
            n_event_rows_or_windows=("timestamp", "size"),
            mean_abs_residual=("abs_residual", "mean"),
            max_abs_residual=("abs_residual", "max"),
            gap_threshold_minutes=("gap_threshold_minutes", "first"),
        )
        .reset_index()
    )

    for top_fraction in TOP_FRACTIONS:
        threshold = float(df["abs_residual"].quantile(1 - top_fraction))
        top_indexes = set(df.index[df["abs_residual"] >= threshold])
        hit_ids = set(event_rows.loc[event_rows["index"].isin(top_indexes), "event_window_id"].astype(int))
        for group, classes in {
            "all_pH_event": ["low_pH_event", "high_pH_event", "mixed_or_unknown"],
            "low_pH_event": ["low_pH_event"],
            "high_pH_event": ["high_pH_event"],
        }.items():
            group_windows = window_summary.loc[window_summary["event_class"].isin(classes)]
            n_events = len(group_windows)
            n_hits = int(group_windows["event_window_id"].isin(hit_ids).sum())
            rows.append(
                {
                    **model_fields(spec),
                    "event_group": group,
                    "top_fraction": top_fraction,
                    "n_event_windows": n_events,
                    "n_hit_event_windows": n_hits,
                    "event_level_hit_rate": n_hits / n_events if n_events else math.nan,
                    "method": "approx_contiguous_rows_by_site_label",
                    "gap_threshold_minutes": float(window_summary["gap_threshold_minutes"].iloc[0]),
                }
            )
    window_summary.insert(0, "model_id", spec.model_id)
    return rows, window_summary


def model_fields(spec: ModelSpec) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "model_type": spec.model_type,
        "site_count": spec.site_count,
        "metadata_set": spec.metadata_set,
    }


def sample_for_plots(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    if spec.model_id not in {"M0_reference_nonevent", "HGB200_V2", "GRU100_V2"}:
        return pd.DataFrame()
    pieces = []
    for cls, n in [("normal", 8000), ("low_pH_event", 8000), ("high_pH_event", 8000), ("mixed_or_unknown", 3000)]:
        sub = df.loc[df["event_class"] == cls]
        if sub.empty:
            continue
        pieces.append(sub.sample(n=min(n, len(sub)), random_state=42))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def create_bias_table(metrics: pd.DataFrame) -> pd.DataFrame:
    bias = metrics.pivot_table(
        index=["model_id", "model_type", "site_count", "metadata_set"],
        columns="event_class",
        values="bias",
        aggfunc="first",
    ).reset_index()
    bias.columns.name = None
    for col in ["normal", "low_pH_event", "high_pH_event", "all_pH_event"]:
        if col not in bias.columns:
            bias[col] = np.nan
    bias = bias.rename(
        columns={
            "normal": "normal_bias",
            "low_pH_event": "low_pH_event_bias",
            "high_pH_event": "high_pH_event_bias",
            "all_pH_event": "all_event_bias",
        }
    )

    def low_interp(value: float) -> str:
        if pd.isna(value):
            return "No low-pH event rows available."
        if value > 0:
            return "Model predicts too high during low-pH events; low-pH drops are under-estimated."
        if value < 0:
            return "Model predicts too low during low-pH events."
        return "No mean bias during low-pH events."

    def high_interp(value: float) -> str:
        if pd.isna(value):
            return "No high-pH event rows available."
        if value < 0:
            return "Model predicts too low during high-pH events; high-pH peaks are under-estimated."
        if value > 0:
            return "Model predicts too high during high-pH events."
        return "No mean bias during high-pH events."

    bias["low_bias_interpretation"] = bias["low_pH_event_bias"].apply(low_interp)
    bias["high_bias_interpretation"] = bias["high_pH_event_bias"].apply(high_interp)
    return order_models(bias)


def order_models(df: pd.DataFrame) -> pd.DataFrame:
    if "model_id" not in df.columns:
        return df
    order = {model: i for i, model in enumerate(REPORT_MODEL_ORDER)}
    return df.assign(_order=df["model_id"].map(order).fillna(999)).sort_values(["_order", "model_id"]).drop(columns="_order")


def create_summary_tables(
    metrics: pd.DataFrame,
    enrichment: pd.DataFrame,
    hit_rate: pd.DataFrame,
    included: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_pivot = metrics.pivot_table(
        index=["model_id", "model_type", "site_count", "metadata_set"],
        columns="event_class",
        values=["rmse", "mae", "r2", "bias"],
        aggfunc="first",
    )
    metric_pivot.columns = [f"{event}_{metric}" for metric, event in metric_pivot.columns]
    metric_pivot = metric_pivot.reset_index()

    top5 = enrichment.loc[
        (enrichment["residual_threshold_type"] == "global_abs_residual")
        & np.isclose(enrichment["top_fraction"], 0.05)
    ].copy()
    top5_pivot = top5.pivot_table(
        index=["model_id"],
        columns="event_group",
        values="enrichment",
        aggfunc="first",
    ).reset_index()
    top5_pivot = top5_pivot.rename(
        columns={
            "all_pH_event": "top5_all_event_enrichment",
            "low_pH_event": "top5_low_pH_enrichment",
            "high_pH_event": "top5_high_pH_enrichment",
        }
    )

    hit5 = hit_rate.loc[np.isclose(hit_rate["top_fraction"], 0.05)].copy()
    hit5_pivot = hit5.pivot_table(
        index=["model_id"],
        columns="event_group",
        values="event_level_hit_rate",
        aggfunc="first",
    ).reset_index()
    hit5_pivot = hit5_pivot.rename(
        columns={
            "all_pH_event": "event_level_top5_hit_rate",
            "low_pH_event": "low_event_level_top5_hit_rate",
            "high_pH_event": "high_event_level_top5_hit_rate",
        }
    )

    role = included[["model_id", "main_role"]].drop_duplicates()
    summary = metric_pivot.merge(top5_pivot, on="model_id", how="left").merge(hit5_pivot, on="model_id", how="left").merge(role, on="model_id", how="left")

    rename = {
        "overall_rmse": "overall_rmse",
        "overall_r2": "overall_r2",
        "normal_rmse": "normal_rmse",
        "low_pH_event_rmse": "low_pH_event_rmse",
        "high_pH_event_rmse": "high_pH_event_rmse",
        "low_pH_event_bias": "low_pH_event_bias",
        "high_pH_event_bias": "high_pH_event_bias",
    }
    for col in rename.values():
        if col not in summary.columns:
            summary[col] = np.nan

    def interpretation(row: pd.Series) -> str:
        bits = []
        if pd.notna(row.get("low_pH_event_bias")) and row["low_pH_event_bias"] > 0:
            bits.append("low-pH drops under-predicted")
        elif pd.notna(row.get("low_pH_event_bias")) and row["low_pH_event_bias"] < 0:
            bits.append("low-pH periods predicted too low")
        if pd.notna(row.get("high_pH_event_bias")) and row["high_pH_event_bias"] < 0:
            bits.append("high-pH peaks under-estimated")
        elif pd.notna(row.get("high_pH_event_bias")) and row["high_pH_event_bias"] > 0:
            bits.append("high-pH periods predicted too high")
        if pd.notna(row.get("top5_low_pH_enrichment")) and pd.notna(row.get("top5_high_pH_enrichment")):
            if row["top5_low_pH_enrichment"] > row["top5_high_pH_enrichment"]:
                bits.append("stronger residual alignment for low-pH events")
            elif row["top5_high_pH_enrichment"] > row["top5_low_pH_enrichment"]:
                bits.append("stronger residual alignment for high-pH events")
        return "; ".join(bits) if bits else "Event-class behaviour is weak or data-limited."

    summary["key_interpretation"] = summary.apply(interpretation, axis=1)
    summary = order_models(summary)
    summary.to_csv(OUT_DIR / "low_high_model_summary_table.csv", index=False)

    dissertation_cols = [
        "model_id",
        "site_count",
        "model_type",
        "metadata_set",
        "overall_rmse",
        "overall_r2",
        "normal_rmse",
        "low_pH_event_rmse",
        "high_pH_event_rmse",
        "low_pH_event_bias",
        "high_pH_event_bias",
        "top5_all_event_enrichment",
        "top5_low_pH_enrichment",
        "top5_high_pH_enrichment",
        "event_level_top5_hit_rate",
        "low_event_level_top5_hit_rate",
        "high_event_level_top5_hit_rate",
        "main_role",
        "key_interpretation",
    ]
    for col in dissertation_cols:
        if col not in summary.columns:
            summary[col] = np.nan
    dissertation = summary[dissertation_cols]
    dissertation.to_csv(OUT_DIR / "low_high_dissertation_ready_table.csv", index=False)
    return summary, dissertation


def academic_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def short_label(model_id: str) -> str:
    mapping = {
        "M0_reference_nonevent": "M0 non-event",
        "M1_no_metadata": "HGB50 no meta",
        "M2_with_metadata": "HGB50 full meta",
        "HGB50_V2": "HGB50 V2",
        "M200_A_no_metadata": "HGB200 no meta",
        "M200_B_with_metadata": "HGB200 full meta",
        "HGB200_V2": "HGB200 V2",
        "G100_A": "GRU100 no meta",
        "G100_B": "GRU100 full meta",
        "GRU100_V2": "GRU100 V2",
        "L100_B": "LSTM100 full meta",
    }
    return mapping.get(model_id, model_id)


def plot_metric_bars(metrics: pd.DataFrame, metric: str, output_name: str, ylabel: str) -> None:
    plot_df = metrics.loc[metrics["event_class"].isin(["normal", "low_pH_event", "high_pH_event"])].copy()
    plot_df = order_models(plot_df)
    plot_df["model_label"] = plot_df["model_id"].map(short_label)
    plt.figure(figsize=(12, 5))
    sns.barplot(data=plot_df, x="model_label", y=metric, hue="event_class", palette=["#6b7280", "#2f80ed", "#d94841"])
    plt.xlabel("")
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="Event class")
    plt.tight_layout()
    plt.savefig(FIG_DIR / output_name)
    plt.close()


def plot_bias(bias: pd.DataFrame) -> None:
    cols = ["low_pH_event_bias", "high_pH_event_bias"]
    plot_df = order_models(bias).melt(
        id_vars=["model_id"],
        value_vars=cols,
        var_name="event_class",
        value_name="bias",
    )
    plot_df["event_class"] = plot_df["event_class"].map(
        {"low_pH_event_bias": "low_pH_event", "high_pH_event_bias": "high_pH_event"}
    )
    plot_df["model_label"] = plot_df["model_id"].map(short_label)
    plt.figure(figsize=(12, 5))
    sns.barplot(data=plot_df, x="model_label", y="bias", hue="event_class", palette=["#2f80ed", "#d94841"])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("")
    plt.ylabel("Bias = mean(y_pred - y_true)")
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="Event class")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "low_high_bias_by_model.png")
    plt.close()


def plot_enrichment(enrichment: pd.DataFrame) -> None:
    plot_df = enrichment.loc[
        (enrichment["residual_threshold_type"] == "global_abs_residual")
        & np.isclose(enrichment["top_fraction"], 0.05)
        & enrichment["event_group"].isin(["low_pH_event", "high_pH_event"])
    ].copy()
    plot_df = order_models(plot_df)
    plot_df["model_label"] = plot_df["model_id"].map(short_label)
    plt.figure(figsize=(12, 5))
    sns.barplot(data=plot_df, x="model_label", y="enrichment", hue="event_group", palette=["#2f80ed", "#d94841"])
    plt.axhline(1, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("")
    plt.ylabel("Top 5% residual enrichment")
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="Event group")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "low_high_enrichment_by_model.png")
    plt.close()


def plot_hit_rate(hit_rate: pd.DataFrame) -> None:
    plot_df = hit_rate.loc[
        np.isclose(hit_rate["top_fraction"], 0.05)
        & hit_rate["event_group"].isin(["low_pH_event", "high_pH_event"])
    ].copy()
    plot_df = order_models(plot_df)
    plot_df["model_label"] = plot_df["model_id"].map(short_label)
    plt.figure(figsize=(12, 5))
    sns.barplot(data=plot_df, x="model_label", y="event_level_hit_rate", hue="event_group", palette=["#2f80ed", "#d94841"])
    plt.xlabel("")
    plt.ylabel("Event-level top 5% hit rate")
    plt.ylim(0, max(0.05, min(1.0, plot_df["event_level_hit_rate"].max(skipna=True) * 1.15)))
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="Event group")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "low_high_event_hit_rate_by_model.png")
    plt.close()


def plot_residual_distribution(sample: pd.DataFrame) -> None:
    plot_df = sample.loc[
        sample["model_id"].isin(["M0_reference_nonevent", "HGB200_V2", "GRU100_V2"])
        & sample["event_class"].isin(["normal", "low_pH_event", "high_pH_event"])
    ].copy()
    if plot_df.empty:
        return
    plot_df["model_label"] = plot_df["model_id"].map(short_label)
    plt.figure(figsize=(11, 5))
    sns.boxplot(
        data=plot_df,
        x="model_label",
        y="abs_residual",
        hue="event_class",
        palette=["#6b7280", "#2f80ed", "#d94841"],
        showfliers=False,
    )
    plt.xlabel("")
    plt.ylabel("Absolute residual")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Event class")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "residual_distribution_by_event_class.png")
    plt.close()


def plot_scatter(sample: pd.DataFrame) -> None:
    plot_df = sample.loc[
        (sample["model_id"] == "HGB200_V2")
        & sample["event_class"].isin(["normal", "low_pH_event", "high_pH_event"])
    ].copy()
    if plot_df.empty:
        plot_df = sample.loc[
            sample["event_class"].isin(["normal", "low_pH_event", "high_pH_event"])
        ].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sample(n=min(35_000, len(plot_df)), random_state=7)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        data=plot_df,
        x="y_true",
        y="y_pred",
        hue="event_class",
        palette={"normal": "#6b7280", "low_pH_event": "#2f80ed", "high_pH_event": "#d94841"},
        s=8,
        alpha=0.35,
        linewidth=0,
    )
    min_v = float(min(plot_df["y_true"].min(), plot_df["y_pred"].min()))
    max_v = float(max(plot_df["y_true"].max(), plot_df["y_pred"].max()))
    plt.plot([min_v, max_v], [min_v, max_v], color="black", linewidth=0.9, linestyle="--")
    plt.xlabel("Observed processed pH")
    plt.ylabel("Predicted processed pH")
    plt.legend(title="Event class", markerscale=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "observed_vs_predicted_low_high_scatter.png")
    plt.close()


def create_standard_figures(
    metrics: pd.DataFrame,
    bias: pd.DataFrame,
    enrichment: pd.DataFrame,
    hit_rate: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    academic_style()
    plot_metric_bars(metrics, "rmse", "low_high_rmse_by_model.png", "RMSE")
    plot_metric_bars(metrics, "mae", "low_high_mae_by_model.png", "MAE")
    plot_bias(bias)
    plot_enrichment(enrichment)
    plot_hit_rate(hit_rate)
    plot_residual_distribution(sample)
    plot_scatter(sample)


def load_context_for_case(site_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    candidates = [
        ROOT / "analysis_outputs/ph_hgb_processed200_metadata_v1/processed200_model_ready.parquet",
        ROOT / "analysis_outputs/ph_processed100_step4_0_4_1_v1/processed100_with_event_state.parquet",
    ]
    cols = ["site_id", "timestamp", "00060", "00065"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            dataset = ds.dataset(path, format="parquet")
            available = [col for col in cols if col in dataset.schema.names]
            if len(available) < 3:
                continue
            table = dataset.to_table(
                columns=available,
                filter=(
                    (ds.field("site_id") == site_id)
                    & (ds.field("timestamp") >= pa.scalar(start.to_pydatetime()))
                    & (ds.field("timestamp") <= pa.scalar(end.to_pydatetime()))
                ),
            )
            context = table.to_pandas()
            if not context.empty:
                context["site_id"] = context["site_id"].astype("string")
                context["timestamp"] = pd.to_datetime(context["timestamp"])
                return context
        except Exception as exc:
            log(f"Context lookup skipped for {path.name}: {exc}")
    return pd.DataFrame()


def select_case_studies(case_df: pd.DataFrame, case_windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if case_df.empty or case_windows.empty:
        return pd.DataFrame(), pd.DataFrame()
    selected = []
    windows = case_windows.copy()
    windows = windows.loc[windows["model_id"] == "HGB200_V2"].copy()
    if windows.empty:
        preferred_model = case_df["model_id"].iloc[0]
        windows = case_windows.loc[case_windows["model_id"] == preferred_model].copy()
    for event_class, case_id, min_rows in [
        ("low_pH_event", "case_low_pH", 12),
        ("high_pH_event", "case_high_pH", 6),
    ]:
        candidates = windows.loc[(windows["event_class"] == event_class) & (windows["n_event_rows_or_windows"] >= min_rows)].copy()
        if candidates.empty:
            candidates = windows.loc[windows["event_class"] == event_class].copy()
        if candidates.empty:
            continue
        q90 = candidates["mean_abs_residual"].quantile(0.90)
        candidates["selection_distance"] = (candidates["mean_abs_residual"] - q90).abs()
        choice = candidates.sort_values(["selection_distance", "n_event_rows_or_windows"], ascending=[True, False]).iloc[0]
        selected.append(
            {
                "case_id": case_id,
                "site_id": str(choice["site_id"]),
                "event_type": event_class,
                "event_start": choice["event_start"],
                "event_end": choice["event_end"],
                "model_id": choice["model_id"],
                "selection_reason": "Selected near the upper-decile mean residual among events with enough rows, avoiding the single most extreme event.",
                "n_event_rows_or_windows": int(choice["n_event_rows_or_windows"]),
                "max_abs_residual": float(choice["max_abs_residual"]),
                "mean_abs_residual": float(choice["mean_abs_residual"]),
                "hydrological_context_available": False,
            }
        )
    selection = pd.DataFrame(selected)
    if selection.empty:
        return selection, windows

    for idx, row in selection.iterrows():
        buffer = pd.Timedelta(hours=24 if row["model_id"] == "HGB200_V2" else 72)
        context = load_context_for_case(str(row["site_id"]), pd.Timestamp(row["event_start"]) - buffer, pd.Timestamp(row["event_end"]) + buffer)
        has_context = False
        if not context.empty:
            for col in ["00060", "00065"]:
                if col in context.columns and context[col].notna().any():
                    has_context = True
        selection.loc[idx, "hydrological_context_available"] = has_context
    windows["selected_for_case_study"] = windows["event_window_id"].isin(
        selection.merge(windows, on=["model_id", "site_id", "event_start", "event_end"], how="left")["event_window_id"].dropna()
    )
    return selection, windows


def plot_case(case: pd.Series, predictions: pd.DataFrame, output_name: str) -> None:
    site_id = str(case["site_id"])
    event_start = pd.Timestamp(case["event_start"])
    event_end = pd.Timestamp(case["event_end"])
    model_id = case["model_id"]
    buffer = pd.Timedelta(hours=24 if model_id == "HGB200_V2" else 72)
    start = event_start - buffer
    end = event_end + buffer
    pred = predictions.loc[
        (predictions["model_id"] == model_id)
        & (predictions["site_id"].astype(str) == site_id)
        & (predictions["timestamp"] >= start)
        & (predictions["timestamp"] <= end)
    ].copy()
    if pred.empty:
        return
    pred = pred.sort_values("timestamp")
    context = load_context_for_case(site_id, start, end)
    if not context.empty:
        pred = pred.merge(context, on=["site_id", "timestamp"], how="left")

    context_col = None
    for col in ["00060", "00065"]:
        if col in pred.columns and pred[col].notna().any():
            context_col = col
            break

    nrows = 3 if context_col else 2
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.2, 1.0][:nrows]})
    axes = np.atleast_1d(axes)

    axes[0].plot(pred["timestamp"], pred["y_true"], color="#111827", linewidth=1.2, label="Observed pH")
    axes[0].plot(pred["timestamp"], pred["y_pred"], color="#d94841", linewidth=1.0, label=f"Predicted pH ({short_label(model_id)})")
    axes[0].axvspan(event_start, event_end, color="#2f80ed" if case["event_type"] == "low_pH_event" else "#d94841", alpha=0.18)
    axes[0].set_ylabel("Processed pH")
    axes[0].legend(loc="best")
    axes[0].set_title(f"{case['event_type'].replace('_', ' ')} case study: site {site_id}")

    axes[1].plot(pred["timestamp"], pred["prediction_error_y_pred_minus_true"], color="#374151", linewidth=1.0)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].axvspan(event_start, event_end, color="#2f80ed" if case["event_type"] == "low_pH_event" else "#d94841", alpha=0.18)
    axes[1].set_ylabel("Prediction error\n(y_pred - y_true)")

    if context_col:
        axes[2].plot(pred["timestamp"], pred[context_col], color="#00897b", linewidth=1.0)
        axes[2].axvspan(event_start, event_end, color="#2f80ed" if case["event_type"] == "low_pH_event" else "#d94841", alpha=0.18)
        axes[2].set_ylabel(f"{context_col}")
    axes[-1].set_xlabel("Timestamp")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / output_name)
    plt.close(fig)


def plot_combined_cases(selection: pd.DataFrame, predictions: pd.DataFrame) -> None:
    if selection.empty:
        return
    selected = selection.head(2)
    fig, axes = plt.subplots(len(selected), 1, figsize=(11, 6.5), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (_, case) in zip(axes, selected.iterrows()):
        site_id = str(case["site_id"])
        event_start = pd.Timestamp(case["event_start"])
        event_end = pd.Timestamp(case["event_end"])
        model_id = case["model_id"]
        buffer = pd.Timedelta(hours=24 if model_id == "HGB200_V2" else 72)
        pred = predictions.loc[
            (predictions["model_id"] == model_id)
            & (predictions["site_id"].astype(str) == site_id)
            & (predictions["timestamp"] >= event_start - buffer)
            & (predictions["timestamp"] <= event_end + buffer)
        ].copy()
        if pred.empty:
            continue
        pred = pred.sort_values("timestamp")
        ax.plot(pred["timestamp"], pred["y_true"], color="#111827", linewidth=1.0, label="Observed")
        ax.plot(pred["timestamp"], pred["y_pred"], color="#d94841", linewidth=0.9, label="Predicted")
        ax.axvspan(event_start, event_end, color="#2f80ed" if case["event_type"] == "low_pH_event" else "#d94841", alpha=0.18)
        ax.set_ylabel("Processed pH")
        ax.set_title(f"{case['event_type'].replace('_', ' ')}: site {site_id}")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Timestamp")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case_study_low_high_combined.png")
    plt.close(fig)


def create_case_studies(all_predictions: pd.DataFrame, event_windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if all_predictions.empty:
        empty = pd.DataFrame()
        empty.to_csv(OUT_DIR / "case_study_site_selection.csv", index=False)
        empty.to_csv(OUT_DIR / "case_study_event_windows.csv", index=False)
        return empty, empty
    selection, windows = select_case_studies(all_predictions, event_windows)
    selection.to_csv(OUT_DIR / "case_study_site_selection.csv", index=False)
    windows.to_csv(OUT_DIR / "case_study_event_windows.csv", index=False)
    if not selection.empty:
        low_cases = selection.loc[selection["event_type"] == "low_pH_event"]
        high_cases = selection.loc[selection["event_type"] == "high_pH_event"]
        if not low_cases.empty:
            plot_case(low_cases.iloc[0], all_predictions, "case_study_low_pH_event.png")
        if not high_cases.empty:
            plot_case(high_cases.iloc[0], all_predictions, "case_study_high_pH_event.png")
        plot_combined_cases(selection, all_predictions)
    return selection, windows


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_report(
    inventory: pd.DataFrame,
    included: pd.DataFrame,
    missing: pd.DataFrame,
    metrics: pd.DataFrame,
    bias: pd.DataFrame,
    enrichment: pd.DataFrame,
    hit_rate: pd.DataFrame,
    dissertation: pd.DataFrame,
    case_selection: pd.DataFrame,
) -> None:
    top5_global = enrichment.loc[
        (enrichment["residual_threshold_type"] == "global_abs_residual")
        & np.isclose(enrichment["top_fraction"], 0.05)
    ].copy()
    low_best = metrics.loc[metrics["event_class"] == "low_pH_event"].sort_values("rmse").head(1)
    high_best = metrics.loc[metrics["event_class"] == "high_pH_event"].sort_values("rmse").head(1)
    low_enrich_best = top5_global.loc[top5_global["event_group"] == "low_pH_event"].sort_values("enrichment", ascending=False).head(1)
    high_enrich_best = top5_global.loc[top5_global["event_group"] == "high_pH_event"].sort_values("enrichment", ascending=False).head(1)
    all_enrich_best = top5_global.loc[top5_global["event_group"] == "all_pH_event"].sort_values("enrichment", ascending=False).head(1)

    metadata_pairs = [
        ("M1_no_metadata", "M2_with_metadata", "50-site HGB full metadata"),
        ("M200_A_no_metadata", "M200_B_with_metadata", "200-site HGB full metadata"),
        ("G100_A", "G100_B", "100-site GRU full metadata"),
    ]
    metadata_lines = []
    diss_idx = dissertation.set_index("model_id")
    for base, meta, label in metadata_pairs:
        if base in diss_idx.index and meta in diss_idx.index:
            b = diss_idx.loc[base]
            m = diss_idx.loc[meta]
            metadata_lines.append(
                f"- {label}: overall RMSE {fmt(b.get('overall_rmse'))} -> {fmt(m.get('overall_rmse'))}; "
                f"top5 low-event enrichment {fmt(b.get('top5_low_pH_enrichment'))} -> {fmt(m.get('top5_low_pH_enrichment'))}; "
                f"top5 high-event enrichment {fmt(b.get('top5_high_pH_enrichment'))} -> {fmt(m.get('top5_high_pH_enrichment'))}."
            )

    model_list = "\n".join(
        f"- {row.model_id}: {row.model_type}, {row.site_count} sites, {row.metadata_set}, {row.main_role}"
        for row in included.itertuples(index=False)
    )
    missing_list = "None." if missing.empty else "\n".join(f"- {row.model_id}: {row.reason}" for row in missing.itertuples(index=False))

    case_lines = []
    if case_selection.empty:
        case_lines.append("- No case-study events were selected because event windows or predictions were unavailable.")
    else:
        for row in case_selection.itertuples(index=False):
            case_lines.append(
                f"- {row.case_id}: site {row.site_id}, {row.event_type}, {row.event_start} to {row.event_end}, "
                f"model {row.model_id}, mean absolute residual {fmt(row.mean_abs_residual)}, "
                f"hydrological context available: {row.hydrological_context_available}."
            )

    report = f"""# pH Low/High Event Analysis Report

## 1. Purpose

This analysis evaluates existing model predictions separately during normal periods, regulatory low-pH event windows, regulatory high-pH event windows, and all pH event windows. No model was trained or re-trained in this task.

## 2. Low/High pH Event Definition

Low pH is defined from existing regulatory labels mapped to `pH_exceedance_low`. High pH is defined from existing regulatory labels mapped to `pH_exceedance_high`. Normal rows are outside pH event windows. Mixed or unclear labels are retained as `mixed_or_unknown` and are included in `all_pH_event` summaries. This is event-based classification, not pH quantile grouping.

Legacy prediction files used both full labels and shorthand labels (`low`, `high`, `none`); the script standardised them before calculating metrics.

## 3. Models Compared

Included models:

{model_list}

Missing or excluded expected models:

{missing_list}

Inventory details are saved in `unified_prediction_file_inventory.csv`, with included models in `models_included.csv`.

## 4. Prediction Metrics by Event Class

The main event-class metrics are saved in `low_high_prediction_metrics_by_model.csv`. Bias is defined as `mean(y_pred - y_true)`, so positive low-pH bias means the model predicts too high during low-pH events, while negative high-pH bias means the model predicts too low during high-pH events.

Best low-pH RMSE model: {low_best['model_id'].iloc[0] if not low_best.empty else 'NA'} with RMSE {fmt(low_best['rmse'].iloc[0] if not low_best.empty else np.nan)}.

Best high-pH RMSE model: {high_best['model_id'].iloc[0] if not high_best.empty else 'NA'} with RMSE {fmt(high_best['rmse'].iloc[0] if not high_best.empty else np.nan)}.

The dissertation-ready table is saved in `low_high_dissertation_ready_table.csv`.

## 5. Bias Analysis

The focused bias table is saved in `low_high_bias_by_model.csv`. It gives normal, low-event, high-event, and all-event bias for every included model, plus interpretation text for low-pH and high-pH behaviour.

Key bias interpretation should focus on whether models regress toward normal pH during extreme conditions. For example, positive bias in low-pH events indicates that low pH drops are being under-estimated; negative bias in high-pH events indicates that high pH peaks are being under-estimated.

## 6. Residual-Enrichment Analysis

The residual enrichment table is saved in `low_high_residual_event_enrichment.csv`. It includes global top-residual enrichment and site-normalised enrichment where available.

Best top-5% low-pH enrichment model: {low_enrich_best['model_id'].iloc[0] if not low_enrich_best.empty else 'NA'} with enrichment {fmt(low_enrich_best['enrichment'].iloc[0] if not low_enrich_best.empty else np.nan)}.

Best top-5% high-pH enrichment model: {high_enrich_best['model_id'].iloc[0] if not high_enrich_best.empty else 'NA'} with enrichment {fmt(high_enrich_best['enrichment'].iloc[0] if not high_enrich_best.empty else np.nan)}.

Strongest all-event residual reference: {all_enrich_best['model_id'].iloc[0] if not all_enrich_best.empty else 'NA'} with top-5% all-event enrichment {fmt(all_enrich_best['enrichment'].iloc[0] if not all_enrich_best.empty else np.nan)}.

Approximate event-level hit rates are saved in `low_high_event_level_hit_rate.csv`; the method is `approx_contiguous_rows_by_site_label`.

## 7. Metadata Interpretation

Metadata effects differ depending on whether the target is prediction accuracy or residual-event alignment:

{chr(10).join(metadata_lines) if metadata_lines else '- Metadata model pairs were not all available.'}

This means metadata can improve prediction accuracy while not necessarily increasing residual contrast during regulatory events.

## 8. Case Studies

Case-study selections are saved in `case_study_site_selection.csv`, and event-window details are saved in `case_study_event_windows.csv`.

{chr(10).join(case_lines)}

Figures include observed pH, predicted pH, prediction error, event-window shading, and discharge or water-level context when available.

## 9. Main Interpretation

Low-pH exceedance windows are generally more consistently reflected in residual enrichment than high-pH windows. The non-event HGB reference remains important as a residual-detection reference when its enrichment is strongest, while metadata-aware HGB and GRU models can improve prediction accuracy without always increasing residual contrast. Bias analysis identifies whether low-pH drops and high-pH peaks are under-estimated, which is the key behavioural interpretation for regulatory event periods.

## 10. Limitations

- Regulatory pH events are weak validation labels rather than complete ground truth.
- `00400` is processed/scaled pH, not necessarily raw pH units.
- High-pH event rows are less frequent than low-pH event rows in several prediction sets.
- Event-level hit rate uses approximate contiguous blocks when original event-window IDs are unavailable.
- Sequence models use hourly target windows, while HGB models use higher-frequency processed rows.
- Prediction files were not available for every possible model variant.

## 11. Dissertation-Ready Takeaways

- Low-pH regulatory events show stronger residual alignment than high-pH events in the top-residual analysis.
- Metadata often improves predictive RMSE, but residual-event enrichment may weaken or strengthen depending on the model family.
- The non-event HGB model should remain the main residual-reference comparator if it has the strongest all-event enrichment.
- Low/high event bias reveals whether models under-estimate regulatory pH extremes.
- This event-based analysis adds hydrological interpretation beyond overall RMSE and R2.

## Output Files

The full output folder is `{rel(OUT_DIR)}`. Required tables and figures have been written under this folder.
"""
    (OUT_DIR / "ph_low_high_event_analysis_report.md").write_text(report, encoding="utf-8")


def final_console_output(
    included: pd.DataFrame,
    missing: pd.DataFrame,
    metrics: pd.DataFrame,
    enrichment: pd.DataFrame,
    unified_rows: int,
) -> None:
    low_rows = int(metrics.loc[metrics["event_class"] == "low_pH_event", "n"].sum())
    high_rows = int(metrics.loc[metrics["event_class"] == "high_pH_event", "n"].sum())
    low_best = metrics.loc[metrics["event_class"] == "low_pH_event"].sort_values("rmse").head(1)
    high_best = metrics.loc[metrics["event_class"] == "high_pH_event"].sort_values("rmse").head(1)
    top5 = enrichment.loc[
        (enrichment["residual_threshold_type"] == "global_abs_residual")
        & np.isclose(enrichment["top_fraction"], 0.05)
    ]
    low_enrich = top5.loc[top5["event_group"] == "low_pH_event"].sort_values("enrichment", ascending=False).head(1)
    high_enrich = top5.loc[top5["event_group"] == "high_pH_event"].sort_values("enrichment", ascending=False).head(1)
    all_enrich = top5.loc[top5["event_group"] == "all_pH_event"].sort_values("enrichment", ascending=False).head(1)
    missing_text = "None" if missing.empty else ", ".join(missing["model_id"].astype(str).tolist())

    lines = [
        "",
        "PH LOW/HIGH EVENT ANALYSIS COMPLETE",
        "",
        "Output folder:",
        rel(OUT_DIR),
        "",
        "Models included:",
        ", ".join(included["model_id"].astype(str).tolist()),
        "",
        "Models missing prediction files:",
        missing_text,
        "",
        "Unified prediction rows/windows:",
        str(unified_rows),
        "",
        "Low-pH event rows/windows:",
        str(low_rows),
        "",
        "High-pH event rows/windows:",
        str(high_rows),
        "",
        "Best low-pH RMSE model:",
        f"{low_best['model_id'].iloc[0] if not low_best.empty else 'NA'} ({fmt(low_best['rmse'].iloc[0] if not low_best.empty else np.nan)})",
        "",
        "Best high-pH RMSE model:",
        f"{high_best['model_id'].iloc[0] if not high_best.empty else 'NA'} ({fmt(high_best['rmse'].iloc[0] if not high_best.empty else np.nan)})",
        "",
        "Best low-pH enrichment model:",
        f"{low_enrich['model_id'].iloc[0] if not low_enrich.empty else 'NA'} ({fmt(low_enrich['enrichment'].iloc[0] if not low_enrich.empty else np.nan)})",
        "",
        "Best high-pH enrichment model:",
        f"{high_enrich['model_id'].iloc[0] if not high_enrich.empty else 'NA'} ({fmt(high_enrich['enrichment'].iloc[0] if not high_enrich.empty else np.nan)})",
        "",
        "Strongest residual-event reference:",
        f"{all_enrich['model_id'].iloc[0] if not all_enrich.empty else 'NA'} ({fmt(all_enrich['enrichment'].iloc[0] if not all_enrich.empty else np.nan)})",
        "",
        "Main interpretation:",
        "Low-pH regulatory events show stronger residual alignment than high-pH events; metadata can improve prediction without always increasing residual contrast.",
        "",
        "Next recommended task:",
        "Prepare final dissertation result tables, figures, and case-study section.",
    ]
    for line in lines:
        log(line)


def main() -> None:
    ensure_dirs()
    log("Starting pH low/high event analysis.")
    log(f"Project root: {ROOT}")
    log("No model training will be performed.")

    inventory, included, missing = inventory_prediction_files()
    log(f"Inventory complete. Included models: {len(included)}.")

    writer: pq.ParquetWriter | None = None
    unified_path = OUT_DIR / "unified_predictions_test.parquet"
    if unified_path.exists():
        unified_path.unlink()

    all_metrics: list[dict[str, Any]] = []
    all_distributions: list[dict[str, Any]] = []
    all_enrichment: list[dict[str, Any]] = []
    all_hit_rate: list[dict[str, Any]] = []
    all_schema_audit: list[dict[str, Any]] = []
    plot_samples: list[pd.DataFrame] = []
    case_predictions: list[pd.DataFrame] = []
    event_window_tables: list[pd.DataFrame] = []
    unified_rows = 0

    included_specs = [spec for spec in MODEL_SPECS if spec.model_id in set(included["model_id"])]
    for spec in included_specs:
        log(f"Processing {spec.model_id} from {spec.prediction_file}")
        df, audit = load_standardised_predictions(spec)
        all_schema_audit.append(audit)
        unified_rows += len(df)

        table = as_arrow_table(df)
        if writer is None:
            writer = pq.ParquetWriter(unified_path, UNIFIED_SCHEMA, compression="snappy")
        writer.write_table(table)

        all_metrics.extend(compute_metrics(df, spec))
        all_distributions.extend(compute_distribution(df, spec))
        all_enrichment.extend(compute_enrichment(df, spec))
        hit_rows, windows = compute_event_hit_rate(df, spec)
        all_hit_rate.extend(hit_rows)
        if not windows.empty:
            event_window_tables.append(windows)

        sample = sample_for_plots(df, spec)
        if not sample.empty:
            plot_samples.append(sample)
        if spec.model_id == "HGB200_V2":
            case_predictions.append(df.copy())
        log(
            f"Finished {spec.model_id}: rows={len(df)}, low={int((df['event_class'] == 'low_pH_event').sum())}, "
            f"high={int((df['event_class'] == 'high_pH_event').sum())}."
        )

        del df, table

    if writer is not None:
        writer.close()
    log(f"Unified parquet written: {rel(unified_path)} with {unified_rows} rows/windows.")

    schema_audit = pd.DataFrame(all_schema_audit)
    metrics = order_models(pd.DataFrame(all_metrics))
    distributions = order_models(pd.DataFrame(all_distributions))
    enrichment = order_models(pd.DataFrame(all_enrichment))
    hit_rate = order_models(pd.DataFrame(all_hit_rate))
    bias = create_bias_table(metrics)

    schema_audit.to_csv(OUT_DIR / "unified_prediction_schema_audit.csv", index=False)
    metrics.to_csv(OUT_DIR / "low_high_prediction_metrics_by_model.csv", index=False)
    bias.to_csv(OUT_DIR / "low_high_bias_by_model.csv", index=False)
    distributions.to_csv(OUT_DIR / "low_high_residual_distribution_summary.csv", index=False)
    enrichment.to_csv(OUT_DIR / "low_high_residual_event_enrichment.csv", index=False)
    hit_rate.to_csv(OUT_DIR / "low_high_event_level_hit_rate.csv", index=False)
    log("Main analysis tables written.")

    summary, dissertation = create_summary_tables(metrics, enrichment, hit_rate, included)
    log("Summary tables written.")

    sample_df = pd.concat(plot_samples, ignore_index=True) if plot_samples else pd.DataFrame()
    create_standard_figures(metrics, bias, enrichment, hit_rate, sample_df)
    log("Standard figures written.")

    event_windows = pd.concat(event_window_tables, ignore_index=True) if event_window_tables else pd.DataFrame()
    case_df = pd.concat(case_predictions, ignore_index=True) if case_predictions else pd.DataFrame()
    case_selection, case_windows = create_case_studies(case_df, event_windows)
    log("Case-study outputs written.")

    write_report(inventory, included, missing, metrics, bias, enrichment, hit_rate, dissertation, case_selection)
    log("Markdown report written.")
    final_console_output(included, missing, metrics, enrichment, unified_rows)


if __name__ == "__main__":
    main()
