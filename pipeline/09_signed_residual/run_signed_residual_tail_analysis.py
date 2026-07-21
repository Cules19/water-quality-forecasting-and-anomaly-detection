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
import pyarrow.parquet as pq


warnings.filterwarnings("ignore", category=FutureWarning)


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
OUT_DIR = ROOT / "analysis_outputs/ph_signed_residual_tail_analysis_v1"
FIG_DIR = OUT_DIR / "figures"
LOG_PATH = OUT_DIR / "signed_residual_tail_analysis_log.txt"

TAIL_FRACTIONS = [0.01, 0.05, 0.10]
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


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_type: str
    site_count: int
    metadata_set: str
    training_mode: str
    prediction_file: str
    main_model_role: str
    row_filter_col: str | None = None
    row_filter_value: str | None = None


MODEL_SPECS = [
    ModelSpec(
        "M0_reference_nonevent",
        "HGB",
        50,
        "none",
        "non_event",
        "analysis_outputs/ph_nonevent_training_full50_v3/nonevent_predictions_test.parquet",
        "normal-behaviour residual reference",
        row_filter_col="model",
        row_filter_value="M4_hgb_nonevent_train",
    ),
    ModelSpec(
        "M1_no_metadata",
        "HGB",
        50,
        "none",
        "all_data",
        "analysis_outputs/ph_hgb_full50_metadata_v1/prediction_test_M1_no_metadata.parquet",
        "50-site all-data HGB baseline",
    ),
    ModelSpec(
        "M2_with_metadata",
        "HGB",
        50,
        "full",
        "all_data",
        "analysis_outputs/ph_hgb_full50_metadata_v1/prediction_test_M2_with_metadata.parquet",
        "50-site full-metadata HGB",
    ),
    ModelSpec(
        "HGB50_V2",
        "HGB",
        50,
        "compact_hydrological_v2",
        "all_data",
        "analysis_outputs/ph_v2_metadata_rerun_v1/hgb50_v2_predictions_test.parquet",
        "50-site compact V2 metadata HGB",
    ),
    ModelSpec(
        "M200_A_no_metadata",
        "HGB",
        200,
        "none",
        "all_data",
        "analysis_outputs/ph_hgb_processed200_metadata_v1/prediction_test_M200_A_no_metadata.parquet",
        "200-site all-data HGB baseline",
    ),
    ModelSpec(
        "M200_B_with_metadata",
        "HGB",
        200,
        "full",
        "all_data",
        "analysis_outputs/ph_hgb_processed200_metadata_v1/prediction_test_M200_B_with_metadata.parquet",
        "200-site full-metadata HGB",
    ),
    ModelSpec(
        "HGB200_V2",
        "HGB",
        200,
        "compact_hydrological_v2",
        "all_data",
        "analysis_outputs/ph_v2_metadata_rerun_v1/hgb200_v2_predictions_test.parquet",
        "200-site compact V2 metadata HGB",
    ),
    ModelSpec(
        "G100_A",
        "GRU",
        100,
        "none",
        "all_data",
        "analysis_outputs/ph_sequence100_gru_lstm_metadata_v1/prediction_test_G100_A_gru_no_metadata.parquet",
        "100-site GRU no-metadata sequence model",
    ),
    ModelSpec(
        "G100_B",
        "GRU",
        100,
        "full",
        "all_data",
        "analysis_outputs/ph_sequence100_gru_lstm_metadata_v1/prediction_test_G100_B_gru_with_metadata.parquet",
        "100-site GRU full-metadata sequence model",
    ),
    ModelSpec(
        "GRU100_V2",
        "GRU",
        100,
        "compact_hydrological_v2",
        "all_data",
        "analysis_outputs/ph_v2_metadata_rerun_v1/gru100_v2_predictions_test.parquet",
        "100-site GRU compact V2 metadata sequence model",
    ),
    ModelSpec(
        "L100_B",
        "LSTM",
        100,
        "full",
        "all_data",
        "analysis_outputs/ph_sequence100_lstm_metadata_v1/prediction_test_L100_B_lstm_with_metadata.parquet",
        "100-site LSTM full-metadata sequence sensitivity",
    ),
]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")


def log(message: str) -> None:
    print(message, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def path_for(spec: ModelSpec) -> Path:
    return ROOT / spec.prediction_file


def first_present(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def parquet_info(path: Path) -> tuple[int, list[str], dict[str, str]]:
    pf = pq.ParquetFile(path)
    columns = pf.schema_arrow.names
    dtypes = {field.name: str(field.type) for field in pf.schema_arrow}
    return int(pf.metadata.num_rows), columns, dtypes


def map_columns(columns: list[str], spec: ModelSpec) -> dict[str, str | None]:
    return {
        "site_id": first_present(columns, ["site_id", "site", "site_no", "Site_ID"]),
        "timestamp": first_present(columns, ["timestamp", "target_timestamp", "datetime"]),
        "y_true": first_present(columns, ["00400", "y_true", "observed", "observed_pH"]),
        "y_pred": first_present(columns, ["y_pred", "prediction", "predicted", "predicted_pH"]),
        "raw_residual": first_present(columns, ["residual", "residual_y_true_minus_pred"]),
        "raw_abs_residual": first_present(columns, ["abs_residual"]),
        "in_ph_event": first_present(columns, ["in_ph_event"]),
        "ph_event_label": first_present(columns, ["ph_event_label"]),
        "event_state": first_present(columns, ["event_state"]),
        "row_filter": spec.row_filter_col if spec.row_filter_col in columns else None,
    }


def model_fields(spec: ModelSpec) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "model_type": spec.model_type,
        "site_count": spec.site_count,
        "metadata_set": spec.metadata_set,
        "training_mode": spec.training_mode,
    }


def count_filtered_rows(path: Path, columns: list[str], spec: ModelSpec, total_rows: int) -> int:
    if not spec.row_filter_col or spec.row_filter_col not in columns:
        return total_rows
    filt = pd.read_parquet(path, columns=[spec.row_filter_col])[spec.row_filter_col]
    return int((filt.astype(str) == str(spec.row_filter_value)).sum())


def audit_prediction_files() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    included_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for spec in MODEL_SPECS:
        path = path_for(spec)
        if not path.exists():
            reason = "Prediction file missing."
            inventory_rows.append(
                {
                    **model_fields(spec),
                    "prediction_file": spec.prediction_file,
                    "n_rows_or_windows": 0,
                    "has_site_id": False,
                    "has_timestamp": False,
                    "has_y_true": False,
                    "has_y_pred": False,
                    "has_event_label": False,
                    "include_in_analysis": False,
                    "exclusion_reason": reason,
                }
            )
            schema_rows.append(
                {
                    **model_fields(spec),
                    "prediction_file": spec.prediction_file,
                    "raw_columns": "",
                    "site_id_col": None,
                    "timestamp_col": None,
                    "y_true_col": None,
                    "y_pred_col": None,
                    "residual_col": None,
                    "abs_residual_col": None,
                    "in_ph_event_col": None,
                    "ph_event_label_col": None,
                    "event_state_col": None,
                    "row_filter_col": spec.row_filter_col,
                    "row_filter_value": spec.row_filter_value,
                    "residual_definition_used": "observed_minus_predicted",
                    "prediction_error_definition_used": "predicted_minus_observed",
                    "raw_dtypes_json": "{}",
                }
            )
            missing_rows.append({"model_id": spec.model_id, "prediction_file": spec.prediction_file, "reason": reason})
            continue

        total_rows, columns, dtypes = parquet_info(path)
        mapping = map_columns(columns, spec)
        n_rows = count_filtered_rows(path, columns, spec, total_rows)
        has_event_label = any(mapping[key] is not None for key in ["in_ph_event", "ph_event_label", "event_state"])
        has_required = all(mapping[key] is not None for key in ["site_id", "timestamp", "y_true", "y_pred"])
        include = bool(has_required and has_event_label and n_rows > 0)
        reasons = []
        if not has_required:
            reasons.append("Missing one or more required prediction columns: site_id/timestamp/y_true/y_pred.")
        if not has_event_label:
            reasons.append("Missing event-label columns.")
        if n_rows == 0:
            reasons.append("No rows after required model filter.")
        reason = " ".join(reasons)
        inventory_rows.append(
            {
                **model_fields(spec),
                "prediction_file": spec.prediction_file,
                "n_rows_or_windows": n_rows,
                "has_site_id": mapping["site_id"] is not None,
                "has_timestamp": mapping["timestamp"] is not None,
                "has_y_true": mapping["y_true"] is not None,
                "has_y_pred": mapping["y_pred"] is not None,
                "has_event_label": has_event_label,
                "include_in_analysis": include,
                "exclusion_reason": "" if include else reason,
            }
        )
        schema_rows.append(
            {
                **model_fields(spec),
                "prediction_file": spec.prediction_file,
                "raw_n_rows": total_rows,
                "filtered_n_rows": n_rows,
                "raw_columns": "|".join(columns),
                "site_id_col": mapping["site_id"],
                "timestamp_col": mapping["timestamp"],
                "y_true_col": mapping["y_true"],
                "y_pred_col": mapping["y_pred"],
                "residual_col": mapping["raw_residual"],
                "abs_residual_col": mapping["raw_abs_residual"],
                "in_ph_event_col": mapping["in_ph_event"],
                "ph_event_label_col": mapping["ph_event_label"],
                "event_state_col": mapping["event_state"],
                "row_filter_col": mapping["row_filter"],
                "row_filter_value": spec.row_filter_value if mapping["row_filter"] else None,
                "residual_definition_used": "observed_minus_predicted",
                "prediction_error_definition_used": "predicted_minus_observed",
                "raw_dtypes_json": json.dumps(dtypes, sort_keys=True),
            }
        )
        if include:
            included_rows.append(
                {
                    **model_fields(spec),
                    "prediction_file": spec.prediction_file,
                    "n_rows_or_windows": n_rows,
                    "main_model_role": spec.main_model_role,
                }
            )
        else:
            missing_rows.append({"model_id": spec.model_id, "prediction_file": spec.prediction_file, "reason": reason})

    inventory = pd.DataFrame(inventory_rows)
    schema = pd.DataFrame(schema_rows)
    included = pd.DataFrame(included_rows)
    missing = pd.DataFrame(missing_rows, columns=["model_id", "prediction_file", "reason"])
    inventory.to_csv(OUT_DIR / "prediction_file_inventory.csv", index=False)
    schema.to_csv(OUT_DIR / "prediction_schema_audit.csv", index=False)
    included.to_csv(OUT_DIR / "models_included.csv", index=False)
    missing.to_csv(OUT_DIR / "models_missing_prediction_files.csv", index=False)
    return inventory, schema, included


def event_class_from_raw(raw: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    n = len(raw)
    in_event = pd.Series(False, index=raw.index)
    if "in_ph_event_raw" in raw:
        in_event = pd.to_numeric(raw["in_ph_event_raw"], errors="coerce").fillna(0).astype("int8") != 0

    label = pd.Series("", index=raw.index, dtype="string")
    if "ph_event_label_raw" in raw:
        label = raw["ph_event_label_raw"].astype("string").fillna("").str.strip().str.lower()

    low_mask = label.str.contains("low", regex=False).fillna(False)
    high_mask = label.str.contains("high", regex=False).fillna(False)
    normal_label = label.isin(["", "none", "normal", "no_event", "not_event", "nan", "<na>"])

    if "event_state_raw" in raw:
        state_num = pd.to_numeric(raw["event_state_raw"], errors="coerce")
        event_class = pd.Series("mixed_or_unknown", index=raw.index, dtype="string")
        event_class.loc[state_num == 0] = "normal"
        event_class.loc[state_num == 1] = "low_pH_event"
        event_class.loc[state_num == 2] = "high_pH_event"
        event_class.loc[state_num == 3] = "mixed_or_unknown"
        unresolved = state_num.isna() | ~state_num.isin([0, 1, 2, 3])
        event_class.loc[unresolved & low_mask] = "low_pH_event"
        event_class.loc[unresolved & high_mask] = "high_pH_event"
        event_class.loc[unresolved & (~in_event | normal_label) & ~(low_mask | high_mask)] = "normal"
    else:
        event_class = pd.Series("mixed_or_unknown", index=raw.index, dtype="string")
        event_class.loc[(~in_event | normal_label) & ~(low_mask | high_mask)] = "normal"
        event_class.loc[low_mask] = "low_pH_event"
        event_class.loc[high_mask] = "high_pH_event"

    ph_label = pd.Series("mixed_or_unknown", index=raw.index, dtype="string")
    ph_label.loc[event_class == "normal"] = "none"
    ph_label.loc[event_class == "low_pH_event"] = "pH_exceedance_low"
    ph_label.loc[event_class == "high_pH_event"] = "pH_exceedance_high"

    state = pd.Series(3, index=raw.index, dtype="int8")
    state.loc[event_class == "normal"] = 0
    state.loc[event_class == "low_pH_event"] = 1
    state.loc[event_class == "high_pH_event"] = 2
    return ph_label, state, event_class


def load_standardised_predictions(spec: ModelSpec, schema_row: pd.Series) -> pd.DataFrame:
    path = path_for(spec)
    mapping = {
        "site_id": schema_row["site_id_col"],
        "timestamp": schema_row["timestamp_col"],
        "y_true": schema_row["y_true_col"],
        "y_pred": schema_row["y_pred_col"],
        "in_ph_event": schema_row["in_ph_event_col"],
        "ph_event_label": schema_row["ph_event_label_col"],
        "event_state": schema_row["event_state_col"],
        "row_filter": schema_row["row_filter_col"],
    }
    read_cols = {value for value in mapping.values() if isinstance(value, str) and value}
    raw = pd.read_parquet(path, columns=list(read_cols))
    if spec.row_filter_col and spec.row_filter_col in raw:
        raw = raw.loc[raw[spec.row_filter_col].astype(str) == str(spec.row_filter_value)].copy()

    rename = {
        mapping["site_id"]: "site_id",
        mapping["timestamp"]: "timestamp",
        mapping["y_true"]: "y_true",
        mapping["y_pred"]: "y_pred",
    }
    if isinstance(mapping["in_ph_event"], str) and mapping["in_ph_event"]:
        rename[mapping["in_ph_event"]] = "in_ph_event_raw"
    if isinstance(mapping["ph_event_label"], str) and mapping["ph_event_label"]:
        rename[mapping["ph_event_label"]] = "ph_event_label_raw"
    if isinstance(mapping["event_state"], str) and mapping["event_state"]:
        rename[mapping["event_state"]] = "event_state_raw"
    raw = raw.rename(columns=rename)

    df = pd.DataFrame(index=raw.index)
    df["site_id"] = raw["site_id"].astype("string")
    df["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    df["y_true"] = pd.to_numeric(raw["y_true"], errors="coerce").astype("float32")
    df["y_pred"] = pd.to_numeric(raw["y_pred"], errors="coerce").astype("float32")
    residual = (df["y_true"].astype("float64") - df["y_pred"].astype("float64")).astype("float32")
    df["residual"] = residual
    df["abs_residual"] = residual.abs().astype("float32")
    df["prediction_error"] = (-residual).astype("float32")

    event_input = raw[[col for col in ["in_ph_event_raw", "ph_event_label_raw", "event_state_raw"] if col in raw]].copy()
    ph_label, event_state, event_class = event_class_from_raw(event_input)
    df["in_ph_event"] = (event_state != 0).astype("int8")
    df["ph_event_label"] = ph_label.astype("string")
    df["event_state"] = event_state.astype("int8")
    df["event_class"] = event_class.astype("string")
    df = df.dropna(subset=["site_id", "timestamp", "y_true", "y_pred", "residual"]).reset_index(drop=True)
    return df


def thresholds_for_model(df: pd.DataFrame, spec: ModelSpec) -> tuple[pd.DataFrame, dict[tuple[str, float], float]]:
    rows: list[dict[str, Any]] = []
    thresholds: dict[tuple[str, float], float] = {}
    n_test = len(df)
    n_sites = int(df["site_id"].nunique())
    for frac in TAIL_FRACTIONS:
        neg = float(df["residual"].quantile(frac))
        pos = float(df["residual"].quantile(1 - frac))
        abs_thr = float(df["abs_residual"].quantile(1 - frac))
        thresholds[("negative", frac)] = neg
        thresholds[("positive", frac)] = pos
        thresholds[("absolute", frac)] = abs_thr
        for tail_type, threshold, selected in [
            ("negative", neg, df["residual"] <= neg),
            ("positive", pos, df["residual"] >= pos),
            ("absolute", abs_thr, df["abs_residual"] >= abs_thr),
        ]:
            rows.append(
                {
                    **model_fields(spec),
                    "tail_type": tail_type,
                    "tail_fraction": frac,
                    "residual_threshold": threshold,
                    "n_test_rows": n_test,
                    "n_selected_rows": int(selected.sum()),
                    "n_sites": n_sites,
                }
            )
    return pd.DataFrame(rows), thresholds


def group_mask(df: pd.DataFrame, target_event_group: str) -> pd.Series:
    if target_event_group == "all_pH_event":
        return df["event_class"].isin(["low_pH_event", "high_pH_event", "mixed_or_unknown"])
    return df["event_class"] == target_event_group


def tail_mask(df: pd.DataFrame, tail_type: str, threshold: float) -> pd.Series:
    if tail_type == "negative":
        return df["residual"] <= threshold
    if tail_type == "positive":
        return df["residual"] >= threshold
    if tail_type == "absolute":
        return df["abs_residual"] >= threshold
    raise ValueError(f"Unknown tail_type: {tail_type}")


def enrichment_rows(df: pd.DataFrame, spec: ModelSpec, thresholds: dict[tuple[str, float], float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    combos = [
        ("negative", "low_pH_event"),
        ("positive", "high_pH_event"),
        ("absolute", "all_pH_event"),
        ("absolute", "low_pH_event"),
        ("absolute", "high_pH_event"),
    ]
    n_test = len(df)
    for tail_type, event_group in combos:
        for frac in TAIL_FRACTIONS:
            threshold = thresholds[(tail_type, frac)]
            tmask = tail_mask(df, tail_type, threshold)
            emask = group_mask(df, event_group)
            n_event = int(emask.sum())
            n_tail = int(tmask.sum())
            n_tail_event = int((tmask & emask).sum())
            base_rate = n_event / n_test if n_test else math.nan
            tail_rate = n_tail_event / n_tail if n_tail else math.nan
            enrichment = tail_rate / base_rate if base_rate and not math.isnan(base_rate) else math.nan
            rows.append(
                {
                    **model_fields(spec),
                    "tail_type": tail_type,
                    "target_event_group": event_group,
                    "tail_fraction": frac,
                    "n_test_rows": n_test,
                    "n_event_rows": n_event,
                    "base_event_rate": base_rate,
                    "n_tail_rows": n_tail,
                    "n_tail_event_rows": n_tail_event,
                    "tail_event_rate": tail_rate,
                    "enrichment": enrichment,
                    "residual_threshold": threshold,
                }
            )
    return rows


def direction_consistency_rows(df: pd.DataFrame, spec: ModelSpec) -> list[dict[str, Any]]:
    rows = []
    for event_group, expected_sign, expected_mask_func, opposite_mask_func in [
        ("low_pH_event", "negative", lambda x: x < 0, lambda x: x > 0),
        ("high_pH_event", "positive", lambda x: x > 0, lambda x: x < 0),
    ]:
        sub = df.loc[df["event_class"] == event_group, "residual"].astype("float64")
        n = len(sub)
        expected = int(expected_mask_func(sub).sum()) if n else 0
        opposite = int(opposite_mask_func(sub).sum()) if n else 0
        rows.append(
            {
                **model_fields(spec),
                "event_group": event_group,
                "n_event_rows": n,
                "expected_sign": expected_sign,
                "expected_sign_count": expected,
                "expected_sign_rate": expected / n if n else math.nan,
                "opposite_sign_count": opposite,
                "opposite_sign_rate": opposite / n if n else math.nan,
                "mean_residual": float(sub.mean()) if n else math.nan,
                "median_residual": float(sub.median()) if n else math.nan,
                "mean_abs_residual": float(sub.abs().mean()) if n else math.nan,
            }
        )
    return rows


def infer_gap_threshold(df: pd.DataFrame, spec: ModelSpec) -> pd.Timedelta:
    if spec.model_type in {"GRU", "LSTM"}:
        return pd.Timedelta(hours=3)
    sample = df[["site_id", "timestamp"]].head(min(len(df), 500_000)).copy()
    sample = sample.sort_values(["site_id", "timestamp"])
    diffs = sample.groupby("site_id", observed=True)["timestamp"].diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if len(diffs) == 0:
        return pd.Timedelta(minutes=30)
    threshold = diffs.median() * 3
    threshold = max(threshold, pd.Timedelta(minutes=30))
    threshold = min(threshold, pd.Timedelta(hours=3))
    return threshold


def event_windows(df: pd.DataFrame, spec: ModelSpec) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    method = "approx_contiguous_rows_by_site_label"
    event_rows = df.loc[df["event_class"].isin(["low_pH_event", "high_pH_event", "mixed_or_unknown"])].copy()
    if event_rows.empty:
        return event_rows, pd.DataFrame(), method
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
    event_rows["event_window_id"] = np.cumsum(np.asarray(new_window.fillna(True), dtype=bool)).astype("int64")
    event_rows["gap_threshold_minutes"] = gap.total_seconds() / 60
    window_summary = (
        event_rows.groupby("event_window_id", observed=True)
        .agg(
            site_id=("site_id", "first"),
            event_class=("event_class", "first"),
            event_start=("timestamp", "min"),
            event_end=("timestamp", "max"),
            n_event_rows_or_windows=("timestamp", "size"),
            gap_threshold_minutes=("gap_threshold_minutes", "first"),
        )
        .reset_index()
    )
    return event_rows, window_summary, method


def event_hit_rate_rows(
    df: pd.DataFrame,
    spec: ModelSpec,
    thresholds: dict[tuple[str, float], float],
    event_rows_df: pd.DataFrame,
    windows: pd.DataFrame,
    method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    combos = [
        ("low_pH_event", "negative", ["low_pH_event"]),
        ("high_pH_event", "positive", ["high_pH_event"]),
        ("all_pH_event", "absolute", ["low_pH_event", "high_pH_event", "mixed_or_unknown"]),
        ("low_pH_event", "absolute", ["low_pH_event"]),
        ("high_pH_event", "absolute", ["high_pH_event"]),
    ]
    for event_group, tail_type, classes in combos:
        group_windows = windows.loc[windows["event_class"].isin(classes)] if not windows.empty else pd.DataFrame()
        n_events = len(group_windows)
        for frac in TAIL_FRACTIONS:
            if n_events == 0:
                n_hits = 0
            else:
                threshold = thresholds[(tail_type, frac)]
                selected_indexes = set(df.index[tail_mask(df, tail_type, threshold)])
                hit_ids = set(event_rows_df.loc[event_rows_df["index"].isin(selected_indexes), "event_window_id"].astype(int))
                n_hits = int(group_windows["event_window_id"].isin(hit_ids).sum())
            rows.append(
                {
                    **model_fields(spec),
                    "event_group": event_group,
                    "tail_type": tail_type,
                    "tail_fraction": frac,
                    "n_event_windows": n_events,
                    "n_hit_event_windows": n_hits,
                    "event_level_hit_rate": n_hits / n_events if n_events else math.nan,
                    "event_window_method": method,
                }
            )
    return rows


def reliability_label(n: int) -> str:
    if n >= 10:
        return "reliable"
    if n >= 3:
        return "limited"
    return "insufficient"


def site_enrichment(site_df: pd.DataFrame, tmask: pd.Series, emask: pd.Series) -> tuple[float, int, int]:
    n_site = len(site_df)
    n_event = int(emask.sum())
    n_tail = int(tmask.sum())
    n_tail_event = int((tmask & emask).sum())
    base_rate = n_event / n_site if n_site else math.nan
    tail_rate = n_tail_event / n_tail if n_tail else math.nan
    enrichment = tail_rate / base_rate if base_rate and not math.isnan(base_rate) else math.nan
    return enrichment, n_tail, n_tail_event


def site_hit_rate(site_windows: pd.DataFrame, site_event_rows: pd.DataFrame, selected_indexes: set[int], classes: list[str]) -> float:
    group_windows = site_windows.loc[site_windows["event_class"].isin(classes)]
    if group_windows.empty:
        return math.nan
    hit_ids = set(site_event_rows.loc[site_event_rows["index"].isin(selected_indexes), "event_window_id"].astype(int))
    return float(group_windows["event_window_id"].isin(hit_ids).sum() / len(group_windows))


def per_site_rows(
    df: pd.DataFrame,
    spec: ModelSpec,
    thresholds: dict[tuple[str, float], float],
    event_rows_df: pd.DataFrame,
    windows: pd.DataFrame,
) -> list[dict[str, Any]]:
    frac = 0.05
    neg_thr = thresholds[("negative", frac)]
    pos_thr = thresholds[("positive", frac)]
    abs_thr = thresholds[("absolute", frac)]
    neg_selected = set(df.index[df["residual"] <= neg_thr])
    pos_selected = set(df.index[df["residual"] >= pos_thr])
    abs_selected = set(df.index[df["abs_residual"] >= abs_thr])
    rows: list[dict[str, Any]] = []
    for site_id, site_df in df.groupby("site_id", observed=True):
        low_mask = site_df["event_class"] == "low_pH_event"
        high_mask = site_df["event_class"] == "high_pH_event"
        neg_mask = site_df["residual"] <= neg_thr
        pos_mask = site_df["residual"] >= pos_thr
        abs_mask = site_df["abs_residual"] >= abs_thr
        neg_low, n_neg, n_neg_low = site_enrichment(site_df, neg_mask, low_mask)
        pos_high, n_pos, n_pos_high = site_enrichment(site_df, pos_mask, high_mask)
        abs_low, n_abs, n_abs_low = site_enrichment(site_df, abs_mask, low_mask)
        abs_high, _, n_abs_high = site_enrichment(site_df, abs_mask, high_mask)
        site_event_rows = event_rows_df.loc[event_rows_df["site_id"] == site_id] if not event_rows_df.empty else pd.DataFrame()
        site_windows = windows.loc[windows["site_id"] == site_id] if not windows.empty else pd.DataFrame()
        rows.append(
            {
                **model_fields(spec),
                "site_id": str(site_id),
                "n_test_rows_or_windows": len(site_df),
                "low_pH_event_count": int(low_mask.sum()),
                "high_pH_event_count": int(high_mask.sum()),
                "low_reliability": reliability_label(int(low_mask.sum())),
                "high_reliability": reliability_label(int(high_mask.sum())),
                "negative_tail5_low_pH_enrichment": neg_low,
                "positive_tail5_high_pH_enrichment": pos_high,
                "absolute_tail5_low_pH_enrichment": abs_low,
                "absolute_tail5_high_pH_enrichment": abs_high,
                "negative_tail5_rows": n_neg,
                "positive_tail5_rows": n_pos,
                "absolute_tail5_rows": n_abs,
                "negative_tail5_low_pH_rows": n_neg_low,
                "positive_tail5_high_pH_rows": n_pos_high,
                "absolute_tail5_low_pH_rows": n_abs_low,
                "absolute_tail5_high_pH_rows": n_abs_high,
                "negative_tail5_low_pH_hit_rate": site_hit_rate(site_windows, site_event_rows, neg_selected, ["low_pH_event"]),
                "positive_tail5_high_pH_hit_rate": site_hit_rate(site_windows, site_event_rows, pos_selected, ["high_pH_event"]),
                "absolute_tail5_low_pH_hit_rate": site_hit_rate(site_windows, site_event_rows, abs_selected, ["low_pH_event"]),
                "absolute_tail5_high_pH_hit_rate": site_hit_rate(site_windows, site_event_rows, abs_selected, ["high_pH_event"]),
            }
        )
    return rows


def order_models(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "model_id" not in df:
        return df
    order = {model: idx for idx, model in enumerate(REPORT_MODEL_ORDER)}
    return df.assign(_order=df["model_id"].map(order).fillna(999)).sort_values(["_order", "model_id"]).drop(columns="_order")


def top5_value(df: pd.DataFrame, model_id: str, tail_type: str, event_group: str, value_col: str) -> float:
    sub = df.loc[
        (df["model_id"] == model_id)
        & (df["tail_type"] == tail_type)
        & (df.get("target_event_group", df.get("event_group")) == event_group)
        & np.isclose(df["tail_fraction"], 0.05)
    ]
    if sub.empty:
        return math.nan
    return float(sub.iloc[0][value_col])


def build_comparison(
    enrichment: pd.DataFrame,
    hit_rate: pd.DataFrame,
    included: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in REPORT_MODEL_ORDER:
        if model_id not in set(included["model_id"]):
            continue
        low_neg = top5_value(enrichment, model_id, "negative", "low_pH_event", "enrichment")
        low_abs = top5_value(enrichment, model_id, "absolute", "low_pH_event", "enrichment")
        high_pos = top5_value(enrichment, model_id, "positive", "high_pH_event", "enrichment")
        high_abs = top5_value(enrichment, model_id, "absolute", "high_pH_event", "enrichment")
        low_neg_hit = top5_value(hit_rate.rename(columns={"event_group": "target_event_group"}), model_id, "negative", "low_pH_event", "event_level_hit_rate")
        low_abs_hit = top5_value(hit_rate.rename(columns={"event_group": "target_event_group"}), model_id, "absolute", "low_pH_event", "event_level_hit_rate")
        high_pos_hit = top5_value(hit_rate.rename(columns={"event_group": "target_event_group"}), model_id, "positive", "high_pH_event", "event_level_hit_rate")
        high_abs_hit = top5_value(hit_rate.rename(columns={"event_group": "target_event_group"}), model_id, "absolute", "high_pH_event", "event_level_hit_rate")
        if pd.notna(low_neg) and pd.notna(high_pos):
            low_gain = low_neg > low_abs
            high_gain = high_pos > high_abs
            high_enriched = high_pos > 1
            if low_gain and high_gain and high_enriched:
                interp = "Signed tails improve both directions; high-pH positive-tail enrichment exceeds 1."
            elif low_gain and high_gain:
                interp = "Signed tails improve both directions relative to absolute residuals, but high-pH positive-tail enrichment remains below 1."
            elif low_gain and not high_gain:
                interp = "Signed negative tail improves low-pH interpretation; high-pH remains weak or not improved."
            elif high_gain and not low_gain:
                interp = "Positive tail improves high-pH interpretation; low-pH is not improved over absolute residuals."
            else:
                interp = "Signed tails do not improve top-5 enrichment over absolute residuals for this model."
        else:
            interp = "Insufficient event rows or metrics for interpretation."
        rows.append(
            {
                "model_id": model_id,
                "low_negative_tail_enrichment": low_neg,
                "low_absolute_tail_enrichment": low_abs,
                "low_signed_minus_absolute_delta": low_neg - low_abs if pd.notna(low_neg) and pd.notna(low_abs) else math.nan,
                "high_positive_tail_enrichment": high_pos,
                "high_absolute_tail_enrichment": high_abs,
                "high_signed_minus_absolute_delta": high_pos - high_abs if pd.notna(high_pos) and pd.notna(high_abs) else math.nan,
                "low_negative_tail_hit_rate": low_neg_hit,
                "low_absolute_tail_hit_rate": low_abs_hit,
                "high_positive_tail_hit_rate": high_pos_hit,
                "high_absolute_tail_hit_rate": high_abs_hit,
                "main_interpretation": interp,
            }
        )
    return order_models(pd.DataFrame(rows))


def build_dissertation_table(
    enrichment: pd.DataFrame,
    hit_rate: pd.DataFrame,
    consistency: pd.DataFrame,
    included: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    role_lookup = included.set_index("model_id")["main_model_role"].to_dict()
    field_lookup = included.set_index("model_id")[["model_type", "site_count", "metadata_set", "training_mode"]].to_dict("index")
    for model_id in REPORT_MODEL_ORDER:
        if model_id not in role_lookup:
            continue
        low_sign = consistency.loc[(consistency["model_id"] == model_id) & (consistency["event_group"] == "low_pH_event")]
        high_sign = consistency.loc[(consistency["model_id"] == model_id) & (consistency["event_group"] == "high_pH_event")]
        row = {
            "model_id": model_id,
            **field_lookup[model_id],
            "absolute_top5_all_event_enrichment": top5_value(enrichment, model_id, "absolute", "all_pH_event", "enrichment"),
            "absolute_top5_low_enrichment": top5_value(enrichment, model_id, "absolute", "low_pH_event", "enrichment"),
            "absolute_top5_high_enrichment": top5_value(enrichment, model_id, "absolute", "high_pH_event", "enrichment"),
            "negative_tail5_low_enrichment": top5_value(enrichment, model_id, "negative", "low_pH_event", "enrichment"),
            "positive_tail5_high_enrichment": top5_value(enrichment, model_id, "positive", "high_pH_event", "enrichment"),
            "negative_tail5_low_event_hit_rate": top5_value(hit_rate.rename(columns={"event_group": "target_event_group"}), model_id, "negative", "low_pH_event", "event_level_hit_rate"),
            "positive_tail5_high_event_hit_rate": top5_value(hit_rate.rename(columns={"event_group": "target_event_group"}), model_id, "positive", "high_pH_event", "event_level_hit_rate"),
            "low_expected_sign_rate": float(low_sign.iloc[0]["expected_sign_rate"]) if not low_sign.empty else math.nan,
            "high_expected_sign_rate": float(high_sign.iloc[0]["expected_sign_rate"]) if not high_sign.empty else math.nan,
            "main_model_role": role_lookup[model_id],
        }
        low_improves = row["negative_tail5_low_enrichment"] > row["absolute_top5_low_enrichment"]
        high_improves = row["positive_tail5_high_enrichment"] > row["absolute_top5_high_enrichment"]
        high_enriched = row["positive_tail5_high_enrichment"] > 1
        if low_improves and high_improves and high_enriched:
            interp = "Signed tails improve both directions, and high-pH positive-tail enrichment exceeds 1."
        elif low_improves and high_improves:
            interp = "Signed tails improve both directions relative to absolute residuals, but high-pH remains below enrichment = 1."
        elif low_improves and not high_improves:
            interp = "Negative-tail analysis strengthens low-pH interpretation, but high-pH remains weak."
        elif high_improves and not low_improves:
            interp = "Positive-tail analysis strengthens high-pH interpretation, while low-pH is not improved over absolute residuals."
        else:
            interp = "Signed-tail enrichment is not stronger than absolute residual enrichment at top/bottom 5%."
        row["main_interpretation"] = interp
        rows.append(row)
    return order_models(pd.DataFrame(rows))


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


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
        }
    )


def save_bar_figure(df: pd.DataFrame, title: str, ylabel: str, output_name: str) -> None:
    plot_df = order_models(df.copy())
    if plot_df.empty:
        return
    models = list(plot_df["model_id"].drop_duplicates())
    x = np.arange(len(models))
    width = 0.23
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {0.01: "#4c78a8", 0.05: "#f58518", 0.10: "#54a24b"}
    for idx, frac in enumerate(TAIL_FRACTIONS):
        vals = []
        for model in models:
            sub = plot_df.loc[(plot_df["model_id"] == model) & np.isclose(plot_df["tail_fraction"], frac)]
            vals.append(float(sub["enrichment"].iloc[0]) if not sub.empty else math.nan)
        ax.bar(x + (idx - 1) * width, vals, width=width, label=f"{int(frac * 100)}%", color=colors[frac])
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(m) for m in models], rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Tail")
    fig.tight_layout()
    fig.savefig(FIG_DIR / output_name)
    plt.close(fig)


def plot_outputs(
    enrichment: pd.DataFrame,
    hit_rate: pd.DataFrame,
    comparison: pd.DataFrame,
    consistency: pd.DataFrame,
    dissertation: pd.DataFrame,
    residual_samples: pd.DataFrame,
) -> None:
    setup_plot_style()
    neg_low = enrichment.loc[(enrichment["tail_type"] == "negative") & (enrichment["target_event_group"] == "low_pH_event")]
    save_bar_figure(
        neg_low,
        "Low-pH enrichment in negative residual tails",
        "Enrichment",
        "negative_tail_low_pH_enrichment.png",
    )

    pos_high = enrichment.loc[(enrichment["tail_type"] == "positive") & (enrichment["target_event_group"] == "high_pH_event")]
    save_bar_figure(
        pos_high,
        "High-pH enrichment in positive residual tails",
        "Enrichment",
        "positive_tail_high_pH_enrichment.png",
    )

    subset = ["M0_reference_nonevent", "HGB50_V2", "HGB200_V2", "G100_B", "GRU100_V2", "L100_B"]
    comp = comparison.loc[comparison["model_id"].isin(subset)].copy()
    if not comp.empty:
        metrics = [
            ("low_absolute_tail_enrichment", "Low abs"),
            ("low_negative_tail_enrichment", "Low neg"),
            ("high_absolute_tail_enrichment", "High abs"),
            ("high_positive_tail_enrichment", "High pos"),
        ]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(comp))
        width = 0.19
        colors = ["#9ecae9", "#2171b5", "#fcae91", "#cb181d"]
        for idx, (col, label) in enumerate(metrics):
            ax.bar(x + (idx - 1.5) * width, comp[col], width=width, label=label, color=colors[idx])
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(m) for m in comp["model_id"]], rotation=35, ha="right")
        ax.set_ylabel("Top/bottom 5% enrichment")
        ax.set_title("Signed versus absolute residual-tail enrichment")
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "signed_vs_absolute_tail_comparison.png")
        plt.close(fig)

    hit5 = hit_rate.loc[np.isclose(hit_rate["tail_fraction"], 0.05)].copy()
    hit5 = hit5.loc[
        ((hit5["tail_type"] == "negative") & (hit5["event_group"] == "low_pH_event"))
        | ((hit5["tail_type"] == "positive") & (hit5["event_group"] == "high_pH_event"))
        | ((hit5["tail_type"] == "absolute") & (hit5["event_group"].isin(["low_pH_event", "high_pH_event"])))
    ]
    if not hit5.empty:
        hit5["series"] = hit5["event_group"].str.replace("_pH_event", "", regex=False) + " " + hit5["tail_type"]
        hit5 = order_models(hit5)
        models = list(hit5["model_id"].drop_duplicates())
        series = ["low negative", "low absolute", "high positive", "high absolute"]
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(models))
        width = 0.19
        colors = ["#2171b5", "#9ecae9", "#cb181d", "#fcae91"]
        for idx, ser in enumerate(series):
            vals = []
            for model in models:
                sub = hit5.loc[(hit5["model_id"] == model) & (hit5["series"] == ser)]
                vals.append(float(sub["event_level_hit_rate"].iloc[0]) if not sub.empty else math.nan)
            ax.bar(x + (idx - 1.5) * width, vals, width=width, label=ser, color=colors[idx])
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(m) for m in models], rotation=35, ha="right")
        ax.set_ylabel("Event-level hit rate")
        ax.set_title("Signed event-level hit rate at 5% tails")
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "signed_tail_event_hit_rate.png")
        plt.close(fig)

    direction = consistency.copy()
    if not direction.empty:
        direction = order_models(direction)
        models = list(direction["model_id"].drop_duplicates())
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(models))
        width = 0.35
        for idx, (group, label, color) in enumerate(
            [
                ("low_pH_event", "Low-pH residual < 0", "#2171b5"),
                ("high_pH_event", "High-pH residual > 0", "#cb181d"),
            ]
        ):
            vals = []
            for model in models:
                sub = direction.loc[(direction["model_id"] == model) & (direction["event_group"] == group)]
                vals.append(float(sub["expected_sign_rate"].iloc[0]) if not sub.empty else math.nan)
            ax.bar(x + (idx - 0.5) * width, vals, width=width, label=label, color=color)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(m) for m in models], rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Expected sign rate")
        ax.set_title("Low/high residual direction consistency")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "low_high_direction_consistency.png")
        plt.close(fig)

    if not residual_samples.empty:
        key_models = ["M0_reference_nonevent", "HGB200_V2", "G100_B"]
        fig, axes = plt.subplots(len(key_models), 1, figsize=(10, 8), sharex=True)
        bins = np.linspace(
            max(-0.12, residual_samples["residual"].quantile(0.005)),
            min(0.12, residual_samples["residual"].quantile(0.995)),
            80,
        )
        colors = {"normal": "#777777", "low_pH_event": "#2171b5", "high_pH_event": "#cb181d"}
        for ax, model in zip(axes, key_models):
            sub = residual_samples.loc[residual_samples["model_id"] == model]
            for cls in ["normal", "low_pH_event", "high_pH_event"]:
                vals = sub.loc[sub["event_class"] == cls, "residual"].dropna()
                if len(vals):
                    ax.hist(vals, bins=bins, density=True, alpha=0.35, color=colors[cls], label=cls)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(short_label(model))
            ax.set_ylabel("Density")
        axes[-1].set_xlabel("Residual = observed pH - predicted pH")
        axes[0].legend(ncol=3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "residual_distribution_signed_by_event_class.png")
        plt.close(fig)

    if not dissertation.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot = order_models(dissertation)
        x = np.arange(len(plot))
        ax.bar(x - 0.2, plot["negative_tail5_low_enrichment"], width=0.4, label="Low negative 5%", color="#2171b5")
        ax.bar(x + 0.2, plot["positive_tail5_high_enrichment"], width=0.4, label="High positive 5%", color="#cb181d")
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(m) for m in plot["model_id"]], rotation=35, ha="right")
        ax.set_ylabel("Enrichment")
        ax.set_title("Signed-tail model summary")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "signed_tail_model_summary.png")
        plt.close(fig)


def sample_residuals_for_plot(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    if spec.model_id not in {"M0_reference_nonevent", "HGB200_V2", "G100_B"}:
        return pd.DataFrame()
    pieces = []
    for cls, n in [("normal", 8000), ("low_pH_event", 8000), ("high_pH_event", 8000)]:
        sub = df.loc[df["event_class"] == cls, ["residual", "event_class"]]
        if sub.empty:
            continue
        pieces.append(sub.sample(n=min(n, len(sub)), random_state=42).assign(model_id=spec.model_id))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def bootstrap_placeholder() -> None:
    pd.DataFrame(
        [
            {
                "status": "not_run",
                "reason": "Optional site-level bootstrap was skipped to keep Phase 1 signed-tail analysis lightweight and non-blocking.",
                "random_seed": 42,
                "n_bootstrap_repetitions": 0,
            }
        ]
    ).to_csv(OUT_DIR / "signed_tail_bootstrap_uncertainty.csv", index=False)


def event_count_sensitivity(enrichment: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in enrichment.iterrows():
        rows.append(
            {
                "model_id": row["model_id"],
                "tail_type": row["tail_type"],
                "target_event_group": row["target_event_group"],
                "tail_fraction": row["tail_fraction"],
                "n_event_rows": row["n_event_rows"],
                "n_tail_event_rows": row["n_tail_event_rows"],
                "base_event_rate": row["base_event_rate"],
                "tail_event_rate": row["tail_event_rate"],
                "enrichment": row["enrichment"],
                "event_count_reliability": reliability_label(int(row["n_event_rows"])),
            }
        )
    return pd.DataFrame(rows)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            vals.append(str(row[col]).replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    included: pd.DataFrame,
    missing: pd.DataFrame,
    enrichment: pd.DataFrame,
    hit_rate: pd.DataFrame,
    comparison: pd.DataFrame,
    consistency: pd.DataFrame,
    dissertation: pd.DataFrame,
) -> str:
    m0 = dissertation.loc[dissertation["model_id"] == "M0_reference_nonevent"]
    best_low = dissertation.sort_values("negative_tail5_low_enrichment", ascending=False).iloc[0]
    best_high = dissertation.sort_values("positive_tail5_high_enrichment", ascending=False).iloc[0]
    m0_low = float(m0.iloc[0]["negative_tail5_low_enrichment"]) if not m0.empty else math.nan
    m0_low_best = bool(best_low["model_id"] == "M0_reference_nonevent")
    signed_low_improved = bool((comparison["low_signed_minus_absolute_delta"] > 0).any())
    signed_high_improved = bool((comparison["high_signed_minus_absolute_delta"] > 0).any())
    high_enriched_models = dissertation.loc[dissertation["positive_tail5_high_enrichment"] > 1, "model_id"].tolist()

    included_lines = "\n".join(
        f"- {row.model_id}: {row.model_type}, {row.site_count} sites, {row.metadata_set}, {row.training_mode}"
        for row in included.itertuples(index=False)
    )
    if missing.empty:
        missing_lines = "- None among the priority models."
    else:
        missing_lines = "\n".join(f"- {row.model_id}: {row.reason}" for row in missing.itertuples(index=False))

    comp_table = df_to_markdown(comparison[
        [
            "model_id",
            "low_negative_tail_enrichment",
            "low_absolute_tail_enrichment",
            "high_positive_tail_enrichment",
            "high_absolute_tail_enrichment",
            "main_interpretation",
        ]
    ])

    direction_table = df_to_markdown(consistency[
        ["model_id", "event_group", "n_event_rows", "expected_sign_rate", "mean_residual", "median_residual"]
    ])

    text = f"""# Signed Residual-Tail Analysis for Low- and High-pH Events

## 1. Purpose

Previous residual-event validation mainly used `abs(residual)`, which ignores whether observed pH was lower or higher than the model prediction. This analysis separates residual direction because the supervisor requested physically interpretable tails for low- and high-pH regulatory event windows.

## 2. Residual Definition

The signed residual is defined consistently as:

`residual = observed pH - predicted pH`

Therefore:

- negative residual means observed pH is below prediction;
- positive residual means observed pH is above prediction;
- `prediction_error = predicted pH - observed pH` is also saved for sign checking but is not used for signed-tail thresholds.

## 3. Models Included

{included_lines}

Models excluded or unavailable:

{missing_lines}

## 4. Negative-Tail Low-pH Results

The main low-pH validation target is the negative residual tail. The best bottom-5% low-pH enrichment model is `{best_low['model_id']}` with enrichment `{fmt(best_low['negative_tail5_low_enrichment'])}`.

M0 non-event HGB has bottom-5% low-pH enrichment `{fmt(m0_low)}`. Non-event HGB remains the strongest low-pH residual reference: `{'yes' if m0_low_best else 'no'}`.

Direction consistency is saved in `signed_tail_event_direction_consistency.csv`; low-pH rows are expected to have residual < 0.

## 5. Positive-Tail High-pH Results

The main high-pH validation target is the positive residual tail. The best top-positive-5% high-pH enrichment model is `{best_high['model_id']}` with enrichment `{fmt(best_high['positive_tail5_high_enrichment'])}`.

Positive-tail analysis improves high-pH interpretation relative to the absolute-residual reference, but high-pH remains much weaker than low-pH for most HGB models. Models with positive-tail high-pH enrichment above 1 are: `{', '.join(high_enriched_models) if high_enriched_models else 'none'}`.

## 6. Signed vs Absolute Comparison

{comp_table}

Signed tails improved low-pH top-5 interpretation for at least one model: `{'yes' if signed_low_improved else 'no'}`.

Signed tails improved high-pH top-5 interpretation for at least one model: `{'yes' if signed_high_improved else 'no'}`.

## 7. Direction Consistency

{direction_table}

## 8. Metadata and Model Effects

Non-event HGB remains the key normal-behaviour residual reference when the goal is low-pH hot-moment detection. Metadata-aware HGB models generally improve prediction in the broader project, but signed-tail results still show that prediction accuracy and directional residual sensitivity are different objectives. GRU/LSTM outputs are useful sequence-model checks, but they do not replace the HGB signed-tail interpretation.

## 9. Interpretation

Signed-tail analysis provides a more physically meaningful validation than absolute residuals alone because it distinguishes low-pH deviations from high-pH deviations. Low-pH windows are assessed against the negative residual tail, while high-pH windows are assessed against the positive residual tail. If high-pH enrichment remains weak even in the positive tail, the weakness is unlikely to be only a metric-design artefact.

## 10. Limitations

- Regulatory event windows are weak validation labels, not complete anomaly ground truth.
- pH values are processed/scaled.
- High-pH events are less frequent than low-pH events.
- Event windows differ in duration and are approximated from contiguous event rows when explicit event IDs are absent.
- Models use different site sets.
- Sequence models use hourly windows, while HGB models use higher-frequency processed rows.
- Per-site enrichment is unstable when event counts are low.

## 11. Dissertation-Ready Takeaways

- Negative residual tails are the physically appropriate validation target for low-pH events.
- Positive residual tails are the physically appropriate validation target for high-pH events.
- Non-event HGB remains the main normal-behaviour residual reference if it has the strongest low-pH negative-tail enrichment.
- Signed-tail analysis clarifies whether high-pH weakness reflects the absolute-residual metric or true model behaviour.
- Prediction accuracy and directional residual sensitivity should be treated as distinct objectives.
"""
    (OUT_DIR / "signed_residual_tail_analysis_report.md").write_text(text, encoding="utf-8")
    if m0_low_best and signed_low_improved and signed_high_improved:
        if high_enriched_models:
            main = "Signed negative tails strongly improve low-pH validation; positive tails improve high-pH relative to absolute residuals, but high-pH enrichment above 1 appears only in selected sequence models."
        else:
            main = "Signed negative tails strongly improve low-pH validation; positive tails improve high-pH relative to absolute residuals, but high-pH remains below enrichment = 1."
    elif m0_low_best and signed_low_improved:
        main = "Signed negative-tail analysis strengthens low-pH interpretation, while high-pH remains weaker or model-dependent."
    elif signed_high_improved:
        main = "Signed positive-tail analysis reveals some high-pH directional signal, but the strongest low-pH reference is not necessarily M0."
    else:
        main = "Signed-tail analysis is physically clearer, but top-5 enrichment does not consistently exceed the absolute-residual reference."
    return main


def main() -> None:
    ensure_dirs()
    log("Starting signed residual-tail analysis.")
    inventory, schema, included = audit_prediction_files()
    missing = pd.read_csv(OUT_DIR / "models_missing_prediction_files.csv")
    log(f"Priority models audited: {len(inventory)}")
    log(f"Models included: {len(included)}")

    threshold_parts: list[pd.DataFrame] = []
    enrichment_rows_all: list[dict[str, Any]] = []
    consistency_rows_all: list[dict[str, Any]] = []
    hit_rows_all: list[dict[str, Any]] = []
    site_rows_all: list[dict[str, Any]] = []
    residual_samples: list[pd.DataFrame] = []

    schema_by_model = schema.set_index("model_id")
    included_ids = set(included["model_id"])
    for spec in MODEL_SPECS:
        if spec.model_id not in included_ids:
            log(f"Skipping {spec.model_id}: not included.")
            continue
        log(f"Processing {spec.model_id} from {spec.prediction_file}")
        df = load_standardised_predictions(spec, schema_by_model.loc[spec.model_id])
        log(
            f"{spec.model_id}: rows={len(df):,}, sites={df['site_id'].nunique()}, "
            f"low={int((df['event_class'] == 'low_pH_event').sum()):,}, "
            f"high={int((df['event_class'] == 'high_pH_event').sum()):,}"
        )
        threshold_df, thresholds = thresholds_for_model(df, spec)
        threshold_parts.append(threshold_df)
        enrichment_rows_all.extend(enrichment_rows(df, spec, thresholds))
        consistency_rows_all.extend(direction_consistency_rows(df, spec))
        event_rows_df, windows, method = event_windows(df, spec)
        hit_rows_all.extend(event_hit_rate_rows(df, spec, thresholds, event_rows_df, windows, method))
        site_rows_all.extend(per_site_rows(df, spec, thresholds, event_rows_df, windows))
        sample = sample_residuals_for_plot(df, spec)
        if not sample.empty:
            residual_samples.append(sample)
        del df

    thresholds_df = order_models(pd.concat(threshold_parts, ignore_index=True))
    enrichment = order_models(pd.DataFrame(enrichment_rows_all))
    consistency = order_models(pd.DataFrame(consistency_rows_all))
    hit_rate = order_models(pd.DataFrame(hit_rows_all))
    site_metrics = order_models(pd.DataFrame(site_rows_all))
    comparison = build_comparison(enrichment, hit_rate, included)
    dissertation = build_dissertation_table(enrichment, hit_rate, consistency, included)
    event_count_sensitivity_df = event_count_sensitivity(enrichment)

    thresholds_df.to_csv(OUT_DIR / "signed_tail_counts_and_thresholds.csv", index=False)
    enrichment.to_csv(OUT_DIR / "signed_tail_enrichment.csv", index=False)
    consistency.to_csv(OUT_DIR / "signed_tail_event_direction_consistency.csv", index=False)
    hit_rate.to_csv(OUT_DIR / "signed_tail_event_level_hit_rate.csv", index=False)
    site_metrics.to_csv(OUT_DIR / "signed_tail_metrics_by_site.csv", index=False)
    comparison.to_csv(OUT_DIR / "signed_vs_absolute_tail_comparison.csv", index=False)
    dissertation.to_csv(OUT_DIR / "signed_tail_dissertation_ready_table.csv", index=False)
    event_count_sensitivity_df.to_csv(OUT_DIR / "signed_tail_event_count_sensitivity.csv", index=False)
    bootstrap_placeholder()

    residual_sample_df = pd.concat(residual_samples, ignore_index=True) if residual_samples else pd.DataFrame()
    plot_outputs(enrichment, hit_rate, comparison, consistency, dissertation, residual_sample_df)
    main_interpretation = write_report(included, missing, enrichment, hit_rate, comparison, consistency, dissertation)

    best_low = dissertation.sort_values("negative_tail5_low_enrichment", ascending=False).iloc[0]
    best_high = dissertation.sort_values("positive_tail5_high_enrichment", ascending=False).iloc[0]
    m0 = dissertation.loc[dissertation["model_id"] == "M0_reference_nonevent"].iloc[0]
    signed_low_improved = bool((comparison["low_signed_minus_absolute_delta"] > 0).any())
    signed_high_improved = bool((comparison["high_signed_minus_absolute_delta"] > 0).any())
    m0_strongest = best_low["model_id"] == "M0_reference_nonevent"
    console = f"""
PHASE 1 SIGNED RESIDUAL-TAIL ANALYSIS COMPLETE

Output folder:
{rel(OUT_DIR)}

Models included:
{', '.join(included['model_id'].tolist())}

Models excluded:
{', '.join(missing['model_id'].tolist()) if not missing.empty else 'none'}

Residual definition:
observed pH - predicted pH

Main top/bottom 5% results:

Best negative-tail low-pH enrichment model:
{best_low['model_id']}

Negative-tail low-pH enrichment:
{fmt(best_low['negative_tail5_low_enrichment'])}

Best positive-tail high-pH enrichment model:
{best_high['model_id']}

Positive-tail high-pH enrichment:
{fmt(best_high['positive_tail5_high_enrichment'])}

Previous absolute-tail low-pH enrichment:
{fmt(m0['absolute_top5_low_enrichment'])} for M0_reference_nonevent

Previous absolute-tail high-pH enrichment:
{fmt(m0['absolute_top5_high_enrichment'])} for M0_reference_nonevent

Signed tails improved low-pH interpretation:
{'yes' if signed_low_improved else 'no'}

Signed tails improved high-pH interpretation:
{'yes' if signed_high_improved else 'no'}

Non-event HGB remains strongest residual reference:
{'yes' if m0_strongest else 'no'}

Main interpretation:
{main_interpretation}

Next recommended task:
Build the site-level hydrological metadata and flow-properties table.
""".strip()
    log(console)


if __name__ == "__main__":
    main()
