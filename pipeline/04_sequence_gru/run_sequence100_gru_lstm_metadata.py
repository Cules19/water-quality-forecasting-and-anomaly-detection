from __future__ import annotations

import json
import math
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
ANALYSIS_DIR = ROOT / "analysis_outputs"
OUT_DIR = ANALYSIS_DIR / "ph_sequence100_gru_lstm_metadata_v1"
INPUT_DIR = ANALYSIS_DIR / "ph_processed100_step4_0_4_1_v1"
BASE_PARQUET = INPUT_DIR / "processed100_with_event_state.parquet"
METADATA_PARQUET = INPUT_DIR / "processed100_with_event_state_metadata.parquet"
METADATA_FEATURE_SPEC = INPUT_DIR / "processed100_metadata_feature_set.json"

TARGET = "00400"
SITE_COL = "site_id"
TIME_COL = "timestamp"
SPLIT_COL = "split"
EVENT_COL = "in_ph_event"
EVENT_LABEL_COL = "ph_event_label"
EVENT_STATE_COL = "event_state"
SEQUENCE_LENGTH = 24
RANDOM_SEED = 42

SENSOR_FEATURES = ["00010", "00095", "63680", "00060", "00065"]
DYNAMIC_FEATURES = [
    "00010",
    "00095",
    "63680",
    "00060",
    "00065",
    "sin_hour",
    "cos_hour",
    "sin_day",
    "cos_day",
    "month",
    "hour",
    "day_of_year",
]
REQUIRED_BASE_COLUMNS = [
    SITE_COL,
    TIME_COL,
    SPLIT_COL,
    TARGET,
    *SENSOR_FEATURES,
    "sin_hour",
    "cos_hour",
    "sin_day",
    "cos_day",
    "month",
    "hour",
    "day_of_year",
    EVENT_COL,
    EVENT_LABEL_COL,
    EVENT_STATE_COL,
]
NUMERIC_AGG_COLUMNS = [TARGET, *SENSOR_FEATURES]
FINAL_NON_MISSING_COLUMNS = [TARGET, *DYNAMIC_FEATURES, SITE_COL, TIME_COL, SPLIT_COL]
SPLIT_VALUES = ["train", "val", "test"]
TOP_FRACTIONS = [0.01, 0.05, 0.10]

BATCH_ROWS = 1_000_000
SCALER_CHUNK_WINDOWS = 25_000
PREDICT_BATCH_SIZE = 4096
TRAIN_BATCH_SIZE = 2048
MAX_TRAIN_WINDOWS_CPU = 1_000_000
EARLY_STOP_VAL_WINDOWS_CPU = 500_000
MAX_EPOCHS_CPU = 8
EARLY_STOPPING_PATIENCE = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_SIZE = 64
NUM_LAYERS = 1
MLP_HIDDEN_SIZE = 64
DROPOUT = 0.0

MODEL_SPECS = [
    {
        "model_id": "G100_A",
        "model_name": "G100_A_gru_no_metadata",
        "recurrent_type": "gru",
        "metadata_used": False,
        "history_file": "training_history_G100_A_gru_no_metadata.csv",
        "model_file": "model_G100_A_gru_no_metadata.pt",
        "val_prediction_file": "prediction_val_G100_A_gru_no_metadata.parquet",
        "test_prediction_file": "prediction_test_G100_A_gru_no_metadata.parquet",
    },
    {
        "model_id": "G100_B",
        "model_name": "G100_B_gru_with_metadata",
        "recurrent_type": "gru",
        "metadata_used": True,
        "history_file": "training_history_G100_B_gru_with_metadata.csv",
        "model_file": "model_G100_B_gru_with_metadata.pt",
        "val_prediction_file": "prediction_val_G100_B_gru_with_metadata.parquet",
        "test_prediction_file": "prediction_test_G100_B_gru_with_metadata.parquet",
    },
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message: str) -> None:
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_stdout(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, handle)
        sys.stderr = Tee(original_stderr, handle)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def log(message: str) -> None:
    print(message, flush=True)


def normalise_site_id(value: object) -> str:
    raw = str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    return raw.zfill(8) if raw.isdigit() else raw


def normalise_split(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"validation", "valid"}:
        return "val"
    if text in {"testing"}:
        return "test"
    return text


def normalise_event_label(value: object) -> str:
    text = str(value).strip().lower()
    if "low" in text:
        return "pH_exceedance_low"
    if "high" in text:
        return "pH_exceedance_high"
    if "mixed" in text or "unknown" in text:
        return "pH_exceedance_mixed"
    return "none"


def compact_label(value: object) -> str:
    label = normalise_event_label(value)
    if "low" in label:
        return "low"
    if "high" in label:
        return "high"
    if "mixed" in label:
        return "mixed"
    return "none"


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
        else:
            display[col] = display[col].astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.to_numpy()]
    return "\n".join([header, separator, *body])


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")


def ensure_inputs() -> dict[str, object]:
    log("[INFO] Checking Phase 4 inputs and schemas")
    for path in [BASE_PARQUET, METADATA_PARQUET, METADATA_FEATURE_SPEC]:
        if not path.exists():
            raise FileNotFoundError(path)
    base_pf = pq.ParquetFile(BASE_PARQUET)
    metadata_pf = pq.ParquetFile(METADATA_PARQUET)
    base_cols = set(base_pf.schema_arrow.names)
    metadata_cols = set(metadata_pf.schema_arrow.names)
    missing_base = [col for col in REQUIRED_BASE_COLUMNS if col not in base_cols]
    if missing_base:
        raise RuntimeError(f"Missing required base columns: {missing_base}")

    spec = json.loads(METADATA_FEATURE_SPEC.read_text(encoding="utf-8"))
    metadata_features = list(spec.get("metadata_features_used", []))
    missing_meta = [col for col in metadata_features if col not in metadata_cols]
    if missing_meta:
        raise RuntimeError(f"Missing metadata features in metadata parquet: {missing_meta}")

    forbidden_exact = {
        TARGET,
        EVENT_COL,
        EVENT_LABEL_COL,
        EVENT_STATE_COL,
        "event_state_name",
        "sample_weight_first_version",
        "residual",
        "prediction",
        "y_pred",
    }
    forbidden_keyword = [
        "event",
        "incident",
        "exceedance",
        "residual",
        "prediction",
        "00400",
        "ph_mean",
        "ph_std",
        "target",
    ]
    dynamic_forbidden = sorted(set(DYNAMIC_FEATURES).intersection(forbidden_exact))
    metadata_forbidden = [
        col
        for col in metadata_features
        if col in forbidden_exact or any(keyword in col.lower() for keyword in forbidden_keyword)
    ]
    if TARGET in DYNAMIC_FEATURES or dynamic_forbidden or metadata_forbidden:
        raise RuntimeError(
            f"Leakage feature check failed. dynamic={dynamic_forbidden}; metadata={metadata_forbidden[:20]}"
        )

    return {
        "base_rows": int(base_pf.metadata.num_rows),
        "base_row_groups": int(base_pf.metadata.num_row_groups),
        "metadata_rows": int(metadata_pf.metadata.num_rows),
        "metadata_row_groups": int(metadata_pf.metadata.num_row_groups),
        "metadata_features": metadata_features,
        "metadata_spec": spec,
    }


def aggregate_batch_to_hourly_parts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[SITE_COL] = df[SITE_COL].map(normalise_site_id).astype("string")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df[SPLIT_COL] = df[SPLIT_COL].map(normalise_split).astype("string")
    df = df.dropna(subset=[SITE_COL, TIME_COL])
    df["hour_timestamp"] = df[TIME_COL].dt.floor("h")
    df = df.dropna(subset=["hour_timestamp"])

    for col in NUMERIC_AGG_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    labels = df[EVENT_LABEL_COL].map(compact_label).astype("string")
    event_state = pd.to_numeric(df[EVENT_STATE_COL], errors="coerce").fillna(0).astype("int16")
    in_event = pd.to_numeric(df[EVENT_COL], errors="coerce").fillna(0).astype("int8")
    df["_event_flag"] = ((in_event > 0) | event_state.gt(0) | labels.isin(["low", "high", "mixed"])).astype("int8")
    df["_low_flag"] = (labels.eq("low") | event_state.eq(1)).astype("int8")
    df["_high_flag"] = (labels.eq("high") | event_state.eq(2)).astype("int8")
    df["_mixed_flag"] = (labels.eq("mixed") | event_state.eq(3)).astype("int8")
    for split in SPLIT_VALUES:
        df[f"_split_count_{split}"] = df[SPLIT_COL].eq(split).astype("int32")

    agg_kwargs = {"n_source_rows": (SITE_COL, "size")}
    for col in NUMERIC_AGG_COLUMNS:
        agg_kwargs[f"{col}_sum"] = (col, "sum")
        agg_kwargs[f"{col}_count"] = (col, "count")
    for flag in ["_event_flag", "_low_flag", "_high_flag", "_mixed_flag"]:
        agg_kwargs[flag] = (flag, "max")
    for split in SPLIT_VALUES:
        agg_kwargs[f"split_count_{split}"] = (f"_split_count_{split}", "sum")

    out = (
        df.groupby([SITE_COL, "hour_timestamp"], observed=True, sort=True)
        .agg(**agg_kwargs)
        .reset_index()
        .rename(columns={"hour_timestamp": TIME_COL})
    )
    return out


def build_hourly_dataset(input_info: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly_path = OUT_DIR / "sequence100_hourly_rows.parquet"
    if hourly_path.exists():
        log("[INFO] Loading existing hourly rows from this Phase 5 folder")
        hourly = pd.read_parquet(hourly_path)
        summary = make_hourly_summary(hourly, input_info, split_method="loaded_existing")
        summary.to_csv(OUT_DIR / "sequence100_hourly_dataset_summary.csv", index=False, encoding="utf-8-sig")
        return hourly, summary

    log("[INFO] Building hourly aggregate parts from processed100 base parquet")
    temp_path = OUT_DIR / "_sequence100_hourly_aggregate_parts.tmp.parquet"
    if temp_path.exists():
        temp_path.unlink()
    pf = pq.ParquetFile(BASE_PARQUET)
    writer: pq.ParquetWriter | None = None
    total_parts = 0
    started = perf_counter()
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=BATCH_ROWS, columns=REQUIRED_BASE_COLUMNS), start=1):
        batch_df = batch.to_pandas()
        agg = aggregate_batch_to_hourly_parts(batch_df)
        total_parts += len(agg)
        table = pa.Table.from_pandas(agg, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(temp_path, table.schema, compression="snappy")
        writer.write_table(table)
        log(
            f"[INFO] Hourly part batch {batch_idx}/{pf.metadata.num_row_groups}: "
            f"source_rows={len(batch_df):,} hourly_parts={len(agg):,}"
        )
    if writer is not None:
        writer.close()
    if not temp_path.exists():
        raise RuntimeError("Hourly aggregate part file was not created.")

    log(f"[INFO] Finalising hourly dataset from {total_parts:,} hourly aggregate parts")
    parts = pd.read_parquet(temp_path)
    parts[SITE_COL] = parts[SITE_COL].map(normalise_site_id).astype("string")
    parts[TIME_COL] = pd.to_datetime(parts[TIME_COL], errors="coerce")
    agg_kwargs = {"n_source_rows": ("n_source_rows", "sum")}
    for col in NUMERIC_AGG_COLUMNS:
        agg_kwargs[f"{col}_sum"] = (f"{col}_sum", "sum")
        agg_kwargs[f"{col}_count"] = (f"{col}_count", "sum")
    for flag in ["_event_flag", "_low_flag", "_high_flag", "_mixed_flag"]:
        agg_kwargs[flag] = (flag, "max")
    for split in SPLIT_VALUES:
        agg_kwargs[f"split_count_{split}"] = (f"split_count_{split}", "sum")

    grouped = (
        parts.groupby([SITE_COL, TIME_COL], observed=True, sort=True)
        .agg(**agg_kwargs)
        .reset_index()
    )
    del parts

    hourly = grouped[[SITE_COL, TIME_COL, "n_source_rows"]].copy()
    for col in NUMERIC_AGG_COLUMNS:
        counts = grouped[f"{col}_count"].replace(0, np.nan)
        hourly[col] = grouped[f"{col}_sum"] / counts

    split_counts = grouped[[f"split_count_{split}" for split in SPLIT_VALUES]].to_numpy(dtype="int64")
    split_idx = split_counts.argmax(axis=1)
    split_total_nonzero = (split_counts > 0).sum(axis=1)
    hourly[SPLIT_COL] = np.asarray(SPLIT_VALUES, dtype=object)[split_idx]
    hourly["mixed_source_split_hour"] = (split_total_nonzero > 1).astype("int8")

    low = grouped["_low_flag"].astype(bool).to_numpy()
    high = grouped["_high_flag"].astype(bool).to_numpy()
    mixed = grouped["_mixed_flag"].astype(bool).to_numpy()
    any_event = grouped["_event_flag"].astype(bool).to_numpy() | low | high | mixed
    event_state = np.zeros(len(hourly), dtype="int8")
    event_state[high] = 2
    event_state[low] = 1
    event_state[(low & high) | mixed] = 3
    labels = np.full(len(hourly), "none", dtype=object)
    labels[event_state == 1] = "pH_exceedance_low"
    labels[event_state == 2] = "pH_exceedance_high"
    labels[event_state == 3] = "pH_exceedance_mixed"
    hourly[EVENT_COL] = any_event.astype("int8")
    hourly[EVENT_LABEL_COL] = labels
    hourly[EVENT_STATE_COL] = event_state

    timestamp = pd.to_datetime(hourly[TIME_COL], errors="coerce")
    hourly["hour"] = timestamp.dt.hour.astype("int16")
    hourly["month"] = timestamp.dt.month.astype("int16")
    hourly["day_of_year"] = timestamp.dt.dayofyear.astype("int16")
    hourly["sin_hour"] = np.sin(2 * np.pi * hourly["hour"].astype("float64") / 24.0)
    hourly["cos_hour"] = np.cos(2 * np.pi * hourly["hour"].astype("float64") / 24.0)
    hourly["sin_day"] = np.sin(2 * np.pi * hourly["day_of_year"].astype("float64") / 365.25)
    hourly["cos_day"] = np.cos(2 * np.pi * hourly["day_of_year"].astype("float64") / 365.25)

    before_drop = len(hourly)
    hourly = hourly.dropna(subset=FINAL_NON_MISSING_COLUMNS).copy()
    hourly = hourly.sort_values([SITE_COL, TIME_COL], kind="mergesort").reset_index(drop=True)
    hourly.insert(0, "hourly_row_idx", np.arange(len(hourly), dtype="int64"))
    duplicate_keys = int(hourly.duplicated([SITE_COL, TIME_COL]).sum())
    if duplicate_keys:
        raise RuntimeError(f"Duplicate hourly site/timestamp keys after final aggregation: {duplicate_keys}")

    column_order = [
        "hourly_row_idx",
        SITE_COL,
        TIME_COL,
        SPLIT_COL,
        TARGET,
        *DYNAMIC_FEATURES,
        EVENT_COL,
        EVENT_LABEL_COL,
        EVENT_STATE_COL,
        "n_source_rows",
        "mixed_source_split_hour",
    ]
    hourly = hourly[column_order]
    write_parquet(hourly, hourly_path)
    temp_path.unlink(missing_ok=True)
    log(
        f"[INFO] Hourly dataset complete: rows={len(hourly):,}; "
        f"dropped_missing_hourly_rows={before_drop - len(hourly):,}; "
        f"elapsed_seconds={perf_counter() - started:.1f}"
    )

    summary = make_hourly_summary(hourly, input_info, split_method="majority_source_row_split_per_site_hour")
    summary.to_csv(OUT_DIR / "sequence100_hourly_dataset_summary.csv", index=False, encoding="utf-8-sig")
    return hourly, summary


def make_hourly_summary(hourly: pd.DataFrame, input_info: dict[str, object], split_method: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    timestamp = pd.to_datetime(hourly[TIME_COL], errors="coerce")
    labels = hourly[EVENT_LABEL_COL].map(compact_label)
    rows.append(
        {
            "scope": "overall",
            "split": "all",
            "original_rows": int(input_info["base_rows"]),
            "hourly_rows": int(len(hourly)),
            "sites": int(hourly[SITE_COL].nunique()),
            "train_hourly_rows": int(hourly[SPLIT_COL].eq("train").sum()),
            "val_hourly_rows": int(hourly[SPLIT_COL].eq("val").sum()),
            "test_hourly_rows": int(hourly[SPLIT_COL].eq("test").sum()),
            "event_hourly_rows": int(hourly[EVENT_COL].sum()),
            "low_event_hourly_rows": int(labels.eq("low").sum()),
            "high_event_hourly_rows": int(labels.eq("high").sum()),
            "mixed_event_hourly_rows": int(labels.eq("mixed").sum()),
            "timestamp_min": timestamp.min(),
            "timestamp_max": timestamp.max(),
            "mixed_source_split_hours": int(hourly["mixed_source_split_hour"].sum()) if "mixed_source_split_hour" in hourly.columns else np.nan,
            "split_method": split_method,
        }
    )
    for split, group in hourly.groupby(SPLIT_COL, sort=True, observed=True):
        group_labels = group[EVENT_LABEL_COL].map(compact_label)
        group_ts = pd.to_datetime(group[TIME_COL], errors="coerce")
        rows.append(
            {
                "scope": "split",
                "split": split,
                "original_rows": int(input_info["base_rows"]),
                "hourly_rows": int(len(group)),
                "sites": int(group[SITE_COL].nunique()),
                "train_hourly_rows": int(len(group)) if split == "train" else 0,
                "val_hourly_rows": int(len(group)) if split == "val" else 0,
                "test_hourly_rows": int(len(group)) if split == "test" else 0,
                "event_hourly_rows": int(group[EVENT_COL].sum()),
                "low_event_hourly_rows": int(group_labels.eq("low").sum()),
                "high_event_hourly_rows": int(group_labels.eq("high").sum()),
                "mixed_event_hourly_rows": int(group_labels.eq("mixed").sum()),
                "timestamp_min": group_ts.min(),
                "timestamp_max": group_ts.max(),
                "mixed_source_split_hours": int(group["mixed_source_split_hour"].sum()) if "mixed_source_split_hour" in group.columns else np.nan,
                "split_method": split_method,
            }
        )
    return pd.DataFrame(rows)


def create_window_index(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_path = OUT_DIR / "sequence100_window_index.parquet"
    if window_path.exists():
        log("[INFO] Loading existing 24h window index from this Phase 5 folder")
        windows = pd.read_parquet(window_path)
        summary = make_window_summary(windows, hourly, sampling_info={})
        summary.to_csv(OUT_DIR / "sequence100_window_dataset_summary.csv", index=False, encoding="utf-8-sig")
        return windows, summary

    log("[INFO] Building strict 24-hour window index")
    rows: list[pd.DataFrame] = []
    candidate_windows = 0
    invalid_time_gap = 0
    invalid_split_crossing = 0
    one_hour = np.timedelta64(1, "h")

    for site_id, group in hourly.groupby(SITE_COL, sort=True, observed=True):
        group = group.sort_values(TIME_COL).reset_index(drop=True)
        n = len(group)
        if n < SEQUENCE_LENGTH:
            continue
        candidate_windows += n - SEQUENCE_LENGTH + 1
        ts = group[TIME_COL].to_numpy(dtype="datetime64[ns]")
        diffs = np.diff(ts)
        diff_ok = np.r_[False, diffs == one_hour]
        continuous = pd.Series(diff_ok.astype("int16")).rolling(SEQUENCE_LENGTH - 1).sum().to_numpy() == SEQUENCE_LENGTH - 1

        split_values = group[SPLIT_COL].astype(str).to_numpy()
        split_same_prev = np.r_[False, split_values[1:] == split_values[:-1]]
        same_split = pd.Series(split_same_prev.astype("int16")).rolling(SEQUENCE_LENGTH - 1).sum().to_numpy() == SEQUENCE_LENGTH - 1

        target_positions = np.arange(SEQUENCE_LENGTH - 1, n, dtype="int64")
        continuous_target = continuous[target_positions]
        same_split_target = same_split[target_positions]
        valid = continuous_target & same_split_target
        invalid_time_gap += int((~continuous_target).sum())
        invalid_split_crossing += int((continuous_target & ~same_split_target).sum())
        if not valid.any():
            continue

        target_pos = target_positions[valid]
        start_pos = target_pos - SEQUENCE_LENGTH + 1
        labels = group[EVENT_LABEL_COL].map(compact_label).astype(str).to_numpy()
        low_roll = pd.Series((labels == "low").astype("int16")).rolling(SEQUENCE_LENGTH).sum().to_numpy()
        high_roll = pd.Series((labels == "high").astype("int16")).rolling(SEQUENCE_LENGTH).sum().to_numpy()
        event_roll = pd.Series(group[EVENT_COL].astype("int16")).rolling(SEQUENCE_LENGTH).sum().to_numpy()
        low_count = low_roll[target_pos].astype("int16")
        high_count = high_roll[target_pos].astype("int16")
        event_count = event_roll[target_pos].astype("int16")
        summary_label = np.full(len(target_pos), "none", dtype=object)
        summary_label[(low_count > 0) & (high_count == 0)] = "pH_exceedance_low"
        summary_label[(high_count > 0) & (low_count == 0)] = "pH_exceedance_high"
        summary_label[(low_count > 0) & (high_count > 0)] = "pH_exceedance_mixed"

        part = pd.DataFrame(
            {
                SITE_COL: str(site_id),
                "target_timestamp": ts[target_pos],
                SPLIT_COL: split_values[target_pos],
                "target_value": group[TARGET].to_numpy(dtype="float32")[target_pos],
                EVENT_COL: group[EVENT_COL].to_numpy(dtype="int8")[target_pos],
                EVENT_LABEL_COL: group[EVENT_LABEL_COL].astype(str).to_numpy()[target_pos],
                EVENT_STATE_COL: group[EVENT_STATE_COL].to_numpy(dtype="int8")[target_pos],
                "window_start_timestamp": ts[start_pos],
                "window_end_timestamp": ts[target_pos],
                "n_steps": SEQUENCE_LENGTH,
                "has_any_event_in_window": event_count > 0,
                "window_event_label_summary": summary_label,
                "hourly_row_start_idx": group["hourly_row_idx"].to_numpy(dtype="int64")[start_pos],
                "hourly_row_end_idx": group["hourly_row_idx"].to_numpy(dtype="int64")[target_pos],
            }
        )
        rows.append(part)

    windows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if windows.empty:
        raise RuntimeError("No valid 24-hour windows were created.")
    windows.insert(0, "window_id", np.arange(1, len(windows) + 1, dtype="int64"))
    write_parquet(windows, window_path)
    audit = pd.DataFrame(
        [
            {
                "n_candidate_windows": int(candidate_windows),
                "n_valid_windows": int(len(windows)),
                "n_invalid_due_to_time_gap": int(invalid_time_gap),
                "n_invalid_due_to_split_crossing": int(invalid_split_crossing),
                "strict_hourly_continuity": True,
                "fallback_max_gap_2h_used": False,
            }
        ]
    )
    audit.to_csv(OUT_DIR / "sequence100_window_validity_audit.csv", index=False, encoding="utf-8-sig")
    validate_window_positions(hourly, windows)
    summary = make_window_summary(windows, hourly, sampling_info={})
    summary.to_csv(OUT_DIR / "sequence100_window_dataset_summary.csv", index=False, encoding="utf-8-sig")
    log(f"[INFO] Window index complete: valid_windows={len(windows):,}")
    return windows, summary


def validate_window_positions(hourly: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    log("[INFO] Validating window row pointers")
    row_idx = hourly["hourly_row_idx"].to_numpy(dtype="int64")
    if not np.array_equal(row_idx, np.arange(len(hourly), dtype="int64")):
        raise RuntimeError("hourly_row_idx is not contiguous.")
    start = windows["hourly_row_start_idx"].to_numpy(dtype="int64")
    end = windows["hourly_row_end_idx"].to_numpy(dtype="int64")
    hourly_time = hourly[TIME_COL].to_numpy(dtype="datetime64[ns]")
    hourly_site = hourly[SITE_COL].astype(str).to_numpy()
    hourly_split = hourly[SPLIT_COL].astype(str).to_numpy()
    window_site = windows[SITE_COL].astype(str).to_numpy()
    window_split = windows[SPLIT_COL].astype(str).to_numpy()
    target_time = windows["target_timestamp"].to_numpy(dtype="datetime64[ns]")
    start_time = windows["window_start_timestamp"].to_numpy(dtype="datetime64[ns]")
    offsets = np.arange(SEQUENCE_LENGTH, dtype="int64")
    counters = {
        "end_minus_start_is_23": int((end - start != SEQUENCE_LENGTH - 1).sum()),
        "first_timestamp_matches_window_start": int((hourly_time[start] != start_time).sum()),
        "last_timestamp_matches_target": int((hourly_time[end] != target_time).sum()),
        "start_site_matches_window_site": int((hourly_site[start] != window_site).sum()),
        "end_site_matches_window_site": int((hourly_site[end] != window_site).sum()),
        "start_split_matches_window_split": int((hourly_split[start] != window_split).sum()),
        "end_split_matches_window_split": int((hourly_split[end] != window_split).sum()),
        "internal_non_hourly_steps": 0,
        "internal_cross_site_windows": 0,
        "internal_cross_split_windows": 0,
    }
    one_hour = np.timedelta64(1, "h")
    chunk = 250_000
    for lo in range(0, len(windows), chunk):
        hi = min(lo + chunk, len(windows))
        pos = start[lo:hi, None] + offsets[None, :]
        times = hourly_time[pos]
        sites = hourly_site[pos]
        splits = hourly_split[pos]
        counters["internal_non_hourly_steps"] += int((np.diff(times, axis=1) != one_hour).sum())
        counters["internal_cross_site_windows"] += int((sites != window_site[lo:hi, None]).any(axis=1).sum())
        counters["internal_cross_split_windows"] += int((splits != window_split[lo:hi, None]).any(axis=1).sum())

    checks = pd.DataFrame(
        [
            {
                "check_name": key,
                "status": "PASS" if value == 0 else "FAIL",
                "value": int(value),
            }
            for key, value in counters.items()
        ]
    )
    checks.to_csv(OUT_DIR / "sequence100_window_position_checks.csv", index=False, encoding="utf-8-sig")
    if checks["status"].eq("FAIL").any():
        raise RuntimeError("Window row pointer validation failed.")
    return checks


def make_window_summary(
    windows: pd.DataFrame,
    hourly: pd.DataFrame,
    sampling_info: dict[str, object],
) -> pd.DataFrame:
    labels = windows[EVENT_LABEL_COL].map(compact_label)
    rows = [
        {
            "scope": "overall",
            "split": "all",
            "total_valid_windows": int(len(windows)),
            "train_windows": int(windows[SPLIT_COL].eq("train").sum()),
            "val_windows": int(windows[SPLIT_COL].eq("val").sum()),
            "test_windows": int(windows[SPLIT_COL].eq("test").sum()),
            "event_target_windows": int(windows[EVENT_COL].sum()),
            "low_event_target_windows": int(labels.eq("low").sum()),
            "high_event_target_windows": int(labels.eq("high").sum()),
            "mixed_event_target_windows": int(labels.eq("mixed").sum()),
            "sites_retained": int(windows[SITE_COL].nunique()),
            "sites_dropped_due_to_insufficient_windows": int(hourly[SITE_COL].nunique() - windows[SITE_COL].nunique()),
            "average_windows_per_site": float(len(windows) / windows[SITE_COL].nunique()),
            **sampling_info,
        }
    ]
    for split, group in windows.groupby(SPLIT_COL, sort=True, observed=True):
        group_labels = group[EVENT_LABEL_COL].map(compact_label)
        rows.append(
            {
                "scope": "split",
                "split": split,
                "total_valid_windows": int(len(group)),
                "train_windows": int(len(group)) if split == "train" else 0,
                "val_windows": int(len(group)) if split == "val" else 0,
                "test_windows": int(len(group)) if split == "test" else 0,
                "event_target_windows": int(group[EVENT_COL].sum()),
                "low_event_target_windows": int(group_labels.eq("low").sum()),
                "high_event_target_windows": int(group_labels.eq("high").sum()),
                "mixed_event_target_windows": int(group_labels.eq("mixed").sum()),
                "sites_retained": int(group[SITE_COL].nunique()),
                "sites_dropped_due_to_insufficient_windows": np.nan,
                "average_windows_per_site": float(len(group) / group[SITE_COL].nunique()),
                **sampling_info,
            }
        )
    return pd.DataFrame(rows)


def load_metadata_from_metadata_parquet(metadata_features: list[str], required_sites: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    log("[INFO] Extracting static metadata from processed100_with_event_state_metadata.parquet")
    required_set = set(required_sites)
    found: dict[str, dict[str, object]] = {}
    pf = pq.ParquetFile(METADATA_PARQUET)
    columns = [SITE_COL, *metadata_features]
    batches_scanned = 0
    rows_scanned = 0
    for batch in pf.iter_batches(batch_size=250_000, columns=columns):
        batches_scanned += 1
        df = batch.to_pandas()
        rows_scanned += len(df)
        df[SITE_COL] = df[SITE_COL].map(normalise_site_id).astype(str)
        df = df[df[SITE_COL].isin(required_set)]
        if df.empty:
            continue
        for _, row in df.drop_duplicates(SITE_COL, keep="first").iterrows():
            site_id = str(row[SITE_COL])
            if site_id not in found:
                found[site_id] = {col: row[col] for col in metadata_features}
        if len(found) == len(required_set):
            break
    missing = sorted(required_set - set(found))
    if missing:
        raise RuntimeError(f"Metadata parquet did not provide all required sites: {missing[:10]}")
    meta = pd.DataFrame([{SITE_COL: site, **found[site]} for site in sorted(found)])
    for col in metadata_features:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    extraction_info = {
        "metadata_source": str(METADATA_PARQUET.relative_to(ROOT)),
        "metadata_extraction_strategy": "first static row per site from metadata parquet",
        "metadata_batches_scanned": int(batches_scanned),
        "metadata_rows_scanned": int(rows_scanned),
        "metadata_sites_extracted": int(len(meta)),
    }
    return meta, extraction_info


def build_arrays_and_encodings(
    hourly: pd.DataFrame,
    windows: pd.DataFrame,
    metadata_features: list[str],
) -> dict[str, object]:
    log("[INFO] Building in-memory modelling arrays")
    feature_matrix = hourly[DYNAMIC_FEATURES].to_numpy(dtype="float32")
    if not np.isfinite(feature_matrix).all():
        raise RuntimeError("Non-finite dynamic feature values detected before scaling.")
    start_pos = windows["hourly_row_start_idx"].to_numpy(dtype="int64")
    target_values = windows["target_value"].to_numpy(dtype="float32")
    train_mask = windows[SPLIT_COL].eq("train").to_numpy()
    train_sites = sorted(windows.loc[train_mask, SITE_COL].astype(str).unique().tolist())
    all_sites = sorted(windows[SITE_COL].astype(str).unique().tolist())
    missing_train_sites = sorted(set(all_sites) - set(train_sites))
    if missing_train_sites:
        raise RuntimeError(f"Val/test sites absent from train site encoding: {missing_train_sites[:10]}")
    site_mapping = {site_id: idx for idx, site_id in enumerate(train_sites)}
    site_codes = windows[SITE_COL].astype(str).map(site_mapping).to_numpy(dtype="int64")
    embedding_dim = min(16, int(math.ceil(math.sqrt(len(site_mapping)))) + 1)
    site_encoding = {
        "site_to_code": site_mapping,
        "n_sites": int(len(site_mapping)),
        "embedding_dim": int(embedding_dim),
        "encoding_fit_on": "train window sites only",
        "all_val_test_sites_seen_in_train": True,
    }
    (OUT_DIR / "sequence100_site_id_encoding.json").write_text(json.dumps(site_encoding, indent=2), encoding="utf-8")

    meta, extraction_info = load_metadata_from_metadata_parquet(metadata_features, train_sites)
    meta = meta.set_index(SITE_COL).loc[train_sites].reset_index()
    raw_meta = meta[metadata_features].to_numpy(dtype="float32")
    finite_mask = np.isfinite(raw_meta)
    missing_counts = (~finite_mask).sum(axis=0)
    train_mean = np.nanmean(np.where(finite_mask, raw_meta, np.nan), axis=0)
    train_mean = np.where(np.isfinite(train_mean), train_mean, 0.0).astype("float32")
    filled_meta = np.where(finite_mask, raw_meta, train_mean[None, :]).astype("float32")
    meta_scaler = StandardScaler()
    metadata_scaled = meta_scaler.fit_transform(filled_meta).astype("float32")
    metadata_scaler_info = {
        **extraction_info,
        "metadata_features": metadata_features,
        "n_metadata_features": int(len(metadata_features)),
        "fit_scope": "train sites only, one metadata vector per site",
        "nonfinite_values_filled_with_train_site_mean": {
            col: int(count) for col, count in zip(metadata_features, missing_counts)
        },
        "mean": {col: float(value) for col, value in zip(metadata_features, meta_scaler.mean_)},
        "scale": {col: float(value) for col, value in zip(metadata_features, meta_scaler.scale_)},
    }
    (OUT_DIR / "sequence100_metadata_scaler_info.json").write_text(
        json.dumps(metadata_scaler_info, indent=2), encoding="utf-8"
    )

    return {
        "feature_matrix": feature_matrix,
        "start_pos": start_pos,
        "target_values": target_values,
        "site_codes": site_codes,
        "site_mapping": site_mapping,
        "site_embedding_dim": embedding_dim,
        "metadata_by_site_code": metadata_scaled,
        "metadata_features": metadata_features,
    }


def sample_indices_site_balanced(
    windows: pd.DataFrame,
    indices: np.ndarray,
    cap: int,
    random_seed: int,
    label: str,
) -> tuple[np.ndarray, pd.DataFrame, bool]:
    if len(indices) <= cap:
        summary = (
            windows.iloc[indices]
            .groupby(SITE_COL, observed=True)
            .size()
            .reset_index(name=f"available_{label}_windows")
        )
        summary[f"sampled_{label}_windows"] = summary[f"available_{label}_windows"]
        summary["sample_fraction"] = 1.0
        return indices.copy(), summary, False

    rng = np.random.default_rng(random_seed)
    meta = windows.iloc[indices][[SITE_COL]].copy()
    meta["_row_index"] = indices
    groups = [(str(site), group["_row_index"].to_numpy(dtype="int64")) for site, group in meta.groupby(SITE_COL, sort=True)]
    total = len(indices)
    desired = []
    for site, arr in groups:
        exact = len(arr) / total * cap
        desired.append([site, arr, int(math.floor(exact)), exact - math.floor(exact)])
    allocated = sum(item[2] for item in desired)
    remaining = cap - allocated
    for item in sorted(desired, key=lambda x: x[3], reverse=True):
        if remaining <= 0:
            break
        if item[2] < len(item[1]):
            item[2] += 1
            remaining -= 1

    sampled_parts = []
    rows = []
    for site, arr, n_take, _frac in desired:
        n_take = min(max(1, n_take), len(arr))
        sampled = rng.choice(arr, size=n_take, replace=False)
        sampled.sort()
        sampled_parts.append(sampled)
        rows.append(
            {
                SITE_COL: site,
                f"available_{label}_windows": int(len(arr)),
                f"sampled_{label}_windows": int(len(sampled)),
                "sample_fraction": float(len(sampled) / len(arr)),
            }
        )
    sampled_indices = np.concatenate(sampled_parts)
    sampled_indices.sort()
    return sampled_indices, pd.DataFrame(rows), True


def fit_dynamic_scaler(
    feature_matrix: np.ndarray,
    start_pos: np.ndarray,
    train_sample_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    log("[INFO] Fitting dynamic StandardScaler on sampled train windows only")
    scaler = StandardScaler()
    offsets = np.arange(SEQUENCE_LENGTH, dtype="int64")
    for lo in range(0, len(train_sample_indices), SCALER_CHUNK_WINDOWS):
        hi = min(lo + SCALER_CHUNK_WINDOWS, len(train_sample_indices))
        idx = train_sample_indices[lo:hi]
        pos = start_pos[idx, None] + offsets[None, :]
        scaler.partial_fit(feature_matrix[pos].reshape(-1, len(DYNAMIC_FEATURES)))

    scaled = np.empty_like(feature_matrix, dtype="float32")
    row_chunk = 1_000_000
    for lo in range(0, len(feature_matrix), row_chunk):
        hi = min(lo + row_chunk, len(feature_matrix))
        scaled[lo:hi] = scaler.transform(feature_matrix[lo:hi]).astype("float32")
    if not np.isfinite(scaled).all():
        raise RuntimeError("Non-finite dynamic feature values after scaling.")

    info = {
        "scaler": "StandardScaler",
        "fit_scope": "sampled train windows only",
        "target_scaled": False,
        "target_scaling_decision": "00400 is already processed/scaled in the prepared dataset; target was not double-scaled.",
        "sequence_length": SEQUENCE_LENGTH,
        "dynamic_features": DYNAMIC_FEATURES,
        "n_fit_windows": int(len(train_sample_indices)),
        "n_fit_rows": int(len(train_sample_indices) * SEQUENCE_LENGTH),
        "mean": {col: float(value) for col, value in zip(DYNAMIC_FEATURES, scaler.mean_)},
        "scale": {col: float(value) for col, value in zip(DYNAMIC_FEATURES, scaler.scale_)},
    }
    (OUT_DIR / "sequence100_dynamic_scaler_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return scaled, info


class SequenceRegressor(nn.Module):
    def __init__(
        self,
        recurrent_type: str,
        n_dynamic_features: int,
        n_sites: int,
        site_embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        mlp_hidden_size: int,
        metadata_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.recurrent_type = recurrent_type
        if recurrent_type == "gru":
            self.recurrent = nn.GRU(
                input_size=n_dynamic_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.0 if num_layers == 1 else dropout,
            )
        elif recurrent_type == "lstm":
            self.recurrent = nn.LSTM(
                input_size=n_dynamic_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.0 if num_layers == 1 else dropout,
            )
        else:
            raise ValueError(recurrent_type)
        self.site_embedding = nn.Embedding(n_sites, site_embedding_dim)
        head_input = hidden_size + site_embedding_dim + metadata_dim
        layers: list[nn.Module] = [nn.Linear(head_input, mlp_hidden_size), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(mlp_hidden_size, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, site_code: torch.Tensor, metadata: torch.Tensor | None = None) -> torch.Tensor:
        output = self.recurrent(x)
        hidden = output[1]
        if self.recurrent_type == "lstm":
            hidden = hidden[0]
        final_hidden = hidden[-1]
        site_embedding = self.site_embedding(site_code)
        pieces = [final_hidden, site_embedding]
        if metadata is not None:
            pieces.append(metadata)
        combined = torch.cat(pieces, dim=1)
        return self.head(combined).squeeze(1)


def iter_model_batches(
    indices: np.ndarray,
    scaled_features: np.ndarray,
    start_pos: np.ndarray,
    target_values: np.ndarray,
    site_codes: np.ndarray,
    metadata_by_site_code: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
):
    order = indices.copy()
    if shuffle:
        rng.shuffle(order)
    offsets = np.arange(SEQUENCE_LENGTH, dtype="int64")
    for lo in range(0, len(order), batch_size):
        batch_indices = order[lo : lo + batch_size]
        pos = start_pos[batch_indices, None] + offsets[None, :]
        x = torch.from_numpy(scaled_features[pos])
        y = torch.from_numpy(target_values[batch_indices].astype("float32"))
        site_code_np = site_codes[batch_indices].astype("int64")
        site_code = torch.from_numpy(site_code_np)
        if metadata_by_site_code is not None:
            metadata = torch.from_numpy(metadata_by_site_code[site_code_np])
        else:
            metadata = None
        yield x, site_code, metadata, y


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(y) & np.isfinite(pred)
    n = int(mask.sum())
    if n == 0:
        return {"n_windows": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan}
    y = y[mask]
    pred = pred[mask]
    residual = y - pred
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "n_windows": n,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
    }


def evaluate_model(
    model: nn.Module,
    indices: np.ndarray,
    arrays: dict[str, object],
    metadata_used: bool,
    device: torch.device,
    batch_size: int = PREDICT_BATCH_SIZE,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    loss_sum = 0.0
    n = 0
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(RANDOM_SEED)
    metadata = arrays["metadata_by_site_code"] if metadata_used else None
    with torch.no_grad():
        for x, site_code, meta, y in iter_model_batches(
            indices,
            arrays["scaled_features"],
            arrays["start_pos"],
            arrays["target_values"],
            arrays["site_codes"],
            metadata,
            batch_size=batch_size,
            shuffle=False,
            rng=rng,
        ):
            x = x.to(device)
            site_code = site_code.to(device)
            y = y.to(device)
            meta = meta.to(device) if meta is not None else None
            pred = model(x, site_code, meta)
            loss = loss_fn(pred, y)
            loss_sum += float(loss.item()) * len(y)
            n += len(y)
            preds.append(pred.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
    y_arr = np.concatenate(ys) if ys else np.array([], dtype="float32")
    pred_arr = np.concatenate(preds) if preds else np.array([], dtype="float32")
    return loss_sum / n if n else np.nan, y_arr, pred_arr


def train_sequence_model(
    spec: dict[str, object],
    arrays: dict[str, object],
    train_sample_indices: np.ndarray,
    val_early_indices: np.ndarray,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame, dict[str, object]]:
    model_id = str(spec["model_id"])
    model_name = str(spec["model_name"])
    metadata_used = bool(spec["metadata_used"])
    log(f"[INFO] Training {model_id}: {model_name}")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    metadata_dim = len(arrays["metadata_features"]) if metadata_used else 0
    model = SequenceRegressor(
        recurrent_type=str(spec["recurrent_type"]),
        n_dynamic_features=len(DYNAMIC_FEATURES),
        n_sites=len(arrays["site_mapping"]),
        site_embedding_dim=int(arrays["site_embedding_dim"]),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        mlp_hidden_size=MLP_HIDDEN_SIZE,
        metadata_dim=metadata_dim,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    best_rmse = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    metadata = arrays["metadata_by_site_code"] if metadata_used else None
    model_path = OUT_DIR / str(spec["model_file"])

    for epoch in range(1, MAX_EPOCHS_CPU + 1):
        epoch_start = perf_counter()
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        for x, site_code, meta, y in iter_model_batches(
            train_sample_indices,
            arrays["scaled_features"],
            arrays["start_pos"],
            arrays["target_values"],
            arrays["site_codes"],
            metadata,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            rng=rng,
        ):
            x = x.to(device)
            site_code = site_code.to(device)
            y = y.to(device)
            meta = meta.to(device) if meta is not None else None
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, site_code, meta)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(y)
            n_train += len(y)
        train_loss = train_loss_sum / n_train if n_train else np.nan
        val_loss, y_val, pred_val = evaluate_model(
            model,
            val_early_indices,
            arrays,
            metadata_used=metadata_used,
            device=device,
            batch_size=PREDICT_BATCH_SIZE,
        )
        val_metrics = metric_dict(y_val, pred_val)
        val_rmse = float(val_metrics["rmse"])
        elapsed = perf_counter() - epoch_start
        rows.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "n_train_windows_epoch": int(n_train),
                "n_validation_windows_early_stop": int(len(val_early_indices)),
                "epoch_time_seconds": elapsed,
                "learning_rate": LEARNING_RATE,
            }
        )
        log(
            f"[INFO] {model_id} epoch={epoch} train_loss={train_loss:.6f} "
            f"val_rmse={val_rmse:.6f} val_r2={val_metrics['r2']:.6f} "
            f"seconds={elapsed:.1f}"
        )
        if np.isfinite(val_rmse) and val_rmse < best_rmse - 1e-7:
            best_rmse = val_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_spec": spec,
                    "dynamic_features": DYNAMIC_FEATURES,
                    "metadata_features": arrays["metadata_features"] if metadata_used else [],
                    "site_mapping": arrays["site_mapping"],
                },
                model_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            log(f"[INFO] Early stopping triggered for {model_id}")
            break

    history = pd.DataFrame(rows)
    history.to_csv(OUT_DIR / str(spec["history_file"]), index=False, encoding="utf-8-sig")
    if best_epoch < 0:
        raise RuntimeError(f"{model_id} did not produce a finite validation RMSE.")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_info = {
        "model_id": model_id,
        "best_epoch": int(best_epoch),
        "best_early_stop_val_rmse": float(best_rmse),
        "epochs_completed": int(len(history)),
        "completed": True,
    }
    return model, history, train_info


def prediction_frame(
    spec: dict[str, object],
    windows: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    meta = windows.iloc[indices].reset_index(drop=True)
    pred = pd.DataFrame(
        {
            "model_id": spec["model_id"],
            "model_name": spec["model_name"],
            SITE_COL: meta[SITE_COL].astype(str),
            "target_timestamp": meta["target_timestamp"],
            SPLIT_COL: meta[SPLIT_COL].astype(str),
            TARGET: y_true.astype("float64"),
            "y_pred": y_pred.astype("float64"),
            EVENT_COL: meta[EVENT_COL].astype("int8").to_numpy(),
            EVENT_LABEL_COL: meta[EVENT_LABEL_COL].astype(str).to_numpy(),
            EVENT_STATE_COL: meta[EVENT_STATE_COL].astype("int8").to_numpy(),
        }
    )
    pred["residual"] = pred[TARGET] - pred["y_pred"]
    pred["abs_residual"] = pred["residual"].abs()
    return pred


def predict_and_save(
    model: nn.Module,
    spec: dict[str, object],
    arrays: dict[str, object],
    windows: pd.DataFrame,
    indices: np.ndarray,
    split: str,
    device: torch.device,
) -> pd.DataFrame:
    log(f"[INFO] Predicting {split} windows for {spec['model_id']}")
    _loss, y_true, y_pred = evaluate_model(
        model,
        indices,
        arrays,
        metadata_used=bool(spec["metadata_used"]),
        device=device,
        batch_size=PREDICT_BATCH_SIZE,
    )
    pred = prediction_frame(spec, windows, indices, y_true, y_pred)
    out_name = str(spec["val_prediction_file"] if split == "val" else spec["test_prediction_file"])
    write_parquet(pred, OUT_DIR / out_name)
    return pred


def exact_top_mask(values: np.ndarray, fraction: float) -> tuple[np.ndarray, float]:
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=bool), np.nan
    n_top = max(1, int(math.ceil(n * fraction)))
    order = np.argpartition(values, n - n_top)[n - n_top :]
    mask = np.zeros(n, dtype=bool)
    mask[order] = True
    threshold = float(np.min(values[order]))
    return mask, threshold


def row_level_enrichment(test_predictions: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[float, np.ndarray]]]:
    residual_rows = []
    label_rows = []
    top_masks: dict[str, dict[float, np.ndarray]] = {}
    for model_name, pred in test_predictions.items():
        model_id = str(pred["model_id"].iloc[0])
        metadata_used = bool("with_metadata" in model_name)
        is_event = pred[EVENT_COL].astype(bool).to_numpy()
        labels = pred[EVENT_LABEL_COL].map(compact_label).to_numpy()
        abs_residual = pred["abs_residual"].to_numpy(dtype="float64")
        base_event_rate = float(is_event.mean()) if len(pred) else np.nan
        top_masks[model_name] = {}
        for fraction in TOP_FRACTIONS:
            mask, threshold = exact_top_mask(abs_residual, fraction)
            top_masks[model_name][fraction] = mask
            top_event = int((mask & is_event).sum())
            top_rate = float(is_event[mask].mean()) if mask.any() else np.nan
            residual_rows.append(
                {
                    "model_id": model_id,
                    "model_name": model_name,
                    "top_fraction": fraction,
                    "n_test_windows": int(len(pred)),
                    "n_event_windows": int(is_event.sum()),
                    "base_event_rate": base_event_rate,
                    "n_top_windows": int(mask.sum()),
                    "n_top_event_windows": top_event,
                    "top_event_rate": top_rate,
                    "enrichment": top_rate / base_event_rate if base_event_rate and np.isfinite(base_event_rate) else np.nan,
                    "abs_residual_threshold": threshold,
                    "metadata_used": metadata_used,
                }
            )
            for label in ["low", "high"]:
                label_mask = labels == label
                base_rate = float(label_mask.mean()) if len(pred) else np.nan
                top_label = int((mask & label_mask).sum())
                top_label_rate = float(label_mask[mask].mean()) if mask.any() else np.nan
                label_rows.append(
                    {
                        "model_id": model_id,
                        "model_name": model_name,
                        "event_label": f"pH_exceedance_{label}",
                        "top_fraction": fraction,
                        "n_test_windows": int(len(pred)),
                        "n_label_windows": int(label_mask.sum()),
                        "base_label_rate": base_rate,
                        "n_top_windows": int(mask.sum()),
                        "n_top_label_windows": top_label,
                        "top_label_rate": top_label_rate,
                        "enrichment": top_label_rate / base_rate if base_rate and np.isfinite(base_rate) else np.nan,
                        "metadata_used": metadata_used,
                    }
                )
    return pd.DataFrame(residual_rows), pd.DataFrame(label_rows), top_masks


def approximate_event_level_hit_rate(
    test_predictions: dict[str, pd.DataFrame],
    top_masks: dict[str, dict[float, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for model_name, pred in test_predictions.items():
        model_id = str(pred["model_id"].iloc[0])
        base = pred[[SITE_COL, "target_timestamp", EVENT_LABEL_COL]].copy()
        base["target_timestamp"] = pd.to_datetime(base["target_timestamp"], errors="coerce")
        base["label_compact"] = base[EVENT_LABEL_COL].map(compact_label)
        base = base.reset_index(drop=True)
        for fraction in TOP_FRACTIONS:
            work = base.copy()
            work["top_residual"] = top_masks[model_name][fraction]
            events = work[work["label_compact"].isin(["low", "high"])].copy()
            if events.empty:
                for group_name in ["all", "low", "high"]:
                    rows.append(
                        {
                            "model_id": model_id,
                            "model_name": model_name,
                            "top_fraction": fraction,
                            "event_group": group_name,
                            "n_event_windows": 0,
                            "n_hit_event_windows": 0,
                            "event_level_hit_rate": np.nan,
                            "method": "approx_contiguous_target_windows_by_site_label",
                        }
                    )
                continue
            events = events.sort_values([SITE_COL, "label_compact", "target_timestamp"]).reset_index(drop=True)
            same_site = events[SITE_COL].eq(events[SITE_COL].shift())
            same_label = events["label_compact"].eq(events["label_compact"].shift())
            one_hour = events["target_timestamp"].diff().eq(pd.Timedelta(hours=1))
            new_block = ~(same_site & same_label & one_hour)
            events["event_block_id"] = new_block.cumsum()
            block = (
                events.groupby(["event_block_id", "label_compact"], observed=True)
                .agg(hit=("top_residual", "max"), n_rows=("top_residual", "size"))
                .reset_index()
            )
            for group_name, group in [
                ("all", block),
                ("low", block[block["label_compact"].eq("low")]),
                ("high", block[block["label_compact"].eq("high")]),
            ]:
                n_blocks = int(len(group))
                n_hit = int(group["hit"].sum()) if n_blocks else 0
                rows.append(
                    {
                        "model_id": model_id,
                        "model_name": model_name,
                        "top_fraction": fraction,
                        "event_group": group_name,
                        "n_event_windows": n_blocks,
                        "n_hit_event_windows": n_hit,
                        "event_level_hit_rate": n_hit / n_blocks if n_blocks else np.nan,
                        "method": "approx_contiguous_target_windows_by_site_label",
                    }
                )
    return pd.DataFrame(rows)


def compute_all_metrics(
    val_predictions: dict[str, pd.DataFrame],
    test_predictions: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    site_rows = []
    for split, pred_map in [("val", val_predictions), ("test", test_predictions)]:
        for model_name, pred in pred_map.items():
            metrics = metric_dict(pred[TARGET].to_numpy(), pred["y_pred"].to_numpy())
            metric_rows.append(
                {
                    "model_id": str(pred["model_id"].iloc[0]),
                    "model_name": model_name,
                    "split": split,
                    "n_windows": int(metrics["n_windows"]),
                    "n_sites": int(pred[SITE_COL].nunique()),
                    "metadata_used": bool("with_metadata" in model_name),
                    "sequence_length": SEQUENCE_LENGTH,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                }
            )
    for model_name, pred in test_predictions.items():
        for site_id, group in pred.groupby(SITE_COL, sort=True, observed=True):
            metrics = metric_dict(group[TARGET].to_numpy(), group["y_pred"].to_numpy())
            labels = group[EVENT_LABEL_COL].map(compact_label)
            site_rows.append(
                {
                    "model_id": str(group["model_id"].iloc[0]),
                    "model_name": model_name,
                    SITE_COL: site_id,
                    "n_test_windows": int(len(group)),
                    "n_test_event_windows": int(group[EVENT_COL].sum()),
                    "n_test_low_event_windows": int(labels.eq("low").sum()),
                    "n_test_high_event_windows": int(labels.eq("high").sum()),
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                }
            )
    comparison = pd.DataFrame(metric_rows)
    site_metrics = pd.DataFrame(site_rows)
    residual_event, low_high, top_masks = row_level_enrichment(test_predictions)
    event_hit = approximate_event_level_hit_rate(test_predictions, top_masks)
    comparison.to_csv(OUT_DIR / "sequence_model_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    residual_event.to_csv(OUT_DIR / "sequence_residual_event_validation.csv", index=False, encoding="utf-8-sig")
    low_high.to_csv(OUT_DIR / "sequence_low_high_event_enrichment.csv", index=False, encoding="utf-8-sig")
    event_hit.to_csv(OUT_DIR / "sequence_event_level_hit_rate.csv", index=False, encoding="utf-8-sig")
    site_metrics.to_csv(OUT_DIR / "sequence_site_level_metrics.csv", index=False, encoding="utf-8-sig")
    return comparison, residual_event, low_high, event_hit, site_metrics


def same_row_hgb_comparison() -> pd.DataFrame:
    candidates = []
    for path in ANALYSIS_DIR.rglob("*.parquet"):
        lower = str(path).lower()
        if "hgb" in lower and "100" in lower and ("prediction" in lower or "pred" in lower):
            if OUT_DIR not in path.parents:
                candidates.append(path)
    if not candidates:
        out = pd.DataFrame(
            [
                {
                    "status": "skipped_no_100site_hgb_predictions",
                    "message": "No same-row 100-site HGB comparison was performed because 100-site HGB predictions were not available.",
                }
            ]
        )
        out.to_csv(OUT_DIR / "sequence_hgb_same_row_comparison.csv", index=False, encoding="utf-8-sig")
        return out
    out = pd.DataFrame(
        [
            {
                "status": "skipped_candidate_files_found_manual_review_needed",
                "candidate_files": ";".join(str(path.relative_to(ROOT)) for path in candidates[:20]),
            }
        ]
    )
    out.to_csv(OUT_DIR / "sequence_hgb_same_row_comparison.csv", index=False, encoding="utf-8-sig")
    return out


def write_feature_spec(
    metadata_features: list[str],
    input_info: dict[str, object],
    train_sampling_used: bool,
    val_sampling_used: bool,
) -> None:
    spec = {
        "phase": "Phase 5 sequence100 GRU/LSTM metadata",
        "source_no_metadata_timeseries": str(BASE_PARQUET.relative_to(ROOT)),
        "source_metadata_timeseries": str(METADATA_PARQUET.relative_to(ROOT)),
        "metadata_feature_spec_source": str(METADATA_FEATURE_SPEC.relative_to(ROOT)),
        "target": TARGET,
        "dynamic_sequence_features": DYNAMIC_FEATURES,
        "static_metadata_features": metadata_features,
        "site_embedding_used": True,
        "sequence_length_hours": SEQUENCE_LENGTH,
        "target_position": "last step t",
        "window_input_range": "t-23h through t",
        "uses_ph_history": False,
        "uses_event_labels_as_features": False,
        "event_columns_evaluation_only": [EVENT_COL, EVENT_LABEL_COL, EVENT_STATE_COL, "event_state_name"],
        "forbidden_predictor_columns": [
            TARGET,
            EVENT_COL,
            EVENT_LABEL_COL,
            EVENT_STATE_COL,
            "event_state_name",
            "sample_weight_first_version",
            "event counts",
            "event ratios",
            "incident fields",
            "exceedance fields",
            "residuals",
            "prediction columns",
            "model metrics",
        ],
        "hourly_aggregation": {
            "numeric_dynamic_and_target": "mean within site-hour",
            "time_features": "recomputed from hourly timestamp",
            "event_label": "conservative event label retained only for evaluation",
            "split": "majority source row split per site-hour",
        },
        "input_base_rows": int(input_info["base_rows"]),
        "training_sampling_used": bool(train_sampling_used),
        "early_stop_validation_sampling_used": bool(val_sampling_used),
    }
    (OUT_DIR / "sequence100_feature_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")


def write_model_config(
    device: torch.device,
    train_count: int,
    train_sample_count: int,
    val_count: int,
    val_early_count: int,
    l100_status: str,
) -> None:
    config = {
        "random_seed": RANDOM_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "sequence_length": SEQUENCE_LENGTH,
        "dynamic_features": DYNAMIC_FEATURES,
        "models": MODEL_SPECS,
        "optional_lstm": {
            "model_id": "L100_B",
            "status": l100_status,
            "reason": "Optional extension skipped so required GRU comparison completes on CPU-only runtime.",
        },
        "hyperparameters": {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "mlp_hidden_size": MLP_HIDDEN_SIZE,
            "dropout": DROPOUT,
            "batch_size": TRAIN_BATCH_SIZE,
            "prediction_batch_size": PREDICT_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS_CPU,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "loss": "MSE",
            "optimizer": "Adam",
        },
        "runtime_controls": {
            "original_train_windows": int(train_count),
            "sampled_train_windows": int(train_sample_count),
            "train_window_cap_cpu": int(MAX_TRAIN_WINDOWS_CPU),
            "original_val_windows": int(val_count),
            "early_stop_val_windows": int(val_early_count),
            "early_stop_val_window_cap_cpu": int(EARLY_STOP_VAL_WINDOWS_CPU),
            "full_val_predictions_saved": True,
            "full_test_predictions_saved": True,
        },
    }
    (OUT_DIR / "model_config_gru_lstm.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def save_figures(
    comparison: pd.DataFrame,
    residual_event: pd.DataFrame,
    low_high: pd.DataFrame,
    event_hit: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    test_predictions: dict[str, pd.DataFrame],
) -> None:
    log("[INFO] Saving figures")
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"G100_A_gru_no_metadata": "#4C78A8", "G100_B_gru_with_metadata": "#F58518"}
    test_metrics = comparison[comparison[SPLIT_COL].eq("test")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(test_metrics["model_id"], test_metrics["rmse"], color=[colors.get(m, "#777777") for m in test_metrics["model_name"]])
    axes[0].set_ylabel("Test RMSE")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(test_metrics["model_id"], test_metrics["r2"], color=[colors.get(m, "#777777") for m in test_metrics["model_name"]])
    axes[1].set_ylabel("Test R2")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "sequence_rmse_r2_comparison.png", dpi=300)
    plt.close(fig)

    top5 = residual_event[np.isclose(residual_event["top_fraction"], 0.05)]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(top5["model_id"], top5["enrichment"], color=[colors.get(m, "#777777") for m in top5["model_name"]])
    ax.axhline(1, color="#111111", linestyle="--", linewidth=1)
    ax.set_ylabel("Top 5% event enrichment")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "sequence_top5_enrichment_comparison.png", dpi=300)
    plt.close(fig)

    top5_hit = event_hit[np.isclose(event_hit["top_fraction"], 0.05) & event_hit["event_group"].eq("all")]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(top5_hit["model_id"], top5_hit["event_level_hit_rate"], color=[colors.get(m, "#777777") for m in top5_hit["model_name"]])
    ax.set_ylabel("Top 5% event-level hit rate")
    ax.set_ylim(0, max(0.05, float(top5_hit["event_level_hit_rate"].max()) * 1.2 if not top5_hit.empty else 0.05))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "sequence_event_hit_rate_comparison.png", dpi=300)
    plt.close(fig)

    top5_labels = low_high[np.isclose(low_high["top_fraction"], 0.05)].copy()
    top5_labels["label_short"] = top5_labels["event_label"].map(lambda x: "low" if "low" in x else "high")
    pivot = top5_labels.pivot(index="model_id", columns="label_short", values="enrichment").reindex(["G100_A", "G100_B"])
    fig, ax = plt.subplots(figsize=(6.2, 4))
    pivot.plot(kind="bar", ax=ax, color=["#E45756", "#72B7B2"])
    ax.axhline(1, color="#111111", linestyle="--", linewidth=1)
    ax.set_ylabel("Top 5% enrichment")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "sequence_low_high_enrichment_comparison.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model_name, history in histories.items():
        axes[0].plot(history["epoch"], history["train_loss"], marker="o", label=model_name)
        axes[1].plot(history["epoch"], history["val_rmse"], marker="o", label=model_name)
    axes[0].set_ylabel("Train MSE")
    axes[1].set_ylabel("Validation RMSE")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "sequence_training_curves.png", dpi=300)
    plt.close(fig)

    case = test_predictions["G100_B_gru_with_metadata"].copy()
    case["target_timestamp"] = pd.to_datetime(case["target_timestamp"], errors="coerce")
    event_counts = case.groupby(SITE_COL)[EVENT_COL].sum().sort_values(ascending=False)
    case_site = str(event_counts.index[0]) if not event_counts.empty else str(case[SITE_COL].iloc[0])
    site_case = case[case[SITE_COL].eq(case_site)].sort_values("target_timestamp")
    event_rows = site_case[site_case[EVENT_COL].astype(bool)]
    if not event_rows.empty:
        center = event_rows["target_timestamp"].iloc[0]
        site_case = site_case[
            (site_case["target_timestamp"] >= center - pd.Timedelta(hours=96))
            & (site_case["target_timestamp"] <= center + pd.Timedelta(hours=96))
        ]
    else:
        site_case = site_case.head(240)
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(site_case["target_timestamp"], site_case[TARGET], label="observed", color="#111111", linewidth=1.4)
    ax1.plot(site_case["target_timestamp"], site_case["y_pred"], label="predicted", color="#F58518", linewidth=1.2)
    event_points = site_case[site_case[EVENT_COL].astype(bool)]
    if not event_points.empty:
        ax1.scatter(event_points["target_timestamp"], event_points[TARGET], s=16, color="#E45756", label="event target")
    ax1.set_ylabel("Processed pH")
    ax2 = ax1.twinx()
    ax2.plot(site_case["target_timestamp"], site_case["abs_residual"], label="abs residual", color="#4C78A8", alpha=0.6)
    ax2.set_ylabel("Absolute residual")
    ax1.set_title(f"Case-study site {case_site}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "sequence_observed_predicted_case_study.png", dpi=300)
    plt.close(fig)


def top5_value(df: pd.DataFrame, model_id: str, column: str, event_group: str | None = None, label: str | None = None) -> float:
    subset = df[np.isclose(df["top_fraction"], 0.05) & df["model_id"].eq(model_id)]
    if event_group is not None and "event_group" in subset.columns:
        subset = subset[subset["event_group"].eq(event_group)]
    if label is not None and "event_label" in subset.columns:
        subset = subset[subset["event_label"].str.contains(label, case=False, na=False)]
    if subset.empty:
        return float("nan")
    return float(subset[column].iloc[0])


def write_report(
    hourly_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    residual_event: pd.DataFrame,
    low_high: pd.DataFrame,
    event_hit: pd.DataFrame,
    site_metrics: pd.DataFrame,
    hgb_comparison: pd.DataFrame,
    train_sampling_used: bool,
    val_sampling_used: bool,
    l100_status: str,
) -> tuple[str, str]:
    test = comparison[comparison[SPLIT_COL].eq("test")].set_index("model_id")
    rmse_a = float(test.loc["G100_A", "rmse"])
    rmse_b = float(test.loc["G100_B", "rmse"])
    r2_a = float(test.loc["G100_A", "r2"])
    r2_b = float(test.loc["G100_B", "r2"])
    enrich_a = top5_value(residual_event, "G100_A", "enrichment")
    enrich_b = top5_value(residual_event, "G100_B", "enrichment")
    hit_a = top5_value(event_hit, "G100_A", "event_level_hit_rate", event_group="all")
    hit_b = top5_value(event_hit, "G100_B", "event_level_hit_rate", event_group="all")
    low_a = top5_value(low_high, "G100_A", "enrichment", label="low")
    low_b = top5_value(low_high, "G100_B", "enrichment", label="low")
    high_a = top5_value(low_high, "G100_A", "enrichment", label="high")
    high_b = top5_value(low_high, "G100_B", "enrichment", label="high")

    metadata_improved_rmse = rmse_b < rmse_a
    metadata_improved_r2 = r2_b > r2_a
    metadata_improved_enrichment = enrich_b > enrich_a
    if metadata_improved_rmse and metadata_improved_r2 and metadata_improved_enrichment:
        interpretation = (
            "Metadata improved both GRU prediction and residual-event alignment, suggesting the temporal model used site context "
            "without fully absorbing event deviations."
        )
    elif metadata_improved_rmse and metadata_improved_r2 and not metadata_improved_enrichment:
        interpretation = (
            "Metadata improved GRU prediction but weakened residual-event enrichment, matching the earlier HGB trade-off pattern."
        )
    elif not metadata_improved_rmse and not metadata_improved_r2:
        interpretation = (
            "The metadata GRU did not improve prediction over the no-metadata GRU; the sequence model is not yet a clear replacement for HGB."
        )
    else:
        interpretation = "Metadata effects are mixed: prediction and residual-event behaviour do not move in the same direction."

    difficult_sites = site_metrics.sort_values("rmse", ascending=False).head(10)
    site_pivot = site_metrics.pivot(index=SITE_COL, columns="model_id", values="rmse").dropna()
    site_pivot["rmse_delta_B_minus_A"] = site_pivot.get("G100_B", np.nan) - site_pivot.get("G100_A", np.nan)
    improved_sites = int((site_pivot["rmse_delta_B_minus_A"] < 0).sum())
    worsened_sites = int((site_pivot["rmse_delta_B_minus_A"] > 0).sum())

    report = f"""# Phase 5 Sequence100 GRU/LSTM Metadata Report

## 1. Purpose

This phase tests 24-hour temporal sequence models for processed pH prediction across 100 sites, comparing a GRU without metadata (G100_A) against a GRU with static metadata (G100_B).

## 2. Dataset

- No-metadata time-series source: `{BASE_PARQUET.relative_to(ROOT)}`
- Metadata source: `{METADATA_PARQUET.relative_to(ROOT)}`
- Metadata feature specification: `{METADATA_FEATURE_SPEC.relative_to(ROOT)}`
- Hourly aggregation: numeric variables averaged by site-hour; time features recomputed from the hourly timestamp.
- Split handling: majority source-row split within each site-hour.
- Valid window rule: strict 24 consecutive hourly rows, one site, one split.

Hourly summary:

{dataframe_to_markdown(hourly_summary)}

Window summary:

{dataframe_to_markdown(window_summary)}

## 3. Feature Design

- Dynamic sequence features: `{DYNAMIC_FEATURES}`
- Target: `{TARGET}` at the final timestamp `t`
- pH history in input: no
- Event labels in input: no
- Site embedding: yes
- Static metadata in G100_B: {len(json.loads((OUT_DIR / "sequence100_feature_spec.json").read_text(encoding="utf-8"))["static_metadata_features"])} safe fields from the Phase 4 feature set
- Event columns are used only after prediction for residual-event validation.

## 4. Model Setup

- G100_A: 1-layer GRU, hidden size {HIDDEN_SIZE}, site embedding, no metadata.
- G100_B: same GRU plus static metadata vector.
- L100_B: {l100_status}; it is optional and did not block completion.
- Optimizer: Adam, learning rate {LEARNING_RATE}, weight decay {WEIGHT_DECAY}, MSE loss.
- Runtime: CPU-only was detected if CUDA was unavailable; max epochs set to {MAX_EPOCHS_CPU} with patience {EARLY_STOPPING_PATIENCE}.
- Training sampling used: {train_sampling_used}
- Early-stop validation sampling used: {val_sampling_used}
- Full validation and test predictions were saved for both GRU models.

## 5. Prediction Performance

{dataframe_to_markdown(comparison)}

- Metadata improved GRU RMSE: {'yes' if metadata_improved_rmse else 'no'}
- Metadata improved GRU R2: {'yes' if metadata_improved_r2 else 'no'}
- G100_A test RMSE/R2: {rmse_a:.6f} / {r2_a:.6f}
- G100_B test RMSE/R2: {rmse_b:.6f} / {r2_b:.6f}

## 6. Residual-Event Validation

Row-level residual-event enrichment:

{dataframe_to_markdown(residual_event)}

Low/high enrichment:

{dataframe_to_markdown(low_high)}

Approximate event-level hit rate:

{dataframe_to_markdown(event_hit)}

- Metadata improved top 5% residual-event enrichment: {'yes' if metadata_improved_enrichment else 'no'}
- G100_A / G100_B top 5% enrichment: {enrich_a:.6f} / {enrich_b:.6f}
- G100_A / G100_B top 5% event-level hit rate: {hit_a:.6f} / {hit_b:.6f}
- Low-pH top 5% enrichment G100_A / G100_B: {low_a:.6f} / {low_b:.6f}
- High-pH top 5% enrichment G100_A / G100_B: {high_a:.6f} / {high_b:.6f}

## 7. Site-Level Behaviour

- Sites with lower test RMSE after adding metadata: {improved_sites}
- Sites with higher test RMSE after adding metadata: {worsened_sites}

Most difficult test sites by RMSE:

{dataframe_to_markdown(difficult_sites)}

## 8. Comparison With HGB Findings

The previous HGB experiments showed that metadata improved pH prediction but often weakened residual-event alignment. The same-row 100-site HGB comparison status is:

{dataframe_to_markdown(hgb_comparison)}

Because no 100-site HGB prediction files were available, this Phase 5 comparison to HGB is qualitative rather than same-row.

## 9. Interpretation

{interpretation}

## 10. Limitations

- Sequence models use hourly aggregated data rather than the original sub-hourly cadence.
- No pH history was used, so the GRU tests temporal exogenous forcing rather than autoregressive pH persistence.
- Event labels are weak validation labels and are not model inputs.
- CPU-only runtime required capped site-balanced training windows and capped validation windows for early stopping.
- Event-level hit rate uses approximate contiguous target-window blocks by site and event label.
- No same-row 100-site HGB prediction comparison was available.

## 11. Next Step

If prediction gain and residual-event alignment are both acceptable, proceed to LSTM or FiLM metadata conditioning. If the metadata GRU mainly improves RMSE while weakening residual-event sensitivity, keep HGB/non-event HGB as the detection reference and use GRU results as a temporal robustness check.
"""
    (OUT_DIR / "phase5_sequence100_gru_lstm_metadata_report.md").write_text(report, encoding="utf-8")
    return interpretation, "Proceed to LSTM/FiLM only if the GRU metadata trade-off is acceptable; otherwise keep HGB/non-event HGB as the main detection reference."


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    input_info = ensure_inputs()
    metadata_features = input_info["metadata_features"]

    hourly, hourly_summary = build_hourly_dataset(input_info)
    windows, window_summary = create_window_index(hourly)

    train_indices = np.where(windows[SPLIT_COL].eq("train").to_numpy())[0].astype("int64")
    val_indices = np.where(windows[SPLIT_COL].eq("val").to_numpy())[0].astype("int64")
    test_indices = np.where(windows[SPLIT_COL].eq("test").to_numpy())[0].astype("int64")
    if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
        raise RuntimeError("Train/val/test windows are all required.")

    train_sample_indices, train_sample_summary, train_sampling_used = sample_indices_site_balanced(
        windows, train_indices, MAX_TRAIN_WINDOWS_CPU, RANDOM_SEED, "train"
    )
    val_early_indices, val_sample_summary, val_sampling_used = sample_indices_site_balanced(
        windows, val_indices, EARLY_STOP_VAL_WINDOWS_CPU, RANDOM_SEED + 1, "val_early_stop"
    )
    train_sample_summary.to_csv(OUT_DIR / "sequence100_training_sample_summary.csv", index=False, encoding="utf-8-sig")
    val_sample_summary.to_csv(OUT_DIR / "sequence100_early_stop_validation_sample_summary.csv", index=False, encoding="utf-8-sig")
    sampling_info = {
        "original_train_window_count": int(len(train_indices)),
        "sampled_train_window_count": int(len(train_sample_indices)),
        "train_sampling_method": "site-balanced proportional sampling without replacement" if train_sampling_used else "all train windows",
        "train_sampling_random_seed": RANDOM_SEED if train_sampling_used else np.nan,
        "original_val_window_count": int(len(val_indices)),
        "early_stop_val_window_count": int(len(val_early_indices)),
        "early_stop_val_sampling_method": "site-balanced proportional sampling without replacement" if val_sampling_used else "all validation windows",
    }
    window_summary = make_window_summary(windows, hourly, sampling_info=sampling_info)
    window_summary.to_csv(OUT_DIR / "sequence100_window_dataset_summary.csv", index=False, encoding="utf-8-sig")

    arrays = build_arrays_and_encodings(hourly, windows, metadata_features)
    scaled_features, dynamic_scaler_info = fit_dynamic_scaler(
        arrays["feature_matrix"],
        arrays["start_pos"],
        train_sample_indices,
    )
    arrays["scaled_features"] = scaled_features

    write_feature_spec(metadata_features, input_info, train_sampling_used, val_sampling_used)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    l100_status = "skipped_cpu_only_optional"
    write_model_config(device, len(train_indices), len(train_sample_indices), len(val_indices), len(val_early_indices), l100_status)

    histories: dict[str, pd.DataFrame] = {}
    train_infos = []
    val_predictions: dict[str, pd.DataFrame] = {}
    test_predictions: dict[str, pd.DataFrame] = {}
    for spec in MODEL_SPECS:
        model, history, train_info = train_sequence_model(spec, arrays, train_sample_indices, val_early_indices, device)
        histories[str(spec["model_name"])] = history
        train_infos.append(train_info)
        val_predictions[str(spec["model_name"])] = predict_and_save(model, spec, arrays, windows, val_indices, "val", device)
        test_predictions[str(spec["model_name"])] = predict_and_save(model, spec, arrays, windows, test_indices, "test", device)
    pd.DataFrame(train_infos).to_csv(OUT_DIR / "sequence100_model_training_completion.csv", index=False, encoding="utf-8-sig")

    comparison, residual_event, low_high, event_hit, site_metrics = compute_all_metrics(val_predictions, test_predictions)
    hgb_comparison = same_row_hgb_comparison()
    save_figures(comparison, residual_event, low_high, event_hit, histories, test_predictions)
    interpretation, next_step = write_report(
        hourly_summary,
        window_summary,
        comparison,
        residual_event,
        low_high,
        event_hit,
        site_metrics,
        hgb_comparison,
        train_sampling_used,
        val_sampling_used,
        l100_status,
    )

    test = comparison[comparison[SPLIT_COL].eq("test")].set_index("model_id")
    rmse_a = float(test.loc["G100_A", "rmse"])
    rmse_b = float(test.loc["G100_B", "rmse"])
    r2_a = float(test.loc["G100_A", "r2"])
    r2_b = float(test.loc["G100_B", "r2"])
    enrich_a = top5_value(residual_event, "G100_A", "enrichment")
    enrich_b = top5_value(residual_event, "G100_B", "enrichment")
    hit_a = top5_value(event_hit, "G100_A", "event_level_hit_rate", event_group="all")
    hit_b = top5_value(event_hit, "G100_B", "event_level_hit_rate", event_group="all")
    low_a = top5_value(low_high, "G100_A", "enrichment", label="low")
    low_b = top5_value(low_high, "G100_B", "enrichment", label="low")
    high_a = top5_value(low_high, "G100_A", "enrichment", label="high")
    high_b = top5_value(low_high, "G100_B", "enrichment", label="high")
    metadata_improved_rmse = rmse_b < rmse_a
    metadata_improved_r2 = r2_b > r2_a
    metadata_improved_enrichment = enrich_b > enrich_a
    overall_hourly = hourly_summary[hourly_summary["scope"].eq("overall")].iloc[0]
    overall_windows = window_summary[window_summary["scope"].eq("overall")].iloc[0]

    final_text = f"""PHASE 5 SEQUENCE100 GRU/LSTM METADATA COMPLETE

Output folder:
{OUT_DIR.relative_to(ROOT).as_posix()}/

Hourly rows:
{int(overall_hourly['hourly_rows'])}

Valid 24h windows:
{int(overall_windows['total_valid_windows'])}

Train windows:
{int(overall_windows['train_windows'])}

Validation windows:
{int(overall_windows['val_windows'])}

Test windows:
{int(overall_windows['test_windows'])}

Models completed:
G100_A GRU no metadata: yes
G100_B GRU with metadata: yes
L100_B LSTM with metadata: skipped

G100_A test RMSE:
{rmse_a:.6f}

G100_B test RMSE:
{rmse_b:.6f}

G100_A test R2:
{r2_a:.6f}

G100_B test R2:
{r2_b:.6f}

G100_A top 5% enrichment:
{enrich_a:.6f}

G100_B top 5% enrichment:
{enrich_b:.6f}

G100_A event-level top 5% hit rate:
{hit_a:.6f}

G100_B event-level top 5% hit rate:
{hit_b:.6f}

G100_A low-pH top 5% enrichment:
{low_a:.6f}

G100_B low-pH top 5% enrichment:
{low_b:.6f}

G100_A high-pH top 5% enrichment:
{high_a:.6f}

G100_B high-pH top 5% enrichment:
{high_b:.6f}

Metadata improved GRU RMSE:
{'yes' if metadata_improved_rmse else 'no'}

Metadata improved GRU R2:
{'yes' if metadata_improved_r2 else 'no'}

Metadata improved GRU residual-event enrichment:
{'yes' if metadata_improved_enrichment else 'no'}

Main interpretation:
{interpretation}

Next recommended task:
{next_step}

Elapsed seconds:
{perf_counter() - started:.1f}
"""
    print(final_text)
    (OUT_DIR / "sequence100_final_console_summary.txt").write_text(final_text, encoding="utf-8")


if __name__ == "__main__":
    with tee_stdout(OUT_DIR / "sequence100_gru_lstm_metadata_log.txt"):
        main()
