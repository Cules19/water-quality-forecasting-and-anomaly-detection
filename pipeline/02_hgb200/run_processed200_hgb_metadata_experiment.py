#!/usr/bin/env python
"""Phase 3B strict 200-site processed HGB metadata experiment.

This script uses data/processed as the time-series source and
data/reference/sites_register.json as the static metadata source. It avoids raw
or clean-data rebuilding. Event labels are created only for validation after
prediction, never as model predictors.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "analysis_outputs"
OUTPUT_DIR = ANALYSIS_DIR / "ph_hgb_processed200_metadata_v1"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REGISTRY_PATH = PROJECT_ROOT / "data" / "reference" / "sites_register.json"
EVENT_FILE = PROJECT_ROOT / "data" / "reference" / "regulatory_events.csv"
FIGURES_DIR = OUTPUT_DIR / "figures"
TMP_DIR = OUTPUT_DIR / "_tmp_memmap"
LOG_PATH = OUTPUT_DIR / "processed200_hgb_metadata_log.txt"

PREVIOUS_SCHEMA_SUMMARY = ANALYSIS_DIR / "ph_processed_metadata_availability_check_v1" / "processed_schema_summary.csv"
FULL50_SUMMARY_PATH = ANALYSIS_DIR / "ph_full50_model_summary_with_m0_v1" / "full50_m0_m1_m2_summary.csv"
FULL50_METRICS_PATH = ANALYSIS_DIR / "ph_hgb_full50_metadata_v1" / "model_comparison_metrics.csv"
FULL50_RESIDUAL_PATH = ANALYSIS_DIR / "ph_hgb_full50_metadata_v1" / "residual_event_validation.csv"
FULL50_LOW_HIGH_PATH = ANALYSIS_DIR / "ph_hgb_full50_metadata_v1" / "low_high_event_enrichment.csv"
FULL50_HIT_PATH = ANALYSIS_DIR / "ph_hgb_full50_metadata_v1" / "event_level_hit_rate.csv"

TARGET_SITE_COUNT = 200
TARGET_COL = "00400"
SITE_COL = "site_id"
TIME_COL = "timestamp"
SPLIT_COL = "split"
EVENT_COL = "in_ph_event"
EVENT_LABEL_COL = "ph_event_label"
SITE_CODE_COL = "site_id_code"

REQUIRED_CODES = ["00400", "00010", "00095", "63680", "00060", "00065"]
EXOGENOUS_COLUMNS = ["00010", "00095", "63680", "00060", "00065"]
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
BASE_FEATURES = [*DYNAMIC_FEATURES, SITE_CODE_COL]
MODEL_READY_COLUMNS = [
    SITE_COL,
    TIME_COL,
    SPLIT_COL,
    TARGET_COL,
    *EXOGENOUS_COLUMNS,
    "sin_hour",
    "cos_hour",
    "sin_day",
    "cos_day",
    "month",
    "hour",
    "day_of_year",
    EVENT_COL,
    EVENT_LABEL_COL,
]

PH_EVENT_LABELS = ["pH_exceedance_low", "pH_exceedance_high"]
THRESHOLDS = [0.01, 0.05, 0.10]

M200_A_ID = "M200_A_no_metadata"
M200_A_NAME = "Processed200 all-data HGB without metadata"
M200_B_ID = "M200_B_with_metadata"
M200_B_NAME = "Processed200 all-data HGB with metadata"

HGB_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "random_state": 42,
    "early_stopping": False,
}

# sklearn's HistGradientBoostingRegressor in this environment validates X as
# float64. With 38.5M train rows, using every computed metadata field would
# require more than local memory. We still compute and audit all safe metadata,
# but use this priority subset for the HGB metadata model.
MAX_METADATA_FEATURES_FOR_HGB = 18
METADATA_FEATURE_PRIORITY = [
    "meta_official_latitude",
    "meta_official_longitude",
    "meta_official_elevation",
    "meta_official_drainage_area",
    "meta_official_average_discharge",
    "meta_official_catchment_average_precip_mm_yr",
    "meta_official_site_type_code",
    "meta_official_land_cover_code",
    "meta_train_n_rows",
    "meta_train_coverage_days",
    "meta_train_dominant_interval_minutes",
    "meta_train_00010_median",
    "meta_train_00095_median",
    "meta_train_63680_median",
    "meta_train_00060_median",
    "meta_train_00065_median",
    "meta_train_00010_std",
    "meta_train_00095_std",
]

ALIAS_CANDIDATES = {
    SITE_COL: ["site_id", "Site_ID", "SITE_ID", "site", "site_no", "station_id"],
    TIME_COL: ["timestamp", "datetime", "date_time", "time", "TIMESTAMP"],
    TARGET_COL: ["00400", "pH", "ph"],
    "00010": ["00010", "Temperature_C", "temperature", "temp"],
    "00095": ["00095", "Specific_Conductance_uS_cm", "specific_conductance"],
    "63680": ["63680", "Turbidity_FNU", "turbidity", "Turbidity [NTU]*.1"],
    "00060": ["00060", "discharge", "Discharge_cfs"],
    "00065": ["00065", "Gage_Height_ft", "gage_height", "gage_height_ft"],
}

FORBIDDEN_FEATURE_PATTERNS = [
    "00400",
    "ph",
    "target",
    "label",
    "event",
    "incident",
    "exceedance",
    "low_ph",
    "high_ph",
    "residual",
    "prediction",
    "pred",
    "rmse",
    "mae",
    "r2",
    "test",
    "val",
    "validation",
    "future",
]


# ---------------------------------------------------------------------------
# Logging and output helpers
# ---------------------------------------------------------------------------


def setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def ensure_output_path(path: Path) -> Path:
    resolved = path.resolve()
    output_root = OUTPUT_DIR.resolve()
    if output_root != resolved and output_root not in resolved.parents:
        raise ValueError(f"Refusing to write outside output folder: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    path = ensure_output_path(OUTPUT_DIR / filename)
    df.to_csv(path, index=False)
    logging.info("Wrote %s (%s rows, %s columns)", path, len(df), len(df.columns))
    return path


def write_json(payload: dict[str, Any], filename: str) -> Path:
    path = ensure_output_path(OUTPUT_DIR / filename)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logging.info("Wrote %s", path)
    return path


def write_text(text: str, filename: str) -> Path:
    path = ensure_output_path(OUTPUT_DIR / filename)
    path.write_text(text, encoding="utf-8")
    logging.info("Wrote %s", path)
    return path


def write_parquet(df: pd.DataFrame, filename: str) -> Path:
    path = ensure_output_path(OUTPUT_DIR / filename)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    logging.info("Wrote %s (%s rows, %s columns)", path, len(df), len(df.columns))
    return path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def normalise_site_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    text = re.sub(r"\s+", "", text)
    if text.isdigit() and len(text) < 8:
        return text.zfill(8)
    return text


def site_id_from_filename(path: Path) -> str:
    match = re.match(r"^([A-Za-z]?\d{5,}|E\d+[A-Za-z]?)", path.stem)
    return normalise_site_id(match.group(1)) if match else ""


def safe_float(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        if isinstance(value, str) and value.strip() == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def sanitize_token(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def infer_existing_column(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    lower_lookup = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for col in columns:
            if candidate_lower in col.lower():
                return col
    if required:
        raise ValueError(f"Could not infer column from candidates {candidates} in columns {columns}")
    return None


def dominant_interval_minutes(timestamps: pd.Series) -> float:
    values = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()
    if len(values) < 2:
        return np.nan
    diffs = values.diff().dt.total_seconds().div(60)
    diffs = diffs[(diffs > 0) & diffs.notna()]
    if diffs.empty:
        return np.nan
    mode = diffs.round(3).mode(dropna=True)
    return float(mode.iloc[0]) if not mode.empty else float(diffs.median())


def threshold_label(fraction: float) -> str:
    return f"top_{int(round(fraction * 100))}pct"


def normalise_event_label(value: Any) -> str:
    text = str(value).strip().lower()
    if "low" in text:
        return "pH_exceedance_low"
    if "high" in text:
        return "pH_exceedance_high"
    return "none"


# ---------------------------------------------------------------------------
# Candidate audit and site selection
# ---------------------------------------------------------------------------


def timestamp_range_from_parquet_stats(pf: pq.ParquetFile, time_col: str | None) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    if time_col is None:
        return pd.NaT, pd.NaT, "no timestamp column"
    names = list(pf.schema_arrow.names)
    if time_col not in names:
        return pd.NaT, pd.NaT, "timestamp column absent from schema"
    idx = names.index(time_col)
    mins: list[Any] = []
    maxs: list[Any] = []
    for row_group_idx in range(pf.metadata.num_row_groups):
        column = pf.metadata.row_group(row_group_idx).column(idx)
        stats = column.statistics
        if stats is None or not stats.has_min_max:
            continue
        mins.append(stats.min)
        maxs.append(stats.max)
    if mins and maxs:
        ts_min = pd.to_datetime(min(mins), errors="coerce")
        ts_max = pd.to_datetime(max(maxs), errors="coerce")
        return ts_min, ts_max, "timestamp range from parquet footer statistics"

    dates = re.findall(r"(20\d{6})", pf.metadata.created_by or "")
    return pd.NaT, pd.NaT, "timestamp footer statistics unavailable"


def timestamp_range_from_filename(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = re.findall(r"(20\d{6})", path.stem)
    if len(dates) >= 2:
        return pd.to_datetime(dates[-2], format="%Y%m%d", errors="coerce"), pd.to_datetime(dates[-1], format="%Y%m%d", errors="coerce")
    return pd.NaT, pd.NaT


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        logging.warning("Registry not found: %s", REGISTRY_PATH)
        return {}
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    logging.info("Loaded registry with %s site entries", len(registry))
    return registry


def inspect_processed_parquet_files(registry: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(PROCESSED_DIR.glob("*.parquet")):
        try:
            pf = pq.ParquetFile(path)
            columns = list(pf.schema_arrow.names)
            n_rows = int(pf.metadata.num_rows)
        except Exception as exc:
            logging.warning("Failed to inspect parquet %s: %s", path, exc)
            continue
        site_id = site_id_from_filename(path)
        time_col = infer_existing_column(columns, ALIAS_CANDIDATES[TIME_COL], required=False)
        ts_min, ts_max, ts_note = timestamp_range_from_parquet_stats(pf, time_col)
        if pd.isna(ts_min) or pd.isna(ts_max):
            fn_min, fn_max = timestamp_range_from_filename(path)
            ts_min = fn_min if pd.isna(ts_min) else ts_min
            ts_max = fn_max if pd.isna(ts_max) else ts_max
            if pd.notna(fn_min) and pd.notna(fn_max):
                ts_note = f"{ts_note}; timestamp range estimated from filename"
        coverage = float((ts_max - ts_min).total_seconds() / 86400.0) if pd.notna(ts_min) and pd.notna(ts_max) else np.nan
        flags = {f"has_{code}": code in columns for code in REQUIRED_CODES}
        has_all = all(flags[f"has_{code}"] for code in REQUIRED_CODES)
        rows.append(
            {
                "site_id": site_id,
                "processed_file": rel(path),
                **flags,
                "has_all_required_variables": bool(has_all),
                "n_rows_estimated": n_rows,
                "timestamp_min": ts_min,
                "timestamp_max": ts_max,
                "coverage_days_estimated": coverage,
                "metadata_match": site_id in registry,
                "selected_for_processed200": False,
                "selection_rank": np.nan,
                "selection_reason": "",
                "timestamp_range_note": ts_note,
            }
        )
    audit = pd.DataFrame(rows)
    if audit.empty:
        raise RuntimeError(f"No parquet files found in {PROCESSED_DIR}")

    # One best parquet per site; then strict full-feature selection.
    audit = audit.sort_values(["site_id", "has_all_required_variables", "n_rows_estimated"], ascending=[True, False, False])
    audit = audit.drop_duplicates("site_id", keep="first").reset_index(drop=True)
    strict = audit[audit["has_all_required_variables"]].copy()
    strict = strict.sort_values(
        ["metadata_match", "n_rows_estimated", "coverage_days_estimated", "site_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected_sites = set(strict.head(TARGET_SITE_COUNT)["site_id"])
    rank_map = {site: idx + 1 for idx, site in enumerate(strict.head(TARGET_SITE_COUNT)["site_id"].tolist())}
    audit["selected_for_processed200"] = audit["site_id"].isin(selected_sites)
    audit["selection_rank"] = audit["site_id"].map(rank_map)
    audit.loc[audit["selected_for_processed200"], "selection_reason"] = (
        "strict full-feature parquet; registry matched where available; ranked by row count and estimated coverage"
    )
    audit.loc[~audit["has_all_required_variables"], "selection_reason"] = "not strict full-feature candidate"
    audit.loc[audit["has_all_required_variables"] & ~audit["selected_for_processed200"], "selection_reason"] = (
        "strict candidate but outside top 200 by row count/coverage"
    )
    audit = audit.sort_values(["has_all_required_variables", "selected_for_processed200", "selection_rank", "site_id"], ascending=[False, False, True, True])
    return audit.reset_index(drop=True)


def build_selected_sites(audit: pd.DataFrame) -> pd.DataFrame:
    selected = audit[audit["selected_for_processed200"]].copy()
    selected = selected.sort_values("selection_rank").reset_index(drop=True)
    return selected[
        [
            "site_id",
            "selection_rank",
            "processed_file",
            "metadata_match",
            "n_rows_estimated",
            "coverage_days_estimated",
            "selection_reason",
        ]
    ]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def load_ph_events() -> pd.DataFrame:
    if not EVENT_FILE.exists():
        logging.warning("Regulatory event file not found: %s", EVENT_FILE)
        return pd.DataFrame(columns=["site_id_norm", "incident_start", "incident_end", "ph_event_label"])
    events = pd.read_csv(EVENT_FILE, dtype={"site_id": "string"}, low_memory=False)
    events["site_id_norm"] = events["site_id"].map(normalise_site_id)
    events = events[events["label"].isin(PH_EVENT_LABELS)].copy()
    events["incident_start"] = pd.to_datetime(events["incident_start"], errors="coerce")
    events["incident_end"] = pd.to_datetime(events["incident_end"], errors="coerce")
    events = events.dropna(subset=["incident_start", "incident_end"])
    events["ph_event_label"] = events["label"].map(normalise_event_label)
    logging.info("Loaded %s pH regulatory event rows across %s sites", len(events), events["site_id_norm"].nunique())
    return events[["site_id_norm", "incident_start", "incident_end", "ph_event_label"]]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(df[TIME_COL], errors="coerce")
    df["hour"] = timestamp.dt.hour.astype("int16")
    df["month"] = timestamp.dt.month.astype("int16")
    df["day_of_year"] = timestamp.dt.dayofyear.astype("int16")
    df["sin_hour"] = np.sin(2 * math.pi * df["hour"] / 24.0).astype("float32")
    df["cos_hour"] = np.cos(2 * math.pi * df["hour"] / 24.0).astype("float32")
    df["sin_day"] = np.sin(2 * math.pi * df["day_of_year"] / 365.25).astype("float32")
    df["cos_day"] = np.cos(2 * math.pi * df["day_of_year"] / 365.25).astype("float32")
    return df


def add_chronological_split(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    train_end = int(math.floor(n * 0.70))
    val_end = int(math.floor(n * 0.85))
    split = np.empty(n, dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"
    split[val_end:] = "test"
    df[SPLIT_COL] = split
    return df


def add_event_labels_for_site(df: pd.DataFrame, site_events: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(TIME_COL, kind="mergesort").copy()
    df[EVENT_COL] = np.int8(0)
    df[EVENT_LABEL_COL] = "none"
    if site_events.empty or df.empty:
        return df
    events = site_events.sort_values("incident_start")[["incident_start", "incident_end", "ph_event_label"]].copy()
    aligned = pd.merge_asof(
        df[[TIME_COL]].sort_values(TIME_COL),
        events,
        left_on=TIME_COL,
        right_on="incident_start",
        direction="backward",
    )
    in_event = aligned["incident_end"].notna() & (aligned[TIME_COL] <= aligned["incident_end"])
    df[EVENT_COL] = in_event.astype("int8").to_numpy()
    df[EVENT_LABEL_COL] = aligned["ph_event_label"].where(in_event, "none").fillna("none").to_numpy()
    return df


def build_column_mapping(path: Path) -> tuple[dict[str, str], list[str]]:
    pf = pq.ParquetFile(path)
    columns = list(pf.schema_arrow.names)
    mapping: dict[str, str] = {}
    for target, candidates in ALIAS_CANDIDATES.items():
        found = infer_existing_column(columns, candidates, required=False)
        if found is not None:
            mapping[found] = target
    missing = [col for col in [TIME_COL, TARGET_COL, *EXOGENOUS_COLUMNS] if col not in mapping.values()]
    if missing:
        raise ValueError(f"{path.name} is missing required columns/aliases after strict selection: {missing}")
    read_cols = list(dict.fromkeys(mapping.keys()))
    return mapping, read_cols


def clean_site_frame(raw: pd.DataFrame, mapping: dict[str, str], fallback_site_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if TIME_COL not in raw.columns and getattr(raw.index, "name", None) in {TIME_COL, "datetime"}:
        raw = raw.reset_index()
    elif TIME_COL not in raw.columns and isinstance(raw.index, pd.DatetimeIndex):
        raw = raw.reset_index().rename(columns={"index": TIME_COL})
    alias_used = {source: target for source, target in mapping.items() if source != target}
    df = raw.rename(columns=mapping).copy()
    if SITE_COL not in df.columns:
        df[SITE_COL] = fallback_site_id
    else:
        df[SITE_COL] = df[SITE_COL].map(normalise_site_id)
        df[SITE_COL] = df[SITE_COL].replace("", fallback_site_id).fillna(fallback_site_id)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

    for col in [TARGET_COL, *EXOGENOUS_COLUMNS]:
        values = pd.to_numeric(df[col], errors="coerce")
        values = values.mask(values <= -999.0)
        if col == TARGET_COL:
            values = values.mask((values < 0.0) | (values > 14.0))
        df[col] = values.astype("float32")

    initial_rows = len(df)
    missing_time_or_target = int(df[TIME_COL].isna().sum() + df[TARGET_COL].isna().sum())
    missing_predictor_rows = int(df[EXOGENOUS_COLUMNS].isna().any(axis=1).sum())
    df = df.dropna(subset=[TIME_COL, TARGET_COL, *EXOGENOUS_COLUMNS]).copy()
    df = df.sort_values([SITE_COL, TIME_COL], kind="mergesort")
    duplicate_count = int(df.duplicated([SITE_COL, TIME_COL]).sum())
    if duplicate_count:
        df = df.drop_duplicates([SITE_COL, TIME_COL], keep="first")
    cleaned_rows = len(df)
    df = add_chronological_split(df)
    df = add_time_features(df)
    for col in [TARGET_COL, *EXOGENOUS_COLUMNS, "sin_hour", "cos_hour", "sin_day", "cos_day"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    for col in ["month", "hour", "day_of_year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("int16")
    stats = {
        "initial_rows": int(initial_rows),
        "cleaned_rows": int(cleaned_rows),
        "dropped_rows": int(initial_rows - cleaned_rows),
        "missing_time_or_target_rows_before_cleaning": missing_time_or_target,
        "missing_predictor_rows_before_cleaning": missing_predictor_rows,
        "duplicate_site_timestamp_count": duplicate_count,
        "alias_mapping_used": "; ".join(f"{src}->{dst}" for src, dst in alias_used.items()) if alias_used else "",
    }
    return df, stats


def train_metadata_for_site(df: pd.DataFrame) -> dict[str, Any]:
    train = df[df[SPLIT_COL].eq("train")].copy()
    record: dict[str, Any] = {"site_id": str(df[SITE_COL].iloc[0]) if not df.empty else ""}
    if train.empty:
        record.update(
            {
                "meta_train_n_rows": 0,
                "meta_train_coverage_days": np.nan,
                "meta_train_dominant_interval_minutes": np.nan,
            }
        )
    else:
        start = train[TIME_COL].min()
        end = train[TIME_COL].max()
        record.update(
            {
                "meta_train_n_rows": int(len(train)),
                "meta_train_coverage_days": float((end - start).total_seconds() / 86400.0) if pd.notna(start) and pd.notna(end) else np.nan,
                "meta_train_dominant_interval_minutes": dominant_interval_minutes(train[TIME_COL]),
            }
        )
    for col in EXOGENOUS_COLUMNS:
        series = pd.to_numeric(train[col], errors="coerce") if not train.empty else pd.Series(dtype="float32")
        q25 = series.quantile(0.25) if len(series) else np.nan
        q75 = series.quantile(0.75) if len(series) else np.nan
        record.update(
            {
                f"meta_train_{col}_median": series.median() if len(series) else np.nan,
                f"meta_train_{col}_mean": series.mean() if len(series) else np.nan,
                f"meta_train_{col}_std": series.std() if len(series) else np.nan,
                f"meta_train_{col}_iqr": q75 - q25 if pd.notna(q25) and pd.notna(q75) else np.nan,
                f"meta_train_{col}_p05": series.quantile(0.05) if len(series) else np.nan,
                f"meta_train_{col}_p95": series.quantile(0.95) if len(series) else np.nan,
                f"meta_train_{col}_missing_rate": float(series.isna().mean()) if len(series) else np.nan,
            }
        )
    return record


def build_processed200_dataset(selected: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_path = ensure_output_path(OUTPUT_DIR / "processed200_model_ready.parquet")
    if output_path.exists():
        output_path.unlink()
    event_groups = {site: group for site, group in events.groupby("site_id_norm")}
    writer: pq.ParquetWriter | None = None
    train_meta_rows: list[dict[str, Any]] = []
    site_build_rows: list[dict[str, Any]] = []
    totals = {
        "n_rows": 0,
        "train_rows": 0,
        "val_rows": 0,
        "test_rows": 0,
        "duplicate_site_timestamp_rows_removed": 0,
        "n_test_event_rows": 0,
        "n_test_low_event_rows": 0,
        "n_test_high_event_rows": 0,
    }
    timestamp_min = pd.NaT
    timestamp_max = pd.NaT
    final_sites: list[str] = []
    split_method = "chronological per site: first 70% train, next 15% val, final 15% test"

    try:
        for idx, row in selected.iterrows():
            site_id = normalise_site_id(row["site_id"])
            path = PROJECT_ROOT / str(row["processed_file"]).replace("/", "\\")
            logging.info("Reading selected site %s/%s: %s", idx + 1, len(selected), path.name)
            mapping, read_cols = build_column_mapping(path)
            raw = pd.read_parquet(path, columns=read_cols)
            site_df, stats = clean_site_frame(raw, mapping, site_id)
            del raw
            gc.collect()
            site_df = site_df[site_df[SITE_COL].eq(site_id)].copy()
            site_df = add_event_labels_for_site(site_df, event_groups.get(site_id, pd.DataFrame()))
            if site_df.empty:
                site_build_rows.append({"site_id": site_id, "processed_file": rel(path), "final_rows": 0, "dropped_after_cleaning": True, **stats})
                continue
            final_sites.append(site_id)
            train_meta_rows.append(train_metadata_for_site(site_df))
            split_counts = site_df[SPLIT_COL].value_counts().to_dict()
            test = site_df[site_df[SPLIT_COL].eq("test")]
            totals["n_rows"] += int(len(site_df))
            totals["train_rows"] += int(split_counts.get("train", 0))
            totals["val_rows"] += int(split_counts.get("val", 0))
            totals["test_rows"] += int(split_counts.get("test", 0))
            totals["duplicate_site_timestamp_rows_removed"] += int(stats["duplicate_site_timestamp_count"])
            totals["n_test_event_rows"] += int(test[EVENT_COL].sum())
            totals["n_test_low_event_rows"] += int(test[EVENT_LABEL_COL].map(normalise_event_label).eq("pH_exceedance_low").sum())
            totals["n_test_high_event_rows"] += int(test[EVENT_LABEL_COL].map(normalise_event_label).eq("pH_exceedance_high").sum())
            site_min = site_df[TIME_COL].min()
            site_max = site_df[TIME_COL].max()
            timestamp_min = site_min if pd.isna(timestamp_min) or site_min < timestamp_min else timestamp_min
            timestamp_max = site_max if pd.isna(timestamp_max) or site_max > timestamp_max else timestamp_max
            site_build_rows.append(
                {
                    "site_id": site_id,
                    "processed_file": rel(path),
                    "final_rows": int(len(site_df)),
                    "train_rows": int(split_counts.get("train", 0)),
                    "val_rows": int(split_counts.get("val", 0)),
                    "test_rows": int(split_counts.get("test", 0)),
                    "test_event_rows": int(test[EVENT_COL].sum()),
                    "dropped_after_cleaning": False,
                    **stats,
                }
            )
            site_df = site_df[MODEL_READY_COLUMNS].copy()
            table = pa.Table.from_pandas(site_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
            del site_df, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    site_build = pd.DataFrame(site_build_rows)
    train_meta = pd.DataFrame(train_meta_rows)
    summary = pd.DataFrame(
        [
            {
                "n_rows": int(totals["n_rows"]),
                "n_sites": int(len(final_sites)),
                "train_rows": int(totals["train_rows"]),
                "val_rows": int(totals["val_rows"]),
                "test_rows": int(totals["test_rows"]),
                "timestamp_min": timestamp_min,
                "timestamp_max": timestamp_max,
                "duplicate_site_timestamp_count": 0,
                "duplicate_site_timestamp_rows_removed": int(totals["duplicate_site_timestamp_rows_removed"]),
                "missing_target_after_cleaning": 0,
                "missing_predictors_after_cleaning": 0,
                "n_test_event_rows": int(totals["n_test_event_rows"]),
                "n_test_low_event_rows": int(totals["n_test_low_event_rows"]),
                "n_test_high_event_rows": int(totals["n_test_high_event_rows"]),
                "final_site_count": int(len(final_sites)),
                "sites_dropped_after_row_cleaning": int(TARGET_SITE_COUNT - len(final_sites)),
                "split_method": split_method,
            }
        ]
    )
    write_csv(summary, "processed200_dataset_summary.csv")
    write_csv(site_build, "processed200_site_build_audit.csv")
    logging.info("Built model-ready dataset with %s rows and %s sites", totals["n_rows"], len(final_sites))
    return summary, train_meta, site_build


# ---------------------------------------------------------------------------
# Metadata construction
# ---------------------------------------------------------------------------


def extract_dominant_land_cover(catchment: dict[str, Any]) -> tuple[str, float]:
    land_cover = catchment.get("watershed_land_cover")
    if not isinstance(land_cover, dict) or not land_cover:
        return "unknown", np.nan
    items = [(str(k), safe_float(v)) for k, v in land_cover.items()]
    items = [(k, v) for k, v in items if pd.notna(v)]
    if not items:
        return "unknown", np.nan
    key, pct = max(items, key=lambda item: item[1])
    return key, pct


def build_official_metadata(site_ids: list[str], registry: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for site_id in site_ids:
        entry = registry.get(site_id, {})
        source_description = parse_jsonish(entry.get("source_description")) if isinstance(entry, dict) else {}
        discharge_stats = (
            entry.get("discharge_description", {}).get("statistics", {})
            if isinstance(entry.get("discharge_description"), dict)
            else {}
        )
        catchment = entry.get("catchment_analysis", {}) if isinstance(entry, dict) else {}
        if not isinstance(catchment, dict):
            catchment = {}
        dominant_land_cover, dominant_land_cover_pct = extract_dominant_land_cover(catchment)
        drainage_area = entry.get("drainage_area_km2", source_description.get("drainage_area_sqkm")) if isinstance(entry, dict) else np.nan
        rows.append(
            {
                "site_id": site_id,
                "meta_official_source": entry.get("source") if isinstance(entry, dict) else None,
                "meta_official_units": entry.get("units") if isinstance(entry, dict) else None,
                "meta_official_basin": entry.get("basin_cd") if isinstance(entry, dict) else None,
                "meta_official_country": entry.get("country_cd") if isinstance(entry, dict) else None,
                "meta_official_state": source_description.get("state_cd"),
                "meta_official_county": source_description.get("county_cd"),
                "meta_official_site_type": source_description.get("site_type_cd") or (entry.get("site_type") if isinstance(entry, dict) else None),
                "meta_official_status": entry.get("status") if isinstance(entry, dict) else None,
                "meta_official_latitude": safe_float(entry.get("latitude")) if isinstance(entry, dict) else np.nan,
                "meta_official_longitude": safe_float(entry.get("longitude")) if isinstance(entry, dict) else np.nan,
                "meta_official_elevation": safe_float(entry.get("elevation")) if isinstance(entry, dict) else np.nan,
                "meta_official_drainage_area": safe_float(drainage_area),
                "meta_official_average_discharge": safe_float(discharge_stats.get("mean_discharge")),
                "meta_official_catchment_average_precip_mm_yr": safe_float(catchment.get("average_precip_mm_yr")),
                "meta_official_catchment_average_evapo_trans_mm_yr": safe_float(catchment.get("average_evapo_trans_mm_yr")),
                "meta_official_catchment_population_est_2020": safe_float(catchment.get("population_est_2020")),
                "meta_official_catchment_watershed_elevation_mean_m": safe_float(catchment.get("watershed_elevation_mean_m")),
                "meta_official_catchment_watershed_dams_no": safe_float(catchment.get("watershed_dams_no")),
                "meta_official_catchment_watershed_dams_cap_m3": safe_float(catchment.get("watershed_dams_cap_m3")),
                "meta_official_catchment_irrigation_area_km2": safe_float(catchment.get("irrigation_area_km2")),
                "meta_official_catchment_irrigation_area_pct": safe_float(catchment.get("irrigation_area_pct")),
                "meta_official_land_cover": dominant_land_cover,
                "meta_official_land_cover_pct": dominant_land_cover_pct,
                "metadata_match": site_id in registry,
            }
        )
    return pd.DataFrame(rows)


def feature_leakage_reason(feature: str) -> str | None:
    lower = feature.lower()
    allowed_train_exact = {
        "meta_train_n_rows",
        "meta_train_coverage_days",
        "meta_train_dominant_interval_minutes",
    }
    allowed_train_prefixes = [f"meta_train_{code}_" for code in EXOGENOUS_COLUMNS]
    if lower in allowed_train_exact or any(lower.startswith(prefix) for prefix in allowed_train_prefixes):
        if "00400" in lower:
            return "contains target parameter code 00400"
        return None
    if "00400" in lower:
        return "contains target parameter code 00400"
    if "low_ph" in lower or "high_ph" in lower:
        return "contains low/high pH marker"
    if re.search(r"(^|_)ph($|_)", lower) or lower.startswith("ph_") or lower.endswith("_ph") or "has_ph" in lower:
        return "contains pH/target marker"
    tokens = set(re.split(r"[^a-z0-9]+", lower))
    for token in FORBIDDEN_FEATURE_PATTERNS:
        if token in {"ph", "00400", "low_ph", "high_ph"}:
            continue
        if token in tokens or token in lower:
            return f"contains forbidden marker `{token}`"
    if "val" in tokens:
        return "contains validation-set marker"
    return None


def select_hgb_metadata_features(computed_features: list[str]) -> list[str]:
    selected = [feature for feature in METADATA_FEATURE_PRIORITY if feature in computed_features]
    if len(selected) < MAX_METADATA_FEATURES_FOR_HGB:
        for feature in computed_features:
            if feature in selected:
                continue
            if feature.startswith("meta_missing_"):
                continue
            selected.append(feature)
            if len(selected) >= MAX_METADATA_FEATURES_FOR_HGB:
                break
    return selected[:MAX_METADATA_FEATURES_FOR_HGB]


def make_model_ready_metadata(
    official: pd.DataFrame,
    train_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    meta = official.merge(train_meta, on="site_id", how="left")
    audit_rows: list[dict[str, Any]] = []
    feature_cols: list[str] = []
    official_features: list[str] = []
    train_features: list[str] = []
    missing_features: list[str] = []
    categorical_maps: dict[str, dict[str, int]] = {}

    categorical_cols = [
        "meta_official_source",
        "meta_official_units",
        "meta_official_basin",
        "meta_official_country",
        "meta_official_state",
        "meta_official_county",
        "meta_official_site_type",
        "meta_official_status",
        "meta_official_land_cover",
    ]
    numeric_cols = [
        c
        for c in meta.columns
        if c.startswith("meta_official_")
        and c not in categorical_cols
        and c not in {"meta_official_land_cover"}
    ]
    train_cols = [c for c in meta.columns if c.startswith("meta_train_")]

    for col in categorical_cols:
        if col not in meta.columns:
            continue
        missing = meta[col].isna() | meta[col].astype("string").str.strip().eq("")
        miss_col = f"meta_missing_{col.replace('meta_', '')}"
        meta[miss_col] = missing.astype("int8")
        meta[col] = meta[col].astype("string").fillna("unknown").replace("", "unknown")
        values = sorted(meta[col].dropna().astype(str).unique().tolist())
        mapping = {value: idx for idx, value in enumerate(values)}
        code_col = f"{col}_code"
        meta[code_col] = meta[col].astype(str).map(mapping).fillna(mapping.get("unknown", 0)).astype("int16")
        categorical_maps[col] = mapping
        audit_rows.append(
            {
                "metadata_field": col,
                "source": "official_registry",
                "dtype_group": "categorical",
                "missing_count_before_imputation": int(missing.sum()),
                "missing_rate_before_imputation": float(missing.mean()) if len(meta) else np.nan,
                "model_feature_name": code_col,
                "missing_indicator": miss_col,
                "used_in_model": True,
            }
        )
        for candidate in [code_col, miss_col]:
            reason = feature_leakage_reason(candidate)
            if reason is None:
                feature_cols.append(candidate)
                official_features.append(candidate)
                if candidate == miss_col:
                    missing_features.append(candidate)

    for col in numeric_cols + train_cols:
        if col not in meta.columns or col == "metadata_match":
            continue
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
        missing = meta[col].isna()
        miss_col = f"meta_missing_{col.replace('meta_', '')}"
        if missing.any():
            meta[miss_col] = missing.astype("int8")
            fill_value = meta[col].median()
            if pd.isna(fill_value):
                fill_value = 0.0
            meta[col] = meta[col].fillna(fill_value)
        else:
            fill_value = np.nan
        meta[col] = meta[col].astype("float32")
        reason = feature_leakage_reason(col)
        used = reason is None
        if used:
            feature_cols.append(col)
            if col.startswith("meta_train_"):
                train_features.append(col)
            else:
                official_features.append(col)
        if missing.any() and feature_leakage_reason(miss_col) is None:
            meta[miss_col] = meta[miss_col].astype("int8")
            feature_cols.append(miss_col)
            missing_features.append(miss_col)
            if col.startswith("meta_train_"):
                train_features.append(miss_col)
            else:
                official_features.append(miss_col)
        audit_rows.append(
            {
                "metadata_field": col,
                "source": "train_only_exogenous" if col.startswith("meta_train_") else "official_registry",
                "dtype_group": "numeric",
                "missing_count_before_imputation": int(missing.sum()),
                "missing_rate_before_imputation": float(missing.mean()) if len(meta) else np.nan,
                "model_feature_name": col,
                "missing_indicator": miss_col if missing.any() else "",
                "imputation_value": fill_value,
                "used_in_model": used,
                "leakage_screen_reason": reason or "passed",
            }
        )

    # Preserve order and final leakage screen.
    final_features: list[str] = []
    leakage_rows: list[dict[str, Any]] = []
    for feature in dict.fromkeys(feature_cols):
        reason = feature_leakage_reason(feature)
        if reason:
            leakage_rows.append({"feature_name": feature, "status": "excluded", "reason": reason})
        else:
            final_features.append(feature)
            leakage_rows.append({"feature_name": feature, "status": "kept", "reason": "passed final leakage screen"})

    audit = pd.DataFrame(audit_rows)
    leakage_screen = pd.DataFrame(leakage_rows)
    write_csv(leakage_screen, "processed200_metadata_leakage_screen.csv")
    hgb_features = select_hgb_metadata_features(final_features)
    metadata_feature_set = {
        "metadata_features_used": hgb_features,
        "metadata_features_computed_after_leakage_screen": final_features,
        "official_metadata_features": [f for f in hgb_features if f in official_features],
        "train_derived_metadata_features": [f for f in hgb_features if f in train_features],
        "missingness_features": [f for f in hgb_features if f in missing_features],
        "computed_official_metadata_features": [f for f in final_features if f in official_features],
        "computed_train_derived_metadata_features": [f for f in final_features if f in train_features],
        "hgb_metadata_feature_selection_note": (
            "All safe metadata fields were computed and audited, but the HGB metadata model uses a compact "
            f"{len(hgb_features)}-feature priority subset because this sklearn build validates HGB inputs as "
            "float64 and the full 38.5M-row training matrix would exceed local memory."
        ),
        "categorical_encoding_maps": categorical_maps,
        "official_metadata_source": rel(REGISTRY_PATH),
        "train_derived_source": "processed200_model_ready.parquet split == train",
        "leakage_controls": {
            "no_target_00400_metadata": True,
            "no_event_metadata": True,
            "no_residual_or_prediction_metadata": True,
            "train_derived_uses_only": EXOGENOUS_COLUMNS,
        },
    }
    display_cols = [
        "meta_official_latitude",
        "meta_official_longitude",
        "meta_official_drainage_area",
        "meta_official_average_discharge",
        "meta_official_site_type",
        "meta_official_land_cover",
        "metadata_match",
    ]
    return_cols = list(dict.fromkeys(["site_id", *final_features, *display_cols]))
    return meta[return_cols].copy(), audit, metadata_feature_set


def write_model_ready_with_metadata(metadata_ready: pd.DataFrame, metadata_features: list[str]) -> None:
    source_path = OUTPUT_DIR / "processed200_model_ready.parquet"
    output_path = ensure_output_path(OUTPUT_DIR / "processed200_model_ready_with_metadata.parquet")
    if output_path.exists():
        output_path.unlink()
    meta_index = metadata_ready.set_index("site_id")[metadata_features]
    pf = pq.ParquetFile(source_path)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        for batch in pf.iter_batches(batch_size=250_000):
            df = batch.to_pandas()
            joined = df.join(meta_index, on=SITE_COL)
            if joined[metadata_features].isna().any().any():
                missing_sites = joined.loc[joined[metadata_features].isna().any(axis=1), SITE_COL].drop_duplicates().head(10).tolist()
                raise ValueError(f"Metadata merge introduced missing values for sites: {missing_sites}")
            table = pa.Table.from_pandas(joined, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += len(joined)
            del df, joined, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    logging.info("Wrote metadata-merged parquet with %s rows", total_rows)


# ---------------------------------------------------------------------------
# Feature matrices, model training, and predictions
# ---------------------------------------------------------------------------


def fit_site_encoder(model_ready_path: Path) -> dict[str, int]:
    train_sites: set[str] = set()
    pf = pq.ParquetFile(model_ready_path)
    for batch in pf.iter_batches(columns=[SITE_COL, SPLIT_COL], batch_size=1_000_000):
        df = batch.to_pandas()
        train_sites.update(df.loc[df[SPLIT_COL].eq("train"), SITE_COL].astype(str).tolist())
    site_map = {site: idx for idx, site in enumerate(sorted(train_sites))}
    if not site_map:
        raise ValueError("No training sites found for site encoder.")
    logging.info("Fitted site encoder on %s train sites", len(site_map))
    return site_map


def split_counts_from_model_ready(model_ready_path: Path) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    pf = pq.ParquetFile(model_ready_path)
    for batch in pf.iter_batches(columns=[SPLIT_COL], batch_size=1_000_000):
        split = batch.to_pandas()[SPLIT_COL]
        value_counts = split.value_counts().to_dict()
        for key in counts:
            counts[key] += int(value_counts.get(key, 0))
    return counts


def metadata_matrix(metadata_ready: pd.DataFrame, metadata_features: list[str], site_map: dict[str, int]) -> np.ndarray:
    indexed = metadata_ready.set_index("site_id")
    ordered_sites = [site for site, _ in sorted(site_map.items(), key=lambda item: item[1])]
    missing_sites = [site for site in ordered_sites if site not in indexed.index]
    if missing_sites:
        raise ValueError(f"Missing metadata for train sites: {missing_sites[:10]}")
    matrix = indexed.loc[ordered_sites, metadata_features].apply(pd.to_numeric, errors="coerce")
    if matrix.isna().any().any():
        missing = matrix.isna().sum()
        raise ValueError(f"Model-ready metadata contains NaN values: {missing[missing > 0].to_dict()}")
    return matrix.astype("float32").to_numpy(copy=True)


def build_feature_block(
    df: pd.DataFrame,
    feature_names: list[str],
    site_map: dict[str, int],
    meta_matrix: np.ndarray | None,
    metadata_features: list[str],
) -> np.ndarray:
    n = len(df)
    x = np.empty((n, len(feature_names)), dtype=np.float32)
    col_pos = 0
    for feature in DYNAMIC_FEATURES:
        if feature in feature_names:
            x[:, col_pos] = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=np.float32, copy=False)
            col_pos += 1
    if SITE_CODE_COL in feature_names:
        site_codes = df[SITE_COL].astype(str).map(site_map).to_numpy(dtype=np.int16)
        x[:, col_pos] = site_codes.astype(np.float32)
        col_pos += 1
    remaining = feature_names[col_pos:]
    if remaining:
        if meta_matrix is None:
            raise ValueError("metadata matrix required for metadata features")
        site_codes = df[SITE_COL].astype(str).map(site_map).to_numpy(dtype=np.int16)
        meta_positions = [metadata_features.index(feature) for feature in remaining]
        x[:, col_pos:] = meta_matrix[site_codes][:, meta_positions]
    return x


def create_train_memmap(
    model_id: str,
    feature_names: list[str],
    site_map: dict[str, int],
    meta_matrix: np.ndarray | None,
    metadata_features: list[str],
    n_train_rows: int,
) -> tuple[np.memmap, np.memmap, Path, Path]:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    x_path = ensure_output_path(TMP_DIR / f"{model_id}_X_train_float32.dat")
    y_path = ensure_output_path(TMP_DIR / f"{model_id}_y_train_float32.dat")
    if x_path.exists():
        x_path.unlink()
    if y_path.exists():
        y_path.unlink()
    x = np.memmap(x_path, mode="w+", dtype=np.float32, shape=(n_train_rows, len(feature_names)))
    y = np.memmap(y_path, mode="w+", dtype=np.float32, shape=(n_train_rows,))
    pf = pq.ParquetFile(OUTPUT_DIR / "processed200_model_ready.parquet")
    offset = 0
    columns = list(dict.fromkeys([SITE_COL, SPLIT_COL, TARGET_COL, *DYNAMIC_FEATURES]))
    for batch in pf.iter_batches(columns=columns, batch_size=500_000):
        df = batch.to_pandas()
        train = df[df[SPLIT_COL].eq("train")]
        if train.empty:
            continue
        n = len(train)
        x[offset : offset + n, :] = build_feature_block(train, feature_names, site_map, meta_matrix, metadata_features)
        y[offset : offset + n] = pd.to_numeric(train[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        offset += n
        del df, train
        gc.collect()
    if offset != n_train_rows:
        raise RuntimeError(f"Expected {n_train_rows} train rows but wrote {offset}")
    x.flush()
    y.flush()
    logging.info("Created train memmap for %s: %s rows x %s features", model_id, n_train_rows, len(feature_names))
    return x, y, x_path, y_path


def train_hgb_model(model_id: str, x_train: np.memmap, y_train: np.memmap, params: dict[str, Any]) -> HistGradientBoostingRegressor:
    logging.info("Training %s with HGB params: %s", model_id, params)
    started = time.time()
    model = HistGradientBoostingRegressor(**params)
    model.fit(x_train, y_train)
    logging.info("Finished %s in %.1f seconds", model_id, time.time() - started)
    return model


def predict_split_to_parquet(
    model: HistGradientBoostingRegressor,
    model_id: str,
    split: str,
    feature_names: list[str],
    site_map: dict[str, int],
    meta_matrix: np.ndarray | None,
    metadata_features: list[str],
    output_name: str,
) -> None:
    output_path = ensure_output_path(OUTPUT_DIR / output_name)
    if output_path.exists():
        output_path.unlink()
    pf = pq.ParquetFile(OUTPUT_DIR / "processed200_model_ready.parquet")
    columns = list(dict.fromkeys([SITE_COL, TIME_COL, SPLIT_COL, TARGET_COL, EVENT_COL, EVENT_LABEL_COL, *DYNAMIC_FEATURES]))
    writer: pq.ParquetWriter | None = None
    total = 0
    try:
        for batch in pf.iter_batches(columns=columns, batch_size=500_000):
            df = batch.to_pandas()
            work = df[df[SPLIT_COL].eq(split)].copy()
            if work.empty:
                continue
            x = build_feature_block(work, feature_names, site_map, meta_matrix, metadata_features)
            y_pred = model.predict(x).astype("float32")
            y_true = pd.to_numeric(work[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32, copy=False)
            out = work[[SITE_COL, TIME_COL, SPLIT_COL, TARGET_COL, EVENT_COL, EVENT_LABEL_COL]].copy()
            out["y_pred"] = y_pred
            out["residual"] = (y_true - y_pred).astype("float32")
            out["abs_residual"] = np.abs(out["residual"].to_numpy(dtype=np.float32)).astype("float32")
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
            total += len(out)
            del df, work, x, y_pred, y_true, out, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    logging.info("Wrote %s predictions for %s rows to %s", split, total, output_path)


def run_one_model(
    model_id: str,
    feature_names: list[str],
    site_map: dict[str, int],
    meta_matrix: np.ndarray | None,
    metadata_features: list[str],
    split_counts: dict[str, int],
    params: dict[str, Any],
) -> None:
    x_train, y_train, x_path, y_path = create_train_memmap(
        model_id=model_id,
        feature_names=feature_names,
        site_map=site_map,
        meta_matrix=meta_matrix,
        metadata_features=metadata_features,
        n_train_rows=split_counts["train"],
    )
    try:
        model = train_hgb_model(model_id, x_train, y_train, params)
    finally:
        del x_train, y_train
        gc.collect()
        for path in [x_path, y_path]:
            if path.exists():
                path.unlink()
    prefix = "M200_A_no_metadata" if model_id == M200_A_ID else "M200_B_with_metadata"
    predict_split_to_parquet(
        model,
        model_id,
        "val",
        feature_names,
        site_map,
        meta_matrix,
        metadata_features,
        f"prediction_val_{prefix}.parquet",
    )
    predict_split_to_parquet(
        model,
        model_id,
        "test",
        feature_names,
        site_map,
        meta_matrix,
        metadata_features,
        f"prediction_test_{prefix}.parquet",
    )
    del model
    gc.collect()


# ---------------------------------------------------------------------------
# Metrics and validation
# ---------------------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.square(y_true - np.mean(y_true))))
    if np.isclose(denom, 0.0):
        return np.nan
    return float(1.0 - np.sum(np.square(y_true - y_pred)) / denom)


def top_residual_mask(abs_residual: np.ndarray, fraction: float) -> np.ndarray:
    n = len(abs_residual)
    if n == 0:
        return np.zeros(0, dtype=bool)
    n_top = max(1, int(math.ceil(n * fraction)))
    idx = np.argpartition(abs_residual, -n_top)[-n_top:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def split_metric_row(model_id: str, model_name: str, split: str, pred: pd.DataFrame, metadata_used: bool) -> dict[str, Any]:
    y_true = pred[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
    y_pred = pred["y_pred"].to_numpy(dtype=np.float32, copy=False)
    return {
        "model_id": model_id,
        "model_name": model_name,
        "split": split,
        "n_rows": int(len(pred)),
        "n_sites": int(pred[SITE_COL].nunique()),
        "metadata_used": bool(metadata_used),
        "site_id_used": True,
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score_manual(y_true, y_pred),
    }


def residual_event_rows(model_id: str, pred: pd.DataFrame) -> list[dict[str, Any]]:
    abs_residual = pred["abs_residual"].to_numpy(dtype=np.float32, copy=False)
    event = pred[EVENT_COL].to_numpy(dtype=np.int8, copy=False) > 0
    n_test = len(pred)
    n_event = int(event.sum())
    base_rate = n_event / n_test if n_test else np.nan
    rows: list[dict[str, Any]] = []
    for fraction in THRESHOLDS:
        top = top_residual_mask(abs_residual, fraction)
        n_top = int(top.sum())
        n_top_event = int((event & top).sum())
        top_rate = n_top_event / n_top if n_top else np.nan
        rows.append(
            {
                "model_id": model_id,
                "threshold": threshold_label(fraction),
                "top_fraction": fraction,
                "n_test_rows": int(n_test),
                "n_event_rows": n_event,
                "base_event_rate": base_rate,
                "n_top_rows": n_top,
                "n_top_event_rows": n_top_event,
                "top_event_rate": top_rate,
                "enrichment": top_rate / base_rate if base_rate and not np.isclose(base_rate, 0) else np.nan,
            }
        )
    return rows


def low_high_enrichment_rows(model_id: str, pred: pd.DataFrame) -> list[dict[str, Any]]:
    abs_residual = pred["abs_residual"].to_numpy(dtype=np.float32, copy=False)
    labels = pred[EVENT_LABEL_COL].map(normalise_event_label).to_numpy()
    n_test = len(pred)
    rows: list[dict[str, Any]] = []
    for fraction in THRESHOLDS:
        top = top_residual_mask(abs_residual, fraction)
        n_top = int(top.sum())
        for label in PH_EVENT_LABELS:
            label_mask = labels == label
            n_label = int(label_mask.sum())
            base_rate = n_label / n_test if n_test else np.nan
            n_top_label = int((label_mask & top).sum())
            top_rate = n_top_label / n_top if n_top else np.nan
            rows.append(
                {
                    "model_id": model_id,
                    "threshold": threshold_label(fraction),
                    "top_fraction": fraction,
                    "event_label": label,
                    "n_label_rows": n_label,
                    "base_label_rate": base_rate,
                    "n_top_rows": n_top,
                    "n_top_label_rows": n_top_label,
                    "top_label_rate": top_rate,
                    "label_enrichment": top_rate / base_rate if base_rate and not np.isclose(base_rate, 0) else np.nan,
                }
            )
    return rows


def event_level_hit_rate_rows(model_id: str, pred: pd.DataFrame) -> list[dict[str, Any]]:
    work = pred[[SITE_COL, TIME_COL, EVENT_COL, EVENT_LABEL_COL, "abs_residual"]].copy()
    work["event_label_norm"] = work[EVENT_LABEL_COL].map(normalise_event_label)
    work = work.sort_values([SITE_COL, TIME_COL], kind="mergesort").reset_index(drop=True)
    event_mask = (work[EVENT_COL].to_numpy(dtype=np.int8, copy=False) > 0) & (work["event_label_norm"].to_numpy() != "none")
    prev_site = work[SITE_COL].astype("string").shift(1)
    prev_label = work["event_label_norm"].shift(1)
    prev_event = pd.Series(event_mask).shift(1).fillna(False).to_numpy(dtype=bool)
    block_start = event_mask & (
        (~prev_event)
        | (work[SITE_COL].astype("string") != prev_site).to_numpy()
        | (work["event_label_norm"] != prev_label).to_numpy()
    )
    work["event_window_id"] = np.where(event_mask, np.cumsum(block_start), np.nan)
    n_windows = int(work.loc[event_mask, "event_window_id"].dropna().nunique())
    abs_residual = work["abs_residual"].to_numpy(dtype=np.float32, copy=False)
    rows: list[dict[str, Any]] = []
    for fraction in THRESHOLDS:
        top = top_residual_mask(abs_residual, fraction)
        hit_ids = work.loc[event_mask & top, "event_window_id"].dropna().astype("int64")
        n_hit = int(hit_ids.nunique())
        rows.append(
            {
                "model_id": model_id,
                "threshold": threshold_label(fraction),
                "top_fraction": fraction,
                "event_window_method": "approximate_contiguous_blocks_within_site_and_label_on_sorted_test_rows",
                "n_event_windows": n_windows,
                "n_event_windows_hit": n_hit,
                "event_level_hit_rate": n_hit / n_windows if n_windows else np.nan,
            }
        )
    return rows


def site_level_metric_rows(model_id: str, model_name: str, pred: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for site_id, group in pred.groupby(SITE_COL, sort=True, observed=True):
        y_true = group[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
        y_pred = group["y_pred"].to_numpy(dtype=np.float32, copy=False)
        rows.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "site_id": str(site_id),
                "n_test_rows": int(len(group)),
                "n_test_event_rows": int(group[EVENT_COL].sum()),
                "rmse": rmse(y_true, y_pred),
                "mae": mae(y_true, y_pred),
                "r2": r2_score_manual(y_true, y_pred),
            }
        )
    return rows


def load_prediction(path: Path) -> pd.DataFrame:
    cols = [SITE_COL, TIME_COL, SPLIT_COL, TARGET_COL, "y_pred", "residual", "abs_residual", EVENT_COL, EVENT_LABEL_COL]
    df = pd.read_parquet(path, columns=cols)
    df[SITE_COL] = df[SITE_COL].astype("string")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    return df


def compute_all_metrics(metadata_ready: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = [
        (M200_A_ID, M200_A_NAME, False, "M200_A_no_metadata"),
        (M200_B_ID, M200_B_NAME, True, "M200_B_with_metadata"),
    ]
    metric_rows: list[dict[str, Any]] = []
    residual_rows_all: list[dict[str, Any]] = []
    low_high_rows_all: list[dict[str, Any]] = []
    event_hit_rows_all: list[dict[str, Any]] = []
    site_rows_all: list[dict[str, Any]] = []
    for model_id, model_name, metadata_used, prefix in specs:
        val = load_prediction(OUTPUT_DIR / f"prediction_val_{prefix}.parquet")
        test = load_prediction(OUTPUT_DIR / f"prediction_test_{prefix}.parquet")
        metric_rows.append(split_metric_row(model_id, model_name, "val", val, metadata_used))
        metric_rows.append(split_metric_row(model_id, model_name, "test", test, metadata_used))
        residual_rows_all.extend(residual_event_rows(model_id, test))
        low_high_rows_all.extend(low_high_enrichment_rows(model_id, test))
        event_hit_rows_all.extend(event_level_hit_rate_rows(model_id, test))
        site_rows_all.extend(site_level_metric_rows(model_id, model_name, test))
        del val, test
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    residual = pd.DataFrame(residual_rows_all)
    low_high = pd.DataFrame(low_high_rows_all)
    event_hit = pd.DataFrame(event_hit_rows_all)
    site_metrics = pd.DataFrame(site_rows_all)
    effect = build_site_level_effect(site_metrics, metadata_ready)
    write_csv(metrics, "model_comparison_metrics.csv")
    write_csv(residual, "residual_event_validation.csv")
    write_csv(low_high, "low_high_event_enrichment.csv")
    write_csv(event_hit, "event_level_hit_rate.csv")
    write_csv(site_metrics, "site_level_metrics.csv")
    write_csv(effect, "site_level_metadata_effect.csv")
    return metrics, residual, low_high, event_hit, site_metrics, effect


def build_site_level_effect(site_metrics: pd.DataFrame, metadata_ready: pd.DataFrame) -> pd.DataFrame:
    a = site_metrics[site_metrics["model_id"].eq(M200_A_ID)].copy()
    b = site_metrics[site_metrics["model_id"].eq(M200_B_ID)].copy()
    effect = a[["site_id", "rmse", "r2", "n_test_rows", "n_test_event_rows"]].merge(
        b[["site_id", "rmse", "r2"]],
        on="site_id",
        how="inner",
        suffixes=("_M200_A_no_metadata", "_M200_B_with_metadata"),
    )
    effect["rmse_delta_B_minus_A"] = effect["rmse_M200_B_with_metadata"] - effect["rmse_M200_A_no_metadata"]
    effect["r2_delta_B_minus_A"] = effect["r2_M200_B_with_metadata"] - effect["r2_M200_A_no_metadata"]
    keep_meta = [
        "site_id",
        "meta_official_latitude",
        "meta_official_longitude",
        "meta_official_drainage_area",
        "meta_official_average_discharge",
        "meta_official_site_type",
        "meta_official_land_cover",
    ]
    available = [col for col in keep_meta if col in metadata_ready.columns]
    effect = effect.merge(metadata_ready[available], on="site_id", how="left")
    return effect.sort_values("rmse_delta_B_minus_A").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures and report
# ---------------------------------------------------------------------------


def savefig(fig: Any, filename: str) -> None:
    png = ensure_output_path(FIGURES_DIR / filename)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    logging.info("Wrote figure %s", png)


def create_figures(
    metrics: pd.DataFrame,
    residual: pd.DataFrame,
    low_high: pd.DataFrame,
    event_hit: pd.DataFrame,
    site_effect: pd.DataFrame,
    metadata_audit: pd.DataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning("Matplotlib unavailable; skipping figures: %s", exc)
        return

    colors = {M200_A_ID: "#4C78A8", M200_B_ID: "#F58518"}
    labels = {M200_A_ID: "M200_A no metadata", M200_B_ID: "M200_B with metadata"}
    test_metrics = metrics[metrics["split"].eq("test")].set_index("model_id").loc[[M200_A_ID, M200_B_ID]].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    x = np.arange(len(test_metrics))
    axes[0].bar(x, test_metrics["rmse"], color=[colors[m] for m in test_metrics["model_id"]])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[m] for m in test_metrics["model_id"]], rotation=20, ha="right")
    axes[0].set_ylabel("Test RMSE")
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.7)
    axes[1].bar(x, test_metrics["r2"], color=[colors[m] for m in test_metrics["model_id"]])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels[m] for m in test_metrics["model_id"]], rotation=20, ha="right")
    axes[1].set_ylabel("Test R2")
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.7)
    fig.tight_layout()
    savefig(fig, "rmse_r2_processed200_comparison.png")
    plt.close(fig)

    top5 = residual[residual["threshold"].eq("top_5pct")].set_index("model_id").loc[[M200_A_ID, M200_B_ID]].reset_index()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(np.arange(len(top5)), top5["enrichment"], color=[colors[m] for m in top5["model_id"]])
    ax.set_xticks(np.arange(len(top5)))
    ax.set_xticklabels([labels[m] for m in top5["model_id"]], rotation=20, ha="right")
    ax.set_ylabel("Top 5% event enrichment")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    savefig(fig, "top5_enrichment_processed200_comparison.png")
    plt.close(fig)

    hit5 = event_hit[event_hit["threshold"].eq("top_5pct")].set_index("model_id").loc[[M200_A_ID, M200_B_ID]].reset_index()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(np.arange(len(hit5)), hit5["event_level_hit_rate"], color=[colors[m] for m in hit5["model_id"]])
    ax.set_xticks(np.arange(len(hit5)))
    ax.set_xticklabels([labels[m] for m in hit5["model_id"]], rotation=20, ha="right")
    ax.set_ylabel("Event-level top 5% hit rate")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    savefig(fig, "event_hit_rate_processed200_comparison.png")
    plt.close(fig)

    lh5 = low_high[low_high["threshold"].eq("top_5pct")].copy()
    pivot = lh5.pivot(index="event_label", columns="model_id", values="label_enrichment").reindex(PH_EVENT_LABELS)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(pivot.index))
    width = 0.36
    ax.bar(x - width / 2, pivot[M200_A_ID], width=width, color=colors[M200_A_ID], label=labels[M200_A_ID])
    ax.bar(x + width / 2, pivot[M200_B_ID], width=width, color=colors[M200_B_ID], label=labels[M200_B_ID])
    ax.set_xticks(x)
    ax.set_xticklabels(["Low pH", "High pH"])
    ax.set_ylabel("Top 5% label enrichment")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    savefig(fig, "low_high_enrichment_processed200_comparison.png")
    plt.close(fig)

    effect = site_effect.sort_values("rmse_delta_B_minus_A").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(len(effect)), effect["rmse_delta_B_minus_A"], color=np.where(effect["rmse_delta_B_minus_A"] < 0, "#4C78A8", "#E45756"))
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Sites sorted by RMSE delta")
    ax.set_ylabel("RMSE delta B - A")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    savefig(fig, "site_level_rmse_delta_processed200.png")
    plt.close(fig)

    miss = metadata_audit.copy()
    miss = miss[miss["missing_rate_before_imputation"].notna()].sort_values("missing_rate_before_imputation", ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(np.arange(len(miss)), miss["missing_rate_before_imputation"], color="#72B7B2")
    ax.set_yticks(np.arange(len(miss)))
    ax.set_yticklabels(miss["metadata_field"])
    ax.invert_yaxis()
    ax.set_xlabel("Missingness before imputation")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    savefig(fig, "processed200_metadata_missingness.png")
    plt.close(fig)

    create_optional_full50_figure(metrics, residual, low_high, event_hit)


def create_optional_full50_figure(metrics: pd.DataFrame, residual: pd.DataFrame, low_high: pd.DataFrame, event_hit: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows: list[dict[str, Any]] = []
    if FULL50_METRICS_PATH.exists():
        m = pd.read_csv(FULL50_METRICS_PATH)
        r = pd.read_csv(FULL50_RESIDUAL_PATH) if FULL50_RESIDUAL_PATH.exists() else pd.DataFrame()
        h = pd.read_csv(FULL50_HIT_PATH) if FULL50_HIT_PATH.exists() else pd.DataFrame()
        lh = pd.read_csv(FULL50_LOW_HIGH_PATH) if FULL50_LOW_HIGH_PATH.exists() else pd.DataFrame()
        for old_id, label in [("M1_no_metadata", "50 M1"), ("M2_with_metadata", "50 M2")]:
            test = m[(m["model_id"].eq(old_id)) & (m["split"].eq("test"))]
            if test.empty:
                continue
            row = {"label": label, "rmse": float(test["rmse"].iloc[0]), "r2": float(test["r2"].iloc[0])}
            top5 = r[(r["model_id"].eq(old_id)) & (r["threshold"].eq("top_5pct"))] if not r.empty else pd.DataFrame()
            hit5 = h[(h["model_id"].eq(old_id)) & (h["threshold"].eq("top_5pct"))] if not h.empty else pd.DataFrame()
            low5 = lh[(lh["model_id"].eq(old_id)) & (lh["threshold"].eq("top_5pct") & lh["event_label"].eq("pH_exceedance_low"))] if not lh.empty else pd.DataFrame()
            high5 = lh[(lh["model_id"].eq(old_id)) & (lh["threshold"].eq("top_5pct") & lh["event_label"].eq("pH_exceedance_high"))] if not lh.empty else pd.DataFrame()
            row["top5"] = float(top5["enrichment"].iloc[0]) if not top5.empty else np.nan
            row["hit5"] = float(hit5["event_level_hit_rate"].iloc[0]) if not hit5.empty else np.nan
            row["low5"] = float(low5["label_enrichment"].iloc[0]) if not low5.empty else np.nan
            row["high5"] = float(high5["label_enrichment"].iloc[0]) if not high5.empty else np.nan
            rows.append(row)
    for model_id, label in [(M200_A_ID, "200 A"), (M200_B_ID, "200 B")]:
        test = metrics[(metrics["model_id"].eq(model_id)) & (metrics["split"].eq("test"))]
        row = {"label": label, "rmse": float(test["rmse"].iloc[0]), "r2": float(test["r2"].iloc[0])}
        row["top5"] = float(residual[(residual["model_id"].eq(model_id)) & (residual["threshold"].eq("top_5pct"))]["enrichment"].iloc[0])
        row["hit5"] = float(event_hit[(event_hit["model_id"].eq(model_id)) & (event_hit["threshold"].eq("top_5pct"))]["event_level_hit_rate"].iloc[0])
        row["low5"] = float(low_high[(low_high["model_id"].eq(model_id)) & (low_high["threshold"].eq("top_5pct") & low_high["event_label"].eq("pH_exceedance_low"))]["label_enrichment"].iloc[0])
        row["high5"] = float(low_high[(low_high["model_id"].eq(model_id)) & (low_high["threshold"].eq("top_5pct") & low_high["event_label"].eq("pH_exceedance_high"))]["label_enrichment"].iloc[0])
        rows.append(row)
    comp = pd.DataFrame(rows)
    if len(comp) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    x = np.arange(len(comp))
    axes[0].bar(x, comp["rmse"], color="#4C78A8")
    axes[0].set_title("RMSE")
    axes[1].bar(x, comp["r2"], color="#F58518")
    axes[1].set_title("R2")
    axes[2].bar(x, comp["top5"], color="#72B7B2")
    axes[2].set_title("Top 5% enrichment")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(comp["label"], rotation=25, ha="right")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    fig.tight_layout()
    savefig(fig, "processed200_vs_full50_summary.png")
    plt.close(fig)


def format_float(value: Any, digits: int = 6) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "NA"


def write_report(
    candidate_audit: pd.DataFrame,
    selected: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    metadata_audit: pd.DataFrame,
    metadata_feature_set: dict[str, Any],
    metrics: pd.DataFrame,
    residual: pd.DataFrame,
    low_high: pd.DataFrame,
    event_hit: pd.DataFrame,
    site_effect: pd.DataFrame,
) -> str:
    ds = dataset_summary.iloc[0]
    test_metrics = metrics[metrics["split"].eq("test")].set_index("model_id")
    val_metrics = metrics[metrics["split"].eq("val")].set_index("model_id")
    top5 = residual[residual["threshold"].eq("top_5pct")].set_index("model_id")
    hit5 = event_hit[event_hit["threshold"].eq("top_5pct")].set_index("model_id")
    low5 = low_high[(low_high["threshold"].eq("top_5pct")) & (low_high["event_label"].eq("pH_exceedance_low"))].set_index("model_id")
    high5 = low_high[(low_high["threshold"].eq("top_5pct")) & (low_high["event_label"].eq("pH_exceedance_high"))].set_index("model_id")

    a_rmse = float(test_metrics.loc[M200_A_ID, "rmse"])
    b_rmse = float(test_metrics.loc[M200_B_ID, "rmse"])
    a_r2 = float(test_metrics.loc[M200_A_ID, "r2"])
    b_r2 = float(test_metrics.loc[M200_B_ID, "r2"])
    a_top5 = float(top5.loc[M200_A_ID, "enrichment"])
    b_top5 = float(top5.loc[M200_B_ID, "enrichment"])
    interpretation = (
        "At 200 sites, metadata improves pH prediction but weakens residual-event alignment, confirming the trade-off between predictive accuracy and residual sensitivity observed in the 50-site experiment."
        if b_rmse < a_rmse and b_r2 > a_r2 and b_top5 < a_top5
        else "At 200 sites, metadata improves both pH prediction and residual-event alignment, suggesting that site/catchment context becomes more useful as site heterogeneity increases."
        if b_rmse < a_rmse and b_r2 > a_r2 and b_top5 > a_top5
        else "At 200 sites, metadata provides limited additional benefit, possibly because site identity and dynamic predictors already capture much observed-site heterogeneity."
    )
    n_improved = int((site_effect["rmse_delta_B_minus_A"] < 0).sum())
    n_worsened = int((site_effect["rmse_delta_B_minus_A"] > 0).sum())
    strict_candidates = int(candidate_audit["has_all_required_variables"].sum())
    metadata_features = metadata_feature_set.get("metadata_features_used", [])
    processed_sites_scanned = candidate_audit["site_id"].dropna().astype(str)
    processed_sites_scanned = int(processed_sites_scanned[processed_sites_scanned.ne("")].nunique())

    comp_rows = [
        ["200-site M200_A no metadata", a_rmse, a_r2, a_top5, float(hit5.loc[M200_A_ID, "event_level_hit_rate"]), float(low5.loc[M200_A_ID, "label_enrichment"]), float(high5.loc[M200_A_ID, "label_enrichment"])],
        ["200-site M200_B with metadata", b_rmse, b_r2, b_top5, float(hit5.loc[M200_B_ID, "event_level_hit_rate"]), float(low5.loc[M200_B_ID, "label_enrichment"]), float(high5.loc[M200_B_ID, "label_enrichment"])],
    ]
    if FULL50_METRICS_PATH.exists():
        fm = pd.read_csv(FULL50_METRICS_PATH)
        fr = pd.read_csv(FULL50_RESIDUAL_PATH) if FULL50_RESIDUAL_PATH.exists() else pd.DataFrame()
        fh = pd.read_csv(FULL50_HIT_PATH) if FULL50_HIT_PATH.exists() else pd.DataFrame()
        fl = pd.read_csv(FULL50_LOW_HIGH_PATH) if FULL50_LOW_HIGH_PATH.exists() else pd.DataFrame()
        for model_id, label in [("M1_no_metadata", "50-site M1 no metadata"), ("M2_with_metadata", "50-site M2 with metadata")]:
            mt = fm[(fm["model_id"].eq(model_id)) & (fm["split"].eq("test"))]
            if mt.empty:
                continue
            r5 = fr[(fr["model_id"].eq(model_id)) & (fr["threshold"].eq("top_5pct"))] if not fr.empty else pd.DataFrame()
            h5 = fh[(fh["model_id"].eq(model_id)) & (fh["threshold"].eq("top_5pct"))] if not fh.empty else pd.DataFrame()
            lo = fl[(fl["model_id"].eq(model_id)) & (fl["threshold"].eq("top_5pct")) & (fl["event_label"].eq("pH_exceedance_low"))] if not fl.empty else pd.DataFrame()
            hi = fl[(fl["model_id"].eq(model_id)) & (fl["threshold"].eq("top_5pct")) & (fl["event_label"].eq("pH_exceedance_high"))] if not fl.empty else pd.DataFrame()
            comp_rows.insert(
                0 if model_id == "M1_no_metadata" else 1,
                [
                    label,
                    float(mt["rmse"].iloc[0]),
                    float(mt["r2"].iloc[0]),
                    float(r5["enrichment"].iloc[0]) if not r5.empty else np.nan,
                    float(h5["event_level_hit_rate"].iloc[0]) if not h5.empty else np.nan,
                    float(lo["label_enrichment"].iloc[0]) if not lo.empty else np.nan,
                    float(hi["label_enrichment"].iloc[0]) if not hi.empty else np.nan,
                ],
            )

    comp_table = "| Model | Test RMSE | Test R2 | Top5 enrichment | Event-level top5 hit | Low-pH top5 | High-pH top5 |\n"
    comp_table += "|---|---:|---:|---:|---:|---:|---:|\n"
    for row in comp_rows:
        comp_table += f"| {row[0]} | {format_float(row[1])} | {format_float(row[2])} | {format_float(row[3])} | {format_float(row[4])} | {format_float(row[5])} | {format_float(row[6])} |\n"

    report = f"""# Phase 3B Processed200 HGB Metadata Experiment

## 1. Purpose

This phase expands the full-data HGB metadata comparison from 50 sites to a strict processed-data 200-site experiment. The comparison is M200_A without metadata versus M200_B with safe official and train-only exogenous metadata.

## 2. Data Source

- Main time-series data source: `data/processed`
- Static metadata source: `data/reference/sites_register.json`
- Regulatory events: `data/reference/regulatory_events.csv`, validation only

## 3. Site Selection

- Processed parquet sites scanned: {processed_sites_scanned:,}
- Strict full-feature candidates: {strict_candidates:,}
- Selected for processed200 attempt: {len(selected):,}
- Final modelled sites after row-level cleaning: {int(ds['final_site_count']):,}
- Sites dropped after row-level cleaning: {int(ds['sites_dropped_after_row_cleaning']):,}

Selection used only processed parquet schemas and metadata/footer row counts. Sites were not selected by model performance, residual metrics, or pH-event availability.

## 4. Dataset Construction

- Final rows: {int(ds['n_rows']):,}
- Train rows: {int(ds['train_rows']):,}
- Validation rows: {int(ds['val_rows']):,}
- Test rows: {int(ds['test_rows']):,}
- Time range: {ds['timestamp_min']} to {ds['timestamp_max']}
- Split method: {ds['split_method']}
- Duplicate site-timestamp rows after cleaning: {int(ds['duplicate_site_timestamp_count']):,}
- Duplicate site-timestamp rows removed during cleaning: {int(ds.get('duplicate_site_timestamp_rows_removed', 0)):,}
- Missing target after cleaning: {int(ds['missing_target_after_cleaning']):,}
- Missing required predictors after cleaning: {int(ds['missing_predictors_after_cleaning']):,}
- Test pH event rows: {int(ds['n_test_event_rows']):,}
- Test low-pH event rows: {int(ds['n_test_low_event_rows']):,}
- Test high-pH event rows: {int(ds['n_test_high_event_rows']):,}

## 5. Metadata Construction

Official metadata came from `sites_register.json`. Train-only derived metadata used only the training rows and only exogenous predictors: `{', '.join(EXOGENOUS_COLUMNS)}`. No target pH, event labels, residuals, predictions, validation statistics, or test statistics were used.

- Metadata match rate among final modelled sites: {float(selected['metadata_match'].mean()):.1%}
- Metadata features used by M200_B: {len(metadata_features):,}
- Safe metadata features computed after leakage screen: {len(metadata_feature_set.get('metadata_features_computed_after_leakage_screen', [])):,}
- Metadata fields audited before imputation: {len(metadata_audit):,}

M200_B uses a compact priority subset of metadata features because this local sklearn HGB implementation validates inputs as float64; using every computed metadata feature with 38.5M training rows would exceed local memory. The full computed metadata table is still saved and audited.

## 6. Model Setup

Both models used `HistGradientBoostingRegressor` with identical hyperparameters:

```json
{json.dumps(HGB_PARAMS, indent=2)}
```

Training used all rows where `split == "train"`. Event labels, pH lag, previous pH, pH summaries, residual fields, and prediction fields were excluded from predictors.

## 7. Prediction Performance

Validation metrics:

- M200_A RMSE/R2: {format_float(val_metrics.loc[M200_A_ID, 'rmse'])} / {format_float(val_metrics.loc[M200_A_ID, 'r2'])}
- M200_B RMSE/R2: {format_float(val_metrics.loc[M200_B_ID, 'rmse'])} / {format_float(val_metrics.loc[M200_B_ID, 'r2'])}

Test metrics:

- M200_A RMSE/R2: {format_float(a_rmse)} / {format_float(a_r2)}
- M200_B RMSE/R2: {format_float(b_rmse)} / {format_float(b_r2)}

Metadata improved RMSE at 200 sites: {'yes' if b_rmse < a_rmse else 'no'}.
Metadata improved R2 at 200 sites: {'yes' if b_r2 > a_r2 else 'no'}.

## 8. Residual-Event Validation

- M200_A top 5% enrichment: {format_float(a_top5)}
- M200_B top 5% enrichment: {format_float(b_top5)}
- M200_A event-level top 5% hit rate: {format_float(hit5.loc[M200_A_ID, 'event_level_hit_rate'])}
- M200_B event-level top 5% hit rate: {format_float(hit5.loc[M200_B_ID, 'event_level_hit_rate'])}
- M200_A low-pH top 5% enrichment: {format_float(low5.loc[M200_A_ID, 'label_enrichment'])}
- M200_B low-pH top 5% enrichment: {format_float(low5.loc[M200_B_ID, 'label_enrichment'])}
- M200_A high-pH top 5% enrichment: {format_float(high5.loc[M200_A_ID, 'label_enrichment'])}
- M200_B high-pH top 5% enrichment: {format_float(high5.loc[M200_B_ID, 'label_enrichment'])}

Event-level hit rate uses approximate contiguous event blocks within each site and event label on sorted test rows.

Metadata {'strengthened' if b_top5 > a_top5 else 'weakened or did not strengthen'} residual-event alignment at the top 5% row threshold. Low-pH enrichment is {'stronger' if float(low5.loc[M200_B_ID, 'label_enrichment']) > float(high5.loc[M200_B_ID, 'label_enrichment']) else 'not stronger'} than high-pH enrichment for M200_B.

## 9. Site-Level Effects

- Sites improved by metadata by RMSE: {n_improved:,}
- Sites worsened by metadata by RMSE: {n_worsened:,}

Most improved sites are the top rows of `site_level_metadata_effect.csv`; positive RMSE delta means metadata worsened that site.

## 10. Comparison with 50-Site Result

{comp_table}

## 11. Interpretation

{interpretation}

## 12. Limitations

- The strict full-feature candidate count is tight around 200 sites.
- Train-derived metadata uses processed exogenous variables after the strict row-cleaning step.
- Regulatory pH event labels are weak validation, not complete ground truth.
- Direct 300-site expansion may require alias-normalised variables or a relaxed predictor set.
- HGB remains a tabular baseline before GRU/LSTM sequence models.
- Optional permutation importance was skipped because the required full-row 200-site experiment already used 55M rows and two full HGB fits.

## 13. Next Step

If the 200-site experiment is stable, proceed to GRU/LSTM with metadata. If direct 300-site expansion is still required, run an alias-normalised 300-site eligibility check first and keep non-event HGB as a residual-detection reference.
"""
    write_text(report, "step3b_processed200_hgb_metadata_report.md")
    return interpretation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()
    started = time.time()
    logging.info("Starting Phase 3B processed200 HGB metadata experiment")
    registry = load_registry()

    candidate_audit = inspect_processed_parquet_files(registry)
    selected = build_selected_sites(candidate_audit)
    write_csv(candidate_audit, "processed200_candidate_site_audit.csv")
    write_csv(selected, "processed200_selected_sites.csv")
    if len(selected) < TARGET_SITE_COUNT:
        logging.warning("Only %s strict selected sites available; proceeding with largest valid strict set.", len(selected))

    events = load_ph_events()
    dataset_summary, train_meta, site_build = build_processed200_dataset(selected, events)
    final_site_ids = site_build.loc[~site_build["dropped_after_cleaning"], "site_id"].astype(str).tolist()
    selected_final = selected[selected["site_id"].isin(final_site_ids)].copy()

    official = build_official_metadata(final_site_ids, registry)
    metadata_ready, metadata_audit, metadata_feature_set = make_model_ready_metadata(official, train_meta)
    write_csv(metadata_audit, "processed200_metadata_audit.csv")
    write_json(metadata_feature_set, "processed200_metadata_feature_set.json")
    write_csv(metadata_ready, "processed200_site_metadata_model_ready.csv")
    metadata_features = metadata_feature_set["metadata_features_used"]

    write_model_ready_with_metadata(metadata_ready, metadata_features)
    base_pf = pq.ParquetFile(OUTPUT_DIR / "processed200_model_ready.parquet")
    meta_pf = pq.ParquetFile(OUTPUT_DIR / "processed200_model_ready_with_metadata.parquet")
    if base_pf.metadata.num_rows != meta_pf.metadata.num_rows:
        raise RuntimeError("Metadata merge changed row count.")

    feature_a = BASE_FEATURES
    feature_b = [*BASE_FEATURES, *metadata_features]
    write_json({"model_id": M200_A_ID, "features": feature_a, "n_features": len(feature_a)}, "feature_list_M200_A_no_metadata.json")
    write_json({"model_id": M200_B_ID, "features": feature_b, "n_features": len(feature_b)}, "feature_list_M200_B_with_metadata.json")
    write_json(
        {
            "model": "HistGradientBoostingRegressor",
            "hgb_params": HGB_PARAMS,
            "target": TARGET_COL,
            "dynamic_features": DYNAMIC_FEATURES,
            "site_code_feature": SITE_CODE_COL,
            "metadata_features": metadata_features,
            "training_rows_rule": 'all rows where split == "train"',
            "validation_rows_rule": 'all rows where split == "val"',
            "test_rows_rule": 'all rows where split == "test"',
        },
        "model_config.json",
    )

    model_ready_path = OUTPUT_DIR / "processed200_model_ready.parquet"
    site_map = fit_site_encoder(model_ready_path)
    split_counts = split_counts_from_model_ready(model_ready_path)
    meta_mat = metadata_matrix(metadata_ready, metadata_features, site_map)

    run_one_model(M200_A_ID, feature_a, site_map, None, metadata_features, split_counts, HGB_PARAMS)
    run_one_model(M200_B_ID, feature_b, site_map, meta_mat, metadata_features, split_counts, HGB_PARAMS)

    metrics, residual, low_high, event_hit, site_metrics, site_effect = compute_all_metrics(metadata_ready)
    create_figures(metrics, residual, low_high, event_hit, site_effect, metadata_audit)
    interpretation = write_report(
        candidate_audit,
        selected_final,
        dataset_summary,
        metadata_audit,
        metadata_feature_set,
        metrics,
        residual,
        low_high,
        event_hit,
        site_effect,
    )

    if TMP_DIR.exists():
        try:
            shutil.rmtree(TMP_DIR)
        except PermissionError as exc:
            logging.warning("Could not remove temporary memmap directory %s: %s", TMP_DIR, exc)

    test_metrics = metrics[metrics["split"].eq("test")].set_index("model_id")
    top5 = residual[residual["threshold"].eq("top_5pct")].set_index("model_id")
    hit5 = event_hit[event_hit["threshold"].eq("top_5pct")].set_index("model_id")
    low5 = low_high[(low_high["threshold"].eq("top_5pct")) & (low_high["event_label"].eq("pH_exceedance_low"))].set_index("model_id")
    high5 = low_high[(low_high["threshold"].eq("top_5pct")) & (low_high["event_label"].eq("pH_exceedance_high"))].set_index("model_id")
    ds = dataset_summary.iloc[0]
    processed_sites_scanned = candidate_audit["site_id"].dropna().astype(str)
    processed_sites_scanned = int(processed_sites_scanned[processed_sites_scanned.ne("")].nunique())
    logging.info("Phase 3B completed in %.1f seconds", time.time() - started)

    print(
        f"""
PHASE 3B PROCESSED200 HGB METADATA COMPLETE

Output folder:
{rel(OUTPUT_DIR)}/

Processed sites scanned:
{processed_sites_scanned}

Strict full-feature candidate sites:
{int(candidate_audit['has_all_required_variables'].sum())}

Selected sites attempted:
{len(selected)}

Final modelled sites:
{int(ds['final_site_count'])}

Final rows:
{int(ds['n_rows'])}

Train rows:
{int(ds['train_rows'])}

Validation rows:
{int(ds['val_rows'])}

Test rows:
{int(ds['test_rows'])}

Metadata match rate:
{float(selected_final['metadata_match'].mean()) if len(selected_final) else np.nan:.3f}

M200_A test RMSE:
{float(test_metrics.loc[M200_A_ID, 'rmse']):.6f}

M200_B test RMSE:
{float(test_metrics.loc[M200_B_ID, 'rmse']):.6f}

M200_A test R2:
{float(test_metrics.loc[M200_A_ID, 'r2']):.6f}

M200_B test R2:
{float(test_metrics.loc[M200_B_ID, 'r2']):.6f}

M200_A top 5% enrichment:
{float(top5.loc[M200_A_ID, 'enrichment']):.6f}

M200_B top 5% enrichment:
{float(top5.loc[M200_B_ID, 'enrichment']):.6f}

M200_A event-level top 5% hit rate:
{float(hit5.loc[M200_A_ID, 'event_level_hit_rate']):.6f}

M200_B event-level top 5% hit rate:
{float(hit5.loc[M200_B_ID, 'event_level_hit_rate']):.6f}

M200_A low-pH top 5% enrichment:
{float(low5.loc[M200_A_ID, 'label_enrichment']):.6f}

M200_B low-pH top 5% enrichment:
{float(low5.loc[M200_B_ID, 'label_enrichment']):.6f}

M200_A high-pH top 5% enrichment:
{float(high5.loc[M200_A_ID, 'label_enrichment']):.6f}

M200_B high-pH top 5% enrichment:
{float(high5.loc[M200_B_ID, 'label_enrichment']):.6f}

Metadata improved RMSE:
{'yes' if float(test_metrics.loc[M200_B_ID, 'rmse']) < float(test_metrics.loc[M200_A_ID, 'rmse']) else 'no'}

Metadata improved R2:
{'yes' if float(test_metrics.loc[M200_B_ID, 'r2']) > float(test_metrics.loc[M200_A_ID, 'r2']) else 'no'}

Metadata improved residual-event enrichment:
{'yes' if float(top5.loc[M200_B_ID, 'enrichment']) > float(top5.loc[M200_A_ID, 'enrichment']) else 'no'}

Main interpretation:
{interpretation}

Next recommended task:
Proceed to GRU/LSTM with metadata, or run alias-normalised 300-site eligibility check if 300-site expansion is still required.
""".strip()
    )


if __name__ == "__main__":
    main()
