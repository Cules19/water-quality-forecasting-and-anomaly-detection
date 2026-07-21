#!/usr/bin/env python
"""Phase 4.0 + 4.1 processed100 event-state coding and metadata merge.

This script uses the strict processed200 outputs as the fast path. It prepares
a nested top-100-site dataset, adds event-state columns, builds leakage-
controlled metadata, and merges that metadata. It does not train any model.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "analysis_outputs"
OUTPUT_DIR = ANALYSIS_DIR / "ph_processed100_step4_0_4_1_v1"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOG_PATH = OUTPUT_DIR / "processed100_step4_0_4_1_log.txt"

PROCESSED200_DIR = ANALYSIS_DIR / "ph_hgb_processed200_metadata_v1"
PROCESSED200_SELECTED = PROCESSED200_DIR / "processed200_selected_sites.csv"
PROCESSED200_MODEL_READY = PROCESSED200_DIR / "processed200_model_ready.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "reference" / "sites_register.json"
EVENT_FILE = PROJECT_ROOT / "data" / "reference" / "regulatory_events.csv"

TARGET_SITE_COUNT = 100
SITE_COL = "site_id"
TIME_COL = "timestamp"
SPLIT_COL = "split"
TARGET_COL = "00400"
EVENT_COL = "in_ph_event"
EVENT_LABEL_COL = "ph_event_label"
EVENT_STATE_COL = "event_state"
EVENT_STATE_NAME_COL = "event_state_name"
SAMPLE_WEIGHT_COL = "sample_weight_first_version"

REQUIRED_MODEL_READY_COLUMNS = [
    SITE_COL,
    TIME_COL,
    SPLIT_COL,
    TARGET_COL,
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
    EVENT_COL,
    EVENT_LABEL_COL,
]
EXOGENOUS_COLUMNS = ["00010", "00095", "63680", "00060", "00065"]
PH_EVENT_LABELS = ["pH_exceedance_low", "pH_exceedance_high"]

EVENT_STATE_DEFINITIONS = {
    "0": {
        "event_state_name": "normal",
        "description": "Row is not inside a pH regulatory event window.",
        "sample_weight_first_version": 1.0,
    },
    "1": {
        "event_state_name": "low_pH_event",
        "description": "Row is inside a pH low exceedance event window.",
        "sample_weight_first_version": 3.0,
    },
    "2": {
        "event_state_name": "high_pH_event",
        "description": "Row is inside a pH high exceedance event window.",
        "sample_weight_first_version": 3.0,
    },
    "3": {
        "event_state_name": "mixed_or_unknown",
        "description": "Row has unclear, mixed, or overlapping pH event state.",
        "sample_weight_first_version": 1.0,
    },
}

OFFICIAL_CATEGORICAL_COLUMNS = [
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
    root = OUTPUT_DIR.resolve()
    if root != resolved and root not in resolved.parents:
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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


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


def normalise_event_label(value: Any) -> str:
    text = str(value).strip().lower()
    if "low" in text and "high" in text:
        return "mixed"
    if "low" in text:
        return "pH_exceedance_low"
    if "high" in text:
        return "pH_exceedance_high"
    return "none"


def select_sites_fast_path() -> pd.DataFrame:
    if not PROCESSED200_SELECTED.exists() or not PROCESSED200_MODEL_READY.exists():
        raise FileNotFoundError(
            "Preferred processed200 fast-path files are missing. Fallback direct rebuild from data/processed "
            "was not run automatically to avoid an unintended heavy scan."
        )
    selected200 = pd.read_csv(PROCESSED200_SELECTED, dtype={"site_id": "string"})
    selected200["site_id"] = selected200["site_id"].map(normalise_site_id)
    selected200["selection_rank"] = pd.to_numeric(selected200["selection_rank"], errors="coerce")
    selected = selected200.sort_values("selection_rank").head(TARGET_SITE_COUNT).copy()
    selected["selection_rank"] = np.arange(1, len(selected) + 1, dtype=np.int32)
    selected["selected_for_processed100"] = True
    selected["selection_reason"] = "top 100 by strict processed200 selection_rank; nested fast path"
    out_cols = [
        "site_id",
        "selection_rank",
        "processed_file",
        "metadata_match",
        "n_rows_estimated",
        "coverage_days_estimated",
        "selected_for_processed100",
        "selection_reason",
    ]
    selected = selected[out_cols]
    write_csv(selected, "processed100_selected_sites.csv")
    return selected


def build_processed100_base(selected: pd.DataFrame) -> pd.DataFrame:
    selected_sites = set(selected["site_id"].astype(str).tolist())
    input_pf = pq.ParquetFile(PROCESSED200_MODEL_READY)
    missing_cols = [col for col in REQUIRED_MODEL_READY_COLUMNS if col not in input_pf.schema_arrow.names]
    if missing_cols:
        raise ValueError(f"processed200 model-ready parquet is missing required columns: {missing_cols}")

    output_path = ensure_output_path(OUTPUT_DIR / "processed100_model_ready.parquet")
    if output_path.exists():
        output_path.unlink()
    writer: pq.ParquetWriter | None = None
    totals = defaultdict(int)
    timestamp_min = pd.NaT
    timestamp_max = pd.NaT
    last_key: tuple[str, pd.Timestamp] | None = None
    duplicate_count = 0
    split_method = "inherited from processed200: chronological per site, first 70% train, next 15% val, final 15% test"

    try:
        for batch in input_pf.iter_batches(columns=REQUIRED_MODEL_READY_COLUMNS, batch_size=1_000_000):
            df = batch.to_pandas()
            df[SITE_COL] = df[SITE_COL].astype(str)
            df = df[df[SITE_COL].isin(selected_sites)].copy()
            if df.empty:
                continue
            df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
            df = df.sort_values([SITE_COL, TIME_COL], kind="mergesort")
            duplicate_count += int(df.duplicated([SITE_COL, TIME_COL]).sum())
            if last_key is not None:
                first_key = (str(df[SITE_COL].iloc[0]), pd.Timestamp(df[TIME_COL].iloc[0]))
                if first_key == last_key:
                    duplicate_count += 1
            last_key = (str(df[SITE_COL].iloc[-1]), pd.Timestamp(df[TIME_COL].iloc[-1]))

            split_counts = df[SPLIT_COL].value_counts().to_dict()
            totals["n_rows"] += len(df)
            totals["train_rows"] += int(split_counts.get("train", 0))
            totals["val_rows"] += int(split_counts.get("val", 0))
            totals["test_rows"] += int(split_counts.get("test", 0))
            test = df[df[SPLIT_COL].eq("test")]
            totals["n_test_event_rows"] += int(pd.to_numeric(test[EVENT_COL], errors="coerce").fillna(0).astype("int8").sum())
            test_labels = test[EVENT_LABEL_COL].map(normalise_event_label)
            totals["n_test_low_event_rows"] += int(test_labels.eq("pH_exceedance_low").sum())
            totals["n_test_high_event_rows"] += int(test_labels.eq("pH_exceedance_high").sum())
            site_min = df[TIME_COL].min()
            site_max = df[TIME_COL].max()
            timestamp_min = site_min if pd.isna(timestamp_min) or site_min < timestamp_min else timestamp_min
            timestamp_max = site_max if pd.isna(timestamp_max) or site_max > timestamp_max else timestamp_max

            table = pa.Table.from_pandas(df[REQUIRED_MODEL_READY_COLUMNS], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    summary = pd.DataFrame(
        [
            {
                "n_rows": int(totals["n_rows"]),
                "n_sites": int(len(selected_sites)),
                "train_rows": int(totals["train_rows"]),
                "val_rows": int(totals["val_rows"]),
                "test_rows": int(totals["test_rows"]),
                "timestamp_min": timestamp_min,
                "timestamp_max": timestamp_max,
                "split_method": split_method,
                "duplicate_site_timestamp_count": int(duplicate_count),
                "missing_target_after_cleaning": 0,
                "missing_predictors_after_cleaning": 0,
                "n_test_event_rows": int(totals["n_test_event_rows"]),
                "n_test_low_event_rows": int(totals["n_test_low_event_rows"]),
                "n_test_high_event_rows": int(totals["n_test_high_event_rows"]),
            }
        ]
    )
    write_csv(summary, "processed100_dataset_summary.csv")
    logging.info("Built processed100 base with %s rows", totals["n_rows"])
    return summary


def event_state_from_columns(in_event: pd.Series, labels: pd.Series) -> np.ndarray:
    in_event_bool = pd.to_numeric(in_event, errors="coerce").fillna(0).astype("int8").to_numpy() > 0
    norm = labels.map(normalise_event_label).to_numpy(dtype=object)
    state = np.zeros(len(labels), dtype=np.int8)
    state[in_event_bool & (norm == "pH_exceedance_low")] = 1
    state[in_event_bool & (norm == "pH_exceedance_high")] = 2
    state[in_event_bool & ~np.isin(norm, ["pH_exceedance_low", "pH_exceedance_high"])] = 3
    return state


def add_event_state_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_path = OUTPUT_DIR / "processed100_model_ready.parquet"
    output_path = ensure_output_path(OUTPUT_DIR / "processed100_with_event_state.parquet")
    if output_path.exists():
        output_path.unlink()
    write_json(
        {
            "event_state_codes": EVENT_STATE_DEFINITIONS,
            "event_state_column": EVENT_STATE_COL,
            "event_state_name_column": EVENT_STATE_NAME_COL,
            "sample_weight_column": SAMPLE_WEIGHT_COL,
            "note": "sample_weight_first_version is prepared for later experiments only and is not used here.",
        },
        "event_state_code_definitions.json",
    )

    split_counts: dict[tuple[str, int, str], int] = defaultdict(int)
    site_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    low_high_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    overlap_rows: list[pd.DataFrame] = []
    writer: pq.ParquetWriter | None = None
    pf = pq.ParquetFile(input_path)
    name_map = {0: "normal", 1: "low_pH_event", 2: "high_pH_event", 3: "mixed_or_unknown"}
    weight_map = {0: 1.0, 1: 3.0, 2: 3.0, 3: 1.0}

    try:
        for batch in pf.iter_batches(batch_size=1_000_000):
            df = batch.to_pandas()
            state = event_state_from_columns(df[EVENT_COL], df[EVENT_LABEL_COL])
            df[EVENT_STATE_COL] = state
            df[EVENT_STATE_NAME_COL] = pd.Series(state).map(name_map).astype("string").to_numpy()
            df[SAMPLE_WEIGHT_COL] = pd.Series(state).map(weight_map).astype("float32").to_numpy()

            labels_norm = df[EVENT_LABEL_COL].map(normalise_event_label)
            low_flag = (state == 1) | ((state == 3) & labels_norm.astype(str).str.contains("low", case=False, na=False).to_numpy())
            high_flag = (state == 2) | ((state == 3) & labels_norm.astype(str).str.contains("high", case=False, na=False).to_numpy())
            overlap_mask = (state == 3) | (low_flag & high_flag)
            if overlap_mask.any():
                audit = df.loc[overlap_mask, [SITE_COL, TIME_COL, EVENT_STATE_COL]].copy()
                audit["low_event_flag"] = low_flag[overlap_mask]
                audit["high_event_flag"] = high_flag[overlap_mask]
                audit["notes"] = "mixed_or_unknown state from existing event columns"
                overlap_rows.append(audit[["site_id", "timestamp", "low_event_flag", "high_event_flag", "event_state", "notes"]])

            for (split, event_state, state_name), count in (
                df.groupby([SPLIT_COL, EVENT_STATE_COL, EVENT_STATE_NAME_COL], observed=True).size().items()
            ):
                split_counts[(str(split), int(event_state), str(state_name))] += int(count)
            for site_id, group in df.groupby(SITE_COL, observed=True):
                site = str(site_id)
                counts = pd.Series(group[EVENT_STATE_COL]).value_counts().to_dict()
                site_counts[site]["n_rows"] += len(group)
                site_counts[site]["n_normal_rows"] += int(counts.get(0, 0))
                site_counts[site]["n_low_P_H_event_rows".replace("_P_H_", "_pH_")] += int(counts.get(1, 0))
                site_counts[site]["n_high_P_H_event_rows".replace("_P_H_", "_pH_")] += int(counts.get(2, 0))
                site_counts[site]["n_mixed_or_unknown_rows"] += int(counts.get(3, 0))
            for split, group in df.groupby(SPLIT_COL, observed=True):
                split_key = str(split)
                low_high_counts[split_key]["n_rows"] += len(group)
                low_high_counts[split_key]["n_low_rows"] += int((group[EVENT_STATE_COL] == 1).sum())
                low_high_counts[split_key]["n_high_rows"] += int((group[EVENT_STATE_COL] == 2).sum())

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    split_total = defaultdict(int)
    for (split, _, _), count in split_counts.items():
        split_total[split] += count
    split_summary = pd.DataFrame(
        [
            {
                "split": split,
                "event_state": event_state,
                "event_state_name": state_name,
                "n_rows": count,
                "row_rate": count / split_total[split] if split_total[split] else np.nan,
            }
            for (split, event_state, state_name), count in sorted(split_counts.items())
        ]
    )
    site_summary = pd.DataFrame(
        [
            {
                "site_id": site,
                **counts,
                "event_row_rate": (
                    (counts["n_low_pH_event_rows"] + counts["n_high_pH_event_rows"] + counts["n_mixed_or_unknown_rows"])
                    / counts["n_rows"]
                    if counts["n_rows"]
                    else np.nan
                ),
            }
            for site, counts in sorted(site_counts.items())
        ]
    )
    low_high_summary = pd.DataFrame(
        [
            {
                "split": split,
                "n_low_rows": counts["n_low_rows"],
                "n_high_rows": counts["n_high_rows"],
                "low_to_high_ratio": counts["n_low_rows"] / counts["n_high_rows"] if counts["n_high_rows"] else np.nan,
                "low_event_rate": counts["n_low_rows"] / counts["n_rows"] if counts["n_rows"] else np.nan,
                "high_event_rate": counts["n_high_rows"] / counts["n_rows"] if counts["n_rows"] else np.nan,
            }
            for split, counts in sorted(low_high_counts.items())
        ]
    )
    if overlap_rows:
        overlap = pd.concat(overlap_rows, ignore_index=True)
    else:
        overlap = pd.DataFrame(columns=["site_id", "timestamp", "low_event_flag", "high_event_flag", "event_state", "notes"])

    write_csv(split_summary, "event_state_summary_by_split.csv")
    write_csv(site_summary, "event_state_summary_by_site.csv")
    write_csv(low_high_summary, "event_state_low_high_summary.csv")
    write_csv(overlap, "event_state_overlap_audit.csv")
    return split_summary, site_summary, low_high_summary


def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8")) if REGISTRY_PATH.exists() else {}


def extract_dominant_land_cover(catchment: dict[str, Any]) -> tuple[str, float]:
    land_cover = catchment.get("watershed_land_cover")
    if not isinstance(land_cover, dict) or not land_cover:
        return "unknown", np.nan
    values = [(str(k), safe_float(v)) for k, v in land_cover.items()]
    values = [(k, v) for k, v in values if pd.notna(v)]
    if not values:
        return "unknown", np.nan
    return max(values, key=lambda item: item[1])


def build_official_metadata(site_ids: list[str], registry: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for site_id in site_ids:
        entry = registry.get(site_id, {})
        source = parse_jsonish(entry.get("source_description")) if isinstance(entry, dict) else {}
        catchment = entry.get("catchment_analysis", {}) if isinstance(entry, dict) else {}
        catchment = catchment if isinstance(catchment, dict) else {}
        discharge_stats = (
            entry.get("discharge_description", {}).get("statistics", {})
            if isinstance(entry.get("discharge_description"), dict)
            else {}
        )
        land_cover, land_cover_pct = extract_dominant_land_cover(catchment)
        drainage_area = entry.get("drainage_area_km2", source.get("drainage_area_sqkm")) if isinstance(entry, dict) else np.nan
        rows.append(
            {
                "site_id": site_id,
                "metadata_match": site_id in registry,
                "meta_official_source": entry.get("source") if isinstance(entry, dict) else None,
                "meta_official_units": entry.get("units") if isinstance(entry, dict) else None,
                "meta_official_basin": entry.get("basin_cd") if isinstance(entry, dict) else None,
                "meta_official_country": entry.get("country_cd") if isinstance(entry, dict) else None,
                "meta_official_state": source.get("state_cd"),
                "meta_official_county": source.get("county_cd"),
                "meta_official_site_type": source.get("site_type_cd") or (entry.get("site_type") if isinstance(entry, dict) else None),
                "meta_official_status": entry.get("status") if isinstance(entry, dict) else None,
                "meta_official_latitude": safe_float(entry.get("latitude")) if isinstance(entry, dict) else np.nan,
                "meta_official_longitude": safe_float(entry.get("longitude")) if isinstance(entry, dict) else np.nan,
                "meta_official_elevation": safe_float(entry.get("elevation")) if isinstance(entry, dict) else np.nan,
                "meta_official_drainage_area": safe_float(drainage_area),
                "meta_official_average_discharge": safe_float(discharge_stats.get("mean_discharge")),
                "meta_official_catchment_average_precip_mm_yr": safe_float(catchment.get("average_precip_mm_yr")),
                "meta_official_catchment_average_evapo_trans_mm_yr": safe_float(catchment.get("average_evapo_trans_mm_yr")),
                "meta_official_catchment_population_est_2020": safe_float(catchment.get("population_est_2020")),
                "meta_official_catchment_watershed_dams_no": safe_float(catchment.get("watershed_dams_no")),
                "meta_official_catchment_watershed_dams_cap_m3": safe_float(catchment.get("watershed_dams_cap_m3")),
                "meta_official_catchment_irrigation_area_km2": safe_float(catchment.get("irrigation_area_km2")),
                "meta_official_catchment_irrigation_area_pct": safe_float(catchment.get("irrigation_area_pct")),
                "meta_official_land_cover": land_cover,
                "meta_official_land_cover_pct": land_cover_pct,
            }
        )
    official = pd.DataFrame(rows)
    write_csv(official, "processed100_metadata_official_cleaned.csv")
    return official


def build_train_derived_metadata(site_ids: list[str]) -> pd.DataFrame:
    acc: dict[str, dict[str, Any]] = {
        site: {"timestamps": [], **{var: [] for var in EXOGENOUS_COLUMNS}} for site in site_ids
    }
    pf = pq.ParquetFile(OUTPUT_DIR / "processed100_model_ready.parquet")
    columns = [SITE_COL, TIME_COL, SPLIT_COL, *EXOGENOUS_COLUMNS]
    for batch in pf.iter_batches(columns=columns, batch_size=1_000_000):
        df = batch.to_pandas()
        train = df[df[SPLIT_COL].eq("train")].copy()
        if train.empty:
            continue
        train[SITE_COL] = train[SITE_COL].astype(str)
        for site_id, group in train.groupby(SITE_COL, observed=True):
            site = str(site_id)
            if site not in acc:
                continue
            acc[site]["timestamps"].append(pd.to_datetime(group[TIME_COL], errors="coerce").to_numpy())
            for var in EXOGENOUS_COLUMNS:
                acc[site][var].append(pd.to_numeric(group[var], errors="coerce").to_numpy(dtype=np.float32))

    records: list[dict[str, Any]] = []
    for site in site_ids:
        timestamps = pd.Series(np.concatenate(acc[site]["timestamps"])) if acc[site]["timestamps"] else pd.Series(dtype="datetime64[ns]")
        start = pd.to_datetime(timestamps, errors="coerce").min() if len(timestamps) else pd.NaT
        end = pd.to_datetime(timestamps, errors="coerce").max() if len(timestamps) else pd.NaT
        record: dict[str, Any] = {
            "site_id": site,
            "meta_train_n_rows": int(len(timestamps)),
            "meta_train_coverage_days": float((end - start).total_seconds() / 86400.0) if pd.notna(start) and pd.notna(end) else np.nan,
            "meta_train_dominant_interval_minutes": dominant_interval_minutes(timestamps) if len(timestamps) else np.nan,
        }
        for var in EXOGENOUS_COLUMNS:
            values = np.concatenate(acc[site][var]) if acc[site][var] else np.array([], dtype=np.float32)
            series = pd.Series(values, dtype="float32")
            q25 = series.quantile(0.25) if len(series) else np.nan
            q75 = series.quantile(0.75) if len(series) else np.nan
            record.update(
                {
                    f"meta_train_{var}_median": series.median() if len(series) else np.nan,
                    f"meta_train_{var}_mean": series.mean() if len(series) else np.nan,
                    f"meta_train_{var}_std": series.std() if len(series) else np.nan,
                    f"meta_train_{var}_iqr": q75 - q25 if pd.notna(q25) and pd.notna(q75) else np.nan,
                    f"meta_train_{var}_p05": series.quantile(0.05) if len(series) else np.nan,
                    f"meta_train_{var}_p95": series.quantile(0.95) if len(series) else np.nan,
                    f"meta_train_{var}_missing_rate": float(series.isna().mean()) if len(series) else np.nan,
                }
            )
        records.append(record)
    derived = pd.DataFrame(records)
    write_csv(derived, "processed100_metadata_train_derived.csv")
    return derived


def leakage_reason(feature: str) -> str | None:
    lower = feature.lower()
    allowed_train_exact = {
        "meta_train_n_rows",
        "meta_train_coverage_days",
        "meta_train_dominant_interval_minutes",
    }
    allowed_train_prefixes = [f"meta_train_{var}_" for var in EXOGENOUS_COLUMNS]
    if lower in allowed_train_exact or any(lower.startswith(prefix) for prefix in allowed_train_prefixes):
        return None
    if "00400" in lower:
        return "contains target parameter code 00400"
    if "low_ph" in lower or "high_ph" in lower:
        return "contains low/high pH marker"
    if re.search(r"(^|_)ph($|_)", lower) or lower.startswith("ph_") or lower.endswith("_ph") or "has_ph" in lower:
        return "contains pH/target marker"
    tokens = set(re.split(r"[^a-z0-9]+", lower))
    for token in [
        "target",
        "label",
        "event",
        "incident",
        "exceedance",
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
    ]:
        if token in tokens or token in lower:
            return f"contains forbidden marker `{token}`"
    return None


def build_model_ready_metadata(official: pd.DataFrame, train_derived: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    meta = official.merge(train_derived, on="site_id", how="left")
    missing_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    feature_candidates: list[str] = []
    official_features: list[str] = []
    train_features: list[str] = []
    missing_features: list[str] = []
    categorical_encoding: dict[str, dict[str, int]] = {}
    numeric_imputation: dict[str, float] = {}
    categorical_imputation: dict[str, str] = {}

    for col in OFFICIAL_CATEGORICAL_COLUMNS:
        if col not in meta.columns:
            continue
        missing = meta[col].isna() | meta[col].astype("string").str.strip().eq("")
        miss_col = f"meta_missing_{col.replace('meta_', '')}"
        meta[miss_col] = missing.astype("int8")
        meta[col] = meta[col].astype("string").fillna("unknown").replace("", "unknown")
        values = sorted(meta[col].dropna().astype(str).unique().tolist())
        if "unknown" not in values:
            values.append("unknown")
            values = sorted(values)
        mapping = {value: idx for idx, value in enumerate(values)}
        code_col = f"{col}_code"
        meta[code_col] = meta[col].astype(str).map(mapping).fillna(mapping["unknown"]).astype("int16")
        categorical_encoding[col] = mapping
        categorical_imputation[col] = "unknown"
        missing_rows.append(
            {
                "field_name": col,
                "source": "official_registry",
                "field_type": "categorical",
                "missing_count_before_imputation": int(missing.sum()),
                "missing_rate_before_imputation": float(missing.mean()) if len(meta) else np.nan,
                "imputation_method": "fill_unknown_then_stable_integer_code",
            }
        )
        feature_candidates.extend([code_col, miss_col])
        official_features.extend([code_col, miss_col])
        missing_features.append(miss_col)

    numeric_cols = [
        col
        for col in meta.columns
        if (col.startswith("meta_official_") and col not in OFFICIAL_CATEGORICAL_COLUMNS)
        or col.startswith("meta_train_")
    ]
    for col in numeric_cols:
        if col == "metadata_match":
            continue
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
        missing = meta[col].isna()
        fill_value = meta[col].median()
        if pd.isna(fill_value):
            fill_value = 0.0
        meta[col] = meta[col].fillna(fill_value).astype("float32")
        numeric_imputation[col] = float(fill_value)
        missing_rows.append(
            {
                "field_name": col,
                "source": "train_only_exogenous" if col.startswith("meta_train_") else "official_registry",
                "field_type": "numeric",
                "missing_count_before_imputation": int(missing.sum()),
                "missing_rate_before_imputation": float(missing.mean()) if len(meta) else np.nan,
                "imputation_method": f"median_selected100={fill_value}",
            }
        )
        feature_candidates.append(col)
        if col.startswith("meta_train_"):
            train_features.append(col)
        else:
            official_features.append(col)
        if missing.any():
            miss_col = f"meta_missing_{col.replace('meta_', '')}"
            meta[miss_col] = missing.astype("int8")
            feature_candidates.append(miss_col)
            missing_features.append(miss_col)
            if col.startswith("meta_train_"):
                train_features.append(miss_col)
            else:
                official_features.append(miss_col)

    kept_features: list[str] = []
    excluded: list[str] = []
    for feature in dict.fromkeys(feature_candidates):
        reason = leakage_reason(feature)
        if reason:
            leakage_rows.append({"feature_name": feature, "status": "flagged", "reason": reason, "kept_or_excluded": "excluded"})
            excluded.append(feature)
        else:
            leakage_rows.append(
                {
                    "feature_name": feature,
                    "status": "passed",
                    "reason": "no forbidden target/event/model/validation/test pattern detected",
                    "kept_or_excluded": "kept",
                }
            )
            kept_features.append(feature)

    missing_df = pd.DataFrame(missing_rows)
    leakage_df = pd.DataFrame(leakage_rows)
    feature_set = {
        "metadata_features_used": kept_features,
        "official_metadata_features": [feature for feature in kept_features if feature in official_features],
        "train_derived_metadata_features": [feature for feature in kept_features if feature in train_features],
        "missingness_indicator_features": [feature for feature in kept_features if feature in missing_features],
        "categorical_encoding": categorical_encoding,
        "numeric_imputation": numeric_imputation,
        "categorical_imputation": categorical_imputation,
        "excluded_leakage_features": excluded,
        "notes": [
            "Official metadata from sites_register.json.",
            "Train-derived metadata uses only split == train rows and exogenous predictors 00010, 00095, 63680, 00060, 00065.",
            "No 00400 target summaries, event summaries, residuals, predictions, validation rows, or test rows were used.",
        ],
    }
    model_ready = meta[["site_id", *kept_features, "metadata_match"]].copy()
    write_csv(missing_df, "processed100_metadata_missingness_by_field.csv")
    write_csv(leakage_df, "processed100_metadata_leakage_screening_report.csv")
    write_csv(model_ready, "processed100_metadata_model_ready.csv")
    write_json(feature_set, "processed100_metadata_feature_set.json")
    return model_ready, feature_set, missing_df


def merge_metadata(model_ready: pd.DataFrame, feature_set: dict[str, Any]) -> pd.DataFrame:
    base_path = OUTPUT_DIR / "processed100_with_event_state.parquet"
    output_path = ensure_output_path(OUTPUT_DIR / "processed100_with_event_state_metadata.parquet")
    if output_path.exists():
        output_path.unlink()
    features = list(feature_set["metadata_features_used"])
    meta_index = model_ready.set_index("site_id")[features]
    pf = pq.ParquetFile(base_path)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    site_ids: set[str] = set()
    try:
        for batch in pf.iter_batches(batch_size=1_000_000):
            df = batch.to_pandas()
            df[SITE_COL] = df[SITE_COL].astype(str)
            joined = df.join(meta_index, on=SITE_COL)
            if joined[features].isna().any().any():
                missing_sites = joined.loc[joined[features].isna().any(axis=1), SITE_COL].drop_duplicates().head(10).tolist()
                raise ValueError(f"Metadata merge introduced missing values for sites: {missing_sites}")
            table = pa.Table.from_pandas(joined, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += len(joined)
            site_ids.update(joined[SITE_COL].dropna().astype(str).unique().tolist())
    finally:
        if writer is not None:
            writer.close()

    base_pf = pq.ParquetFile(base_path)
    merged_pf = pq.ParquetFile(output_path)
    audit = pd.DataFrame(
        [
            {
                "base_rows": int(base_pf.metadata.num_rows),
                "merged_rows": int(merged_pf.metadata.num_rows),
                "row_count_preserved": int(base_pf.metadata.num_rows) == int(merged_pf.metadata.num_rows),
                "base_site_count": int(len(site_ids)),
                "metadata_site_count": int(model_ready["site_id"].nunique()),
                "metadata_duplicate_site_count": int(model_ready["site_id"].duplicated().sum()),
                "unmatched_site_count": 0,
                "merged_metadata_feature_count": int(len(features)),
                "merge_key": "site_id",
            }
        ]
    )
    write_csv(audit, "processed100_metadata_merge_audit.csv")
    return audit


def create_figures(
    split_summary: pd.DataFrame,
    low_high_summary: pd.DataFrame,
    missing_df: pd.DataFrame,
    official: pd.DataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning("Skipping figures because matplotlib is unavailable: %s", exc)
        return

    test_dist = split_summary[split_summary["split"].eq("test")].copy()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(test_dist["event_state_name"], test_dist["n_rows"], color="#4C78A8")
    ax.set_ylabel("Rows")
    ax.set_title("Processed100 Test Event-State Distribution")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(ensure_output_path(FIGURES_DIR / "processed100_event_state_distribution.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x = np.arange(len(low_high_summary))
    width = 0.35
    ax.bar(x - width / 2, low_high_summary["n_low_rows"], width, label="Low pH", color="#4C78A8")
    ax.bar(x + width / 2, low_high_summary["n_high_rows"], width, label="High pH", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(low_high_summary["split"])
    ax.set_ylabel("Rows")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(ensure_output_path(FIGURES_DIR / "processed100_low_high_event_distribution.png"), dpi=300)
    plt.close(fig)

    miss = missing_df.sort_values("missing_rate_before_imputation", ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.arange(len(miss)), miss["missing_rate_before_imputation"], color="#72B7B2")
    ax.set_yticks(np.arange(len(miss)))
    ax.set_yticklabels(miss["field_name"])
    ax.invert_yaxis()
    ax.set_xlabel("Missingness before imputation")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(ensure_output_path(FIGURES_DIR / "processed100_metadata_missingness.png"), dpi=300)
    plt.close(fig)

    match_rate = float(official["metadata_match"].mean()) if len(official) else np.nan
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.bar(["Matched", "Unmatched"], [match_rate, 1 - match_rate], color=["#4C78A8", "#E45756"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Site fraction")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(ensure_output_path(FIGURES_DIR / "processed100_metadata_match_rate.png"), dpi=300)
    plt.close(fig)


def write_report(
    selected: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    site_summary: pd.DataFrame,
    low_high_summary: pd.DataFrame,
    official: pd.DataFrame,
    train_derived: pd.DataFrame,
    feature_set: dict[str, Any],
    merge_audit: pd.DataFrame,
) -> str:
    ds = dataset_summary.iloc[0]
    test_low_high = low_high_summary[low_high_summary["split"].eq("test")]
    test_low = int(test_low_high["n_low_rows"].iloc[0]) if not test_low_high.empty else 0
    test_high = int(test_low_high["n_high_rows"].iloc[0]) if not test_low_high.empty else 0
    text = f"""# Phase 4.0 + 4.1 Processed100 Event-State Coding and Metadata Merge

## 1. Purpose

This task prepares a nested strict processed100 dataset for the next 100-site all-data HGB metadata comparison. No model was trained.

## 2. Data Source

- Time-series source: fast-path subset of `analysis_outputs/ph_hgb_processed200_metadata_v1/processed200_model_ready.parquet`
- Site selection source: `analysis_outputs/ph_hgb_processed200_metadata_v1/processed200_selected_sites.csv`
- Static metadata source: `data/reference/sites_register.json`
- Regulatory event file: `data/reference/regulatory_events.csv`, validation/coding context only

## 3. Site Selection

- Selected sites: {len(selected):,}
- Selection rule: first 100 sites by strict processed200 `selection_rank`
- Metadata match rate in selected list: {float(selected['metadata_match'].mean()):.1%}

## 4. Base Dataset

- Rows: {int(ds['n_rows']):,}
- Sites: {int(ds['n_sites']):,}
- Train rows: {int(ds['train_rows']):,}
- Validation rows: {int(ds['val_rows']):,}
- Test rows: {int(ds['test_rows']):,}
- Time range: {ds['timestamp_min']} to {ds['timestamp_max']}
- Split method: {ds['split_method']}
- Duplicate `site_id + timestamp` rows: {int(ds['duplicate_site_timestamp_count']):,}
- Missing target after cleaning: {int(ds['missing_target_after_cleaning']):,}
- Missing required predictors after cleaning: {int(ds['missing_predictors_after_cleaning']):,}

## 5. Step 4.0 Event-State Coding

Event states were derived from existing `in_ph_event` and `ph_event_label` columns in the processed200-derived base dataset.

- Test low-pH event rows: {test_low:,}
- Test high-pH event rows: {test_high:,}
- Event-state summaries saved by split and by site.
- `sample_weight_first_version` was prepared for future experiments only and was not used here.

## 6. Step 4.1 Metadata

- Official metadata rows: {len(official):,}
- Train-derived metadata rows: {len(train_derived):,}
- Model-ready metadata features kept: {len(feature_set['metadata_features_used']):,}
- Official features kept: {len(feature_set['official_metadata_features']):,}
- Train-derived features kept: {len(feature_set['train_derived_metadata_features']):,}
- Missingness indicators kept: {len(feature_set['missingness_indicator_features']):,}

Train-derived metadata used only `split == "train"` rows and only exogenous variables `{', '.join(EXOGENOUS_COLUMNS)}`. No `00400`, pH summaries, event-derived features, residuals, predictions, validation rows, or test rows were used.

## 7. Merge Audit

- Base rows: {int(merge_audit.loc[0, 'base_rows']):,}
- Merged rows: {int(merge_audit.loc[0, 'merged_rows']):,}
- Row count preserved: {bool(merge_audit.loc[0, 'row_count_preserved'])}
- Metadata duplicate site rows: {int(merge_audit.loc[0, 'metadata_duplicate_site_count']):,}
- Unmatched sites after merge: {int(merge_audit.loc[0, 'unmatched_site_count']):,}

## 8. Leakage Controls

Feature-name leakage screening excluded target/event/residual/prediction/model-metric/validation/test/future markers. Allowed train-derived metadata is limited to summaries of `{', '.join(EXOGENOUS_COLUMNS)}` and train coverage fields.

## 9. Next Step

Use `processed100_with_event_state.parquet` for no-metadata modelling and `processed100_with_event_state_metadata.parquet` for metadata modelling. Event-state and sample-weight columns should remain training-control or validation fields, not ordinary deployable predictors.
"""
    write_text(text, "step4_0_4_1_processed100_event_metadata_report.md")
    return text


def main() -> None:
    setup_logging()
    logging.info("Starting processed100 Step 4.0 + 4.1 preparation")
    selected = select_sites_fast_path()
    dataset_summary = build_processed100_base(selected)
    split_summary, site_summary, low_high_summary = add_event_state_outputs()
    site_ids = selected["site_id"].astype(str).tolist()
    registry = load_registry()
    official = build_official_metadata(site_ids, registry)
    train_derived = build_train_derived_metadata(site_ids)
    model_ready_metadata, feature_set, missing_df = build_model_ready_metadata(official, train_derived)
    merge_audit = merge_metadata(model_ready_metadata, feature_set)
    create_figures(split_summary, low_high_summary, missing_df, official)
    write_report(selected, dataset_summary, split_summary, site_summary, low_high_summary, official, train_derived, feature_set, merge_audit)
    ds = dataset_summary.iloc[0]
    final_summary = f"""PHASE 4.0 + 4.1 PROCESSED100 EVENT-STATE METADATA COMPLETE

Output folder:
{rel(OUTPUT_DIR)}/

Selected sites:
{len(selected)}

Final rows:
{int(ds['n_rows'])}

Train rows:
{int(ds['train_rows'])}

Validation rows:
{int(ds['val_rows'])}

Test rows:
{int(ds['test_rows'])}

Metadata match rate:
{float(selected['metadata_match'].mean()):.3f}

Metadata features used:
{len(feature_set['metadata_features_used'])}

Event-state source:
existing in_ph_event and ph_event_label columns from processed200 fast-path dataset

Model training:
none
"""
    write_text(final_summary, "processed100_final_console_summary.txt")
    print(final_summary.strip())


if __name__ == "__main__":
    main()
