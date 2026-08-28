#!/usr/bin/env python
"""Build the leak-safe reduced-predictor ~1000-site processed dataset.

The pipeline reads the stage-00 candidate manifest, processes each selected
clean file independently, and writes two versioned per-site Parquet layers:

* native_clean_by_site: native-unit cleaned observations plus audit flags;
* model_ready_by_site: observed-pH complete rows, chronological splits,
  leak-safe scaling, time features, and evaluation-only pH event fields.

Existing ``data/processed`` files and historical analysis outputs are never
modified.  Per-site outputs and checkpoints are written atomically so an
interrupted run can safely resume.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from importlib import metadata as importlib_metadata

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "common1000_processing_v1.json"

SITE_COL = "site_id"
TIME_COL = "timestamp"
TARGET_CODE = "00400"
PREDICTOR_CODES = ["00010", "00095", "00300", "63680"]
REQUIRED_CODES = [TARGET_CODE, *PREDICTOR_CODES]

ALIASES: dict[str, list[str]] = {
    SITE_COL: ["site_id", "Site_ID", "SITE_ID", "site", "site_no", "station_id"],
    TIME_COL: ["timestamp", "TIMESTAMP", "datetime", "date_time", "time"],
    "00400": ["00400", "pH", "ph"],
    "00010": ["00010", "Temperature_C", "temperature", "temp"],
    "00095": ["00095", "Specific_Conductance_uS_cm", "specific_conductance"],
    "00300": ["00300", "Dissolved_Oxygen_mg_L", "dissolved_oxygen"],
    "63680": ["63680", "Turbidity_FNU", "turbidity", "Turbidity [NTU]*.1"],
}

QC_STATUS_DEFINITION = {
    "-1": "source column has no QC field",
    "0": "QC pass (0)",
    "1": "QC fail (1); value excluded",
    "2": "QC missing or unrecognised while value is present; value excluded",
    "3": "source value missing",
}
RANGE_FLAG_DEFINITION = {
    "0": "no range problem",
    "1": "outside hard physical/declared range; value excluded",
    "2": "above soft high threshold; value retained and flagged",
    "3": "non-finite or <= -999 sentinel; value excluded",
}

MODEL_FEATURES = [
    "00010",
    "00095",
    "00300",
    "63680",
    "sin_hour",
    "cos_hour",
    "sin_day",
    "cos_day",
    "month",
    "hour",
    "day_of_year",
]
EVENT_COLUMNS = ["in_ph_event", "ph_event_label", "ph_event_direction", "event_id"]
FORBIDDEN_FEATURE_TOKENS = {
    "00400",
    "ph",
    "target",
    "label",
    "event",
    "incident",
    "residual",
    "prediction",
    "test",
    "validation",
    "future",
}


_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_PROJECT_ROOT: Path | None = None
_WORKER_OUTPUT_ROOT: Path | None = None
_WORKER_EVENTS: dict[str, list[dict[str, Any]]] = {}
_WORKER_CONFIG_HASH = ""
_WORKER_SCRIPT_HASH = ""
_WORKER_RESUME = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--resume", action="store_true", help="Reuse validated per-site checkpoints.")
    parser.add_argument("--smoke-only", action="store_true", help="Run the four-network smoke test only.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke test after it has already passed.")
    parser.add_argument("--workers", type=int, default=None, help="Override configured process count.")
    parser.add_argument("--site-limit", type=int, default=None, help="Development-only deterministic site limit.")
    return parser.parse_args()


def normalise_site_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    text = re.sub(r"\s+", "", text)
    return text.zfill(8) if text.isdigit() and len(text) < 8 else text


def safe_name(site_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", site_id) or "unknown_site"


def resolve_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serialisable: {type(value)!r}")


def atomic_write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_parquet(df: pd.DataFrame, path: Path, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, temporary, compression=compression, use_dictionary=True)
    os.replace(temporary, path)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("target_code") != TARGET_CODE:
        raise ValueError(f"target_code must be {TARGET_CODE}")
    if list(config.get("predictor_codes", [])) != PREDICTOR_CODES:
        raise ValueError(f"predictor_codes must be {PREDICTOR_CODES}")
    fractions = [float(value) for value in config.get("split_fractions", [])]
    if len(fractions) != 3 or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("split_fractions must contain three values that sum to 1")
    if any(value <= 0 for value in fractions):
        raise ValueError("split_fractions must all be positive")
    configured_features = set(MODEL_FEATURES)
    for feature in configured_features:
        lower = feature.lower()
        tokens = set(re.split(r"[^a-z0-9]+", lower))
        if feature == TARGET_CODE or tokens & FORBIDDEN_FEATURE_TOKENS:
            raise ValueError(f"Forbidden leakage-prone model feature: {feature}")
    if config.get("events", {}).get("fields_are_predictors") is not False:
        raise ValueError("Event fields must be configured as non-predictors")


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config not found: {resolved}")
    config = json.loads(resolved.read_text(encoding="utf-8-sig"))
    validate_config(config)
    return config, resolved


def infer_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    actual = list(columns)
    exact = set(actual)
    for candidate in candidates:
        if candidate in exact:
            return candidate
    lower_map = {str(column).lower(): str(column) for column in actual}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def source_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        return list(pq.ParquetFile(path).schema_arrow.names)
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    return list(pd.read_csv(path, sep=separator, nrows=0).columns)


def build_source_mapping(path: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    columns = source_columns(path)
    mapping: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        found = infer_existing_column(columns, aliases)
        if found is not None:
            mapping[canonical] = found
    missing = [code for code in [TIME_COL, *REQUIRED_CODES] if code not in mapping]
    if missing:
        raise ValueError(f"Missing required source columns/aliases: {missing}")
    qc_mapping: dict[str, str] = {}
    for code in REQUIRED_CODES:
        source = mapping[code]
        found = infer_existing_column(columns, [f"{source}_qc", f"{code}_qc"])
        if found is not None:
            qc_mapping[code] = found
    read_columns = list(dict.fromkeys([*mapping.values(), *qc_mapping.values()]))
    return mapping, qc_mapping, read_columns


def read_source_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=separator, usecols=columns, low_memory=False)


def parse_timestamps(values: pd.Series) -> tuple[pd.Series, str]:
    parsed = pd.to_datetime(values, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert("UTC").dt.tz_localize(None), "utc_normalized_from_aware"
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        # Mixed timezone strings are uncommon in the current clean corpus.  If
        # present, make the conversion explicit and auditable rather than
        # leaving object timestamps in the processed layer.
        parsed_utc = pd.to_datetime(values, errors="coerce", utc=True)
        return parsed_utc.dt.tz_convert("UTC").dt.tz_localize(None), "utc_normalized_from_mixed"
    return parsed, "source_naive_unknown"


def do_unit_family(network: str) -> str:
    return "ea_percent_saturation_like_unverified" if str(network).startswith("EA") else "reported_mg_l"


def hard_range_for(code: str, network: str, config: dict[str, Any]) -> tuple[float | None, float | None]:
    ranges = config["hard_ranges"]
    if code == "00300":
        key = "00300_ea_percent_saturation_like" if str(network).startswith("EA") else "00300_mg_l"
    else:
        key = code
    low, high = ranges[key]
    return (None if low is None else float(low), None if high is None else float(high))


def clean_numeric_column(
    raw: pd.Series,
    qc: pd.Series | None,
    code: str,
    network: str,
    config: dict[str, Any],
) -> tuple[pd.Series, np.ndarray, np.ndarray, dict[str, int]]:
    values = pd.to_numeric(raw, errors="coerce").astype("float64")
    raw_present = values.notna().to_numpy()
    nonfinite = raw_present & ~np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))
    sentinel = raw_present & values.le(-999.0).to_numpy()

    if qc is None:
        qc_status = np.where(raw_present, -1, 3).astype("int8")
        qc_invalid = np.zeros(len(values), dtype=bool)
    else:
        qc_numeric = pd.to_numeric(qc, errors="coerce")
        qc_status = np.full(len(values), 3, dtype="int8")
        present_index = raw_present
        qc_status[present_index & qc_numeric.eq(0.0).to_numpy()] = 0
        qc_status[present_index & qc_numeric.eq(1.0).to_numpy()] = 1
        recognised = qc_numeric.eq(0.0).to_numpy() | qc_numeric.eq(1.0).to_numpy()
        qc_status[present_index & ~recognised] = 2
        qc_invalid = present_index & (qc_status != 0)

    low, high = hard_range_for(code, network, config)
    hard_invalid = np.zeros(len(values), dtype=bool)
    if low is not None:
        hard_invalid |= raw_present & values.lt(low).to_numpy()
    if high is not None:
        hard_invalid |= raw_present & values.gt(high).to_numpy()

    soft_threshold = config.get("soft_high_flags", {}).get(code)
    soft_high = (
        raw_present & values.gt(float(soft_threshold)).to_numpy()
        if soft_threshold is not None
        else np.zeros(len(values), dtype=bool)
    )

    range_flag = np.zeros(len(values), dtype="int8")
    range_flag[soft_high] = 2
    range_flag[hard_invalid] = 1
    range_flag[nonfinite | sentinel] = 3

    invalid = qc_invalid | hard_invalid | nonfinite | sentinel
    cleaned = values.mask(invalid).astype("float32")
    stats = {
        f"{code}_source_nonmissing": int(raw_present.sum()),
        f"{code}_qc_fail": int((qc_status == 1).sum()),
        f"{code}_qc_unknown": int((qc_status == 2).sum()),
        f"{code}_hard_range_invalid": int(hard_invalid.sum()),
        f"{code}_sentinel_or_nonfinite": int((nonfinite | sentinel).sum()),
        f"{code}_soft_high_flag": int(soft_high.sum()),
        f"{code}_clean_observed": int(cleaned.notna().sum()),
    }
    return cleaned, qc_status, range_flag, stats


def interpolate_short_internal_gaps(
    timestamps: pd.Series,
    values: pd.Series,
    max_gap_hours: float,
) -> tuple[pd.Series, np.ndarray, float]:
    """Linearly fill only internally bracketed gaps spanning <= max_gap_hours."""

    if values.notna().sum() < 2 or values.isna().sum() == 0:
        return values.astype("float32"), np.zeros(len(values), dtype=bool), 0.0
    work = pd.Series(values.to_numpy(dtype="float64", na_value=np.nan), index=pd.DatetimeIndex(timestamps))
    observed = work.notna()
    observation_time = pd.Series(work.index, index=work.index).where(observed)
    previous_time = observation_time.ffill()
    next_time = observation_time.bfill()
    span_hours = (next_time - previous_time).dt.total_seconds().div(3600.0)
    eligible = work.isna() & previous_time.notna() & next_time.notna() & span_hours.le(max_gap_hours)
    interpolated = work.interpolate(method="time", limit_area="inside")
    imputed = eligible & interpolated.notna()
    filled = work.copy()
    filled.loc[imputed] = interpolated.loc[imputed]
    max_used = float(span_hours.loc[imputed].max()) if imputed.any() else 0.0
    return pd.Series(filled.to_numpy(dtype="float32")), imputed.to_numpy(dtype=bool), max_used


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(df[TIME_COL], errors="raise")
    hour = timestamp.dt.hour.astype("int16")
    month = timestamp.dt.month.astype("int16")
    day = timestamp.dt.dayofyear.astype("int16")
    df["hour"] = hour
    df["month"] = month
    df["day_of_year"] = day
    df["sin_hour"] = np.sin(2.0 * math.pi * hour / 24.0).astype("float32")
    df["cos_hour"] = np.cos(2.0 * math.pi * hour / 24.0).astype("float32")
    df["sin_day"] = np.sin(2.0 * math.pi * day / 365.25).astype("float32")
    df["cos_day"] = np.cos(2.0 * math.pi * day / 365.25).astype("float32")
    return df


def chronological_split(n_rows: int, fractions: list[float]) -> np.ndarray:
    train_end = int(math.floor(n_rows * fractions[0]))
    validation_end = int(math.floor(n_rows * (fractions[0] + fractions[1])))
    split = np.empty(n_rows, dtype=object)
    split[:train_end] = "train"
    split[train_end:validation_end] = "val"
    split[validation_end:] = "test"
    return split


def fixed_minmax(values: pd.Series, low: float, high: float) -> pd.Series:
    if high <= low:
        raise ValueError(f"Invalid MinMax bounds: {low}, {high}")
    return ((pd.to_numeric(values, errors="coerce") - low) / (high - low)).astype("float32")


def assign_event_fields(
    timestamps: pd.Series,
    events: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    n_rows = len(timestamps)
    result = pd.DataFrame(index=np.arange(n_rows))
    if n_rows == 0 or not events:
        result["in_ph_event"] = np.zeros(n_rows, dtype=bool)
        result["ph_event_label"] = np.full(n_rows, "none", dtype=object)
        result["ph_event_direction"] = np.full(n_rows, "none", dtype=object)
        result["event_id"] = np.zeros(n_rows, dtype="int64")
        return result, {"matched_event_rows": 0, "matched_low_rows": 0, "matched_high_rows": 0, "matched_mixed_rows": 0}

    timestamp_values = pd.to_datetime(timestamps).to_numpy(dtype="datetime64[ns]")
    low_diff = np.zeros(n_rows + 1, dtype="int32")
    high_diff = np.zeros(n_rows + 1, dtype="int32")
    event_ids = np.zeros(n_rows, dtype="int64")

    for event in events:
        start = np.datetime64(pd.Timestamp(event["incident_start"]), "ns")
        end = np.datetime64(pd.Timestamp(event["incident_end"]), "ns")
        left = int(np.searchsorted(timestamp_values, start, side="left"))
        right = int(np.searchsorted(timestamp_values, end, side="right"))
        if left >= right:
            continue
        target_diff = low_diff if event["direction"] == "low" else high_diff
        target_diff[left] += 1
        target_diff[right] -= 1
        event_ids[left:right] = int(event["event_id"])

    low_active = np.cumsum(low_diff[:-1]) > 0
    high_active = np.cumsum(high_diff[:-1]) > 0
    mixed = low_active & high_active
    low_only = low_active & ~high_active
    high_only = high_active & ~low_active
    in_event = low_active | high_active

    labels = np.full(n_rows, "none", dtype=object)
    directions = np.full(n_rows, "none", dtype=object)
    labels[low_only] = "pH_exceedance_low"
    directions[low_only] = "low"
    labels[high_only] = "pH_exceedance_high"
    directions[high_only] = "high"
    labels[mixed] = "mixed"
    directions[mixed] = "mixed"
    event_ids[mixed] = -1

    result["in_ph_event"] = in_event
    result["ph_event_label"] = labels
    result["ph_event_direction"] = directions
    result["event_id"] = event_ids
    stats = {
        "matched_event_rows": int(in_event.sum()),
        "matched_low_rows": int(low_only.sum()),
        "matched_high_rows": int(high_only.sum()),
        "matched_mixed_rows": int(mixed.sum()),
    }
    return result, stats


def load_ph_events(path: Path, config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Event file not found: {path}")
    usecols = ["site_id", "parameter", "label", "incident_start", "incident_end"]
    events = pd.read_csv(path, usecols=usecols, dtype={"site_id": "string"}, low_memory=False)
    event_config = config["events"]
    events = events.loc[events["parameter"].astype(str).str.strip().eq(event_config["parameter_exact"])].copy()
    events = events.loc[events["label"].isin(event_config["allowed_labels"])].copy()
    events["site_id"] = events["site_id"].map(normalise_site_id)
    events["incident_start"] = pd.to_datetime(events["incident_start"], errors="coerce")
    events["incident_end"] = pd.to_datetime(events["incident_end"], errors="coerce")
    events = events.dropna(subset=["site_id", "incident_start", "incident_end"])
    events = events.loc[events["incident_end"].ge(events["incident_start"])].copy()
    events = events.sort_values(["site_id", "incident_start", "incident_end", "label"], kind="mergesort").reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1, dtype="int64")
    events["direction"] = np.where(events["label"].eq("pH_exceedance_low"), "low", "high")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for site_id, group in events.groupby("site_id", sort=False):
        grouped[str(site_id)] = group[
            ["event_id", "incident_start", "incident_end", "direction", "label"]
        ].to_dict(orient="records")
    summary = {
        "source_rows_after_exact_filter": int(len(events)),
        "source_sites_after_exact_filter": int(events["site_id"].nunique()),
        "low_events": int(events["direction"].eq("low").sum()),
        "high_events": int(events["direction"].eq("high").sum()),
        "event_file_sha256": sha256_file(path),
    }
    return grouped, summary


def determine_quality_tier(
    coverage_days: float,
    usable_rows: int,
    usable_fraction: float,
    split_counts: dict[str, int],
    config: dict[str, Any],
) -> tuple[str, str]:
    primary = config["quality_tiers"]["primary_strict"]
    primary_checks = {
        "coverage_days": coverage_days >= float(primary["minimum_coverage_days"]),
        "usable_rows": usable_rows >= int(primary["minimum_usable_rows"]),
        "usable_fraction": usable_fraction >= float(primary["minimum_usable_fraction"]),
        "train_rows": split_counts.get("train", 0) >= int(primary["minimum_train_rows"]),
        "validation_rows": split_counts.get("val", 0) >= int(primary["minimum_validation_rows"]),
        "test_rows": split_counts.get("test", 0) >= int(primary["minimum_test_rows"]),
    }
    if all(primary_checks.values()):
        return "primary_strict", "all strict quality thresholds passed"

    extended = config["quality_tiers"]["extended_30d"]
    if coverage_days >= float(extended["minimum_coverage_days"]) and usable_rows >= int(extended["minimum_usable_rows"]):
        failed = ";".join(key for key, passed in primary_checks.items() if not passed)
        return "extended_30d", f"strict thresholds failed: {failed}"

    reasons: list[str] = []
    if coverage_days < float(extended["minimum_coverage_days"]):
        reasons.append("coverage_under_30d")
    if usable_rows < int(extended["minimum_usable_rows"]):
        reasons.append("usable_rows_under_300")
    return "supplementary", ";".join(reasons) or "supplementary quality tier"


def model_minmax_bounds(code: str, network: str, config: dict[str, Any]) -> tuple[float, float]:
    bounds = config["fixed_minmax_bounds"]
    if code == "00300":
        key = "00300_ea_percent_saturation_like" if str(network).startswith("EA") else "00300_mg_l"
    else:
        key = code
    low, high = bounds[key]
    return float(low), float(high)


def build_native_and_model(
    raw: pd.DataFrame,
    mapping: dict[str, str],
    qc_mapping: dict[str, str],
    site_id: str,
    network: str,
    events: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    initial_rows = int(len(raw))
    parsed_time, timezone_status = parse_timestamps(raw[mapping[TIME_COL]])
    timestamp_invalid_rows = int(parsed_time.isna().sum())

    source_site_mismatch_rows = 0
    if SITE_COL in mapping:
        source_ids = raw[mapping[SITE_COL]].map(normalise_site_id)
        source_site_mismatch_rows = int((source_ids.ne("") & source_ids.ne(site_id)).sum())

    work = pd.DataFrame(
        {
            TIME_COL: parsed_time,
            "_source_order": np.arange(initial_rows, dtype="int64"),
        }
    )
    audit_counts: dict[str, Any] = {}
    for code in REQUIRED_CODES:
        cleaned, qc_status, range_flag, stats = clean_numeric_column(
            raw[mapping[code]],
            raw[qc_mapping[code]] if code in qc_mapping else None,
            code,
            network,
            config,
        )
        work[code] = cleaned
        work[f"qc_status_{code}"] = qc_status
        work[f"range_flag_{code}"] = range_flag
        audit_counts.update(stats)

    del raw
    work = work.loc[work[TIME_COL].notna()].copy()
    work["_valid_required_count"] = work[REQUIRED_CODES].notna().sum(axis=1).astype("int8")
    work = work.sort_values(
        [TIME_COL, "_valid_required_count", "_source_order"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    duplicate_rows = int(work.duplicated(TIME_COL).sum())
    work = work.drop_duplicates(TIME_COL, keep="first")
    work = work.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
    work = work.drop(columns=["_valid_required_count", "_source_order"])

    max_gap = float(config["max_interpolation_gap_hours"])
    maximum_interpolation_span_used = 0.0
    for code in REQUIRED_CODES:
        work[f"observed_{code}"] = work[code].notna()
    for code in PREDICTOR_CODES:
        filled, imputed, maximum_used = interpolate_short_internal_gaps(work[TIME_COL], work[code], max_gap)
        work[f"{code}_filled"] = filled
        work[f"imputed_{code}"] = imputed
        audit_counts[f"{code}_imputed"] = int(imputed.sum())
        maximum_interpolation_span_used = max(maximum_interpolation_span_used, maximum_used)

    usable_mask = work[TARGET_CODE].notna()
    for code in PREDICTOR_CODES:
        usable_mask &= work[f"{code}_filled"].notna()
    usable_positions = np.flatnonzero(usable_mask.to_numpy())
    model = pd.DataFrame(
        {
            SITE_COL: np.full(len(usable_positions), site_id, dtype=object),
            TIME_COL: work.loc[usable_mask, TIME_COL].to_numpy(),
            "00400_native": work.loc[usable_mask, TARGET_CODE].to_numpy(dtype="float32"),
        }
    )
    for code in PREDICTOR_CODES:
        model[f"{code}_native"] = work.loc[usable_mask, f"{code}_filled"].to_numpy(dtype="float32")

    fractions = [float(value) for value in config["split_fractions"]]
    model["split"] = chronological_split(len(model), fractions)
    split_counts = {key: int(value) for key, value in model["split"].value_counts().to_dict().items()}
    for name in ["train", "val", "test"]:
        split_counts.setdefault(name, 0)

    # Fixed teacher-compatible transforms are independent of future rows.
    for code in [TARGET_CODE, "00010", "00300", "63680"]:
        native_col = "00400_native" if code == TARGET_CODE else f"{code}_native"
        low, high = model_minmax_bounds(code, network, config)
        model[code] = fixed_minmax(model[native_col], low, high)

    conductance_train = pd.to_numeric(
        model.loc[model["split"].eq("train"), "00095_native"], errors="coerce"
    )
    conductance_mean = float(conductance_train.mean()) if len(conductance_train) else 0.0
    conductance_scale = float(conductance_train.std(ddof=0)) if len(conductance_train) else 1.0
    zero_variance_conductance = not np.isfinite(conductance_scale) or conductance_scale <= 0.0
    if zero_variance_conductance:
        conductance_scale = 1.0
    model["00095"] = (
        (pd.to_numeric(model["00095_native"], errors="coerce") - conductance_mean) / conductance_scale
    ).astype("float32")

    model = add_time_features(model)
    for code in REQUIRED_CODES:
        model[f"observed_{code}"] = work.loc[usable_mask, f"observed_{code}"].to_numpy(dtype=bool)
        model[f"qc_status_{code}"] = work.loc[usable_mask, f"qc_status_{code}"].to_numpy(dtype="int8")
        model[f"range_flag_{code}"] = work.loc[usable_mask, f"range_flag_{code}"].to_numpy(dtype="int8")
    for code in PREDICTOR_CODES:
        model[f"imputed_{code}"] = work.loc[usable_mask, f"imputed_{code}"].to_numpy(dtype=bool)

    unit_family = do_unit_family(network)
    model["do_unit_family"] = unit_family
    event_frame, event_stats = assign_event_fields(model[TIME_COL], events)
    for column in EVENT_COLUMNS:
        model[column] = event_frame[column].to_numpy()

    native_columns = [SITE_COL, TIME_COL]
    native = pd.DataFrame({SITE_COL: np.full(len(work), site_id, dtype=object), TIME_COL: work[TIME_COL]})
    native["network"] = network
    native["do_unit_family"] = unit_family
    for code in REQUIRED_CODES:
        native[code] = work[code].astype("float32")
        native[f"observed_{code}"] = work[f"observed_{code}"].astype(bool)
        native[f"qc_status_{code}"] = work[f"qc_status_{code}"].astype("int8")
        native[f"range_flag_{code}"] = work[f"range_flag_{code}"].astype("int8")
    for code in PREDICTOR_CODES:
        native[f"{code}_filled"] = work[f"{code}_filled"].astype("float32")
        native[f"imputed_{code}"] = work[f"imputed_{code}"].astype(bool)
    del work

    usable_rows = int(len(model))
    deduplicated_rows = int(len(native))
    usable_fraction = float(usable_rows / deduplicated_rows) if deduplicated_rows else 0.0
    if usable_rows >= 2:
        coverage_days = float((model[TIME_COL].iloc[-1] - model[TIME_COL].iloc[0]).total_seconds() / 86400.0)
    else:
        coverage_days = 0.0
    tier, tier_reason = determine_quality_tier(
        coverage_days,
        usable_rows,
        usable_fraction,
        split_counts,
        config,
    )

    # Acceptance assertions are intentionally close to the data-write boundary.
    assert native[TIME_COL].is_monotonic_increasing
    assert native[TIME_COL].is_unique
    assert model[TIME_COL].is_monotonic_increasing
    assert model[TIME_COL].is_unique
    assert model["observed_00400"].all()
    assert model[[TARGET_CODE, *PREDICTOR_CODES]].notna().all(axis=None)
    assert maximum_interpolation_span_used <= max_gap + 1e-9
    assert not set(EVENT_COLUMNS) & set(MODEL_FEATURES)

    stats: dict[str, Any] = {
        "initial_rows": initial_rows,
        "invalid_timestamp_rows": timestamp_invalid_rows,
        "source_site_mismatch_rows": source_site_mismatch_rows,
        "duplicate_timestamp_rows_removed": duplicate_rows,
        "native_clean_rows": deduplicated_rows,
        "usable_model_rows": usable_rows,
        "usable_complete_fraction": usable_fraction,
        "usable_start": model[TIME_COL].iloc[0] if usable_rows else None,
        "usable_end": model[TIME_COL].iloc[-1] if usable_rows else None,
        "usable_coverage_days": coverage_days,
        "train_rows": split_counts["train"],
        "validation_rows": split_counts["val"],
        "test_rows": split_counts["test"],
        "quality_tier": tier,
        "quality_tier_reason": tier_reason,
        "timezone_status": timezone_status,
        "do_unit_family": unit_family,
        "maximum_interpolation_span_used_hours": maximum_interpolation_span_used,
        "conductance_train_mean": conductance_mean,
        "conductance_train_scale": conductance_scale,
        "conductance_zero_variance_fallback": zero_variance_conductance,
        **audit_counts,
        **event_stats,
    }
    scaler = {
        "site_id": site_id,
        "network": network,
        "quality_tier": tier,
        "split_fractions": fractions,
        "fixed_minmax": {
            code: {
                "low": model_minmax_bounds(code, network, config)[0],
                "high": model_minmax_bounds(code, network, config)[1],
                "clip": False,
            }
            for code in [TARGET_CODE, "00010", "00300", "63680"]
        },
        "00095_standard_scaler": {
            "fit_scope": "rows where split == train only",
            "mean": conductance_mean,
            "scale": conductance_scale,
            "zero_variance_fallback": zero_variance_conductance,
            "n_train_rows": split_counts["train"],
        },
        "do_unit_family": unit_family,
        "inverse_transform_available": True,
    }
    return native, model, stats, scaler


def worker_initialiser(
    config: dict[str, Any],
    project_root: str,
    output_root: str,
    events: dict[str, list[dict[str, Any]]],
    config_hash: str,
    script_hash: str,
    resume: bool,
) -> None:
    global _WORKER_CONFIG, _WORKER_PROJECT_ROOT, _WORKER_OUTPUT_ROOT
    global _WORKER_EVENTS, _WORKER_CONFIG_HASH, _WORKER_SCRIPT_HASH, _WORKER_RESUME
    _WORKER_CONFIG = config
    _WORKER_PROJECT_ROOT = Path(project_root)
    _WORKER_OUTPUT_ROOT = Path(output_root)
    _WORKER_EVENTS = events
    _WORKER_CONFIG_HASH = config_hash
    _WORKER_SCRIPT_HASH = script_hash
    _WORKER_RESUME = resume


def checkpoint_paths(output_root: Path, site_id: str) -> dict[str, Path]:
    name = safe_name(site_id)
    return {
        "native": output_root / "native_clean_by_site" / f"{name}.parquet",
        "model": output_root / "model_ready_by_site" / f"{name}.parquet",
        "scaler": output_root / "scalers" / f"{name}.json",
        "checkpoint": output_root / "checkpoints" / "sites" / f"{name}.json",
        "success": output_root / "checkpoints" / "site_success" / f"{name}._SUCCESS",
    }


def validated_resume_checkpoint(
    paths: dict[str, Path],
    source_path: Path,
    config_hash: str,
    script_hash: str,
) -> dict[str, Any] | None:
    if not paths["checkpoint"].exists() or not paths["success"].exists():
        return None
    try:
        checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    source_stat = source_path.stat()
    checks = [
        checkpoint.get("status") == "success",
        checkpoint.get("config_sha256") == config_hash,
        checkpoint.get("script_sha256") == script_hash,
        int(checkpoint.get("source_size_bytes", -1)) == int(source_stat.st_size),
        int(checkpoint.get("source_mtime_ns", -1)) == int(source_stat.st_mtime_ns),
        all(paths[key].exists() for key in ["native", "model", "scaler"]),
    ]
    if not all(checks):
        return None
    checkpoint["resumed_from_checkpoint"] = True
    return checkpoint


def process_site(row: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_CONFIG is None or _WORKER_PROJECT_ROOT is None or _WORKER_OUTPUT_ROOT is None:
        raise RuntimeError("Worker was not initialised")
    config = _WORKER_CONFIG
    site_id = normalise_site_id(row.get("site_id"))
    network = str(row.get("network", ""))
    selected_file = str(row.get("selected_file", ""))
    source_path = resolve_path(selected_file, _WORKER_PROJECT_ROOT)
    paths = checkpoint_paths(_WORKER_OUTPUT_ROOT, site_id)
    started = time.perf_counter()

    base: dict[str, Any] = {
        "site_id": site_id,
        "candidate_rank": int(float(row.get("candidate_rank", 0) or 0)),
        "network": network,
        "selected_file": project_relative(source_path),
        "source_format": source_path.suffix.lower().lstrip("."),
        "status": "failed",
        "resumed_from_checkpoint": False,
        "config_sha256": _WORKER_CONFIG_HASH,
        "script_sha256": _WORKER_SCRIPT_HASH,
    }

    try:
        if not source_path.exists():
            raise FileNotFoundError(f"Selected source file does not exist: {source_path}")
        if _WORKER_RESUME:
            checkpoint = validated_resume_checkpoint(
                paths, source_path, _WORKER_CONFIG_HASH, _WORKER_SCRIPT_HASH
            )
            if checkpoint is not None:
                return checkpoint

        paths["success"].unlink(missing_ok=True)
        source_stat = source_path.stat()
        source_hash = sha256_file(source_path) if config["output"].get("hash_source_files", True) else ""
        mapping, qc_mapping, read_columns = build_source_mapping(source_path)
        raw = read_source_frame(source_path, read_columns)
        native, model, stats, scaler = build_native_and_model(
            raw,
            mapping,
            qc_mapping,
            site_id,
            network,
            _WORKER_EVENTS.get(site_id, []),
            config,
        )
        del raw

        compression = str(config["output"].get("parquet_compression", "zstd"))
        atomic_write_parquet(native, paths["native"], compression)
        atomic_write_parquet(model, paths["model"], compression)
        atomic_write_json(scaler, paths["scaler"])

        native_hash = sha256_file(paths["native"])
        model_hash = sha256_file(paths["model"])
        scaler_hash = sha256_file(paths["scaler"])
        elapsed = time.perf_counter() - started
        result = {
            **base,
            **stats,
            "status": "success",
            "source_size_bytes": int(source_stat.st_size),
            "source_mtime_ns": int(source_stat.st_mtime_ns),
            "source_sha256": source_hash,
            "native_output": project_relative(paths["native"]),
            "native_output_size_bytes": int(paths["native"].stat().st_size),
            "native_output_sha256": native_hash,
            "model_output": project_relative(paths["model"]),
            "model_output_size_bytes": int(paths["model"].stat().st_size),
            "model_output_sha256": model_hash,
            "scaler_output": project_relative(paths["scaler"]),
            "scaler_output_sha256": scaler_hash,
            "elapsed_seconds": elapsed,
            "error_type": "",
            "error_message": "",
            "traceback": "",
        }
        atomic_write_json(result, paths["checkpoint"])
        atomic_write_text(
            f"site_id={site_id}\nconfig_sha256={_WORKER_CONFIG_HASH}\nscript_sha256={_WORKER_SCRIPT_HASH}\n",
            paths["success"],
        )
        return result
    except Exception as exc:  # noqa: BLE001 - failures must become auditable rows
        paths["success"].unlink(missing_ok=True)
        result = {
            **base,
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            if source_path.exists():
                stat = source_path.stat()
                result["source_size_bytes"] = int(stat.st_size)
                result["source_mtime_ns"] = int(stat.st_mtime_ns)
            atomic_write_json(result, paths["checkpoint"])
        except Exception:  # pragma: no cover - preserve the original error
            pass
        return result


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    df.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def validate_site_output(result: dict[str, Any], output_root: Path) -> list[str]:
    problems: list[str] = []
    if result.get("status") != "success":
        return problems
    site_id = str(result["site_id"])
    paths = checkpoint_paths(output_root, site_id)
    for key in ["native", "model", "scaler", "success"]:
        if not paths[key].exists():
            problems.append(f"{site_id}: missing {key} output")
    if problems:
        return problems
    native_pf = pq.ParquetFile(paths["native"])
    model_pf = pq.ParquetFile(paths["model"])
    if native_pf.metadata.num_rows != int(result.get("native_clean_rows", -1)):
        problems.append(f"{site_id}: native row count mismatch")
    if model_pf.metadata.num_rows != int(result.get("usable_model_rows", -1)):
        problems.append(f"{site_id}: model row count mismatch")
    required_model_columns = {
        SITE_COL,
        TIME_COL,
        "split",
        TARGET_CODE,
        *PREDICTOR_CODES,
        "00400_native",
        *MODEL_FEATURES,
        *EVENT_COLUMNS,
    }
    missing = sorted(required_model_columns - set(model_pf.schema_arrow.names))
    if missing:
        problems.append(f"{site_id}: missing model columns {missing}")
    try:
        scaler = json.loads(paths["scaler"].read_text(encoding="utf-8"))
        if scaler["00095_standard_scaler"]["fit_scope"] != "rows where split == train only":
            problems.append(f"{site_id}: conductance scaler scope is not train-only")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"{site_id}: unreadable scaler JSON ({exc})")
    return problems


def smoke_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in manifest.sort_values(["network", "candidate_rank"]).groupby("network", sort=True):
        rows.append(group.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def run_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    output_root: Path,
    events: dict[str, list[dict[str, Any]]],
    config_hash: str,
    script_hash: str,
    resume: bool,
    workers: int,
    label: str,
) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    initializer_args = (
        config,
        str(PROJECT_ROOT),
        str(output_root),
        events,
        config_hash,
        script_hash,
        resume,
    )
    results: list[dict[str, Any]] = []
    total = len(rows)
    if workers <= 1:
        worker_initialiser(*initializer_args)
        for index, row in enumerate(rows, start=1):
            result = process_site(row)
            results.append(result)
            logging.info(
                "%s %s/%s site=%s status=%s tier=%s rows=%s elapsed=%.1fs",
                label,
                index,
                total,
                result.get("site_id"),
                result.get("status"),
                result.get("quality_tier", ""),
                result.get("usable_model_rows", ""),
                float(result.get("elapsed_seconds", 0.0)),
            )
        return results

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=worker_initialiser,
        initargs=initializer_args,
    ) as executor:
        future_to_site = {executor.submit(process_site, row): row["site_id"] for row in rows}
        for index, future in enumerate(concurrent.futures.as_completed(future_to_site), start=1):
            site_id = future_to_site[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - executor-level failure
                result = {
                    "site_id": site_id,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            logging.info(
                "%s %s/%s site=%s status=%s tier=%s rows=%s elapsed=%.1fs",
                label,
                index,
                total,
                result.get("site_id"),
                result.get("status"),
                result.get("quality_tier", ""),
                result.get("usable_model_rows", ""),
                float(result.get("elapsed_seconds", 0.0)),
            )
    return results


def environment_report(output_root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(PROJECT_ROOT)
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "operating_system": platform.platform(),
        "python_version": sys.version,
        "logical_cpu_count": os.cpu_count(),
        "disk_total_gb": disk.total / 1024**3,
        "disk_free_gb": disk.free / 1024**3,
        "packages": {},
    }
    try:
        import psutil  # type: ignore

        payload["ram_total_gb"] = psutil.virtual_memory().total / 1024**3
    except Exception:
        payload["ram_total_gb"] = None
    for package in ["numpy", "pandas", "pyarrow", "scikit-learn", "joblib", "torch"]:
        try:
            payload["packages"][package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            payload["packages"][package] = None
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        payload["nvidia_smi"] = gpu.stdout.strip() if gpu.returncode == 0 else "unavailable"
    except Exception:
        payload["nvidia_smi"] = "unavailable"
    atomic_write_json(payload, output_root / "00_environment" / "environment_report.json")
    return payload


def aggregate_statuses(
    results: list[dict[str, Any]],
    manifest: pd.DataFrame,
    output_root: Path,
    config: dict[str, Any],
    config_path: Path,
    config_hash: str,
    script_hash: str,
    event_summary: dict[str, Any],
    environment: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    reports_dir = output_root / "reports"
    cohorts_dir = output_root / "cohorts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cohorts_dir.mkdir(parents=True, exist_ok=True)

    status = pd.DataFrame(results)
    if "site_id" not in status:
        status["site_id"] = pd.Series(dtype="string")
    status["site_id"] = status["site_id"].map(normalise_site_id).astype("string")
    status = status.sort_values(["candidate_rank", "site_id"], na_position="last").reset_index(drop=True)
    write_csv(status, reports_dir / "site_processing_status.csv")
    write_csv(status, reports_dir / "site_quality_audit.csv")

    successful = status.loc[status["status"].eq("success")].copy()
    failed = status.loc[~status["status"].eq("success")].copy()
    if not failed.empty:
        write_csv(failed, reports_dir / "failed_sites.csv")
    else:
        write_csv(pd.DataFrame(columns=["site_id", "network", "error_type", "error_message"]), reports_dir / "failed_sites.csv")

    tier_order = ["primary_strict", "extended_30d", "supplementary"]
    for tier in tier_order:
        tier_frame = successful.loc[successful["quality_tier"].eq(tier)].copy()
        write_csv(tier_frame, cohorts_dir / f"{tier}_sites.csv")

    cohort_counts = (
        successful.groupby(["network", "quality_tier"], dropna=False)
        .size()
        .rename("site_count")
        .reset_index()
    )
    if not cohort_counts.empty:
        cohort_counts["quality_tier"] = pd.Categorical(
            cohort_counts["quality_tier"], categories=tier_order, ordered=True
        )
        cohort_counts = cohort_counts.sort_values(["network", "quality_tier"])
        cohort_counts["quality_tier"] = cohort_counts["quality_tier"].astype("string")
    write_csv(cohort_counts, reports_dir / "cohort_counts_by_network.csv")

    metric_suffixes = [
        "source_nonmissing",
        "qc_fail",
        "qc_unknown",
        "hard_range_invalid",
        "sentinel_or_nonfinite",
        "soft_high_flag",
        "clean_observed",
        "imputed",
    ]
    variable_rows: list[dict[str, Any]] = []
    for code in REQUIRED_CODES:
        row: dict[str, Any] = {"parameter_code": code}
        for suffix in metric_suffixes:
            column = f"{code}_{suffix}"
            row[suffix] = int(pd.to_numeric(successful.get(column, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        variable_rows.append(row)
    variable_summary = pd.DataFrame(variable_rows)
    write_csv(variable_summary, reports_dir / "variable_processing_summary.csv")

    input_hashes = successful[
        [
            "site_id",
            "selected_file",
            "source_size_bytes",
            "source_mtime_ns",
            "source_sha256",
        ]
    ].copy() if not successful.empty else pd.DataFrame(
        columns=["site_id", "selected_file", "source_size_bytes", "source_mtime_ns", "source_sha256"]
    )
    write_csv(input_hashes, reports_dir / "input_file_hashes.csv")
    output_hashes = successful[
        [
            "site_id",
            "native_output",
            "native_output_size_bytes",
            "native_output_sha256",
            "model_output",
            "model_output_size_bytes",
            "model_output_sha256",
            "scaler_output",
            "scaler_output_sha256",
        ]
    ].copy() if not successful.empty else pd.DataFrame(
        columns=["site_id", "native_output", "native_output_sha256", "model_output", "model_output_sha256"]
    )
    write_csv(output_hashes, reports_dir / "output_file_hashes.csv")

    event_network = successful.groupby("network", dropna=False)[
        ["matched_event_rows", "matched_low_rows", "matched_high_rows", "matched_mixed_rows"]
    ].sum(numeric_only=True).reset_index() if not successful.empty else pd.DataFrame(
        columns=["network", "matched_event_rows", "matched_low_rows", "matched_high_rows", "matched_mixed_rows"]
    )
    write_csv(event_network, reports_dir / "event_match_by_network.csv")

    problems: list[str] = []
    expected_sites = set(manifest["site_id"].astype(str))
    result_sites = set(status["site_id"].astype(str))
    if status["site_id"].duplicated().any():
        problems.append("site_processing_status contains duplicate site IDs")
    if expected_sites != result_sites:
        missing = sorted(expected_sites - result_sites)
        unexpected = sorted(result_sites - expected_sites)
        problems.append(f"status/manifest site mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    if len(status) != len(manifest):
        problems.append(f"status row count {len(status)} does not equal manifest row count {len(manifest)}")

    for result in results:
        problems.extend(validate_site_output(result, output_root))

    tier_total = int(successful["quality_tier"].isin(tier_order).sum()) if not successful.empty else 0
    if tier_total != len(successful):
        problems.append("quality tier counts do not reconcile to successful sites")
    if len(failed):
        problems.append(f"{len(failed)} sites failed processing")

    counts = {
        "manifest_sites": int(len(manifest)),
        "successful_sites": int(len(successful)),
        "failed_sites": int(len(failed)),
        "primary_strict_sites": int(successful["quality_tier"].eq("primary_strict").sum()) if not successful.empty else 0,
        "extended_30d_sites": int(successful["quality_tier"].eq("extended_30d").sum()) if not successful.empty else 0,
        "supplementary_sites": int(successful["quality_tier"].eq("supplementary").sum()) if not successful.empty else 0,
        "native_clean_rows": int(pd.to_numeric(successful.get("native_clean_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "usable_model_rows": int(pd.to_numeric(successful.get("usable_model_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "train_rows": int(pd.to_numeric(successful.get("train_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "validation_rows": int(pd.to_numeric(successful.get("validation_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "test_rows": int(pd.to_numeric(successful.get("test_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "matched_event_rows": int(pd.to_numeric(successful.get("matched_event_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
    }
    summary = {
        "run_id": config["run_id"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "success" if not problems else "validation_failed",
        "counts": counts,
        "config_path": project_relative(config_path),
        "config_sha256": config_hash,
        "script_path": project_relative(SCRIPT_PATH),
        "script_sha256": script_hash,
        "candidate_manifest": project_relative(resolve_path(config["candidate_manifest"])),
        "event_source_summary": event_summary,
        "environment": environment,
        "quality_tier_definitions": config["quality_tiers"],
        "model_feature_columns": MODEL_FEATURES,
        "evaluation_only_event_columns": EVENT_COLUMNS,
        "validation_problems": problems,
    }
    atomic_write_json(summary, reports_dir / "run_summary.json")
    atomic_write_json(
        {
            "qc_status": QC_STATUS_DEFINITION,
            "range_flag": RANGE_FLAG_DEFINITION,
            "do_unit_families": {
                "ea_percent_saturation_like_unverified": "EA 00300 retained as a percent-saturation-like signal; not converted to mg/L",
                "reported_mg_l": "00300 retained as reported mg/L-like concentration for non-EA networks",
            },
            "event_fields_are_predictors": False,
            "target_history_is_predictor": False,
        },
        reports_dir / "field_definitions.json",
    )
    run_manifest = {
        "run_id": config["run_id"],
        "created_at": summary["generated_at"],
        "status": summary["status"],
        "counts": counts,
        "config_sha256": config_hash,
        "script_sha256": script_hash,
        "aggregate_outputs": [
            project_relative(reports_dir / name)
            for name in [
                "site_processing_status.csv",
                "site_quality_audit.csv",
                "variable_processing_summary.csv",
                "cohort_counts_by_network.csv",
                "event_match_by_network.csv",
                "input_file_hashes.csv",
                "output_file_hashes.csv",
                "run_summary.json",
                "field_definitions.json",
            ]
        ],
    }
    atomic_write_json(run_manifest, output_root / "run_manifest.json")

    report = f"""# Common1000 processed-data run report

Generated: {summary['generated_at']}

## Outcome

- Manifest sites: {counts['manifest_sites']:,}
- Successful sites: {counts['successful_sites']:,}
- Failed sites: {counts['failed_sites']:,}
- Primary strict sites: {counts['primary_strict_sites']:,}
- Extended 30-day sites: {counts['extended_30d_sites']:,}
- Supplementary sites: {counts['supplementary_sites']:,}
- Native cleaned rows: {counts['native_clean_rows']:,}
- Model-ready rows: {counts['usable_model_rows']:,}
- Train / validation / test rows: {counts['train_rows']:,} / {counts['validation_rows']:,} / {counts['test_rows']:,}
- Matched pH-event rows: {counts['matched_event_rows']:,}

## Scientific safeguards

- Target pH is observed, never interpolated, in model-ready rows.
- Predictor interpolation is internal-only and capped at six hours.
- Conductance scaling is fitted on each site's training rows only.
- EA dissolved oxygen is retained as a separately labelled percent-saturation-like unit family.
- Event fields are evaluation-only and are excluded from the declared feature list.

## Validation

Run status: **{summary['status']}**

Validation problems: {len(problems):,}. See `reports/run_summary.json` and `reports/failed_sites.csv`.
"""
    atomic_write_text(report, reports_dir / "FINAL_REPORT.md")
    return summary, problems


def setup_logging(output_root: Path) -> None:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    output_root = resolve_path(config["output_dir"])
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. Use --resume or choose a new run_id/output_dir."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging(output_root)
    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Output root: %s", output_root)

    config_hash = sha256_json(config)
    script_hash = sha256_file(SCRIPT_PATH)
    atomic_write_json(config, output_root / "effective_config.json")
    environment = environment_report(output_root)
    if float(environment["disk_free_gb"]) < 40.0:
        raise RuntimeError("Free disk space is below the configured 40 GB safety threshold")

    manifest_path = resolve_path(config["candidate_manifest"])
    manifest = pd.read_csv(manifest_path, dtype={"site_id": "string"})
    manifest["site_id"] = manifest["site_id"].map(normalise_site_id).astype("string")
    manifest["candidate_rank"] = pd.to_numeric(manifest["candidate_rank"], errors="raise").astype("int64")
    manifest["row_count"] = pd.to_numeric(manifest.get("row_count"), errors="coerce")
    if manifest["site_id"].duplicated().any():
        duplicates = manifest.loc[manifest["site_id"].duplicated(), "site_id"].tolist()
        raise ValueError(f"Candidate manifest contains duplicate site IDs: {duplicates[:10]}")
    for selected_file in manifest["selected_file"]:
        if not resolve_path(str(selected_file)).exists():
            raise FileNotFoundError(f"Manifest source file missing: {selected_file}")
    if args.site_limit is not None:
        manifest = manifest.sort_values("candidate_rank").head(int(args.site_limit)).copy()
        logging.warning("Development site limit is active: %s sites", len(manifest))
    logging.info("Loaded candidate manifest with %s sites", len(manifest))

    events_path = resolve_path(config["events_path"])
    events, event_summary = load_ph_events(events_path, config)
    logging.info(
        "Loaded %s exact pH events across %s sites",
        event_summary["source_rows_after_exact_filter"],
        event_summary["source_sites_after_exact_filter"],
    )

    workers = int(args.workers or config["default_workers"])
    workers = max(1, workers)
    smoke_output = output_root / "00_smoke"
    if not args.skip_smoke:
        smoke = smoke_rows(manifest)
        smoke_results = run_rows(
            smoke.to_dict(orient="records"),
            config,
            smoke_output,
            events,
            config_hash,
            script_hash,
            args.resume,
            min(workers, len(smoke)),
            "SMOKE",
        )
        smoke_problems: list[str] = []
        for result in smoke_results:
            if result.get("status") != "success":
                smoke_problems.append(
                    f"{result.get('site_id')}: {result.get('error_type')} {result.get('error_message')}"
                )
            smoke_problems.extend(validate_site_output(result, smoke_output))
        atomic_write_json(
            {"results": smoke_results, "validation_problems": smoke_problems},
            smoke_output / "smoke_summary.json",
        )
        if smoke_problems:
            raise RuntimeError(f"Smoke test failed: {smoke_problems[:5]}")
        atomic_write_text("four-network smoke test passed\n", smoke_output / "_SUCCESS")
        logging.info("Four-network smoke test passed")
    elif not (smoke_output / "_SUCCESS").exists():
        logging.warning("Smoke test was skipped without an existing smoke _SUCCESS marker")

    if args.smoke_only:
        logging.info("Smoke-only run completed")
        return

    ordered = manifest.sort_values(["row_count", "candidate_rank"], ascending=[False, True], na_position="last")
    full_results = run_rows(
        ordered.to_dict(orient="records"),
        config,
        output_root,
        events,
        config_hash,
        script_hash,
        args.resume,
        workers,
        "FULL",
    )
    summary, problems = aggregate_statuses(
        full_results,
        manifest,
        output_root,
        config,
        config_path,
        config_hash,
        script_hash,
        event_summary,
        environment,
    )
    if problems:
        atomic_write_text("\n".join(problems) + "\n", output_root / "_VALIDATION_FAILED")
        logging.error("Run validation failed with %s problems", len(problems))
        raise SystemExit(2)
    (output_root / "_VALIDATION_FAILED").unlink(missing_ok=True)
    atomic_write_text(
        f"run_id={config['run_id']}\nconfig_sha256={config_hash}\nscript_sha256={script_hash}\n",
        output_root / "_SUCCESS",
    )
    logging.info("Run completed successfully: %s", summary["counts"])


if __name__ == "__main__":
    main()
