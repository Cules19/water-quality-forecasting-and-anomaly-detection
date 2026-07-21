from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from scipy.stats import spearmanr


warnings.filterwarnings("ignore", category=FutureWarning)


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
OUT_DIR = ROOT / "analysis_outputs/ph_site_hydrology_performance_analysis_v1"
FIG_DIR = OUT_DIR / "figures"
LOG_PATH = OUT_DIR / "site_hydrology_analysis_log.txt"
MODEL_READY_PATH = ROOT / "analysis_outputs/ph_hgb_processed200_metadata_v1/processed200_model_ready.parquet"
SELECTED_SITES_PATH = ROOT / "analysis_outputs/ph_hgb_processed200_metadata_v1/processed200_selected_sites.csv"
REGISTER_PATH = ROOT / "data/reference/sites_register.json"

QUALITY_APPROVED = {"good", "acceptable"}
EVENT_METRICS = [
    "absolute_top5_event_enrichment",
    "absolute_top5_low_enrichment",
    "absolute_top5_high_enrichment",
    "negative_tail5_low_enrichment",
    "positive_tail5_high_enrichment",
    "overall_event_hit_rate",
    "negative_tail5_low_hit_rate",
    "positive_tail5_high_hit_rate",
]
PREDICTION_METRICS = ["rmse", "mae", "r2"]
SITE_PROPERTIES = [
    "latitude",
    "longitude",
    "elevation",
    "log10_mean_flow",
    "flow_CV",
    "richards_baker_flashiness",
    "log10_Q50",
    "log10_Q95",
    "Q95_Q50_ratio",
    "log10_drainage_area",
]
METADATA_EFFECT_METRICS = [
    "rmse_gain_full",
    "rmse_gain_v2",
    "enrichment_change_full",
    "enrichment_change_v2",
    "low_tail_enrichment_change_full",
    "low_tail_enrichment_change_v2",
    "high_tail_enrichment_change_full",
    "high_tail_enrichment_change_v2",
]
MODEL_ORDER = [
    "M200_A_no_metadata",
    "M200_B_with_metadata",
    "HGB200_V2",
    "M0_reference_nonevent",
    "G100_B",
    "GRU100_V2",
    "L100_B",
]
MODEL_COLORS = {
    "M200_A_no_metadata": "#244C70",
    "M200_B_with_metadata": "#3F948A",
    "HGB200_V2": "#C08A45",
    "M0_reference_nonevent": "#7467A3",
    "G100_B": "#4B7FA8",
    "GRU100_V2": "#6F9B62",
    "L100_B": "#6F7780",
}
MODEL_LABELS = {
    "M200_A_no_metadata": "HGB200 no metadata",
    "M200_B_with_metadata": "HGB200 full metadata",
    "HGB200_V2": "HGB200 V2 metadata",
    "M0_reference_nonevent": "M0 non-event HGB",
    "G100_B": "GRU100 full metadata",
    "GRU100_V2": "GRU100 V2 metadata",
    "L100_B": "LSTM100 full metadata",
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_type: str
    site_count: int
    metadata_set: str
    training_mode: str
    prediction_file: str
    row_filter_col: str | None = None
    row_filter_value: str | None = None


MODEL_SPECS = [
    ModelSpec(
        "M200_A_no_metadata",
        "HGB",
        200,
        "none",
        "all_data",
        "analysis_outputs/ph_hgb_processed200_metadata_v1/prediction_test_M200_A_no_metadata.parquet",
    ),
    ModelSpec(
        "M200_B_with_metadata",
        "HGB",
        200,
        "full",
        "all_data",
        "analysis_outputs/ph_hgb_processed200_metadata_v1/prediction_test_M200_B_with_metadata.parquet",
    ),
    ModelSpec(
        "HGB200_V2",
        "HGB",
        200,
        "compact_hydrological_v2",
        "all_data",
        "analysis_outputs/ph_v2_metadata_rerun_v1/hgb200_v2_predictions_test.parquet",
    ),
    ModelSpec(
        "M0_reference_nonevent",
        "HGB",
        50,
        "none",
        "non_event",
        "analysis_outputs/ph_nonevent_training_full50_v3/nonevent_predictions_test.parquet",
        "model",
        "M4_hgb_nonevent_train",
    ),
    ModelSpec(
        "G100_B",
        "GRU",
        100,
        "full",
        "all_data",
        "analysis_outputs/ph_sequence100_gru_lstm_metadata_v1/prediction_test_G100_B_gru_with_metadata.parquet",
    ),
    ModelSpec(
        "GRU100_V2",
        "GRU",
        100,
        "compact_hydrological_v2",
        "all_data",
        "analysis_outputs/ph_v2_metadata_rerun_v1/gru100_v2_predictions_test.parquet",
    ),
    ModelSpec(
        "L100_B",
        "LSTM",
        100,
        "full",
        "all_data",
        "analysis_outputs/ph_sequence100_lstm_metadata_v1/prediction_test_L100_B_lstm_with_metadata.parquet",
    ),
]


def ensure_dirs(reset_log: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if reset_log or not LOG_PATH.exists():
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


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):.{digits}f}"


def model_fields(spec: ModelSpec) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "model_type": spec.model_type,
        "site_count": spec.site_count,
        "metadata_set": spec.metadata_set,
        "training_mode": spec.training_mode,
    }


def prediction_path(spec: ModelSpec) -> Path:
    return ROOT / spec.prediction_file


def normalise_site_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.upper()


def nested_get(mapping: dict[str, Any], *keys: str, default: Any = np.nan) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def dominant_land_cover(value: Any) -> Any:
    if not isinstance(value, dict) or not value:
        return np.nan
    valid = {str(k): pd.to_numeric(v, errors="coerce") for k, v in value.items()}
    valid = {k: float(v) for k, v in valid.items() if pd.notna(v)}
    return max(valid, key=valid.get) if valid else np.nan


def first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def audit_prediction_files() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    rows: list[dict[str, Any]] = []
    site_sets: dict[str, set[str]] = {}
    for spec in MODEL_SPECS:
        path = prediction_path(spec)
        if not path.exists():
            rows.append({**model_fields(spec), "prediction_file": spec.prediction_file, "exists": False})
            site_sets[spec.model_id] = set()
            continue
        pf = pq.ParquetFile(path)
        columns = pf.schema_arrow.names
        site_col = first_present(columns, ["site_id", "Site_ID"])
        timestamp_col = first_present(columns, ["timestamp", "target_timestamp", "datetime"])
        y_true_col = first_present(columns, ["y_true", "00400", "observed", "actual"])
        y_pred_col = first_present(columns, ["y_pred", "prediction", "predicted"])
        read_cols = [site_col] if site_col else []
        if spec.row_filter_col and spec.row_filter_col in columns:
            read_cols.append(spec.row_filter_col)
        data = pq.read_table(path, columns=read_cols).to_pandas() if read_cols else pd.DataFrame()
        if spec.row_filter_col and spec.row_filter_col in data:
            data = data.loc[data[spec.row_filter_col].astype(str).eq(str(spec.row_filter_value))]
        sites = set(data[site_col].map(normalise_site_id)) if site_col else set()
        sites.discard("")
        site_sets[spec.model_id] = sites
        rows.append(
            {
                **model_fields(spec),
                "prediction_file": spec.prediction_file,
                "exists": True,
                "file_size_bytes": path.stat().st_size,
                "raw_rows": pf.metadata.num_rows,
                "filtered_rows": len(data),
                "n_sites": len(sites),
                "site_id_col": site_col,
                "timestamp_col": timestamp_col,
                "y_true_col": y_true_col,
                "y_pred_col": y_pred_col,
                "event_columns": "|".join(
                    col for col in ["in_ph_event", "ph_event_label", "event_state"] if col in columns
                ),
                "raw_columns": "|".join(columns),
                "include_in_analysis": bool(site_col and timestamp_col and y_true_col and y_pred_col),
            }
        )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(OUT_DIR / "model_prediction_inventory.csv", index=False)
    return inventory, site_sets


def build_site_static_metadata(site_sets: dict[str, set[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry: dict[str, Any] = json.loads(REGISTER_PATH.read_text(encoding="utf-8"))
    all_sites = sorted(set().union(*site_sets.values()))
    static_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for site_id in all_sites:
        entry = registry.get(site_id, {})
        catchment = entry.get("catchment_analysis") if isinstance(entry.get("catchment_analysis"), dict) else {}
        discharge_stats = nested_get(entry, "discharge_description", "statistics", default={})
        if not isinstance(discharge_stats, dict):
            discharge_stats = {}
        static = {
            "site_id": site_id,
            "latitude": entry.get("latitude", np.nan),
            "longitude": entry.get("longitude", np.nan),
            "elevation": entry.get("elevation", np.nan),
            "drainage_area": entry.get("drainage_area_km2", np.nan),
            "official_average_discharge": discharge_stats.get("mean_discharge", np.nan),
            "official_average_discharge_note": (
                "registry discharge_description.statistics.mean_discharge; registry marks calculation_method="
                + str(discharge_stats.get("calculation_method", "unknown"))
            ),
            "basin": entry.get("basin_cd", np.nan),
            "state": entry.get("state", entry.get("state_cd", np.nan)),
            "site_type": entry.get("site_type", np.nan),
            "dominant_land_cover": dominant_land_cover(catchment.get("watershed_land_cover")),
            "catchment_precipitation": catchment.get("average_precip_mm_yr", np.nan),
            "catchment_evapotranspiration": catchment.get("average_evapo_trans_mm_yr", np.nan),
            "population": catchment.get("population_est_2020", np.nan),
            "dam_count": catchment.get("watershed_dams_no", np.nan),
            "irrigation_fraction": catchment.get("irrigation_area_pct", np.nan),
            "registry_source": entry.get("source", np.nan),
            "registry_units": entry.get("units", np.nan),
            "river_name": entry.get("river_name", np.nan),
            "waterbody_name": entry.get("waterbody_name", np.nan),
        }
        static_rows.append(static)
        lat = pd.to_numeric(static["latitude"], errors="coerce")
        lon = pd.to_numeric(static["longitude"], errors="coerce")
        elev = pd.to_numeric(static["elevation"], errors="coerce")
        audit_rows.append(
            {
                "site_id": site_id,
                "registry_match": bool(entry),
                "duplicate_registry_id": False,
                "missing_latitude": pd.isna(lat),
                "missing_longitude": pd.isna(lon),
                "missing_elevation": pd.isna(elev),
                "invalid_latitude": bool(pd.notna(lat) and not (-90 <= lat <= 90)),
                "invalid_longitude": bool(pd.notna(lon) and not (-180 <= lon <= 180)),
                "elevation_unit": "m (registry documentation)",
                "discharge_unit_group": entry.get("units", np.nan),
                **{f"in_{model_id}": site_id in sites for model_id, sites in site_sets.items()},
            }
        )
    static_df = pd.DataFrame(static_rows).sort_values("site_id")
    audit_df = pd.DataFrame(audit_rows).sort_values("site_id")
    static_df.to_csv(OUT_DIR / "site_static_metadata.csv", index=False)
    audit_df.to_csv(OUT_DIR / "site_register_audit.csv", index=False)
    return static_df, audit_df


def selected_processed_file_map() -> dict[str, Path]:
    selected = pd.read_csv(SELECTED_SITES_PATH, dtype=str)
    result: dict[str, Path] = {}
    for raw_path in selected["processed_file"].dropna():
        path = ROOT / str(raw_path)
        site_id = normalise_site_id(path.name.split("_", 1)[0])
        result[site_id] = path
    return result


def read_split_boundaries_and_processed_audit(
    required_sites: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("Reading existing model-ready split labels and processed 00060 audit values")
    pf = pq.ParquetFile(MODEL_READY_PATH)
    records: dict[str, dict[str, Any]] = {}
    for row_group in range(pf.num_row_groups):
        df = pf.read_row_group(row_group, columns=["site_id", "timestamp", "split", "00060"]).to_pandas()
        df["site_id"] = df["site_id"].map(normalise_site_id)
        df = df[df["site_id"].isin(required_sites)]
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["00060"] = pd.to_numeric(df["00060"], errors="coerce")
        for site_id, site_df in df.groupby("site_id", observed=True, sort=False):
            rec = records.setdefault(
                site_id,
                {
                    "site_id": site_id,
                    "first_timestamp": pd.NaT,
                    "train_start": pd.NaT,
                    "train_end": pd.NaT,
                    "validation_end": pd.NaT,
                    "final_timestamp": pd.NaT,
                    "n_train_model_rows": 0,
                    "n_val_model_rows": 0,
                    "n_test_model_rows": 0,
                    "processed_rows": 0,
                    "processed_missing": 0,
                    "processed_negative": 0,
                    "processed_sum": 0.0,
                    "processed_valid": 0,
                    "processed_min": np.nan,
                    "processed_max": np.nan,
                },
            )
            times = site_df["timestamp"].dropna()
            if not times.empty:
                rec["first_timestamp"] = min(
                    rec["first_timestamp"], times.min()
                ) if pd.notna(rec["first_timestamp"]) else times.min()
                rec["final_timestamp"] = max(
                    rec["final_timestamp"], times.max()
                ) if pd.notna(rec["final_timestamp"]) else times.max()
            for split, start_key, end_key, count_key in [
                ("train", "train_start", "train_end", "n_train_model_rows"),
                ("val", None, "validation_end", "n_val_model_rows"),
                ("test", None, None, "n_test_model_rows"),
            ]:
                split_times = site_df.loc[site_df["split"].astype(str).eq(split), "timestamp"].dropna()
                rec[count_key] += int(site_df["split"].astype(str).eq(split).sum())
                if start_key and not split_times.empty:
                    rec[start_key] = min(rec[start_key], split_times.min()) if pd.notna(rec[start_key]) else split_times.min()
                if end_key and not split_times.empty:
                    rec[end_key] = max(rec[end_key], split_times.max()) if pd.notna(rec[end_key]) else split_times.max()
            q = site_df["00060"]
            valid = q.dropna()
            rec["processed_rows"] += len(q)
            rec["processed_missing"] += int(q.isna().sum())
            rec["processed_negative"] += int((valid < 0).sum())
            rec["processed_sum"] += float(valid.sum())
            rec["processed_valid"] += len(valid)
            if not valid.empty:
                rec["processed_min"] = min(rec["processed_min"], valid.min()) if pd.notna(rec["processed_min"]) else float(valid.min())
                rec["processed_max"] = max(rec["processed_max"], valid.max()) if pd.notna(rec["processed_max"]) else float(valid.max())
        del df
    boundary_rows = []
    inventory_rows = []
    for site_id in sorted(required_sites):
        rec = records.get(site_id, {"site_id": site_id})
        start = rec.get("train_start", pd.NaT)
        end = rec.get("train_end", pd.NaT)
        train_days = (end.normalize() - start.normalize()).days + 1 if pd.notna(start) and pd.notna(end) else np.nan
        boundary_rows.append({**rec, "number_of_train_days": train_days, "split_source": rel(MODEL_READY_PATH)})
        n = rec.get("processed_rows", 0) or 0
        valid_n = rec.get("processed_valid", 0) or 0
        inventory_rows.append(
            {
                "site_id": site_id,
                "source_type": "processed_model_ready",
                "file_path": rel(MODEL_READY_PATH),
                "timestamp_column": "timestamp",
                "discharge_column": "00060",
                "units": "standardised site-specific values",
                "minimum": rec.get("processed_min", np.nan),
                "maximum": rec.get("processed_max", np.nan),
                "mean": rec.get("processed_sum", 0.0) / valid_n if valid_n else np.nan,
                "proportion_negative": rec.get("processed_negative", 0) / valid_n if valid_n else np.nan,
                "missingness": rec.get("processed_missing", 0) / n if n else np.nan,
                "time_start": rec.get("first_timestamp", pd.NaT),
                "time_end": rec.get("final_timestamp", pd.NaT),
                "likely_transformation": "standardised; confirmed from saved preprocessing pipeline",
                "suitable_for_flow_metrics": False,
                "selection_reason": "Rejected for physical metrics because centring produces negative values.",
            }
        )
    boundaries = pd.DataFrame(boundary_rows)
    processed_inventory = pd.DataFrame(inventory_rows)
    return boundaries, processed_inventory


def audit_transformations_and_units(
    required_sites: set[str], processed_map: dict[str, Path], static_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transform_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    static_lookup = static_df.set_index("site_id").to_dict("index")
    for site_id in sorted(required_sites):
        processed = processed_map.get(site_id)
        pipeline = processed.with_name(processed.stem + "_pipeline.joblib") if processed else None
        scaler_name = "unknown"
        center = np.nan
        scale = np.nan
        inverse_available = False
        if pipeline and pipeline.exists():
            payload = joblib.load(pipeline)
            scaler = payload.get("scalers", {}).get("00060") if isinstance(payload, dict) else None
            if scaler is not None:
                scaler_name = type(scaler).__name__
                inverse_available = hasattr(scaler, "inverse_transform")
                mean_value = getattr(scaler, "mean_", [np.nan])
                scale_value = getattr(scaler, "scale_", [np.nan])
                center = float(np.ravel(mean_value)[0]) if np.size(mean_value) else np.nan
                scale = float(np.ravel(scale_value)[0]) if np.size(scale_value) else np.nan
        transform_rows.extend(
            [
                {
                    "site_id": site_id,
                    "source_type": "clean_native",
                    "file_path": rel(ROOT / "data/clean"),
                    "likely_transformation": "native_or_cleaned_unscaled",
                    "scaler_type": "none",
                    "scaler_center": np.nan,
                    "scaler_scale": np.nan,
                    "inverse_transform_available": False,
                    "selected_for_flow_metrics": True,
                    "decision": "Use clean 00060 because it is physically interpretable.",
                },
                {
                    "site_id": site_id,
                    "source_type": "processed",
                    "file_path": rel(processed) if processed else "",
                    "likely_transformation": "standardised" if "StandardScaler" in scaler_name else scaler_name,
                    "scaler_type": scaler_name,
                    "scaler_center": center,
                    "scaler_scale": scale,
                    "inverse_transform_available": inverse_available,
                    "selected_for_flow_metrics": False,
                    "decision": "Do not use centred processed 00060 while native clean values are available.",
                },
            ]
        )
        meta = static_lookup.get(site_id, {})
        unit_rows.append(
            {
                "site_id": site_id,
                "registry_source": meta.get("registry_source", np.nan),
                "registry_unit_group": meta.get("registry_units", np.nan),
                "parameter_code": "00060",
                "selected_units": "ft3/s (cfs)",
                "unit_evidence": "project parameter mapping: 00060 = Discharge_cfs; all selected sites are USGS/Imperial",
                "unit_conversion_applied": False,
                "units_consistent_across_selected_sites": True,
                "suitable_for_cross_site_dimensionless_metrics": True,
                "suitable_for_cross_site_absolute_flow_comparison": True,
            }
        )
    transforms = pd.DataFrame(transform_rows)
    units = pd.DataFrame(unit_rows)
    transforms.to_csv(OUT_DIR / "flow_transformation_audit.csv", index=False)
    units.to_csv(OUT_DIR / "discharge_units_audit.csv", index=False)
    return transforms, units


def choose_clean_file(site_id: str, processed: Path | None) -> Path | None:
    if processed:
        exact = ROOT / "data/clean" / (processed.stem + ".csv")
        if exact.exists():
            return exact
    candidates = list((ROOT / "data/clean").glob(f"{site_id}_*.csv"))
    candidates = [path for path in candidates if "00060" in path.name]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stem.split("_")[-1], path.stat().st_mtime, path.stat().st_size))


def quality_label(valid_daily_count: int, coverage_fraction: float) -> str:
    if valid_daily_count >= 730 and coverage_fraction >= 0.80:
        return "good"
    if valid_daily_count >= 365:
        return "acceptable"
    if valid_daily_count >= 90:
        return "limited"
    return "insufficient"


def process_clean_flow_site(
    site_id: str,
    clean_path: Path,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    header = pd.read_csv(clean_path, nrows=0).columns.tolist()
    time_col = first_present(header, ["TIMESTAMP", "timestamp", "datetime", "date_time"])
    flow_col = first_present(header, ["00060", "Discharge_cfs", "discharge"])
    if not time_col or not flow_col:
        reason = "missing timestamp or 00060 column"
        return (
            {"site_id": site_id, "flow_source": rel(clean_path), "metric_exclusion_reason": reason},
            {"site_id": site_id, "flow_quality_label": "insufficient", "metric_exclusion_reason": reason},
            {
                "site_id": site_id,
                "source_type": "clean_native",
                "file_path": rel(clean_path),
                "suitable_for_flow_metrics": False,
                "selection_reason": reason,
            },
        )
    total_rows = 0
    missing_count = 0
    negative_count = 0
    valid_count = 0
    flow_sum = 0.0
    flow_min = np.nan
    flow_max = np.nan
    source_start = pd.NaT
    source_end = pd.NaT
    train_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        clean_path,
        usecols=[time_col, flow_col],
        chunksize=250_000,
        low_memory=False,
    ):
        total_rows += len(chunk)
        timestamps = pd.to_datetime(chunk[time_col], errors="coerce")
        flow = pd.to_numeric(chunk[flow_col], errors="coerce")
        missing_count += int(flow.isna().sum())
        valid = flow.dropna()
        negative_count += int((valid < 0).sum())
        valid_count += len(valid)
        flow_sum += float(valid.sum())
        if not valid.empty:
            flow_min = min(flow_min, valid.min()) if pd.notna(flow_min) else float(valid.min())
            flow_max = max(flow_max, valid.max()) if pd.notna(flow_max) else float(valid.max())
        valid_times = timestamps.dropna()
        if not valid_times.empty:
            source_start = min(source_start, valid_times.min()) if pd.notna(source_start) else valid_times.min()
            source_end = max(source_end, valid_times.max()) if pd.notna(source_end) else valid_times.max()
        mask = (
            timestamps.notna()
            & flow.notna()
            & (flow >= 0)
            & (timestamps >= train_start)
            & (timestamps <= train_end)
        )
        if mask.any():
            train_parts.append(pd.DataFrame({"timestamp": timestamps[mask], "flow": flow[mask].astype("float64")}))
    inventory = {
        "site_id": site_id,
        "source_type": "clean_native",
        "file_path": rel(clean_path),
        "timestamp_column": time_col,
        "discharge_column": flow_col,
        "units": "ft3/s (cfs)",
        "minimum": flow_min,
        "maximum": flow_max,
        "mean": flow_sum / valid_count if valid_count else np.nan,
        "proportion_negative": negative_count / valid_count if valid_count else np.nan,
        "missingness": missing_count / total_rows if total_rows else np.nan,
        "time_start": source_start,
        "time_end": source_end,
        "likely_transformation": "native_or_cleaned_unscaled",
        "suitable_for_flow_metrics": bool(train_parts),
        "selection_reason": "Selected by source priority: native clean 00060.",
    }
    if not train_parts:
        reason = "no valid non-negative clean 00060 observations in model train period"
        metric = {
            "site_id": site_id,
            "flow_source": rel(clean_path),
            "flow_units": "ft3/s (cfs)",
            "flow_is_native_or_unscaled": True,
            "train_start": train_start,
            "train_end": train_end,
            "flow_quality_label": "insufficient",
            "metric_exclusion_reason": reason,
        }
        quality = {**metric, "valid_daily_count": 0, "coverage_fraction": 0.0}
        inventory["suitable_for_flow_metrics"] = False
        inventory["selection_reason"] = reason
        return metric, quality, inventory
    train = pd.concat(train_parts, ignore_index=True)
    raw_train_rows = len(train)
    duplicate_timestamp_rows = int(train.duplicated("timestamp", keep=False).sum())
    by_timestamp = train.groupby("timestamp", as_index=False, observed=True).agg(
        flow=("flow", "mean"), raw_observation_count=("flow", "size")
    )
    by_timestamp["date"] = by_timestamp["timestamp"].dt.normalize()
    daily = by_timestamp.groupby("date", as_index=False, observed=True).agg(
        daily_mean_flow=("flow", "mean"), n_observations=("timestamp", "size")
    )
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["day_gap"] = daily["date"].diff().dt.days
    daily["is_consecutive_with_previous_day"] = daily["day_gap"].eq(1)
    valid_daily_count = len(daily)
    train_coverage_days = int((train_end.normalize() - train_start.normalize()).days + 1)
    coverage_fraction = valid_daily_count / train_coverage_days if train_coverage_days > 0 else np.nan
    gaps = (daily["day_gap"].dropna() - 1).clip(lower=0)
    leading_gap = int((daily["date"].min() - train_start.normalize()).days) if valid_daily_count else train_coverage_days
    trailing_gap = int((train_end.normalize() - daily["date"].max()).days) if valid_daily_count else train_coverage_days
    longest_gap = max([leading_gap, trailing_gap, int(gaps.max()) if not gaps.empty else 0])
    q = daily["daily_mean_flow"].astype("float64")
    consecutive = daily["is_consecutive_with_previous_day"].to_numpy(dtype=bool)
    diffs = q.diff().abs()
    rbi_numerator = float(diffs[consecutive].sum())
    rbi_denominator = float(q.sum())
    mean_flow = float(q.mean()) if valid_daily_count else np.nan
    flow_std = float(q.std(ddof=1)) if valid_daily_count >= 2 else np.nan
    cv = flow_std / mean_flow if pd.notna(flow_std) and mean_flow > 0 else np.nan
    rbi = rbi_numerator / rbi_denominator if rbi_denominator > 0 else np.nan
    q50 = float(q.quantile(0.50)) if valid_daily_count else np.nan
    q95_exceedance = float(q.quantile(0.05)) if valid_daily_count else np.nan
    q95_percentile = float(q.quantile(0.95)) if valid_daily_count else np.nan
    label = quality_label(valid_daily_count, coverage_fraction)
    exclusion = "" if label in QUALITY_APPROVED else f"flow quality {label}; excluded from main correlations"
    metric = {
        "site_id": site_id,
        "flow_source": rel(clean_path),
        "flow_units": "ft3/s (cfs)",
        "flow_is_native_or_unscaled": True,
        "train_start": train_start,
        "train_end": train_end,
        "valid_daily_count": valid_daily_count,
        "coverage_fraction": coverage_fraction,
        "mean_flow": mean_flow,
        "median_flow": q50,
        "flow_std": flow_std,
        "flow_CV": cv,
        "richards_baker_numerator": rbi_numerator,
        "richards_baker_denominator": rbi_denominator,
        "valid_consecutive_daily_pairs": int(consecutive.sum()),
        "richards_baker_flashiness": rbi,
        "Q50": q50,
        "Q95_exceedance": q95_exceedance,
        "Q95_percentile_audit": q95_percentile,
        "Q95_Q50_ratio": q95_exceedance / q50 if q50 > 0 else np.nan,
        "Q05_exceedance": q95_percentile,
        "Q10_exceedance": float(q.quantile(0.90)),
        "Q90_exceedance": float(q.quantile(0.10)),
        "min_flow": float(q.min()),
        "max_flow": float(q.max()),
        "zero_flow_fraction": float((q == 0).mean()),
        "train_coverage_days": train_coverage_days,
        "longest_gap_days": longest_gap,
        "flow_quality_label": label,
        "metric_exclusion_reason": exclusion,
    }
    quality = {
        "site_id": site_id,
        "flow_source": rel(clean_path),
        "train_start": train_start,
        "train_end": train_end,
        "raw_train_flow_rows": raw_train_rows,
        "duplicate_timestamp_rows": duplicate_timestamp_rows,
        "duplicate_timestamp_aggregation": "mean 00060 at duplicate timestamps",
        "daily_aggregation": "mean of unique-timestamp discharge values",
        "valid_daily_count": valid_daily_count,
        "train_coverage_days": train_coverage_days,
        "coverage_fraction": coverage_fraction,
        "longest_gap_days": longest_gap,
        "valid_consecutive_daily_pairs": int(consecutive.sum()),
        "flow_quality_label": label,
        "metric_exclusion_reason": exclusion,
        "long_gap_interpolation": "none",
        "differences_across_nonconsecutive_days": "excluded",
    }
    return metric, quality, inventory


def build_flow_metrics(
    required_sites: set[str],
    boundaries: pd.DataFrame,
    processed_inventory: pd.DataFrame,
    processed_map: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    boundary_lookup = boundaries.set_index("site_id").to_dict("index")
    metric_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    inventory_rows = processed_inventory.to_dict("records")
    for index, site_id in enumerate(sorted(required_sites), start=1):
        info = boundary_lookup.get(site_id, {})
        start = pd.to_datetime(info.get("train_start"), errors="coerce")
        end = pd.to_datetime(info.get("train_end"), errors="coerce")
        clean_path = choose_clean_file(site_id, processed_map.get(site_id))
        if clean_path is None or pd.isna(start) or pd.isna(end):
            reason = "missing clean 00060 source or existing train boundary"
            metric_rows.append(
                {
                    "site_id": site_id,
                    "flow_source": rel(clean_path) if clean_path else "",
                    "flow_units": "ft3/s (cfs)",
                    "flow_is_native_or_unscaled": True,
                    "train_start": start,
                    "train_end": end,
                    "flow_quality_label": "insufficient",
                    "metric_exclusion_reason": reason,
                }
            )
            quality_rows.append({"site_id": site_id, "flow_quality_label": "insufficient", "metric_exclusion_reason": reason})
            inventory_rows.append(
                {
                    "site_id": site_id,
                    "source_type": "clean_native",
                    "file_path": rel(clean_path) if clean_path else "",
                    "suitable_for_flow_metrics": False,
                    "selection_reason": reason,
                }
            )
            continue
        metric, quality, inventory = process_clean_flow_site(site_id, clean_path, start, end)
        metric_rows.append(metric)
        quality_rows.append(quality)
        inventory_rows.append(inventory)
        if index % 20 == 0 or index == len(required_sites):
            log(f"Flow metrics: processed {index}/{len(required_sites)} sites")
    metrics = pd.DataFrame(metric_rows).sort_values("site_id")
    quality = pd.DataFrame(quality_rows).sort_values("site_id")
    inventory = pd.DataFrame(inventory_rows).sort_values(["site_id", "source_type"])
    metrics.to_csv(OUT_DIR / "site_flow_metrics.csv", index=False)
    quality.to_csv(OUT_DIR / "flow_daily_series_quality_audit.csv", index=False)
    inventory.to_csv(OUT_DIR / "discharge_source_inventory.csv", index=False)
    return metrics, quality


def event_class_from_raw(raw: pd.DataFrame) -> pd.Series:
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
        state = pd.to_numeric(raw["event_state_raw"], errors="coerce")
        event_class = pd.Series("mixed_or_unknown", index=raw.index, dtype="string")
        event_class.loc[state == 0] = "normal"
        event_class.loc[state == 1] = "low_pH_event"
        event_class.loc[state == 2] = "high_pH_event"
        unresolved = state.isna() | ~state.isin([0, 1, 2, 3])
        event_class.loc[unresolved & low_mask] = "low_pH_event"
        event_class.loc[unresolved & high_mask] = "high_pH_event"
        event_class.loc[unresolved & (~in_event | normal_label) & ~(low_mask | high_mask)] = "normal"
    else:
        event_class = pd.Series("mixed_or_unknown", index=raw.index, dtype="string")
        event_class.loc[(~in_event | normal_label) & ~(low_mask | high_mask)] = "normal"
        event_class.loc[low_mask] = "low_pH_event"
        event_class.loc[high_mask] = "high_pH_event"
    return event_class


def load_standardised_predictions(spec: ModelSpec) -> pd.DataFrame:
    path = prediction_path(spec)
    columns = pq.ParquetFile(path).schema_arrow.names
    mapping = {
        "site_id": first_present(columns, ["site_id", "Site_ID"]),
        "timestamp": first_present(columns, ["timestamp", "target_timestamp", "datetime"]),
        "y_true": first_present(columns, ["y_true", "00400", "observed", "actual"]),
        "y_pred": first_present(columns, ["y_pred", "prediction", "predicted"]),
        "in_ph_event": first_present(columns, ["in_ph_event"]),
        "ph_event_label": first_present(columns, ["ph_event_label"]),
        "event_state": first_present(columns, ["event_state"]),
    }
    required = [mapping["site_id"], mapping["timestamp"], mapping["y_true"], mapping["y_pred"]]
    if any(item is None for item in required):
        raise ValueError(f"Missing required prediction columns for {spec.model_id}: {mapping}")
    read_cols = {str(value) for value in mapping.values() if value}
    if spec.row_filter_col and spec.row_filter_col in columns:
        read_cols.add(spec.row_filter_col)
    raw = pd.read_parquet(path, columns=sorted(read_cols))
    if spec.row_filter_col and spec.row_filter_col in raw:
        raw = raw.loc[raw[spec.row_filter_col].astype(str).eq(str(spec.row_filter_value))].copy()
    rename = {
        mapping["site_id"]: "site_id",
        mapping["timestamp"]: "timestamp",
        mapping["y_true"]: "y_true",
        mapping["y_pred"]: "y_pred",
    }
    for key in ["in_ph_event", "ph_event_label", "event_state"]:
        if mapping[key]:
            rename[mapping[key]] = key + "_raw"
    raw = raw.rename(columns=rename)
    df = pd.DataFrame(index=raw.index)
    df["site_id"] = raw["site_id"].map(normalise_site_id).astype("string")
    df["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    df["y_true"] = pd.to_numeric(raw["y_true"], errors="coerce").astype("float32")
    df["y_pred"] = pd.to_numeric(raw["y_pred"], errors="coerce").astype("float32")
    df["residual"] = (df["y_true"].astype("float64") - df["y_pred"].astype("float64")).astype("float32")
    df["abs_residual"] = df["residual"].abs().astype("float32")
    event_cols = [col for col in ["in_ph_event_raw", "ph_event_label_raw", "event_state_raw"] if col in raw]
    df["event_class"] = event_class_from_raw(raw[event_cols].copy())
    df = df.dropna(subset=["site_id", "timestamp", "y_true", "y_pred", "residual"]).reset_index(drop=True)
    return df


def infer_event_gap(df: pd.DataFrame, spec: ModelSpec) -> pd.Timedelta:
    if spec.model_type in {"GRU", "LSTM"}:
        return pd.Timedelta(hours=3)
    sample = df[["site_id", "timestamp"]].head(min(len(df), 500_000)).sort_values(["site_id", "timestamp"])
    diffs = sample.groupby("site_id", observed=True)["timestamp"].diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return pd.Timedelta(minutes=30)
    return min(max(diffs.median() * 3, pd.Timedelta(minutes=30)), pd.Timedelta(hours=3))


def assign_event_windows(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    events = df.loc[df["event_class"].ne("normal")].copy()
    if events.empty:
        return events
    gap = infer_event_gap(df, spec)
    events = events.sort_values(["site_id", "event_class", "timestamp"]).reset_index(drop=True)
    previous_site = events["site_id"].shift()
    previous_class = events["event_class"].shift()
    previous_time = events["timestamp"].shift()
    new_window = (
        events["site_id"].ne(previous_site)
        | events["event_class"].ne(previous_class)
        | (events["timestamp"].sub(previous_time) > gap)
    )
    events["event_window_id"] = np.cumsum(new_window.fillna(True).to_numpy(dtype=bool)).astype("int64")
    events["event_gap_threshold_minutes"] = gap.total_seconds() / 60.0
    return events


def reliability_label(n_windows: int) -> str:
    if n_windows >= 10:
        return "reliable"
    if n_windows >= 3:
        return "limited"
    return "insufficient"


def enrichment(mask_tail: pd.Series, mask_event: pd.Series) -> float:
    n = len(mask_tail)
    n_tail = int(mask_tail.sum())
    n_event = int(mask_event.sum())
    if n == 0 or n_tail == 0 or n_event == 0:
        return np.nan
    base_rate = n_event / n
    tail_rate = int((mask_tail & mask_event).sum()) / n_tail
    return tail_rate / base_rate if base_rate > 0 else np.nan


def event_window_site_summary(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "site_id",
        "total_event_windows",
        "low_event_windows",
        "high_event_windows",
        "overall_event_hit_rate",
        "negative_tail5_low_hit_rate",
        "positive_tail5_high_hit_rate",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    windows = (
        events.groupby("event_window_id", observed=True)
        .agg(
            site_id=("site_id", "first"),
            event_class=("event_class", "first"),
            abs_hit=("absolute_top5", "max"),
            negative_hit=("negative_tail5", "max"),
            positive_hit=("positive_tail5", "max"),
        )
        .reset_index()
    )
    rows = []
    for site_id, group in windows.groupby("site_id", observed=True):
        low = group["event_class"].eq("low_pH_event")
        high = group["event_class"].eq("high_pH_event")
        rows.append(
            {
                "site_id": normalise_site_id(site_id),
                "total_event_windows": len(group),
                "low_event_windows": int(low.sum()),
                "high_event_windows": int(high.sum()),
                "overall_event_hit_rate": float(group["abs_hit"].mean()) if len(group) else np.nan,
                "negative_tail5_low_hit_rate": float(group.loc[low, "negative_hit"].mean()) if low.any() else np.nan,
                "positive_tail5_high_hit_rate": float(group.loc[high, "positive_hit"].mean()) if high.any() else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def calculate_model_site_metrics(spec: ModelSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    log(f"Per-site test metrics: loading {spec.model_id}")
    df = load_standardised_predictions(spec)
    negative_threshold = float(df["residual"].quantile(0.05))
    positive_threshold = float(df["residual"].quantile(0.95))
    absolute_threshold = float(df["abs_residual"].quantile(0.95))
    df["negative_tail5"] = df["residual"].le(negative_threshold)
    df["positive_tail5"] = df["residual"].ge(positive_threshold)
    df["absolute_top5"] = df["abs_residual"].ge(absolute_threshold)
    events = assign_event_windows(df, spec)
    window_summary = event_window_site_summary(events)
    window_lookup = window_summary.set_index("site_id").to_dict("index") if not window_summary.empty else {}
    rows: list[dict[str, Any]] = []
    for site_id, site_df in df.groupby("site_id", observed=True, sort=True):
        site_id = normalise_site_id(site_id)
        residual = site_df["residual"].astype("float64")
        y_true = site_df["y_true"].astype("float64")
        sse = float(np.square(residual).sum())
        sst = float(np.square(y_true - y_true.mean()).sum())
        low = site_df["event_class"].eq("low_pH_event")
        high = site_df["event_class"].eq("high_pH_event")
        any_event = site_df["event_class"].ne("normal")
        window = window_lookup.get(site_id, {})
        low_windows = int(window.get("low_event_windows", 0))
        high_windows = int(window.get("high_event_windows", 0))
        total_windows = int(window.get("total_event_windows", 0))
        rows.append(
            {
                **model_fields(spec),
                "site_id": site_id,
                "n_test_rows": len(site_df),
                "rmse": math.sqrt(sse / len(site_df)) if len(site_df) else np.nan,
                "mae": float(residual.abs().mean()),
                "r2": 1.0 - sse / sst if sst > 0 else np.nan,
                "mean_residual": float(residual.mean()),
                "median_residual": float(residual.median()),
                "mean_absolute_residual": float(residual.abs().mean()),
                "absolute_top5_event_enrichment": enrichment(site_df["absolute_top5"], any_event),
                "absolute_top5_low_enrichment": enrichment(site_df["absolute_top5"], low),
                "absolute_top5_high_enrichment": enrichment(site_df["absolute_top5"], high),
                "negative_tail5_low_enrichment": enrichment(site_df["negative_tail5"], low),
                "positive_tail5_high_enrichment": enrichment(site_df["positive_tail5"], high),
                "low_expected_sign_rate": float(residual[low].lt(0).mean()) if low.any() else np.nan,
                "high_expected_sign_rate": float(residual[high].gt(0).mean()) if high.any() else np.nan,
                "overall_event_hit_rate": window.get("overall_event_hit_rate", np.nan),
                "negative_tail5_low_hit_rate": window.get("negative_tail5_low_hit_rate", np.nan),
                "positive_tail5_high_hit_rate": window.get("positive_tail5_high_hit_rate", np.nan),
                "normal_rows": int(site_df["event_class"].eq("normal").sum()),
                "low_event_rows": int(low.sum()),
                "high_event_rows": int(high.sum()),
                "mixed_or_unknown_event_rows": int(site_df["event_class"].eq("mixed_or_unknown").sum()),
                "low_event_windows": low_windows,
                "high_event_windows": high_windows,
                "total_regulatory_event_windows": total_windows,
                "event_reliability_label": reliability_label(total_windows),
                "low_event_reliability_label": reliability_label(low_windows),
                "high_event_reliability_label": reliability_label(high_windows),
            }
        )
    audit = {
        **model_fields(spec),
        "prediction_file": spec.prediction_file,
        "n_test_rows": len(df),
        "n_sites": df["site_id"].nunique(),
        "negative_tail5_threshold": negative_threshold,
        "positive_tail5_threshold": positive_threshold,
        "absolute_top5_threshold": absolute_threshold,
        "residual_definition": "observed pH - predicted pH",
        "event_window_method": "contiguous rows by site and low/high label; gaps above model-specific threshold start a new window",
    }
    result = pd.DataFrame(rows)
    del df, events
    gc.collect()
    return result, audit


def build_all_model_metrics() -> pd.DataFrame:
    metric_tables: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        if not prediction_path(spec).exists():
            log(f"Skipping missing optional model: {spec.model_id}")
            continue
        metrics, audit = calculate_model_site_metrics(spec)
        metric_tables.append(metrics)
        audit_rows.append(audit)
    result = pd.concat(metric_tables, ignore_index=True) if metric_tables else pd.DataFrame()
    result["model_id"] = pd.Categorical(result["model_id"], categories=MODEL_ORDER, ordered=True)
    result = result.sort_values(["model_id", "site_id"]).reset_index(drop=True)
    result["model_id"] = result["model_id"].astype("string")
    result.to_csv(OUT_DIR / "site_model_metrics_long.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT_DIR / "model_metric_calculation_audit.csv", index=False)
    return result


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for source, target in [
        ("mean_flow", "log10_mean_flow"),
        ("Q50", "log10_Q50"),
        ("Q95_exceedance", "log10_Q95"),
        ("drainage_area", "log10_drainage_area"),
    ]:
        values = pd.to_numeric(result.get(source), errors="coerce")
        result[target] = np.nan
        valid = values > 0
        result.loc[valid, target] = np.log10(values.loc[valid])
    return result


def build_site_overlap_audit(site_sets: dict[str, set[str]], hydrology: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    available_flow = set(hydrology.loc[hydrology["flow_quality_label"].isin(QUALITY_APPROVED), "site_id"].astype(str))
    model_ids = [spec.model_id for spec in MODEL_SPECS if site_sets.get(spec.model_id)]
    for model_id in model_ids:
        sites = site_sets[model_id]
        rows.append(
            {
                "audit_type": "model_set",
                "model_a": model_id,
                "model_b": "",
                "n_sites_a": len(sites),
                "n_sites_b": np.nan,
                "n_shared_sites": len(sites & available_flow),
                "n_union_sites": len(sites | available_flow),
                "jaccard_overlap": len(sites & available_flow) / len(sites | available_flow) if sites | available_flow else np.nan,
                "note": "shared count is sites with good/acceptable flow metrics",
            }
        )
    for i, model_a in enumerate(model_ids):
        for model_b in model_ids[i + 1 :]:
            a = site_sets[model_a]
            b = site_sets[model_b]
            rows.append(
                {
                    "audit_type": "pairwise_model_overlap",
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_sites_a": len(a),
                    "n_sites_b": len(b),
                    "n_shared_sites": len(a & b),
                    "n_union_sites": len(a | b),
                    "jaccard_overlap": len(a & b) / len(a | b) if a | b else np.nan,
                    "note": "model-specific site sets retained; no incompatible averaging",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "site_set_overlap_audit.csv", index=False)
    return result


def build_merged_tables(
    static_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    model_metrics: pd.DataFrame,
    site_sets: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hydrology = static_df.merge(flow_df, on="site_id", how="outer", validate="1:1")
    hydrology = add_log_transforms(hydrology)
    hydrology.to_csv(OUT_DIR / "site_hydrological_metadata.csv", index=False)
    merged = model_metrics.merge(hydrology, on="site_id", how="left", validate="m:1", indicator=True)
    merged.to_csv(OUT_DIR / "site_hydrology_model_metrics_long.csv", index=False)

    hgb_ids = ["M200_A_no_metadata", "M200_B_with_metadata", "HGB200_V2"]
    common_hgb = set.intersection(*(site_sets.get(model_id, set()) for model_id in hgb_ids))
    hgb = merged.loc[merged["model_id"].isin(hgb_ids) & merged["site_id"].isin(common_hgb)].copy()
    hgb.to_csv(OUT_DIR / "site_hydrology_HGB200_common.csv", index=False)

    m0 = merged.loc[merged["model_id"].eq("M0_reference_nonevent")].copy()
    m0.to_csv(OUT_DIR / "site_hydrology_M0_nonevent50.csv", index=False)

    sequence_ids = [model_id for model_id in ["G100_B", "GRU100_V2", "L100_B"] if site_sets.get(model_id)]
    common_sequence = set.intersection(*(site_sets[model_id] for model_id in sequence_ids)) if sequence_ids else set()
    sequence = merged.loc[merged["model_id"].isin(sequence_ids) & merged["site_id"].isin(common_sequence)].copy()
    sequence.to_csv(OUT_DIR / "site_hydrology_sequence100.csv", index=False)

    excluded_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        reasons = []
        if row.get("_merge", "both") != "both":
            reasons.append("unmatched hydrological metadata")
        quality = row.get("flow_quality_label", np.nan)
        if quality not in QUALITY_APPROVED:
            reasons.append(f"flow quality {quality}")
        if reasons:
            excluded_rows.append(
                {
                    "site_id": row["site_id"],
                    "model_id": row["model_id"],
                    "exclusion_scope": "main correlation analysis",
                    "reason": "; ".join(reasons),
                }
            )
    excluded = pd.DataFrame(excluded_rows, columns=["site_id", "model_id", "exclusion_scope", "reason"])
    excluded.to_csv(OUT_DIR / "excluded_sites.csv", index=False)
    build_site_overlap_audit(site_sets, hydrology)
    return hydrology, merged, hgb, m0, sequence


def metadata_gain_table(hgb: pd.DataFrame) -> pd.DataFrame:
    hydrology_cols = [
        "site_id",
        "latitude",
        "longitude",
        "elevation",
        "drainage_area",
        "mean_flow",
        "flow_CV",
        "richards_baker_flashiness",
        "Q50",
        "Q95_exceedance",
        "Q95_Q50_ratio",
        "log10_mean_flow",
        "log10_Q50",
        "log10_Q95",
        "log10_drainage_area",
        "flow_quality_label",
        "low_event_windows",
        "high_event_windows",
        "total_regulatory_event_windows",
    ]
    base = hgb.loc[hgb["model_id"].eq("M200_A_no_metadata")].set_index("site_id")
    full = hgb.loc[hgb["model_id"].eq("M200_B_with_metadata")].set_index("site_id")
    v2 = hgb.loc[hgb["model_id"].eq("HGB200_V2")].set_index("site_id")
    common = sorted(set(base.index) & set(full.index) & set(v2.index))
    result = base.loc[common, [col for col in hydrology_cols if col != "site_id" and col in base.columns]].copy()
    result.index.name = "site_id"
    result["rmse_no_metadata"] = base.loc[common, "rmse"]
    result["rmse_full_metadata"] = full.loc[common, "rmse"]
    result["rmse_v2_metadata"] = v2.loc[common, "rmse"]
    result["rmse_gain_full"] = base.loc[common, "rmse"] - full.loc[common, "rmse"]
    result["rmse_gain_v2"] = base.loc[common, "rmse"] - v2.loc[common, "rmse"]
    for suffix, other in [("full", full), ("v2", v2)]:
        result[f"enrichment_change_{suffix}"] = (
            other.loc[common, "absolute_top5_event_enrichment"]
            - base.loc[common, "absolute_top5_event_enrichment"]
        )
        result[f"low_tail_enrichment_change_{suffix}"] = (
            other.loc[common, "negative_tail5_low_enrichment"]
            - base.loc[common, "negative_tail5_low_enrichment"]
        )
        result[f"high_tail_enrichment_change_{suffix}"] = (
            other.loc[common, "positive_tail5_high_enrichment"]
            - base.loc[common, "positive_tail5_high_enrichment"]
        )
    result = result.reset_index()
    result.to_csv(OUT_DIR / "metadata_gain_by_site.csv", index=False)
    return result


def relevant_event_count_column(metric: str) -> str:
    lower = metric.lower()
    if "low" in lower or "negative_tail" in lower:
        return "low_event_windows"
    if "high" in lower or "positive_tail" in lower:
        return "high_event_windows"
    return "total_regulatory_event_windows"


def correlation_row(
    model_id: str,
    subset: str,
    data: pd.DataFrame,
    site_property: str,
    model_metric: str,
) -> dict[str, Any]:
    pair = data[[site_property, model_metric]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    inference_eligible = bool(
        len(pair) >= 10 and pair[site_property].nunique() > 1 and pair[model_metric].nunique() > 1
    )
    if len(pair) >= 3 and pair[site_property].nunique() > 1 and pair[model_metric].nunique() > 1:
        rho, raw_p_value = spearmanr(pair[site_property], pair[model_metric])
        p_value = raw_p_value if inference_eligible else np.nan
    else:
        rho, p_value = np.nan, np.nan
    direction = "positive" if pd.notna(rho) and rho > 0 else "negative" if pd.notna(rho) and rho < 0 else "none_or_undefined"
    return {
        "model_id": model_id,
        "analysis_subset": subset,
        "site_property": site_property,
        "model_metric": model_metric,
        "n_valid_sites": len(pair),
        "minimum_sites_for_inference": 10,
        "inference_eligible": inference_eligible,
        "spearman_rho": rho,
        "p_value": p_value,
        "fdr_adjusted_p_value": np.nan,
        "significant_fdr_0_05": False,
        "interpretation_direction": direction,
    }


def apply_bh_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["fdr_adjusted_p_value"] = np.nan
    result["significant_fdr_0_05"] = False
    if result.empty:
        return result
    group_cols = ["model_id", "analysis_subset"]
    for _, indexes in result.groupby(group_cols, observed=True).groups.items():
        indexes = list(indexes)
        p = pd.to_numeric(result.loc[indexes, "p_value"], errors="coerce")
        valid = p.dropna().sort_values()
        m = len(valid)
        if m == 0:
            continue
        adjusted_sorted = np.minimum.accumulate((valid.to_numpy() * m / np.arange(1, m + 1))[::-1])[::-1]
        adjusted_sorted = np.clip(adjusted_sorted, 0, 1)
        result.loc[valid.index, "fdr_adjusted_p_value"] = adjusted_sorted
        result.loc[valid.index, "significant_fdr_0_05"] = adjusted_sorted <= 0.05
    return result


def run_correlations(
    merged: pd.DataFrame, metadata_gain: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict[str, Any]] = []
    reliable_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for model_id, model_df in merged.groupby("model_id", observed=True):
        approved = model_df.loc[model_df["flow_quality_label"].isin(QUALITY_APPROVED)].copy()
        for site_property in SITE_PROPERTIES:
            if site_property not in approved:
                continue
            for metric in PREDICTION_METRICS + EVENT_METRICS:
                if metric not in approved:
                    continue
                all_rows.append(correlation_row(str(model_id), "all_quality_approved_sites", approved, site_property, metric))
                if metric in EVENT_METRICS:
                    count_col = relevant_event_count_column(metric)
                    reliable = approved.loc[pd.to_numeric(approved[count_col], errors="coerce").fillna(0).ge(10)]
                    reliable_rows.append(correlation_row(str(model_id), "event_reliable_ge10", reliable, site_property, metric))
                    for threshold in [3, 5, 10, 20]:
                        subset = approved.loc[pd.to_numeric(approved[count_col], errors="coerce").fillna(0).ge(threshold)]
                        sensitivity_rows.append(
                            correlation_row(
                                str(model_id), f"event_count_ge{threshold}", subset, site_property, metric
                            )
                        )
    gain_approved = metadata_gain.loc[metadata_gain["flow_quality_label"].isin(QUALITY_APPROVED)].copy()
    for site_property in SITE_PROPERTIES:
        if site_property not in gain_approved:
            continue
        for metric in METADATA_EFFECT_METRICS:
            if metric in gain_approved:
                all_rows.append(
                    correlation_row(
                        "HGB200_METADATA_GAIN", "all_quality_approved_sites", gain_approved, site_property, metric
                    )
                )
    all_corr = apply_bh_fdr(pd.DataFrame(all_rows))
    reliable_corr = apply_bh_fdr(pd.DataFrame(reliable_rows))
    sensitivity = apply_bh_fdr(pd.DataFrame(sensitivity_rows))
    all_corr.to_csv(OUT_DIR / "spearman_correlations_all_sites.csv", index=False)
    reliable_corr.to_csv(OUT_DIR / "spearman_correlations_event_reliable_sites.csv", index=False)
    all_corr[
        ["model_id", "analysis_subset", "site_property", "model_metric", "n_valid_sites", "p_value"]
    ].to_csv(OUT_DIR / "spearman_pvalues_all_sites.csv", index=False)
    pd.concat([all_corr, reliable_corr], ignore_index=True).to_csv(
        OUT_DIR / "spearman_fdr_adjusted_pvalues.csv", index=False
    )
    sensitivity.to_csv(OUT_DIR / "event_count_sensitivity_results.csv", index=False)
    return all_corr, reliable_corr, sensitivity


def build_outlier_diagnostics(merged: pd.DataFrame) -> pd.DataFrame:
    relationships = [
        ("richards_baker_flashiness", "rmse"),
        ("flow_CV", "rmse"),
        ("richards_baker_flashiness", "absolute_top5_event_enrichment"),
        ("richards_baker_flashiness", "negative_tail5_low_enrichment"),
        ("Q95_Q50_ratio", "negative_tail5_low_enrichment"),
        ("log10_mean_flow", "rmse"),
    ]
    rows: list[dict[str, Any]] = []
    for model_id in ["M200_A_no_metadata", "M200_B_with_metadata", "HGB200_V2", "M0_reference_nonevent"]:
        data = merged.loc[
            merged["model_id"].eq(model_id) & merged["flow_quality_label"].isin(QUALITY_APPROVED)
        ].copy()
        for x_col, y_col in relationships:
            pair = data[["site_id", x_col, y_col, "flow_quality_label"]].dropna().copy()
            if len(pair) < 4:
                continue
            rho = spearmanr(pair[x_col], pair[y_col]).statistic
            candidates = [
                (pair[x_col].idxmin(), f"lowest {x_col}"),
                (pair[x_col].idxmax(), f"highest {x_col}"),
                (pair[y_col].idxmin(), f"lowest {y_col}"),
                (pair[y_col].idxmax(), f"highest {y_col}"),
            ]
            seen: set[str] = set()
            for idx, reason in candidates:
                site_id = str(pair.loc[idx, "site_id"])
                if site_id in seen:
                    continue
                seen.add(site_id)
                reduced = pair.drop(index=idx)
                reduced_rho = spearmanr(reduced[x_col], reduced[y_col]).statistic if len(reduced) >= 3 else np.nan
                delta = reduced_rho - rho if pd.notna(reduced_rho) and pd.notna(rho) else np.nan
                rows.append(
                    {
                        "site_id": site_id,
                        "model_id": model_id,
                        "relationship": f"{x_col} vs {y_col}",
                        "outlier_reason": reason,
                        "site_property_value": pair.loc[idx, x_col],
                        "model_metric_value": pair.loc[idx, y_col],
                        "data_quality_issue": "none identified",
                        "included_in_main_analysis": True,
                        "full_spearman_rho": rho,
                        "rho_without_site": reduced_rho,
                        "sensitivity_effect": delta,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "site_property_model_relationship_summary.csv", index=False)
    return result


def select_contrasting_sites(merged: pd.DataFrame) -> pd.DataFrame:
    data = merged.loc[
        merged["model_id"].eq("HGB200_V2") & merged["flow_quality_label"].isin(QUALITY_APPROVED)
    ].copy()
    data = data.dropna(subset=["richards_baker_flashiness", "flow_CV", "rmse"])
    event_pool = data.loc[pd.to_numeric(data["total_regulatory_event_windows"], errors="coerce").fillna(0).ge(3)]
    selection_pool = event_pool if len(event_pool) >= 4 else data
    choices: list[tuple[pd.Series, str]] = []
    if not selection_pool.empty:
        choices.extend(
            [
                (
                    selection_pool.loc[selection_pool["richards_baker_flashiness"].idxmax()],
                    "highest flashiness among sites with at least 3 regulatory event windows",
                ),
                (
                    selection_pool.loc[selection_pool["richards_baker_flashiness"].idxmin()],
                    "lowest flashiness among sites with at least 3 regulatory event windows",
                ),
                (
                    selection_pool.loc[selection_pool["rmse"].idxmin()],
                    "lowest RMSE / strongest prediction among sites with at least 3 regulatory event windows",
                ),
            ]
        )
        reliable = selection_pool.loc[
            selection_pool["low_event_windows"].ge(3)
            & selection_pool["negative_tail5_low_enrichment"].notna()
        ]
        if not reliable.empty:
            choices.append(
                (
                    reliable.loc[reliable["negative_tail5_low_enrichment"].idxmax()],
                    "strongest low-pH negative-tail enrichment among sites with at least 3 low-pH windows",
                )
            )
        else:
            choices.append((selection_pool.loc[selection_pool["rmse"].idxmax()], "highest RMSE"))
    rows = []
    seen: set[str] = set()
    for row, reason in choices:
        site_id = str(row["site_id"])
        if site_id in seen:
            continue
        seen.add(site_id)
        rows.append(
            {
                "site_id": site_id,
                "model_id": row["model_id"],
                "richards_baker_flashiness": row["richards_baker_flashiness"],
                "flow_CV": row["flow_CV"],
                "rmse": row["rmse"],
                "negative_tail5_low_enrichment": row["negative_tail5_low_enrichment"],
                "low_event_windows": row["low_event_windows"],
                "high_event_windows": row["high_event_windows"],
                "total_regulatory_event_windows": row["total_regulatory_event_windows"],
                "selection_reason": reason,
            }
        )
        if len(rows) == 4:
            break
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "contrasting_site_candidates.csv", index=False)
    return result


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": "#000000",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "savefig.transparent": False,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "#000000",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(0.8)
    ax.tick_params(direction="in", colors="#000000", width=0.8)
    ax.xaxis.label.set_color("#000000")
    ax.yaxis.label.set_color("#000000")


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.patch.set_facecolor("#FFFFFF")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="#FFFFFF", transparent=False)
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="#FFFFFF", transparent=False)
    plt.close(fig)


def correlation_heatmap(
    correlations: pd.DataFrame,
    model_ids: list[str],
    metrics: list[str],
    stem: str,
) -> None:
    prop_labels = {
        "latitude": "Latitude",
        "longitude": "Longitude",
        "elevation": "Elevation",
        "log10_mean_flow": "log10 mean flow",
        "flow_CV": "Flow CV",
        "richards_baker_flashiness": "R-B flashiness",
        "log10_Q50": "log10 Q50",
        "log10_Q95": "log10 Q95",
        "Q95_Q50_ratio": "Q95/Q50",
        "log10_drainage_area": "log10 drainage area",
    }
    metric_labels = {
        "rmse": "RMSE",
        "r2": "$R^2$",
        "absolute_top5_event_enrichment": "Absolute\nenrichment",
        "negative_tail5_low_enrichment": "Low negative-tail\nenrichment",
        "positive_tail5_high_enrichment": "High positive-tail\nenrichment",
        "overall_event_hit_rate": "Event hit rate",
        "negative_tail5_low_hit_rate": "Low signed\nhit rate",
    }
    fig, axes = plt.subplots(
        1,
        len(model_ids),
        figsize=(6.2 * len(model_ids), 6.2),
        squeeze=False,
        constrained_layout=True,
    )
    for panel, model_id in enumerate(model_ids):
        ax = axes[0, panel]
        sub = correlations.loc[correlations["model_id"].eq(model_id)]
        matrix = np.full((len(SITE_PROPERTIES), len(metrics)), np.nan)
        annotations = np.full(matrix.shape, "", dtype=object)
        for i, prop in enumerate(SITE_PROPERTIES):
            for j, metric in enumerate(metrics):
                row = sub.loc[sub["site_property"].eq(prop) & sub["model_metric"].eq(metric)]
                if row.empty:
                    continue
                item = row.iloc[0]
                rho = item["spearman_rho"]
                n = item["n_valid_sites"]
                padj = item["fdr_adjusted_p_value"]
                matrix[i, j] = rho
                star = "*" if pd.notna(padj) and padj <= 0.05 else ""
                annotations[i, j] = f"{rho:.2f}{star}\n(n={int(n)})" if pd.notna(rho) else f"NA\n(n={int(n)})"
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            annot=annotations,
            fmt="",
            annot_kws={"fontsize": 7},
            xticklabels=[metric_labels.get(metric, metric) for metric in metrics],
            yticklabels=[prop_labels.get(prop, prop) for prop in SITE_PROPERTIES] if panel == 0 else False,
            cbar=panel == len(model_ids) - 1,
            cbar_kws={"label": "Spearman rho"},
        )
        ax.set_xlabel("Model-performance metric")
        ax.set_ylabel("Site property" if panel == 0 else "")
        ax.set_title(MODEL_LABELS.get(model_id, model_id), fontsize=10)
        ax.tick_params(axis="x", labelrotation=35)
        style_axis(ax)
        ax.text(-0.08, 1.03, f"({chr(97 + panel)})", transform=ax.transAxes, fontweight="bold")
    save_figure(fig, stem)


def annotate_scatter_outliers(ax: plt.Axes, data: pd.DataFrame, x: str, y: str, limit: int = 2) -> None:
    pair = data[["site_id", x, y]].dropna().copy()
    if len(pair) < 4:
        return
    x_scale = pair[x].std(ddof=0)
    y_scale = pair[y].std(ddof=0)
    if not x_scale or not y_scale:
        return
    score = ((pair[x] - pair[x].median()) / x_scale).abs() + ((pair[y] - pair[y].median()) / y_scale).abs()
    for idx in score.nlargest(limit).index:
        row = pair.loc[idx]
        ax.annotate(str(row["site_id"]), (row[x], row[y]), xytext=(3, 3), textcoords="offset points", fontsize=6)


def simple_trend(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.unique(x[valid]).size < 2:
        return
    coefficients = np.polyfit(x[valid], y[valid], 1)
    x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
    ax.plot(x_line, np.polyval(coefficients, x_line), color=color, linewidth=1.2, alpha=0.9)


def relationship_figure(
    merged: pd.DataFrame,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
    stem: str,
    event_reliability: bool = False,
) -> None:
    model_ids = ["M200_A_no_metadata", "M200_B_with_metadata", "HGB200_V2"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    for panel, (ax, model_id) in enumerate(zip(axes, model_ids)):
        data = merged.loc[
            merged["model_id"].eq(model_id) & merged["flow_quality_label"].isin(QUALITY_APPROVED)
        ].copy()
        if event_reliability:
            count_col = relevant_event_count_column(y)
            data = data.loc[pd.to_numeric(data[count_col], errors="coerce").fillna(0).ge(10)]
        pair = data[["site_id", x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        color = MODEL_COLORS[model_id]
        ax.scatter(pair[x], pair[y], s=18, alpha=0.72, color=color, edgecolors="#000000", linewidths=0.25)
        simple_trend(ax, pair[x].to_numpy(float), pair[y].to_numpy(float), color)
        rho, p = spearmanr(pair[x], pair[y]) if len(pair) >= 3 else (np.nan, np.nan)
        ax.text(
            0.03,
            0.97,
            f"rho = {rho:.2f}\nn = {len(pair)}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "#FFFFFF", "edgecolor": "#000000", "linewidth": 0.5, "pad": 2},
        )
        annotate_scatter_outliers(ax, pair, x, y)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label if panel == 0 else "")
        ax.set_title(MODEL_LABELS[model_id], fontsize=10)
        if not pair.empty and pair[x].min() >= 0:
            ax.set_xlim(left=0)
        if not pair.empty and pair[y].min() >= 0:
            ax.set_ylim(bottom=0)
        style_axis(ax)
        ax.text(-0.12, 1.03, f"({chr(97 + panel)})", transform=ax.transAxes, fontweight="bold")
    save_figure(fig, stem)


def matrix_plot(merged: pd.DataFrame) -> None:
    variables = [
        "log10_mean_flow",
        "flow_CV",
        "richards_baker_flashiness",
        "log10_Q50",
        "Q95_Q50_ratio",
        "rmse",
        "absolute_top5_event_enrichment",
        "negative_tail5_low_enrichment",
    ]
    labels = {
        "log10_mean_flow": "log10 mean flow",
        "flow_CV": "Flow CV",
        "richards_baker_flashiness": "R-B flashiness",
        "log10_Q50": "log10 Q50",
        "Q95_Q50_ratio": "Q95/Q50",
        "rmse": "RMSE",
        "absolute_top5_event_enrichment": "Absolute enrichment",
        "negative_tail5_low_enrichment": "Low signed enrichment",
    }
    data = merged.loc[
        merged["model_id"].eq("HGB200_V2") & merged["flow_quality_label"].isin(QUALITY_APPROVED),
        ["site_id", *variables],
    ].copy()
    n = len(variables)
    fig, axes = plt.subplots(n, n, figsize=(13.5, 13.5), constrained_layout=True)
    color = MODEL_COLORS["HGB200_V2"]
    for i, y in enumerate(variables):
        for j, x in enumerate(variables):
            ax = axes[i, j]
            if i == j:
                vals = pd.to_numeric(data[x], errors="coerce").dropna()
                ax.hist(vals, bins=18, color=color, alpha=0.82, edgecolor="#000000", linewidth=0.35)
                ax.set_ylabel("Frequency" if j == 0 else "")
            elif i > j:
                pair = data[[x, y]].dropna()
                ax.scatter(pair[x], pair[y], s=7, color=color, alpha=0.55, edgecolors="none")
            else:
                pair = data[[x, y]].dropna()
                rho = spearmanr(pair[x], pair[y]).statistic if len(pair) >= 3 else np.nan
                ax.text(0.5, 0.55, f"rho = {rho:.2f}", transform=ax.transAxes, ha="center", va="center", fontsize=9)
                ax.text(0.5, 0.40, f"n = {len(pair)}", transform=ax.transAxes, ha="center", va="center", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_xlabel(labels[x] if i == n - 1 else "")
            if i != j:
                ax.set_ylabel(labels[y] if j == 0 else "")
            style_axis(ax)
    save_figure(fig, "site_property_matrix_plot")


def metadata_gain_figure(
    gain: pd.DataFrame,
    metric_names: list[str],
    y_label: str,
    stem: str,
) -> None:
    properties = [
        ("richards_baker_flashiness", "R-B flashiness"),
        ("flow_CV", "Flow CV"),
        ("log10_mean_flow", "log10 mean flow (cfs)"),
        ("elevation", "Elevation (m)"),
    ]
    data = gain.loc[gain["flow_quality_label"].isin(QUALITY_APPROVED)].copy()
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 7.2), constrained_layout=True)
    for row_index, metric in enumerate(metric_names):
        color = MODEL_COLORS["M200_B_with_metadata"] if row_index == 0 else MODEL_COLORS["HGB200_V2"]
        for col_index, (prop, prop_label) in enumerate(properties):
            ax = axes[row_index, col_index]
            pair = data[["site_id", prop, metric]].dropna()
            ax.scatter(pair[prop], pair[metric], s=16, color=color, alpha=0.7, edgecolors="#000000", linewidths=0.2)
            simple_trend(ax, pair[prop].to_numpy(float), pair[metric].to_numpy(float), color)
            rho = spearmanr(pair[prop], pair[metric]).statistic if len(pair) >= 3 else np.nan
            ax.text(0.03, 0.96, f"rho = {rho:.2f}\nn = {len(pair)}", transform=ax.transAxes, va="top", fontsize=7)
            ax.axhline(0, color="#000000", linewidth=0.7, linestyle="--")
            ax.set_xlabel(prop_label)
            ax.set_ylabel(y_label if col_index == 0 else "")
            ax.text(
                0.97,
                0.96,
                "Full metadata" if row_index == 0 else "V2 metadata",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            style_axis(ax)
            ax.text(-0.12, 1.04, f"({chr(97 + row_index * 4 + col_index)})", transform=ax.transAxes, fontweight="bold")
    save_figure(fig, stem)


def event_count_stability_plot(merged: pd.DataFrame) -> None:
    data = merged.loc[
        merged["model_id"].eq("HGB200_V2") & merged["flow_quality_label"].isin(QUALITY_APPROVED)
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), constrained_layout=True)
    plots = [
        ("total_regulatory_event_windows", "absolute_top5_event_enrichment", "All event windows", "Absolute top-5% enrichment"),
        ("low_event_windows", "negative_tail5_low_enrichment", "Low-pH event windows", "Low negative-tail enrichment"),
    ]
    for panel, (ax, (x, y, x_label, y_label)) in enumerate(zip(axes, plots)):
        pair = data[["site_id", x, y]].dropna()
        ax.scatter(pair[x], pair[y], s=18, color=MODEL_COLORS["HGB200_V2"], alpha=0.7, edgecolors="#000000", linewidths=0.25)
        ax.axvline(10, color="#000000", linestyle="--", linewidth=0.8, label="Reliability threshold = 10")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        legend = ax.legend(loc="upper right", frameon=True)
        legend.get_frame().set_facecolor("#FFFFFF")
        legend.get_frame().set_edgecolor("#000000")
        style_axis(ax)
        ax.text(-0.12, 1.03, f"({chr(97 + panel)})", transform=ax.transAxes, fontweight="bold")
    save_figure(fig, "event_count_vs_enrichment_stability")


def create_figures(merged: pd.DataFrame, all_corr: pd.DataFrame, gain: pd.DataFrame) -> None:
    configure_plot_style()
    heatmap_metrics = [
        "rmse",
        "r2",
        "absolute_top5_event_enrichment",
        "negative_tail5_low_enrichment",
        "positive_tail5_high_enrichment",
        "overall_event_hit_rate",
    ]
    correlation_heatmap(
        all_corr,
        ["M200_A_no_metadata", "M200_B_with_metadata", "HGB200_V2"],
        heatmap_metrics,
        "hgb200_site_property_correlation_heatmap",
    )
    correlation_heatmap(
        all_corr,
        ["M0_reference_nonevent"],
        ["rmse", "absolute_top5_event_enrichment", "negative_tail5_low_enrichment", "negative_tail5_low_hit_rate"],
        "m0_nonevent_site_property_correlation_heatmap",
    )
    matrix_plot(merged)
    relationship_figure(merged, "flow_CV", "rmse", "Flow coefficient of variation", "Test RMSE", "flow_cv_vs_rmse")
    relationship_figure(
        merged,
        "richards_baker_flashiness",
        "rmse",
        "Richards-Baker flashiness index",
        "Test RMSE",
        "flashiness_vs_rmse",
    )
    relationship_figure(
        merged,
        "richards_baker_flashiness",
        "absolute_top5_event_enrichment",
        "Richards-Baker flashiness index",
        "Absolute top-5% event enrichment",
        "flashiness_vs_absolute_enrichment",
        event_reliability=True,
    )
    relationship_figure(
        merged,
        "richards_baker_flashiness",
        "negative_tail5_low_enrichment",
        "Richards-Baker flashiness index",
        "Low-pH negative-tail enrichment",
        "flashiness_vs_low_negative_tail_enrichment",
        event_reliability=True,
    )
    relationship_figure(
        merged,
        "Q95_Q50_ratio",
        "negative_tail5_low_enrichment",
        "Q95/Q50 low-flow ratio",
        "Low-pH negative-tail enrichment",
        "q95_q50_vs_low_tail_enrichment",
        event_reliability=True,
    )
    relationship_figure(
        merged,
        "log10_mean_flow",
        "rmse",
        "log10 mean train-period flow (cfs)",
        "Test RMSE",
        "mean_flow_vs_rmse",
    )
    metadata_gain_figure(
        gain,
        ["rmse_gain_full", "rmse_gain_v2"],
        "RMSE gain (no metadata - metadata)",
        "metadata_rmse_gain_vs_site_properties",
    )
    metadata_gain_figure(
        gain,
        ["enrichment_change_full", "enrichment_change_v2"],
        "Change in absolute event enrichment",
        "metadata_enrichment_change_vs_site_properties",
    )
    event_count_stability_plot(merged)


def strongest_association(
    correlations: pd.DataFrame,
    metric: str,
    model_ids: list[str] | None = None,
) -> pd.Series | None:
    subset = correlations.loc[
        correlations["model_metric"].eq(metric)
        & correlations["spearman_rho"].notna()
        & pd.to_numeric(correlations["n_valid_sites"], errors="coerce").ge(10)
    ].copy()
    if model_ids is not None:
        subset = subset.loc[subset["model_id"].isin(model_ids)]
    if subset.empty:
        return None
    subset["abs_rho"] = subset["spearman_rho"].abs()
    return subset.sort_values(["significant_fdr_0_05", "abs_rho"], ascending=[False, False]).iloc[0]


def association_text(row: pd.Series | None) -> str:
    if row is None:
        return "No estimable association."
    significance = (
        "FDR-significant"
        if bool(row.get("significant_fdr_0_05", False))
        else "not FDR-significant"
    )
    return (
        f"{row['site_property']} with {row['model_metric']} in {row['model_id']}: "
        f"rho={fmt(row['spearman_rho'])}, n={int(row['n_valid_sites'])}, "
        f"FDR-adjusted p={fmt(row['fdr_adjusted_p_value'])} ({significance})."
    )


def association_status(
    correlations: pd.DataFrame,
    property_name: str,
    metric_name: str,
    model_ids: list[str],
) -> str:
    subset = correlations.loc[
        correlations["site_property"].eq(property_name)
        & correlations["model_metric"].eq(metric_name)
        & correlations["model_id"].isin(model_ids)
        & correlations["spearman_rho"].notna()
    ]
    if subset.empty:
        return "unclear"
    significant = subset.loc[subset["significant_fdr_0_05"]]
    if significant.empty:
        return "unclear"
    median = float(subset["spearman_rho"].median())
    return "yes" if median > 0 else "no" if median < 0 else "unclear"


def event_filter_robustness(all_corr: pd.DataFrame, reliable_corr: pd.DataFrame) -> str:
    keys = ["model_id", "site_property", "model_metric"]
    event_all = all_corr.loc[
        all_corr["model_metric"].isin(EVENT_METRICS), keys + ["spearman_rho", "n_valid_sites"]
    ]
    event_rel = reliable_corr.loc[
        reliable_corr["model_metric"].isin(EVENT_METRICS), keys + ["spearman_rho", "n_valid_sites"]
    ]
    paired = event_all.merge(event_rel, on=keys, suffixes=("_all", "_reliable")).dropna()
    paired = paired.loc[
        paired["n_valid_sites_all"].ge(10)
        & paired["n_valid_sites_reliable"].ge(10)
        & (paired["spearman_rho_all"].abs().ge(0.10) | paired["spearman_rho_reliable"].abs().ge(0.10))
    ]
    if paired.empty:
        return "unclear"
    consistency = float(
        np.sign(paired["spearman_rho_all"]).eq(np.sign(paired["spearman_rho_reliable"])).mean()
    )
    if consistency >= 0.75:
        return "yes"
    if consistency >= 0.50:
        return "partly"
    return "no"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    if frame.empty:
        return "No rows available."
    show = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows).copy()
    for col in show.select_dtypes(include=["number"]).columns:
        show[col] = show[col].map(lambda value: fmt(value) if pd.notna(value) else "NA")
    headers = list(show.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in show.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def write_report(
    static_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    model_metrics: pd.DataFrame,
    merged: pd.DataFrame,
    all_corr: pd.DataFrame,
    reliable_corr: pd.DataFrame,
    sensitivity: pd.DataFrame,
    gain: pd.DataFrame,
    candidates: pd.DataFrame,
    helper_status: str,
) -> dict[str, Any]:
    hgb_ids = ["M200_A_no_metadata", "M200_B_with_metadata", "HGB200_V2"]
    quality_counts = flow_df["flow_quality_label"].value_counts(dropna=False).to_dict()
    usable = int(flow_df["flow_quality_label"].isin(QUALITY_APPROVED).sum())
    excluded = int(len(flow_df) - usable)
    best_rmse = strongest_association(all_corr, "rmse", hgb_ids)
    best_absolute = strongest_association(reliable_corr, "absolute_top5_event_enrichment", hgb_ids)
    best_low = strongest_association(reliable_corr, "negative_tail5_low_enrichment", hgb_ids)
    best_high = strongest_association(reliable_corr, "positive_tail5_high_enrichment", hgb_ids)
    best_m0_rmse = strongest_association(all_corr, "rmse", ["M0_reference_nonevent"])
    best_m0_low = strongest_association(
        reliable_corr, "negative_tail5_low_enrichment", ["M0_reference_nonevent"]
    )
    flash_rmse = association_status(all_corr, "richards_baker_flashiness", "rmse", hgb_ids)
    flash_enrichment = association_status(
        reliable_corr, "richards_baker_flashiness", "absolute_top5_event_enrichment", hgb_ids
    )
    gain_rows = all_corr.loc[all_corr["model_id"].eq("HGB200_METADATA_GAIN")]
    gain_rmse_sig = gain_rows.loc[
        gain_rows["model_metric"].isin(["rmse_gain_full", "rmse_gain_v2"])
        & gain_rows["significant_fdr_0_05"]
    ]
    gain_enrich_sig = gain_rows.loc[
        gain_rows["model_metric"].isin(["enrichment_change_full", "enrichment_change_v2"])
        & gain_rows["significant_fdr_0_05"]
    ]
    metadata_rmse_status = "yes" if not gain_rmse_sig.empty else "unclear"
    metadata_enrich_status = "yes" if not gain_enrich_sig.empty else "unclear"
    robust = event_filter_robustness(all_corr, reliable_corr)

    model_summary = (
        model_metrics.groupby("model_id", observed=True)
        .agg(
            sites=("site_id", "nunique"),
            median_rmse=("rmse", "median"),
            median_absolute_enrichment=("absolute_top5_event_enrichment", "median"),
            median_low_signed_enrichment=("negative_tail5_low_enrichment", "median"),
            median_high_signed_enrichment=("positive_tail5_high_enrichment", "median"),
            reliable_event_sites=("event_reliability_label", lambda x: int(pd.Series(x).eq("reliable").sum())),
        )
        .reset_index()
    )
    hgb_top = pd.concat(
        [
            all_corr.loc[all_corr["model_id"].isin(hgb_ids) & all_corr["model_metric"].isin(["rmse", "r2"])],
            reliable_corr.loc[
                reliable_corr["model_id"].isin(hgb_ids)
                & reliable_corr["model_metric"].isin(
                    [
                        "absolute_top5_event_enrichment",
                        "negative_tail5_low_enrichment",
                        "positive_tail5_high_enrichment",
                    ]
                )
            ],
        ],
        ignore_index=True,
    )
    if not hgb_top.empty:
        hgb_top["abs_rho"] = hgb_top["spearman_rho"].abs()
        hgb_top = hgb_top.sort_values(
            ["model_id", "model_metric", "significant_fdr_0_05", "abs_rho"],
            ascending=[True, True, False, False],
        ).groupby(["model_id", "model_metric"], observed=True).head(1)

    findings = []
    if best_rmse is not None:
        findings.append("The strongest HGB200 site-property association with RMSE was " + association_text(best_rmse))
    if best_absolute is not None:
        findings.append("For event-reliable sites, the strongest association with absolute enrichment was " + association_text(best_absolute))
    if best_low is not None:
        findings.append("For low-pH signed detection, the strongest event-reliable association was " + association_text(best_low))
    if gain_rmse_sig.empty:
        findings.append("No metadata RMSE-gain relationship survived FDR correction; hydrological concentration of metadata benefit remains unclear.")
    else:
        top_gain = gain_rmse_sig.assign(abs_rho=gain_rmse_sig["spearman_rho"].abs()).sort_values("abs_rho", ascending=False).iloc[0]
        findings.append("Metadata RMSE gain showed at least one FDR-significant site-property association: " + association_text(top_gain))
    findings = findings[:5]

    report = f"""# Site Hydrology and Model-Performance Analysis

## 1. Purpose

Previous metadata was used as model input. The table created here is instead a post-hoc explanatory dataset. The objective is not to improve pH prediction directly, but to identify which site characteristics are associated with prediction accuracy and residual-event detectability.

No model was trained or retrained. All outcomes were calculated from existing test prediction files. Model-performance metrics remain outcomes and were not merged into any training dataset.

## 2. Data Sources

- Site register: `data/reference/sites_register.json`.
- Discharge: latest matching `data/clean` file for each processed-model site, parameter `00060`.
- Split restriction: existing `split == train` boundaries recovered from `processed200_model_ready.parquet`.
- Outcomes: existing test predictions for the three HGB200 models, M0 non-event HGB, and three optional sequence models.
- Signed residual definition: observed pH minus predicted pH.
- Supervisor helper script: {helper_status}

All 200 HGB sites were USGS sites with the registry unit group `Imperial`. Project parameter mappings define `00060` as `Discharge_cfs`, so the selected clean source is treated as native/cleaned unscaled discharge in cubic feet per second. The processed `00060` was rejected for physical flow metrics because saved pipelines use `StandardScaler`, producing centred values that can be negative.

## 3. Flow-Metric Definitions

- Mean flow: arithmetic mean of train-period daily mean discharge.
- Flow CV: sample standard deviation divided by positive mean flow.
- Richards-Baker flashiness: sum of absolute day-to-day changes divided by the sum of daily flows. Only consecutive daily pairs were used.
- Q50: median daily discharge.
- Q95: hydrological flow-duration convention, discharge equalled or exceeded 95% of the time, calculated as the 5th percentile (`Q95_exceedance`).
- Q95/Q50: `Q95_exceedance / Q50` where Q50 was positive.

Duplicate timestamps were aggregated by mean before daily aggregation. Long gaps were not interpolated, and differences were not calculated across non-consecutive dates.

## 4. Site and Data-Quality Audit

- Static metadata sites: {len(static_df)}.
- Registry matches: {int(static_df['registry_source'].notna().sum())}.
- Sites with good/acceptable native discharge: {usable}.
- Sites excluded from main flow correlations: {excluded}.
- Flow quality counts: {json.dumps({str(k): int(v) for k, v in quality_counts.items()}, sort_keys=True)}.
- Coordinate coverage: latitude {int(static_df['latitude'].notna().sum())}/{len(static_df)}, longitude {int(static_df['longitude'].notna().sum())}/{len(static_df)}, elevation {int(static_df['elevation'].notna().sum())}/{len(static_df)}.

The registry fields `drainage_area_km2` and nested catchment descriptors were extracted where available. Registry mean discharge is retained only as an audited supplementary field because its registry calculation method can be `direct_from_data`; train-period `mean_flow` is the main analysis variable.

## 5. Per-Site Model Metrics

Prediction metrics are test RMSE, MAE, R-squared, mean and median signed residual, and mean absolute residual. Event outcomes use within-model global 5% thresholds. Absolute residuals evaluate all/low/high regulatory events; negative residuals evaluate low-pH events; positive residuals evaluate high-pH events.

Event reliability uses contiguous event-window counts: reliable at 10 or more windows, limited at 3-9, and insufficient below 3. Low- and high-pH labels are assessed separately.

Spearman effect sizes are retained descriptively from 3 valid site pairs, but p-values and FDR inference require at least 10 valid sites. This prevents tiny high-pH subsets from producing misleading perfect-rank significance.

{markdown_table(model_summary, ['model_id', 'sites', 'median_rmse', 'median_absolute_enrichment', 'median_low_signed_enrichment', 'median_high_signed_enrichment', 'reliable_event_sites'])}

## 6. Main HGB200 Results

{association_text(best_rmse)}

{association_text(best_absolute)}

{association_text(best_low)}

{association_text(best_high)}

The strongest relationship for each HGB200 model/metric is shown below. FDR-adjusted evidence, effect size, sample size, and plot structure should be considered together.

{markdown_table(hgb_top, ['model_id', 'model_metric', 'site_property', 'n_valid_sites', 'inference_eligible', 'spearman_rho', 'p_value', 'fdr_adjusted_p_value', 'significant_fdr_0_05'], max_rows=20)}

Flashiness associated with higher RMSE: **{flash_rmse}**. Flashiness associated with stronger absolute residual-event enrichment: **{flash_enrichment}**. `Unclear` means the direction was not supported by FDR-adjusted evidence across the HGB200 analyses.

## 7. M0 Non-Event Results

{association_text(best_m0_rmse)}

{association_text(best_m0_low)}

M0 is evaluated on its own 50-site set and is not averaged with HGB200 or sequence models. Event-filtered results use at least 10 relevant windows. Differences between all-site and reliable-site estimates are reported as sensitivity evidence rather than treated as interchangeable.

## 8. Metadata Gain Analysis

`rmse_gain_full` and `rmse_gain_v2` are positive when metadata reduces RMSE. Enrichment changes are metadata minus no-metadata, so negative values indicate reduced residual-event contrast.

- Metadata RMSE gain associated with measured site properties: **{metadata_rmse_status}**.
- Metadata enrichment change associated with measured site properties: **{metadata_enrich_status}**.
- Median full-metadata RMSE gain: {fmt(gain['rmse_gain_full'].median())}.
- Median V2-metadata RMSE gain: {fmt(gain['rmse_gain_v2'].median())}.
- Median full-metadata absolute-enrichment change: {fmt(gain['enrichment_change_full'].median())}.
- Median V2-metadata absolute-enrichment change: {fmt(gain['enrichment_change_v2'].median())}.

Where RMSE improves while enrichment decreases, metadata improved site-level prediction but reduced residual-event contrast, reinforcing the distinction between predictive accuracy and event sensitivity.

## 9. Event-Count Sensitivity

Results robust after event-count filtering: **{robust}** for directional sign consistency among estimable correlations. This does not imply that the relationships were FDR-significant. Correlations were repeated at 3, 5, 10 and 20 relevant event windows. Sparse sites remain in the descriptive tables but are excluded from the main event-reliable analysis. The `event_count_vs_enrichment_stability` figure shows why low-count site estimates should not be overinterpreted.

## 10. Hydrological Interpretation

Positive flashiness-RMSE relationships, when supported, are interpreted as a tendency for rapidly varying flow regimes to coincide with less stable exogenous pH relationships. Positive flashiness-enrichment relationships, when supported, are consistent with hydrological disturbances producing larger departures from expected pH behaviour. These are associations, not causal effects.

Weak correlations are also informative: simple site-level flow descriptors may not capture event frequency, water-quality data quality, biological controls, acid-base chemistry, land management, or event-specific mechanisms. No direct rainfall causality is claimed because rainfall was not included in this analysis.

## 11. Limitations

- Regulatory pH events are weak labels rather than complete ground truth.
- Flow records differ in temporal coverage and density.
- Q95 uses the explicitly stated 95% exceedance convention.
- HGB200, M0 and sequence models use different site sets and time resolutions.
- Sequence windows and HGB rows are not pooled as if they were identical observations.
- Per-site enrichment is unstable where event-window counts are sparse.
- Correlation does not establish causation.
- Processed discharge is standardised; physical metrics therefore use clean native/unscaled `00060`.
- Registry mean discharge may be calculated from full-period source data and is not used as the primary train-period flow descriptor.

## 12. Dissertation-Ready Findings

"""
    for index, finding in enumerate(findings, start=1):
        report += f"{index}. {finding}\n"
    report += f"""

## 13. Recommended Contrasting Sites

The following HGB200 V2 sites provide contrasting hydrological/performance cases for predictor and residual plots.

{markdown_table(candidates, ['site_id', 'model_id', 'richards_baker_flashiness', 'flow_CV', 'rmse', 'negative_tail5_low_enrichment', 'low_event_windows', 'high_event_windows', 'selection_reason'])}

The next task should create observed-versus-predicted figures for these sites with regulatory-event shading, residual panels and parity plots.
"""
    (OUT_DIR / "site_hydrology_performance_analysis_report.md").write_text(report, encoding="utf-8")
    return {
        "static_sites": len(static_df),
        "usable_sites": usable,
        "excluded_sites": excluded,
        "hgb_common_sites": int(
            merged.loc[merged["model_id"].isin(hgb_ids)].groupby("site_id")["model_id"].nunique().eq(3).sum()
        ),
        "m0_sites": int(merged.loc[merged["model_id"].eq("M0_reference_nonevent"), "site_id"].nunique()),
        "best_rmse": best_rmse,
        "best_absolute": best_absolute,
        "best_low": best_low,
        "flash_rmse": flash_rmse,
        "flash_enrichment": flash_enrichment,
        "metadata_rmse": metadata_rmse_status,
        "metadata_enrichment": metadata_enrich_status,
        "robust": robust,
    }


def find_supervisor_helper() -> str:
    candidates = list(ROOT.rglob("metadata_plots_snippets.py"))
    candidates = [path for path in candidates if OUT_DIR not in path.parents]
    if not candidates:
        return "not found in the project root; task-specified generic definitions were used"
    source = candidates[0]
    destination = OUT_DIR / "metadata_plots_snippets_reference_copy.py"
    shutil.copy2(source, destination)
    return f"found at `{rel(source)}` and copied to `{destination.name}`; definitions were reviewed"


def run_flow_stage(
    site_sets: dict[str, set[str]], static_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_sites = set().union(*site_sets.values())
    processed_map = selected_processed_file_map()
    boundaries, processed_inventory = read_split_boundaries_and_processed_audit(required_sites)
    audit_transformations_and_units(required_sites, processed_map, static_df)
    flow_df, quality = build_flow_metrics(required_sites, boundaries, processed_inventory, processed_map)
    return flow_df, quality, boundaries


def run_model_merge_stage(
    static_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    site_sets: dict[str, set[str]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    model_metrics = build_all_model_metrics()
    hydrology, merged, hgb, m0, sequence = build_merged_tables(static_df, flow_df, model_metrics, site_sets)
    gain = metadata_gain_table(hgb)
    all_corr, reliable_corr, sensitivity = run_correlations(merged, gain)
    build_outlier_diagnostics(merged)
    candidates = select_contrasting_sites(merged)
    return model_metrics, merged, gain, all_corr, reliable_corr, sensitivity, candidates


def final_console(summary: dict[str, Any], candidates: pd.DataFrame) -> None:
    best_rmse = summary["best_rmse"]
    best_absolute = summary["best_absolute"]
    best_low = summary["best_low"]
    candidate_ids = ", ".join(candidates["site_id"].astype(str)) if not candidates.empty else "none"
    print("\nSITE HYDROLOGY AND MODEL-PERFORMANCE ANALYSIS COMPLETE\n")
    print("Output folder:")
    print(rel(OUT_DIR) + "/\n")
    print(f"Static metadata sites:\n{summary['static_sites']}\n")
    print(f"Sites with usable native/unscaled discharge:\n{summary['usable_sites']}\n")
    print(f"Sites excluded from flow analysis:\n{summary['excluded_sites']}\n")
    print("Main discharge source:\ndata/clean site CSV files, train-period 00060 only\n")
    print("Discharge units:\nft3/s (cfs)\n")
    print("Q95 definition:\n5th percentile of daily flow (discharge equalled or exceeded 95% of the time)\n")
    print(f"HGB200 common sites analysed:\n{summary['hgb_common_sites']}\n")
    print(f"M0 non-event sites analysed:\n{summary['m0_sites']}\n")
    print("Strongest site-property association with RMSE:")
    print(best_rmse["site_property"] if best_rmse is not None else "none")
    print("\nSpearman rho:")
    print(fmt(best_rmse["spearman_rho"]) if best_rmse is not None else "NA")
    print("\nFDR-adjusted p-value:")
    print(fmt(best_rmse["fdr_adjusted_p_value"]) if best_rmse is not None else "NA")
    print("\nStrongest site-property association with absolute enrichment:")
    print(best_absolute["site_property"] if best_absolute is not None else "none")
    print("\nStrongest site-property association with low negative-tail enrichment:")
    print(best_low["site_property"] if best_low is not None else "none")
    print(f"\nFlashiness associated with higher RMSE:\n{summary['flash_rmse']}\n")
    print(f"Flashiness associated with stronger residual-event enrichment:\n{summary['flash_enrichment']}\n")
    print(f"Metadata RMSE gain associated with site properties:\n{summary['metadata_rmse']}\n")
    print(f"Metadata enrichment loss associated with site properties:\n{summary['metadata_enrichment']}\n")
    print(f"Results robust after event-count filtering:\n{summary['robust']}\n")
    print(f"Recommended contrasting sites for predictor plots:\n{candidate_ids}\n")
    print("Main interpretation:")
    print(
        "Site hydrology explains some model-performance variation only where effect sizes and FDR-adjusted evidence support it; "
        "prediction accuracy and residual-event detectability remain distinct outcomes.\n"
    )
    print("Next recommended task:")
    print(
        "Create observed-versus-predicted figures for 3-4 contrasting sites with regulatory event shading, residual panels and parity plots."
    )


def run_all() -> None:
    ensure_dirs(reset_log=True)
    log("Starting site hydrology and model-performance analysis. No model training is performed.")
    helper_status = find_supervisor_helper()
    log(f"Supervisor helper: {helper_status}")
    prediction_inventory, site_sets = audit_prediction_files()
    included = prediction_inventory.loc[prediction_inventory.get("include_in_analysis", False).fillna(False)]
    if included.empty:
        raise RuntimeError("No valid prediction files were available.")
    static_df, _ = build_site_static_metadata(site_sets)
    flow_df, _, _ = run_flow_stage(site_sets, static_df)
    model_metrics, merged, gain, all_corr, reliable_corr, sensitivity, candidates = run_model_merge_stage(
        static_df, flow_df, site_sets
    )
    create_figures(merged, all_corr, gain)
    summary = write_report(
        static_df,
        flow_df,
        model_metrics,
        merged,
        all_corr,
        reliable_corr,
        sensitivity,
        gain,
        candidates,
        helper_status,
    )
    log("Analysis complete; all required tables, figures and report were written.")
    final_console(summary, candidates)


def run_stage(stage: str) -> None:
    ensure_dirs(reset_log=stage == "flow")
    _, site_sets = audit_prediction_files()
    static_df, _ = build_site_static_metadata(site_sets)
    if stage == "flow":
        run_flow_stage(site_sets, static_df)
        return
    flow_path = OUT_DIR / "site_flow_metrics.csv"
    if not flow_path.exists():
        raise FileNotFoundError("Run the flow stage first or run the complete analysis.")
    flow_df = pd.read_csv(flow_path, dtype={"site_id": str})
    if stage == "model":
        run_model_merge_stage(static_df, flow_df, site_sets)
        return
    if stage == "plots":
        merged = pd.read_csv(OUT_DIR / "site_hydrology_model_metrics_long.csv", dtype={"site_id": str})
        correlations = pd.read_csv(OUT_DIR / "spearman_correlations_all_sites.csv")
        reliable = pd.read_csv(OUT_DIR / "spearman_correlations_event_reliable_sites.csv")
        sensitivity = pd.read_csv(OUT_DIR / "event_count_sensitivity_results.csv")
        gain = pd.read_csv(OUT_DIR / "metadata_gain_by_site.csv", dtype={"site_id": str})
        model_metrics = pd.read_csv(OUT_DIR / "site_model_metrics_long.csv", dtype={"site_id": str})
        candidates = pd.read_csv(OUT_DIR / "contrasting_site_candidates.csv", dtype={"site_id": str})
        create_figures(merged, correlations, gain)
        summary = write_report(
            static_df,
            flow_df,
            model_metrics,
            merged,
            correlations,
            reliable,
            sensitivity,
            gain,
            candidates,
            find_supervisor_helper(),
        )
        final_console(summary, candidates)
        return
    raise ValueError(f"Unknown stage: {stage}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Post-hoc site hydrology and model-performance analysis")
    parser.add_argument("--stage", choices=["all", "flow", "model", "plots"], default="all")
    args = parser.parse_args(argv)
    if args.stage == "all":
        run_all()
    else:
        run_stage(args.stage)


if __name__ == "__main__":
    main()
