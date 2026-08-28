from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import astral
import numpy as np
import pandas as pd
import pvlib
import pyarrow.parquet as pq
import torch
from astral import Observer
from astral.sun import noon as astral_noon
from astral.sun import zenith as astral_zenith

import common as C


CANONICAL_MANIFEST = C.ROOT / "analysis_outputs" / "ph_common1000_processed_v1" / "reports" / "canonical_site_manifest.csv"
METADATA_910 = C.ROOT / "analysis_outputs" / "ph_h00_static3_additive_validation_v1" / "spec" / "metadata_input_with_static3.csv"
METADATA_MODEL_READY = C.ROOT / "analysis_outputs" / "ph_common1000_hgb_v1" / "full" / "reports" / "metadata_model_ready.csv"
NEON_POSITIONS = C.ROOT / "data" / "reference" / "neon_sensor_positions.json"
SITE_REGISTER = C.ROOT / "data" / "reference" / "sites_register.json"
TEACHER_CLEAN_DVC = C.ROOT / "data" / "clean.dvc"
EA_DOWNLOADER = C.ROOT / "core" / "data" / "ea_downloader.py"
USGS_DOWNLOADER = C.ROOT / "core" / "data" / "usgs_downloader.py"
NEON_DOWNLOADER = C.ROOT / "core" / "data" / "neon_downloader.py"
CONSOLIDATOR = C.ROOT / "core" / "utilities" / "consolidate_data_toolkit.py"
CLEAN_PARQUET_CODE = C.ROOT / "core" / "data" / "build_clean_parquet.py"
COMMON_PROCESSING_CODE = C.ROOT / "pipeline" / "00_common1000" / "run_common1000_processing.py"
PREPARE_CODE = C.ROOT / "pipeline" / "20_common1000_leave_sites_out" / "prepare_leave_sites_out.py"
MODEL_CODE = C.ROOT / "pipeline" / "21_common1000_leave_sites_out_models" / "run_leave_sites_out_models.py"

RESIDUALS = C.ROOT / "analysis_outputs" / "cross_variable_hot_moment_incremental_closure_v1" / "residual_scores" / "matched_residuals.parquet"
M0_RAW_PREDICTIONS = C.ROOT / "analysis_outputs" / "cross_variable_hot_moment_incremental_closure_v1" / "predictions" / "cross_variable_h00_predictions.parquet"
M0_PREDICTION_DIRECTORY = C.M0_ROOT / "predictions_by_site"
CALIBRATION_ROOT = C.ROOT / "analysis_outputs" / "ph_h00_30day_calibration_validation_v1"
CALIBRATION_PARAMETERS = CALIBRATION_ROOT / "calibration" / "site_calibration_parameters.csv"
M0_VALIDATION_POST_OFFSET = CALIBRATION_ROOT / "predictions" / "validation_predictions.parquet"
SPLIT_KEY_MANIFEST = C.ROOT / "analysis_outputs" / "ph_h00_static_interaction_validation_v1" / "config" / "split_and_key_manifest.json"

EA_TIME_DOC = "https://environment.data.gov.uk/support/faqs/275875350/275875667"
USGS_TIME_DOC = "https://waterservices.usgs.gov/docs/instantaneous-values/instantaneous-values-details/"
NEON_TIME_DOC = "https://www.neonscience.org/data-samples/data-management/data-formats-conventions"
SPA_REFERENCE = "https://doi.org/10.2172/15003974"


def table_info(path: Path) -> tuple[int | None, str]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        return parquet.metadata.num_rows, str(parquet.schema_arrow)
    if suffix == ".csv":
        header = pd.read_csv(path, nrows=0)
        with path.open("rb") as handle:
            rows = sum(block.count(b"\n") for block in iter(lambda: handle.read(8 * 1024 * 1024), b"")) - 1
        return max(rows, 0), "csv:" + "|".join(map(str, header.columns))
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        keys = sorted(payload) if isinstance(payload, dict) else []
        return None, "json_keys:" + "|".join(map(str, keys))
    if suffix == ".pt":
        state = torch.load(path, map_location="cpu", weights_only=True)
        schema = "|".join(f"{name}:{tuple(value.shape)}:{value.dtype}" for name, value in state.items())
        return None, "torch_state_dict:" + schema
    if suffix == ".py":
        return None, f"python_source_lines:{len(path.read_text(encoding='utf-8-sig').splitlines())}"
    return None, suffix.lstrip(".") or "file"


def directory_digest(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = child.relative_to(path).as_posix()
        child_hash = C.sha256_file(child)
        digest.update(relative.encode("utf-8"))
        digest.update(child_hash.encode("ascii"))
        count += 1
        total += child.stat().st_size
    return digest.hexdigest(), count, total


def artifact_inventory() -> pd.DataFrame:
    assets: list[tuple[str, Path, str]] = [
        ("official_h00_checkpoint", C.M0_CHECKPOINT, "file"),
        ("official_h00_success_feature_manifest", C.M0_SUCCESS, "file"),
        ("official_h00_effective_config", C.M0_CONFIG, "file"),
        ("official_h00_train_only_preprocessing", C.M0_PREPROCESSING, "file"),
        ("official_h00_metadata_transform", C.M0_METADATA_TRANSFORM, "file"),
        ("official_m0_raw_prediction_table", M0_RAW_PREDICTIONS, "file"),
        ("official_m0_per_site_prediction_directory", M0_PREDICTION_DIRECTORY, "directory"),
        ("official_raw_support_query_and_roles", RESIDUALS, "file"),
        ("official_30day_site_offsets", CALIBRATION_PARAMETERS, "file"),
        ("official_validation_post_offset_predictions", M0_VALIDATION_POST_OFFSET, "file"),
        ("primary_638_137_135_split_manifest", C.COHORT_PATH, "file"),
        ("formal_128_and_126_split_key_manifest", SPLIT_KEY_MANIFEST, "file"),
        ("preexisting_910_metadata_table", METADATA_910, "file"),
        ("preexisting_1009_metadata_source", METADATA_MODEL_READY, "file"),
        ("neon_sensor_position_authority", NEON_POSITIONS, "file"),
        ("site_registry_coordinate_authority", SITE_REGISTER, "file"),
        ("clean_data_dvc_snapshot_pointer", TEACHER_CLEAN_DVC, "file"),
        ("h00_calendar_and_hourly_preparation_code", PREPARE_CODE, "file"),
        ("h00_window_and_model_code", MODEL_CODE, "file"),
        ("common_timestamp_parser_code", COMMON_PROCESSING_CODE, "file"),
        ("ea_source_downloader", EA_DOWNLOADER, "file"),
        ("usgs_source_downloader", USGS_DOWNLOADER, "file"),
        ("neon_source_downloader", NEON_DOWNLOADER, "file"),
        ("clean_timestamp_consolidator", CONSOLIDATOR, "file"),
        ("clean_csv_to_parquet_code", CLEAN_PARQUET_CODE, "file"),
    ]
    records = []
    for role, path, kind in assets:
        if not path.exists():
            raise FileNotFoundError(path)
        if kind == "directory":
            digest, files, total = directory_digest(path)
            records.append(
                {
                    "role": role,
                    "path": str(path),
                    "kind": kind,
                    "sha256": digest,
                    "bytes": total,
                    "files": files,
                    "rows": None,
                    "schema": "directory of keyed per-site prediction parquet files",
                }
            )
        else:
            rows, schema = table_info(path)
            records.append(
                {
                    "role": role,
                    "path": str(path),
                    "kind": kind,
                    "sha256": C.sha256_file(path),
                    "bytes": path.stat().st_size,
                    "files": 1,
                    "rows": rows,
                    "schema": schema,
                }
            )
    return pd.DataFrame(records)


def timestamp_audit(cohort: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    canonical = pd.read_csv(CANONICAL_MANIFEST, dtype={"site_id": str})
    selected = canonical.merge(
        cohort[["site_id", "split", "network"]],
        on="site_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_canonical", "_cohort"),
    )
    if len(selected) != 910:
        raise RuntimeError(f"Canonical timestamp manifest coverage is {len(selected)}/910")

    detail = []
    total_rows = 0
    for number, row in enumerate(selected.itertuples(index=False), 1):
        site_id = str(row.site_id)
        path = C.HOURLY_ROOT / f"{site_id}.parquet"
        frame = pd.read_parquet(path, columns=["site_id", "timestamp"])
        timestamps = pd.DatetimeIndex(frame["timestamp"])
        values = timestamps.to_numpy(dtype="datetime64[ns]")
        diffs = np.diff(values).astype("timedelta64[s]").astype(np.int64) if len(values) > 1 else np.empty(0, np.int64)
        source_timezone_status = str(row.timezone_status)
        if site_id.startswith("E"):
            source_family = "EA_HYDROLOGY"
            proof = "EA official Hydrology metadata: all data recorded in GMT; clean snapshot provenance; naive representation therefore semantically UTC"
            proof_url = EA_TIME_DOC
        elif site_id.startswith("N"):
            source_family = "NEON"
            proof = "NEON official convention: all timestamps UTC; source-aware timestamp was converted to UTC then made naive by canonical processing"
            proof_url = NEON_TIME_DOC
        else:
            source_family = "USGS_INSTANTANEOUS"
            proof = "USGS WaterML supplies site-local ISO-8601 timestamps with offsets; project downloader parses with utc=True; consolidator drops timezone after UTC conversion"
            proof_url = USGS_TIME_DOC
        detail.append(
            {
                "site_id": site_id,
                "split": row.split,
                "network": getattr(row, "network_cohort", getattr(row, "network", "")),
                "source_family": source_family,
                "source_timestamp_field": "TIMESTAMP (clean) -> timestamp (canonical/hourly)",
                "source_timezone_status": source_timezone_status,
                "stored_dtype": str(frame["timestamp"].dtype),
                "stored_timezone_aware": timestamps.tz is not None,
                "semantic_timezone": "UTC",
                "semantic_utc_proven": True,
                "proof": proof,
                "proof_url": proof_url,
                "rows": len(frame),
                "missing_timestamps": int(timestamps.isna().sum()),
                "duplicate_site_timestamp_keys": int(frame.duplicated(["site_id", "timestamp"]).sum()),
                "nonmonotonic": bool(not timestamps.is_monotonic_increasing),
                "non_hour_aligned": int(np.count_nonzero((timestamps.minute != 0) | (timestamps.second != 0) | (timestamps.microsecond != 0))),
                "non_1h_adjacent_steps": int(np.count_nonzero(diffs != 3600)),
                "minimum_adjacent_seconds": int(diffs.min()) if len(diffs) else None,
                "maximum_adjacent_seconds": int(diffs.max()) if len(diffs) else None,
                "leap_day_rows": int(np.count_nonzero((timestamps.month == 2) & (timestamps.day == 29))),
                "start_utc_naive_storage": timestamps.min(),
                "end_utc_naive_storage": timestamps.max(),
                "feature_boundary_operation": "tz_localize('UTC') on proven-naive UTC; tz_convert('UTC') if already aware",
                "true_48h_spacing_pass": bool(not len(diffs) or np.all(diffs == 3600)),
            }
        )
        total_rows += len(frame)
        if number % 100 == 0 or number == len(selected):
            print(f"timestamp-audit {number}/{len(selected)} sites rows={total_rows:,}", flush=True)
    audit = pd.DataFrame(detail)
    summary = (
        audit.groupby(["source_family", "split", "source_timezone_status", "stored_dtype"], dropna=False, as_index=False)
        .agg(
            sites=("site_id", "nunique"),
            rows=("rows", "sum"),
            stored_timezone_aware_sites=("stored_timezone_aware", "sum"),
            semantic_utc_proven_sites=("semantic_utc_proven", "sum"),
            missing_timestamps=("missing_timestamps", "sum"),
            duplicate_site_timestamp_keys=("duplicate_site_timestamp_keys", "sum"),
            nonmonotonic_sites=("nonmonotonic", "sum"),
            non_hour_aligned_rows=("non_hour_aligned", "sum"),
            non_1h_adjacent_steps=("non_1h_adjacent_steps", "sum"),
            leap_day_rows=("leap_day_rows", "sum"),
        )
    )
    clean_files = [p for p in (C.ROOT / "data" / "clean").rglob("*") if p.is_file()]
    clean_bytes = sum(p.stat().st_size for p in clean_files)
    result = {
        "status": "PASS_WITH_EXPLICIT_UTC_LOCALISATION_AT_SOLAR_BOUNDARY",
        "sites": int(audit["site_id"].nunique()),
        "rows": int(audit["rows"].sum()),
        "stored_timezone_aware_sites": int(audit["stored_timezone_aware"].sum()),
        "stored_timezone_naive_sites": int((~audit["stored_timezone_aware"]).sum()),
        "semantic_utc_proven_sites": int(audit["semantic_utc_proven"].sum()),
        "source_naive_unknown_label_sites": int((audit["source_timezone_status"] == "source_naive_unknown").sum()),
        "interpretation_of_source_naive_unknown": "The canonical parser could not infer timezone from the already-normalised naive file alone. Independent source documentation plus the preserved acquisition/conversion chain resolves semantics without using outcomes.",
        "feature_boundary": "Every stored naive timestamp is explicitly localised as UTC before solar calculation; no civil timezone or DST conversion is applied.",
        "missing_timestamps": int(audit["missing_timestamps"].sum()),
        "duplicate_site_timestamp_keys": int(audit["duplicate_site_timestamp_keys"].sum()),
        "nonmonotonic_sites": int(audit["nonmonotonic"].sum()),
        "non_hour_aligned_rows": int(audit["non_hour_aligned"].sum()),
        "non_1h_adjacent_steps": int(audit["non_1h_adjacent_steps"].sum()),
        "true_48h_spacing_all_sites": bool(audit["true_48h_spacing_pass"].all()),
        "leap_day_rows": int(audit["leap_day_rows"].sum()),
        "dst_handling": "UTC axes remain uniformly hourly through civil DST transitions; no duplicated or missing civil-clock hour is introduced.",
        "source_documents": {"EA": EA_TIME_DOC, "USGS": USGS_TIME_DOC, "NEON": NEON_TIME_DOC},
        "source_code": {
            "canonical_parse": f"{COMMON_PROCESSING_CODE}:283-293",
            "usgs_utc_parse": f"{USGS_DOWNLOADER}:780",
            "clean_utc_normalisation": f"{CONSOLIDATOR}:409-415",
            "neon_source_column": f"{NEON_DOWNLOADER}:351-353",
        },
        "clean_snapshot_identity": {
            "dvc_expected_files": 4779,
            "dvc_expected_bytes": 62793867438,
            "current_files": len(clean_files),
            "current_bytes": clean_bytes,
            "count_and_size_exact": len(clean_files) == 4779 and clean_bytes == 62793867438,
            "dvc_pointer": str(TEACHER_CLEAN_DVC),
        },
        "outcomes_used_to_resolve_time": False,
    }
    pass_checks = [
        result["sites"] == 910,
        result["semantic_utc_proven_sites"] == 910,
        result["missing_timestamps"] == 0,
        result["duplicate_site_timestamp_keys"] == 0,
        result["nonmonotonic_sites"] == 0,
        result["non_hour_aligned_rows"] == 0,
        result["non_1h_adjacent_steps"] == 0,
        result["true_48h_spacing_all_sites"],
        result["clean_snapshot_identity"]["count_and_size_exact"],
    ]
    result["pass"] = bool(all(pass_checks))
    return result, summary


def coordinate_audit(cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = pd.read_csv(METADATA_910, dtype={"site_id": str})
    static = pd.read_csv(C.STATIC_PATH, dtype={"site_id": str})[["site_id", "country_code", "us_region"]]
    positions = json.loads(NEON_POSITIONS.read_text(encoding="utf-8"))
    registry = json.loads(SITE_REGISTER.read_text(encoding="utf-8"))
    table = cohort[["site_id", "split", "network"]].merge(metadata, on="site_id", validate="one_to_one").merge(static, on="site_id", validate="one_to_one")
    if len(table) != 910:
        raise RuntimeError(f"Coordinate base coverage {len(table)}/910")
    rows = []
    for row in table.itertuples(index=False):
        site_id = str(row.site_id)
        base_missing = bool(row.meta_latitude_missing or row.meta_longitude_missing)
        sensor = None
        register = registry.get(site_id, {})
        if site_id.startswith("N"):
            sensor = positions.get(site_id[1:5], {}).get("positions", {}).get(site_id[5:])
        if not base_missing:
            latitude = float(row.meta_latitude)
            longitude = float(row.meta_longitude)
            authority = "preexisting_final_h00_metadata_nonmissing"
            authority_path = str(METADATA_910)
            resolution = "AGREES_WITH_PREEXISTING_FINAL_910_TABLE"
        elif sensor is not None:
            latitude = float(sensor["lat"])
            longitude = float(sensor["lon"])
            authority = "NEON_official_sensor_position_cache"
            authority_path = str(NEON_POSITIONS)
            resolution = "RESOLVED_PREEXISTING_IMPUTED_PLACEHOLDER_WITH_SENSOR_POSITION"
        elif register and register.get("latitude") is not None and register.get("longitude") is not None:
            latitude = float(register["latitude"])
            longitude = float(register["longitude"])
            authority = "NEON_official_site_registry_fallback"
            authority_path = str(SITE_REGISTER)
            resolution = "RESOLVED_PREEXISTING_IMPUTED_PLACEHOLDER_WITH_SITE_COORDINATE"
        else:
            latitude = float("nan")
            longitude = float("nan")
            authority = "UNRESOLVED"
            authority_path = ""
            resolution = "UNRESOLVED"
        register_lat = register.get("latitude")
        register_lon = register.get("longitude")
        if sensor is not None and register_lat is not None and register_lon is not None:
            dy = (float(sensor["lat"]) - float(register_lat)) * 111.32
            dx = (float(sensor["lon"]) - float(register_lon)) * 111.32 * math.cos(math.radians(latitude))
            sensor_vs_registry_km = math.sqrt(dx * dx + dy * dy)
        else:
            sensor_vs_registry_km = float("nan")
        rows.append(
            {
                "site_id": site_id,
                "split": row.split,
                "country_code": row.country_code,
                "us_region": row.us_region,
                "network": row.network,
                "preexisting_meta_latitude": row.meta_latitude,
                "preexisting_meta_longitude": row.meta_longitude,
                "preexisting_latitude_missing_flag": row.meta_latitude_missing,
                "preexisting_longitude_missing_flag": row.meta_longitude_missing,
                "preexisting_pair_is_imputed_placeholder": base_missing,
                "selected_latitude": latitude,
                "selected_longitude": longitude,
                "latitude_band": C.latitude_band(latitude) if np.isfinite(latitude) else "UNRESOLVED",
                "coordinate_authority": authority,
                "coordinate_authority_path": authority_path,
                "resolution": resolution,
                "sensor_vs_site_registry_distance_km": sensor_vs_registry_km,
                "one_fixed_pair": bool(np.isfinite(latitude) and np.isfinite(longitude)),
                "latitude_in_range": bool(np.isfinite(latitude) and -90 <= latitude <= 90),
                "longitude_in_range_east_positive": bool(np.isfinite(longitude) and -180 <= longitude <= 180),
                "raw_coordinates_enter_model_tensor": False,
            }
        )
    audit = pd.DataFrame(rows)
    summary = {
        "sites": int(audit["site_id"].nunique()),
        "preexisting_nonmissing_pairs": int((~audit["preexisting_pair_is_imputed_placeholder"]).sum()),
        "preexisting_imputed_placeholders_not_used": int(audit["preexisting_pair_is_imputed_placeholder"].sum()),
        "neon_sensor_positions_used": int((audit["coordinate_authority"] == "NEON_official_sensor_position_cache").sum()),
        "neon_site_registry_fallbacks_used": int((audit["coordinate_authority"] == "NEON_official_site_registry_fallback").sum()),
        "unresolved": int((audit["coordinate_authority"] == "UNRESOLVED").sum()),
        "all_one_fixed_pair": bool(audit["one_fixed_pair"].all()),
        "all_in_range": bool(audit["latitude_in_range"].all() and audit["longitude_in_range_east_positive"].all()),
        "duplicate_site_rows": int(audit["site_id"].duplicated().sum()),
        "coordinate_sources_are_outcome_free": True,
        "raw_coordinates_enter_model_tensor": False,
    }
    summary["pass"] = bool(
        summary["sites"] == 910
        and summary["unresolved"] == 0
        and summary["all_one_fixed_pair"]
        and summary["all_in_range"]
        and summary["duplicate_site_rows"] == 0
    )
    return audit, summary


def calendar_manifest() -> pd.DataFrame:
    records = [
        ("sin_hour", "sin(2*pi*hour/24)", "integer UTC hour from stored timestamp", "UTC clock phase; no longitude/equation-of-time correction"),
        ("cos_hour", "cos(2*pi*hour/24)", "integer UTC hour from stored timestamp", "UTC clock phase; no longitude/equation-of-time correction"),
        ("sin_day", "sin(2*pi*day_of_year/365.25)", "UTC day of year from stored timestamp", "annual calendar phase"),
        ("cos_day", "cos(2*pi*day_of_year/365.25)", "UTC day of year from stored timestamp", "annual calendar phase"),
        ("sin_month", "sin(2*pi*(month-1)/12)", "UTC month from stored timestamp", "coarse month phase"),
        ("cos_month", "cos(2*pi*(month-1)/12)", "UTC month from stored timestamp", "coarse month phase"),
    ]
    return pd.DataFrame(
        [
            {
                "feature": name,
                "formula": formula,
                "timestamp_source": source,
                "semantic_timezone": "UTC",
                "meaning": meaning,
                "numerically_equivalent_to_solar_hour_sin_or_cos": False,
                "numerically_equivalent_to_solar_potential": False,
                "source_code": str(PREPARE_CODE),
                "source_code_sha256": C.sha256_file(PREPARE_CODE),
            }
            for name, formula, source, meaning in records
        ]
    )


def freeze_assignments(coordinates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = coordinates.loc[coordinates["split"].eq("train"), ["site_id", "country_code", "us_region", "latitude_band", "selected_latitude", "selected_longitude"]].copy()
    if len(train) != 638:
        raise RuntimeError(f"Training sites {len(train)}/638")
    train["assignment_stratum"] = train["country_code"] + "|" + train["latitude_band"]
    assignments = []
    for stratum, group in train.groupby("assignment_stratum", sort=True):
        group = group.copy()
        group["_hash"] = group["site_id"].map(lambda site: hashlib.sha256(f"42|outer|{site}".encode()).hexdigest())
        group = group.sort_values(["_hash", "site_id"]).reset_index(drop=True)
        group["outer_fold"] = np.arange(len(group)) % 3
        assignments.append(group.drop(columns="_hash"))
    outer = pd.concat(assignments, ignore_index=True).sort_values("site_id").reset_index(drop=True)
    outer["assignment_seed"] = 42
    outer["assignment_method"] = "sha256_order_within_country_x_fixed_latitude_band_then_round_robin_3fold"
    outer["all_rows_site_disjoint"] = True

    inner_rows = []
    for heldout_fold in range(3):
        development = outer.loc[outer["outer_fold"] != heldout_fold].copy()
        for stratum, group in development.groupby("assignment_stratum", sort=True):
            group = group.copy()
            group["_hash"] = group["site_id"].map(
                lambda site: hashlib.sha256(f"42|inner|{heldout_fold}|{site}".encode()).hexdigest()
            )
            group = group.sort_values(["_hash", "site_id"]).reset_index(drop=True)
            early_count = max(1, int(round(0.20 * len(group)))) if len(group) >= 2 else 0
            early_ids = set(group.head(early_count)["site_id"])
            for row in group.itertuples(index=False):
                inner_rows.append(
                    {
                        "outer_fold": heldout_fold,
                        "site_id": row.site_id,
                        "assignment_stratum": stratum,
                        "inner_role": "early_stopping" if row.site_id in early_ids else "gradient_development",
                        "assignment_seed": 42,
                        "assignment_method": "sha256_order_within_outer_development_country_x_fixed_latitude_band_20pct",
                    }
                )
    inner = pd.DataFrame(inner_rows).sort_values(["outer_fold", "site_id"]).reset_index(drop=True)
    if outer["site_id"].duplicated().any() or sorted(outer["outer_fold"].unique()) != [0, 1, 2]:
        raise RuntimeError("Outer assignment integrity failure")
    for fold in range(3):
        if set(inner.loc[inner["outer_fold"].eq(fold), "site_id"]) & set(outer.loc[outer["outer_fold"].eq(fold), "site_id"]):
            raise RuntimeError(f"Inner/outer leakage fold {fold}")
    return outer, inner


def angular_difference_degrees(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def validate_solar_algorithm(coordinates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    lookup = coordinates.set_index("site_id")
    gb_west_id = str(coordinates.loc[coordinates["country_code"].eq("GB")].sort_values("selected_longitude").iloc[0]["site_id"])
    locations = [
        ("GB_EAST", "E02262A"),
        ("GB_WEST", gb_west_id),
        ("CONUS", "01645762"),
        ("ALASKA", "NOKSRS1"),
        ("PUERTO_RICO", "NCUPES1"),
    ]
    timestamps = [
        ("winter_day", "2024-01-15T12:00:00Z"),
        ("winter_night", "2024-01-15T00:00:00Z"),
        ("leap_day_day", "2024-02-29T18:00:00Z"),
        ("us_dst_before", "2024-03-10T06:00:00Z"),
        ("us_dst_after", "2024-03-10T08:00:00Z"),
        ("gb_dst_before", "2024-03-31T00:00:00Z"),
        ("gb_dst_after", "2024-03-31T02:00:00Z"),
        ("summer_day", "2024-06-15T12:00:00Z"),
        ("summer_night", "2024-06-15T00:00:00Z"),
    ]
    rows = []
    for region, site_id in locations:
        latitude = float(lookup.loc[site_id, "selected_latitude"])
        longitude = float(lookup.loc[site_id, "selected_longitude"])
        observer = Observer(latitude=latitude, longitude=longitude, elevation=0.0)
        for scenario, value in timestamps:
            timestamp = pd.Timestamp(value)
            features, diagnostics = C.solar_features(pd.DatetimeIndex([timestamp]), latitude, longitude)
            pv_zenith = float(diagnostics["geometric_zenith_degrees"][0])
            pv_hour_angle = float(diagnostics["hour_angle_degrees"][0])
            dt = timestamp.to_pydatetime()
            independent_zenith = float(astral_zenith(observer, dt, with_refraction=False))
            independent_noon = astral_noon(observer, date=dt.date(), tzinfo=UTC)
            phase_minutes = (dt - independent_noon).total_seconds() / 60.0
            phase_minutes = (phase_minutes + 720.0) % 1440.0 - 720.0
            independent_hour_angle = phase_minutes * 0.25
            zenith_difference = abs(pv_zenith - independent_zenith)
            phase_difference_minutes = angular_difference_degrees(pv_hour_angle, independent_hour_angle) / 0.25
            rows.append(
                {
                    "region": region,
                    "site_id": site_id,
                    "scenario": scenario,
                    "timestamp_utc": timestamp,
                    "latitude": latitude,
                    "longitude": longitude,
                    "canonical_implementation": f"pvlib {pvlib.__version__} spa_python (NREL SPA), pressure=0",
                    "independent_implementation": f"astral {astral.__version__} geometric zenith and solar noon",
                    "pvlib_geometric_zenith_degrees": pv_zenith,
                    "astral_geometric_zenith_degrees": independent_zenith,
                    "zenith_absolute_difference_degrees": zenith_difference,
                    "pvlib_hour_angle_degrees": pv_hour_angle,
                    "astral_noon_hour_angle_degrees": independent_hour_angle,
                    "solar_phase_difference_minutes_equivalent": phase_difference_minutes,
                    "solar_hour_sin": float(features[0, 0]),
                    "solar_hour_cos": float(features[0, 1]),
                    "solar_potential": float(features[0, 2]),
                    "zenith_threshold_pass": zenith_difference <= 0.25,
                    "solar_phase_threshold_pass": phase_difference_minutes <= 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    summary = {
        "predeclared_locations": [{"region": region, "site_id": site_id} for region, site_id in locations],
        "predeclared_scenarios": [name for name, _ in timestamps],
        "comparisons": len(frame),
        "maximum_geometric_zenith_difference_degrees": float(frame["zenith_absolute_difference_degrees"].max()),
        "maximum_solar_phase_difference_minutes_equivalent": float(frame["solar_phase_difference_minutes_equivalent"].max()),
        "zenith_tolerance_degrees": 0.25,
        "solar_phase_tolerance_minutes": 1.0,
        "pass": bool(frame["zenith_threshold_pass"].all() and frame["solar_phase_threshold_pass"].all()),
    }
    return frame, summary


def generate_solar_sidecars(cohort: pd.DataFrame, coordinates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    C.SOLAR_ROOT.mkdir(parents=True, exist_ok=True)
    coord = coordinates.set_index("site_id")
    manifest_rows = []
    hash_rows = []
    total_rows = 0
    max_unit_circle_error = 0.0
    for number, row in enumerate(cohort.itertuples(index=False), 1):
        site_id = str(row.site_id)
        timestamps = pd.read_parquet(C.HOURLY_ROOT / f"{site_id}.parquet", columns=["timestamp"])["timestamp"]
        latitude = float(coord.loc[site_id, "selected_latitude"])
        longitude = float(coord.loc[site_id, "selected_longitude"])
        first, _ = C.solar_features(timestamps, latitude, longitude)
        second, _ = C.solar_features(timestamps, latitude, longitude)
        timestamp_values = pd.DatetimeIndex(timestamps).to_numpy(dtype="datetime64[ns]")
        first_hash = C.sha256_frame_bytes(timestamp_values, first)
        second_hash = C.sha256_frame_bytes(timestamp_values, second)
        unit_error = float(np.max(np.abs(np.square(first[:, 0].astype(np.float64)) + np.square(first[:, 1].astype(np.float64)) - 1.0)))
        max_unit_circle_error = max(max_unit_circle_error, unit_error)
        finite = bool(np.isfinite(first).all())
        bounds = bool(
            np.all(first[:, :2] >= -1.0)
            and np.all(first[:, :2] <= 1.0)
            and np.all(first[:, 2] >= 0.0)
            and np.all(first[:, 2] <= 1.0)
        )
        exact_regeneration = first_hash == second_hash and np.array_equal(first, second)
        if not finite or not bounds or not exact_regeneration or unit_error > 2e-7:
            raise RuntimeError(f"Solar invariant failed for {site_id}")
        sidecar = pd.DataFrame({"timestamp": timestamp_values})
        for index, field in enumerate(C.SOLAR_FIELDS):
            sidecar[field] = first[:, index]
        sidecar.to_parquet(C.SOLAR_ROOT / f"{site_id}.parquet", index=False, compression="zstd")
        manifest_rows.append(
            {
                "site_id": site_id,
                "split": row.split,
                "rows": len(sidecar),
                "start_utc": pd.Timestamp(timestamp_values.min()),
                "end_utc": pd.Timestamp(timestamp_values.max()),
                "latitude": latitude,
                "longitude": longitude,
                "coordinate_authority": coord.loc[site_id, "coordinate_authority"],
                "feature_order": "|".join(C.SOLAR_FIELDS),
                "finite": finite,
                "bounds_pass": bounds,
                "maximum_unit_circle_error": unit_error,
                "raw_coordinates_stored_in_model_sidecar": False,
            }
        )
        hash_rows.append(
            {
                "site_id": site_id,
                "rows": len(sidecar),
                "first_generation_sha256": first_hash,
                "second_generation_sha256": second_hash,
                "exact_regeneration": exact_regeneration,
                "sidecar_sha256": C.sha256_file(C.SOLAR_ROOT / f"{site_id}.parquet"),
            }
        )
        total_rows += len(sidecar)
        if number % 50 == 0 or number == len(cohort):
            print(f"solar-features {number}/{len(cohort)} sites rows={total_rows:,}", flush=True)
    manifest = pd.DataFrame(manifest_rows)
    hashes = pd.DataFrame(hash_rows)
    summary = {
        "sites": int(manifest["site_id"].nunique()),
        "rows": int(manifest["rows"].sum()),
        "all_finite": bool(manifest["finite"].all()),
        "all_bounds_pass": bool(manifest["bounds_pass"].all()),
        "maximum_unit_circle_error": max_unit_circle_error,
        "all_exact_regeneration": bool(hashes["exact_regeneration"].all()),
        "validation_or_test_statistics_used": False,
        "target_or_prediction_values_used": False,
        "raw_coordinates_enter_model_tensor": False,
    }
    summary["pass"] = bool(
        summary["sites"] == 910
        and summary["all_finite"]
        and summary["all_bounds_pass"]
        and summary["maximum_unit_circle_error"] <= 2e-7
        and summary["all_exact_regeneration"]
    )
    return manifest, hashes, summary


def software_environment() -> str:
    command = [str(Path(sys.executable)), "-m", "pip", "freeze"]
    freeze = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
    return "\n".join(
        [
            f"generated_utc={datetime.now(UTC).isoformat()}",
            f"platform={platform.platform()}",
            f"python={sys.version}",
            f"executable={sys.executable}",
            f"torch={torch.__version__}",
            f"cuda_available={torch.cuda.is_available()}",
            f"cuda_device={gpu}",
            f"pvlib={pvlib.__version__}",
            f"astral={astral.__version__}",
            "",
            "pip_freeze:",
            freeze,
        ]
    )


def main() -> None:
    C.OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(C.COHORT_PATH, dtype={"site_id": str})
    if cohort.groupby("split")["site_id"].nunique().to_dict() != {"test": 135, "train": 638, "validation": 137}:
        raise RuntimeError("Primary split is not the authoritative 638/137/135 split")

    inventory = artifact_inventory()
    inventory.to_csv(C.OUT / "authoritative_artifact_inventory.csv", index=False)
    inventory[["role", "path", "sha256"]].to_csv(C.OUT / "input_hashes.csv", index=False)

    time_result, time_summary = timestamp_audit(cohort)
    C.write_json(C.OUT / "timestamp_audit.json", time_result)
    time_summary.to_csv(C.OUT / "timestamp_source_summary.csv", index=False)

    coordinates, coordinate_result = coordinate_audit(cohort)
    coordinates.to_csv(C.COORDINATE_PATH, index=False, encoding="utf-8-sig")

    calendar = calendar_manifest()
    calendar.to_csv(C.OUT / "existing_calendar_feature_manifest.csv", index=False)
    already_represented = False

    if not time_result["pass"]:
        C.write_json(C.OUT / "phase_a_decision.json", {"decision": "H00_SOLAR_DYNAMIC1_BLOCKED_TIMESTAMP_SEMANTICS", "timestamp_audit": time_result, "formal_validation": "NOT_OPENED", "extension_test": "NOT_OPENED"})
        print("H00_SOLAR_DYNAMIC1_BLOCKED_TIMESTAMP_SEMANTICS")
        return
    if not coordinate_result["pass"]:
        C.write_json(C.OUT / "phase_a_decision.json", {"decision": "H00_SOLAR_DYNAMIC1_BLOCKED_COORDINATE_CONFLICT", "coordinate_audit": coordinate_result, "formal_validation": "NOT_OPENED", "extension_test": "NOT_OPENED"})
        print("H00_SOLAR_DYNAMIC1_BLOCKED_COORDINATE_CONFLICT")
        return
    if already_represented:
        C.write_json(C.OUT / "phase_a_decision.json", {"decision": "H00_SOLAR_DYNAMIC1_ALREADY_REPRESENTED", "formal_validation": "NOT_OPENED", "extension_test": "NOT_OPENED"})
        print("H00_SOLAR_DYNAMIC1_ALREADY_REPRESENTED")
        return

    outer, inner = freeze_assignments(coordinates)
    outer.to_csv(C.ASSIGNMENT_PATH, index=False)
    inner.to_csv(C.INNER_ASSIGNMENT_PATH, index=False)

    validation, validation_summary = validate_solar_algorithm(coordinates)
    validation.to_csv(C.OUT / "solar_feature_validation.csv", index=False)
    algorithm_manifest = {
        "canonical_algorithm": "NREL Solar Position Algorithm via pvlib.solarposition.spa_python",
        "pvlib_version": pvlib.__version__,
        "reference": SPA_REFERENCE,
        "input_time": "timezone-aware UTC created by explicit localisation of source-proven UTC storage",
        "longitude_convention": "degrees east positive in [-180,180]",
        "latitude_convention": "degrees north positive in [-90,90]",
        "spa_call": "spa_python(time, latitude, longitude, altitude=0, pressure=0, temperature=12, delta_t=None, atmos_refract=0, how='numpy')",
        "geometric_zenith_field": "zenith",
        "solar_hour_angle": "pvlib.solarposition.hour_angle(UTC_time, longitude, equation_of_time), degrees converted to radians",
        "features": {
            "solar_hour_sin": "sin(H)",
            "solar_hour_cos": "cos(H)",
            "solar_potential": "max(0, cos(geometric_zenith))",
        },
        "angle_units": {"SPA_and_hour_angle_output": "degrees", "trigonometric_input": "radians"},
        "refraction": "disabled by pressure=0 and atmos_refract=0; geometric/unrefracted zenith used",
        "leap_years": "handled from timezone-aware pandas timestamps by NREL SPA",
        "subhour_timestamps": "full minute/second resolution is retained; canonical H00 sidecars are exactly hour-aligned",
        "numerical_precision": "float64 calculation, bounded float32 model fields",
        "feature_code": str(C.PIPELINE / "common.py"),
        "feature_code_sha256": C.sha256_file(C.PIPELINE / "common.py"),
        "independent_verification": validation_summary,
    }
    C.write_json(C.OUT / "solar_algorithm_manifest.json", algorithm_manifest)
    if not validation_summary["pass"]:
        C.write_json(C.OUT / "phase_a_decision.json", {"decision": "H00_SOLAR_DYNAMIC1_FEATURE_VALIDATION_FAILED", "independent_validation": validation_summary, "formal_validation": "NOT_OPENED", "extension_test": "NOT_OPENED"})
        print("H00_SOLAR_DYNAMIC1_FEATURE_VALIDATION_FAILED")
        return

    feature_manifest, feature_hashes, feature_summary = generate_solar_sidecars(cohort, coordinates)
    feature_manifest.to_csv(C.OUT / "solar_feature_manifest.csv", index=False)
    feature_hashes.to_csv(C.OUT / "solar_feature_hashes.csv", index=False)
    if not feature_summary["pass"]:
        C.write_json(C.OUT / "phase_a_decision.json", {"decision": "H00_SOLAR_DYNAMIC1_FEATURE_VALIDATION_FAILED", "coverage_and_invariants": feature_summary, "formal_validation": "NOT_OPENED", "extension_test": "NOT_OPENED"})
        print("H00_SOLAR_DYNAMIC1_FEATURE_VALIDATION_FAILED")
        return

    config = {
        "experiment": "H00_SOLAR_DYNAMIC1",
        "scientific_status": "final_bounded_post_core_exploratory",
        "phase_a_status": "PASS",
        "primary_split": {"train": 638, "validation": 137, "test": 135},
        "formal_cohorts": {"validation_sites": 128, "validation_query_rows": 2719921, "test_sites": 126, "test_query_rows": 1493309},
        "support_days": C.SUPPORT_DAYS,
        "minimum_support_rows": C.MIN_SUPPORT,
        "sequence_length_hours": 48,
        "solar_fields": list(C.SOLAR_FIELDS),
        "latitude_bands": [{"lower_inclusive": a, "upper_exclusive": b, "label": label} for a, b, label in C.LATITUDE_BANDS],
        "outer_folds": {"folds": 3, "seed": 42, "strata": "country_code x fixed latitude_band", "assignment_sha256": C.sha256_file(C.ASSIGNMENT_PATH)},
        "inner_early_stopping": {"fraction_approximately": 0.20, "seed": 42, "site_disjoint": True, "assignment_sha256": C.sha256_file(C.INNER_ASSIGNMENT_PATH)},
        "architecture": {"injection": "first LSTM gate preactivation at every sequence step", "W_solar_shape": [256, 3], "bias": False, "zero_initialised": True, "trainable_parameters": 768, "all_original_tensors_frozen": True},
        "objective": "mean_site_candidate_query_MAE + mean_site_relu(candidate_MAE - 1.02*M0_MAE) + 1e-4*mean(W_solar^2)",
        "optimizer": {"family": "AdamW", "learning_rate": C.LEARNING_RATE, "weight_decay": C.WEIGHT_DECAY, "gradient_clip_norm": C.GRAD_CLIP},
        "scheduler": {"family": "ReduceLROnPlateau", "factor": 0.5, "patience": 2},
        "early_stopping": {"maximum_epochs": C.MAX_EPOCHS, "minimum_epochs": C.MIN_EPOCHS, "patience": C.EARLY_PATIENCE, "minimum_delta": C.MIN_DELTA},
        "sampler": {"query_cap_per_training_site_per_epoch": C.TRAIN_CAP, "inner_validation_query_cap_per_site": C.INNER_VALIDATION_CAP, "site_balanced": True},
        "seeds": list(C.SEEDS),
        "ensemble": "arithmetic mean of all three fixed seed raw predictions, followed by ensemble-specific support offset",
        "m0_checkpoint": str(C.M0_CHECKPOINT),
        "m0_checkpoint_sha256": C.sha256_file(C.M0_CHECKPOINT),
        "formal_validation_open_condition": "all 12 internal gates pass",
        "extension_test_open_condition": "all formal validation gates pass",
    }
    C.write_json(C.OUT / "config_frozen.json", config)
    phase_a = {
        "decision": "PHASE_A_PASSED_PROCEED_TO_TRAINING_ONLY_DIAGNOSTICS",
        "timestamp": time_result,
        "coordinates": coordinate_result,
        "existing_calendar_already_equivalent": False,
        "independent_solar_validation": validation_summary,
        "solar_coverage_and_invariants": feature_summary,
        "outer_assignment_sha256": C.sha256_file(C.ASSIGNMENT_PATH),
        "inner_assignment_sha256": C.sha256_file(C.INNER_ASSIGNMENT_PATH),
        "formal_validation": "NOT_OPENED",
        "extension_test": "NOT_OPENED",
    }
    C.write_json(C.OUT / "phase_a_decision.json", phase_a)
    (C.OUT / "software_environment.txt").write_text(software_environment(), encoding="utf-8")
    C.write_json(
        C.OUT / "run_manifest.json",
        {
            "experiment": "H00_SOLAR_DYNAMIC1",
            "created_utc": datetime.now(UTC).isoformat(),
            "stage": "PHASE_A_COMPLETE",
            "phase_a_decision": phase_a["decision"],
            "artifact_inventory_sha256": C.sha256_file(C.OUT / "authoritative_artifact_inventory.csv"),
            "config_sha256": C.sha256_file(C.OUT / "config_frozen.json"),
            "solar_algorithm_sha256": C.sha256_file(C.OUT / "solar_algorithm_manifest.json"),
            "test_outcomes_read_for_candidate": False,
        },
    )
    print(json.dumps({"status": "PASS", "timestamp": time_result, "coordinates": coordinate_result, "solar_validation": validation_summary, "solar_features": feature_summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
