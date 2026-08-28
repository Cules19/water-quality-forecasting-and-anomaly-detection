"""Merge teacher-Archive rainfall features into Common1000 pH hourly files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-per-strategy", type=int, default=0)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def bool_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def cohort_manifest(config: dict[str, Any]) -> pd.DataFrame:
    match = pd.read_csv(
        resolve(config["cohort_match_path"]),
        dtype={"site_id": "string"},
        encoding="utf-8-sig",
    )
    split = pd.read_csv(
        resolve(config["site_split_path"]),
        dtype={"site_id": "string"},
        encoding="utf-8-sig",
    )
    split = split.loc[
        :,
        [
            "site_id",
            "status",
            "reason",
            "network",
            "hourly_rows",
            "target_observed_hours",
            "coverage_days",
            "start",
            "end",
            "split",
        ],
    ].rename(
        columns={
            "status": "ph_status",
            "reason": "ph_reason",
            "hourly_rows": "ph_hourly_rows",
            "target_observed_hours": "ph_target_observed_hours",
            "coverage_days": "ph_coverage_days",
            "start": "ph_start",
            "end": "ph_end",
        }
    )
    manifest = match.merge(split, on="site_id", how="left", validate="one_to_one")
    if "network_y" in manifest.columns:
        manifest["network"] = manifest["network_y"].fillna(
            manifest.get("network_x")
        )
        manifest = manifest.drop(
            columns=[
                column
                for column in ("network_x", "network_y")
                if column in manifest.columns
            ]
        )
    direct = bool_values(manifest["preliminary_eligible_30d_80pct"])
    legacy = bool_values(manifest["existing_00045_varies"]) & ~direct
    manifest["rainfall_strategy"] = np.select(
        [direct, legacy],
        ["canonical_archive", "supplemental_existing_00045"],
        default="excluded",
    )
    manifest["primary_archive_cohort"] = direct
    manifest["extended_candidate_cohort"] = direct | legacy
    manifest["selected_rainfall_path"] = np.where(
        direct,
        manifest["file_path"].fillna(""),
        np.where(
            legacy,
            manifest["selected_file_resolved"].fillna(""),
            "",
        ),
    )
    manifest["selected_rainfall_source"] = np.where(
        direct,
        manifest["source"].fillna(""),
        np.where(legacy, "Existing_USGS_00045", ""),
    )
    manifest["exclusion_reason"] = ""
    excluded = ~manifest["extended_candidate_cohort"]
    constant_legacy = (
        bool_values(manifest["selected_file_has_existing_00045"])
        & ~bool_values(manifest["existing_00045_varies"])
        & excluded
    )
    manifest.loc[constant_legacy, "exclusion_reason"] = (
        "existing 00045 is constant or has no rain variation"
    )
    manifest.loc[excluded & ~constant_legacy, "exclusion_reason"] = manifest.loc[
        excluded & ~constant_legacy, "match_status"
    ].fillna("no usable rainfall source")
    manifest = manifest.sort_values("site_id", kind="mergesort").reset_index(drop=True)
    expected = int(config["expected_ph_sites"])
    if len(manifest) != expected or manifest["site_id"].nunique() != expected:
        raise ValueError(
            f"Expected {expected} unique pH sites, found {len(manifest)} rows and "
            f"{manifest['site_id'].nunique()} unique identifiers"
        )
    counts = manifest["rainfall_strategy"].value_counts().to_dict()
    expected_counts = {
        "canonical_archive": int(config["expected_primary_archive_sites"]),
        "supplemental_existing_00045": int(
            config["expected_supplemental_existing_00045_sites"]
        ),
        "excluded": int(config["expected_excluded_sites"]),
    }
    if counts != expected_counts:
        raise ValueError(f"Cohort counts differ: expected {expected_counts}, found {counts}")
    return manifest


def load_archive_features(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_parquet(
        path,
        columns=[
            "TIMESTAMP",
            "00045",
            "00045_observed_hour",
            "00045_source_missing",
            "precip_api",
            "precip_6h",
        ],
    )
    timestamps = pd.to_datetime(frame["TIMESTAMP"], errors="coerce", utc=True)
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("Canonical Archive feature file has invalid or duplicate timestamps")
    feature = pd.DataFrame(
        {
            "00045": pd.to_numeric(frame["00045"], errors="coerce").to_numpy(
                dtype=np.float64
            ),
            "00045_observed_hour": frame["00045_observed_hour"]
            .astype(bool)
            .to_numpy(),
            "00045_source_missing": frame["00045_source_missing"]
            .astype(bool)
            .to_numpy(),
            "precip_api": pd.to_numeric(
                frame["precip_api"], errors="coerce"
            ).to_numpy(dtype=np.float64),
            "precip_6h": pd.to_numeric(
                frame["precip_6h"], errors="coerce"
            ).to_numpy(dtype=np.float64),
        },
        index=pd.DatetimeIndex(timestamps),
    )
    stats = {
        "native_input_unit": "mm/hour",
        "hourly_aggregation": "already hourly",
        "unit_conversion_factor_to_mm": 1.0,
        "rain_source_start_utc": feature.index.min().isoformat(),
        "rain_source_end_utc": feature.index.max().isoformat(),
        "rain_source_hours": int(len(feature)),
        "rain_source_observed_hours": int(feature["00045_observed_hour"].sum()),
        "rain_source_gap_hours_filled_zero": int(
            feature["00045_source_missing"].sum()
        ),
    }
    return feature, stats


def load_legacy_features(
    path: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_parquet(path, columns=["TIMESTAMP", "00045"])
    timestamps = pd.to_datetime(source["TIMESTAMP"], errors="coerce", utc=True)
    values = pd.to_numeric(source["00045"], errors="coerce")
    values = values.where(np.isfinite(values) & values.ge(0))
    valid_timestamp = timestamps.notna()
    hourly_key = timestamps.loc[valid_timestamp].dt.floor("h")
    hourly_inches = values.loc[valid_timestamp].groupby(hourly_key).sum(min_count=1)
    hourly_inches = hourly_inches[
        ~hourly_inches.index.duplicated(keep="last")
    ].sort_index()
    if hourly_inches.empty:
        raise ValueError("No valid hourly legacy 00045 values")
    full_index = pd.date_range(
        hourly_inches.index.min(),
        hourly_inches.index.max(),
        freq="h",
        tz="UTC",
    )
    reindexed_inches = hourly_inches.reindex(full_index)
    observed = reindexed_inches.notna()
    rainfall_mm = (
        reindexed_inches.fillna(0.0)
        * float(config["legacy_existing_00045_mm_per_inch"])
    )
    precip_api = rainfall_mm.ewm(
        halflife=float(config["teacher_archive_api_halflife_hours"]),
        adjust=False,
    ).mean()
    precip_6h = rainfall_mm.rolling(
        int(config["teacher_archive_rolling_window_hours"]),
        min_periods=1,
    ).sum()
    feature = pd.DataFrame(
        {
            "00045": rainfall_mm.to_numpy(dtype=np.float64),
            "00045_observed_hour": observed.to_numpy(dtype=bool),
            "00045_source_missing": (~observed).to_numpy(dtype=bool),
            "precip_api": precip_api.to_numpy(dtype=np.float64),
            "precip_6h": precip_6h.to_numpy(dtype=np.float64),
        },
        index=full_index,
    )
    stats = {
        "native_input_unit": config["legacy_existing_00045_native_unit"],
        "hourly_aggregation": config["legacy_existing_00045_hourly_aggregation"],
        "unit_conversion_factor_to_mm": float(
            config["legacy_existing_00045_mm_per_inch"]
        ),
        "rain_source_rows": int(len(source)),
        "rain_source_valid_timestamp_rows": int(valid_timestamp.sum()),
        "rain_source_hours": int(len(feature)),
        "rain_source_start_utc": full_index.min().isoformat(),
        "rain_source_end_utc": full_index.max().isoformat(),
        "rain_source_observed_hours": int(observed.sum()),
        "rain_source_gap_hours_filled_zero": int((~observed).sum()),
    }
    return feature, stats


def build_one(
    row: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    site_id = str(row["site_id"])
    strategy = str(row["rainfall_strategy"])
    prepared_path = resolve(config["prepared_ph_root"]) / f"{site_id}.parquet"
    archive_path = (
        resolve(config["teacher_archive_feature_root"]) / f"{site_id}.parquet"
    )
    legacy_path = Path(str(row["selected_file_resolved"]))
    output_path = (
        output_root / "hourly_ph_rainfall_by_site" / f"{site_id}.parquet"
    )
    checkpoint_path = output_root / "checkpoints" / f"{site_id}.json"
    if resume and output_path.exists() and checkpoint_path.exists():
        return load_json(checkpoint_path)

    result: dict[str, Any] = {
        "site_id": site_id,
        "network": row.get("network"),
        "split": row.get("split"),
        "rainfall_strategy": strategy,
        "rainfall_source": row.get("selected_rainfall_source"),
        "prepared_ph_file": str(prepared_path),
        "rainfall_input_file": str(
            archive_path if strategy == "canonical_archive" else legacy_path
        ),
        "output_file": str(output_path),
        "primary_archive_cohort": bool(row["primary_archive_cohort"]),
        "extended_candidate_cohort": bool(row["extended_candidate_cohort"]),
        "status": "failed",
        "error_type": "",
        "error_message": "",
    }
    try:
        if not prepared_path.exists():
            raise FileNotFoundError(prepared_path)
        ph = pd.read_parquet(prepared_path)
        ph_timestamp = pd.to_datetime(ph["timestamp"], errors="coerce")
        if ph_timestamp.isna().any() or ph_timestamp.duplicated().any():
            raise ValueError("Prepared pH file has invalid or duplicate timestamps")
        if not ph_timestamp.is_monotonic_increasing:
            raise ValueError("Prepared pH timestamps are not monotonically increasing")
        if ph["site_id"].astype(str).ne(site_id).any():
            raise ValueError("Prepared pH site identifier differs from file name")
        utc_keys = pd.DatetimeIndex(ph_timestamp).tz_localize("UTC")

        if strategy == "canonical_archive":
            if not archive_path.exists():
                raise FileNotFoundError(archive_path)
            feature, source_stats = load_archive_features(archive_path)
        elif strategy == "supplemental_existing_00045":
            if not legacy_path.exists():
                raise FileNotFoundError(legacy_path)
            feature, source_stats = load_legacy_features(legacy_path, config)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        aligned = feature.reindex(utc_keys)
        rainfall = aligned["00045"].to_numpy(dtype=np.float64)
        observed = (
            aligned["00045_observed_hour"].fillna(False).to_numpy(dtype=bool)
        )
        source_missing = (
            aligned["00045_source_missing"].fillna(True).to_numpy(dtype=bool)
        )
        api = aligned["precip_api"].to_numpy(dtype=np.float64)
        rolling = aligned["precip_6h"].to_numpy(dtype=np.float64)
        feature_available = np.isfinite(api) & np.isfinite(rolling)
        rainfall_missing = (~observed).astype(np.int8)
        target_observed = ph["target_00400_observed"].astype(bool).to_numpy()
        rainfall_values = rainfall[np.isfinite(rainfall)]
        api_values = api[np.isfinite(api)]
        rolling_values = rolling[np.isfinite(rolling)]
        available_positions = np.flatnonzero(feature_available)
        overlap_start = (
            utc_keys[available_positions[0]] if len(available_positions) else pd.NaT
        )
        overlap_end = (
            utc_keys[available_positions[-1]] if len(available_positions) else pd.NaT
        )
        overlap_span_days = (
            float((overlap_end - overlap_start).total_seconds() / 86400.0)
            if pd.notna(overlap_start) and pd.notna(overlap_end)
            else 0.0
        )
        minimum_hours = int(config["minimum_feature_overlap_hours"])
        minimum_days = float(config["minimum_overlap_days"])
        rainfall_varies = (
            bool(pd.Series(rainfall_values).nunique(dropna=True) > 1)
            if len(rainfall_values)
            else False
        )
        negative_hours = int(
            np.less(rainfall_values, 0).sum()
            + np.less(api_values, 0).sum()
            + np.less(rolling_values, 0).sum()
        )
        ready = bool(
            feature_available.sum() >= minimum_hours
            and overlap_span_days >= minimum_days
            and rainfall_varies
            and negative_hours == 0
        )

        merged = ph.copy()
        merged["00045"] = rainfall.astype(np.float32)
        merged["00045_observed_hour"] = observed
        merged["00045_missing"] = rainfall_missing
        merged["precip_api"] = api.astype(np.float32)
        merged["precip_api_missing"] = (~np.isfinite(api)).astype(np.int8)
        merged["precip_6h"] = rolling.astype(np.float32)
        merged["precip_6h_missing"] = (~np.isfinite(rolling)).astype(np.int8)
        merged["precip_feature_available"] = feature_available
        merged["precip_source"] = pd.Series(
            [str(row["selected_rainfall_source"])] * len(merged),
            dtype="string",
        )
        merged["precip_unit"] = pd.Series(
            [str(config["rainfall_unit"])] * len(merged),
            dtype="string",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        merged.to_parquet(
            temporary,
            index=False,
            compression=str(config["parquet_compression"]),
        )
        os.replace(temporary, output_path)
        result.update(source_stats)
        result.update(
            {
                "status": "success",
                "model_hours": int(len(ph)),
                "target_observed_hours": int(target_observed.sum()),
                "feature_available_hours": int(feature_available.sum()),
                "feature_available_fraction_all_model_hours": float(
                    feature_available.mean()
                ),
                "target_observed_with_feature_hours": int(
                    (target_observed & feature_available).sum()
                ),
                "rainfall_observed_hours_on_model_axis": int(observed.sum()),
                "rainfall_missing_flag_hours_on_model_axis": int(
                    rainfall_missing.sum()
                ),
                "source_missing_filled_zero_hours_on_model_axis": int(
                    (source_missing & feature_available).sum()
                ),
                "outside_rainfall_period_hours": int((~feature_available).sum()),
                "feature_overlap_start_utc": (
                    overlap_start.isoformat() if pd.notna(overlap_start) else ""
                ),
                "feature_overlap_end_utc": (
                    overlap_end.isoformat() if pd.notna(overlap_end) else ""
                ),
                "feature_overlap_span_days": overlap_span_days,
                "rainfall_unique_values_on_model_axis": int(
                    pd.Series(rainfall_values).nunique(dropna=True)
                ),
                "rainfall_varies_on_model_axis": rainfall_varies,
                "rainfall_minimum_mm": (
                    float(np.min(rainfall_values)) if len(rainfall_values) else None
                ),
                "rainfall_maximum_mm": (
                    float(np.max(rainfall_values)) if len(rainfall_values) else None
                ),
                "precip_api_minimum": (
                    float(np.min(api_values)) if len(api_values) else None
                ),
                "precip_api_maximum": (
                    float(np.max(api_values)) if len(api_values) else None
                ),
                "precip_6h_minimum": (
                    float(np.min(rolling_values)) if len(rolling_values) else None
                ),
                "precip_6h_maximum": (
                    float(np.max(rolling_values)) if len(rolling_values) else None
                ),
                "negative_feature_hours": negative_hours,
                "ready_for_common_key_experiment": ready,
                "output_rows": int(len(merged)),
                "output_size_bytes": int(output_path.stat().st_size),
                "output_sha256": sha256(output_path),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "ready_for_common_key_experiment": False,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
    write_json(checkpoint_path, result)
    return result


def grouped_summary(status: pd.DataFrame, field: str) -> pd.DataFrame:
    return (
        status.groupby(field, dropna=False)
        .agg(
            candidate_sites=("site_id", "size"),
            successful_sites=("status", lambda values: int(values.eq("success").sum())),
            ready_sites=("ready_for_common_key_experiment", "sum"),
            model_hours=("model_hours", "sum"),
            target_observed_hours=("target_observed_hours", "sum"),
            feature_available_hours=("feature_available_hours", "sum"),
            target_observed_with_feature_hours=(
                "target_observed_with_feature_hours",
                "sum",
            ),
            rainfall_observed_hours=(
                "rainfall_observed_hours_on_model_axis",
                "sum",
            ),
            outside_rainfall_period_hours=("outside_rainfall_period_hours", "sum"),
            negative_feature_hours=("negative_feature_hours", "sum"),
            median_feature_coverage=(
                "feature_available_fraction_all_model_hours",
                "median",
            ),
            minimum_feature_coverage=(
                "feature_available_fraction_all_model_hours",
                "min",
            ),
        )
        .reset_index()
    )


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_json(config_path)
    manifest = cohort_manifest(config)
    candidates = manifest.loc[manifest["extended_candidate_cohort"]].copy()
    if args.smoke_per_strategy > 0:
        candidates = (
            candidates.sort_values(["rainfall_strategy", "site_id"], kind="mergesort")
            .groupby("rainfall_strategy", group_keys=False)
            .head(int(args.smoke_per_strategy))
            .copy()
        )
    base_output = resolve(config["output_root"])
    output_root = (
        base_output / f"smoke_{int(args.smoke_per_strategy)}_per_strategy"
        if args.smoke_per_strategy > 0
        else base_output / "full"
    )
    for name in ("hourly_ph_rainfall_by_site", "checkpoints", "reports"):
        (output_root / name).mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(config["workers"]))) as executor:
        futures = [
            executor.submit(
                build_one,
                row,
                config,
                output_root,
                bool(args.resume),
            )
            for row in candidates.to_dict(orient="records")
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"completed {completed}/{len(futures)}", flush=True)

    status = pd.DataFrame(results).sort_values("site_id", kind="mergesort")
    reports = output_root / "reports"
    write_csv(reports / "merge_qa_by_site.csv", status)
    full_manifest = manifest.merge(
        status[
            [
                "site_id",
                "status",
                "ready_for_common_key_experiment",
                "output_file",
            ]
        ].rename(columns={"status": "merge_status"}),
        on="site_id",
        how="left",
        validate="one_to_one",
    )
    full_manifest["merge_status"] = full_manifest["merge_status"].fillna("not_built")
    full_manifest["ready_for_common_key_experiment"] = (
        full_manifest["ready_for_common_key_experiment"].fillna(False).astype(bool)
    )
    write_csv(reports / "cohort_manifest.csv", full_manifest)
    ready = status.loc[
        status["status"].eq("success")
        & status["ready_for_common_key_experiment"].astype(bool)
    ].copy()
    write_csv(reports / "extended_experiment_cohort.csv", ready)
    primary = ready.loc[ready["rainfall_strategy"].eq("canonical_archive")].copy()
    write_csv(reports / "primary_archive_experiment_cohort.csv", primary)
    write_csv(
        reports / "supplemental_existing_00045_cohort.csv",
        ready.loc[
            ready["rainfall_strategy"].eq("supplemental_existing_00045")
        ].copy(),
    )
    write_csv(
        reports / "excluded_sites.csv",
        full_manifest.loc[
            ~full_manifest["extended_candidate_cohort"],
            [
                "site_id",
                "network",
                "split",
                "match_status",
                "selected_file_has_existing_00045",
                "existing_00045_nonmissing_rows",
                "existing_00045_unique_value_count",
                "existing_00045_varies",
                "exclusion_reason",
            ],
        ].copy(),
    )
    write_csv(reports / "strategy_summary.csv", grouped_summary(status, "rainfall_strategy"))
    write_csv(reports / "split_summary.csv", grouped_summary(status, "split"))
    parameters = pd.DataFrame(
        [
            {
                "parameter": "primary_cohort_rule",
                "value": config["primary_cohort_rule"],
                "unit": "",
            },
            {
                "parameter": "extended_cohort_rule",
                "value": config["extended_cohort_rule"],
                "unit": "",
            },
            {
                "parameter": "precip_api_half_life",
                "value": config["teacher_archive_api_halflife_hours"],
                "unit": "hours",
            },
            {
                "parameter": "precip_6h_window",
                "value": config["teacher_archive_rolling_window_hours"],
                "unit": "hours",
            },
            {
                "parameter": "legacy_00045_conversion",
                "value": config["legacy_existing_00045_mm_per_inch"],
                "unit": "mm/inch",
            },
            {
                "parameter": "minimum_feature_overlap",
                "value": config["minimum_feature_overlap_hours"],
                "unit": "hours",
            },
            {
                "parameter": "minimum_overlap",
                "value": config["minimum_overlap_days"],
                "unit": "days",
            },
        ]
    )
    write_csv(reports / "parameters.csv", parameters)

    success = status["status"].eq("success")
    summary = {
        "config": str(config_path),
        "mode": "smoke" if args.smoke_per_strategy > 0 else "full",
        "total_ph_sites": int(len(manifest)),
        "requested_candidate_sites": int(len(status)),
        "successful_candidate_sites": int(success.sum()),
        "failed_candidate_sites": int((~success).sum()),
        "primary_archive_sites_generated": int(
            status.loc[success, "rainfall_strategy"].eq("canonical_archive").sum()
        ),
        "supplemental_existing_00045_sites_generated": int(
            status.loc[success, "rainfall_strategy"]
            .eq("supplemental_existing_00045")
            .sum()
        ),
        "excluded_sites": int((~manifest["extended_candidate_cohort"]).sum()),
        "primary_archive_sites_ready": int(len(primary)),
        "extended_sites_ready": int(len(ready)),
        "ready_site_split_counts": {
            str(name): int(count)
            for name, count in ready.groupby("split").size().items()
        },
        "total_model_hours": int(
            pd.to_numeric(status["model_hours"], errors="coerce").fillna(0).sum()
        ),
        "total_feature_available_hours": int(
            pd.to_numeric(
                status["feature_available_hours"], errors="coerce"
            ).fillna(0).sum()
        ),
        "total_target_observed_with_feature_hours": int(
            pd.to_numeric(
                status["target_observed_with_feature_hours"],
                errors="coerce",
            ).fillna(0).sum()
        ),
        "negative_feature_hours": int(
            pd.to_numeric(
                status["negative_feature_hours"], errors="coerce"
            ).fillna(0).sum()
        ),
        "teacher_archive_formula": {
            "precip_api": config["teacher_archive_api_formula"],
            "precip_6h": config["teacher_archive_rolling_formula"],
        },
        "merged_directory": str(output_root / "hourly_ph_rainfall_by_site"),
        "original_ph_files_modified": False,
        "rainfall_source_files_modified": False,
        "site_splits_modified": False,
        "models_trained": False,
    }
    write_json(reports / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if bool(success.all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
