"""Audit downloaded hourly precipitation files and match them to the common1000 cohort.

This stage is read-only with respect to source data. It reconstructs hourly
timestamps from year/month/day/hour because the source ``datetime`` column is
date-only, selects the latest export by the timestamp embedded in the filename,
and writes auditable CSV/JSON reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FILENAME_RE = re.compile(
    r"^(?P<site_id>.+?)_precipitation_from_(?P<source>.+)_"
    r"(?P<export_stamp>\d{8}_\d{6})\.csv$"
)
REQUIRED_COLUMNS = [
    "datetime",
    "precipitation_mm",
    "date",
    "year",
    "month",
    "day",
    "hour",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precip-dir", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def utc_timestamp(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce", utc=True)


def finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def scan_file(path: Path, cohort_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> dict[str, Any]:
    match = FILENAME_RE.match(path.name)
    if match is None:
        return {
            "file_name": path.name,
            "file_path": str(path),
            "file_size_bytes": path.stat().st_size,
            "file_name_valid": False,
            "scan_error": "filename does not match expected pattern",
        }

    site_id = match.group("site_id")
    source = match.group("source")
    export_stamp = pd.to_datetime(match.group("export_stamp"), format="%Y%m%d_%H%M%S")
    base: dict[str, Any] = {
        "file_name": path.name,
        "file_path": str(path),
        "site_id": site_id,
        "source": source,
        "export_timestamp": export_stamp.isoformat(),
        "file_size_bytes": path.stat().st_size,
        "file_name_valid": True,
        "sha256": file_sha256(path),
        "scan_error": "",
    }

    try:
        frame = pd.read_csv(path, low_memory=False)
        base["columns"] = ";".join(str(column) for column in frame.columns)
        base["schema_valid"] = list(frame.columns) == REQUIRED_COLUMNS
        base["missing_required_columns"] = ";".join(
            column for column in REQUIRED_COLUMNS if column not in frame.columns
        )
        if base["missing_required_columns"]:
            raise ValueError(f"missing columns: {base['missing_required_columns']}")

        values = pd.to_numeric(frame["precipitation_mm"], errors="coerce")
        finite = values.where(np.isfinite(values))
        timestamps = pd.to_datetime(
            {
                "year": pd.to_numeric(frame["year"], errors="coerce"),
                "month": pd.to_numeric(frame["month"], errors="coerce"),
                "day": pd.to_numeric(frame["day"], errors="coerce"),
                "hour": pd.to_numeric(frame["hour"], errors="coerce"),
            },
            errors="coerce",
            utc=True,
        )
        valid_timestamp_mask = timestamps.notna()
        timestamp_value = pd.DataFrame(
            {
                "timestamp": timestamps[valid_timestamp_mask],
                "precipitation_mm": finite[valid_timestamp_mask],
            }
        )
        duplicate_timestamp_rows = int(
            timestamp_value["timestamp"].duplicated(keep="last").sum()
        )
        unique_hourly = (
            timestamp_value.drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp", kind="mergesort")
            .reset_index(drop=True)
        )
        unique_timestamps = unique_hourly["timestamp"]
        first_timestamp = unique_timestamps.min() if len(unique_timestamps) else pd.NaT
        last_timestamp = unique_timestamps.max() if len(unique_timestamps) else pd.NaT
        expected_hours = (
            int((last_timestamp - first_timestamp).total_seconds() // 3600) + 1
            if pd.notna(first_timestamp) and pd.notna(last_timestamp)
            else 0
        )
        diffs = unique_timestamps.diff().dt.total_seconds().div(3600)
        gap_diffs = diffs[diffs > 1]

        base.update(
            {
                "row_count": int(len(frame)),
                "valid_timestamp_rows": int(valid_timestamp_mask.sum()),
                "invalid_timestamp_rows": int((~valid_timestamp_mask).sum()),
                "duplicate_timestamp_rows": duplicate_timestamp_rows,
                "timestamps_monotonic_non_decreasing": bool(
                    timestamps[valid_timestamp_mask].is_monotonic_increasing
                ),
                "unique_hour_count": int(len(unique_hourly)),
                "start_timestamp_utc": (
                    first_timestamp.isoformat() if pd.notna(first_timestamp) else ""
                ),
                "end_timestamp_utc": (
                    last_timestamp.isoformat() if pd.notna(last_timestamp) else ""
                ),
                "expected_hour_count": expected_hours,
                "hourly_timestamp_coverage": (
                    float(len(unique_hourly) / expected_hours) if expected_hours else None
                ),
                "gap_count": int(len(gap_diffs)),
                "missing_hours_inside_span": int(
                    np.maximum(gap_diffs.to_numpy(dtype=float) - 1, 0).sum()
                ),
                "maximum_gap_hours": finite_or_none(gap_diffs.max()),
                "precip_nonmissing_rows": int(finite.notna().sum()),
                "precip_missing_rows": int(finite.isna().sum()),
                "precip_zero_rows": int(finite.eq(0).sum()),
                "precip_positive_rows": int(finite.gt(0).sum()),
                "precip_negative_rows": int(finite.lt(0).sum()),
                "precip_minimum_mm": finite_or_none(finite.min()),
                "precip_maximum_mm": finite_or_none(finite.max()),
                "precip_sum_mm": finite_or_none(finite.sum(min_count=1)),
                "precip_unique_value_count": int(finite.nunique(dropna=True)),
                "datetime_column_is_date_only": bool(
                    frame["datetime"]
                    .astype("string")
                    .dropna()
                    .str.fullmatch(r"\d{4}-\d{2}-\d{2}")
                    .all()
                ),
            }
        )

        window = cohort_windows.get(site_id)
        if window is not None and len(unique_hourly):
            usable_start, usable_end = window
            overlap_start = max(first_timestamp, usable_start)
            overlap_end = min(last_timestamp, usable_end)
            if overlap_end >= overlap_start:
                expected_overlap_hours = (
                    int((overlap_end.floor("h") - overlap_start.ceil("h")).total_seconds() // 3600)
                    + 1
                )
                expected_overlap_hours = max(0, expected_overlap_hours)
                overlap_mask = unique_hourly["timestamp"].between(
                    overlap_start.ceil("h"), overlap_end.floor("h"), inclusive="both"
                )
                overlap = unique_hourly.loc[overlap_mask]
                overlap_values = overlap["precipitation_mm"]
                base.update(
                    {
                        "cohort_overlap_start_utc": overlap_start.isoformat(),
                        "cohort_overlap_end_utc": overlap_end.isoformat(),
                        "cohort_expected_overlap_hours": expected_overlap_hours,
                        "cohort_overlap_unique_hours": int(len(overlap)),
                        "cohort_overlap_nonmissing_hours": int(overlap_values.notna().sum()),
                        "cohort_overlap_positive_hours": int(overlap_values.gt(0).sum()),
                        "cohort_overlap_unique_value_count": int(
                            overlap_values.nunique(dropna=True)
                        ),
                        "cohort_overlap_hour_coverage": (
                            float(len(overlap) / expected_overlap_hours)
                            if expected_overlap_hours
                            else None
                        ),
                        "cohort_overlap_nonmissing_coverage": (
                            float(overlap_values.notna().sum() / expected_overlap_hours)
                            if expected_overlap_hours
                            else None
                        ),
                    }
                )
    except Exception as exc:  # noqa: BLE001
        base["scan_error"] = f"{type(exc).__name__}: {exc}"
    return base


def registry_frame(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = []
    for key, value in payload.items():
        precip = value.get("precipitation_description") or {}
        precip_file = value.get("precipitation_data_file_name") or precip.get("data_file") or ""
        rows.append(
            {
                "site_id": str(value.get("SITE_ID") or key),
                "registry_source": value.get("source"),
                "registry_country_code": value.get("country_cd"),
                "registry_latitude": value.get("latitude"),
                "registry_longitude": value.get("longitude"),
                "registry_has_precipitation": bool(value.get("has_precipitation", False)),
                "registry_precipitation_source": value.get("linked_precipitation_source"),
                "registry_precipitation_file": Path(precip_file).name if precip_file else "",
                "registry_precipitation_data_points": precip.get("data_points"),
                "registry_start_date": value.get("start_date"),
                "registry_end_date": value.get("end_date"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(args.cohort, dtype={"site_id": "string"}, encoding="utf-8-sig")
    cohort["site_id"] = cohort["site_id"].astype("string")
    cohort["usable_start_utc"] = pd.to_datetime(cohort["usable_start"], errors="coerce", utc=True)
    cohort["usable_end_utc"] = pd.to_datetime(cohort["usable_end"], errors="coerce", utc=True)
    cohort_windows = {
        str(row.site_id): (row.usable_start_utc, row.usable_end_utc)
        for row in cohort.itertuples()
        if pd.notna(row.usable_start_utc) and pd.notna(row.usable_end_utc)
    }

    paths = sorted(args.precip_dir.glob("*.csv"))
    file_rows = []
    for number, path in enumerate(paths, start=1):
        file_rows.append(scan_file(path, cohort_windows))
        if number % 100 == 0 or number == len(paths):
            print(f"scanned {number}/{len(paths)}", flush=True)
    files = pd.DataFrame(file_rows)

    valid_files = files.loc[files["file_name_valid"].fillna(False)].copy()
    valid_files["export_timestamp_sort"] = pd.to_datetime(
        valid_files["export_timestamp"], errors="coerce"
    )
    canonical_index = (
        valid_files.sort_values(
            ["site_id", "export_timestamp_sort", "file_name"], kind="mergesort"
        )
        .groupby("site_id", sort=False)
        .tail(1)
        .index
    )
    files["is_canonical_latest_export"] = False
    files.loc[canonical_index, "is_canonical_latest_export"] = True

    duplicate_summary = (
        valid_files.groupby("site_id", as_index=False)
        .agg(
            export_copy_count=("file_name", "size"),
            distinct_content_hash_count=("sha256", "nunique"),
            earliest_export_timestamp=("export_timestamp_sort", "min"),
            latest_export_timestamp=("export_timestamp_sort", "max"),
        )
    )
    duplicate_summary["all_export_copies_identical"] = (
        duplicate_summary["distinct_content_hash_count"] == 1
    )

    canonical = files.loc[files["is_canonical_latest_export"]].copy()
    canonical = canonical.merge(duplicate_summary, on="site_id", how="left")
    registry = registry_frame(args.registry)
    sites = canonical.merge(registry, on="site_id", how="left", validate="one_to_one")
    sites["registry_match"] = sites["registry_source"].notna()
    sites["registry_filename_matches_canonical"] = (
        sites["registry_precipitation_file"].fillna("").eq(sites["file_name"].fillna(""))
    )

    cohort_columns = [
        "site_id",
        "candidate_rank",
        "network",
        "selected_file",
        "usable_start",
        "usable_end",
        "usable_coverage_days",
        "usable_model_rows",
        "quality_tier",
        "canonical_physical_site_id",
        "is_duplicate_alias",
        "preferred_record",
    ]
    cohort_match = cohort[cohort_columns].copy()
    cohort_match["selected_filename_mentions_00045"] = cohort_match[
        "selected_file"
    ].astype("string").str.contains(r"(?:^|_)00045(?:_|\.|$)", regex=True, na=False)
    existing_precip_rows = []
    for row in cohort_match.itertuples():
        selected_path = Path(str(row.selected_file))
        if not selected_path.is_absolute():
            selected_path = args.project_root / selected_path
        status: dict[str, Any] = {
            "site_id": str(row.site_id),
            "selected_file_resolved": str(selected_path),
            "selected_file_exists": selected_path.exists(),
            "selected_file_has_existing_00045": False,
            "existing_00045_nonmissing_rows": 0,
            "existing_00045_positive_rows": 0,
            "existing_00045_negative_rows": 0,
            "existing_00045_unique_value_count": 0,
            "existing_00045_varies": False,
            "existing_00045_error": "",
        }
        try:
            if selected_path.exists() and selected_path.suffix.lower() == ".parquet":
                columns = set(pq.read_schema(selected_path).names)
                status["selected_file_has_existing_00045"] = "00045" in columns
                if "00045" in columns:
                    values = pd.to_numeric(
                        pd.read_parquet(selected_path, columns=["00045"])["00045"],
                        errors="coerce",
                    )
                    values = values.where(np.isfinite(values))
                    status.update(
                        {
                            "existing_00045_nonmissing_rows": int(values.notna().sum()),
                            "existing_00045_positive_rows": int(values.gt(0).sum()),
                            "existing_00045_negative_rows": int(values.lt(0).sum()),
                            "existing_00045_unique_value_count": int(
                                values.nunique(dropna=True)
                            ),
                            "existing_00045_varies": bool(
                                values.nunique(dropna=True) > 1
                            ),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            status["existing_00045_error"] = f"{type(exc).__name__}: {exc}"
        existing_precip_rows.append(status)
    cohort_match = cohort_match.merge(
        pd.DataFrame(existing_precip_rows), on="site_id", how="left", validate="one_to_one"
    )
    site_columns = [
        "site_id",
        "source",
        "file_name",
        "file_path",
        "file_size_bytes",
        "export_copy_count",
        "distinct_content_hash_count",
        "all_export_copies_identical",
        "row_count",
        "schema_valid",
        "scan_error",
        "datetime_column_is_date_only",
        "unique_hour_count",
        "start_timestamp_utc",
        "end_timestamp_utc",
        "expected_hour_count",
        "hourly_timestamp_coverage",
        "duplicate_timestamp_rows",
        "gap_count",
        "missing_hours_inside_span",
        "maximum_gap_hours",
        "precip_nonmissing_rows",
        "precip_missing_rows",
        "precip_zero_rows",
        "precip_positive_rows",
        "precip_negative_rows",
        "precip_minimum_mm",
        "precip_maximum_mm",
        "precip_sum_mm",
        "precip_unique_value_count",
        "cohort_overlap_start_utc",
        "cohort_overlap_end_utc",
        "cohort_expected_overlap_hours",
        "cohort_overlap_unique_hours",
        "cohort_overlap_nonmissing_hours",
        "cohort_overlap_positive_hours",
        "cohort_overlap_unique_value_count",
        "cohort_overlap_hour_coverage",
        "cohort_overlap_nonmissing_coverage",
        "registry_match",
        "registry_source",
        "registry_country_code",
        "registry_latitude",
        "registry_longitude",
        "registry_has_precipitation",
        "registry_precipitation_source",
        "registry_precipitation_file",
        "registry_precipitation_data_points",
        "registry_filename_matches_canonical",
    ]
    for column in site_columns:
        if column not in sites:
            sites[column] = np.nan
    cohort_match = cohort_match.merge(
        sites[site_columns], on="site_id", how="left", validate="one_to_one"
    )
    cohort_match["precipitation_file_match"] = cohort_match["file_name"].notna()
    cohort_match["has_time_overlap"] = (
        pd.to_numeric(cohort_match["cohort_expected_overlap_hours"], errors="coerce")
        .fillna(0)
        .gt(0)
    )
    cohort_match["overlap_days"] = (
        pd.to_numeric(cohort_match["cohort_expected_overlap_hours"], errors="coerce")
        .fillna(0)
        .div(24)
    )
    cohort_match["rainfall_varies_in_overlap"] = (
        pd.to_numeric(cohort_match["cohort_overlap_unique_value_count"], errors="coerce")
        .fillna(0)
        .gt(1)
    )
    cohort_match["preliminary_eligible_30d_80pct"] = (
        cohort_match["precipitation_file_match"]
        & cohort_match["overlap_days"].ge(30)
        & cohort_match["rainfall_varies_in_overlap"]
        & pd.to_numeric(
            cohort_match["cohort_overlap_nonmissing_coverage"], errors="coerce"
        )
        .fillna(0)
        .ge(0.8)
    )
    cohort_match["any_preliminary_dynamic_rainfall"] = (
        cohort_match["preliminary_eligible_30d_80pct"]
        | cohort_match["existing_00045_varies"].fillna(False)
    )
    cohort_match["match_status"] = np.select(
        [
            ~cohort_match["precipitation_file_match"],
            ~cohort_match["has_time_overlap"],
            cohort_match["overlap_days"].lt(30),
            ~cohort_match["rainfall_varies_in_overlap"],
            pd.to_numeric(
                cohort_match["cohort_overlap_nonmissing_coverage"], errors="coerce"
            )
            .fillna(0)
            .lt(0.8),
        ],
        [
            "no precipitation file",
            "no time overlap",
            "time overlap under 30 days",
            "rainfall does not vary in overlap",
            "overlap coverage under 80%",
        ],
        default="preliminary eligible",
    )

    network_summary = (
        cohort_match.groupby("network", dropna=False)
        .agg(
            cohort_sites=("site_id", "size"),
            exact_file_matches=("precipitation_file_match", "sum"),
            time_overlap_sites=("has_time_overlap", "sum"),
            preliminary_eligible_sites=("preliminary_eligible_30d_80pct", "sum"),
            existing_00045_sites=("selected_file_has_existing_00045", "sum"),
            existing_00045_varying_sites=("existing_00045_varies", "sum"),
            any_preliminary_dynamic_rainfall=("any_preliminary_dynamic_rainfall", "sum"),
        )
        .reset_index()
    )
    source_summary = (
        sites.groupby("source", dropna=False)
        .agg(
            unique_precipitation_sites=("site_id", "size"),
            exported_files=("export_copy_count", "sum"),
            sites_with_nonidentical_exports=(
                "all_export_copies_identical",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            sites_with_scan_errors=("scan_error", lambda values: int(values.fillna("").ne("").sum())),
            sites_with_negative_values=("precip_negative_rows", lambda values: int(pd.to_numeric(values, errors="coerce").fillna(0).gt(0).sum())),
        )
        .reset_index()
    )
    match_status_summary = (
        cohort_match.groupby("match_status", dropna=False)
        .size()
        .rename("site_count")
        .reset_index()
        .sort_values("site_count", ascending=False)
    )

    summary = {
        "precipitation_directory": str(args.precip_dir),
        "downloaded_csv_files": int(len(files)),
        "unique_precipitation_sites": int(len(sites)),
        "precipitation_sources": {
            str(row.source): int(row.unique_precipitation_sites)
            for row in source_summary.itertuples()
        },
        "files_with_scan_errors": int(files["scan_error"].fillna("").ne("").sum()),
        "canonical_sites_with_schema_errors": int(
            (~sites["schema_valid"].fillna(False)).sum()
        ),
        "canonical_sites_with_negative_precipitation": int(
            pd.to_numeric(sites["precip_negative_rows"], errors="coerce")
            .fillna(0)
            .gt(0)
            .sum()
        ),
        "sites_with_multiple_exports": int(sites["export_copy_count"].gt(1).sum()),
        "sites_with_nonidentical_exports": int(
            (~sites["all_export_copies_identical"].astype(bool)).sum()
        ),
        "cohort_sites": int(len(cohort_match)),
        "cohort_exact_file_matches": int(cohort_match["precipitation_file_match"].sum()),
        "cohort_unmatched_sites": int((~cohort_match["precipitation_file_match"]).sum()),
        "cohort_time_overlap_sites": int(cohort_match["has_time_overlap"].sum()),
        "cohort_overlap_at_least_30_days": int(cohort_match["overlap_days"].ge(30).sum()),
        "cohort_preliminary_eligible_30d_80pct": int(
            cohort_match["preliminary_eligible_30d_80pct"].sum()
        ),
        "cohort_sites_already_with_00045": int(
            cohort_match["selected_file_has_existing_00045"].sum()
        ),
        "cohort_sites_with_varying_existing_00045": int(
            cohort_match["existing_00045_varies"].sum()
        ),
        "cohort_sites_with_any_preliminary_dynamic_rainfall": int(
            cohort_match["any_preliminary_dynamic_rainfall"].sum()
        ),
        "cohort_sites_without_any_preliminary_dynamic_rainfall": int(
            (~cohort_match["any_preliminary_dynamic_rainfall"]).sum()
        ),
        "selection_rule": "latest export timestamp embedded in filename",
        "preliminary_eligibility_rule": (
            "exact site-id match; >=30 days of time overlap; rainfall varies; "
            ">=80% nonmissing hourly coverage in overlap"
        ),
    }

    files = files.drop(columns=["export_timestamp_sort"], errors="ignore")
    files.to_csv(args.output_dir / "precipitation_file_inventory.csv", index=False, encoding="utf-8-sig")
    sites.to_csv(args.output_dir / "precipitation_site_inventory.csv", index=False, encoding="utf-8-sig")
    cohort_match.to_csv(args.output_dir / "common1000_precipitation_match.csv", index=False, encoding="utf-8-sig")
    cohort_match.loc[~cohort_match["any_preliminary_dynamic_rainfall"]].to_csv(
        args.output_dir / "common1000_precipitation_unresolved.csv",
        index=False,
        encoding="utf-8-sig",
    )
    network_summary.to_csv(args.output_dir / "network_summary.csv", index=False, encoding="utf-8-sig")
    source_summary.to_csv(args.output_dir / "source_summary.csv", index=False, encoding="utf-8-sig")
    match_status_summary.to_csv(args.output_dir / "match_status_summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
