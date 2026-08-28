"""Build read-only, model-time-aligned hourly rainfall sidecars for Common1000."""

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
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke-per-source",
        type=int,
        default=0,
        help="Deterministically use the first N sites from each rainfall source.",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


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


def bool_series(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False)


def canonical_sites(config: dict[str, Any], smoke_per_source: int) -> pd.DataFrame:
    audit = pd.read_csv(
        resolve(config["audit_match_path"]),
        dtype={"site_id": "string"},
        encoding="utf-8-sig",
    )
    eligible = bool_series(audit["preliminary_eligible_30d_80pct"])
    sites = audit.loc[eligible].copy()
    if smoke_per_source > 0:
        sites = (
            sites.sort_values(["source", "site_id"], kind="mergesort")
            .groupby("source", group_keys=False)
            .head(smoke_per_source)
            .copy()
        )
    sites = sites.sort_values(["source", "site_id"], kind="mergesort").reset_index(drop=True)
    expected = int(config["expected_primary_sites"])
    if smoke_per_source == 0 and len(sites) != expected:
        raise ValueError(f"Expected {expected} eligible sites, found {len(sites)}")

    split = pd.read_csv(
        resolve(config["site_split_path"]),
        dtype={"site_id": "string"},
        encoding="utf-8-sig",
    )
    split_columns = [
        "site_id",
        "split",
        "network",
        "hourly_rows",
        "target_observed_hours",
        "coverage_days",
        "start",
        "end",
    ]
    sites = sites.merge(
        split[split_columns],
        on="site_id",
        how="left",
        suffixes=("", "_prepared"),
        validate="one_to_one",
    )
    if sites["split"].isna().any():
        missing = sites.loc[sites["split"].isna(), "site_id"].tolist()
        raise ValueError(f"Eligible sites absent from pH split report: {missing[:10]}")
    return sites


def load_rainfall(path: Path) -> tuple[pd.Series, dict[str, Any]]:
    frame = pd.read_csv(
        path,
        usecols=["precipitation_mm", "year", "month", "day", "hour"],
        low_memory=False,
    )
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
    values = pd.to_numeric(frame["precipitation_mm"], errors="coerce")
    values = values.where(np.isfinite(values))
    negative_rows = int(values.lt(0).sum())
    values = values.mask(values.lt(0))
    valid = timestamps.notna()
    rainfall = pd.Series(
        values.loc[valid].to_numpy(dtype=np.float64),
        index=pd.DatetimeIndex(timestamps.loc[valid]),
        name="00045",
    )
    duplicate_rows = int(rainfall.index.duplicated(keep="last").sum())
    rainfall = rainfall[~rainfall.index.duplicated(keep="last")].sort_index()
    return rainfall, {
        "source_rows": int(len(frame)),
        "source_valid_timestamp_rows": int(valid.sum()),
        "source_invalid_timestamp_rows": int((~valid).sum()),
        "source_duplicate_timestamp_rows": duplicate_rows,
        "source_negative_rainfall_rows": negative_rows,
        "source_nonmissing_rainfall_rows": int(values.notna().sum()),
        "source_start_utc": rainfall.index.min().isoformat() if len(rainfall) else "",
        "source_end_utc": rainfall.index.max().isoformat() if len(rainfall) else "",
    }


def model_keys(path: Path) -> tuple[pd.Series, pd.DatetimeIndex]:
    frame = pd.read_parquet(path, columns=["timestamp"])
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"Invalid prepared timestamp rows: {int(timestamps.isna().sum())}")
    if timestamps.duplicated().any():
        raise ValueError(f"Duplicate prepared timestamps: {int(timestamps.duplicated().sum())}")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Prepared timestamps are not monotonically increasing")
    keys = pd.DatetimeIndex(timestamps)
    if keys.tz is None:
        utc_keys = keys.tz_localize("UTC")
    else:
        utc_keys = keys.tz_convert("UTC")
    return timestamps, utc_keys


def build_one(
    row: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    site_id = str(row["site_id"])
    source_path = Path(str(row["file_path"]))
    target_name = str(config["target_name"])
    prepared_path = (
        resolve(config["prepared_root"])
        / "hourly_native_by_site"
        / target_name
        / f"{site_id}.parquet"
    )
    output_path = output_root / "hourly_rainfall_by_site" / f"{site_id}.parquet"
    checkpoint_path = output_root / "checkpoints" / f"{site_id}.json"
    if resume and output_path.exists() and checkpoint_path.exists():
        return load_json(checkpoint_path)

    result: dict[str, Any] = {
        "site_id": site_id,
        "network": row.get("network"),
        "split": row.get("split"),
        "source": row.get("source"),
        "canonical_file": source_path.name,
        "canonical_file_path": str(source_path),
        "canonical_file_sha256": row.get("sha256"),
        "prepared_file": str(prepared_path),
        "sidecar_file": str(output_path),
        "rainfall_unit": config["rainfall_unit"],
        "timezone_assumption": config["timezone_assumption"],
        "status": "failed",
        "error_type": "",
        "error_message": "",
    }
    try:
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if not prepared_path.exists():
            raise FileNotFoundError(prepared_path)
        result["canonical_file_sha256"] = (
            str(row.get("sha256"))
            if row.get("sha256") not in (None, "", np.nan)
            else sha256(source_path)
        )
        rainfall, source_stats = load_rainfall(source_path)
        model_timestamps, utc_keys = model_keys(prepared_path)
        aligned_values = rainfall.reindex(utc_keys).to_numpy(dtype=np.float64)
        observed = np.isfinite(aligned_values)
        values = aligned_values.astype(np.float32)
        sidecar = pd.DataFrame(
            {
                "timestamp": model_timestamps.to_numpy(),
                "site_id": pd.Series([site_id] * len(model_timestamps), dtype="string"),
                "00045": values,
                "00045_observed_hour": observed,
                "00045_missing": (~observed).astype(np.int8),
            }
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        sidecar.to_parquet(
            temporary,
            index=False,
            compression=str(config["parquet_compression"]),
        )
        os.replace(temporary, output_path)

        model_start = utc_keys.min()
        model_end = utc_keys.max()
        rain_start = rainfall.index.min()
        rain_end = rainfall.index.max()
        overlap_start = max(model_start, rain_start)
        overlap_end = min(model_end, rain_end)
        expected_overlap_hours = (
            int((overlap_end - overlap_start).total_seconds() // 3600) + 1
            if overlap_end >= overlap_start
            else 0
        )
        overlap_mask = (utc_keys >= overlap_start) & (utc_keys <= overlap_end)
        overlap_observed = int(observed[overlap_mask].sum())
        nonmissing_values = values[observed]
        first_observed = utc_keys[observed].min() if observed.any() else pd.NaT
        last_observed = utc_keys[observed].max() if observed.any() else pd.NaT
        result.update(source_stats)
        result.update(
            {
                "status": "success",
                "model_hour_count": int(len(model_timestamps)),
                "model_start_utc": model_start.isoformat(),
                "model_end_utc": model_end.isoformat(),
                "aligned_observed_hours": int(observed.sum()),
                "aligned_missing_hours": int((~observed).sum()),
                "aligned_zero_hours": int(np.equal(nonmissing_values, 0).sum()),
                "aligned_positive_hours": int(np.greater(nonmissing_values, 0).sum()),
                "aligned_negative_hours": int(np.less(nonmissing_values, 0).sum()),
                "aligned_unique_rainfall_values": int(
                    pd.Series(nonmissing_values).nunique(dropna=True)
                ),
                "aligned_maximum_mm_per_hour": (
                    float(np.max(nonmissing_values)) if len(nonmissing_values) else None
                ),
                "aligned_observed_fraction_all_model_hours": (
                    float(observed.mean()) if len(observed) else 0.0
                ),
                "overlap_start_utc": overlap_start.isoformat() if expected_overlap_hours else "",
                "overlap_end_utc": overlap_end.isoformat() if expected_overlap_hours else "",
                "expected_overlap_hours": expected_overlap_hours,
                "observed_hours_within_overlap": overlap_observed,
                "observed_fraction_within_overlap": (
                    float(overlap_observed / expected_overlap_hours)
                    if expected_overlap_hours
                    else 0.0
                ),
                "first_aligned_observed_utc": (
                    first_observed.isoformat() if pd.notna(first_observed) else ""
                ),
                "last_aligned_observed_utc": (
                    last_observed.isoformat() if pd.notna(last_observed) else ""
                ),
                "sidecar_rows": int(len(sidecar)),
                "sidecar_size_bytes": int(output_path.stat().st_size),
                "sidecar_sha256": sha256(output_path),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
    write_json(checkpoint_path, result)
    return result


def grouped_summary(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    return (
        frame.groupby(field, dropna=False)
        .agg(
            sites=("site_id", "size"),
            success_sites=("status", lambda values: int(values.eq("success").sum())),
            model_hours=("model_hour_count", "sum"),
            observed_rainfall_hours=("aligned_observed_hours", "sum"),
            missing_rainfall_hours=("aligned_missing_hours", "sum"),
            positive_rainfall_hours=("aligned_positive_hours", "sum"),
            median_all_model_coverage=(
                "aligned_observed_fraction_all_model_hours",
                "median",
            ),
            minimum_all_model_coverage=(
                "aligned_observed_fraction_all_model_hours",
                "min",
            ),
            minimum_overlap_coverage=("observed_fraction_within_overlap", "min"),
        )
        .reset_index()
    )


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_json(config_path)
    sites = canonical_sites(config, int(args.smoke_per_source))
    base_output = resolve(config["output_root"])
    output_root = (
        base_output / f"smoke_{int(args.smoke_per_source)}_per_source"
        if args.smoke_per_source > 0
        else base_output / "full"
    )
    for directory in ("hourly_rainfall_by_site", "checkpoints", "reports"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)

    rows = sites.to_dict(orient="records")
    results = []
    workers = max(1, int(config["workers"]))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(build_one, row, config, output_root, bool(args.resume)): row["site_id"]
            for row in rows
        }
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                print(f"completed {completed}/{len(futures)}", flush=True)

    status = pd.DataFrame(results).sort_values("site_id", kind="mergesort")
    report_dir = output_root / "reports"
    write_csv(report_dir / "alignment_qa_by_site.csv", status)

    mapping_columns = [
        "site_id",
        "network",
        "split",
        "source",
        "canonical_file",
        "canonical_file_path",
        "canonical_file_sha256",
        "rainfall_unit",
        "timezone_assumption",
        "prepared_file",
        "sidecar_file",
        "status",
    ]
    write_csv(report_dir / "canonical_rainfall_mapping.csv", status[mapping_columns])
    for field, name in (
        ("source", "source_summary.csv"),
        ("network", "network_summary.csv"),
        ("split", "split_summary.csv"),
    ):
        write_csv(report_dir / name, grouped_summary(status, field))

    success = status["status"].eq("success")
    minimum_overlap_coverage = float(config["minimum_overlap_nonmissing_coverage"])
    minimum_overlap_hours = int(float(config["minimum_overlap_days"]) * 24)
    ready = (
        success
        & pd.to_numeric(status["expected_overlap_hours"], errors="coerce")
        .fillna(0)
        .ge(minimum_overlap_hours)
        & pd.to_numeric(status["observed_fraction_within_overlap"], errors="coerce")
        .fillna(0)
        .ge(minimum_overlap_coverage)
        & pd.to_numeric(status["aligned_unique_rainfall_values"], errors="coerce")
        .fillna(0)
        .gt(1)
    )
    readiness = status.loc[
        :,
        [
            "site_id",
            "network",
            "split",
            "source",
            "status",
            "model_hour_count",
            "aligned_observed_hours",
            "aligned_missing_hours",
            "aligned_observed_fraction_all_model_hours",
            "expected_overlap_hours",
            "observed_hours_within_overlap",
            "observed_fraction_within_overlap",
            "aligned_unique_rainfall_values",
            "aligned_maximum_mm_per_hour",
            "sidecar_file",
        ],
    ].copy()
    readiness["ready_for_common_key_experiment"] = ready.to_numpy()
    write_csv(report_dir / "common_key_readiness.csv", readiness)

    summary = {
        "config": str(config_path),
        "mode": "smoke" if args.smoke_per_source > 0 else "full",
        "requested_sites": int(len(status)),
        "success_sites": int(success.sum()),
        "failed_sites": int((~success).sum()),
        "ready_for_common_key_experiment": int(ready.sum()),
        "total_model_hours": int(
            pd.to_numeric(status["model_hour_count"], errors="coerce").fillna(0).sum()
        ),
        "total_aligned_observed_rainfall_hours": int(
            pd.to_numeric(status["aligned_observed_hours"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "total_aligned_missing_rainfall_hours": int(
            pd.to_numeric(status["aligned_missing_hours"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "total_aligned_positive_rainfall_hours": int(
            pd.to_numeric(status["aligned_positive_hours"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "sites_with_negative_aligned_rainfall": int(
            pd.to_numeric(status["aligned_negative_hours"], errors="coerce")
            .fillna(0)
            .gt(0)
            .sum()
        ),
        "all_model_hour_coverage_quantiles": {
            str(quantile): float(
                pd.to_numeric(
                    status["aligned_observed_fraction_all_model_hours"],
                    errors="coerce",
                ).quantile(quantile)
            )
            for quantile in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
        "within_overlap_coverage_quantiles": {
            str(quantile): float(
                pd.to_numeric(
                    status["observed_fraction_within_overlap"],
                    errors="coerce",
                ).quantile(quantile)
            )
            for quantile in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
        "sidecar_directory": str(output_root / "hourly_rainfall_by_site"),
        "raw_sources_modified": False,
        "prepared_water_quality_modified": False,
        "models_trained": False,
    }
    write_json(report_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if bool(success.all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
