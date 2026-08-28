"""Apply the teacher Archive rainfall features to all unique precipitation sites."""

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
    parser.add_argument(
        "--smoke-per-source",
        type=int,
        default=0,
        help="Use the first N canonical sites from each precipitation source.",
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


def canonical_sites(config: dict[str, Any], smoke_per_source: int) -> pd.DataFrame:
    inventory = pd.read_csv(
        resolve(config["site_inventory_path"]),
        dtype={"site_id": "string"},
        encoding="utf-8-sig",
    )
    canonical = (
        inventory["is_canonical_latest_export"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("true")
    )
    sites = inventory.loc[canonical].copy()
    if sites["site_id"].duplicated().any():
        duplicate_ids = sites.loc[sites["site_id"].duplicated(), "site_id"].tolist()
        raise ValueError(f"Canonical inventory contains duplicate site IDs: {duplicate_ids[:10]}")
    if smoke_per_source > 0:
        sites = (
            sites.sort_values(["source", "site_id"], kind="mergesort")
            .groupby("source", group_keys=False)
            .head(smoke_per_source)
            .copy()
        )
    sites = sites.sort_values(["source", "site_id"], kind="mergesort").reset_index(
        drop=True
    )
    expected = int(config["expected_unique_sites"])
    if smoke_per_source == 0 and len(sites) != expected:
        raise ValueError(f"Expected {expected} canonical sites, found {len(sites)}")
    return sites


def load_complete_hourly(path: Path) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
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
    values = values.where(np.isfinite(values) & values.ge(0))
    valid_timestamp = timestamps.notna()
    rainfall = pd.Series(
        values.loc[valid_timestamp].to_numpy(dtype=np.float64),
        index=pd.DatetimeIndex(timestamps.loc[valid_timestamp]),
        name="precip_mm",
    )
    duplicate_rows = int(rainfall.index.duplicated(keep="last").sum())
    rainfall = rainfall[~rainfall.index.duplicated(keep="last")].sort_index()
    if rainfall.empty:
        raise ValueError(f"No valid rainfall timestamps in {path}")
    full_index = pd.date_range(
        rainfall.index.min(),
        rainfall.index.max(),
        freq="h",
        tz="UTC",
    )
    complete = rainfall.reindex(full_index)
    observed = complete.notna()
    filled = complete.fillna(0.0)
    return complete, filled, {
        "source_rows": int(len(frame)),
        "valid_timestamp_rows": int(valid_timestamp.sum()),
        "invalid_timestamp_rows": int((~valid_timestamp).sum()),
        "duplicate_timestamp_rows": duplicate_rows,
        "complete_hour_count": int(len(complete)),
        "observed_rainfall_hours": int(observed.sum()),
        "source_missing_or_gap_hours_filled_zero": int((~observed).sum()),
        "source_start_utc": full_index.min().isoformat(),
        "source_end_utc": full_index.max().isoformat(),
    }


def build_one(
    row: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    site_id = str(row["site_id"])
    source_path = Path(str(row["file_path"]))
    output_path = output_root / "hourly_precip_features_by_site" / f"{site_id}.parquet"
    checkpoint_path = output_root / "checkpoints" / f"{site_id}.json"
    if resume and output_path.exists() and checkpoint_path.exists():
        return load_json(checkpoint_path)

    result: dict[str, Any] = {
        "site_id": site_id,
        "source": row.get("source"),
        "canonical_file": source_path.name,
        "canonical_file_path": str(source_path),
        "canonical_file_sha256": row.get("sha256"),
        "export_timestamp": row.get("export_timestamp"),
        "export_copy_count": row.get("export_copy_count"),
        "distinct_content_hash_count": row.get("distinct_content_hash_count"),
        "latest_export_timestamp": row.get("latest_export_timestamp"),
        "all_export_copies_identical": row.get("all_export_copies_identical"),
        "output_file": str(output_path),
        "status": "failed",
        "error_type": "",
        "error_message": "",
        "teacher_archive_exact_formula": True,
        "api_halflife_hours": float(config["api_halflife_hours"]),
        "rolling_window_hours": int(config["rolling_window_hours"]),
        "missing_rainfall_fill_value": float(config["missing_rainfall_fill_value"]),
    }
    try:
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        complete, rainfall, source_stats = load_complete_hourly(source_path)
        observed = complete.notna().to_numpy(dtype=bool)

        # Exact feature definitions from Archive/data_and_metadata_snippets.py.
        precip_api = rainfall.ewm(
            halflife=float(config["api_halflife_hours"]),
            adjust=False,
        ).mean()
        precip_6h = rainfall.rolling(
            int(config["rolling_window_hours"]),
            min_periods=1,
        ).sum()

        rainfall_values = rainfall.to_numpy(dtype=np.float32)
        api_values = precip_api.to_numpy(dtype=np.float32)
        rolling_values = precip_6h.to_numpy(dtype=np.float32)
        output = pd.DataFrame(
            {
                str(config["timestamp_column"]): rainfall.index,
                "site_id": pd.Series([site_id] * len(rainfall), dtype="string"),
                str(config["rainfall_input_column"]): rainfall_values,
                str(config["rainfall_model_column"]): rainfall_values,
                "00045_observed_hour": observed,
                "00045_source_missing": (~observed).astype(np.int8),
                str(config["api_output_column"]): api_values,
                str(config["rolling_output_column"]): rolling_values,
            }
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        output.to_parquet(
            temporary,
            index=False,
            compression=str(config["parquet_compression"]),
        )
        os.replace(temporary, output_path)

        result.update(source_stats)
        result.update(
            {
                "status": "success",
                "filled_rainfall_zero_hours": int(np.equal(rainfall_values, 0).sum()),
                "filled_rainfall_positive_hours": int(np.greater(rainfall_values, 0).sum()),
                "filled_rainfall_negative_hours": int(np.less(rainfall_values, 0).sum()),
                "precip_api_nonmissing_hours": int(np.isfinite(api_values).sum()),
                "precip_api_minimum": float(np.min(api_values)),
                "precip_api_maximum": float(np.max(api_values)),
                "precip_6h_nonmissing_hours": int(np.isfinite(rolling_values).sum()),
                "precip_6h_minimum": float(np.min(rolling_values)),
                "precip_6h_maximum": float(np.max(rolling_values)),
                "output_rows": int(len(output)),
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
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
    write_json(checkpoint_path, result)
    return result


def source_summary(status: pd.DataFrame) -> pd.DataFrame:
    return (
        status.groupby("source", dropna=False)
        .agg(
            sites=("site_id", "size"),
            successful_sites=("status", lambda values: int(values.eq("success").sum())),
            source_rows=("source_rows", "sum"),
            complete_hours=("complete_hour_count", "sum"),
            observed_rainfall_hours=("observed_rainfall_hours", "sum"),
            filled_missing_or_gap_hours=(
                "source_missing_or_gap_hours_filled_zero",
                "sum",
            ),
            zero_rainfall_hours=("filled_rainfall_zero_hours", "sum"),
            positive_rainfall_hours=("filled_rainfall_positive_hours", "sum"),
            negative_rainfall_hours=("filled_rainfall_negative_hours", "sum"),
            precip_api_nonmissing_hours=("precip_api_nonmissing_hours", "sum"),
            precip_6h_nonmissing_hours=("precip_6h_nonmissing_hours", "sum"),
            minimum_precip_api=("precip_api_minimum", "min"),
            maximum_precip_api=("precip_api_maximum", "max"),
            maximum_precip_6h=("precip_6h_maximum", "max"),
            output_size_bytes=("output_size_bytes", "sum"),
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
    for directory in ("hourly_precip_features_by_site", "checkpoints", "reports"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)

    rows = sites.to_dict(orient="records")
    results: list[dict[str, Any]] = []
    workers = max(1, int(config["workers"]))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                build_one,
                row,
                config,
                output_root,
                bool(args.resume),
            ): row["site_id"]
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
    write_csv(report_dir / "processing_qa_by_site.csv", status)
    write_csv(report_dir / "source_summary.csv", source_summary(status))
    mapping_columns = [
        "site_id",
        "source",
        "canonical_file",
        "canonical_file_path",
        "canonical_file_sha256",
        "export_timestamp",
        "export_copy_count",
        "distinct_content_hash_count",
        "latest_export_timestamp",
        "all_export_copies_identical",
        "output_file",
        "status",
    ]
    write_csv(report_dir / "canonical_mapping.csv", status[mapping_columns])
    write_csv(
        report_dir / "duplicate_export_sites.csv",
        status.loc[
            pd.to_numeric(status["export_copy_count"], errors="coerce").fillna(0).gt(1),
            mapping_columns,
        ],
    )
    parameters = pd.DataFrame(
        [
            {
                "teacher_archive_source": config["teacher_archive_source"],
                "api_output_column": config["api_output_column"],
                "api_halflife_hours": config["api_halflife_hours"],
                "api_formula": config["api_formula"],
                "rolling_output_column": config["rolling_output_column"],
                "rolling_window_hours": config["rolling_window_hours"],
                "rolling_formula": config["rolling_formula"],
                "missing_rainfall_fill_value": config["missing_rainfall_fill_value"],
                "canonical_selection_rule": config["canonical_selection_rule"],
            }
        ]
    )
    write_csv(report_dir / "parameters.csv", parameters)

    success = status["status"].eq("success")
    summary = {
        "config": str(config_path),
        "mode": "smoke" if args.smoke_per_source > 0 else "full",
        "requested_unique_sites": int(len(status)),
        "successful_sites": int(success.sum()),
        "failed_sites": int((~success).sum()),
        "source_site_counts": {
            str(source): int(count)
            for source, count in status.groupby("source").size().items()
        },
        "total_complete_hours": int(
            pd.to_numeric(status["complete_hour_count"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "total_filled_missing_or_gap_hours": int(
            pd.to_numeric(
                status["source_missing_or_gap_hours_filled_zero"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "sites_with_repeated_exports": int(
            pd.to_numeric(status["export_copy_count"], errors="coerce")
            .fillna(0)
            .gt(1)
            .sum()
        ),
        "teacher_archive_formula": {
            "precip_api": config["api_formula"],
            "precip_6h": config["rolling_formula"],
        },
        "output_directory": str(output_root / "hourly_precip_features_by_site"),
        "raw_sources_modified": False,
        "water_quality_data_modified": False,
        "models_trained": False,
    }
    write_json(report_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if bool(success.all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
