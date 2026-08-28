"""Extend prior-format hourly rainfall sidecars to all canonical rain stations."""

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
    parser.add_argument("--smoke-per-source", type=int, default=0)
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


def selected_sites(config: dict[str, Any], smoke_per_source: int) -> pd.DataFrame:
    report = pd.read_csv(
        resolve(config["teacher_archive_processing_report"]),
        dtype={"site_id": "string"},
        encoding="utf-8-sig",
    )
    sites = report.loc[report["status"].eq("success")].copy()
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
    expected = int(config["expected_sites"])
    if smoke_per_source == 0 and len(sites) != expected:
        raise ValueError(f"Expected {expected} sites, found {len(sites)}")
    if sites["site_id"].duplicated().any():
        raise ValueError("Site report contains duplicate station identifiers")
    return sites


def build_one(
    row: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    site_id = str(row["site_id"])
    source_path = Path(str(row["canonical_file_path"]))
    teacher_path = Path(str(row["output_file"]))
    output_path = output_root / "hourly_rainfall_by_site" / f"{site_id}.parquet"
    checkpoint_path = output_root / "checkpoints" / f"{site_id}.json"
    if resume and output_path.exists() and checkpoint_path.exists():
        return load_json(checkpoint_path)

    result: dict[str, Any] = {
        "site_id": site_id,
        "network": "all_precipitation_stations",
        "split": "all",
        "source": row.get("source"),
        "canonical_file": source_path.name,
        "canonical_file_path": str(source_path),
        "canonical_file_sha256": row.get("canonical_file_sha256"),
        "prepared_file": str(output_path),
        "sidecar_file": str(output_path),
        "teacher_archive_file": str(teacher_path),
        "status": "failed",
        "error_type": "",
        "error_message": "",
    }
    try:
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if not teacher_path.exists():
            raise FileNotFoundError(teacher_path)
        source = pd.read_parquet(
            teacher_path,
            columns=[
                "TIMESTAMP",
                "site_id",
                "00045",
                "00045_observed_hour",
                "00045_source_missing",
            ],
        )
        timestamp = pd.to_datetime(source["TIMESTAMP"], errors="coerce", utc=True)
        if timestamp.isna().any():
            raise ValueError("Teacher-Archive output contains invalid timestamps")
        timestamp = timestamp.dt.tz_convert("UTC").dt.tz_localize(None)
        rainfall = pd.to_numeric(source["00045"], errors="coerce").astype("float32")
        observed = source["00045_observed_hour"].astype(bool).to_numpy()
        missing = source["00045_source_missing"].astype(bool).to_numpy()
        if np.any(observed == missing):
            raise ValueError("Observed and missing flags are not complements")
        if rainfall.isna().to_numpy().any():
            raise ValueError("Canonical full-hour rainfall unexpectedly contains nulls")
        if (rainfall.to_numpy() < 0).any():
            raise ValueError("Canonical full-hour rainfall contains negative values")
        if source["site_id"].astype(str).ne(site_id).any():
            raise ValueError("Station identifier differs inside source feature file")

        sidecar = pd.DataFrame(
            {
                "timestamp": timestamp.to_numpy(),
                "site_id": pd.Series([site_id] * len(source), dtype="string"),
                "00045": rainfall.to_numpy(dtype=np.float32),
                "00045_observed_hour": observed,
                "00045_missing": missing.astype(np.int8),
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
        result.update(
            {
                "status": "success",
                "sidecar_rows": int(len(sidecar)),
                "start_utc": timestamp.min().isoformat(),
                "end_utc": timestamp.max().isoformat(),
                "observed_rainfall_hours": int(observed.sum()),
                "missing_rainfall_hours": int(missing.sum()),
                "zero_rainfall_hours": int(rainfall.eq(0).sum()),
                "positive_rainfall_hours": int(rainfall.gt(0).sum()),
                "negative_rainfall_hours": int(rainfall.lt(0).sum()),
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


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_json(config_path)
    sites = selected_sites(config, int(args.smoke_per_source))
    base_output = resolve(config["output_root"])
    output_root = (
        base_output / f"smoke_{int(args.smoke_per_source)}_per_source"
        if args.smoke_per_source > 0
        else base_output / "full"
    )
    for name in ("hourly_rainfall_by_site", "checkpoints", "reports"):
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
            for row in sites.to_dict(orient="records")
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"completed {completed}/{len(futures)}", flush=True)

    status = pd.DataFrame(results).sort_values("site_id", kind="mergesort")
    reports = output_root / "reports"
    write_csv(reports / "alignment_qa_by_site.csv", status)
    source_summary = (
        status.groupby("source", dropna=False)
        .agg(
            sites=("site_id", "size"),
            successful_sites=("status", lambda values: int(values.eq("success").sum())),
            sidecar_rows=("sidecar_rows", "sum"),
            observed_rainfall_hours=("observed_rainfall_hours", "sum"),
            missing_rainfall_hours=("missing_rainfall_hours", "sum"),
            negative_rainfall_hours=("negative_rainfall_hours", "sum"),
        )
        .reset_index()
    )
    write_csv(reports / "source_summary.csv", source_summary)
    success = status["status"].eq("success")
    summary = {
        "config": str(config_path),
        "mode": "smoke" if args.smoke_per_source > 0 else "full",
        "requested_sites": int(len(status)),
        "successful_sites": int(success.sum()),
        "failed_sites": int((~success).sum()),
        "source_site_counts": {
            str(source): int(count)
            for source, count in status.groupby("source").size().items()
        },
        "total_hours": int(
            pd.to_numeric(status["sidecar_rows"], errors="coerce").fillna(0).sum()
        ),
        "total_missing_rainfall_hours": int(
            pd.to_numeric(status["missing_rainfall_hours"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "total_negative_rainfall_hours": int(
            pd.to_numeric(status["negative_rainfall_hours"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "sidecar_directory": str(output_root / "hourly_rainfall_by_site"),
        "existing_910_site_outputs_modified": False,
        "raw_sources_modified": False,
        "models_trained": False,
    }
    write_json(reports / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if bool(success.all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
