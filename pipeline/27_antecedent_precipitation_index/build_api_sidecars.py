"""Build causal antecedent precipitation index sidecars for Common1000."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import lfilter


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


def api_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    warmup_half_lives = float(config["warmup_half_lives"])
    for item in config["api_variants"]:
        name = str(item["name"])
        half_life = int(item["half_life_hours"])
        if half_life <= 0:
            raise ValueError(f"Invalid half-life for {name}: {half_life}")
        variants.append(
            {
                "name": name,
                "half_life_hours": half_life,
                "alpha": float(0.5 ** (1.0 / half_life)),
                "warmup_hours": int(math.ceil(half_life * warmup_half_lives)),
            }
        )
    names = [item["name"] for item in variants]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate API names: {names}")
    return variants


def canonical_sites(config: dict[str, Any], smoke_per_source: int) -> pd.DataFrame:
    report = pd.read_csv(
        resolve(config["rainfall_alignment_report"]),
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
    expected = int(config["expected_primary_sites"])
    if smoke_per_source == 0 and len(sites) != expected:
        raise ValueError(f"Expected {expected} successful sites, found {len(sites)}")
    return sites


def load_rainfall(path: Path) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
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
        name="00045",
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
    full_rainfall = rainfall.reindex(full_index)
    observed = full_rainfall.notna()
    starts = observed & ~observed.shift(fill_value=False)
    segment_id = starts.cumsum().where(observed)
    segment_sizes = full_rainfall.groupby(segment_id, dropna=True).size()
    stats = {
        "source_rows": int(len(frame)),
        "source_valid_timestamp_rows": int(valid_timestamp.sum()),
        "source_duplicate_timestamp_rows": duplicate_rows,
        "source_full_hour_count": int(len(full_rainfall)),
        "source_gap_or_missing_hours": int(full_rainfall.isna().sum()),
        "source_observed_segments": int(segment_sizes.size),
        "source_shortest_observed_segment_hours": (
            int(segment_sizes.min()) if not segment_sizes.empty else 0
        ),
        "source_longest_observed_segment_hours": (
            int(segment_sizes.max()) if not segment_sizes.empty else 0
        ),
        "source_start_utc": full_index.min().isoformat(),
        "source_end_utc": full_index.max().isoformat(),
    }
    return rainfall, full_rainfall, stats


def model_keys(path: Path) -> tuple[pd.Series, pd.DatetimeIndex]:
    frame = pd.read_parquet(path, columns=["timestamp"])
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"Invalid prepared timestamp rows: {int(timestamps.isna().sum())}")
    if timestamps.duplicated().any():
        raise ValueError(
            f"Duplicate prepared timestamps: {int(timestamps.duplicated().sum())}"
        )
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Prepared timestamps are not monotonically increasing")
    keys = pd.DatetimeIndex(timestamps)
    utc_keys = keys.tz_localize("UTC") if keys.tz is None else keys.tz_convert("UTC")
    return timestamps, utc_keys


def observed_segments(observed: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], observed.astype(bool), [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[0::2], changes[1::2], strict=True))


def compute_api(
    full_rainfall: pd.Series,
    alpha: float,
    warmup_hours: int,
) -> tuple[pd.Series, pd.Series]:
    rainfall_values = full_rainfall.to_numpy(dtype=np.float64)
    observed = np.isfinite(rainfall_values)
    api_values = np.full(len(rainfall_values), np.nan, dtype=np.float64)
    warmup = np.zeros(len(rainfall_values), dtype=bool)
    for start, end in observed_segments(observed):
        segment = rainfall_values[start:end]
        recurrence = lfilter([1.0], [1.0, -alpha], segment)
        positions = np.arange(len(segment))
        segment_warmup = positions < warmup_hours
        warmup[start:end] = segment_warmup
        valid = ~segment_warmup
        api_values[start:end][valid] = recurrence[valid]
    return (
        pd.Series(api_values, index=full_rainfall.index),
        pd.Series(warmup, index=full_rainfall.index),
    )


def validate_base_sidecar(
    base_path: Path,
    model_timestamps: pd.Series,
    aligned_rainfall: np.ndarray,
) -> None:
    base = pd.read_parquet(
        base_path,
        columns=[
            "timestamp",
            "00045",
            "00045_observed_hour",
            "00045_missing",
        ],
    )
    base_timestamps = pd.to_datetime(base["timestamp"], errors="coerce")
    if not base_timestamps.equals(model_timestamps):
        raise ValueError("Stage-26 rainfall timestamps differ from prepared timestamps")
    expected = aligned_rainfall.astype(np.float32)
    actual = pd.to_numeric(base["00045"], errors="coerce").to_numpy(dtype=np.float32)
    if not np.array_equal(np.isnan(expected), np.isnan(actual)):
        raise ValueError("Stage-26 rainfall missingness differs from canonical source")
    finite = np.isfinite(expected)
    if finite.any() and not np.array_equal(expected[finite], actual[finite]):
        raise ValueError("Stage-26 rainfall values differ from canonical source")
    observed = np.isfinite(expected)
    if not np.array_equal(
        base["00045_observed_hour"].astype(bool).to_numpy(),
        observed,
    ):
        raise ValueError("Stage-26 observed flag is inconsistent")
    if not np.array_equal(
        pd.to_numeric(base["00045_missing"], errors="coerce").to_numpy(dtype=np.int8),
        (~observed).astype(np.int8),
    ):
        raise ValueError("Stage-26 missing flag is inconsistent")


def build_one(
    row: dict[str, Any],
    config: dict[str, Any],
    variants: list[dict[str, Any]],
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    site_id = str(row["site_id"])
    source_path = Path(str(row["canonical_file_path"]))
    prepared_path = Path(str(row["prepared_file"]))
    base_sidecar_path = Path(str(row["sidecar_file"]))
    output_path = output_root / "hourly_api_by_site" / f"{site_id}.parquet"
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
        "canonical_file_sha256": row.get("canonical_file_sha256"),
        "prepared_file": str(prepared_path),
        "base_rainfall_sidecar": str(base_sidecar_path),
        "api_sidecar_file": str(output_path),
        "rainfall_unit": config["rainfall_unit"],
        "timezone_assumption": config["timezone_assumption"],
        "status": "failed",
        "error_type": "",
        "error_message": "",
    }
    for variant in variants:
        name = variant["name"]
        result.update(
            {
                f"{name}_half_life_hours": int(variant["half_life_hours"]),
                f"{name}_alpha": float(variant["alpha"]),
                f"{name}_warmup_required_hours": int(variant["warmup_hours"]),
                f"{name}_valid_hours": 0,
                f"{name}_missing_hours": 0,
                f"{name}_warmup_hours_on_model_axis": 0,
                f"{name}_valid_fraction_all_model_hours": 0.0,
                f"{name}_valid_fraction_observed_rainfall_hours": 0.0,
                f"{name}_unique_values": 0,
                f"{name}_minimum": None,
                f"{name}_maximum": None,
                f"{name}_negative_hours": 0,
                f"{name}_first_valid_utc": "",
                f"{name}_last_valid_utc": "",
            }
        )

    try:
        for required in (source_path, prepared_path, base_sidecar_path):
            if not required.exists():
                raise FileNotFoundError(required)

        rainfall, full_rainfall, source_stats = load_rainfall(source_path)
        model_timestamps, utc_keys = model_keys(prepared_path)
        aligned_rainfall = rainfall.reindex(utc_keys).to_numpy(dtype=np.float64)
        validate_base_sidecar(
            base_sidecar_path,
            model_timestamps,
            aligned_rainfall,
        )

        observed_rainfall = np.isfinite(aligned_rainfall)
        sidecar_data: dict[str, Any] = {
            "timestamp": model_timestamps.to_numpy(),
            "site_id": pd.Series([site_id] * len(model_timestamps), dtype="string"),
            "00045": aligned_rainfall.astype(np.float32),
            "00045_observed_hour": observed_rainfall,
            "00045_missing": (~observed_rainfall).astype(np.int8),
        }
        api_valid_masks: list[np.ndarray] = []

        for variant in variants:
            name = variant["name"]
            api, warmup = compute_api(
                full_rainfall,
                alpha=float(variant["alpha"]),
                warmup_hours=int(variant["warmup_hours"]),
            )
            aligned_api = api.reindex(utc_keys).to_numpy(dtype=np.float64)
            aligned_warmup = (
                warmup.reindex(utc_keys, fill_value=False).to_numpy(dtype=bool)
            )
            valid_api = np.isfinite(aligned_api)
            api_valid_masks.append(valid_api)
            output_api = aligned_api.astype(np.float32)
            sidecar_data[name] = output_api
            sidecar_data[f"{name}_valid"] = valid_api
            sidecar_data[f"{name}_missing"] = (~valid_api).astype(np.int8)
            sidecar_data[f"{name}_warmup"] = aligned_warmup

            finite_values = output_api[valid_api]
            first_valid = utc_keys[valid_api].min() if valid_api.any() else pd.NaT
            last_valid = utc_keys[valid_api].max() if valid_api.any() else pd.NaT
            valid_hours = int(valid_api.sum())
            result.update(
                {
                    f"{name}_valid_hours": valid_hours,
                    f"{name}_missing_hours": int((~valid_api).sum()),
                    f"{name}_warmup_hours_on_model_axis": int(aligned_warmup.sum()),
                    f"{name}_valid_fraction_all_model_hours": (
                        float(valid_api.mean()) if len(valid_api) else 0.0
                    ),
                    f"{name}_valid_fraction_observed_rainfall_hours": (
                        float(valid_hours / observed_rainfall.sum())
                        if observed_rainfall.any()
                        else 0.0
                    ),
                    f"{name}_unique_values": int(
                        pd.Series(finite_values).nunique(dropna=True)
                    ),
                    f"{name}_minimum": (
                        float(np.min(finite_values)) if len(finite_values) else None
                    ),
                    f"{name}_maximum": (
                        float(np.max(finite_values)) if len(finite_values) else None
                    ),
                    f"{name}_negative_hours": int(
                        np.less(finite_values, 0).sum()
                    ),
                    f"{name}_first_valid_utc": (
                        first_valid.isoformat() if pd.notna(first_valid) else ""
                    ),
                    f"{name}_last_valid_utc": (
                        last_valid.isoformat() if pd.notna(last_valid) else ""
                    ),
                }
            )

        common_valid = np.logical_and.reduce(api_valid_masks)
        first_common = utc_keys[common_valid].min() if common_valid.any() else pd.NaT
        last_common = utc_keys[common_valid].max() if common_valid.any() else pd.NaT
        sidecar = pd.DataFrame(sidecar_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        sidecar.to_parquet(
            temporary,
            index=False,
            compression=str(config["parquet_compression"]),
        )
        os.replace(temporary, output_path)

        result.update(source_stats)
        result.update(
            {
                "status": "success",
                "model_hour_count": int(len(model_timestamps)),
                "model_start_utc": utc_keys.min().isoformat(),
                "model_end_utc": utc_keys.max().isoformat(),
                "aligned_observed_rainfall_hours": int(observed_rainfall.sum()),
                "aligned_missing_rainfall_hours": int((~observed_rainfall).sum()),
                "common_all_scale_valid_hours": int(common_valid.sum()),
                "common_all_scale_valid_fraction_all_model_hours": (
                    float(common_valid.mean()) if len(common_valid) else 0.0
                ),
                "common_all_scale_first_valid_utc": (
                    first_common.isoformat() if pd.notna(first_common) else ""
                ),
                "common_all_scale_last_valid_utc": (
                    last_common.isoformat() if pd.notna(last_common) else ""
                ),
                "api_sidecar_rows": int(len(sidecar)),
                "api_sidecar_size_bytes": int(output_path.stat().st_size),
                "api_sidecar_sha256": sha256(output_path),
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


def coverage_long(
    status: pd.DataFrame,
    variants: list[dict[str, Any]],
    minimum_valid_hours: int,
) -> pd.DataFrame:
    def integer_or_zero(value: Any) -> int:
        numeric = pd.to_numeric(value, errors="coerce")
        return 0 if pd.isna(numeric) else int(numeric)

    rows: list[dict[str, Any]] = []
    for _, site in status.iterrows():
        for variant in variants:
            name = variant["name"]
            valid_hours = integer_or_zero(site.get(f"{name}_valid_hours"))
            unique_values = integer_or_zero(site.get(f"{name}_unique_values"))
            negative_hours = integer_or_zero(site.get(f"{name}_negative_hours"))
            if site.get("status") != "success":
                readiness_reason = "build failed"
            elif negative_hours > 0:
                readiness_reason = "negative API value"
            elif unique_values <= 1:
                readiness_reason = "API has no variation"
            elif valid_hours < minimum_valid_hours:
                readiness_reason = (
                    f"fewer than {minimum_valid_hours} valid hours after warm-up"
                )
            else:
                readiness_reason = "ready"
            rows.append(
                {
                    "site_id": site["site_id"],
                    "network": site.get("network"),
                    "split": site.get("split"),
                    "source": site.get("source"),
                    "status": site.get("status"),
                    "api_feature": name,
                    "half_life_hours": int(variant["half_life_hours"]),
                    "alpha": float(variant["alpha"]),
                    "warmup_required_hours": int(variant["warmup_hours"]),
                    "model_hours": site.get("model_hour_count"),
                    "observed_rainfall_hours": site.get(
                        "aligned_observed_rainfall_hours"
                    ),
                    "api_valid_hours": valid_hours,
                    "api_missing_hours": site.get(f"{name}_missing_hours"),
                    "api_warmup_hours_on_model_axis": site.get(
                        f"{name}_warmup_hours_on_model_axis"
                    ),
                    "api_valid_fraction_all_model_hours": site.get(
                        f"{name}_valid_fraction_all_model_hours"
                    ),
                    "api_valid_fraction_observed_rainfall_hours": site.get(
                        f"{name}_valid_fraction_observed_rainfall_hours"
                    ),
                    "api_unique_values": unique_values,
                    "api_minimum": site.get(f"{name}_minimum"),
                    "api_maximum": site.get(f"{name}_maximum"),
                    "api_negative_hours": negative_hours,
                    "first_valid_utc": site.get(f"{name}_first_valid_utc"),
                    "last_valid_utc": site.get(f"{name}_last_valid_utc"),
                    "ready_for_experiment": bool(
                        site.get("status") == "success"
                        and valid_hours >= minimum_valid_hours
                        and unique_values > 1
                        and negative_hours == 0
                    ),
                    "readiness_reason": readiness_reason,
                }
            )
    return pd.DataFrame(rows)


def grouped_scale_summary(long: pd.DataFrame, field: str | None) -> pd.DataFrame:
    group_fields = ["api_feature"] if field is None else [field, "api_feature"]
    return (
        long.groupby(group_fields, dropna=False)
        .agg(
            sites=("site_id", "size"),
            successful_sites=("status", lambda values: int(values.eq("success").sum())),
            ready_sites=("ready_for_experiment", "sum"),
            not_ready_sites=(
                "ready_for_experiment",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            zero_valid_hour_sites=(
                "api_valid_hours",
                lambda values: int(pd.to_numeric(values, errors="coerce").fillna(0).eq(0).sum()),
            ),
            below_80pct_full_axis_sites=(
                "api_valid_fraction_all_model_hours",
                lambda values: int(
                    pd.to_numeric(values, errors="coerce").fillna(0).lt(0.8).sum()
                ),
            ),
            model_hours=("model_hours", "sum"),
            observed_rainfall_hours=("observed_rainfall_hours", "sum"),
            api_valid_hours=("api_valid_hours", "sum"),
            api_missing_hours=("api_missing_hours", "sum"),
            warmup_hours_on_model_axis=("api_warmup_hours_on_model_axis", "sum"),
            median_full_axis_coverage=(
                "api_valid_fraction_all_model_hours",
                "median",
            ),
            minimum_full_axis_coverage=(
                "api_valid_fraction_all_model_hours",
                "min",
            ),
            median_observed_rainfall_coverage=(
                "api_valid_fraction_observed_rainfall_hours",
                "median",
            ),
            minimum_observed_rainfall_coverage=(
                "api_valid_fraction_observed_rainfall_hours",
                "min",
            ),
            negative_api_hours=("api_negative_hours", "sum"),
        )
        .reset_index()
    )


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_json(config_path)
    variants = api_variants(config)
    sites = canonical_sites(config, int(args.smoke_per_source))
    base_output = resolve(config["output_root"])
    output_root = (
        base_output / f"smoke_{int(args.smoke_per_source)}_per_source"
        if args.smoke_per_source > 0
        else base_output / "full"
    )
    for directory in ("hourly_api_by_site", "checkpoints", "reports"):
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
                variants,
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
    write_csv(report_dir / "api_qa_by_site.csv", status)

    minimum_valid_hours = int(config["minimum_valid_hours_for_experiment"])
    long = coverage_long(status, variants, minimum_valid_hours)
    write_csv(report_dir / "api_coverage_long.csv", long)
    parameters = pd.DataFrame(
        [
            {
                "api_feature": variant["name"],
                "half_life_hours": int(variant["half_life_hours"]),
                "alpha": float(variant["alpha"]),
                "warmup_half_lives": float(config["warmup_half_lives"]),
                "warmup_required_hours": int(variant["warmup_hours"]),
                "warmup_required_days": float(variant["warmup_hours"] / 24),
                "minimum_valid_hours_for_experiment": minimum_valid_hours,
                "minimum_valid_days_for_experiment": float(
                    minimum_valid_hours / 24
                ),
                "formula": config["api_definition"],
            }
            for variant in variants
        ]
    )
    write_csv(report_dir / "api_parameters.csv", parameters)
    write_csv(
        report_dir / "api_not_ready_by_scale.csv",
        long.loc[~long["ready_for_experiment"]].copy(),
    )
    write_csv(
        report_dir / "api_low_coverage_by_scale.csv",
        long.loc[long["api_valid_fraction_all_model_hours"].lt(0.8)].copy(),
    )
    write_csv(
        report_dir / "api_scale_summary.csv",
        grouped_scale_summary(long, None),
    )
    for field in ("source", "network", "split"):
        write_csv(
            report_dir / f"{field}_scale_summary.csv",
            grouped_scale_summary(long, field),
        )

    readiness_columns = [
        "site_id",
        "network",
        "split",
        "source",
        "status",
        "model_hour_count",
        "aligned_observed_rainfall_hours",
        "aligned_missing_rainfall_hours",
        "common_all_scale_valid_hours",
        "common_all_scale_valid_fraction_all_model_hours",
        "common_all_scale_first_valid_utc",
        "common_all_scale_last_valid_utc",
        "api_sidecar_file",
    ]
    readiness = status.loc[:, readiness_columns].copy()
    ready_columns: list[str] = []
    for variant in variants:
        name = variant["name"]
        for suffix in (
            "valid_hours",
            "missing_hours",
            "warmup_hours_on_model_axis",
            "valid_fraction_all_model_hours",
            "valid_fraction_observed_rainfall_hours",
            "first_valid_utc",
            "last_valid_utc",
        ):
            readiness[f"{name}_{suffix}"] = status[f"{name}_{suffix}"]
        per_scale = long.loc[
            long["api_feature"].eq(name),
            ["site_id", "ready_for_experiment"],
        ].set_index("site_id")["ready_for_experiment"]
        ready_column = f"{name}_ready_for_experiment"
        readiness[ready_column] = readiness["site_id"].map(per_scale).fillna(False)
        ready_columns.append(ready_column)
    readiness["all_scales_ready_for_experiment"] = readiness[ready_columns].all(axis=1)
    write_csv(report_dir / "api_readiness_by_site.csv", readiness)
    write_csv(
        report_dir / "api_all_scales_not_ready.csv",
        readiness.loc[~readiness["all_scales_ready_for_experiment"]].copy(),
    )

    success = status["status"].eq("success")
    scale_summary = pd.read_csv(report_dir / "api_scale_summary.csv")
    summary = {
        "config": str(config_path),
        "mode": "smoke" if args.smoke_per_source > 0 else "full",
        "requested_sites": int(len(status)),
        "success_sites": int(success.sum()),
        "failed_sites": int((~success).sum()),
        "api_definition": config["api_definition"],
        "decay_definition": config["decay_definition"],
        "warmup_half_lives": float(config["warmup_half_lives"]),
        "minimum_valid_hours_for_experiment": minimum_valid_hours,
        "scale_ready_sites": {
            str(row["api_feature"]): int(row["ready_sites"])
            for _, row in scale_summary.iterrows()
        },
        "all_scales_ready_sites": int(
            readiness["all_scales_ready_for_experiment"].sum()
        ),
        "total_model_hours": int(
            pd.to_numeric(status["model_hour_count"], errors="coerce").fillna(0).sum()
        ),
        "total_observed_rainfall_hours": int(
            pd.to_numeric(
                status["aligned_observed_rainfall_hours"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "scale_totals": {
            str(row["api_feature"]): {
                "valid_hours": int(row["api_valid_hours"]),
                "missing_hours": int(row["api_missing_hours"]),
                "warmup_hours_on_model_axis": int(
                    row["warmup_hours_on_model_axis"]
                ),
                "negative_api_hours": int(row["negative_api_hours"]),
            }
            for _, row in scale_summary.iterrows()
        },
        "api_sidecar_directory": str(output_root / "hourly_api_by_site"),
        "raw_sources_modified": False,
        "prepared_water_quality_modified": False,
        "base_rainfall_sidecars_modified": False,
        "models_trained": False,
    }
    write_json(report_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if bool(success.all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
