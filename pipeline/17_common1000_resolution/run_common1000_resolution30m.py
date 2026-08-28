from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Common1000 30-minute GRU-FiLM experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, relative: str) -> Any:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def direction(low: bool, high: bool, mixed: bool) -> str:
    if mixed or (low and high):
        return "mixed"
    if low:
        return "low"
    if high:
        return "high"
    return "none"


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(frame["timestamp"])
    fractional_hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    frame["hour"] = fractional_hour.astype(np.float32)
    frame["month"] = timestamp.dt.month.astype(np.int16)
    frame["day_of_year"] = timestamp.dt.dayofyear.astype(np.int16)
    frame["sin_hour"] = np.sin(2 * np.pi * fractional_hour / 24).astype(np.float32)
    frame["cos_hour"] = np.cos(2 * np.pi * fractional_hour / 24).astype(np.float32)
    frame["sin_day"] = np.sin(2 * np.pi * frame["day_of_year"] / 365.25).astype(np.float32)
    frame["cos_day"] = np.cos(2 * np.pi * frame["day_of_year"] / 365.25).astype(np.float32)
    return frame


def aggregate_resolution(raw: pd.DataFrame, site_id: str, config: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    frequency = f"{int(config['resolution_minutes'])}min"
    sensors = ["00010", "00095", "00300", "63680"]
    target = str(config["target_scaled"])
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")
    raw["bin_timestamp"] = raw["timestamp"].dt.floor(frequency)
    raw["event_low"] = raw["ph_event_direction"].astype("string").str.lower().eq("low")
    raw["event_high"] = raw["ph_event_direction"].astype("string").str.lower().eq("high")
    raw["event_mixed"] = raw["ph_event_direction"].astype("string").str.lower().eq("mixed")
    grouped = raw.groupby("bin_timestamp", sort=True, observed=True)
    numeric = grouped[[target, *sensors]].mean().astype(np.float32)
    split_first = grouped["split"].first().astype("string")
    split_unique = grouped["split"].nunique()
    flags = grouped[["event_low", "event_high", "event_mixed"]].any()
    raw["event_id_numeric"] = pd.to_numeric(raw["event_id"], errors="coerce")
    event_rows = raw[raw["event_id_numeric"].ge(0)]
    event_group = event_rows.groupby("bin_timestamp", sort=True, observed=True)["event_id_numeric"]
    event_id = event_group.first().reindex(numeric.index).fillna(-1).astype(np.int64)
    event_nunique = event_group.nunique().reindex(numeric.index).fillna(0)
    event_id.loc[event_nunique.gt(1)] = -2
    result = numeric.reset_index().rename(columns={"bin_timestamp": "timestamp"})
    result["site_id"] = site_id
    result["split"] = split_first.to_numpy()
    result["split_nunique"] = split_unique.to_numpy(dtype=np.int16)
    result["event_low"] = flags["event_low"].to_numpy(dtype=bool)
    result["event_high"] = flags["event_high"].to_numpy(dtype=bool)
    result["event_mixed"] = flags["event_mixed"].to_numpy(dtype=bool)
    result["event_id"] = event_id.to_numpy(dtype=np.int64)
    mixed_bins = int(result["split_nunique"].ne(1).sum())
    result = result[result["split_nunique"].eq(1) & result["split"].isin(SPLITS)].copy()
    result["ph_event_direction"] = pd.Series(
        [direction(bool(lo), bool(hi), bool(mix)) for lo, hi, mix in zip(result["event_low"], result["event_high"], result["event_mixed"])],
        index=result.index,
        dtype="string",
    )
    result["in_ph_event"] = result["ph_event_direction"].ne("none")
    result.loc[~result["in_ph_event"], "event_id"] = -1
    result = add_time_features(result)
    result = result.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    keep = ["site_id", "timestamp", "split", target, *config["dynamic_features"], "in_ph_event", "ph_event_direction", "event_id"]
    result = result[keep]
    if result[[target, *config["dynamic_features"]]].isna().any().any():
        raise ValueError(f"30-minute aggregation produced missing values for {site_id}")
    if result["timestamp"].duplicated().any() or not result["timestamp"].is_monotonic_increasing:
        raise ValueError(f"Invalid 30-minute keys for {site_id}")
    return result, mixed_bins


def valid_windows(frame: pd.DataFrame, steps: int, resolution_minutes: int) -> pd.DataFrame:
    columns = ["local_end", "timestamp", "split", "in_ph_event", "ph_event_direction", "event_id"]
    if len(frame) < steps:
        return pd.DataFrame(columns=columns)
    timestamp = pd.to_datetime(frame["timestamp"]).to_numpy(dtype="datetime64[ns]")
    split = frame["split"].astype(str).to_numpy()
    expected = np.timedelta64(resolution_minutes, "m")
    new = np.empty(len(frame), dtype=bool)
    new[0] = True
    new[1:] = (np.diff(timestamp) != expected) | (split[1:] != split[:-1])
    run = np.cumsum(new, dtype=np.int64)
    position = pd.Series(run).groupby(run, sort=False).cumcount().to_numpy(dtype=np.int64) + 1
    index = np.flatnonzero(position >= steps)
    result = frame.loc[index, columns[1:]].copy()
    result.insert(0, "local_end", index.astype(np.int64))
    return result.reset_index(drop=True)


def build_parquet_shards(cohort: pd.DataFrame, output: Path, config: dict[str, Any], resume: bool) -> pd.DataFrame:
    shard_dir = output / "sequence_data" / "hourly_by_site"
    window_dir = output / "sequence_data" / "window_index_by_site"
    checkpoint_dir = output / "checkpoints" / "parquet_shards"
    for directory in (shard_dir, window_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    source_columns = ["site_id", "timestamp", "split", config["target_scaled"], "00010", "00095", "00300", "63680", "ph_event_direction", "event_id"]
    rows: list[dict[str, Any]] = []
    for number, row in enumerate(cohort.itertuples(index=False), start=1):
        site_id = str(row.site_id)
        shard = shard_dir / f"{site_id}.parquet"
        window_path = window_dir / f"{site_id}.parquet"
        checkpoint = checkpoint_dir / f"{site_id}.json"
        if resume and shard.exists() and window_path.exists() and checkpoint.exists():
            rows.append(load_json(checkpoint))
            continue
        source = resolve(str(row.model_output))
        raw = pd.read_parquet(source, columns=source_columns)
        half_hour, mixed = aggregate_resolution(raw, site_id, config)
        windows = valid_windows(half_hour, int(config["sequence_length_hours"]), int(config["resolution_minutes"]))
        write_parquet(shard, half_hour)
        write_parquet(window_path, windows)
        counts = windows["split"].value_counts().to_dict() if len(windows) else {}
        result = {
            "site_id": site_id,
            "network": str(row.network),
            "source_model_rows": len(raw),
            "hourly_rows": len(half_hour),
            "mixed_split_boundary_hours_removed": mixed,
            "train_windows": int(counts.get("train", 0)),
            "val_windows": int(counts.get("val", 0)),
            "test_windows": int(counts.get("test", 0)),
            "hourly_start": pd.Timestamp(half_hour["timestamp"].min()).isoformat() if len(half_hour) else None,
            "hourly_end": pd.Timestamp(half_hour["timestamp"].max()).isoformat() if len(half_hour) else None,
        }
        write_json(checkpoint, result)
        rows.append(result)
        del raw, half_hour, windows
        gc.collect()
        if number % 25 == 0 or number == len(cohort):
            print(f"[30m shards] {number}/{len(cohort)}", flush=True)
    return pd.DataFrame(rows).sort_values("site_id", kind="mergesort").reset_index(drop=True)


def validate_arrays(arrays: dict[str, np.ndarray], steps: int, resolution_minutes: int) -> list[str]:
    problems: list[str] = []
    span = (steps - 1) * resolution_minutes * 60 * 1_000_000_000
    for split in SPLITS:
        ends_all = arrays[f"{split}_ends"]
        if len(ends_all) == 0:
            problems.append(f"{split} has no windows")
            continue
        for start in range(0, len(ends_all), 1_000_000):
            ends = np.asarray(ends_all[start : start + 1_000_000], dtype=np.int64)
            first = ends - steps + 1
            if np.any(first < 0):
                problems.append(f"{split} starts before array")
                break
            if np.any(arrays["site_code"][ends] != arrays["site_code"][first]):
                problems.append(f"{split} crosses sites")
                break
            if np.any(arrays["timestamp_ns"][ends] - arrays["timestamp_ns"][first] != span):
                problems.append(f"{split} is not {resolution_minutes}-minute continuous")
                break
    return problems


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = observed - predicted
    denominator = float(np.square(observed - observed.mean()).sum())
    return {
        "n_rows": len(observed),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1 - np.square(residual).sum() / denominator) if denominator > 0 else np.nan,
    }


def common_30m_target_comparison(half_path: Path, hourly_path: Path, output: Path) -> pd.DataFrame:
    half = pd.read_parquet(half_path, columns=["site_id", "timestamp", "observed_ph", "predicted_ph", "in_ph_event"])
    hourly = pd.read_parquet(hourly_path, columns=["site_id", "timestamp", "predicted_ph"])
    half["hour_timestamp"] = pd.to_datetime(half["timestamp"]).dt.floor("h")
    hourly = hourly.rename(columns={"timestamp": "hour_timestamp", "predicted_ph": "hourly_prediction"})
    hourly["hour_timestamp"] = pd.to_datetime(hourly["hour_timestamp"])
    common = half.merge(hourly, on=["site_id", "hour_timestamp"], how="inner", validate="many_to_one")
    rows = []
    for model_id, column in (("GRU_FILM_V2_N1009_HOURLY_REPEATED", "hourly_prediction"), ("GRU_FILM_V2_N1009_30MIN", "predicted_ph")):
        row = {"model_id": model_id, "target_resolution_minutes": 30, "n_sites": common["site_id"].nunique(), **metric_values(common["observed_ph"], common[column])}
        residual = np.abs(common["observed_ph"].to_numpy(dtype=float) - common[column].to_numpy(dtype=float))
        n_take = max(1, int(np.ceil(len(common) * 0.05)))
        selected = np.argpartition(residual, len(residual) - n_take)[len(residual) - n_take :]
        base = float(common["in_ph_event"].mean())
        selected_rate = float(common.iloc[selected]["in_ph_event"].mean())
        row["top5_event_enrichment"] = selected_rate / base if base > 0 else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    write_csv(output / "reports" / "common_30m_target_resolution_comparison.csv", result)
    return result


def save_resolution_comparison_plot(comparison: pd.DataFrame, output: Path) -> None:
    labels = ["Hourly model\nrepeated", "30-minute\nGRU-FiLM"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, comparison["rmse"], color=["#9ecae1", "#1f77b4"])
    axes[0].set_ylabel("RMSE (native pH)")
    axes[0].set_title("Same 30-minute targets")
    axes[1].bar(labels, comparison["top5_event_enrichment"], color=["#fdae6b", "#e6550d"])
    axes[1].set_ylabel("Top-5% event enrichment")
    axes[1].set_title("Same 30-minute targets")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "figures" / "hourly_vs_30min_common_target_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    output = resolve(config["output_root"])
    if args.postprocess_only:
        comparison = pd.read_csv(output / "reports" / "common_30m_target_resolution_comparison.csv")
        save_resolution_comparison_plot(comparison, output)
        hash_path = output / "reports" / "artifact_hashes.csv"
        hashes = pd.read_csv(hash_path) if hash_path.exists() else pd.DataFrame(columns=["role", "path", "sha256"])
        updates = {
            "config": (str(config_path), sha256_file(config_path)),
            "script": (str(SCRIPT_PATH), sha256_file(SCRIPT_PATH)),
        }
        for role, (path_value, digest) in updates.items():
            hashes = hashes.loc[~hashes["role"].eq(role)]
            hashes = pd.concat([hashes, pd.DataFrame([{"role": role, "path": path_value, "sha256": digest}])], ignore_index=True)
        write_csv(hash_path, hashes)
        print("Resolution comparison figure regenerated.")
        return
    success = output / "_SUCCESS"
    if args.resume and success.exists():
        print(f"Already complete: {success}")
        return
    if success.exists():
        success.unlink()
    for directory in ("reports", "models", "predictions", "figures", "sequence_data"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    base = load_module("common1000_sequence_base", "pipeline/12_common1000_sequence/run_common1000_sequence_models.py")
    film = load_module("common1000_film_base", "pipeline/14_common1000_film/run_common1000_film_scaling.py")
    cohort = pd.read_csv(resolve(config["hgb_root"]) / "reports" / "cohort.csv", dtype={"site_id": "string"})
    write_csv(output / "reports" / "cohort.csv", cohort)
    site_summary = build_parquet_shards(cohort, output, config, args.resume)
    write_csv(output / "reports" / "resolution_window_site_summary.csv", site_summary)
    manifest = base.assemble_sequence_arrays(cohort, site_summary, output, config, args.resume)
    arrays = base.load_arrays(manifest)
    problems = validate_arrays(arrays, int(config["sequence_length_hours"]), int(config["resolution_minutes"]))
    if problems:
        raise RuntimeError(f"30-minute window validation failed: {problems}")
    metadata, metadata_names = film.build_v2_metadata(len(cohort), manifest, arrays, config, output)
    train_ends = np.asarray(arrays["train_sample_ends"], dtype=np.int64)
    val_ends = np.asarray(arrays["val_early_ends"], dtype=np.int64)
    mean, scale = film.fit_dynamic_scaler(train_ends, arrays, config, output / "sequence_data" / "dynamic_scaler.json")
    spec = {
        "model_id": "GRU_FILM_V2_N1009_30MIN",
        "site_count": len(cohort),
        "recurrent_type": "gru",
        "conditioning": "film",
        "training_scope": "all_data",
    }
    model, history, resource = film.train_model(spec, train_ends, val_ends, arrays, mean, scale, metadata, config, output, device, args.resume)
    overall, network, site, tail = film.evaluate_model(model, spec, arrays, manifest, cohort, mean, scale, metadata, config, output, device)
    overall_frame = pd.DataFrame(overall)
    write_csv(output / "reports" / "model_metrics.csv", overall_frame)
    write_csv(output / "reports" / "network_metrics.csv", pd.DataFrame(network))
    write_csv(output / "reports" / "site_metrics.csv", pd.DataFrame(site))
    write_csv(output / "reports" / "residual_tail_metrics.csv", pd.DataFrame(tail))
    write_json(output / "reports" / "feature_manifest.json", {"dynamic_features": config["dynamic_features"], "metadata_features": metadata_names, "forbidden_predictors": config["forbidden_predictors"]})
    half_test = output / "predictions" / f"prediction_test_{spec['model_id']}.parquet"
    comparison = common_30m_target_comparison(half_test, resolve(config["hourly_reference_prediction"]), output)
    save_resolution_comparison_plot(comparison, output)
    test_row = overall_frame[overall_frame["split"].eq("test")].iloc[0]
    report = f"""# Common1000 30-minute GRU-FiLM resolution report

## Design

- Sites: {len(cohort):,}.
- Resolution: {config['resolution_minutes']} minutes; sequence steps: {config['sequence_length_hours']} ({config['sequence_duration_hours']} hours).
- Train/validation/test windows: {manifest['split_window_counts']['train']:,} / {manifest['split_window_counts']['val']:,} / {manifest['split_window_counts']['test']:,}.
- Sampled training windows: {manifest['training_sample_windows']:,}; same 5,000-per-site cap as the hourly experiment.
- Data path: per-site Parquet shards -> disk-backed arrays -> BF16 batches of {config['train_batch_size']:,}.

## Native-resolution test result

- RMSE: {test_row.rmse:.6f}; MAE: {test_row.mae:.6f}; R2: {test_row.r2:.6f}.

## Common 30-minute target comparison

{comparison.to_csv(index=False)}

The hourly prediction is repeated across the two half-hour targets only for this common-target diagnostic. The native-resolution result remains the primary 30-minute model result.
"""
    (output / "reports" / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    expected = manifest["split_window_counts"]
    for split in ("val", "test"):
        path = output / "predictions" / f"prediction_{split}_{spec['model_id']}.parquet"
        if not path.exists() or pq.ParquetFile(path).metadata.num_rows != int(expected[split]):
            problems.append(f"{split} prediction row mismatch")
    if comparison["n_rows"].nunique() != 1 or comparison["n_sites"].nunique() != 1:
        problems.append("common-resolution comparison keys differ")
    if not np.isfinite(overall_frame[["rmse", "mae", "r2"]].to_numpy(dtype=float)).all():
        problems.append("non-finite model metrics")
    hashes = [
        {"role": "config", "path": str(config_path), "sha256": sha256_file(config_path)},
        {"role": "script", "path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        {"role": "model", "path": str(output / "models" / spec["model_id"] / "best_model.pt"), "sha256": sha256_file(output / "models" / spec["model_id"] / "best_model.pt")},
    ]
    write_csv(output / "reports" / "artifact_hashes.csv", pd.DataFrame(hashes))
    summary = {
        "status": "success" if not problems else "failed_validation",
        "sites": len(cohort),
        "resolution_minutes": int(config["resolution_minutes"]),
        "sequence_steps": int(config["sequence_length_hours"]),
        "sequence_duration_hours": int(config["sequence_duration_hours"]),
        "shard_rows": int(site_summary["hourly_rows"].sum()),
        "window_counts": manifest["split_window_counts"],
        "sampled_training_windows": manifest["training_sample_windows"],
        "epochs_completed": len(history),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "bf16": torch.cuda.is_bf16_supported(),
        "peak_gpu_memory_bytes": resource.get("peak_gpu_memory_bytes"),
        "validation_problems": problems,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "reports" / "run_summary.json", summary)
    if problems:
        raise RuntimeError(f"Resolution validation failed: {problems}")
    write_json(success, {"status": "success", "sites": len(cohort), "completed_at_epoch": time.time()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
