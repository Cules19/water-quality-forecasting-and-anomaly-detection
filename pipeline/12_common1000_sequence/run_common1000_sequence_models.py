#!/usr/bin/env python
"""Build hourly 24-step windows and train controlled GRU/LSTM pH models."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}
CODE_TO_SPLIT = {value: key for key, value in SPLIT_TO_CODE.items()}
DIRECTION_TO_CODE = {"none": 0, "low": 1, "high": 2, "mixed": 3}
CODE_TO_DIRECTION = {value: key for key, value in DIRECTION_TO_CODE.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "common1000_sequence_v1.json",
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary, compression="snappy")
    os.replace(temporary, path)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_config(config: dict[str, Any]) -> None:
    features = list(config["dynamic_features"])
    forbidden = set(config["forbidden_predictors"])
    overlap = sorted(set(features) & forbidden)
    if overlap:
        raise ValueError(f"Forbidden predictors present: {overlap}")
    if len(features) != len(set(features)):
        raise ValueError("Dynamic feature list contains duplicates")
    if int(config["sequence_length_hours"]) < 2:
        raise ValueError("Sequence length must be at least two")


def model_source_path(cohort_row: pd.Series) -> Path:
    value = Path(str(cohort_row["model_output"]))
    return value if value.is_absolute() else PROJECT_ROOT / value


def event_direction_from_flags(low: bool, high: bool, mixed: bool) -> str:
    if mixed or (low and high):
        return "mixed"
    if low:
        return "low"
    if high:
        return "high"
    return "none"


def add_time_features(hourly: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(hourly["timestamp"])
    hourly["hour"] = timestamp.dt.hour.astype("int16")
    hourly["month"] = timestamp.dt.month.astype("int16")
    hourly["day_of_year"] = timestamp.dt.dayofyear.astype("int16")
    hourly["sin_hour"] = np.sin(2.0 * np.pi * hourly["hour"] / 24.0).astype("float32")
    hourly["cos_hour"] = np.cos(2.0 * np.pi * hourly["hour"] / 24.0).astype("float32")
    hourly["sin_day"] = np.sin(2.0 * np.pi * hourly["day_of_year"] / 365.25).astype("float32")
    hourly["cos_day"] = np.cos(2.0 * np.pi * hourly["day_of_year"] / 365.25).astype("float32")
    return hourly


def aggregate_site_hourly(raw: pd.DataFrame, site_id: str, config: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    sensor_features = [feature for feature in config["dynamic_features"] if feature in {"00010", "00095", "00300", "63680"}]
    target = config["target_scaled"]
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")
    raw["hour_timestamp"] = raw["timestamp"].dt.floor("h")
    raw["event_low"] = raw["ph_event_direction"].astype("string").str.lower().eq("low")
    raw["event_high"] = raw["ph_event_direction"].astype("string").str.lower().eq("high")
    raw["event_mixed"] = raw["ph_event_direction"].astype("string").str.lower().eq("mixed")
    grouped = raw.groupby("hour_timestamp", sort=True, observed=True)
    numeric = grouped[[target, *sensor_features]].mean().astype("float32")
    split_first = grouped["split"].first().astype("string")
    split_unique = grouped["split"].nunique()
    event_flags = grouped[["event_low", "event_high", "event_mixed"]].any()

    raw["event_id_numeric"] = pd.to_numeric(raw["event_id"], errors="coerce")
    event_rows = raw[raw["event_id_numeric"].ge(0)]
    event_grouped = event_rows.groupby("hour_timestamp", sort=True, observed=True)["event_id_numeric"]
    event_id = event_grouped.first().reindex(numeric.index).fillna(-1).astype("int64")
    event_id_nunique = event_grouped.nunique().reindex(numeric.index).fillna(0)
    event_id.loc[event_id_nunique.gt(1)] = -2
    hourly = numeric.reset_index().rename(columns={"hour_timestamp": "timestamp"})
    hourly["site_id"] = site_id
    hourly["split"] = split_first.to_numpy()
    hourly["split_nunique"] = split_unique.to_numpy(dtype=np.int16)
    hourly["event_low"] = event_flags["event_low"].to_numpy(dtype=bool)
    hourly["event_high"] = event_flags["event_high"].to_numpy(dtype=bool)
    hourly["event_mixed"] = event_flags["event_mixed"].to_numpy(dtype=bool)
    hourly["event_id"] = event_id.to_numpy(dtype=np.int64)
    mixed_boundary_hours = int((hourly["split_nunique"] != 1).sum())
    hourly = hourly[hourly["split_nunique"].eq(1) & hourly["split"].isin(SPLIT_TO_CODE)].copy()
    directions = [
        event_direction_from_flags(bool(low), bool(high), bool(mixed))
        for low, high, mixed in zip(hourly["event_low"], hourly["event_high"], hourly["event_mixed"])
    ]
    hourly["ph_event_direction"] = pd.Series(directions, index=hourly.index, dtype="string")
    hourly["in_ph_event"] = hourly["ph_event_direction"].ne("none")
    hourly.loc[~hourly["in_ph_event"], "event_id"] = -1
    hourly = add_time_features(hourly)
    hourly = hourly.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    keep = [
        "site_id",
        "timestamp",
        "split",
        target,
        *config["dynamic_features"],
        "in_ph_event",
        "ph_event_direction",
        "event_id",
    ]
    hourly = hourly[keep]
    if hourly[[target, *config["dynamic_features"]]].isna().any().any():
        raise ValueError(f"Hourly aggregation produced missing model values for {site_id}")
    if hourly["timestamp"].duplicated().any() or not hourly["timestamp"].is_monotonic_increasing:
        raise ValueError(f"Invalid hourly time key for {site_id}")
    return hourly, mixed_boundary_hours


def valid_window_index(hourly: pd.DataFrame, sequence_length: int) -> pd.DataFrame:
    if len(hourly) < sequence_length:
        return pd.DataFrame(columns=["local_end", "timestamp", "split", "in_ph_event", "ph_event_direction", "event_id"])
    timestamp = pd.to_datetime(hourly["timestamp"]).to_numpy(dtype="datetime64[ns]")
    split = hourly["split"].astype(str).to_numpy()
    new_run = np.empty(len(hourly), dtype=bool)
    new_run[0] = True
    new_run[1:] = (np.diff(timestamp) != np.timedelta64(1, "h")) | (split[1:] != split[:-1])
    run_id = np.cumsum(new_run, dtype=np.int64)
    run_position = pd.Series(run_id).groupby(run_id, sort=False).cumcount().to_numpy(dtype=np.int64) + 1
    index = np.flatnonzero(run_position >= sequence_length)
    result = hourly.loc[index, ["timestamp", "split", "in_ph_event", "ph_event_direction", "event_id"]].copy()
    result.insert(0, "local_end", index.astype(np.int64))
    return result.reset_index(drop=True)


def build_hourly_files(
    cohort: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any],
    resume: bool,
) -> pd.DataFrame:
    hourly_dir = output_dir / "sequence_data" / "hourly_by_site"
    window_dir = output_dir / "sequence_data" / "window_index_by_site"
    hourly_dir.mkdir(parents=True, exist_ok=True)
    window_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    source_columns = [
        "site_id",
        "timestamp",
        "split",
        config["target_scaled"],
        "00010",
        "00095",
        "00300",
        "63680",
        "ph_event_direction",
        "event_id",
    ]
    sequence_length = int(config["sequence_length_hours"])
    for number, (_, cohort_row) in enumerate(cohort.iterrows(), start=1):
        site_id = str(cohort_row["site_id"])
        hourly_path = hourly_dir / f"{site_id}.parquet"
        window_path = window_dir / f"{site_id}.parquet"
        checkpoint = output_dir / "checkpoints" / "hourly" / f"{site_id}.json"
        if resume and checkpoint.exists() and hourly_path.exists() and window_path.exists():
            rows.append(load_json(checkpoint))
            continue
        source = model_source_path(cohort_row)
        raw = pd.read_parquet(source, columns=source_columns)
        hourly, boundary_hours = aggregate_site_hourly(raw, site_id, config)
        windows = valid_window_index(hourly, sequence_length)
        write_parquet(hourly_path, hourly)
        write_parquet(window_path, windows)
        counts = windows["split"].value_counts().to_dict() if len(windows) else {}
        result = {
            "site_id": site_id,
            "network": str(cohort_row["network"]),
            "source_model_rows": int(len(raw)),
            "hourly_rows": int(len(hourly)),
            "mixed_split_boundary_hours_removed": boundary_hours,
            "train_windows": int(counts.get("train", 0)),
            "val_windows": int(counts.get("val", 0)),
            "test_windows": int(counts.get("test", 0)),
            "hourly_start": hourly["timestamp"].min() if len(hourly) else None,
            "hourly_end": hourly["timestamp"].max() if len(hourly) else None,
        }
        write_json(checkpoint, result)
        rows.append(result)
        del raw, hourly, windows
        gc.collect()
        if number % 25 == 0 or number == len(cohort):
            logging.info("Hourly/window build: %s/%s sites", number, len(cohort))
    summary = pd.DataFrame(rows).sort_values("site_id", kind="mergesort").reset_index(drop=True)
    return summary


def deterministic_positions(n_rows: int, cap: int) -> np.ndarray:
    if n_rows <= 0 or cap <= 0:
        return np.empty(0, dtype=np.int64)
    if n_rows <= cap:
        return np.arange(n_rows, dtype=np.int64)
    result = np.linspace(0, n_rows - 1, cap, dtype=np.int64)
    if len(np.unique(result)) != cap:
        raise RuntimeError("Deterministic sample contains duplicate positions")
    return result


def open_npy(path: Path, dtype: Any, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def assemble_sequence_arrays(
    cohort: pd.DataFrame,
    site_summary: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    data_dir = output_dir / "sequence_data" / "arrays"
    manifest_path = data_dir / "array_manifest.json"
    required_paths = {
        "features": data_dir / "features.npy",
        "target": data_dir / "target.npy",
        "timestamp_ns": data_dir / "timestamp_ns.npy",
        "site_code": data_dir / "site_code.npy",
        "event_direction": data_dir / "event_direction.npy",
        "event_id": data_dir / "event_id.npy",
        "train_ends": data_dir / "train_ends.npy",
        "val_ends": data_dir / "val_ends.npy",
        "test_ends": data_dir / "test_ends.npy",
        "train_sample_ends": data_dir / "train_sample_ends.npy",
        "val_early_ends": data_dir / "val_early_ends.npy",
    }
    if resume and manifest_path.exists() and all(path.exists() for path in required_paths.values()):
        manifest = load_json(manifest_path)
        manifest["paths"] = {key: str(path) for key, path in required_paths.items()}
        return manifest

    data_dir.mkdir(parents=True, exist_ok=True)
    site_order = cohort["site_id"].astype(str).tolist()
    site_map = {site_id: index for index, site_id in enumerate(site_order)}
    summary_index = site_summary.set_index("site_id")
    total_hourly = int(site_summary["hourly_rows"].sum())
    split_totals = {
        "train": int(site_summary["train_windows"].sum()),
        "val": int(site_summary["val_windows"].sum()),
        "test": int(site_summary["test_windows"].sum()),
    }
    n_features = len(config["dynamic_features"])
    features = open_npy(required_paths["features"], np.float32, (total_hourly, n_features))
    target = open_npy(required_paths["target"], np.float32, (total_hourly,))
    timestamp_ns = open_npy(required_paths["timestamp_ns"], np.int64, (total_hourly,))
    site_code = open_npy(required_paths["site_code"], np.int16, (total_hourly,))
    event_direction = open_npy(required_paths["event_direction"], np.int8, (total_hourly,))
    event_id = open_npy(required_paths["event_id"], np.int64, (total_hourly,))
    split_arrays = {
        split: open_npy(required_paths[f"{split}_ends"], np.int64, (count,))
        for split, count in split_totals.items()
    }
    split_offsets = {split: 0 for split in SPLIT_TO_CODE}
    train_samples: list[np.ndarray] = []
    val_samples: list[np.ndarray] = []
    hourly_offset = 0
    training_audit: list[dict[str, Any]] = []
    hourly_dir = output_dir / "sequence_data" / "hourly_by_site"
    window_dir = output_dir / "sequence_data" / "window_index_by_site"
    train_cap = int(config["train_window_cap_per_site"])
    val_cap = int(config["early_stop_validation_cap_per_site"])

    for number, site_id in enumerate(site_order, start=1):
        hourly = pd.read_parquet(hourly_dir / f"{site_id}.parquet")
        windows = pd.read_parquet(window_dir / f"{site_id}.parquet")
        n_hourly = len(hourly)
        stop = hourly_offset + n_hourly
        features[hourly_offset:stop] = hourly[config["dynamic_features"]].to_numpy(dtype=np.float32)
        target[hourly_offset:stop] = pd.to_numeric(hourly[config["target_scaled"]], errors="raise").to_numpy(dtype=np.float32)
        timestamp_ns[hourly_offset:stop] = (
            pd.to_datetime(hourly["timestamp"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        )
        site_code[hourly_offset:stop] = site_map[site_id]
        event_direction[hourly_offset:stop] = hourly["ph_event_direction"].map(DIRECTION_TO_CODE).fillna(0).to_numpy(dtype=np.int8)
        event_id[hourly_offset:stop] = pd.to_numeric(hourly["event_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        sampled_counts: dict[str, int] = {}
        available_counts: dict[str, int] = {}
        for split in SPLIT_TO_CODE:
            local = windows.loc[windows["split"].eq(split), "local_end"].to_numpy(dtype=np.int64)
            global_ends = local + hourly_offset
            start = split_offsets[split]
            split_arrays[split][start : start + len(global_ends)] = global_ends
            split_offsets[split] += len(global_ends)
            available_counts[split] = len(global_ends)
            if split == "train":
                selected = global_ends[deterministic_positions(len(global_ends), train_cap)]
                train_samples.append(selected)
                sampled_counts[split] = len(selected)
            elif split == "val":
                selected = global_ends[deterministic_positions(len(global_ends), val_cap)]
                val_samples.append(selected)
                sampled_counts[split] = len(selected)
        training_audit.append(
            {
                "site_id": site_id,
                "network": summary_index.loc[site_id, "network"],
                "available_train_windows": available_counts.get("train", 0),
                "sampled_train_windows": sampled_counts.get("train", 0),
                "available_val_windows": available_counts.get("val", 0),
                "early_stop_val_windows": sampled_counts.get("val", 0),
                "test_windows": available_counts.get("test", 0),
            }
        )
        hourly_offset = stop
        del hourly, windows
        if number % 50 == 0 or number == len(site_order):
            logging.info("Array assembly: %s/%s sites", number, len(site_order))

    if hourly_offset != total_hourly or split_offsets != split_totals:
        raise RuntimeError(f"Array assembly count mismatch: hourly={hourly_offset}/{total_hourly}, splits={split_offsets}/{split_totals}")
    train_sample = np.concatenate(train_samples) if train_samples else np.empty(0, dtype=np.int64)
    val_sample = np.concatenate(val_samples) if val_samples else np.empty(0, dtype=np.int64)
    np.save(required_paths["train_sample_ends"], train_sample)
    np.save(required_paths["val_early_ends"], val_sample)
    write_csv(output_dir / "reports" / "training_window_sample_audit.csv", pd.DataFrame(training_audit))
    for array in [features, target, timestamp_ns, site_code, event_direction, event_id, *split_arrays.values()]:
        array.flush()
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
    manifest = {
        "total_hourly_rows": total_hourly,
        "split_window_counts": split_totals,
        "training_sample_windows": int(len(train_sample)),
        "early_stop_validation_windows": int(len(val_sample)),
        "n_sites": len(site_order),
        "n_dynamic_features": n_features,
        "site_order": site_order,
        "site_map": site_map,
        "paths": {key: str(path) for key, path in required_paths.items()},
    }
    write_json(manifest_path, {key: value for key, value in manifest.items() if key != "paths"})
    return manifest


def load_arrays(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.load(path, mmap_mode="r") for key, path in manifest["paths"].items()}


def validate_window_arrays(arrays: dict[str, np.ndarray], sequence_length: int) -> list[str]:
    problems: list[str] = []
    expected_span_ns = (sequence_length - 1) * 3600 * 1_000_000_000
    for split in ("train", "val", "test"):
        ends = arrays[f"{split}_ends"]
        if len(ends) == 0:
            problems.append(f"{split} has no valid windows")
            continue
        starts = np.asarray(ends, dtype=np.int64) - sequence_length + 1
        end_times = np.asarray(arrays["timestamp_ns"][ends], dtype=np.int64)
        start_times = np.asarray(arrays["timestamp_ns"][starts], dtype=np.int64)
        if np.any(end_times - start_times != expected_span_ns):
            problems.append(f"{split} contains a non-continuous 24-hour window")
        if np.any(arrays["site_code"][ends] != arrays["site_code"][starts]):
            problems.append(f"{split} contains a cross-site window")
    return problems


def build_metadata_matrix(
    cohort: pd.DataFrame,
    hgb_mode_dir: Path,
    output_dir: Path,
) -> tuple[np.ndarray, list[str]]:
    metadata = pd.read_csv(hgb_mode_dir / "reports" / "metadata_model_ready.csv", dtype={"site_id": "string"})
    metadata = metadata.set_index("site_id").loc[cohort["site_id"].astype(str)].reset_index()
    feature_columns = [column for column in metadata.columns if column not in {"site_id", "metadata_match"}]
    numeric_columns = [column for column in feature_columns if not column.endswith("_code")]
    categorical_columns = [column for column in feature_columns if column.endswith("_code")]
    parts: list[np.ndarray] = []
    names: list[str] = []
    spec: dict[str, Any] = {"numeric": {}, "categorical": {}}
    for column in numeric_columns:
        values = pd.to_numeric(metadata[column], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        if column.endswith("_missing"):
            transformed = values
            spec["numeric"][column] = {"mean": 0.0, "scale": 1.0, "binary_missing_flag": True}
        else:
            mean = float(values.mean(dtype=np.float64))
            scale = float(values.std(dtype=np.float64))
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            transformed = (values - mean) / scale
            spec["numeric"][column] = {"mean": mean, "scale": scale, "binary_missing_flag": False}
        parts.append(transformed[:, None].astype(np.float32))
        names.append(column)
    for column in categorical_columns:
        values = pd.to_numeric(metadata[column], errors="coerce").fillna(-1).astype(int)
        categories = sorted(values.unique().tolist())
        one_hot = np.zeros((len(values), len(categories)), dtype=np.float32)
        mapping = {category: index for index, category in enumerate(categories)}
        for row, value in enumerate(values):
            one_hot[row, mapping[int(value)]] = 1.0
        parts.append(one_hot)
        one_hot_names = [f"{column}__{category}" for category in categories]
        names.extend(one_hot_names)
        spec["categorical"][column] = {"categories": categories, "one_hot_features": one_hot_names}
    matrix = np.concatenate(parts, axis=1) if parts else np.empty((len(metadata), 0), dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Sequence metadata matrix contains non-finite values")
    np.save(output_dir / "sequence_data" / "metadata_matrix.npy", matrix)
    write_json(output_dir / "sequence_data" / "metadata_transform.json", {"feature_names": names, **spec})
    return matrix, names


def fit_dynamic_scaler(
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    output_dir: Path,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray]:
    path = output_dir / "sequence_data" / "dynamic_scaler.json"
    if resume and path.exists():
        payload = load_json(path)
        return np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["scale"], dtype=np.float32)
    ends = arrays["train_sample_ends"]
    features = arrays["features"]
    length = int(config["sequence_length_hours"])
    batch = int(config["scaler_batch_windows"])
    offsets = np.arange(length - 1, -1, -1, dtype=np.int64)
    n = 0
    total = np.zeros(features.shape[1], dtype=np.float64)
    total_sq = np.zeros(features.shape[1], dtype=np.float64)
    for start in range(0, len(ends), batch):
        positions = np.asarray(ends[start : start + batch])[:, None] - offsets[None, :]
        values = np.asarray(features[positions], dtype=np.float32).reshape(-1, features.shape[1]).astype(np.float64)
        n += len(values)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
    mean = total / n
    variance = np.maximum(total_sq / n - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    payload = {
        "features": config["dynamic_features"],
        "fit_training_windows": int(len(ends)),
        "fit_exposure_rows_including_overlap": int(n),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    write_json(path, payload)
    return mean.astype(np.float32), scale.astype(np.float32)


class SequenceRegressor(nn.Module):
    def __init__(
        self,
        recurrent_type: str,
        n_dynamic: int,
        hidden_size: int,
        layers: int,
        n_sites: int,
        site_embedding_size: int,
        metadata_size: int,
        mlp_hidden: int,
    ) -> None:
        super().__init__()
        recurrent_class = nn.GRU if recurrent_type == "gru" else nn.LSTM
        self.encoder = recurrent_class(n_dynamic, hidden_size, num_layers=layers, batch_first=True)
        self.site_embedding = nn.Embedding(n_sites, site_embedding_size)
        head_input = hidden_size + site_embedding_size + metadata_size
        self.head = nn.Sequential(nn.Linear(head_input, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))
        self.metadata_size = metadata_size

    def forward(self, dynamic: torch.Tensor, site_code: torch.Tensor, metadata: torch.Tensor | None) -> torch.Tensor:
        output, hidden = self.encoder(dynamic)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        final = hidden[-1]
        embedding = self.site_embedding(site_code)
        pieces = [final, embedding]
        if self.metadata_size:
            if metadata is None:
                raise ValueError("Metadata tensor required")
            pieces.append(metadata)
        return self.head(torch.cat(pieces, dim=1)).squeeze(1)


def batch_numpy(
    ends: np.ndarray,
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    metadata_matrix: np.ndarray,
    sequence_length: int,
    metadata_used: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    offsets = np.arange(sequence_length - 1, -1, -1, dtype=np.int64)
    positions = ends[:, None] - offsets[None, :]
    dynamic = np.asarray(arrays["features"][positions], dtype=np.float32)
    dynamic = (dynamic - mean[None, None, :]) / scale[None, None, :]
    site = np.asarray(arrays["site_code"][ends], dtype=np.int64)
    metadata = metadata_matrix[site] if metadata_used else None
    target = np.asarray(arrays["target"][ends], dtype=np.float32)
    return np.ascontiguousarray(dynamic), site, np.ascontiguousarray(metadata) if metadata is not None else None, target


@torch.no_grad()
def predict_scaled(
    model: SequenceRegressor,
    ends: np.ndarray,
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    metadata_matrix: np.ndarray,
    config: dict[str, Any],
    metadata_used: bool,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    batch_size = int(config["prediction_batch_size"])
    predictions = np.empty(len(ends), dtype=np.float32)
    for start in range(0, len(ends), batch_size):
        stop = min(len(ends), start + batch_size)
        batch_ends = np.asarray(ends[start:stop], dtype=np.int64)
        dynamic, site, metadata, _target = batch_numpy(
            batch_ends,
            arrays,
            mean,
            scale,
            metadata_matrix,
            int(config["sequence_length_hours"]),
            metadata_used,
        )
        dynamic_tensor = torch.from_numpy(dynamic).to(device)
        site_tensor = torch.from_numpy(site).to(device)
        metadata_tensor = torch.from_numpy(metadata).to(device) if metadata is not None else None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(dynamic_tensor, site_tensor, metadata_tensor)
        predictions[start:stop] = prediction.float().cpu().numpy()
    return predictions


def create_model(spec: dict[str, Any], config: dict[str, Any], n_sites: int, metadata_size: int, device: torch.device) -> SequenceRegressor:
    return SequenceRegressor(
        recurrent_type=str(spec["recurrent_type"]),
        n_dynamic=len(config["dynamic_features"]),
        hidden_size=int(config["hidden_size"]),
        layers=int(config["recurrent_layers"]),
        n_sites=n_sites,
        site_embedding_size=int(config["site_embedding_size"]),
        metadata_size=metadata_size if bool(spec["metadata_used"]) else 0,
        mlp_hidden=int(config["mlp_hidden_size"]),
    ).to(device)


def train_model(
    spec: dict[str, Any],
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    metadata_matrix: np.ndarray,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    resume: bool,
) -> tuple[SequenceRegressor, pd.DataFrame]:
    model_id = str(spec["model_id"])
    model_path = output_dir / "models" / f"{model_id}.pt"
    history_path = output_dir / "reports" / f"training_history_{model_id}.csv"
    success_path = output_dir / "models" / f"{model_id}_SUCCESS.json"
    model = create_model(spec, config, len(metadata_matrix), metadata_matrix.shape[1], device)
    if resume and model_path.exists() and success_path.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        logging.info("Resumed completed model %s", model_id)
        return model, history

    set_reproducible_seed(int(config["random_seed"]))
    model = create_model(spec, config, len(metadata_matrix), metadata_matrix.shape[1], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    train_ends = np.asarray(arrays["train_sample_ends"], dtype=np.int64)
    val_ends = np.asarray(arrays["val_early_ends"], dtype=np.int64)
    batch_size = int(config["train_batch_size"])
    rng = np.random.default_rng(int(config["random_seed"]))
    best_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []
    started = time.time()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(config["maximum_epochs"]) + 1):
        model.train()
        order = rng.permutation(len(train_ends))
        train_sq = 0.0
        train_n = 0
        for start in range(0, len(order), batch_size):
            selected = train_ends[order[start : start + batch_size]]
            dynamic, site, metadata, target = batch_numpy(
                selected,
                arrays,
                mean,
                scale,
                metadata_matrix,
                int(config["sequence_length_hours"]),
                bool(spec["metadata_used"]),
            )
            dynamic_tensor = torch.from_numpy(dynamic).to(device)
            site_tensor = torch.from_numpy(site).to(device)
            metadata_tensor = torch.from_numpy(metadata).to(device) if metadata is not None else None
            target_tensor = torch.from_numpy(target).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model(dynamic_tensor, site_tensor, metadata_tensor)
                loss = nn.functional.mse_loss(prediction, target_tensor)
            loss.backward()
            optimizer.step()
            train_sq += float(loss.detach().cpu()) * len(selected)
            train_n += len(selected)

        val_prediction = predict_scaled(
            model,
            val_ends,
            arrays,
            mean,
            scale,
            metadata_matrix,
            config,
            bool(spec["metadata_used"]),
            device,
        )
        val_target = np.asarray(arrays["target"][val_ends], dtype=np.float32)
        val_loss = float(np.mean(np.square(val_target.astype(np.float64) - val_prediction.astype(np.float64))))
        row = {
            "model_id": model_id,
            "epoch": epoch,
            "train_mse_scaled": train_sq / train_n,
            "validation_mse_scaled": val_loss,
            "validation_rmse_native_ph": math.sqrt(val_loss) * float(config["target_inverse_multiplier"]),
            "elapsed_seconds": time.time() - started,
        }
        history_rows.append(row)
        logging.info(
            "%s epoch %s train_mse=%.7f val_rmse_native=%.5f",
            model_id,
            epoch,
            row["train_mse_scaled"],
            row["validation_rmse_native_ph"],
        )
        if val_loss < best_loss - 1e-10:
            best_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_spec": spec,
                    "best_epoch": best_epoch,
                    "best_validation_mse_scaled": best_loss,
                    "config_name": config["name"],
                },
                model_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config["early_stopping_patience"]):
                break
    history = pd.DataFrame(history_rows)
    write_csv(history_path, history)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    write_json(
        success_path,
        {
            "model_id": model_id,
            "best_epoch": best_epoch,
            "best_validation_mse_scaled": best_loss,
            "elapsed_seconds": time.time() - started,
        },
    )
    return model, history


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    observed64 = observed.astype(np.float64)
    predicted64 = predicted.astype(np.float64)
    residual = observed64 - predicted64
    denominator = float(np.square(observed64 - observed64.mean()).sum())
    return {
        "n_rows": int(len(observed)),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "mae": float(np.abs(residual).mean()),
        "r2": float(1.0 - np.square(residual).sum() / denominator) if denominator > 0 else np.nan,
    }


def prediction_frame(
    model_id: str,
    split: str,
    ends: np.ndarray,
    predicted_scaled: np.ndarray,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    cohort: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    site_codes = np.asarray(arrays["site_code"][ends], dtype=np.int64)
    site_order = np.asarray(manifest["site_order"], dtype=object)
    network_by_site = cohort.set_index("site_id").loc[manifest["site_order"], "network"].astype(str).to_numpy(dtype=object)
    observed_scaled = np.asarray(arrays["target"][ends], dtype=np.float32)
    multiplier = float(config["target_inverse_multiplier"])
    observed = observed_scaled * multiplier
    predicted = predicted_scaled * multiplier
    direction_codes = np.asarray(arrays["event_direction"][ends], dtype=np.int8)
    frame = pd.DataFrame(
        {
            "model_id": model_id,
            "site_id": site_order[site_codes],
            "network": network_by_site[site_codes],
            "timestamp": pd.to_datetime(np.asarray(arrays["timestamp_ns"][ends], dtype=np.int64), unit="ns"),
            "split": split,
            "observed_scaled": observed_scaled,
            "predicted_scaled": predicted_scaled,
            "observed_ph": observed,
            "predicted_ph": predicted,
            "residual_observed_minus_predicted": observed - predicted,
            "abs_residual": np.abs(observed - predicted),
            "in_ph_event": direction_codes != 0,
            "ph_event_direction": [CODE_TO_DIRECTION[int(value)] for value in direction_codes],
            "event_id": np.asarray(arrays["event_id"][ends], dtype=np.int64),
        }
    )
    return frame


def tail_analysis(frame: pd.DataFrame, model_id: str, fractions: Iterable[float]) -> list[dict[str, Any]]:
    residual = frame["residual_observed_minus_predicted"].to_numpy(dtype=np.float32)
    absolute = np.abs(residual)
    event = frame["in_ph_event"].astype(bool).to_numpy()
    directions = frame["ph_event_direction"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    base_event_rate = float(event.mean()) if len(event) else np.nan
    for fraction in fractions:
        thresholds = {
            "absolute": float(np.quantile(absolute, 1 - fraction)),
            "negative": float(np.quantile(residual, fraction)),
            "positive": float(np.quantile(residual, 1 - fraction)),
        }
        masks = {
            "absolute": absolute >= thresholds["absolute"],
            "negative": residual <= thresholds["negative"],
            "positive": residual >= thresholds["positive"],
        }
        for tail_type, mask in masks.items():
            selected_event_rate = float(event[mask].mean()) if mask.any() else np.nan
            rows.append(
                {
                    "model_id": model_id,
                    "tail_fraction_requested": fraction,
                    "tail_type": tail_type,
                    "threshold_native_ph": thresholds[tail_type],
                    "n_test_rows": len(frame),
                    "n_selected_rows": int(mask.sum()),
                    "base_event_rate": base_event_rate,
                    "selected_event_rate": selected_event_rate,
                    "event_enrichment": selected_event_rate / base_event_rate if base_event_rate > 0 else np.nan,
                    "selected_low_rows": int(((directions == "low") & mask).sum()),
                    "selected_high_rows": int(((directions == "high") & mask).sum()),
                    "selected_mixed_rows": int(((directions == "mixed") & mask).sum()),
                }
            )
    return rows


def evaluate_and_save_model(
    model: SequenceRegressor,
    spec: dict[str, Any],
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    metadata_matrix: np.ndarray,
    manifest: dict[str, Any],
    cohort: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model_id = str(spec["model_id"])
    overall_rows: list[dict[str, Any]] = []
    network_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        ends = arrays[f"{split}_ends"]
        prediction = predict_scaled(
            model,
            ends,
            arrays,
            mean,
            scale,
            metadata_matrix,
            config,
            bool(spec["metadata_used"]),
            device,
        )
        frame = prediction_frame(model_id, split, ends, prediction, arrays, manifest, cohort, config)
        write_parquet(output_dir / "predictions" / f"prediction_{split}_{model_id}.parquet", frame)
        metric = {"model_id": model_id, "split": split, "metric_unit": "native pH", "n_sites": frame["site_id"].nunique()}
        metric.update(metric_values(frame["observed_ph"].to_numpy(), frame["predicted_ph"].to_numpy()))
        overall_rows.append(metric)
        for network, group in frame.groupby("network", sort=True, observed=True):
            row = {"model_id": model_id, "split": split, "network": network, "metric_unit": "native pH", "n_sites": group["site_id"].nunique()}
            row.update(metric_values(group["observed_ph"].to_numpy(), group["predicted_ph"].to_numpy()))
            network_rows.append(row)
        for site_id, group in frame.groupby("site_id", sort=True, observed=True):
            row = {
                "model_id": model_id,
                "split": split,
                "site_id": site_id,
                "network": group["network"].iloc[0],
                "n_event_rows": int(group["in_ph_event"].sum()),
            }
            row.update(metric_values(group["observed_ph"].to_numpy(), group["predicted_ph"].to_numpy()))
            site_rows.append(row)
        if split == "test":
            tail_rows.extend(tail_analysis(frame, model_id, [float(value) for value in config["residual_tail_fractions"]]))
        del frame, prediction
        gc.collect()
    return overall_rows, network_rows, site_rows, tail_rows


def report_text(
    mode: str,
    summary: pd.DataFrame,
    manifest: dict[str, Any],
    metrics: pd.DataFrame,
    device: torch.device,
) -> str:
    test = metrics[metrics["split"].eq("test")]
    lines = [
        f"# Common1000 sequence-model {mode} report",
        "",
        "## Dataset",
        "",
        f"- Sites: {manifest['n_sites']:,}",
        f"- Hourly rows: {manifest['total_hourly_rows']:,}",
        f"- Train/validation/test windows: {manifest['split_window_counts']['train']:,} / {manifest['split_window_counts']['val']:,} / {manifest['split_window_counts']['test']:,}",
        f"- Sampled training windows: {manifest['training_sample_windows']:,}",
        f"- Device: {device}",
        "",
        "## Test metrics (hourly target, native pH)",
        "",
        "| Model | RMSE | MAE | R2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in test.itertuples(index=False):
        lines.append(f"| {row.model_id} | {row.rmse:.6f} | {row.mae:.6f} | {row.r2:.6f} |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- These models predict the final hour of a valid 24-hour exogenous sequence.",
            "- No current or historical pH and no event field is used as an input.",
            "- Site embedding means this is same-site chronological evaluation, not unseen-site generalisation.",
            "- Hourly sequence metrics are not numerically interchangeable with native-row HGB metrics.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    validate_config(config)
    output_dir = resolve_path(config["output_root"]) / args.mode
    setup_logging(output_dir)
    success_path = output_dir / "_SUCCESS"
    if args.resume and success_path.exists():
        logging.info("Sequence run already complete: %s", success_path)
        return
    if success_path.exists():
        success_path.unlink()
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Common1000 sequence run")
    device = torch.device("cuda")
    set_reproducible_seed(int(config["random_seed"]))
    hgb_mode_dir = resolve_path(config["hgb_root"]) / args.mode
    cohort = pd.read_csv(hgb_mode_dir / "reports" / "cohort.csv", dtype={"site_id": "string"})
    write_csv(output_dir / "reports" / "cohort.csv", cohort)
    logging.info("Starting sequence run mode=%s sites=%s device=%s", args.mode, len(cohort), torch.cuda.get_device_name(0))

    site_summary = build_hourly_files(cohort, output_dir, config, args.resume)
    write_csv(output_dir / "reports" / "hourly_window_site_summary.csv", site_summary)
    manifest = assemble_sequence_arrays(cohort, site_summary, output_dir, config, args.resume)
    arrays = load_arrays(manifest)
    window_array_problems = validate_window_arrays(arrays, int(config["sequence_length_hours"]))
    if window_array_problems:
        raise RuntimeError(f"Window array validation failed before training: {window_array_problems}")
    metadata_matrix, metadata_names = build_metadata_matrix(cohort, hgb_mode_dir, output_dir)
    mean, scale = fit_dynamic_scaler(arrays, config, output_dir, args.resume)
    write_json(
        output_dir / "reports" / "feature_manifest.json",
        {
            "dynamic_features": config["dynamic_features"],
            "metadata_features": metadata_names,
            "site_embedding_used": True,
            "sequence_length_hours": config["sequence_length_hours"],
            "target": config["target_scaled"],
            "forbidden_inputs": config["forbidden_predictors"],
        },
    )

    all_overall: list[dict[str, Any]] = []
    all_network: list[dict[str, Any]] = []
    all_site: list[dict[str, Any]] = []
    all_tail: list[dict[str, Any]] = []
    histories: dict[str, int] = {}
    for spec in config["models"]:
        model, history = train_model(spec, arrays, mean, scale, metadata_matrix, config, output_dir, device, args.resume)
        histories[str(spec["model_id"])] = len(history)
        overall, network, site, tail = evaluate_and_save_model(
            model,
            spec,
            arrays,
            mean,
            scale,
            metadata_matrix,
            manifest,
            cohort,
            config,
            output_dir,
            device,
        )
        all_overall.extend(overall)
        all_network.extend(network)
        all_site.extend(site)
        all_tail.extend(tail)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    metrics = pd.DataFrame(all_overall)
    network_metrics = pd.DataFrame(all_network)
    site_metrics = pd.DataFrame(all_site)
    tail_metrics = pd.DataFrame(all_tail)
    write_csv(output_dir / "reports" / "model_comparison_metrics.csv", metrics)
    write_csv(output_dir / "reports" / "network_metrics.csv", network_metrics)
    write_csv(output_dir / "reports" / "site_level_metrics.csv", site_metrics)
    write_csv(output_dir / "reports" / "residual_tail_analysis.csv", tail_metrics)

    problems: list[str] = list(window_array_problems)
    for spec in config["models"]:
        model_id = str(spec["model_id"])
        for split in ("val", "test"):
            path = output_dir / "predictions" / f"prediction_{split}_{model_id}.parquet"
            if not path.exists():
                problems.append(f"missing prediction file {path.name}")
                continue
            expected = int(manifest["split_window_counts"][split])
            actual = pq.ParquetFile(path).metadata.num_rows
            if actual != expected:
                problems.append(f"{path.name} rows {actual} != {expected}")
    if not np.isfinite(metrics[["rmse", "mae", "r2"]].to_numpy(dtype=float)).all():
        problems.append("non-finite overall metrics")
    hashes = [
        {"role": "config", "path": str(config_path), "sha256": sha256_file(config_path)},
        {"role": "script", "path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
    ]
    for spec in config["models"]:
        model_path = output_dir / "models" / f"{spec['model_id']}.pt"
        hashes.append({"role": f"model_{spec['model_id']}", "path": str(model_path), "sha256": sha256_file(model_path)})
    write_csv(output_dir / "reports" / "artifact_hashes.csv", pd.DataFrame(hashes))
    run_summary = {
        "status": "success" if not problems else "failed_validation",
        "mode": args.mode,
        "sites": len(cohort),
        "hourly_rows": manifest["total_hourly_rows"],
        "window_counts": manifest["split_window_counts"],
        "sampled_training_windows": manifest["training_sample_windows"],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bf16": torch.cuda.is_bf16_supported(),
        "epochs_completed": histories,
        "validation_problems": problems,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "reports" / "run_summary.json", run_summary)
    atomic_text(output_dir / "reports" / "FINAL_REPORT.md", report_text(args.mode, site_summary, manifest, metrics, device))
    if problems:
        raise RuntimeError(f"Sequence validation failed: {problems}")
    write_json(success_path, {"status": "success", "mode": args.mode, "sites": len(cohort), "completed_at_epoch": time.time()})
    logging.info("Completed sequence mode=%s in %.1f seconds", args.mode, time.time() - started)


if __name__ == "__main__":
    main()
