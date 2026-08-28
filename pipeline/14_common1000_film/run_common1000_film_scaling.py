#!/usr/bin/env python
"""Common1000 FiLM ablation, site-count scaling and non-event sequence control."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DIRECTION_TO_NAME = {0: "none", 1: "low", 2: "high", 3: "mixed"}
HOUR_NS = 3_600_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "common1000_film_scaling_v1.json",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary, compression="zstd")
    temporary.replace(path)


def setup_logging(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config: dict[str, Any]) -> None:
    counts = [int(value) for value in config["site_counts"]]
    if counts != sorted(set(counts)) or counts[-1] != 1009 or int(config["ablation_site_count"]) not in counts:
        raise ValueError("site_counts must be unique ascending values ending at 1009 and include the ablation count")
    if len(config["dynamic_features"]) != 11:
        raise ValueError("Expected the existing 11 Common1000 dynamic features")
    if int(config["sequence_length_hours"]) != 24:
        raise ValueError("This controlled experiment requires the existing 24-hour windows")


def array_paths(sequence_dir: Path) -> dict[str, Path]:
    root = sequence_dir / "sequence_data" / "arrays"
    return {
        "features": root / "features.npy",
        "target": root / "target.npy",
        "timestamp_ns": root / "timestamp_ns.npy",
        "site_code": root / "site_code.npy",
        "event_direction": root / "event_direction.npy",
        "event_id": root / "event_id.npy",
        "train_ends": root / "train_ends.npy",
        "val_ends": root / "val_ends.npy",
        "test_ends": root / "test_ends.npy",
        "train_sample_ends": root / "train_sample_ends.npy",
        "val_early_ends": root / "val_early_ends.npy",
    }


def load_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], pd.DataFrame]:
    sequence_dir = resolve(config["sequence_input"])
    paths = array_paths(sequence_dir)
    required = [
        sequence_dir / "_SUCCESS",
        sequence_dir / "sequence_data" / "arrays" / "array_manifest.json",
        sequence_dir / "reports" / "cohort.csv",
        sequence_dir / "reports" / "feature_manifest.json",
        resolve(config["metadata_input"]),
        *paths.values(),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    manifest = load_json(sequence_dir / "sequence_data" / "arrays" / "array_manifest.json")
    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    cohort = pd.read_csv(sequence_dir / "reports" / "cohort.csv", dtype={"site_id": "string"})
    cohort["site_id"] = cohort["site_id"].astype(str)
    if len(cohort) != 1009 or int(manifest["n_sites"]) != 1009:
        raise ValueError("Expected the validated 1009-site full sequence cohort")
    if cohort["site_id"].tolist() != [str(value) for value in manifest["site_order"]]:
        raise ValueError("Cohort order does not match array site codes")
    feature_manifest = load_json(sequence_dir / "reports" / "feature_manifest.json")
    if feature_manifest["dynamic_features"] != config["dynamic_features"]:
        raise ValueError("Dynamic feature order differs from the completed sequence run")
    return manifest, arrays, cohort


def subset_ends(ends: np.ndarray, arrays: dict[str, np.ndarray], site_count: int) -> np.ndarray:
    values = np.asarray(ends, dtype=np.int64)
    return values[np.asarray(arrays["site_code"][values], dtype=np.int64) < site_count]


def validate_windows(ends: np.ndarray, arrays: dict[str, np.ndarray], site_count: int, length: int = 24) -> list[str]:
    problems: list[str] = []
    for start in range(0, len(ends), 1_000_000):
        current = np.asarray(ends[start : start + 1_000_000], dtype=np.int64)
        first = current - length + 1
        if np.any(first < 0):
            problems.append("window starts before array")
            break
        if np.any(arrays["site_code"][current] >= site_count):
            problems.append("window contains a site outside the cohort")
            break
        if np.any(arrays["site_code"][current] != arrays["site_code"][first]):
            problems.append("window crosses sites")
            break
        if np.any(arrays["timestamp_ns"][current] - arrays["timestamp_ns"][first] != (length - 1) * HOUR_NS):
            problems.append("window is not hourly continuous")
            break
    return problems


def train_exogenous_summaries(
    arrays: dict[str, np.ndarray],
    site_count: int,
    dynamic_names: list[str],
    sensor_names: list[str],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    ends = subset_ends(arrays["train_sample_ends"], arrays, site_count)
    codes = np.asarray(arrays["site_code"][ends], dtype=np.int64)
    order = np.argsort(codes, kind="stable")
    ends = ends[order]
    codes = codes[order]
    sensor_indices = [dynamic_names.index(name) for name in sensor_names]
    rows = np.full((site_count, len(sensor_names) * 2 + 1), np.nan, dtype=np.float64)
    counts = np.bincount(codes, minlength=site_count)
    offsets = np.r_[0, np.cumsum(counts)]
    for site in range(site_count):
        site_ends = ends[offsets[site] : offsets[site + 1]]
        if len(site_ends) == 0:
            continue
        values = np.asarray(arrays["features"][site_ends][:, sensor_indices], dtype=np.float64)
        rows[site, : len(sensor_names)] = np.median(values, axis=0)
        rows[site, len(sensor_names) : 2 * len(sensor_names)] = values.std(axis=0)
        timestamps = np.asarray(arrays["timestamp_ns"][site_ends], dtype=np.int64)
        rows[site, -1] = (timestamps.max() - timestamps.min()) / (24 * HOUR_NS)
    names = [f"meta_train_{name}_median" for name in sensor_names]
    names += [f"meta_train_{name}_std" for name in sensor_names]
    names += ["meta_train_coverage_days"]
    return rows, names, {"sampled_train_windows": int(len(ends)), "per_site_counts": counts.tolist()}


def standardize_columns(values: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    transformed = np.empty_like(values, dtype=np.float32)
    specs: list[dict[str, float]] = []
    for index in range(values.shape[1]):
        column = values[:, index].astype(np.float64)
        finite = np.isfinite(column)
        median = float(np.median(column[finite])) if finite.any() else 0.0
        column[~finite] = median
        mean = float(column.mean())
        scale = float(column.std())
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        transformed[:, index] = ((column - mean) / scale).astype(np.float32)
        specs.append({"imputation_median": median, "mean": mean, "scale": scale})
    return transformed, specs


def build_v2_metadata(
    site_count: int,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    output: Path,
) -> tuple[np.ndarray, list[str]]:
    raw = pd.read_csv(resolve(config["metadata_input"]), dtype={"site_id": "string"})
    raw["site_id"] = raw["site_id"].astype(str)
    raw = raw.set_index("site_id").loc[[str(value) for value in manifest["site_order"][:site_count]]]
    numeric_names = [str(value) for value in config["v2_official_numeric"]]
    categorical_names = [str(value) for value in config["v2_official_categorical"]]
    missing = [name for name in [*numeric_names, *categorical_names] if name not in raw.columns]
    if missing:
        raise ValueError(f"Missing V2 metadata fields: {missing}")
    numeric = raw[numeric_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    train_values, train_names, train_spec = train_exogenous_summaries(
        arrays,
        site_count,
        config["dynamic_features"],
        config["v2_train_summary_sensors"],
    )
    numeric_all = np.concatenate([numeric, train_values], axis=1)
    numeric_matrix, numeric_specs = standardize_columns(numeric_all)
    feature_names = [*numeric_names, *train_names]
    parts = [numeric_matrix]
    categorical_spec: dict[str, Any] = {}
    for name in categorical_names:
        values = pd.to_numeric(raw[name], errors="coerce").fillna(-1).astype(int)
        categories = sorted(values.unique().tolist())
        one_hot = np.zeros((site_count, len(categories)), dtype=np.float32)
        mapping = {value: index for index, value in enumerate(categories)}
        for row, value in enumerate(values):
            one_hot[row, mapping[int(value)]] = 1.0
        names = [f"{name}__{value}" for value in categories]
        parts.append(one_hot)
        feature_names.extend(names)
        categorical_spec[name] = {"categories": categories, "features": names}
    matrix = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("V2 metadata contains non-finite values")
    metadata_dir = output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    np.save(metadata_dir / f"metadata_N{site_count}.npy", matrix)
    write_json(
        metadata_dir / f"metadata_N{site_count}.json",
        {
            "site_count": site_count,
            "feature_names": feature_names,
            "numeric": {name: spec for name, spec in zip([*numeric_names, *train_names], numeric_specs)},
            "categorical": categorical_spec,
            "train_summary_audit": train_spec,
            "forbidden_inputs": config["forbidden_predictors"],
        },
    )
    return matrix, feature_names


def fit_dynamic_scaler(
    train_ends: np.ndarray,
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    length = int(config["sequence_length_hours"])
    offsets = np.arange(length - 1, -1, -1, dtype=np.int64)
    n = 0
    total = np.zeros(arrays["features"].shape[1], dtype=np.float64)
    total_sq = np.zeros_like(total)
    batch = int(config["scaler_batch_windows"])
    for start in range(0, len(train_ends), batch):
        ends = np.asarray(train_ends[start : start + batch], dtype=np.int64)
        positions = ends[:, None] - offsets[None, :]
        values = np.asarray(arrays["features"][positions], dtype=np.float64).reshape(-1, len(total))
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        n += len(values)
    mean = total / n
    scale = np.sqrt(np.maximum(total_sq / n - np.square(mean), 0.0))
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    write_json(
        path,
        {
            "fit_training_windows": int(len(train_ends)),
            "fit_exposure_rows_including_overlap": int(n),
            "features": config["dynamic_features"],
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        },
    )
    return mean.astype(np.float32), scale.astype(np.float32)


class ConditionedSequenceRegressor(nn.Module):
    def __init__(
        self,
        recurrent_type: str,
        conditioning: str,
        n_dynamic: int,
        hidden_size: int,
        layers: int,
        n_sites: int,
        site_embedding_size: int,
        metadata_size: int,
        film_hidden: int,
        mlp_hidden: int,
    ) -> None:
        super().__init__()
        recurrent_class = nn.GRU if recurrent_type == "gru" else nn.LSTM
        self.encoder = recurrent_class(n_dynamic, hidden_size, num_layers=layers, batch_first=True)
        self.site_embedding = nn.Embedding(n_sites, site_embedding_size)
        self.conditioning = conditioning
        if conditioning == "film":
            self.metadata_encoder = nn.Sequential(
                nn.Linear(metadata_size, film_hidden),
                nn.ReLU(),
                nn.Linear(film_hidden, hidden_size * 2),
            )
            final_layer = self.metadata_encoder[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
            head_input = hidden_size + site_embedding_size
        elif conditioning == "concat":
            self.metadata_encoder = None
            head_input = hidden_size + site_embedding_size + metadata_size
        else:
            raise ValueError(f"Unsupported conditioning: {conditioning}")
        self.head = nn.Sequential(nn.Linear(head_input, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))
        self.hidden_size = hidden_size

    def film_parameters(self, metadata: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.metadata_encoder is None:
            raise ValueError("FiLM parameters requested for concatenation model")
        gamma_beta = self.metadata_encoder(metadata)
        gamma_delta, beta = gamma_beta.chunk(2, dim=1)
        return 1.0 + gamma_delta, beta

    def forward(self, dynamic: torch.Tensor, site_code: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        output, _hidden = self.encoder(dynamic)
        embedding = self.site_embedding(site_code)
        if self.conditioning == "film":
            gamma, beta = self.film_parameters(metadata)
            output = gamma[:, None, :] * output + beta[:, None, :]
            pieces = [output[:, -1, :], embedding]
        else:
            pieces = [output[:, -1, :], embedding, metadata]
        return self.head(torch.cat(pieces, dim=1)).squeeze(1)


def model_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def batch_numpy(
    ends: np.ndarray,
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    metadata: np.ndarray,
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.arange(length - 1, -1, -1, dtype=np.int64)
    positions = ends[:, None] - offsets[None, :]
    dynamic = np.asarray(arrays["features"][positions], dtype=np.float32)
    dynamic = (dynamic - mean[None, None, :]) / scale[None, None, :]
    site = np.asarray(arrays["site_code"][ends], dtype=np.int64)
    target = np.asarray(arrays["target"][ends], dtype=np.float32)
    return (
        np.ascontiguousarray(dynamic),
        site,
        np.ascontiguousarray(metadata[site]),
        target,
    )


def create_model(spec: dict[str, Any], config: dict[str, Any], n_sites: int, metadata_size: int, device: torch.device) -> ConditionedSequenceRegressor:
    embedding_size = min(16, math.ceil(math.sqrt(n_sites)) + 1)
    return ConditionedSequenceRegressor(
        recurrent_type=str(spec["recurrent_type"]),
        conditioning=str(spec["conditioning"]),
        n_dynamic=len(config["dynamic_features"]),
        hidden_size=int(config["hidden_size"]),
        layers=int(config["recurrent_layers"]),
        n_sites=n_sites,
        site_embedding_size=embedding_size,
        metadata_size=metadata_size,
        film_hidden=int(config["film_hidden_size"]),
        mlp_hidden=int(config["mlp_hidden_size"]),
    ).to(device)


@torch.no_grad()
def predict_scaled(
    model: ConditionedSequenceRegressor,
    ends: np.ndarray,
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    metadata: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    prediction = np.empty(len(ends), dtype=np.float32)
    batch_size = int(config["prediction_batch_size"])
    length = int(config["sequence_length_hours"])
    for start in range(0, len(ends), batch_size):
        stop = min(start + batch_size, len(ends))
        selected = np.asarray(ends[start:stop], dtype=np.int64)
        dynamic, site, meta, _target = batch_numpy(selected, arrays, mean, scale, metadata, length)
        dynamic_tensor = torch.from_numpy(dynamic).to(device)
        site_tensor = torch.from_numpy(site).to(device)
        metadata_tensor = torch.from_numpy(meta).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            values = model(dynamic_tensor, site_tensor, metadata_tensor)
        prediction[start:stop] = values.float().cpu().numpy()
    return prediction


def train_model(
    spec: dict[str, Any],
    train_ends: np.ndarray,
    val_ends: np.ndarray,
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    metadata: np.ndarray,
    config: dict[str, Any],
    output: Path,
    device: torch.device,
    resume: bool,
) -> tuple[ConditionedSequenceRegressor, pd.DataFrame, dict[str, Any]]:
    model_id = str(spec["model_id"])
    model_dir = output / "models" / model_id
    best_path = model_dir / "best_model.pt"
    final_path = model_dir / "final_model.pt"
    success_path = model_dir / "_SUCCESS.json"
    history_path = output / "reports" / f"training_history_{model_id}.csv"
    resource_path = output / "reports" / f"resource_usage_{model_id}.json"
    if resume and success_path.exists() and best_path.exists():
        model = create_model(spec, config, metadata.shape[0], metadata.shape[1], device)
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, pd.read_csv(history_path), load_json(resource_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = model_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    seed = int(config["random_seed"])
    set_seed(seed)
    model = create_model(spec, config, metadata.shape[0], metadata.shape[1], device)
    initial_hash = model_state_hash(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    batch_size = int(config["train_batch_size"])
    length = int(config["sequence_length_hours"])
    rng = np.random.default_rng(seed)
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    rows: list[dict[str, Any]] = []
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    logging.info(
        "Training %s train=%s val=%s metadata=%s device=%s",
        model_id,
        f"{len(train_ends):,}",
        f"{len(val_ends):,}",
        metadata.shape[1],
        device,
    )
    for epoch in range(1, int(config["maximum_epochs"]) + 1):
        model.train()
        order = rng.permutation(len(train_ends))
        train_sse = 0.0
        train_n = 0
        for start in range(0, len(order), batch_size):
            selected = train_ends[order[start : start + batch_size]]
            dynamic, site, meta, target = batch_numpy(selected, arrays, mean, scale, metadata, length)
            dynamic_tensor = torch.from_numpy(dynamic).to(device)
            site_tensor = torch.from_numpy(site).to(device)
            metadata_tensor = torch.from_numpy(meta).to(device)
            target_tensor = torch.from_numpy(target).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model(dynamic_tensor, site_tensor, metadata_tensor)
                loss = nn.functional.mse_loss(prediction, target_tensor)
            loss.backward()
            optimizer.step()
            train_sse += float(loss.detach().cpu()) * len(selected)
            train_n += len(selected)
        val_prediction = predict_scaled(model, val_ends, arrays, mean, scale, metadata, config, device)
        val_target = np.asarray(arrays["target"][val_ends], dtype=np.float64)
        val_mse = float(np.square(val_target - val_prediction.astype(np.float64)).mean())
        row = {
            "model_id": model_id,
            "epoch": epoch,
            "train_mse_scaled": train_sse / train_n,
            "validation_mse_scaled": val_mse,
            "validation_rmse_native_ph": math.sqrt(val_mse) * float(config["target_inverse_multiplier"]),
            "elapsed_seconds": time.time() - started,
        }
        rows.append(row)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_spec": spec,
            "epoch": epoch,
            "validation_mse_scaled": val_mse,
            "initial_state_sha256": initial_hash,
        }
        torch.save(checkpoint, checkpoints / f"epoch_{epoch:02d}.pt")
        logging.info(
            "%s epoch=%s train_mse=%.7f val_rmse_native=%.5f",
            model_id,
            epoch,
            row["train_mse_scaled"],
            row["validation_rmse_native_ph"],
        )
        if val_mse < best_loss - 1e-10:
            best_loss = val_mse
            best_epoch = epoch
            stale_epochs = 0
            torch.save(checkpoint, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["early_stopping_patience"]):
                break
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_spec": spec,
            "epoch": rows[-1]["epoch"],
            "validation_mse_scaled": rows[-1]["validation_mse_scaled"],
            "initial_state_sha256": initial_hash,
        },
        final_path,
    )
    history = pd.DataFrame(rows)
    write_csv(history_path, history)
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    resource = {
        "model_id": model_id,
        "site_count": int(spec["site_count"]),
        "training_scope": spec["training_scope"],
        "train_windows": int(len(train_ends)),
        "validation_windows_for_early_stopping": int(len(val_ends)),
        "batch_size": batch_size,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_rmse_native_ph": math.sqrt(best_loss) * float(config["target_inverse_multiplier"]),
        "initial_state_sha256": initial_hash,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "bf16": bool(device.type == "cuda" and torch.cuda.is_bf16_supported()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    write_json(resource_path, resource)
    write_json(success_path, resource)
    return model, history, resource


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    observed64 = np.asarray(observed, dtype=np.float64)
    predicted64 = np.asarray(predicted, dtype=np.float64)
    residual = observed64 - predicted64
    denominator = float(np.square(observed64 - observed64.mean()).sum())
    return {
        "n_rows": int(len(observed64)),
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
    multiplier: float,
) -> pd.DataFrame:
    site_codes = np.asarray(arrays["site_code"][ends], dtype=np.int64)
    site_order = np.asarray(manifest["site_order"], dtype=object)
    networks = cohort.set_index("site_id").loc[[str(value) for value in manifest["site_order"]], "network"].astype(str).to_numpy(dtype=object)
    observed_scaled = np.asarray(arrays["target"][ends], dtype=np.float32)
    observed = observed_scaled * multiplier
    predicted = np.asarray(predicted_scaled, dtype=np.float32) * multiplier
    direction = np.asarray(arrays["event_direction"][ends], dtype=np.int8)
    return pd.DataFrame(
        {
            "model_id": model_id,
            "site_id": site_order[site_codes],
            "network": networks[site_codes],
            "timestamp": pd.to_datetime(np.asarray(arrays["timestamp_ns"][ends], dtype=np.int64), unit="ns"),
            "split": split,
            "observed_scaled": observed_scaled,
            "predicted_scaled": np.asarray(predicted_scaled, dtype=np.float32),
            "observed_ph": observed,
            "predicted_ph": predicted,
            "residual_observed_minus_predicted": observed - predicted,
            "abs_residual": np.abs(observed - predicted),
            "in_ph_event": direction != 0,
            "ph_event_direction": [DIRECTION_TO_NAME[int(value)] for value in direction],
            "event_id": np.asarray(arrays["event_id"][ends], dtype=np.int64),
        }
    )


def tail_analysis(frame: pd.DataFrame, fractions: Iterable[float]) -> list[dict[str, Any]]:
    residual = frame["residual_observed_minus_predicted"].to_numpy(dtype=np.float64)
    event = frame["in_ph_event"].astype(bool).to_numpy()
    direction = frame["ph_event_direction"].astype(str).to_numpy()
    base_rate = float(event.mean())
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        thresholds = {
            "absolute": float(np.quantile(np.abs(residual), 1 - fraction)),
            "negative": float(np.quantile(residual, fraction)),
            "positive": float(np.quantile(residual, 1 - fraction)),
        }
        masks = {
            "absolute": np.abs(residual) >= thresholds["absolute"],
            "negative": residual <= thresholds["negative"],
            "positive": residual >= thresholds["positive"],
        }
        for tail_type, mask in masks.items():
            selected_rate = float(event[mask].mean()) if mask.any() else np.nan
            rows.append(
                {
                    "model_id": frame["model_id"].iloc[0],
                    "tail_fraction_requested": float(fraction),
                    "tail_type": tail_type,
                    "threshold_native_ph": thresholds[tail_type],
                    "n_test_rows": len(frame),
                    "n_selected_rows": int(mask.sum()),
                    "base_event_rate": base_rate,
                    "selected_event_rate": selected_rate,
                    "event_enrichment": selected_rate / base_rate if base_rate > 0 else np.nan,
                    "selected_low_rows": int(((direction == "low") & mask).sum()),
                    "selected_high_rows": int(((direction == "high") & mask).sum()),
                    "selected_mixed_rows": int(((direction == "mixed") & mask).sum()),
                }
            )
    return rows


def evaluate_model(
    model: ConditionedSequenceRegressor,
    spec: dict[str, Any],
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    cohort: pd.DataFrame,
    mean: np.ndarray,
    scale: np.ndarray,
    metadata: np.ndarray,
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model_id = str(spec["model_id"])
    site_count = int(spec["site_count"])
    overall_rows: list[dict[str, Any]] = []
    network_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        ends = subset_ends(arrays[f"{split}_ends"], arrays, site_count)
        predicted = predict_scaled(model, ends, arrays, mean, scale, metadata, config, device)
        frame = prediction_frame(
            model_id,
            split,
            ends,
            predicted,
            arrays,
            manifest,
            cohort,
            float(config["target_inverse_multiplier"]),
        )
        if frame.isna().any().any() or frame.duplicated(["site_id", "timestamp"]).any():
            raise ValueError(f"Invalid prediction frame for {model_id}/{split}")
        write_parquet(output / "predictions" / f"prediction_{split}_{model_id}.parquet", frame)
        overall_rows.append(
            {
                "model_id": model_id,
                "site_count": site_count,
                "recurrent_type": spec["recurrent_type"],
                "conditioning": spec["conditioning"],
                "training_scope": spec["training_scope"],
                "split": split,
                "n_sites": int(frame["site_id"].nunique()),
                **metric_values(frame["observed_ph"], frame["predicted_ph"]),
            }
        )
        for network, group in frame.groupby("network", sort=True):
            network_rows.append(
                {
                    "model_id": model_id,
                    "site_count": site_count,
                    "training_scope": spec["training_scope"],
                    "split": split,
                    "network": network,
                    "n_sites": int(group["site_id"].nunique()),
                    **metric_values(group["observed_ph"], group["predicted_ph"]),
                }
            )
        for site_id, group in frame.groupby("site_id", sort=True):
            site_rows.append(
                {
                    "model_id": model_id,
                    "site_count": site_count,
                    "training_scope": spec["training_scope"],
                    "split": split,
                    "site_id": site_id,
                    "network": group["network"].iloc[0],
                    **metric_values(group["observed_ph"], group["predicted_ph"]),
                }
            )
        if split == "test":
            tail_rows.extend(tail_analysis(frame, config["residual_tail_fractions"]))
        del frame, predicted
        gc.collect()
    return overall_rows, network_rows, site_rows, tail_rows


def non_event_audit(train_ends: np.ndarray, arrays: dict[str, np.ndarray], manifest: dict[str, Any]) -> pd.DataFrame:
    site = np.asarray(arrays["site_code"][train_ends], dtype=np.int64)
    direction = np.asarray(arrays["event_direction"][train_ends], dtype=np.int8)
    frame = pd.DataFrame({"site_code": site, "direction_code": direction})
    counts = frame.groupby(["site_code", "direction_code"]).size().unstack(fill_value=0)
    for value in range(4):
        if value not in counts.columns:
            counts[value] = 0
    counts = counts[[0, 1, 2, 3]].reset_index()
    counts.columns = ["site_code", "retained_nonevent", "removed_low", "removed_high", "removed_mixed"]
    result = pd.DataFrame({"site_code": np.arange(len(manifest["site_order"]), dtype=int)})
    result["site_id"] = [str(value) for value in manifest["site_order"]]
    result = result.merge(counts, on="site_code", how="left").fillna(0)
    count_columns = ["retained_nonevent", "removed_low", "removed_high", "removed_mixed"]
    result[count_columns] = result[count_columns].astype(int)
    result["all_sampled_train"] = result[count_columns].sum(axis=1)
    result["removed_any_event"] = result[["removed_low", "removed_high", "removed_mixed"]].sum(axis=1)
    result["removed_fraction"] = np.where(
        result["all_sampled_train"] > 0,
        result["removed_any_event"] / result["all_sampled_train"],
        np.nan,
    )
    return result


def same_prediction_keys(path_a: Path, path_b: Path) -> bool:
    a = pd.read_parquet(path_a, columns=["site_id", "timestamp", "observed_ph"])
    b = pd.read_parquet(path_b, columns=["site_id", "timestamp", "observed_ph"])
    return len(a) == len(b) and a.equals(b)


def save_plots(
    output: Path,
    overall: pd.DataFrame,
    tails: pd.DataFrame,
    winner_base: str,
    nonevent_id: str,
) -> None:
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    test = overall[overall["split"].eq("test")]
    ablation = test[test["site_count"].eq(200)].sort_values("rmse")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ablation["model_id"], ablation["rmse"], color="#4C78A8")
    ax.set_ylabel("Test RMSE (native pH)")
    ax.set_title("N=200 concatenation versus FiLM ablation")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "n200_film_ablation_rmse.png", dpi=200)
    plt.close(fig)

    scale = test[
        test["model_id"].str.match(rf"^{winner_base}_N(50|200|500|1009)$")
    ].sort_values("site_count")
    tail = tails[
        tails["model_id"].isin(scale["model_id"])
        & tails["tail_fraction_requested"].eq(0.05)
        & tails["tail_type"].eq("absolute")
    ][["model_id", "event_enrichment"]]
    scale = scale.merge(tail, on="model_id", how="left")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(scale["site_count"], scale["rmse"], marker="o", color="#4C78A8", label="RMSE")
    ax2.plot(scale["site_count"], scale["event_enrichment"], marker="s", color="#E45756", label="Top-5% enrichment")
    ax1.set_xlabel("Number of sites")
    ax1.set_ylabel("Test RMSE (native pH)", color="#4C78A8")
    ax2.set_ylabel("Absolute top-5% event enrichment", color="#E45756")
    ax1.set_title(f"{winner_base} site-count scaling")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "site_scaling_rmse_enrichment.png", dpi=200)
    plt.close(fig)

    comparison = test[test["model_id"].isin([f"{winner_base}_N1009", nonevent_id])]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(comparison["training_scope"], comparison["rmse"], color=["#4C78A8", "#C08A45"])
    ax.set_ylabel("Test RMSE (native pH)")
    ax.set_title("N=1009 all-data versus target-row non-event training")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "nonevent_sequence_rmse.png", dpi=200)
    plt.close(fig)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def report_text(
    overall: pd.DataFrame,
    tails: pd.DataFrame,
    winner_base: str,
    winner_n200: str,
    nonevent_id: str,
    problems: list[str],
) -> str:
    n200 = overall[(overall["site_count"].eq(200)) & overall["split"].eq("test")].sort_values("rmse")
    scale = overall[
        overall["split"].eq("test") & overall["model_id"].str.match(rf"^{winner_base}_N(50|200|500|1009)$")
    ].sort_values("site_count")
    all_id = f"{winner_base}_N1009"
    pair = overall[overall["split"].eq("test") & overall["model_id"].isin([all_id, nonevent_id])].set_index("model_id")
    tail_pair = tails[
        tails["model_id"].isin([all_id, nonevent_id])
        & tails["tail_fraction_requested"].eq(0.05)
        & tails["tail_type"].isin(["absolute", "negative", "positive"])
    ]
    lines = [
        "# Common1000 large-machine FiLM and scaling report",
        "",
        "## N=200 architecture ablation",
        "",
        "| Model | RMSE | MAE | R2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in n200.itertuples(index=False):
        lines.append(f"| {row.model_id} | {row.rmse:.6f} | {row.mae:.6f} | {row.r2:.6f} |")
    lines.extend(
        [
            "",
            f"FiLM winner selected by N=200 validation RMSE: `{winner_n200}`.",
            "",
            "## Site-count scaling",
            "",
            "| Sites | Model | Test rows | RMSE | R2 |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in scale.itertuples(index=False):
        lines.append(f"| {row.site_count} | {row.model_id} | {row.n_rows:,} | {row.rmse:.6f} | {row.r2:.6f} |")
    if len(pair) == 2:
        all_row = pair.loc[all_id]
        non_row = pair.loc[nonevent_id]
        lines.extend(
            [
                "",
                "## N=1009 controlled non-event sequence result",
                "",
                f"- All-data RMSE/R2: {all_row.rmse:.6f} / {all_row.r2:.6f}.",
                f"- Non-event RMSE/R2: {non_row.rmse:.6f} / {non_row.r2:.6f}.",
                f"- RMSE change (non-event minus all): {non_row.rmse - all_row.rmse:+.6f}.",
            ]
        )
        for tail_type in ("absolute", "negative", "positive"):
            subset = tail_pair[tail_pair["tail_type"].eq(tail_type)].set_index("model_id")
            if all_id in subset.index and nonevent_id in subset.index:
                lines.append(
                    f"- {tail_type} top-5% enrichment: "
                    f"{subset.loc[all_id, 'event_enrichment']:.3f} -> {subset.loc[nonevent_id, 'event_enrichment']:.3f}."
                )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- FiLM modulates every recurrent output state with metadata-derived gamma and beta.",
            "- No pH history or event field is a predictor.",
            "- All cohorts are nested prefixes of the same quality-ranked site order.",
            "- Scaling metrics use each cohort's own included test sites; event windows are incomplete weak labels.",
            "- This stage completes site-count scaling at hourly resolution; finer-resolution rebuilding is a separate later experiment.",
            "",
            "## Validation",
            "",
            f"- Problems: {problems if problems else 'none'}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    validate_config(config)
    output = resolve(config["output_root"])
    success_path = output / "_SUCCESS"
    if args.resume and success_path.exists():
        print(f"Validated FiLM/scaling output already exists: {success_path}")
        return
    if output.exists() and not args.resume:
        expected = resolve(config["output_root"]).resolve()
        if output.resolve() != expected:
            raise RuntimeError(f"Refusing to clear unexpected path: {output}")
        shutil.rmtree(output)
    for name in ("models", "predictions", "reports", "metadata", "scalers", "figures"):
        (output / name).mkdir(parents=True, exist_ok=True)
    setup_logging(output)
    started = time.time()
    manifest, arrays, cohort = load_inputs(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The large-machine FiLM experiment requires the available CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The configured large-machine experiment requires BF16 support")
    logging.info("Starting FiLM/scaling experiment on %s", torch.cuda.get_device_name(device))

    problems: list[str] = []
    overall_rows: list[dict[str, Any]] = []
    network_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    resources: dict[str, dict[str, Any]] = {}
    cohort_inputs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def prepare(site_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if site_count in cohort_inputs:
            return cohort_inputs[site_count]
        train_ends = subset_ends(arrays["train_sample_ends"], arrays, site_count)
        val_early = subset_ends(arrays["val_early_ends"], arrays, site_count)
        for name, ends in (("train", train_ends), ("validation", val_early)):
            current_problems = validate_windows(ends, arrays, site_count, int(config["sequence_length_hours"]))
            if current_problems:
                raise ValueError(f"N{site_count} {name} windows invalid: {current_problems}")
        metadata, metadata_names = build_v2_metadata(site_count, manifest, arrays, config, output)
        mean, scale = fit_dynamic_scaler(
            train_ends,
            arrays,
            config,
            output / "scalers" / f"dynamic_scaler_N{site_count}.json",
        )
        write_csv(output / "reports" / f"cohort_N{site_count}.csv", cohort.iloc[:site_count].copy())
        write_json(
            output / "reports" / f"feature_manifest_N{site_count}.json",
            {
                "dynamic_features": config["dynamic_features"],
                "metadata_features": metadata_names,
                "target": "00400",
                "forbidden_predictors": config["forbidden_predictors"],
            },
        )
        cohort_inputs[site_count] = (train_ends, val_early, metadata, mean, scale)
        return cohort_inputs[site_count]

    def run_spec(spec: dict[str, Any], train_override: np.ndarray | None = None) -> None:
        site_count = int(spec["site_count"])
        train_ends, val_early, metadata, mean, scale = prepare(site_count)
        fit_ends = train_override if train_override is not None else train_ends
        model, _history, resource = train_model(
            spec,
            fit_ends,
            val_early,
            arrays,
            mean,
            scale,
            metadata,
            config,
            output,
            device,
            args.resume,
        )
        resources[str(spec["model_id"])] = resource
        overall, networks, sites, tails = evaluate_model(
            model,
            spec,
            arrays,
            manifest,
            cohort,
            mean,
            scale,
            metadata,
            config,
            output,
            device,
        )
        overall_rows.extend(overall)
        network_rows.extend(networks)
        site_rows.extend(sites)
        tail_rows.extend(tails)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    ablation_n = int(config["ablation_site_count"])
    ablation_specs = []
    for recurrent in ("gru", "lstm"):
        for conditioning in ("concat", "film"):
            ablation_specs.append(
                {
                    "model_id": f"{recurrent.upper()}_{conditioning.upper()}_V2_N{ablation_n}",
                    "recurrent_type": recurrent,
                    "conditioning": conditioning,
                    "site_count": ablation_n,
                    "training_scope": "all_data",
                }
            )
    for spec in ablation_specs:
        run_spec(spec)

    ablation_metrics = pd.DataFrame(overall_rows)
    film_val = ablation_metrics[
        ablation_metrics["split"].eq("val") & ablation_metrics["conditioning"].eq("film")
    ].sort_values("rmse")
    if len(film_val) != 2:
        raise RuntimeError("N200 FiLM ablation did not produce both recurrent cells")
    winner_n200 = str(film_val.iloc[0]["model_id"])
    winner_recurrent = str(film_val.iloc[0]["recurrent_type"])
    winner_base = f"{winner_recurrent.upper()}_FILM_V2"
    write_json(
        output / "reports" / "n200_winner_selection.json",
        {
            "selection_split": "validation",
            "selection_metric": "RMSE native pH",
            "winner_model_id": winner_n200,
            "winner_recurrent_type": winner_recurrent,
            "candidates": film_val[["model_id", "rmse", "mae", "r2"]].to_dict(orient="records"),
        },
    )
    logging.info("N200 FiLM validation winner: %s", winner_n200)

    for site_count in [int(value) for value in config["site_counts"] if int(value) != ablation_n]:
        run_spec(
            {
                "model_id": f"{winner_base}_N{site_count}",
                "recurrent_type": winner_recurrent,
                "conditioning": "film",
                "site_count": site_count,
                "training_scope": "all_data",
            }
        )

    full_id = f"{winner_base}_N1009"
    full_train, _full_val, _metadata, _mean, _scale = prepare(1009)
    event_direction = np.asarray(arrays["event_direction"][full_train], dtype=np.int8)
    nonevent_train = full_train[event_direction == 0]
    audit = non_event_audit(full_train, arrays, manifest)
    write_csv(output / "reports" / "nonevent_training_by_site.csv", audit)
    nonevent_id = f"{winner_base}_N1009_NONEVENT"
    run_spec(
        {
            "model_id": nonevent_id,
            "recurrent_type": winner_recurrent,
            "conditioning": "film",
            "site_count": 1009,
            "training_scope": "non_event_target_only",
        },
        train_override=nonevent_train,
    )
    if resources[full_id]["initial_state_sha256"] != resources[nonevent_id]["initial_state_sha256"]:
        problems.append("all-data and non-event N1009 models do not share identical initialisation")

    overall = pd.DataFrame(overall_rows)
    networks = pd.DataFrame(network_rows)
    sites = pd.DataFrame(site_rows)
    tails = pd.DataFrame(tail_rows)
    write_csv(output / "reports" / "model_metrics.csv", overall)
    write_csv(output / "reports" / "network_metrics.csv", networks)
    write_csv(output / "reports" / "site_metrics.csv", sites)
    write_csv(output / "reports" / "residual_tail_metrics.csv", tails)
    write_csv(output / "reports" / "resource_usage.csv", pd.DataFrame(resources.values()))

    scaling_ids = [f"{winner_base}_N{value}" for value in config["site_counts"]]
    scaling = overall[overall["model_id"].isin(scaling_ids) & overall["split"].eq("test")].copy()
    scaling = scaling.merge(
        tails[
            tails["model_id"].isin(scaling_ids)
            & tails["tail_fraction_requested"].eq(0.05)
            & tails["tail_type"].eq("absolute")
        ][["model_id", "event_enrichment"]],
        on="model_id",
        how="left",
    )
    write_csv(output / "reports" / "site_scaling_curve.csv", scaling)

    # Fair-key assertions for N200 and the N1009 controlled pair.
    n200_ids = [str(spec["model_id"]) for spec in ablation_specs]
    for split in ("val", "test"):
        reference = output / "predictions" / f"prediction_{split}_{n200_ids[0]}.parquet"
        for model_id in n200_ids[1:]:
            if not same_prediction_keys(reference, output / "predictions" / f"prediction_{split}_{model_id}.parquet"):
                problems.append(f"N200 {split} prediction keys differ for {model_id}")
        if not same_prediction_keys(
            output / "predictions" / f"prediction_{split}_{full_id}.parquet",
            output / "predictions" / f"prediction_{split}_{nonevent_id}.parquet",
        ):
            problems.append(f"N1009 all/non-event {split} keys differ")
    if len(nonevent_train) + int((event_direction != 0).sum()) != len(full_train):
        problems.append("non-event training endpoint reconciliation failed")
    if set(scaling["site_count"].unique()) != set(int(value) for value in config["site_counts"]):
        problems.append("site scaling curve is incomplete")
    if not np.isfinite(overall[["rmse", "mae", "r2"]].to_numpy(dtype=float)).all():
        problems.append("overall metrics contain non-finite values")
    for count in config["site_counts"]:
        manifest_path = output / "reports" / f"feature_manifest_N{count}.json"
        payload = load_json(manifest_path)
        inputs = [*payload["dynamic_features"], *payload["metadata_features"]]
        if any(value in inputs for value in ["00400", "in_ph_event", "event_id"]):
            problems.append(f"N{count} feature leakage")

    save_plots(output, overall, tails, winner_base, nonevent_id)
    report = report_text(overall, tails, winner_base, winner_n200, nonevent_id, problems)
    (output / "reports" / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "status": "success" if not problems else "failed_validation",
        "winner_n200": winner_n200,
        "winner_base": winner_base,
        "nonevent_model": nonevent_id,
        "site_counts": config["site_counts"],
        "models_trained": overall["model_id"].drop_duplicates().tolist(),
        "full_train_windows": int(len(full_train)),
        "nonevent_train_windows": int(len(nonevent_train)),
        "event_target_windows_removed": int((event_direction != 0).sum()),
        "validation_problems": problems,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "bf16": True,
    }
    write_json(output / "reports" / "run_summary.json", summary)
    hash_paths = [
        config_path,
        SCRIPT_PATH,
        resolve(config["sequence_input"]) / "sequence_data" / "arrays" / "array_manifest.json",
        output / "reports" / "model_metrics.csv",
        output / "reports" / "site_scaling_curve.csv",
        output / "reports" / "residual_tail_metrics.csv",
        output / "reports" / "FINAL_REPORT.md",
    ]
    hashes = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in hash_paths
    ]
    write_csv(output / "reports" / "artifact_hashes.csv", pd.DataFrame(hashes))
    if problems:
        raise RuntimeError(f"Final validation failed: {problems}")
    write_json(
        success_path,
        {
            "status": "success",
            "winner": winner_base,
            "completed_at_unix": time.time(),
            "report": str((output / "reports" / "FINAL_REPORT.md").resolve()),
        },
    )
    logging.info("Completed FiLM/scaling/non-event workflow in %.1f seconds", time.time() - started)


if __name__ == "__main__":
    main()
