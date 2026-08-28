"""Run a common-key rainfall ablation on the 910 direct teacher-Archive sites."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = (
    PROJECT_ROOT
    / "pipeline"
    / "21_common1000_leave_sites_out_models"
    / "run_leave_sites_out_models.py"
)
SPEC = importlib.util.spec_from_file_location("leave_site_models", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)
PREPARE = BASE.PREPARE
CALENDAR_FEATURES = tuple(PREPARE.CALENDAR_FEATURES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Generate validation predictions only; intended for hyperparameter selection.",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    base_path = config.get("base_config")
    if not base_path:
        return config
    resolved_base = resolve(base_path)
    if resolved_base.resolve() == path.resolve():
        raise ValueError(f"Configuration inherits itself: {path}")
    base = load_config(resolved_base)
    base.update({key: value for key, value in config.items() if key != "base_config"})
    base["base_config"] = str(resolved_base)
    return base


def write_json(path: Path, payload: Any) -> None:
    BASE.write_json(path, payload)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    BASE.write_csv(path, frame)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    BASE.write_parquet(path, frame)


def sha256(path: Path) -> str:
    return BASE.sha256(path)


def transform_values(values: np.ndarray, transform: str) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    if transform == "identity":
        return result
    if transform == "log1p_nonnegative":
        result[result < 0] = np.nan
        return np.log1p(result)
    raise ValueError(f"Unknown feature transform: {transform}")


def key_digest(site_id: str, timestamps: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(site_id.encode("utf-8"))
    digest.update(np.asarray(timestamps, dtype="datetime64[ns]").view("int64").tobytes())
    return digest.hexdigest()


def target_values(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    source_column = str(
        config.get(
            "target_source_column",
            f"target_{config['target_code']}_native",
        )
    )
    target = pd.to_numeric(frame[source_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    target = target * float(config.get("target_source_multiplier", 1.0))
    observed_column = config.get("target_observed_column")
    if observed_column:
        observed = frame[str(observed_column)].fillna(False).to_numpy(dtype=bool)
        target = np.where(observed, target, np.nan)
    return target


def common_indices(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    target = target_values(frame, config)
    horizon = int(config["horizon_hours"])
    sequence = int(config["sequence_length_hours"])
    indices = np.flatnonzero(np.isfinite(target))
    indices = indices[indices - horizon - sequence + 1 >= 0]
    if not len(indices):
        return indices

    missing = np.column_stack(
        [
            frame[f"x_{code}_missing"].to_numpy(dtype=np.int8)
            for code in config["base_dynamic_codes"]
        ]
    )
    hourly_missing = missing.sum(axis=1).astype(np.int64)
    prefix = np.concatenate(([0], np.cumsum(hourly_missing, dtype=np.int64)))
    end = indices - horizon
    start = end - sequence + 1
    missing_total = prefix[end + 1] - prefix[start]
    fraction = missing_total / float(sequence * len(config["base_dynamic_codes"]))
    valid = fraction <= float(config["maximum_dynamic_missing_fraction_per_window"])

    if bool(config["require_precipitation_for_every_input_hour"]):
        available = frame["precip_feature_available"].fillna(False).to_numpy(dtype=bool)
        available_prefix = np.concatenate(([0], np.cumsum(available, dtype=np.int64)))
        available_count = available_prefix[end + 1] - available_prefix[start]
        valid &= available_count == sequence

    return indices[valid]


def comparison_indices(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> np.ndarray:
    """Return target keys shared across structural hyperparameter variants."""
    anchor_sequence = int(
        config.get(
            "comparison_anchor_sequence_length_hours",
            config["sequence_length_hours"],
        )
    )
    model_sequence = int(config["sequence_length_hours"])
    if model_sequence > anchor_sequence:
        raise ValueError(
            "sequence_length_hours cannot exceed "
            "comparison_anchor_sequence_length_hours"
        )
    anchor_config = {**config, "sequence_length_hours": anchor_sequence}
    return common_indices(frame, anchor_config)


def select_smoke_split(split: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    pieces = []
    seed = int(config["random_seed"])
    for split_name in ("train", "validation", "test"):
        group = split[split["split"].eq(split_name)].copy()
        group["_key"] = group["site_id"].map(
            lambda value: hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
        )
        count = int(config["smoke_sites_per_split"][split_name])
        pieces.append(group.sort_values("_key").head(count).drop(columns="_key"))
    return pd.concat(pieces, ignore_index=True)


def fit_preprocessing(
    split: pd.DataFrame,
    input_root: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    definitions = config["feature_definitions"]
    feature_sums = {name: 0.0 for name in definitions}
    feature_sum_sq = {name: 0.0 for name in definitions}
    feature_counts = {name: 0 for name in definitions}
    target_sum = 0.0
    target_sum_sq = 0.0
    target_count = 0
    key_rows = []

    for row in split.itertuples(index=False):
        site_id = str(row.site_id)
        frame = pd.read_parquet(input_root / f"{site_id}.parquet")
        index = comparison_indices(frame, config)
        timestamps = pd.to_datetime(frame.loc[index, "timestamp"]).to_numpy(dtype="datetime64[ns]")
        key_rows.append(
            {
                "site_id": site_id,
                "split": row.split,
                "common_target_windows": len(index),
                "first_common_target": timestamps.min() if len(timestamps) else pd.NaT,
                "last_common_target": timestamps.max() if len(timestamps) else pd.NaT,
                "common_key_sha256": key_digest(site_id, timestamps),
            }
        )
        if row.split != "train":
            continue

        available = frame["precip_feature_available"].fillna(False).to_numpy(dtype=bool)
        for name, definition in definitions.items():
            raw = pd.to_numeric(
                frame[definition["source_column"]], errors="coerce"
            ).to_numpy(dtype=np.float64)
            values = transform_values(raw, definition["transform"])
            values = values[available & np.isfinite(values)]
            feature_counts[name] += len(values)
            feature_sums[name] += float(values.sum())
            feature_sum_sq[name] += float(np.square(values).sum())

        target = target_values(frame, config)
        values = target[index]
        values = values[np.isfinite(values)]
        target_count += len(values)
        target_sum += float(values.sum())
        target_sum_sq += float(np.square(values).sum())

    features = {}
    for name in definitions:
        count = feature_counts[name]
        if count == 0:
            raise RuntimeError(f"No training values for {name}")
        mean = feature_sums[name] / count
        variance = max(feature_sum_sq[name] / count - mean**2, 0.0)
        scale = math.sqrt(variance)
        features[name] = {
            "source_column": definitions[name]["source_column"],
            "transform": definitions[name]["transform"],
            "mean": mean,
            "scale": scale if scale >= 1e-8 else 1.0,
            "fit_nonmissing_rows": count,
        }
    if target_count == 0:
        raise RuntimeError("No common training targets")
    target_mean = target_sum / target_count
    target_variance = max(target_sum_sq / target_count - target_mean**2, 0.0)
    target_scale = math.sqrt(target_variance)
    preprocessing = {
        "fit_split": "train_sites_only",
        "fit_time_domain": "teacher_archive_feature_available_hours",
        "feature_transform_order": "exact Archive feature -> configured transform -> training-only standardization",
        "imputation": "training-site mean with missing indicator retained",
        "features": features,
        "target": {
            "mean": target_mean,
            "scale": target_scale if target_scale >= 1e-8 else 1.0,
            "fit_common_target_rows": target_count,
        },
    }
    return preprocessing, pd.DataFrame(key_rows)


def site_payload(
    path: Path,
    feature_names: list[str],
    preprocessing: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    columns: list[np.ndarray] = []
    output_names: list[str] = []
    for name in feature_names:
        stat = preprocessing["features"][name]
        raw = pd.to_numeric(frame[stat["source_column"]], errors="coerce").to_numpy(
            dtype=np.float64
        )
        values = transform_values(raw, stat["transform"])
        missing = ~np.isfinite(values)
        values = np.where(missing, float(stat["mean"]), values)
        columns.extend(
            [
                ((values - float(stat["mean"])) / float(stat["scale"])).astype(np.float32),
                missing.astype(np.float32),
            ]
        )
        output_names.extend([f"{name}_standardized", f"{name}_missing"])

    if (
        bool(config.get("include_do_source_family_indicator", True))
        and "x_00300_percent_family" in frame
    ):
        columns.append(frame["x_00300_percent_family"].to_numpy(dtype=np.float32))
        output_names.append("x_00300_percent_family")
    # Existing experiments retain the historical six calendar channels when
    # `calendar_features` is absent.  Explicit experiment configs may now
    # request a strict subset (including none) for controlled calendar
    # ablations without changing any sensor preprocessing.
    for field in tuple(config.get("calendar_features", CALENDAR_FEATURES)):
        columns.append(frame[field].to_numpy(dtype=np.float32))
        output_names.append(field)

    target = target_values(frame, config).astype(np.float32)
    if bool(config.get("include_target_history", True)):
        filled = (
            pd.Series(target)
            .ffill(limit=int(config["target_history_fill_limit_hours"]))
            .to_numpy(dtype=np.float32)
        )
        history_missing = ~np.isfinite(filled)
        target_mean = float(preprocessing["target"]["mean"])
        target_scale = float(preprocessing["target"]["scale"])
        filled = np.where(history_missing, target_mean, filled)
        columns.extend(
            [
                ((filled - target_mean) / target_scale).astype(np.float32),
                history_missing.astype(np.float32),
            ]
        )
        output_names.extend(
            ["target_history_standardized", "target_history_missing"]
        )

    return {
        "features": np.column_stack(columns).astype(np.float32),
        "feature_names": output_names,
        "target": target,
        "timestamp": pd.DatetimeIndex(frame["timestamp"]).to_numpy(dtype="datetime64[ns]"),
        "in_event": frame["in_event"].to_numpy(dtype=bool),
        "event_direction": frame["event_direction"].astype(str).to_numpy(),
        "event_id": frame["event_id"].to_numpy(dtype=np.int64),
        "frame": frame,
    }


def capped_indices(
    indices: np.ndarray,
    cap: int,
    seed: int,
) -> np.ndarray:
    if cap > 0 and len(indices) > cap:
        return np.sort(np.random.default_rng(seed).choice(indices, cap, replace=False))
    return indices


def window_batches(
    payload: dict[str, Any],
    indices: np.ndarray,
    metadata: np.ndarray,
    config: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    order = indices.copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    sequence = int(config["sequence_length_hours"])
    horizon = int(config["horizon_hours"])
    offsets = np.arange(sequence - 1, -1, -1, dtype=np.int64)
    for start in range(0, len(order), batch_size):
        target_index = order[start : start + batch_size]
        input_end = target_index - horizon
        positions = input_end[:, None] - offsets[None, :]
        dynamic = np.ascontiguousarray(payload["features"][positions])
        meta = np.ascontiguousarray(np.tile(metadata, (len(target_index), 1)))
        target = np.ascontiguousarray(payload["target"][target_index])
        yield dynamic, meta, target, target_index


def validation_rmse(
    model: nn.Module,
    site_ids: list[str],
    feature_names: list[str],
    metadata_map: dict[str, np.ndarray],
    preprocessing: dict[str, Any],
    config: dict[str, Any],
    input_root: Path,
    settings: dict[str, Any],
    device: torch.device,
) -> float:
    model.eval()
    squared = 0.0
    count = 0
    mean = float(preprocessing["target"]["mean"])
    scale = float(preprocessing["target"]["scale"])
    with torch.no_grad():
        for number, site_id in enumerate(site_ids):
            payload = site_payload(
                input_root / f"{site_id}.parquet", feature_names, preprocessing, config
            )
            index = comparison_indices(payload["frame"], config)
            index = capped_indices(
                index,
                int(settings["validation_window_cap_per_site"]),
                int(config["random_seed"]) + number,
            )
            for dynamic, meta, target, _ in window_batches(
                payload,
                index,
                metadata_map[site_id],
                config,
                int(settings["prediction_batch_size"]),
                False,
                0,
            ):
                prediction = (
                    model(
                        torch.from_numpy(dynamic).to(device),
                        torch.from_numpy(meta).to(device),
                    )
                    .float()
                    .cpu()
                    .numpy()
                    * scale
                    + mean
                )
                squared += float(np.square(prediction - target).sum())
                count += len(target)
    return math.sqrt(squared / count) if count else float("inf")


def train_model(
    model: nn.Module,
    train_ids: list[str],
    validation_ids: list[str],
    feature_names: list[str],
    metadata_map: dict[str, np.ndarray],
    preprocessing: dict[str, Any],
    config: dict[str, Any],
    input_root: Path,
    settings: dict[str, Any],
    device: torch.device,
    train_non_event: bool,
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.get("learning_rate_scheduler_factor", 0.5)),
            patience=int(config.get("learning_rate_scheduler_patience", 2)),
        )
        if bool(config.get("learning_rate_scheduler_enabled", False))
        else None
    )
    loss_function = nn.MSELoss()
    gradient_clip_norm = float(config.get("gradient_clip_norm", 0.0))
    minimum_epochs = int(config.get("minimum_epochs", 1))
    minimum_delta = float(config.get("early_stopping_min_delta", 1e-8))
    target_mean = float(preprocessing["target"]["mean"])
    target_scale = float(preprocessing["target"]["scale"])
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = 0
    stale = 0
    event_removed = 0
    history = []

    for epoch in range(1, int(settings["maximum_epochs"]) + 1):
        model.train()
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_loss = 0.0
        epoch_n = 0
        order = train_ids.copy()
        random.Random(int(config["random_seed"]) + epoch).shuffle(order)
        for site_number, site_id in enumerate(order):
            payload = site_payload(
                input_root / f"{site_id}.parquet", feature_names, preprocessing, config
            )
            all_index = comparison_indices(payload["frame"], config)
            index = (
                all_index[~payload["in_event"][all_index]]
                if train_non_event
                else all_index
            )
            if epoch == 1:
                event_removed += int(len(all_index) - len(index))
            cap = int(settings["train_window_cap_per_site"])
            if cap > 0 and len(index) > cap:
                index = np.random.default_rng(
                    int(config["random_seed"]) + epoch * 100003 + site_number
                ).choice(index, cap, replace=False)
            for dynamic, meta, target, _ in window_batches(
                payload,
                index,
                metadata_map[site_id],
                config,
                int(settings["train_batch_size"]),
                True,
                int(config["random_seed"]) + epoch + site_number,
            ):
                dynamic_tensor = torch.from_numpy(dynamic).to(device)
                meta_tensor = torch.from_numpy(meta).to(device)
                target_tensor = torch.from_numpy(
                    ((target - target_mean) / target_scale).astype(np.float32)
                ).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    prediction = model(dynamic_tensor, meta_tensor)
                    loss = loss_function(prediction, target_tensor)
                loss.backward()
                if gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
                epoch_loss += float(loss.item()) * len(target)
                epoch_n += len(target)

        score = validation_rmse(
            model,
            validation_ids,
            feature_names,
            metadata_map,
            preprocessing,
            config,
            input_root,
            settings,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse_scaled": epoch_loss / max(epoch_n, 1),
                "train_windows": epoch_n,
                "validation_rmse_native": score,
                "learning_rate": epoch_learning_rate,
            }
        )
        print(
            f"{feature_names[-1]} epoch={epoch} train_n={epoch_n} val_rmse={score:.6f}",
            flush=True,
        )
        if score < best_validation - minimum_delta:
            best_validation = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step(score)
        if (
            epoch >= minimum_epochs
            and stale >= int(config["early_stopping_patience"])
        ):
            break

    model.load_state_dict(best_state)
    return (
        model,
        pd.DataFrame(history),
        {
            "best_validation_rmse_native": best_validation,
            "best_epoch": int(best_epoch),
            "epochs_ran": int(len(history)),
            "train_event_windows_removed": event_removed,
            "gradient_clip_norm": gradient_clip_norm,
            "learning_rate_scheduler_enabled": scheduler is not None,
        },
    )


def prediction_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return PREPARE.metric_values(
        observed.astype(np.float64), predicted.astype(np.float64)
    )


def predict_model(
    model: nn.Module,
    split: pd.DataFrame,
    feature_names: list[str],
    metadata_map: dict[str, np.ndarray],
    preprocessing: dict[str, Any],
    config: dict[str, Any],
    input_root: Path,
    settings: dict[str, Any],
    output: Path,
    run_id: str,
    device: torch.device,
    model_spec: dict[str, Any],
    train_non_event: bool,
    prediction_splits: tuple[str, ...] = ("validation", "test"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    site_rows = []
    overall_parts: dict[str, dict[str, list[np.ndarray]]] = {
        split_name: {"observed": [], "predicted": []}
        for split_name in prediction_splits
    }
    target_mean = float(preprocessing["target"]["mean"])
    target_scale = float(preprocessing["target"]["scale"])
    horizon = int(config["horizon_hours"])

    for split_name in prediction_splits:
        for site_id in split.loc[split["split"].eq(split_name), "site_id"].astype(str):
            payload = site_payload(
                input_root / f"{site_id}.parquet", feature_names, preprocessing, config
            )
            index = comparison_indices(payload["frame"], config)
            predicted_parts = []
            index_parts = []
            with torch.no_grad():
                for dynamic, meta, _target, target_index in window_batches(
                    payload,
                    index,
                    metadata_map[site_id],
                    config,
                    int(settings["prediction_batch_size"]),
                    False,
                    0,
                ):
                    prediction = (
                        model(
                            torch.from_numpy(dynamic).to(device),
                            torch.from_numpy(meta).to(device),
                        )
                        .float()
                        .cpu()
                        .numpy()
                        * target_scale
                        + target_mean
                    )
                    predicted_parts.append(prediction)
                    index_parts.append(target_index)
            if not index_parts:
                continue
            index = np.concatenate(index_parts)
            predicted = np.concatenate(predicted_parts)
            observed = payload["target"][index]
            input_end = index - horizon
            frame = payload["frame"]
            prediction_frame = pd.DataFrame(
                {
                    "site_id": site_id,
                    "timestamp": payload["timestamp"][index],
                    "split": split_name,
                    "target": config["target"],
                    "horizon_hours": horizon,
                    "sequence_length_hours": int(
                        config["sequence_length_hours"]
                    ),
                    "model_id": run_id,
                    "observed_native": observed,
                    "predicted_native": predicted.astype(np.float32),
                    "in_event": payload["in_event"][index],
                    "event_direction": payload["event_direction"][index],
                    "event_id": payload["event_id"][index],
                    "input_end_rain_mm": pd.to_numeric(
                        frame["00045"], errors="coerce"
                    ).to_numpy(dtype=np.float32)[input_end],
                    "input_end_precip_api": pd.to_numeric(
                        frame["precip_api"], errors="coerce"
                    ).to_numpy(dtype=np.float32)[input_end],
                    "input_end_precip_6h": pd.to_numeric(
                        frame["precip_6h"], errors="coerce"
                    ).to_numpy(dtype=np.float32)[input_end],
                }
            )
            write_parquet(
                output
                / "predictions_by_site"
                / run_id
                / split_name
                / f"{site_id}.parquet",
                prediction_frame,
            )
            metric = prediction_metrics(observed, predicted)
            site_rows.append(
                {
                    "run_id": run_id,
                    "target": config["target"],
                    "split": split_name,
                    "site_id": site_id,
                    "horizon_hours": horizon,
                    "sequence_length_hours": int(
                        config["sequence_length_hours"]
                    ),
                    **metric,
                }
            )
            overall_parts[split_name]["observed"].append(observed)
            overall_parts[split_name]["predicted"].append(predicted)

    sites = pd.DataFrame(site_rows)
    overall_rows = []
    for split_name in prediction_splits:
        observed_parts = overall_parts[split_name]["observed"]
        predicted_parts = overall_parts[split_name]["predicted"]
        if not observed_parts:
            continue
        metric = prediction_metrics(
            np.concatenate(observed_parts), np.concatenate(predicted_parts)
        )
        subset = sites[sites["split"].eq(split_name)]
        finite_nse = pd.to_numeric(subset["nse"], errors="coerce")
        overall_rows.append(
            {
                "run_id": run_id,
                "target": config["target"],
                "split": split_name,
                "horizon_hours": horizon,
                "sequence_length_hours": int(
                    config["sequence_length_hours"]
                ),
                "comparison_anchor_sequence_length_hours": int(
                    config.get(
                        "comparison_anchor_sequence_length_hours",
                        config["sequence_length_hours"],
                    )
                ),
                "recurrent_type": str(model_spec["recurrent_type"]),
                "recurrent_layers": int(model_spec["recurrent_layers"]),
                "conditioning": str(model_spec["conditioning"]),
                "metadata_variant": str(model_spec["metadata_variant"]),
                "target_history": bool(model_spec["target_history"]),
                "train_non_event": train_non_event,
                **metric,
                "sites": int(subset["site_id"].nunique()),
                "macro_site_rmse": float(subset["rmse"].mean()),
                "macro_site_mae": float(subset["mae"].mean()),
                "macro_site_nse": float(finite_nse[np.isfinite(finite_nse)].mean()),
            }
        )
    return pd.DataFrame(overall_rows), sites


def exact_top_mask(values: np.ndarray, fraction: float, largest: bool) -> np.ndarray:
    count = max(1, int(math.ceil(len(values) * fraction)))
    order = np.argsort(values, kind="mergesort")
    selected = order[-count:] if largest else order[:count]
    mask = np.zeros(len(values), dtype=bool)
    mask[selected] = True
    return mask


def evaluate_test_predictions(
    output: Path,
    run_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enrichment_rows = []
    rain_rows = []
    for run_id in run_ids:
        files = sorted((output / "predictions_by_site" / run_id / "test").glob("*.parquet"))
        prediction = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        observed = prediction["observed_native"].to_numpy(dtype=np.float64)
        predicted = prediction["predicted_native"].to_numpy(dtype=np.float64)
        residual = predicted - observed
        absolute = np.abs(residual)
        event = prediction["in_event"].to_numpy(dtype=bool)
        low = prediction["event_direction"].astype(str).eq("low").to_numpy()
        high = prediction["event_direction"].astype(str).eq("high").to_numpy()
        absolute_mask = exact_top_mask(absolute, 0.05, True)
        negative_mask = exact_top_mask(residual, 0.05, False)
        positive_mask = exact_top_mask(residual, 0.05, True)
        base_event = float(event.mean())
        base_low = float(low.mean())
        base_high = float(high.mean())
        enrichment_rows.append(
            {
                "run_id": run_id,
                "test_rows": len(prediction),
                "event_rows": int(event.sum()),
                "base_event_rate": base_event,
                "absolute_top5_event_rate": float(event[absolute_mask].mean()),
                "absolute_top5_event_enrichment": (
                    float(event[absolute_mask].mean() / base_event)
                    if base_event > 0
                    else np.nan
                ),
                "low_event_rows": int(low.sum()),
                "negative_top5_low_enrichment": (
                    float(low[negative_mask].mean() / base_low)
                    if base_low > 0
                    else np.nan
                ),
                "high_event_rows": int(high.sum()),
                "positive_top5_high_enrichment": (
                    float(high[positive_mask].mean() / base_high)
                    if base_high > 0
                    else np.nan
                ),
            }
        )
        for rain_column in (
            "input_end_rain_mm",
            "input_end_precip_api",
            "input_end_precip_6h",
        ):
            rain = pd.to_numeric(prediction[rain_column], errors="coerce")
            valid = rain.notna().to_numpy()
            wet = rain.to_numpy(dtype=np.float64) > 0
            rain_rows.append(
                {
                    "run_id": run_id,
                    "rain_variable": rain_column,
                    "n": int(valid.sum()),
                    "spearman_abs_residual": float(
                        pd.Series(absolute[valid]).corr(
                            pd.Series(rain.to_numpy(dtype=np.float64)[valid]),
                            method="spearman",
                        )
                    ),
                    "spearman_signed_residual": float(
                        pd.Series(residual[valid]).corr(
                            pd.Series(rain.to_numpy(dtype=np.float64)[valid]),
                            method="spearman",
                        )
                    ),
                    "dry_rows": int((valid & ~wet).sum()),
                    "wet_rows": int((valid & wet).sum()),
                    "dry_rmse": float(np.sqrt(np.mean(np.square(residual[valid & ~wet])))),
                    "wet_rmse": float(np.sqrt(np.mean(np.square(residual[valid & wet])))),
                }
            )
    return pd.DataFrame(enrichment_rows), pd.DataFrame(rain_rows)


def audit_predictions(
    output: Path,
    run_ids: list[str],
    common_keys: pd.DataFrame,
    prediction_splits: tuple[str, ...] = ("validation", "test"),
) -> list[str]:
    problems = []
    expected = common_keys.set_index(["split", "site_id"])
    reference: dict[tuple[str, str], np.ndarray] = {}
    for run_id in run_ids:
        for split_name in prediction_splits:
            for path in sorted(
                (output / "predictions_by_site" / run_id / split_name).glob("*.parquet")
            ):
                site_id = path.stem
                frame = pd.read_parquet(path, columns=["site_id", "timestamp"])
                timestamps = pd.to_datetime(frame["timestamp"]).to_numpy(dtype="datetime64[ns]")
                row = expected.loc[(split_name, site_id)]
                if len(frame) != int(row["common_target_windows"]):
                    problems.append(f"wrong prediction row count: {run_id}/{split_name}/{site_id}")
                if key_digest(site_id, timestamps) != str(row["common_key_sha256"]):
                    problems.append(f"wrong prediction keys: {run_id}/{split_name}/{site_id}")
                key = (split_name, site_id)
                if key not in reference:
                    reference[key] = timestamps
                elif not np.array_equal(reference[key], timestamps):
                    problems.append(f"model key mismatch: {run_id}/{split_name}/{site_id}")
                if problems:
                    return problems
    return problems


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = resolve(args.config)
    config = load_config(config_path)
    output_mode = (
        f"{args.mode}_validation_only" if args.validation_only else args.mode
    )
    output = resolve(config["output_root"]) / output_mode
    input_root = resolve(config["merged_input_root"])
    for directory in ("models", "reports", "predictions_by_site", "metadata"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    write_json(output / "reports" / "effective_config.json", config)

    cohort = pd.read_csv(
        resolve(config["primary_cohort_path"]), dtype={"site_id": "string"}
    )
    cohort["site_id"] = cohort["site_id"].map(PREPARE.normalise_site_id)
    cohort = cohort[
        cohort["primary_archive_cohort"].fillna(False).astype(bool)
        & cohort["ready_for_common_key_experiment"].fillna(False).astype(bool)
    ][["site_id", "split", "network"]].drop_duplicates("site_id")
    if len(cohort) != 910:
        raise RuntimeError(f"Expected 910 primary Archive sites, found {len(cohort)}")
    if args.validation_only:
        # Validation-only selection must not open held-out test time-series
        # files while fitting preprocessing or constructing common keys.
        cohort = cohort.loc[cohort["split"].isin(("train", "validation"))].copy()
    if args.mode == "smoke":
        cohort = select_smoke_split(cohort, config)

    preprocessing, common_keys = fit_preprocessing(cohort, input_root, config)
    write_json(output / "reports" / "train_only_preprocessing.json", preprocessing)
    write_csv(output / "reports" / "common_key_manifest.csv", common_keys)
    split_summary = (
        common_keys.groupby("split", as_index=False)
        .agg(
            sites=("site_id", "nunique"),
            common_target_windows=("common_target_windows", "sum"),
            minimum_windows_per_site=("common_target_windows", "min"),
            median_windows_per_site=("common_target_windows", "median"),
        )
    )
    write_csv(output / "reports" / "common_key_split_summary.csv", split_summary)
    if (common_keys["common_target_windows"] <= 0).any():
        bad = common_keys.loc[
            common_keys["common_target_windows"] <= 0, "site_id"
        ].tolist()
        raise RuntimeError(f"Sites without common windows: {bad[:10]}")

    metadata_source = BASE.merge_metadata(config)
    metadata_fields = [
        field
        for field in config["metadata_fields"]
        if field not in set(config["coordinate_fields"])
    ]
    metadata_map, metadata_parameters = BASE.fit_metadata(
        metadata_source,
        cohort,
        metadata_fields,
        set(config["metadata_log1p_fields"]),
    )
    metadata_filename = (
        f"{str(config.get('target', 'ph'))}_"
        "no_coordinates_train_only_transform.json"
    )
    write_json(output / "metadata" / metadata_filename, metadata_parameters)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This experiment requires CUDA")
    settings = config[args.mode]
    prediction_splits = ("validation",) if args.validation_only else ("validation", "test")
    selected = set(args.model)
    all_overall = []
    all_sites = []
    run_records = []
    completed_run_ids = []

    base_model_spec = {
        "recurrent_layers": int(config.get("recurrent_layers", 1)),
        "conditioning": "film",
        "metadata_variant": "no_coordinates",
        "target_history": bool(config.get("include_target_history", True)),
        "train_non_event": True,
    }
    for model_definition in config["models"]:
        model_id = model_definition["model_id"]
        if selected and model_id not in selected:
            continue
        allowed_training_overrides = {
            "learning_rate",
            "weight_decay",
            "dropout",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "minimum_epochs",
            "learning_rate_scheduler_enabled",
            "learning_rate_scheduler_factor",
            "learning_rate_scheduler_patience",
            "gradient_clip_norm",
        }
        training_overrides = dict(model_definition.get("training_overrides", {}))
        unsupported_overrides = set(training_overrides) - allowed_training_overrides
        if unsupported_overrides:
            raise ValueError(
                f"Unsupported training_overrides for {model_id}: "
                f"{sorted(unsupported_overrides)}"
            )
        model_config = {**config, **training_overrides}
        model_config["sequence_length_hours"] = int(
            model_definition.get(
                "sequence_length_hours",
                config["sequence_length_hours"],
            )
        )
        model_config["include_target_history"] = bool(
            model_definition.get(
                "target_history",
                config.get("include_target_history", True),
            )
        )
        anchor_sequence = int(
            model_config.get(
                "comparison_anchor_sequence_length_hours",
                model_config["sequence_length_hours"],
            )
        )
        if int(model_config["sequence_length_hours"]) > anchor_sequence:
            raise ValueError(
                f"{model_id} sequence_length_hours exceeds the common-key "
                f"anchor ({anchor_sequence})"
            )
        feature_names = list(model_definition["feature_names"])
        train_non_event = bool(model_definition.get("train_non_event", True))
        recurrent_type = str(
            model_definition.get(
                "recurrent_type",
                config.get("recurrent_type", "gru"),
            )
        ).lower()
        if recurrent_type not in {"gru", "lstm"}:
            raise ValueError(
                f"Unsupported recurrent_type for {model_id}: {recurrent_type}"
            )
        run_id = str(
            model_definition.get(
                "run_id",
                f"{str(config.get('target', 'ph'))}_{model_id}_H"
                f"{int(model_config['horizon_hours']):02d}",
            )
        )
        current_model_spec = {
            **base_model_spec,
            "model_id": model_id,
            "recurrent_type": recurrent_type,
            "recurrent_layers": int(
                model_definition.get(
                    "recurrent_layers",
                    base_model_spec["recurrent_layers"],
                )
            ),
            "train_non_event": train_non_event,
            "target_history": bool(model_config["include_target_history"]),
        }
        first_site = str(cohort.iloc[0]["site_id"])
        sample = site_payload(
            input_root / f"{first_site}.parquet",
            feature_names,
            preprocessing,
            model_config,
        )
        run_seed = int(model_config["random_seed"]) + int(
            model_config["horizon_hours"]
        ) * 1000
        BASE.set_seed(run_seed)
        model = BASE.SiteAgnosticFiLMRNN(
            current_model_spec,
            sample["features"].shape[1],
            len(metadata_map[first_site]),
            model_config,
        ).to(device)
        model_path = output / "models" / f"{run_id}.pt"
        success_path = output / "models" / f"{run_id}_SUCCESS.json"
        if args.resume and model_path.exists() and success_path.exists():
            model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
            training_info = load_json(success_path)
        else:
            model, history, training_info = train_model(
                model,
                cohort.loc[cohort["split"].eq("train"), "site_id"].astype(str).tolist(),
                cohort.loc[
                    cohort["split"].eq("validation"), "site_id"
                ].astype(str).tolist(),
                feature_names,
                metadata_map,
                preprocessing,
                model_config,
                input_root,
                settings,
                device,
                train_non_event,
            )
            torch.save(model.state_dict(), model_path)
            write_csv(output / "reports" / f"training_history_{run_id}.csv", history)
            training_info.update(
                {
                    "run_id": run_id,
                    "feature_set": feature_names,
                    "input_feature_names": sample["feature_names"],
                    "dynamic_size": sample["features"].shape[1],
                    "metadata_size": len(metadata_map[first_site]),
                    "recurrent_type": recurrent_type,
                    "recurrent_layers": int(
                        current_model_spec["recurrent_layers"]
                    ),
                    "dropout": float(model_config.get("dropout", 0.0)),
                    "learning_rate": float(model_config["learning_rate"]),
                    "weight_decay": float(model_config["weight_decay"]),
                    "sequence_length_hours": int(
                        model_config["sequence_length_hours"]
                    ),
                    "comparison_anchor_sequence_length_hours": anchor_sequence,
                    "trainable_parameters": int(
                        sum(
                            parameter.numel()
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        )
                    ),
                    "no_site_embedding": True,
                    "include_do_source_family_indicator": bool(
                        config.get("include_do_source_family_indicator", True)
                    ),
                    "common_key_manifest_sha256": sha256(
                        output / "reports" / "common_key_manifest.csv"
                    ),
                }
            )
            write_json(success_path, training_info)

        overall, sites = predict_model(
            model,
            cohort,
            feature_names,
            metadata_map,
            preprocessing,
            model_config,
            input_root,
            settings,
            output,
            run_id,
            device,
            current_model_spec,
            train_non_event,
            prediction_splits,
        )
        overall["train_event_windows_removed"] = int(
            training_info["train_event_windows_removed"]
        )
        overall["feature_set"] = "|".join(feature_names)
        write_csv(output / "reports" / f"overall_metrics_{run_id}.csv", overall)
        write_csv(output / "reports" / f"site_metrics_{run_id}.csv", sites)
        all_overall.append(overall)
        all_sites.append(sites)
        run_records.append(
            {
                "run_id": run_id,
                "model_id": model_id,
                "recurrent_type": recurrent_type,
                "recurrent_layers": int(
                    current_model_spec["recurrent_layers"]
                ),
                "trainable_parameters": int(
                    sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    )
                ),
                "feature_set": "|".join(feature_names),
                "train_non_event": train_non_event,
                "learning_rate": float(model_config["learning_rate"]),
                "weight_decay": float(model_config["weight_decay"]),
                "dropout": float(model_config.get("dropout", 0.0)),
                "sequence_length_hours": int(
                    model_config["sequence_length_hours"]
                ),
                "comparison_anchor_sequence_length_hours": anchor_sequence,
                "site_count": len(cohort),
            }
        )
        completed_run_ids.append(run_id)

    overall_all = pd.concat(all_overall, ignore_index=True)
    site_all = pd.concat(all_sites, ignore_index=True)
    write_csv(output / "reports" / "model_metrics_all.csv", overall_all)
    write_csv(output / "reports" / "site_metrics_all.csv", site_all)
    write_csv(output / "reports" / "run_manifest.csv", pd.DataFrame(run_records))
    evaluate_events = bool(config.get("evaluate_event_diagnostics", True))
    if args.validation_only or not evaluate_events:
        enrichment = pd.DataFrame()
        rain_relationship = pd.DataFrame()
    else:
        enrichment, rain_relationship = evaluate_test_predictions(
            output, completed_run_ids
        )
    write_csv(output / "reports" / "event_enrichment_top5.csv", enrichment)
    write_csv(
        output / "reports" / "rain_residual_relationship.csv", rain_relationship
    )

    problems = audit_predictions(
        output,
        completed_run_ids,
        common_keys,
        prediction_splits,
    )
    if not np.isfinite(
        overall_all[["rmse", "mae", "macro_site_rmse"]].to_numpy(dtype=np.float64)
    ).all():
        problems.append("non-finite primary metrics")
    evaluation_split = "validation" if args.validation_only else "test"
    evaluation = overall_all[
        overall_all["split"].eq(evaluation_split)
    ].sort_values("rmse")
    report_lines = [
        "# Common-key teacher-Archive rainfall ablation",
        "",
        f"Mode: {args.mode}. Evaluation split: {evaluation_split}. "
        f"Sites: {len(cohort)}. Training modes: "
        + ", ".join(
            f"{row['model_id']}={row['recurrent_type']}/"
            f"{'non-event' if row['train_non_event'] else 'all-data'}"
            for row in run_records
        )
        + ".",
        "Causal input windows by model: "
        + ", ".join(
            f"{row['model_id']}={row['sequence_length_hours']}h"
            for row in run_records
        )
        + f". All models share target timestamps anchored to "
        f"{int(config.get('comparison_anchor_sequence_length_hours', config['sequence_length_hours']))}h eligibility.",
        "The primary cohort contains only direct canonical teacher-Archive rainfall sites.",
        "",
        PREPARE.frame_to_markdown(
            evaluation[
                [
                    "run_id",
                    "sequence_length_hours",
                    "n",
                    "sites",
                    "rmse",
                    "mae",
                    "nse",
                    "macro_site_rmse",
                    "macro_site_nse",
                ]
            ]
        ),
        "",
    ]
    if not args.validation_only and evaluate_events:
        report_lines.extend(
            [
                "## Top-5% residual/event enrichment",
                "",
                PREPARE.frame_to_markdown(enrichment),
                "",
                "## Rainfall/residual relationship",
                "",
                PREPARE.frame_to_markdown(rain_relationship),
                "",
            ]
        )
    elif args.validation_only:
        report_lines.extend(
            [
                "Hyperparameters are ranked on validation data only. "
                "No test predictions or test metrics were generated.",
                "",
            ]
        )
    else:
        report_lines.extend(
            [
                "Event diagnostics were intentionally disabled for this target; "
                "the upstream event labels describe pH events and are not valid "
                "DO-event labels.",
                "",
            ]
        )
    (output / "reports" / "FINAL_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    audit = {
        "status": "pass" if not problems else "fail",
        "mode": args.mode,
        "output_mode": output_mode,
        "validation_only_tuning": bool(args.validation_only),
        "prediction_splits": list(prediction_splits),
        "test_predictions_generated": not args.validation_only,
        "test_metrics_generated": not args.validation_only,
        "event_diagnostics_generated": (
            not args.validation_only and evaluate_events
        ),
        "sites": len(cohort),
        "expected_full_sites": 910,
        "site_split_counts": cohort.groupby("split").size().to_dict(),
        "common_keys_shared_by_all_models": not problems,
        "comparison_anchor_sequence_length_hours": int(
            config.get(
                "comparison_anchor_sequence_length_hours",
                config["sequence_length_hours"],
            )
        ),
        "sequence_lengths_by_model": {
            row["model_id"]: row["sequence_length_hours"]
            for row in run_records
        },
        "training_only_preprocessing": True,
        "target_history_by_model": {
            row["model_id"]: bool(
                next(
                    model.get(
                        "target_history",
                        config.get("include_target_history", True),
                    )
                    for model in config["models"]
                    if model["model_id"] == row["model_id"]
                )
            )
            for row in run_records
        },
        "target_history_is_causal_when_enabled": True,
        "training_modes": {
            row["model_id"]: (
                "non_event" if row["train_non_event"] else "all_data"
            )
            for row in run_records
        },
        "recurrent_types": {
            row["model_id"]: row["recurrent_type"] for row in run_records
        },
        "teacher_aligned_optimisation": {
            "dropout": float(config.get("dropout", 0.0)),
            "gradient_clip_norm": float(config.get("gradient_clip_norm", 0.0)),
            "scheduler_enabled": bool(
                config.get("learning_rate_scheduler_enabled", False)
            ),
            "scheduler_factor": float(
                config.get("learning_rate_scheduler_factor", 0.5)
            ),
            "scheduler_patience": int(
                config.get("learning_rate_scheduler_patience", 2)
            ),
        },
        "teacher_archive_api_formula_preserved_upstream": True,
        "gpu": torch.cuda.get_device_name(0),
        "bf16": torch.cuda.is_bf16_supported(),
        "problems": problems,
        "elapsed_seconds": time.time() - started,
        "config_sha256": sha256(config_path),
        "effective_config_sha256": sha256(
            output / "reports" / "effective_config.json"
        ),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(output / "reports" / "independent_audit.json", audit)
    if problems:
        raise RuntimeError(problems)
    write_json(output / "_SUCCESS", audit)
    print("\n".join(report_lines), flush=True)


if __name__ == "__main__":
    main()
