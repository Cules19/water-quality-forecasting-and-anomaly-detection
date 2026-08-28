#!/usr/bin/env python
"""Fair same-window baselines for the Common1000 sequence cohort."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
MAIN_MODEL_IDS = [
    "B0_site_mean",
    "B1_seasonal_mean",
    "B2_persistence_1h",
    "B3_R0_ridge",
    "B3_R1_ridge_metadata",
    "B4_H0_hgb",
    "B4_H1_hgb_metadata",
]
NONEVENT_MODEL_IDS = [
    "B3_R0_ridge_nonevent",
    "B3_R1_ridge_metadata_nonevent",
    "B4_H0_hgb_nonevent",
    "B4_H1_hgb_metadata_nonevent",
]
SEQUENCE_MODEL_IDS = ["GRU_S0", "GRU_S1", "LSTM_S1"]
CODE_TO_DIRECTION = {0: "none", 1: "low", 2: "high", 3: "mixed"}
HOUR_NS = 3_600_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "common1000_baseline_v1.json",
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
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


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(output_dir / "run.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "sequence_root",
        "output_root",
        "dynamic_features",
        "ridge_alpha_grid",
        "hgb_parameters",
        "target_inverse_multiplier",
        "residual_tail_fractions",
        "forbidden_predictors",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing config fields: {missing}")
    if len(config["dynamic_features"]) != 11:
        raise ValueError("Common1000 baseline expects the same 11 dynamic features as the sequence models")
    if any(float(value) <= 0 for value in config["ridge_alpha_grid"]):
        raise ValueError("Ridge alpha values must be positive")


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
    }


def load_inputs(sequence_dir: Path, config: dict[str, Any], mode: str) -> tuple[dict[str, Any], dict[str, np.ndarray], pd.DataFrame, np.ndarray, list[str]]:
    required = [
        sequence_dir / "_SUCCESS",
        sequence_dir / "sequence_data" / "arrays" / "array_manifest.json",
        sequence_dir / "sequence_data" / "metadata_matrix.npy",
        sequence_dir / "sequence_data" / "metadata_transform.json",
        sequence_dir / "reports" / "cohort.csv",
        sequence_dir / "reports" / "feature_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    paths = array_paths(sequence_dir)
    missing.extend(str(path) for path in paths.values() if not path.exists())
    if missing:
        raise FileNotFoundError(f"Missing sequence inputs: {missing}")
    manifest = load_json(sequence_dir / "sequence_data" / "arrays" / "array_manifest.json")
    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    cohort = pd.read_csv(sequence_dir / "reports" / "cohort.csv", dtype={"site_id": "string"})
    cohort["site_id"] = cohort["site_id"].astype(str)
    metadata = np.load(sequence_dir / "sequence_data" / "metadata_matrix.npy", mmap_mode="r")
    metadata_names = load_json(sequence_dir / "sequence_data" / "metadata_transform.json")["feature_names"]
    sequence_features = load_json(sequence_dir / "reports" / "feature_manifest.json")["dynamic_features"]
    if sequence_features != config["dynamic_features"]:
        raise ValueError(f"Dynamic feature mismatch: sequence={sequence_features}, baseline={config['dynamic_features']}")
    if len(cohort) != int(manifest["n_sites"]) or metadata.shape != (len(cohort), len(metadata_names)):
        raise ValueError("Cohort, site order and metadata matrix are not aligned")
    if mode == "full":
        expectations = {
            "n_sites": config["expected_full_sites"],
            "training_sample_windows": config["expected_full_train_sample_windows"],
        }
        for field, expected in expectations.items():
            if int(manifest[field]) != int(expected):
                raise ValueError(f"Unexpected full {field}: {manifest[field]} != {expected}")
        if len(arrays["val_ends"]) != int(config["expected_full_validation_windows"]):
            raise ValueError("Unexpected full validation window count")
        if len(arrays["test_ends"]) != int(config["expected_full_test_windows"]):
            raise ValueError("Unexpected full test window count")
    return manifest, arrays, cohort, metadata, metadata_names


def validate_windows(arrays: dict[str, np.ndarray]) -> list[str]:
    problems: list[str] = []
    n_rows = len(arrays["target"])
    for name in ("train_sample_ends", "val_ends", "test_ends"):
        ends = arrays[name]
        if len(ends) == 0:
            problems.append(f"{name} is empty")
            continue
        for start in range(0, len(ends), 1_000_000):
            values = np.asarray(ends[start : start + 1_000_000], dtype=np.int64)
            if np.any(values < 23) or np.any(values >= n_rows):
                problems.append(f"{name} contains out-of-bound endpoints")
                break
            if np.any(arrays["timestamp_ns"][values] - arrays["timestamp_ns"][values - 23] != 23 * HOUR_NS):
                problems.append(f"{name} contains non-continuous 24-hour windows")
                break
            if np.any(arrays["site_code"][values] != arrays["site_code"][values - 23]):
                problems.append(f"{name} contains cross-site windows")
                break
    return problems


def feature_matrix(ends: np.ndarray, arrays: dict[str, np.ndarray], metadata: np.ndarray, include_metadata: bool) -> np.ndarray:
    dynamic = np.asarray(arrays["features"][ends], dtype=np.float32)
    if not include_metadata:
        return dynamic
    site = np.asarray(arrays["site_code"][ends], dtype=np.int64)
    return np.concatenate([dynamic, np.asarray(metadata[site], dtype=np.float32)], axis=1)


def batches(values: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(values), size):
        yield np.asarray(values[start : start + size], dtype=np.int64)


def fit_simple_statistics(arrays: dict[str, np.ndarray], manifest: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    ends = np.asarray(arrays["train_sample_ends"], dtype=np.int64)
    site = np.asarray(arrays["site_code"][ends], dtype=np.int64)
    target = np.asarray(arrays["target"][ends], dtype=np.float64)
    n_sites = int(manifest["n_sites"])
    site_count = np.bincount(site, minlength=n_sites)
    site_sum = np.bincount(site, weights=target, minlength=n_sites)
    global_mean = float(target.mean())
    site_mean = np.full(n_sites, global_mean, dtype=np.float64)
    has_training = site_count > 0
    site_mean[has_training] = site_sum[has_training] / site_count[has_training]
    month_index = feature_names.index("month")
    hour_index = feature_names.index("hour")
    month = np.rint(np.asarray(arrays["features"][ends, month_index])).astype(np.int16)
    hour = np.rint(np.asarray(arrays["features"][ends, hour_index])).astype(np.int16)
    if np.any((month < 1) | (month > 12) | (hour < 0) | (hour > 23)):
        raise ValueError("Invalid month/hour values in sampled training endpoints")
    site_month_key = site * 12 + month - 1
    site_month_hour_key = site_month_key * 24 + hour
    sm_count = np.bincount(site_month_key, minlength=n_sites * 12)
    sm_sum = np.bincount(site_month_key, weights=target, minlength=n_sites * 12)
    smh_count = np.bincount(site_month_hour_key, minlength=n_sites * 12 * 24)
    smh_sum = np.bincount(site_month_hour_key, weights=target, minlength=n_sites * 12 * 24)
    return {
        "global_mean": global_mean,
        "site_has_sampled_training": has_training,
        "site_mean": site_mean.astype(np.float64),
        "site_month_count": sm_count,
        "site_month_sum": sm_sum,
        "site_month_hour_count": smh_count,
        "site_month_hour_sum": smh_sum,
    }


def predict_site_mean(ends: np.ndarray, arrays: dict[str, np.ndarray], statistics: dict[str, Any]) -> np.ndarray:
    site = np.asarray(arrays["site_code"][ends], dtype=np.int64)
    return np.asarray(statistics["site_mean"][site], dtype=np.float32)


def predict_seasonal(ends: np.ndarray, arrays: dict[str, np.ndarray], statistics: dict[str, Any], feature_names: list[str]) -> np.ndarray:
    site = np.asarray(arrays["site_code"][ends], dtype=np.int64)
    month = np.rint(np.asarray(arrays["features"][ends, feature_names.index("month")])).astype(np.int16)
    hour = np.rint(np.asarray(arrays["features"][ends, feature_names.index("hour")])).astype(np.int16)
    sm_key = site * 12 + month - 1
    smh_key = sm_key * 24 + hour
    result = np.asarray(statistics["site_mean"][site], dtype=np.float64)
    sm_count = statistics["site_month_count"][sm_key]
    sm_mask = sm_count > 0
    result[sm_mask] = statistics["site_month_sum"][sm_key[sm_mask]] / sm_count[sm_mask]
    smh_count = statistics["site_month_hour_count"][smh_key]
    smh_mask = smh_count > 0
    result[smh_mask] = statistics["site_month_hour_sum"][smh_key[smh_mask]] / smh_count[smh_mask]
    return result.astype(np.float32)


def validate_persistence(ends: np.ndarray, arrays: dict[str, np.ndarray]) -> list[str]:
    problems: list[str] = []
    for current in batches(ends, 1_000_000):
        previous = current - 1
        if np.any(previous < 0):
            problems.append("Persistence contains a missing previous row")
            break
        if np.any(arrays["site_code"][current] != arrays["site_code"][previous]):
            problems.append("Persistence crosses site boundaries")
            break
        if np.any(arrays["timestamp_ns"][current] - arrays["timestamp_ns"][previous] != HOUR_NS):
            problems.append("Persistence is not exactly one hour")
            break
    return problems


def predict_persistence(ends: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    problems = validate_persistence(ends, arrays)
    if problems:
        raise ValueError(problems)
    return np.asarray(arrays["target"][np.asarray(ends, dtype=np.int64) - 1], dtype=np.float32)


def ridge_sufficient_statistics(
    ends: np.ndarray,
    arrays: dict[str, np.ndarray],
    metadata: np.ndarray,
    include_metadata: bool,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    n = 0
    feature_sum: np.ndarray | None = None
    feature_sq: np.ndarray | None = None
    target_sum = 0.0
    for current in batches(ends, batch_size):
        x = feature_matrix(current, arrays, metadata, include_metadata).astype(np.float64)
        y = np.asarray(arrays["target"][current], dtype=np.float64)
        if feature_sum is None:
            feature_sum = np.zeros(x.shape[1], dtype=np.float64)
            feature_sq = np.zeros(x.shape[1], dtype=np.float64)
        feature_sum += x.sum(axis=0)
        feature_sq += np.square(x).sum(axis=0)
        target_sum += float(y.sum())
        n += len(current)
    assert feature_sum is not None and feature_sq is not None and n > 0
    mean = feature_sum / n
    variance = np.maximum(feature_sq / n - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    target_mean = target_sum / n
    xtx = np.zeros((len(mean), len(mean)), dtype=np.float64)
    xty = np.zeros(len(mean), dtype=np.float64)
    for current in batches(ends, batch_size):
        x = feature_matrix(current, arrays, metadata, include_metadata).astype(np.float64)
        x = (x - mean) / scale
        y = np.asarray(arrays["target"][current], dtype=np.float64) - target_mean
        xtx += x.T @ x
        xty += x.T @ y
    return mean, scale, target_mean, xtx, xty


def ridge_predict(model: dict[str, Any], ends: np.ndarray, arrays: dict[str, np.ndarray], metadata: np.ndarray, batch_size: int) -> np.ndarray:
    prediction = np.empty(len(ends), dtype=np.float32)
    offset = 0
    for current in batches(ends, batch_size):
        x = feature_matrix(current, arrays, metadata, bool(model["include_metadata"])).astype(np.float64)
        x = (x - model["mean"]) / model["scale"]
        values = model["target_mean"] + x @ model["coefficient"]
        prediction[offset : offset + len(current)] = values.astype(np.float32)
        offset += len(current)
    return prediction


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


def fit_ridge(
    model_id: str,
    train_ends: np.ndarray,
    val_ends: np.ndarray,
    arrays: dict[str, np.ndarray],
    metadata: np.ndarray,
    include_metadata: bool,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logging.info("Fitting %s on %s training endpoints", model_id, f"{len(train_ends):,}")
    batch_size = int(config["fit_batch_rows"])
    mean, scale, target_mean, xtx, xty = ridge_sufficient_statistics(
        train_ends, arrays, metadata, include_metadata, batch_size
    )
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for alpha in [float(value) for value in config["ridge_alpha_grid"]]:
        coefficient = np.linalg.solve(xtx + alpha * np.eye(len(mean), dtype=np.float64), xty)
        model = {
            "model_id": model_id,
            "include_metadata": include_metadata,
            "alpha": alpha,
            "mean": mean,
            "scale": scale,
            "target_mean": target_mean,
            "coefficient": coefficient,
        }
        prediction = ridge_predict(model, val_ends, arrays, metadata, int(config["prediction_batch_rows"]))
        metrics = metric_values(arrays["target"][val_ends], prediction)
        row = {"model_id": model_id, "alpha": alpha, **metrics}
        candidates.append(row)
        if best is None or row["rmse"] < best["validation_rmse"]:
            model["validation_rmse"] = row["rmse"]
            best = model
    assert best is not None
    joblib.dump(best, output_dir / "models" / f"{model_id}.joblib", compress=3)
    return best, candidates


def hgb_training_matrix(
    path: Path,
    ends: np.ndarray,
    arrays: dict[str, np.ndarray],
    metadata: np.ndarray,
    include_metadata: bool,
    batch_size: int,
) -> np.memmap:
    n_features = arrays["features"].shape[1] + (metadata.shape[1] if include_metadata else 0)
    if path.exists():
        path.unlink()
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(ends), n_features))
    offset = 0
    for current in batches(ends, batch_size):
        matrix[offset : offset + len(current)] = feature_matrix(current, arrays, metadata, include_metadata)
        offset += len(current)
    matrix.flush()
    return matrix


def fit_hgb(
    model_id: str,
    train_ends: np.ndarray,
    arrays: dict[str, np.ndarray],
    metadata: np.ndarray,
    include_metadata: bool,
    config: dict[str, Any],
    output_dir: Path,
) -> HistGradientBoostingRegressor:
    logging.info("Fitting %s on %s training endpoints", model_id, f"{len(train_ends):,}")
    temporary_dir = output_dir / "_tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = temporary_dir / f"X_{model_id}.npy"
    x = hgb_training_matrix(
        matrix_path, train_ends, arrays, metadata, include_metadata, int(config["fit_batch_rows"])
    )
    y = np.asarray(arrays["target"][train_ends], dtype=np.float32)
    model = HistGradientBoostingRegressor(**config["hgb_parameters"])
    model.fit(x, y)
    joblib.dump(model, output_dir / "models" / f"{model_id}.joblib", compress=3)
    del x, y
    gc.collect()
    matrix_path.unlink(missing_ok=True)
    return model


def hgb_predict(
    model: HistGradientBoostingRegressor,
    ends: np.ndarray,
    arrays: dict[str, np.ndarray],
    metadata: np.ndarray,
    include_metadata: bool,
    batch_size: int,
) -> np.ndarray:
    prediction = np.empty(len(ends), dtype=np.float32)
    offset = 0
    for current in batches(ends, batch_size):
        x = feature_matrix(current, arrays, metadata, include_metadata)
        prediction[offset : offset + len(current)] = model.predict(x).astype(np.float32)
        offset += len(current)
    return prediction


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
    cohort_index = cohort.set_index("site_id").loc[manifest["site_order"]]
    network_by_site = cohort_index["network"].astype(str).to_numpy(dtype=object)
    observed_scaled = np.asarray(arrays["target"][ends], dtype=np.float32)
    observed = observed_scaled * multiplier
    predicted = np.asarray(predicted_scaled, dtype=np.float32) * multiplier
    direction_codes = np.asarray(arrays["event_direction"][ends], dtype=np.int8)
    return pd.DataFrame(
        {
            "model_id": model_id,
            "site_id": site_order[site_codes],
            "network": network_by_site[site_codes],
            "timestamp": pd.to_datetime(np.asarray(arrays["timestamp_ns"][ends], dtype=np.int64), unit="ns"),
            "split": split,
            "observed_scaled": observed_scaled,
            "predicted_scaled": np.asarray(predicted_scaled, dtype=np.float32),
            "observed_ph": observed,
            "predicted_ph": predicted,
            "residual_observed_minus_predicted": observed - predicted,
            "abs_residual": np.abs(observed - predicted),
            "in_ph_event": direction_codes != 0,
            "ph_event_direction": [CODE_TO_DIRECTION[int(value)] for value in direction_codes],
            "event_id": np.asarray(arrays["event_id"][ends], dtype=np.int64),
        }
    )


def metric_tables(frame: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model_id = str(frame["model_id"].iloc[0])
    split = str(frame["split"].iloc[0])
    overall = {
        "model_id": model_id,
        "split": split,
        "metric_unit": "native pH",
        "n_sites": int(frame["site_id"].nunique()),
        **metric_values(frame["observed_ph"].to_numpy(), frame["predicted_ph"].to_numpy()),
    }
    networks: list[dict[str, Any]] = []
    sites: list[dict[str, Any]] = []
    for network, group in frame.groupby("network", sort=True, observed=True):
        networks.append(
            {
                "model_id": model_id,
                "split": split,
                "network": network,
                "metric_unit": "native pH",
                "n_sites": int(group["site_id"].nunique()),
                **metric_values(group["observed_ph"].to_numpy(), group["predicted_ph"].to_numpy()),
            }
        )
    for site_id, group in frame.groupby("site_id", sort=True, observed=True):
        sites.append(
            {
                "model_id": model_id,
                "split": split,
                "site_id": site_id,
                "network": group["network"].iloc[0],
                "n_event_rows": int(group["in_ph_event"].sum()),
                **metric_values(group["observed_ph"].to_numpy(), group["predicted_ph"].to_numpy()),
            }
        )
    return overall, networks, sites


def tail_masks(residual: np.ndarray, fraction: float) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    absolute = np.abs(residual)
    thresholds = {
        "absolute": float(np.quantile(absolute, 1 - fraction)),
        "negative": float(np.quantile(residual, fraction)),
        "positive": float(np.quantile(residual, 1 - fraction)),
    }
    return thresholds, {
        "absolute": absolute >= thresholds["absolute"],
        "negative": residual <= thresholds["negative"],
        "positive": residual >= thresholds["positive"],
    }


def tail_analysis(frame: pd.DataFrame, fractions: Iterable[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_id = str(frame["model_id"].iloc[0])
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", frame)]
    scopes.extend(("network", str(network), group) for network, group in frame.groupby("network", sort=True, observed=True))
    tail_rows: list[dict[str, Any]] = []
    hit_rows: list[dict[str, Any]] = []
    for scope_type, scope_value, group in scopes:
        residual = group["residual_observed_minus_predicted"].to_numpy(dtype=np.float64)
        event = group["in_ph_event"].astype(bool).to_numpy()
        direction = group["ph_event_direction"].astype(str).to_numpy()
        base_event_rate = float(event.mean()) if len(event) else np.nan
        valid_event = group["in_ph_event"].astype(bool) & group["event_id"].ge(0)
        event_keys = group.loc[valid_event, ["site_id", "event_id", "ph_event_direction"]].drop_duplicates()
        for fraction in fractions:
            thresholds, masks = tail_masks(residual, float(fraction))
            for tail_type, mask in masks.items():
                selected_event_rate = float(event[mask].mean()) if mask.any() else np.nan
                tail_rows.append(
                    {
                        "model_id": model_id,
                        "scope_type": scope_type,
                        "scope_value": scope_value,
                        "tail_fraction_requested": float(fraction),
                        "tail_type": tail_type,
                        "threshold_native_ph": thresholds[tail_type],
                        "n_test_rows": len(group),
                        "n_selected_rows": int(mask.sum()),
                        "base_event_rate": base_event_rate,
                        "selected_event_rate": selected_event_rate,
                        "event_enrichment": selected_event_rate / base_event_rate if base_event_rate > 0 else np.nan,
                        "selected_low_rows": int(((direction == "low") & mask).sum()),
                        "selected_high_rows": int(((direction == "high") & mask).sum()),
                        "selected_mixed_rows": int(((direction == "mixed") & mask).sum()),
                    }
                )
                selected = group.loc[mask & valid_event.to_numpy(), ["site_id", "event_id", "ph_event_direction"]].drop_duplicates()
                total_events = len(event_keys)
                hit_rows.append(
                    {
                        "model_id": model_id,
                        "scope_type": scope_type,
                        "scope_value": scope_value,
                        "tail_fraction_requested": float(fraction),
                        "tail_type": tail_type,
                        "n_event_windows": total_events,
                        "n_hit_event_windows": len(selected),
                        "event_window_hit_rate": len(selected) / total_events if total_events else np.nan,
                        "n_low_hits": int((selected["ph_event_direction"] == "low").sum()),
                        "n_high_hits": int((selected["ph_event_direction"] == "high").sum()),
                        "n_mixed_hits": int((selected["ph_event_direction"] == "mixed").sum()),
                    }
                )
    return tail_rows, hit_rows


def validate_prediction_frame(frame: pd.DataFrame, expected_rows: int) -> list[str]:
    problems: list[str] = []
    if len(frame) != expected_rows:
        problems.append(f"row count {len(frame)} != {expected_rows}")
    if frame.isna().any().any():
        problems.append("prediction frame contains missing values")
    if frame.duplicated(["site_id", "timestamp"]).any():
        problems.append("prediction frame contains duplicate site/timestamp keys")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(numeric).all():
        problems.append("prediction frame contains non-finite values")
    if not np.allclose(frame["observed_scaled"].to_numpy(dtype=float) * 14.0, frame["observed_ph"], atol=1e-6):
        problems.append("scaled/native observed pH mismatch")
    residual = frame["observed_ph"].to_numpy() - frame["predicted_ph"].to_numpy()
    if not np.allclose(residual, frame["residual_observed_minus_predicted"], atol=1e-7):
        problems.append("residual identity mismatch")
    return problems


def reference_sequence_frame(
    sequence_dir: Path,
    model_id: str,
    split: str,
    expected: pd.DataFrame,
) -> pd.DataFrame:
    path = sequence_dir / "predictions" / f"prediction_{split}_{model_id}.parquet"
    frame = pd.read_parquet(path)
    if len(frame) != len(expected):
        raise ValueError(f"{model_id}/{split} row count does not match baseline endpoint set")
    if not frame[["site_id", "timestamp"]].reset_index(drop=True).equals(
        expected[["site_id", "timestamp"]].reset_index(drop=True)
    ):
        raise ValueError(f"{model_id}/{split} keys are not exactly aligned")
    if not np.allclose(frame["observed_ph"], expected["observed_ph"], atol=1e-6):
        raise ValueError(f"{model_id}/{split} observed targets are not aligned")
    return frame


def evaluate_frame(
    frame: pd.DataFrame,
    fractions: list[float],
    overall_rows: list[dict[str, Any]],
    network_rows: list[dict[str, Any]],
    site_rows: list[dict[str, Any]],
    tail_rows: list[dict[str, Any]],
    hit_rows: list[dict[str, Any]],
) -> None:
    overall, network, site = metric_tables(frame)
    overall_rows.append(overall)
    network_rows.extend(network)
    site_rows.extend(site)
    if str(frame["split"].iloc[0]) == "test":
        tails, hits = tail_analysis(frame, fractions)
        tail_rows.extend(tails)
        hit_rows.extend(hits)


def model_manifest(dynamic_names: list[str], metadata_names: list[str], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "models": {
            "B0_site_mean": {"inputs": ["site_id aggregate from sampled training target"], "historical_ph": False},
            "B1_seasonal_mean": {"inputs": ["site_id", "month", "hour", "sampled training target aggregates"], "historical_ph": False},
            "B2_persistence_1h": {"inputs": ["previous-hour 00400"], "historical_ph": True},
            "B3_R0_ridge": {"inputs": dynamic_names, "historical_ph": False},
            "B3_R1_ridge_metadata": {"inputs": [*dynamic_names, *metadata_names], "historical_ph": False},
            "B4_H0_hgb": {"inputs": dynamic_names, "historical_ph": False},
            "B4_H1_hgb_metadata": {"inputs": [*dynamic_names, *metadata_names], "historical_ph": False},
        },
        "nonevent_variants": NONEVENT_MODEL_IDS,
        "forbidden_predictors": config["forbidden_predictors"],
        "interpretation": "B2 is a forecasting-only reference; all other model features exclude current/historical pH and event fields.",
    }


def validate_feature_manifest(payload: dict[str, Any], config: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    forbidden_exact = {"00400", "00400_native", "in_ph_event", "ph_event_label", "ph_event_direction", "event_id"}
    forbidden_markers = ["residual", "prediction", "future"]
    for model_id, spec in payload["models"].items():
        if model_id == "B2_persistence_1h":
            continue
        inputs = [str(value) for value in spec["inputs"]]
        if forbidden_exact.intersection(inputs):
            problems.append(f"{model_id} contains a forbidden exact predictor")
        for value in inputs:
            if any(marker in value.lower() for marker in forbidden_markers):
                problems.append(f"{model_id} contains forbidden-like predictor {value}")
    return problems


def paired_comparisons(overall: pd.DataFrame) -> pd.DataFrame:
    test = overall[overall["split"].eq("test")].set_index("model_id")
    pairs = [
        ("LSTM_S1", "B0_site_mean", "LSTM versus site mean"),
        ("LSTM_S1", "B1_seasonal_mean", "LSTM versus seasonal mean"),
        ("LSTM_S1", "B3_R1_ridge_metadata", "LSTM versus metadata Ridge"),
        ("LSTM_S1", "B4_H1_hgb_metadata", "LSTM versus metadata HGB"),
        ("GRU_S0", "B3_R0_ridge", "GRU versus Ridge without metadata"),
        ("GRU_S0", "B4_H0_hgb", "GRU versus HGB without metadata"),
        ("B3_R1_ridge_metadata", "B3_R0_ridge", "Ridge metadata effect"),
        ("B4_H1_hgb_metadata", "B4_H0_hgb", "HGB metadata effect"),
        ("B3_R0_ridge_nonevent", "B3_R0_ridge", "Ridge non-event training effect"),
        ("B3_R1_ridge_metadata_nonevent", "B3_R1_ridge_metadata", "Metadata Ridge non-event effect"),
        ("B4_H0_hgb_nonevent", "B4_H0_hgb", "HGB non-event training effect"),
        ("B4_H1_hgb_metadata_nonevent", "B4_H1_hgb_metadata", "Metadata HGB non-event effect"),
    ]
    rows: list[dict[str, Any]] = []
    for candidate, reference, label in pairs:
        if candidate not in test.index or reference not in test.index:
            continue
        candidate_row = test.loc[candidate]
        reference_row = test.loc[reference]
        rows.append(
            {
                "comparison": label,
                "candidate_model": candidate,
                "reference_model": reference,
                "candidate_rmse": candidate_row["rmse"],
                "reference_rmse": reference_row["rmse"],
                "rmse_change_candidate_minus_reference": candidate_row["rmse"] - reference_row["rmse"],
                "candidate_r2": candidate_row["r2"],
                "reference_r2": reference_row["r2"],
                "r2_change_candidate_minus_reference": candidate_row["r2"] - reference_row["r2"],
            }
        )
    return pd.DataFrame(rows)


def macro_site_metrics(site_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_id, split), group in site_metrics.groupby(["model_id", "split"], sort=False):
        rows.append(
            {
                "model_id": model_id,
                "split": split,
                "n_sites": len(group),
                "median_site_rmse": group["rmse"].median(),
                "median_site_mae": group["mae"].median(),
                "median_site_r2": group["r2"].median(),
                "positive_site_r2_count": int((group["r2"] > 0).sum()),
                "positive_site_r2_fraction": float((group["r2"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def site_baseline_improvement(site_metrics: pd.DataFrame) -> pd.DataFrame:
    test = site_metrics[site_metrics["split"].eq("test")]
    reference = test[test["model_id"].eq("B0_site_mean")][["site_id", "rmse", "r2"]].rename(
        columns={"rmse": "site_mean_rmse", "r2": "site_mean_r2"}
    )
    rows: list[dict[str, Any]] = []
    for model_id, group in test.groupby("model_id", sort=False):
        merged = group.merge(reference, on="site_id", how="inner")
        rows.append(
            {
                "model_id": model_id,
                "n_common_sites": len(merged),
                "sites_better_rmse_than_site_mean": int((merged["rmse"] < merged["site_mean_rmse"]).sum()),
                "fraction_better_rmse_than_site_mean": float((merged["rmse"] < merged["site_mean_rmse"]).mean()),
                "median_rmse_change_vs_site_mean": float((merged["rmse"] - merged["site_mean_rmse"]).median()),
                "median_r2_change_vs_site_mean": float((merged["r2"] - merged["site_mean_r2"]).median()),
            }
        )
    return pd.DataFrame(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hash_manifest(paths: Iterable[Path], category: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.exists() and path.is_file():
            rows.append(
                {
                    "category": category,
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def report_text(mode: str, manifest: dict[str, Any], overall: pd.DataFrame, macro: pd.DataFrame, comparisons: pd.DataFrame, problems: list[str]) -> str:
    test = overall[overall["split"].eq("test")].sort_values("rmse")
    lines = [
        f"# Common1000 fair baseline {mode} report",
        "",
        "## Dataset",
        "",
        f"- Sites: {manifest['n_sites']:,}",
        f"- Sampled training windows: {manifest['training_sample_windows']:,}",
        f"- Validation/test windows: {manifest['split_window_counts']['val']:,} / {manifest['split_window_counts']['test']:,}",
        "- GRU/LSTM predictions are reused and key-aligned; they are not retrained.",
        "",
        "## Test ranking (native pH)",
        "",
        "| Model | RMSE | MAE | R2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in test.itertuples(index=False):
        lines.append(f"| {row.model_id} | {row.rmse:.6f} | {row.mae:.6f} | {row.r2:.6f} |")
    lines.extend(["", "## Interpretation", ""])
    if not comparisons.empty:
        for row in comparisons.head(8).itertuples(index=False):
            lines.append(
                f"- {row.comparison}: RMSE change {row.rmse_change_candidate_minus_reference:+.6f}; "
                f"R2 change {row.r2_change_candidate_minus_reference:+.6f}."
            )
    lstm_macro = macro[(macro["model_id"].eq("LSTM_S1")) & (macro["split"].eq("test"))]
    if not lstm_macro.empty:
        row = lstm_macro.iloc[0]
        lines.append(
            f"- LSTM_S1 positive per-site R2: {int(row['positive_site_r2_count'])}/{int(row['n_sites'])}; "
            f"median site R2 {row['median_site_r2']:.6f}."
        )
    lines.extend(
        [
            "- B2 persistence uses previous-hour pH and is a forecasting-only reference.",
            "- Event labels are evaluation-only. Non-event variants remove event target rows only from training.",
            "- All strict comparisons use identical validation/test endpoint keys.",
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
    sequence_dir = resolve_path(config["sequence_root"]) / args.mode
    output_dir = resolve_path(config["output_root"]) / args.mode
    success_path = output_dir / "_SUCCESS"
    if args.resume and success_path.exists():
        print(f"Validated baseline output already exists: {success_path}")
        return
    if output_dir.exists():
        expected_parent = resolve_path(config["output_root"]).resolve()
        if output_dir.resolve().parent != expected_parent:
            raise RuntimeError(f"Refusing to clear unexpected output path: {output_dir}")
        shutil.rmtree(output_dir)
    for directory in ("models", "predictions", "reports", "_tmp"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    started = time.time()
    logging.info("Starting Common1000 fair baseline mode=%s", args.mode)

    manifest, arrays, cohort, metadata, metadata_names = load_inputs(sequence_dir, config, args.mode)
    problems = validate_windows(arrays)
    feature_manifest = model_manifest(config["dynamic_features"], metadata_names, config)
    problems.extend(validate_feature_manifest(feature_manifest, config))
    write_json(output_dir / "reports" / "model_input_manifest.json", feature_manifest)
    if problems:
        raise ValueError(f"Input validation failed: {problems}")

    multiplier = float(config["target_inverse_multiplier"])
    batch_size = int(config["prediction_batch_rows"])
    train_ends = np.asarray(arrays["train_sample_ends"], dtype=np.int64)
    val_ends = np.asarray(arrays["val_ends"], dtype=np.int64)
    test_ends = np.asarray(arrays["test_ends"], dtype=np.int64)
    event_train_mask = np.asarray(arrays["event_direction"][train_ends], dtype=np.int8) != 0
    nonevent_train_ends = train_ends[~event_train_mask]
    training_audit = pd.DataFrame(
        [
            {
                "mode": args.mode,
                "all_sampled_training_windows": len(train_ends),
                "event_training_windows_removed": int(event_train_mask.sum()),
                "nonevent_training_windows": len(nonevent_train_ends),
                "validation_windows_unchanged": len(val_ends),
                "test_windows_unchanged": len(test_ends),
            }
        ]
    )
    write_csv(output_dir / "reports" / "nonevent_training_audit.csv", training_audit)

    logging.info("Building site and seasonal training-only aggregates")
    statistics = fit_simple_statistics(arrays, manifest, config["dynamic_features"])
    joblib.dump(statistics, output_dir / "models" / "B0_B1_training_aggregates.joblib", compress=3)
    for split, ends in (("val", val_ends), ("test", test_ends)):
        persistence_problems = validate_persistence(ends, arrays)
        if persistence_problems:
            raise ValueError(f"{split} persistence validation failed: {persistence_problems}")

    models: dict[str, dict[str, Any]] = {
        "B0_site_mean": {"kind": "site_mean", "object": statistics, "include_metadata": False},
        "B1_seasonal_mean": {"kind": "seasonal", "object": statistics, "include_metadata": False},
        "B2_persistence_1h": {"kind": "persistence", "object": None, "include_metadata": False},
    }
    alpha_rows: list[dict[str, Any]] = []
    ridge_specs = [
        ("B3_R0_ridge", train_ends, False),
        ("B3_R1_ridge_metadata", train_ends, True),
        ("B3_R0_ridge_nonevent", nonevent_train_ends, False),
        ("B3_R1_ridge_metadata_nonevent", nonevent_train_ends, True),
    ]
    for model_id, fit_ends, include_metadata in ridge_specs:
        model, rows = fit_ridge(
            model_id,
            fit_ends,
            val_ends,
            arrays,
            metadata,
            include_metadata,
            config,
            output_dir,
        )
        models[model_id] = {"kind": "ridge", "object": model, "include_metadata": include_metadata}
        alpha_rows.extend(rows)
    write_csv(output_dir / "reports" / "ridge_alpha_validation.csv", pd.DataFrame(alpha_rows))

    hgb_specs = [
        ("B4_H0_hgb", train_ends, False),
        ("B4_H1_hgb_metadata", train_ends, True),
        ("B4_H0_hgb_nonevent", nonevent_train_ends, False),
        ("B4_H1_hgb_metadata_nonevent", nonevent_train_ends, True),
    ]
    for model_id, fit_ends, include_metadata in hgb_specs:
        model = fit_hgb(model_id, fit_ends, arrays, metadata, include_metadata, config, output_dir)
        models[model_id] = {"kind": "hgb", "object": model, "include_metadata": include_metadata}

    overall_rows: list[dict[str, Any]] = []
    network_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    hit_rows: list[dict[str, Any]] = []
    fractions = [float(value) for value in config["residual_tail_fractions"]]
    expected_frames: dict[str, pd.DataFrame] = {}

    def predict_spec(spec: dict[str, Any], ends: np.ndarray) -> np.ndarray:
        if spec["kind"] == "site_mean":
            return predict_site_mean(ends, arrays, spec["object"])
        if spec["kind"] == "seasonal":
            return predict_seasonal(ends, arrays, spec["object"], config["dynamic_features"])
        if spec["kind"] == "persistence":
            return predict_persistence(ends, arrays)
        if spec["kind"] == "ridge":
            return ridge_predict(spec["object"], ends, arrays, metadata, batch_size)
        if spec["kind"] == "hgb":
            return hgb_predict(
                spec["object"], ends, arrays, metadata, bool(spec["include_metadata"]), batch_size
            )
        raise ValueError(f"Unknown model kind: {spec['kind']}")

    for model_id in [*MAIN_MODEL_IDS, *NONEVENT_MODEL_IDS]:
        spec = models[model_id]
        for split, ends in (("val", val_ends), ("test", test_ends)):
            logging.info("Predicting %s split=%s rows=%s", model_id, split, f"{len(ends):,}")
            predicted = predict_spec(spec, ends)
            frame = prediction_frame(model_id, split, ends, predicted, arrays, manifest, cohort, multiplier)
            frame_problems = validate_prediction_frame(frame, len(ends))
            if frame_problems:
                raise ValueError(f"{model_id}/{split} validation failed: {frame_problems}")
            write_parquet(output_dir / "predictions" / f"prediction_{split}_{model_id}.parquet", frame)
            evaluate_frame(frame, fractions, overall_rows, network_rows, site_rows, tail_rows, hit_rows)
            if model_id == "B0_site_mean":
                expected_frames[split] = frame[["site_id", "timestamp", "observed_ph"]].copy()
            del predicted, frame
            gc.collect()

    logging.info("Aligning and independently re-evaluating existing GRU/LSTM predictions")
    for model_id in SEQUENCE_MODEL_IDS:
        for split in ("val", "test"):
            frame = reference_sequence_frame(sequence_dir, model_id, split, expected_frames[split])
            frame_problems = validate_prediction_frame(frame, len(expected_frames[split]))
            if frame_problems:
                raise ValueError(f"{model_id}/{split} reference validation failed: {frame_problems}")
            evaluate_frame(frame, fractions, overall_rows, network_rows, site_rows, tail_rows, hit_rows)
            del frame
            gc.collect()

    overall = pd.DataFrame(overall_rows)
    networks = pd.DataFrame(network_rows)
    sites = pd.DataFrame(site_rows)
    tails = pd.DataFrame(tail_rows)
    hits = pd.DataFrame(hit_rows)
    catalog = {
        **{model_id: ("baseline", "all_training") for model_id in MAIN_MODEL_IDS},
        **{model_id: ("baseline", "nonevent_training") for model_id in NONEVENT_MODEL_IDS},
        **{model_id: ("sequence_reference", "existing_model") for model_id in SEQUENCE_MODEL_IDS},
    }
    for frame in (overall, networks, sites, tails, hits):
        frame.insert(1, "model_family", frame["model_id"].map(lambda value: catalog[str(value)][0]))
        frame.insert(2, "training_scope", frame["model_id"].map(lambda value: catalog[str(value)][1]))
    macro = macro_site_metrics(sites)
    improvements = site_baseline_improvement(sites)
    comparisons = paired_comparisons(overall)
    write_csv(output_dir / "reports" / "model_comparison_metrics.csv", overall)
    write_csv(output_dir / "reports" / "network_metrics.csv", networks)
    write_csv(output_dir / "reports" / "site_level_metrics.csv", sites)
    write_csv(output_dir / "reports" / "macro_site_metrics.csv", macro)
    write_csv(output_dir / "reports" / "site_mean_improvement.csv", improvements)
    write_csv(output_dir / "reports" / "paired_model_comparisons.csv", comparisons)
    write_csv(output_dir / "reports" / "residual_tail_analysis.csv", tails)
    write_csv(output_dir / "reports" / "event_window_hit_rates.csv", hits)
    write_csv(
        output_dir / "reports" / "nonevent_model_metrics.csv",
        overall[overall["model_id"].isin(NONEVENT_MODEL_IDS)],
    )

    site_status_path = sequence_dir / "reports" / "hourly_window_site_summary.csv"
    site_status = pd.read_csv(site_status_path, dtype={"site_id": "string"})
    site_status["site_mean_global_fallback"] = site_status["train_windows"].eq(0)
    site_status["baseline_status"] = np.where(site_status["test_windows"].gt(0), "success", "no_test_window")
    site_status["baseline_status_reason"] = np.where(
        site_status["test_windows"].gt(0), "", "no valid same-split continuous 24-hour test window"
    )
    site_status.loc[site_status["train_windows"].eq(0), "baseline_status_reason"] = (
        "no valid train or test window; site/global aggregate fallback is defined but never evaluated"
    )
    write_csv(output_dir / "reports" / "site_status.csv", site_status)
    if len(site_status) != int(manifest["n_sites"]):
        problems.append("site status does not cover the complete cohort")
    if args.mode == "full" and int(site_status["test_windows"].eq(0).sum()) != 6:
        problems.append("full run does not contain the expected six no-test-window sites")

    # Final report-level consistency checks.
    expected_model_count = len(MAIN_MODEL_IDS) + len(NONEVENT_MODEL_IDS) + len(SEQUENCE_MODEL_IDS)
    for split, expected_rows in (("val", len(val_ends)), ("test", len(test_ends))):
        subset = overall[overall["split"].eq(split)]
        if len(subset) != expected_model_count or not subset["n_rows"].eq(expected_rows).all():
            problems.append(f"{split} metric row coverage mismatch")
    if not np.isfinite(overall[["rmse", "mae", "r2"]].to_numpy(dtype=float)).all():
        problems.append("overall metrics contain non-finite values")
    if len(nonevent_train_ends) + int(event_train_mask.sum()) != len(train_ends):
        problems.append("non-event training row reconciliation failed")

    report = report_text(args.mode, manifest, overall, macro, comparisons, problems)
    (output_dir / "reports" / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    elapsed = time.time() - started
    run_summary = {
        "status": "success" if not problems else "failed_validation",
        "mode": args.mode,
        "sites": int(manifest["n_sites"]),
        "sampled_training_windows": len(train_ends),
        "nonevent_training_windows": len(nonevent_train_ends),
        "validation_windows": len(val_ends),
        "test_windows": len(test_ends),
        "baseline_models": [*MAIN_MODEL_IDS, *NONEVENT_MODEL_IDS],
        "sequence_reference_models": SEQUENCE_MODEL_IDS,
        "validation_problems": problems,
        "elapsed_seconds": elapsed,
    }
    write_json(output_dir / "reports" / "run_summary.json", run_summary)

    input_hash_paths = [
        config_path,
        SCRIPT_PATH,
        sequence_dir / "sequence_data" / "arrays" / "array_manifest.json",
        sequence_dir / "sequence_data" / "metadata_matrix.npy",
        sequence_dir / "sequence_data" / "metadata_transform.json",
        sequence_dir / "reports" / "feature_manifest.json",
        sequence_dir / "predictions" / "prediction_test_LSTM_S1.parquet",
        *array_paths(sequence_dir).values(),
    ]
    output_hash_paths = [
        output_dir / "reports" / "model_comparison_metrics.csv",
        output_dir / "reports" / "network_metrics.csv",
        output_dir / "reports" / "site_level_metrics.csv",
        output_dir / "reports" / "residual_tail_analysis.csv",
        output_dir / "reports" / "event_window_hit_rates.csv",
        output_dir / "reports" / "FINAL_REPORT.md",
    ]
    hashes = pd.DataFrame(
        [*hash_manifest(input_hash_paths, "input"), *hash_manifest(output_hash_paths, "output")]
    )
    write_csv(output_dir / "reports" / "artifact_hashes.csv", hashes)
    shutil.rmtree(output_dir / "_tmp", ignore_errors=True)
    if problems:
        raise RuntimeError(f"Final validation failed: {problems}")
    write_json(
        success_path,
        {
            "status": "success",
            "mode": args.mode,
            "completed_at_unix": time.time(),
            "elapsed_seconds": elapsed,
            "report": str((output_dir / "reports" / "FINAL_REPORT.md").resolve()),
        },
    )
    logging.info("Completed Common1000 baseline mode=%s in %.1f seconds", args.mode, elapsed)


if __name__ == "__main__":
    main()
