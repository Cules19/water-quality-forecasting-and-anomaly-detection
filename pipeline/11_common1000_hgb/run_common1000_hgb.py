#!/usr/bin/env python
"""Leak-safe HGB baselines for the Common1000 processed pH cohort.

M0 uses four scaled sensor variables and seven calendar/cyclic variables. M1
adds only static registry/manifest metadata. Training rows are deterministically
time-stratified and capped per site; validation and test rows are all evaluated.
Predictions and metrics are reported in native pH units.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
MODEL_IDS = ("M0", "M1")
SPLITS = ("val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "common1000_hgb_v1.json",
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "processed_root",
        "cohort_manifest",
        "registry_path",
        "output_root",
        "base_features",
        "target_scaled",
        "target_native",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Configuration is missing keys: {missing}")
    return config


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary, compression="snappy")
    os.replace(temporary, path)


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def dominant_land_cover(catchment: dict[str, Any]) -> tuple[str, float]:
    values = catchment.get("watershed_land_cover")
    if not isinstance(values, dict):
        return "unknown", np.nan
    candidates = [(str(key), safe_float(value)) for key, value in values.items()]
    candidates = [(key, value) for key, value in candidates if np.isfinite(value)]
    if not candidates:
        return "unknown", np.nan
    return max(candidates, key=lambda item: item[1])


def is_true(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})


def largest_remainder_quotas(counts: pd.Series, total: int) -> dict[str, int]:
    raw = counts.astype(float) / float(counts.sum()) * total
    quotas = np.floor(raw).astype(int)
    for key in counts.index:
        if counts[key] > 0 and quotas[key] == 0:
            quotas[key] = 1
    while int(quotas.sum()) > total:
        removable = [key for key in counts.index if quotas[key] > 1]
        key = min(removable, key=lambda item: raw[item] - quotas[item])
        quotas[key] -= 1
    while int(quotas.sum()) < total:
        available = [key for key in counts.index if quotas[key] < counts[key]]
        key = max(available, key=lambda item: raw[item] - quotas[item])
        quotas[key] += 1
    return {str(key): int(value) for key, value in quotas.items()}


def select_cohort(manifest_path: Path, mode: str, config: dict[str, Any]) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype={"site_id": "string", "canonical_physical_site_id": "string"})
    required = {
        "site_id",
        "status",
        "quality_tier",
        "preferred_record",
        "canonical_physical_site_id",
        "network",
        "candidate_rank",
        "train_rows",
        "validation_rows",
        "test_rows",
        "model_output",
        "do_unit_family",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Cohort manifest missing columns: {missing}")
    cohort = manifest[
        manifest["status"].eq("success")
        & manifest["quality_tier"].eq("primary_strict")
        & is_true(manifest["preferred_record"])
    ].copy()
    cohort["site_id"] = cohort["site_id"].astype("string")
    cohort = cohort.sort_values(["candidate_rank", "site_id"], kind="mergesort").reset_index(drop=True)
    if cohort["canonical_physical_site_id"].duplicated().any():
        duplicate = cohort.loc[cohort["canonical_physical_site_id"].duplicated(False), "canonical_physical_site_id"].tolist()
        raise ValueError(f"Preferred cohort still has physical-site duplicates: {duplicate[:10]}")
    expected = int(config["expected_full_sites"])
    if len(cohort) != expected:
        raise ValueError(f"Expected {expected} strict preferred sites, found {len(cohort)}")
    if mode == "full":
        cohort["selection_reason"] = "all strict preferred unique physical sites"
        return cohort

    target = int(config["smoke_site_count"])
    counts = cohort.groupby("network", observed=True).size()
    quotas = largest_remainder_quotas(counts, target)
    pieces: list[pd.DataFrame] = []
    for network, group in cohort.groupby("network", sort=True, observed=True):
        group = group.sort_values(["candidate_rank", "site_id"], kind="mergesort").reset_index(drop=True)
        quota = quotas[str(network)]
        positions = np.linspace(0, len(group) - 1, quota, dtype=np.int64)
        selected = group.iloc[np.unique(positions)].copy()
        selected["selection_reason"] = f"network-stratified smoke sample ({quota} from {len(group)})"
        pieces.append(selected)
    smoke = pd.concat(pieces, ignore_index=True).sort_values(["candidate_rank", "site_id"], kind="mergesort")
    if len(smoke) != target:
        raise RuntimeError(f"Smoke selector expected {target} sites, selected {len(smoke)}")
    return smoke.reset_index(drop=True)


def feature_leakage_reason(feature: str, forbidden: Iterable[str]) -> str | None:
    lower = feature.lower()
    tokens = set(re.split(r"[^a-z0-9]+", lower))
    for marker in forbidden:
        marker_lower = str(marker).lower()
        if marker_lower == "ph":
            if "ph" in tokens or lower.startswith("ph_") or lower.endswith("_ph"):
                return "contains pH marker"
            continue
        if marker_lower in tokens or marker_lower in lower:
            return f"contains forbidden marker {marker_lower}"
    return None


def assert_safe_features(features: list[str], config: dict[str, Any]) -> None:
    reasons = {feature: feature_leakage_reason(feature, config["forbidden_feature_markers"]) for feature in features}
    failed = {feature: reason for feature, reason in reasons.items() if reason}
    if failed:
        raise ValueError(f"Feature leakage screen failed: {failed}")
    forbidden_exact = {
        config["target_scaled"],
        config["target_native"],
        "in_ph_event",
        "ph_event_label",
        "ph_event_direction",
        "event_id",
    }
    overlap = sorted(set(features) & forbidden_exact)
    if overlap:
        raise ValueError(f"Forbidden columns in predictor list: {overlap}")


def registry_metadata(cohort: pd.DataFrame, registry_path: Path) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in cohort.itertuples(index=False):
        site_id = str(row.site_id)
        entry = registry.get(site_id, {})
        if not isinstance(entry, dict):
            entry = {}
        source_description = parse_jsonish(entry.get("source_description"))
        catchment = entry.get("catchment_analysis") if isinstance(entry.get("catchment_analysis"), dict) else {}
        land_cover, land_cover_pct = dominant_land_cover(catchment)
        drainage = entry.get("drainage_area_km2", source_description.get("drainage_area_sqkm"))
        rows.append(
            {
                "site_id": site_id,
                "metadata_match": site_id in registry,
                "meta_latitude": safe_float(entry.get("latitude")),
                "meta_longitude": safe_float(entry.get("longitude")),
                "meta_elevation": safe_float(entry.get("elevation")),
                "meta_drainage_area_km2": safe_float(drainage),
                "meta_average_precip_mm_yr": safe_float(catchment.get("average_precip_mm_yr")),
                "meta_average_evapo_trans_mm_yr": safe_float(catchment.get("average_evapo_trans_mm_yr")),
                "meta_population_est_2020": safe_float(catchment.get("population_est_2020")),
                "meta_watershed_dams_no": safe_float(catchment.get("watershed_dams_no")),
                "meta_watershed_dams_cap_m3": safe_float(catchment.get("watershed_dams_cap_m3")),
                "meta_irrigation_area_km2": safe_float(catchment.get("irrigation_area_km2")),
                "meta_irrigation_area_pct": safe_float(catchment.get("irrigation_area_pct")),
                "meta_dominant_land_cover_pct": land_cover_pct,
                "meta_registry_source": str(entry.get("source") or "unknown"),
                "meta_network": str(row.network or "unknown"),
                "meta_site_type": str(source_description.get("site_type_cd") or entry.get("site_type") or "unknown"),
                "meta_dominant_land_cover": land_cover,
                "meta_do_unit_family": str(row.do_unit_family or "unknown"),
            }
        )
    raw = pd.DataFrame(rows)
    numeric_raw = [column for column in raw.columns if column.startswith("meta_") and pd.api.types.is_numeric_dtype(raw[column])]
    categorical_raw = [
        "meta_registry_source",
        "meta_network",
        "meta_site_type",
        "meta_dominant_land_cover",
        "meta_do_unit_family",
    ]
    model = raw[["site_id", "metadata_match"]].copy()
    audit_rows: list[dict[str, Any]] = []
    feature_names: list[str] = []
    category_names: list[str] = []
    category_maps: dict[str, dict[str, int]] = {}
    numeric_imputation: dict[str, float] = {}

    for column in numeric_raw:
        values = pd.to_numeric(raw[column], errors="coerce")
        missing = values.isna()
        median = float(values.median()) if values.notna().any() else 0.0
        model[column] = values.fillna(median).astype("float32")
        missing_column = f"{column}_missing"
        model[missing_column] = missing.astype("int8")
        feature_names.extend([column, missing_column])
        numeric_imputation[column] = median
        audit_rows.append(
            {
                "field": column,
                "type": "numeric",
                "missing_count": int(missing.sum()),
                "missing_rate": float(missing.mean()),
                "imputation": median,
                "model_features": f"{column}|{missing_column}",
            }
        )

    for column in categorical_raw:
        values = raw[column].astype("string").fillna("unknown").replace("", "unknown")
        categories = sorted(values.astype(str).unique().tolist())
        mapping = {value: index for index, value in enumerate(categories)}
        code_column = f"{column}_code"
        model[code_column] = values.astype(str).map(mapping).astype("int16")
        feature_names.append(code_column)
        category_names.append(code_column)
        category_maps[column] = mapping
        audit_rows.append(
            {
                "field": column,
                "type": "categorical",
                "missing_count": int(values.eq("unknown").sum()),
                "missing_rate": float(values.eq("unknown").mean()),
                "imputation": "unknown category",
                "model_features": code_column,
                "n_categories": len(categories),
            }
        )

    if model[feature_names].isna().any().any():
        raise ValueError("Model-ready static metadata contains missing values")
    metadata_spec = {
        "numeric_imputation_medians": numeric_imputation,
        "categorical_maps": category_maps,
        "metadata_features": feature_names,
        "categorical_features": category_names,
        "leakage_controls": [
            "no target/pH fields",
            "no event fields",
            "no residual or model-performance fields",
            "only static registry fields, network and dissolved-oxygen unit family",
            "registry discharge time-series summaries are excluded because they may include validation/test periods",
        ],
    }
    return model, feature_names, category_names, {"audit": audit_rows, "spec": metadata_spec, "raw": raw}


def model_file_path(processed_root: Path, cohort_row: pd.Series) -> Path:
    configured = Path(str(cohort_row["model_output"]))
    if configured.is_absolute():
        return configured
    candidate = PROJECT_ROOT / configured
    if candidate.exists():
        return candidate
    return processed_root / "model_ready_by_site" / f"{cohort_row['site_id']}.parquet"


def deterministic_time_positions(n_rows: int, cap: int) -> np.ndarray:
    if n_rows <= 0 or cap <= 0:
        return np.empty(0, dtype=np.int64)
    if n_rows <= cap:
        return np.arange(n_rows, dtype=np.int64)
    positions = np.linspace(0, n_rows - 1, cap, dtype=np.int64)
    if len(np.unique(positions)) != cap:
        raise RuntimeError("Deterministic sampler produced duplicate positions")
    return positions


def read_split_table(path: Path, split: str, columns: list[str]) -> pd.DataFrame:
    table = pq.read_table(path, columns=list(dict.fromkeys(columns)), filters=[("split", "=", split)])
    return table.to_pandas()


def build_training_memmaps(
    cohort: pd.DataFrame,
    processed_root: Path,
    base_features: list[str],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[np.memmap, np.memmap, pd.DataFrame, Path, Path]:
    cap = int(config["train_rows_per_site_cap"])
    expected_rows = int(np.minimum(pd.to_numeric(cohort["train_rows"]).to_numpy(dtype=np.int64), cap).sum())
    temporary_dir = output_dir / "_tmp_memmap"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    x_path = temporary_dir / "X_train_base_float32.dat"
    y_path = temporary_dir / "y_train_scaled_float32.dat"
    for path in (x_path, y_path):
        if path.exists():
            path.unlink()
    x = np.memmap(x_path, mode="w+", dtype=np.float32, shape=(expected_rows, len(base_features)))
    y = np.memmap(y_path, mode="w+", dtype=np.float32, shape=(expected_rows,))
    audit_rows: list[dict[str, Any]] = []
    offset = 0
    target = config["target_scaled"]
    target_native = config["target_native"]
    multiplier = float(config["target_inverse_multiplier"])
    columns = ["split", "timestamp", target, target_native, "observed_00400", *base_features]

    for site_index, (_, cohort_row) in enumerate(cohort.iterrows(), start=1):
        site_id = str(cohort_row["site_id"])
        path = model_file_path(processed_root, cohort_row)
        if not path.exists():
            raise FileNotFoundError(f"Missing model-ready site file: {path}")
        frame = read_split_table(path, "train", columns)
        positions = deterministic_time_positions(len(frame), cap)
        sample = frame.iloc[positions]
        base = sample[base_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        target_values = pd.to_numeric(sample[target], errors="coerce").to_numpy(dtype=np.float32)
        target_native_values = pd.to_numeric(sample[target_native], errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(base).all() or not np.isfinite(target_values).all():
            raise ValueError(f"Non-finite training value at site {site_id}")
        if not sample["observed_00400"].astype(bool).all():
            raise ValueError(f"Interpolated/unobserved target found in training data for {site_id}")
        inverse_error = float(np.max(np.abs(target_values * multiplier - target_native_values))) if len(sample) else 0.0
        if inverse_error > 2e-5:
            raise ValueError(f"Target inverse transform mismatch at {site_id}: {inverse_error}")
        n_sample = len(sample)
        x[offset : offset + n_sample] = base
        y[offset : offset + n_sample] = target_values
        audit_rows.append(
            {
                "site_id": site_id,
                "network": cohort_row["network"],
                "available_train_rows": int(len(frame)),
                "sampled_train_rows": int(n_sample),
                "sampling_fraction": float(n_sample / len(frame)) if len(frame) else np.nan,
                "sample_start": sample["timestamp"].iloc[0] if n_sample else pd.NaT,
                "sample_end": sample["timestamp"].iloc[-1] if n_sample else pd.NaT,
                "memmap_start": int(offset),
                "memmap_stop": int(offset + n_sample),
                "target_inverse_max_abs_error": inverse_error,
            }
        )
        offset += n_sample
        if site_index % 50 == 0 or site_index == len(cohort):
            logging.info("Training sample: %s/%s sites, %s rows", site_index, len(cohort), f"{offset:,}")
        del frame, sample, base, target_values, target_native_values
        gc.collect()
    if offset != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} sampled rows, wrote {offset}")
    x.flush()
    y.flush()
    return x, y, pd.DataFrame(audit_rows), x_path, y_path


def build_m1_memmap(
    base: np.memmap,
    training_audit: pd.DataFrame,
    metadata: pd.DataFrame,
    metadata_features: list[str],
    output_dir: Path,
) -> tuple[np.memmap, Path]:
    path = output_dir / "_tmp_memmap" / "X_train_M1_float32.dat"
    if path.exists():
        path.unlink()
    n_rows, n_base = base.shape
    result = np.memmap(path, mode="w+", dtype=np.float32, shape=(n_rows, n_base + len(metadata_features)))
    chunk = 500_000
    for start in range(0, n_rows, chunk):
        stop = min(n_rows, start + chunk)
        result[start:stop, :n_base] = base[start:stop]
    indexed = metadata.set_index("site_id")
    for row in training_audit.itertuples(index=False):
        values = indexed.loc[str(row.site_id), metadata_features].to_numpy(dtype=np.float32)
        result[int(row.memmap_start) : int(row.memmap_stop), n_base:] = values
    result.flush()
    return result, path


def fit_or_load_models(
    x_base: np.memmap,
    y: np.memmap,
    training_audit: pd.DataFrame,
    metadata: pd.DataFrame,
    metadata_features: list[str],
    metadata_categorical: list[str],
    base_features: list[str],
    config: dict[str, Any],
    output_dir: Path,
    resume: bool,
) -> tuple[dict[str, HistGradientBoostingRegressor], dict[str, list[str]]]:
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    features = {"M0": list(base_features), "M1": [*base_features, *metadata_features]}
    models: dict[str, HistGradientBoostingRegressor] = {}
    params = dict(config["hgb_parameters"])
    params["random_state"] = int(config["random_seed"])

    model_path = models_dir / "M0.joblib"
    if resume and model_path.exists():
        models["M0"] = joblib.load(model_path)
        logging.info("Resumed M0 from %s", model_path)
    else:
        logging.info("Training M0 on %s rows x %s features", f"{len(y):,}", x_base.shape[1])
        started = time.time()
        model = HistGradientBoostingRegressor(**params)
        model.fit(x_base, y)
        joblib.dump(model, model_path, compress=3)
        models["M0"] = model
        logging.info("M0 trained in %.1f seconds", time.time() - started)

    model_path = models_dir / "M1.joblib"
    if resume and model_path.exists():
        models["M1"] = joblib.load(model_path)
        logging.info("Resumed M1 from %s", model_path)
    else:
        x_m1, x_m1_path = build_m1_memmap(x_base, training_audit, metadata, metadata_features, output_dir)
        try:
            categorical_mask = np.array([name in set(metadata_categorical) for name in features["M1"]], dtype=bool)
            logging.info("Training M1 on %s rows x %s features", f"{len(y):,}", x_m1.shape[1])
            started = time.time()
            model = HistGradientBoostingRegressor(categorical_features=categorical_mask, **params)
            model.fit(x_m1, y)
            joblib.dump(model, model_path, compress=3)
            models["M1"] = model
            logging.info("M1 trained in %.1f seconds", time.time() - started)
        finally:
            del x_m1
            gc.collect()
            if x_m1_path.exists():
                x_m1_path.unlink()
    return models, features


def empty_accumulator() -> dict[str, float]:
    return {"n": 0, "sum_abs": 0.0, "sum_sq": 0.0, "sum_y": 0.0, "sum_y2": 0.0}


def update_accumulator(acc: dict[str, float], observed: np.ndarray, predicted: np.ndarray) -> None:
    residual = observed.astype(np.float64) - predicted.astype(np.float64)
    acc["n"] += int(len(observed))
    acc["sum_abs"] += float(np.abs(residual).sum(dtype=np.float64))
    acc["sum_sq"] += float(np.square(residual).sum(dtype=np.float64))
    values = observed.astype(np.float64)
    acc["sum_y"] += float(values.sum(dtype=np.float64))
    acc["sum_y2"] += float(np.square(values).sum(dtype=np.float64))


def accumulator_row(acc: dict[str, float]) -> dict[str, Any]:
    n = int(acc["n"])
    if n == 0:
        return {"n_rows": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    denominator = acc["sum_y2"] - (acc["sum_y"] ** 2 / n)
    return {
        "n_rows": n,
        "rmse": math.sqrt(acc["sum_sq"] / n),
        "mae": acc["sum_abs"] / n,
        "r2": 1.0 - acc["sum_sq"] / denominator if denominator > 0 else np.nan,
    }


def merge_accumulator(target: dict[str, float], source: dict[str, Any]) -> None:
    for key in target:
        target[key] += float(source[key])


def prediction_columns(config: dict[str, Any], base_features: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            [
                "site_id",
                "timestamp",
                "split",
                config["target_scaled"],
                config["target_native"],
                "observed_00400",
                *base_features,
                "in_ph_event",
                "ph_event_label",
                "ph_event_direction",
                "event_id",
            ]
        )
    )


def write_site_prediction(path: Path, chunks: list[pd.DataFrame]) -> None:
    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    atomic_write_parquet(path, frame)


def evaluate_models(
    models: dict[str, HistGradientBoostingRegressor],
    cohort: pd.DataFrame,
    processed_root: Path,
    metadata: pd.DataFrame,
    metadata_features: list[str],
    base_features: list[str],
    config: dict[str, Any],
    output_dir: Path,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled = {(model, split): empty_accumulator() for model in MODEL_IDS for split in SPLITS}
    networks: dict[tuple[str, str, str], dict[str, float]] = defaultdict(empty_accumulator)
    site_metric_rows: list[dict[str, Any]] = []
    metadata_indexed = metadata.set_index("site_id")
    checkpoint_dir = output_dir / "checkpoints" / "evaluation"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    columns = prediction_columns(config, base_features)
    multiplier = float(config["target_inverse_multiplier"])

    for site_number, (_, cohort_row) in enumerate(cohort.iterrows(), start=1):
        site_id = str(cohort_row["site_id"])
        network = str(cohort_row["network"])
        checkpoint_path = checkpoint_dir / f"{site_id}.json"
        prediction_paths = {
            model_id: output_dir / "predictions" / "test" / model_id / f"{site_id}.parquet"
            for model_id in MODEL_IDS
        }
        if resume and checkpoint_path.exists() and all(path.exists() for path in prediction_paths.values()):
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            for model_id in MODEL_IDS:
                for split in SPLITS:
                    acc = checkpoint["accumulators"][model_id][split]
                    merge_accumulator(pooled[(model_id, split)], acc)
                    merge_accumulator(networks[(model_id, split, network)], acc)
                    metric = {"model_id": model_id, "split": split, "site_id": site_id, "network": network}
                    metric.update(accumulator_row(acc))
                    metric["n_event_rows"] = int(checkpoint["event_rows"][model_id][split])
                    site_metric_rows.append(metric)
            logging.info("Evaluation resumed: %s", site_id)
            continue

        path = model_file_path(processed_root, cohort_row)
        site_acc = {model: {split: empty_accumulator() for split in SPLITS} for model in MODEL_IDS}
        event_rows = {model: {split: 0 for split in SPLITS} for model in MODEL_IDS}
        prediction_chunks: dict[str, list[pd.DataFrame]] = {model: [] for model in MODEL_IDS}
        metadata_values = metadata_indexed.loc[site_id, metadata_features].to_numpy(dtype=np.float32)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=columns, batch_size=int(config["prediction_batch_rows"])):
            frame = batch.to_pandas()
            frame = frame[frame["split"].isin(SPLITS)]
            if frame.empty:
                continue
            if not frame["observed_00400"].astype(bool).all():
                raise ValueError(f"Unobserved target in evaluation data for {site_id}")
            x_base = frame[base_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            if not np.isfinite(x_base).all():
                raise ValueError(f"Non-finite predictor in evaluation data for {site_id}")
            x_m1 = np.empty((len(frame), len(base_features) + len(metadata_features)), dtype=np.float32)
            x_m1[:, : len(base_features)] = x_base
            x_m1[:, len(base_features) :] = metadata_values
            observed_native = pd.to_numeric(frame[config["target_native"]], errors="coerce").to_numpy(dtype=np.float32)
            predictions = {
                "M0": models["M0"].predict(x_base).astype(np.float32),
                "M1": models["M1"].predict(x_m1).astype(np.float32),
            }
            for model_id, predicted_scaled in predictions.items():
                predicted_native = predicted_scaled * multiplier
                if not np.isfinite(predicted_native).all():
                    raise ValueError(f"Non-finite prediction for {model_id} at {site_id}")
                for split in SPLITS:
                    mask = frame["split"].eq(split).to_numpy()
                    if not mask.any():
                        continue
                    update_accumulator(site_acc[model_id][split], observed_native[mask], predicted_native[mask])
                    event_rows[model_id][split] += int(frame.loc[mask, "in_ph_event"].astype(bool).sum())
                test_mask = frame["split"].eq("test").to_numpy()
                if test_mask.any():
                    output = frame.loc[
                        test_mask,
                        ["site_id", "timestamp", "split", config["target_scaled"], config["target_native"], "in_ph_event", "ph_event_label", "ph_event_direction", "event_id"],
                    ].copy()
                    output = output.rename(columns={config["target_scaled"]: "observed_scaled", config["target_native"]: "observed_ph"})
                    output["predicted_scaled"] = predicted_scaled[test_mask]
                    output["predicted_ph"] = predicted_native[test_mask]
                    output["residual_observed_minus_predicted"] = output["observed_ph"].to_numpy(dtype=np.float32) - predicted_native[test_mask]
                    output["abs_residual"] = np.abs(output["residual_observed_minus_predicted"].to_numpy(dtype=np.float32))
                    prediction_chunks[model_id].append(output)
            del frame, x_base, x_m1, observed_native, predictions
            gc.collect()

        for model_id in MODEL_IDS:
            write_site_prediction(prediction_paths[model_id], prediction_chunks[model_id])
            for split in SPLITS:
                merge_accumulator(pooled[(model_id, split)], site_acc[model_id][split])
                merge_accumulator(networks[(model_id, split, network)], site_acc[model_id][split])
                metric = {"model_id": model_id, "split": split, "site_id": site_id, "network": network}
                metric.update(accumulator_row(site_acc[model_id][split]))
                metric["n_event_rows"] = event_rows[model_id][split]
                site_metric_rows.append(metric)
        write_json(
            checkpoint_path,
            {
                "site_id": site_id,
                "network": network,
                "accumulators": site_acc,
                "event_rows": event_rows,
                "prediction_files": {key: str(value) for key, value in prediction_paths.items()},
            },
        )
        if site_number % 25 == 0 or site_number == len(cohort):
            logging.info("Evaluation: %s/%s sites", site_number, len(cohort))
        del prediction_chunks
        gc.collect()

    comparison_rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        for split in SPLITS:
            row = {"model_id": model_id, "split": split, "n_sites": len(cohort), "metric_unit": "native pH"}
            row.update(accumulator_row(pooled[(model_id, split)]))
            comparison_rows.append(row)
    network_rows: list[dict[str, Any]] = []
    for (model_id, split, network), acc in sorted(networks.items()):
        row = {"model_id": model_id, "split": split, "network": network, "metric_unit": "native pH"}
        row.update(accumulator_row(acc))
        network_rows.append(row)
    return pd.DataFrame(comparison_rows), pd.DataFrame(site_metric_rows), pd.DataFrame(network_rows)


def normalise_event_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "mixed" in text:
        return "mixed"
    if "low" in text:
        return "low"
    if "high" in text:
        return "high"
    return "none"


def event_tail_analysis(
    cohort: pd.DataFrame,
    output_dir: Path,
    fractions: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tail_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        paths = [output_dir / "predictions" / "test" / model_id / f"{site_id}.parquet" for site_id in cohort["site_id"].astype(str)]
        residual_parts: list[np.ndarray] = []
        for path in paths:
            residual_parts.append(pq.read_table(path, columns=["residual_observed_minus_predicted"])[0].to_numpy(zero_copy_only=False).astype(np.float32))
        residual = np.concatenate(residual_parts)
        absolute = np.abs(residual)
        thresholds = {
            fraction: {
                "absolute": float(np.quantile(absolute, 1.0 - fraction)),
                "negative": float(np.quantile(residual, fraction)),
                "positive": float(np.quantile(residual, 1.0 - fraction)),
            }
            for fraction in fractions
        }
        totals = {"rows": 0, "event": 0, "low": 0, "high": 0, "mixed": 0}
        selected: dict[tuple[float, str], dict[str, int]] = {}
        event_ids: set[tuple[str, int]] = set()
        hit_event_ids: dict[float, set[tuple[str, int]]] = {fraction: set() for fraction in fractions}
        offset = 0
        for site_id, path in zip(cohort["site_id"].astype(str), paths):
            frame = pd.read_parquet(path, columns=["in_ph_event", "ph_event_label", "event_id", "residual_observed_minus_predicted"])
            n = len(frame)
            site_residual = residual[offset : offset + n]
            offset += n
            labels = frame["ph_event_label"].map(normalise_event_label).to_numpy()
            event = frame["in_ph_event"].astype(bool).to_numpy()
            totals["rows"] += n
            totals["event"] += int(event.sum())
            for label in ("low", "high", "mixed"):
                totals[label] += int((labels == label).sum())
            valid_event_id = event & frame["event_id"].notna().to_numpy()
            for value in frame.loc[valid_event_id, "event_id"].astype("int64").unique():
                event_ids.add((site_id, int(value)))
            for fraction in fractions:
                absolute_mask = np.abs(site_residual) >= thresholds[fraction]["absolute"]
                negative_mask = site_residual <= thresholds[fraction]["negative"]
                positive_mask = site_residual >= thresholds[fraction]["positive"]
                for name, mask in (("absolute", absolute_mask), ("negative", negative_mask), ("positive", positive_mask)):
                    key = (fraction, name)
                    stats = selected.setdefault(key, {"rows": 0, "event": 0, "low": 0, "high": 0, "mixed": 0})
                    stats["rows"] += int(mask.sum())
                    stats["event"] += int((event & mask).sum())
                    for label in ("low", "high", "mixed"):
                        stats[label] += int(((labels == label) & mask).sum())
                hit_mask = valid_event_id & absolute_mask
                for value in frame.loc[hit_mask, "event_id"].astype("int64").unique():
                    hit_event_ids[fraction].add((site_id, int(value)))
        if offset != len(residual):
            raise RuntimeError("Residual/event scan row count mismatch")

        for fraction in fractions:
            for tail_type in ("absolute", "negative", "positive"):
                stats = selected[(fraction, tail_type)]
                selected_rate = stats["event"] / stats["rows"] if stats["rows"] else np.nan
                base_rate = totals["event"] / totals["rows"] if totals["rows"] else np.nan
                tail_rows.append(
                    {
                        "model_id": model_id,
                        "tail_fraction_requested": fraction,
                        "tail_type": tail_type,
                        "threshold_native_ph": thresholds[fraction][tail_type],
                        "n_test_rows": totals["rows"],
                        "n_selected_rows": stats["rows"],
                        "actual_selected_fraction": stats["rows"] / totals["rows"],
                        "base_event_rate": base_rate,
                        "selected_event_rate": selected_rate,
                        "event_enrichment": selected_rate / base_rate if base_rate > 0 else np.nan,
                        "selected_low_rows": stats["low"],
                        "selected_high_rows": stats["high"],
                        "selected_mixed_rows": stats["mixed"],
                    }
                )
            absolute_stats = selected[(fraction, "absolute")]
            direction_rows.append(
                {
                    "model_id": model_id,
                    "tail_fraction_requested": fraction,
                    "n_event_ids": len(event_ids),
                    "n_event_ids_hit_by_absolute_tail": len(hit_event_ids[fraction]),
                    "event_id_hit_rate": len(hit_event_ids[fraction]) / len(event_ids) if event_ids else np.nan,
                    "low_base_rate": totals["low"] / totals["rows"],
                    "low_rate_in_negative_tail": selected[(fraction, "negative")]["low"] / selected[(fraction, "negative")]["rows"],
                    "low_rate_in_absolute_tail": absolute_stats["low"] / absolute_stats["rows"],
                    "high_base_rate": totals["high"] / totals["rows"],
                    "high_rate_in_positive_tail": selected[(fraction, "positive")]["high"] / selected[(fraction, "positive")]["rows"],
                    "high_rate_in_absolute_tail": absolute_stats["high"] / absolute_stats["rows"],
                }
            )
        del residual, absolute, residual_parts
        gc.collect()
    return pd.DataFrame(tail_rows), pd.DataFrame(direction_rows)


def create_report(
    output_dir: Path,
    mode: str,
    cohort: pd.DataFrame,
    training_audit: pd.DataFrame,
    comparison: pd.DataFrame,
    metadata: pd.DataFrame,
) -> str:
    test = comparison[comparison["split"].eq("test")].set_index("model_id")
    lines = [
        f"# Common1000 HGB {mode} report",
        "",
        "## Outcome",
        "",
        f"- Sites: {len(cohort):,}",
        f"- Sampled training rows: {int(training_audit['sampled_train_rows'].sum()):,}",
        f"- Full validation rows: {int(comparison.loc[comparison['split'].eq('val'), 'n_rows'].max()):,}",
        f"- Full test rows: {int(comparison.loc[comparison['split'].eq('test'), 'n_rows'].max()):,}",
        f"- Registry matches: {int(metadata['metadata_match'].sum()):,}/{len(metadata):,}",
        "",
        "## Test metrics (native pH units)",
        "",
        "| Model | RMSE | MAE | R2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model_id in MODEL_IDS:
        row = test.loc[model_id]
        lines.append(f"| {model_id} | {row['rmse']:.6f} | {row['mae']:.6f} | {row['r2']:.6f} |")
    lines.extend(
        [
            "",
            "## Safeguards",
            "",
            "- pH target is never a predictor and is observed rather than interpolated.",
            "- Event labels, event IDs, predictions and residuals are evaluation-only.",
            "- Training is time-stratified within each site's training split and capped per site.",
            "- Validation and test metrics use every available row and are reported after inverse scaling to native pH.",
            "- M1 metadata is static; missing numeric fields are cohort-median imputed with explicit missingness flags.",
        ]
    )
    return "\n".join(lines) + "\n"


def cleanup_memmaps(arrays: list[np.memmap], paths: list[Path]) -> None:
    for array in arrays:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
    gc.collect()
    for path in paths:
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    processed_root = resolve_path(config["processed_root"])
    output_root = resolve_path(config["output_root"])
    output_dir = output_root / args.mode
    setup_logging(output_dir)
    success_path = output_dir / "_SUCCESS"
    if success_path.exists() and args.resume:
        logging.info("Run already complete: %s", success_path)
        return
    if success_path.exists():
        success_path.unlink()

    started = time.time()
    logging.info("Starting Common1000 HGB mode=%s", args.mode)
    cohort = select_cohort(resolve_path(config["cohort_manifest"]), args.mode, config)
    write_csv(output_dir / "reports" / "cohort.csv", cohort)
    base_features = list(config["base_features"])
    assert_safe_features(base_features, config)

    metadata, metadata_features, metadata_categorical, metadata_payload = registry_metadata(
        cohort, resolve_path(config["registry_path"])
    )
    assert_safe_features(metadata_features, config)
    write_csv(output_dir / "reports" / "metadata_model_ready.csv", metadata)
    write_csv(output_dir / "reports" / "metadata_raw.csv", metadata_payload.pop("raw"))
    write_csv(output_dir / "reports" / "metadata_feature_audit.csv", pd.DataFrame(metadata_payload.pop("audit")))
    write_json(output_dir / "reports" / "metadata_spec.json", metadata_payload["spec"])
    write_json(
        output_dir / "reports" / "feature_lists.json",
        {
            "M0": base_features,
            "M1": [*base_features, *metadata_features],
            "M1_categorical_features": metadata_categorical,
            "forbidden_columns": [config["target_scaled"], config["target_native"], "in_ph_event", "ph_event_label", "ph_event_direction", "event_id"],
        },
    )

    x_base, y, training_audit, x_path, y_path = build_training_memmaps(
        cohort, processed_root, base_features, config, output_dir
    )
    write_csv(output_dir / "reports" / "training_sample_audit.csv", training_audit)
    try:
        models, feature_sets = fit_or_load_models(
            x_base,
            y,
            training_audit,
            metadata,
            metadata_features,
            metadata_categorical,
            base_features,
            config,
            output_dir,
            args.resume,
        )
    finally:
        cleanup_memmaps([x_base, y], [x_path, y_path])

    comparison, site_metrics, network_metrics = evaluate_models(
        models,
        cohort,
        processed_root,
        metadata,
        metadata_features,
        base_features,
        config,
        output_dir,
        args.resume,
    )
    write_csv(output_dir / "reports" / "model_comparison_metrics.csv", comparison)
    write_csv(output_dir / "reports" / "site_level_metrics.csv", site_metrics)
    write_csv(output_dir / "reports" / "network_metrics.csv", network_metrics)

    tails, directions = event_tail_analysis(cohort, output_dir, [float(value) for value in config["residual_tail_fractions"]])
    write_csv(output_dir / "reports" / "residual_tail_analysis.csv", tails)
    write_csv(output_dir / "reports" / "directional_event_analysis.csv", directions)

    expected_test = int(pd.to_numeric(cohort["test_rows"]).sum())
    expected_val = int(pd.to_numeric(cohort["validation_rows"]).sum())
    validation_problems: list[str] = []
    for model_id in MODEL_IDS:
        model_test = int(comparison.loc[(comparison["model_id"].eq(model_id)) & (comparison["split"].eq("test")), "n_rows"].iloc[0])
        model_val = int(comparison.loc[(comparison["model_id"].eq(model_id)) & (comparison["split"].eq("val")), "n_rows"].iloc[0])
        if model_test != expected_test:
            validation_problems.append(f"{model_id} test rows {model_test} != {expected_test}")
        if model_val != expected_val:
            validation_problems.append(f"{model_id} validation rows {model_val} != {expected_val}")
        prediction_count = len(list((output_dir / "predictions" / "test" / model_id).glob("*.parquet")))
        if prediction_count != len(cohort):
            validation_problems.append(f"{model_id} prediction files {prediction_count} != {len(cohort)}")
    if training_audit["site_id"].nunique() != len(cohort):
        validation_problems.append("training sample audit does not cover every cohort site")
    if not np.isfinite(comparison[["rmse", "mae", "r2"]].to_numpy(dtype=float)).all():
        validation_problems.append("non-finite comparison metric")

    hashes = [
        {"role": "config", "path": str(config_path), "sha256": sha256_file(config_path)},
        {"role": "script", "path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
    ]
    for model_id in MODEL_IDS:
        model_path = output_dir / "models" / f"{model_id}.joblib"
        hashes.append({"role": f"model_{model_id}", "path": str(model_path), "sha256": sha256_file(model_path)})
    write_csv(output_dir / "reports" / "artifact_hashes.csv", pd.DataFrame(hashes))
    summary = {
        "status": "success" if not validation_problems else "failed_validation",
        "mode": args.mode,
        "sites": len(cohort),
        "networks": cohort.groupby("network", observed=True).size().to_dict(),
        "sampled_training_rows": int(training_audit["sampled_train_rows"].sum()),
        "full_validation_rows": expected_val,
        "full_test_rows": expected_test,
        "models": list(MODEL_IDS),
        "features": feature_sets,
        "metrics_unit": "native pH",
        "validation_problems": validation_problems,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "reports" / "run_summary.json", summary)
    atomic_write_text(output_dir / "reports" / "FINAL_REPORT.md", create_report(output_dir, args.mode, cohort, training_audit, comparison, metadata))
    if validation_problems:
        raise RuntimeError(f"Post-run validation failed: {validation_problems}")
    write_json(success_path, {"status": "success", "mode": args.mode, "sites": len(cohort), "completed_at_epoch": time.time()})
    logging.info("Completed mode=%s in %.1f seconds", args.mode, time.time() - started)


if __name__ == "__main__":
    main()
