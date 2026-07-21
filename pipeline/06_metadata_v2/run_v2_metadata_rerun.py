from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
ANALYSIS_DIR = ROOT / "analysis_outputs"
OUT_DIR = ANALYSIS_DIR / "ph_v2_metadata_rerun_v1"
FIG_DIR = OUT_DIR / "figures"
TMP_DIR = OUT_DIR / "_tmp_memmap"

HGB50_PATH = ANALYSIS_DIR / "ph_metadata_audit_v1" / "ph_baseline_ready_v3_with_metadata.parquet"
HGB200_PATH = ANALYSIS_DIR / "ph_hgb_processed200_metadata_v1" / "processed200_model_ready_with_metadata.parquet"
PROCESSED100_METADATA_TABLE = ANALYSIS_DIR / "ph_processed100_step4_0_4_1_v1" / "processed100_metadata_model_ready.csv"
PROCESSED100_METADATA_SPEC = ANALYSIS_DIR / "ph_processed100_step4_0_4_1_v1" / "processed100_metadata_feature_set.json"
PHASE5_GRU_DIR = ANALYSIS_DIR / "ph_sequence100_gru_lstm_metadata_v1"
PHASE5B_LSTM_DIR = ANALYSIS_DIR / "ph_sequence100_lstm_metadata_v1"
FULL50_SUMMARY_DIR = ANALYSIS_DIR / "ph_full50_model_summary_with_m0_v1"
HGB50_PREV_DIR = ANALYSIS_DIR / "ph_hgb_full50_metadata_v1"
HGB200_PREV_DIR = ANALYSIS_DIR / "ph_hgb_processed200_metadata_v1"

TARGET = "00400"
SITE_COL = "site_id"
TIME_COL = "timestamp"
SPLIT_COL = "split"
EVENT_COL = "in_ph_event"
EVENT_LABEL_COL = "ph_event_label"
EVENT_STATE_COL = "event_state"
SITE_CODE_COL = "site_id_code"
SEQUENCE_LENGTH = 24
RANDOM_SEED = 42

DYNAMIC_FEATURES = [
    "00010",
    "00095",
    "63680",
    "00060",
    "00065",
    "sin_hour",
    "cos_hour",
    "sin_day",
    "cos_day",
    "month",
    "hour",
    "day_of_year",
]
BASE_HGB_FEATURES = [*DYNAMIC_FEATURES, SITE_CODE_COL]
TOP_FRACTIONS = [0.01, 0.05, 0.10]
PH_EVENT_LABELS = ["pH_exceedance_low", "pH_exceedance_high"]

HGB_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "random_state": 42,
    "early_stopping": False,
}

TRAIN_BATCH_SIZE = 2048
PREDICT_BATCH_SIZE = 4096
MAX_TRAIN_WINDOWS_CPU = 1_000_000
EARLY_STOP_VAL_WINDOWS_CPU = 500_000
MAX_EPOCHS_CPU = 8
EARLY_STOPPING_PATIENCE = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRU_HIDDEN_SIZE = 64
GRU_NUM_LAYERS = 1
GRU_MLP_HIDDEN = 64
GRU_DROPOUT = 0.0

RECOMMENDED_OFFICIAL = [
    "meta_official_latitude",
    "meta_official_longitude",
    "meta_official_elevation_m",
    "meta_official_drainage_area_km2",
    "meta_official_official_average_discharge",
    "meta_official_catchment_average_precip_mm_yr",
    "meta_official_catchment_average_evapo_trans_mm_yr",
    "meta_official_catchment_population_est_2020",
    "meta_official_catchment_watershed_dams_no",
    "meta_official_catchment_watershed_dams_cap_m3",
    "meta_official_catchment_irrigation_area_km2",
    "meta_official_catchment_irrigation_area_pct",
    "meta_official_dominant_land_cover_code",
    "meta_official_dominant_land_cover_pct",
    "meta_official_site_type_code",
    "meta_official_basin_cd_code",
]
RECOMMENDED_TRAIN_DERIVED = [
    "meta_train_00010_median",
    "meta_train_00010_std",
    "meta_train_00095_median",
    "meta_train_00095_std",
    "meta_train_63680_median",
    "meta_train_63680_std",
    "meta_train_00060_median",
    "meta_train_00060_std",
    "meta_train_00065_median",
    "meta_train_00065_std",
    "meta_train_coverage_days",
    "meta_train_dominant_interval_minutes",
]
ALIASES = {
    "meta_official_elevation_m": ["meta_official_elevation_m", "meta_official_elevation"],
    "meta_official_drainage_area_km2": ["meta_official_drainage_area_km2", "meta_official_drainage_area"],
    "meta_official_official_average_discharge": [
        "meta_official_official_average_discharge",
        "meta_official_average_discharge",
    ],
    "meta_official_dominant_land_cover_code": [
        "meta_official_dominant_land_cover_code",
        "meta_official_land_cover_code",
    ],
    "meta_official_dominant_land_cover_pct": [
        "meta_official_dominant_land_cover_pct",
        "meta_official_land_cover_pct",
    ],
    "meta_official_basin_cd_code": ["meta_official_basin_cd_code", "meta_official_basin_code"],
}
FORBIDDEN_SUBSTRINGS = [
    "00400",
    "target",
    "label",
    "event",
    "incident",
    "exceedance",
    "low_ph",
    "high_ph",
    "residual",
    "prediction",
    "pred",
    "rmse",
    "mae",
    "r2",
    "test",
    "val",
    "validation",
    "future",
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, msg: str) -> None:
        for stream in self.streams:
            stream.write(msg)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_stdout(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, handle)
        sys.stderr = Tee(old_stderr, handle)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def log(message: str) -> None:
    print(message, flush=True)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def normalise_site_id(value: object) -> str:
    raw = str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    return raw.zfill(8) if raw.isdigit() else raw


def normalise_event_label(value: object) -> str:
    text = str(value).strip().lower()
    if "low" in text:
        return "pH_exceedance_low"
    if "high" in text:
        return "pH_exceedance_high"
    if "mixed" in text:
        return "pH_exceedance_mixed"
    return "none"


def compact_label(value: object) -> str:
    label = normalise_event_label(value)
    if "low" in label:
        return "low"
    if "high" in label:
        return "high"
    if "mixed" in label:
        return "mixed"
    return "none"


def threshold_label(fraction: float) -> str:
    return f"top_{int(round(fraction * 100))}pct"


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
        else:
            display[col] = display[col].astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.to_numpy()]
    return "\n".join([header, separator, *body])


def schema_names(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def leakage_reason(feature: str) -> str | None:
    lower = feature.lower()
    if feature in RECOMMENDED_TRAIN_DERIVED:
        return None
    if "ph" in lower and "precip" not in lower:
        return "contains pH marker"
    for token in FORBIDDEN_SUBSTRINGS:
        if token in lower:
            return f"contains forbidden marker: {token}"
    return None


def match_v2_features(dataset_scope: str, available_cols: set[str]) -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    leak_rows: list[dict[str, object]] = []
    used: list[str] = []
    for rec in [*RECOMMENDED_OFFICIAL, *RECOMMENDED_TRAIN_DERIVED]:
        candidates = ALIASES.get(rec, [rec])
        matched = next((col for col in candidates if col in available_cols), "")
        source = "official_core_hydrological" if rec in RECOMMENDED_OFFICIAL else "train_only_exogenous_background"
        available = bool(matched)
        reason = "" if available else "recommended V2 field unavailable in this dataset scope"
        leak = leakage_reason(matched or rec)
        used_in_v2 = available and leak is None
        if used_in_v2:
            used.append(matched)
        rows.append(
            {
                "dataset_scope": dataset_scope,
                "recommended_feature_name": rec,
                "matched_column_name": matched,
                "available": available,
                "source": source,
                "reason_if_missing": reason,
                "used_in_v2": used_in_v2,
            }
        )
        leak_rows.append(
            {
                "dataset_scope": dataset_scope,
                "feature_name": matched or rec,
                "recommended_feature_name": rec,
                "status": "PASS" if leak is None and available else ("MISSING" if not available else "EXCLUDED"),
                "reason": "" if leak is None else leak,
                "used_in_v2": used_in_v2,
            }
        )
    return used, rows, leak_rows


def build_v2_feature_definitions() -> dict[str, list[str]]:
    scope_paths = {
        "hgb50": HGB50_PATH,
        "hgb200": HGB200_PATH,
        "gru100": ANALYSIS_DIR / "ph_processed100_step4_0_4_1_v1" / "processed100_with_event_state_metadata.parquet",
    }
    availability_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    selected: dict[str, list[str]] = {}
    for scope, path in scope_paths.items():
        cols = set(schema_names(path))
        features, rows, leak_rows = match_v2_features(scope, cols)
        selected[scope] = features
        availability_rows.extend(rows)
        leakage_rows.extend(leak_rows)
        log(f"[INFO] V2 features for {scope}: {len(features)}")

    pd.DataFrame(availability_rows).to_csv(OUT_DIR / "v2_metadata_feature_availability.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(leakage_rows).to_csv(OUT_DIR / "v2_metadata_leakage_screening.csv", index=False, encoding="utf-8-sig")
    definition = {
        "name": "compact_hydrological_v2_metadata",
        "description": "Core official hydrological metadata plus selected train-only exogenous site-background summaries.",
        "recommended_official_features": RECOMMENDED_OFFICIAL,
        "recommended_train_only_exogenous_features": RECOMMENDED_TRAIN_DERIVED,
        "alias_mapping": ALIASES,
        "selected_features_by_scope": selected,
        "excluded_by_design": [
            "meta_missing_* fields",
            "source/units/status/country/state/county administrative codes",
            "train-derived mean/p05/p95/iqr/missing_rate",
            "pH-derived features",
            "event-derived features",
            "residual/prediction/model-metric features",
            "validation/test/future-derived features",
        ],
    }
    (OUT_DIR / "v2_metadata_feature_definition.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")
    return selected


def scan_split_counts_and_site_map(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    log(f"[INFO] Scanning split counts and train sites: {rel(path)}")
    counts = {"train": 0, "val": 0, "test": 0}
    train_sites: set[str] = set()
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=[SITE_COL, SPLIT_COL], batch_size=1_000_000):
        df = batch.to_pandas()
        split = df[SPLIT_COL].astype(str)
        for key in counts:
            counts[key] += int(split.eq(key).sum())
        sites = df.loc[split.eq("train"), SITE_COL].map(normalise_site_id).astype(str).unique().tolist()
        train_sites.update(sites)
        del df
    site_map = {site: idx for idx, site in enumerate(sorted(train_sites))}
    return counts, site_map


def build_hgb_block(df: pd.DataFrame, feature_names: list[str], site_map: dict[str, int], v2_features: list[str]) -> np.ndarray:
    x = np.empty((len(df), len(feature_names)), dtype=np.float32)
    pos = 0
    for feature in DYNAMIC_FEATURES:
        x[:, pos] = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        pos += 1
    site_codes = df[SITE_COL].map(normalise_site_id).astype(str).map(site_map).to_numpy(dtype=np.int32)
    if np.any(pd.isna(site_codes)):
        raise RuntimeError("Site code mapping produced missing values.")
    x[:, pos] = site_codes.astype(np.float32)
    pos += 1
    for feature in v2_features:
        x[:, pos] = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        pos += 1
    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite HGB feature block detected.")
    return x


def create_hgb_train_memmap(
    model_id: str,
    path: Path,
    feature_names: list[str],
    v2_features: list[str],
    split_counts: dict[str, int],
    site_map: dict[str, int],
) -> tuple[np.memmap, np.memmap, Path, Path]:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    x_path = TMP_DIR / f"{model_id}_X_train_float32.dat"
    y_path = TMP_DIR / f"{model_id}_y_train_float32.dat"
    for p in [x_path, y_path]:
        if p.exists():
            p.unlink()
    n_train = split_counts["train"]
    x_train = np.memmap(x_path, mode="w+", dtype=np.float32, shape=(n_train, len(feature_names)))
    y_train = np.memmap(y_path, mode="w+", dtype=np.float32, shape=(n_train,))
    cols = list(dict.fromkeys([SITE_COL, SPLIT_COL, TARGET, *DYNAMIC_FEATURES, *v2_features]))
    pf = pq.ParquetFile(path)
    offset = 0
    for batch_idx, batch in enumerate(pf.iter_batches(columns=cols, batch_size=500_000), start=1):
        df = batch.to_pandas()
        work = df[df[SPLIT_COL].astype(str).eq("train")].copy()
        if not work.empty:
            n = len(work)
            x_train[offset : offset + n] = build_hgb_block(work, feature_names, site_map, v2_features)
            y_train[offset : offset + n] = pd.to_numeric(work[TARGET], errors="coerce").to_numpy(dtype=np.float32, copy=False)
            offset += n
        if batch_idx % 20 == 0:
            log(f"[INFO] {model_id} memmap train rows written: {offset:,}")
        del df, work
        gc.collect()
    if offset != n_train:
        raise RuntimeError(f"{model_id} expected {n_train:,} train rows but wrote {offset:,}")
    x_train.flush()
    y_train.flush()
    log(f"[INFO] {model_id} train memmap complete: {n_train:,} rows x {len(feature_names)} features")
    return x_train, y_train, x_path, y_path


def train_hgb(model_id: str, x_train: np.memmap, y_train: np.memmap) -> HistGradientBoostingRegressor:
    log(f"[INFO] Training {model_id} HGB on {x_train.shape[0]:,} rows x {x_train.shape[1]} features")
    started = perf_counter()
    model = HistGradientBoostingRegressor(**HGB_PARAMS)
    model.fit(x_train, y_train)
    log(f"[INFO] Finished {model_id} HGB in {perf_counter() - started:.1f}s")
    return model


def predict_hgb_split(
    model: HistGradientBoostingRegressor,
    model_id: str,
    path: Path,
    split: str,
    feature_names: list[str],
    v2_features: list[str],
    site_map: dict[str, int],
    out_path: Path,
) -> None:
    if out_path.exists():
        out_path.unlink()
    cols = list(dict.fromkeys([SITE_COL, TIME_COL, SPLIT_COL, TARGET, EVENT_COL, EVENT_LABEL_COL, *DYNAMIC_FEATURES, *v2_features]))
    pf = pq.ParquetFile(path)
    writer: pq.ParquetWriter | None = None
    total = 0
    for batch in pf.iter_batches(columns=cols, batch_size=500_000):
        df = batch.to_pandas()
        work = df[df[SPLIT_COL].astype(str).eq(split)].copy()
        if work.empty:
            del df, work
            continue
        x = build_hgb_block(work, feature_names, site_map, v2_features)
        y_pred = model.predict(x).astype("float32")
        y_true = pd.to_numeric(work[TARGET], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        pred = work[[SITE_COL, TIME_COL, SPLIT_COL, TARGET, EVENT_COL, EVENT_LABEL_COL]].copy()
        pred[SITE_COL] = pred[SITE_COL].map(normalise_site_id).astype("string")
        pred[TIME_COL] = pd.to_datetime(pred[TIME_COL], errors="coerce")
        pred["y_pred"] = y_pred
        pred["residual"] = (y_true - y_pred).astype("float32")
        pred["abs_residual"] = np.abs(pred["residual"].to_numpy(dtype=np.float32)).astype("float32")
        table = pa.Table.from_pandas(pred, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)
        total += len(pred)
        del df, work, x, y_pred, y_true, pred, table
        gc.collect()
    if writer is not None:
        writer.close()
    log(f"[INFO] Wrote {model_id} {split} predictions: {total:,} rows -> {rel(out_path)}")


def run_hgb_v2(
    scope: str,
    model_id: str,
    model_name: str,
    path: Path,
    v2_features: list[str],
    test_prediction_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log(f"[INFO] Running {model_id}")
    split_counts, site_map = scan_split_counts_and_site_map(path)
    feature_names = [*BASE_HGB_FEATURES, *v2_features]
    x_train, y_train, x_path, y_path = create_hgb_train_memmap(model_id, path, feature_names, v2_features, split_counts, site_map)
    try:
        model = train_hgb(model_id, x_train, y_train)
    finally:
        del x_train, y_train
        gc.collect()
        for p in [x_path, y_path]:
            if p.exists():
                p.unlink()
    val_path = OUT_DIR / f"{model_id.lower()}_predictions_val.parquet"
    test_path = OUT_DIR / test_prediction_name
    predict_hgb_split(model, model_id, path, "val", feature_names, v2_features, site_map, val_path)
    predict_hgb_split(model, model_id, path, "test", feature_names, v2_features, site_map, test_path)
    del model
    gc.collect()
    val_pred = load_prediction(val_path)
    test_pred = load_prediction(test_path)
    metrics = pd.DataFrame(
        [
            prediction_metric_row(model_id, model_name, "val", val_pred, True, "rows"),
            prediction_metric_row(model_id, model_name, "test", test_pred, True, "rows"),
        ]
    )
    residual = residual_event_table(model_id, model_name, test_pred, "rows")
    low_high = low_high_table(model_id, model_name, test_pred, "rows")
    event_hit = event_level_hit_table(model_id, model_name, test_pred)
    site_metrics = site_level_table(model_id, model_name, test_pred)
    return metrics, residual, low_high, event_hit, site_metrics


def load_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df[SITE_COL] = df[SITE_COL].map(normalise_site_id).astype("string")
    time_col = "target_timestamp" if "target_timestamp" in df.columns else TIME_COL
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[EVENT_LABEL_COL] = df[EVENT_LABEL_COL].map(normalise_event_label).astype("string")
    return df


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(pred)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    y = y[mask]
    pred = pred[mask]
    residual = y - pred
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": n,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
    }


def prediction_metric_row(model_id: str, model_name: str, split: str, pred: pd.DataFrame, metadata_used: bool, unit: str) -> dict[str, object]:
    metrics = metric_values(pred[TARGET].to_numpy(), pred["y_pred"].to_numpy())
    return {
        "model_id": model_id,
        "model_name": model_name,
        "split": split,
        "n_rows": int(metrics["n"]),
        "n_sites": int(pred[SITE_COL].nunique()),
        "metadata_used": metadata_used,
        "sequence_length": SEQUENCE_LENGTH if unit == "windows" else np.nan,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
    }


def exact_top_mask(values: np.ndarray, fraction: float) -> tuple[np.ndarray, float]:
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=bool), np.nan
    n_top = max(1, int(math.ceil(n * fraction)))
    idx = np.argpartition(values, n - n_top)[n - n_top :]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask, float(np.min(values[idx]))


def residual_event_table(model_id: str, model_name: str, pred: pd.DataFrame, unit: str) -> pd.DataFrame:
    rows = []
    is_event = pred[EVENT_COL].astype(bool).to_numpy()
    abs_residual = pred["abs_residual"].to_numpy(dtype=np.float64)
    base = float(is_event.mean()) if len(pred) else np.nan
    for fraction in TOP_FRACTIONS:
        mask, threshold = exact_top_mask(abs_residual, fraction)
        top_rate = float(is_event[mask].mean()) if mask.any() else np.nan
        rows.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "threshold": threshold_label(fraction),
                "top_fraction": fraction,
                "n_test_rows": int(len(pred)),
                "n_event_rows": int(is_event.sum()),
                "base_event_rate": base,
                "n_top_rows": int(mask.sum()),
                "n_top_event_rows": int((mask & is_event).sum()),
                "top_event_rate": top_rate,
                "enrichment": top_rate / base if base and np.isfinite(base) else np.nan,
                "abs_residual_threshold": threshold,
                "unit": unit,
            }
        )
    return pd.DataFrame(rows)


def low_high_table(model_id: str, model_name: str, pred: pd.DataFrame, unit: str) -> pd.DataFrame:
    rows = []
    labels = pred[EVENT_LABEL_COL].map(compact_label).to_numpy()
    abs_residual = pred["abs_residual"].to_numpy(dtype=np.float64)
    for fraction in TOP_FRACTIONS:
        mask, _threshold = exact_top_mask(abs_residual, fraction)
        for label in ["low", "high"]:
            label_mask = labels == label
            base = float(label_mask.mean()) if len(pred) else np.nan
            top_rate = float(label_mask[mask].mean()) if mask.any() else np.nan
            rows.append(
                {
                    "model_id": model_id,
                    "model_name": model_name,
                    "threshold": threshold_label(fraction),
                    "top_fraction": fraction,
                    "event_label": f"pH_exceedance_{label}",
                    "n_label_rows": int(label_mask.sum()),
                    "base_label_rate": base,
                    "n_top_rows": int(mask.sum()),
                    "n_top_label_rows": int((mask & label_mask).sum()),
                    "top_label_rate": top_rate,
                    "label_enrichment": top_rate / base if base and np.isfinite(base) else np.nan,
                    "unit": unit,
                }
            )
    return pd.DataFrame(rows)


def event_level_hit_table(model_id: str, model_name: str, pred: pd.DataFrame) -> pd.DataFrame:
    time_col = "target_timestamp" if "target_timestamp" in pred.columns else TIME_COL
    work = pred[[SITE_COL, time_col, EVENT_COL, EVENT_LABEL_COL, "abs_residual"]].copy()
    work = work.rename(columns={time_col: "event_time"})
    work["event_time"] = pd.to_datetime(work["event_time"], errors="coerce")
    work["label_norm"] = work[EVENT_LABEL_COL].map(compact_label)
    work = work.sort_values([SITE_COL, "event_time"], kind="mergesort").reset_index(drop=True)
    event_mask = work[EVENT_COL].astype(bool).to_numpy() & work["label_norm"].isin(["low", "high"]).to_numpy()
    prev_event = pd.Series(event_mask).shift(1).fillna(False).to_numpy(dtype=bool)
    prev_site = work[SITE_COL].shift(1)
    prev_label = work["label_norm"].shift(1)
    block_start = event_mask & (
        (~prev_event)
        | (work[SITE_COL] != prev_site).to_numpy()
        | (work["label_norm"] != prev_label).to_numpy()
    )
    work["event_window_id"] = np.where(event_mask, np.cumsum(block_start), np.nan)
    n_windows = int(work.loc[event_mask, "event_window_id"].dropna().nunique())
    abs_residual = work["abs_residual"].to_numpy(dtype=np.float64)
    rows = []
    for fraction in TOP_FRACTIONS:
        top, _threshold = exact_top_mask(abs_residual, fraction)
        hit = work.loc[event_mask & top, "event_window_id"].dropna().astype("int64")
        n_hit = int(hit.nunique())
        rows.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "threshold": threshold_label(fraction),
                "top_fraction": fraction,
                "event_window_method": "approximate_contiguous_blocks_within_site_and_label_on_sorted_test_rows",
                "n_event_windows": n_windows,
                "n_event_windows_hit": n_hit,
                "event_level_hit_rate": n_hit / n_windows if n_windows else np.nan,
            }
        )
    return pd.DataFrame(rows)


def site_level_table(model_id: str, model_name: str, pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site_id, group in pred.groupby(SITE_COL, sort=True, observed=True):
        metrics = metric_values(group[TARGET].to_numpy(), group["y_pred"].to_numpy())
        rows.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "site_id": site_id,
                "n_test_rows": int(len(group)),
                "n_test_event_rows": int(group[EVENT_COL].astype(bool).sum()),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )
    return pd.DataFrame(rows)


def sample_indices_site_balanced(
    windows: pd.DataFrame,
    indices: np.ndarray,
    cap: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, bool]:
    if len(indices) <= cap:
        summary = windows.iloc[indices].groupby(SITE_COL, observed=True).size().reset_index(name="available_windows")
        summary["sampled_windows"] = summary["available_windows"]
        summary["sample_fraction"] = 1.0
        return indices.copy(), summary, False
    rng = np.random.default_rng(seed)
    meta = windows.iloc[indices][[SITE_COL]].copy()
    meta["_idx"] = indices
    parts = []
    rows = []
    total = len(indices)
    desired = []
    for site_id, group in meta.groupby(SITE_COL, sort=True, observed=True):
        arr = group["_idx"].to_numpy(dtype=np.int64)
        exact = len(arr) / total * cap
        desired.append([site_id, arr, int(math.floor(exact)), exact - math.floor(exact)])
    remainder = cap - sum(item[2] for item in desired)
    for item in sorted(desired, key=lambda row: row[3], reverse=True):
        if remainder <= 0:
            break
        if item[2] < len(item[1]):
            item[2] += 1
            remainder -= 1
    for site_id, arr, n_take, _frac in desired:
        n_take = min(max(1, n_take), len(arr))
        sampled = rng.choice(arr, size=n_take, replace=False)
        sampled.sort()
        parts.append(sampled)
        rows.append(
            {
                SITE_COL: site_id,
                "available_windows": int(len(arr)),
                "sampled_windows": int(len(sampled)),
                "sample_fraction": float(len(sampled) / len(arr)),
            }
        )
    sampled_indices = np.concatenate(parts)
    sampled_indices.sort()
    return sampled_indices, pd.DataFrame(rows), True


class GRUV2Regressor(nn.Module):
    def __init__(self, n_dynamic: int, n_sites: int, site_embedding_dim: int, metadata_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(n_dynamic, GRU_HIDDEN_SIZE, num_layers=GRU_NUM_LAYERS, batch_first=True, dropout=0.0)
        self.site_embedding = nn.Embedding(n_sites, site_embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN_SIZE + site_embedding_dim + metadata_dim, GRU_MLP_HIDDEN),
            nn.ReLU(),
            nn.Linear(GRU_MLP_HIDDEN, 1),
        )

    def forward(self, x: torch.Tensor, site_code: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        final_hidden = hidden[-1]
        site_emb = self.site_embedding(site_code)
        return self.head(torch.cat([final_hidden, site_emb, metadata], dim=1)).squeeze(1)


def iter_gru_batches(
    indices: np.ndarray,
    arrays: dict[str, object],
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
):
    order = indices.copy()
    if shuffle:
        rng.shuffle(order)
    offsets = np.arange(SEQUENCE_LENGTH, dtype=np.int64)
    scaled = arrays["scaled_features"]
    starts = arrays["start_pos"]
    y = arrays["target_values"]
    site_codes = arrays["site_codes"]
    meta = arrays["metadata_by_site_code"]
    for lo in range(0, len(order), batch_size):
        idx = order[lo : lo + batch_size]
        site_code_np = site_codes[idx].astype("int64")
        pos = starts[idx, None] + offsets[None, :]
        yield (
            torch.from_numpy(scaled[pos]),
            torch.from_numpy(site_code_np),
            torch.from_numpy(meta[site_code_np]),
            torch.from_numpy(y[idx].astype("float32")),
        )


def evaluate_gru(model: nn.Module, indices: np.ndarray, arrays: dict[str, object], device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss_fn = nn.MSELoss()
    loss_sum = 0.0
    n = 0
    preds = []
    ys = []
    rng = np.random.default_rng(RANDOM_SEED)
    with torch.no_grad():
        for x, site_code, meta, y in iter_gru_batches(indices, arrays, PREDICT_BATCH_SIZE, False, rng):
            x = x.to(device)
            site_code = site_code.to(device)
            meta = meta.to(device)
            y = y.to(device)
            pred = model(x, site_code, meta)
            loss = loss_fn(pred, y)
            loss_sum += float(loss.item()) * len(y)
            n += len(y)
            preds.append(pred.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
    return loss_sum / n if n else np.nan, np.concatenate(ys), np.concatenate(preds)


def apply_phase5_dynamic_scaler(hourly: pd.DataFrame) -> np.ndarray:
    info = read_json(PHASE5_GRU_DIR / "sequence100_dynamic_scaler_info.json")
    feature_matrix = hourly[DYNAMIC_FEATURES].to_numpy(dtype=np.float32)
    mean = np.array([info["mean"][feature] for feature in DYNAMIC_FEATURES], dtype=np.float32)
    scale = np.array([info["scale"][feature] for feature in DYNAMIC_FEATURES], dtype=np.float32)
    scale = np.where(scale == 0, 1.0, scale)
    scaled = np.empty_like(feature_matrix, dtype=np.float32)
    for lo in range(0, len(feature_matrix), 1_000_000):
        hi = min(lo + 1_000_000, len(feature_matrix))
        scaled[lo:hi] = ((feature_matrix[lo:hi] - mean[None, :]) / scale[None, :]).astype("float32")
    return scaled


def build_gru_arrays(v2_features: list[str]) -> tuple[pd.DataFrame, dict[str, object], np.ndarray, np.ndarray, pd.DataFrame, bool, bool]:
    log("[INFO] Loading Phase 5 sequence rows/windows for GRU100_V2")
    hourly = pd.read_parquet(PHASE5_GRU_DIR / "sequence100_hourly_rows.parquet")
    windows = pd.read_parquet(PHASE5_GRU_DIR / "sequence100_window_index.parquet")
    hourly[SITE_COL] = hourly[SITE_COL].map(normalise_site_id).astype("string")
    windows[SITE_COL] = windows[SITE_COL].map(normalise_site_id).astype("string")
    windows["target_timestamp"] = pd.to_datetime(windows["target_timestamp"], errors="coerce")
    windows[EVENT_LABEL_COL] = windows[EVENT_LABEL_COL].map(normalise_event_label).astype("string")

    site_encoding = read_json(PHASE5_GRU_DIR / "sequence100_site_id_encoding.json")
    site_map = {str(k): int(v) for k, v in site_encoding["site_to_code"].items()}
    site_embedding_dim = int(site_encoding["embedding_dim"])
    site_codes = windows[SITE_COL].astype(str).map(site_map).to_numpy(dtype=np.int64)
    sites_by_code = [site for site, _ in sorted(site_map.items(), key=lambda item: item[1])]

    meta_df = pd.read_csv(PROCESSED100_METADATA_TABLE, dtype={SITE_COL: "string"})
    meta_df[SITE_COL] = meta_df[SITE_COL].map(normalise_site_id).astype(str)
    meta_df = meta_df.set_index(SITE_COL).loc[sites_by_code]
    raw_meta = meta_df[v2_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    metadata_by_site_code = scaler.fit_transform(raw_meta).astype("float32")
    meta_scaler_info = {
        "fit_scope": "train sites only; one V2 vector per site",
        "metadata_features": v2_features,
        "n_metadata_features": len(v2_features),
        "mean": {feature: float(value) for feature, value in zip(v2_features, scaler.mean_)},
        "scale": {feature: float(value) for feature, value in zip(v2_features, scaler.scale_)},
    }
    (OUT_DIR / "gru100_v2_metadata_scaler_info.json").write_text(json.dumps(meta_scaler_info, indent=2), encoding="utf-8")

    arrays = {
        "scaled_features": apply_phase5_dynamic_scaler(hourly),
        "start_pos": windows["hourly_row_start_idx"].to_numpy(dtype=np.int64),
        "target_values": windows["target_value"].to_numpy(dtype=np.float32),
        "site_codes": site_codes,
        "site_mapping": site_map,
        "site_embedding_dim": site_embedding_dim,
        "metadata_by_site_code": metadata_by_site_code,
        "metadata_features": v2_features,
    }
    train_idx = np.where(windows[SPLIT_COL].eq("train").to_numpy())[0].astype(np.int64)
    val_idx = np.where(windows[SPLIT_COL].eq("val").to_numpy())[0].astype(np.int64)
    test_idx = np.where(windows[SPLIT_COL].eq("test").to_numpy())[0].astype(np.int64)
    train_sample, train_summary, train_sampling = sample_indices_site_balanced(windows, train_idx, MAX_TRAIN_WINDOWS_CPU, RANDOM_SEED)
    val_early, val_summary, val_sampling = sample_indices_site_balanced(windows, val_idx, EARLY_STOP_VAL_WINDOWS_CPU, RANDOM_SEED + 1)
    train_summary.to_csv(OUT_DIR / "gru100_v2_training_sample_summary.csv", index=False, encoding="utf-8-sig")
    val_summary.to_csv(OUT_DIR / "gru100_v2_early_stop_validation_sample_summary.csv", index=False, encoding="utf-8-sig")
    split_index = pd.DataFrame(
        [
            {
                "split": "train",
                "n_windows": len(train_idx),
                "sampled_for_training_or_early_stop": len(train_sample),
                "sampling_used": train_sampling,
            },
            {
                "split": "val",
                "n_windows": len(val_idx),
                "sampled_for_training_or_early_stop": len(val_early),
                "sampling_used": val_sampling,
            },
            {"split": "test", "n_windows": len(test_idx), "sampled_for_training_or_early_stop": len(test_idx), "sampling_used": False},
        ]
    )
    split_index.to_csv(OUT_DIR / "gru100_v2_window_sampling_summary.csv", index=False, encoding="utf-8-sig")
    arrays["train_idx"] = train_idx
    arrays["val_idx"] = val_idx
    arrays["test_idx"] = test_idx
    arrays["train_sample"] = train_sample
    arrays["val_early"] = val_early
    return windows, arrays, val_idx, test_idx, split_index, train_sampling, val_sampling


def train_gru100_v2(arrays: dict[str, object], device: torch.device) -> tuple[nn.Module, pd.DataFrame]:
    log("[INFO] Training GRU100_V2")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    model = GRUV2Regressor(
        n_dynamic=len(DYNAMIC_FEATURES),
        n_sites=len(arrays["site_mapping"]),
        site_embedding_dim=int(arrays["site_embedding_dim"]),
        metadata_dim=len(arrays["metadata_features"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(RANDOM_SEED)
    best_rmse = math.inf
    best_epoch = -1
    no_improve = 0
    rows = []
    model_path = OUT_DIR / "model_GRU100_V2.pt"
    for epoch in range(1, MAX_EPOCHS_CPU + 1):
        start = perf_counter()
        model.train()
        loss_sum = 0.0
        n_train = 0
        for x, site_code, meta, y in iter_gru_batches(arrays["train_sample"], arrays, TRAIN_BATCH_SIZE, True, rng):
            x = x.to(device)
            site_code = site_code.to(device)
            meta = meta.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, site_code, meta)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(y)
            n_train += len(y)
        train_loss = loss_sum / n_train
        _val_loss, y_val, pred_val = evaluate_gru(model, arrays["val_early"], arrays, device)
        metrics = metric_values(y_val, pred_val)
        val_rmse = float(metrics["rmse"])
        rows.append(
            {
                "model_id": "GRU100_V2",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_rmse": metrics["rmse"],
                "val_mae": metrics["mae"],
                "val_r2": metrics["r2"],
                "n_train_windows_epoch": n_train,
                "n_validation_windows_early_stop": len(arrays["val_early"]),
                "epoch_time_seconds": perf_counter() - start,
            }
        )
        log(f"[INFO] GRU100_V2 epoch={epoch} train_loss={train_loss:.6f} val_rmse={val_rmse:.6f} val_r2={metrics['r2']:.6f}")
        if np.isfinite(val_rmse) and val_rmse < best_rmse - 1e-7:
            best_rmse = val_rmse
            best_epoch = epoch
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "metadata_features": arrays["metadata_features"]}, model_path)
        else:
            no_improve += 1
        if no_improve >= EARLY_STOPPING_PATIENCE:
            log("[INFO] Early stopping GRU100_V2")
            break
    history = pd.DataFrame(rows)
    history.to_csv(OUT_DIR / "gru100_v2_training_history.csv", index=False, encoding="utf-8-sig")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, history


def make_gru_prediction_frame(windows: pd.DataFrame, indices: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    meta = windows.iloc[indices].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "model_id": "GRU100_V2",
            "model_name": "100-site 24h GRU with compact V2 metadata",
            SITE_COL: meta[SITE_COL].astype(str),
            "target_timestamp": meta["target_timestamp"],
            SPLIT_COL: meta[SPLIT_COL].astype(str),
            TARGET: y_true.astype("float64"),
            "y_pred": y_pred.astype("float64"),
            EVENT_COL: meta[EVENT_COL].astype("int8").to_numpy(),
            EVENT_LABEL_COL: meta[EVENT_LABEL_COL].astype(str).to_numpy(),
            EVENT_STATE_COL: meta[EVENT_STATE_COL].astype("int8").to_numpy(),
        }
    )
    out["residual"] = out[TARGET] - out["y_pred"]
    out["abs_residual"] = out["residual"].abs()
    return out


def run_gru100_v2(v2_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    windows, arrays, val_idx, test_idx, _split_index, _train_sampling, _val_sampling = build_gru_arrays(v2_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model, _history = train_gru100_v2(arrays, device)
    _loss_val, y_val, pred_val = evaluate_gru(model, val_idx, arrays, device)
    _loss_test, y_test, pred_test = evaluate_gru(model, test_idx, arrays, device)
    val_pred = make_gru_prediction_frame(windows, val_idx, y_val, pred_val)
    test_pred = make_gru_prediction_frame(windows, test_idx, y_test, pred_test)
    write_parquet(test_pred, OUT_DIR / "gru100_v2_predictions_test.parquet")
    metrics = pd.DataFrame(
        [
            prediction_metric_row("GRU100_V2", "100-site 24h GRU with compact V2 metadata", "val", val_pred, True, "windows"),
            prediction_metric_row("GRU100_V2", "100-site 24h GRU with compact V2 metadata", "test", test_pred, True, "windows"),
        ]
    )
    residual = residual_event_table("GRU100_V2", "100-site 24h GRU with compact V2 metadata", test_pred, "windows")
    low_high = low_high_table("GRU100_V2", "100-site 24h GRU with compact V2 metadata", test_pred, "windows")
    event_hit = event_level_hit_table("GRU100_V2", "100-site 24h GRU with compact V2 metadata", test_pred)
    site_metrics = site_level_table("GRU100_V2", "100-site 24h GRU with compact V2 metadata", test_pred)
    return metrics, residual, low_high, event_hit, site_metrics


def save_metric_outputs(prefix: str, metrics: pd.DataFrame, residual: pd.DataFrame, low_high: pd.DataFrame, event_hit: pd.DataFrame, site: pd.DataFrame | None = None) -> None:
    metrics.to_csv(OUT_DIR / f"{prefix}_model_metrics.csv", index=False, encoding="utf-8-sig")
    residual.to_csv(OUT_DIR / f"{prefix}_residual_event_validation.csv", index=False, encoding="utf-8-sig")
    low_high.to_csv(OUT_DIR / f"{prefix}_low_high_event_enrichment.csv", index=False, encoding="utf-8-sig")
    event_hit.to_csv(OUT_DIR / f"{prefix}_event_level_hit_rate.csv", index=False, encoding="utf-8-sig")
    if site is not None:
        site.to_csv(OUT_DIR / f"{prefix}_site_level_metrics.csv", index=False, encoding="utf-8-sig")


def extract_top(residual: pd.DataFrame, model_id: str, fraction: float = 0.05) -> float:
    row = residual[residual["model_id"].eq(model_id) & np.isclose(pd.to_numeric(residual["top_fraction"]), fraction)]
    return float(row["enrichment"].iloc[0]) if not row.empty else np.nan


def extract_hit(event_hit: pd.DataFrame, model_id: str, fraction: float = 0.05) -> float:
    row = event_hit[event_hit["model_id"].eq(model_id) & np.isclose(pd.to_numeric(event_hit["top_fraction"]), fraction)]
    if "event_group" in row.columns:
        row = row[row["event_group"].eq("all")]
    return float(row["event_level_hit_rate"].iloc[0]) if not row.empty else np.nan


def extract_low_high(low_high: pd.DataFrame, model_id: str, label: str, fraction: float = 0.05) -> float:
    value_col = "label_enrichment" if "label_enrichment" in low_high.columns else "enrichment"
    row = low_high[
        low_high["model_id"].eq(model_id)
        & np.isclose(pd.to_numeric(low_high["top_fraction"]), fraction)
        & low_high["event_label"].str.contains(label, case=False, na=False)
    ]
    return float(row[value_col].iloc[0]) if not row.empty else np.nan


def final_row(
    model_id: str,
    site_count: int,
    model_type: str,
    metadata_set: str,
    metrics: pd.DataFrame,
    residual: pd.DataFrame,
    low_high: pd.DataFrame,
    event_hit: pd.DataFrame,
    role: str,
    interpretation: str,
) -> dict[str, object]:
    test = metrics[metrics["model_id"].eq(model_id) & metrics["split"].eq("test")]
    if test.empty:
        return {
            "model_id": model_id,
            "site_count": site_count,
            "model_type": model_type,
            "metadata_set": metadata_set,
            "test_rmse": np.nan,
            "test_mae": np.nan,
            "test_r2": np.nan,
            "top1_event_enrichment": np.nan,
            "top5_event_enrichment": np.nan,
            "top10_event_enrichment": np.nan,
            "event_level_top5_hit_rate": np.nan,
            "low_pH_top5_enrichment": np.nan,
            "high_pH_top5_enrichment": np.nan,
            "model_role": role,
            "interpretation": interpretation,
        }
    row = test.iloc[0]
    return {
        "model_id": model_id,
        "site_count": site_count,
        "model_type": model_type,
        "metadata_set": metadata_set,
        "test_rmse": float(row["rmse"]),
        "test_mae": float(row["mae"]),
        "test_r2": float(row["r2"]),
        "top1_event_enrichment": extract_top(residual, model_id, 0.01),
        "top5_event_enrichment": extract_top(residual, model_id, 0.05),
        "top10_event_enrichment": extract_top(residual, model_id, 0.10),
        "event_level_top5_hit_rate": extract_hit(event_hit, model_id, 0.05),
        "low_pH_top5_enrichment": extract_low_high(low_high, model_id, "low", 0.05),
        "high_pH_top5_enrichment": extract_low_high(low_high, model_id, "high", 0.05),
        "model_role": role,
        "interpretation": interpretation,
    }


def load_prev_hgb50_rows() -> list[dict[str, object]]:
    main = pd.read_csv(FULL50_SUMMARY_DIR / "full50_m0_m1_m2_main_comparison.csv")
    rows = []
    for _, r in main.iterrows():
        rows.append(
            {
                "model_id": r["model_id"],
                "site_count": 50,
                "model_type": "HGB",
                "metadata_set": "none" if str(r["metadata_used"]).lower() == "no" else ("full" if str(r["metadata_used"]).lower() == "yes" else "none"),
                "test_rmse": float(r["test_rmse"]),
                "test_mae": float(r["test_mae"]),
                "test_r2": float(r["test_r2"]),
                "top1_event_enrichment": float(r["top1_event_enrichment"]),
                "top5_event_enrichment": float(r["top5_event_enrichment"]),
                "top10_event_enrichment": float(r["top10_event_enrichment"]),
                "event_level_top5_hit_rate": float(r["event_level_top5_hit_rate"]),
                "low_pH_top5_enrichment": float(r["low_pH_top5_enrichment"]),
                "high_pH_top5_enrichment": float(r["high_pH_top5_enrichment"]),
                "model_role": r["model_role"],
                "interpretation": r["interpretation"],
            }
        )
    return rows


def load_prev_hgb200_rows() -> list[dict[str, object]]:
    metrics = pd.read_csv(HGB200_PREV_DIR / "model_comparison_metrics.csv")
    residual = pd.read_csv(HGB200_PREV_DIR / "residual_event_validation.csv")
    low_high = pd.read_csv(HGB200_PREV_DIR / "low_high_event_enrichment.csv")
    event_hit = pd.read_csv(HGB200_PREV_DIR / "event_level_hit_rate.csv")
    return [
        final_row("M200_A_no_metadata", 200, "HGB", "none", metrics, residual, low_high, event_hit, "200-site no-metadata HGB baseline", "previous no-metadata reference"),
        final_row("M200_B_with_metadata", 200, "HGB", "full", metrics, residual, low_high, event_hit, "200-site full metadata HGB reference", "previous full metadata reference"),
    ]


def load_prev_gru_lstm_rows() -> list[dict[str, object]]:
    rows = []
    metrics = pd.read_csv(PHASE5_GRU_DIR / "sequence_model_comparison_metrics.csv")
    residual = pd.read_csv(PHASE5_GRU_DIR / "sequence_residual_event_validation.csv")
    low_high = pd.read_csv(PHASE5_GRU_DIR / "sequence_low_high_event_enrichment.csv")
    event_hit = pd.read_csv(PHASE5_GRU_DIR / "sequence_event_level_hit_rate.csv")
    rows.append(final_row("G100_A", 100, "GRU", "none", metrics, residual, low_high, event_hit, "100-site GRU no-metadata baseline", "previous no-metadata GRU reference"))
    rows.append(final_row("G100_B", 100, "GRU", "full", metrics, residual, low_high, event_hit, "100-site GRU full metadata reference", "previous full metadata GRU reference"))
    if (PHASE5B_LSTM_DIR / "lstm100_model_metrics.csv").exists():
        lm = pd.read_csv(PHASE5B_LSTM_DIR / "lstm100_model_metrics.csv")
        lr = pd.read_csv(PHASE5B_LSTM_DIR / "lstm100_residual_event_validation.csv")
        ll = pd.read_csv(PHASE5B_LSTM_DIR / "lstm100_low_high_event_enrichment.csv")
        lh = pd.read_csv(PHASE5B_LSTM_DIR / "lstm100_event_level_hit_rate.csv")
        rows.append(final_row("L100_B", 100, "LSTM", "full", lm, lr, ll, lh, "100-site LSTM full metadata sensitivity", "previous LSTM sensitivity reference"))
    return rows


def build_final_comparison(
    hgb50: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    hgb200: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    gru: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    rows.extend(load_prev_hgb50_rows())
    rows.extend(load_prev_hgb200_rows())
    rows.extend(load_prev_gru_lstm_rows())
    h50m, h50r, h50l, h50h = hgb50
    h200m, h200r, h200l, h200h = hgb200
    grum, grur, grul, gruh = gru
    rows.append(final_row("HGB50_V2", 50, "HGB", "V2", h50m, h50r, h50l, h50h, "50-site compact V2 HGB metadata ablation", "new V2 metadata rerun"))
    rows.append(final_row("HGB200_V2", 200, "HGB", "V2", h200m, h200r, h200l, h200h, "200-site compact V2 HGB metadata ablation", "new V2 metadata rerun"))
    rows.append(final_row("GRU100_V2", 100, "GRU", "V2", grum, grur, grul, gruh, "100-site compact V2 GRU metadata ablation", "new V2 metadata rerun"))
    final = pd.DataFrame(rows)
    final.to_csv(OUT_DIR / "v2_final_model_comparison.csv", index=False, encoding="utf-8-sig")

    ablation_rows = []
    scopes = [
        ("HGB50", "M1_no_metadata", "M2_with_metadata", "HGB50_V2"),
        ("HGB200", "M200_A_no_metadata", "M200_B_with_metadata", "HGB200_V2"),
        ("GRU100", "G100_A", "G100_B", "GRU100_V2"),
    ]
    idx = final.set_index("model_id")
    for scope, no_id, full_id, v2_id in scopes:
        no = idx.loc[no_id]
        full = idx.loc[full_id]
        v2 = idx.loc[v2_id]
        if v2["test_rmse"] <= full["test_rmse"] * 1.02:
            interp = "V2 preserves most or all full metadata prediction benefit with better interpretability."
        elif v2["test_rmse"] < no["test_rmse"]:
            interp = "V2 is better than no metadata but loses some full metadata prediction benefit."
        else:
            interp = "V2 does not improve over no metadata; broader site-profile fields may be driving metadata gains."
        ablation_rows.append(
            {
                "comparison_scope": scope,
                "no_metadata_rmse": no["test_rmse"],
                "full_metadata_rmse": full["test_rmse"],
                "v2_metadata_rmse": v2["test_rmse"],
                "v2_vs_no_metadata_rmse_delta": v2["test_rmse"] - no["test_rmse"],
                "v2_vs_full_metadata_rmse_delta": v2["test_rmse"] - full["test_rmse"],
                "no_metadata_top5_enrichment": no["top5_event_enrichment"],
                "full_metadata_top5_enrichment": full["top5_event_enrichment"],
                "v2_metadata_top5_enrichment": v2["top5_event_enrichment"],
                "v2_vs_no_metadata_top5_delta": v2["top5_event_enrichment"] - no["top5_event_enrichment"],
                "v2_vs_full_metadata_top5_delta": v2["top5_event_enrichment"] - full["top5_event_enrichment"],
                "main_interpretation": interp,
            }
        )
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(OUT_DIR / "v2_metadata_ablation_summary.csv", index=False, encoding="utf-8-sig")
    return final, ablation


def create_figures(final: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    def pred_fig(scope: str, model_ids: list[str], filename: str) -> None:
        data = final[final["model_id"].isin(model_ids)].set_index("model_id").loc[model_ids].reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].bar(data["metadata_set"], data["test_rmse"], color=["#4C78A8", "#F58518", "#54A24B"][: len(data)])
        axes[0].set_ylabel("Test RMSE")
        axes[0].set_title(scope)
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].bar(data["metadata_set"], data["test_r2"], color=["#4C78A8", "#F58518", "#54A24B"][: len(data)])
        axes[1].set_ylabel("Test R2")
        axes[1].grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=300)
        plt.close(fig)

    pred_fig("HGB50", ["M1_no_metadata", "M2_with_metadata", "HGB50_V2"], "v2_hgb50_prediction_comparison.png")
    pred_fig("HGB200", ["M200_A_no_metadata", "M200_B_with_metadata", "HGB200_V2"], "v2_hgb200_prediction_comparison.png")
    pred_fig("GRU100", ["G100_A", "G100_B", "GRU100_V2"], "v2_gru_prediction_comparison.png")

    key = final[final["model_id"].isin(["M1_no_metadata", "M2_with_metadata", "HGB50_V2", "M200_A_no_metadata", "M200_B_with_metadata", "HGB200_V2", "G100_A", "G100_B", "GRU100_V2"])].copy()
    key["label"] = key["model_id"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(key["label"], key["top5_event_enrichment"], color="#4C78A8")
    ax.axhline(1, color="#111111", linestyle="--", linewidth=1)
    ax.set_ylabel("Top 5% event enrichment")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v2_top5_enrichment_comparison.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(key["label"], key["event_level_top5_hit_rate"], color="#F58518")
    ax.set_ylabel("Event-level top 5% hit rate")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v2_event_hit_rate_comparison.png", dpi=300)
    plt.close(fig)

    low_high = key[["model_id", "low_pH_top5_enrichment", "high_pH_top5_enrichment"]].set_index("model_id")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    low_high.plot(kind="bar", ax=ax, color=["#E45756", "#72B7B2"])
    ax.axhline(1, color="#111111", linestyle="--", linewidth=1)
    ax.set_ylabel("Top 5% enrichment")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v2_low_high_enrichment_comparison.png", dpi=300)
    plt.close(fig)


def write_report(final: pd.DataFrame, ablation: pd.DataFrame, selected: dict[str, list[str]]) -> tuple[str, str, str]:
    idx = final.set_index("model_id")
    hgb50_v2 = idx.loc["HGB50_V2"]
    hgb200_v2 = idx.loc["HGB200_V2"]
    gru_v2 = idx.loc["GRU100_V2"]
    preserve_count = int((ablation["v2_vs_full_metadata_rmse_delta"] <= ablation["full_metadata_rmse"].abs() * 0.02).sum())
    if preserve_count >= 2:
        main_interp = "The compact V2 metadata set preserves most of the benefit of full metadata while being substantially more interpretable. V2 is therefore preferred for dissertation-facing metadata models."
        recommended = "V2"
    elif (ablation["v2_vs_no_metadata_rmse_delta"] < 0).any():
        main_interp = "V2 gives a clearer hydrological interpretation but loses some predictive benefit compared with full metadata. It is useful as an interpretable metadata ablation, while full metadata remains an upper-bound reference."
        recommended = "V2 for interpretation; full metadata as upper-bound"
    else:
        main_interp = "The predictive gain from metadata appears to depend on broader site-profile features rather than only compact hydrological descriptors."
        recommended = "full metadata"
    next_step = "Prepare final dissertation result tables, figures, and case studies."

    report = f"""# V2 Metadata Re-Run Report

## 1. Purpose

This phase replaces the broad full metadata set with a compact, hydrologically interpretable V2 metadata set. Hyperparameters were not tuned; the corresponding previous HGB/GRU settings were reused.

## 2. V2 Metadata Definition

V2 = core official hydrological metadata + selected train-only exogenous site-background summaries.

Selected features by scope:

- HGB50: {len(selected['hgb50'])} features
- HGB200: {len(selected['hgb200'])} features
- GRU100: {len(selected['gru100'])} features

Exact feature lists are saved in `v2_metadata_feature_definition.json`, and availability/mapping details are saved in `v2_metadata_feature_availability.csv`.

## 3. Leakage Control

- No pH-derived metadata was used.
- No event-derived metadata was used.
- No residual, prediction, or model-metric fields were used.
- No validation/test statistics were used.
- Train-derived metadata uses only train rows and only exogenous variables.
- Event labels were used only after prediction for residual-event validation.

## 4. HGB50 V2 Result

- Test RMSE: {hgb50_v2['test_rmse']:.6f}
- Test R2: {hgb50_v2['test_r2']:.6f}
- Top 5% enrichment: {hgb50_v2['top5_event_enrichment']:.6f}
- Event-level top 5% hit rate: {hgb50_v2['event_level_top5_hit_rate']:.6f}
- Low/high top 5% enrichment: {hgb50_v2['low_pH_top5_enrichment']:.6f} / {hgb50_v2['high_pH_top5_enrichment']:.6f}

## 5. HGB200 V2 Result

- Test RMSE: {hgb200_v2['test_rmse']:.6f}
- Test R2: {hgb200_v2['test_r2']:.6f}
- Top 5% enrichment: {hgb200_v2['top5_event_enrichment']:.6f}
- Event-level top 5% hit rate: {hgb200_v2['event_level_top5_hit_rate']:.6f}
- Low/high top 5% enrichment: {hgb200_v2['low_pH_top5_enrichment']:.6f} / {hgb200_v2['high_pH_top5_enrichment']:.6f}

## 6. GRU100 V2 Result

- Test RMSE: {gru_v2['test_rmse']:.6f}
- Test R2: {gru_v2['test_r2']:.6f}
- Top 5% enrichment: {gru_v2['top5_event_enrichment']:.6f}
- Event-level top 5% hit rate: {gru_v2['event_level_top5_hit_rate']:.6f}
- Low/high top 5% enrichment: {gru_v2['low_pH_top5_enrichment']:.6f} / {gru_v2['high_pH_top5_enrichment']:.6f}

## 7. Optional LSTM100 V2 Result

LSTM100_V2 skipped because LSTM full metadata did not outperform GRU and this task prioritises framework completion over tuning.

## 8. Final Comparison

{dataframe_to_markdown(final)}

## 9. Ablation Summary

{dataframe_to_markdown(ablation)}

## 10. Main Interpretation

{main_interp}

## 11. Recommended Final Model Roles

- Non-event HGB = residual-detection reference.
- 200-site HGB V2 or full metadata = tabular prediction model, depending on V2/full trade-off.
- 100-site GRU V2 or full metadata = temporal sequence model, depending on V2/full trade-off.
- LSTM = sensitivity only.

## 12. Next Step

{next_step}
"""
    (OUT_DIR / "phase_v2_metadata_rerun_report.md").write_text(report, encoding="utf-8")
    return main_interp, recommended, next_step


def write_optional_lstm_placeholders() -> None:
    msg = "LSTM100_V2 skipped because LSTM full metadata did not outperform GRU and this task prioritises framework completion over tuning."
    placeholder = pd.DataFrame([{"model_id": "LSTM100_V2", "status": "skipped", "reason": msg}])
    for name in [
        "optional_lstm100_v2_model_metrics.csv",
        "optional_lstm100_v2_residual_event_validation.csv",
        "optional_lstm100_v2_low_high_event_enrichment.csv",
        "optional_lstm100_v2_event_level_hit_rate.csv",
        "optional_lstm100_v2_predictions_test.parquet",
    ]:
        if name.endswith(".csv"):
            placeholder.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")
    (OUT_DIR / "optional_lstm100_v2_skipped.txt").write_text(msg, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    selected = build_v2_feature_definitions()

    hgb50_metrics, hgb50_residual, hgb50_low_high, hgb50_hit, hgb50_site = run_hgb_v2(
        scope="hgb50",
        model_id="HGB50_V2",
        model_name="50-site all-data HGB with compact V2 metadata",
        path=HGB50_PATH,
        v2_features=selected["hgb50"],
        test_prediction_name="hgb50_v2_predictions_test.parquet",
    )
    save_metric_outputs("hgb50_v2", hgb50_metrics, hgb50_residual, hgb50_low_high, hgb50_hit, hgb50_site)

    hgb200_metrics, hgb200_residual, hgb200_low_high, hgb200_hit, hgb200_site = run_hgb_v2(
        scope="hgb200",
        model_id="HGB200_V2",
        model_name="200-site all-data HGB with compact V2 metadata",
        path=HGB200_PATH,
        v2_features=selected["hgb200"],
        test_prediction_name="hgb200_v2_predictions_test.parquet",
    )
    save_metric_outputs("hgb200_v2", hgb200_metrics, hgb200_residual, hgb200_low_high, hgb200_hit, hgb200_site)

    gru_metrics, gru_residual, gru_low_high, gru_hit, gru_site = run_gru100_v2(selected["gru100"])
    save_metric_outputs("gru100_v2", gru_metrics, gru_residual, gru_low_high, gru_hit, gru_site)

    write_optional_lstm_placeholders()
    final, ablation = build_final_comparison(
        (hgb50_metrics, hgb50_residual, hgb50_low_high, hgb50_hit),
        (hgb200_metrics, hgb200_residual, hgb200_low_high, hgb200_hit),
        (gru_metrics, gru_residual, gru_low_high, gru_hit),
    )
    create_figures(final)
    interpretation, recommended, next_step = write_report(final, ablation, selected)

    h50 = final.set_index("model_id").loc["HGB50_V2"]
    h200 = final.set_index("model_id").loc["HGB200_V2"]
    gru = final.set_index("model_id").loc["GRU100_V2"]
    preserve = "yes" if (ablation["v2_vs_full_metadata_rmse_delta"] <= ablation["full_metadata_rmse"].abs() * 0.02).sum() >= 2 else "partial"
    final_text = f"""V2 METADATA RE-RUN COMPLETE

Output folder:
{rel(OUT_DIR)}/

V2 metadata features used:
HGB50={len(selected['hgb50'])}; HGB200={len(selected['hgb200'])}; GRU100={len(selected['gru100'])}

HGB50_V2 completed:
yes

HGB200_V2 completed:
yes

GRU100_V2 completed:
yes

LSTM100_V2 completed:
skipped

Key results:

HGB50_V2 test RMSE:
{h50['test_rmse']:.6f}

HGB50_V2 top5 enrichment:
{h50['top5_event_enrichment']:.6f}

HGB200_V2 test RMSE:
{h200['test_rmse']:.6f}

HGB200_V2 top5 enrichment:
{h200['top5_event_enrichment']:.6f}

GRU100_V2 test RMSE:
{gru['test_rmse']:.6f}

GRU100_V2 top5 enrichment:
{gru['top5_event_enrichment']:.6f}

Does V2 preserve most of full metadata prediction benefit:
{preserve}

Does V2 improve interpretability:
yes

Recommended final metadata set:
{recommended}

Next recommended task:
{next_step}

Elapsed seconds:
{perf_counter() - start:.1f}
"""
    print(final_text)
    (OUT_DIR / "v2_final_console_summary.txt").write_text(final_text, encoding="utf-8")
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    with tee_stdout(OUT_DIR / "v2_metadata_rerun_log.txt"):
        main()
