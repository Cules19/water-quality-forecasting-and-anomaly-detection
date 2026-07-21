from __future__ import annotations

import json
import math
import os
import random
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
from torch import nn


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
ANALYSIS_DIR = ROOT / "analysis_outputs"
OUT_DIR = ANALYSIS_DIR / "ph_sequence100_lstm_metadata_v1"
PHASE5_DIR = ANALYSIS_DIR / "ph_sequence100_gru_lstm_metadata_v1"
INPUT_DIR = ANALYSIS_DIR / "ph_processed100_step4_0_4_1_v1"
METADATA_PARQUET = INPUT_DIR / "processed100_with_event_state_metadata.parquet"
METADATA_FEATURE_SPEC = INPUT_DIR / "processed100_metadata_feature_set.json"

TARGET = "00400"
SITE_COL = "site_id"
TIME_COL = "timestamp"
SPLIT_COL = "split"
EVENT_COL = "in_ph_event"
EVENT_LABEL_COL = "ph_event_label"
EVENT_STATE_COL = "event_state"
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
TOP_FRACTIONS = [0.01, 0.05, 0.10]

TRAIN_BATCH_SIZE = 2048
PREDICT_BATCH_SIZE = 4096
MAX_TRAIN_WINDOWS_CPU = 1_000_000
EARLY_STOP_VAL_WINDOWS_CPU = 500_000
MAX_EPOCHS_CPU = 8
EARLY_STOPPING_PATIENCE = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_SIZE = 64
NUM_LAYERS = 1
SITE_EMBEDDING_DIM = 16
MLP_HIDDEN_SIZE = 64
DROPOUT = 0.0

MODEL_ID = "L100_B"
MODEL_NAME = "L100_B_lstm_with_metadata"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message: str) -> None:
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_stdout(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, handle)
        sys.stderr = Tee(original_stderr, handle)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def log(message: str) -> None:
    print(message, flush=True)


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
    if "mixed" in text or "unknown" in text:
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


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_inputs() -> None:
    required = [
        PHASE5_DIR / "sequence100_hourly_rows.parquet",
        PHASE5_DIR / "sequence100_window_index.parquet",
        PHASE5_DIR / "sequence100_hourly_dataset_summary.csv",
        PHASE5_DIR / "sequence100_window_dataset_summary.csv",
        PHASE5_DIR / "sequence100_dynamic_scaler_info.json",
        PHASE5_DIR / "sequence100_metadata_scaler_info.json",
        PHASE5_DIR / "sequence100_site_id_encoding.json",
        PHASE5_DIR / "sequence_model_comparison_metrics.csv",
        PHASE5_DIR / "sequence_residual_event_validation.csv",
        PHASE5_DIR / "sequence_low_high_event_enrichment.csv",
        PHASE5_DIR / "sequence_event_level_hit_rate.csv",
        PHASE5_DIR / "prediction_val_G100_B_gru_with_metadata.parquet",
        PHASE5_DIR / "prediction_test_G100_B_gru_with_metadata.parquet",
        METADATA_PARQUET,
        METADATA_FEATURE_SPEC,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Phase 5B inputs: {missing}")


def copy_reused_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly_summary = pd.read_csv(PHASE5_DIR / "sequence100_hourly_dataset_summary.csv")
    window_summary = pd.read_csv(PHASE5_DIR / "sequence100_window_dataset_summary.csv")
    hourly_summary["phase5b_action"] = "reused_phase5_hourly_rows"
    hourly_summary["source_file"] = str((PHASE5_DIR / "sequence100_hourly_rows.parquet").relative_to(ROOT))
    window_summary["phase5b_action"] = "reused_phase5_window_index"
    window_summary["source_file"] = str((PHASE5_DIR / "sequence100_window_index.parquet").relative_to(ROOT))
    hourly_summary.to_csv(OUT_DIR / "lstm100_hourly_dataset_summary.csv", index=False, encoding="utf-8-sig")
    window_summary.to_csv(OUT_DIR / "lstm100_window_dataset_summary.csv", index=False, encoding="utf-8-sig")
    return hourly_summary, window_summary


def load_hourly_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    log("[INFO] Loading reusable Phase 5 hourly rows and window index")
    hourly = pd.read_parquet(PHASE5_DIR / "sequence100_hourly_rows.parquet")
    windows = pd.read_parquet(PHASE5_DIR / "sequence100_window_index.parquet")
    hourly[SITE_COL] = hourly[SITE_COL].map(normalise_site_id).astype("string")
    hourly[TIME_COL] = pd.to_datetime(hourly[TIME_COL], errors="coerce")
    windows[SITE_COL] = windows[SITE_COL].map(normalise_site_id).astype("string")
    windows["target_timestamp"] = pd.to_datetime(windows["target_timestamp"], errors="coerce")
    windows[SPLIT_COL] = windows[SPLIT_COL].astype("string")
    windows[EVENT_LABEL_COL] = windows[EVENT_LABEL_COL].map(normalise_event_label).astype("string")

    row_idx = hourly["hourly_row_idx"].to_numpy(dtype="int64")
    if not np.array_equal(row_idx, np.arange(len(hourly), dtype="int64")):
        raise RuntimeError("Phase 5 hourly_row_idx is not contiguous.")
    if TARGET in DYNAMIC_FEATURES:
        raise RuntimeError("Target 00400 cannot be a dynamic input feature.")
    forbidden = {TARGET, EVENT_COL, EVENT_LABEL_COL, EVENT_STATE_COL, "event_state_name", "sample_weight_first_version"}
    used_forbidden = sorted(forbidden.intersection(DYNAMIC_FEATURES))
    if used_forbidden:
        raise RuntimeError(f"Forbidden dynamic features detected: {used_forbidden}")
    return hourly, windows


def sample_indices_site_balanced(
    windows: pd.DataFrame,
    indices: np.ndarray,
    cap: int,
    random_seed: int,
    label: str,
) -> tuple[np.ndarray, pd.DataFrame, bool]:
    if len(indices) <= cap:
        summary = (
            windows.iloc[indices]
            .groupby(SITE_COL, observed=True)
            .size()
            .reset_index(name=f"available_{label}_windows")
        )
        summary[f"sampled_{label}_windows"] = summary[f"available_{label}_windows"]
        summary["sample_fraction"] = 1.0
        return indices.copy(), summary, False

    rng = np.random.default_rng(random_seed)
    meta = windows.iloc[indices][[SITE_COL]].copy()
    meta["_row_index"] = indices
    groups = [(str(site), group["_row_index"].to_numpy(dtype="int64")) for site, group in meta.groupby(SITE_COL, sort=True)]
    total = len(indices)
    desired = []
    for site, arr in groups:
        exact = len(arr) / total * cap
        desired.append([site, arr, int(math.floor(exact)), exact - math.floor(exact)])
    allocated = sum(item[2] for item in desired)
    remaining = cap - allocated
    for item in sorted(desired, key=lambda item: item[3], reverse=True):
        if remaining <= 0:
            break
        if item[2] < len(item[1]):
            item[2] += 1
            remaining -= 1

    parts = []
    rows = []
    for site, arr, n_take, _frac in desired:
        n_take = min(max(1, n_take), len(arr))
        sampled = rng.choice(arr, size=n_take, replace=False)
        sampled.sort()
        parts.append(sampled)
        rows.append(
            {
                SITE_COL: site,
                f"available_{label}_windows": int(len(arr)),
                f"sampled_{label}_windows": int(len(sampled)),
                "sample_fraction": float(len(sampled) / len(arr)),
            }
        )
    sampled_indices = np.concatenate(parts)
    sampled_indices.sort()
    return sampled_indices, pd.DataFrame(rows), True


def apply_dynamic_scaler(feature_matrix: np.ndarray, scaler_info: dict[str, object]) -> np.ndarray:
    log("[INFO] Applying reusable Phase 5 dynamic scaler")
    mean = np.array([scaler_info["mean"][feature] for feature in DYNAMIC_FEATURES], dtype="float32")
    scale = np.array([scaler_info["scale"][feature] for feature in DYNAMIC_FEATURES], dtype="float32")
    scale = np.where(scale == 0, 1.0, scale).astype("float32")
    scaled = np.empty_like(feature_matrix, dtype="float32")
    row_chunk = 1_000_000
    for lo in range(0, len(feature_matrix), row_chunk):
        hi = min(lo + row_chunk, len(feature_matrix))
        scaled[lo:hi] = ((feature_matrix[lo:hi] - mean[None, :]) / scale[None, :]).astype("float32")
    if not np.isfinite(scaled).all():
        raise RuntimeError("Non-finite dynamic values after scaling.")
    out_info = dict(scaler_info)
    out_info["phase5b_action"] = "reused_phase5_dynamic_scaler_mean_scale"
    out_info["source_file"] = str((PHASE5_DIR / "sequence100_dynamic_scaler_info.json").relative_to(ROOT))
    (OUT_DIR / "lstm100_dynamic_scaler_info.json").write_text(json.dumps(out_info, indent=2), encoding="utf-8")
    return scaled


def extract_metadata(metadata_features: list[str], sites_by_code: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    log("[INFO] Extracting static metadata rows from metadata parquet")
    required = set(sites_by_code)
    found: dict[str, dict[str, object]] = {}
    pf = pq.ParquetFile(METADATA_PARQUET)
    columns = [SITE_COL, *metadata_features]
    batches_scanned = 0
    rows_scanned = 0
    for batch in pf.iter_batches(batch_size=250_000, columns=columns):
        batches_scanned += 1
        df = batch.to_pandas()
        rows_scanned += len(df)
        df[SITE_COL] = df[SITE_COL].map(normalise_site_id).astype(str)
        df = df[df[SITE_COL].isin(required)]
        if df.empty:
            continue
        for _, row in df.drop_duplicates(SITE_COL, keep="first").iterrows():
            site_id = str(row[SITE_COL])
            if site_id not in found:
                found[site_id] = {col: row[col] for col in metadata_features}
        if len(found) == len(required):
            break
    missing = sorted(required - set(found))
    if missing:
        raise RuntimeError(f"Missing metadata rows for sites: {missing[:10]}")
    meta = pd.DataFrame([{SITE_COL: site, **found[site]} for site in sites_by_code])
    for col in metadata_features:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    info = {
        "metadata_source": str(METADATA_PARQUET.relative_to(ROOT)),
        "metadata_extraction_strategy": "first static row per site, matching Phase 5",
        "metadata_batches_scanned": int(batches_scanned),
        "metadata_rows_scanned": int(rows_scanned),
        "metadata_sites_extracted": int(len(meta)),
    }
    return meta, info


def scale_metadata(
    metadata_features: list[str],
    sites_by_code: list[str],
    phase5_metadata_scaler: dict[str, object],
) -> np.ndarray:
    meta, extraction_info = extract_metadata(metadata_features, sites_by_code)
    raw = meta[metadata_features].to_numpy(dtype="float32")
    mean = np.array([phase5_metadata_scaler["mean"][feature] for feature in metadata_features], dtype="float32")
    scale = np.array([phase5_metadata_scaler["scale"][feature] for feature in metadata_features], dtype="float32")
    scale = np.where(scale == 0, 1.0, scale).astype("float32")
    filled = np.where(np.isfinite(raw), raw, mean[None, :]).astype("float32")
    scaled = ((filled - mean[None, :]) / scale[None, :]).astype("float32")
    if not np.isfinite(scaled).all():
        raise RuntimeError("Non-finite metadata values after scaling.")
    out_info = dict(phase5_metadata_scaler)
    out_info.update(extraction_info)
    out_info["phase5b_action"] = "reused_phase5_metadata_scaler_mean_scale"
    out_info["source_file"] = str((PHASE5_DIR / "sequence100_metadata_scaler_info.json").relative_to(ROOT))
    (OUT_DIR / "lstm100_metadata_scaler_info.json").write_text(json.dumps(out_info, indent=2), encoding="utf-8")
    return scaled


def build_arrays(hourly: pd.DataFrame, windows: pd.DataFrame) -> dict[str, object]:
    log("[INFO] Building arrays for LSTM")
    feature_matrix = hourly[DYNAMIC_FEATURES].to_numpy(dtype="float32")
    if not np.isfinite(feature_matrix).all():
        raise RuntimeError("Non-finite dynamic features before scaling.")

    phase5_dynamic_scaler = read_json(PHASE5_DIR / "sequence100_dynamic_scaler_info.json")
    scaled_features = apply_dynamic_scaler(feature_matrix, phase5_dynamic_scaler)
    del feature_matrix

    phase5_site_encoding = read_json(PHASE5_DIR / "sequence100_site_id_encoding.json")
    site_mapping = {str(site): int(code) for site, code in phase5_site_encoding["site_to_code"].items()}
    train_sites = sorted(windows.loc[windows[SPLIT_COL].eq("train"), SITE_COL].astype(str).unique().tolist())
    if sorted(site_mapping) != train_sites:
        raise RuntimeError("Phase 5 site mapping does not match Phase 5B train sites.")
    all_sites = sorted(windows[SITE_COL].astype(str).unique().tolist())
    unknown = sorted(set(all_sites) - set(site_mapping))
    if unknown:
        raise RuntimeError(f"Val/test sites absent from train site encoding: {unknown[:10]}")
    site_codes = windows[SITE_COL].astype(str).map(site_mapping).to_numpy(dtype="int64")
    sites_by_code = [site for site, _code in sorted(site_mapping.items(), key=lambda item: item[1])]

    feature_spec = read_json(METADATA_FEATURE_SPEC)
    metadata_features = list(feature_spec.get("metadata_features_used", []))
    forbidden_keywords = ["event", "incident", "exceedance", "residual", "prediction", "00400", "ph_mean", "ph_std", "target"]
    forbidden_metadata = [col for col in metadata_features if any(keyword in col.lower() for keyword in forbidden_keywords)]
    if forbidden_metadata:
        raise RuntimeError(f"Forbidden metadata features detected: {forbidden_metadata[:20]}")
    phase5_metadata_scaler = read_json(PHASE5_DIR / "sequence100_metadata_scaler_info.json")
    metadata_by_site_code = scale_metadata(metadata_features, sites_by_code, phase5_metadata_scaler)

    site_encoding_out = {
        "site_to_code": site_mapping,
        "n_sites": int(len(site_mapping)),
        "embedding_dim": SITE_EMBEDDING_DIM,
        "encoding_fit_on": "train window sites only",
        "mapping_source": str((PHASE5_DIR / "sequence100_site_id_encoding.json").relative_to(ROOT)),
        "phase5b_action": "reused_phase5_site_mapping_with_embedding_dim_16",
        "all_val_test_sites_seen_in_train": True,
    }
    (OUT_DIR / "lstm100_site_id_encoding.json").write_text(json.dumps(site_encoding_out, indent=2), encoding="utf-8")

    return {
        "scaled_features": scaled_features,
        "start_pos": windows["hourly_row_start_idx"].to_numpy(dtype="int64"),
        "target_values": windows["target_value"].to_numpy(dtype="float32"),
        "site_codes": site_codes,
        "site_mapping": site_mapping,
        "metadata_features": metadata_features,
        "metadata_by_site_code": metadata_by_site_code,
    }


class LSTMMetadataRegressor(nn.Module):
    def __init__(
        self,
        n_dynamic_features: int,
        n_sites: int,
        metadata_dim: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_dynamic_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.0,
        )
        self.site_embedding = nn.Embedding(n_sites, SITE_EMBEDDING_DIM)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + SITE_EMBEDDING_DIM + metadata_dim, MLP_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(MLP_HIDDEN_SIZE, 1),
        )

    def forward(self, x: torch.Tensor, site_code: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        _, (hidden, _cell) = self.lstm(x)
        final_hidden = hidden[-1]
        site_embedding = self.site_embedding(site_code)
        combined = torch.cat([final_hidden, site_embedding, metadata], dim=1)
        return self.head(combined).squeeze(1)


def iter_batches(
    indices: np.ndarray,
    arrays: dict[str, object],
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
):
    order = indices.copy()
    if shuffle:
        rng.shuffle(order)
    offsets = np.arange(SEQUENCE_LENGTH, dtype="int64")
    scaled_features = arrays["scaled_features"]
    start_pos = arrays["start_pos"]
    target_values = arrays["target_values"]
    site_codes = arrays["site_codes"]
    metadata_by_site_code = arrays["metadata_by_site_code"]
    for lo in range(0, len(order), batch_size):
        batch_indices = order[lo : lo + batch_size]
        site_code_np = site_codes[batch_indices].astype("int64")
        pos = start_pos[batch_indices, None] + offsets[None, :]
        yield (
            torch.from_numpy(scaled_features[pos]),
            torch.from_numpy(site_code_np),
            torch.from_numpy(metadata_by_site_code[site_code_np]),
            torch.from_numpy(target_values[batch_indices].astype("float32")),
        )


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(y) & np.isfinite(pred)
    n = int(mask.sum())
    if n == 0:
        return {"n_windows": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan}
    y = y[mask]
    pred = pred[mask]
    residual = y - pred
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "n_windows": n,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
    }


def evaluate_model(
    model: nn.Module,
    indices: np.ndarray,
    arrays: dict[str, object],
    device: torch.device,
    batch_size: int = PREDICT_BATCH_SIZE,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    loss_sum = 0.0
    n = 0
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(RANDOM_SEED)
    with torch.no_grad():
        for x, site_code, metadata, y in iter_batches(indices, arrays, batch_size, shuffle=False, rng=rng):
            x = x.to(device)
            site_code = site_code.to(device)
            metadata = metadata.to(device)
            y = y.to(device)
            pred = model(x, site_code, metadata)
            loss = loss_fn(pred, y)
            loss_sum += float(loss.item()) * len(y)
            n += len(y)
            preds.append(pred.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
    y_arr = np.concatenate(ys) if ys else np.array([], dtype="float32")
    pred_arr = np.concatenate(preds) if preds else np.array([], dtype="float32")
    return loss_sum / n if n else np.nan, y_arr, pred_arr


def train_lstm(
    arrays: dict[str, object],
    train_sample_indices: np.ndarray,
    val_early_indices: np.ndarray,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame, dict[str, object]]:
    log("[INFO] Training L100_B LSTM with metadata")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    model = LSTMMetadataRegressor(
        n_dynamic_features=len(DYNAMIC_FEATURES),
        n_sites=len(arrays["site_mapping"]),
        metadata_dim=len(arrays["metadata_features"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    best_rmse = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    model_path = OUT_DIR / "model_L100_B_lstm_with_metadata.pt"

    for epoch in range(1, MAX_EPOCHS_CPU + 1):
        epoch_start = perf_counter()
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        for x, site_code, metadata, y in iter_batches(
            train_sample_indices,
            arrays,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            rng=rng,
        ):
            x = x.to(device)
            site_code = site_code.to(device)
            metadata = metadata.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, site_code, metadata)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(y)
            n_train += len(y)
        train_loss = train_loss_sum / n_train if n_train else np.nan
        val_loss, y_val, pred_val = evaluate_model(model, val_early_indices, arrays, device)
        val_metrics = metric_dict(y_val, pred_val)
        val_rmse = float(val_metrics["rmse"])
        elapsed = perf_counter() - epoch_start
        rows.append(
            {
                "model_id": MODEL_ID,
                "model_name": MODEL_NAME,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "n_train_windows_epoch": int(n_train),
                "n_validation_windows_early_stop": int(len(val_early_indices)),
                "epoch_time_seconds": elapsed,
                "learning_rate": LEARNING_RATE,
            }
        )
        log(
            f"[INFO] L100_B epoch={epoch} train_loss={train_loss:.6f} "
            f"val_rmse={val_rmse:.6f} val_r2={val_metrics['r2']:.6f} seconds={elapsed:.1f}"
        )
        if np.isfinite(val_rmse) and val_rmse < best_rmse - 1e-7:
            best_rmse = val_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_id": MODEL_ID,
                    "model_name": MODEL_NAME,
                    "dynamic_features": DYNAMIC_FEATURES,
                    "metadata_features": arrays["metadata_features"],
                    "site_mapping": arrays["site_mapping"],
                    "site_embedding_dim": SITE_EMBEDDING_DIM,
                },
                model_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            log("[INFO] Early stopping triggered for L100_B")
            break

    history = pd.DataFrame(rows)
    history.to_csv(OUT_DIR / "training_history_L100_B_lstm_with_metadata.csv", index=False, encoding="utf-8-sig")
    if best_epoch < 0:
        raise RuntimeError("L100_B did not produce a finite validation RMSE.")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_info = {
        "model_id": MODEL_ID,
        "best_epoch": int(best_epoch),
        "best_early_stop_val_rmse": float(best_rmse),
        "epochs_completed": int(len(history)),
        "completed": True,
    }
    return model, history, train_info


def prediction_frame(
    windows: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    meta = windows.iloc[indices].reset_index(drop=True)
    pred = pd.DataFrame(
        {
            "model_id": MODEL_ID,
            "model_name": MODEL_NAME,
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
    pred["residual"] = pred[TARGET] - pred["y_pred"]
    pred["abs_residual"] = pred["residual"].abs()
    return pred


def predict_and_save(
    model: nn.Module,
    arrays: dict[str, object],
    windows: pd.DataFrame,
    indices: np.ndarray,
    split: str,
    device: torch.device,
) -> pd.DataFrame:
    log(f"[INFO] Predicting {split} windows for L100_B")
    _loss, y_true, y_pred = evaluate_model(model, indices, arrays, device)
    pred = prediction_frame(windows, indices, y_true, y_pred)
    out_name = f"prediction_{split}_L100_B_lstm_with_metadata.parquet"
    write_parquet(pred, OUT_DIR / out_name)
    return pred


def exact_top_mask(values: np.ndarray, fraction: float) -> tuple[np.ndarray, float]:
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=bool), np.nan
    n_top = max(1, int(math.ceil(n * fraction)))
    order = np.argpartition(values, n - n_top)[n - n_top :]
    mask = np.zeros(n, dtype=bool)
    mask[order] = True
    threshold = float(np.min(values[order]))
    return mask, threshold


def compute_lstm_metrics(
    val_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[float, np.ndarray]]:
    metric_rows = []
    for split, pred in [("val", val_pred), ("test", test_pred)]:
        metrics = metric_dict(pred[TARGET].to_numpy(), pred["y_pred"].to_numpy())
        metric_rows.append(
            {
                "model_id": MODEL_ID,
                "model_name": MODEL_NAME,
                "split": split,
                "n_windows": int(metrics["n_windows"]),
                "n_sites": int(pred[SITE_COL].nunique()),
                "metadata_used": True,
                "sequence_length": SEQUENCE_LENGTH,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )
    metrics_df = pd.DataFrame(metric_rows)

    site_rows = []
    for site_id, group in test_pred.groupby(SITE_COL, sort=True, observed=True):
        metrics = metric_dict(group[TARGET].to_numpy(), group["y_pred"].to_numpy())
        labels = group[EVENT_LABEL_COL].map(compact_label)
        site_rows.append(
            {
                "model_id": MODEL_ID,
                "model_name": MODEL_NAME,
                SITE_COL: site_id,
                "n_test_windows": int(len(group)),
                "n_test_event_windows": int(group[EVENT_COL].sum()),
                "n_test_low_event_windows": int(labels.eq("low").sum()),
                "n_test_high_event_windows": int(labels.eq("high").sum()),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )
    site_metrics = pd.DataFrame(site_rows)

    residual_rows = []
    low_high_rows = []
    top_masks: dict[float, np.ndarray] = {}
    is_event = test_pred[EVENT_COL].astype(bool).to_numpy()
    labels = test_pred[EVENT_LABEL_COL].map(compact_label).to_numpy()
    abs_residual = test_pred["abs_residual"].to_numpy(dtype="float64")
    base_event_rate = float(is_event.mean()) if len(test_pred) else np.nan
    for fraction in TOP_FRACTIONS:
        mask, threshold = exact_top_mask(abs_residual, fraction)
        top_masks[fraction] = mask
        top_event = int((mask & is_event).sum())
        top_rate = float(is_event[mask].mean()) if mask.any() else np.nan
        residual_rows.append(
            {
                "model_id": MODEL_ID,
                "model_name": MODEL_NAME,
                "top_fraction": fraction,
                "n_test_windows": int(len(test_pred)),
                "n_event_windows": int(is_event.sum()),
                "base_event_rate": base_event_rate,
                "n_top_windows": int(mask.sum()),
                "n_top_event_windows": top_event,
                "top_event_rate": top_rate,
                "enrichment": top_rate / base_event_rate if base_event_rate and np.isfinite(base_event_rate) else np.nan,
                "abs_residual_threshold": threshold,
                "metadata_used": True,
            }
        )
        for label in ["low", "high"]:
            label_mask = labels == label
            base_rate = float(label_mask.mean()) if len(test_pred) else np.nan
            top_label = int((mask & label_mask).sum())
            top_label_rate = float(label_mask[mask].mean()) if mask.any() else np.nan
            low_high_rows.append(
                {
                    "model_id": MODEL_ID,
                    "model_name": MODEL_NAME,
                    "event_label": f"pH_exceedance_{label}",
                    "top_fraction": fraction,
                    "n_test_windows": int(len(test_pred)),
                    "n_label_windows": int(label_mask.sum()),
                    "base_label_rate": base_rate,
                    "n_top_windows": int(mask.sum()),
                    "n_top_label_windows": top_label,
                    "top_label_rate": top_label_rate,
                    "enrichment": top_label_rate / base_rate if base_rate and np.isfinite(base_rate) else np.nan,
                    "metadata_used": True,
                }
            )

    event_rows = []
    base = test_pred[[SITE_COL, "target_timestamp", EVENT_LABEL_COL]].copy()
    base["target_timestamp"] = pd.to_datetime(base["target_timestamp"], errors="coerce")
    base["label_compact"] = base[EVENT_LABEL_COL].map(compact_label)
    for fraction, mask in top_masks.items():
        work = base.copy()
        work["top_residual"] = mask
        events = work[work["label_compact"].isin(["low", "high"])].copy()
        if events.empty:
            groups = [("all", events), ("low", events), ("high", events)]
        else:
            events = events.sort_values([SITE_COL, "label_compact", "target_timestamp"]).reset_index(drop=True)
            same_site = events[SITE_COL].eq(events[SITE_COL].shift())
            same_label = events["label_compact"].eq(events["label_compact"].shift())
            one_hour = events["target_timestamp"].diff().eq(pd.Timedelta(hours=1))
            events["event_block_id"] = (~(same_site & same_label & one_hour)).cumsum()
            block = (
                events.groupby(["event_block_id", "label_compact"], observed=True)
                .agg(hit=("top_residual", "max"), n_rows=("top_residual", "size"))
                .reset_index()
            )
            groups = [
                ("all", block),
                ("low", block[block["label_compact"].eq("low")]),
                ("high", block[block["label_compact"].eq("high")]),
            ]
        for event_group, group in groups:
            n_blocks = int(len(group))
            n_hit = int(group["hit"].sum()) if n_blocks else 0
            event_rows.append(
                {
                    "model_id": MODEL_ID,
                    "model_name": MODEL_NAME,
                    "top_fraction": fraction,
                    "event_group": event_group,
                    "n_event_windows": n_blocks,
                    "n_hit_event_windows": n_hit,
                    "event_level_hit_rate": n_hit / n_blocks if n_blocks else np.nan,
                    "method": "approx_contiguous_target_windows_by_site_label",
                }
            )

    residual_event = pd.DataFrame(residual_rows)
    low_high = pd.DataFrame(low_high_rows)
    event_hit = pd.DataFrame(event_rows)
    metrics_df.to_csv(OUT_DIR / "lstm100_model_metrics.csv", index=False, encoding="utf-8-sig")
    residual_event.to_csv(OUT_DIR / "lstm100_residual_event_validation.csv", index=False, encoding="utf-8-sig")
    low_high.to_csv(OUT_DIR / "lstm100_low_high_event_enrichment.csv", index=False, encoding="utf-8-sig")
    event_hit.to_csv(OUT_DIR / "lstm100_event_level_hit_rate.csv", index=False, encoding="utf-8-sig")
    site_metrics.to_csv(OUT_DIR / "lstm100_site_level_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics_df, residual_event, low_high, event_hit, site_metrics, top_masks


def top5_value(df: pd.DataFrame, model_id: str, column: str, event_group: str | None = None, label: str | None = None) -> float:
    subset = df[np.isclose(pd.to_numeric(df["top_fraction"], errors="coerce"), 0.05) & df["model_id"].eq(model_id)]
    if event_group is not None and "event_group" in subset.columns:
        subset = subset[subset["event_group"].eq(event_group)]
    if label is not None and "event_label" in subset.columns:
        subset = subset[subset["event_label"].str.contains(label, case=False, na=False)]
    if subset.empty:
        return float("nan")
    return float(subset[column].iloc[0])


def build_gru_lstm_comparison(
    lstm_metrics: pd.DataFrame,
    lstm_residual: pd.DataFrame,
    lstm_low_high: pd.DataFrame,
    lstm_event_hit: pd.DataFrame,
) -> pd.DataFrame:
    gru_metrics = pd.read_csv(PHASE5_DIR / "sequence_model_comparison_metrics.csv")
    gru_residual = pd.read_csv(PHASE5_DIR / "sequence_residual_event_validation.csv")
    gru_low_high = pd.read_csv(PHASE5_DIR / "sequence_low_high_event_enrichment.csv")
    gru_event_hit = pd.read_csv(PHASE5_DIR / "sequence_event_level_hit_rate.csv")
    g_metrics = gru_metrics[gru_metrics["model_id"].eq("G100_B")]
    l_metrics = lstm_metrics[lstm_metrics["model_id"].eq("L100_B")]
    rows = []
    for split in ["val", "test"]:
        g_row = g_metrics[g_metrics[SPLIT_COL].eq(split)].iloc[0]
        l_row = l_metrics[l_metrics[SPLIT_COL].eq(split)].iloc[0]
        for metric, direction in [("rmse", "lower"), ("mae", "lower"), ("r2", "higher")]:
            g = float(g_row[metric])
            l = float(l_row[metric])
            better = "L100_B" if (l < g if direction == "lower" else l > g) else "G100_B"
            rows.append(
                {
                    "comparison_group": "prediction",
                    "split": split,
                    "metric": metric,
                    "higher_or_lower_is_better": direction,
                    "G100_B_gru_with_metadata": g,
                    "L100_B_lstm_with_metadata": l,
                    "delta_L100_B_minus_G100_B": l - g,
                    "better_model": better,
                }
            )

    for fraction in TOP_FRACTIONS:
        g_row = gru_residual[gru_residual["model_id"].eq("G100_B") & np.isclose(gru_residual["top_fraction"], fraction)].iloc[0]
        l_row = lstm_residual[np.isclose(lstm_residual["top_fraction"], fraction)].iloc[0]
        rows.append(
            {
                "comparison_group": "row_residual_event",
                "split": "test",
                "metric": f"top_{int(fraction * 100)}pct_enrichment",
                "higher_or_lower_is_better": "higher",
                "G100_B_gru_with_metadata": float(g_row["enrichment"]),
                "L100_B_lstm_with_metadata": float(l_row["enrichment"]),
                "delta_L100_B_minus_G100_B": float(l_row["enrichment"]) - float(g_row["enrichment"]),
                "better_model": "L100_B" if float(l_row["enrichment"]) > float(g_row["enrichment"]) else "G100_B",
            }
        )

    for label in ["low", "high"]:
        g_row = gru_low_high[
            gru_low_high["model_id"].eq("G100_B")
            & np.isclose(gru_low_high["top_fraction"], 0.05)
            & gru_low_high["event_label"].str.contains(label, case=False, na=False)
        ].iloc[0]
        l_row = lstm_low_high[
            np.isclose(lstm_low_high["top_fraction"], 0.05)
            & lstm_low_high["event_label"].str.contains(label, case=False, na=False)
        ].iloc[0]
        rows.append(
            {
                "comparison_group": "low_high_event",
                "split": "test",
                "metric": f"{label}_top_5pct_enrichment",
                "higher_or_lower_is_better": "higher",
                "G100_B_gru_with_metadata": float(g_row["enrichment"]),
                "L100_B_lstm_with_metadata": float(l_row["enrichment"]),
                "delta_L100_B_minus_G100_B": float(l_row["enrichment"]) - float(g_row["enrichment"]),
                "better_model": "L100_B" if float(l_row["enrichment"]) > float(g_row["enrichment"]) else "G100_B",
            }
        )

    for group in ["all", "low", "high"]:
        g_row = gru_event_hit[
            gru_event_hit["model_id"].eq("G100_B")
            & np.isclose(gru_event_hit["top_fraction"], 0.05)
            & gru_event_hit["event_group"].eq(group)
        ].iloc[0]
        l_row = lstm_event_hit[
            np.isclose(lstm_event_hit["top_fraction"], 0.05)
            & lstm_event_hit["event_group"].eq(group)
        ].iloc[0]
        rows.append(
            {
                "comparison_group": "event_level_hit",
                "split": "test",
                "metric": f"{group}_top_5pct_hit_rate",
                "higher_or_lower_is_better": "higher",
                "G100_B_gru_with_metadata": float(g_row["event_level_hit_rate"]),
                "L100_B_lstm_with_metadata": float(l_row["event_level_hit_rate"]),
                "delta_L100_B_minus_G100_B": float(l_row["event_level_hit_rate"]) - float(g_row["event_level_hit_rate"]),
                "better_model": "L100_B" if float(l_row["event_level_hit_rate"]) > float(g_row["event_level_hit_rate"]) else "G100_B",
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT_DIR / "gru_lstm_metadata_comparison.csv", index=False, encoding="utf-8-sig")
    return comparison


def save_figures(
    history: pd.DataFrame,
    lstm_metrics: pd.DataFrame,
    lstm_residual: pd.DataFrame,
    lstm_low_high: pd.DataFrame,
    lstm_event_hit: pd.DataFrame,
    lstm_test_pred: pd.DataFrame,
) -> None:
    log("[INFO] Saving Phase 5B figures")
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    gru_metrics = pd.read_csv(PHASE5_DIR / "sequence_model_comparison_metrics.csv")
    gru_residual = pd.read_csv(PHASE5_DIR / "sequence_residual_event_validation.csv")
    gru_low_high = pd.read_csv(PHASE5_DIR / "sequence_low_high_event_enrichment.csv")
    gru_event_hit = pd.read_csv(PHASE5_DIR / "sequence_event_level_hit_rate.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", color="#4C78A8")
    axes[0].set_ylabel("Train MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[1].plot(history["epoch"], history["val_rmse"], marker="o", color="#F58518")
    axes[1].set_ylabel("Validation RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "lstm100_training_curves.png", dpi=300)
    plt.close(fig)

    combined_metrics = pd.concat(
        [
            gru_metrics[gru_metrics["model_id"].eq("G100_B")],
            lstm_metrics,
        ],
        ignore_index=True,
    )
    test_metrics = combined_metrics[combined_metrics[SPLIT_COL].eq("test")].copy()
    colors = ["#4C78A8", "#F58518"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(test_metrics["model_id"], test_metrics["rmse"], color=colors)
    axes[0].set_ylabel("Test RMSE")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(test_metrics["model_id"], test_metrics["r2"], color=colors)
    axes[1].set_ylabel("Test R2")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "gru_lstm_rmse_r2_comparison.png", dpi=300)
    plt.close(fig)

    top5 = pd.concat(
        [
            gru_residual[gru_residual["model_id"].eq("G100_B") & np.isclose(gru_residual["top_fraction"], 0.05)],
            lstm_residual[np.isclose(lstm_residual["top_fraction"], 0.05)],
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(top5["model_id"], top5["enrichment"], color=colors)
    ax.axhline(1, color="#111111", linestyle="--", linewidth=1)
    ax.set_ylabel("Top 5% event enrichment")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "gru_lstm_top5_enrichment_comparison.png", dpi=300)
    plt.close(fig)

    top5_hit = pd.concat(
        [
            gru_event_hit[
                gru_event_hit["model_id"].eq("G100_B")
                & np.isclose(gru_event_hit["top_fraction"], 0.05)
                & gru_event_hit["event_group"].eq("all")
            ],
            lstm_event_hit[np.isclose(lstm_event_hit["top_fraction"], 0.05) & lstm_event_hit["event_group"].eq("all")],
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(top5_hit["model_id"], top5_hit["event_level_hit_rate"], color=colors)
    ax.set_ylabel("Top 5% event-level hit rate")
    ax.set_ylim(0, max(0.05, float(top5_hit["event_level_hit_rate"].max()) * 1.2))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "gru_lstm_event_hit_rate_comparison.png", dpi=300)
    plt.close(fig)

    top5_labels = pd.concat(
        [
            gru_low_high[gru_low_high["model_id"].eq("G100_B") & np.isclose(gru_low_high["top_fraction"], 0.05)],
            lstm_low_high[np.isclose(lstm_low_high["top_fraction"], 0.05)],
        ],
        ignore_index=True,
    )
    top5_labels["label_short"] = top5_labels["event_label"].map(lambda value: "low" if "low" in str(value).lower() else "high")
    pivot = top5_labels.pivot(index="model_id", columns="label_short", values="enrichment").reindex(["G100_B", "L100_B"])
    fig, ax = plt.subplots(figsize=(6.2, 4))
    pivot.plot(kind="bar", ax=ax, color=["#E45756", "#72B7B2"])
    ax.axhline(1, color="#111111", linestyle="--", linewidth=1)
    ax.set_ylabel("Top 5% enrichment")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "gru_lstm_low_high_enrichment_comparison.png", dpi=300)
    plt.close(fig)

    case = lstm_test_pred.copy()
    case["target_timestamp"] = pd.to_datetime(case["target_timestamp"], errors="coerce")
    event_counts = case.groupby(SITE_COL)[EVENT_COL].sum().sort_values(ascending=False)
    case_site = str(event_counts.index[0]) if not event_counts.empty else str(case[SITE_COL].iloc[0])
    site_case = case[case[SITE_COL].eq(case_site)].sort_values("target_timestamp")
    event_rows = site_case[site_case[EVENT_COL].astype(bool)]
    if not event_rows.empty:
        center = event_rows["target_timestamp"].iloc[0]
        site_case = site_case[
            (site_case["target_timestamp"] >= center - pd.Timedelta(hours=96))
            & (site_case["target_timestamp"] <= center + pd.Timedelta(hours=96))
        ]
    else:
        site_case = site_case.head(240)
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(site_case["target_timestamp"], site_case[TARGET], label="observed", color="#111111", linewidth=1.4)
    ax1.plot(site_case["target_timestamp"], site_case["y_pred"], label="LSTM predicted", color="#F58518", linewidth=1.2)
    event_points = site_case[site_case[EVENT_COL].astype(bool)]
    if not event_points.empty:
        ax1.scatter(event_points["target_timestamp"], event_points[TARGET], s=16, color="#E45756", label="event target")
    ax1.set_ylabel("Processed pH")
    ax2 = ax1.twinx()
    ax2.plot(site_case["target_timestamp"], site_case["abs_residual"], label="abs residual", color="#4C78A8", alpha=0.6)
    ax2.set_ylabel("Absolute residual")
    ax1.set_title(f"LSTM case-study site {case_site}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "lstm100_observed_predicted_case_study.png", dpi=300)
    plt.close(fig)


def write_feature_and_config(
    arrays: dict[str, object],
    train_count: int,
    train_sample_count: int,
    val_count: int,
    val_early_count: int,
    device: torch.device,
) -> None:
    feature_spec = {
        "phase": "Phase 5B LSTM100 metadata",
        "reused_phase5_sequence_dataset": True,
        "phase5_source_folder": str(PHASE5_DIR.relative_to(ROOT)),
        "metadata_source": str(METADATA_PARQUET.relative_to(ROOT)),
        "metadata_feature_spec_source": str(METADATA_FEATURE_SPEC.relative_to(ROOT)),
        "target": TARGET,
        "dynamic_sequence_features": DYNAMIC_FEATURES,
        "metadata_features": arrays["metadata_features"],
        "sequence_length_hours": SEQUENCE_LENGTH,
        "site_embedding": {"used": True, "embedding_dim": SITE_EMBEDDING_DIM},
        "uses_ph_history": False,
        "uses_event_labels_as_features": False,
        "excluded_event_features": [
            EVENT_COL,
            EVENT_LABEL_COL,
            EVENT_STATE_COL,
            "event_state_name",
            "sample_weight_first_version",
            "event counts",
            "event ratios",
            "incident fields",
            "exceedance fields",
        ],
        "excluded_target_derived_features": [
            TARGET,
            "previous pH",
            "pH lag",
            "rolling pH",
            "site pH mean",
            "site pH standard deviation",
            "residuals",
            "prediction columns",
            "model metrics",
        ],
    }
    (OUT_DIR / "lstm100_feature_spec.json").write_text(json.dumps(feature_spec, indent=2), encoding="utf-8")

    config = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "recurrent_type": "lstm",
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "random_seed": RANDOM_SEED,
        "sequence_length": SEQUENCE_LENGTH,
        "hyperparameters": {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "site_embedding_dim": SITE_EMBEDDING_DIM,
            "mlp_hidden_size": MLP_HIDDEN_SIZE,
            "dropout": DROPOUT,
            "batch_size": TRAIN_BATCH_SIZE,
            "prediction_batch_size": PREDICT_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS_CPU,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "loss": "MSE",
            "optimizer": "Adam",
        },
        "runtime_controls": {
            "original_train_windows": int(train_count),
            "sampled_train_windows": int(train_sample_count),
            "train_window_cap_cpu": int(MAX_TRAIN_WINDOWS_CPU),
            "original_val_windows": int(val_count),
            "early_stop_val_windows": int(val_early_count),
            "early_stop_val_window_cap_cpu": int(EARLY_STOP_VAL_WINDOWS_CPU),
            "full_val_predictions_saved": True,
            "full_test_predictions_saved": True,
            "gru_retrained": False,
        },
    }
    (OUT_DIR / "model_config_lstm100_metadata.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def write_report(
    hourly_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
    history: pd.DataFrame,
    lstm_metrics: pd.DataFrame,
    lstm_residual: pd.DataFrame,
    lstm_low_high: pd.DataFrame,
    lstm_event_hit: pd.DataFrame,
    comparison: pd.DataFrame,
    train_sampling_used: bool,
    val_sampling_used: bool,
) -> tuple[str, str]:
    pred_test = comparison[(comparison["comparison_group"].eq("prediction")) & (comparison["split"].eq("test"))]
    rmse_row = pred_test[pred_test["metric"].eq("rmse")].iloc[0]
    r2_row = pred_test[pred_test["metric"].eq("r2")].iloc[0]
    enrich_row = comparison[comparison["metric"].eq("top_5pct_enrichment")].iloc[0]
    hit_row = comparison[comparison["metric"].eq("all_top_5pct_hit_rate")].iloc[0]
    low_row = comparison[comparison["metric"].eq("low_top_5pct_enrichment")].iloc[0]
    high_row = comparison[comparison["metric"].eq("high_top_5pct_enrichment")].iloc[0]

    lstm_improved_rmse = rmse_row["better_model"] == "L100_B"
    lstm_improved_r2 = r2_row["better_model"] == "L100_B"
    lstm_improved_enrichment = enrich_row["better_model"] == "L100_B"
    if lstm_improved_rmse and lstm_improved_r2 and lstm_improved_enrichment:
        interpretation = "LSTM with metadata improves both prediction and residual-event alignment over the metadata GRU."
    elif lstm_improved_rmse and lstm_improved_r2 and not lstm_improved_enrichment:
        interpretation = "LSTM improves prediction over the metadata GRU but weakens residual-event alignment."
    elif not lstm_improved_rmse and not lstm_improved_r2 and lstm_improved_enrichment:
        interpretation = "LSTM weakens prediction but improves residual-event alignment, suggesting a prediction-detection trade-off."
    else:
        interpretation = "LSTM does not clearly improve over the metadata GRU; the GRU remains the stronger sequence baseline."
    next_step = (
        "Use the better of G100_B and L100_B as the sequence-model result; only proceed to FiLM/attention if the LSTM gain justifies extra complexity."
    )

    report = f"""# Phase 5B LSTM100 Metadata Report

## 1. Purpose

This phase trains one additional sequence model, `L100_B_lstm_with_metadata`, and compares it with the existing `G100_B_gru_with_metadata` from Phase 5. GRU models were not retrained.

## 2. Dataset Reuse

- Reused Phase 5 hourly rows: yes
- Reused Phase 5 24h window index: yes
- Reused Phase 5 dynamic scaler mean/scale: yes
- Reused Phase 5 metadata scaler mean/scale: yes
- Metadata source: `{METADATA_PARQUET.relative_to(ROOT)}`
- Event labels were used only for evaluation.
- pH history and `00400` were not used as sequence inputs.

Hourly summary:

{dataframe_to_markdown(hourly_summary)}

Window summary:

{dataframe_to_markdown(window_summary)}

## 3. Feature Design

- Dynamic sequence features: `{DYNAMIC_FEATURES}`
- Static metadata features: {len(read_json(OUT_DIR / "lstm100_feature_spec.json")["metadata_features"])}
- Site embedding dimension: {SITE_EMBEDDING_DIM}
- Sequence length: {SEQUENCE_LENGTH} hourly steps
- Target: `{TARGET}` at the final timestamp

## 4. Training Setup

- Model: 1-layer LSTM, hidden size {HIDDEN_SIZE}, MLP hidden size {MLP_HIDDEN_SIZE}.
- Optimizer: Adam, learning rate {LEARNING_RATE}, weight decay {WEIGHT_DECAY}.
- CPU-only runtime: {not torch.cuda.is_available()}
- Training sampling used: {train_sampling_used}
- Early-stop validation sampling used: {val_sampling_used}
- Full validation/test predictions were saved.

Training history:

{dataframe_to_markdown(history)}

## 5. LSTM Prediction Metrics

{dataframe_to_markdown(lstm_metrics)}

## 6. LSTM Residual-Event Validation

Row-level enrichment:

{dataframe_to_markdown(lstm_residual)}

Low/high enrichment:

{dataframe_to_markdown(lstm_low_high)}

Approximate event-level hit rate:

{dataframe_to_markdown(lstm_event_hit)}

## 7. GRU vs LSTM Metadata Comparison

{dataframe_to_markdown(comparison)}

- LSTM improved test RMSE over G100_B: {'yes' if lstm_improved_rmse else 'no'}
- LSTM improved test R2 over G100_B: {'yes' if lstm_improved_r2 else 'no'}
- LSTM improved top 5% residual-event enrichment over G100_B: {'yes' if lstm_improved_enrichment else 'no'}
- G100_B/L100_B test RMSE: {float(rmse_row['G100_B_gru_with_metadata']):.6f} / {float(rmse_row['L100_B_lstm_with_metadata']):.6f}
- G100_B/L100_B test R2: {float(r2_row['G100_B_gru_with_metadata']):.6f} / {float(r2_row['L100_B_lstm_with_metadata']):.6f}
- G100_B/L100_B top 5% enrichment: {float(enrich_row['G100_B_gru_with_metadata']):.6f} / {float(enrich_row['L100_B_lstm_with_metadata']):.6f}
- G100_B/L100_B event-level top 5% hit rate: {float(hit_row['G100_B_gru_with_metadata']):.6f} / {float(hit_row['L100_B_lstm_with_metadata']):.6f}
- G100_B/L100_B low-pH top 5% enrichment: {float(low_row['G100_B_gru_with_metadata']):.6f} / {float(low_row['L100_B_lstm_with_metadata']):.6f}
- G100_B/L100_B high-pH top 5% enrichment: {float(high_row['G100_B_gru_with_metadata']):.6f} / {float(high_row['L100_B_lstm_with_metadata']):.6f}

## 8. Interpretation

{interpretation}

## 9. Limitations

- The LSTM uses the same hourly aggregated 24h windows as Phase 5.
- CPU-only runtime required the same capped site-balanced train sampling strategy used in Phase 5.
- Event-level hit rate is approximated by contiguous target-window blocks within site and event label.
- The comparison is strongest against G100_B because both models use the same windows, scaler logic, metadata features, and full val/test targets.

## 10. Next Step

{next_step}
"""
    (OUT_DIR / "phase5b_lstm100_metadata_report.md").write_text(report, encoding="utf-8")
    return interpretation, next_step


def final_console(
    hourly_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    interpretation: str,
    next_step: str,
    elapsed: float,
) -> str:
    overall_hourly = hourly_summary[hourly_summary["scope"].eq("overall")].iloc[0]
    overall_windows = window_summary[window_summary["scope"].eq("overall")].iloc[0]
    pred_test = comparison[(comparison["comparison_group"].eq("prediction")) & (comparison["split"].eq("test"))]
    rmse_row = pred_test[pred_test["metric"].eq("rmse")].iloc[0]
    r2_row = pred_test[pred_test["metric"].eq("r2")].iloc[0]
    enrich_row = comparison[comparison["metric"].eq("top_5pct_enrichment")].iloc[0]
    hit_row = comparison[comparison["metric"].eq("all_top_5pct_hit_rate")].iloc[0]
    low_row = comparison[comparison["metric"].eq("low_top_5pct_enrichment")].iloc[0]
    high_row = comparison[comparison["metric"].eq("high_top_5pct_enrichment")].iloc[0]
    text = f"""PHASE 5B LSTM100 METADATA COMPLETE

Output folder:
{OUT_DIR.relative_to(ROOT).as_posix()}/

Reused Phase 5 hourly/window dataset:
yes

Hourly rows:
{int(overall_hourly['hourly_rows'])}

Valid 24h windows:
{int(overall_windows['total_valid_windows'])}

Train windows:
{int(overall_windows['train_windows'])}

Validation windows:
{int(overall_windows['val_windows'])}

Test windows:
{int(overall_windows['test_windows'])}

Models completed:
G100_B GRU with metadata: reused
L100_B LSTM with metadata: yes

G100_B test RMSE:
{float(rmse_row['G100_B_gru_with_metadata']):.6f}

L100_B test RMSE:
{float(rmse_row['L100_B_lstm_with_metadata']):.6f}

G100_B test R2:
{float(r2_row['G100_B_gru_with_metadata']):.6f}

L100_B test R2:
{float(r2_row['L100_B_lstm_with_metadata']):.6f}

G100_B top 5% enrichment:
{float(enrich_row['G100_B_gru_with_metadata']):.6f}

L100_B top 5% enrichment:
{float(enrich_row['L100_B_lstm_with_metadata']):.6f}

G100_B event-level top 5% hit rate:
{float(hit_row['G100_B_gru_with_metadata']):.6f}

L100_B event-level top 5% hit rate:
{float(hit_row['L100_B_lstm_with_metadata']):.6f}

G100_B low-pH top 5% enrichment:
{float(low_row['G100_B_gru_with_metadata']):.6f}

L100_B low-pH top 5% enrichment:
{float(low_row['L100_B_lstm_with_metadata']):.6f}

G100_B high-pH top 5% enrichment:
{float(high_row['G100_B_gru_with_metadata']):.6f}

L100_B high-pH top 5% enrichment:
{float(high_row['L100_B_lstm_with_metadata']):.6f}

LSTM improved RMSE over GRU:
{'yes' if rmse_row['better_model'] == 'L100_B' else 'no'}

LSTM improved R2 over GRU:
{'yes' if r2_row['better_model'] == 'L100_B' else 'no'}

LSTM improved residual-event enrichment over GRU:
{'yes' if enrich_row['better_model'] == 'L100_B' else 'no'}

Main interpretation:
{interpretation}

Next recommended task:
{next_step}

Elapsed seconds:
{elapsed:.1f}
"""
    (OUT_DIR / "lstm100_final_console_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return text


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    check_inputs()
    hourly_summary, window_summary = copy_reused_summaries()
    hourly, windows = load_hourly_windows()

    train_indices = np.where(windows[SPLIT_COL].eq("train").to_numpy())[0].astype("int64")
    val_indices = np.where(windows[SPLIT_COL].eq("val").to_numpy())[0].astype("int64")
    test_indices = np.where(windows[SPLIT_COL].eq("test").to_numpy())[0].astype("int64")
    train_sample_indices, train_sample_summary, train_sampling_used = sample_indices_site_balanced(
        windows, train_indices, MAX_TRAIN_WINDOWS_CPU, RANDOM_SEED, "train"
    )
    val_early_indices, val_sample_summary, val_sampling_used = sample_indices_site_balanced(
        windows, val_indices, EARLY_STOP_VAL_WINDOWS_CPU, RANDOM_SEED + 1, "val_early_stop"
    )
    train_sample_summary.to_csv(OUT_DIR / "lstm100_training_sample_summary.csv", index=False, encoding="utf-8-sig")
    val_sample_summary.to_csv(OUT_DIR / "lstm100_early_stop_validation_sample_summary.csv", index=False, encoding="utf-8-sig")

    arrays = build_arrays(hourly, windows)
    del hourly

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    write_feature_and_config(arrays, len(train_indices), len(train_sample_indices), len(val_indices), len(val_early_indices), device)

    model, history, train_info = train_lstm(arrays, train_sample_indices, val_early_indices, device)
    pd.DataFrame([train_info]).to_csv(OUT_DIR / "lstm100_model_training_completion.csv", index=False, encoding="utf-8-sig")

    val_pred = predict_and_save(model, arrays, windows, val_indices, "val", device)
    test_pred = predict_and_save(model, arrays, windows, test_indices, "test", device)
    lstm_metrics, lstm_residual, lstm_low_high, lstm_event_hit, lstm_site_metrics, top_masks = compute_lstm_metrics(val_pred, test_pred)
    comparison = build_gru_lstm_comparison(lstm_metrics, lstm_residual, lstm_low_high, lstm_event_hit)
    save_figures(history, lstm_metrics, lstm_residual, lstm_low_high, lstm_event_hit, test_pred)
    interpretation, next_step = write_report(
        hourly_summary,
        window_summary,
        history,
        lstm_metrics,
        lstm_residual,
        lstm_low_high,
        lstm_event_hit,
        comparison,
        train_sampling_used,
        val_sampling_used,
    )
    final_console(hourly_summary, window_summary, comparison, interpretation, next_step, perf_counter() - started)


if __name__ == "__main__":
    with tee_stdout(OUT_DIR / "lstm100_metadata_log.txt"):
        main()
