from __future__ import annotations

import gc
import importlib.util
import json
import math
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
OUT_DIR = ROOT / "analysis_outputs/ph_current_machine_supervisor_tasks_v1/04_nonevent_gru100"
BASE_DIR = ROOT / "analysis_outputs/ph_sequence100_gru_lstm_metadata_v1"
PHASE4_DIR = ROOT / "analysis_outputs/ph_processed100_step4_0_4_1_v1"
BASE_SCRIPT = ROOT / "pipeline/04_sequence_gru/run_sequence100_gru_lstm_metadata.py"
HOURLY_PATH = BASE_DIR / "sequence100_hourly_rows.parquet"
WINDOW_PATH = BASE_DIR / "sequence100_window_index.parquet"
BASE_PRED_PATH = BASE_DIR / "prediction_test_G100_B_gru_with_metadata.parquet"
BASE_CONFIG_PATH = BASE_DIR / "model_config_gru_lstm.json"
DYNAMIC_SCALER_PATH = BASE_DIR / "sequence100_dynamic_scaler_info.json"
METADATA_SCALER_PATH = BASE_DIR / "sequence100_metadata_scaler_info.json"
SITE_ENCODING_PATH = BASE_DIR / "sequence100_site_id_encoding.json"
METADATA_TABLE_PATH = PHASE4_DIR / "processed100_metadata_model_ready.csv"
M0_PREDICTION_METRICS_PATH = ROOT / "analysis_outputs/ph_full50_model_summary_with_m0_v1/full50_m0_m1_m2_prediction_metrics.csv"
M0_SIGNED_METRICS_PATH = ROOT / "analysis_outputs/ph_signed_residual_tail_analysis_v1/signed_tail_dissertation_ready_table.csv"
LOG_PATH = OUT_DIR / "nonevent_gru_training_log.txt"

MODEL_ID = "GRU100_FULLMETA_NONEVENT_TARGETONLY"
MODEL_NAME = "GRU100 full metadata, non-event target-only training"
CHECKPOINT_NAME = f"{MODEL_ID}.pt"
PREDICTION_NAME = f"prediction_test_{MODEL_ID}.parquet"
HISTORY_NAME = "nonevent_gru_training_history.csv"
TOP_FRACTIONS = [0.01, 0.05, 0.10]


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, message: str) -> None:
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_output(path: Path):
    with path.open("w", encoding="utf-8") as handle:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, handle), Tee(stderr, handle)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = stdout, stderr


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("sequence100_baseline", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import baseline script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT_DIR = OUT_DIR
    return module


def normalise_site_id(value: Any) -> str:
    raw = str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    return raw.zfill(8) if raw.isdigit() else raw


def compact_event_label(value: Any) -> str:
    text = str(value).lower()
    if "low" in text:
        return "low"
    if "high" in text:
        return "high"
    if "mixed" in text or "unknown" in text:
        return "mixed"
    return "normal"


def exact_top_mask(values: np.ndarray, fraction: float, largest: bool = True) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype="float64")
    n_take = max(1, int(math.ceil(len(values) * fraction)))
    order_values = values if largest else -values
    selected = np.argpartition(order_values, len(values) - n_take)[len(values) - n_take :]
    mask = np.zeros(len(values), dtype=bool)
    mask[selected] = True
    threshold = float(values[selected].min() if largest else values[selected].max())
    return mask, threshold


def load_inputs(base: Any) -> tuple[pd.DataFrame, dict[str, object]]:
    print("[INFO] Loading compact baseline window index")
    window_cols = [
        "site_id", "target_timestamp", "split", "target_value", "in_ph_event", "ph_event_label",
        "event_state", "hourly_row_start_idx",
    ]
    windows = pd.read_parquet(WINDOW_PATH, columns=window_cols)
    windows["site_id"] = windows["site_id"].map(normalise_site_id)
    windows["split"] = windows["split"].astype("category")
    windows["ph_event_label"] = windows["ph_event_label"].astype("category")
    windows["target_timestamp"] = pd.to_datetime(windows["target_timestamp"], errors="coerce")

    with DYNAMIC_SCALER_PATH.open("r", encoding="utf-8") as handle:
        dynamic_info = json.load(handle)
    dynamic_features = list(dynamic_info["dynamic_features"])
    if dynamic_features != list(base.DYNAMIC_FEATURES):
        raise RuntimeError("Baseline dynamic feature order differs from imported architecture script.")
    print("[INFO] Loading baseline hourly dynamic feature matrix")
    hourly = pd.read_parquet(HOURLY_PATH, columns=dynamic_features)
    features = hourly[dynamic_features].to_numpy(dtype="float32")
    del hourly
    means = np.array([dynamic_info["mean"][col] for col in dynamic_features], dtype="float64")
    scales = np.array([dynamic_info["scale"][col] for col in dynamic_features], dtype="float64")
    for lo in range(0, len(features), 1_000_000):
        hi = min(lo + 1_000_000, len(features))
        features[lo:hi] = (features[lo:hi] - means) / scales
    if not np.isfinite(features).all():
        raise RuntimeError("Non-finite dynamic values after applying baseline scaler.")

    site_info = json.loads(SITE_ENCODING_PATH.read_text(encoding="utf-8"))
    site_mapping = {normalise_site_id(k): int(v) for k, v in site_info["site_to_code"].items()}
    site_codes = windows["site_id"].map(site_mapping)
    if site_codes.isna().any():
        raise RuntimeError("Window site absent from the exact G100_B site encoding.")

    metadata_info = json.loads(METADATA_SCALER_PATH.read_text(encoding="utf-8"))
    metadata_features = list(metadata_info["metadata_features"])
    metadata = pd.read_csv(METADATA_TABLE_PATH, dtype={"site_id": str})
    metadata["site_id"] = metadata["site_id"].map(normalise_site_id)
    metadata = metadata.set_index("site_id").loc[list(site_mapping)].reset_index()
    raw_metadata = metadata[metadata_features].to_numpy(dtype="float32")
    meta_mean = np.array([metadata_info["mean"][col] for col in metadata_features], dtype="float64")
    meta_scale = np.array([metadata_info["scale"][col] for col in metadata_features], dtype="float64")
    raw_metadata = np.where(np.isfinite(raw_metadata), raw_metadata, meta_mean[None, :])
    scaled_metadata = ((raw_metadata - meta_mean) / meta_scale).astype("float32")
    arrays: dict[str, object] = {
        "scaled_features": features,
        "start_pos": windows["hourly_row_start_idx"].to_numpy(dtype="int64"),
        "target_values": windows["target_value"].to_numpy(dtype="float32"),
        "site_codes": site_codes.to_numpy(dtype="int64"),
        "site_mapping": site_mapping,
        "site_embedding_dim": int(site_info["embedding_dim"]),
        "metadata_by_site_code": scaled_metadata,
        "metadata_features": metadata_features,
    }
    scaler_audit = {
        "dynamic_scaler_reused": str(DYNAMIC_SCALER_PATH.relative_to(ROOT)),
        "metadata_scaler_reused": str(METADATA_SCALER_PATH.relative_to(ROOT)),
        "site_encoding_reused": str(SITE_ENCODING_PATH.relative_to(ROOT)),
        "dynamic_feature_count": len(dynamic_features),
        "metadata_feature_count": len(metadata_features),
        "site_count": len(site_mapping),
    }
    return windows, arrays | {"scaler_audit": scaler_audit}


def audit_and_sample(base: Any, windows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    train_mask = windows["split"].eq("train").to_numpy()
    val_indices = np.where(windows["split"].eq("val").to_numpy())[0].astype("int64")
    test_indices = np.where(windows["split"].eq("test").to_numpy())[0].astype("int64")
    event = pd.to_numeric(windows["in_ph_event"], errors="coerce").fillna(0).eq(1).to_numpy()
    eligible_indices = np.where(train_mask & ~event)[0].astype("int64")
    removed_indices = np.where(train_mask & event)[0].astype("int64")
    sampled_train, sample_summary, _ = base.sample_indices_site_balanced(
        windows, eligible_indices, base.MAX_TRAIN_WINDOWS_CPU, base.RANDOM_SEED, "nonevent_train"
    )
    val_early, val_summary, _ = base.sample_indices_site_balanced(
        windows, val_indices, base.EARLY_STOP_VAL_WINDOWS_CPU, base.RANDOM_SEED + 1, "val_early_stop"
    )
    sample_summary.to_csv(OUT_DIR / "nonevent_gru_train_sampling_by_site.csv", index=False)
    val_summary.to_csv(OUT_DIR / "nonevent_gru_val_sampling_by_site.csv", index=False)

    removed = windows.iloc[removed_indices]
    removed_label = removed["ph_event_label"].map(compact_event_label)
    rows = [
        {
            "scope": "overall",
            "site_id": "ALL",
            "original_train_windows": int(train_mask.sum()),
            "removed_event_target_windows": len(removed_indices),
            "retained_nonevent_train_windows": len(eligible_indices),
            "sampled_train_windows": len(sampled_train),
            "percentage_removed": 100 * len(removed_indices) / int(train_mask.sum()),
            "removed_low_pH_targets": int(removed_label.eq("low").sum()),
            "removed_high_pH_targets": int(removed_label.eq("high").sum()),
            "removed_mixed_targets": int(removed_label.eq("mixed").sum()),
            "sites_with_no_remaining_train_windows": 0,
            "validation_windows_unchanged": len(val_indices),
            "test_windows_unchanged": len(test_indices),
            "train_validation_test_site_overlap": len(
                set(windows.loc[train_mask, "site_id"])
                & set(windows.loc[windows["split"].eq("val"), "site_id"])
                & set(windows.loc[windows["split"].eq("test"), "site_id"])
            ),
        }
    ]
    for site_id, group in windows.loc[train_mask].groupby("site_id", sort=True, observed=True):
        event_site = pd.to_numeric(group["in_ph_event"], errors="coerce").fillna(0).eq(1)
        labels = group.loc[event_site, "ph_event_label"].map(compact_event_label)
        rows.append(
            {
                "scope": "site",
                "site_id": site_id,
                "original_train_windows": len(group),
                "removed_event_target_windows": int(event_site.sum()),
                "retained_nonevent_train_windows": int((~event_site).sum()),
                "sampled_train_windows": int(sample_summary.loc[sample_summary["site_id"].eq(site_id), "sampled_nonevent_train_windows"].iloc[0]),
                "percentage_removed": 100 * event_site.mean(),
                "removed_low_pH_targets": int(labels.eq("low").sum()),
                "removed_high_pH_targets": int(labels.eq("high").sum()),
                "removed_mixed_targets": int(labels.eq("mixed").sum()),
                "sites_with_no_remaining_train_windows": int((~event_site).sum() == 0),
                "validation_windows_unchanged": int((windows["split"].eq("val") & windows["site_id"].eq(site_id)).sum()),
                "test_windows_unchanged": int((windows["split"].eq("test") & windows["site_id"].eq(site_id)).sum()),
                "train_validation_test_site_overlap": True,
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT_DIR / "nonevent_gru_window_audit.csv", index=False)
    if audit.loc[audit["scope"].eq("site"), "sites_with_no_remaining_train_windows"].sum() != 0:
        raise RuntimeError("At least one site lost all non-event training windows.")
    return sampled_train, val_early, test_indices, audit


def train_and_predict(
    base: Any,
    windows: pd.DataFrame,
    arrays: dict[str, object],
    train_indices: np.ndarray,
    val_early_indices: np.ndarray,
    test_indices: np.ndarray,
) -> pd.DataFrame:
    spec = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "recurrent_type": "gru",
        "metadata_used": True,
        "history_file": HISTORY_NAME,
        "model_file": CHECKPOINT_NAME,
        "val_prediction_file": "unused_validation_predictions.parquet",
        "test_prediction_file": PREDICTION_NAME,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model, _history, info = base.train_sequence_model(spec, arrays, train_indices, val_early_indices, device)
    print(f"[INFO] Training complete at epoch {info['best_epoch']}; predicting unchanged full test set")
    _loss, y_true, y_pred = base.evaluate_model(model, test_indices, arrays, True, device, base.PREDICT_BATCH_SIZE)
    meta = windows.iloc[test_indices].reset_index(drop=True)
    prediction = pd.DataFrame(
        {
            "model_id": MODEL_ID,
            "model_name": MODEL_NAME,
            "site_id": meta["site_id"].astype(str),
            "target_timestamp": meta["target_timestamp"],
            "split": "test",
            "00400": y_true.astype("float64"),
            "y_pred": y_pred.astype("float64"),
            "in_ph_event": meta["in_ph_event"].astype("int8"),
            "ph_event_label": meta["ph_event_label"].astype(str),
            "event_state": meta["event_state"].astype("int8"),
        }
    )
    prediction["residual"] = prediction["00400"] - prediction["y_pred"]
    prediction["abs_residual"] = prediction["residual"].abs()
    prediction.to_parquet(OUT_DIR / PREDICTION_NAME, index=False)
    return prediction


def verify_baseline_reconstruction(base: Any, arrays: dict[str, object], test_indices: np.ndarray) -> None:
    checkpoint = torch.load(BASE_DIR / "model_G100_B_gru_with_metadata.pt", map_location="cpu")
    model = base.SequenceRegressor(
        recurrent_type="gru",
        n_dynamic_features=len(base.DYNAMIC_FEATURES),
        n_sites=len(arrays["site_mapping"]),
        site_embedding_dim=int(arrays["site_embedding_dim"]),
        hidden_size=base.HIDDEN_SIZE,
        num_layers=base.NUM_LAYERS,
        mlp_hidden_size=base.MLP_HIDDEN_SIZE,
        metadata_dim=len(arrays["metadata_features"]),
        dropout=base.DROPOUT,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    check_indices = test_indices[:4096]
    _loss, y_true, reconstructed = base.evaluate_model(model, check_indices, arrays, True, torch.device("cpu"), 4096)
    saved = pd.read_parquet(BASE_PRED_PATH, columns=["00400", "y_pred"]).head(len(check_indices))
    max_target_difference = float(np.max(np.abs(y_true - saved["00400"].to_numpy(dtype="float32"))))
    max_prediction_difference = float(np.max(np.abs(reconstructed - saved["y_pred"].to_numpy(dtype="float32"))))
    result = pd.DataFrame([
        {
            "n_windows_checked": len(check_indices),
            "max_absolute_target_difference": max_target_difference,
            "max_absolute_prediction_difference": max_prediction_difference,
            "tolerance": 1e-5,
            "passed": max_target_difference <= 1e-7 and max_prediction_difference <= 1e-5,
            "purpose": "prove exact baseline feature order, scalers, metadata vectors, site encoding, and architecture reconstruction",
        }
    ])
    result.to_csv(OUT_DIR / "baseline_reconstruction_audit.csv", index=False)
    if not bool(result.loc[0, "passed"]):
        raise RuntimeError(f"Baseline reconstruction failed: max prediction difference {max_prediction_difference}")
    print(f"[INFO] Baseline reconstruction passed; max prediction difference={max_prediction_difference:.3g}")


def prediction_metrics(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_id, pred in predictions.items():
        residual = pred["00400"].to_numpy(dtype="float64") - pred["y_pred"].to_numpy(dtype="float64")
        target = pred["00400"].to_numpy(dtype="float64")
        sse = np.square(residual).sum()
        sst = np.square(target - target.mean()).sum()
        rows.append(
            {
                "model_id": model_id,
                "n_test_windows": len(pred),
                "n_sites": pred["site_id"].nunique(),
                "rmse": math.sqrt(float(np.mean(np.square(residual)))),
                "mae": float(np.mean(np.abs(residual))),
                "r2": float(1 - sse / sst),
                "mean_signed_residual": float(np.mean(residual)),
                "median_signed_residual": float(np.median(residual)),
                "test_set_definition": "unchanged complete G100_B test windows",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "nonevent_gru_prediction_metrics.csv", index=False)
    return result


def event_blocks(pred: pd.DataFrame, row_mask: np.ndarray, detection_mask: np.ndarray) -> pd.DataFrame:
    events = pred.loc[row_mask, ["site_id", "target_timestamp", "ph_event_label"]].copy()
    events["hit"] = detection_mask[row_mask]
    events["label"] = events["ph_event_label"].map(compact_event_label)
    events = events.sort_values(["site_id", "label", "target_timestamp"])
    new = (
        events["site_id"].ne(events["site_id"].shift())
        | events["label"].ne(events["label"].shift())
        | events["target_timestamp"].diff().ne(pd.Timedelta(hours=1))
    )
    events["event_window_id"] = new.cumsum()
    return events.groupby(["event_window_id", "label"], observed=True).agg(hit=("hit", "max"), n_rows=("hit", "size")).reset_index()


def residual_metrics(predictions: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    absolute_rows, signed_rows, hit_rows = [], [], []
    for model_id, pred in predictions.items():
        labels = pred["ph_event_label"].map(compact_event_label).to_numpy()
        is_event = pred["in_ph_event"].astype(bool).to_numpy()
        low, high, mixed = labels == "low", labels == "high", labels == "mixed"
        residual = pred["residual"].to_numpy(dtype="float64")
        abs_residual = np.abs(residual)
        for fraction in TOP_FRACTIONS:
            abs_mask, abs_threshold = exact_top_mask(abs_residual, fraction, largest=True)
            neg_mask, neg_threshold = exact_top_mask(residual, fraction, largest=False)
            pos_mask, pos_threshold = exact_top_mask(residual, fraction, largest=True)
            base_event = is_event.mean()
            abs_rate = is_event[abs_mask].mean()
            absolute_rows.append(
                {
                    "model_id": model_id,
                    "tail_fraction": fraction,
                    "n_test_windows": len(pred),
                    "n_event_rows": int(is_event.sum()),
                    "base_event_rate": base_event,
                    "n_selected_rows": int(abs_mask.sum()),
                    "n_selected_event_rows": int((abs_mask & is_event).sum()),
                    "tail_event_rate": abs_rate,
                    "absolute_event_enrichment": abs_rate / base_event if base_event > 0 else np.nan,
                    "absolute_residual_threshold": abs_threshold,
                    "mean_absolute_residual": float(abs_residual.mean()),
                }
            )
            low_rate = low[neg_mask].mean()
            high_rate = high[pos_mask].mean()
            signed_rows.append(
                {
                    "model_id": model_id,
                    "tail_fraction": fraction,
                    "negative_residual_threshold": neg_threshold,
                    "positive_residual_threshold": pos_threshold,
                    "low_pH_event_rows": int(low.sum()),
                    "high_pH_event_rows": int(high.sum()),
                    "mixed_event_rows": int(mixed.sum()),
                    "base_low_pH_rate": low.mean(),
                    "negative_tail_low_pH_rate": low_rate,
                    "negative_tail_low_pH_enrichment": low_rate / low.mean() if low.mean() > 0 else np.nan,
                    "base_high_pH_rate": high.mean(),
                    "positive_tail_high_pH_rate": high_rate,
                    "positive_tail_high_pH_enrichment": high_rate / high.mean() if high.mean() > 0 else np.nan,
                    "low_pH_expected_sign_rate": float((residual[low] < 0).mean()) if low.any() else np.nan,
                    "high_pH_expected_sign_rate": float((residual[high] > 0).mean()) if high.any() else np.nan,
                    "event_count_warning": "high-pH estimates may be unstable" if high.sum() < 100 else "none",
                }
            )
            for detection_type, event_group, event_mask, detection in [
                ("absolute", "all", is_event, abs_mask),
                ("negative", "low", low, neg_mask),
                ("positive", "high", high, pos_mask),
                ("absolute", "mixed", mixed, abs_mask),
            ]:
                blocks = event_blocks(pred, event_mask, detection) if event_mask.any() else pd.DataFrame()
                hit_rows.append(
                    {
                        "model_id": model_id,
                        "tail_fraction": fraction,
                        "detection_type": detection_type,
                        "event_group": event_group,
                        "n_event_windows": len(blocks),
                        "n_hit_event_windows": int(blocks["hit"].sum()) if not blocks.empty else 0,
                        "event_hit_rate": float(blocks["hit"].mean()) if not blocks.empty else np.nan,
                        "event_window_method": "contiguous hourly target rows by site and event label",
                    }
                )
    absolute = pd.DataFrame(absolute_rows)
    signed = pd.DataFrame(signed_rows)
    hits = pd.DataFrame(hit_rows)
    absolute.to_csv(OUT_DIR / "nonevent_gru_residual_metrics.csv", index=False)
    signed.to_csv(OUT_DIR / "nonevent_gru_signed_tail_metrics.csv", index=False)
    hits.to_csv(OUT_DIR / "nonevent_gru_event_hit_rates.csv", index=False)
    return absolute, signed, hits


def configure_plots() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def comparison_bar(frame: pd.DataFrame, value: str, stem: str, ylabel: str) -> None:
    order = ["G100_B", MODEL_ID]
    values = [float(frame.loc[frame["model_id"].eq(model), value].iloc[0]) for model in order]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    bars = ax.bar(["All-data G100_B", "Non-event target-only"], values, color=["#4B7FA8", "#C08A45"], width=0.62)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.45)
    ax.bar_label(bars, labels=[f"{x:.3f}" for x in values], padding=3, fontsize=8)
    save_fig(fig, stem)


def build_figures(
    metrics: pd.DataFrame,
    absolute: pd.DataFrame,
    signed: pd.DataFrame,
    hits: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> None:
    configure_plots()
    comparison_bar(metrics, "rmse", "gru_all_vs_nonevent_rmse", "Test RMSE")
    comparison_bar(metrics, "r2", "gru_all_vs_nonevent_r2", "Test R2")
    comparison_bar(absolute.loc[absolute["tail_fraction"].eq(0.05)], "absolute_event_enrichment",
                   "gru_all_vs_nonevent_absolute_enrichment", "Top-5% absolute event enrichment")
    comparison_bar(signed.loc[signed["tail_fraction"].eq(0.05)], "negative_tail_low_pH_enrichment",
                   "gru_all_vs_nonevent_low_negative_tail_enrichment", "Negative-tail 5% low-pH enrichment")
    comparison_bar(signed.loc[signed["tail_fraction"].eq(0.05)], "positive_tail_high_pH_enrichment",
                   "gru_all_vs_nonevent_high_positive_tail_enrichment", "Positive-tail 5% high-pH enrichment")
    hit = hits.loc[
        hits["tail_fraction"].eq(0.05) & hits["detection_type"].eq("absolute") & hits["event_group"].eq("all")
    ]
    comparison_bar(hit, "event_hit_rate", "gru_all_vs_nonevent_event_hit_rate", "Top-5% event-window hit rate")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for model_id, color in [("G100_B", "#4B7FA8"), (MODEL_ID, "#C08A45")]:
        residual = predictions[model_id]["residual"].to_numpy(dtype="float64")
        bounds = np.quantile(residual, [0.005, 0.995])
        ax.hist(residual[(residual >= bounds[0]) & (residual <= bounds[1])], bins=100, density=True,
                histtype="step", linewidth=1.1, color=color, label="All-data G100_B" if model_id == "G100_B" else "Non-event target-only")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Residual (observed - predicted)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(color="#DDDDDD", linewidth=0.4)
    save_fig(fig, "gru_all_vs_nonevent_residual_distribution")


def value_at(frame: pd.DataFrame, model_id: str, column: str, fraction: float = 0.05, **filters: str) -> float:
    mask = frame["model_id"].eq(model_id)
    if "tail_fraction" in frame:
        mask &= frame["tail_fraction"].eq(fraction)
    for key, value in filters.items():
        mask &= frame[key].eq(value)
    return float(frame.loc[mask, column].iloc[0])


def write_config(base: Any, audit: pd.DataFrame, arrays: dict[str, object], elapsed: float | None = None) -> None:
    baseline_config = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    overall = audit.loc[audit["scope"].eq("overall")].iloc[0]
    config = {
        "model_id": MODEL_ID,
        "baseline_model_id": "G100_B",
        "single_experimental_change": "exclude train windows whose target row has in_ph_event == 1",
        "validation_windows": "complete and unchanged for final evaluation; baseline 500,000-window early-stop sample retained",
        "test_windows": "complete and unchanged",
        "event_features_used": False,
        "strict_24h_sensitivity": "skipped to prioritise required target-only model on CPU",
        "baseline_config": baseline_config,
        "scaler_and_encoding_audit": arrays["scaler_audit"],
        "window_counts": {key: int(overall[key]) for key in [
            "original_train_windows", "removed_event_target_windows", "retained_nonevent_train_windows",
            "sampled_train_windows", "validation_windows_unchanged", "test_windows_unchanged",
        ]},
        "runtime_seconds": elapsed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
    }
    (OUT_DIR / "nonevent_gru100_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def write_report(
    audit: pd.DataFrame,
    metrics: pd.DataFrame,
    absolute: pd.DataFrame,
    signed: pd.DataFrame,
    hits: pd.DataFrame,
) -> dict[str, Any]:
    overall = audit.loc[audit["scope"].eq("overall")].iloc[0]
    base_rmse = value_at(metrics, "G100_B", "rmse")
    new_rmse = value_at(metrics, MODEL_ID, "rmse")
    base_r2 = value_at(metrics, "G100_B", "r2")
    new_r2 = value_at(metrics, MODEL_ID, "r2")
    base_abs = value_at(absolute, "G100_B", "absolute_event_enrichment")
    new_abs = value_at(absolute, MODEL_ID, "absolute_event_enrichment")
    base_low = value_at(signed, "G100_B", "negative_tail_low_pH_enrichment")
    new_low = value_at(signed, MODEL_ID, "negative_tail_low_pH_enrichment")
    base_high = value_at(signed, "G100_B", "positive_tail_high_pH_enrichment")
    new_high = value_at(signed, MODEL_ID, "positive_tail_high_pH_enrichment")
    base_hit = value_at(hits, "G100_B", "event_hit_rate", detection_type="absolute", event_group="all")
    new_hit = value_at(hits, MODEL_ID, "event_hit_rate", detection_type="absolute", event_group="all")
    low_improved, abs_improved, hit_improved = new_low > base_low, new_abs > base_abs, new_hit > base_hit
    detection = "yes" if abs_improved and low_improved and hit_improved else "partly" if any([abs_improved, low_improved, hit_improved]) else "no"
    m0_prediction = pd.read_csv(M0_PREDICTION_METRICS_PATH).loc[lambda x: x["model_id"].eq("M0_reference_nonevent")].iloc[0]
    m0_signed = pd.read_csv(M0_SIGNED_METRICS_PATH).loc[lambda x: x["model_id"].eq("M0_reference_nonevent")].iloc[0]
    report = f"""# GRU100 full-metadata non-event target-only experiment

## Controlled design

This experiment uses the exact G100_B architecture, 100 sites, 24-hour sequence, 12 exogenous/time inputs, 73 metadata features, site encoding, baseline scalers, site-balanced 1,000,000-window CPU sample, seed 42, optimiser and early stopping settings. The only experimental change is removal from **training only** of windows whose target row has `in_ph_event == 1`. Event fields are not model inputs. Validation and test windows remain complete.

- Original train windows: {int(overall['original_train_windows']):,}
- Event-target train windows removed: {int(overall['removed_event_target_windows']):,} ({overall['percentage_removed']:.2f}%)
- Retained non-event train windows: {int(overall['retained_nonevent_train_windows']):,}
- Sampled train windows: {int(overall['sampled_train_windows']):,}
- Unchanged test windows: {int(overall['test_windows_unchanged']):,}
- Sites lost: {int(overall['sites_with_no_remaining_train_windows'])}

## Fair all-data versus non-event comparison

| Metric | G100_B all-data | Non-event target-only |
|---|---:|---:|
| RMSE | {base_rmse:.6f} | {new_rmse:.6f} |
| R2 | {base_r2:.6f} | {new_r2:.6f} |
| Absolute top-5% enrichment | {base_abs:.3f} | {new_abs:.3f} |
| Low-pH negative-tail 5% enrichment | {base_low:.3f} | {new_low:.3f} |
| High-pH positive-tail 5% enrichment | {base_high:.3f} | {new_high:.3f} |
| Absolute top-5% event-window hit rate | {base_hit:.3f} | {new_hit:.3f} |

## Contextual M0 reference

M0 non-event HGB had RMSE {float(m0_prediction['rmse']):.6f}, absolute top-5% enrichment {float(m0_signed['absolute_top5_all_event_enrichment']):.3f}, low-pH negative-tail enrichment {float(m0_signed['negative_tail5_low_enrichment']):.3f}, and high-pH positive-tail enrichment {float(m0_signed['positive_tail5_high_enrichment']):.3f}. These values are contextual only because M0 uses 50 sites and native processed rows rather than the fixed GRU100 hourly test windows.

## Answers

1. Non-event training {'improved' if abs_improved else 'did not improve'} top-5% absolute enrichment.
2. It {'improved' if low_improved else 'did not improve'} low-pH negative-tail enrichment.
3. It {'improved' if hit_improved else 'did not improve'} event-level top-5% hit rate.
4. The RMSE change was {new_rmse - base_rmse:+.6f}; the R2 change was {new_r2 - base_r2:+.6f}.
5. The HGB non-event finding is reproduced by GRU **{detection}** across the three primary detection criteria.
6. The low-pH change was {new_low - base_low:+.3f} enrichment units; the result should be read alongside the high-pH change of {new_high - base_high:+.3f}.
7. High-pH evidence remains constrained by fewer event windows; the signed-tail and hit-rate tables preserve the counts.
8. Target-only exclusion is the completed teacher-aligned test. Strict 24-hour exclusion was skipped because this machine is CPU-only and the optional sensitivity must not delay the main model.

The fair causal comparison is therefore G100_B all-data versus the target-only non-event GRU on identical test rows.
"""
    (OUT_DIR / "nonevent_gru_report.md").write_text(report, encoding="utf-8")
    return {
        "all_rmse": base_rmse, "new_rmse": new_rmse, "all_r2": base_r2, "new_r2": new_r2,
        "all_abs5": base_abs, "new_abs5": new_abs, "all_low5": base_low, "new_low5": new_low,
        "all_high5": base_high, "new_high5": new_high, "all_hit5": base_hit, "new_hit5": new_hit,
        "detection_improved": detection,
    }


def main() -> None:
    started = perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tee_output(LOG_PATH):
        print("[INFO] Starting controlled non-event GRU100 target-only experiment")
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        base = load_baseline_module()
        windows, arrays = load_inputs(base)
        train_indices, val_early, test_indices, audit = audit_and_sample(base, windows)
        verify_baseline_reconstruction(base, arrays, test_indices)
        write_config(base, audit, arrays)
        prediction = train_and_predict(base, windows, arrays, train_indices, val_early, test_indices)
        baseline = pd.read_parquet(BASE_PRED_PATH)
        baseline["model_id"] = "G100_B"
        if not baseline[["site_id", "target_timestamp"]].equals(prediction[["site_id", "target_timestamp"]]):
            raise RuntimeError("All-data and non-event GRU prediction rows are not the same unchanged test set.")
        predictions = {"G100_B": baseline, MODEL_ID: prediction}
        metrics = prediction_metrics(predictions)
        absolute, signed, hits = residual_metrics(predictions)
        build_figures(metrics, absolute, signed, hits, predictions)
        status = write_report(audit, metrics, absolute, signed, hits)
        elapsed = perf_counter() - started
        write_config(base, audit, arrays, elapsed=elapsed)
        (OUT_DIR / "nonevent_gru_status.json").write_text(json.dumps(status | {"runtime_seconds": elapsed}, indent=2), encoding="utf-8")
        print("NON-EVENT GRU100 TARGET-ONLY COMPLETE")
        print(f"Runtime seconds: {elapsed:.1f}")
        print(f"G100_B RMSE: {status['all_rmse']:.6f}")
        print(f"Non-event RMSE: {status['new_rmse']:.6f}")
        print(f"G100_B top-5 enrichment: {status['all_abs5']:.3f}")
        print(f"Non-event top-5 enrichment: {status['new_abs5']:.3f}")
        del baseline, prediction, predictions, windows, arrays
        gc.collect()


if __name__ == "__main__":
    main()
