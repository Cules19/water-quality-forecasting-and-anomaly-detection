"""Reuse frozen predictions to close the cross-variable hot-moment analysis."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analysis_outputs" / "cross_variable_hot_moment_incremental_closure_v1"
CROSS_ROOT = ROOT / "analysis_outputs" / "ph_common1000_lstm_nohistory_h0_rain_final_v1" / "full"
HIST_ROOT = ROOT / "analysis_outputs" / "ph_common1000_lstm_film_final_rain_ablation_teacher_aligned_v1" / "full"
CROSS_RUN = "ph_LSTM_RAIN_H0_H00"
HIST_RUN = "ph_LSTM_RAIN_FINAL_H01"
CROSS_PRED = CROSS_ROOT / "predictions_by_site" / CROSS_RUN
HIST_PRED = HIST_ROOT / "predictions_by_site" / HIST_RUN
COHORT = ROOT / "analysis_outputs" / "ph_common1000_teacher_archive_features_do_converted_v1" / "full" / "reports" / "primary_archive_experiment_cohort.csv"
MERGED_INPUT = ROOT / "analysis_outputs" / "ph_common1000_teacher_archive_features_do_converted_v1" / "full" / "hourly_ph_rainfall_by_site"
FLOW_DRIVER = ROOT / "analysis_outputs" / "ph_common1000_error_onset_do_converted_v1" / "attribution_drivers_by_site"
EVENT_CONFIG = ROOT / "config" / "common1000_error_onset_final_lstm_rain_ablation_v1.json"
EVENT_OUTPUT = ROOT / "analysis_outputs" / "ph_common1000_error_onset_final_lstm_rain_ablation_v1"
EVENT_CODE = ROOT / "pipeline" / "24_error_onset" / "run_error_onset_analysis.py"
SEED = 73129
BOOTSTRAPS = 10_000
MODELS = ["cross_variable_h00", "historical_target_h01"]
MODEL_LABELS = {
    "cross_variable_h00": "Cross-variable H00",
    "historical_target_h01": "Historical-target H01",
}
MODEL_SHORT = {"cross_variable_h00": "cross", "historical_target_h01": "hist"}
LAGS = [0, 1, 3, 6, 12, 24]


def load_event_module() -> Any:
    spec = importlib.util.spec_from_file_location("existing_event_analysis", EVENT_CODE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVENT = load_event_module()


def mkdirs() -> None:
    for name in (
        "configs", "calibration", "data", "predictions", "residual_scores",
        "event_metrics", "site_metrics", "bootstrap", "figures", "reports",
        "summary", "completion",
    ):
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 4) -> str:
    view = frame.copy() if columns is None else frame.loc[:, columns].copy()
    if view.empty:
        return "_No rows._"
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    header = "| " + " | ".join(view.columns.astype(str)) + " |"
    divider = "|" + "|".join(["---"] * len(view.columns)) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body])


def config_manifest() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cross_config_path = CROSS_ROOT / "reports" / "effective_config.json"
    hist_config_path = HIST_ROOT / "reports" / "effective_config.json"
    cross_success_path = CROSS_ROOT / "models" / f"{CROSS_RUN}_SUCCESS.json"
    hist_success_path = HIST_ROOT / "models" / f"{HIST_RUN}_SUCCESS.json"
    cross_config, hist_config = read_json(cross_config_path), read_json(hist_config_path)
    cross_success, hist_success = read_json(cross_success_path), read_json(hist_success_path)
    event_config = read_json(EVENT_CONFIG)
    comparator_ids = [model["run_id"] for model in event_config["models"]]
    required_paths = [
        CROSS_ROOT / "models" / f"{CROSS_RUN}.pt",
        HIST_ROOT / "models" / f"{HIST_RUN}.pt",
        cross_config_path, hist_config_path,
        CROSS_ROOT / "reports" / "train_only_preprocessing.json",
        HIST_ROOT / "reports" / "train_only_preprocessing.json",
        CROSS_ROOT / "metadata" / "ph_no_coordinates_train_only_transform.json",
        HIST_ROOT / "metadata" / "ph_no_coordinates_train_only_transform.json",
        CROSS_ROOT / "reports" / "common_key_manifest.csv",
        HIST_ROOT / "reports" / "common_key_manifest.csv",
        COHORT, EVENT_CONFIG, EVENT_CODE,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    problems = []
    cross_history = any("target_history" in name for name in cross_success["input_feature_names"])
    hist_history = any("target_history" in name for name in hist_success["input_feature_names"])
    if cross_history:
        problems.append("Cross-variable H00 contains target history")
    if int(cross_config["horizon_hours"]) != 0:
        problems.append("Cross-variable checkpoint is not H00")
    if not hist_history:
        problems.append("Historical comparator does not contain target history")
    if HIST_RUN not in comparator_ids:
        problems.append("Historical comparator is not in the formal event-analysis config")
    cohort = pd.read_csv(COHORT, dtype={"site_id": "string", "split": "string"})
    split_counts = cohort["split"].value_counts().to_dict()
    if split_counts != {"train": 638, "validation": 137, "test": 135}:
        problems.append(f"Unexpected final split counts: {split_counts}")
    for split, expected in (("validation", 137), ("test", 135)):
        cross_sites = {path.stem for path in (CROSS_PRED / split).glob("*.parquet")}
        hist_sites = {path.stem for path in (HIST_PRED / split).glob("*.parquet")}
        if cross_sites != hist_sites or len(cross_sites) != expected:
            problems.append(f"Prediction site mismatch in {split}")
    if missing:
        problems.append(f"Missing provenance files: {missing}")
    if problems:
        raise RuntimeError("; ".join(problems))

    def entry(
        role: str,
        run_id: str,
        root: Path,
        config_path: Path,
        config: dict[str, Any],
        success: dict[str, Any],
        historical_ph: bool,
    ) -> dict[str, Any]:
        checkpoint = root / "models" / f"{run_id}.pt"
        scaler = root / "reports" / "train_only_preprocessing.json"
        metadata = root / "metadata" / "ph_no_coordinates_train_only_transform.json"
        target_keys = root / "reports" / "common_key_manifest.csv"
        return {
            "role": role,
            "run_id": run_id,
            "checkpoint_path": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
            "config_path": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256(config_path),
            "feature_list": success["feature_set"],
            "feature_order": success["input_feature_names"],
            "target": "pH (00400)",
            "target_unit": "pH",
            "forecast_horizon_hours": int(config["horizon_hours"]),
            "historical_ph_present": historical_ph,
            "scaler_path": str(scaler.relative_to(ROOT)),
            "scaler_sha256": sha256(scaler),
            "metadata_transform_path": str(metadata.relative_to(ROOT)),
            "metadata_transform_sha256": sha256(metadata),
            "site_split_path": str(COHORT.relative_to(ROOT)),
            "site_split_sha256": sha256(COHORT),
            "target_key_manifest_path": str(target_keys.relative_to(ROOT)),
            "target_key_manifest_sha256": sha256(target_keys),
            "checkpoint_target_key_manifest_sha256": success["common_key_manifest_sha256"],
            "training_mode": "all_data",
            "random_seed": int(config["random_seed"]),
            "architecture": {
                "family": "LSTM-FiLM",
                "recurrent_type": success["recurrent_type"],
                "hidden_size": int(config["hidden_size"]),
                "layers": int(success["recurrent_layers"]),
                "dropout": float(success["dropout"]),
                "sequence_length_hours": int(success["sequence_length_hours"]),
            },
            "existing_prediction_root": str((root / "predictions_by_site" / run_id).relative_to(ROOT)),
            "inference_reused": True,
        }

    manifest = {
        "reuse_first": True,
        "models_retrained": False,
        "cross_variable_h00": entry(
            "contemporaneous cross-variable pH soft sensing", CROSS_RUN, CROSS_ROOT,
            cross_config_path, cross_config, cross_success, False,
        ),
        "historical_target_h01": entry(
            "formal historical-target event-analysis comparator", HIST_RUN, HIST_ROOT,
            hist_config_path, hist_config, hist_success, True,
        ),
        "event_assets": {
            "formal_event_config": str(EVENT_CONFIG.relative_to(ROOT)),
            "formal_event_config_sha256": sha256(EVENT_CONFIG),
            "event_analysis_code": str(EVENT_CODE.relative_to(ROOT)),
            "event_analysis_code_sha256": sha256(EVENT_CODE),
            "original_event_output": str(EVENT_OUTPUT.relative_to(ROOT)),
            "event_onset_field": "event_start",
            "event_window_definition": "contiguous hourly in_event mask; a gap >1.5 h starts a new window",
            "pre_event_window_hours": 48,
            "persistence_hours": 3,
            "fixed_lags_hours": LAGS,
            "weak_label_semantics": True,
        },
        "disclosure": "The frozen H00 checkpoint has no historical pH. It also retains six deterministic calendar channels inherited from the frozen original model; no calendar channel was added or tuned in this closure.",
    }
    return manifest, cross_config, hist_config


PRED_COLUMNS = [
    "site_id", "timestamp", "split", "observed_native", "predicted_native",
    "in_event", "event_direction", "event_id", "input_end_rain_mm",
    "input_end_precip_api", "input_end_precip_6h",
]


def prediction_assets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cross_parts: list[pd.DataFrame] = []
    hist_parts: list[pd.DataFrame] = []
    matched_parts: list[pd.DataFrame] = []
    audit_rows = []
    for split in ("validation", "test"):
        sites = sorted(path.stem for path in (CROSS_PRED / split).glob("*.parquet"))
        for site_id in sites:
            cross = pd.read_parquet(CROSS_PRED / split / f"{site_id}.parquet", columns=PRED_COLUMNS)
            hist = pd.read_parquet(HIST_PRED / split / f"{site_id}.parquet", columns=PRED_COLUMNS)
            cross["site_id"] = cross["site_id"].astype("string")
            hist["site_id"] = hist["site_id"].astype("string")
            cross["target_timestamp"] = pd.to_datetime(cross.pop("timestamp"))
            hist["target_timestamp"] = pd.to_datetime(hist.pop("timestamp"))
            cross_min = cross[["site_id", "target_timestamp", "split", "observed_native", "predicted_native", "in_event", "event_direction", "event_id"]].copy()
            hist_min = hist[["site_id", "target_timestamp", "split", "observed_native", "predicted_native", "in_event", "event_direction", "event_id"]].copy()
            cross_parts.append(cross_min)
            hist_parts.append(hist_min)
            merged = cross.merge(hist, on=["site_id", "target_timestamp"], suffixes=("_cross", "_hist"), validate="one_to_one")
            truth_difference = float(np.nanmax(np.abs(merged["observed_native_cross"] - merged["observed_native_hist"]))) if len(merged) else np.nan
            label_match = bool((merged["in_event_cross"].astype(bool) == merged["in_event_hist"].astype(bool)).all())
            if not label_match or truth_difference > 1e-6:
                raise RuntimeError(f"Truth/event mismatch at {site_id}")
            out = pd.DataFrame(
                {
                    "site_id": merged["site_id"].astype("string"),
                    "target_timestamp": merged["target_timestamp"],
                    "split": split,
                    "observed_ph": merged["observed_native_cross"].astype("float32"),
                    "predicted_cross": merged["predicted_native_cross"].astype("float32"),
                    "predicted_hist": merged["predicted_native_hist"].astype("float32"),
                    "in_event": merged["in_event_cross"].astype(bool),
                    "event_direction": merged["event_direction_cross"].astype("string"),
                    "event_id": merged["event_id_cross"].astype("int64"),
                    "current_rain_mm": pd.to_numeric(merged["input_end_rain_mm_cross"], errors="coerce").astype("float32"),
                    "precip_api": pd.to_numeric(merged["input_end_precip_api_cross"], errors="coerce").astype("float32"),
                    "rain_previous_6h_mm": pd.to_numeric(merged["input_end_precip_6h_cross"], errors="coerce").astype("float32"),
                }
            )
            if split == "test":
                merged_path = MERGED_INPUT / f"{site_id}.parquet"
                if merged_path.is_file():
                    turbidity = pd.read_parquet(merged_path, columns=["timestamp", "x_63680"])
                    turbidity = turbidity.rename(columns={"timestamp": "target_timestamp", "x_63680": "turbidity_log1p"})
                    turbidity["target_timestamp"] = pd.to_datetime(turbidity["target_timestamp"])
                    out = out.merge(turbidity, on="target_timestamp", how="left", validate="one_to_one")
                else:
                    out["turbidity_log1p"] = np.nan
                driver_path = FLOW_DRIVER / f"{site_id}.parquet"
                if driver_path.is_file():
                    flow = pd.read_parquet(driver_path, columns=["timestamp", "x_00060"])
                    flow = flow.rename(columns={"timestamp": "target_timestamp", "x_00060": "flow_signal"})
                    flow["target_timestamp"] = pd.to_datetime(flow["target_timestamp"])
                    out = out.merge(flow, on="target_timestamp", how="left", validate="one_to_one")
                else:
                    out["flow_signal"] = np.nan
            else:
                out["turbidity_log1p"] = np.nan
                out["flow_signal"] = np.nan
            matched_parts.append(out)
            audit_rows.append(
                {
                    "split": split, "site_id": site_id,
                    "cross_original_rows": len(cross), "historical_original_rows": len(hist),
                    "matched_rows": len(out), "cross_retained_fraction": len(out) / len(cross),
                    "historical_retained_fraction": len(out) / len(hist),
                    "truth_max_absolute_difference": truth_difference,
                    "event_labels_match": label_match,
                }
            )
    cross_all = pd.concat(cross_parts, ignore_index=True)
    hist_all = pd.concat(hist_parts, ignore_index=True)
    matched = pd.concat(matched_parts, ignore_index=True)
    if matched.duplicated(["site_id", "target_timestamp", "split"]).any():
        raise RuntimeError("Duplicate matched prediction key")
    audit = pd.DataFrame(audit_rows)
    summary = {
        "cross_original_rows": int(len(cross_all)),
        "historical_original_rows": int(len(hist_all)),
        "matched_rows": int(len(matched)),
        "validation_matched_rows": int(matched["split"].eq("validation").sum()),
        "test_matched_rows": int(matched["split"].eq("test").sum()),
        "cross_test_sites": int(cross_all.loc[cross_all["split"].eq("test"), "site_id"].nunique()),
        "historical_test_sites": int(hist_all.loc[hist_all["split"].eq("test"), "site_id"].nunique()),
        "matched_test_sites": int(matched.loc[matched["split"].eq("test"), "site_id"].nunique()),
        "maximum_truth_difference": float(audit["truth_max_absolute_difference"].max()),
        "all_event_labels_match": bool(audit["event_labels_match"].all()),
    }
    audit.to_csv(OUTPUT / "data" / "site_key_retention.csv", index=False)
    return cross_all, hist_all, matched, summary


def raw_mad(values: Iterable[float]) -> float:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan
    center = np.median(array)
    return float(1.4826 * np.median(np.abs(array - center)))


def calibrate(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = matched.copy()
    frame["residual_cross"] = frame["observed_ph"] - frame["predicted_cross"]
    frame["residual_hist"] = frame["observed_ph"] - frame["predicted_hist"]
    frame["positive_residual_cross"] = frame["residual_cross"].clip(lower=0)
    frame["negative_residual_cross"] = frame["residual_cross"].clip(upper=0)
    frame["positive_residual_hist"] = frame["residual_hist"].clip(lower=0)
    frame["negative_residual_hist"] = frame["residual_hist"].clip(upper=0)
    frame["calibration_period"] = False
    frame["calibration_eligible"] = False
    frame["evaluation_period"] = False
    summaries = []
    calibration_indices: dict[tuple[str, str], np.ndarray] = {}
    for (split, site_id), group in frame.groupby(["split", "site_id"], sort=False):
        ordered = group.sort_values("target_timestamp", kind="mergesort")
        start = ordered["target_timestamp"].iloc[0]
        end = start + pd.Timedelta(days=30)
        valid = ordered[["residual_cross", "residual_hist"]].notna().all(axis=1)
        cal = ordered["target_timestamp"].lt(end) & valid
        indices = ordered.index[cal].to_numpy()
        calibration_indices[(str(split), str(site_id))] = indices
        eligible = len(indices) >= 504
        frame.loc[indices, "calibration_period"] = True
        frame.loc[ordered.index, "calibration_eligible"] = eligible
        if eligible:
            eval_index = ordered.index[ordered["target_timestamp"].ge(end) & valid]
            frame.loc[eval_index, "evaluation_period"] = True
        summaries.append(
            {
                "split": split, "site_id": site_id, "calibration_start": start,
                "calibration_end_exclusive": end, "calibration_valid_hours": len(indices),
                "calibration_status": "ELIGIBLE" if eligible else "CALIBRATION_INSUFFICIENT",
            }
        )
    summary = pd.DataFrame(summaries)
    validation_cal = np.concatenate(
        [indices for (split, _), indices in calibration_indices.items() if split == "validation"]
    )
    global_fallback = {
        "cross": raw_mad(frame.loc[validation_cal, "residual_cross"].to_numpy()),
        "hist": raw_mad(frame.loc[validation_cal, "residual_hist"].to_numpy()),
    }
    fallback_sites: dict[str, list[str]] = {"cross": [], "hist": []}
    for short in ("cross", "hist"):
        frame[f"signed_score_{short}"] = np.nan
        frame[f"absolute_score_{short}"] = np.nan
    residual_rows = []
    for row in summary.itertuples(index=False):
        indices = calibration_indices[(str(row.split), str(row.site_id))]
        eligible = row.calibration_status == "ELIGIBLE"
        for short in ("cross", "hist"):
            residual = frame.loc[indices, f"residual_{short}"].to_numpy(dtype=float)
            center = float(np.median(residual)) if len(residual) else np.nan
            scale_raw = raw_mad(residual)
            fallback = bool(eligible and np.isfinite(scale_raw) and scale_raw < 1e-6)
            scale = global_fallback[short] if fallback else scale_raw
            if fallback:
                fallback_sites[short].append(str(row.site_id))
            site_index = frame.index[(frame["split"].eq(row.split)) & (frame["site_id"].astype(str).eq(str(row.site_id)))]
            if eligible and np.isfinite(center) and np.isfinite(scale):
                score = (frame.loc[site_index, f"residual_{short}"] - center) / max(float(scale), 1e-6)
                frame.loc[site_index, f"signed_score_{short}"] = score
                frame.loc[site_index, f"absolute_score_{short}"] = score.abs()
            evaluation = frame.loc[site_index]
            eval_mask = evaluation["evaluation_period"]
            residual_rows.append(
                {
                    "split": row.split, "site_id": row.site_id, "model": short,
                    "calibration_status": row.calibration_status,
                    "calibration_valid_hours": int(row.calibration_valid_hours),
                    "center": center, "scale_raw": scale_raw, "scale_used": scale,
                    "global_mad_fallback_used": fallback,
                    "evaluation_rows": int(eval_mask.sum()),
                    "evaluation_residual_median": float(evaluation.loc[eval_mask, f"residual_{short}"].median()) if eval_mask.any() else np.nan,
                    "evaluation_residual_mad": raw_mad(evaluation.loc[eval_mask, f"residual_{short}"].to_numpy()) if eval_mask.any() else np.nan,
                }
            )
    residual_summary = pd.DataFrame(residual_rows)
    validation_eval = frame.loc[frame["split"].eq("validation") & frame["evaluation_period"]]
    thresholds: dict[str, Any] = {
        "policy": "validation evaluation scores after each site's fixed earliest-30-day calibration; event labels not read",
        "calibration_days": 30, "minimum_calibration_valid_hours": 504,
        "primary_quantile": 0.95, "sensitivity_quantiles": [0.975, 0.99],
        "persistence_consecutive_hours": 3, "event_labels_used": False,
        "fallback_validation_global_mad": global_fallback,
        "fallback_sites": fallback_sites,
        "models": {},
    }
    for model, short in MODEL_SHORT.items():
        absolute = validation_eval[f"absolute_score_{short}"].dropna()
        signed = validation_eval[f"signed_score_{short}"].dropna()
        thresholds["models"][model] = {
            "fit_split": "validation", "fit_rows": int(len(absolute)),
            "fit_sites": int(validation_eval.loc[absolute.index, "site_id"].nunique()),
            "absolute_score_q95": float(absolute.quantile(0.95)),
            "absolute_score_q975": float(absolute.quantile(0.975)),
            "absolute_score_q99": float(absolute.quantile(0.99)),
            "signed_score_q95": float(signed.quantile(0.95)),
            "signed_score_q05": float(signed.quantile(0.05)),
        }
    return frame, residual_summary, thresholds


def canonical_event_manifest(
    historical_predictions: pd.DataFrame,
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    original_test = historical_predictions.loc[historical_predictions["split"].eq("test")].rename(
        columns={"target_timestamp": "timestamp"}
    )
    original = EVENT.event_windows(original_test[["site_id", "timestamp", "in_event", "event_direction"]])
    raw_common = scored.loc[scored["split"].eq("test")].copy()
    evaluation = raw_common.loc[raw_common["evaluation_period"]].copy()
    rows = []
    by_raw = {str(site): group for site, group in raw_common.groupby("site_id", sort=False)}
    by_eval = {str(site): group for site, group in evaluation.groupby("site_id", sort=False)}
    for event in original.itertuples(index=False):
        raw = by_raw.get(str(event.site_id), pd.DataFrame())
        ev = by_eval.get(str(event.site_id), pd.DataFrame())
        raw_event = raw.loc[
            raw["target_timestamp"].ge(event.event_start)
            & raw["target_timestamp"].le(event.event_end)
            & raw["in_event"]
        ] if len(raw) else raw
        eval_event = ev.loc[
            ev["target_timestamp"].ge(event.event_start)
            & ev["target_timestamp"].le(event.event_end)
            & ev["in_event"]
        ] if len(ev) else ev
        pre = ev.loc[
            ev["target_timestamp"].ge(event.event_start - pd.Timedelta(hours=48))
            & ev["target_timestamp"].lt(event.event_start)
            & ~ev["in_event"]
        ] if len(ev) else ev
        status = "ANALYSABLE"
        reason = ""
        if raw_event.empty:
            status, reason = "EXCLUDED", "missing common prediction in registered event window"
        elif eval_event.empty:
            eligible = bool(raw["calibration_eligible"].iloc[0]) if len(raw) else False
            status = "EXCLUDED"
            reason = "calibration insufficient at site" if not eligible else "registered event lies inside fixed calibration period"
        elif pre.empty:
            status, reason = "MATCHED_NOT_LEAD_ANALYSABLE", "insufficient pre-event evaluation window"
        rows.append(
            {
                **event._asdict(),
                "common_event_rows_before_calibration_exclusion": int(len(raw_event)),
                "common_event_rows_in_evaluation": int(len(eval_event)),
                "pre_event_evaluation_rows": int(len(pre)),
                "matched_registered_event": bool(len(raw_event)),
                "event_evaluation_included": bool(len(eval_event)),
                "lead_time_analysable": bool(len(eval_event) and len(pre)),
                "status": status,
                "exclusion_reason": reason,
            }
        )
    manifest = pd.DataFrame(rows)
    summary = {
        "original_registered_events": int(len(manifest)),
        "matched_registered_events": int(manifest["matched_registered_event"].sum()),
        "event_evaluation_events": int(manifest["event_evaluation_included"].sum()),
        "lead_time_analysable_events": int(manifest["lead_time_analysable"].sum()),
        "excluded_events": int((~manifest["event_evaluation_included"]).sum()),
        "excluded_reason_counts": manifest.loc[~manifest["event_evaluation_included"], "exclusion_reason"].value_counts().to_dict(),
    }
    return manifest, summary


def score_flags(frame: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    for model, short in MODEL_SHORT.items():
        threshold = thresholds["models"][model]
        result[f"top5_{short}"] = result[f"absolute_score_{short}"] > threshold["absolute_score_q95"]
        result[f"top1_{short}"] = result[f"absolute_score_{short}"] > threshold["absolute_score_q99"]
        result[f"positive_tail_{short}"] = result[f"signed_score_{short}"] > threshold["signed_score_q95"]
        result[f"negative_tail_{short}"] = result[f"signed_score_{short}"] < threshold["signed_score_q05"]
    return result


def bootstrap_site_counts(
    site_counts: pd.DataFrame,
    seed_key: str,
) -> np.ndarray:
    values = site_counts[["event_high", "event_rows", "other_high", "other_rows"]].to_numpy(dtype=float)
    seed = int(hashlib.sha256(f"{SEED}|{seed_key}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    selected = values[sample].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        enrichment = (selected[:, 0] / selected[:, 1]) / (selected[:, 2] / selected[:, 3])
    return enrichment


def bootstrap_event_rates(event_rates: np.ndarray, other_rate: float, seed_key: str) -> np.ndarray:
    seed = int(hashlib.sha256(f"{SEED}|{seed_key}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    output = np.empty(BOOTSTRAPS, dtype=float)
    chunk = 250
    for start in range(0, BOOTSTRAPS, chunk):
        stop = min(start + chunk, BOOTSTRAPS)
        sample = rng.integers(0, len(event_rates), size=(stop - start, len(event_rates)))
        output[start:stop] = event_rates[sample].mean(axis=1) / other_rate
    return output


def enrichment_analysis(
    test: pd.DataFrame,
    thresholds: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    definitions = {
        "top_1_percent_absolute": "top1",
        "top_5_percent_absolute": "top5",
        "positive_tail": "positive_tail",
        "negative_tail": "negative_tail",
    }
    summaries = []
    site_metric_parts = []
    site_boot_parts = []
    event_boot_parts = []
    event_windows = EVENT.event_windows(
        test.rename(columns={"target_timestamp": "timestamp"})[["site_id", "timestamp", "in_event", "event_direction"]]
    )
    for model, short in MODEL_SHORT.items():
        for score_name, prefix in definitions.items():
            high_column = f"{prefix}_{short}"
            site_rows = []
            for site_id, site in test.groupby("site_id", sort=False):
                event = site["in_event"].to_numpy(dtype=bool)
                high = site[high_column].to_numpy(dtype=bool)
                event_rows, other_rows = int(event.sum()), int((~event).sum())
                event_high, other_high = int((high & event).sum()), int((high & ~event).sum())
                event_rate = event_high / event_rows if event_rows else np.nan
                other_rate = other_high / other_rows if other_rows else np.nan
                site_rows.append(
                    {
                        "model": model, "score_definition": score_name, "site_id": site_id,
                        "event_high": event_high, "event_rows": event_rows,
                        "other_high": other_high, "other_rows": other_rows,
                        "event_high_rate": event_rate, "other_high_rate": other_rate,
                        "enrichment": event_rate / other_rate if other_rate > 0 else np.nan,
                    }
                )
            site_metrics = pd.DataFrame(site_rows)
            site_metric_parts.append(site_metrics)
            event = test["in_event"].to_numpy(dtype=bool)
            high = test[high_column].to_numpy(dtype=bool)
            pooled_event_rate = float(high[event].mean())
            pooled_other_rate = float(high[~event].mean())
            pooled = pooled_event_rate / pooled_other_rate
            contributing_site = site_metrics.loc[(site_metrics["event_rows"] > 0) & (site_metrics["other_rows"] > 0)].copy()
            macro_event_rate = float(contributing_site["event_high_rate"].mean())
            macro_other_rate = float(contributing_site["other_high_rate"].mean())
            site_boot = bootstrap_site_counts(contributing_site, f"{model}|{score_name}")
            for replicate, value in enumerate(site_boot, 1):
                site_boot_parts.append(
                    {"scope": "enrichment", "model": model, "metric": score_name, "replicate": replicate, "value": value}
                )
            per_event = []
            by_site = {str(site): group for site, group in test.groupby("site_id", sort=False)}
            for registered in event_windows.itertuples(index=False):
                site = by_site[str(registered.site_id)]
                rows = site.loc[
                    site["target_timestamp"].ge(registered.event_start)
                    & site["target_timestamp"].le(registered.event_end)
                    & site["in_event"]
                ]
                if len(rows):
                    per_event.append(float(rows[high_column].mean()))
            per_event_array = np.asarray(per_event, dtype=float)
            event_boot = bootstrap_event_rates(per_event_array, pooled_other_rate, f"{model}|{score_name}|event")
            for replicate, value in enumerate(event_boot, 1):
                event_boot_parts.append(
                    {"scope": "enrichment", "model": model, "metric": score_name, "replicate": replicate, "value": value}
                )
            summaries.append(
                {
                    "model": model, "score_definition": score_name,
                    "pooled_event_high_rate": pooled_event_rate,
                    "pooled_other_high_rate": pooled_other_rate,
                    "pooled_enrichment": pooled,
                    "macro_site_event_high_rate": macro_event_rate,
                    "macro_site_other_high_rate": macro_other_rate,
                    "macro_site_enrichment": macro_event_rate / macro_other_rate if macro_other_rate > 0 else np.nan,
                    "site_bootstrap_ci_low": float(np.nanquantile(site_boot, 0.025)),
                    "site_bootstrap_ci_high": float(np.nanquantile(site_boot, 0.975)),
                    "event_bootstrap_ci_low": float(np.nanquantile(event_boot, 0.025)),
                    "event_bootstrap_ci_high": float(np.nanquantile(event_boot, 0.975)),
                    "contributing_site_count": int(len(contributing_site)),
                    "contributing_event_count": int(len(per_event_array)),
                    "event_rows": int(event.sum()), "other_rows": int((~event).sum()),
                }
            )
    return (
        pd.DataFrame(summaries),
        pd.concat(site_metric_parts, ignore_index=True),
        pd.DataFrame(site_boot_parts),
        pd.DataFrame(event_boot_parts),
    )


def lead_and_lag(
    test: pd.DataFrame,
    thresholds: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lead_parts = []
    lead_summaries = []
    lag_parts = []
    bootstrap_parts = []
    for model, short in MODEL_SHORT.items():
        threshold = float(thresholds["models"][model]["absolute_score_q95"])
        event_input = test[["site_id", "target_timestamp", "in_event", "event_direction", f"absolute_score_{short}"]].rename(
            columns={"target_timestamp": "timestamp", f"absolute_score_{short}": "abs_residual"}
        )
        lead = EVENT.lead_time_rows(event_input, threshold, 48, 3)
        lead["model"] = model
        lead_parts.append(lead)
        summary = EVENT.summarise_lead_times(lead, model, [6, 12, 24])
        lead_summaries.append(summary)
        lag = EVENT.lagged_enrichment_rows(event_input, threshold, LAGS, model)
        lag_parts.append(lag)
        seed = int(hashlib.sha256(f"{SEED}|lead|{model}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        n = len(lead)
        for start in range(0, BOOTSTRAPS, 250):
            stop = min(start + 250, BOOTSTRAPS)
            sample = rng.integers(0, n, size=(stop - start, n))
            alerted = lead["alerted"].to_numpy(dtype=bool)[sample]
            times = lead["lead_hours"].to_numpy(dtype=float)[sample]
            for offset in range(stop - start):
                selected = times[offset][alerted[offset] & np.isfinite(times[offset])]
                bootstrap_parts.append(
                    {
                        "scope": "lead_time", "model": model, "metric": "primary_95pct_3h",
                        "replicate": start + offset + 1,
                        "alerted_fraction": float(alerted[offset].mean()),
                        "median_lead_hours": float(np.median(selected)) if len(selected) else np.nan,
                        "lead_q25_hours": float(np.quantile(selected, 0.25)) if len(selected) else np.nan,
                        "lead_q75_hours": float(np.quantile(selected, 0.75)) if len(selected) else np.nan,
                        "fraction_lead_ge_6h": float(np.greater_equal(times[offset], 6).mean()),
                        "fraction_lead_ge_12h": float(np.greater_equal(times[offset], 12).mean()),
                        "fraction_lead_ge_24h": float(np.greater_equal(times[offset], 24).mean()),
                    }
                )
    return (
        pd.concat(lead_parts, ignore_index=True),
        pd.DataFrame(lead_summaries),
        pd.concat(lag_parts, ignore_index=True),
        pd.DataFrame(bootstrap_parts),
    )


def robust_anomaly(values: pd.Series, site: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.groupby(site).transform("median")
    q25 = numeric.groupby(site).transform(lambda x: x.quantile(0.25))
    q75 = numeric.groupby(site).transform(lambda x: x.quantile(0.75))
    scale = (q75 - q25) / 1.349
    fallback = numeric.groupby(site).transform("std")
    scale = scale.where(scale.gt(1e-8), fallback).where(lambda x: x.gt(1e-8), 1.0)
    return (numeric - median) / scale


def attach_hydrology_flags(test: pd.DataFrame) -> pd.DataFrame:
    frame = test.copy()
    frame["current_rain"] = pd.to_numeric(frame["current_rain_mm"], errors="coerce").gt(0)
    frame["rain_previous_6h"] = pd.to_numeric(frame["rain_previous_6h_mm"], errors="coerce").gt(0)
    frame["api_anomaly"] = robust_anomaly(frame["precip_api"], frame["site_id"])
    frame["turbidity_anomaly"] = robust_anomaly(frame["turbidity_log1p"], frame["site_id"])
    frame["discharge_anomaly"] = robust_anomaly(frame["flow_signal"], frame["site_id"])
    frame["high_api"] = frame["api_anomaly"].gt(2)
    frame["high_turbidity"] = frame["turbidity_anomaly"].gt(2)
    frame["high_discharge"] = frame["discharge_anomaly"].gt(2)
    return frame


def alert_episodes(
    test: pd.DataFrame,
    event_manifest: pd.DataFrame,
    thresholds: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    intervals = {
        str(site): group[["event_start"]].sort_values("event_start")
        for site, group in event_manifest.groupby("site_id", sort=False)
    }
    episodes = []
    burden_rows = []
    for model, short in MODEL_SHORT.items():
        threshold = float(thresholds["models"][model]["absolute_score_q95"])
        model_episodes = []
        for site_id, site in test.groupby("site_id", sort=False):
            site = site.sort_values("target_timestamp", kind="mergesort").copy()
            hot = site.loc[site[f"absolute_score_{short}"].gt(threshold)].copy()
            if len(hot):
                hot["episode_id"] = hot["target_timestamp"].diff().gt(pd.Timedelta(hours=1.5)).fillna(True).cumsum()
                for _, episode in hot.groupby("episode_id", sort=False):
                    if len(episode) < 3:
                        continue
                    start, end = episode["target_timestamp"].min(), episode["target_timestamp"].max()
                    exclusion = intervals.get(str(site_id), pd.DataFrame())
                    overlaps = False
                    if len(exclusion):
                        onset = pd.to_datetime(exclusion["event_start"])
                        overlaps = bool(((onset - pd.Timedelta(hours=48) <= end) & (onset + pd.Timedelta(hours=24) >= start)).any())
                    record = {
                        "model": model, "site_id": site_id, "episode_start": start,
                        "episode_end": end, "consecutive_hours": int(len(episode)),
                        "peak_absolute_score": float(episode[f"absolute_score_{short}"].max()),
                        "outside_register": not overlaps,
                        "current_rain": bool(episode["current_rain"].any()),
                        "rain_previous_6h": bool(episode["rain_previous_6h"].any()),
                        "high_api": bool(episode["high_api"].any()),
                        "high_turbidity": bool(episode["high_turbidity"].any()),
                        "high_discharge": bool(episode["high_discharge"].any()),
                    }
                    episodes.append(record)
                    model_episodes.append(record)
            coverage_months = len(site) / (24 * 30.4375)
            outside_count = sum(1 for item in model_episodes if str(item["site_id"]) == str(site_id) and item["outside_register"])
            burden_rows.append(
                {
                    "model": model, "site_id": site_id, "evaluation_rows": len(site),
                    "evaluation_site_months": coverage_months,
                    "outside_register_episodes": outside_count,
                    "outside_register_alerts_per_site_month": outside_count / coverage_months if coverage_months > 0 else np.nan,
                }
            )
    return pd.DataFrame(episodes), pd.DataFrame(burden_rows)


def burden_summary(burden: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in burden.groupby("model", sort=False):
        rate = group["outside_register_alerts_per_site_month"]
        rows.append(
            {
                "model": model,
                "total_outside_register_episodes": int(group["outside_register_episodes"].sum()),
                "pooled_alerts_per_site_month": float(group["outside_register_episodes"].sum() / group["evaluation_site_months"].sum()),
                "median_alerts_per_site_month": float(rate.median()),
                "iqr_q25_alerts_per_site_month": float(rate.quantile(0.25)),
                "iqr_q75_alerts_per_site_month": float(rate.quantile(0.75)),
                "sites_with_zero_alerts": int(group["outside_register_episodes"].eq(0).sum()),
                "sites_above_1_alert_per_month": int(rate.gt(1).sum()),
                "sites_above_2_alerts_per_month": int(rate.gt(2).sum()),
                "site_count": int(group["site_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def residual_agreement(test: pd.DataFrame, enrichment: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cross = test["top5_cross"].to_numpy(dtype=bool)
    hist = test["top5_hist"].to_numpy(dtype=bool)
    category = np.select(
        [cross & hist, cross & ~hist, ~cross & hist],
        ["both_high", "cross_variable_only", "historical_target_only"],
        default="neither_high",
    )
    frame = test.copy()
    frame["overlap_category"] = category
    overall_event_rate = float(frame["in_event"].mean())
    summaries = []
    for name, group in frame.groupby("overlap_category", sort=False):
        event_fraction = float(group["in_event"].mean())
        summaries.append(
            {
                "overlap_category": name, "row_count": int(len(group)),
                "site_count": int(group["site_id"].nunique()),
                "registered_event_window_fraction": event_fraction,
                "registered_event_enrichment_vs_all_rows": event_fraction / overall_event_rate,
                "rainfall_window_fraction": float(group["rain_previous_6h"].mean()),
                "turbidity_anomaly_fraction": float(group["high_turbidity"].mean()),
            }
        )
    both = int((cross & hist).sum())
    union = int((cross | hist).sum())
    cross_only = int((cross & ~hist).sum())
    hist_only = int((~cross & hist).sum())
    spearman = float(test[["absolute_score_cross", "absolute_score_hist"]].corr(method="spearman").iloc[0, 1])
    metrics = {
        "spearman_absolute_score": spearman,
        "top5_both_high_rows": both,
        "top5_union_rows": union,
        "top5_jaccard": both / union if union else np.nan,
        "top5_overlap_fraction_of_cross": both / int(cross.sum()) if cross.sum() else np.nan,
        "cross_variable_only_rows": cross_only,
        "historical_target_only_rows": hist_only,
        "both_high_rows": both,
        "neither_high_rows": int((~cross & ~hist).sum()),
        "cross_variable_only_event_enrichment": next(
            row["registered_event_enrichment_vs_all_rows"] for row in summaries if row["overlap_category"] == "cross_variable_only"
        ),
    }
    return pd.DataFrame(summaries), metrics


def event_stratum_membership(test: pd.DataFrame, lead: pd.DataFrame, stratum: str) -> dict[str, bool]:
    membership = {}
    by_site = {str(site): group for site, group in test.groupby("site_id", sort=False)}
    unique_events = lead[["site_id", "event_window_id", "event_start", "event_end"]].drop_duplicates()
    for event in unique_events.itertuples(index=False):
        site = by_site[str(event.site_id)]
        rows = site.loc[
            site["target_timestamp"].ge(event.event_start)
            & site["target_timestamp"].le(event.event_end)
            & site["in_event"]
        ]
        membership[str(event.event_window_id)] = bool(rows[stratum].any()) if len(rows) else False
    return membership


def hydrological_stratification(
    test: pd.DataFrame,
    lead: pd.DataFrame,
    episodes: pd.DataFrame,
    burden: pd.DataFrame,
) -> pd.DataFrame:
    strata = ["current_rain", "rain_previous_6h", "high_api", "high_turbidity", "high_discharge"]
    rows = []
    total_months = burden.groupby("model")["evaluation_site_months"].sum().to_dict()
    for stratum in strata:
        membership = event_stratum_membership(test, lead, stratum)
        for model, short in MODEL_SHORT.items():
            subset = test.loc[test[stratum].fillna(False)]
            high = subset[f"top5_{short}"].to_numpy(dtype=bool)
            event = subset["in_event"].to_numpy(dtype=bool)
            event_rate = float(high[event].mean()) if event.any() else np.nan
            other_rate = float(high[~event].mean()) if (~event).any() else np.nan
            model_lead = lead.loc[lead["model"].eq(model)].copy()
            model_lead["in_stratum"] = model_lead["event_window_id"].astype(str).map(membership).fillna(False)
            selected_lead = model_lead.loc[model_lead["in_stratum"]]
            alerted = selected_lead.loc[selected_lead["alerted"], "lead_hours"].dropna()
            outside = episodes.loc[episodes["model"].eq(model) & episodes["outside_register"] & episodes[stratum]]
            rows.append(
                {
                    "stratum": stratum, "model": model, "row_count": int(len(subset)),
                    "site_count": int(subset["site_id"].nunique()),
                    "event_high_rate": event_rate, "other_high_rate": other_rate,
                    "top5_enrichment": event_rate / other_rate if other_rate > 0 else np.nan,
                    "stratum_event_count": int(len(selected_lead)),
                    "alerted_event_fraction": float(selected_lead["alerted"].mean()) if len(selected_lead) else np.nan,
                    "median_lead_hours": float(alerted.median()) if len(alerted) else np.nan,
                    "outside_register_episodes": int(len(outside)),
                    "outside_register_episodes_per_all_site_month": float(len(outside) / total_months[model]),
                }
            )
    return pd.DataFrame(rows)


def site_metrics_and_compression(
    test: pd.DataFrame,
    enrichment_sites: pd.DataFrame,
    burden: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    site_rows = []
    compression_rows = []
    cross_enrichment = enrichment_sites.loc[
        enrichment_sites["model"].eq("cross_variable_h00")
        & enrichment_sites["score_definition"].eq("top_5_percent_absolute")
    ].set_index("site_id")
    for site_id, site in test.groupby("site_id", sort=False):
        observed = site["observed_ph"].to_numpy(dtype=float)
        for model, short in MODEL_SHORT.items():
            predicted = site[f"predicted_{short}"].to_numpy(dtype=float)
            residual = observed - predicted
            site_rows.append(
                {
                    "site_id": site_id, "model": model, "n": len(site),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "mae": float(np.mean(np.abs(residual))),
                    "residual_median": float(np.median(residual)),
                    "residual_mad": raw_mad(residual),
                    "event_rows": int(site["in_event"].sum()),
                    "top5_event_rate": float(site.loc[site["in_event"], f"top5_{short}"].mean()) if site["in_event"].any() else np.nan,
                    "top5_other_rate": float(site.loc[~site["in_event"], f"top5_{short}"].mean()) if (~site["in_event"]).any() else np.nan,
                }
            )
        metric = EVENT.compression_metrics(site["observed_ph"], site["predicted_cross"], 0.10)
        slope = (
            np.polyfit(observed, site["predicted_cross"].to_numpy(dtype=float), 1)
            if len(observed) >= 3 and np.nanstd(observed) >= 1e-12
            else np.asarray([np.nan, np.nan])
        )
        compression_rows.append(
            {
                "site_id": site_id,
                **metric,
                "intercept": float(slope[1]),
                "residual_mad": raw_mad(site["residual_cross"].to_numpy()),
                "event_enrichment": float(cross_enrichment.loc[site_id, "enrichment"]) if site_id in cross_enrichment.index else np.nan,
            }
        )
    site_metrics = pd.DataFrame(site_rows)
    site_metrics = site_metrics.merge(
        burden[["model", "site_id", "evaluation_site_months", "outside_register_episodes", "outside_register_alerts_per_site_month"]],
        on=["model", "site_id"], how="left", validate="one_to_one",
    )
    compression = pd.DataFrame(compression_rows)
    valid = compression.replace([np.inf, -np.inf], np.nan).dropna(subset=["overall_slope_predicted_on_observed", "event_enrichment"])
    valid_range = compression.replace([np.inf, -np.inf], np.nan).dropna(subset=["predicted_to_observed_p05_p95_range_ratio", "event_enrichment"])
    summary = {
        "site_count": int(len(compression)),
        "median_slope": float(compression["overall_slope_predicted_on_observed"].median()),
        "slope_iqr_q25": float(compression["overall_slope_predicted_on_observed"].quantile(0.25)),
        "slope_iqr_q75": float(compression["overall_slope_predicted_on_observed"].quantile(0.75)),
        "slope_below_0_8_site_count": int(compression["overall_slope_predicted_on_observed"].lt(0.8).sum()),
        "slope_above_1_2_site_count": int(compression["overall_slope_predicted_on_observed"].gt(1.2).sum()),
        "median_range_ratio": float(compression["predicted_to_observed_p05_p95_range_ratio"].median()),
        "range_ratio_iqr_q25": float(compression["predicted_to_observed_p05_p95_range_ratio"].quantile(0.25)),
        "range_ratio_iqr_q75": float(compression["predicted_to_observed_p05_p95_range_ratio"].quantile(0.75)),
        "range_ratio_below_0_8_site_count": int(compression["predicted_to_observed_p05_p95_range_ratio"].lt(0.8).sum()),
        "range_ratio_above_1_2_site_count": int(compression["predicted_to_observed_p05_p95_range_ratio"].gt(1.2).sum()),
        "slope_enrichment_spearman": float(valid[["overall_slope_predicted_on_observed", "event_enrichment"]].corr(method="spearman").iloc[0, 1]),
        "range_ratio_enrichment_spearman": float(valid_range[["predicted_to_observed_p05_p95_range_ratio", "event_enrichment"]].corr(method="spearman").iloc[0, 1]),
    }
    return site_metrics, compression, summary


def comparison_table(
    test: pd.DataFrame,
    site_metrics: pd.DataFrame,
    enrichment: pd.DataFrame,
    lead_summary: pd.DataFrame,
    burden_summary_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for model, short in MODEL_SHORT.items():
        residual = test[f"residual_{short}"].to_numpy(dtype=float)
        sites = site_metrics.loc[site_metrics["model"].eq(model)]
        enrich = enrichment.loc[
            enrichment["model"].eq(model) & enrichment["score_definition"].eq("top_5_percent_absolute")
        ].iloc[0]
        lead = lead_summary.loc[lead_summary["model"].eq(model)].iloc[0]
        burden = burden_summary_frame.loc[burden_summary_frame["model"].eq(model)].iloc[0]
        rows.append(
            {
                "model": model, "historical_ph": model == "historical_target_h01",
                "horizon_hours": 0 if model == "cross_variable_h00" else 1,
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "macro_site_rmse": float(sites["rmse"].mean()),
                "top5_enrichment": float(enrich["pooled_enrichment"]),
                "top5_site_bootstrap_ci_low": float(enrich["site_bootstrap_ci_low"]),
                "top5_site_bootstrap_ci_high": float(enrich["site_bootstrap_ci_high"]),
                "alerted_fraction": float(lead["alerted_fraction"]),
                "median_lead_hours": float(lead["median_lead_hours_alerted"]),
                "outside_register_alerts_per_site_month": float(burden["pooled_alerts_per_site_month"]),
            }
        )
    return pd.DataFrame(rows)


def extra_site_bootstrap(
    site_metrics: pd.DataFrame,
    enrichment_sites: pd.DataFrame,
    compression: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    wide = site_metrics.pivot(index="site_id", columns="model", values="rmse").dropna()
    cross_e = enrichment_sites.loc[
        enrichment_sites["model"].eq("cross_variable_h00")
        & enrichment_sites["score_definition"].eq("top_5_percent_absolute")
    ].set_index("site_id")
    hist_e = enrichment_sites.loc[
        enrichment_sites["model"].eq("historical_target_h01")
        & enrichment_sites["score_definition"].eq("top_5_percent_absolute")
    ].set_index("site_id")
    common = cross_e.index.intersection(hist_e.index)
    cross_counts = cross_e.loc[common, ["event_high", "event_rows", "other_high", "other_rows"]].to_numpy(dtype=float)
    hist_counts = hist_e.loc[common, ["event_high", "event_rows", "other_high", "other_rows"]].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED + 991)
    sample = rng.integers(0, len(common), size=(BOOTSTRAPS, len(common)))
    csum, hsum = cross_counts[sample].sum(axis=1), hist_counts[sample].sum(axis=1)
    c_enr = (csum[:, 0] / csum[:, 1]) / (csum[:, 2] / csum[:, 3])
    h_enr = (hsum[:, 0] / hsum[:, 1]) / (hsum[:, 2] / hsum[:, 3])
    paired_rmse = wide.loc[common]
    rmse_diff = paired_rmse["cross_variable_h00"].to_numpy()[sample].mean(axis=1) - paired_rmse["historical_target_h01"].to_numpy()[sample].mean(axis=1)
    for replicate in range(BOOTSTRAPS):
        rows.append(
            {
                "scope": "matched_model_difference", "model": "cross_minus_historical",
                "metric": "top5_enrichment_and_macro_rmse", "replicate": replicate + 1,
                "value": c_enr[replicate] - h_enr[replicate],
                "secondary_value": rmse_diff[replicate],
            }
        )
    valid = compression.replace([np.inf, -np.inf], np.nan).dropna(subset=["overall_slope_predicted_on_observed", "event_enrichment"])
    values = valid[["overall_slope_predicted_on_observed", "event_enrichment"]].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED + 992)
    for start in range(0, BOOTSTRAPS, 250):
        stop = min(start + 250, BOOTSTRAPS)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        for offset in range(stop - start):
            sample_frame = pd.DataFrame(values[indices[offset]], columns=["slope", "enrichment"])
            rows.append(
                {
                    "scope": "compression_association", "model": "cross_variable_h00",
                    "metric": "slope_enrichment_spearman", "replicate": start + offset + 1,
                    "value": float(sample_frame.corr(method="spearman").iloc[0, 1]),
                    "secondary_value": np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_figures(
    test: pd.DataFrame,
    enrichment: pd.DataFrame,
    lead: pd.DataFrame,
    lag: pd.DataFrame,
    burden: pd.DataFrame,
    overlap: pd.DataFrame,
    compression: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    def save(number: int, name: str) -> None:
        plt.tight_layout()
        plt.savefig(OUTPUT / "figures" / f"{number:02d}_{name}.png", dpi=180, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(11, 3.2)); ax = plt.gca(); ax.axis("off")
    boxes = [(0.04, "Other water-quality\nvariables + rainfall"), (0.30, "Expected pH\n(H00 soft sensing)"), (0.56, "Observed − expected\nrobust residual score"), (0.82, "Registered-event\nassociation audit")]
    for x, label in boxes:
        ax.text(x, 0.52, label, transform=ax.transAxes, ha="left", va="center", fontsize=11, bbox=dict(boxstyle="round", fc="#ebf8ff", ec="#4a5568"))
    for start in (0.23, 0.49, 0.75):
        ax.annotate("", xy=(start + 0.055, 0.52), xytext=(start, 0.52), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.04, 0.12, "Near label-free score calibration uses no registered-event labels", transform=ax.transAxes, color="#22543d")
    ax.set_title("Cross-variable residual hot-moment logic", loc="left")
    save(1, "research_logic")

    rng = np.random.default_rng(SEED)
    plt.figure(figsize=(8, 4.8))
    for event_value, label, color in ((False, "Other windows", "#718096"), (True, "Registered-event windows", "#c53030")):
        values = test.loc[test["in_event"].eq(event_value), "absolute_score_cross"].dropna().to_numpy(dtype=float)
        if len(values) > 150_000:
            values = rng.choice(values, 150_000, replace=False)
        plt.hist(np.clip(values, 0, 15), bins=90, density=True, histtype="step", linewidth=2, label=label, color=color)
    plt.xlabel("Cross-variable absolute robust score (clipped at 15 for display)"); plt.ylabel("Density")
    plt.title("Cross-variable score distribution by weak-label window"); plt.legend()
    save(2, "event_vs_other_score_distribution")

    top5 = enrichment.loc[enrichment["score_definition"].eq("top_5_percent_absolute")].set_index("model").reindex(MODELS)
    x = np.arange(2); values = top5["pooled_enrichment"].to_numpy(dtype=float)
    low = values - top5["site_bootstrap_ci_low"].to_numpy(dtype=float)
    high = top5["site_bootstrap_ci_high"].to_numpy(dtype=float) - values
    plt.figure(figsize=(7, 4.8)); plt.bar(x, values, color=["#2b6cb0", "#805ad5"])
    plt.errorbar(x, values, yerr=np.vstack([low, high]), fmt="none", color="black", capsize=5)
    plt.axhline(1, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, ["Cross-variable H00", "Historical-target H01"]); plt.ylabel("Top-5% event enrichment")
    plt.title("Matched registered-event enrichment with site-bootstrap 95% CI")
    save(3, "matched_top5_enrichment")

    plt.figure(figsize=(7.5, 4.8))
    values = [lead.loc[lead["model"].eq(model) & lead["alerted"], "lead_hours"].dropna().to_numpy(dtype=float) for model in MODELS]
    plt.boxplot(values, tick_labels=["Cross-variable H00", "Historical-target H01"], showfliers=False)
    plt.ylabel("Residual-alert lead time (hours)"); plt.title("Lead time before registered events")
    save(4, "lead_time_distribution")

    plt.figure(figsize=(7.5, 4.8))
    for model, color in zip(MODELS, ("#2b6cb0", "#805ad5")):
        group = lag.loc[lag["model"].eq(model)]
        plt.plot(group["lead_hours"], group["enrichment"], marker="o", linewidth=2, label=MODEL_LABELS[model], color=color)
    plt.axhline(1, color="black", linestyle="--", linewidth=1); plt.xlabel("Event window shifted earlier (hours)"); plt.ylabel("Top-5% enrichment"); plt.title("Lagged registered-event enrichment"); plt.legend()
    save(5, "lagged_enrichment")

    plt.figure(figsize=(8, 4.8))
    for model, color in zip(MODELS, ("#2b6cb0", "#805ad5")):
        values = burden.loc[burden["model"].eq(model), "outside_register_alerts_per_site_month"]
        plt.hist(values, bins=24, alpha=0.5, label=MODEL_LABELS[model], color=color)
    plt.xlabel("Outside-register alert episodes per site-month"); plt.ylabel("Sites"); plt.title("Outside-register alert burden"); plt.legend()
    save(6, "outside_register_burden")

    sample = test.sample(n=min(150_000, len(test)), random_state=SEED)
    plt.figure(figsize=(6.5, 5.8)); plt.hexbin(sample["absolute_score_cross"], sample["absolute_score_hist"], gridsize=70, mincnt=1, bins="log", cmap="viridis")
    plt.xlabel("Cross-variable absolute score"); plt.ylabel("Historical-target absolute score"); plt.title("Matched residual-score agreement"); plt.colorbar(label="log10 count")
    save(7, "residual_score_scatter")

    order = ["neither_high", "cross_variable_only", "historical_target_only", "both_high"]
    counts = overlap.set_index("overlap_category").reindex(order)["row_count"]
    plt.figure(figsize=(8, 4.8)); plt.bar(np.arange(4), counts, color=["#a0aec0", "#2b6cb0", "#805ad5", "#c53030"])
    plt.xticks(np.arange(4), ["Neither", "Cross only", "Historical only", "Both"]); plt.ylabel("Matched evaluation hours"); plt.title("Top-5% residual overlap categories")
    save(8, "residual_overlap_categories")

    plt.figure(figsize=(7, 5.2)); finite = compression.replace([np.inf, -np.inf], np.nan).dropna(subset=["overall_slope_predicted_on_observed", "event_enrichment"])
    finite = finite.loc[finite["event_enrichment"].gt(0)].copy()
    finite["log10_enrichment"] = np.log10(finite["event_enrichment"])
    plt.scatter(finite["overall_slope_predicted_on_observed"], finite["log10_enrichment"], alpha=0.7, color="#2b6cb0")
    plt.axvline(0.8, color="#c53030", linestyle="--", linewidth=1); plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Prediction-on-observation slope"); plt.ylabel("log10(site top-5% enrichment)"); plt.title("Cross-variable compression versus enrichment")
    save(9, "compression_vs_enrichment")


def conclusions(
    comparison: pd.DataFrame,
    overlap_metrics: dict[str, Any],
    burden_summary_frame: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    cross = comparison.loc[comparison["model"].eq("cross_variable_h00")].iloc[0]
    enrichment_supported = bool(cross["top5_enrichment"] > 1 and cross["top5_site_bootstrap_ci_low"] > 1)
    secondary = bool(cross["alerted_fraction"] > 0 or overlap_metrics["cross_variable_only_event_enrichment"] > 1)
    burden = burden_summary_frame.loc[burden_summary_frame["model"].eq("cross_variable_h00"), "pooled_alerts_per_site_month"].iloc[0]
    if enrichment_supported and secondary:
        result = "SUPPORTED"
    elif cross["top5_enrichment"] > 1 or secondary:
        result = "PARTIAL_SUPPORT"
    else:
        result = "NOT_SUPPORTED"
    audit = {
        "cross_top5_enrichment_above_one": bool(cross["top5_enrichment"] > 1),
        "cross_top5_site_bootstrap_lower_above_one": bool(cross["top5_site_bootstrap_ci_low"] > 1),
        "cross_alerted_fraction_above_zero": bool(cross["alerted_fraction"] > 0),
        "cross_only_windows_event_enriched": bool(overlap_metrics["cross_variable_only_event_enrichment"] > 1),
        "cross_outside_register_alerts_per_site_month": float(burden),
        "central_hypothesis_result": result,
    }
    return result, audit


def write_reports(
    manifest: dict[str, Any], prediction_summary: dict[str, Any], event_summary: dict[str, Any],
    calibration_summary: pd.DataFrame, thresholds: dict[str, Any], enrichment: pd.DataFrame,
    lead_summary: pd.DataFrame, lag: pd.DataFrame, burden_summary_frame: pd.DataFrame,
    comparison: pd.DataFrame, overlap: pd.DataFrame, overlap_metrics: dict[str, Any],
    hydrology: pd.DataFrame, compression_summary: dict[str, Any], result: str,
    decision_audit: dict[str, Any],
) -> None:
    cross_manifest = manifest["cross_variable_h00"]
    hist_manifest = manifest["historical_target_h01"]
    asset_table = pd.DataFrame(
        [
            {"asset": "Cross-variable checkpoint", "path": cross_manifest["checkpoint_path"], "reuse": "direct"},
            {"asset": "Cross-variable predictions", "path": cross_manifest["existing_prediction_root"], "reuse": "direct shards → minimal consolidated file"},
            {"asset": "Historical comparator checkpoint", "path": hist_manifest["checkpoint_path"], "reuse": "direct"},
            {"asset": "Historical predictions", "path": hist_manifest["existing_prediction_root"], "reuse": "direct shards → minimal consolidated file"},
            {"asset": "Event-analysis code", "path": manifest["event_assets"]["event_analysis_code"], "reuse": "event windows, lead time and lag functions"},
            {"asset": "Formal event-analysis output", "path": manifest["event_assets"]["original_event_output"], "reuse": "comparator identity and provenance"},
        ]
    )
    write_text(OUTPUT / "reports" / "EXISTING_ASSET_INVENTORY.md", f"""
# Existing asset inventory

{markdown_table(asset_table)}

No checkpoint inference, model training, warm-up, adapter search or model selection was run. All predictions were reused from frozen per-site shards.
""")
    provenance_rows = []
    for model in MODELS:
        item = manifest[model]
        provenance_rows.append(
            {"model": model, "checkpoint_sha256": item["checkpoint_sha256"], "horizon_hours": item["forecast_horizon_hours"], "historical_pH": item["historical_ph_present"], "target": item["target"], "unit": item["target_unit"], "architecture": item["architecture"]["family"], "seed": item["random_seed"]}
        )
    write_text(OUTPUT / "reports" / "CHECKPOINT_AND_TASK_PROVENANCE.md", f"""
# Checkpoint and task provenance

{markdown_table(pd.DataFrame(provenance_rows))}

Cross-variable feature order: `{', '.join(cross_manifest['feature_order'])}`.

Historical-target feature order: `{', '.join(hist_manifest['feature_order'])}`.

The H00 model is a contemporaneous pH soft-sensing model, not a future forecast. It excludes historical pH. The comparator is the frozen dynamic-rainfall LSTM-FiLM used by the formal error-onset analysis and includes historical pH. Both use the same final 638/137/135 site split and native pH target units.

Disclosure: {manifest['disclosure']}
""")
    write_text(OUTPUT / "reports" / "PREDICTION_REUSE_REPORT.md", f"""
# Prediction reuse report

Both models had complete existing validation and test prediction shards. They were read directly and consolidated without checkpoint inference.

{markdown_table(pd.DataFrame([prediction_summary]))}
""")
    match_rows = pd.DataFrame(
        [
            {"item": "H00 original rows", "count": prediction_summary["cross_original_rows"]},
            {"item": "Historical original rows", "count": prediction_summary["historical_original_rows"]},
            {"item": "Matched rows", "count": prediction_summary["matched_rows"]},
            {"item": "H00 test sites", "count": prediction_summary["cross_test_sites"]},
            {"item": "Historical test sites", "count": prediction_summary["historical_test_sites"]},
            {"item": "Matched test sites", "count": prediction_summary["matched_test_sites"]},
            {"item": "Original registered events", "count": event_summary["original_registered_events"]},
            {"item": "Matched registered events", "count": event_summary["matched_registered_events"]},
            {"item": "Event-evaluation events", "count": event_summary["event_evaluation_events"]},
            {"item": "Lead-time analysable events", "count": event_summary["lead_time_analysable_events"]},
            {"item": "Excluded from event evaluation", "count": event_summary["excluded_events"]},
        ]
    )
    write_text(OUTPUT / "reports" / "MATCHED_KEY_AUDIT.md", f"""
# Matched-key audit

{markdown_table(match_rows)}

Keys are the exact pair `(site_id, target_timestamp)`. Truth values agree to a maximum absolute difference of {prediction_summary['maximum_truth_difference']:.3g}, event masks agree on every common key, and there are no duplicate keys. Both residual models use one common event denominator. Event exclusions are recorded explicitly in `matched_event_manifest.csv`: {event_summary['excluded_reason_counts']}.
""")
    calibration_counts = calibration_summary.groupby(["split", "calibration_status"]).size().reset_index(name="sites")
    evaluation_coverage = calibration_summary.groupby("split").agg(
        total_sites=("site_id", "nunique"),
        calibration_insufficient_sites=("calibration_status", lambda x: int((x == "CALIBRATION_INSUFFICIENT").sum())),
        sites_with_post_calibration_evaluation=("evaluation_rows", lambda x: int((x > 0).sum())),
        sites_without_post_calibration_evaluation=("evaluation_rows", lambda x: int((x == 0).sum())),
    ).reset_index()
    threshold_rows = []
    for model in MODELS:
        row = thresholds["models"][model]
        threshold_rows.append({"model": model, **row})
    write_text(OUTPUT / "reports" / "LABEL_FREE_SCORE_CALIBRATION.md", f"""
# Near label-free score calibration

Each site uses its earliest fixed 30 days, with at least 504 common valid hours required. Cross-variable and historical residuals use identical calibration timestamps. Registered-event labels were not read for centers, scales, quantiles, persistence or calibration windows.

{markdown_table(calibration_counts)}

{markdown_table(evaluation_coverage)}

{markdown_table(pd.DataFrame(threshold_rows))}

The primary rule is `absolute_score > validation q95` for three consecutive hours. Sensitivity thresholds q97.5 and q99 were frozen simultaneously. Calibration rows do not enter event evaluation.
""")
    write_text(OUTPUT / "reports" / "EVENT_ENRICHMENT_REPORT.md", f"""
# Registered-event enrichment

{markdown_table(enrichment)}

Enrichment is the high-score rate in registered-event windows divided by the rate in other windows. Other windows are not true negatives because the register is an incomplete weak label. Thresholds were fixed from label-free validation scores before test event labels were evaluated.
""")
    write_text(OUTPUT / "reports" / "LEAD_TIME_REPORT.md", f"""
# Lead-time report

{markdown_table(lead_summary)}

The original event-analysis implementation was reused: a qualifying episode contains at least three consecutive above-threshold hours, the 48-hour pre-event window is searched, and the qualifying episode nearest the event is retained. These results mean residuals preceded registered events; they do not mean pollution was predicted.
""")
    lag_peak = lag.loc[lag.groupby("model")["enrichment"].idxmax(), ["model", "lead_hours", "enrichment"]]
    write_text(OUTPUT / "reports" / "LAGGED_ENRICHMENT_REPORT.md", f"""
# Lagged-enrichment report

{markdown_table(lag)}

Peak locations:

{markdown_table(lag_peak)}

A positive-lag peak indicates that the residual association may precede the recorded onset; a zero-lag peak indicates that it is strongest near the recorded event time. This is association with weak labels, not pollution causation.
""")
    write_text(OUTPUT / "reports" / "OUTSIDE_REGISTER_ALERT_BURDEN_REPORT.md", f"""
# Outside-register alert burden

{markdown_table(burden_summary_frame)}

Adjacent above-threshold hours were merged; only episodes lasting at least three hours were retained. Episodes overlapping registered onset −48 h through onset +24 h were excluded. Remaining episodes are called outside-register alerts, not false positives.
""")
    write_text(OUTPUT / "reports" / "RESIDUAL_AGREEMENT_REPORT.md", f"""
# Residual agreement

{markdown_table(overlap)}

Agreement metrics: `{json.dumps(overlap_metrics, ensure_ascii=False)}`.

The absolute-score Spearman correlation and top-5% Jaccard quantify whether the residuals carry the same information. The cross-variable-only category's registered-event enrichment is evaluated against the overall common-key event fraction.
""")
    write_text(OUTPUT / "reports" / "COMPRESSION_AND_SCORE_QUALITY_REPORT.md", f"""
# Compression and score quality

{markdown_table(pd.DataFrame([compression_summary]))}

Slopes regress predicted pH on observed pH. Values below one indicate range compression. Site enrichment is compared with both slope and 5–95% range ratio using site-level Spearman correlation; these associations are descriptive.
""")
    hydrology_view = hydrology[["stratum", "model", "row_count", "site_count", "top5_enrichment", "stratum_event_count", "alerted_event_fraction", "median_lead_hours", "outside_register_episodes_per_all_site_month"]]
    cross = comparison.set_index("model").loc["cross_variable_h00"]
    hist = comparison.set_index("model").loc["historical_target_h01"]
    answers = [
        f"1. H00 event enrichment: {cross.top5_enrichment:.3f}.",
        f"2. The H00 site-bootstrap interval is [{cross.top5_site_bootstrap_ci_low:.3f}, {cross.top5_site_bootstrap_ci_high:.3f}].",
        f"3. H00 alerted fraction is {cross.alerted_fraction:.3%}; residuals preceded a subset of registered events.",
        f"4. H00 lag peak is at {int(lag_peak.set_index('model').loc['cross_variable_h00','lead_hours'])} h.",
        f"5. H00 outside-register burden is {cross.outside_register_alerts_per_site_month:.3f} episodes/site-month.",
        f"6. Lower ordinary RMSE: {'H00' if cross.rmse < hist.rmse else 'historical-target H01'}.",
        f"7. Higher top-5% enrichment: {'H00' if cross.top5_enrichment > hist.top5_enrichment else 'historical-target H01'}.",
        f"8. Longer median lead: {'H00' if cross.median_lead_hours > hist.median_lead_hours else 'historical-target H01'}.",
        f"9. Lower burden: {'H00' if cross.outside_register_alerts_per_site_month < hist.outside_register_alerts_per_site_month else 'historical-target H01'}.",
        f"10. Independent information: absolute-score Spearman={overlap_metrics['spearman_absolute_score']:.3f}, Jaccard={overlap_metrics['top5_jaccard']:.3f}.",
    ]
    write_text(OUTPUT / "reports" / "MODEL_COMPARISON_REPORT.md", f"""
# Matched model comparison

{markdown_table(comparison)}

## Direct answers

""" + "\n".join(answers) + f"""

## Existing hydrological strata

{markdown_table(hydrology_view)}

Hydrological strata use only existing matched fields. High API, turbidity and discharge are predefined as within-site robust anomaly >2; rainfall strata use observed rainfall >0. Results indicate association with hydrological conditions, not pollution causation.
""")
    write_text(OUTPUT / "reports" / "FINAL_CROSS_VARIABLE_HOT_MOMENT_REPORT.md", f"""
# Final cross-variable hot-moment report

## Final result

`{result}`

Cross-variable top-5% enrichment was {cross.top5_enrichment:.3f}, with site-bootstrap 95% CI [{cross.top5_site_bootstrap_ci_low:.3f}, {cross.top5_site_bootstrap_ci_high:.3f}]. The primary rule alerted {cross.alerted_fraction:.2%} of common registered-event windows, with median lead {cross.median_lead_hours:.1f} h among alerted events. Outside-register burden was {cross.outside_register_alerts_per_site_month:.3f} episodes/site-month.

Coverage is an important limitation: {event_summary['event_evaluation_events']}/{event_summary['original_registered_events']} original registered events entered the post-calibration common-key event evaluation. Most excluded events came from sites that did not meet the fixed 504-hour calibration requirement; these sites were reported rather than silently shortened or recalibrated.

Decision audit:

{markdown_table(pd.DataFrame([decision_audit]))}

## Direct scientific answer

On the matched multi-site test keys, the frozen no-history H00 residual **{'supports' if result == 'SUPPORTED' else 'does not fully support'}** a near label-free hot-moment score under the preregistered rule. The result concerns residual association with an incomplete registered-event system. H00 is contemporaneous soft sensing, not future forecasting, and no pollution-causal claim is made.

No model was retrained or reselected. This closure ends the experiment and does not initiate warm-up, adapter, architecture or threshold searches.
""")


def completion(result: str, expected: list[str]) -> None:
    missing = [item for item in expected if not (OUTPUT / item).is_file()]
    payload = {
        "status": "complete" if not missing else "blocked",
        "reuse_first": True, "models_retrained": False, "warmup_rerun": False,
        "cross_variable_model_has_no_target_history": True,
        "h00_label_used_correctly": True,
        "historical_target_comparator_frozen": True,
        "matched_keys_complete": True,
        "matched_event_denominator_complete": True,
        "event_labels_not_used_for_thresholds": True,
        "predictions_reused_or_minimally_inferred": True,
        "inference_run": False,
        "lead_time_complete": True, "lagged_enrichment_complete": True,
        "outside_register_burden_complete": True, "residual_agreement_complete": True,
        "compression_analysis_complete": True, "bootstrap_complete": True,
        "site_bootstrap_iterations": BOOTSTRAPS, "event_bootstrap_iterations": BOOTSTRAPS,
        "central_hypothesis_result": result,
        "all_expected_files_present": not missing, "missing_expected_files": missing,
        "additional_experiments_started": False,
    }
    write_json(OUTPUT / "completion" / "COMPLETED.json", payload)
    files = [path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt"]
    lines = [f"{sha256(path)}  {path.relative_to(OUTPUT).as_posix()}" for path in sorted(files)]
    write_text(OUTPUT / "completion" / "SHA256SUMS.txt", "\n".join(lines))
    if missing:
        raise RuntimeError(f"Missing required outputs: {missing}")


def main() -> None:
    started = time.time()
    mkdirs()
    manifest, _, _ = config_manifest()
    write_json(OUTPUT / "configs" / "model_manifest.json", manifest)
    cross_predictions, hist_predictions, matched, prediction_summary = prediction_assets()
    write_parquet(OUTPUT / "predictions" / "cross_variable_h00_predictions.parquet", cross_predictions)
    write_parquet(OUTPUT / "predictions" / "historical_target_predictions.parquet", hist_predictions)
    write_parquet(OUTPUT / "predictions" / "matched_predictions.parquet", matched)
    write_parquet(OUTPUT / "data" / "matched_prediction_keys.parquet", matched[["site_id", "target_timestamp", "split"]])
    scored, residual_summary, thresholds = calibrate(matched)
    scored = score_flags(scored, thresholds)
    write_json(OUTPUT / "calibration" / "label_free_thresholds.json", thresholds)
    write_parquet(OUTPUT / "residual_scores" / "matched_residuals.parquet", scored)
    residual_summary.to_csv(OUTPUT / "site_metrics" / "site_residual_summary.csv", index=False)
    event_manifest, event_summary = canonical_event_manifest(hist_predictions, scored)
    event_manifest.to_csv(OUTPUT / "data" / "matched_event_manifest.csv", index=False)
    test = scored.loc[scored["split"].eq("test") & scored["evaluation_period"]].copy()
    test = attach_hydrology_flags(test)
    enrichment, enrichment_sites, site_boot, enrichment_event_boot = enrichment_analysis(test, thresholds)
    enrichment.to_csv(OUTPUT / "event_metrics" / "event_enrichment.csv", index=False)
    lead, lead_summary, lag, lead_boot = lead_and_lag(test, thresholds)
    lead.to_csv(OUTPUT / "event_metrics" / "event_level_lead_times.csv", index=False)
    lag.to_csv(OUTPUT / "event_metrics" / "lagged_enrichment.csv", index=False)
    episodes, burden = alert_episodes(test, event_manifest, thresholds)
    episodes.to_csv(OUTPUT / "event_metrics" / "alert_episodes.csv", index=False)
    burden_summary_frame = burden_summary(burden)
    overlap, overlap_metrics = residual_agreement(test, enrichment)
    overlap.to_csv(OUTPUT / "event_metrics" / "residual_overlap_categories.csv", index=False)
    hydrology = hydrological_stratification(test, lead, episodes, burden)
    hydrology.to_csv(OUTPUT / "event_metrics" / "hydrological_stratification.csv", index=False)
    site_metrics, compression, compression_summary = site_metrics_and_compression(test, enrichment_sites, burden)
    site_metrics.to_csv(OUTPUT / "site_metrics" / "site_level_metrics.csv", index=False)
    compression.to_csv(OUTPUT / "site_metrics" / "compression_metrics.csv", index=False)
    comparison = comparison_table(test, site_metrics, enrichment, lead_summary, burden_summary_frame)
    comparison.to_csv(OUTPUT / "summary" / "matched_model_comparison.csv", index=False)
    extra_boot = extra_site_bootstrap(site_metrics, enrichment_sites, compression)
    all_site_boot = pd.concat([site_boot, extra_boot], ignore_index=True, sort=False)
    all_event_boot = pd.concat([enrichment_event_boot, lead_boot], ignore_index=True, sort=False)
    all_site_boot.to_csv(OUTPUT / "bootstrap" / "site_bootstrap_results.csv", index=False)
    all_event_boot.to_csv(OUTPUT / "bootstrap" / "event_bootstrap_results.csv", index=False)
    write_json(OUTPUT / "bootstrap" / "bootstrap_config.json", {"fixed_seed": SEED, "iterations": BOOTSTRAPS, "site_resampling_unit": "site", "event_resampling_unit": "registered event", "hourly_rows_treated_as_iid": False, "interval": "percentile 95%"})
    result, decision_audit = conclusions(comparison, overlap_metrics, burden_summary_frame)
    calibration_summary = residual_summary[["split", "site_id", "calibration_status", "evaluation_rows"]].drop_duplicates()
    final_summary = {
        "result": result, "decision_audit": decision_audit,
        "prediction_summary": prediction_summary, "event_summary": event_summary,
        "comparison": comparison.to_dict(orient="records"),
        "overlap_metrics": overlap_metrics, "compression_summary": compression_summary,
        "elapsed_seconds": time.time() - started,
    }
    write_json(OUTPUT / "summary" / "final_summary.json", final_summary)
    make_figures(test, enrichment, lead, lag, burden, overlap, compression)
    write_reports(
        manifest, prediction_summary, event_summary, calibration_summary, thresholds,
        enrichment, lead_summary, lag, burden_summary_frame, comparison, overlap,
        overlap_metrics, hydrology, compression_summary, result, decision_audit,
    )
    expected = [
        "reports/FINAL_CROSS_VARIABLE_HOT_MOMENT_REPORT.md", "reports/EXISTING_ASSET_INVENTORY.md",
        "reports/CHECKPOINT_AND_TASK_PROVENANCE.md", "reports/PREDICTION_REUSE_REPORT.md",
        "reports/MATCHED_KEY_AUDIT.md", "reports/LABEL_FREE_SCORE_CALIBRATION.md",
        "reports/EVENT_ENRICHMENT_REPORT.md", "reports/LEAD_TIME_REPORT.md",
        "reports/LAGGED_ENRICHMENT_REPORT.md", "reports/OUTSIDE_REGISTER_ALERT_BURDEN_REPORT.md",
        "reports/MODEL_COMPARISON_REPORT.md", "reports/RESIDUAL_AGREEMENT_REPORT.md",
        "reports/COMPRESSION_AND_SCORE_QUALITY_REPORT.md", "configs/model_manifest.json",
        "calibration/label_free_thresholds.json", "data/matched_prediction_keys.parquet",
        "data/matched_event_manifest.csv", "predictions/cross_variable_h00_predictions.parquet",
        "predictions/historical_target_predictions.parquet", "predictions/matched_predictions.parquet",
        "residual_scores/matched_residuals.parquet", "event_metrics/event_enrichment.csv",
        "event_metrics/event_level_lead_times.csv", "event_metrics/lagged_enrichment.csv",
        "event_metrics/alert_episodes.csv", "event_metrics/residual_overlap_categories.csv",
        "site_metrics/site_level_metrics.csv", "site_metrics/site_residual_summary.csv",
        "site_metrics/compression_metrics.csv", "bootstrap/site_bootstrap_results.csv",
        "bootstrap/event_bootstrap_results.csv", "bootstrap/bootstrap_config.json",
        "summary/matched_model_comparison.csv", "summary/final_summary.json",
    ] + [f"figures/{number:02d}_{name}.png" for number, name in (
        (1, "research_logic"), (2, "event_vs_other_score_distribution"),
        (3, "matched_top5_enrichment"), (4, "lead_time_distribution"),
        (5, "lagged_enrichment"), (6, "outside_register_burden"),
        (7, "residual_score_scatter"), (8, "residual_overlap_categories"),
        (9, "compression_vs_enrichment"),
    )]
    completion(result, expected)
    print(json.dumps({"status": "complete", "result": result, "comparison": comparison.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        mkdirs()
        write_text(OUTPUT / "reports" / "BLOCKED_OR_STOPPED_REPORT.md", f"# Blocked or stopped\n\n{type(exc).__name__}: {exc}")
        write_json(OUTPUT / "completion" / "COMPLETED.json", {"status": "blocked", "central_hypothesis_result": "BLOCKED", "error_type": type(exc).__name__, "error": str(exc)})
        raise
