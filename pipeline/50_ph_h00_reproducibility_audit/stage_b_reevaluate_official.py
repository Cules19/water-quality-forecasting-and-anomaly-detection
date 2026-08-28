from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "ph_h00_reproducibility_audit_v1"
OFFICIAL = ROOT / "analysis_outputs" / "ph_common1000_lstm_nohistory_h0_rain_final_v1" / "full"
CHECKPOINT = OFFICIAL / "models" / "ph_LSTM_RAIN_H0_H00.pt"
SOURCE_PRED = OFFICIAL / "predictions_by_site" / "ph_LSTM_RAIN_H0_H00" / "validation"
FORMAL = ROOT / "analysis_outputs" / "ph_h00_hydrological_memory_validation_v1"
TRAIN_CODE = ROOT / "pipeline" / "31_common1000_rainfall_ablation" / "run_rainfall_ablation.py"
INPUT_ROOT = ROOT / "analysis_outputs" / "ph_common1000_teacher_archive_features_do_converted_v1" / "full" / "hourly_ph_rainfall_by_site"
COHORT_PATH = ROOT / "analysis_outputs" / "ph_common1000_teacher_archive_features_do_converted_v1" / "full" / "reports" / "primary_archive_experiment_cohort.csv"
CALIBRATION_DAYS = 30
MINIMUM_VALID_HOURS = 504


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = load_module("h00_repro_current_trainer", TRAIN_CODE)


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = observed.astype(np.float64) - predicted.astype(np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
    }


def calibrate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    result["calibration_period"] = False
    result["evaluation_period"] = False
    result["pred_cal_official"] = np.nan
    rows: list[dict[str, Any]] = []
    for site_id, group in result.groupby("site_id", sort=False):
        start = group["timestamp"].min()
        end_exclusive = start + pd.Timedelta(days=CALIBRATION_DAYS)
        cal_index = group.index[group["timestamp"].lt(end_exclusive)]
        eval_index = group.index[group["timestamp"].ge(end_exclusive)]
        if len(cal_index) < MINIMUM_VALID_HOURS or len(eval_index) == 0:
            continue
        valid = result.loc[cal_index, ["observed_ph", "pred_raw_official"]].dropna()
        if len(valid) < MINIMUM_VALID_HOURS:
            continue
        offset = float((valid["observed_ph"] - valid["pred_raw_official"]).mean())
        result.loc[cal_index, "calibration_period"] = True
        result.loc[eval_index, "evaluation_period"] = True
        result.loc[group.index, "pred_cal_official"] = result.loc[group.index, "pred_raw_official"] + offset
        rows.append(
            {
                "site_id": str(site_id),
                "method": "OFFICIAL",
                "calibration_start": start,
                "calibration_end": result.loc[cal_index, "timestamp"].max(),
                "calibration_end_exclusive": end_exclusive,
                "calibration_valid_hours": len(valid),
                "evaluation_start": result.loc[eval_index, "timestamp"].min(),
                "evaluation_valid_hours": len(eval_index),
                "mean_residual_offset": offset,
                "offset_updated_after_calibration": False,
            }
        )
    return result, pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = frame.loc[
        frame["evaluation_period"]
        & frame[["observed_ph", "pred_cal_official"]].notna().all(axis=1)
    ].copy()
    site_rows = []
    for site_id, site in evaluation.groupby("site_id", sort=False):
        values = metric_values(site["observed_ph"].to_numpy(float), site["pred_cal_official"].to_numpy(float))
        site_rows.append({"site_id": str(site_id), **values, "sample_count": len(site)})
    sites = pd.DataFrame(site_rows)
    pooled = metric_values(evaluation["observed_ph"].to_numpy(float), evaluation["pred_cal_official"].to_numpy(float))
    summary = pd.DataFrame(
        [
            {
                "method": "OFFICIAL_FRESH",
                "pooled_rmse": pooled["rmse"],
                "pooled_mae": pooled["mae"],
                "macro_site_rmse": float(sites["rmse"].mean()),
                "macro_site_mae": float(sites["mae"].mean()),
                "site_count": int(sites["site_id"].nunique()),
                "row_count": len(evaluation),
            }
        ]
    )
    return sites, summary


def main() -> None:
    if not (OUT / "audit" / "evidence_preservation_summary.json").exists():
        raise RuntimeError("Stage A evidence preservation must complete first")
    preservation = json.loads((OUT / "audit" / "evidence_preservation_summary.json").read_text(encoding="utf-8"))
    if preservation["decision"] != "EVIDENCE_PRESERVED":
        raise RuntimeError("Evidence preservation did not pass")

    config = json.loads((OFFICIAL / "reports" / "effective_config.json").read_text(encoding="utf-8-sig"))
    preprocessing = json.loads((OFFICIAL / "reports" / "train_only_preprocessing.json").read_text(encoding="utf-8-sig"))
    cohort = pd.read_csv(COHORT_PATH, dtype={"site_id": "string"})
    cohort["site_id"] = cohort["site_id"].map(TRAIN.PREPARE.normalise_site_id)
    cohort = cohort[
        cohort["primary_archive_cohort"].fillna(False).astype(bool)
        & cohort["ready_for_common_key_experiment"].fillna(False).astype(bool)
        & cohort["split"].isin(["train", "validation"])
    ][["site_id", "split", "network"]].drop_duplicates("site_id")
    split_counts = cohort["split"].value_counts().to_dict()
    if split_counts != {"train": 638, "validation": 137}:
        raise RuntimeError(f"Unexpected no-test cohort: {split_counts}")

    metadata_source = TRAIN.BASE.merge_metadata(config)
    metadata_fields = [field for field in config["metadata_fields"] if field not in set(config["coordinate_fields"])]
    metadata_map, metadata_parameters = TRAIN.BASE.fit_metadata(
        metadata_source, cohort, metadata_fields, set(config["metadata_log1p_fields"])
    )
    stored_metadata = json.loads((OFFICIAL / "metadata" / "ph_no_coordinates_train_only_transform.json").read_text(encoding="utf-8-sig"))
    if metadata_parameters != stored_metadata:
        raise RuntimeError("Current metadata transformation differs from frozen official transformation")

    feature_names = list(config["models"][0]["feature_names"])
    validation_ids = cohort.loc[cohort["split"].eq("validation"), "site_id"].astype(str).tolist()
    first_site = validation_ids[0]
    sample = TRAIN.site_payload(INPUT_ROOT / f"{first_site}.parquet", feature_names, preprocessing, config)
    model_spec = {
        "model_id": "LSTM_RAIN_H0",
        "recurrent_type": "lstm",
        "recurrent_layers": int(config.get("recurrent_layers", 1)),
        "conditioning": "film",
        "metadata_variant": "no_coordinates",
        "target_history": False,
        "train_non_event": False,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TRAIN.BASE.SiteAgnosticFiLMRNN(
        model_spec, sample["features"].shape[1], len(metadata_map[first_site]), config
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    work = OUT / "predictions" / "fresh_inference_work"
    work.mkdir(parents=True, exist_ok=True)
    existing_fresh = list((work / "predictions_by_site" / "official_fresh" / "validation").glob("*.parquet"))
    if len(existing_fresh) == len(validation_ids):
        overall = pd.DataFrame()
    else:
        overall, _ = TRAIN.predict_model(
            model=model,
            split=cohort,
            feature_names=feature_names,
            metadata_map=metadata_map,
            preprocessing=preprocessing,
            config=config,
            input_root=INPUT_ROOT,
            settings=config["full"],
            output=work,
            run_id="official_fresh",
            device=device,
            model_spec=model_spec,
            train_non_event=False,
            prediction_splits=("validation",),
        )

    fresh_parts = []
    raw_diffs = []
    for site_id in validation_ids:
        fresh_path = work / "predictions_by_site" / "official_fresh" / "validation" / f"{site_id}.parquet"
        stored_path = SOURCE_PRED / f"{site_id}.parquet"
        fresh = pd.read_parquet(fresh_path)
        stored = pd.read_parquet(stored_path, columns=["site_id", "timestamp", "observed_native", "predicted_native"])
        fresh["site_id"] = fresh["site_id"].astype(str)
        stored["site_id"] = stored["site_id"].astype(str)
        fresh["timestamp"] = pd.to_datetime(fresh["timestamp"])
        stored["timestamp"] = pd.to_datetime(stored["timestamp"])
        if not fresh[["site_id", "timestamp"]].equals(stored[["site_id", "timestamp"]]):
            raise RuntimeError(f"Fresh/stored raw key mismatch: {site_id}")
        diff = np.abs(fresh["predicted_native"].to_numpy(float) - stored["predicted_native"].to_numpy(float))
        raw_diffs.append(diff)
        fresh_parts.append(
            fresh.rename(columns={"observed_native": "observed_ph", "predicted_native": "pred_raw_official"})
        )
    raw_diff = np.concatenate(raw_diffs)
    fresh_full_key = pd.concat(fresh_parts, ignore_index=True).sort_values(["site_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    fresh_full_key.to_parquet(OUT / "predictions" / "official_fresh_full_key_validation.parquet", index=False, compression="zstd")

    stored_formal = pd.read_parquet(FORMAL / "predictions" / "formal_reference_validation.parquet")
    stored_formal["site_id"] = stored_formal["site_id"].astype(str)
    stored_formal["timestamp"] = pd.to_datetime(stored_formal["timestamp"])
    formal_keys = stored_formal[["site_id", "timestamp"]]
    key_audit = fresh_full_key[["site_id", "timestamp"]].merge(
        formal_keys, on=["site_id", "timestamp"], how="outer", indicator=True, validate="one_to_one"
    )
    key_counts = key_audit["_merge"].value_counts().to_dict()
    if int(key_counts.get("right_only", 0)) != 0:
        raise RuntimeError("Frozen formal key is absent from fresh inference")
    fresh_all = fresh_full_key.merge(formal_keys, on=["site_id", "timestamp"], how="inner", validate="one_to_one")
    fresh_all, offsets = calibrate(fresh_all)
    sites, summary = summarize(fresh_all)
    fresh_all.to_parquet(OUT / "predictions" / "official_fresh_validation.parquet", index=False, compression="zstd")
    offsets.to_csv(OUT / "calibration" / "model_specific_offsets.csv", index=False)
    sites.to_csv(OUT / "metrics" / "recomputed_site_metrics.csv", index=False)
    summary.to_csv(OUT / "metrics" / "recomputed_method_summary.csv", index=False)

    compare = fresh_all.merge(
        stored_formal[["site_id", "timestamp", "pred_raw_official", "pred_cal_official", "calibration_period", "evaluation_period"]],
        on=["site_id", "timestamp"], how="outer", suffixes=("_fresh", "_stored"), indicator=True, validate="one_to_one"
    )
    if not compare["_merge"].eq("both").all():
        raise RuntimeError("Fresh/stored frozen-formal key mismatch")
    cal_mask = compare[["pred_cal_official_fresh", "pred_cal_official_stored"]].notna().all(axis=1)
    cal_diff = np.abs(compare.loc[cal_mask, "pred_cal_official_fresh"].to_numpy(float) - compare.loc[cal_mask, "pred_cal_official_stored"].to_numpy(float))

    stored_offsets = pd.read_csv(FORMAL / "calibration" / "model_specific_site_offsets.csv", dtype={"site_id": str})
    stored_offsets = stored_offsets.loc[stored_offsets["method"].eq("OFFICIAL"), ["site_id", "mean_residual_offset"]]
    offset_compare = offsets[["site_id", "mean_residual_offset"]].merge(stored_offsets, on="site_id", suffixes=("_fresh", "_stored"), validate="one_to_one")
    offset_compare["absolute_difference"] = (offset_compare["mean_residual_offset_fresh"] - offset_compare["mean_residual_offset_stored"]).abs()
    offset_compare.to_csv(OUT / "calibration" / "offset_comparison.csv", index=False)

    expected = {
        "pooled_rmse": 0.3110,
        "pooled_mae": 0.2173,
        "macro_site_rmse": 0.2199,
        "macro_site_mae": 0.1720,
        "site_count": 128,
        "row_count": 2_719_921,
    }
    actual = summary.iloc[0].to_dict()
    tolerance_pass = all(
        abs(float(actual[key]) - expected[key]) <= 1e-4
        for key in ("pooled_rmse", "pooled_mae", "macro_site_rmse", "macro_site_mae")
    ) and int(actual["site_count"]) == expected["site_count"] and int(actual["row_count"]) == expected["row_count"]
    comparison = pd.DataFrame(
        [
            {"comparison": "stored_raw_vs_fresh_raw", "row_count": len(raw_diff), "maximum_absolute_prediction_difference": float(raw_diff.max()), "median_absolute_prediction_difference": float(np.median(raw_diff)), "mean_absolute_prediction_difference": float(raw_diff.mean()), "fraction_exact": float(np.mean(raw_diff == 0))},
            {"comparison": "stored_calibrated_vs_fresh_calibrated", "row_count": len(cal_diff), "maximum_absolute_prediction_difference": float(cal_diff.max()), "median_absolute_prediction_difference": float(np.median(cal_diff)), "mean_absolute_prediction_difference": float(cal_diff.mean()), "fraction_exact": float(np.mean(cal_diff == 0))},
        ]
    )
    comparison.to_csv(OUT / "audit" / "prediction_file_comparison.csv", index=False)

    sampled = compare.loc[cal_mask].iloc[:: max(1, len(compare.loc[cal_mask]) // 100_000)]
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(sampled["pred_cal_official_stored"], sampled["pred_cal_official_fresh"], s=2, alpha=0.15, rasterized=True)
    low = float(min(sampled["pred_cal_official_stored"].min(), sampled["pred_cal_official_fresh"].min()))
    high = float(max(sampled["pred_cal_official_stored"].max(), sampled["pred_cal_official_fresh"].max()))
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set(xlabel="Stored official calibrated prediction", ylabel="Fresh calibrated prediction", title="Official H00 deterministic inference reproduction")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "01_official_vs_fresh_prediction_scatter.png", dpi=180)
    plt.close(fig)

    audit = {
        "decision": "OFFICIAL_EVALUATION_REPRODUCED" if tolerance_pass else "OFFICIAL_EVALUATION_NOT_REPRODUCED",
        "device": str(device),
        "checkpoint": str(CHECKPOINT),
        "frozen_scaler_used": True,
        "current_metadata_transform_matches_frozen": True,
        "test_time_series_accessed": False,
        "raw_inference_summary": overall.to_dict(orient="records"),
        "expected": expected,
        "actual": actual,
        "fresh_inference_tolerance": 1e-4,
        "metric_tolerance_passed": tolerance_pass,
        "raw_prediction_max_abs_difference": float(raw_diff.max()),
        "raw_prediction_median_abs_difference": float(np.median(raw_diff)),
        "calibrated_prediction_max_abs_difference": float(cal_diff.max()),
        "calibrated_prediction_median_abs_difference": float(np.median(cal_diff)),
        "offset_max_abs_difference": float(offset_compare["absolute_difference"].max()),
        "fresh_full_key_rows": len(fresh_full_key),
        "frozen_formal_key_rows": len(formal_keys),
        "fresh_only_nonformal_rows": int(key_counts.get("left_only", 0)),
        "frozen_formal_keys_missing_from_fresh": int(key_counts.get("right_only", 0)),
    }
    (OUT / "audit" / "official_reevaluation_summary.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    report = f"""# Official H00 Checkpoint Re-evaluation\n\n## Decision\n\n**{audit['decision']}**\n\nThe frozen official checkpoint was loaded with the current inference code, the frozen input and target scalers, the frozen 638/137 train-validation cohort, and no test time series. The earliest 30 calendar days at each site were used to estimate a mean residual offset, with at least 504 valid hours required.\n\n## Recomputed formal validation result\n\n| metric | expected | recomputed |\n| --- | ---: | ---: |\n| pooled RMSE | {expected['pooled_rmse']:.4f} | {actual['pooled_rmse']:.6f} |\n| pooled MAE | {expected['pooled_mae']:.4f} | {actual['pooled_mae']:.6f} |\n| macro-site RMSE | {expected['macro_site_rmse']:.4f} | {actual['macro_site_rmse']:.6f} |\n| macro-site MAE | {expected['macro_site_mae']:.4f} | {actual['macro_site_mae']:.6f} |\n| eligible sites | {expected['site_count']} | {int(actual['site_count'])} |\n| evaluation rows | {expected['row_count']:,} | {int(actual['row_count']):,} |\n\nThe maximum raw prediction difference from the stored file was {raw_diff.max():.3g}; the median was {np.median(raw_diff):.3g}. The maximum calibrated-prediction difference was {cal_diff.max():.3g}; the maximum offset difference was {offset_compare['absolute_difference'].max():.3g}. Fresh inference generated {len(fresh_full_key):,} official keys. The frozen formal comparison contains {len(formal_keys):,} keys; {int(key_counts.get('left_only', 0)):,} additional official-only keys were excluded to preserve the preregistered formal cohort.\n"""
    (OUT / "reports" / "OFFICIAL_CHECKPOINT_REEVALUATION_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2, default=str))
    if not tolerance_pass:
        raise RuntimeError("OFFICIAL_EVALUATION_NOT_REPRODUCED")


if __name__ == "__main__":
    main()
