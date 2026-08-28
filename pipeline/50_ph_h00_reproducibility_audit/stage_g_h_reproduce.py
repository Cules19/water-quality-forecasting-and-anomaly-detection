from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "ph_h00_reproducibility_audit_v1"
OFFICIAL = ROOT / "analysis_outputs" / "ph_common1000_lstm_nohistory_h0_rain_final_v1" / "full"
INPUT_ROOT = ROOT / "analysis_outputs" / "ph_common1000_teacher_archive_features_do_converted_v1" / "full" / "hourly_ph_rainfall_by_site"
COHORT_PATH = ROOT / "analysis_outputs" / "ph_common1000_teacher_archive_features_do_converted_v1" / "full" / "reports" / "primary_archive_experiment_cohort.csv"
TRAIN_CODE = ROOT / "pipeline" / "31_common1000_rainfall_ablation" / "run_rainfall_ablation.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = load_module("h00_minimal_reproduction_trainer", TRAIN_CODE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["primary", "repeat"], required=True)
    return parser.parse_args()


def compare_states(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, Any]:
    names_match = list(left) == list(right)
    shape_match = set(left) == set(right) and all(tuple(left[name].shape) == tuple(right[name].shape) for name in left)
    exact = shape_match and all(torch.equal(left[name], right[name]) for name in left)
    maximum = 0.0
    mean_values = []
    if shape_match:
        for name in left:
            delta = (left[name].detach().cpu().float() - right[name].detach().cpu().float()).abs()
            if delta.numel():
                maximum = max(maximum, float(delta.max()))
                mean_values.append(float(delta.mean()))
    return {
        "tensor_names_and_order_match": names_match,
        "tensor_shapes_match": shape_match,
        "tensor_exact_match": exact,
        "maximum_absolute_tensor_difference": maximum if shape_match else None,
        "mean_absolute_tensor_difference": float(np.mean(mean_values)) if mean_values else 0.0,
    }


def main() -> None:
    args = parse_args()
    audit = load_json(OUT / "audit" / "stages_c_to_f_summary.json")
    if not audit.get("stages_c_to_f_complete"):
        raise RuntimeError("Stages A-F must complete first")
    if args.run == "repeat":
        primary_summary_path = OUT / "training_reproduction" / "repro_seed42_summary.json"
        if not primary_summary_path.exists() or not load_json(primary_summary_path).get("reproduction_passed"):
            raise RuntimeError("Primary seed42 reproduction has not passed; repeat is forbidden")

    config = load_json(OFFICIAL / "reports" / "effective_config.json")
    config["random_seed"] = 42
    preprocessing = load_json(OFFICIAL / "reports" / "train_only_preprocessing.json")
    cohort = pd.read_csv(COHORT_PATH, dtype={"site_id": "string"})
    cohort["site_id"] = cohort["site_id"].map(TRAIN.PREPARE.normalise_site_id)
    cohort = cohort[
        cohort["primary_archive_cohort"].fillna(False).astype(bool)
        & cohort["ready_for_common_key_experiment"].fillna(False).astype(bool)
        & cohort["split"].isin(["train", "validation"])
    ][["site_id", "split", "network"]].drop_duplicates("site_id")
    if cohort["split"].value_counts().to_dict() != {"train": 638, "validation": 137}:
        raise RuntimeError("Cohort mismatch")
    train_ids = cohort.loc[cohort["split"].eq("train"), "site_id"].astype(str).tolist()
    validation_ids = cohort.loc[cohort["split"].eq("validation"), "site_id"].astype(str).tolist()

    metadata_source = TRAIN.BASE.merge_metadata(config)
    fields = [field for field in config["metadata_fields"] if field not in set(config["coordinate_fields"])]
    metadata_map, metadata_parameters = TRAIN.BASE.fit_metadata(metadata_source, cohort, fields, set(config["metadata_log1p_fields"]))
    frozen_metadata = load_json(OFFICIAL / "metadata" / "ph_no_coordinates_train_only_transform.json")
    if metadata_parameters != frozen_metadata:
        raise RuntimeError("Metadata transform drift")

    feature_names = ["x_00010", "x_00095", "x_00300", "x_63680", "rain_00045"]
    first_site = train_ids[0]
    sample = TRAIN.site_payload(INPUT_ROOT / f"{first_site}.parquet", feature_names, preprocessing, config)
    model_spec = {
        "model_id": "R0_REPRO",
        "recurrent_type": "lstm",
        "recurrent_layers": 1,
        "conditioning": "film",
        "metadata_variant": "no_coordinates",
        "target_history": False,
        "train_non_event": False,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Full matched reproduction requires CUDA")
    TRAIN.BASE.set_seed(42)
    model = TRAIN.BASE.SiteAgnosticFiLMRNN(model_spec, sample["features"].shape[1], len(metadata_map[first_site]), config).to(device)
    initial_hash = state_digest(model.state_dict())
    started = time.time()
    model, history, training_info = TRAIN.train_model(
        model=model,
        train_ids=train_ids,
        validation_ids=validation_ids,
        feature_names=feature_names,
        metadata_map=metadata_map,
        preprocessing=preprocessing,
        config=config,
        input_root=INPUT_ROOT,
        settings=config["full"],
        device=device,
        train_non_event=False,
    )
    elapsed = time.time() - started

    if args.run == "primary":
        history_path = OUT / "training_reproduction" / "repro_seed42_history.csv"
        checkpoint_path = OUT / "training_reproduction" / "repro_seed42_best.pt"
        summary_path = OUT / "training_reproduction" / "repro_seed42_summary.json"
    else:
        history_path = OUT / "training_reproduction" / "repro_seed42_repeat_history.csv"
        checkpoint_path = OUT / "training_reproduction" / "repro_seed42_repeat_best.pt"
        summary_path = OUT / "training_reproduction" / "repro_seed42_repeat_summary.json"
    history.to_csv(history_path, index=False)
    torch.save(model.state_dict(), checkpoint_path)

    official_state = torch.load(OFFICIAL / "models" / "ph_LSTM_RAIN_H0_H00.pt", map_location="cpu", weights_only=True)
    current_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    comparison = compare_states(current_state, official_state)
    expected_metrics = load_json(OUT / "audit" / "official_reevaluation_summary.json")["actual"]
    official_best = load_json(OFFICIAL / "models" / "ph_LSTM_RAIN_H0_H00_SUCCESS.json")
    val_relative_difference = abs(float(training_info["best_validation_rmse_native"]) - float(official_best["best_validation_rmse_native"])) / float(official_best["best_validation_rmse_native"])
    reproduction_passed = bool(
        val_relative_difference <= 0.01
        and comparison["tensor_exact_match"]
        and int(training_info["best_epoch"]) == int(official_best["best_epoch"])
    )
    result = {
        "run": args.run,
        "seed": 42,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "initial_state_sha256": initial_hash,
        "best_state_tensor_sha256": state_digest(current_state),
        "checkpoint_file_sha256": sha256(checkpoint_path),
        "best_epoch": int(training_info["best_epoch"]),
        "epochs_ran": int(training_info["epochs_ran"]),
        "best_validation_rmse_native": float(training_info["best_validation_rmse_native"]),
        "validation_rmse_relative_difference_from_official": val_relative_difference,
        "optimizer_steps_per_epoch": 2739,
        "actual_optimizer_steps": 2739 * int(training_info["epochs_ran"]),
        "checkpoint_comparison_to_official": comparison,
        "formal_validation_metrics": expected_metrics if comparison["tensor_exact_match"] else None,
        "formal_prediction_comparison": "exact by identical state dict, frozen scalers, identical validation keys and the already verified deterministic current inference pipeline" if comparison["tensor_exact_match"] else "not inferred",
        "reproduction_passed": reproduction_passed,
        "test_time_series_accessed": False,
        "parameter_updates_during_evaluation": False,
    }
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    config_record = {
        "run": args.run,
        "seed": 42,
        "official_effective_config_sha256": sha256(OFFICIAL / "reports" / "effective_config.json"),
        "official_input_target_scaler_sha256": sha256(OFFICIAL / "reports" / "train_only_preprocessing.json"),
        "official_metadata_scaler_sha256": sha256(OFFICIAL / "metadata" / "ph_no_coordinates_train_only_transform.json"),
        "raw_no_test_manifest_sha256": sha256(OUT / "hashes" / "raw_data_sha256_manifest.txt"),
        "features": feature_names,
        "sequence_length_hours": 48,
        "horizon_hours": 0,
        "sampler": "site-sequential, per-site uniform without replacement, cap 2500 windows/site/epoch",
        "samples_per_epoch": 1_310_464,
        "optimizer_steps_per_epoch": 2739,
        "maximum_epochs": 20,
        "checkpoint_selection": "minimum validation_rmse_native; restore best state",
        "test_time_series_accessed": False,
    }
    config_name = "repro_seed42_config.json" if args.run == "primary" else "repro_seed42_repeat_config.json"
    (OUT / "training_reproduction" / config_name).write_text(json.dumps(config_record, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not reproduction_passed:
        raise RuntimeError("Seed42 reproduction failed; repeat must not run")


if __name__ == "__main__":
    main()
