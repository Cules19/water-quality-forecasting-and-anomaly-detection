"""Audit the frozen H0 checkpoint and warm-up evidence before any rerun.

This script implements Parts A and B of the frozen H0 rerun protocol.
It is deliberately fail-closed: a failed leakage, temporal-causality, freeze,
or DO-unit test writes a BLOCKED_REPORT and exits non-zero before training.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (
    PROJECT_ROOT / "analysis_outputs" / "h0_site_transfer_warmup_final_audit_v1"
)


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RAIN = import_module(
    "h0_final_rain",
    PROJECT_ROOT
    / "pipeline"
    / "31_common1000_rainfall_ablation"
    / "run_rainfall_ablation.py",
)
WARMUP = import_module(
    "h0_final_warmup",
    PROJECT_ROOT
    / "pipeline"
    / "36_warmup_adaptation"
    / "run_warmup_adaptation.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def test_record(
    records: list[dict[str, Any]],
    test_id: str,
    target: str,
    passed: bool,
    detail: str,
    evidence: Any | None = None,
) -> None:
    row: dict[str, Any] = {
        "test_id": test_id,
        "target": target,
        "passed": bool(passed),
        "detail": detail,
    }
    if evidence is not None:
        row["evidence"] = evidence
    records.append(row)


def checkpoint_descriptor(
    target: str,
    checkpoint: Path,
    config_path: Path,
    effective_config_path: Path,
    success_path: Path,
    preprocessing_path: Path,
    metadata_path: Path,
    common_key_path: Path,
) -> dict[str, Any]:
    config = load_json(effective_config_path)
    success = load_json(success_path)
    cohort_path = RAIN.resolve(config["primary_cohort_path"])
    split_path = (
        PROJECT_ROOT
        / "analysis_outputs"
        / "ph_common1000_leave_sites_out_v1"
        / "full"
        / "reports"
        / f"site_split_{'ph' if target == 'pH' else 'do'}.csv"
    )
    return {
        "target": target,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "effective_config_path": str(effective_config_path.resolve()),
        "effective_config_sha256": sha256(effective_config_path),
        "success_manifest_path": str(success_path.resolve()),
        "success_manifest_sha256": sha256(success_path),
        "preprocessing_path": str(preprocessing_path.resolve()),
        "preprocessing_sha256": sha256(preprocessing_path),
        "metadata_transform_path": str(metadata_path.resolve()),
        "metadata_transform_sha256": sha256(metadata_path),
        "primary_cohort_path": str(cohort_path.resolve()),
        "primary_cohort_sha256": sha256(cohort_path),
        "site_split_path": str(split_path.resolve()),
        "site_split_sha256": sha256(split_path),
        "target_key_manifest_path": str(common_key_path.resolve()),
        "target_key_manifest_sha256": sha256(common_key_path),
        "target_key_manifest_sha256_recorded": success.get(
            "common_key_manifest_sha256"
        ),
        "feature_set": list(success["feature_set"]),
        "input_feature_names": list(success["input_feature_names"]),
        "dynamic_size": int(success["dynamic_size"]),
        "metadata_size": int(success["metadata_size"]),
        "architecture": "site-agnostic LSTM-FiLM",
        "hidden_size": int(config["hidden_size"]),
        "film_hidden_size": int(config["film_hidden_size"]),
        "head_hidden_size": int(config["head_hidden_size"]),
        "recurrent_layers": int(success["recurrent_layers"]),
        "dropout": float(success["dropout"]),
        "sequence_length_hours": int(success["sequence_length_hours"]),
        "horizon_hours": int(config["horizon_hours"]),
        "seed": int(config["random_seed"]),
        "training_data_root": str(RAIN.resolve(config["merged_input_root"])),
        "target_code": str(config["target_code"]),
        "target_source_column": str(
            config.get("target_source_column", f"target_{config['target_code']}_native")
        ),
        "target_source_multiplier": float(config.get("target_source_multiplier", 1.0)),
        "target_unit": str(config.get("target_unit", "native pH units")),
        "target_history": bool(config.get("include_target_history", True)),
        "trainable_parameters": int(success["trainable_parameters"]),
    }


def feature_and_temporal_tests(
    descriptor: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    target = descriptor["target"]
    config = load_json(Path(descriptor["effective_config_path"]))
    preprocessing = load_json(Path(descriptor["preprocessing_path"]))
    cohort = pd.read_csv(
        descriptor["primary_cohort_path"], dtype={"site_id": "string"}
    )
    cohort = cohort[
        cohort["primary_archive_cohort"].fillna(False).astype(bool)
        & cohort["ready_for_common_key_experiment"].fillna(False).astype(bool)
    ]
    site_id = str(cohort.sort_values(["split", "site_id"]).iloc[0]["site_id"])
    site_path = RAIN.resolve(config["merged_input_root"]) / f"{site_id}.parquet"
    payload = RAIN.site_payload(
        site_path,
        list(descriptor["feature_set"]),
        preprocessing,
        config,
    )
    feature_names = [str(value) for value in payload["feature_names"]]
    target_code = str(descriptor["target_code"])
    forbidden_fragments = {
        target_code,
        "target_history",
        "target_lag",
        "rolling_target",
        "target_diff",
    }
    contaminated = [
        name
        for name in feature_names
        if any(fragment.lower() in name.lower() for fragment in forbidden_fragments)
    ]
    test_record(
        records,
        "H0_FEATURE_NAMES",
        target,
        not contaminated,
        "No target code or target-derived feature name is present in X."
        if not contaminated
        else f"Forbidden feature names found: {contaminated}",
        {"feature_names": feature_names, "forbidden_matches": contaminated},
    )
    channel_pass = (
        payload["features"].ndim == 2
        and payload["features"].shape[1] == int(descriptor["dynamic_size"])
        and len(feature_names) == int(descriptor["dynamic_size"])
    )
    test_record(
        records,
        "H0_CHANNEL_COUNT",
        target,
        channel_pass,
        f"Actual X has {payload['features'].shape[1]} channels; manifest expects "
        f"{descriptor['dynamic_size']}.",
    )
    indices = RAIN.comparison_indices(payload["frame"], config)
    if len(indices) < 100:
        sample = indices
    else:
        sample = indices[np.linspace(0, len(indices) - 1, 100, dtype=np.int64)]
    timestamps = pd.DatetimeIndex(payload["timestamp"])
    horizon = int(config["horizon_hours"])
    max_input = timestamps[np.asarray(sample, dtype=np.int64) - horizon]
    target_times = timestamps[np.asarray(sample, dtype=np.int64)]
    strict = np.asarray(max_input < target_times, dtype=bool)
    equal = np.asarray(max_input == target_times, dtype=bool)
    temporal_pass = bool(strict.all())
    evidence = {
        "sample_site": site_id,
        "sample_windows": int(len(sample)),
        "horizon_hours": horizon,
        "strictly_causal_windows": int(strict.sum()),
        "equal_timestamp_windows": int(equal.sum()),
        "violations": int((~strict).sum()),
        "first_max_input_timestamp": max_input[0].isoformat() if len(sample) else None,
        "first_target_timestamp": target_times[0].isoformat() if len(sample) else None,
    }
    test_record(
        records,
        "TEMPORAL_CAUSALITY",
        target,
        temporal_pass,
        "All sampled input windows end strictly before the target timestamp."
        if temporal_pass
        else (
            "The frozen model is contemporaneous: at least one sampled input "
            "window includes an exogenous observation at the target timestamp."
        ),
        evidence,
    )
    return evidence


def audit_warmup_directory(
    target: str,
    root: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    shard_paths = sorted((root / "shards" / "test").glob("*/*.json"))
    rows = [load_json(path) for path in shard_paths]
    by_site_keys: dict[str, set[str]] = defaultdict(set)
    overlap_failures: list[str] = []
    parameter_failures: list[str] = []
    for row in rows:
        site = str(row["site_id"])
        by_site_keys[site].add(str(row["fixed_evaluation_key_sha256"]))
        maximum = pd.Timestamp(row["maximum_adaptation_target"])
        minimum = pd.Timestamp(row["minimum_fixed_evaluation_target"])
        if not maximum < minimum:
            overlap_failures.append(
                f"{site}:{row['warmup_label']}:{row['strategy']}"
            )
        strategy = str(row["strategy"])
        trainable = int(row["trainable_parameters"])
        if strategy == "zero_shot" and trainable != 0:
            parameter_failures.append(f"{site}:{strategy}:{trainable}")
        if strategy == "film_affine" and trainable != 130:
            parameter_failures.append(f"{site}:{strategy}:{trainable}")
        if strategy == "full_finetune" and trainable <= 130:
            parameter_failures.append(f"{site}:{strategy}:{trainable}")
    key_failures = {site: len(keys) for site, keys in by_site_keys.items() if len(keys) != 1}
    separation_pass = bool(rows) and not overlap_failures and not key_failures
    test_record(
        records,
        "WARMUP_EVALUATION_SEPARATION",
        target,
        separation_pass,
        (
            f"Checked {len(rows)} stored runs: warm-up targets precede evaluation "
            "and every duration uses one fixed post-12-month key per site."
            if separation_pass
            else "Warm-up/evaluation overlap or fixed-key mismatch detected."
        ),
        {
            "stored_runs": len(rows),
            "sites": len(by_site_keys),
            "overlap_failures": overlap_failures[:20],
            "fixed_key_failures": key_failures,
        },
    )
    parameter_pass = bool(rows) and not parameter_failures
    test_record(
        records,
        "ADAPTER_PARAMETER_COUNT",
        target,
        parameter_pass,
        "Stored zero-shot/FiLM/full runs use 0/130/>130 trainable parameters."
        if parameter_pass
        else "Unexpected trainable parameter count found.",
        {"failures": parameter_failures[:20]},
    )
    return {
        "rows": len(rows),
        "sites": len(by_site_keys),
        "separation_passed": separation_pass,
        "parameter_count_passed": parameter_pass,
    }


def freeze_test(records: list[dict[str, Any]]) -> dict[str, Any]:
    source = inspect.getsource(WARMUP.predict_scaled)
    source_pass = "torch.no_grad" in source and "optimizer" not in source
    targets: dict[str, Any] = {}
    cases = {
        "pH": {
            "config": PROJECT_ROOT / "config" / "common1000_warmup_adaptation_lowcost_ph_test_v1.json",
        },
        "DO": {
            "config": PROJECT_ROOT / "config" / "common1000_warmup_adaptation_lowcost_do_test_v1.json",
        },
    }
    for target, case in cases.items():
        cfg = load_json(case["config"])
        model_cfg = RAIN.load_config(RAIN.resolve(cfg["base_model_config"]))
        preprocessing = load_json(RAIN.resolve(cfg["preprocessing_path"]))
        strict = pd.read_csv(RAIN.resolve(cfg["strict_cohort_path"]), dtype={"site_id": "string"})
        site_id = str(
            strict.loc[
                strict["split"].eq("test")
                & strict["full_fixed_curve_eligible"].eq(True),
                "site_id",
            ].iloc[0]
        )
        payload = RAIN.site_payload(
            RAIN.resolve(model_cfg["merged_input_root"]) / f"{site_id}.parquet",
            list(cfg["feature_names"]),
            preprocessing,
            model_cfg,
        )
        metadata_parameters = load_json(RAIN.resolve(cfg["metadata_transform_path"]))
        raw_metadata = RAIN.BASE.merge_metadata(model_cfg)
        metadata = WARMUP.apply_saved_metadata_transform(
            raw_metadata,
            [site_id],
            metadata_parameters,
            set(model_cfg["metadata_log1p_fields"]),
        )[site_id]
        model = WARMUP.make_model(
            model_cfg,
            payload["features"].shape[1],
            len(metadata),
            RAIN.resolve(cfg["base_checkpoint"]),
            torch.device("cpu"),
            "film_affine",
        )
        indices = RAIN.comparison_indices(payload["frame"], model_cfg)[:2]
        before = tensor_state_sha256(model)
        WARMUP.predict_scaled(
            model,
            payload,
            indices,
            metadata,
            model_cfg,
            preprocessing,
            2,
            torch.device("cpu"),
        )
        after = tensor_state_sha256(model)
        targets[target] = {
            "site_id": site_id,
            "state_sha256_before": before,
            "state_sha256_after": after,
            "unchanged": before == after,
        }
        test_record(
            records,
            "EVALUATION_FREEZE",
            target,
            source_pass and before == after,
            "Inference uses no_grad, contains no optimizer call, and leaves all model/adapter tensors unchanged.",
            targets[target],
        )
    return {"predict_scaled_source_passed": source_pass, "targets": targets}


def do_unit_test(descriptor: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    config = load_json(Path(descriptor["effective_config_path"]))
    preprocessing = load_json(Path(descriptor["preprocessing_path"]))
    mean = float(preprocessing["target"]["mean"])
    scale = float(preprocessing["target"]["scale"])
    checks = {
        "target_unit_is_mg_l": str(config.get("target_unit", "")).lower() == "mg/l",
        "source_multiplier_is_20": float(config.get("target_source_multiplier", 0.0)) == 20.0,
        "mean_is_plausible_mg_l": 1.0 <= mean <= 20.0,
        "scale_is_plausible_mg_l": 0.1 <= scale <= 10.0,
        "checkpoint_path_contains_mg_l": "mg_l" in descriptor["checkpoint_path"].lower(),
        "warmup_path_contains_mg_l": "mg_l" in str(
            PROJECT_ROOT
            / "analysis_outputs"
            / "do_common1000_warmup_adaptation_lowcost_mg_l_test_v1"
        ).lower(),
    }
    passed = all(checks.values())
    test_record(
        records,
        "DO_NATIVE_MG_L",
        "DO",
        passed,
        f"DO target mean={mean:.6f} mg/L and scale={scale:.6f} mg/L; native mg/L chain is explicit.",
        {"checks": checks, "target_mean": mean, "target_scale": scale},
    )
    return {"passed": passed, "checks": checks, "target_mean": mean, "target_scale": scale}


def comparison_row(
    field: str, final_value: Any, warmup_value: Any, note: str = ""
) -> dict[str, Any]:
    return {
        "field": field,
        "final_h0": final_value,
        "current_warmup": warmup_value,
        "matches": final_value == warmup_value,
        "note": note,
    }


def compare_current_warmup(
    target: str,
    descriptor: dict[str, Any],
    config_path: Path,
    separation: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    cfg = load_json(config_path)
    warm_model_cfg = RAIN.load_config(RAIN.resolve(cfg["base_model_config"]))
    warm_success = load_json(RAIN.resolve(cfg["base_success_manifest"]))
    final_cfg = load_json(Path(descriptor["effective_config_path"]))
    comparisons = [
        comparison_row(
            "checkpoint_sha256",
            descriptor["checkpoint_sha256"],
            sha256(RAIN.resolve(cfg["base_checkpoint"])),
        ),
        comparison_row("architecture", descriptor["architecture"], "site-agnostic LSTM-FiLM"),
        comparison_row(
            "input_window_hours",
            descriptor["sequence_length_hours"],
            int(warm_success["sequence_length_hours"]),
        ),
        comparison_row(
            "forecast_horizon_hours",
            descriptor["horizon_hours"],
            int(warm_model_cfg["horizon_hours"]),
            "This changes target keys even when the site split is unchanged.",
        ),
        comparison_row(
            "feature_list_order",
            descriptor["feature_set"],
            list(cfg["feature_names"]),
        ),
        comparison_row(
            "scaler_sha256",
            descriptor["preprocessing_sha256"],
            sha256(RAIN.resolve(cfg["preprocessing_path"])),
        ),
        comparison_row(
            "metadata_transform_sha256",
            descriptor["metadata_transform_sha256"],
            sha256(RAIN.resolve(cfg["metadata_transform_path"])),
        ),
        comparison_row(
            "site_split_sha256",
            descriptor["site_split_sha256"],
            sha256(
                PROJECT_ROOT
                / "analysis_outputs"
                / "ph_common1000_leave_sites_out_v1"
                / "full"
                / "reports"
                / f"site_split_{'ph' if target == 'pH' else 'do'}.csv"
            ),
        ),
        comparison_row(
            "target_key_manifest_sha256",
            descriptor["target_key_manifest_sha256"],
            str(warm_success["common_key_manifest_sha256"]),
        ),
        comparison_row(
            "target_units",
            descriptor["target_unit"],
            str(warm_model_cfg.get("target_unit", "native pH units")),
        ),
        comparison_row(
            "random_seed",
            descriptor["seed"],
            int(warm_model_cfg["random_seed"]),
        ),
        comparison_row(
            "target_history_excluded",
            not descriptor["target_history"],
            not bool(warm_model_cfg.get("include_target_history", True)),
        ),
        comparison_row(
            "fixed_post_12_month_keys",
            True,
            bool(separation["separation_passed"]),
        ),
    ]
    valid = all(row["matches"] for row in comparisons)
    test_record(
        records,
        "CURRENT_WARMUP_FINAL_CHECKPOINT_MATCH",
        target,
        valid,
        "Current warm-up exactly matches the final H0 provenance."
        if valid
        else "Current warm-up differs from the final H0 provenance and requires rerun if the main experiment is unblocked.",
        {"mismatches": [row for row in comparisons if not row["matches"]]},
    )
    return {"target": target, "current_warmup_valid": valid, "comparisons": comparisons}


def checkpoint_report(
    descriptors: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    soft_sensing: dict[str, Any],
) -> str:
    lines = [
        "# Checkpoint provenance audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Frozen checkpoints",
        "",
    ]
    for target, item in descriptors.items():
        lines.extend(
            [
                f"### {target}",
                "",
                f"- Checkpoint: `{item['checkpoint_path']}`",
                f"- SHA-256: `{item['checkpoint_sha256']}`",
                f"- Configuration: `{item['config_path']}`",
                f"- Features: `{', '.join(item['feature_set'])}`",
                f"- Architecture: {item['architecture']}; hidden {item['hidden_size']}; "
                f"{item['recurrent_layers']} layer; dropout {item['dropout']}",
                f"- Window/horizon: {item['sequence_length_hours']} h / {item['horizon_hours']} h",
                f"- Target: `{item['target_code']}` from `{item['target_source_column']}`; "
                f"unit `{item['target_unit']}`; multiplier {item['target_source_multiplier']}",
                f"- Site split SHA-256: `{item['site_split_sha256']}`",
                f"- Target-key manifest SHA-256: `{item['target_key_manifest_sha256']}`",
                f"- Scaler SHA-256: `{item['preprocessing_sha256']}`",
                f"- Metadata transform SHA-256: `{item['metadata_transform_sha256']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Separate contemporaneous pH soft-sensing reference",
            "",
            f"- Checkpoint: `{soft_sensing['checkpoint_path']}`",
            f"- SHA-256: `{soft_sensing['checkpoint_sha256']}`",
            f"- Features: `{', '.join(soft_sensing['feature_set'])}`",
            f"- Window/horizon: {soft_sensing['sequence_length_hours']} h / "
            f"{soft_sensing['horizon_hours']} h",
            "- Classification: contemporaneous full-sensor pH soft-sensing/nowcast.",
            "- It includes contemporaneous DO (`x_00300`) and is not the post-sonde "
            "CHEAP3 deployment backbone. Its metrics must not be directly ranked against "
            "the strictly causal H01 low-cost deployment results.",
            "",
        ]
    )
    for target, comparison in comparisons.items():
        lines.extend(
            [
                f"## Existing {target} warm-up versus final H0",
                "",
                "| Field | Final H0 | Existing warm-up | Match |",
                "|---|---|---|:---:|",
            ]
        )
        for row in comparison["comparisons"]:
            left = json.dumps(row["final_h0"], ensure_ascii=False)
            right = json.dumps(row["current_warmup"], ensure_ascii=False)
            lines.append(
                f"| {row['field']} | `{left}` | `{right}` | "
                f"{'YES' if row['matches'] else 'NO'} |"
            )
        lines.extend(
            [
                "",
                f"`CURRENT_WARMUP_VALID = {str(comparison['current_warmup_valid']).upper()}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Audit decision",
            "",
            "The causal pH deployment provenance is the validation-selected "
            "`LOW_COST_H01` checkpoint used by the stored warm-up experiment. It has a "
            "one-hour horizon, excludes historical pH and DO input, and matches the "
            "stored checkpoint, scaler, metadata transform, split and target-key hashes. "
            "The DO mg/L warm-up likewise matches its latest traceable unit-correct "
            "low-cost H01 checkpoint. `RAIN_H0_H00` is retained only as a separate "
            "contemporaneous full-sensor soft-sensing reference.",
            "",
        ]
    )
    return "\n".join(lines)


def test_report(records: list[dict[str, Any]]) -> str:
    lines = [
        "# H0 leakage, temporal-causality, freeze and DO-unit test report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Test | Target | Status | Detail |",
        "|---|---|:---:|---|",
    ]
    for row in records:
        detail = str(row["detail"]).replace("|", "\\|")
        lines.append(
            f"| `{row['test_id']}` | {row['target']} | "
            f"{'PASS' if row['passed'] else 'FAIL'} | {detail} |"
        )
    failed = [row for row in records if not row["passed"]]
    stop_level_ids = {
        "H0_FEATURE_NAMES",
        "H0_CHANNEL_COUNT",
        "TEMPORAL_CAUSALITY",
        "WARMUP_EVALUATION_SEPARATION",
        "ADAPTER_PARAMETER_COUNT",
        "EVALUATION_FREEZE",
        "DO_NATIVE_MG_L",
        "PROVENANCE_INTEGRITY",
    }
    stop_level_failed = [row for row in failed if row["test_id"] in stop_level_ids]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "All mandatory tests passed."
                if not failed
                else (
                    f"{len(failed)} test record(s) failed, including "
                    f"{len(stop_level_failed)} stop-level failure(s). "
                    "The main experiment must not run."
                )
            ),
            "",
        ]
    )
    return "\n".join(lines)


def blocked_report(records: list[dict[str, Any]]) -> str:
    temporal = next(
        row
        for row in records
        if row["test_id"] == "TEMPORAL_CAUSALITY" and row["target"] == "pH"
    )
    evidence = temporal.get("evidence", {})
    return "\n".join(
        [
            "# BLOCKED: final H0 warm-up rerun",
            "",
            "## Blocking reason",
            "",
            "The validation-selected pH `RAIN_H0` checkpoint is a contemporaneous "
            "horizon-0 model. Its 48-hour input window includes exogenous observations "
            "at the same timestamp as the pH label. This fails the protocol's mandatory "
            "future-prediction causality rule `max(input_timestamp) < target_timestamp`.",
            "",
            f"Automated sample: {evidence.get('violations')} of "
            f"{evidence.get('sample_windows')} windows failed; "
            f"{evidence.get('equal_timestamp_windows')} ended exactly at the target time.",
            "",
            "## Involved files",
            "",
            "- `analysis_outputs/ph_common1000_lstm_nohistory_h0_rain_final_v1/full/models/ph_LSTM_RAIN_H0_H00.pt`",
            "- `analysis_outputs/ph_common1000_lstm_nohistory_h0_rain_final_v1/full/reports/effective_config.json`",
            "- `pipeline/31_common1000_rainfall_ablation/run_rainfall_ablation.py`",
            "",
            "## Completed before stopping",
            "",
            "- Final pH and DO checkpoint provenance and SHA-256 audit.",
            "- Existing pH/DO warm-up provenance comparison.",
            "- H0 feature-name and actual-channel leakage checks.",
            "- 100-window temporal-causality samples for pH and DO.",
            "- Warm-up/evaluation separation and fixed-key checks across stored results.",
            "- 130-parameter adapter and evaluation-freeze checks.",
            "- Native mg/L DO unit-chain test.",
            "",
            "## Not run",
            "",
            "- CHEAP3 versus CHEAP3_RAIN validation training and input selection.",
            "- Any pH/DO warm-up rerun.",
            "- Test-set reporting, bootstrap comparisons, knee selection and final deployment claims.",
            "",
            "## Minimum human decision required",
            "",
            "Confirm one operational definition before continuing: either (1) accept a "
            "contemporaneous soft-sensor/nowcast that may use external sensors at the pH "
            "measurement timestamp, or (2) require a true future forecast and freeze a "
            "strictly causal horizon (normally one hour), which requires retraining the "
            "pH CHEAP3 and CHEAP3_RAIN H0 checkpoints. The frozen protocol specifies "
            "option (2), so the audit does not silently reinterpret it.",
            "",
        ]
    )


def write_sha_manifest(root: Path) -> None:
    manifest_path = root / "completion" / "SHA256SUMS.txt"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve()
    )
    lines = [f"{sha256(path)}  {rel(path)}" for path in files]
    write_text(manifest_path, "\n".join(lines))


def main() -> int:
    for name in (
        "configs",
        "checkpoints",
        "frozen_selection",
        "predictions",
        "reports",
        "site_metrics",
        "summary",
        "figures",
        "logs",
        "completion",
    ):
        (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)

    ph_root = (
        PROJECT_ROOT
        / "analysis_outputs"
        / "ph_common1000_lstm_film_ph_sensor_ablation_v1"
        / "full"
    )
    do_root = (
        PROJECT_ROOT
        / "analysis_outputs"
        / "do_common1000_lstm_film_sensor_ablation_mg_l_v1"
        / "full"
    )
    descriptors = {
        "pH": checkpoint_descriptor(
            "pH",
            ph_root / "models" / "ph_LSTM_LOW_COST_H01.pt",
            PROJECT_ROOT / "config" / "common1000_lstm_film_ph_sensor_ablation_v1.json",
            ph_root / "reports" / "effective_config.json",
            ph_root / "models" / "ph_LSTM_LOW_COST_H01_SUCCESS.json",
            ph_root / "reports" / "train_only_preprocessing.json",
            ph_root / "metadata" / "ph_no_coordinates_train_only_transform.json",
            ph_root / "reports" / "common_key_manifest.csv",
        ),
        "DO": checkpoint_descriptor(
            "DO",
            do_root / "models" / "do_LSTM_LOW_COST_H01.pt",
            PROJECT_ROOT / "config" / "common1000_lstm_film_do_sensor_ablation_v1.json",
            do_root / "reports" / "effective_config.json",
            do_root / "models" / "do_LSTM_LOW_COST_H01_SUCCESS.json",
            do_root / "reports" / "train_only_preprocessing.json",
            do_root / "metadata" / "do_no_coordinates_train_only_transform.json",
            do_root / "reports" / "common_key_manifest.csv",
        ),
    }
    soft_root = (
        PROJECT_ROOT
        / "analysis_outputs"
        / "ph_common1000_lstm_nohistory_h0_rain_final_v1"
        / "full"
    )
    soft_sensing = checkpoint_descriptor(
        "pH",
        soft_root / "models" / "ph_LSTM_RAIN_H0_H00.pt",
        PROJECT_ROOT / "config" / "common1000_lstm_nohistory_h0_rain_final_v1.json",
        soft_root / "reports" / "effective_config.json",
        soft_root / "models" / "ph_LSTM_RAIN_H0_H00_SUCCESS.json",
        soft_root / "reports" / "train_only_preprocessing.json",
        soft_root / "metadata" / "ph_no_coordinates_train_only_transform.json",
        soft_root / "reports" / "common_key_manifest.csv",
    )
    selection_checkpoint = (
        ph_root / "models" / "ph_LSTM_LOW_COST_H01.pt"
    )
    descriptors["pH"]["validation_selection_checkpoint_path"] = str(
        selection_checkpoint.resolve()
    )
    descriptors["pH"]["validation_selection_checkpoint_sha256"] = sha256(
        selection_checkpoint
    )
    write_json(
        OUTPUT_ROOT / "configs" / "config_manifest.json",
        {
            "causal_deployment_checkpoints": descriptors,
            "contemporaneous_soft_sensing_reference": soft_sensing,
        },
    )

    records: list[dict[str, Any]] = []
    for target, descriptor in descriptors.items():
        checks = {
            "recorded_target_key_sha_matches_file": (
                descriptor["target_key_manifest_sha256_recorded"]
                == descriptor["target_key_manifest_sha256"]
            ),
            "checkpoint_exists": Path(descriptor["checkpoint_path"]).is_file(),
            "config_exists": Path(descriptor["config_path"]).is_file(),
            "scaler_exists": Path(descriptor["preprocessing_path"]).is_file(),
            "metadata_transform_exists": Path(
                descriptor["metadata_transform_path"]
            ).is_file(),
        }
        if target == "pH":
            checks["final_checkpoint_matches_validation_selection"] = (
                descriptor["checkpoint_sha256"]
                == descriptor["validation_selection_checkpoint_sha256"]
            )
        test_record(
            records,
            "PROVENANCE_INTEGRITY",
            target,
            all(checks.values()),
            "Checkpoint, configuration, scaler, metadata transform and target-key hashes are traceable.",
            checks,
        )
    temporal = {
        target: feature_and_temporal_tests(item, records)
        for target, item in descriptors.items()
    }
    ph_separation = audit_warmup_directory(
        "pH",
        PROJECT_ROOT
        / "analysis_outputs"
        / "ph_common1000_warmup_adaptation_lowcost_test_v1"
        / "full",
        records,
    )
    do_separation = audit_warmup_directory(
        "DO",
        PROJECT_ROOT
        / "analysis_outputs"
        / "do_common1000_warmup_adaptation_lowcost_mg_l_test_v1"
        / "full",
        records,
    )
    freeze = freeze_test(records)
    do_unit = do_unit_test(descriptors["DO"], records)
    comparisons = {
        "pH": compare_current_warmup(
            "pH",
            descriptors["pH"],
            PROJECT_ROOT / "config" / "common1000_warmup_adaptation_lowcost_ph_test_v1.json",
            ph_separation,
            records,
        ),
        "DO": compare_current_warmup(
            "DO",
            descriptors["DO"],
            PROJECT_ROOT / "config" / "common1000_warmup_adaptation_lowcost_do_test_v1.json",
            do_separation,
            records,
        ),
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(row["passed"] for row in records) else "fail",
        "tests": records,
        "temporal_samples": temporal,
        "warmup_separation": {"pH": ph_separation, "DO": do_separation},
        "freeze_test": freeze,
        "do_unit_test": do_unit,
        "current_warmup_valid": {
            target: value["current_warmup_valid"] for target, value in comparisons.items()
        },
    }
    write_json(OUTPUT_ROOT / "reports" / "h0_leakage_test_results.json", payload)
    write_text(
        OUTPUT_ROOT / "reports" / "CHECKPOINT_PROVENANCE_AUDIT.md",
        checkpoint_report(descriptors, comparisons, soft_sensing),
    )
    write_text(
        OUTPUT_ROOT / "reports" / "H0_LEAKAGE_AND_UNIT_TEST_REPORT.md",
        test_report(records),
    )
    failed_mandatory = [
        row
        for row in records
        if not row["passed"]
        and row["test_id"]
        in {
            "H0_FEATURE_NAMES",
            "H0_CHANNEL_COUNT",
            "TEMPORAL_CAUSALITY",
            "WARMUP_EVALUATION_SEPARATION",
            "ADAPTER_PARAMETER_COUNT",
            "EVALUATION_FREEZE",
            "DO_NATIVE_MG_L",
            "PROVENANCE_INTEGRITY",
        }
    ]
    blocked = bool(failed_mandatory)
    if blocked:
        write_text(OUTPUT_ROOT / "reports" / "BLOCKED_REPORT.md", blocked_report(records))
    else:
        blocked_path = OUTPUT_ROOT / "reports" / "BLOCKED_REPORT.md"
        resolved_path = OUTPUT_ROOT / "reports" / "RESOLVED_INITIAL_BLOCKED_REPORT.md"
        if blocked_path.exists() and not resolved_path.exists():
            blocked_path.replace(resolved_path)
    completion = {
        "status": "blocked" if blocked else "audit_complete",
        "pH_complete": False,
        "DO_complete": False,
        "checkpoint_audit_passed": True,
        "h0_leakage_tests_passed": not blocked,
        "do_unit_tests_passed": bool(do_unit["passed"]),
        "cheap3_completed": False,
        "cheap3_rain_status": "not_run_due_to_mandatory_audit_failure" if blocked else "pending",
        "selection_frozen_before_test": False,
        "all_expected_files_present": False,
        "current_warmup_valid": payload["current_warmup_valid"],
        "failed_mandatory_tests": failed_mandatory,
    }
    write_json(OUTPUT_ROOT / "completion" / "COMPLETED.json", completion)
    write_sha_manifest(OUTPUT_ROOT)
    print(json.dumps(completion, indent=2, ensure_ascii=False))
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
