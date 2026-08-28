#!/usr/bin/env python
"""Independent audit for the Common1000 FiLM/scaling/non-event workflow."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "common1000_film_scaling_v1.json",
    )
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def metric_values(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    y64 = np.asarray(y, dtype=np.float64)
    p64 = np.asarray(p, dtype=np.float64)
    residual = y64 - p64
    sse = float(np.square(residual).sum())
    denominator = float(np.square(y64 - y64.mean()).sum())
    return math.sqrt(sse / len(y64)), float(np.abs(residual).mean()), 1 - sse / denominator


def key_hash(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame[["site_id", "timestamp", "observed_ph"]], index=False).to_numpy()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = resolve(config["output_root"])
    reports = output / "reports"
    summary = json.loads((reports / "run_summary.json").read_text(encoding="utf-8"))
    overall_report = pd.read_csv(reports / "model_metrics.csv")
    network_report = pd.read_csv(reports / "network_metrics.csv")
    fixed_report = pd.read_csv(reports / "fixed_common_site_scaling_metrics.csv")
    cohort = pd.read_csv(resolve(config["sequence_input"]) / "reports" / "cohort.csv", dtype={"site_id": "string"})
    site_order = cohort["site_id"].astype(str).tolist()
    problems: list[str] = []
    checks: list[dict[str, Any]] = []
    frames: dict[tuple[str, str], pd.DataFrame] = {}

    for model_id in summary["models_trained"]:
        for split in ("val", "test"):
            frame = pd.read_parquet(output / "predictions" / f"prediction_{split}_{model_id}.parquet")
            frame["site_id"] = frame["site_id"].astype(str)
            nulls = int(frame.isna().sum().sum())
            duplicates = int(frame.duplicated(["site_id", "timestamp"]).sum())
            numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64, copy=False)
            nonfinite = int((~np.isfinite(numeric)).sum())
            scale_error = float(np.max(np.abs(frame["observed_scaled"].to_numpy(dtype=float) * 14 - frame["observed_ph"])))
            residual_error = float(
                np.max(
                    np.abs(
                        frame["observed_ph"].to_numpy()
                        - frame["predicted_ph"].to_numpy()
                        - frame["residual_observed_minus_predicted"].to_numpy()
                    )
                )
            )
            rmse, mae, r2 = metric_values(frame["observed_ph"], frame["predicted_ph"])
            expected = overall_report[
                overall_report["model_id"].eq(model_id) & overall_report["split"].eq(split)
            ].iloc[0]
            metric_delta = max(abs(rmse - expected.rmse), abs(mae - expected.mae), abs(r2 - expected.r2))
            network_delta = 0.0
            for network, group in frame.groupby("network", sort=False):
                a, b, c = metric_values(group["observed_ph"], group["predicted_ph"])
                expected_network = network_report[
                    network_report["model_id"].eq(model_id)
                    & network_report["split"].eq(split)
                    & network_report["network"].eq(network)
                ].iloc[0]
                network_delta = max(
                    network_delta,
                    abs(a - expected_network.rmse),
                    abs(b - expected_network.mae),
                    abs(c - expected_network.r2),
                    abs(len(group) - expected_network.n_rows),
                )
            if nulls or duplicates or nonfinite or scale_error > 1e-6 or residual_error > 1e-7 or metric_delta > 2e-10 or network_delta > 2e-9:
                problems.append(f"{model_id}/{split}: prediction/report audit failed")
            checks.append(
                {
                    "model_id": model_id,
                    "split": split,
                    "rows": len(frame),
                    "sites": int(frame["site_id"].nunique()),
                    "time_min": str(frame["timestamp"].min()),
                    "time_max": str(frame["timestamp"].max()),
                    "nulls": nulls,
                    "duplicates": duplicates,
                    "nonfinite": nonfinite,
                    "metric_max_delta": metric_delta,
                    "network_max_delta": network_delta,
                }
            )
            frames[(model_id, split)] = frame

    n200_models = [model for model in summary["models_trained"] if model.endswith("_N200")]
    for split in ("val", "test"):
        reference = key_hash(frames[(n200_models[0], split)])
        for model_id in n200_models[1:]:
            if not np.array_equal(reference, key_hash(frames[(model_id, split)])):
                problems.append(f"N200 {split} keys differ for {model_id}")
    full_id = f"{summary['winner_base']}_N1009"
    nonevent_id = str(summary["nonevent_model"])
    for split in ("val", "test"):
        if not np.array_equal(key_hash(frames[(full_id, split)]), key_hash(frames[(nonevent_id, split)])):
            problems.append(f"N1009 all/non-event {split} keys differ")

    resources = pd.read_csv(reports / "resource_usage.csv").set_index("model_id")
    if resources.loc[full_id, "initial_state_sha256"] != resources.loc[nonevent_id, "initial_state_sha256"]:
        problems.append("N1009 all/non-event initial states differ")
    if int(summary["full_train_windows"]) - int(summary["event_target_windows_removed"]) != int(summary["nonevent_train_windows"]):
        problems.append("non-event training counts do not reconcile")

    # Recompute fixed-common-site tables by filtering the already saved predictions.
    for row in fixed_report.itertuples(index=False):
        frame = frames[(row.model_id, "test")]
        allowed = set(site_order[: int(row.common_test_site_count)])
        subset = frame[frame["site_id"].isin(allowed)]
        a, b, c = metric_values(subset["observed_ph"], subset["predicted_ph"])
        # BF16 recurrent kernels can differ at ~1e-7 when the same rows are
        # evaluated in a different surrounding batch composition.
        if abs(len(subset) - row.n_rows) > 0 or max(abs(a - row.rmse), abs(b - row.mae), abs(c - row.r2)) > 5e-7:
            problems.append(f"fixed common-site metric mismatch: {row.model_id}/N{row.common_test_site_count}")

    for count in config["site_counts"]:
        feature_manifest = json.loads(
            (reports / f"feature_manifest_N{count}.json").read_text(encoding="utf-8")
        )
        inputs = [*feature_manifest["dynamic_features"], *feature_manifest["metadata_features"]]
        if any(value in inputs for value in ["00400", "in_ph_event", "event_id"]):
            problems.append(f"N{count} feature manifest leaks target/event fields")
        scaler = json.loads((output / "scalers" / f"dynamic_scaler_N{count}.json").read_text(encoding="utf-8"))
        expected_train = overall_report[overall_report["model_id"].eq(f"{summary['winner_base']}_N{count}")]
        if expected_train.empty:
            problems.append(f"N{count} scaling model missing")
        if int(scaler["fit_training_windows"]) <= 0:
            problems.append(f"N{count} scaler has no training windows")

    if summary.get("validation_problems") != [] or not (output / "_SUCCESS").exists():
        problems.append("run summary or _SUCCESS is invalid")
    audit = {
        "status": "pass" if not problems else "fail",
        "prediction_checks": checks,
        "n200_models": n200_models,
        "all_data_model": full_id,
        "nonevent_model": nonevent_id,
        "problems": problems,
    }
    (reports / "independent_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": audit["status"], "files_checked": len(checks), "problems": problems}, ensure_ascii=False))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
