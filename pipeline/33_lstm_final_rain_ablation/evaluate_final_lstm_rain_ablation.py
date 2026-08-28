"""Evaluate the frozen LSTM-FiLM baseline/rain test ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if not np.isfinite(value) else f"{value:.6g}"
            )
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def bootstrap_mean_ci(
    values: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    random = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1000):
        stop = min(start + 1000, replicates)
        sample = random.integers(
            0,
            len(values),
            size=(stop - start, len(values)),
        )
        means[start:stop] = values[sample].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    experiment = resolve(args.experiment_root) / "full"
    reports = experiment / "reports"
    overall = pd.read_csv(reports / "model_metrics_all.csv")
    sites = pd.read_csv(
        reports / "site_metrics_all.csv",
        dtype={"site_id": "string"},
    )
    manifest = pd.read_csv(reports / "run_manifest.csv")

    roles: dict[str, str] = {}
    for row in manifest.itertuples(index=False):
        role = "rain" if "rain_00045" in str(row.feature_set) else "base"
        if role in roles:
            raise RuntimeError(f"Duplicate model role: {role}")
        roles[role] = str(row.run_id)
    if set(roles) != {"base", "rain"}:
        raise RuntimeError(f"Unexpected model roles: {roles}")

    base_run = roles["base"]
    rain_run = roles["rain"]
    test_overall = overall[overall["split"].eq("test")].set_index("run_id")
    test_sites = sites[sites["split"].eq("test")].copy()
    problems: list[str] = []

    base_sites = test_sites.loc[
        test_sites["run_id"].eq(base_run),
        ["site_id", "rmse", "mae", "nse"],
    ]
    rain_sites = test_sites.loc[
        test_sites["run_id"].eq(rain_run),
        ["site_id", "rmse", "mae", "nse"],
    ]
    paired = base_sites.merge(
        rain_sites,
        on="site_id",
        suffixes=("_base", "_rain"),
        validate="one_to_one",
    )
    if len(paired) != 135:
        problems.append(f"expected 135 paired test sites, found {len(paired)}")
    paired["rmse_delta_rain_minus_base"] = (
        paired["rmse_rain"] - paired["rmse_base"]
    )
    paired["mae_delta_rain_minus_base"] = (
        paired["mae_rain"] - paired["mae_base"]
    )
    paired_delta = paired["rmse_delta_rain_minus_base"].to_numpy(
        dtype=np.float64
    )
    paired_ci = bootstrap_mean_ci(
        paired_delta,
        int(args.bootstrap_replicates),
        int(args.seed),
    )
    base_overall_rmse = float(test_overall.loc[base_run, "rmse"])
    rain_overall_rmse = float(test_overall.loc[rain_run, "rmse"])
    paired_summary = pd.DataFrame(
        [
            {
                "base_run": base_run,
                "rain_run": rain_run,
                "test_sites": len(paired),
                "base_overall_rmse": base_overall_rmse,
                "rain_overall_rmse": rain_overall_rmse,
                "overall_rmse_change_fraction": (
                    rain_overall_rmse / base_overall_rmse - 1.0
                ),
                "mean_site_rmse_delta_rain_minus_base": float(
                    paired_delta.mean()
                ),
                "median_site_rmse_delta_rain_minus_base": float(
                    np.median(paired_delta)
                ),
                "mean_site_rmse_delta_ci_low": paired_ci[0],
                "mean_site_rmse_delta_ci_high": paired_ci[1],
                "rain_improved_sites": int((paired_delta < 0).sum()),
                "base_improved_sites": int((paired_delta > 0).sum()),
                "tied_sites": int((paired_delta == 0).sum()),
            }
        ]
    )

    strata = (
        "all",
        "event",
        "non_event",
        "low_event",
        "high_event",
        "current_rain",
        "current_dry",
        "wet_previous_6h",
        "dry_previous_6h",
    )
    pooled = {
        name: {
            "n": 0,
            "base_sse": 0.0,
            "rain_sse": 0.0,
            "sites": set(),
        }
        for name in strata
    }
    stratum_site_rows = []
    base_paths = sorted(
        (experiment / "predictions_by_site" / base_run / "test").glob(
            "*.parquet"
        )
    )
    rain_paths = {
        path.stem: path
        for path in (
            experiment / "predictions_by_site" / rain_run / "test"
        ).glob("*.parquet")
    }
    if len(base_paths) != 135 or len(rain_paths) != 135:
        problems.append(
            f"expected 135 prediction files per model; "
            f"base={len(base_paths)}, rain={len(rain_paths)}"
        )

    columns = [
        "site_id",
        "timestamp",
        "observed_native",
        "predicted_native",
        "in_event",
        "event_direction",
        "input_end_rain_mm",
        "input_end_precip_6h",
    ]
    for site_number, base_path in enumerate(base_paths):
        site_id = base_path.stem
        if site_id not in rain_paths:
            problems.append(f"missing rain prediction file for {site_id}")
            continue
        base = pd.read_parquet(base_path, columns=columns)
        rain = pd.read_parquet(rain_paths[site_id], columns=columns)
        if len(base) != len(rain):
            problems.append(f"row mismatch for {site_id}")
            continue
        if not np.array_equal(
            pd.to_datetime(base["timestamp"]).to_numpy(dtype="datetime64[ns]"),
            pd.to_datetime(rain["timestamp"]).to_numpy(dtype="datetime64[ns]"),
        ):
            problems.append(f"timestamp mismatch for {site_id}")
            continue
        observed = base["observed_native"].to_numpy(dtype=np.float64)
        rain_observed = rain["observed_native"].to_numpy(dtype=np.float64)
        if not np.allclose(observed, rain_observed, equal_nan=True):
            problems.append(f"observed target mismatch for {site_id}")
            continue
        base_error = (
            base["predicted_native"].to_numpy(dtype=np.float64) - observed
        )
        rain_error = (
            rain["predicted_native"].to_numpy(dtype=np.float64) - observed
        )
        finite = (
            np.isfinite(observed)
            & np.isfinite(base_error)
            & np.isfinite(rain_error)
        )
        event = base["in_event"].fillna(False).to_numpy(dtype=bool)
        direction = base["event_direction"].astype(str).to_numpy()
        current_rain = pd.to_numeric(
            base["input_end_rain_mm"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        rain_6h = pd.to_numeric(
            base["input_end_precip_6h"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        masks = {
            "all": finite,
            "event": finite & event,
            "non_event": finite & ~event,
            "low_event": finite & (direction == "low"),
            "high_event": finite & (direction == "high"),
            "current_rain": finite & np.isfinite(current_rain) & (current_rain > 0),
            "current_dry": finite & np.isfinite(current_rain) & (current_rain == 0),
            "wet_previous_6h": finite & np.isfinite(rain_6h) & (rain_6h > 0),
            "dry_previous_6h": finite & np.isfinite(rain_6h) & (rain_6h == 0),
        }
        for stratum_number, (name, mask) in enumerate(masks.items()):
            n = int(mask.sum())
            if not n:
                continue
            base_sse = float(np.square(base_error[mask]).sum())
            rain_sse = float(np.square(rain_error[mask]).sum())
            base_rmse = float(np.sqrt(base_sse / n))
            rain_rmse = float(np.sqrt(rain_sse / n))
            pooled[name]["n"] += n
            pooled[name]["base_sse"] += base_sse
            pooled[name]["rain_sse"] += rain_sse
            pooled[name]["sites"].add(site_id)
            stratum_site_rows.append(
                {
                    "stratum": name,
                    "site_id": site_id,
                    "n": n,
                    "base_rmse": base_rmse,
                    "rain_rmse": rain_rmse,
                    "rmse_delta_rain_minus_base": rain_rmse - base_rmse,
                }
            )

    stratum_sites = pd.DataFrame(stratum_site_rows)
    stratum_rows = []
    for number, name in enumerate(strata):
        values = pooled[name]
        n = int(values["n"])
        if not n:
            continue
        base_rmse = float(np.sqrt(values["base_sse"] / n))
        rain_rmse = float(np.sqrt(values["rain_sse"] / n))
        subset = stratum_sites[stratum_sites["stratum"].eq(name)]
        delta = subset["rmse_delta_rain_minus_base"].to_numpy(
            dtype=np.float64
        )
        ci_low, ci_high = bootstrap_mean_ci(
            delta,
            int(args.bootstrap_replicates),
            int(args.seed) + number + 1,
        )
        stratum_rows.append(
            {
                "stratum": name,
                "rows": n,
                "sites": len(values["sites"]),
                "base_pooled_rmse": base_rmse,
                "rain_pooled_rmse": rain_rmse,
                "pooled_rmse_change_fraction": (
                    rain_rmse / base_rmse - 1.0
                ),
                "mean_site_rmse_delta_rain_minus_base": float(delta.mean()),
                "mean_site_rmse_delta_ci_low": ci_low,
                "mean_site_rmse_delta_ci_high": ci_high,
                "rain_improved_sites": int((delta < 0).sum()),
                "base_improved_sites": int((delta > 0).sum()),
            }
        )
    stratified = pd.DataFrame(stratum_rows)

    training_rows = []
    for row in manifest.itertuples(index=False):
        success = json.loads(
            (experiment / "models" / f"{row.run_id}_SUCCESS.json").read_text(
                encoding="utf-8"
            )
        )
        training_rows.append(
            {
                "run_id": row.run_id,
                "feature_role": (
                    "rain" if "rain_00045" in str(row.feature_set) else "base"
                ),
                "best_epoch": int(success["best_epoch"]),
                "epochs_ran": int(success["epochs_ran"]),
                "capped_validation_rmse_at_best_epoch": float(
                    success["best_validation_rmse_native"]
                ),
                "trainable_parameters": int(row.trainable_parameters),
            }
        )
    training = pd.DataFrame(training_rows)

    write_csv(reports / "final_lstm_paired_site_metrics.csv", paired)
    write_csv(reports / "final_lstm_paired_summary.csv", paired_summary)
    write_csv(reports / "final_lstm_stratified_metrics.csv", stratified)
    write_csv(reports / "final_lstm_stratum_site_metrics.csv", stratum_sites)
    write_csv(reports / "final_lstm_training_summary.csv", training)

    primary = paired_summary.iloc[0]
    event_row = stratified[stratified["stratum"].eq("event")].iloc[0]
    wet_row = stratified[
        stratified["stratum"].eq("wet_previous_6h")
    ].iloc[0]
    report = "\n".join(
        [
            "# Frozen LSTM-FiLM rainfall ablation",
            "",
            "The model configuration was frozen before this final test evaluation: "
            "48-hour input, learning rate 0.0003, hidden size 64, one LSTM layer, "
            "dropout 0.1, all-data training and teacher-aligned DO conversion.",
            "",
            "BASE and RAIN use identical held-out sites and target timestamps. "
            "Negative paired RMSE deltas favour RAIN. Confidence intervals are "
            "95% site-block bootstrap intervals.",
            "",
            "## Overall held-out test metrics",
            "",
            markdown_table(
                test_overall.reset_index()[
                    [
                        "run_id",
                        "rmse",
                        "mae",
                        "nse",
                        "macro_site_rmse",
                        "macro_site_nse",
                    ]
                ].sort_values("rmse")
            ),
            "",
            "## Paired site comparison",
            "",
            markdown_table(paired_summary),
            "",
            "## Test strata",
            "",
            markdown_table(stratified),
            "",
            "## Training summary",
            "",
            markdown_table(training),
            "",
            "## Interpretation",
            "",
            f"Adding rainfall changed overall test RMSE by "
            f"{float(primary['overall_rmse_change_fraction']) * 100:+.2f}% and "
            f"improved {int(primary['rain_improved_sites'])}/135 held-out sites. "
            f"The paired mean site-RMSE interval was "
            f"[{float(primary['mean_site_rmse_delta_ci_low']):.6f}, "
            f"{float(primary['mean_site_rmse_delta_ci_high']):.6f}].",
            "",
            f"Within labelled event hours, rainfall changed pooled RMSE by "
            f"{float(event_row['pooled_rmse_change_fraction']) * 100:+.2f}%. "
            f"Within hours having precipitation in the previous six hours, it "
            f"changed pooled RMSE by "
            f"{float(wet_row['pooled_rmse_change_fraction']) * 100:+.2f}%.",
            "",
            "This is the frozen one-seed final ablation. It tests the rainfall "
            "feature under one pre-specified model configuration; it does not "
            "estimate seed-to-seed optimisation variance.",
            "",
        ]
    )
    (reports / "FINAL_LSTM_RAIN_ABLATION_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    audit = {
        "status": "pass" if not problems else "fail",
        "roles": roles,
        "paired_test_sites": len(paired),
        "test_prediction_files_per_model": {
            "base": len(base_paths),
            "rain": len(rain_paths),
        },
        "prediction_keys_identical": not any(
            "mismatch" in problem for problem in problems
        ),
        "configuration_frozen_before_test": True,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": int(args.seed),
        "problems": problems,
        "manifest_sha256": hashlib.sha256(
            (reports / "run_manifest.csv").read_bytes()
        ).hexdigest(),
    }
    write_json(reports / "final_lstm_rain_ablation_audit.json", audit)
    if problems:
        raise RuntimeError(problems)
    print(report)


if __name__ == "__main__":
    main()
