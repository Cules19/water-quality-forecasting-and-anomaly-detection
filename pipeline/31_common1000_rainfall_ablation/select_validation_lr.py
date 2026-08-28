"""Rank an LSTM-FiLM learning-rate sweep without reading test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_json(config_path)
    output = resolve(config["output_root"]) / f"{args.mode}_validation_only"
    reports = output / "reports"

    overall = pd.read_csv(reports / "model_metrics_all.csv")
    sites = pd.read_csv(reports / "site_metrics_all.csv")
    run_manifest = pd.read_csv(reports / "run_manifest.csv")
    training_audit = load_json(reports / "independent_audit.json")

    problems: list[str] = []
    if set(overall["split"].astype(str)) != {"validation"}:
        problems.append("overall metrics include a non-validation split")
    if set(sites["split"].astype(str)) != {"validation"}:
        problems.append("site metrics include a non-validation split")
    if training_audit.get("prediction_splits") != ["validation"]:
        problems.append("training audit does not declare validation-only predictions")
    if training_audit.get("test_predictions_generated") is not False:
        problems.append("training audit says test predictions were generated")
    if training_audit.get("test_metrics_generated") is not False:
        problems.append("training audit says test metrics were generated")

    test_artifacts = [
        path
        for path in (output / "predictions_by_site").rglob("*")
        if "test" in {part.lower() for part in path.parts}
    ]
    if test_artifacts:
        problems.append(f"found {len(test_artifacts)} test prediction artifacts")

    learning_rates = {
        f"ph_{model['model_id']}_H01": float(
            model.get("training_overrides", {}).get(
                "learning_rate", config["learning_rate"]
            )
        )
        for model in config["models"]
    }
    expected_runs = set(learning_rates)
    observed_runs = set(overall["run_id"].astype(str))
    if observed_runs != expected_runs:
        problems.append(
            f"run mismatch: expected={sorted(expected_runs)}, observed={sorted(observed_runs)}"
        )

    ranking = overall.copy()
    ranking["learning_rate"] = ranking["run_id"].map(learning_rates)
    ranking = ranking.merge(
        run_manifest[
            ["run_id", "recurrent_type", "dropout", "trainable_parameters"]
        ],
        on="run_id",
        how="left",
    )
    training_rows = []
    for run_id in ranking["run_id"].astype(str):
        info = load_json(output / "models" / f"{run_id}_SUCCESS.json")
        training_rows.append(
            {
                "run_id": run_id,
                "best_epoch": int(info["best_epoch"]),
                "epochs_ran": int(info["epochs_ran"]),
                "capped_validation_rmse": float(
                    info["best_validation_rmse_native"]
                ),
                "final_learning_rate": float(
                    pd.read_csv(
                        reports / f"training_history_{run_id}.csv"
                    )["learning_rate"].iloc[-1]
                ),
            }
        )
    ranking = ranking.merge(pd.DataFrame(training_rows), on="run_id", how="left")
    ranking = ranking.sort_values(
        ["rmse", "macro_site_rmse", "learning_rate"], kind="mergesort"
    ).reset_index(drop=True)
    ranking.insert(0, "validation_rank", range(1, len(ranking) + 1))
    best_rmse = float(ranking.loc[0, "rmse"])
    ranking["rmse_above_best_pct"] = (
        (ranking["rmse"] - best_rmse) / best_rmse * 100.0
    )

    selected_columns = [
        "validation_rank",
        "run_id",
        "learning_rate",
        "n",
        "sites",
        "rmse",
        "mae",
        "nse",
        "macro_site_rmse",
        "macro_site_nse",
        "best_epoch",
        "epochs_ran",
        "capped_validation_rmse",
        "final_learning_rate",
        "rmse_above_best_pct",
    ]
    ranking[selected_columns].to_csv(
        reports / "lr_sweep_validation_ranking.csv", index=False
    )

    best = ranking.iloc[0]
    best_run_id = str(best["run_id"])
    site_rmse = sites.pivot(
        index="site_id", columns="run_id", values="rmse"
    ).sort_index()
    pairwise_rows = []
    random = np.random.default_rng(42)
    for candidate_run_id in ranking["run_id"].astype(str):
        if candidate_run_id == best_run_id:
            continue
        paired = site_rmse[[best_run_id, candidate_run_id]].dropna()
        delta = (
            paired[candidate_run_id] - paired[best_run_id]
        ).to_numpy(dtype=np.float64)
        bootstrap_mean = delta[
            random.integers(0, len(delta), size=(20000, len(delta)))
        ].mean(axis=1)
        candidate_overall_rmse = float(
            ranking.loc[
                ranking["run_id"].eq(candidate_run_id), "rmse"
            ].iloc[0]
        )
        pairwise_rows.append(
            {
                "selected_run_id": best_run_id,
                "candidate_run_id": candidate_run_id,
                "paired_sites": len(delta),
                "candidate_minus_selected_overall_rmse": (
                    candidate_overall_rmse - best_rmse
                ),
                "candidate_rmse_above_selected_pct": (
                    (candidate_overall_rmse - best_rmse) / best_rmse * 100.0
                ),
                "mean_site_rmse_delta_candidate_minus_selected": float(
                    delta.mean()
                ),
                "median_site_rmse_delta_candidate_minus_selected": float(
                    np.median(delta)
                ),
                "bootstrap_95_ci_low": float(
                    np.quantile(bootstrap_mean, 0.025)
                ),
                "bootstrap_95_ci_high": float(
                    np.quantile(bootstrap_mean, 0.975)
                ),
                "selected_better_sites": int((delta > 0).sum()),
                "candidate_better_sites": int((delta < 0).sum()),
                "tied_sites": int((delta == 0).sum()),
            }
        )
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(
        reports / "lr_sweep_validation_pairwise_bootstrap.csv", index=False
    )

    audit = {
        "status": "pass" if not problems else "fail",
        "selection_split": "validation",
        "primary_metric": "overall_rmse",
        "test_prediction_files_read": 0,
        "test_metrics_read": 0,
        "test_prediction_artifacts_found": len(test_artifacts),
        "selected_run_id": str(best["run_id"]),
        "selected_learning_rate": float(best["learning_rate"]),
        "selected_validation_rmse": float(best["rmse"]),
        "problems": problems,
    }
    (reports / "lr_sweep_selection_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    report = [
        "# LSTM-FiLM rainfall learning-rate sweep",
        "",
        "Selection used validation data only. No test prediction or test metric "
        "was read or generated during this sweep.",
        "",
        frame_to_markdown(ranking[selected_columns]),
        "",
        "## Paired validation-site check",
        "",
        frame_to_markdown(pairwise),
        "",
        f"Selected learning rate: {float(best['learning_rate']):.8g} "
        f"({best['run_id']}); validation RMSE = {float(best['rmse']):.8f}.",
        "",
        "The selected learning rate is a tuning result, not a final test result. "
        "The test split remains sealed until the full LSTM-FiLM configuration is frozen.",
        "",
        f"Audit status: {audit['status']}.",
        "",
    ]
    (reports / "LSTM_LR_SWEEP_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    if problems:
        raise RuntimeError(problems)
    print("\n".join(report))


if __name__ == "__main__":
    main()
