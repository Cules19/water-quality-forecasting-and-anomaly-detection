"""Select an LSTM-FiLM input window using validation data only."""

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
            values.append(f"{value:.8g}" if isinstance(value, float) else str(value))
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
    selection_config = config["hyperparameter_selection"]

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

    sequence_lengths = {
        f"ph_{model['model_id']}_H01": int(model["sequence_length_hours"])
        for model in config["models"]
    }
    if set(overall["run_id"].astype(str)) != set(sequence_lengths):
        problems.append("configured and observed run IDs differ")
    if overall.groupby("split")["n"].nunique().max() != 1:
        problems.append("models do not share the same number of validation targets")
    if overall.groupby("split")["sites"].nunique().max() != 1:
        problems.append("models do not share the same validation sites")

    ranking = overall.copy()
    ranking["sequence_length_hours"] = ranking["run_id"].map(sequence_lengths)
    ranking = ranking.merge(
        run_manifest[
            ["run_id", "learning_rate", "dropout", "trainable_parameters"]
        ],
        on="run_id",
        how="left",
    )
    training_rows = []
    for run_id in ranking["run_id"].astype(str):
        info = load_json(output / "models" / f"{run_id}_SUCCESS.json")
        history = pd.read_csv(reports / f"training_history_{run_id}.csv")
        training_rows.append(
            {
                "run_id": run_id,
                "best_epoch": int(info["best_epoch"]),
                "epochs_ran": int(info["epochs_ran"]),
                "capped_validation_rmse": float(
                    info["best_validation_rmse_native"]
                ),
                "final_learning_rate": float(history["learning_rate"].iloc[-1]),
            }
        )
    ranking = ranking.merge(pd.DataFrame(training_rows), on="run_id", how="left")
    ranking = ranking.sort_values(
        ["rmse", "macro_site_rmse", "sequence_length_hours"],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "validation_rank", range(1, len(ranking) + 1))
    raw_best_rmse = float(ranking.loc[0, "rmse"])
    ranking["rmse_above_raw_best_pct"] = (
        (ranking["rmse"] - raw_best_rmse) / raw_best_rmse * 100.0
    )

    reference_sequence = int(selection_config["reference_sequence_length_hours"])
    reference_run_id = next(
        run_id
        for run_id, sequence in sequence_lengths.items()
        if sequence == reference_sequence
    )
    site_rmse = sites.pivot(
        index="site_id", columns="run_id", values="rmse"
    ).sort_index()
    random = np.random.default_rng(42)
    comparison_rows = []
    for candidate_run_id, candidate_sequence in sequence_lengths.items():
        if candidate_run_id == reference_run_id:
            continue
        paired = site_rmse[[reference_run_id, candidate_run_id]].dropna()
        delta = (
            paired[candidate_run_id] - paired[reference_run_id]
        ).to_numpy(dtype=np.float64)
        bootstrap_mean = delta[
            random.integers(0, len(delta), size=(20000, len(delta)))
        ].mean(axis=1)
        reference_rmse = float(
            ranking.loc[ranking["run_id"].eq(reference_run_id), "rmse"].iloc[0]
        )
        candidate_rmse = float(
            ranking.loc[ranking["run_id"].eq(candidate_run_id), "rmse"].iloc[0]
        )
        comparison_rows.append(
            {
                "reference_run_id": reference_run_id,
                "candidate_run_id": candidate_run_id,
                "candidate_sequence_length_hours": candidate_sequence,
                "paired_sites": len(delta),
                "candidate_minus_reference_overall_rmse": (
                    candidate_rmse - reference_rmse
                ),
                "candidate_improvement_vs_reference_pct": (
                    (reference_rmse - candidate_rmse) / reference_rmse * 100.0
                ),
                "mean_site_rmse_delta_candidate_minus_reference": float(
                    delta.mean()
                ),
                "median_site_rmse_delta_candidate_minus_reference": float(
                    np.median(delta)
                ),
                "bootstrap_95_ci_low": float(
                    np.quantile(bootstrap_mean, 0.025)
                ),
                "bootstrap_95_ci_high": float(
                    np.quantile(bootstrap_mean, 0.975)
                ),
                "candidate_better_sites": int((delta < 0).sum()),
                "reference_better_sites": int((delta > 0).sum()),
                "tied_sites": int((delta == 0).sum()),
            }
        )
    comparisons = pd.DataFrame(comparison_rows).sort_values(
        "candidate_sequence_length_hours"
    )

    raw_best_run_id = str(ranking.loc[0, "run_id"])
    selected_run_id = raw_best_run_id
    decision_reason = "The 48-hour reference has the lowest validation RMSE."
    if raw_best_run_id != reference_run_id:
        comparison = comparisons[
            comparisons["candidate_run_id"].eq(raw_best_run_id)
        ].iloc[0]
        raw_best_macro = float(ranking.loc[0, "macro_site_rmse"])
        reference_macro = float(
            ranking.loc[
                ranking["run_id"].eq(reference_run_id), "macro_site_rmse"
            ].iloc[0]
        )
        improvement_threshold = float(
            selection_config["near_tie_threshold_pct"]
        )
        qualifies = (
            float(comparison["candidate_improvement_vs_reference_pct"])
            >= improvement_threshold
            and float(comparison["bootstrap_95_ci_high"]) < 0
            and raw_best_macro < reference_macro
        )
        if qualifies:
            decision_reason = (
                "The raw-best window exceeds the pre-registered improvement "
                "threshold, improves macro RMSE, and has a paired-site bootstrap "
                "interval entirely below zero."
            )
        else:
            selected_run_id = reference_run_id
            decision_reason = (
                "The raw-best window did not satisfy all pre-registered evidence "
                "criteria, so the teacher-aligned 48-hour reference was retained."
            )

    selected = ranking[ranking["run_id"].eq(selected_run_id)].iloc[0]
    ranking_columns = [
        "validation_rank",
        "run_id",
        "sequence_length_hours",
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
        "rmse_above_raw_best_pct",
    ]
    ranking[ranking_columns].to_csv(
        reports / "window_sweep_validation_ranking.csv", index=False
    )
    comparisons.to_csv(
        reports / "window_sweep_validation_pairwise_bootstrap.csv", index=False
    )

    audit = {
        "status": "pass" if not problems else "fail",
        "selection_split": "validation",
        "primary_metric": "overall_rmse",
        "common_target_key_anchor_hours": int(
            config["comparison_anchor_sequence_length_hours"]
        ),
        "test_prediction_files_read": 0,
        "test_metrics_read": 0,
        "test_prediction_artifacts_found": len(test_artifacts),
        "raw_best_run_id": raw_best_run_id,
        "selected_run_id": selected_run_id,
        "selected_sequence_length_hours": int(
            selected["sequence_length_hours"]
        ),
        "selected_validation_rmse": float(selected["rmse"]),
        "decision_reason": decision_reason,
        "problems": problems,
    }
    (reports / "window_sweep_selection_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    report = [
        "# LSTM-FiLM rainfall input-window sweep",
        "",
        "All models used identical target timestamps anchored to 96-hour "
        "eligibility. Selection used validation data only; no test prediction "
        "or test metric was read or generated.",
        "",
        frame_to_markdown(ranking[ranking_columns]),
        "",
        "## Paired comparison with the 48-hour reference",
        "",
        frame_to_markdown(comparisons),
        "",
        f"Raw best: {raw_best_run_id}. Selected: {selected_run_id} "
        f"({int(selected['sequence_length_hours'])} hours), validation RMSE "
        f"= {float(selected['rmse']):.8f}.",
        "",
        decision_reason,
        "",
        f"Audit status: {audit['status']}.",
        "",
    ]
    (reports / "LSTM_WINDOW_SWEEP_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    if problems:
        raise RuntimeError(problems)
    print("\n".join(report))


if __name__ == "__main__":
    main()
