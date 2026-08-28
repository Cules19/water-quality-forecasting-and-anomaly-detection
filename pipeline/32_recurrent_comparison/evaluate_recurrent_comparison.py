"""Paired site-level audit for the matched GRU/LSTM x baseline/rain experiment."""

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
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1000):
        stop = min(start + 1000, replicates)
        sample = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[sample].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def training_summary(
    experiment: Path,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for row in manifest.itertuples(index=False):
        success_path = experiment / "models" / f"{row.run_id}_SUCCESS.json"
        success = json.loads(success_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": row.run_id,
                "recurrent_type": row.recurrent_type,
                "feature_role": (
                    "rain" if "rain_00045" in str(row.feature_set) else "base"
                ),
                "trainable_parameters": int(row.trainable_parameters),
                "best_epoch": int(success["best_epoch"]),
                "epochs_ran": int(success["epochs_ran"]),
                "capped_validation_rmse_at_best_epoch": float(
                    success["best_validation_rmse_native"]
                ),
            }
        )
    return pd.DataFrame(rows)


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
    test_overall = overall.loc[overall["split"].eq("test")].set_index("run_id")
    test_sites = sites.loc[sites["split"].eq("test")].copy()

    roles: dict[tuple[str, str], str] = {}
    for row in manifest.itertuples(index=False):
        feature_role = "rain" if "rain_00045" in str(row.feature_set) else "base"
        key = (str(row.recurrent_type), feature_role)
        if key in roles:
            raise RuntimeError(f"Duplicate model role: {key}")
        roles[key] = str(row.run_id)
    expected_roles = {
        ("gru", "base"),
        ("gru", "rain"),
        ("lstm", "base"),
        ("lstm", "rain"),
    }
    if set(roles) != expected_roles:
        raise RuntimeError(f"Unexpected model roles: {set(roles)}")

    comparisons = [
        ("rain_effect_within_gru", roles[("gru", "base")], roles[("gru", "rain")]),
        (
            "rain_effect_within_lstm",
            roles[("lstm", "base")],
            roles[("lstm", "rain")],
        ),
        (
            "lstm_vs_gru_without_rain",
            roles[("gru", "base")],
            roles[("lstm", "base")],
        ),
        (
            "lstm_vs_gru_with_rain",
            roles[("gru", "rain")],
            roles[("lstm", "rain")],
        ),
    ]
    rows = []
    problems: list[str] = []
    for number, (name, reference, challenger) in enumerate(comparisons):
        reference_sites = test_sites.loc[
            test_sites["run_id"].eq(reference),
            ["site_id", "rmse", "mae", "nse"],
        ]
        challenger_sites = test_sites.loc[
            test_sites["run_id"].eq(challenger),
            ["site_id", "rmse", "mae", "nse"],
        ]
        paired = reference_sites.merge(
            challenger_sites,
            on="site_id",
            suffixes=("_reference", "_challenger"),
            validate="one_to_one",
        )
        if len(paired) != 135:
            problems.append(f"{name}: expected 135 paired test sites, found {len(paired)}")
        delta = (
            paired["rmse_challenger"] - paired["rmse_reference"]
        ).to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_mean_ci(
            delta,
            int(args.bootstrap_replicates),
            int(args.seed) + number,
        )
        reference_rmse = float(test_overall.loc[reference, "rmse"])
        challenger_rmse = float(test_overall.loc[challenger, "rmse"])
        rows.append(
            {
                "comparison": name,
                "reference_run": reference,
                "challenger_run": challenger,
                "negative_delta_favors_challenger": True,
                "test_sites": int(len(paired)),
                "reference_overall_rmse": reference_rmse,
                "challenger_overall_rmse": challenger_rmse,
                "overall_rmse_change_fraction": (
                    challenger_rmse / reference_rmse - 1.0
                ),
                "mean_site_rmse_delta": float(delta.mean()),
                "median_site_rmse_delta": float(np.median(delta)),
                "mean_site_rmse_delta_ci_low": ci_low,
                "mean_site_rmse_delta_ci_high": ci_high,
                "challenger_improved_sites": int((delta < 0).sum()),
                "reference_improved_sites": int((delta > 0).sum()),
                "tied_sites": int((delta == 0).sum()),
            }
        )
    comparison = pd.DataFrame(rows)
    training = training_summary(experiment, manifest)
    write_csv(reports / "recurrent_comparison_pairs.csv", comparison)
    write_csv(reports / "recurrent_training_summary.csv", training)

    primary = comparison.loc[
        comparison["comparison"].eq("rain_effect_within_gru")
    ].iloc[0]
    architecture = comparison.loc[
        comparison["comparison"].eq("lstm_vs_gru_with_rain")
    ].iloc[0]
    report = "\n".join(
        [
            "# Matched GRU/LSTM recurrent comparison",
            "",
            "All four models use the same 910-site cohort, disjoint site splits, "
            "48-hour causal windows, target keys, FiLM metadata, target history, "
            "training budget and validation-only checkpoint selection.",
            "",
            "A negative paired RMSE delta favours the challenger. Confidence intervals "
            "are 95% site-block bootstrap intervals over the 135 held-out test sites.",
            "",
            "## Test metrics",
            "",
            markdown_table(
                test_overall.reset_index()[
                    [
                        "run_id",
                        "recurrent_type",
                        "rmse",
                        "mae",
                        "nse",
                        "macro_site_rmse",
                        "macro_site_nse",
                    ]
                ].sort_values("rmse")
            ),
            "",
            "## Paired site comparisons",
            "",
            markdown_table(comparison),
            "",
            "## Training summary",
            "",
            markdown_table(training),
            "",
            "## Primary interpretation",
            "",
            f"Within GRU, adding dynamic rainfall changed overall test RMSE by "
            f"{float(primary['overall_rmse_change_fraction']) * 100:+.2f}% and improved "
            f"{int(primary['challenger_improved_sites'])}/135 sites. The site-block "
            f"mean RMSE-delta interval was "
            f"[{float(primary['mean_site_rmse_delta_ci_low']):.6f}, "
            f"{float(primary['mean_site_rmse_delta_ci_high']):.6f}].",
            "",
            f"With rainfall included, switching from GRU to LSTM changed overall test "
            f"RMSE by {float(architecture['overall_rmse_change_fraction']) * 100:+.2f}%.",
            "",
            "These are one-seed architecture results. They establish the matched "
            "comparison but do not yet quantify seed-to-seed optimisation variance.",
        ]
    ) + "\n"
    (reports / "RECURRENT_COMPARISON_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    audit = {
        "status": "pass" if not problems else "fail",
        "roles": {f"{key[0]}_{key[1]}": value for key, value in roles.items()},
        "test_sites_per_pair": comparison.set_index("comparison")[
            "test_sites"
        ].to_dict(),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": int(args.seed),
        "validation_only_checkpoint_selection": True,
        "problems": problems,
        "manifest_sha256": hashlib.sha256(
            (reports / "run_manifest.csv").read_bytes()
        ).hexdigest(),
    }
    write_json(reports / "recurrent_comparison_audit.json", audit)
    if problems:
        raise RuntimeError(problems)
    print(report)


if __name__ == "__main__":
    main()
