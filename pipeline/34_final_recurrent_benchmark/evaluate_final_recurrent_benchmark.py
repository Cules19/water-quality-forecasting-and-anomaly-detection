"""Compare matched frozen-setting GRU-FiLM and LSTM-FiLM experiments."""

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
    parser.add_argument("--lstm-root", required=True)
    parser.add_argument("--gru-root", required=True)
    parser.add_argument("--output-root", required=True)
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
    for column in display:
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
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def read_experiment(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reports = root / "full" / "reports"
    overall = pd.read_csv(reports / "model_metrics_all.csv")
    sites = pd.read_csv(
        reports / "site_metrics_all.csv",
        dtype={"site_id": "string"},
    )
    manifest = pd.read_csv(reports / "run_manifest.csv")
    return overall, sites, manifest


def main() -> None:
    args = parse_args()
    lstm_root = resolve(args.lstm_root)
    gru_root = resolve(args.gru_root)
    output = resolve(args.output_root)
    reports = output / "reports"

    lstm_overall, lstm_sites, lstm_manifest = read_experiment(lstm_root)
    gru_overall, gru_sites, gru_manifest = read_experiment(gru_root)
    overall = pd.concat([lstm_overall, gru_overall], ignore_index=True)
    sites = pd.concat([lstm_sites, gru_sites], ignore_index=True)
    manifest = pd.concat([lstm_manifest, gru_manifest], ignore_index=True)

    roles: dict[tuple[str, str], str] = {}
    run_roots: dict[str, Path] = {}
    for root, frame in ((lstm_root, lstm_manifest), (gru_root, gru_manifest)):
        for row in frame.itertuples(index=False):
            role = "rain" if "rain_00045" in str(row.feature_set) else "base"
            key = (str(row.recurrent_type), role)
            if key in roles:
                raise RuntimeError(f"duplicate role: {key}")
            roles[key] = str(row.run_id)
            run_roots[str(row.run_id)] = root
    expected = {
        ("gru", "base"),
        ("gru", "rain"),
        ("lstm", "base"),
        ("lstm", "rain"),
    }
    if set(roles) != expected:
        raise RuntimeError(f"unexpected roles: {set(roles)}")

    problems: list[str] = []
    common_manifests = [
        root / "full" / "reports" / "common_key_manifest.csv"
        for root in (lstm_root, gru_root)
    ]
    common_hashes = [
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in common_manifests
    ]
    if len(set(common_hashes)) != 1:
        problems.append("GRU and LSTM common-key manifests differ")

    comparisons = [
        ("rain_effect_within_gru", roles[("gru", "base")], roles[("gru", "rain")]),
        (
            "rain_effect_within_lstm",
            roles[("lstm", "base")],
            roles[("lstm", "rain")],
        ),
        (
            "gru_vs_lstm_without_rain",
            roles[("lstm", "base")],
            roles[("gru", "base")],
        ),
        (
            "gru_vs_lstm_with_rain",
            roles[("lstm", "rain")],
            roles[("gru", "rain")],
        ),
    ]
    test_overall = overall[overall["split"].eq("test")].set_index("run_id")
    test_sites = sites[sites["split"].eq("test")].copy()
    comparison_rows = []
    for number, (name, reference, challenger) in enumerate(comparisons):
        reference_sites = test_sites.loc[
            test_sites["run_id"].eq(reference),
            ["site_id", "rmse"],
        ]
        challenger_sites = test_sites.loc[
            test_sites["run_id"].eq(challenger),
            ["site_id", "rmse"],
        ]
        paired = reference_sites.merge(
            challenger_sites,
            on="site_id",
            suffixes=("_reference", "_challenger"),
            validate="one_to_one",
        )
        if len(paired) != 135:
            problems.append(f"{name}: paired sites={len(paired)}, expected=135")
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
        comparison_rows.append(
            {
                "comparison": name,
                "reference_run": reference,
                "challenger_run": challenger,
                "negative_delta_favors_challenger": True,
                "test_sites": len(paired),
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
    comparison = pd.DataFrame(comparison_rows)

    site_rmse = test_sites.pivot(
        index="site_id",
        columns="run_id",
        values="rmse",
    ).dropna()
    gru_rain_effect = (
        site_rmse[roles[("gru", "rain")]]
        - site_rmse[roles[("gru", "base")]]
    )
    lstm_rain_effect = (
        site_rmse[roles[("lstm", "rain")]]
        - site_rmse[roles[("lstm", "base")]]
    )
    interaction_values = (
        gru_rain_effect - lstm_rain_effect
    ).to_numpy(dtype=np.float64)
    interaction_ci = bootstrap_mean_ci(
        interaction_values,
        int(args.bootstrap_replicates),
        int(args.seed) + 100,
    )
    interaction = pd.DataFrame(
        [
            {
                "contrast": "(GRU_RAIN-GRU_BASE)-(LSTM_RAIN-LSTM_BASE)",
                "positive_means_rain_is_less_favorable_for_gru": True,
                "paired_sites": len(site_rmse),
                "mean_rmse_difference_in_differences": float(
                    interaction_values.mean()
                ),
                "median_rmse_difference_in_differences": float(
                    np.median(interaction_values)
                ),
                "bootstrap_95_ci_low": interaction_ci[0],
                "bootstrap_95_ci_high": interaction_ci[1],
                "sites_where_rain_effect_is_more_favorable_for_lstm": int(
                    (interaction_values > 0).sum()
                ),
                "sites_where_rain_effect_is_more_favorable_for_gru": int(
                    (interaction_values < 0).sum()
                ),
            }
        ]
    )

    training_rows = []
    for row in manifest.itertuples(index=False):
        success = json.loads(
            (
                run_roots[str(row.run_id)]
                / "full"
                / "models"
                / f"{row.run_id}_SUCCESS.json"
            ).read_text(encoding="utf-8")
        )
        training_rows.append(
            {
                "run_id": row.run_id,
                "recurrent_type": row.recurrent_type,
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
    test_table = test_overall.reset_index()[
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

    write_csv(reports / "final_recurrent_test_metrics.csv", test_table)
    write_csv(reports / "final_recurrent_paired_comparisons.csv", comparison)
    write_csv(reports / "final_recurrent_architecture_rain_interaction.csv", interaction)
    write_csv(reports / "final_recurrent_training_summary.csv", training)

    report = "\n".join(
        [
            "# Matched frozen-setting GRU/LSTM benchmark",
            "",
            "GRU and LSTM use identical data, site splits, target keys, 48-hour "
            "windows, hidden size, learning rate, dropout and training budget. "
            "The GRU experiment was requested after inspecting the frozen LSTM "
            "test results, so architecture comparisons are explicitly post-hoc.",
            "",
            "## Held-out test metrics",
            "",
            markdown_table(test_table),
            "",
            "## Paired site comparisons",
            "",
            markdown_table(comparison),
            "",
            "## Architecture by rainfall interaction",
            "",
            markdown_table(interaction),
            "",
            "## Training summary",
            "",
            markdown_table(training),
            "",
        ]
    )
    (reports / "FINAL_RECURRENT_BENCHMARK_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    audit = {
        "status": "pass" if not problems else "fail",
        "post_hoc_architecture_benchmark": True,
        "common_key_manifest_identical": len(set(common_hashes)) == 1,
        "common_key_manifest_sha256": common_hashes[0],
        "roles": {f"{key[0]}_{key[1]}": value for key, value in roles.items()},
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": int(args.seed),
        "architecture_rain_interaction_ci": [
            interaction_ci[0],
            interaction_ci[1],
        ],
        "problems": problems,
    }
    write_json(reports / "final_recurrent_benchmark_audit.json", audit)
    if problems:
        raise RuntimeError(problems)
    print(report)


if __name__ == "__main__":
    main()
