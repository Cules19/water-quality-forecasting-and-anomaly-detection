"""Interpret and freeze the validation-only warm-up adaptation protocol."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import nct, t


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = Path(__file__).resolve().parent / "run_warmup_adaptation.py"
SPEC = importlib.util.spec_from_file_location("warmup_runner_eval", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            PROJECT_ROOT
            / "config"
            / "common1000_warmup_adaptation_validation_v1.json"
        ),
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def paired_frame(
    site_metrics: pd.DataFrame, challenger: str, reference: str
) -> pd.DataFrame:
    keys = ["site_id", "warmup_order", "warmup_label"]
    left = site_metrics.loc[
        site_metrics["strategy"].eq(challenger), keys + ["rmse"]
    ].rename(columns={"rmse": "challenger_rmse"})
    right = site_metrics.loc[
        site_metrics["strategy"].eq(reference), keys + ["rmse"]
    ].rename(columns={"rmse": "reference_rmse"})
    paired = left.merge(right, on=keys, validate="one_to_one")
    paired["delta_rmse"] = (
        paired["challenger_rmse"] - paired["reference_rmse"]
    )
    return paired


def approximate_paired_power(
    delta: np.ndarray, future_sites: int, alpha: float = 0.05
) -> dict[str, float]:
    delta = np.asarray(delta, dtype=np.float64)
    standard_deviation = float(delta.std(ddof=1))
    effect_size = (
        float(delta.mean() / standard_deviation)
        if standard_deviation > 0
        else float("nan")
    )
    degrees_freedom = int(future_sites) - 1
    critical = float(t.ppf(1.0 - alpha / 2.0, degrees_freedom))
    noncentrality = effect_size * math.sqrt(future_sites)
    power = float(
        nct.cdf(-critical, degrees_freedom, noncentrality)
        + 1.0
        - nct.cdf(critical, degrees_freedom, noncentrality)
    )
    return {
        "validation_mean_delta_rmse": float(delta.mean()),
        "validation_site_delta_sd": standard_deviation,
        "paired_standardized_effect": effect_size,
        "prospective_test_sites": int(future_sites),
        "approximate_two_sided_power_alpha_0_05": power,
    }


def make_curve_figure(
    curve: pd.DataFrame,
    path: Path,
    title: str = "Validation-only new-site warm-up curve",
    ylabel: str = "Macro site RMSE",
) -> None:
    labels = (
        curve[["warmup_order", "warmup_label"]]
        .drop_duplicates()
        .sort_values("warmup_order")
    )
    positions = np.arange(len(labels))
    palette = {
        "zero_shot": "#505050",
        "film_affine": "#1676B8",
        "full_finetune": "#D36B27",
    }
    display = {
        "zero_shot": "Zero-shot",
        "film_affine": "FiLM + affine (130 parameters)",
        "full_finetune": "Full fine-tuning",
    }
    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    for strategy in ("zero_shot", "film_affine", "full_finetune"):
        subset = curve.loc[curve["strategy"].eq(strategy)].sort_values(
            "warmup_order"
        )
        axis.plot(
            positions,
            subset["macro_site_rmse"],
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=palette[strategy],
            label=display[strategy],
        )
    axis.set_xticks(positions, labels["warmup_label"].str.replace("_", " "))
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Available local warm-up data")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def make_delta_figure(
    paired: pd.DataFrame,
    path: Path,
    title: str = "Paired site effect with 95% site-bootstrap intervals",
) -> None:
    labels = (
        paired[["warmup_order", "warmup_label"]]
        .drop_duplicates()
        .sort_values("warmup_order")
    )
    positions = np.arange(len(labels), dtype=np.float64)
    palette = {
        "film_affine": "#1676B8",
        "full_finetune": "#D36B27",
    }
    display = {
        "film_affine": "FiLM + affine",
        "full_finetune": "Full fine-tuning",
    }
    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    for strategy, shift in (("film_affine", -0.08), ("full_finetune", 0.08)):
        subset = paired.loc[paired["strategy"].eq(strategy)].sort_values(
            "warmup_order"
        )
        mean = subset["mean_site_delta_rmse_adapted_minus_zero"].to_numpy()
        lower = subset["bootstrap_95_ci_lower"].to_numpy()
        upper = subset["bootstrap_95_ci_upper"].to_numpy()
        axis.errorbar(
            positions + shift,
            mean,
            yerr=np.vstack([mean - lower, upper - mean]),
            fmt="o-",
            capsize=3,
            linewidth=1.8,
            color=palette[strategy],
            label=display[strategy],
        )
    axis.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    axis.set_xticks(positions, labels["warmup_label"].str.replace("_", " "))
    axis.set_ylabel("Site RMSE delta vs zero-shot (negative is better)")
    axis.set_xlabel("Available local warm-up data")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_json(config_path)
    root = resolve(config["output_root"]) / args.mode
    reports = root / "reports"
    figures = root / "figures"
    site_metrics = pd.read_csv(reports / "warmup_site_metrics.csv")
    curve = pd.read_csv(reports / "warmup_curve_summary.csv")
    paired = pd.read_csv(reports / "warmup_paired_site_summary.csv")
    adaptation_audit = load_json(reports / "warmup_adaptation_audit.json")
    if adaptation_audit["status"] != "pass":
        raise RuntimeError("The source warm-up adaptation audit did not pass")

    strict_cohort = pd.read_csv(
        resolve(config["strict_cohort_path"]),
        dtype={"site_id": "string", "split": "string"},
    )
    prospective_test_sites = int(
        (
            strict_cohort["split"].eq("test")
            & strict_cohort["full_fixed_curve_eligible"].eq(True)
        ).sum()
    )
    validation_sites = int(adaptation_audit["sites"])
    target_display = str(config.get("target_display_name", "target"))
    power_rows = []
    for strategy in ("film_affine", "full_finetune"):
        comparison = paired_frame(site_metrics, strategy, "zero_shot")
        for (order, label), group in comparison.groupby(
            ["warmup_order", "warmup_label"], sort=False
        ):
            power_rows.append(
                {
                    "warmup_order": order,
                    "warmup_label": label,
                    "strategy": strategy,
                    **approximate_paired_power(
                        group["delta_rmse"].to_numpy(),
                        prospective_test_sites,
                    ),
                }
            )
    power = pd.DataFrame(power_rows).sort_values(
        ["warmup_order", "strategy"]
    )

    film_vs_full = paired_frame(
        site_metrics, "film_affine", "full_finetune"
    )
    comparison_rows = []
    for (order, label), group in film_vs_full.groupby(
        ["warmup_order", "warmup_label"], sort=False
    ):
        delta = group["delta_rmse"].to_numpy(dtype=np.float64)
        lower, upper = RUNNER.bootstrap_mean_ci(
            delta,
            int(config["bootstrap_replicates"]),
            RUNNER.stable_seed(config["random_seed"], "film_vs_full", label),
        )
        comparison_rows.append(
            {
                "warmup_order": order,
                "warmup_label": label,
                "mean_site_delta_rmse_film_minus_full": float(delta.mean()),
                "bootstrap_95_ci_lower": lower,
                "bootstrap_95_ci_upper": upper,
                "sites_film_better": int((delta < 0).sum()),
                "sites_full_better": int((delta > 0).sum()),
            }
        )
    film_full = pd.DataFrame(comparison_rows)

    make_curve_figure(
        curve,
        figures / "warmup_validation_curve.png",
        f"Validation-only {target_display} warm-up curve "
        f"({validation_sites} fixed sites)",
        f"Macro site RMSE ({target_display})",
    )
    make_delta_figure(paired, figures / "warmup_validation_paired_delta.png")

    twelve_curve = curve.loc[
        curve["warmup_label"].eq("12_months")
    ].set_index("strategy")
    twelve_paired = paired.loc[
        paired["warmup_label"].eq("12_months")
        & paired["strategy"].eq("film_affine")
    ].iloc[0]
    twelve_power = power.loc[
        power["warmup_label"].eq("12_months")
        & power["strategy"].eq("film_affine")
    ].iloc[0]
    twelve_film_full = film_full.loc[
        film_full["warmup_label"].eq("12_months")
    ].iloc[0]
    robust_labels = paired.loc[
        paired["strategy"].eq("film_affine")
        & paired["bootstrap_95_ci_upper"].lt(0),
        "warmup_label",
    ].tolist()
    pooled_improvement = 100.0 * (
        twelve_curve.loc["zero_shot", "pooled_rmse"]
        - twelve_curve.loc["film_affine", "pooled_rmse"]
    ) / twelve_curve.loc["zero_shot", "pooled_rmse"]
    macro_improvement = 100.0 * (
        twelve_curve.loc["zero_shot", "macro_site_rmse"]
        - twelve_curve.loc["film_affine", "macro_site_rmse"]
    ) / twelve_curve.loc["zero_shot", "macro_site_rmse"]
    mae_improvement = 100.0 * (
        twelve_curve.loc["zero_shot", "macro_site_mae"]
        - twelve_curve.loc["film_affine", "macro_site_mae"]
    ) / twelve_curve.loc["zero_shot", "macro_site_mae"]

    frozen_protocol = {
        "status": "frozen_on_validation",
        "primary_strategy": "film_affine",
        "primary_strategy_reason": (
            "The supervisor-prespecified low-parameter site adaptation is the "
            "primary deployment strategy; its validation effect is reported "
            "without requiring a favourable result."
        ),
        "base_checkpoint": str(resolve(config["base_checkpoint"])),
        "base_checkpoint_sha256": adaptation_audit["base_checkpoint_sha256"],
        "warmup_labels": config["full"]["warmup_labels"],
        "fixed_evaluation_after_months": 12,
        "local_validation_fraction": config["local_validation_fraction"],
        "minimum_local_validation_targets": config[
            "minimum_local_validation_targets"
        ],
        "maximum_local_validation_targets": config[
            "maximum_local_validation_targets"
        ],
        "maximum_epochs": config["full"]["maximum_epochs"],
        "minimum_epochs": config["full"]["minimum_epochs"],
        "early_stopping_patience": config["full"][
            "early_stopping_patience"
        ],
        "film_affine_learning_rate": config["full"][
            "film_affine_learning_rate"
        ],
        "full_finetune_learning_rate": config["full"][
            "full_finetune_learning_rate"
        ],
        "weight_decay": config["full"]["weight_decay"],
        "gradient_clip_norm": config["full"]["gradient_clip_norm"],
        "test_site_count": prospective_test_sites,
        "test_status": "not_run",
        "no_further_adaptation_tuning_after_test": True,
    }

    audit = {
        "status": "pass",
        "source_site_rows": len(site_metrics),
        "expected_source_site_rows": (
            validation_sites
            * len(config["full"]["warmup_labels"])
            * len(config["strategies"])
        ),
        "zero_shot_rmse_constant_within_every_site": bool(
            (
                site_metrics.loc[
                    site_metrics["strategy"].eq("zero_shot")
                ]
                .groupby("site_id")["rmse"]
                .nunique()
                .eq(1)
            ).all()
        ),
        "all_strategies_use_identical_evaluated_target_count": bool(
            (
                site_metrics.groupby(
                    ["site_id", "warmup_label"]
                )["fixed_evaluation_targets"]
                .nunique()
                .eq(1)
            ).all()
        ),
        "film_affine_trainable_parameters": int(
            site_metrics.loc[
                site_metrics["strategy"].eq("film_affine"),
                "trainable_parameters",
            ].max()
        ),
        "full_finetune_trainable_parameters": int(
            site_metrics.loc[
                site_metrics["strategy"].eq("full_finetune"),
                "trainable_parameters",
            ].max()
        ),
        "test_metrics_read": False,
        "prospective_test_sites": prospective_test_sites,
    }
    if (
        len(site_metrics) != audit["expected_source_site_rows"]
        or not audit["zero_shot_rmse_constant_within_every_site"]
        or not audit["all_strategies_use_identical_evaluated_target_count"]
        or audit["film_affine_trainable_parameters"] != 130
        or audit["full_finetune_trainable_parameters"]
        <= audit["film_affine_trainable_parameters"]
    ):
        audit["status"] = "fail"

    RUNNER.write_csv(reports / "warmup_prospective_test_power.csv", power)
    RUNNER.write_csv(reports / "warmup_film_vs_full_paired.csv", film_full)
    RUNNER.write_json(reports / "frozen_adaptation_protocol.json", frozen_protocol)
    RUNNER.write_json(reports / "warmup_validation_interpretation_audit.json", audit)

    report = "\n".join(
        [
            "# Warm-up validation interpretation and frozen protocol",
            "",
            "## Main validation finding",
            "",
            f"At 12 months, the 130-parameter FiLM + affine adapter reduced "
            f"pooled RMSE by **{pooled_improvement:.2f}%**, macro site RMSE by "
            f"**{macro_improvement:.2f}%**, and macro site MAE by "
            f"**{mae_improvement:.2f}%** relative to zero-shot.",
            "",
            f"The site-level mean RMSE change was "
            f"**{twelve_paired['mean_site_delta_rmse_adapted_minus_zero']:.6f}** "
            f"with a 95% site-bootstrap interval of "
            f"[{twelve_paired['bootstrap_95_ci_lower']:.6f}, "
            f"{twelve_paired['bootstrap_95_ci_upper']:.6f}]. "
            f"Adaptation improved {int(twelve_paired['sites_improved'])}/"
            f"{validation_sites} sites.",
            "",
            f"At 12 months, FiLM + affine was better than full fine-tuning at "
            f"{int(twelve_film_full['sites_film_better'])}/"
            f"{validation_sites} sites and trained "
            f"{audit['film_affine_trainable_parameters']:,} rather than "
            f"{audit['full_finetune_trainable_parameters']:,} parameters. "
            "FiLM + affine is frozen as the deployment-oriented primary "
            "strategy; full fine-tuning remains a comparator.",
            "",
            "## Duration interpretation",
            "",
            (
                "FiLM + affine had a site-bootstrap interval excluding zero at: "
                f"**{', '.join(robust_labels)}**."
                if robust_labels
                else
                "No FiLM + affine duration had a site-bootstrap interval "
                "excluding zero; the method remains a deployment comparator, "
                "not a claimed improvement."
            ),
            "",
            "## Prospective test precision",
            "",
            f"Using the validation-site effect distribution only, a conventional "
            f"paired-test approximation gives about "
            f"**{100.0 * twelve_power['approximate_two_sided_power_alpha_0_05']:.1f}%** "
            f"power for the {prospective_test_sites}-site test cohort at the "
            "12-month low-cost effect. Duration-specific estimates are retained "
            "in the power table and are planning diagnostics, not guarantees.",
            "",
            "## Frozen decision",
            "",
            "- Primary adaptation: site-specific FiLM offsets + positive output affine.",
            "- Primary comparison: fixed post-12-month target keys.",
            "- Zero-shot and full fine-tuning remain prespecified comparators.",
            "- Learning rates, epoch selection, regularisation and all data cutoffs "
            "are frozen before test evaluation.",
            f"- The {prospective_test_sites} test-site files and labels have not "
            "been read by this stage.",
            "",
        ]
    )
    (reports / "WARMUP_VALIDATION_INTERPRETATION.md").write_text(
        report, encoding="utf-8"
    )
    if audit["status"] != "pass":
        raise RuntimeError("Warm-up validation interpretation audit failed")
    print(
        f"Validation protocol frozen: 12-month FiLM improvement "
        f"{pooled_improvement:.2f}%; prospective test sites="
        f"{prospective_test_sites}"
    )


if __name__ == "__main__":
    main()
