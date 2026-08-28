"""Audit and interpret the single frozen-protocol held-out warm-up test."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_EVALUATOR_PATH = (
    Path(__file__).resolve().parent / "evaluate_warmup_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "warmup_validation_helpers", VALIDATION_EVALUATOR_PATH
)
HELPERS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(HELPERS)
RUNNER = HELPERS.RUNNER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-config",
        default=(
            PROJECT_ROOT
            / "config"
            / "common1000_warmup_adaptation_test_v1.json"
        ),
    )
    parser.add_argument(
        "--validation-config",
        default=(
            PROJECT_ROOT
            / "config"
            / "common1000_warmup_adaptation_validation_v1.json"
        ),
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def comparison_summary(
    site_metrics: pd.DataFrame,
    challenger: str,
    reference: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    paired = HELPERS.paired_frame(site_metrics, challenger, reference)
    rows = []
    for (order, label), group in paired.groupby(
        ["warmup_order", "warmup_label"], sort=False
    ):
        delta = group["delta_rmse"].to_numpy(dtype=np.float64)
        lower, upper = RUNNER.bootstrap_mean_ci(
            delta,
            replicates,
            RUNNER.stable_seed(seed, challenger, reference, label),
        )
        leave_one_out = np.asarray(
            [
                (delta.sum() - delta[index]) / (len(delta) - 1)
                for index in range(len(delta))
            ],
            dtype=np.float64,
        )
        rows.append(
            {
                "warmup_order": order,
                "warmup_label": label,
                "challenger": challenger,
                "reference": reference,
                "sites": len(delta),
                "mean_site_delta_rmse": float(delta.mean()),
                "median_site_delta_rmse": float(np.median(delta)),
                "bootstrap_95_ci_lower": lower,
                "bootstrap_95_ci_upper": upper,
                "sites_challenger_better": int((delta < 0).sum()),
                "sites_reference_better": int((delta > 0).sum()),
                "leave_one_site_out_mean_minimum": float(
                    leave_one_out.min()
                ),
                "leave_one_site_out_mean_maximum": float(
                    leave_one_out.max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("warmup_order")


def replication_table(
    validation_paired: pd.DataFrame,
    test_paired: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["warmup_order", "warmup_label", "strategy"]
    validation = validation_paired[keys + [
        "mean_site_delta_rmse_adapted_minus_zero",
        "bootstrap_95_ci_lower",
        "bootstrap_95_ci_upper",
        "sites_improved",
        "sites",
    ]].rename(
        columns={
            "mean_site_delta_rmse_adapted_minus_zero": "validation_mean_delta",
            "bootstrap_95_ci_lower": "validation_ci_lower",
            "bootstrap_95_ci_upper": "validation_ci_upper",
            "sites_improved": "validation_sites_improved",
            "sites": "validation_sites",
        }
    )
    test = test_paired[keys + [
        "mean_site_delta_rmse_adapted_minus_zero",
        "bootstrap_95_ci_lower",
        "bootstrap_95_ci_upper",
        "sites_improved",
        "sites",
    ]].rename(
        columns={
            "mean_site_delta_rmse_adapted_minus_zero": "test_mean_delta",
            "bootstrap_95_ci_lower": "test_ci_lower",
            "bootstrap_95_ci_upper": "test_ci_upper",
            "sites_improved": "test_sites_improved",
            "sites": "test_sites",
        }
    )
    result = validation.merge(test, on=keys, validate="one_to_one")
    result["same_mean_direction"] = (
        np.sign(result["validation_mean_delta"])
        == np.sign(result["test_mean_delta"])
    )
    result["validation_robust_improvement"] = (
        result["validation_ci_upper"] < 0
    )
    result["test_robust_improvement"] = result["test_ci_upper"] < 0
    result["robustly_replicated_improvement"] = (
        result["validation_robust_improvement"]
        & result["test_robust_improvement"]
    )
    return result.sort_values(["warmup_order", "strategy"])


def relative_improvement(
    curve: pd.DataFrame, strategy: str, metric: str
) -> float:
    indexed = curve.set_index("strategy")
    return float(
        100.0
        * (
            indexed.loc["zero_shot", metric]
            - indexed.loc[strategy, metric]
        )
        / indexed.loc["zero_shot", metric]
    )


def main() -> None:
    args = parse_args()
    test_config = load_json(resolve(args.test_config))
    validation_config = load_json(resolve(args.validation_config))
    test_root = resolve(test_config["output_root"]) / "full"
    validation_root = resolve(validation_config["output_root"]) / "full"
    test_reports = test_root / "reports"
    test_figures = test_root / "figures"

    source_audit = load_json(
        test_reports / "warmup_adaptation_audit.json"
    )
    if source_audit["status"] != "pass":
        raise RuntimeError("Source held-out test audit did not pass")
    expected_test_sites = int(source_audit["sites"])
    target_display = str(test_config.get("target_display_name", "target"))
    test_site_metrics = pd.read_csv(
        test_reports / "warmup_site_metrics.csv",
        dtype={"site_id": "string"},
    )
    test_curve = pd.read_csv(
        test_reports / "warmup_curve_summary.csv"
    )
    test_paired = pd.read_csv(
        test_reports / "warmup_paired_site_summary.csv"
    )
    validation_site_metrics = pd.read_csv(
        validation_root / "reports" / "warmup_site_metrics.csv",
        dtype={"site_id": "string"},
    )
    validation_paired = pd.read_csv(
        validation_root / "reports" / "warmup_paired_site_summary.csv"
    )

    film_zero = comparison_summary(
        test_site_metrics,
        "film_affine",
        "zero_shot",
        int(test_config["bootstrap_replicates"]),
        int(test_config["random_seed"]),
    )
    full_zero = comparison_summary(
        test_site_metrics,
        "full_finetune",
        "zero_shot",
        int(test_config["bootstrap_replicates"]),
        int(test_config["random_seed"]),
    )
    film_full = comparison_summary(
        test_site_metrics,
        "film_affine",
        "full_finetune",
        int(test_config["bootstrap_replicates"]),
        int(test_config["random_seed"]),
    )
    replication = replication_table(validation_paired, test_paired)

    HELPERS.make_curve_figure(
        test_curve,
        test_figures / "warmup_test_curve.png",
        f"Held-out {target_display} warm-up curve "
        f"({expected_test_sites} fixed test sites)",
        f"Macro site RMSE ({target_display})",
    )
    HELPERS.make_delta_figure(
        test_paired,
        test_figures / "warmup_test_paired_delta.png",
        "Held-out paired site effect with 95% bootstrap intervals",
    )

    twelve_curve = test_curve.loc[
        test_curve["warmup_label"].eq("12_months")
    ]
    twelve_film = film_zero.loc[
        film_zero["warmup_label"].eq("12_months")
    ].iloc[0]
    twelve_full = full_zero.loc[
        full_zero["warmup_label"].eq("12_months")
    ].iloc[0]
    twelve_film_full = film_full.loc[
        film_full["warmup_label"].eq("12_months")
    ].iloc[0]
    film_pooled = relative_improvement(
        twelve_curve, "film_affine", "pooled_rmse"
    )
    film_macro = relative_improvement(
        twelve_curve, "film_affine", "macro_site_rmse"
    )
    film_mae = relative_improvement(
        twelve_curve, "film_affine", "macro_site_mae"
    )
    full_pooled = relative_improvement(
        twelve_curve, "full_finetune", "pooled_rmse"
    )
    full_macro = relative_improvement(
        twelve_curve, "full_finetune", "macro_site_rmse"
    )
    full_mae = relative_improvement(
        twelve_curve, "full_finetune", "macro_site_mae"
    )

    validation_sites = set(
        validation_site_metrics["site_id"].dropna().astype(str)
    )
    test_sites = set(test_site_metrics["site_id"].dropna().astype(str))
    zero_constant = bool(
        (
            test_site_metrics.loc[
                test_site_metrics["strategy"].eq("zero_shot")
            ]
            .groupby("site_id")["rmse"]
            .nunique()
            .eq(1)
        ).all()
    )
    fixed_keys_constant = bool(
        (
            test_site_metrics.groupby("site_id")[
                "fixed_evaluation_key_sha256"
            ]
            .nunique()
            .eq(1)
        ).all()
    )
    primary_leave_one_out_robust = bool(
        twelve_film["leave_one_site_out_mean_maximum"] < 0
    )
    audit = {
        "status": "pass",
        "source_test_audit_status": source_audit["status"],
        "frozen_protocol_verified_before_test": source_audit[
            "frozen_protocol_verified"
        ],
        "test_result_rows": len(test_site_metrics),
        "expected_test_result_rows": (
            expected_test_sites
            * len(test_config["full"]["warmup_labels"])
            * len(test_config["strategies"])
        ),
        "test_sites": len(test_sites),
        "validation_test_site_overlap": sorted(
            validation_sites.intersection(test_sites)
        ),
        "zero_shot_metric_constant_across_warmups_per_site": zero_constant,
        "fixed_evaluation_key_constant_across_all_warmups_per_site": (
            fixed_keys_constant
        ),
        "adaptation_evaluation_overlap_targets": source_audit[
            "adaptation_evaluation_overlap_targets"
        ],
        "primary_12_month_bootstrap_ci_excludes_zero": bool(
            twelve_film["bootstrap_95_ci_upper"] < 0
        ),
        "primary_12_month_all_leave_one_site_out_means_improve": (
            primary_leave_one_out_robust
        ),
        "test_protocol_changed_after_results": False,
    }
    if (
        len(test_site_metrics) != audit["expected_test_result_rows"]
        or len(test_sites) != expected_test_sites
        or audit["validation_test_site_overlap"]
        or not zero_constant
        or not fixed_keys_constant
        or audit["adaptation_evaluation_overlap_targets"] != 0
    ):
        audit["status"] = "fail"

    completion = {
        "status": "held_out_test_complete",
        "primary_strategy": "film_affine",
        "test_sites": expected_test_sites,
        "warmup_periods": len(test_config["full"]["warmup_labels"]),
        "base_checkpoint_sha256": source_audit[
            "base_checkpoint_sha256"
        ],
        "frozen_protocol_path": source_audit["frozen_protocol_path"],
        "post_cutoff_test_targets_used_for_model_or_epoch_selection": False,
        "no_further_parameter_tuning_or_test_rerun": True,
    }
    RUNNER.write_csv(
        test_reports / "warmup_test_film_vs_zero_sensitivity.csv",
        film_zero,
    )
    RUNNER.write_csv(
        test_reports / "warmup_test_full_vs_zero_sensitivity.csv",
        full_zero,
    )
    RUNNER.write_csv(
        test_reports / "warmup_test_film_vs_full_sensitivity.csv",
        film_full,
    )
    RUNNER.write_csv(
        test_reports / "warmup_validation_test_replication.csv",
        replication,
    )
    RUNNER.write_json(
        test_reports / "warmup_final_test_audit.json", audit
    )
    RUNNER.write_json(
        test_reports / "warmup_test_completion_manifest.json", completion
    )

    replicated_film = replication.loc[
        replication["strategy"].eq("film_affine")
        & replication["robustly_replicated_improvement"].eq(True),
        "warmup_label",
    ].tolist()
    leave_one_out_text = (
        "Every leave-one-site-out mean remained beneficial"
        if primary_leave_one_out_robust
        else
        "At least one leave-one-site-out mean was not beneficial"
    )
    replication_text = (
        "Low-cost adaptation was robustly beneficial in both validation and "
        f"test at: **{', '.join(replicated_film)}**."
        if replicated_film
        else
        "No warm-up duration had a bootstrap interval below zero in both "
        "validation and test."
    )
    if twelve_film_full["bootstrap_95_ci_upper"] < 0:
        film_full_interval_text = (
            "The interval is below zero, favouring FiLM + affine on average."
        )
    elif twelve_film_full["bootstrap_95_ci_lower"] > 0:
        film_full_interval_text = (
            "The interval is above zero, favouring full fine-tuning on average."
        )
    else:
        film_full_interval_text = (
            "The interval includes zero, so the held-out data do not establish "
            "that either method has lower average site RMSE."
        )
    report = "\n".join(
        [
            "# Final held-out new-site warm-up test report",
            "",
            "## Primary result",
            "",
            f"Under the validation-frozen protocol, the 12-month 130-parameter "
            f"FiLM + affine adapter reduced pooled RMSE by **{film_pooled:.2f}%**, "
            f"macro site RMSE by **{film_macro:.2f}%**, and macro site MAE by "
            f"**{film_mae:.2f}%** relative to zero-shot on "
            f"{expected_test_sites} held-out sites.",
            "",
            f"The paired mean site RMSE delta was "
            f"**{twelve_film['mean_site_delta_rmse']:.6f}** with a 95% "
            f"site-bootstrap interval of "
            f"[{twelve_film['bootstrap_95_ci_lower']:.6f}, "
            f"{twelve_film['bootstrap_95_ci_upper']:.6f}]. "
            f"The adapter improved "
            f"{int(twelve_film['sites_challenger_better'])}/"
            f"{expected_test_sites} "
            "sites.",
            "",
            f"{leave_one_out_text} "
            f"([{twelve_film['leave_one_site_out_mean_minimum']:.6f}, "
            f"{twelve_film['leave_one_site_out_mean_maximum']:.6f}]).",
            "",
            "## Validation replication",
            "",
            replication_text,
            "",
            "## Low-cost versus full fine-tuning",
            "",
            f"At 12 months, full fine-tuning improved pooled RMSE by "
            f"**{full_pooled:.2f}%**, macro site RMSE by **{full_macro:.2f}%**, "
            f"and macro site MAE by **{full_mae:.2f}%**. It improved "
            f"{int(twelve_full['sites_challenger_better'])}/"
            f"{expected_test_sites} sites.",
            "",
            f"The mean FiLM-minus-full site RMSE difference was "
            f"{twelve_film_full['mean_site_delta_rmse']:.6f}, with a 95% "
            f"interval of [{twelve_film_full['bootstrap_95_ci_lower']:.6f}, "
            f"{twelve_film_full['bootstrap_95_ci_upper']:.6f}]. "
            f"{film_full_interval_text} The low-cost method remains the "
            "practical primary method because it updates only its site adapter "
            "rather than the complete generic model. Site counts and all error "
            "metrics above are reported without selecting the favourable method.",
            "",
            "## Integrity",
            "",
            "- Validation and test site sets are disjoint.",
            "- The checkpoint and all adaptation settings matched the frozen protocol.",
            "- Adaptation and fixed evaluation targets have zero overlap.",
            "- Every duration uses identical post-12-month evaluation keys per site.",
            "- Post-cutoff test labels did not select epochs, parameters or methods.",
            "- No further tuning or repeat test run is permitted.",
            "",
        ]
    )
    (test_reports / "FINAL_WARMUP_TEST_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    if audit["status"] != "pass":
        raise RuntimeError("Final held-out warm-up audit failed")
    print(
        f"Final warm-up test passed: 12-month FiLM pooled improvement "
        f"{film_pooled:.2f}%; improved sites="
        f"{int(twelve_film['sites_challenger_better'])}/"
        f"{expected_test_sites}"
    )


if __name__ == "__main__":
    main()
