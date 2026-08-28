"""Audit data feasibility and leak-safe time keys for new-site warm-up curves."""

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
    parser.add_argument(
        "--config",
        default=PROJECT_ROOT / "config" / "common1000_warmup_feasibility_v1.json",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def date_offset(specification: dict[str, int]) -> pd.DateOffset:
    allowed = {"days", "months"}
    unknown = set(specification) - allowed
    if unknown:
        raise ValueError(f"Unsupported calendar offset fields: {sorted(unknown)}")
    if len(specification) != 1:
        raise ValueError("Each calendar offset must contain exactly one of days or months")
    key, value = next(iter(specification.items()))
    if int(value) <= 0:
        raise ValueError("Calendar offset must be positive")
    return pd.DateOffset(**{key: int(value)})


def key_digest(site_id: str, timestamps: pd.DatetimeIndex | np.ndarray) -> str:
    values = np.asarray(timestamps, dtype="datetime64[ns]")
    digest = hashlib.sha256()
    digest.update(str(site_id).encode("utf-8"))
    digest.update(values.view("int64").tobytes())
    return digest.hexdigest()


def interval_timestamps(
    timestamps: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> pd.DatetimeIndex:
    mask = timestamps >= start
    if end is not None:
        mask &= timestamps < end
    return timestamps[mask]


def coverage_fraction(count: int, start: pd.Timestamp, end: pd.Timestamp) -> float:
    possible_hours = (end - start).total_seconds() / 3600.0
    if possible_hours <= 0:
        return float("nan")
    return float(count / possible_hours)


def build_site_period_rows(
    site_id: str,
    split: str,
    timestamps: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    timestamps = pd.DatetimeIndex(timestamps).sort_values()
    if not len(timestamps):
        raise ValueError(f"{site_id}: no common target timestamps")
    if timestamps.has_duplicates:
        raise ValueError(f"{site_id}: duplicate common target timestamps")

    deployment_start = pd.Timestamp(timestamps[0])
    fixed_evaluation_start = deployment_start + date_offset(
        config["fixed_evaluation_after"]
    )
    fixed_evaluation = interval_timestamps(
        timestamps, fixed_evaluation_start, None
    )
    minimum_adaptation = int(config["minimum_adaptation_targets"])
    minimum_adaptation_coverage = float(
        config["minimum_adaptation_target_coverage"]
    )
    minimum_evaluation = int(config["minimum_evaluation_targets"])
    immediate_days = int(config["immediate_evaluation_days"])

    rows: list[dict[str, Any]] = []
    previous_cutoff: pd.Timestamp | None = None
    for order, period in enumerate(config["warmup_periods"], start=1):
        specification = {
            key: int(period[key]) for key in ("days", "months") if key in period
        }
        cutoff = deployment_start + date_offset(specification)
        if previous_cutoff is not None and cutoff <= previous_cutoff:
            raise ValueError("Warm-up periods must be strictly increasing")
        previous_cutoff = cutoff

        adaptation = interval_timestamps(timestamps, deployment_start, cutoff)
        immediate_end = cutoff + pd.DateOffset(days=immediate_days)
        immediate_evaluation = interval_timestamps(
            timestamps, cutoff, immediate_end
        )
        adaptation_coverage = coverage_fraction(
            len(adaptation), deployment_start, cutoff
        )
        adaptation_count_eligible = len(adaptation) >= minimum_adaptation
        adaptation_coverage_eligible = (
            adaptation_coverage >= minimum_adaptation_coverage
        )
        adaptation_eligible = (
            adaptation_count_eligible and adaptation_coverage_eligible
        )

        adaptation_evaluation_overlap = int(
            len(np.intersect1d(adaptation.values, fixed_evaluation.values))
        )
        adaptation_immediate_overlap = int(
            len(np.intersect1d(adaptation.values, immediate_evaluation.values))
        )
        maximum_adaptation_target = (
            pd.Timestamp(adaptation[-1]) if len(adaptation) else pd.NaT
        )
        minimum_fixed_evaluation_target = (
            pd.Timestamp(fixed_evaluation[0]) if len(fixed_evaluation) else pd.NaT
        )
        minimum_immediate_evaluation_target = (
            pd.Timestamp(immediate_evaluation[0])
            if len(immediate_evaluation)
            else pd.NaT
        )

        rows.append(
            {
                "site_id": site_id,
                "split": split,
                "warmup_order": order,
                "warmup_label": period["label"],
                "deployment_start": deployment_start,
                "warmup_cutoff_exclusive": cutoff,
                "warmup_calendar_hours": int(
                    round((cutoff - deployment_start).total_seconds() / 3600.0)
                ),
                "adaptation_targets": len(adaptation),
                "adaptation_target_coverage": adaptation_coverage,
                "adaptation_key_sha256": key_digest(site_id, adaptation),
                "maximum_adaptation_target": maximum_adaptation_target,
                "immediate_evaluation_start": cutoff,
                "immediate_evaluation_end_exclusive": immediate_end,
                "immediate_evaluation_targets": len(immediate_evaluation),
                "immediate_evaluation_key_sha256": key_digest(
                    site_id, immediate_evaluation
                ),
                "minimum_immediate_evaluation_target": (
                    minimum_immediate_evaluation_target
                ),
                "fixed_evaluation_start": fixed_evaluation_start,
                "fixed_evaluation_targets": len(fixed_evaluation),
                "fixed_evaluation_key_sha256": key_digest(
                    site_id, fixed_evaluation
                ),
                "minimum_fixed_evaluation_target": minimum_fixed_evaluation_target,
                "adaptation_count_eligible": adaptation_count_eligible,
                "adaptation_coverage_eligible": adaptation_coverage_eligible,
                "adaptation_eligible": adaptation_eligible,
                "immediate_curve_eligible": (
                    adaptation_eligible
                    and len(immediate_evaluation) >= minimum_evaluation
                ),
                "fixed_curve_eligible": (
                    adaptation_eligible
                    and len(fixed_evaluation) >= minimum_evaluation
                ),
                "adaptation_immediate_overlap_targets": (
                    adaptation_immediate_overlap
                ),
                "adaptation_fixed_overlap_targets": adaptation_evaluation_overlap,
            }
        )

    all_periods_fixed_eligible = all(
        bool(row["fixed_curve_eligible"]) for row in rows
    )
    site_summary = {
        "site_id": site_id,
        "split": split,
        "deployment_start": deployment_start,
        "last_common_target": pd.Timestamp(timestamps[-1]),
        "common_target_count": len(timestamps),
        "common_target_span_days": (
            pd.Timestamp(timestamps[-1]) - deployment_start
        ).total_seconds()
        / 86400.0,
        "common_key_sha256": key_digest(site_id, timestamps),
        "fixed_evaluation_start": fixed_evaluation_start,
        "fixed_evaluation_targets": len(fixed_evaluation),
        "fixed_evaluation_key_sha256": key_digest(site_id, fixed_evaluation),
        "full_fixed_curve_eligible": all_periods_fixed_eligible,
    }
    return rows, site_summary


def summarize_by_period(
    details: pd.DataFrame, site_summary: pd.DataFrame
) -> pd.DataFrame:
    full_counts = (
        site_summary.groupby("split", sort=False)["full_fixed_curve_eligible"]
        .sum()
        .astype(int)
        .to_dict()
    )
    rows = []
    for (split, order, label), group in details.groupby(
        ["split", "warmup_order", "warmup_label"], sort=False
    ):
        rows.append(
            {
                "split": split,
                "warmup_order": order,
                "warmup_label": label,
                "total_sites": group["site_id"].nunique(),
                "adaptation_eligible_sites": int(group["adaptation_eligible"].sum()),
                "immediate_curve_eligible_sites": int(
                    group["immediate_curve_eligible"].sum()
                ),
                "fixed_curve_eligible_sites": int(
                    group["fixed_curve_eligible"].sum()
                ),
                "full_fixed_curve_sites": int(full_counts[split]),
                "median_adaptation_targets": float(
                    group["adaptation_targets"].median()
                ),
                "median_immediate_evaluation_targets": float(
                    group["immediate_evaluation_targets"].median()
                ),
                "median_fixed_evaluation_targets": float(
                    group["fixed_evaluation_targets"].median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "warmup_order"])


def threshold_sensitivity(
    details: pd.DataFrame, thresholds: list[int], minimum_coverage: float
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        for split, split_group in details.groupby("split", sort=False):
            period_fixed = (
                (split_group["adaptation_targets"] >= threshold)
                & (
                    split_group["adaptation_target_coverage"]
                    >= minimum_coverage
                )
                & (split_group["fixed_evaluation_targets"] >= threshold)
            )
            period_immediate = (
                (split_group["adaptation_targets"] >= threshold)
                & (
                    split_group["adaptation_target_coverage"]
                    >= minimum_coverage
                )
                & (split_group["immediate_evaluation_targets"] >= threshold)
            )
            fixed_by_site = period_fixed.groupby(split_group["site_id"]).all()
            rows.append(
                {
                    "split": split,
                    "minimum_adaptation_and_evaluation_targets": threshold,
                    "minimum_adaptation_target_coverage": minimum_coverage,
                    "full_fixed_curve_sites": int(fixed_by_site.sum()),
                    "minimum_period_fixed_eligible_sites": int(
                        period_fixed.groupby(split_group["warmup_label"]).sum().min()
                    ),
                    "minimum_period_immediate_eligible_sites": int(
                        period_immediate.groupby(
                            split_group["warmup_label"]
                        ).sum().min()
                    ),
                }
            )
    return pd.DataFrame(rows)


def report_markdown(
    config: dict[str, Any],
    summary: pd.DataFrame,
    site_summary: pd.DataFrame,
    audit: dict[str, Any],
) -> str:
    minimum_adaptation = int(config["minimum_adaptation_targets"])
    minimum_adaptation_coverage = float(
        config["minimum_adaptation_target_coverage"]
    )
    minimum_evaluation = int(config["minimum_evaluation_targets"])
    validation_sites = int(
        site_summary.loc[
            site_summary["split"].eq("validation"), "full_fixed_curve_eligible"
        ].sum()
    )
    test_sites = int(
        site_summary.loc[
            site_summary["split"].eq("test"), "full_fixed_curve_eligible"
        ].sum()
    )

    display = summary[
        [
            "split",
            "warmup_label",
            "adaptation_eligible_sites",
            "immediate_curve_eligible_sites",
            "fixed_curve_eligible_sites",
            "full_fixed_curve_sites",
        ]
    ].copy()
    table_lines = [
        "| Split | Warm-up | Adaptation eligible | Immediate 30-day curve | Fixed-key curve | Complete fixed-key cohort |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in display.itertuples(index=False):
        table_lines.append(
            f"| {row.split} | {row.warmup_label} | "
            f"{row.adaptation_eligible_sites} | "
            f"{row.immediate_curve_eligible_sites} | "
            f"{row.fixed_curve_eligible_sites} | "
            f"{row.full_fixed_curve_sites} |"
        )

    return "\n".join(
        [
            "# Warm-up curve data-feasibility audit",
            "",
            "## Outcome",
            "",
            f"The strict fixed-key warm-up curve is feasible on **{validation_sites} "
            f"validation sites** and **{test_sites} test sites**. A site must have "
            f"at least {minimum_adaptation} common adaptation targets in every "
            f"warm-up period, at least {minimum_adaptation_coverage:.0%} valid-target "
            f"coverage in every period, and at least {minimum_evaluation} common "
            "targets after the 12-month boundary.",
            "",
            "No model was trained and no warm-up performance metric was inspected in "
            "this audit.",
            "",
            "## Eligibility counts",
            "",
            *table_lines,
            "",
            "## Primary experimental time split",
            "",
            "- Site deployment begins at its first valid 48-hour common-target key.",
            "- Adaptation labels are restricted to `[deployment_start, warmup_cutoff)`.",
            "- The primary evaluation set is the exact same set of valid target keys "
            "at or after `deployment_start + 12 months` for every warm-up duration.",
            "- The immediate 30-day evaluation is retained as an operational "
            "sensitivity analysis; its sites and timestamps change with warm-up "
            "duration and therefore it is not the primary controlled comparison.",
            "- The recurrent core is trained on training sites only. Validation sites "
            "are used to choose adaptation settings; test sites are evaluated only "
            "after the protocol is frozen.",
            (
                "- During evaluation, causal target history may include observations "
                "before the forecast hour, but evaluation labels never update model "
                "parameters."
                if bool(config.get("include_target_history", True))
                else
                "- Target history is excluded from the model inputs; evaluation "
                "labels never enter a forecast input or update model parameters."
            ),
            "",
            "## Leakage checks",
            "",
            f"- Adaptation/evaluation key overlap: "
            f"{audit['adaptation_evaluation_overlap_targets']}.",
            f"- Source common-key digest mismatches: "
            f"{audit['common_key_digest_mismatches']}.",
            f"- Sites with duplicate timestamps: "
            f"{audit['duplicate_timestamp_sites']}.",
            f"- Overall audit status: **{audit['status']}**.",
            "",
            "## Recommendation",
            "",
            "Use the complete fixed-key cohort for the thesis-facing warm-up curve. "
            "Report the immediate 30-day curve secondarily, with changing cohort size "
            "shown at every duration. Freeze the adaptation method on validation sites "
            "before producing test metrics.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_json(config_path)
    output_root = resolve(config["output_root"])
    reports = output_root / "reports"

    cohort = pd.read_csv(
        resolve(config["primary_cohort_path"]),
        dtype={"site_id": "string", "split": "string"},
    )
    manifest = pd.read_csv(
        resolve(config["common_key_manifest_path"]),
        dtype={"site_id": "string", "split": "string"},
    )
    prediction_root = resolve(config["reference_prediction_root"])

    detail_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    digest_mismatches: list[str] = []
    duplicate_sites: list[str] = []
    missing_prediction_sites: list[str] = []

    for split in config["evaluation_splits"]:
        expected = int(config["expected_sites"][split])
        cohort_sites = sorted(
            cohort.loc[cohort["split"].eq(split), "site_id"].astype(str).unique()
        )
        if len(cohort_sites) != expected:
            raise RuntimeError(
                f"{split}: expected {expected} cohort sites, found {len(cohort_sites)}"
            )

        for site_id in cohort_sites:
            path = prediction_root / split / f"{site_id}.parquet"
            if not path.exists():
                missing_prediction_sites.append(site_id)
                continue
            frame = pd.read_parquet(path, columns=["site_id", "timestamp"])
            timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
            if timestamps.has_duplicates:
                duplicate_sites.append(site_id)

            rows, site_summary = build_site_period_rows(
                site_id, split, timestamps, config
            )
            detail_rows.extend(rows)
            site_rows.append(site_summary)

            expected_row = manifest.loc[
                manifest["site_id"].eq(site_id) & manifest["split"].eq(split)
            ]
            if len(expected_row) != 1:
                digest_mismatches.append(site_id)
            else:
                expected_digest = str(expected_row.iloc[0]["common_key_sha256"])
                expected_count = int(expected_row.iloc[0]["common_target_windows"])
                if (
                    site_summary["common_key_sha256"] != expected_digest
                    or site_summary["common_target_count"] != expected_count
                ):
                    digest_mismatches.append(site_id)

    if missing_prediction_sites:
        raise RuntimeError(
            f"Missing prediction keys for {len(missing_prediction_sites)} sites"
        )

    details = pd.DataFrame(detail_rows).sort_values(
        ["split", "site_id", "warmup_order"]
    )
    sites = pd.DataFrame(site_rows).sort_values(["split", "site_id"])
    summary = summarize_by_period(details, sites)
    sensitivity = threshold_sensitivity(
        details,
        [int(value) for value in config["threshold_sensitivity_targets"]],
        float(config["minimum_adaptation_target_coverage"]),
    )
    strict_cohort = sites.loc[sites["full_fixed_curve_eligible"]].copy()

    overlap = int(
        details[
            [
                "adaptation_immediate_overlap_targets",
                "adaptation_fixed_overlap_targets",
            ]
        ]
        .to_numpy()
        .sum()
    )
    expected_total = sum(int(value) for value in config["expected_sites"].values())
    audit = {
        "status": "pass",
        "config": str(config_path),
        "source_sites": len(sites),
        "expected_source_sites": expected_total,
        "detail_rows": len(details),
        "expected_detail_rows": expected_total * len(config["warmup_periods"]),
        "missing_prediction_sites": missing_prediction_sites,
        "duplicate_timestamp_sites": len(duplicate_sites),
        "duplicate_timestamp_site_ids": duplicate_sites,
        "common_key_digest_mismatches": len(digest_mismatches),
        "common_key_digest_mismatch_sites": digest_mismatches,
        "adaptation_evaluation_overlap_targets": overlap,
        "validation_full_fixed_curve_sites": int(
            strict_cohort["split"].eq("validation").sum()
        ),
        "test_full_fixed_curve_sites": int(
            strict_cohort["split"].eq("test").sum()
        ),
        "test_labels_or_predictions_used_for_model_selection": False,
    }
    if (
        len(sites) != expected_total
        or len(details) != expected_total * len(config["warmup_periods"])
        or duplicate_sites
        or digest_mismatches
        or overlap
    ):
        audit["status"] = "fail"

    write_csv(reports / "warmup_site_period_feasibility.csv", details)
    write_csv(reports / "warmup_site_summary.csv", sites)
    write_csv(reports / "warmup_period_summary.csv", summary)
    write_csv(reports / "warmup_threshold_sensitivity.csv", sensitivity)
    write_csv(reports / "warmup_strict_fixed_key_cohort.csv", strict_cohort)
    write_json(reports / "warmup_feasibility_audit.json", audit)
    (reports / "WARMUP_FEASIBILITY_REPORT.md").write_text(
        report_markdown(config, summary, sites, audit), encoding="utf-8"
    )

    if audit["status"] != "pass":
        raise RuntimeError(
            "Warm-up feasibility audit failed; inspect warmup_feasibility_audit.json"
        )

    print(
        f"Warm-up audit passed: {len(sites)} sites; "
        f"strict validation={audit['validation_full_fixed_curve_sites']}; "
        f"strict test={audit['test_full_fixed_curve_sites']}"
    )


if __name__ == "__main__":
    main()
