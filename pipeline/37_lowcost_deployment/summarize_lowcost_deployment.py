"""Summarize full/low-cost cores and per-site held-out warm-up curves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            PROJECT_ROOT
            / "config"
            / "common1000_lowcost_deployment_summary_v1.json"
        ),
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def bootstrap_mean_ci(
    values: np.ndarray, replicates: int, seed: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1000):
        count = min(1000, replicates - start)
        sample = generator.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[sample].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def prediction_coverage(
    root: Path, run_id: str, tolerance: float
) -> dict[str, float | int]:
    files = sorted(
        (root / "predictions_by_site" / run_id / "test").glob("*.parquet")
    )
    count = 0
    within = 0
    absolute_sum = 0.0
    square_sum = 0.0
    for path in files:
        frame = pd.read_parquet(
            path, columns=["observed_native", "predicted_native"]
        )
        residual = (
            frame["predicted_native"].to_numpy(dtype=np.float64)
            - frame["observed_native"].to_numpy(dtype=np.float64)
        )
        count += len(residual)
        within += int((np.abs(residual) <= tolerance).sum())
        absolute_sum += float(np.abs(residual).sum())
        square_sum += float(np.square(residual).sum())
    if not count:
        raise RuntimeError(f"No held-out predictions for {run_id}")
    return {
        "evaluated_targets": count,
        "rmse_recomputed": float(np.sqrt(square_sum / count)),
        "mae_recomputed": float(absolute_sum / count),
        "within_tolerance_count": within,
        "within_tolerance_fraction": float(within / count),
    }


def earliest_site_thresholds(
    metrics: pd.DataFrame, target: str, tolerance: float
) -> pd.DataFrame:
    subset = metrics.loc[metrics["strategy"].eq("film_affine")].copy()
    rows = []
    for site_id, group in subset.groupby("site_id"):
        group = group.sort_values("warmup_order")
        achieved = group.loc[group["mae"].le(tolerance)]
        first = achieved.iloc[0] if len(achieved) else None
        rows.append(
            {
                "target": target,
                "site_id": str(site_id),
                "mae_tolerance": tolerance,
                "earliest_mae_threshold_warmup": (
                    str(first["warmup_label"]) if first is not None else "not_achieved"
                ),
                "earliest_mae_threshold_order": (
                    int(first["warmup_order"]) if first is not None else np.nan
                ),
                "minimum_warmup_mae": float(group["mae"].min()),
                "twelve_month_mae": float(
                    group.loc[group["warmup_label"].eq("12_months"), "mae"].iloc[0]
                ),
                "twelve_month_within_tolerance_fraction": float(
                    group.loc[
                        group["warmup_label"].eq("12_months"),
                        "within_tolerance_fraction",
                    ].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_combined_curve(
    summaries: dict[str, pd.DataFrame],
    target_specs: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, len(summaries), figsize=(12, 4.8))
    if len(summaries) == 1:
        axes = [axes]
    colors = {
        "zero_shot": "#555555",
        "film_affine": "#1676B8",
        "full_finetune": "#D36B27",
    }
    for axis, (target, curve) in zip(axes, summaries.items()):
        labels = (
            curve[["warmup_order", "warmup_label"]]
            .drop_duplicates()
            .sort_values("warmup_order")
        )
        x = np.arange(len(labels))
        for strategy in ("zero_shot", "film_affine", "full_finetune"):
            group = curve.loc[curve["strategy"].eq(strategy)].sort_values(
                "warmup_order"
            )
            axis.plot(
                x,
                group["macro_site_mae"],
                marker="o",
                linewidth=2,
                color=colors[strategy],
                label=strategy.replace("_", " "),
            )
        tolerance = float(target_specs[target]["tolerance"])
        axis.axhline(
            tolerance,
            color="#228833",
            linestyle="--",
            linewidth=1.4,
            label=f"MAE target ±{tolerance:g}",
        )
        axis.set_xticks(x, labels["warmup_label"].str.replace("_", " "), rotation=30)
        axis.set_title(target)
        axis.set_xlabel("Local expensive-sensor warm-up")
        axis.set_ylabel(f"Macro site MAE ({target_specs[target]['unit']})")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_site_curves(
    metrics: pd.DataFrame,
    target: str,
    unit: str,
    tolerance: float,
    output: Path,
) -> None:
    subset = metrics.loc[metrics["strategy"].eq("film_affine")].copy()
    for site_id, group in subset.groupby("site_id"):
        group = group.sort_values("warmup_order")
        labels = group["warmup_label"].str.replace("_", " ")
        x = np.arange(len(group))
        figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
        axes[0].plot(x, group["mae"], marker="o", color="#1676B8")
        axes[0].axhline(tolerance, color="#228833", linestyle="--")
        axes[0].set_ylabel(f"MAE ({unit})")
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].plot(
            x,
            group["within_tolerance_fraction"],
            marker="o",
            color="#6A3D9A",
        )
        axes[1].set_ylabel(f"Fraction within ±{tolerance:g}")
        axes[1].set_ylim(0, 1.02)
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].set_xticks(x, labels, rotation=30)
        axes[1].set_xlabel("Local warm-up data")
        figure.suptitle(f"{target} low-cost warm-up — site {site_id}")
        figure.tight_layout()
        path = output / target.lower() / f"{site_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    config = load_json(resolve(args.config))
    output = resolve(config["output_root"])
    reports = output / "reports"
    figures = output / "figures"
    core_rows = []
    paired_rows = []
    warmup_rows = []
    threshold_frames = []
    curve_by_target: dict[str, pd.DataFrame] = {}
    target_specs = {str(row["target"]): row for row in config["targets"]}

    for spec in config["targets"]:
        target = str(spec["target"])
        tolerance = float(spec["tolerance"])
        core_root = resolve(spec["core_root"])
        overall = pd.read_csv(core_root / "reports" / "model_metrics_all.csv")
        site = pd.read_csv(
            core_root / "reports" / "site_metrics_all.csv",
            dtype={"site_id": "string"},
        )
        for role, run_id in (
            ("full_sensors", spec["full_run_id"]),
            ("low_cost", spec["lowcost_run_id"]),
        ):
            row = overall.loc[
                overall["split"].eq("test") & overall["run_id"].eq(run_id)
            ].iloc[0].to_dict()
            row.update(
                {
                    "target_name": target,
                    "sensor_role": role,
                    "absolute_error_tolerance": tolerance,
                    **prediction_coverage(core_root, str(run_id), tolerance),
                }
            )
            core_rows.append(row)

        full = site.loc[
            site["split"].eq("test")
            & site["run_id"].eq(spec["full_run_id"]),
            ["site_id", "rmse"],
        ].rename(columns={"rmse": "full_sensor_rmse"})
        low = site.loc[
            site["split"].eq("test")
            & site["run_id"].eq(spec["lowcost_run_id"]),
            ["site_id", "rmse"],
        ].rename(columns={"rmse": "lowcost_rmse"})
        paired = full.merge(low, on="site_id", validate="one_to_one")
        paired["delta_rmse_lowcost_minus_full"] = (
            paired["lowcost_rmse"] - paired["full_sensor_rmse"]
        )
        delta = paired["delta_rmse_lowcost_minus_full"].to_numpy()
        lower, upper = bootstrap_mean_ci(
            delta,
            int(config["bootstrap_replicates"]),
            stable_seed(config["random_seed"], target, "sensor_ablation"),
        )
        paired_rows.append(
            {
                "target": target,
                "sites": len(paired),
                "mean_site_delta_rmse_lowcost_minus_full": float(delta.mean()),
                "bootstrap_95_ci_lower": lower,
                "bootstrap_95_ci_upper": upper,
                "sites_lowcost_better": int((delta < 0).sum()),
                "sites_full_better": int((delta > 0).sum()),
            }
        )

        warmup_root = resolve(spec["warmup_root"])
        curve = pd.read_csv(
            warmup_root / "reports" / "warmup_curve_summary.csv"
        )
        metrics = pd.read_csv(
            warmup_root / "reports" / "warmup_site_metrics.csv",
            dtype={"site_id": "string"},
        )
        curve.insert(0, "target", target)
        warmup_rows.extend(curve.to_dict(orient="records"))
        curve_by_target[target] = curve
        threshold_frames.append(
            earliest_site_thresholds(metrics, target, tolerance)
        )
        plot_site_curves(
            metrics,
            target,
            str(spec["unit"]),
            tolerance,
            figures / "per_site_warmup",
        )

    core = pd.DataFrame(core_rows)
    paired = pd.DataFrame(paired_rows)
    warmup = pd.DataFrame(warmup_rows)
    thresholds = pd.concat(threshold_frames, ignore_index=True)
    write_csv(reports / "core_sensor_ablation.csv", core)
    write_csv(reports / "paired_site_sensor_ablation.csv", paired)
    write_csv(reports / "heldout_warmup_summary.csv", warmup)
    write_csv(reports / "per_site_earliest_mae_threshold.csv", thresholds)
    plot_combined_curve(
        curve_by_target,
        target_specs,
        figures / "lowcost_warmup_pH_DO.png",
    )

    lines = [
        "# Low-cost pH and DO deployment summary",
        "",
        "All held-out results below exclude target history. Low-cost inputs are "
        "water temperature (00010), specific conductance (00095), turbidity "
        "(63680), and hourly rainfall (00045).",
        "",
        "## Full-sensor versus low-cost core",
        "",
        "| Target | Input | Test RMSE | Test MAE | Within tolerance |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in core.itertuples(index=False):
        lines.append(
            f"| {row.target_name} | {row.sensor_role} | {row.rmse:.4f} | "
            f"{row.mae:.4f} | {row.within_tolerance_fraction:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Twelve-month held-out warm-up",
            "",
            "| Target | Strategy | Macro RMSE | Macro MAE | Pooled within tolerance |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    twelve = warmup.loc[warmup["warmup_label"].eq("12_months")]
    for row in twelve.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.strategy} | {row.macro_site_rmse:.4f} | "
            f"{row.macro_site_mae:.4f} | "
            f"{row.pooled_within_tolerance_fraction:.1%} |"
        )
    lines.extend(
        [
            "",
            "A station's earliest threshold in "
            "`per_site_earliest_mae_threshold.csv` is the first tested duration "
            "whose held-out site MAE is no greater than ±0.2 pH or ±1.0 mg/L DO. "
            "The prediction-level fraction within that absolute-error tolerance "
            "is reported separately; it is not described as classification accuracy.",
            "",
        ]
    )
    (reports / "FINAL_LOW_COST_DEPLOYMENT_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    audit = {
        "status": "pass",
        "targets": sorted(core["target_name"].unique().tolist()),
        "core_test_rows": len(core),
        "paired_sensor_rows": len(paired),
        "warmup_rows": len(warmup),
        "site_threshold_rows": len(thresholds),
        "target_history_used": False,
        "per_site_figures": len(
            list((figures / "per_site_warmup").glob("*/*.png"))
        ),
    }
    if (
        len(core) != 4
        or len(paired) != 2
        or len(warmup) != 42
        or not len(thresholds)
    ):
        audit["status"] = "fail"
    write_json(reports / "independent_audit.json", audit)
    if audit["status"] != "pass":
        raise RuntimeError(audit)
    print(
        f"Deployment summary complete: {len(thresholds)} site-target curves, "
        f"{audit['per_site_figures']} figures"
    )


if __name__ == "__main__":
    main()
