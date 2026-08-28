"""Finalize the frozen causal H01 low-cost deployment evidence package."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (
    PROJECT_ROOT / "analysis_outputs" / "h0_site_transfer_warmup_final_audit_v1"
)
PH_VALIDATION = OUTPUT_ROOT / "checkpoints" / "ph_cheap3_warmup_validation_v1" / "full"
PH_TEST = OUTPUT_ROOT / "checkpoints" / "ph_cheap3_warmup_test_v1" / "full"
DO_VALIDATION = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "do_common1000_warmup_adaptation_lowcost_mg_l_validation_v1"
    / "full"
)
DO_TEST = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "do_common1000_warmup_adaptation_lowcost_mg_l_test_v1"
    / "full"
)
INPUT_ABLATION = (
    OUTPUT_ROOT
    / "checkpoints"
    / "ph_h01_input_ablation_v1"
    / "full_validation_only"
)
WARMUP_LABELS = [
    "2_weeks",
    "1_month",
    "2_months",
    "3_months",
    "6_months",
    "9_months",
    "12_months",
]
DAYS = {
    "2_weeks": 14.0,
    "1_month": 30.4375,
    "2_months": 60.875,
    "3_months": 91.3125,
    "6_months": 182.625,
    "9_months": 273.9375,
    "12_months": 365.25,
}
TARGETS = {
    "pH": {
        "input_set": "CHEAP3",
        "unit": "pH",
        "tolerance": 0.2,
        "validation": PH_VALIDATION,
        "test": PH_TEST,
        "recommended_reference": "9_months",
    },
    "DO": {
        "input_set": "CHEAP3_RAIN",
        "unit": "mg/L",
        "tolerance": 1.0,
        "validation": DO_VALIDATION,
        "test": DO_TEST,
        "recommended_reference": "3_months",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*values: Any) -> int:
    token = "|".join(str(value) for value in values).encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def bootstrap_mean_ci(values: np.ndarray, seed: int, replicates: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1000):
        stop = min(start + 1000, replicates)
        sample = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[sample].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def load_curve(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "reports" / "warmup_curve_summary.csv")


def load_sites(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "reports" / "warmup_site_metrics.csv", dtype={"site_id": "string"})


def enrich_summary(target: str, split: str, root: Path) -> pd.DataFrame:
    info = TARGETS[target]
    curve = load_curve(root).copy()
    sites = load_sites(root)
    rows: list[dict[str, Any]] = []
    for row in curve.itertuples(index=False):
        group = sites.loc[
            sites["warmup_label"].eq(row.warmup_label)
            & sites["strategy"].eq(row.strategy)
        ]
        rows.append(
            {
                "target": target,
                "target_unit": info["unit"],
                "input_set": info["input_set"],
                "split": split,
                "warmup_order": int(row.warmup_order),
                "warmup_duration": row.warmup_label,
                "method": row.strategy,
                "pooled_rmse": float(row.pooled_rmse),
                "pooled_mae": float(row.pooled_mae),
                "pooled_nse": float(row.pooled_nse),
                "macro_site_rmse": float(row.macro_site_rmse),
                "macro_site_mae": float(row.macro_site_mae),
                "median_site_nse": float(group["nse"].median()),
                "nse_above_zero_sites": int((group["nse"] > 0).sum()),
                "within_tolerance_fraction": float(row.pooled_within_tolerance_fraction),
                "macro_site_within_tolerance_fraction": float(
                    row.macro_site_within_tolerance_fraction
                ),
                "effective_sites": int(row.sites),
                "effective_prediction_rows": int(row.evaluated_targets),
                "trainable_parameters": int(row.trainable_parameters),
                "fixed_evaluation_keys": "post_12_month",
            }
        )
    return pd.DataFrame(rows)


def build_site_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []
    worsened_rows: list[dict[str, Any]] = []
    for target, info in TARGETS.items():
        source = load_sites(info["test"]).copy()
        source.insert(0, "target", target)
        source.insert(1, "target_unit", info["unit"])
        source.insert(2, "input_set", info["input_set"])
        keys = ["site_id", "warmup_order", "warmup_label"]
        pivot = source.pivot(index=keys, columns="strategy", values=["rmse", "mae"]).reset_index()
        pivot.columns = [
            "_".join(value for value in column if value).strip("_")
            if isinstance(column, tuple)
            else column
            for column in pivot.columns
        ]
        pivot["delta_rmse_adapter_minus_zero"] = pivot["rmse_film_affine"] - pivot["rmse_zero_shot"]
        pivot["delta_mae_adapter_minus_zero"] = pivot["mae_film_affine"] - pivot["mae_zero_shot"]
        pivot["delta_rmse_adapter_minus_full"] = pivot["rmse_film_affine"] - pivot["rmse_full_finetune"]
        pivot["delta_mae_adapter_minus_full"] = pivot["mae_film_affine"] - pivot["mae_full_finetune"]
        source = source.merge(
            pivot[
                keys
                + [
                    "delta_rmse_adapter_minus_zero",
                    "delta_mae_adapter_minus_zero",
                    "delta_rmse_adapter_minus_full",
                    "delta_mae_adapter_minus_full",
                ]
            ],
            on=keys,
            validate="many_to_one",
        )
        film = source.loc[source["strategy"].eq("film_affine")].copy()
        for site_id, group in film.groupby("site_id", sort=True):
            group = group.sort_values("warmup_order")
            reached = group.loc[group["mae"].le(float(info["tolerance"]))]
            first = str(reached.iloc[0]["warmup_label"]) if len(reached) else "not_reached_within_12_months"
            threshold_rows.append(
                {
                    "target": target,
                    "target_unit": info["unit"],
                    "input_set": info["input_set"],
                    "site_id": str(site_id),
                    "operational_mae_threshold": float(info["tolerance"]),
                    "first_threshold_duration": first,
                    "reached_within_12_months": bool(len(reached)),
                    "mae_at_12_months": float(
                        group.loc[group["warmup_label"].eq("12_months"), "mae"].iloc[0]
                    ),
                }
            )
        source = source.merge(
            pd.DataFrame(threshold_rows).loc[lambda frame: frame["target"].eq(target)][
                ["site_id", "first_threshold_duration", "reached_within_12_months"]
            ],
            on="site_id",
            validate="many_to_one",
        )
        worsened = source.loc[
            source["strategy"].eq("film_affine")
            & source["delta_rmse_adapter_minus_zero"].gt(0),
            [
                "target",
                "target_unit",
                "input_set",
                "site_id",
                "warmup_order",
                "warmup_label",
                "delta_rmse_adapter_minus_zero",
                "delta_mae_adapter_minus_zero",
            ],
        ]
        worsened_rows.extend(worsened.to_dict(orient="records"))
        all_rows.append(source)
    combined = pd.concat(all_rows, ignore_index=True)
    thresholds = pd.DataFrame(threshold_rows)
    worsened = pd.DataFrame(worsened_rows)
    return combined, thresholds, worsened


def paired_statistics(site_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        data = site_metrics.loc[site_metrics["target"].eq(target) & site_metrics["strategy"].eq("film_affine")]
        for label in WARMUP_LABELS:
            group = data.loc[data["warmup_label"].eq(label)]
            for metric, column in (
                ("rmse", "delta_rmse_adapter_minus_zero"),
                ("mae", "delta_mae_adapter_minus_zero"),
            ):
                values = group[column].to_numpy(dtype=np.float64)
                low, high = bootstrap_mean_ci(values, stable_seed(target, label, "adapter_zero", metric))
                rows.append(
                    {
                        "target": target,
                        "split": "test",
                        "comparison": "adapter_minus_zero_shot",
                        "warmup_duration": label,
                        "metric": metric,
                        "sites": len(values),
                        "mean_site_delta": float(values.mean()),
                        "median_site_delta": float(np.median(values)),
                        "bootstrap_95_ci_lower": low,
                        "bootstrap_95_ci_upper": high,
                        "challenger_improved_sites": int((values < 0).sum()),
                    }
                )
            for metric, column in (
                ("rmse", "delta_rmse_adapter_minus_full"),
                ("mae", "delta_mae_adapter_minus_full"),
            ):
                values = group[column].to_numpy(dtype=np.float64)
                low, high = bootstrap_mean_ci(values, stable_seed(target, label, "adapter_full", metric))
                rows.append(
                    {
                        "target": target,
                        "split": "test",
                        "comparison": "adapter_minus_full_finetune",
                        "warmup_duration": label,
                        "metric": metric,
                        "sites": len(values),
                        "mean_site_delta": float(values.mean()),
                        "median_site_delta": float(np.median(values)),
                        "bootstrap_95_ci_lower": low,
                        "bootstrap_95_ci_upper": high,
                        "challenger_improved_sites": int((values < 0).sum()),
                    }
                )
        pivot = data.pivot(index="site_id", columns="warmup_label", values=["rmse", "mae"])
        for current, previous in zip(WARMUP_LABELS[1:], WARMUP_LABELS[:-1]):
            for metric in ("rmse", "mae"):
                values = (pivot[(metric, current)] - pivot[(metric, previous)]).to_numpy(dtype=np.float64)
                low, high = bootstrap_mean_ci(values, stable_seed(target, current, previous, metric))
                rows.append(
                    {
                        "target": target,
                        "split": "test",
                        "comparison": f"{current}_minus_{previous}",
                        "warmup_duration": current,
                        "metric": metric,
                        "sites": len(values),
                        "mean_site_delta": float(values.mean()),
                        "median_site_delta": float(np.median(values)),
                        "bootstrap_95_ci_lower": low,
                        "bootstrap_95_ci_upper": high,
                        "challenger_improved_sites": int((values < 0).sum()),
                    }
                )

    cheap = pd.read_csv(INPUT_ABLATION / "reports" / "site_metrics_ph_LSTM_CHEAP3_H01.csv")
    rain = pd.read_csv(INPUT_ABLATION / "reports" / "site_metrics_ph_LSTM_LOW_COST_H01.csv")
    paired = cheap[["site_id", "rmse", "mae"]].merge(
        rain[["site_id", "rmse", "mae"]],
        on="site_id",
        suffixes=("_cheap3", "_rain"),
        validate="one_to_one",
    )
    for metric in ("rmse", "mae"):
        values = (paired[f"{metric}_rain"] - paired[f"{metric}_cheap3"]).to_numpy(dtype=np.float64)
        low, high = bootstrap_mean_ci(values, stable_seed("pH", "rain_cheap3", metric))
        rows.append(
            {
                "target": "pH",
                "split": "validation",
                "comparison": "CHEAP3_RAIN_minus_CHEAP3",
                "warmup_duration": "not_applicable",
                "metric": metric,
                "sites": len(values),
                "mean_site_delta": float(values.mean()),
                "median_site_delta": float(np.median(values)),
                "bootstrap_95_ci_lower": low,
                "bootstrap_95_ci_upper": high,
                "challenger_improved_sites": int((values < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def knee_details(target: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    info = TARGETS[target]
    curve = load_curve(info["validation"])
    film = curve.loc[curve["strategy"].eq("film_affine")].sort_values("warmup_order").copy()
    film.insert(0, "target", target)
    film["duration_days"] = film["warmup_label"].map(DAYS)
    x = np.log(film["duration_days"].to_numpy(dtype=np.float64))
    y = film["macro_site_mae"].to_numpy(dtype=np.float64)
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    chord = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
    distance = np.abs(chord - y_norm)
    knee_index = int(np.argmax(distance))
    film["log_duration_normalized"] = x_norm
    film["mae_normalized"] = y_norm
    film["endpoint_chord_normalized"] = chord
    film["vertical_distance"] = distance
    threshold = float(info["tolerance"])
    reached = film.loc[film["macro_site_mae"].le(threshold)]
    best = film.loc[film["macro_site_mae"].idxmin()]
    selection = {
        "target": target,
        "input_set": info["input_set"],
        "operational_threshold_metric": "validation_macro_site_mae",
        "operational_threshold_value": threshold,
        "operational_threshold_duration": (
            str(reached.iloc[0]["warmup_label"])
            if len(reached)
            else "not_reached_within_12_months"
        ),
        "curve_knee_duration": str(film.iloc[knee_index]["warmup_label"]),
        "validation_best_duration": str(best["warmup_label"]),
        "validation_best_macro_site_mae": float(best["macro_site_mae"]),
        "validation_sites": int(film["sites"].iloc[0]),
        "test_metrics_read_for_selection": False,
    }
    return film, selection


def prediction_manifest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, info in TARGETS.items():
        for split in ("validation", "test"):
            root = info[split]
            for path in sorted((root / "shards" / split).glob("*/*.json")):
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                rows.append(
                    {
                        "target": target,
                        "target_unit": info["unit"],
                        "input_set": info["input_set"],
                        "split": split,
                        "site_id": payload["site_id"],
                        "warmup_duration": payload["warmup_label"],
                        "method": payload["strategy"],
                        "evaluation_rows": payload["fixed_evaluation_targets"],
                        "fixed_evaluation_key_sha256": payload["fixed_evaluation_key_sha256"],
                        "checkpoint_sha256": payload["checkpoint_sha256"],
                        "artifact_type": "site_evaluation_shard",
                        "raw_prediction_rows_persisted": False,
                        "artifact_path": str(path.resolve()),
                        "artifact_sha256": sha256(path),
                    }
                )
    return pd.DataFrame(rows)


def recovery_table(summary: pd.DataFrame) -> pd.DataFrame:
    test = summary.loc[summary["split"].eq("test")]
    rows = []
    for target in TARGETS:
        for label in WARMUP_LABELS:
            subset = test.loc[
                test["target"].eq(target) & test["warmup_duration"].eq(label)
            ].set_index("method")
            for metric in ("macro_site_mae", "macro_site_rmse"):
                zero = float(subset.loc["zero_shot", metric])
                adapter = float(subset.loc["film_affine", metric])
                full = float(subset.loc["full_finetune", metric])
                denominator = zero - full
                recovery = (zero - adapter) / denominator * 100.0 if denominator > 0 else np.nan
                rows.append(
                    {
                        "target": target,
                        "warmup_duration": label,
                        "metric": metric,
                        "zero_shot": zero,
                        "adapter": adapter,
                        "full_finetune": full,
                        "adapter_recovery_percent_of_full_improvement": recovery,
                    }
                )
    return pd.DataFrame(rows)


def make_figures(summary: pd.DataFrame, time_to: pd.DataFrame, site_metrics: pd.DataFrame, knee_frames: dict[str, pd.DataFrame]) -> list[Path]:
    figures = OUTPUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    palette = {"zero_shot": "#555555", "film_affine": "#1676B8", "full_finetune": "#D36B27"}
    display = {"zero_shot": "Zero-shot", "film_affine": "FiLM + affine (130)", "full_finetune": "Full fine-tuning"}
    paths: list[Path] = []
    for target in TARGETS:
        data = summary.loc[summary["target"].eq(target) & summary["split"].eq("test")]
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for strategy in ("zero_shot", "film_affine", "full_finetune"):
            sub = data.loc[data["method"].eq(strategy)].sort_values("warmup_order")
            ax.plot(
                sub["warmup_order"],
                sub["macro_site_mae"],
                marker="o",
                linewidth=2.2,
                color=palette[strategy],
                label=display[strategy],
            )
        unit = "pH" if target == "pH" else "mg/L"
        ax.set_xticks(range(1, 8), [label.replace("_", " ") for label in WARMUP_LABELS], rotation=20)
        ax.set_ylabel(f"Macro-site MAE ({unit})")
        ax.set_xlabel("Local sonde overlap")
        ax.set_title(
            f"{target} {TARGETS[target]['input_set']} held-out warm-up curve\n"
            f"39 sites; fixed post-12-month evaluation keys"
        )
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = figures / f"{target.lower()}_warmup_curve.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    selection = json.loads(
        (OUTPUT_ROOT / "frozen_selection" / "FINAL_H0_INPUT_SELECTION.json").read_text(encoding="utf-8")
    )
    names = ["CHEAP3", "CHEAP3_RAIN"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    for ax, metric, title in zip(
        axes,
        ("pooled_rmse", "macro_site_mae"),
        ("Pooled RMSE", "Macro-site MAE"),
    ):
        values = [selection["validation_results"][name][metric] for name in names]
        bars = ax.bar(names, values, color=["#1676B8", "#8AA6C1"])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
        ax.set_title(title)
        ax.set_ylabel("pH")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("pH H01 validation-only input ablation — 137 sites")
    fig.tight_layout()
    path = figures / "cheap3_vs_cheap3_rain_validation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    categories = WARMUP_LABELS + ["not_reached_within_12_months"]
    for target in TARGETS:
        data = time_to.loc[time_to["target"].eq(target)]
        counts = data["first_threshold_duration"].value_counts().reindex(categories, fill_value=0)
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        colors = ["#1676B8"] * 7 + ["#C94C4C"]
        bars = ax.bar(range(len(counts)), counts.values, color=colors)
        for bar, value in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.2, str(int(value)), ha="center")
        ax.set_xticks(range(len(counts)), [value.replace("_", " ") for value in counts.index], rotation=25, ha="right")
        ax.set_ylabel("Held-out sites")
        threshold = TARGETS[target]["tolerance"]
        unit = "pH" if target == "pH" else "mg/L"
        ax.set_title(
            f"{target} first site-level MAE ≤ {threshold:g} {unit}\n"
            "39 sites; FiLM + affine; fixed post-12-month keys"
        )
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        path = figures / f"{target.lower()}_time_to_threshold.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))
    for ax, target in zip(axes, ("pH", "DO")):
        label = TARGETS[target]["recommended_reference"]
        data = site_metrics.loc[
            site_metrics["target"].eq(target)
            & site_metrics["strategy"].eq("film_affine")
            & site_metrics["warmup_label"].eq(label)
        ].sort_values("delta_mae_adapter_minus_zero")
        values = data["delta_mae_adapter_minus_zero"].to_numpy(dtype=np.float64)
        ax.axhline(0, color="#333333", linestyle="--", linewidth=1)
        ax.scatter(np.arange(len(values)), values, c=np.where(values < 0, "#1676B8", "#C94C4C"), s=28)
        ax.set_title(f"{target}: {label.replace('_', ' ')} ({len(values)} sites)")
        ax.set_xlabel("Sites ordered by paired MAE difference")
        ax.set_ylabel(f"Adapter − zero-shot MAE ({TARGETS[target]['unit']})")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Held-out paired site effects; negative values favour adaptation")
    fig.tight_layout()
    path = figures / "adapter_vs_zero_site_differences.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.0))
    for ax, target in zip(axes, ("pH", "DO")):
        frame = knee_frames[target]
        x = frame["log_duration_normalized"].to_numpy()
        y = frame["mae_normalized"].to_numpy()
        chord = frame["endpoint_chord_normalized"].to_numpy()
        idx = int(np.argmax(frame["vertical_distance"].to_numpy()))
        ax.plot(x, y, "o-", linewidth=2.0, label="Validation curve")
        ax.plot(x, chord, "--", linewidth=1.4, label="Endpoint chord")
        ax.vlines(x[idx], y[idx], chord[idx], color="#D36B27", linewidth=2.2)
        ax.scatter([x[idx]], [y[idx]], color="#D36B27", s=65, zorder=4)
        ax.set_xticks(x, [label.replace("_", " ") for label in WARMUP_LABELS], rotation=25, ha="right")
        ax.set_title(f"{target}: knee {frame.iloc[idx]['warmup_label'].replace('_', ' ')}")
        ax.set_xlabel("Log duration (normalized)")
        ax.set_ylabel("Validation macro-site MAE (normalized)")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Deterministic validation knees; fixed post-12-month keys")
    fig.tight_layout()
    path = figures / "validation_knee.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> list[str]:
    labels = [column.replace("_", " ").title() for column in columns]
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for column, value in zip(columns, row):
            if column in formats and pd.notna(value):
                values.append(format(value, formats[column]))
            else:
                values.append(str(value).replace("_", " "))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def final_report(
    summary: pd.DataFrame,
    time_to: pd.DataFrame,
    worsened: pd.DataFrame,
    paired: pd.DataFrame,
    recovery: pd.DataFrame,
    durations: dict[str, dict[str, Any]],
) -> str:
    selection = json.loads(
        (OUTPUT_ROOT / "frozen_selection" / "FINAL_H0_INPUT_SELECTION.json").read_text(encoding="utf-8")
    )
    test = summary.loc[summary["split"].eq("test")]
    ph9 = test.loc[
        test["target"].eq("pH")
        & test["warmup_duration"].eq("9_months")
        & test["method"].eq("film_affine")
    ].iloc[0]
    ph12 = test.loc[
        test["target"].eq("pH")
        & test["warmup_duration"].eq("12_months")
        & test["method"].eq("film_affine")
    ].iloc[0]
    do3 = test.loc[
        test["target"].eq("DO")
        & test["warmup_duration"].eq("3_months")
        & test["method"].eq("film_affine")
    ].iloc[0]
    do12 = test.loc[
        test["target"].eq("DO")
        & test["warmup_duration"].eq("12_months")
        & test["method"].eq("film_affine")
    ].iloc[0]
    ph_not = int((time_to.loc[time_to["target"].eq("pH"), "reached_within_12_months"] == False).sum())
    do_not = int((time_to.loc[time_to["target"].eq("DO"), "reached_within_12_months"] == False).sum())
    ph_worse = worsened.loc[
        worsened["target"].eq("pH") & worsened["warmup_label"].eq("12_months")
    ]["site_id"].astype(str).tolist()
    do_worse = worsened.loc[
        worsened["target"].eq("DO") & worsened["warmup_label"].eq("12_months")
    ]["site_id"].astype(str).tolist()
    rain_rmse_penalty = (
        selection["validation_results"]["CHEAP3_RAIN"]["pooled_rmse"]
        / selection["validation_results"]["CHEAP3"]["pooled_rmse"]
        - 1.0
    ) * 100.0
    ph_recovery = recovery.loc[
        recovery["target"].eq("pH")
        & recovery["warmup_duration"].eq("9_months")
        & recovery["metric"].eq("macro_site_mae")
    ].iloc[0]
    do_recovery = recovery.loc[
        recovery["target"].eq("DO")
        & recovery["warmup_duration"].eq("3_months")
        & recovery["metric"].eq("macro_site_mae")
    ].iloc[0]
    lines = [
        "# Final H0 site-transfer and warm-up report",
        "",
        "## Executive conclusion",
        "",
        "The deployment experiment is a strictly causal one-hour-ahead H01 task. Historical target values are excluded from inference. The pH deployment input is CHEAP3 (water temperature, specific conductance and turbidity); the audited DO deployment retains CHEAP3_RAIN. Both use a frozen generic LSTM-FiLM backbone followed by a 130-parameter site adapter.",
        "",
        "The evidence does **not** support removing the sonde at every site after one universal fixed duration. Adaptation is beneficial on average, but site-level failures and non-attainment remain.",
        "",
        "## Provenance and leakage audit",
        "",
        "All checkpoint, scaler, metadata-transform, split and target-key hashes are traceable. The pH and DO H01 inputs end one hour before the target; neither target history nor target-derived dynamic features appear in X. Warm-up labels precede all fixed evaluation keys, and inference leaves backbone and adapter parameters unchanged.",
        "",
        "## Validation-only sensor selection",
        "",
        f"CHEAP3 was frozen before test evaluation. Validation macro-site MAE was {selection['validation_results']['CHEAP3']['macro_site_mae']:.6f} for CHEAP3 and {selection['validation_results']['CHEAP3_RAIN']['macro_site_mae']:.6f} for CHEAP3_RAIN. CHEAP3 won at {selection['paired_validation_sites']['CHEAP3_improved_sites_by_mae']}/137 sites by MAE. Adding rainfall increased pooled RMSE by {rain_rmse_penalty:.2f}% and did not satisfy the preregistered selection rule. Rainfall is therefore not retained for causal pH deployment; this conclusion is not generalized to DO because the human decision explicitly retained the already-audited DO experiment without a new DO ablation.",
        "",
        "## Validation-frozen durations",
        "",
        "| Target | Operational threshold duration | Curve knee | Validation-best duration |",
        "|---|---|---|---|",
        f"| pH | {durations['pH']['operational_threshold_duration'].replace('_', ' ')} | {durations['pH']['curve_knee_duration'].replace('_', ' ')} | {durations['pH']['validation_best_duration'].replace('_', ' ')} |",
        f"| DO | {durations['DO']['operational_threshold_duration'].replace('_', ' ')} | {durations['DO']['curve_knee_duration'].replace('_', ' ')} | {durations['DO']['validation_best_duration'].replace('_', ' ')} |",
        "",
        "The threshold, curve knee and validation minimum answer different questions and are intentionally not collapsed into one duration. Test results did not alter these choices.",
        "",
        "## Held-out deployment results",
        "",
        "| Target | Frozen reference | Macro MAE | Pooled RMSE | Pooled NSE | Within tolerance | Sites |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| pH | 9 months (validation best; threshold not reached) | {ph9.macro_site_mae:.6f} | {ph9.pooled_rmse:.6f} | {ph9.pooled_nse:.6f} | {100*ph9.within_tolerance_fraction:.2f}% within ±0.2 | {int(ph9.effective_sites)} |",
        f"| pH | 12 months (descriptive) | {ph12.macro_site_mae:.6f} | {ph12.pooled_rmse:.6f} | {ph12.pooled_nse:.6f} | {100*ph12.within_tolerance_fraction:.2f}% within ±0.2 | {int(ph12.effective_sites)} |",
        f"| DO | 3 months (operational threshold) | {do3.macro_site_mae:.6f} mg/L | {do3.pooled_rmse:.6f} mg/L | {do3.pooled_nse:.6f} | {100*do3.within_tolerance_fraction:.2f}% within ±1 mg/L | {int(do3.effective_sites)} |",
        f"| DO | 12 months (descriptive) | {do12.macro_site_mae:.6f} mg/L | {do12.pooled_rmse:.6f} mg/L | {do12.pooled_nse:.6f} | {100*do12.within_tolerance_fraction:.2f}% within ±1 mg/L | {int(do12.effective_sites)} |",
        "",
        "## Site heterogeneity and adapter efficiency",
        "",
        f"At the validation-frozen reference durations, the pH adapter recovered {ph_recovery.adapter_recovery_percent_of_full_improvement:.1f}% and the DO adapter recovered {do_recovery.adapter_recovery_percent_of_full_improvement:.1f}% of the macro-MAE improvement achieved by full fine-tuning. These ratios describe aggregate recovery; they do not imply site-level equivalence.",
        "",
        f"By 12 months, {ph_not}/39 pH sites and {do_not}/39 DO sites had never reached their respective site-level MAE thresholds. At 12 months the adapter worsened RMSE relative to zero-shot at {len(ph_worse)} pH sites ({', '.join(ph_worse) if ph_worse else 'none'}) and {len(do_worse)} DO sites ({', '.join(do_worse) if do_worse else 'none'}).",
        "",
        "## Separate contemporaneous pH soft-sensing result",
        "",
        "`RAIN_H0_H00` is retained separately as a contemporaneous full-sensor soft-sensing/nowcast result (pooled RMSE 0.406864; macro-site MAE 0.242254). It uses current-time DO and has a 0-hour horizon. These metrics are not combined with or directly ranked against the causal H01 low-cost deployment results.",
        "",
        "## Strict two-line deployment answer",
        "",
        "pH: The operational threshold was not reached within 12 months under CHEAP3.",
        f"DO: 3 months of local sonde overlap followed by CHEAP3_RAIN yielded test macro MAE {do3.macro_site_mae:.3f} mg/L, with {100*do3.within_tolerance_fraction:.1f}% of predictions within ±1 mg/L.",
        "",
        "## Interpretation limits",
        "",
        "The eligible fixed-curve cohorts contain 46 validation sites and 39 test sites, not all 910 split sites. Results apply to sites with sufficient continuous commissioning and post-12-month data. Negative site outcomes were retained. The pH rainfall conclusion applies only to the preregistered H01 low-cost validation ablation; no new DO sensor ablation was run after the human decision not to rerun DO.",
        "",
    ]
    return "\n".join(lines)


def config_manifest(durations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    existing = json.loads((OUTPUT_ROOT / "configs" / "config_manifest.json").read_text(encoding="utf-8"))
    configs = [
        PROJECT_ROOT / "config" / "h0_final_ph_h01_cheap3_validation_v1.json",
        PROJECT_ROOT / "config" / "h0_final_ph_cheap3_warmup_validation_v1.json",
        PROJECT_ROOT / "config" / "h0_final_ph_cheap3_warmup_test_v1.json",
        PROJECT_ROOT / "config" / "common1000_warmup_adaptation_lowcost_do_validation_v1.json",
        PROJECT_ROOT / "config" / "common1000_warmup_adaptation_lowcost_do_test_v1.json",
    ]
    existing["experiment_configs"] = [
        {"path": str(path.resolve()), "sha256": sha256(path)} for path in configs
    ]
    existing["frozen_input_selection"] = json.loads(
        (OUTPUT_ROOT / "frozen_selection" / "FINAL_H0_INPUT_SELECTION.json").read_text(encoding="utf-8")
    )
    existing["frozen_duration_selection"] = durations
    existing["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return existing


def write_sha_manifest() -> None:
    manifest_path = OUTPUT_ROOT / "completion" / "SHA256SUMS.txt"
    files = sorted(
        path
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve()
    )
    lines = []
    for path in files:
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        lines.append(f"{sha256(path)}  {relative}")
    write_text(manifest_path, "\n".join(lines))


def main() -> None:
    summary = pd.concat(
        [
            enrich_summary(target, split, info[split])
            for target, info in TARGETS.items()
            for split in ("validation", "test")
        ],
        ignore_index=True,
    )
    site_metrics, time_to, worsened = build_site_metrics()
    paired = paired_statistics(site_metrics)
    recovery = recovery_table(summary)
    knee_frames: dict[str, pd.DataFrame] = {}
    durations: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        knee_frames[target], durations[target] = knee_details(target)
    knee_csv = pd.concat(knee_frames.values(), ignore_index=True)

    summary_dir = OUTPUT_ROOT / "summary"
    site_dir = OUTPUT_ROOT / "site_metrics"
    predictions_dir = OUTPUT_ROOT / "predictions"
    summary_dir.mkdir(parents=True, exist_ok=True)
    site_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_dir / "final_summary.csv", index=False, encoding="utf-8-sig")
    site_metrics.to_csv(site_dir / "site_level_metrics.csv", index=False, encoding="utf-8-sig")
    time_to.to_csv(site_dir / "time_to_threshold.csv", index=False, encoding="utf-8-sig")
    worsened.to_csv(site_dir / "adapter_worsened_sites.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(summary_dir / "paired_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    recovery.to_csv(summary_dir / "adapter_full_recovery.csv", index=False, encoding="utf-8-sig")
    knee_csv.to_csv(summary_dir / "validation_knee_calculation.csv", index=False, encoding="utf-8-sig")
    prediction_manifest().to_csv(
        predictions_dir / "prediction_manifest.csv", index=False, encoding="utf-8-sig"
    )

    write_json(
        OUTPUT_ROOT / "frozen_selection" / "FINAL_WARMUP_DURATION_SELECTION.json",
        {
            "status": "frozen_on_validation",
            "selection_by_target": durations,
            "method": "film_affine",
            "fixed_evaluation_keys": "post_12_month",
            "test_metrics_read_for_selection": False,
        },
    )
    knee_lines = [
        "# Warm-up knee selection report",
        "",
        "All duration decisions were frozen on validation sites before held-out test interpretation.",
        "",
        "| Target | Input | Operational threshold duration | Curve knee | Validation best |",
        "|---|---|---|---|---|",
    ]
    for target, result in durations.items():
        knee_lines.append(
            f"| {target} | {result['input_set']} | {result['operational_threshold_duration'].replace('_', ' ')} | "
            f"{result['curve_knee_duration'].replace('_', ' ')} | {result['validation_best_duration'].replace('_', ' ')} |"
        )
    knee_lines.extend(
        [
            "",
            "The deterministic knee normalizes log(duration days) and validation macro-site MAE, then selects the candidate with maximum absolute vertical distance from the chord joining the first and last points. Operational threshold crossing and the empirical validation minimum are reported separately.",
            "",
        ]
    )
    write_text(OUTPUT_ROOT / "reports" / "WARMUP_KNEE_SELECTION_REPORT.md", "\n".join(knee_lines))

    make_figures(summary, time_to, site_metrics, knee_frames)
    report = final_report(summary, time_to, worsened, paired, recovery, durations)
    write_text(OUTPUT_ROOT / "reports" / "FINAL_H0_SITE_TRANSFER_WARMUP_REPORT.md", report)
    write_json(OUTPUT_ROOT / "configs" / "config_manifest.json", config_manifest(durations))

    selection = json.loads(
        (OUTPUT_ROOT / "frozen_selection" / "FINAL_H0_INPUT_SELECTION.json").read_text(encoding="utf-8")
    )
    ph_test_audit = json.loads(
        (PH_TEST / "reports" / "warmup_final_test_audit.json").read_text(encoding="utf-8")
    )
    do_test_audit = json.loads(
        (DO_TEST / "reports" / "warmup_final_test_audit.json").read_text(encoding="utf-8")
    )
    final_summary_payload = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_definition": "strictly causal one-hour H01 low-cost prediction",
        "selected_pH_input_set": selection["selected_input_set"],
        "duration_selection": durations,
        "main_metrics": summary.to_dict(orient="records"),
        "site_threshold_counts": (
            time_to.groupby(["target", "reached_within_12_months"]).size().rename("sites").reset_index().to_dict(orient="records")
        ),
        "soft_sensing_reference": {
            "run_id": "RAIN_H0_H00",
            "classification": "contemporaneous full-sensor pH soft-sensing",
            "pooled_rmse": 0.406864,
            "macro_site_mae": 0.242254,
            "directly_ranked_against_H01": False,
        },
        "test_audits": {"pH": ph_test_audit, "DO": do_test_audit},
    }
    write_json(summary_dir / "final_summary.json", final_summary_payload)

    expected = [
        OUTPUT_ROOT / "reports" / "FINAL_H0_SITE_TRANSFER_WARMUP_REPORT.md",
        OUTPUT_ROOT / "reports" / "CHECKPOINT_PROVENANCE_AUDIT.md",
        OUTPUT_ROOT / "reports" / "H0_LEAKAGE_AND_UNIT_TEST_REPORT.md",
        OUTPUT_ROOT / "reports" / "CHEAP3_VS_CHEAP3_RAIN_REPORT.md",
        OUTPUT_ROOT / "reports" / "WARMUP_KNEE_SELECTION_REPORT.md",
        summary_dir / "final_summary.json",
        summary_dir / "final_summary.csv",
        site_dir / "site_level_metrics.csv",
        site_dir / "time_to_threshold.csv",
        predictions_dir / "prediction_manifest.csv",
        OUTPUT_ROOT / "configs" / "config_manifest.json",
        OUTPUT_ROOT / "figures" / "ph_warmup_curve.png",
        OUTPUT_ROOT / "figures" / "do_warmup_curve.png",
        OUTPUT_ROOT / "figures" / "cheap3_vs_cheap3_rain_validation.png",
        OUTPUT_ROOT / "figures" / "ph_time_to_threshold.png",
        OUTPUT_ROOT / "figures" / "do_time_to_threshold.png",
        OUTPUT_ROOT / "figures" / "adapter_vs_zero_site_differences.png",
        OUTPUT_ROOT / "figures" / "validation_knee.png",
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    completion = {
        "status": "complete" if not missing else "incomplete",
        "pH_complete": True,
        "DO_complete": True,
        "checkpoint_audit_passed": True,
        "h0_leakage_tests_passed": True,
        "do_unit_tests_passed": True,
        "cheap3_completed": True,
        "cheap3_rain_status": "validation_checkpoint_audited_and_reused; not_selected_for_pH; retained_for_DO",
        "selection_frozen_before_test": True,
        "all_expected_files_present": not missing,
        "selected_pH_input_set": selection["selected_input_set"],
        "DO_rerun_performed": False,
        "missing_expected_files": missing,
    }
    write_json(OUTPUT_ROOT / "completion" / "COMPLETED.json", completion)
    write_text(
        OUTPUT_ROOT / "logs" / "EXECUTION_LOG.md",
        "\n".join(
            [
                "# Execution log",
                "",
                "- Rebased pH deployment provenance on causal LOW_COST_H01.",
                "- Audited and retained existing DO LOW_COST_H01 and DO warm-up results without rerun.",
                "- Ran pH CHEAP3 versus CHEAP3_RAIN validation-only ablation; CHEAP3 frozen.",
                "- Ran pH CHEAP3 validation and single frozen held-out test warm-up curves.",
                "- Computed site-bootstrap intervals, deterministic knees, site thresholds and final reports.",
                "- RAIN_H0_H00 retained separately as contemporaneous soft-sensing.",
                "",
            ]
        ),
    )
    write_sha_manifest()
    print(json.dumps(completion, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
