"""Freeze operational threshold, deterministic knee, and validation-best duration."""

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
VALIDATION_ROOT = (
    OUTPUT_ROOT / "checkpoints" / "ph_cheap3_warmup_validation_v1" / "full"
)
DAYS = {
    "2_weeks": 14.0,
    "1_month": 30.4375,
    "2_months": 60.875,
    "3_months": 91.3125,
    "6_months": 182.625,
    "9_months": 273.9375,
    "12_months": 365.25,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    source = VALIDATION_ROOT / "reports" / "warmup_curve_summary.csv"
    curve = pd.read_csv(source)
    film = curve.loc[curve["strategy"].eq("film_affine")].sort_values(
        "warmup_order"
    ).copy()
    film["duration_days"] = film["warmup_label"].map(DAYS)
    x = np.log(film["duration_days"].to_numpy(dtype=np.float64))
    y = film["macro_site_mae"].to_numpy(dtype=np.float64)
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    chord = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
    vertical_distance = np.abs(chord - y_norm)
    knee_index = int(np.argmax(vertical_distance))
    film["log_duration_normalized"] = x_norm
    film["mae_normalized"] = y_norm
    film["endpoint_chord_normalized"] = chord
    film["vertical_distance"] = vertical_distance
    knee_label = str(film.iloc[knee_index]["warmup_label"])
    best_row = film.loc[film["macro_site_mae"].idxmin()]
    threshold_rows = film.loc[film["macro_site_mae"].le(0.2)]
    threshold_label = (
        str(threshold_rows.iloc[0]["warmup_label"])
        if len(threshold_rows)
        else "not_reached_within_12_months"
    )
    details_path = OUTPUT_ROOT / "summary" / "validation_knee_calculation.csv"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    film.to_csv(details_path, index=False, encoding="utf-8-sig")

    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    labels = film["warmup_label"].str.replace("_", " ")
    axis.plot(x_norm, y_norm, "o-", linewidth=2.2, label="Observed validation curve")
    axis.plot(x_norm, chord, "--", linewidth=1.5, label="Endpoint chord")
    axis.vlines(
        x_norm[knee_index],
        y_norm[knee_index],
        chord[knee_index],
        color="#D36B27",
        linewidth=2.2,
        label=f"Maximum vertical distance: {knee_label.replace('_', ' ')}",
    )
    axis.scatter(
        [x_norm[knee_index]], [y_norm[knee_index]], s=70, color="#D36B27", zorder=4
    )
    axis.set_xticks(x_norm, labels, rotation=20)
    axis.set_xlabel("Warm-up duration (log days, normalized)")
    axis.set_ylabel("Validation macro-site MAE (normalized)")
    axis.set_title("pH CHEAP3 validation knee — 46 sites, fixed post-12-month keys")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure_path = OUTPUT_ROOT / "figures" / "validation_knee.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    payload = {
        "status": "frozen_on_validation",
        "target": "pH",
        "input_set": "CHEAP3",
        "primary_method": "film_affine",
        "validation_sites": int(film["sites"].iloc[0]),
        "fixed_evaluation_keys": "post_12_month",
        "operational_threshold": {"metric": "macro_site_mae", "value": 0.2},
        "operational_threshold_duration": threshold_label,
        "curve_knee_duration": knee_label,
        "validation_best_duration": str(best_row["warmup_label"]),
        "validation_best_macro_site_mae": float(best_row["macro_site_mae"]),
        "knee_method": (
            "Normalize log(duration_days) and macro-site MAE to [0,1], draw the "
            "line through the first and last candidate points, and select the "
            "candidate with maximum absolute vertical distance from that line."
        ),
        "knee_vertical_distance": float(vertical_distance[knee_index]),
        "duration_days": DAYS,
        "source_curve_path": str(source.resolve()),
        "source_curve_sha256": sha256(source),
        "calculation_csv_path": str(details_path.resolve()),
        "figure_path": str(figure_path.resolve()),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_metrics_read": False,
    }
    write_json(
        OUTPUT_ROOT / "frozen_selection" / "WARMUP_DURATION_SELECTION.json",
        payload,
    )
    lines = [
        "# Warm-up knee and duration selection",
        "",
        "This decision was frozen on 46 validation sites before test evaluation. All durations use identical post-12-month evaluation keys.",
        "",
        "| Candidate | Adapter macro-site MAE | Normalized vertical distance |",
        "|---|---:|---:|",
    ]
    for row in film.itertuples(index=False):
        lines.append(
            f"| {str(row.warmup_label).replace('_', ' ')} | "
            f"{row.macro_site_mae:.6f} | {row.vertical_distance:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Operational threshold duration: **{threshold_label.replace('_', ' ')}**.",
            f"- Deterministic curve knee: **{knee_label.replace('_', ' ')}**.",
            f"- Validation-best candidate: **{str(best_row['warmup_label']).replace('_', ' ')}** "
            f"(macro-site MAE {float(best_row['macro_site_mae']):.6f}).",
            "",
            "The threshold and knee answer different questions: the threshold tests whether the predefined operational target is achieved, while the knee marks diminishing returns. They are therefore reported separately and are not forced into one duration.",
            "",
        ]
    )
    write_text(
        OUTPUT_ROOT / "reports" / "WARMUP_KNEE_SELECTION_REPORT.md",
        "\n".join(lines),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
