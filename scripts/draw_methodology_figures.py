"""Draw the two frozen methodology figures and their source manifest.

Run from any directory with:
    .venv/Scripts/python.exe scripts/figures/draw_methodology_figures.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "methodology"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
NAVY = "#12355B"
BLUE = "#2A6F97"
LIGHT_BLUE = "#A9D6E5"
TEAL = "#2A9D8F"
ORANGE = "#F4A261"
RED = "#E76F51"
GRAY = "#6B7280"
LIGHT_GRAY = "#E5E7EB"
PALE_GRAY = "#F6F7F9"
WHITE = "#FFFFFF"
BLACK = "#111827"

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.7,
        "savefig.facecolor": WHITE,
        "figure.facecolor": WHITE,
    }
)


def canvas(height_mm: float):
    fig, ax = plt.subplots(figsize=(183 * MM, height_mm * MM))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def text(ax, x, y, s, size=6.0, color=BLACK, weight="normal", ha="left", va="center", **kwargs):
    size = max(5.0, float(size))
    return ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color,
        weight=weight,
        ha=ha,
        va=va,
        linespacing=1.12,
        clip_on=False,
        **kwargs,
    )


def rounded(ax, x, y, w, h, fc=WHITE, ec=LIGHT_GRAY, lw=0.65, radius=0.012, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=0.65, style="-|>", mutation=7, z=3):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def panel(ax, x, y, w, h, label, title):
    rounded(ax, x, y, w, h, fc=WHITE, ec="#CBD5E1", lw=0.8, radius=0.015)
    text(ax, x + 0.012, y + h - 0.028, label, size=8, color=NAVY, weight="bold", va="top")
    text(ax, x + 0.041, y + h - 0.029, title, size=7.0, color=NAVY, weight="bold", va="top")


def chip(ax, x, y, w, h, label, fc=LIGHT_BLUE, color=NAVY, ec=None, size=5.6, weight="bold"):
    rounded(ax, x, y, w, h, fc=fc, ec=ec or fc, lw=0.5, radius=0.010, z=2)
    text(ax, x + w / 2, y + h / 2, label, size=size, color=color, weight=weight, ha="center")


def save_figure(fig, stem):
    metadata = {"Creator": "draw_methodology_figures.py", "Title": stem}
    fig.savefig(OUT / f"{stem}.pdf", metadata=metadata)
    fig.savefig(OUT / f"{stem}.svg", metadata=metadata)
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    plt.close(fig)


def read_json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def read_csv(relative_path: str):
    with (ROOT / relative_path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_local_sources():
    """Fail before drawing if a displayed identity has drifted locally."""
    cohort = read_csv(
        "analysis_outputs/ph_common1000_teacher_archive_features_do_converted_v1/"
        "full/reports/primary_archive_experiment_cohort.csv"
    )
    assert len(cohort) == 910
    assert Counter(row["split"] for row in cohort) == Counter({"train": 638, "validation": 137, "test": 135})
    assert Counter(row["network"] for row in cohort) == Counter(
        {"EA 临时站点": 696, "USGS（8位数字站点）": 158, "N 系列站点": 56}
    )

    h00 = read_json("analysis_outputs/ph_common1000_lstm_nohistory_h0_rain_final_v1/full/reports/effective_config.json")
    assert (h00["target_code"], h00["horizon_hours"], h00["sequence_length_hours"]) == ("00400", 0, 48)
    assert h00["include_target_history"] is False and h00["models"][0]["model_id"] == "LSTM_RAIN_H0"

    historical = read_json(
        "analysis_outputs/ph_common1000_lstm_film_final_rain_ablation_teacher_aligned_v1/"
        "full/reports/effective_config.json"
    )
    assert (historical["target_code"], historical["horizon_hours"], historical["sequence_length_hours"]) == (
        "00400",
        1,
        48,
    )

    low_ph = read_json("analysis_outputs/ph_common1000_lstm_film_ph_sensor_ablation_v1/full/reports/effective_config.json")
    low_do = read_json(
        "analysis_outputs/do_common1000_lstm_film_sensor_ablation_mg_l_v1/full/reports/effective_config.json"
    )
    assert low_ph["models"][1]["model_id"] == "PH_LOW_COST" and low_ph["models"][1]["target_history"] is False
    assert low_do["models"][1]["model_id"] == "DO_LOW_COST" and low_do["models"][1]["target_history"] is False
    assert (low_do["target_code"], low_do["target_unit"], low_do["sequence_length_hours"], low_do["horizon_hours"]) == (
        "00300",
        "mg/L",
        48,
        1,
    )

    checkpoint_paths = [
        "analysis_outputs/ph_common1000_lstm_nohistory_h0_rain_final_v1/full/models/ph_LSTM_RAIN_H0_H00.pt",
        "analysis_outputs/ph_common1000_lstm_film_final_rain_ablation_teacher_aligned_v1/full/models/ph_LSTM_RAIN_FINAL_H01.pt",
        "analysis_outputs/ph_common1000_lstm_film_ph_sensor_ablation_v1/full/models/ph_LSTM_LOW_COST_H01.pt",
        "analysis_outputs/do_common1000_lstm_film_sensor_ablation_mg_l_v1/full/models/do_LSTM_LOW_COST_H01.pt",
    ]
    assert all((ROOT / path).is_file() for path in checkpoint_paths)

    conformal = read_csv("analysis_outputs/common1000_lowcost_conformal_v1/reports/conformal_site_split.csv")
    for target in ("pH", "DO"):
        counts = Counter(row["conformal_split"] for row in conformal if row["target"] == target)
        assert counts == Counter({"calibration": 45, "evaluation": 90})

    warmup = read_json(
        "analysis_outputs/h0_site_transfer_warmup_final_audit_v1/checkpoints/"
        "ph_cheap3_warmup_validation_v1/full/reports/frozen_adaptation_protocol.json"
    )
    assert warmup["warmup_labels"] == [
        "2_weeks",
        "1_month",
        "2_months",
        "3_months",
        "6_months",
        "9_months",
        "12_months",
    ]
    assert warmup["fixed_evaluation_after_months"] == 12

    temporal = (
        ROOT
        / "analysis_outputs/ph_temporal_holdout_site_analysis_v1/reports/"
        "FINAL_PH_TEMPORAL_HOLDOUT_AND_SITE_FAILURE_REPORT.md"
    ).read_text(encoding="utf-8")
    assert "History was `<= T_i-90d`" in temporal
    assert "support was `(T_i-90d, T_i-60d]`" in temporal
    assert "test was `(T_i-60d, T_i]`" in temporal


def draw_figure_1():
    fig, ax = canvas(120)

    # Panel a - sources and cohorts
    x, y, w, h = 0.014, 0.135, 0.304, 0.842
    panel(ax, x, y, w, h, "a", "Sources and fixed cohorts")
    chip(ax, x + 0.020, 0.851, 0.075, 0.044, "EA", fc=LIGHT_BLUE)
    chip(ax, x + 0.114, 0.851, 0.075, 0.044, "USGS", fc="#CFE8F3")
    chip(ax, x + 0.208, 0.851, 0.075, 0.044, "NEON", fc="#D8F0EC", color="#12665D")

    rounded(ax, x + 0.020, 0.666, 0.264, 0.147, fc=PALE_GRAY, ec=LIGHT_GRAY)
    text(ax, x + 0.034, 0.788, "Primary pH cohort", size=6.3, color=NAVY, weight="bold", va="top")
    text(ax, x + 0.034, 0.743, "910 sites", size=7.0, color=NAVY, weight="bold")
    text(ax, x + 0.034, 0.704, "EA 696  |  USGS 158  |  NEON 56", size=5.5, color=GRAY)
    bx, by, bw, bh = x + 0.034, 0.680, 0.235, 0.012
    ax.add_patch(Rectangle((bx, by), bw * 696 / 910, bh, fc=BLUE, ec="none"))
    ax.add_patch(Rectangle((bx + bw * 696 / 910, by), bw * 158 / 910, bh, fc=ORANGE, ec="none"))
    ax.add_patch(Rectangle((bx + bw * 854 / 910, by), bw * 56 / 910, bh, fc=TEAL, ec="none"))

    chip(ax, x + 0.020, 0.590, 0.078, 0.052, "638 train", fc="#DCEAF4", color=NAVY, size=5.4)
    chip(ax, x + 0.108, 0.590, 0.078, 0.052, "137 validation", fc="#FBE5D2", color="#8A4A14", size=5.15)
    chip(ax, x + 0.196, 0.590, 0.088, 0.052, "135 test", fc="#DDF1ED", color="#12665D", size=5.4)

    rounded(ax, x + 0.020, 0.448, 0.264, 0.110, fc=WHITE, ec=LIGHT_GRAY)
    ax.add_patch(Circle((x + 0.056, 0.505), 0.022, fc=BLUE, ec="none"))
    text(ax, x + 0.056, 0.505, "pH", size=6.2, color=WHITE, weight="bold", ha="center")
    text(ax, x + 0.086, 0.514, "pH  |  parameter 00400", size=5.7, color=BLACK, weight="bold")
    ax.add_patch(Circle((x + 0.056, 0.469), 0.022, fc=TEAL, ec="none"))
    text(ax, x + 0.056, 0.469, "DO", size=5.8, color=WHITE, weight="bold", ha="center")
    text(ax, x + 0.086, 0.478, "dissolved oxygen  |  00300", size=5.45, color=BLACK, weight="bold")
    text(ax, x + 0.086, 0.459, "final low-cost test: 135 sites", size=5.15, color=GRAY)

    rounded(ax, x + 0.020, 0.194, 0.264, 0.221, fc="#F8FBFD", ec=LIGHT_GRAY)
    text(ax, x + 0.034, 0.390, "Verified dynamic input families", size=6.1, color=NAVY, weight="bold", va="top")
    rows = [
        (BLUE, "Water sensors", "temperature | conductance | turbidity | non-target DO"),
        (TEAL, "Rainfall", "current hour | 6 h sum | 48 h API"),
        (ORANGE, "Time", "hour | day | month cyclic encodings"),
    ]
    for i, (c, head, detail) in enumerate(rows):
        yy = 0.342 - i * 0.057
        ax.add_patch(Circle((x + 0.045, yy), 0.009, fc=c, ec="none"))
        text(ax, x + 0.065, yy + 0.010, head, size=5.55, color=BLACK, weight="bold")
        text(ax, x + 0.065, yy - 0.012, detail, size=5.0, color=GRAY)

    # Panel b - processing and cohort construction
    x, y, w, h = 0.337, 0.135, 0.315, 0.842
    panel(ax, x, y, w, h, "b", "Frozen processing")
    steps = [
        ("Native records", "station identity | source units | source time"),
        ("Identity, units and UTC", "source-aware normalization"),
        ("Sensor QC + hourly alignment", "causal handling on the hourly axis"),
        ("Fixed 48 h windows", "dynamic missingness masks retained"),
        ("Train-only preprocessing", "scalers and transforms frozen"),
        ("Fixed site-disjoint cohorts", "task-specific keys and denominators"),
    ]
    box_x, box_w, box_h = x + 0.035, w - 0.070, 0.081
    top = 0.858
    for i, (head, detail) in enumerate(steps):
        yy = top - i * 0.108
        fc = "#F8FBFD" if i < 3 else "#F7FBFA"
        ec = LIGHT_BLUE if i < 3 else "#B9E0D9"
        rounded(ax, box_x, yy - box_h, box_w, box_h, fc=fc, ec=ec, lw=0.65)
        ax.add_patch(Circle((box_x + 0.023, yy - box_h / 2), 0.013, fc=BLUE if i < 3 else TEAL, ec="none"))
        text(ax, box_x + 0.049, yy - 0.030, head, size=5.75, color=NAVY, weight="bold")
        text(ax, box_x + 0.049, yy - 0.055, detail, size=4.9, color=GRAY)
        if i < len(steps) - 1:
            arrow(ax, x + w / 2, yy - box_h - 0.005, x + w / 2, yy - 0.103, color=GRAY, lw=0.55)
    rounded(ax, x + 0.035, 0.151, w - 0.070, 0.060, fc="#FFF5EC", ec=ORANGE, lw=0.7)
    text(
        ax,
        x + w / 2,
        0.181,
        "Sensor QC is upstream of, and distinct from,\nlater residual/event analysis.",
        size=5.15,
        color="#8A4A14",
        weight="bold",
        ha="center",
    )

    # Panel c - analytical branches
    x, y, w, h = 0.671, 0.135, 0.315, 0.842
    panel(ax, x, y, w, h, "c", "Branches and outputs")
    text(ax, x + 0.025, 0.895, "Frozen LSTM-FiLM reference family", size=5.7, color=GRAY, weight="bold")
    branches = [
        (BLUE, "Contemporaneous cross-variable H00", "ph_LSTM_RAIN_H0_H00", "pH(t); no pH history | 135 test sites"),
        (ORANGE, "Historical-target one-hour H01", "ph_LSTM_RAIN_FINAL_H01", "pH(t+1); earlier pH allowed | 135 test sites"),
        (TEAL, "Target-history-free low-cost H01", "ph/do_LSTM_LOW_COST_H01", "pH or DO(t+1); target history excluded | 135/target"),
    ]
    for i, (c, head, model, detail) in enumerate(branches):
        yy = 0.850 - i * 0.155
        rounded(ax, x + 0.025, yy - 0.112, w - 0.050, 0.112, fc=WHITE, ec=c, lw=0.75)
        ax.add_patch(Rectangle((x + 0.025, yy - 0.112), 0.010, 0.112, fc=c, ec="none"))
        text(ax, x + 0.050, yy - 0.026, head, size=5.55, color=NAVY, weight="bold")
        text(ax, x + 0.050, yy - 0.055, model, size=4.9, color=c, weight="bold")
        text(ax, x + 0.050, yy - 0.086, detail, size=4.75, color=GRAY)
    arrow(ax, x + w / 2, 0.430, x + w / 2, 0.390, color=GRAY, lw=0.65)
    rounded(ax, x + 0.025, 0.322, w - 0.050, 0.065, fc="#F8FBFD", ec=LIGHT_BLUE)
    text(ax, x + 0.050, 0.354, "Raw estimate", size=5.65, color=NAVY, weight="bold")
    arrow(ax, x + 0.132, 0.354, x + 0.168, 0.354, color=GRAY, lw=0.55)
    chip(ax, x + 0.173, 0.334, 0.094, 0.040, "30-d offset*", fc="#FFF0E2", color="#8A4A14", ec=ORANGE, size=5.0)
    text(ax, x + 0.025, 0.297, "*Only when local target support is available.", size=4.75, color="#8A4A14", weight="bold")
    text(ax, x + 0.025, 0.266, "H00 support-eligible: 128 validation | 126 test", size=4.65, color=GRAY)

    rounded(ax, x + 0.025, 0.180, w - 0.050, 0.064, fc="#F7FBFA", ec="#B9E0D9")
    text(ax, x + 0.040, 0.220, "Outputs", size=5.4, color=NAVY, weight="bold")
    text(ax, x + 0.040, 0.196, "accuracy | residual/event | conformal uncertainty", size=4.8, color=GRAY)
    text(ax, x + w - 0.035, 0.155, "Conformal per target: 45 calibration | 90 evaluation", size=4.55, color=GRAY, ha="right")

    # Subordinate context strip
    rounded(ax, 0.014, 0.025, 0.972, 0.082, fc=PALE_GRAY, ec=LIGHT_GRAY, lw=0.7)
    text(ax, 0.031, 0.080, "Bounded context tests", size=5.7, color=NAVY, weight="bold")
    text(
        ax,
        0.210,
        0.080,
        "local/static  |  upstream catchment  |  solar geometry  |  rainfall/API  |  signed discharge*",
        size=5.15,
        color=BLACK,
    )
    text(
        ax,
        0.031,
        0.047,
        "Separate context only; no outcome claim.  *Flow: 61 sites / 39 groups, training-only available-case.",
        size=4.75,
        color=GRAY,
    )

    save_figure(fig, "Figure_1_study_design")


def input_dot(ax, x, y, color=BLUE, filled=True, radius=0.008):
    ax.add_patch(Circle((x, y), radius, fc=color if filled else WHITE, ec=color, lw=0.7, zorder=5))


def diamond(ax, x, y, color=RED, r=0.012):
    ax.add_patch(Polygon([(x, y + r), (x + r, y), (x, y - r), (x - r, y)], closed=True, fc=color, ec=color, lw=0.5, zorder=6))


def timeline_card(ax, x, y, w, h, title, subtitle, kind):
    rounded(ax, x, y, w, h, fc=WHITE, ec=LIGHT_GRAY, lw=0.65)
    text(ax, x + 0.015, y + h - 0.026, title, size=5.75, color=NAVY, weight="bold", va="top")
    text(ax, x + 0.015, y + h - 0.052, subtitle, size=4.8, color=GRAY, va="top")
    x0, x1 = x + 0.030, x + w - 0.030
    yy = y + 0.150
    cutoff = x0 + 0.74 * (x1 - x0)
    target_x = cutoff if kind == "h00" else x0 + 0.91 * (x1 - x0)

    ax.add_patch(Rectangle((x0, yy - 0.025), cutoff - x0, 0.050, fc="#E8F3F8", ec="none", zorder=1))
    ax.plot([x0, x1], [yy, yy], color=GRAY, lw=0.7, zorder=2)
    ax.plot([cutoff, cutoff], [yy - 0.070, yy + 0.095], color=NAVY, lw=0.7, ls=(0, (2, 2)), zorder=2)
    text(ax, cutoff, yy - 0.060, "cutoff t", size=4.4, color=NAVY, ha="center")
    text(ax, x0, yy + 0.050, "48 h input", size=4.6, color=BLUE, weight="bold")
    for frac in [0.08, 0.23, 0.38, 0.53, 0.68, 0.74]:
        input_dot(ax, x0 + frac * (x1 - x0), yy + 0.020, color=BLUE, filled=True, radius=0.006)

    if kind == "h00":
        for frac in [0.15, 0.36, 0.58, 0.72]:
            input_dot(ax, x0 + frac * (x1 - x0), yy - 0.032, color="#A7AFBC", filled=False, radius=0.006)
        diamond(ax, target_x, yy + 0.072, RED, r=0.010)
        text(ax, target_x, yy + 0.104, "pH(t)", size=4.8, color=RED, weight="bold", ha="center")
        text(ax, x + 0.015, y + 0.058, "Permitted: non-target sensors through t", size=4.75, color=BLACK)
        text(ax, x + 0.015, y + 0.029, "Excluded: current and historical pH", size=4.75, color=GRAY)
    elif kind == "historical":
        for frac in [0.15, 0.36, 0.58, 0.72]:
            input_dot(ax, x0 + frac * (x1 - x0), yy - 0.032, color=ORANGE, filled=True, radius=0.006)
        diamond(ax, target_x, yy + 0.020, RED, r=0.010)
        text(ax, target_x, yy + 0.054, "pH(t+1)", size=4.8, color=RED, weight="bold", ha="center")
        text(ax, x + 0.015, y + 0.058, "Permitted: sensors + earlier pH", size=4.75, color=BLACK)
        text(ax, x + 0.015, y + 0.029, "Forecast target is beyond the cutoff", size=4.75, color=GRAY)
    else:
        for frac in [0.15, 0.36, 0.58, 0.72]:
            input_dot(ax, x0 + frac * (x1 - x0), yy - 0.032, color="#A7AFBC", filled=False, radius=0.006)
        diamond(ax, target_x, yy + 0.020, RED, r=0.010)
        text(ax, target_x, yy + 0.054, "pH/DO(t+1)", size=4.7, color=RED, weight="bold", ha="center")
        text(ax, x + 0.015, y + 0.058, "Permitted: low-cost sensors through t", size=4.75, color=BLACK)
        text(ax, x + 0.015, y + 0.029, "Excluded: pH/DO target history", size=4.75, color=GRAY)


def draw_figure_2():
    fig, ax = canvas(130)

    # Panel a - inference information
    panel(ax, 0.018, 0.545, 0.964, 0.432, "a", "Information available at inference")
    input_dot(ax, 0.670, 0.934, BLUE, True, 0.006)
    text(ax, 0.682, 0.934, "permitted input", size=4.6, color=GRAY)
    input_dot(ax, 0.785, 0.934, "#A7AFBC", False, 0.006)
    text(ax, 0.797, 0.934, "target excluded", size=4.6, color=GRAY)
    diamond(ax, 0.913, 0.934, RED, 0.008)
    text(ax, 0.926, 0.934, "outcome", size=4.6, color=GRAY)

    timeline_card(ax, 0.036, 0.588, 0.292, 0.302, "H00 contemporaneous", "cross-variable pH estimate", "h00")
    timeline_card(ax, 0.354, 0.588, 0.292, 0.302, "Historical-target H01", "one-hour pH forecast", "historical")
    timeline_card(ax, 0.672, 0.588, 0.292, 0.302, "Low-cost H01", "one-hour pH or DO forecast", "lowcost")

    # Panel b - evaluation designs
    panel(ax, 0.018, 0.035, 0.964, 0.472, "b", "Evaluation designs and deployment questions")
    card_y, card_h, card_w = 0.075, 0.365, 0.292
    xs = [0.036, 0.354, 0.672]
    titles = ["Complete-site hold-out", "New-site support / warm-up", "Familiar-site temporal hold-out"]
    questions = ["Transfer to a new site?", "How much local target support is needed?", "Generalisation to a later period?"]
    for xx, tt in zip(xs, titles):
        rounded(ax, xx, card_y, card_w, card_h, fc=WHITE, ec=LIGHT_GRAY, lw=0.65)
        text(ax, xx + 0.015, card_y + card_h - 0.026, tt, size=5.65, color=NAVY, weight="bold", va="top")

    # Complete-site hold-out
    xx = xs[0]
    text(ax, xx + 0.018, 0.350, "All records from each site stay in one split", size=4.65, color=GRAY)
    groups = [
        (xx + 0.020, 0.175, 0.075, 0.125, BLUE, "638", "train"),
        (xx + 0.108, 0.175, 0.075, 0.125, ORANGE, "137", "validation"),
        (xx + 0.196, 0.175, 0.075, 0.125, TEAL, "135", "test"),
    ]
    for gx, gy, gw, gh, c, n, lab in groups:
        for r in range(3):
            for col in range(3):
                ax.add_patch(Circle((gx + 0.014 + col * 0.020, gy + 0.075 + r * 0.017), 0.0055, fc=c, ec="none"))
        text(ax, gx + gw / 2, gy + 0.048, n, size=6.5, color=c, weight="bold", ha="center")
        text(ax, gx + gw / 2, gy + 0.022, lab, size=4.65, color=GRAY, ha="center")
    text(ax, xx + card_w / 2, 0.113, questions[0], size=5.5, color=NAVY, weight="bold", ha="center")

    # New-site support / warm-up
    xx = xs[1]
    text(ax, xx + 0.018, 0.350, "Local labels precede a fixed, non-overlapping query", size=4.65, color=GRAY)
    x0, x1, yy = xx + 0.025, xx + card_w - 0.025, 0.245
    query_start = x0 + 0.70 * (x1 - x0)
    ax.add_patch(Rectangle((x0, yy - 0.030), query_start - x0 - 0.012, 0.060, fc="#DDF1ED", ec="none"))
    ax.add_patch(Rectangle((query_start, yy - 0.030), x1 - query_start, 0.060, fc="#E8F3F8", ec="none"))
    ax.plot([x0, x1], [yy, yy], color=GRAY, lw=0.7)
    text(ax, x0 + 0.5 * (query_start - x0), yy + 0.052, "local support labels", size=4.75, color="#12665D", weight="bold", ha="center")
    text(ax, query_start + 0.5 * (x1 - query_start), yy + 0.052, "fixed query", size=4.75, color=BLUE, weight="bold", ha="center")
    ax.plot([query_start - 0.006, query_start - 0.006], [yy - 0.050, yy + 0.050], color=NAVY, lw=0.65, ls=(0, (2, 2)))
    durations = [(0.07, "2wk"), (0.16, "1"), (0.27, "2"), (0.38, "3"), (0.52, "6"), (0.63, "9"), (0.69, "12*")]
    for frac, lab in durations:
        xt = x0 + frac * (x1 - x0)
        ax.plot([xt, xt], [yy - 0.037, yy - 0.027], color=TEAL, lw=0.55)
        text(ax, xt, yy - 0.057, lab, size=4.0, color=GRAY, ha="center")
    text(ax, x0, 0.166, "1-12 labels are months", size=4.1, color=GRAY)
    text(ax, xx + 0.018, 0.143, "*12-month test estimate is descriptive", size=4.2, color="#8A4A14")
    text(ax, xx + card_w / 2, 0.102, questions[1], size=5.15, color=NAVY, weight="bold", ha="center")

    # Familiar-site temporal hold-out
    xx = xs[2]
    text(ax, xx + 0.018, 0.350, "Chronological blocks; final outcomes stay sealed", size=4.65, color=GRAY)
    x0, x1, yy = xx + 0.025, xx + card_w - 0.025, 0.245
    b1 = x0 + 0.52 * (x1 - x0)
    b2 = x0 + 0.69 * (x1 - x0)
    ax.add_patch(Rectangle((x0, yy - 0.030), b1 - x0, 0.060, fc="#E8F3F8", ec="none"))
    ax.add_patch(Rectangle((b1, yy - 0.030), b2 - b1, 0.060, fc="#FFF0E2", ec="none"))
    ax.add_patch(Rectangle((b2, yy - 0.030), x1 - b2, 0.060, fc="#FDE9E5", ec="none"))
    ax.plot([x0, x1], [yy, yy], color=GRAY, lw=0.7)
    for bx in [b1, b2]:
        ax.plot([bx, bx], [yy - 0.048, yy + 0.048], color=NAVY, lw=0.65, ls=(0, (2, 2)))
    text(ax, (x0 + b1) / 2, yy + 0.052, "earlier history", size=4.65, color=BLUE, weight="bold", ha="center")
    text(ax, (b1 + b2) / 2, yy + 0.078, "support", size=4.55, color="#8A4A14", weight="bold", ha="center")
    text(ax, (b2 + x1) / 2, yy + 0.052, "final hold-out", size=4.35, color=RED, weight="bold", ha="center")
    text(ax, (x0 + b1) / 2, yy - 0.058, "<= Ti-90 d", size=4.15, color=GRAY, ha="center")
    text(ax, (b1 + b2) / 2, yy - 0.058, "30 d", size=4.15, color=GRAY, ha="center")
    text(ax, (b2 + x1) / 2, yy - 0.058, "60 d", size=4.15, color=GRAY, ha="center")
    text(ax, xx + 0.018, 0.164, "Support is used only for offset / baseline", size=4.25, color=GRAY)
    text(ax, xx + card_w / 2, 0.113, questions[2], size=5.15, color=NAVY, weight="bold", ha="center")

    save_figure(fig, "Figure_2_information_design")


def write_manifest():
    rows = [
        ("Figure_1_study_design", "a", "EA 696 | USGS 158 | NEON 56; primary cohort 910", "analysis_outputs/ph_common1000_teacher_archive_features_do_converted_v1/full/reports/primary_archive_experiment_cohort.csv", "network; primary_archive_cohort; status", "Counts are within the final 910-site primary cohort; N-series records are mapped to NEON in the source-category audit."),
        ("Figure_1_study_design", "a", "EA | USGS | NEON source-category mapping", "analysis_outputs/ph_h00_upstream13_complete_case_exploratory_v1/audit/upstream13_missingness_by_site.csv", "network; source_category", "Confirms the displayed monitoring-organization names, including N-series to NEON mapping."),
        ("Figure_1_study_design", "a", "638 train | 137 validation | 135 test", "analysis_outputs/ph_common1000_teacher_archive_features_do_converted_v1/full/reports/primary_archive_experiment_cohort.csv", "split", "Fixed site-disjoint primary split."),
        ("Figure_1_study_design", "a", "pH parameter 00400", "analysis_outputs/ph_common1000_lstm_nohistory_h0_rain_final_v1/full/reports/effective_config.json", "target; target_code", "Retained pH target identity."),
        ("Figure_1_study_design", "a", "dissolved oxygen parameter 00300; mg/L final low-cost target", "analysis_outputs/do_common1000_lstm_film_sensor_ablation_mg_l_v1/full/reports/effective_config.json", "target; target_code; target_unit; target_source_multiplier", "Final low-cost DO configuration."),
        ("Figure_1_study_design", "a", "temperature | conductance | turbidity | DO | rainfall | time encodings", "current_state_inventory_complete/dataset_registry.csv", "D004 station variables and dynamic covariates", "Family-level labels only; individual branch permissions are shown in Figure 2."),
        ("Figure_1_study_design", "a", "current rainfall | 6 h sum | 48 h antecedent precipitation index", "current_state_inventory_complete/dataset_registry.csv", "D004 and D005 dynamic covariates", "Verified rainfall families available in the final prepared archive."),
        ("Figure_1_study_design", "b", "native records -> identity, units and UTC -> sensor QC and hourly alignment", "analysis_outputs/ph_common1000_teacher_archive_features_do_converted_v1/full/reports/primary_archive_experiment_cohort.csv", "native_input_unit; hourly_aggregation; rain_source_start_utc; rain_source_end_utc", "Processing stages summarized without adding a new analysis."),
        ("Figure_1_study_design", "b", "fixed 48 h windows and dynamic missingness masks", "analysis_outputs/ph_common1000_lstm_nohistory_h0_rain_final_v1/full/reports/effective_config.json", "sequence_length_hours; maximum_dynamic_missing_fraction_per_window", "The same 48 h setting is verified in the retained H01 and low-cost configurations."),
        ("Figure_1_study_design", "b", "train-only preprocessing frozen", "analysis_outputs/ph_common1000_lstm_film_final_rain_ablation_teacher_aligned_v1/full/reports/train_only_preprocessing.json", "fit split and preprocessing statistics", "Preprocessing is fit without validation or test outcomes."),
        ("Figure_1_study_design", "b", "fixed site-disjoint cohorts and task-specific keys", "current_state_inventory_complete/split_and_key_registry.csv", "K002-K010", "Registry distinguishes primary, calibration, warm-up, conformal and residual-event cohorts."),
        ("Figure_1_study_design", "b", "sensor QC distinct from residual/event analysis", "analysis_outputs/cross_variable_hot_moment_incremental_closure_v1/reports/EXISTING_ASSET_INVENTORY.md", "historical and H00 prediction/event assets", "Residual/event analysis is downstream of the frozen prediction pipeline."),
        ("Figure_1_study_design", "c", "ph_LSTM_RAIN_H0_H00; pH(t); no target history; 135 test sites", "analysis_outputs/ph_common1000_lstm_nohistory_h0_rain_final_v1/full/reports/effective_config.json", "models[0].model_id; horizon_hours; include_target_history; primary_cohort_path", "Output filename adds the target prefix and H00 suffix, verified against the checkpoint/prediction directory."),
        ("Figure_1_study_design", "c", "Frozen LSTM-FiLM reference family", "current_state_inventory_complete/model_registry.csv", "retained pH H00/H01 and low-cost model-family fields", "Family-level label; exact checkpoint names are separately verified."),
        ("Figure_1_study_design", "c", "ph_LSTM_RAIN_FINAL_H01; pH(t+1); earlier pH allowed; 135 test sites", "analysis_outputs/cross_variable_hot_moment_incremental_closure_v1/reports/CHECKPOINT_AND_TASK_PROVENANCE.md", "Historical-target feature order", "Exact retained checkpoint name is also listed in current_state_inventory_complete/checkpoint_registry.csv."),
        ("Figure_1_study_design", "c", "ph_LSTM_LOW_COST_H01 | do_LSTM_LOW_COST_H01; target history excluded; 135 test sites per target", "analysis_outputs/ph_common1000_lstm_film_ph_sensor_ablation_v1/full/reports/effective_config.json", "models[1].model_id; feature_names; target_history; horizon_hours", "DO identity and mg/L target are independently verified in the paired DO effective configuration."),
        ("Figure_1_study_design", "c", "H00 support eligible: 128 validation | 126 test; 30-day offset conditional on local target support", "current_state_inventory_complete/split_and_key_registry.csv", "K004-K005", "The offset is not shown as a universal model component."),
        ("Figure_1_study_design", "c", "accuracy | residual/event | conformal uncertainty", "current_state_inventory_complete/experiment_registry.csv", "retained analytical work packages", "Output classes only; no result, metric value or ranking is plotted."),
        ("Figure_1_study_design", "c", "Conformal per target: 45 calibration | 90 evaluation", "analysis_outputs/common1000_lowcost_conformal_v1/reports/conformal_site_split.csv", "target; conformal_split", "Counts are per target; the file has 270 rows across pH and DO."),
        ("Figure_1_study_design", "context strip", "local/static | upstream catchment | solar geometry | rainfall/API", "analysis_outputs/ph_h00_static3_additive_validation_v1/reports/FINAL_H00_STATIC3_VALIDATION_REPORT.md", "bounded H00 context work packages", "Labels indicate separate context tests only, with no result claim."),
        ("Figure_1_study_design", "context strip", "upstream catchment", "analysis_outputs/ph_h00_upstream1_v1/FINAL_REPORT.md", "frozen upstream experiment identity", "Separate bounded context test."),
        ("Figure_1_study_design", "context strip", "solar geometry", "analysis_outputs/ph_h00_solar_dynamic1_v1/FINAL_REPORT.md", "frozen solar experiment identity", "Separate bounded context test."),
        ("Figure_1_study_design", "context strip", "signed discharge; 61 sites / 39 groups; training-only available-case", "analysis_outputs/ph_h00_flow_asinh_adapter_exploratory_v1/protocol/FROZEN_FLOW_PROTOCOL.md", "cohort; groups; training-only restrictions", "Stage is not displayed and flow is not shown as a universal input."),
        ("Figure_2_information_design", "a", "H00: 48 h, horizon 0, non-target inputs through t, current/history pH excluded", "analysis_outputs/ph_common1000_lstm_nohistory_h0_rain_final_v1/full/reports/effective_config.json", "sequence_length_hours; horizon_hours; models[0].feature_names; include_target_history", "Timeline matches the retained H00 effective configuration."),
        ("Figure_2_information_design", "a", "Historical-target H01: 48 h, horizon 1, earlier pH permitted", "analysis_outputs/cross_variable_hot_moment_incremental_closure_v1/reports/CHECKPOINT_AND_TASK_PROVENANCE.md", "Historical-target feature order", "Timeline matches the historical comparator checkpoint feature order and H01 configuration."),
        ("Figure_2_information_design", "a", "Low-cost H01: 48 h, horizon 1, pH/DO target history excluded", "analysis_outputs/do_common1000_lstm_film_sensor_ablation_mg_l_v1/full/reports/effective_config.json", "sequence_length_hours; horizon_hours; models[1].feature_names; models[1].target_history", "pH is independently verified in the paired low-cost pH effective configuration."),
        ("Figure_2_information_design", "b", "Complete-site hold-out: 638 train | 137 validation | 135 test", "analysis_outputs/ph_common1000_teacher_archive_features_do_converted_v1/full/reports/primary_archive_experiment_cohort.csv", "site_id; split", "Whole sites remain within one fixed split."),
        ("Figure_2_information_design", "b", "Support durations: 2 weeks, 1, 2, 3, 6, 9 and 12 months; fixed query after 12 months", "analysis_outputs/h0_site_transfer_warmup_final_audit_v1/checkpoints/ph_cheap3_warmup_validation_v1/full/reports/frozen_adaptation_protocol.json", "warmup_labels; fixed_evaluation_after_months", "Support labels precede all fixed evaluation keys."),
        ("Figure_2_information_design", "b", "12-month test estimate is descriptive", "analysis_outputs/h0_site_transfer_warmup_final_audit_v1/reports/FINAL_H0_SITE_TRANSFER_WARMUP_REPORT.md", "Held-out deployment results; interpretation limits", "Explicitly labelled descriptive in the figure."),
        ("Figure_2_information_design", "b", "Temporal hold-out: history <= Ti-90 d; 30 d support; final 60 d hold-out", "analysis_outputs/ph_temporal_holdout_site_analysis_v1/reports/FINAL_PH_TEMPORAL_HOLDOUT_AND_SITE_FAILURE_REPORT.md", "Sections 2-3 site-relative split and leakage audit", "Support is isolated from the final held-out period and used only for offset/B0."),
    ]
    path = OUT / "figure_source_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["figure", "panel", "displayed_value_or_label", "source_path", "source_field_or_section", "notes"])
        writer.writerows(rows)


def main():
    verify_local_sources()
    draw_figure_1()
    draw_figure_2()
    write_manifest()
    print(f"Wrote methodology figures and source manifest to {OUT}")


if __name__ == "__main__":
    main()
