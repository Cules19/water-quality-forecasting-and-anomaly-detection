from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
BASELINE_DIR = ROOT / "analysis_outputs" / "ph_baseline_full50_v3"
DATASET_DIR = ROOT / "analysis_outputs" / "ph_dataset"
OUT_DIR = ROOT / "analysis_outputs" / "ph_residual_event_validation_full50_v3"
FIG_DIR = OUT_DIR / "figures"

PREDICTION_CANDIDATES = [
    BASELINE_DIR / "full50_v3_predictions_test.parquet",
    BASELINE_DIR / "full50_predictions_test.parquet",
    BASELINE_DIR / "predictions_test.parquet",
]
EVENT_WINDOWS_PATH = DATASET_DIR / "ph_event_windows.csv"
BACKUP_EVENT_PATH = ROOT / "data" / "reference" / "regulatory_events.csv"
OVERALL_METRICS_PATH = BASELINE_DIR / "full50_v3_model_metrics_overall.csv"
SITE_METRICS_PATH = BASELINE_DIR / "full50_v3_model_metrics_by_site.csv"

TARGET = "00400"
SITE_COL = "site_id"
TIME_COL = "timestamp"
EVENT_COL = "in_ph_event"
EVENT_LABEL_COL = "ph_event_label"
MODEL_ORDER = [
    "M0_site_mean",
    "M1_seasonal_mean",
    "M2_persistence_1h",
    "M3_ridge",
    "M4_hist_gradient_boosting",
]
MODEL_SHORT = {
    "M0_site_mean": "m0",
    "M1_seasonal_mean": "m1",
    "M2_persistence_1h": "m2",
    "M3_ridge": "m3",
    "M4_hist_gradient_boosting": "m4",
}
TOP_PCTS = [0.005, 0.01, 0.02, 0.05, 0.10]
SITE_TOP_PCTS = [0.01, 0.05, 0.10]
MAIN_MODEL = "M4_hist_gradient_boosting"
COMPARATOR_MODELS = ["M3_ridge", "M2_persistence_1h"]
RANDOM_SEED = 42


def log(message: str) -> None:
    print(message, flush=True)


def normalise_site_id(value: object) -> str:
    raw = str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    return raw.zfill(8) if raw.isdigit() else raw


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{value:.6g}")
        else:
            display[col] = display[col].astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.to_numpy()]
    return "\n".join([header, separator, *body])


def find_prediction_path() -> Path:
    for path in PREDICTION_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No full50 test prediction parquet found in ph_baseline_full50_v3.")


def load_predictions(path: Path) -> pd.DataFrame:
    required = [
        SITE_COL,
        TIME_COL,
        "split",
        "model",
        "y_true",
        "y_pred",
        "residual",
        "abs_residual",
        EVENT_COL,
        EVENT_LABEL_COL,
        "month",
        "hour",
        "day_of_year",
    ]
    schema_cols = pq.ParquetFile(path).schema.names
    missing = [col for col in required if col not in schema_cols]
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {missing}")
    log(f"Loading full50 test predictions: {path}")
    df = pd.read_parquet(path, columns=required)
    df[SITE_COL] = df[SITE_COL].map(normalise_site_id).astype("string")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df["model"] = df["model"].astype("string")
    df["split"] = df["split"].astype("string")
    df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce").fillna(0).astype("int8")
    df[EVENT_LABEL_COL] = df[EVENT_LABEL_COL].fillna("none").astype("string")
    for col in ["y_true", "y_pred", "residual", "abs_residual", "month", "hour", "day_of_year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["residual"] = df["y_true"] - df["y_pred"]
    df["abs_residual"] = df["residual"].abs()
    return df


def load_events() -> tuple[pd.DataFrame, Path]:
    path = EVENT_WINDOWS_PATH if EVENT_WINDOWS_PATH.exists() else BACKUP_EVENT_PATH
    if not path.exists():
        raise FileNotFoundError(f"No pH event window file found: {EVENT_WINDOWS_PATH} or {BACKUP_EVENT_PATH}")
    events = pd.read_csv(path)
    required = [SITE_COL, "label", EVENT_LABEL_COL, "incident_start", "incident_end", "duration_hrs"]
    missing = [col for col in required if col not in events.columns]
    if missing:
        raise ValueError(f"Event window file is missing required columns: {missing}")
    events = events.copy()
    events[SITE_COL] = events[SITE_COL].map(normalise_site_id).astype("string")
    events["incident_start"] = pd.to_datetime(events["incident_start"], errors="coerce")
    events["incident_end"] = pd.to_datetime(events["incident_end"], errors="coerce")
    events["duration_hrs"] = pd.to_numeric(events["duration_hrs"], errors="coerce")
    events[EVENT_LABEL_COL] = events[EVENT_LABEL_COL].fillna("unknown").astype("string")
    events["label"] = events["label"].fillna("").astype("string")
    events = events[
        events["label"].isin(["pH_exceedance_low", "pH_exceedance_high"])
        | events[EVENT_LABEL_COL].isin(["low", "high"])
    ].copy()
    events = events.dropna(subset=[SITE_COL, "incident_start", "incident_end"])
    events = events[events["incident_end"] >= events["incident_start"]].copy()
    events = events.sort_values([SITE_COL, "incident_start", "incident_end"]).reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1)
    events["duration_class"] = pd.cut(
        events["duration_hrs"],
        bins=[-np.inf, 6, 24, np.inf],
        labels=["short_0_6h", "medium_6_24h", "long_24h_plus"],
    ).astype("string")
    events["duration_binary"] = np.where(events["duration_hrs"] < 24, "short_under_24h", "long_24h_plus")
    return events, path


def make_prediction_wide(pred: pd.DataFrame) -> pd.DataFrame:
    base_model = MODEL_ORDER[0]
    base = pred[pred["model"].eq(base_model)].sort_values([SITE_COL, TIME_COL]).reset_index(drop=True)
    base = base[
        [
            SITE_COL,
            TIME_COL,
            "split",
            "y_true",
            EVENT_COL,
            EVENT_LABEL_COL,
            "month",
            "hour",
            "day_of_year",
        ]
    ].copy()
    for model in MODEL_ORDER:
        short = MODEL_SHORT[model]
        group = pred[pred["model"].eq(model)].sort_values([SITE_COL, TIME_COL]).reset_index(drop=True)
        if len(group) != len(base):
            raise ValueError(f"Model {model} row count does not match base model.")
        if not (
            group[SITE_COL].astype(str).to_numpy() == base[SITE_COL].astype(str).to_numpy()
        ).all() or not (group[TIME_COL].to_numpy() == base[TIME_COL].to_numpy()).all():
            raise ValueError(f"Model {model} rows do not align with base site/timestamp order.")
        base[f"pred_{short}"] = group["y_pred"].to_numpy(dtype="float64")
        base[f"resid_{short}"] = group["residual"].to_numpy(dtype="float64")
        base[f"abs_{short}"] = group["abs_residual"].to_numpy(dtype="float64")
    return base


def run_initial_checks(pred: pd.DataFrame, events: pd.DataFrame, pred_path: Path, event_path: Path) -> pd.DataFrame:
    models = sorted(pred["model"].dropna().astype(str).unique().tolist())
    finite_pred = int(np.isfinite(pred["y_pred"].to_numpy(dtype="float64")).sum())
    finite_abs = int(np.isfinite(pred["abs_residual"].to_numpy(dtype="float64")).sum())
    split_values = sorted(pred["split"].dropna().astype(str).unique().tolist())
    event_labels = sorted(pred[EVENT_LABEL_COL].dropna().astype(str).unique().tolist())
    test_min = pred[TIME_COL].min()
    test_max = pred[TIME_COL].max()
    event_overlap = events[(events["incident_end"] >= test_min) & (events["incident_start"] <= test_max)]
    checks = [
        ("prediction_file_exists", pred_path.exists(), str(pred_path)),
        ("event_window_file_exists", event_path.exists(), str(event_path)),
        ("prediction_rows_loaded", len(pred) > 0, f"{len(pred):,}"),
        ("five_expected_models_present", set(MODEL_ORDER).issubset(set(models)), models),
        ("split_is_test_only", split_values == ["test"], split_values),
        ("required_prediction_columns_present", True, "checked on load"),
        ("finite_predictions", finite_pred == len(pred), f"{finite_pred:,}/{len(pred):,}"),
        ("finite_abs_residuals", finite_abs == len(pred), f"{finite_abs:,}/{len(pred):,}"),
        ("event_indicator_present", EVENT_COL in pred.columns, EVENT_COL),
        ("event_labels_present", {"none", "low", "high"}.intersection(event_labels) != set(), event_labels),
        ("event_windows_loaded", len(events) > 0, f"{len(events):,}"),
        ("event_windows_overlap_test_period", len(event_overlap) > 0, f"{len(event_overlap):,} overlapping windows"),
    ]
    out = pd.DataFrame(
        [
            {"check_name": name, "status": "PASS" if passed else "FAIL", "value": value, "notes": value}
            for name, passed, value in checks
        ]
    )
    out.to_csv(OUT_DIR / "residual_event_initial_checks.csv", index=False, encoding="utf-8-sig")
    if (out["status"] == "FAIL").any():
        raise RuntimeError("Critical residual-event validation checks failed:\n" + out.to_string(index=False))
    return out


def row_level_threshold_enrichment(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        group = pred[pred["model"].eq(model)].copy()
        abs_values = group["abs_residual"].to_numpy(dtype="float64")
        event_mask = group[EVENT_COL].to_numpy(dtype="int8") == 1
        n_rows = len(group)
        n_event = int(event_mask.sum())
        background = n_event / n_rows if n_rows else np.nan
        label_background = group[EVENT_LABEL_COL].value_counts(normalize=True).to_dict()
        for pct in TOP_PCTS:
            threshold = float(np.quantile(abs_values, 1 - pct))
            flagged = abs_values >= threshold
            n_flagged = int(flagged.sum())
            n_flagged_event = int(event_mask[flagged].sum())
            precision = n_flagged_event / n_flagged if n_flagged else np.nan
            recall = n_flagged_event / n_event if n_event else np.nan
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision > 0 and recall > 0
                else 0.0
            )
            rows.append(
                {
                    "model": model,
                    "top_pct": pct,
                    "threshold_abs_residual": threshold,
                    "n_rows": n_rows,
                    "n_event_rows": n_event,
                    "background_event_rate": background,
                    "n_flagged_rows": n_flagged,
                    "n_flagged_event_rows": n_flagged_event,
                    "flagged_event_rate": precision,
                    "enrichment": precision / background if background > 0 else np.nan,
                    "event_recall": recall,
                    "f1": f1,
                    "mean_abs_residual_flagged": float(group.loc[flagged, "abs_residual"].mean()),
                    "mean_abs_residual_unflagged": float(group.loc[~flagged, "abs_residual"].mean()),
                }
            )
            flagged_labels = group.loc[flagged, EVENT_LABEL_COL].value_counts(normalize=False)
            for label in ["low", "high"]:
                flagged_count = int(flagged_labels.get(label, 0))
                flagged_rate = flagged_count / n_flagged if n_flagged else np.nan
                base_rate = float(label_background.get(label, 0.0))
                label_rows.append(
                    {
                        "model": model,
                        "top_pct": pct,
                        "event_label": label,
                        "background_label_rate": base_rate,
                        "flagged_label_rate": flagged_rate,
                        "label_enrichment": flagged_rate / base_rate if base_rate > 0 else np.nan,
                        "n_flagged_label_rows": flagged_count,
                    }
                )
    row_out = pd.DataFrame(rows)
    label_out = pd.DataFrame(label_rows)
    row_out.to_csv(OUT_DIR / "row_level_top_residual_enrichment.csv", index=False, encoding="utf-8-sig")
    label_out.to_csv(OUT_DIR / "row_level_event_label_enrichment.csv", index=False, encoding="utf-8-sig")
    return row_out, label_out


def residual_distribution_summary(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    temp = pred.copy()
    temp["event_status"] = np.where(temp[EVENT_COL].eq(1), "event", "nonevent")
    rows = []
    for keys, group in temp.groupby(["model", "event_status", EVENT_LABEL_COL], observed=False):
        model, status, label = keys
        abs_resid = group["abs_residual"].dropna()
        resid = group["residual"].dropna()
        rows.append(
            {
                "model": model,
                "event_status": status,
                "ph_event_label": label,
                "n_rows": len(group),
                "mean_residual": float(resid.mean()),
                "median_residual": float(resid.median()),
                "mean_abs_residual": float(abs_resid.mean()),
                "median_abs_residual": float(abs_resid.median()),
                "abs_residual_p90": float(abs_resid.quantile(0.90)),
                "abs_residual_p95": float(abs_resid.quantile(0.95)),
                "abs_residual_p99": float(abs_resid.quantile(0.99)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "residual_distribution_by_event_status.csv", index=False, encoding="utf-8-sig")

    directional_rows = []
    event_only = temp[temp[EVENT_COL].eq(1)].copy()
    for keys, group in event_only.groupby(["model", EVENT_LABEL_COL], observed=False):
        model, label = keys
        resid = group["residual"].dropna()
        direction_mask = resid.lt(0) if str(label) == "low" else resid.gt(0)
        directional_rows.append(
            {
                "model": model,
                "ph_event_label": label,
                "n_event_rows": len(group),
                "mean_residual": float(resid.mean()),
                "median_residual": float(resid.median()),
                "share_negative_residual": float((resid < 0).mean()),
                "share_positive_residual": float((resid > 0).mean()),
                "direction_consistent_share": float(direction_mask.mean()),
            }
        )
    directional = pd.DataFrame(directional_rows)
    directional.to_csv(OUT_DIR / "directional_residual_summary_by_event_label.csv", index=False, encoding="utf-8-sig")
    return summary, directional


def site_level_enrichment(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (model, site_id), group in pred.groupby(["model", SITE_COL], observed=False):
        abs_values = group["abs_residual"].to_numpy(dtype="float64")
        event_mask = group[EVENT_COL].to_numpy(dtype="int8") == 1
        n_rows = len(group)
        n_event = int(event_mask.sum())
        background = n_event / n_rows if n_rows else np.nan
        for pct in SITE_TOP_PCTS:
            threshold = float(np.quantile(abs_values, 1 - pct))
            flagged = abs_values >= threshold
            n_flagged = int(flagged.sum())
            n_flagged_event = int(event_mask[flagged].sum())
            top_rate = n_flagged_event / n_flagged if n_flagged else np.nan
            rows.append(
                {
                    "model": model,
                    "site_id": site_id,
                    "top_pct": pct,
                    "n_rows": n_rows,
                    "n_event_rows": n_event,
                    "background_event_rate": background,
                    "threshold_abs_residual": threshold,
                    "n_flagged_rows": n_flagged,
                    "top_event_rate": top_rate,
                    "enrichment": top_rate / background if background > 0 else np.nan,
                }
            )
    site_out = pd.DataFrame(rows)
    summary = site_out.groupby(["model", "top_pct"], observed=False).agg(
        n_sites=(SITE_COL, "nunique"),
        median_enrichment=("enrichment", "median"),
        mean_enrichment=("enrichment", "mean"),
        n_sites_enrichment_gt_1=("enrichment", lambda values: int((values > 1).sum())),
        n_sites_enrichment_gt_1_5=("enrichment", lambda values: int((values > 1.5).sum())),
        median_top_event_rate=("top_event_rate", "median"),
        median_background_event_rate=("background_event_rate", "median"),
    ).reset_index()
    site_out.to_csv(OUT_DIR / "site_level_top_residual_enrichment.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "site_level_enrichment_summary.csv", index=False, encoding="utf-8-sig")
    return site_out, summary


def event_overlap_summary(events: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    test_range = base.groupby(SITE_COL)[TIME_COL].agg(["min", "max", "count"]).reset_index()
    merged = events.merge(test_range, on=SITE_COL, how="left")
    merged["overlaps_site_test_period"] = (
        merged["count"].notna()
        & (merged["incident_end"] >= merged["min"])
        & (merged["incident_start"] <= merged["max"])
    )
    summary = pd.DataFrame(
        [
            {
                "n_event_windows_total": len(events),
                "n_event_windows_on_test_sites": int(merged["count"].notna().sum()),
                "n_event_windows_overlapping_test_period": int(merged["overlaps_site_test_period"].sum()),
                "n_sites_with_test_predictions": int(base[SITE_COL].nunique()),
                "test_start_min": base[TIME_COL].min(),
                "test_end_max": base[TIME_COL].max(),
            }
        ]
    )
    summary.to_csv(OUT_DIR / "event_window_overlap_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def event_level_hits(base: pd.DataFrame, events: pd.DataFrame, thresholds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_lookup = {
        (row["model"], float(row["top_pct"])): float(row["threshold_abs_residual"])
        for row in thresholds.to_dict("records")
    }
    event_rows: list[dict[str, object]] = []
    base_sorted = base.sort_values([SITE_COL, TIME_COL]).reset_index(drop=True)
    site_groups = {
        str(site_id): site_df.reset_index(drop=True)
        for site_id, site_df in base_sorted.groupby(SITE_COL, observed=False)
    }

    for event in events.to_dict("records"):
        site_id = str(event[SITE_COL])
        site_df = site_groups.get(site_id)
        if site_df is None or site_df.empty:
            continue
        times = site_df[TIME_COL].to_numpy()
        start = np.datetime64(pd.Timestamp(event["incident_start"]))
        end = np.datetime64(pd.Timestamp(event["incident_end"]))
        left = int(np.searchsorted(times, start, side="left"))
        right = int(np.searchsorted(times, end, side="right"))
        if right <= left:
            continue
        window = site_df.iloc[left:right]
        if window.empty:
            continue
        for model in MODEL_ORDER:
            short = MODEL_SHORT[model]
            abs_values = window[f"abs_{short}"].to_numpy(dtype="float64")
            resid_values = window[f"resid_{short}"].to_numpy(dtype="float64")
            label = str(event[EVENT_LABEL_COL])
            direction_consistent = resid_values < 0 if label == "low" else resid_values > 0
            row = {
                "event_id": event["event_id"],
                "site_id": site_id,
                "model": model,
                "label": event["label"],
                "ph_event_label": label,
                "incident_start": event["incident_start"],
                "incident_end": event["incident_end"],
                "duration_hrs": event["duration_hrs"],
                "duration_class": event["duration_class"],
                "duration_binary": event["duration_binary"],
                "n_test_rows_in_event": len(window),
                "mean_abs_residual": float(np.mean(abs_values)),
                "median_abs_residual": float(np.median(abs_values)),
                "max_abs_residual": float(np.max(abs_values)),
                "mean_residual": float(np.mean(resid_values)),
                "direction_consistent_share": float(np.mean(direction_consistent)),
            }
            for pct in [0.01, 0.05, 0.10]:
                threshold = threshold_lookup[(model, pct)]
                flagged = abs_values >= threshold
                row[f"hit_top_{int(pct * 100)}pct"] = bool(flagged.any())
                row[f"fraction_rows_top_{int(pct * 100)}pct"] = float(flagged.mean())
            event_rows.append(row)
    event_out = pd.DataFrame(event_rows)
    event_out.to_csv(OUT_DIR / "event_level_residual_hits.csv", index=False, encoding="utf-8-sig")

    group_cols = ["model", "ph_event_label", "duration_binary"]
    summary = event_out.groupby(group_cols, observed=False).agg(
        n_events=("event_id", "nunique"),
        median_event_rows=("n_test_rows_in_event", "median"),
        mean_event_rows=("n_test_rows_in_event", "mean"),
        mean_abs_residual=("mean_abs_residual", "mean"),
        median_max_abs_residual=("max_abs_residual", "median"),
        hit_rate_top_1pct=("hit_top_1pct", "mean"),
        hit_rate_top_5pct=("hit_top_5pct", "mean"),
        hit_rate_top_10pct=("hit_top_10pct", "mean"),
        mean_fraction_rows_top_5pct=("fraction_rows_top_5pct", "mean"),
        mean_direction_consistent_share=("direction_consistent_share", "mean"),
    ).reset_index()
    overall = event_out.groupby(["model"], observed=False).agg(
        n_events=("event_id", "nunique"),
        median_event_rows=("n_test_rows_in_event", "median"),
        mean_event_rows=("n_test_rows_in_event", "mean"),
        mean_abs_residual=("mean_abs_residual", "mean"),
        median_max_abs_residual=("max_abs_residual", "median"),
        hit_rate_top_1pct=("hit_top_1pct", "mean"),
        hit_rate_top_5pct=("hit_top_5pct", "mean"),
        hit_rate_top_10pct=("hit_top_10pct", "mean"),
        mean_fraction_rows_top_5pct=("fraction_rows_top_5pct", "mean"),
        mean_direction_consistent_share=("direction_consistent_share", "mean"),
    ).reset_index()
    overall["ph_event_label"] = "all"
    overall["duration_binary"] = "all"
    summary = pd.concat([overall[summary.columns], summary], ignore_index=True)
    summary.to_csv(OUT_DIR / "event_level_hit_summary.csv", index=False, encoding="utf-8-sig")
    return event_out, summary


def model_validation_ranking(row_enrichment: pd.DataFrame, event_summary: pd.DataFrame) -> pd.DataFrame:
    top5 = row_enrichment[row_enrichment["top_pct"].eq(0.05)][
        ["model", "enrichment", "event_recall", "flagged_event_rate"]
    ].rename(
        columns={
            "enrichment": "row_top5_enrichment",
            "event_recall": "row_top5_event_recall",
            "flagged_event_rate": "row_top5_precision",
        }
    )
    top1 = row_enrichment[row_enrichment["top_pct"].eq(0.01)][["model", "enrichment"]].rename(
        columns={"enrichment": "row_top1_enrichment"}
    )
    event_all = event_summary[
        event_summary["ph_event_label"].eq("all") & event_summary["duration_binary"].eq("all")
    ][["model", "hit_rate_top_1pct", "hit_rate_top_5pct", "hit_rate_top_10pct", "mean_direction_consistent_share"]]
    ranking = top5.merge(top1, on="model", how="left").merge(event_all, on="model", how="left")
    ranking["interpretation_score"] = (
        ranking["row_top5_enrichment"].fillna(0)
        + ranking["row_top1_enrichment"].fillna(0)
        + ranking["hit_rate_top_5pct"].fillna(0)
    )
    ranking = ranking.sort_values("interpretation_score", ascending=False)
    ranking.to_csv(OUT_DIR / "residual_event_model_ranking.csv", index=False, encoding="utf-8-sig")
    return ranking


def save_figures(
    pred: pd.DataFrame,
    row_enrichment: pd.DataFrame,
    label_enrichment: pd.DataFrame,
    site_enrichment: pd.DataFrame,
    event_summary: pd.DataFrame,
    directional: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    top = row_enrichment.copy()
    top["top_pct_label"] = (top["top_pct"] * 100).map(lambda value: f"{value:g}%")
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in MODEL_ORDER:
        group = top[top["model"].eq(model)]
        ax.plot(group["top_pct"] * 100, group["enrichment"], marker="o", label=model)
    ax.axhline(1, color="#111111", linestyle="--", linewidth=1)
    ax.set_xlabel("Top absolute residual rows (%)")
    ax.set_ylabel("Event enrichment")
    ax.set_title("Row-level event enrichment across residual thresholds")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_row_level_enrichment_by_threshold.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for model in MODEL_ORDER:
        group = row_enrichment[row_enrichment["model"].eq(model)].sort_values("top_pct")
        ax.plot(group["row_top5_precision"] if "row_top5_precision" in group.columns else group["flagged_event_rate"], group["event_recall"], marker="o", label=model)
    ax.set_xlabel("Precision/event rate among flagged rows")
    ax.set_ylabel("Recall/share of event rows flagged")
    ax.set_title("Residual threshold precision-recall tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_precision_recall_tradeoff.png", dpi=200)
    plt.close(fig)

    top5_site = site_enrichment[site_enrichment["top_pct"].eq(0.05)]
    data = [top5_site.loc[top5_site["model"].eq(model), "enrichment"].dropna().to_numpy() for model in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(data, tick_labels=MODEL_ORDER, showfliers=False)
    ax.axhline(1, color="#111111", linestyle="--", linewidth=1)
    ax.set_ylabel("Site-level top 5% enrichment")
    ax.set_title("Site-level residual-event enrichment")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_site_level_top5_enrichment_boxplot.png", dpi=200)
    plt.close(fig)

    event_all = event_summary[event_summary["ph_event_label"].eq("all") & event_summary["duration_binary"].eq("all")]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(event_all))
    width = 0.25
    ax.bar(x - width, event_all["hit_rate_top_1pct"], width=width, label="top 1%")
    ax.bar(x, event_all["hit_rate_top_5pct"], width=width, label="top 5%")
    ax.bar(x + width, event_all["hit_rate_top_10pct"], width=width, label="top 10%")
    ax.set_xticks(x)
    ax.set_xticklabels(event_all["model"], rotation=25, ha="right")
    ax.set_ylabel("Event-level hit rate")
    ax.set_title("Event-level hit rates by model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_event_level_hit_rate_by_model.png", dpi=200)
    plt.close(fig)

    event_label = event_summary[
        event_summary["ph_event_label"].isin(["low", "high"]) & event_summary["duration_binary"].eq("all")
    ]
    if event_label.empty:
        event_label = event_summary[
            event_summary["ph_event_label"].isin(["low", "high"])
            & event_summary["duration_binary"].isin(["short_under_24h", "long_24h_plus"])
        ]
    if not event_label.empty:
        pivot = event_label.pivot_table(index="model", columns="ph_event_label", values="hit_rate_top_5pct")
        pivot = pivot.reindex(MODEL_ORDER)
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Top 5% event hit rate")
        ax.set_title("Low vs high event hit rate")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_low_high_event_hit_rate.png", dpi=200)
        plt.close(fig)

    sample_parts = []
    for model in MODEL_ORDER:
        model_df = pred[pred["model"].eq(model)]
        sample_parts.append(model_df.sample(min(120_000, len(model_df)), random_state=RANDOM_SEED))
    sample = pd.concat(sample_parts, ignore_index=True)
    sample["event_status"] = np.where(sample[EVENT_COL].eq(1), "event", "nonevent")
    data = []
    labels = []
    for model in MODEL_ORDER:
        for status in ["nonevent", "event"]:
            values = sample.loc[sample["model"].eq(model) & sample["event_status"].eq(status), "abs_residual"]
            data.append(values.to_numpy())
            labels.append(f"{model}\n{status}")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Absolute residual")
    ax.set_title("Absolute residual distribution: event vs non-event rows")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_abs_residual_event_vs_nonevent.png", dpi=200)
    plt.close(fig)

    dir_plot = directional[directional["ph_event_label"].isin(["low", "high"])]
    if not dir_plot.empty:
        pivot = dir_plot.pivot_table(index="model", columns="ph_event_label", values="direction_consistent_share")
        pivot = pivot.reindex(MODEL_ORDER)
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Direction-consistent residual share")
        ax.set_title("Residual direction consistency in pH events")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_directional_residual_by_event_label.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    rank_plot = ranking.sort_values("interpretation_score")
    ax.barh(rank_plot["model"], rank_plot["interpretation_score"], color="#4C78A8")
    ax.set_xlabel("Composite residual-event validation score")
    ax.set_title("Residual-event validation ranking")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_residual_event_model_ranking.png", dpi=200)
    plt.close(fig)

    top_label = label_enrichment[label_enrichment["top_pct"].eq(0.05)].copy()
    pivot = top_label.pivot_table(index="model", columns="event_label", values="label_enrichment")
    pivot = pivot.reindex(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(1, color="#111111", linestyle="--", linewidth=1)
    ax.set_ylabel("Top 5% label enrichment")
    ax.set_title("Low/high label enrichment among top residual rows")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_top5_label_enrichment.png", dpi=200)
    plt.close(fig)


def decide(row_enrichment: pd.DataFrame, event_summary: pd.DataFrame, ranking: pd.DataFrame) -> tuple[str, list[str]]:
    hgb_top5 = row_enrichment[
        row_enrichment["model"].eq(MAIN_MODEL) & row_enrichment["top_pct"].eq(0.05)
    ].iloc[0]
    hgb_top1 = row_enrichment[
        row_enrichment["model"].eq(MAIN_MODEL) & row_enrichment["top_pct"].eq(0.01)
    ].iloc[0]
    hgb_event = event_summary[
        event_summary["model"].eq(MAIN_MODEL)
        & event_summary["ph_event_label"].eq("all")
        & event_summary["duration_binary"].eq("all")
    ].iloc[0]
    reasons = [
        f"HGB top 5% row-level enrichment: {hgb_top5['enrichment']:.3f}.",
        f"HGB top 1% row-level enrichment: {hgb_top1['enrichment']:.3f}.",
        f"HGB event-level top 5% hit rate: {hgb_event['hit_rate_top_5pct']:.3f}.",
    ]
    if hgb_top5["enrichment"] >= 1.2 and hgb_top1["enrichment"] >= 1.2:
        return "RESIDUAL_EVENT_VALIDATION_COMPLETED_MODERATE_EVIDENCE", reasons
    if hgb_top5["enrichment"] >= 1.05 or hgb_top1["enrichment"] >= 1.05:
        return "RESIDUAL_EVENT_VALIDATION_COMPLETED_WEAK_OR_MIXED_EVIDENCE", reasons
    return "RESIDUAL_EVENT_VALIDATION_COMPLETED_NO_CLEAR_ENRICHMENT", reasons


def write_report(
    pred_path: Path,
    event_path: Path,
    pred: pd.DataFrame,
    events: pd.DataFrame,
    checks: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    row_enrichment: pd.DataFrame,
    label_enrichment: pd.DataFrame,
    residual_summary: pd.DataFrame,
    directional: pd.DataFrame,
    site_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    decision: str,
    decision_reasons: list[str],
    elapsed: float,
) -> None:
    top5 = row_enrichment[row_enrichment["top_pct"].eq(0.05)].sort_values("enrichment", ascending=False)
    top1 = row_enrichment[row_enrichment["top_pct"].eq(0.01)].sort_values("enrichment", ascending=False)
    event_all = event_summary[
        event_summary["ph_event_label"].eq("all") & event_summary["duration_binary"].eq("all")
    ].sort_values("hit_rate_top_5pct", ascending=False)
    hgb_label = label_enrichment[
        label_enrichment["model"].eq(MAIN_MODEL) & label_enrichment["top_pct"].eq(0.05)
    ]
    reason_text = "\n".join(f"- {reason}" for reason in decision_reasons)
    report = f"""# Full50 Residual-Event Validation Report

## 1. Purpose

This step tests whether high prediction residuals align with pH regulatory event windows. No new model is trained.

The main interpretation focuses on `{MAIN_MODEL}` as the feature-based residual model, with `M3_ridge` as a linear comparator and `M2_persistence_1h` as a temporal benchmark.

## 2. Inputs

- Prediction file: `{pred_path}`
- Event window file: `{event_path}`
- Output folder: `{OUT_DIR}`
- Prediction rows: {len(pred):,}
- Models: {MODEL_ORDER}
- Event windows loaded: {len(events):,}

Initial checks:

{dataframe_to_markdown(checks)}

Event window overlap summary:

{dataframe_to_markdown(overlap_summary)}

## 3. Row-Level Top Residual Enrichment

Top 5% residual rows:

{dataframe_to_markdown(top5)}

Top 1% residual rows:

{dataframe_to_markdown(top1)}

HGB low/high label enrichment at top 5%:

{dataframe_to_markdown(hgb_label)}

## 4. Residual Distribution And Direction

Residual distribution by event status:

{dataframe_to_markdown(residual_summary, max_rows=30)}

Directional residual summary:

{dataframe_to_markdown(directional)}

For low-pH events, direction-consistent residuals are negative (`y_true < y_pred`). For high-pH events, direction-consistent residuals are positive (`y_true > y_pred`).

## 5. Site-Level Enrichment

Site-level enrichment summary:

{dataframe_to_markdown(site_summary)}

## 6. Event-Level Hit Rate

Event-level summary:

{dataframe_to_markdown(event_all)}

Low/high and short/long event breakdown:

{dataframe_to_markdown(event_summary[event_summary["ph_event_label"].ne("all")], max_rows=40)}

## 7. Model Ranking

{dataframe_to_markdown(ranking)}

Persistence-1h uses historical pH and should be interpreted as a temporal benchmark, not the main co-behaviour hot-moment model.

## 8. Decision

{decision}

{reason_text}

Runtime: {elapsed:.1f} seconds.
"""
    (OUT_DIR / "residual_event_validation_report.md").write_text(report, encoding="utf-8")
    (OUT_DIR / "residual_event_validation_decision.txt").write_text(
        decision + "\n\n" + reason_text + "\n",
        encoding="utf-8",
    )


def main() -> None:
    start = perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pred_path = find_prediction_path()
    pred = load_predictions(pred_path)
    events, event_path = load_events()
    checks = run_initial_checks(pred, events, pred_path, event_path)

    log("Computing row-level residual-event enrichment...")
    row_enrichment, label_enrichment = row_level_threshold_enrichment(pred)
    residual_summary, directional = residual_distribution_summary(pred)
    site_enrichment, site_summary = site_level_enrichment(pred)

    log("Preparing event-level residual hit validation...")
    base = make_prediction_wide(pred)
    overlap_summary = event_overlap_summary(events, base)
    event_hits, event_summary = event_level_hits(base, events, row_enrichment)
    ranking = model_validation_ranking(row_enrichment, event_summary)

    log("Saving figures and report...")
    save_figures(pred, row_enrichment, label_enrichment, site_enrichment, event_summary, directional, ranking)
    decision, reasons = decide(row_enrichment, event_summary, ranking)
    elapsed = perf_counter() - start
    write_report(
        pred_path,
        event_path,
        pred,
        events,
        checks,
        overlap_summary,
        row_enrichment,
        label_enrichment,
        residual_summary,
        directional,
        site_summary,
        event_summary,
        ranking,
        decision,
        reasons,
        elapsed,
    )

    hgb_top5 = row_enrichment[
        row_enrichment["model"].eq(MAIN_MODEL) & row_enrichment["top_pct"].eq(0.05)
    ].iloc[0]
    hgb_event = event_summary[
        event_summary["model"].eq(MAIN_MODEL)
        & event_summary["ph_event_label"].eq("all")
        & event_summary["duration_binary"].eq("all")
    ].iloc[0]
    log(f"input prediction file: {pred_path}")
    log(f"event window file: {event_path}")
    log(f"output folder: {OUT_DIR}")
    log(f"prediction rows: {len(pred):,}")
    log(f"models evaluated: {', '.join(MODEL_ORDER)}")
    log(f"event windows loaded: {len(events):,}")
    log(f"event windows with test rows: {event_hits['event_id'].nunique() if not event_hits.empty else 0:,}")
    log(f"HGB top 5pct row-level enrichment: {float(hgb_top5['enrichment']):.6f}")
    log(f"HGB top 5pct event recall: {float(hgb_top5['event_recall']):.6f}")
    log(f"HGB event-level top 5pct hit rate: {float(hgb_event['hit_rate_top_5pct']):.6f}")
    log(f"best residual-event ranking model: {ranking.iloc[0]['model']}")
    log(f"final decision: {decision}")


if __name__ == "__main__":
    main()
