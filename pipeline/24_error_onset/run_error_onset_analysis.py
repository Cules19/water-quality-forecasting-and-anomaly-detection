"""Leakage-safe residual lead-time, attribution and compression diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "NA" if pd.isna(value) else f"{float(value):.5g}"
            )
        else:
            display[column] = display[column].fillna("NA").astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *body])


def read_prediction_split(root: Path, run_id: str, split: str) -> pd.DataFrame:
    paths = sorted((root / run_id / split).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No prediction shards: {root / run_id / split}")
    columns = [
        "site_id",
        "timestamp",
        "observed_native",
        "predicted_native",
        "in_event",
        "event_direction",
    ]
    parts = [pd.read_parquet(path, columns=columns) for path in paths]
    frame = pd.concat(parts, ignore_index=True)
    frame["site_id"] = frame["site_id"].astype("string")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["in_event"] = frame["in_event"].fillna(False).astype(bool)
    frame["residual"] = (
        pd.to_numeric(frame["observed_native"], errors="coerce")
        - pd.to_numeric(frame["predicted_native"], errors="coerce")
    )
    frame["abs_residual"] = frame["residual"].abs()
    return frame.sort_values(["site_id", "timestamp"], kind="mergesort").reset_index(drop=True)


def validation_threshold(
    calibration: pd.DataFrame,
    tail_fraction: float,
) -> dict[str, float | int | str | bool]:
    normal = calibration.loc[~calibration["in_event"], "abs_residual"].dropna()
    if normal.empty:
        raise ValueError("No normal validation residuals available for threshold fitting")
    return {
        "threshold": float(normal.quantile(1.0 - tail_fraction)),
        "tail_fraction": float(tail_fraction),
        "fit_split": "validation",
        "fit_rows": int(len(normal)),
        "fit_sites": int(calibration.loc[normal.index, "site_id"].nunique()),
        "test_labels_used_for_threshold": False,
    }


def event_windows(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the hourly weak-label mask into contiguous registered windows."""
    rows: list[dict[str, Any]] = []
    for site_id, site in frame.groupby("site_id", sort=False):
        site = site.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        timestamp = site["timestamp"]
        active = site["in_event"].to_numpy(dtype=bool)
        previous_active = np.r_[False, active[:-1]]
        gap = timestamp.diff().dt.total_seconds().div(3600).fillna(np.inf).to_numpy()
        starts = np.flatnonzero(active & ((~previous_active) | (gap > 1.5)))
        for number, start in enumerate(starts):
            later_inactive = np.flatnonzero(~active[start:])
            inactive_stop = (
                start + int(later_inactive[0]) if len(later_inactive) else len(site)
            )
            next_start = int(starts[number + 1]) if number + 1 < len(starts) else len(site)
            stop = min(inactive_stop, next_start)
            segment = site.iloc[start:stop]
            if segment.empty:
                continue
            direction_mode = segment["event_direction"].mode()
            rows.append(
                {
                    "site_id": str(site_id),
                    "event_window_id": f"{site_id}:{timestamp.iloc[start].isoformat()}",
                    "event_start": timestamp.iloc[start],
                    "event_end": segment["timestamp"].iloc[-1],
                    "event_rows": int(len(segment)),
                    "event_direction": str(direction_mode.iloc[0])
                    if not direction_mode.empty
                    else "unknown",
                }
            )
    return pd.DataFrame(rows)


def lead_time_rows(
    evaluation: pd.DataFrame,
    threshold: float,
    maximum_lead_hours: int,
    minimum_consecutive_hours: int = 1,
) -> pd.DataFrame:
    if minimum_consecutive_hours < 1:
        raise ValueError("minimum_consecutive_hours must be at least 1")
    windows = event_windows(evaluation)
    rows: list[dict[str, Any]] = []
    by_site = {
        str(site_id): site.sort_values("timestamp", kind="mergesort")
        for site_id, site in evaluation.groupby("site_id", sort=False)
    }
    for event in windows.itertuples(index=False):
        site = by_site[str(event.site_id)]
        lower = event.event_start - pd.Timedelta(hours=maximum_lead_hours)
        pre = site[
            site["timestamp"].ge(lower)
            & site["timestamp"].lt(event.event_start)
            & ~site["in_event"]
        ].sort_values("timestamp", kind="mergesort").copy()
        hot = pre.loc[
            pre["abs_residual"].ge(threshold),
            ["timestamp", "abs_residual"],
        ].copy()
        if hot.empty:
            qualifying = pd.DataFrame()
        else:
            # A below-threshold or missing intervening hour separates episodes.
            new_episode = hot["timestamp"].diff().gt(pd.Timedelta(hours=1.5))
            hot["episode_id"] = new_episode.fillna(True).cumsum()
            episodes = hot.groupby("episode_id", sort=False).agg(
                alert_start=("timestamp", "min"),
                alert_end=("timestamp", "max"),
                consecutive_hours=("timestamp", "size"),
                peak_abs_residual=("abs_residual", "max"),
            )
            qualifying = episodes.loc[
                episodes["consecutive_hours"].ge(minimum_consecutive_hours)
            ]
        # Use the alert episode nearest the event. This prevents an unrelated
        # isolated tail residual early in a long lookback from inflating lead.
        selected = (
            qualifying.sort_values("alert_end", kind="mergesort").iloc[-1]
            if not qualifying.empty
            else None
        )
        first_alert = selected["alert_start"] if selected is not None else pd.NaT
        last_alert = selected["alert_end"] if selected is not None else pd.NaT
        lead = (
            (event.event_start - first_alert).total_seconds() / 3600.0
            if pd.notna(first_alert)
            else np.nan
        )
        alert_gap = (
            (event.event_start - last_alert).total_seconds() / 3600.0
            if pd.notna(last_alert)
            else np.nan
        )
        rows.append(
            {
                **event._asdict(),
                "analysis_window_start": lower,
                "pre_event_rows": int(len(pre)),
                "pre_event_coverage_hours": float(
                    (event.event_start - pre["timestamp"].min()).total_seconds() / 3600.0
                )
                if not pre.empty
                else 0.0,
                "first_alert": first_alert,
                "last_alert": last_alert,
                "lead_hours": lead,
                "alert_to_event_gap_hours": alert_gap,
                "alerted": bool(pd.notna(first_alert)),
                "threshold": float(threshold),
                "minimum_consecutive_hours": int(minimum_consecutive_hours),
                "selected_episode_hours": (
                    int(selected["consecutive_hours"])
                    if selected is not None
                    else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def summarise_lead_times(
    lead: pd.DataFrame,
    model_label: str,
    reported_hours: list[int],
) -> dict[str, Any]:
    alerted = lead.loc[lead["alerted"], "lead_hours"].dropna()
    analyzable = lead["pre_event_rows"].gt(0)
    analyzable_count = int(analyzable.sum())
    result: dict[str, Any] = {
        "model": model_label,
        "minimum_consecutive_hours": int(
            lead["minimum_consecutive_hours"].iloc[0]
        )
        if len(lead)
        else np.nan,
        "event_windows": int(len(lead)),
        "events_with_pre_event_rows": analyzable_count,
        "alerted_events": int(lead["alerted"].sum()),
        "alerted_fraction": float(lead["alerted"].mean()) if len(lead) else np.nan,
        "alerted_fraction_among_events_with_pre_event_rows": (
            float(lead.loc[analyzable, "alerted"].mean())
            if analyzable_count
            else np.nan
        ),
        "median_lead_hours_alerted": float(alerted.median()) if len(alerted) else np.nan,
        "lead_q25_hours_alerted": float(alerted.quantile(0.25)) if len(alerted) else np.nan,
        "lead_q75_hours_alerted": float(alerted.quantile(0.75)) if len(alerted) else np.nan,
    }
    for hours in reported_hours:
        result[f"fraction_events_alerted_at_least_{hours}h"] = (
            float(lead["lead_hours"].ge(hours).mean()) if len(lead) else np.nan
        )
        result[
            f"fraction_analyzable_events_alerted_at_least_{hours}h"
        ] = (
            float(lead.loc[analyzable, "lead_hours"].ge(hours).mean())
            if analyzable_count
            else np.nan
        )
    return result


def shifted_event_mask(frame: pd.DataFrame, lead_hours: int) -> np.ndarray:
    """Return event(t + lead), so a positive shift tests an earlier alert."""
    event_keys = frame.loc[frame["in_event"], ["site_id", "timestamp"]].copy()
    event_keys["timestamp"] = event_keys["timestamp"] - pd.Timedelta(hours=lead_hours)
    event_index = pd.MultiIndex.from_frame(event_keys.drop_duplicates())
    row_index = pd.MultiIndex.from_frame(frame[["site_id", "timestamp"]])
    return row_index.isin(event_index)


def lagged_enrichment_rows(
    evaluation: pd.DataFrame,
    threshold: float,
    lags: list[int],
    model_label: str,
) -> pd.DataFrame:
    hot = evaluation["abs_residual"].ge(threshold).to_numpy(dtype=bool)
    rows = []
    for lag in lags:
        shifted = shifted_event_mask(evaluation, int(lag))
        base_rate = float(shifted.mean())
        tail_rate = float(shifted[hot].mean()) if hot.any() else np.nan
        rows.append(
            {
                "model": model_label,
                "lead_hours": int(lag),
                "test_rows": int(len(evaluation)),
                "tail_rows": int(hot.sum()),
                "shifted_event_rows": int(shifted.sum()),
                "base_event_rate": base_rate,
                "tail_event_rate": tail_rate,
                "enrichment": tail_rate / base_rate if base_rate > 0 else np.nan,
                "threshold": float(threshold),
            }
        )
    return pd.DataFrame(rows)


def linear_slope(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 3 or np.nanstd(observed[valid]) < 1e-12:
        return np.nan
    return float(np.polyfit(observed[valid], predicted[valid], 1)[0])


def compression_metrics(
    observed: pd.Series,
    predicted: pd.Series,
    tail_fraction: float,
) -> dict[str, float | int]:
    o = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(o) & np.isfinite(p)
    o, p = o[valid], p[valid]
    if len(o) < 10:
        return {
            "n": int(len(o)),
            "overall_slope_predicted_on_observed": np.nan,
            "low_tail_slope": np.nan,
            "high_tail_slope": np.nan,
            "predicted_to_observed_p05_p95_range_ratio": np.nan,
        }
    low, high = np.quantile(o, [tail_fraction, 1.0 - tail_fraction])
    observed_range = float(np.quantile(o, 0.95) - np.quantile(o, 0.05))
    predicted_range = float(np.quantile(p, 0.95) - np.quantile(p, 0.05))
    return {
        "n": int(len(o)),
        "overall_slope_predicted_on_observed": linear_slope(o, p),
        "low_tail_slope": linear_slope(o[o <= low], p[o <= low]),
        "high_tail_slope": linear_slope(o[o >= high], p[o >= high]),
        "observed_tail_low": float(low),
        "observed_tail_high": float(high),
        "observed_p05_p95_range": observed_range,
        "predicted_p05_p95_range": predicted_range,
        "predicted_to_observed_p05_p95_range_ratio": (
            predicted_range / observed_range if observed_range > 0 else np.nan
        ),
    }


def robust_site_anomaly(values: pd.Series, site_ids: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.groupby(site_ids).transform("median")
    q25 = numeric.groupby(site_ids).transform(lambda x: x.quantile(0.25))
    q75 = numeric.groupby(site_ids).transform(lambda x: x.quantile(0.75))
    scale = (q75 - q25) / 1.349
    fallback = numeric.groupby(site_ids).transform("std")
    scale = scale.where(scale.gt(1e-8), fallback).where(lambda x: x.gt(1e-8), 1.0)
    return (numeric - median) / scale


def driver_frame_for_site(
    site_id: str,
    prepared_root: Path,
    driver_root: Path,
) -> pd.DataFrame | None:
    prepared_path = prepared_root / f"{site_id}.parquet"
    driver_path = driver_root / f"{site_id}.parquet"
    if not (prepared_path.exists() and driver_path.exists()):
        return None
    prepared = pd.read_parquet(
        prepared_path,
        columns=["timestamp", "x_63680"],
    )
    drivers = pd.read_parquet(driver_path)
    frame = prepared.merge(drivers, on="timestamp", how="inner", validate="one_to_one")
    native_rain = np.expm1(pd.to_numeric(frame["x_00045"], errors="coerce"))
    frame["precip_6h_native"] = native_rain.rolling(6, min_periods=1).sum()
    # ``x_63680`` in the prepared Common1000 layer is already log1p-transformed.
    # Applying log1p again would attenuate the turbidity anomaly twice.
    frame["turbidity_log1p"] = pd.to_numeric(
        frame["x_63680"], errors="coerce"
    )
    frame["flow_log1p"] = pd.to_numeric(frame["x_00060"], errors="coerce")
    frame["site_id"] = str(site_id)
    return frame[
        ["site_id", "timestamp", "precip_6h_native", "turbidity_log1p", "flow_log1p"]
    ]


def attribute_misses(
    evaluation: pd.DataFrame,
    prepared_root: Path,
    driver_root: Path,
    alpha: float,
    minimum_rows: int,
    model_label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    available_sites = sorted(
        set(evaluation["site_id"].astype(str))
        & {path.stem for path in driver_root.glob("*.parquet")}
    )
    drivers = [
        value
        for site_id in available_sites
        if (value := driver_frame_for_site(site_id, prepared_root, driver_root)) is not None
    ]
    if not drivers:
        return pd.DataFrame(), {"model": model_label, "status": "no_driver_sites"}
    driver = pd.concat(drivers, ignore_index=True)
    joined = evaluation.merge(
        driver,
        on=["site_id", "timestamp"],
        how="inner",
        validate="many_to_one",
    )
    raw_columns = ["precip_6h_native", "turbidity_log1p", "flow_log1p"]
    anomaly_columns = []
    for column in raw_columns:
        name = f"{column}_anomaly"
        joined[name] = robust_site_anomaly(joined[column], joined["site_id"])
        anomaly_columns.append(name)
    joined["abs_residual_anomaly"] = robust_site_anomaly(
        joined["abs_residual"], joined["site_id"]
    )
    complete = joined.dropna(subset=[*anomaly_columns, "abs_residual_anomaly"]).copy()
    if len(complete) < minimum_rows:
        return pd.DataFrame(), {
            "model": model_label,
            "status": "insufficient_rows",
            "joined_rows": int(len(complete)),
        }
    predictors = complete[anomaly_columns].astype(float)
    response = complete["abs_residual_anomaly"].to_numpy(dtype=float)
    model = Ridge(alpha=float(alpha)).fit(predictors, response)
    coefficient = pd.DataFrame(
        {
            "model": model_label,
            "driver": ["rainfall_6h", "turbidity", "flow"],
            "standardized_ridge_coefficient": model.coef_,
            "ridge_alpha": float(alpha),
            "rows": int(len(complete)),
            "sites": int(complete["site_id"].nunique()),
        }
    )
    top = complete["abs_residual"].ge(complete["abs_residual"].quantile(0.95))
    summary = {
        "model": model_label,
        "status": "complete",
        "joined_rows": int(len(complete)),
        "sites": int(complete["site_id"].nunique()),
        "r_squared": float(
            model.score(predictors, response)
        ),
        "top_residual_precip_6h_median_mm": float(
            complete.loc[top, "precip_6h_native"].median()
        ),
        "other_residual_precip_6h_median_mm": float(
            complete.loc[~top, "precip_6h_native"].median()
        ),
        "descriptive_test_covariates_only": True,
    }
    return coefficient, summary


def save_lag_plot(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for label, group in frame.groupby("model", sort=False):
        ax.plot(
            group["lead_hours"],
            group["enrichment"],
            marker="o",
            linewidth=2,
            label=label,
        )
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel("Event window shifted earlier (hours)")
    ax.set_ylabel("Registered-event enrichment in residual tail")
    ax.set_title("Do large residuals lead registered pH event windows?")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_lead_plot(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    groups = list(frame.groupby("model", sort=False))
    values = [
        group.loc[group["alerted"], "lead_hours"].dropna().to_numpy(dtype=float)
        for _, group in groups
    ]
    labels = [label for label, _ in groups]
    if values and any(len(value) for value in values):
        ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.set_ylabel("First-alert lead time (hours)")
    ax.set_title("Lead time among registered event windows with a pre-event alert")
    ax.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_compression_plot(
    model_frames: dict[str, pd.DataFrame],
    site_id: str,
    path: Path,
    tail_fraction: float,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(model_frames),
        figsize=(6.2 * len(model_frames), 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(model_frames) == 1:
        axes = [axes]
    for ax, (label, frame) in zip(axes, model_frames.items()):
        site = frame.loc[frame["site_id"].astype(str).eq(site_id)].copy()
        metric = compression_metrics(
            site["observed_native"],
            site["predicted_native"],
            tail_fraction,
        )
        if len(site) > 20_000:
            site = site.sample(20_000, random_state=42)
        ax.hexbin(
            site["observed_native"],
            site["predicted_native"],
            gridsize=55,
            mincnt=1,
            cmap="viridis",
        )
        low = float(
            np.nanmin([site["observed_native"].min(), site["predicted_native"].min()])
        )
        high = float(
            np.nanmax([site["observed_native"].max(), site["predicted_native"].max()])
        )
        ax.plot([low, high], [low, high], color="white", linestyle="--", linewidth=1.5)
        ax.set_title(
            f"{label}\n"
            f"overall slope={metric['overall_slope_predicted_on_observed']:.3f}, "
            f"range ratio={metric['predicted_to_observed_p05_p95_range_ratio']:.3f}"
        )
        ax.set_xlabel("Observed pH")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Predicted pH")
    fig.suptitle(f"Extreme compression at held-out site {site_id}", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = resolve(args.config)
    config = load_json(config_path)
    prediction_root = resolve(config["prediction_root"])
    prepared_root = resolve(config["prepared_ph_root"])
    driver_root = resolve(config["precipitation_driver_root"])
    output = resolve(config["output_root"])
    reports = output / "reports"
    figures = output / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    thresholds = []
    lead_parts = []
    lead_summaries = []
    lead_any_hour_parts = []
    lead_any_hour_summaries = []
    lag_parts = []
    compression_rows = []
    attribution_parts = []
    attribution_summaries = []
    evaluation_frames: dict[str, pd.DataFrame] = {}
    problems: list[str] = []

    for model_spec in config["models"]:
        run_id = str(model_spec["run_id"])
        label = str(model_spec["label"])
        calibration = read_prediction_split(
            prediction_root,
            run_id,
            str(config["calibration_split"]),
        )
        evaluation = read_prediction_split(
            prediction_root,
            run_id,
            str(config["evaluation_split"]),
        )
        evaluation_frames[label] = evaluation
        threshold_info = validation_threshold(
            calibration,
            float(config["residual_tail_fraction"]),
        )
        threshold_info.update({"run_id": run_id, "model": label})
        thresholds.append(threshold_info)
        threshold = float(threshold_info["threshold"])
        minimum_consecutive_hours = int(
            config.get("lead_alert_minimum_consecutive_hours", 3)
        )

        lead = lead_time_rows(
            evaluation,
            threshold,
            int(config["maximum_lead_hours"]),
            minimum_consecutive_hours,
        )
        lead["run_id"] = run_id
        lead["model"] = label
        lead_parts.append(lead)
        lead_summaries.append(
            summarise_lead_times(
                lead,
                label,
                [int(value) for value in config["reported_lead_hours"]],
            )
        )
        lead_any_hour = lead_time_rows(
            evaluation,
            threshold,
            int(config["maximum_lead_hours"]),
            1,
        )
        lead_any_hour["run_id"] = run_id
        lead_any_hour["model"] = label
        lead_any_hour_parts.append(lead_any_hour)
        lead_any_hour_summaries.append(
            summarise_lead_times(
                lead_any_hour,
                label,
                [int(value) for value in config["reported_lead_hours"]],
            )
        )
        lag_parts.append(
            lagged_enrichment_rows(
                evaluation,
                threshold,
                [int(value) for value in config["lagged_enrichment_hours"]],
                label,
            )
        )

        for scope, frame in (
            ("all_test_sites", evaluation),
            (
                f"site_{config['compression_site_id']}",
                evaluation.loc[
                    evaluation["site_id"]
                    .astype(str)
                    .eq(str(config["compression_site_id"]))
                ],
            ),
        ):
            metric = compression_metrics(
                frame["observed_native"],
                frame["predicted_native"],
                float(config["compression_tail_fraction"]),
            )
            compression_rows.append(
                {
                    "run_id": run_id,
                    "model": label,
                    "scope": scope,
                    **metric,
                }
            )

        coefficient, attribution_summary = attribute_misses(
            evaluation,
            prepared_root,
            driver_root,
            float(config["ridge_alpha"]),
            int(config["minimum_driver_rows"]),
            label,
        )
        if not coefficient.empty:
            attribution_parts.append(coefficient)
        attribution_summaries.append(attribution_summary)

    threshold_frame = pd.DataFrame(thresholds)
    lead_frame = pd.concat(lead_parts, ignore_index=True)
    lead_summary_frame = pd.DataFrame(lead_summaries)
    lead_any_hour_frame = pd.concat(lead_any_hour_parts, ignore_index=True)
    lead_any_hour_summary_frame = pd.DataFrame(lead_any_hour_summaries)
    lag_frame = pd.concat(lag_parts, ignore_index=True)
    compression_frame = pd.DataFrame(compression_rows)
    attribution_frame = (
        pd.concat(attribution_parts, ignore_index=True)
        if attribution_parts
        else pd.DataFrame()
    )
    attribution_summary_frame = pd.DataFrame(attribution_summaries)

    write_csv(reports / "validation_residual_thresholds.csv", threshold_frame)
    write_csv(reports / "event_lead_times.csv", lead_frame)
    write_csv(reports / "event_lead_time_summary.csv", lead_summary_frame)
    write_csv(reports / "event_lead_times_any_hour_sensitivity.csv", lead_any_hour_frame)
    write_csv(
        reports / "event_lead_time_summary_any_hour_sensitivity.csv",
        lead_any_hour_summary_frame,
    )
    write_csv(reports / "lagged_event_enrichment.csv", lag_frame)
    write_csv(reports / "extreme_compression_metrics.csv", compression_frame)
    write_csv(reports / "miss_attribution_coefficients.csv", attribution_frame)
    write_csv(reports / "miss_attribution_summary.csv", attribution_summary_frame)

    save_lag_plot(lag_frame, figures / "lagged_event_enrichment.png")
    save_lead_plot(lead_frame, figures / "event_lead_time_distribution.png")
    save_compression_plot(
        evaluation_frames,
        str(config["compression_site_id"]),
        figures / f"compression_{config['compression_site_id']}.png",
        float(config["compression_tail_fraction"]),
    )

    if threshold_frame["test_labels_used_for_threshold"].any():
        problems.append("test labels were used for residual threshold fitting")
    expected_lead_columns = {
        f"fraction_events_alerted_at_least_{int(value)}h"
        for value in config["reported_lead_hours"]
    }
    if not expected_lead_columns.issubset(set(lead_summary_frame.columns)):
        problems.append("requested lead-time proportions are incomplete")
    if not (
        compression_frame["scope"] == f"site_{config['compression_site_id']}"
    ).any():
        problems.append("requested compression site is absent")
    if lag_frame["enrichment"].isna().all():
        problems.append("lagged enrichment is entirely missing")

    site_compression = compression_frame[
        compression_frame["scope"].eq(f"site_{config['compression_site_id']}")
    ][
        [
            "model",
            "n",
            "overall_slope_predicted_on_observed",
            "low_tail_slope",
            "high_tail_slope",
            "predicted_to_observed_p05_p95_range_ratio",
        ]
    ]
    report = "\n".join(
        [
            "# Error-onset and extreme-compression report",
            "",
            "Residual thresholds were fitted on normal rows from held-out validation sites only. "
            "Registered events are incomplete weak labels; these results measure timing and enrichment, "
            "not event-classification accuracy.",
            "",
            "## Lead-time summary",
            "",
            f"The primary alert is the onset of the alert episode nearest the event "
            f"with at least {int(config.get('lead_alert_minimum_consecutive_hours', 3))} "
            "consecutive above-threshold hourly residuals. This avoids treating an "
            "unrelated isolated residual anywhere in the 48-hour lookback as an early warning.",
            "",
            markdown_table(lead_summary_frame),
            "",
            "### Single-hour alert sensitivity",
            "",
            markdown_table(lead_any_hour_summary_frame),
            "",
            "## Lagged enrichment",
            "",
            markdown_table(
                lag_frame[
                    [
                        "model",
                        "lead_hours",
                        "tail_rows",
                        "shifted_event_rows",
                        "enrichment",
                    ]
                ]
            ),
            "",
            f"## Extreme compression at site {config['compression_site_id']}",
            "",
            "Slopes regress predicted pH on observed pH. Values below 1 indicate regression to the mean.",
            "",
            markdown_table(site_compression),
            "",
            "## Miss attribution on the locally observed rainfall/flow subset",
            "",
            markdown_table(attribution_frame),
            "",
            "Attribution is descriptive: covariates are converted to within-site robust anomalies, "
            "and no event label is used to fit the ridge model.",
        ]
    ) + "\n"
    (reports / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    audit = {
        "status": "pass" if not problems else "fail",
        "threshold_fit_split": str(config["calibration_split"]),
        "test_labels_used_for_threshold": False,
        "models": len(config["models"]),
        "event_windows_analysed": int(len(lead_frame)),
        "lead_alert_minimum_consecutive_hours": int(
            config.get("lead_alert_minimum_consecutive_hours", 3)
        ),
        "compression_site": str(config["compression_site_id"]),
        "attribution_statuses": attribution_summary_frame[
            ["model", "status"]
        ].to_dict(orient="records"),
        "problems": problems,
        "elapsed_seconds": time.time() - started,
        "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(reports / "independent_audit.json", audit)
    if problems:
        raise RuntimeError(problems)
    write_json(output / "_SUCCESS", audit)
    print(report)


if __name__ == "__main__":
    main()
