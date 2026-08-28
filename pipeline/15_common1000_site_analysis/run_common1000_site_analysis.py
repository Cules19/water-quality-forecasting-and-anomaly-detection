from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from scipy.stats import spearmanr


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
FLOW_PROPERTIES = [
    "mean_flow",
    "flow_cv",
    "richards_baker_flashiness",
    "q95_low_flow",
    "q50_median_flow",
    "q95_q50_ratio",
]
STATIC_PROPERTIES = ["latitude", "longitude", "elevation"]
SITE_PROPERTIES = STATIC_PROPERTIES + FLOW_PROPERTIES
PREDICTION_COLUMNS = [
    "site_id",
    "network",
    "timestamp",
    "observed_ph",
    "predicted_ph",
    "abs_residual",
    "in_ph_event",
    "ph_event_direction",
    "event_id",
]
MODEL_COLORS = {
    "GRU_FILM_V2_N1009": "#1f77b4",
    "GRU_FILM_V2_N1009_NONEVENT": "#d62728",
    "GRU_S1_N1009_NONEVENT": "#ff7f0e",
    "LSTM_S1": "#2ca02c",
    "B4_H1_hgb_metadata": "#9467bd",
}
EVENT_COLORS = {"low": "#ef9a9a", "high": "#ffcc80", "mixed": "#ce93d8"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Common1000 site hydrology and model interpretation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_site_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(8) if text.isdigit() else text.upper()


def finite_number(value: Any) -> float:
    converted = pd.to_numeric(value, errors="coerce")
    return float(converted) if pd.notna(converted) and np.isfinite(converted) else np.nan


def positive_log10(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)
    mask = np.isfinite(numeric) & (numeric > 0)
    result[mask] = np.log10(numeric[mask])
    return result


def nested_numeric(mapping: dict[str, Any], candidates: Iterable[str]) -> float:
    for key in candidates:
        if key in mapping:
            value = finite_number(mapping[key])
            if np.isfinite(value):
                return value
    for parent in ("location", "coordinates", "metadata", "site_info"):
        child = mapping.get(parent)
        if isinstance(child, dict):
            value = nested_numeric(child, candidates)
            if np.isfinite(value):
                return value
    return np.nan


def build_static_metadata(config: dict[str, Any], cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    register = load_json(resolve(config["registry_path"]))
    register_lookup = {normalise_site_id(key): value for key, value in register.items()}
    stage11 = pd.read_csv(resolve(config["stage11_metadata_path"]), dtype={"site_id": "string"})
    stage11["site_id"] = stage11["site_id"].map(normalise_site_id)
    stage11 = stage11.set_index("site_id")
    candidate = pd.read_csv(resolve(config["candidate_manifest"]), dtype={"site_id": "string"})
    candidate["site_id"] = candidate["site_id"].map(normalise_site_id)
    candidate = candidate.drop_duplicates("site_id").set_index("site_id")
    rows: list[dict[str, Any]] = []
    for record in cohort.itertuples(index=False):
        site_id = normalise_site_id(record.site_id)
        entry = register_lookup.get(site_id, {})
        direct = bool(entry)
        latitude = nested_numeric(entry, ["latitude", "lat", "site_latitude"]) if direct else np.nan
        longitude = nested_numeric(entry, ["longitude", "lon", "lng", "site_longitude"]) if direct else np.nan
        elevation = nested_numeric(entry, ["elevation", "elevation_m", "altitude"]) if direct else np.nan
        fallback_used = False
        if site_id in stage11.index:
            fallback = stage11.loc[site_id]
            for name, current, fallback_col in (
                ("latitude", latitude, "meta_latitude"),
                ("longitude", longitude, "meta_longitude"),
                ("elevation", elevation, "meta_elevation"),
            ):
                if not np.isfinite(current):
                    value = finite_number(fallback.get(fallback_col, np.nan))
                    if np.isfinite(value):
                        if name == "latitude":
                            latitude = value
                        elif name == "longitude":
                            longitude = value
                        else:
                            elevation = value
                        fallback_used = True
        selected_file = str(candidate.loc[site_id, "selected_file"]) if site_id in candidate.index else ""
        network = str(getattr(record, "network", candidate.loc[site_id, "network"] if site_id in candidate.index else "unknown"))
        rows.append(
            {
                "site_id": site_id,
                "network": network,
                "latitude": latitude,
                "longitude": longitude,
                "elevation": elevation,
                "register_direct_match": direct,
                "stage11_alias_fallback_used": fallback_used,
                "static_metadata_complete": bool(np.isfinite(latitude) and np.isfinite(longitude) and np.isfinite(elevation)),
                "selected_source_file": selected_file,
                "source_has_00060_schema": "00060" in Path(selected_file).name,
            }
        )
    frame = pd.DataFrame(rows)
    audit = (
        frame.groupby("network", dropna=False)
        .agg(
            sites=("site_id", "size"),
            direct_register_matches=("register_direct_match", "sum"),
            complete_static_metadata=("static_metadata_complete", "sum"),
            files_with_00060_schema=("source_has_00060_schema", "sum"),
        )
        .reset_index()
    )
    return frame, audit


def flow_descriptors(daily: pd.Series) -> dict[str, Any]:
    daily = pd.to_numeric(daily, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    daily = daily[daily.ge(0)]
    if daily.empty:
        return {}
    mean_flow = float(daily.mean())
    std = float(daily.std(ddof=1)) if len(daily) > 1 else np.nan
    cv = std / mean_flow if mean_flow > 0 and np.isfinite(std) else np.nan
    gaps = daily.index.to_series().diff().dt.total_seconds().div(86400).to_numpy()
    differences = np.abs(np.diff(daily.to_numpy(dtype=float)))
    consecutive = np.isclose(gaps[1:], 1.0)
    numerator = float(differences[consecutive].sum()) if len(differences) else 0.0
    denominator = float(daily.sum())
    flashiness = numerator / denominator if denominator > 0 else np.nan
    q95 = float(daily.quantile(0.05))
    q50 = float(daily.quantile(0.50))
    return {
        "mean_flow": mean_flow,
        "flow_cv": cv,
        "richards_baker_flashiness": flashiness,
        "q95_low_flow": q95,
        "q50_median_flow": q50,
        "q95_q50_ratio": q95 / q50 if q50 > 0 else np.nan,
        "flow_daily_days": int(len(daily)),
        "flow_date_min": daily.index.min(),
        "flow_date_max": daily.index.max(),
        "flow_span_days": int((daily.index.max() - daily.index.min()).days + 1),
        "flow_daily_coverage_fraction": float(len(daily) / ((daily.index.max() - daily.index.min()).days + 1)),
        "flow_consecutive_pairs": int(consecutive.sum()),
    }


def build_flow_metrics(config: dict[str, Any], static: pd.DataFrame, output: Path, resume: bool) -> pd.DataFrame:
    destination = output / "reports" / "site_flow_metrics.csv"
    if resume and destination.exists():
        return pd.read_csv(destination, dtype={"site_id": "string"})
    model_ready_dir = resolve(config["model_ready_dir"])
    minimum_days = int(config["minimum_daily_flow_days"])
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(static.itertuples(index=False), start=1):
        row: dict[str, Any] = {
            "site_id": record.site_id,
            "network": record.network,
            "flow_source": "native_00060_training_period" if record.source_has_00060_schema else "unavailable",
            "flow_unit": "ft3/s (cfs)" if record.source_has_00060_schema else "",
            "flow_status": "no_00060_schema",
            "flow_source_rows": 0,
            "flow_valid_rows": 0,
            "train_boundary": pd.NaT,
        }
        for name in FLOW_PROPERTIES:
            row[name] = np.nan
        if not record.source_has_00060_schema:
            rows.append(row)
            continue
        source = resolve(record.selected_source_file)
        model_ready = model_ready_dir / f"{record.site_id}.parquet"
        try:
            if not source.exists() or not model_ready.exists():
                raise FileNotFoundError("source or model-ready file missing")
            train = pd.read_parquet(model_ready, columns=["timestamp", "split"])
            train_time = pd.to_datetime(train.loc[train["split"].eq("train"), "timestamp"], errors="coerce")
            boundary = train_time.max()
            del train, train_time
            if pd.isna(boundary):
                raise ValueError("training boundary unavailable")
            schema = pq.ParquetFile(source).schema_arrow.names
            timestamp_col = next((name for name in ("TIMESTAMP", "timestamp", "datetime", "DateTime") if name in schema), None)
            if timestamp_col is None or "00060" not in schema:
                raise ValueError("timestamp/00060 column missing")
            raw = pd.read_parquet(source, columns=[timestamp_col, "00060"])
            row["flow_source_rows"] = int(len(raw))
            timestamp = pd.to_datetime(raw[timestamp_col], errors="coerce", utc=True).dt.tz_localize(None)
            flow = pd.to_numeric(raw["00060"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid = timestamp.notna() & flow.notna() & flow.ge(0) & flow.gt(-999) & timestamp.le(pd.Timestamp(boundary))
            native = pd.DataFrame({"timestamp": timestamp[valid], "flow": flow[valid].astype(float)})
            row["flow_valid_rows"] = int(len(native))
            row["train_boundary"] = boundary
            if native.empty:
                row["flow_status"] = "no_valid_train_00060"
                rows.append(row)
                continue
            native["date"] = native["timestamp"].dt.floor("D")
            daily = native.groupby("date", sort=True, observed=True)["flow"].mean()
            metrics = flow_descriptors(daily)
            row.update(metrics)
            row["flow_status"] = "success" if int(metrics.get("flow_daily_days", 0)) >= minimum_days else "insufficient_daily_flow_days"
        except Exception as exc:  # per-site audit must remain complete
            row["flow_status"] = "error"
            row["flow_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if index % 25 == 0:
            print(f"[flow] {index}/{len(static)} sites", flush=True)
    frame = pd.DataFrame(rows)
    for column in ("mean_flow", "q95_low_flow", "q50_median_flow"):
        frame[f"log10_{column}"] = positive_log10(frame[column])
    write_csv(destination, frame)
    return frame


def exact_global_top_mask(values: np.ndarray, fraction: float) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return np.zeros(0, dtype=bool), np.nan
    n_take = max(1, int(math.ceil(len(values) * fraction)))
    selected = np.argpartition(values, len(values) - n_take)[len(values) - n_take :]
    mask = np.zeros(len(values), dtype=bool)
    mask[selected] = True
    return mask, float(values[selected].min())


def per_site_model_metrics(frame: pd.DataFrame, model: dict[str, Any], fraction: float, minimum_events: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = set(PREDICTION_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{model['model_id']} prediction columns missing: {sorted(missing)}")
    frame = frame.copy()
    frame["site_id"] = frame["site_id"].map(normalise_site_id)
    observed = pd.to_numeric(frame["observed_ph"], errors="coerce").to_numpy(dtype=np.float64)
    predicted = pd.to_numeric(frame["predicted_ph"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError(f"{model['model_id']} contains non-finite predictions")
    residual = observed - predicted
    absolute = np.abs(residual)
    event = frame["in_ph_event"].astype(bool).to_numpy()
    selected, threshold = exact_global_top_mask(absolute, fraction)
    frame["sq_error"] = np.square(residual)
    frame["absolute_error"] = absolute
    frame["event_numeric"] = event.astype(np.int8)
    frame["selected_numeric"] = selected.astype(np.int8)
    frame["selected_event_numeric"] = (selected & event).astype(np.int8)
    frame["observed_numeric"] = observed
    grouped = frame.groupby(["site_id", "network"], sort=True, observed=True)
    metrics = grouped.agg(
        n_rows=("observed_numeric", "size"),
        sum_squared_error=("sq_error", "sum"),
        sum_absolute_error=("absolute_error", "sum"),
        n_event_rows=("event_numeric", "sum"),
        n_global_top5_rows=("selected_numeric", "sum"),
        n_global_top5_event_rows=("selected_event_numeric", "sum"),
        observed_mean=("observed_numeric", "mean"),
        test_time_min=("timestamp", "min"),
        test_time_max=("timestamp", "max"),
    ).reset_index()
    mean_by_row = frame["site_id"].map(metrics.set_index("site_id")["observed_mean"])
    frame["sst_piece"] = np.square(observed - mean_by_row.to_numpy(dtype=float))
    sst = frame.groupby("site_id", sort=True, observed=True)["sst_piece"].sum()
    metrics["rmse"] = np.sqrt(metrics["sum_squared_error"] / metrics["n_rows"])
    metrics["mae"] = metrics["sum_absolute_error"] / metrics["n_rows"]
    metrics["r2"] = 1 - metrics["sum_squared_error"] / metrics["site_id"].map(sst)
    metrics["base_event_rate"] = metrics["n_event_rows"] / metrics["n_rows"]
    metrics["global_top5_event_rate"] = metrics["n_global_top5_event_rows"] / metrics["n_global_top5_rows"].replace(0, np.nan)
    metrics["top5_event_enrichment"] = metrics["global_top5_event_rate"] / metrics["base_event_rate"].replace(0, np.nan)
    metrics["enrichment_reliable_ge_min_events"] = metrics["n_event_rows"].ge(minimum_events) & metrics["n_global_top5_rows"].ge(5)
    metrics.insert(0, "model_id", str(model["model_id"]))
    metrics.insert(1, "model_label", str(model["label"]))
    summary = {
        "model_id": model["model_id"],
        "rows": int(len(frame)),
        "sites": int(frame["site_id"].nunique()),
        "global_top5_threshold_native_ph": threshold,
        "global_event_rate": float(event.mean()),
        "global_top5_event_rate": float(event[selected].mean()),
        "global_top5_event_enrichment": float(event[selected].mean() / event.mean()) if event.mean() > 0 else np.nan,
    }
    return metrics, summary


def build_model_metrics(config: dict[str, Any], output: Path, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination = output / "reports" / "site_model_metrics_long.csv"
    summary_path = output / "reports" / "model_top5_thresholds.csv"
    if resume and destination.exists() and summary_path.exists():
        return pd.read_csv(destination, dtype={"site_id": "string"}), pd.read_csv(summary_path)
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for model in config["performance_models"]:
        path = resolve(model["prediction_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=PREDICTION_COLUMNS)
        metrics, summary = per_site_model_metrics(
            frame,
            model,
            float(config["top_fraction"]),
            int(config["minimum_event_rows_for_reliable_enrichment"]),
        )
        rows.append(metrics)
        summaries.append(summary)
        print(f"[metrics] {model['model_id']}: {len(frame):,} rows", flush=True)
        del frame
    result = pd.concat(rows, ignore_index=True)
    summary = pd.DataFrame(summaries)
    write_csv(destination, result)
    write_csv(summary_path, summary)
    return result, summary


def make_wide_table(static: pd.DataFrame, flow: pd.DataFrame, metrics: pd.DataFrame, config: dict[str, Any], output: Path) -> pd.DataFrame:
    base = static.merge(flow.drop(columns=["network"], errors="ignore"), on="site_id", how="left", validate="one_to_one")
    value_columns = [
        "rmse",
        "mae",
        "r2",
        "n_rows",
        "n_event_rows",
        "base_event_rate",
        "n_global_top5_rows",
        "n_global_top5_event_rows",
        "global_top5_event_rate",
        "top5_event_enrichment",
        "enrichment_reliable_ge_min_events",
    ]
    for model in config["performance_models"]:
        model_id = str(model["model_id"])
        subset = metrics.loc[metrics["model_id"].eq(model_id), ["site_id", *value_columns]].copy()
        subset = subset.rename(columns={column: f"{model_id}__{column}" for column in value_columns})
        base = base.merge(subset, on="site_id", how="left", validate="one_to_one")
    write_csv(output / "reports" / "site_metadata_model_metrics_wide.csv", base)
    long = metrics.merge(base[["site_id", *STATIC_PROPERTIES, *FLOW_PROPERTIES, "flow_status"]], on="site_id", how="left", validate="many_to_one")
    write_csv(output / "reports" / "site_metadata_model_metrics_long.csv", long)
    return base


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    m = len(valid)
    adjusted = np.minimum.accumulate((valid.to_numpy() * m / np.arange(1, m + 1))[::-1])[::-1]
    result.loc[valid.index] = np.minimum(adjusted, 1.0)
    return result


def correlation_row(data: pd.DataFrame, x: str, y: str, model_id: str) -> dict[str, Any]:
    pair = data[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) >= 3 and pair[x].nunique() > 1 and pair[y].nunique() > 1:
        rho, p = spearmanr(pair[x], pair[y])
    else:
        rho, p = np.nan, np.nan
    return {"model_id": model_id, "site_property": x, "model_metric": y, "n_valid_sites": len(pair), "spearman_rho": rho, "p_value": p}


def run_correlations(wide: pd.DataFrame, config: dict[str, Any], output: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in config["performance_models"]:
        model_id = str(model["model_id"])
        for prop in SITE_PROPERTIES:
            for metric in ("rmse", "top5_event_enrichment"):
                outcome = f"{model_id}__{metric}"
                rows.append(correlation_row(wide, prop, outcome, model_id))
    result = pd.DataFrame(rows)
    result["fdr_adjusted_p_value"] = benjamini_hochberg(result["p_value"])
    result["significant_fdr_0_05"] = result["fdr_adjusted_p_value"].le(0.05)
    write_csv(output / "reports" / "spearman_site_property_model_metrics.csv", result)
    return result


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    destination = output / "figures" / f"{name}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "NA" if pd.isna(value) else f"{float(value):.6g}")
        else:
            display[column] = display[column].fillna("NA").astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def create_heatmaps(wide: pd.DataFrame, correlations: pd.DataFrame, config: dict[str, Any], output: Path) -> None:
    labels = {str(model["model_id"]): str(model["label"]) for model in config["performance_models"]}
    columns: list[tuple[str, str]] = []
    for model in config["performance_models"]:
        model_id = str(model["model_id"])
        columns.extend([(model_id, "rmse"), (model_id, "top5_event_enrichment")])
    matrix = np.full((len(SITE_PROPERTIES), len(columns)), np.nan)
    annotations = np.full(matrix.shape, "", dtype=object)
    for i, prop in enumerate(SITE_PROPERTIES):
        for j, (model_id, metric) in enumerate(columns):
            row = correlations[
                correlations["model_id"].eq(model_id)
                & correlations["site_property"].eq(prop)
                & correlations["model_metric"].eq(f"{model_id}__{metric}")
            ]
            if row.empty:
                continue
            value = float(row.iloc[0]["spearman_rho"])
            matrix[i, j] = value
            fdr = float(row.iloc[0]["fdr_adjusted_p_value"])
            star = "*" if np.isfinite(fdr) and fdr <= 0.05 else ""
            annotations[i, j] = f"{value:.2f}{star}\nn={int(row.iloc[0]['n_valid_sites'])}" if np.isfinite(value) else "NA"
    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=[f"{labels[mid]}\n{metric.replace('_', ' ')}" for mid, metric in columns],
        yticklabels=[name.replace("_", " ") for name in SITE_PROPERTIES],
        cbar_kws={"label": "Spearman rho"},
        ax=ax,
    )
    ax.set_title("Which site properties predict model accuracy and event detectability?\n* FDR-adjusted p <= 0.05")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    save_figure(fig, output, "site_property_model_metric_spearman_heatmap")

    primary_accuracy = next(str(m["model_id"]) for m in config["performance_models"] if m.get("primary_accuracy"))
    primary_event = next(str(m["model_id"]) for m in config["performance_models"] if m.get("primary_event"))
    full_columns = [
        *STATIC_PROPERTIES,
        *FLOW_PROPERTIES,
        f"{primary_accuracy}__rmse",
        f"{primary_event}__top5_event_enrichment",
    ]
    full_corr = wide[full_columns].corr(method="spearman", min_periods=10)
    write_csv(output / "reports" / "full_table_spearman_matrix.csv", full_corr.reset_index(names="variable"))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(full_corr, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", square=True, ax=ax)
    ax.set_title("Full site table Spearman correlation matrix")
    save_figure(fig, output, "full_table_spearman_correlation_heatmap")


def create_matrix_plot(wide: pd.DataFrame, config: dict[str, Any], output: Path) -> None:
    primary_accuracy = next(str(m["model_id"]) for m in config["performance_models"] if m.get("primary_accuracy"))
    primary_event = next(str(m["model_id"]) for m in config["performance_models"] if m.get("primary_event"))
    data = wide.copy()
    data["log10 mean flow"] = positive_log10(data["mean_flow"])
    data["log10 Q50"] = positive_log10(data["q50_median_flow"])
    data["log10 Q95"] = positive_log10(data["q95_low_flow"])
    renamed = data.rename(
        columns={
            "elevation": "elevation",
            "flow_cv": "flow CV",
            "richards_baker_flashiness": "RB flashiness",
            f"{primary_accuracy}__rmse": "RMSE",
            f"{primary_event}__top5_event_enrichment": "top-5 enrichment",
        }
    )
    columns = ["elevation", "log10 mean flow", "flow CV", "RB flashiness", "log10 Q50", "log10 Q95", "RMSE", "top-5 enrichment"]
    plot_data = renamed[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(plot_data) < 3:
        raise RuntimeError("Too few complete sites for the requested matrix plot")
    grid = sns.PairGrid(plot_data, vars=columns, corner=True, height=1.55, diag_sharey=False)
    grid.map_lower(sns.scatterplot, s=17, alpha=0.65, color="#2b6f8a", edgecolor="none")
    grid.map_diag(sns.histplot, bins=18, color="#2b6f8a", alpha=0.8)
    grid.figure.suptitle(f"Site properties and model outcomes (complete cases n={len(plot_data)})", y=1.01)
    grid.figure.savefig(output / "figures" / "site_property_model_metric_matrix_plot.png", dpi=220, bbox_inches="tight")
    plt.close(grid.figure)


def choose_contrasting_sites(wide: pd.DataFrame, config: dict[str, Any], output: Path) -> pd.DataFrame:
    accuracy = next(str(m["model_id"]) for m in config["performance_models"] if m.get("primary_accuracy"))
    event_model = next(str(m["model_id"]) for m in config["performance_models"] if m.get("primary_event"))
    rmse_col = f"{accuracy}__rmse"
    enrich_col = f"{event_model}__top5_event_enrichment"
    event_col = f"{event_model}__n_event_rows"
    pool = wide[
        wide["flow_status"].eq("success")
        & wide[rmse_col].notna()
        & wide[event_col].fillna(0).ge(3)
        & wide["richards_baker_flashiness"].notna()
    ].copy()
    if len(pool) < 4:
        pool = wide[wide["flow_status"].eq("success") & wide[rmse_col].notna()].copy()
    selections: list[tuple[str, pd.Series]] = []
    rules = [
        ("flashy", "richards_baker_flashiness", False),
        ("stable", "richards_baker_flashiness", True),
        ("best_prediction", rmse_col, True),
        ("strong_event_detectability", enrich_col, False),
    ]
    used: set[str] = set()
    for reason, column, ascending in rules:
        candidates = pool.loc[~pool["site_id"].isin(used) & pool[column].notna()].sort_values(column, ascending=ascending)
        if candidates.empty:
            continue
        selected = candidates.iloc[0]
        used.add(str(selected["site_id"]))
        selections.append((reason, selected))
    if len(selections) < 4:
        remaining = pool.loc[~pool["site_id"].isin(used)].sort_values("richards_baker_flashiness")
        for _, selected in remaining.iterrows():
            selections.append(("coverage_fill", selected))
            used.add(str(selected["site_id"]))
            if len(selections) == 4:
                break
    rows = []
    for reason, selected in selections[:4]:
        rows.append(
            {
                "site_id": selected["site_id"],
                "network": selected["network"],
                "selection_reason": reason,
                "richards_baker_flashiness": selected["richards_baker_flashiness"],
                "flow_cv": selected["flow_cv"],
                "mean_flow": selected["mean_flow"],
                "all_data_rmse": selected[rmse_col],
                "nonevent_top5_event_enrichment": selected[enrich_col],
                "event_rows": selected[event_col],
            }
        )
    result = pd.DataFrame(rows)
    write_csv(output / "reports" / "contrasting_sites.csv", result)
    return result


def event_spans(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    event = frame.loc[frame["in_ph_event"].astype(bool), ["timestamp", "ph_event_direction"]].copy()
    if event.empty:
        return []
    event = event.sort_values("timestamp")
    event["timestamp"] = pd.to_datetime(event["timestamp"])
    previous = event["timestamp"].shift()
    changed = event["ph_event_direction"].astype(str).ne(event["ph_event_direction"].astype(str).shift())
    new = previous.isna() | event["timestamp"].sub(previous).gt(pd.Timedelta(hours=1.5)) | changed
    event["group"] = new.cumsum()
    spans = []
    for _, group in event.groupby("group", sort=True):
        spans.append((group["timestamp"].min(), group["timestamp"].max() + pd.Timedelta(hours=1), str(group["ph_event_direction"].iloc[0])))
    return spans


def load_site_predictions(config: dict[str, Any], site_id: str) -> tuple[pd.DataFrame, dict[str, str]]:
    merged: pd.DataFrame | None = None
    labels: dict[str, str] = {}
    for model in config["performance_models"]:
        model_id = str(model["model_id"])
        labels[model_id] = str(model["label"])
        path = resolve(model["prediction_path"])
        frame = pd.read_parquet(path, filters=[("site_id", "==", site_id)], columns=PREDICTION_COLUMNS)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        renamed = frame.rename(columns={"predicted_ph": f"predicted__{model_id}"})
        keep = ["site_id", "timestamp", f"predicted__{model_id}"]
        if merged is None:
            keep.extend(["network", "observed_ph", "in_ph_event", "ph_event_direction", "event_id"])
            merged = renamed[keep].copy()
        else:
            check = renamed[["site_id", "timestamp", "observed_ph"]]
            joined = merged[["site_id", "timestamp", "observed_ph"]].merge(check, on=["site_id", "timestamp"], suffixes=("_base", "_new"), how="inner")
            if len(joined) != len(merged) or np.max(np.abs(joined["observed_ph_base"] - joined["observed_ph_new"])) > 1e-6:
                raise ValueError(f"Prediction keys/targets differ for {site_id}/{model_id}")
            merged = merged.merge(renamed[keep], on=["site_id", "timestamp"], how="inner", validate="one_to_one")
    if merged is None or merged.empty:
        raise ValueError(f"No predictions for selected site {site_id}")
    return merged.sort_values("timestamp").reset_index(drop=True), labels


def choose_test_window(frame: pd.DataFrame, event_model: str, days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    events = frame[frame["in_ph_event"].astype(bool)].copy()
    if not events.empty:
        events["score"] = np.abs(events["observed_ph"] - events[f"predicted__{event_model}"])
        centre = pd.Timestamp(events.sort_values("score", ascending=False).iloc[0]["timestamp"])
    else:
        centre = pd.Timestamp(frame.iloc[len(frame) // 2]["timestamp"])
    half = pd.Timedelta(days=days / 2)
    start, stop = centre - half, centre + half
    if start < frame["timestamp"].min():
        start, stop = frame["timestamp"].min(), frame["timestamp"].min() + pd.Timedelta(days=days)
    if stop > frame["timestamp"].max():
        start, stop = frame["timestamp"].max() - pd.Timedelta(days=days), frame["timestamp"].max()
    return start, stop


def create_predictor_figure(frame: pd.DataFrame, labels: dict[str, str], site_row: pd.Series, config: dict[str, Any]) -> plt.Figure:
    model_ids = [str(model["model_id"]) for model in config["performance_models"]]
    event_model = next(str(m["model_id"]) for m in config["performance_models"] if m.get("primary_event"))
    start, stop = choose_test_window(frame, event_model, int(config["predictor_window_days"]))
    window = frame[frame["timestamp"].between(start, stop)].copy()
    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, len(model_ids), height_ratios=[1.45, 1], hspace=0.33, wspace=0.25)
    axis = fig.add_subplot(grid[0, :])
    axis.plot(window["timestamp"], window["observed_ph"], color="black", linewidth=2.0, label="Observed 00400", zorder=5)
    for model_id in model_ids:
        axis.plot(window["timestamp"], window[f"predicted__{model_id}"], linewidth=1.2, alpha=0.9, color=MODEL_COLORS.get(model_id), label=labels[model_id])
    used_labels: set[str] = set()
    for begin, end, direction in event_spans(window):
        compact = direction.lower() if direction.lower() in EVENT_COLORS else "mixed"
        label = f"Regulatory event: {compact}" if compact not in used_labels else None
        used_labels.add(compact)
        axis.axvspan(begin, end, color=EVENT_COLORS[compact], alpha=0.25, label=label, zorder=0)
    axis.set_ylabel("pH (native units)")
    axis.set_title(
        f"Site {site_row.site_id} — {site_row.selection_reason}; "
        f"RB flashiness={site_row.richards_baker_flashiness:.3f}, flow CV={site_row.flow_cv:.3f}\n"
        f"Test window {start:%Y-%m-%d} to {stop:%Y-%m-%d}"
    )
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axis.tick_params(axis="x", rotation=25)
    axis.legend(ncol=3, fontsize=9, loc="best")
    axis.grid(alpha=0.2)

    rng = np.random.default_rng(int(config["random_seed"]))
    sample_index = np.arange(len(frame))
    if len(sample_index) > 5000:
        sample_index = np.sort(rng.choice(sample_index, 5000, replace=False))
    observed = frame["observed_ph"].to_numpy(dtype=float)[sample_index]
    limits = [float(np.nanmin(observed)), float(np.nanmax(observed))]
    for position, model_id in enumerate(model_ids):
        ax = fig.add_subplot(grid[1, position])
        predicted = frame[f"predicted__{model_id}"].to_numpy(dtype=float)[sample_index]
        low = min(limits[0], float(np.nanmin(predicted)))
        high = max(limits[1], float(np.nanmax(predicted)))
        ax.scatter(observed, predicted, s=7, alpha=0.22, color=MODEL_COLORS.get(model_id), rasterized=True)
        ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        rmse = float(np.sqrt(np.mean(np.square(observed - predicted))))
        denominator = float(np.sum(np.square(observed - observed.mean())))
        r2 = 1 - float(np.sum(np.square(observed - predicted))) / denominator if denominator > 0 else np.nan
        ax.set_title(f"{labels[model_id]}\nRMSE={rmse:.3f}, R²={r2:.3f}", fontsize=10)
        ax.set_xlabel("Observed pH")
        if position == 0:
            ax.set_ylabel("Predicted pH")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.grid(alpha=0.2)
    return fig


def create_predictor_plots(candidates: pd.DataFrame, config: dict[str, Any], output: Path) -> pd.DataFrame:
    audit_rows: list[dict[str, Any]] = []
    pdf_path = output / "figures" / "contrasting_site_predictor_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        for row in candidates.itertuples(index=False):
            frame, labels = load_site_predictions(config, str(row.site_id))
            fig = create_predictor_figure(frame, labels, pd.Series(row._asdict()), config)
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(output / "figures" / f"predictor_plot_{row.site_id}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)
            audit_rows.append(
                {
                    "site_id": row.site_id,
                    "prediction_rows": len(frame),
                    "event_rows": int(frame["in_ph_event"].sum()),
                    "test_time_min": frame["timestamp"].min(),
                    "test_time_max": frame["timestamp"].max(),
                    "model_count": len(config["performance_models"]),
                    "keys_exactly_aligned": True,
                }
            )
    audit = pd.DataFrame(audit_rows)
    write_csv(output / "reports" / "predictor_plot_audit.csv", audit)
    return audit


def write_report(
    output: Path,
    static: pd.DataFrame,
    flow: pd.DataFrame,
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    correlations: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    flow_success = flow["flow_status"].eq("success")
    significant = correlations[correlations["significant_fdr_0_05"]]
    strongest = correlations.dropna(subset=["spearman_rho"]).assign(abs_rho=lambda d: d["spearman_rho"].abs()).sort_values("abs_rho", ascending=False).head(12)
    report = f"""# Common1000 site properties and model-performance report

## Coverage

- Common cohort: {len(static):,} sites.
- Complete latitude/longitude/elevation: {int(static['static_metadata_complete'].sum()):,}/{len(static):,} sites.
- Native `00060` schema: {int(static['source_has_00060_schema'].sum()):,}/{len(static):,} sites.
- Successful train-period flow descriptors (minimum daily coverage applied): {int(flow_success.sum()):,}/{len(static):,} sites.
- Flow values are native USGS discharge in ft3/s (cfs); linked or registry-estimated flow was not substituted.

## Definitions

- Mean flow and flow CV use daily mean `00060` in the training period only.
- Richards-Baker flashiness is the sum of absolute changes between consecutive daily means divided by the sum of daily means.
- Q95 is the 5th percentile (flow exceeded 95% of the time); Q50 is the median.
- Per-site top-5% enrichment uses each model's global test-set absolute-residual top-5% threshold, then divides the site's selected event rate by its base event rate.
- Model outcomes and event fields are post-hoc outcomes only and were never supplied to a model.

## Model thresholds

{markdown_table(thresholds)}

## Spearman results

- Tested associations: {len(correlations):,}.
- FDR-significant associations: {len(significant):,}.

Strongest descriptive associations:

{markdown_table(strongest[['model_id','site_property','model_metric','n_valid_sites','spearman_rho','fdr_adjusted_p_value']])}

## Contrasting predictor sites

{markdown_table(candidates)}

## Interpretation boundary

The flow-property analysis is restricted to sites with native `00060`; in this Common1000 cohort those sites are USGS sites. Correlation is descriptive and does not establish causality. Sparse-event sites remain in the table but are flagged and should not drive event-detectability inference.
"""
    (output / "reports" / "FINAL_REPORT.md").write_text(report, encoding="utf-8")


def validate_outputs(output: Path, static: pd.DataFrame, flow: pd.DataFrame, metrics: pd.DataFrame, wide: pd.DataFrame, candidates: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if len(static) != 1009 or static["site_id"].nunique() != 1009:
        problems.append("static table is not the expected 1,009 unique sites")
    if len(flow) != len(static) or flow["site_id"].nunique() != len(static):
        problems.append("flow status table is incomplete")
    if len(wide) != len(static) or wide["site_id"].nunique() != len(static):
        problems.append("wide joined table is incomplete")
    if metrics.groupby("model_id")["site_id"].nunique().min() < 1000:
        problems.append("a model has fewer than 1,000 test sites")
    if len(candidates) != 4 or candidates["site_id"].nunique() != 4:
        problems.append("four unique contrasting sites were not selected")
    expected_figures = [
        "site_property_model_metric_spearman_heatmap.png",
        "full_table_spearman_correlation_heatmap.png",
        "site_property_model_metric_matrix_plot.png",
        "contrasting_site_predictor_plots.pdf",
    ] + [f"predictor_plot_{site}.png" for site in candidates["site_id"]]
    for name in expected_figures:
        if not (output / "figures" / name).exists():
            problems.append(f"missing figure {name}")
    return problems


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    output = resolve(config["output_root"])
    success = output / "_SUCCESS"
    if args.resume and success.exists():
        print(f"Already complete: {success}")
        return
    if success.exists():
        success.unlink()
    (output / "reports").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    started = time.time()
    cohort = pd.read_csv(resolve(config["cohort_path"]), dtype={"site_id": "string"})
    cohort["site_id"] = cohort["site_id"].map(normalise_site_id)
    static, static_audit = build_static_metadata(config, cohort)
    write_csv(output / "reports" / "site_static_metadata.csv", static)
    write_csv(output / "reports" / "static_and_flow_coverage_by_network.csv", static_audit)
    flow = build_flow_metrics(config, static, output, args.resume)
    metrics, thresholds = build_model_metrics(config, output, args.resume)
    wide = make_wide_table(static, flow, metrics, config, output)
    correlations = run_correlations(wide, config, output)
    create_heatmaps(wide, correlations, config, output)
    create_matrix_plot(wide, config, output)
    candidates = choose_contrasting_sites(wide, config, output)
    predictor_audit = create_predictor_plots(candidates, config, output)
    write_report(output, static, flow, metrics, thresholds, correlations, candidates)
    problems = validate_outputs(output, static, flow, metrics, wide, candidates)
    hashes = [
        {"role": "config", "path": str(config_path), "sha256": sha256_file(config_path)},
        {"role": "script", "path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        {"role": "site_register", "path": str(resolve(config["registry_path"])), "sha256": sha256_file(resolve(config["registry_path"]))},
        {"role": "candidate_manifest", "path": str(resolve(config["candidate_manifest"])), "sha256": sha256_file(resolve(config["candidate_manifest"]))},
        {"role": "stage11_metadata", "path": str(resolve(config["stage11_metadata_path"])), "sha256": sha256_file(resolve(config["stage11_metadata_path"]))},
    ]
    for model in config["performance_models"]:
        path = resolve(model["prediction_path"])
        hashes.append({"role": f"prediction_{model['model_id']}", "path": str(path), "sha256": sha256_file(path)})
    write_csv(output / "reports" / "artifact_hashes.csv", pd.DataFrame(hashes))
    summary = {
        "status": "success" if not problems else "failed_validation",
        "sites": len(static),
        "static_complete_sites": int(static["static_metadata_complete"].sum()),
        "native_00060_schema_sites": int(static["source_has_00060_schema"].sum()),
        "successful_flow_sites": int(flow["flow_status"].eq("success").sum()),
        "models": metrics["model_id"].drop_duplicates().tolist(),
        "prediction_sites_min": int(metrics.groupby("model_id")["site_id"].nunique().min()),
        "contrasting_sites": candidates["site_id"].tolist(),
        "predictor_plot_rows": int(predictor_audit["prediction_rows"].sum()),
        "validation_problems": problems,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "reports" / "run_summary.json", summary)
    if problems:
        raise RuntimeError(f"Site analysis validation failed: {problems}")
    write_json(success, {"status": "success", "sites": len(static), "completed_at_epoch": time.time()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
