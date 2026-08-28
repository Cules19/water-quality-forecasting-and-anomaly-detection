"""Build leakage-safe, site-disjoint pH/DO hourly data and baselines.

The source is the Common1000 native clean layer, never the already-scaled model
layer. Predictor gaps are filled causally (past only). Dynamic/target scaling and
baseline statistics are fitted exclusively on training sites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "test")
CALENDAR_FEATURES = ("sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value).__name__}")


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


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table formatter for audit reports."""
    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in frame.itertuples(index=False, name=None):
        values = []
        for value in record:
            if isinstance(value, float):
                values.append("" if not np.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def normalise_site_id(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        integer = str(int(float(text)))
        return integer.zfill(8) if len(integer) <= 8 else integer
    return text.upper()


def deterministic_key(site_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{site_id}".encode("utf-8")).hexdigest()


def assign_site_splits(sites: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.DataFrame:
    if abs(sum(float(fractions[name]) for name in SPLITS) - 1.0) > 1e-9:
        raise ValueError("Site split fractions must sum to one")
    pieces: list[pd.DataFrame] = []
    for network, group in sites.groupby("network", dropna=False, sort=True):
        group = group.copy()
        group["_key"] = group["site_id"].map(lambda value: deterministic_key(str(value), seed))
        group = group.sort_values("_key", kind="mergesort").reset_index(drop=True)
        n = len(group)
        n_train = int(math.floor(n * float(fractions["train"])))
        n_validation = int(math.floor(n * float(fractions["validation"])))
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_validation = max(1, min(n_validation, n - n_train - 1))
        labels = np.full(n, "test", dtype=object)
        labels[:n_train] = "train"
        labels[n_train : n_train + n_validation] = "validation"
        group["split"] = labels
        pieces.append(group.drop(columns="_key"))
    result = pd.concat(pieces, ignore_index=True)
    if result["site_id"].duplicated().any():
        raise ValueError("Duplicate site in site-disjoint split")
    if set(result["split"]) - set(SPLITS):
        raise ValueError("Unexpected split label")
    return result.sort_values(["split", "network", "site_id"], kind="mergesort").reset_index(drop=True)


def causal_fill(values: pd.Series, limit: int) -> pd.Series:
    """Past-only fill. This deliberately never uses bfill or interpolation."""
    return values.ffill(limit=limit)


def do_saturation_mg_l(temp_c: np.ndarray | pd.Series) -> np.ndarray:
    """Freshwater oxygen saturation (mg/L) at 1 atm (Benson-Krause).

    This is the same temperature-only equation used by the supervisor's
    ``models/lstm_film/data.py`` loader.
    """
    temperature_k = np.asarray(temp_c, dtype=np.float64) + 273.15
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        log_saturation = (
            -139.34411
            + 1.575701e5 / temperature_k
            - 6.642308e7 / temperature_k**2
            + 1.243800e10 / temperature_k**3
            - 8.621949e11 / temperature_k**4
        )
        saturation = np.exp(log_saturation)
    saturation[~np.isfinite(temperature_k)] = np.nan
    return saturation


def convert_ea_do_percent_to_mg_l(
    do_percent: pd.Series,
    temp_c: pd.Series,
) -> pd.Series:
    """Convert EA percent-saturation DO to mg/L using matched temperature."""
    percent = pd.to_numeric(do_percent, errors="coerce").astype("float64")
    temperature = pd.to_numeric(temp_c, errors="coerce").astype("float64")
    saturation = do_saturation_mg_l(temperature)
    converted = percent.to_numpy(dtype=np.float64) / 100.0 * saturation
    return pd.Series(converted, index=percent.index, dtype="float64")


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.DatetimeIndex(frame["timestamp"])
    hour = timestamp.hour.to_numpy(dtype=np.float32)
    day = timestamp.dayofyear.to_numpy(dtype=np.float32)
    month = timestamp.month.to_numpy(dtype=np.float32)
    frame["sin_hour"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    frame["cos_hour"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    frame["sin_day"] = np.sin(2 * np.pi * day / 365.25).astype("float32")
    frame["cos_day"] = np.cos(2 * np.pi * day / 365.25).astype("float32")
    frame["sin_month"] = np.sin(2 * np.pi * (month - 1) / 12).astype("float32")
    frame["cos_month"] = np.cos(2 * np.pi * (month - 1) / 12).astype("float32")
    return frame


def fixed_transform(code: str, values: pd.Series, unit_family: str, transforms: dict[str, str]) -> pd.Series:
    transform = transforms[code]
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    if transform == "identity":
        return numeric.astype("float32")
    if transform == "log1p_nonnegative":
        numeric = numeric.mask(numeric < 0)
        return np.log1p(numeric).astype("float32")
    if transform == "unit_family_safe_do":
        divisor = 100.0 if unit_family == "ea_percent_saturation_like_unverified" else 20.0
        return (numeric / divisor).astype("float32")
    if transform == "do_mg_l_div20":
        return (numeric / 20.0).astype("float32")
    raise ValueError(f"Unknown transform {transform} for {code}")


def load_events(path: Path, target_spec: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    columns = ["site_id", "parameter", "label", "incident_start", "incident_end"]
    events = pd.read_csv(path, usecols=columns, dtype={"site_id": "string"})
    events = events[events["parameter"].astype(str).str.strip().eq(target_spec["event_parameter_exact"])].copy()
    events["site_id"] = events["site_id"].map(normalise_site_id)
    events["incident_start"] = pd.to_datetime(events["incident_start"], errors="coerce")
    events["incident_end"] = pd.to_datetime(events["incident_end"], errors="coerce")
    events = events.dropna(subset=["site_id", "incident_start", "incident_end"])
    events = events[events["incident_end"].ge(events["incident_start"])].copy()
    events = events.sort_values(["site_id", "incident_start", "incident_end", "label"], kind="mergesort")
    events["event_id"] = np.arange(1, len(events) + 1, dtype=np.int64)
    result: dict[str, list[dict[str, Any]]] = {}
    for site_id, group in events.groupby("site_id", sort=False):
        result[str(site_id)] = group[["event_id", "incident_start", "incident_end", "label"]].to_dict("records")
    return result


def attach_events(hourly: pd.DataFrame, events: list[dict[str, Any]], target_code: str) -> pd.DataFrame:
    n = len(hourly)
    in_event = np.zeros(n, dtype=bool)
    event_id = np.full(n, -1, dtype=np.int64)
    event_label = np.full(n, "none", dtype=object)
    timestamp = pd.DatetimeIndex(hourly["timestamp"]).to_numpy(dtype="datetime64[ns]")
    for event in events:
        left = int(np.searchsorted(timestamp, np.datetime64(event["incident_start"], "ns"), side="left"))
        right = int(np.searchsorted(timestamp, np.datetime64(event["incident_end"], "ns"), side="right"))
        if right <= left:
            continue
        # ``in_event[left:right]`` is a NumPy view. Copy it before mutating the
        # parent array; otherwise every row in the current interval appears to
        # overlap after the assignment below and is incorrectly labelled mixed.
        overlap = in_event[left:right].copy()
        in_event[left:right] = True
        event_id[left:right] = np.where(overlap, -2, int(event["event_id"]))
        event_label[left:right] = np.where(overlap, "mixed", str(event["label"]))
    labels = pd.Series(event_label, dtype="string").str.lower()
    if target_code == "00400":
        # Object dtype is intentional: a fixed-width ``<U4`` array would
        # silently truncate ``mixed`` to ``mixe``.
        direction = np.where(
            labels.str.contains("low", na=False),
            "low",
            np.where(labels.str.contains("high", na=False), "high", "none"),
        ).astype(object)
    else:
        direction = np.where(in_event, "low", "none").astype(object)
    direction[event_id == -2] = "mixed"
    hourly["in_event"] = in_event
    hourly["event_id"] = event_id
    hourly["event_label"] = event_label
    hourly["event_direction"] = direction
    return hourly


def build_hourly_site(
    source: Path,
    site_id: str,
    target_spec: dict[str, Any],
    transforms: dict[str, str],
    fill_limit: int,
    events: list[dict[str, Any]],
    do_conversion: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = target_spec["target_code"]
    dynamic = list(target_spec["dynamic_codes"])
    do_conversion = do_conversion or {}
    conversion_enabled = bool(do_conversion.get("enabled", False))
    conversion_unit_family = str(
        do_conversion.get(
            "source_unit_family", "ea_percent_saturation_like_unverified"
        )
    )
    temperature_code = str(do_conversion.get("temperature_code", "00010"))
    needed = [
        "timestamp",
        "network",
        "do_unit_family",
        target,
        f"observed_{target}",
        *dynamic,
        *(f"observed_{code}" for code in dynamic),
    ]
    if conversion_enabled and "00300" in dynamic:
        needed.extend([temperature_code, f"observed_{temperature_code}"])
    needed = list(dict.fromkeys(needed))
    available = set(pq.read_schema(source).names)
    missing = [column for column in needed if column not in available]
    if missing:
        raise ValueError(f"Missing source columns for {site_id}: {missing}")
    raw = pd.read_parquet(source, columns=needed)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")
    network = str(raw["network"].dropna().iloc[0]) if raw["network"].notna().any() else "unknown"
    unit_family = str(raw["do_unit_family"].dropna().iloc[0]) if raw["do_unit_family"].notna().any() else "unknown"
    target_values = pd.to_numeric(raw[target], errors="coerce").where(raw[f"observed_{target}"].fillna(False).astype(bool))
    first_target = raw.loc[target_values.notna(), "timestamp"].min()
    last_target = raw.loc[target_values.notna(), "timestamp"].max()
    if pd.isna(first_target) or pd.isna(last_target):
        return pd.DataFrame(), {"site_id": site_id, "status": "excluded", "reason": "no_observed_target", "network": network, "do_unit_family": unit_family}
    raw = raw[raw["timestamp"].between(first_target, last_target)].copy()
    raw["hour"] = raw["timestamp"].dt.floor("h")
    aggregation: dict[str, pd.Series] = {target: target_values.loc[raw.index].groupby(raw["hour"]).mean()}
    observed_counts: dict[str, int] = {}
    conversion_audit: dict[str, Any] = {
        "do_conversion_applied": False,
        "do_conversion_method": "none",
        "do_conversion_input_rows": 0,
        "do_conversion_rows": 0,
        "do_conversion_missing_temperature_rows": 0,
        "do_conversion_input_percent_min": np.nan,
        "do_conversion_input_percent_median": np.nan,
        "do_conversion_input_percent_max": np.nan,
        "do_conversion_output_mg_l_min": np.nan,
        "do_conversion_output_mg_l_median": np.nan,
        "do_conversion_output_mg_l_max": np.nan,
    }
    for code in dynamic:
        values = pd.to_numeric(raw[code], errors="coerce").where(raw[f"observed_{code}"].fillna(False).astype(bool))
        if (
            code == "00300"
            and conversion_enabled
            and unit_family == conversion_unit_family
        ):
            temperature = pd.to_numeric(
                raw[temperature_code], errors="coerce"
            ).where(
                raw[f"observed_{temperature_code}"].fillna(False).astype(bool)
            )
            input_do = values.copy()
            values = convert_ea_do_percent_to_mg_l(input_do, temperature)
            finite_input = input_do[np.isfinite(input_do)]
            finite_output = values[np.isfinite(values)]
            conversion_audit = {
                "do_conversion_applied": True,
                "do_conversion_method": str(
                    do_conversion.get(
                        "method", "benson_krause_freshwater_1atm"
                    )
                ),
                "do_conversion_input_rows": int(input_do.notna().sum()),
                "do_conversion_rows": int(values.notna().sum()),
                "do_conversion_missing_temperature_rows": int(
                    (input_do.notna() & temperature.isna()).sum()
                ),
                "do_conversion_input_percent_min": (
                    float(finite_input.min()) if len(finite_input) else np.nan
                ),
                "do_conversion_input_percent_median": (
                    float(finite_input.median()) if len(finite_input) else np.nan
                ),
                "do_conversion_input_percent_max": (
                    float(finite_input.max()) if len(finite_input) else np.nan
                ),
                "do_conversion_output_mg_l_min": (
                    float(finite_output.min()) if len(finite_output) else np.nan
                ),
                "do_conversion_output_mg_l_median": (
                    float(finite_output.median()) if len(finite_output) else np.nan
                ),
                "do_conversion_output_mg_l_max": (
                    float(finite_output.max()) if len(finite_output) else np.nan
                ),
            }
        aggregation[code] = values.groupby(raw["hour"]).mean()
        observed_counts[code] = int(values.notna().sum())
    index = pd.date_range(first_target.floor("h"), last_target.floor("h"), freq="h")
    hourly = pd.DataFrame(index=index)
    hourly.index.name = "timestamp"
    hourly[f"target_{target}_native"] = aggregation[target].reindex(index).astype("float32")
    hourly[f"target_{target}_observed"] = hourly[f"target_{target}_native"].notna()
    for code in dynamic:
        observed_hourly = aggregation[code].reindex(index)
        transformed = fixed_transform(code, observed_hourly, unit_family, transforms)
        filled = causal_fill(transformed, fill_limit)
        hourly[f"x_{code}"] = filled.astype("float32")
        hourly[f"x_{code}_observed_hour"] = observed_hourly.notna()
        hourly[f"x_{code}_missing"] = filled.isna().astype("int8")
    if target == "00400":
        hourly["x_00300_percent_family"] = np.int8(unit_family == "ea_percent_saturation_like_unverified")
    hourly = hourly.reset_index()
    hourly["site_id"] = site_id
    hourly = add_calendar(hourly)
    hourly = attach_events(hourly, events, target)
    target_hours = int(hourly[f"target_{target}_observed"].sum())
    summary = {
        "site_id": site_id,
        "status": "built",
        "reason": "",
        "network": network,
        "do_unit_family": unit_family,
        "hourly_rows": len(hourly),
        "target_observed_hours": target_hours,
        "coverage_days": (index[-1] - index[0]).total_seconds() / 86400 if len(index) else 0,
        "start": index[0] if len(index) else None,
        "end": index[-1] if len(index) else None,
        "event_target_hours": int((hourly["in_event"] & hourly[f"target_{target}_observed"]).sum()),
        **conversion_audit,
        **{f"{code}_source_observed_rows": observed_counts[code] for code in dynamic},
        **{f"{code}_hourly_missing_after_causal_fill": int(hourly[f"x_{code}_missing"].sum()) for code in dynamic},
    }
    return hourly, summary


def select_smoke(cohort: pd.DataFrame, per_network: int, seed: int) -> pd.DataFrame:
    pieces = []
    for _, group in cohort.groupby("network", dropna=False, sort=True):
        group = group.copy()
        group["_key"] = group["site_id"].map(lambda value: deterministic_key(str(value), seed))
        pieces.append(group.sort_values("_key").head(per_network).drop(columns="_key"))
    return pd.concat(pieces, ignore_index=True)


def prepare_target_sites(
    target_name: str,
    target_spec: dict[str, Any],
    cohort: pd.DataFrame,
    config: dict[str, Any],
    output: Path,
    mode: str,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed = set(target_spec["allowed_do_unit_families"])
    selected = cohort[cohort["do_unit_family"].isin(allowed)].copy()
    if mode == "smoke":
        selected = select_smoke(selected, int(config["smoke_sites_per_network"]), int(config["random_seed"]))
    native_root = resolve(config["native_clean_root"])
    shard_root = output / "hourly_native_by_site" / target_name
    checkpoint_root = output / "checkpoints" / "hourly" / target_name
    event_map = load_events(resolve(config["events_path"]), target_spec)
    summaries: list[dict[str, Any]] = []
    for number, row in enumerate(selected.itertuples(index=False), start=1):
        site_id = str(row.site_id)
        source = native_root / f"{site_id}.parquet"
        shard = shard_root / f"{site_id}.parquet"
        checkpoint = checkpoint_root / f"{site_id}.json"
        if resume and checkpoint.exists() and shard.exists():
            summaries.append(load_json(checkpoint))
            continue
        try:
            hourly, summary = build_hourly_site(
                source,
                site_id,
                target_spec,
                config["fixed_dynamic_transforms"],
                int(config["causal_fill_limit_hours"]),
                event_map.get(normalise_site_id(site_id), []),
                config.get("do_conversion"),
            )
            if len(hourly):
                write_parquet(shard, hourly)
        except Exception as exc:
            summary = {
                "site_id": site_id,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "network": getattr(row, "network", "unknown"),
                "do_unit_family": getattr(row, "do_unit_family", "unknown"),
            }
            if shard.exists():
                shard.unlink()
        write_json(checkpoint, summary)
        summaries.append(summary)
        if number % 100 == 0:
            print(f"{target_name}: built/audited {number}/{len(selected)} sites", flush=True)
    summary_frame = pd.DataFrame(summaries)
    eligible = summary_frame[
        summary_frame["status"].eq("built")
        & pd.to_numeric(summary_frame["target_observed_hours"], errors="coerce").ge(int(config["minimum_target_hours"]))
        & pd.to_numeric(summary_frame["coverage_days"], errors="coerce").ge(float(config["minimum_coverage_days"]))
    ].copy()
    if len(eligible) < 3:
        raise RuntimeError(f"Only {len(eligible)} eligible {target_name} sites")
    split = assign_site_splits(eligible, config["site_split_fractions"], int(config["random_seed"]))
    return summary_frame, split


def streaming_training_statistics(
    target_name: str,
    target_spec: dict[str, Any],
    split: pd.DataFrame,
    output: Path,
) -> dict[str, Any]:
    dynamic = list(target_spec["dynamic_codes"])
    sums = {code: 0.0 for code in dynamic}
    sum_sq = {code: 0.0 for code in dynamic}
    counts = {code: 0 for code in dynamic}
    target_sum = 0.0
    target_sum_sq = 0.0
    target_count = 0
    seasonal_sum: dict[tuple[int, int], float] = defaultdict(float)
    seasonal_count: dict[tuple[int, int], int] = defaultdict(int)
    target_column = f"target_{target_spec['target_code']}_native"
    for site_id in split.loc[split["split"].eq("train"), "site_id"].astype(str):
        frame = pd.read_parquet(output / "hourly_native_by_site" / target_name / f"{site_id}.parquet")
        for code in dynamic:
            values = pd.to_numeric(frame[f"x_{code}"], errors="coerce").to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            counts[code] += len(values)
            sums[code] += float(values.sum())
            sum_sq[code] += float(np.square(values).sum())
        target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(target)
        values = target[valid]
        target_count += len(values)
        target_sum += float(values.sum())
        target_sum_sq += float(np.square(values).sum())
        timestamps = pd.DatetimeIndex(frame.loc[valid, "timestamp"])
        for month, hour, value in zip(timestamps.month, timestamps.hour, values):
            key = (int(month), int(hour))
            seasonal_sum[key] += float(value)
            seasonal_count[key] += 1
    feature_stats = {}
    for code in dynamic:
        mean = sums[code] / counts[code]
        variance = max(sum_sq[code] / counts[code] - mean**2, 0.0)
        scale = math.sqrt(variance)
        feature_stats[code] = {"mean": mean, "scale": scale if scale >= 1e-8 else 1.0, "fit_nonmissing_rows": counts[code]}
    target_mean = target_sum / target_count
    target_variance = max(target_sum_sq / target_count - target_mean**2, 0.0)
    target_scale = math.sqrt(target_variance)
    return {
        "fit_split": "train_sites_only",
        "imputation": "training-site global mean; missing indicator retained",
        "features": feature_stats,
        "target": {"mean": target_mean, "scale": target_scale if target_scale >= 1e-8 else 1.0, "fit_observed_rows": target_count},
        "seasonal_month_hour_mean": {
            f"{month:02d}-{hour:02d}": seasonal_sum[(month, hour)] / seasonal_count[(month, hour)]
            for month, hour in sorted(seasonal_sum)
        },
        "seasonal_month_hour_count": {
            f"{month:02d}-{hour:02d}": seasonal_count[(month, hour)] for month, hour in sorted(seasonal_count)
        },
    }


def valid_window_targets(frame: pd.DataFrame, target_code: str, dynamic: list[str], sequence_length: int, horizon: int, maximum_missing_fraction: float) -> np.ndarray:
    target = pd.to_numeric(frame[f"target_{target_code}_native"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(frame)
    indices = np.flatnonzero(np.isfinite(target))
    indices = indices[indices - horizon - sequence_length + 1 >= 0]
    if not len(indices):
        return indices
    missing = np.column_stack([frame[f"x_{code}_missing"].to_numpy(dtype=np.int8) for code in dynamic])
    hourly_missing = missing.sum(axis=1).astype(np.int64)
    prefix = np.concatenate(([0], np.cumsum(hourly_missing, dtype=np.int64)))
    end = indices - horizon
    start = end - sequence_length + 1
    missing_total = prefix[end + 1] - prefix[start]
    fraction = missing_total / float(sequence_length * len(dynamic))
    return indices[fraction <= maximum_missing_fraction]


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[mask].astype(np.float64)
    predicted = predicted[mask].astype(np.float64)
    if not len(observed):
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "nse": float("nan")}
    residual = predicted - observed
    denominator = float(np.square(observed - observed.mean()).sum())
    return {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "nse": float(1 - np.square(residual).sum() / denominator) if denominator > 0 else float("nan"),
    }


def build_baselines(
    target_name: str,
    target_spec: dict[str, Any],
    split: pd.DataFrame,
    scaler: dict[str, Any],
    config: dict[str, Any],
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_code = target_spec["target_code"]
    target_column = f"target_{target_code}_native"
    dynamic = list(target_spec["dynamic_codes"])
    sequence_length = int(config["sequence_length_hours"])
    maximum_missing = float(config["maximum_dynamic_missing_fraction_per_window"])
    global_mean = float(scaler["target"]["mean"])
    seasonal = scaler["seasonal_month_hour_mean"]
    overall_accumulator: dict[tuple[str, int, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    site_metric_rows: list[dict[str, Any]] = []
    for split_name in ("validation", "test"):
        for site_id in split.loc[split["split"].eq(split_name), "site_id"].astype(str):
            frame = pd.read_parquet(output / "hourly_native_by_site" / target_name / f"{site_id}.parquet")
            target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=np.float64)
            timestamps = pd.DatetimeIndex(frame["timestamp"])
            for horizon in config["forecast_horizons_hours"]:
                horizon = int(horizon)
                indices = valid_window_targets(frame, target_code, dynamic, sequence_length, horizon, maximum_missing)
                if not len(indices):
                    continue
                observed = target[indices]
                month = timestamps.month.to_numpy()[indices]
                hour = timestamps.hour.to_numpy()[indices]
                seasonal_prediction = np.asarray(
                    [seasonal.get(f"{int(m):02d}-{int(h):02d}", global_mean) for m, h in zip(month, hour)],
                    dtype=np.float64,
                )
                global_prediction = np.full(len(indices), global_mean, dtype=np.float64)
                source_index = indices - horizon
                persistence_prediction = target[source_index]
                persistence_eligible = np.isfinite(persistence_prediction)
                prediction = pd.DataFrame(
                    {
                        "site_id": site_id,
                        "timestamp": timestamps.to_numpy()[indices],
                        "split": split_name,
                        "horizon_hours": horizon,
                        "observed_native": observed.astype("float32"),
                        "prediction_global_train_mean": global_prediction.astype("float32"),
                        "prediction_train_seasonal_month_hour": seasonal_prediction.astype("float32"),
                        "prediction_persistence": persistence_prediction.astype("float32"),
                        "persistence_eligible": persistence_eligible,
                        "persistence_source_timestamp": timestamps.to_numpy()[source_index],
                        "in_event": frame["in_event"].to_numpy(dtype=bool)[indices],
                        "event_direction": frame["event_direction"].astype(str).to_numpy()[indices],
                        "event_id": frame["event_id"].to_numpy(dtype=np.int64)[indices],
                    }
                )
                prediction_path = output / "baseline_predictions_by_site" / target_name / f"h{horizon:02d}" / split_name / f"{site_id}.parquet"
                write_parquet(prediction_path, prediction)
                for model, predicted, mask in (
                    ("global_train_mean", global_prediction, np.ones(len(indices), dtype=bool)),
                    ("train_seasonal_month_hour", seasonal_prediction, np.ones(len(indices), dtype=bool)),
                    ("persistence_exact_horizon", persistence_prediction, persistence_eligible),
                ):
                    values = metric_values(observed[mask], predicted[mask])
                    site_metric_rows.append(
                        {
                            "target": target_name,
                            "split": split_name,
                            "horizon_hours": horizon,
                            "model": model,
                            "site_id": site_id,
                            "available_target_windows": len(indices),
                            "evaluated_windows": values["n"],
                            "coverage_fraction": values["n"] / len(indices),
                            **{key: value for key, value in values.items() if key != "n"},
                        }
                    )
                    overall_accumulator[(split_name, horizon, model)].append((observed[mask], predicted[mask]))
    site_metrics = pd.DataFrame(site_metric_rows)
    overall_rows = []
    for (split_name, horizon, model), values in overall_accumulator.items():
        observed = np.concatenate([item[0] for item in values])
        predicted = np.concatenate([item[1] for item in values])
        metric = metric_values(observed, predicted)
        subset = site_metrics[
            site_metrics["split"].eq(split_name)
            & site_metrics["horizon_hours"].eq(horizon)
            & site_metrics["model"].eq(model)
        ]
        overall_rows.append(
            {
                "target": target_name,
                "split": split_name,
                "horizon_hours": horizon,
                "model": model,
                **metric,
                "sites": int(subset["site_id"].nunique()),
                "macro_site_rmse": float(subset["rmse"].mean()),
                "macro_site_mae": float(subset["mae"].mean()),
                "macro_site_nse": float(subset["nse"].mean()),
                "mean_key_coverage": float(subset["coverage_fraction"].mean()),
            }
        )
    return pd.DataFrame(overall_rows), site_metrics


def validate_outputs(
    target_name: str,
    target_spec: dict[str, Any],
    split: pd.DataFrame,
    scaler: dict[str, Any],
    config: dict[str, Any],
    output: Path,
) -> list[str]:
    problems: list[str] = []
    split_sets = {name: set(split.loc[split["split"].eq(name), "site_id"].astype(str)) for name in SPLITS}
    for left in SPLITS:
        for right in SPLITS:
            if left < right and split_sets[left] & split_sets[right]:
                problems.append(f"{target_name}: overlap between {left} and {right}")
    if scaler.get("fit_split") != "train_sites_only":
        problems.append(f"{target_name}: scaler not marked train-sites-only")
    for path in (output / "baseline_predictions_by_site" / target_name).rglob("*.parquet"):
        frame = pd.read_parquet(path, columns=["site_id", "timestamp", "horizon_hours", "persistence_source_timestamp", "persistence_eligible"])
        if frame.duplicated(["site_id", "timestamp", "horizon_hours"]).any():
            problems.append(f"duplicate prediction key: {path}")
            break
        eligible = frame.loc[frame["persistence_eligible"].astype(bool)].copy()
        delta = pd.to_datetime(eligible["timestamp"]) - pd.to_datetime(eligible["persistence_source_timestamp"])
        expected = pd.to_timedelta(eligible["horizon_hours"], unit="h")
        if not delta.eq(expected).all():
            problems.append(f"persistence time alignment failure: {path}")
            break
    forbidden = set(config["forbidden_predictors"])
    if not forbidden:
        problems.append("forbidden predictor assertion is empty")
    return problems


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = resolve(args.config)
    config = load_json(config_path)
    output = resolve(config["output_root"]) / args.mode
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output exists; use --resume or choose a new version: {output}")
    for directory in ("reports", "hourly_native_by_site", "checkpoints", "baseline_predictions_by_site"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(resolve(config["cohort_path"]), dtype={"site_id": "string"})
    cohort["site_id"] = cohort["site_id"].map(normalise_site_id)
    if cohort["site_id"].duplicated().any():
        raise ValueError("Input cohort contains duplicate physical sites")

    all_summary = []
    all_splits = []
    overall_metrics = []
    site_metrics = []
    all_problems: list[str] = []
    for target_name, target_spec in config["targets"].items():
        summary, split = prepare_target_sites(target_name, target_spec, cohort, config, output, args.mode, args.resume)
        write_csv(output / "reports" / f"hourly_site_status_{target_name}.csv", summary)
        write_csv(output / "reports" / f"site_split_{target_name}.csv", split)
        scaler_path = output / "reports" / f"train_only_preprocessing_{target_name}.json"
        if args.resume and scaler_path.exists():
            scaler = load_json(scaler_path)
        else:
            scaler = streaming_training_statistics(target_name, target_spec, split, output)
            scaler["fixed_dynamic_transforms"] = {code: config["fixed_dynamic_transforms"][code] for code in target_spec["dynamic_codes"]}
            scaler["site_split_sha256"] = sha256(output / "reports" / f"site_split_{target_name}.csv")
            write_json(scaler_path, scaler)
        overall, per_site = build_baselines(target_name, target_spec, split, scaler, config, output)
        write_csv(output / "reports" / f"baseline_overall_metrics_{target_name}.csv", overall)
        write_csv(output / "reports" / f"baseline_site_metrics_{target_name}.csv", per_site)
        problems = validate_outputs(target_name, target_spec, split, scaler, config, output)
        all_problems.extend(problems)
        summary["target"] = target_name
        split["target"] = target_name
        all_summary.append(summary)
        all_splits.append(split)
        overall_metrics.append(overall)
        site_metrics.append(per_site)

    combined_summary = pd.concat(all_summary, ignore_index=True)
    combined_splits = pd.concat(all_splits, ignore_index=True)
    combined_overall = pd.concat(overall_metrics, ignore_index=True)
    combined_site = pd.concat(site_metrics, ignore_index=True)
    write_csv(output / "reports" / "hourly_site_status_all_targets.csv", combined_summary)
    write_csv(output / "reports" / "site_splits_all_targets.csv", combined_splits)
    write_csv(output / "reports" / "baseline_overall_metrics_all_targets.csv", combined_overall)
    write_csv(output / "reports" / "baseline_site_metrics_all_targets.csv", combined_site)

    split_counts = combined_splits.groupby(["target", "split"], dropna=False).size().rename("sites").reset_index()
    p1 = combined_overall[
        combined_overall["split"].eq("test") & combined_overall["horizon_hours"].eq(1)
    ].sort_values(["target", "rmse"])
    if bool(config.get("do_conversion", {}).get("enabled", False)):
        do_unit_note = (
            "EA dissolved oxygen is converted from percent saturation to mg/L "
            "with the supervisor's temperature-dependent Benson-Krause "
            "freshwater equation before hourly aggregation."
        )
    else:
        do_unit_note = (
            "EA dissolved oxygen is retained only as a unit-family-safe pH "
            "predictor and is excluded from the native-mg/L DO target experiment."
        )
    report = "\n".join(
        [
            "# Common1000 leave-sites-out preparation and baseline report",
            "",
            "All train/validation/test memberships are mutually exclusive by physical site. No site embedding is defined.",
            "Dynamic preprocessing and target/baseline statistics are fitted on training sites only.",
            "Predictor filling is causal forward-fill for at most six hours; no interpolation or backward-fill is used.",
            do_unit_note,
            "",
            "## Site counts",
            "",
            frame_to_markdown(split_counts),
            "",
            "## One-hour test baselines",
            "",
            frame_to_markdown(p1[["target", "model", "n", "sites", "rmse", "mae", "nse", "macro_site_rmse", "macro_site_nse", "mean_key_coverage"]]),
            "",
            "Persistence is a history-using reference and is evaluated only where the same site's target exists at exactly the requested lead time.",
        ]
    ) + "\n"
    (output / "reports" / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    audit = {
        "status": "pass" if not all_problems else "fail",
        "mode": args.mode,
        "site_disjoint": not any("overlap" in value for value in all_problems),
        "train_only_preprocessing": not any("scaler" in value for value in all_problems),
        "persistence_exact_alignment": not any("persistence" in value for value in all_problems),
        "problems": all_problems,
        "elapsed_seconds": time.time() - started,
        "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(output / "reports" / "independent_audit.json", audit)
    if all_problems:
        raise RuntimeError(f"Leave-sites-out validation failed: {all_problems}")
    write_json(output / "_SUCCESS", audit)
    print(report)


if __name__ == "__main__":
    main()
