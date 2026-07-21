from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
EVENTS_CSV = ROOT / "data" / "reference" / "regulatory_events.csv"
OUT_DIR = ROOT / "analysis_outputs" / "ph_dataset"

TARGET = "00400"
TIME_COL = "timestamp"
SITE_COL = "Site_ID"
INPUT_VARS = ["00010", "00095", "63680", "00060", "00065"]
PH_EVENT_LABELS = ["pH_exceedance_low", "pH_exceedance_high"]
MIN_ROWS = 10_000
TARGET_SITE_COUNT = 50
TARGET_VALID_MIN = 0.0
TARGET_VALID_MAX = 1.0


@dataclass
class Candidate:
    file_path: Path
    file_name: str
    site_id: str
    row_count: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    has_target: bool
    available_inputs: list[str]
    input_count: int
    has_timestamp: bool
    has_site_id: bool
    eligible: bool
    overlap_event_count: int
    overlap_event_duration_hrs: float


def normalise_site_id(value: object) -> str:
    raw = str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    return raw.zfill(8) if raw.isdigit() else raw


def site_from_filename(path: Path) -> str:
    return normalise_site_id(path.stem.split("_")[0])


def load_ph_events() -> pd.DataFrame:
    events = pd.read_csv(EVENTS_CSV, dtype={"site_id": str}, low_memory=False)
    events["site_id_norm"] = events["site_id"].map(normalise_site_id)
    events = events[events["label"].isin(PH_EVENT_LABELS)].copy()
    events["incident_start"] = pd.to_datetime(events["incident_start"], errors="coerce")
    events["incident_end"] = pd.to_datetime(events["incident_end"], errors="coerce")
    events = events.dropna(subset=["incident_start", "incident_end"])
    events["ph_event_label"] = np.where(events["label"].eq("pH_exceedance_low"), "low", "high")
    return events


def read_time_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        table = pq.read_table(path, columns=[TIME_COL])
        times = table.column(TIME_COL).to_pandas()
    except Exception:
        return pd.NaT, pd.NaT
    times = pd.to_datetime(times, errors="coerce")
    if times.notna().any():
        return times.min(), times.max()
    return pd.NaT, pd.NaT


def scan_candidates(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[Candidate] = []
    grouped_events = {site: group for site, group in events.groupby("site_id_norm")}

    for path in sorted(PROCESSED_DIR.glob("*.parquet")):
        parquet_file = pq.ParquetFile(path)
        columns = list(parquet_file.schema_arrow.names)
        site_id = site_from_filename(path)
        start_time, end_time = read_time_range(path) if TIME_COL in columns else (pd.NaT, pd.NaT)
        available_inputs = [col for col in INPUT_VARS if col in columns]
        has_target = TARGET in columns
        has_timestamp = TIME_COL in columns
        has_site_id = SITE_COL in columns
        row_count = int(parquet_file.metadata.num_rows)
        eligible = (
            has_target
            and has_timestamp
            and has_site_id
            and len(available_inputs) >= 3
            and row_count >= MIN_ROWS
            and pd.notna(start_time)
            and pd.notna(end_time)
        )

        overlap_count = 0
        overlap_duration = 0.0
        site_events = grouped_events.get(site_id)
        if eligible and site_events is not None:
            overlaps = site_events[
                (site_events["incident_end"] >= start_time)
                & (site_events["incident_start"] <= end_time)
            ]
            overlap_count = int(len(overlaps))
            overlap_duration = float(overlaps["duration_hrs"].fillna(0).sum())

        rows.append(
            Candidate(
                file_path=path,
                file_name=path.name,
                site_id=site_id,
                row_count=row_count,
                start_time=start_time,
                end_time=end_time,
                has_target=has_target,
                available_inputs=available_inputs,
                input_count=len(available_inputs),
                has_timestamp=has_timestamp,
                has_site_id=has_site_id,
                eligible=eligible,
                overlap_event_count=overlap_count,
                overlap_event_duration_hrs=overlap_duration,
            )
        )

    df = pd.DataFrame(
        [
            {
                "file_path": str(row.file_path),
                "file_name": row.file_name,
                "site_id": row.site_id,
                "row_count": row.row_count,
                "start_time": row.start_time,
                "end_time": row.end_time,
                "has_00400": row.has_target,
                "available_input_variables": ",".join(row.available_inputs),
                "input_variable_count": row.input_count,
                "has_timestamp": row.has_timestamp,
                "has_Site_ID": row.has_site_id,
                "eligible": row.eligible,
                "overlap_ph_event_count": row.overlap_event_count,
                "overlap_ph_event_duration_hrs": row.overlap_event_duration_hrs,
            }
            for row in rows
        ]
    )
    return df


def choose_one_file_per_site(inventory: pd.DataFrame) -> pd.DataFrame:
    eligible = inventory[inventory["eligible"]].copy()
    eligible["start_time"] = pd.to_datetime(eligible["start_time"])
    eligible["end_time"] = pd.to_datetime(eligible["end_time"])
    eligible["timespan_days"] = (eligible["end_time"] - eligible["start_time"]).dt.total_seconds() / 86400

    # One best candidate per site. Prefer more inputs, later data end, longer spans, then more rows.
    eligible = eligible.sort_values(
        ["site_id", "input_variable_count", "end_time", "timespan_days", "row_count"],
        ascending=[True, False, False, False, False],
    )
    return eligible.groupby("site_id", as_index=False).head(1).copy()


def select_sites(best_by_site: pd.DataFrame) -> pd.DataFrame:
    # Prefer the complete common predictor schema. Events are annotation only and
    # must not influence cohort selection.
    all_inputs = best_by_site[best_by_site["input_variable_count"] == len(INPUT_VARS)].copy()
    pool = all_inputs if len(all_inputs) >= TARGET_SITE_COUNT else best_by_site.copy()
    pool = pool.sort_values(
        [
            "input_variable_count",
            "timespan_days",
            "row_count",
            "site_id",
        ],
        ascending=[False, False, False, True],
    )
    selected = pool.head(TARGET_SITE_COUNT).copy()
    selected["selected_for_modelling"] = True
    return selected


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(df[TIME_COL], errors="coerce")
    df["hour"] = timestamp.dt.hour.astype("int16")
    df["month"] = timestamp.dt.month.astype("int16")
    df["day_of_year"] = timestamp.dt.dayofyear.astype("int16")
    df["sin_hour"] = np.sin(2 * math.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * math.pi * df["hour"] / 24.0)
    df["sin_day"] = np.sin(2 * math.pi * df["day_of_year"] / 365.25)
    df["cos_day"] = np.cos(2 * math.pi * df["day_of_year"] / 365.25)
    return df


def add_event_labels(df: pd.DataFrame, site_events: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(TIME_COL).copy()
    df["in_ph_event"] = np.int8(0)
    df["ph_event_label"] = "none"

    if site_events.empty:
        return df

    events = site_events.sort_values("incident_start")[
        ["incident_start", "incident_end", "ph_event_label"]
    ].copy()
    aligned = pd.merge_asof(
        df[[TIME_COL]].sort_values(TIME_COL),
        events,
        left_on=TIME_COL,
        right_on="incident_start",
        direction="backward",
    )
    in_event = aligned["incident_end"].notna() & (aligned[TIME_COL] <= aligned["incident_end"])
    df["in_ph_event"] = in_event.astype("int8").to_numpy()
    labels = aligned["ph_event_label"].where(in_event, "none").fillna("none")
    df["ph_event_label"] = labels.to_numpy()
    return df


def build_dataset(selected: pd.DataFrame, events: pd.DataFrame) -> dict[str, object]:
    output_path = OUT_DIR / "ph_model_dataset.parquet"
    if output_path.exists():
        output_path.unlink()

    selected_sites = set(selected["site_id"])
    selected_events = events[events["site_id_norm"].isin(selected_sites)].copy()
    selected_events = selected_events.merge(
        selected[["site_id", "start_time", "end_time"]],
        left_on="site_id_norm",
        right_on="site_id",
        how="inner",
    )
    selected_events = selected_events[
        (selected_events["incident_end"] >= selected_events["start_time"])
        & (selected_events["incident_start"] <= selected_events["end_time"])
    ].copy()
    selected_events = selected_events.drop(columns=["start_time", "end_time"])

    model_columns = [
        "site_id",
        TIME_COL,
        TARGET,
        *INPUT_VARS,
        "hour",
        "month",
        "day_of_year",
        "sin_hour",
        "cos_hour",
        "sin_day",
        "cos_day",
        "in_ph_event",
        "ph_event_label",
        "source_file",
    ]

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    total_event_rows = 0
    dropped_missing_target_or_time = 0
    dropped_invalid_target = 0
    dropped_missing_inputs = 0
    per_site_rows = []
    ph_min = math.inf
    ph_max = -math.inf

    for _, row in selected.iterrows():
        site_id = row["site_id"]
        path = Path(row["file_path"])
        read_columns = [TIME_COL, SITE_COL, TARGET, *INPUT_VARS]
        df = pd.read_parquet(path, columns=read_columns)
        if TIME_COL not in df.columns and df.index.name == TIME_COL:
            df = df.reset_index()
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df["site_id"] = site_id
        df["source_file"] = path.name

        for col in [TARGET, *INPUT_VARS]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        initial_rows = len(df)
        df = df.dropna(subset=[TIME_COL, TARGET])
        dropped_missing_target_or_time += initial_rows - len(df)
        valid_target = df[TARGET].between(TARGET_VALID_MIN, TARGET_VALID_MAX)
        dropped_invalid_target += int((~valid_target).sum())
        df = df[valid_target].copy()
        before_input_drop = len(df)
        df = df.dropna(subset=INPUT_VARS)
        dropped_missing_inputs += before_input_drop - len(df)
        df = df.drop_duplicates(subset=["site_id", TIME_COL])
        df = add_time_features(df)
        site_events = selected_events[selected_events["site_id_norm"] == site_id]
        df = add_event_labels(df, site_events)
        df = df[model_columns]
        df["site_id"] = df["site_id"].astype("string")
        df["ph_event_label"] = df["ph_event_label"].astype("string")
        df["source_file"] = df["source_file"].astype("string")
        df["in_ph_event"] = df["in_ph_event"].astype("int8")

        rows = len(df)
        event_rows = int(df["in_ph_event"].sum())
        total_rows += rows
        total_event_rows += event_rows
        per_site_rows.append(
            {
                "site_id": site_id,
                "model_rows": rows,
                "event_rows": event_rows,
                "dropped_missing_target_or_time": initial_rows - before_input_drop - int((~valid_target).sum()),
                "dropped_invalid_target": int((~valid_target).sum()),
                "dropped_missing_inputs": before_input_drop - rows,
            }
        )
        if df.empty:
            continue
        ph_min = min(ph_min, float(df[TARGET].min()))
        ph_max = max(ph_max, float(df[TARGET].max()))

        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table)

    if writer is not None:
        writer.close()

    selected_events_out = selected_events[
        [
            "site_id_norm",
            "parameter",
            "label",
            "ph_event_label",
            "severity",
            "description",
            "incident_start",
            "incident_end",
            "duration_hrs",
            "n_points",
        ]
    ].rename(columns={"site_id_norm": "site_id"})

    return {
        "output_path": output_path,
        "selected_events": selected_events_out,
        "total_rows": total_rows,
        "total_event_rows": total_event_rows,
        "dropped_missing_target_or_time": dropped_missing_target_or_time,
        "dropped_invalid_target": dropped_invalid_target,
        "dropped_missing_inputs": dropped_missing_inputs,
        "per_site_rows": pd.DataFrame(per_site_rows),
        "ph_min": ph_min,
        "ph_max": ph_max,
    }


def write_summary(
    inventory: pd.DataFrame,
    best_by_site: pd.DataFrame,
    selected: pd.DataFrame,
    dataset_info: dict[str, object],
) -> None:
    selected_rows = dataset_info["per_site_rows"]
    event_windows = dataset_info["selected_events"]
    model_path = dataset_info["output_path"]
    summary_path = OUT_DIR / "ph_dataset_summary.md"

    selected_start = pd.to_datetime(selected["start_time"]).min()
    selected_end = pd.to_datetime(selected["end_time"]).max()
    all_inputs_count = int((selected["input_variable_count"] == len(INPUT_VARS)).sum())

    text = f"""# pH Modelling Dataset Summary

Generated from `data/processed` on the restored Pollution_ID repository.

## Purpose

This dataset is the first-stage modelling table for the dissertation pH prediction task. It uses processed Parquet files rather than raw/clean CSV files, so the variables are already preprocessed. Initial inspection shows `00400` pH is scaled in processed files rather than stored directly on the 0-14 pH scale. This is suitable for model development, but raw-scale pH plots or pH-unit MAE should later use `data/clean` or an inverse transformation.

## Source Files

- Processed Parquet directory: `{PROCESSED_DIR}`
- Regulatory events file: `{EVENTS_CSV}`

## Candidate Site Scan

- Processed Parquet files scanned: {len(list(PROCESSED_DIR.glob('*.parquet')))}
- Eligible pH site-files: {int(inventory['eligible'].sum())}
- Unique eligible sites after choosing one best file per site: {best_by_site['site_id'].nunique()}
- Selected modelling sites: {selected['site_id'].nunique()}
- Selected sites with all five planned input variables: {all_inputs_count}

## Final Modelling Dataset

- Output file: `{model_path}`
- Rows: {dataset_info['total_rows']:,}
- Sites: {selected['site_id'].nunique()}
- Time range: {selected_start} to {selected_end}
- Target variable: `{TARGET}` processed/scaled pH
- Input variables: {', '.join(INPUT_VARS)}
- Time features: hour, month, day_of_year, sin_hour, cos_hour, sin_day, cos_day
- Rows inside pH regulatory event windows: {dataset_info['total_event_rows']:,}
- Matched pH event windows: {len(event_windows):,}
- Processed pH value range in final dataset: {dataset_info['ph_min']:.4f} to {dataset_info['ph_max']:.4f}
- Dropped rows with missing target/time: {dataset_info['dropped_missing_target_or_time']:,}
- Dropped rows with processed pH outside [{TARGET_VALID_MIN}, {TARGET_VALID_MAX}]: {dataset_info['dropped_invalid_target']:,}
- Dropped rows with missing input variables: {dataset_info['dropped_missing_inputs']:,}

## Important Limitations

- This is a processed/scaled dataset, not raw pH values.
- pH event labels are only for validation of prediction residuals, not the supervised target.
- Event windows are threshold-derived records and should be treated as weak validation rather than complete anomaly ground truth.
- The first version intentionally excludes lag features to keep the dataset simple and stable. Lag/rolling features can be added in the baseline modelling step.
"""
    summary_path.write_text(text, encoding="utf-8")
    selected_rows.to_csv(OUT_DIR / "ph_selected_site_row_counts.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events = load_ph_events()
    inventory = scan_candidates(events)
    best_by_site = choose_one_file_per_site(inventory)
    selected = select_sites(best_by_site)

    inventory["selected_for_modelling"] = inventory["file_path"].isin(set(selected["file_path"]))
    inventory.to_csv(OUT_DIR / "ph_processed_site_inventory.csv", index=False, encoding="utf-8-sig")

    selected.to_csv(OUT_DIR / "ph_selected_sites.csv", index=False, encoding="utf-8-sig")
    dataset_info = build_dataset(selected, events)
    dataset_info["selected_events"].to_csv(OUT_DIR / "ph_event_windows.csv", index=False, encoding="utf-8-sig")
    write_summary(inventory, best_by_site, selected, dataset_info)

    print(f"Output directory: {OUT_DIR}")
    print(f"Selected sites: {selected['site_id'].nunique()}")
    print(f"Rows: {dataset_info['total_rows']:,}")
    print(f"Rows in pH events: {dataset_info['total_event_rows']:,}")
    print(f"Matched pH event windows: {len(dataset_info['selected_events']):,}")
    print(f"Model dataset: {dataset_info['output_path']}")


if __name__ == "__main__":
    main()
