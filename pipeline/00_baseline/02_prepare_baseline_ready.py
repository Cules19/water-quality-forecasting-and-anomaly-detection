from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
INPUT_PATH = ROOT / "analysis_outputs" / "ph_dataset" / "ph_model_dataset.parquet"
OUT_DIR = ROOT / "analysis_outputs" / "ph_baseline_ready"

TARGET = "00400"
SITE_COL = "site_id"
TIME_COL = "timestamp"
SPLIT_COL = "split"
EVENT_COL = "in_ph_event"
EVENT_LABEL_COL = "ph_event_label"

MAIN_FEATURES = [
    "00010",
    "00095",
    "63680",
    "00060",
    "00065",
    "sin_hour",
    "cos_hour",
    "sin_day",
    "cos_day",
    "month",
    SITE_COL,
]
NUMERIC_MAIN_FEATURES = [feature for feature in MAIN_FEATURES if feature != SITE_COL]
OPTIONAL_DIAGNOSTIC_COLUMNS = ["hour", "day_of_year"]
VALIDATION_ONLY_COLUMNS = [EVENT_COL, EVENT_LABEL_COL]
FORBIDDEN_COLUMNS = [
    EVENT_COL,
    EVENT_LABEL_COL,
    "label",
    "severity",
    "description",
    "incident_start",
    "incident_end",
    "duration_hrs",
    "n_points",
]
BASELINE_COLUMNS = [
    SITE_COL,
    TIME_COL,
    SPLIT_COL,
    TARGET,
    "00010",
    "00095",
    "63680",
    "00060",
    "00065",
    "sin_hour",
    "cos_hour",
    "sin_day",
    "cos_day",
    "month",
    "hour",
    "day_of_year",
    EVENT_COL,
    EVENT_LABEL_COL,
]
INPUT_COLUMNS = [col for col in BASELINE_COLUMNS if col != SPLIT_COL]


def normalise_site_id(value: object) -> str:
    raw = str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    return raw.zfill(8) if raw.isdigit() else raw


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{value:.6g}")
        else:
            display[col] = display[col].astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.to_numpy()
    ]
    return "\n".join([header, separator, *body])


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_PATH}")

    available_columns = pq.ParquetFile(INPUT_PATH).schema_arrow.names
    required_missing = [col for col in INPUT_COLUMNS if col not in available_columns]
    if required_missing:
        raise ValueError(f"Input dataset is missing required columns: {required_missing}")

    df = pd.read_parquet(INPUT_PATH, columns=INPUT_COLUMNS)
    df[SITE_COL] = df[SITE_COL].map(normalise_site_id).astype("string")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce").fillna(0).astype("int8")
    df[EVENT_LABEL_COL] = df[EVENT_LABEL_COL].fillna("none").astype("string")
    for col in [TARGET, *NUMERIC_MAIN_FEATURES, *OPTIONAL_DIAGNOSTIC_COLUMNS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values([SITE_COL, TIME_COL]).reset_index(drop=True)
    return df, available_columns


def add_temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    site_sizes = df.groupby(SITE_COL)[TIME_COL].transform("size")
    site_pos = df.groupby(SITE_COL).cumcount()
    train_end = (site_sizes * 0.70).astype("int64")
    val_end = (site_sizes * 0.85).astype("int64")
    split = np.where(site_pos < train_end, "train", np.where(site_pos < val_end, "val", "test"))
    df = df.copy()
    df[SPLIT_COL] = pd.Categorical(split, categories=["train", "val", "test"], ordered=True)
    return df[BASELINE_COLUMNS]


def run_basic_checks(df: pd.DataFrame, available_columns: list[str]) -> pd.DataFrame:
    required_columns_exist = all(col in available_columns for col in INPUT_COLUMNS)
    feature_columns_exist = all(col in df.columns for col in MAIN_FEATURES)
    target_numeric = pd.api.types.is_numeric_dtype(df[TARGET])
    timestamp_parse_failures = int(df[TIME_COL].isna().sum())
    duplicate_count = int(df.duplicated(subset=[SITE_COL, TIME_COL]).sum())
    missing_target = int(df[TARGET].isna().sum())
    missing_numeric_features = int(df[NUMERIC_MAIN_FEATURES].isna().sum().sum())
    missing_site_feature = int(df[SITE_COL].isna().sum())
    forbidden_in_features = sorted(set(MAIN_FEATURES).intersection(FORBIDDEN_COLUMNS + [TARGET, TIME_COL]))

    checks = [
        {
            "check_name": "required_columns_exist",
            "status": "PASS" if required_columns_exist else "FAIL",
            "value": str(required_columns_exist),
            "notes": f"Missing: {[col for col in INPUT_COLUMNS if col not in available_columns]}",
        },
        {
            "check_name": "timestamp_parse_success",
            "status": "PASS" if timestamp_parse_failures == 0 else "FAIL",
            "value": timestamp_parse_failures,
            "notes": "Number of rows where timestamp could not be parsed.",
        },
        {
            "check_name": "target_numeric",
            "status": "PASS" if target_numeric else "FAIL",
            "value": str(df[TARGET].dtype),
            "notes": "Target should be processed/scaled pH.",
        },
        {
            "check_name": "feature_columns_exist",
            "status": "PASS" if feature_columns_exist else "FAIL",
            "value": str(feature_columns_exist),
            "notes": f"Official features: {MAIN_FEATURES}",
        },
        {
            "check_name": "missing_target_count",
            "status": "PASS" if missing_target == 0 else "FAIL",
            "value": missing_target,
            "notes": "Target missing values.",
        },
        {
            "check_name": "missing_feature_count",
            "status": "PASS" if missing_numeric_features == 0 and missing_site_feature == 0 else "FAIL",
            "value": missing_numeric_features + missing_site_feature,
            "notes": "Official baseline numeric features plus site_id.",
        },
        {
            "check_name": "validation_columns_exist",
            "status": "PASS" if all(col in df.columns for col in VALIDATION_ONLY_COLUMNS) else "FAIL",
            "value": str(all(col in df.columns for col in VALIDATION_ONLY_COLUMNS)),
            "notes": "Validation-only columns must be retained but not used as model inputs.",
        },
        {
            "check_name": "duplicate_site_timestamp_count",
            "status": "PASS" if duplicate_count == 0 else "FAIL",
            "value": duplicate_count,
            "notes": "Duplicate site_id + timestamp rows.",
        },
        {
            "check_name": "n_sites",
            "status": "PASS" if df[SITE_COL].nunique() == 50 else "WARN",
            "value": df[SITE_COL].nunique(),
            "notes": "Expected 50 sites unless the upstream dataset changed.",
        },
        {
            "check_name": "n_rows",
            "status": "PASS" if len(df) > 0 else "FAIL",
            "value": len(df),
            "notes": "Total baseline-ready rows.",
        },
        {
            "check_name": "forbidden_features_not_in_main_feature_list",
            "status": "PASS" if not forbidden_in_features else "FAIL",
            "value": forbidden_in_features if forbidden_in_features else "None",
            "notes": "Event labels, timestamp, and target must not be model inputs.",
        },
    ]
    basic = pd.DataFrame(checks)
    basic.to_csv(OUT_DIR / "baseline_ready_basic_check.csv", index=False, encoding="utf-8-sig")
    return basic


def make_split_summaries(df: pd.DataFrame, prefix: str = "ph") -> tuple[pd.DataFrame, pd.DataFrame]:
    temp = df.copy()
    temp["is_low_event"] = (temp[EVENT_COL].eq(1) & temp[EVENT_LABEL_COL].eq("low")).astype("int8")
    temp["is_high_event"] = (temp[EVENT_COL].eq(1) & temp[EVENT_LABEL_COL].eq("high")).astype("int8")

    by_site = temp.groupby([SITE_COL, SPLIT_COL], observed=False).agg(
        n_rows=(TARGET, "size"),
        start_time=(TIME_COL, "min"),
        end_time=(TIME_COL, "max"),
        n_event_rows=(EVENT_COL, "sum"),
        n_low_event_rows=("is_low_event", "sum"),
        n_high_event_rows=("is_high_event", "sum"),
        target_min=(TARGET, "min"),
        target_max=(TARGET, "max"),
        target_mean=(TARGET, "mean"),
        target_std=(TARGET, "std"),
    )
    by_site["event_ratio"] = by_site["n_event_rows"] / by_site["n_rows"]
    by_site = by_site.reset_index()[
        [
            SITE_COL,
            SPLIT_COL,
            "n_rows",
            "start_time",
            "end_time",
            "n_event_rows",
            "event_ratio",
            "n_low_event_rows",
            "n_high_event_rows",
            "target_min",
            "target_max",
            "target_mean",
            "target_std",
        ]
    ]

    total_rows = len(temp)
    overall = temp.groupby(SPLIT_COL, observed=False).agg(
        n_rows=(TARGET, "size"),
        n_sites=(SITE_COL, "nunique"),
        n_event_rows=(EVENT_COL, "sum"),
        n_low_event_rows=("is_low_event", "sum"),
        n_high_event_rows=("is_high_event", "sum"),
        target_min=(TARGET, "min"),
        target_max=(TARGET, "max"),
        target_mean=(TARGET, "mean"),
        target_std=(TARGET, "std"),
    )
    overall["row_ratio"] = overall["n_rows"] / total_rows
    overall["event_ratio"] = overall["n_event_rows"] / overall["n_rows"]
    overall = overall.reset_index()[
        [
            SPLIT_COL,
            "n_rows",
            "row_ratio",
            "n_sites",
            "n_event_rows",
            "event_ratio",
            "n_low_event_rows",
            "n_high_event_rows",
            "target_min",
            "target_max",
            "target_mean",
            "target_std",
        ]
    ]

    by_site.to_csv(OUT_DIR / f"{prefix}_split_summary_by_site.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUT_DIR / f"{prefix}_split_summary_overall.csv", index=False, encoding="utf-8-sig")
    return by_site, overall


def save_feature_set() -> dict[str, object]:
    feature_set = {
        "target": TARGET,
        "main_features": MAIN_FEATURES,
        "optional_diagnostic_columns": OPTIONAL_DIAGNOSTIC_COLUMNS,
        "validation_only_columns": VALIDATION_ONLY_COLUMNS,
        "forbidden_columns": FORBIDDEN_COLUMNS,
    }
    (OUT_DIR / "feature_set_step2_1.json").write_text(
        json.dumps(feature_set, indent=2),
        encoding="utf-8",
    )
    return feature_set


def site_event_category(event_ratio: float) -> str:
    if event_ratio >= 0.50:
        return "EXTREME_EVENT_RATIO"
    if event_ratio >= 0.25:
        return "HIGH_EVENT_RATIO"
    if event_ratio >= 0.10:
        return "MODERATE_EVENT_RATIO"
    return "LOW_EVENT_RATIO"


def select_10site_subset(by_site_summary: pd.DataFrame) -> pd.DataFrame:
    site_total = by_site_summary.groupby(SITE_COL).agg(
        n_rows=("n_rows", "sum"),
        n_event_rows=("n_event_rows", "sum"),
        n_low_event_rows=("n_low_event_rows", "sum"),
        n_high_event_rows=("n_high_event_rows", "sum"),
        start_time=("start_time", "min"),
        end_time=("end_time", "max"),
        target_mean=("target_mean", "mean"),
    )
    site_total["event_ratio"] = site_total["n_event_rows"] / site_total["n_rows"]
    test_summary = by_site_summary[by_site_summary[SPLIT_COL] == "test"].set_index(SITE_COL)
    site_total["test_rows"] = test_summary["n_rows"]
    site_total["test_event_rows"] = test_summary["n_event_rows"]
    site_total["test_low_event_rows"] = test_summary["n_low_event_rows"]
    site_total["test_high_event_rows"] = test_summary["n_high_event_rows"]
    site_total["event_ratio_category"] = site_total["event_ratio"].map(site_event_category)
    median_rows = float(site_total["n_rows"].median())
    site_total["size_distance_from_median"] = (
        (site_total["n_rows"] - median_rows).abs() / median_rows
    )
    site_total["has_test_event"] = site_total["test_event_rows"] > 0
    site_total = site_total.reset_index()

    quotas = {
        "EXTREME_EVENT_RATIO": 2,
        "HIGH_EVENT_RATIO": 2,
        "MODERATE_EVENT_RATIO": 2,
        "LOW_EVENT_RATIO": 4,
    }
    selected_ids: list[str] = []
    selection_notes: dict[str, str] = {}

    for category, quota in quotas.items():
        pool = site_total[
            (site_total["event_ratio_category"] == category)
            & (~site_total[SITE_COL].isin(selected_ids))
        ].copy()
        if pool.empty:
            continue
        pool = pool.sort_values(
            ["has_test_event", "size_distance_from_median", "test_event_rows", "n_event_rows"],
            ascending=[False, True, False, False],
        )
        chosen = pool.head(quota)
        selected_ids.extend(chosen[SITE_COL].tolist())
        for site_id in chosen[SITE_COL]:
            selection_notes[site_id] = f"Quota selection from {category}."

    if len(selected_ids) < 10:
        remaining = site_total[~site_total[SITE_COL].isin(selected_ids)].copy()
        remaining = remaining.sort_values(
            ["has_test_event", "size_distance_from_median", "test_event_rows", "n_event_rows"],
            ascending=[False, True, False, False],
        )
        fill = remaining.head(10 - len(selected_ids))
        selected_ids.extend(fill[SITE_COL].tolist())
        for site_id in fill[SITE_COL]:
            selection_notes[site_id] = "Backfill to reach 10 sites."

    selected = site_total[site_total[SITE_COL].isin(selected_ids)].copy()
    selected["selection_order"] = selected[SITE_COL].map({site_id: idx + 1 for idx, site_id in enumerate(selected_ids)})
    selected["selection_note"] = selected[SITE_COL].map(selection_notes)
    selected = selected.sort_values("selection_order")
    selected.to_csv(OUT_DIR / "ph_10site_selected_sites.csv", index=False, encoding="utf-8-sig")
    return selected


def write_report(
    df: pd.DataFrame,
    basic_checks: pd.DataFrame,
    full_by_site: pd.DataFrame,
    full_overall: pd.DataFrame,
    selected_sites: pd.DataFrame,
    subset_overall: pd.DataFrame,
) -> None:
    test_full = full_overall[full_overall[SPLIT_COL] == "test"].iloc[0]
    val_full = full_overall[full_overall[SPLIT_COL] == "val"].iloc[0]
    train_full = full_overall[full_overall[SPLIT_COL] == "train"].iloc[0]
    test_by_site = full_by_site[full_by_site[SPLIT_COL] == "test"]
    n_zero_test_events = int((test_by_site["n_event_rows"] == 0).sum())
    n_low_test_sites = int((test_by_site["n_low_event_rows"] > 0).sum())
    n_high_test_sites = int((test_by_site["n_high_event_rows"] > 0).sum())
    check_status = "PASS" if (basic_checks["status"] == "FAIL").sum() == 0 else "FAIL"
    selected_table = selected_sites[
        [
            SITE_COL,
            "selection_order",
            "event_ratio_category",
            "n_rows",
            "event_ratio",
            "test_event_rows",
            "selection_note",
        ]
    ]

    text = f"""# Step 2.0 Baseline-Ready pH Dataset Report

## Purpose

This step prepares a reproducible baseline-ready dataset for Step 2.1 10-site baseline modelling. No model was trained in this step.

## Inputs and Outputs

- Input dataset: `{INPUT_PATH}`
- Output folder: `{OUT_DIR}`
- Full baseline-ready dataset: `ph_baseline_ready.parquet`
- 10-site development subset: `ph_10site_baseline_ready.parquet`
- Feature definition file: `feature_set_step2_1.json`

## Official Column Roles

- Target: `{TARGET}` processed/scaled pH
- Main Step 2.1 features: {', '.join(MAIN_FEATURES)}
- Optional diagnostic columns: {', '.join(OPTIONAL_DIAGNOSTIC_COLUMNS)}
- Validation-only columns: {', '.join(VALIDATION_ONLY_COLUMNS)}

`00400` is processed/scaled pH, not raw 0-14 pH. Event labels must not be used as model inputs.

## Basic Validation

- Basic check status: {check_status}
- Rows: {len(df):,}
- Sites: {df[SITE_COL].nunique()}
- Duplicate site-timestamp rows: {int(df.duplicated(subset=[SITE_COL, TIME_COL]).sum()):,}
- Missing target values: {int(df[TARGET].isna().sum()):,}
- Missing numeric feature values: {int(df[NUMERIC_MAIN_FEATURES].isna().sum().sum()):,}

Detailed checks are saved in `baseline_ready_basic_check.csv`.

## Temporal Split Summary

| split | n_rows | row_ratio | n_event_rows | event_ratio |
| --- | --- | --- | --- | --- |
| train | {int(train_full['n_rows'])} | {float(train_full['row_ratio']):.2%} | {int(train_full['n_event_rows'])} | {float(train_full['event_ratio']):.2%} |
| val | {int(val_full['n_rows'])} | {float(val_full['row_ratio']):.2%} | {int(val_full['n_event_rows'])} | {float(val_full['event_ratio']):.2%} |
| test | {int(test_full['n_rows'])} | {float(test_full['row_ratio']):.2%} | {int(test_full['n_event_rows'])} | {float(test_full['event_ratio']):.2%} |

- Test-set event rows: {int(test_full['n_event_rows']):,}
- Test-set event ratio: {float(test_full['event_ratio']):.2%}
- Sites with zero test event rows: {n_zero_test_events}
- Sites with low-pH test events: {n_low_test_sites}
- Sites with high-pH test events: {n_high_test_sites}

The split is chronological within each site: first 70% train, next 15% validation, final 15% test.

## 10-Site Development Subset

The 10-site subset is selected by event-ratio strata to keep the next baseline experiment fast while retaining sites with different event-validation behaviour.

{dataframe_to_markdown(selected_table)}

Subset split summary:

{dataframe_to_markdown(subset_overall)}

## Step 2.1 Usage

Step 2.1 should read `ph_10site_baseline_ready.parquet` for the first baseline experiment and `feature_set_step2_1.json` for the official feature list.

Use only these main features for the first baseline:

{MAIN_FEATURES}

Do not use `in_ph_event` or `ph_event_label` during training. They are retained only for residual-event validation after predictions are produced.
"""
    (OUT_DIR / "baseline_ready_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, available_columns = load_dataset()
    basic_checks = run_basic_checks(df, available_columns)
    df = add_temporal_split(df)

    df.to_parquet(OUT_DIR / "ph_baseline_ready.parquet", index=False)
    df[[SITE_COL, TIME_COL, SPLIT_COL, EVENT_COL, EVENT_LABEL_COL]].to_parquet(
        OUT_DIR / "ph_split_metadata.parquet",
        index=False,
    )

    full_by_site, full_overall = make_split_summaries(df, prefix="ph")
    save_feature_set()

    selected_sites = select_10site_subset(full_by_site)
    selected_ids = set(selected_sites[SITE_COL])
    subset = df[df[SITE_COL].isin(selected_ids)].copy()
    subset.to_parquet(OUT_DIR / "ph_10site_baseline_ready.parquet", index=False)
    subset[[SITE_COL, TIME_COL, SPLIT_COL, EVENT_COL, EVENT_LABEL_COL]].to_parquet(
        OUT_DIR / "ph_10site_split_metadata.parquet",
        index=False,
    )
    subset_by_site, subset_overall = make_split_summaries(subset, prefix="ph_10site")

    write_report(df, basic_checks, full_by_site, full_overall, selected_sites, subset_overall)

    test_full = full_overall[full_overall[SPLIT_COL] == "test"].iloc[0]
    test_by_site = full_by_site[full_by_site[SPLIT_COL] == "test"]
    print(f"full rows: {len(df):,}")
    print(f"full sites: {df[SITE_COL].nunique()}")
    print(f"train rows: {int(full_overall.loc[full_overall[SPLIT_COL].eq('train'), 'n_rows'].iloc[0]):,}")
    print(f"validation rows: {int(full_overall.loc[full_overall[SPLIT_COL].eq('val'), 'n_rows'].iloc[0]):,}")
    print(f"test rows: {int(test_full['n_rows']):,}")
    print(f"test event rows: {int(test_full['n_event_rows']):,}")
    print(f"test event ratio: {float(test_full['event_ratio']):.4%}")
    print(f"sites with zero test event rows: {int((test_by_site['n_event_rows'] == 0).sum())}")
    print(f"10-site subset rows: {len(subset):,}")
    print(f"10-site subset sites: {subset[SITE_COL].nunique()}")
    print(f"selected 10 sites: {', '.join(selected_sites[SITE_COL].tolist())}")
    print(f"output folder path: {OUT_DIR}")


if __name__ == "__main__":
    main()
