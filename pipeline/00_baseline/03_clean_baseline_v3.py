from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
INPUT_DIR = ROOT / "analysis_outputs" / "ph_baseline_ready"
FULL_INPUT = INPUT_DIR / "ph_baseline_ready.parquet"
DEV10_INPUT = INPUT_DIR / "ph_10site_baseline_ready.parquet"
FEATURE_JSON = INPUT_DIR / "feature_set_step2_1.json"
V2_DIR = ROOT / "analysis_outputs" / "ph_baseline_ready_v2"
DEV10_V2_NEAR_SENTINEL = (
    ROOT
    / "analysis_outputs"
    / "ph_baseline_dev10_v2"
    / "dev10_v2_remaining_near_sentinel_check.csv"
)
OUT_DIR = ROOT / "analysis_outputs" / "ph_baseline_ready_v3"
FIG_DIR = OUT_DIR / "figures"

TARGET = "00400"
SITE_COL = "site_id"
TIME_COL = "timestamp"
SPLIT_COL = "split"
EVENT_COL = "in_ph_event"
EVENT_LABEL_COL = "ph_event_label"

OFFICIAL_NUMERIC_FEATURES = [
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
]
OFFICIAL_FEATURES = [*OFFICIAL_NUMERIC_FEATURES, SITE_COL]
AUXILIARY_DIAGNOSTIC_COLUMNS = ["hour", "day_of_year"]
VALIDATION_COLUMNS = [EVENT_COL, EVENT_LABEL_COL]
AUDIT_NUMERIC_COLUMNS = [TARGET, *OFFICIAL_NUMERIC_FEATURES, *AUXILIARY_DIAGNOSTIC_COLUMNS]
REQUIRED_COLUMNS = [
    SITE_COL,
    TIME_COL,
    SPLIT_COL,
    TARGET,
    *OFFICIAL_NUMERIC_FEATURES,
    *AUXILIARY_DIAGNOSTIC_COLUMNS,
    *VALIDATION_COLUMNS,
]
CLEAN_REQUIRED_COLUMNS = [TARGET, *OFFICIAL_NUMERIC_FEATURES, *AUXILIARY_DIAGNOSTIC_COLUMNS]

HARD_SENTINEL_THRESHOLD = -999.0
NEAR_SENTINEL_THRESHOLD = -900.0
EPS = 1e-6


def log(message: str) -> None:
    print(message, flush=True)


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
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.to_numpy()]
    return "\n".join([header, separator, *body])


def parquet_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    return pq.ParquetFile(path).schema_arrow.names


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    df[SITE_COL] = df[SITE_COL].map(normalise_site_id).astype("string")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df[SPLIT_COL] = df[SPLIT_COL].astype("string")
    df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce").fillna(0).astype("int8")
    df[EVENT_LABEL_COL] = df[EVENT_LABEL_COL].fillna("none").astype("string")
    for col in AUDIT_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values([SITE_COL, TIME_COL]).reset_index(drop=True)


def write_rules() -> dict[str, object]:
    rules = {
        "source": "Step 2.1-fix C robust v3 cleaning",
        "general_rules": {
            "missing_or_nonfinite": "invalid for target, official numeric features, hour, and day_of_year",
            "hard_sentinel": f"value <= {HARD_SENTINEL_THRESHOLD}",
            "near_sentinel": f"{HARD_SENTINEL_THRESHOLD} < value <= {NEAR_SENTINEL_THRESHOLD}",
        },
        "column_specific_rules": {
            TARGET: "processed/scaled pH must satisfy 0 <= value <= 1",
            "63680": "turbidity must satisfy value >= 0; this catches near-sentinel values such as -998.87",
            "sin_hour": "-1 <= value <= 1",
            "cos_hour": "-1 <= value <= 1",
            "sin_day": "-1 <= value <= 1",
            "cos_day": "-1 <= value <= 1",
            "month": "1 <= month <= 12 and month is integer-like",
            "hour": "0 <= hour <= 23 and hour is integer-like",
            "day_of_year": "1 <= day_of_year <= 366 and day_of_year is integer-like",
        },
        "row_removal_rule": (
            "Invalid values are converted to missing; rows are dropped if target, "
            "official numeric features, hour, or day_of_year are missing after cleaning."
        ),
        "file_safety": "v1/v2/original files are not modified; all outputs go to ph_baseline_ready_v3",
    }
    (OUT_DIR / "invalid_value_rules_v3.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")
    return rules


def validate_input(dataset: str, path: Path) -> tuple[list[dict[str, object]], pd.DataFrame | None]:
    checks: list[dict[str, object]] = []

    def add(check_name: str, status: str, value: object, notes: str) -> None:
        checks.append(
            {
                "dataset": dataset,
                "check_name": check_name,
                "status": status,
                "value": value,
                "notes": notes,
            }
        )

    exists = path.exists()
    add("input_file_exists", "PASS" if exists else "FAIL", exists, str(path))
    if not exists:
        return checks, None
    columns = parquet_columns(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in columns]
    add("required_columns_exist", "PASS" if not missing else "FAIL", missing if missing else "None", "Required v3 cleaning columns.")
    if missing:
        return checks, None
    try:
        df = load_dataset(path)
        add("dataset_loads_successfully", "PASS", len(df), "Rows loaded.")
    except Exception as exc:
        add("dataset_loads_successfully", "FAIL", type(exc).__name__, str(exc))
        return checks, None

    timestamp_failures = int(df[TIME_COL].isna().sum())
    duplicate_count = int(df.duplicated(subset=[SITE_COL, TIME_COL]).sum())
    split_values = sorted(df[SPLIT_COL].dropna().unique().tolist())
    add("timestamp_parse_success", "PASS" if timestamp_failures == 0 else "FAIL", timestamp_failures, "Rows with unparsed timestamp.")
    add("split_column_exists", "PASS" if SPLIT_COL in df.columns else "FAIL", SPLIT_COL in df.columns, "Existing split must be preserved.")
    add("train_val_test_exist", "PASS" if {"train", "val", "test"}.issubset(split_values) else "FAIL", split_values, "Expected split values.")
    add("target_numeric", "PASS" if pd.api.types.is_numeric_dtype(df[TARGET]) else "FAIL", str(df[TARGET].dtype), "Target is processed/scaled pH.")
    add("validation_columns_exist", "PASS" if all(col in df.columns for col in VALIDATION_COLUMNS) else "FAIL", True, "Event columns are validation-only.")
    add("duplicate_site_timestamp_count", "PASS" if duplicate_count == 0 else "FAIL", duplicate_count, "Critical if non-zero.")
    add("n_rows", "PASS" if len(df) > 0 else "FAIL", len(df), "Input rows.")
    add("n_sites", "PASS" if df[SITE_COL].nunique() > 0 else "FAIL", df[SITE_COL].nunique(), "Input sites.")
    return checks, df


def validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_checks: list[dict[str, object]] = []
    feature_exists = FEATURE_JSON.exists()
    all_checks.append(
        {
            "dataset": "feature_json",
            "check_name": "feature_json_exists",
            "status": "PASS" if feature_exists else "FAIL",
            "value": feature_exists,
            "notes": str(FEATURE_JSON),
        }
    )
    full_checks, full_df = validate_input("full50", FULL_INPUT)
    dev_checks, dev_df = validate_input("dev10", DEV10_INPUT)
    all_checks.extend(full_checks)
    all_checks.extend(dev_checks)
    checks = pd.DataFrame(all_checks)
    checks.to_csv(OUT_DIR / "input_validation_checks_v3.csv", index=False, encoding="utf-8-sig")
    critical = checks[
        checks["check_name"].isin(
            [
                "feature_json_exists",
                "input_file_exists",
                "required_columns_exist",
                "dataset_loads_successfully",
                "timestamp_parse_success",
                "split_column_exists",
                "duplicate_site_timestamp_count",
                "target_numeric",
            ]
        )
        & checks["status"].eq("FAIL")
    ]
    if not critical.empty or full_df is None or dev_df is None:
        raise RuntimeError(
            "Critical v3 input validation failed. See input_validation_checks_v3.csv.\n"
            + critical.to_string(index=False)
        )
    return checks, full_df, dev_df


def integer_like(series: pd.Series) -> pd.Series:
    return np.isclose(series, np.round(series), atol=EPS)


def invalid_masks(df: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    masks: dict[str, dict[str, pd.Series]] = {}
    for col in AUDIT_NUMERIC_COLUMNS:
        series = df[col]
        col_masks: dict[str, pd.Series] = {
            "missing_or_nonfinite": series.isna() | ~np.isfinite(series),
            "hard_sentinel_le_minus_999": series <= HARD_SENTINEL_THRESHOLD,
            "near_sentinel_le_minus_900": (series > HARD_SENTINEL_THRESHOLD) & (series <= NEAR_SENTINEL_THRESHOLD),
        }
        if col == TARGET:
            col_masks["target_outside_0_1"] = (series < 0) | (series > 1)
        if col == "63680":
            col_masks["turbidity_negative"] = series < 0
        if col in ["sin_hour", "cos_hour", "sin_day", "cos_day"]:
            col_masks["cyclic_outside_minus1_1"] = (series < -1 - EPS) | (series > 1 + EPS)
        if col == "month":
            col_masks["month_outside_1_12"] = (series < 1) | (series > 12) | ~integer_like(series)
        if col == "hour":
            col_masks["hour_outside_0_23"] = (series < 0) | (series > 23) | ~integer_like(series)
        if col == "day_of_year":
            col_masks["day_of_year_outside_1_366"] = (series < 1) | (series > 366) | ~integer_like(series)
        masks[col] = col_masks
    return masks


def combined_invalid_by_column(masks: dict[str, dict[str, pd.Series]]) -> dict[str, pd.Series]:
    combined: dict[str, pd.Series] = {}
    for col, rule_masks in masks.items():
        col_mask = None
        for mask in rule_masks.values():
            col_mask = mask.copy() if col_mask is None else (col_mask | mask)
        combined[col] = col_mask if col_mask is not None else pd.Series(False)
    return combined


def audit_by_column_rule(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    masks = invalid_masks(df)
    rows: list[dict[str, object]] = []
    for col, rule_masks in masks.items():
        combined = pd.Series(False, index=df.index)
        for rule, mask in rule_masks.items():
            combined = combined | mask
            rows.append(
                {
                    "dataset": dataset,
                    "column": col,
                    "rule": rule,
                    "n_rows": len(df),
                    "n_invalid_values": int(mask.sum()),
                    "invalid_rate": float(mask.mean()),
                    "min": float(df[col].min()),
                    "p001": float(df[col].quantile(0.001)),
                    "p01": float(df[col].quantile(0.01)),
                    "median": float(df[col].median()),
                    "p99": float(df[col].quantile(0.99)),
                    "max": float(df[col].max()),
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "column": col,
                "rule": "any_invalid_for_column",
                "n_rows": len(df),
                "n_invalid_values": int(combined.sum()),
                "invalid_rate": float(combined.mean()),
                "min": float(df[col].min()),
                "p001": float(df[col].quantile(0.001)),
                "p01": float(df[col].quantile(0.01)),
                "median": float(df[col].median()),
                "p99": float(df[col].quantile(0.99)),
                "max": float(df[col].max()),
            }
        )
    return pd.DataFrame(rows)


def row_invalid_mask(df: pd.DataFrame) -> pd.Series:
    masks = combined_invalid_by_column(invalid_masks(df))
    row_mask = pd.Series(False, index=df.index)
    for col in CLEAN_REQUIRED_COLUMNS:
        row_mask = row_mask | masks[col]
    return row_mask


def row_invalid_reasons(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    masks = invalid_masks(df)
    reason_rows: list[str] = []
    masked_index = df.index[mask]
    for idx in masked_index:
        reasons: list[str] = []
        for col in CLEAN_REQUIRED_COLUMNS:
            for rule, rule_mask in masks[col].items():
                if bool(rule_mask.loc[idx]):
                    reasons.append(f"{col}:{rule}")
        reason_rows.append(";".join(reasons))
    return pd.Series(reason_rows, index=masked_index)


def audit_by_site_split(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    row_mask = row_invalid_mask(df)
    temp = df[[SITE_COL, SPLIT_COL]].copy()
    temp["row_removed_by_v3"] = row_mask.astype("int8")
    out = temp.groupby([SITE_COL, SPLIT_COL], observed=False).agg(
        n_rows=(SITE_COL, "size"),
        n_rows_removed=("row_removed_by_v3", "sum"),
    )
    out["row_removed_rate"] = out["n_rows_removed"] / out["n_rows"]
    out = out.reset_index().sort_values("n_rows_removed", ascending=False)
    out.insert(0, "dataset", dataset)
    return out


def audit_by_site(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    row_mask = row_invalid_mask(df)
    temp = df[[SITE_COL]].copy()
    temp["row_removed_by_v3"] = row_mask.astype("int8")
    out = temp.groupby(SITE_COL).agg(
        n_rows=(SITE_COL, "size"),
        n_rows_removed=("row_removed_by_v3", "sum"),
    )
    out["row_removed_rate"] = out["n_rows_removed"] / out["n_rows"]
    out = out.reset_index().sort_values("n_rows_removed", ascending=False)
    out.insert(0, "dataset", dataset)
    return out


def invalid_examples(df: pd.DataFrame, dataset: str, max_rows: int = 600) -> pd.DataFrame:
    mask = row_invalid_mask(df)
    cols = [SITE_COL, TIME_COL, SPLIT_COL, EVENT_COL, EVENT_LABEL_COL, *AUDIT_NUMERIC_COLUMNS]
    examples = df.loc[mask, cols].copy()
    examples.insert(0, "dataset", dataset)
    if examples.empty:
        examples["invalid_reasons"] = ""
        return examples
    reasons = row_invalid_reasons(df, mask)
    examples["invalid_reasons"] = reasons.to_numpy()
    return examples.sort_values([SITE_COL, TIME_COL]).head(max_rows)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    masks = combined_invalid_by_column(invalid_masks(df))
    cleaned = df.copy()
    for col in CLEAN_REQUIRED_COLUMNS:
        cleaned.loc[masks[col], col] = np.nan
    missing_after = cleaned[CLEAN_REQUIRED_COLUMNS].isna().any(axis=1) | cleaned[SITE_COL].isna()
    cleaned = cleaned.loc[~missing_after].copy()
    cleaned = cleaned.sort_values([SITE_COL, TIME_COL]).reset_index(drop=True)
    return cleaned, missing_after


def retention_summary(before: pd.DataFrame, after: pd.DataFrame, removed_mask: pd.Series, dataset: str) -> dict[str, object]:
    removed = before.loc[removed_mask]
    return {
        "dataset": dataset,
        "n_rows_before": len(before),
        "n_rows_removed": int(removed_mask.sum()),
        "n_rows_after": len(after),
        "row_retention_rate": len(after) / len(before) if len(before) else np.nan,
        "n_sites_before": before[SITE_COL].nunique(),
        "n_sites_after": after[SITE_COL].nunique(),
        "n_splits_before": before[SPLIT_COL].nunique(),
        "n_splits_after": after[SPLIT_COL].nunique(),
        "removed_train_rows": int(removed[SPLIT_COL].eq("train").sum()) if not removed.empty else 0,
        "removed_val_rows": int(removed[SPLIT_COL].eq("val").sum()) if not removed.empty else 0,
        "removed_test_rows": int(removed[SPLIT_COL].eq("test").sum()) if not removed.empty else 0,
        "removed_event_rows": int(removed[EVENT_COL].sum()) if not removed.empty else 0,
    }


def split_summary(df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    temp = df.copy()
    temp["is_low_event"] = (temp[EVENT_COL].eq(1) & temp[EVENT_LABEL_COL].eq("low")).astype("int8")
    temp["is_high_event"] = (temp[EVENT_COL].eq(1) & temp[EVENT_LABEL_COL].eq("high")).astype("int8")
    overall = temp.groupby(SPLIT_COL, sort=False).agg(
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
    overall["event_ratio"] = overall["n_event_rows"] / overall["n_rows"]
    overall = overall.reset_index()
    overall.insert(0, "dataset", dataset)
    by_site = temp.groupby([SITE_COL, SPLIT_COL], observed=False).agg(
        n_rows=(TARGET, "size"),
        n_event_rows=(EVENT_COL, "sum"),
        n_low_event_rows=("is_low_event", "sum"),
        n_high_event_rows=("is_high_event", "sum"),
        start_time=(TIME_COL, "min"),
        end_time=(TIME_COL, "max"),
        target_mean=(TARGET, "mean"),
        target_std=(TARGET, "std"),
    )
    by_site["event_ratio"] = by_site["n_event_rows"] / by_site["n_rows"]
    by_site = by_site.reset_index()
    by_site.insert(0, "dataset", dataset)
    return overall, by_site


def post_clean_validation(df: pd.DataFrame, dataset: str) -> list[dict[str, object]]:
    masks = combined_invalid_by_column(invalid_masks(df))
    remaining_rows = pd.Series(False, index=df.index)
    for col in CLEAN_REQUIRED_COLUMNS:
        remaining_rows = remaining_rows | masks[col]
    return [
        {
            "dataset": dataset,
            "check_name": "n_rows",
            "status": "PASS" if len(df) > 0 else "FAIL",
            "value": len(df),
            "notes": "Rows after v3 cleaning.",
        },
        {
            "dataset": dataset,
            "check_name": "n_sites_preserved",
            "status": "PASS" if df[SITE_COL].nunique() > 0 else "FAIL",
            "value": df[SITE_COL].nunique(),
            "notes": "Sites after v3 cleaning.",
        },
        {
            "dataset": dataset,
            "check_name": "splits_preserved",
            "status": "PASS" if {"train", "val", "test"}.issubset(set(df[SPLIT_COL].unique())) else "FAIL",
            "value": sorted(df[SPLIT_COL].unique().tolist()),
            "notes": "Existing split labels preserved.",
        },
        {
            "dataset": dataset,
            "check_name": "no_missing_clean_required_columns",
            "status": "PASS" if int(df[CLEAN_REQUIRED_COLUMNS].isna().sum().sum()) == 0 else "FAIL",
            "value": int(df[CLEAN_REQUIRED_COLUMNS].isna().sum().sum()),
            "notes": "Target, official numeric features, hour, day_of_year.",
        },
        {
            "dataset": dataset,
            "check_name": "no_invalid_values_remaining",
            "status": "PASS" if int(remaining_rows.sum()) == 0 else "FAIL",
            "value": int(remaining_rows.sum()),
            "notes": "Rows still matching any v3 invalid-value rule.",
        },
        {
            "dataset": dataset,
            "check_name": "no_duplicate_site_timestamp",
            "status": "PASS" if int(df.duplicated(subset=[SITE_COL, TIME_COL]).sum()) == 0 else "FAIL",
            "value": int(df.duplicated(subset=[SITE_COL, TIME_COL]).sum()),
            "notes": "Duplicate site_id + timestamp after cleaning.",
        },
    ]


def v2_comparison(retention: pd.DataFrame) -> pd.DataFrame:
    rows = []
    v2_retention = V2_DIR / "row_retention_summary.csv"
    if v2_retention.exists():
        v2 = pd.read_csv(v2_retention)
        v2 = v2.rename(columns={"n_rows_removed": "v2_rows_removed", "n_rows_after": "v2_rows_after"})
        v3 = retention.rename(columns={"n_rows_removed": "v3_rows_removed", "n_rows_after": "v3_rows_after"})
        comp = v2[["dataset", "v2_rows_removed", "v2_rows_after"]].merge(
            v3[["dataset", "v3_rows_removed", "v3_rows_after"]],
            on="dataset",
            how="outer",
        )
        comp["additional_rows_removed_by_v3"] = comp["v3_rows_removed"] - comp["v2_rows_removed"]
        return comp
    return pd.DataFrame(rows)


def save_figures(audit: pd.DataFrame, site_audit: pd.DataFrame, retention: pd.DataFrame, v2_comp: pd.DataFrame) -> None:
    full_any = audit[
        (audit["dataset"].eq("full50"))
        & (audit["rule"].eq("any_invalid_for_column"))
        & (audit["n_invalid_values"] > 0)
    ].sort_values("n_invalid_values", ascending=False)
    if full_any.empty:
        full_any = audit[(audit["dataset"].eq("full50")) & (audit["rule"].eq("any_invalid_for_column"))]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(full_any["column"], full_any["n_invalid_values"], color="#4C78A8")
    ax.set_title("v3 invalid rows by column, full 50-site source")
    ax.set_xlabel("Column")
    ax.set_ylabel("Invalid values")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_v3_invalid_values_by_column_full50.png", dpi=200)
    plt.close(fig)

    full_rules = audit[(audit["dataset"].eq("full50")) & ~audit["rule"].eq("any_invalid_for_column")]
    rule_counts = full_rules.groupby("rule")["n_invalid_values"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(rule_counts.index, rule_counts.values, color="#F58518")
    ax.set_title("v3 invalid values by rule, full 50-site source")
    ax.set_xlabel("Rule")
    ax.set_ylabel("Invalid values")
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_v3_invalid_values_by_rule_full50.png", dpi=200)
    plt.close(fig)

    full_sites = site_audit[site_audit["dataset"].eq("full50")].head(30)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(full_sites[SITE_COL], full_sites["n_rows_removed"], color="#E45756")
    ax.set_title("v3 removed rows by site, top 30 full 50-site source")
    ax.set_xlabel("Site ID")
    ax.set_ylabel("Rows removed")
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_v3_rows_removed_by_site_full50.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(retention["dataset"], retention["n_rows_removed"], color="#72B7B2")
    ax.set_title("Rows removed by v3 robust cleaning")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Rows removed")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_v3_rows_removed_by_dataset.png", dpi=200)
    plt.close(fig)

    if not v2_comp.empty:
        x = np.arange(len(v2_comp))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width / 2, v2_comp["v2_rows_removed"], width=width, label="v2", color="#BAB0AC")
        ax.bar(x + width / 2, v2_comp["v3_rows_removed"], width=width, label="v3", color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(v2_comp["dataset"])
        ax.set_title("Rows removed: v2 sentinel cleaning vs v3 robust cleaning")
        ax.set_ylabel("Rows removed")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_v2_vs_v3_rows_removed.png", dpi=200)
        plt.close(fig)


def write_report(
    rules: dict[str, object],
    input_checks: pd.DataFrame,
    audit: pd.DataFrame,
    site_split_audit: pd.DataFrame,
    retention: pd.DataFrame,
    post_checks: pd.DataFrame,
    full_summary: pd.DataFrame,
    dev_summary: pd.DataFrame,
    v2_comp: pd.DataFrame,
) -> None:
    full_ret = retention[retention["dataset"].eq("full50")].iloc[0]
    dev_ret = retention[retention["dataset"].eq("dev10")].iloc[0]
    affected_cols = audit[
        (audit["dataset"].eq("full50"))
        & (audit["rule"].eq("any_invalid_for_column"))
        & (audit["n_invalid_values"] > 0)
    ][["column", "n_invalid_values", "invalid_rate", "min", "p001", "p01", "median", "p99", "max"]]
    affected_rules = audit[
        (audit["dataset"].eq("full50"))
        & ~audit["rule"].eq("any_invalid_for_column")
        & (audit["n_invalid_values"] > 0)
    ][["column", "rule", "n_invalid_values", "invalid_rate"]].sort_values("n_invalid_values", ascending=False)
    affected_sites = site_split_audit[
        (site_split_audit["dataset"].eq("full50")) & (site_split_audit["n_rows_removed"] > 0)
    ][[SITE_COL, SPLIT_COL, "n_rows", "n_rows_removed", "row_removed_rate"]].head(25)
    decision = (
        "SAFE_FOR_DEV10_V3_BASELINE_RERUN"
        if (post_checks["status"].eq("FAIL").sum() == 0)
        and int(dev_ret["n_sites_after"]) == 10
        and int(full_ret["n_sites_after"]) == 50
        else "CHECK_V3_DATA_BEFORE_RERUN"
    )

    text = f"""# Step 2.1-fix C Robust v3 Cleaning Report

## Purpose

This step performs robust v3 cleaning for the pH baseline-ready datasets. It does not train or rerun any model. It starts from the original Step 2.0 baseline-ready datasets, preserves the existing temporal split, and preserves the same original 10 development sites.

## File Safety

- Original full 50-site input left unchanged: `{FULL_INPUT}`
- Original dev10 input left unchanged: `{DEV10_INPUT}`
- v2 outputs left unchanged: `{V2_DIR}`
- New v3 output folder: `{OUT_DIR}`

## Rules Applied

```json
{json.dumps(rules, indent=2)}
```

## Input Validation

{dataframe_to_markdown(input_checks)}

## Invalid-Value Audit

Affected full 50-site columns:

{dataframe_to_markdown(affected_cols) if not affected_cols.empty else "No affected columns."}

Affected full 50-site column/rule combinations:

{dataframe_to_markdown(affected_rules) if not affected_rules.empty else "No affected rules."}

Most affected full 50-site site/split combinations:

{dataframe_to_markdown(affected_sites) if not affected_sites.empty else "No affected site/split combinations."}

## Row Retention

{dataframe_to_markdown(retention)}

v2 vs v3 comparison:

{dataframe_to_markdown(v2_comp) if not v2_comp.empty else "No v2 retention summary available for comparison."}

## v3 Output Files

- `ph_baseline_ready_v3.parquet`: cleaned full 50-site baseline-ready dataset.
- `ph_10site_baseline_ready_v3.parquet`: cleaned original dev10 baseline-ready dataset.
- `ph_split_metadata_v3.parquet`: cleaned full split metadata.
- `ph_10site_split_metadata_v3.parquet`: cleaned dev10 split metadata.
- `feature_set_step2_1_v3.json`: copied feature definition file.

## Cleaned Split Summary

Full 50-site v3:

{dataframe_to_markdown(full_summary)}

Dev10 v3:

{dataframe_to_markdown(dev_summary)}

## Post-Clean Validation

{dataframe_to_markdown(post_checks)}

## Decision

{decision}

The next step should rerun the dev10 baseline using `ph_10site_baseline_ready_v3.parquet`. Do not run the full 50-site baseline until the dev10 v3 rerun confirms that Ridge no longer has extreme negative predictions.
"""
    (OUT_DIR / "robust_cleaning_v3_report.md").write_text(text, encoding="utf-8")
    (OUT_DIR / "robust_cleaning_v3_decision.txt").write_text(decision + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rules = write_rules()
    input_checks, full_df, dev_df = validate_inputs()

    log("Auditing v3 invalid values...")
    full_audit = audit_by_column_rule(full_df, "full50")
    dev_audit = audit_by_column_rule(dev_df, "dev10")
    audit = pd.concat([full_audit, dev_audit], ignore_index=True)
    audit.to_csv(OUT_DIR / "invalid_value_audit_by_column_rule_v3.csv", index=False, encoding="utf-8-sig")

    full_site_split = audit_by_site_split(full_df, "full50")
    dev_site_split = audit_by_site_split(dev_df, "dev10")
    site_split = pd.concat([full_site_split, dev_site_split], ignore_index=True)
    site_split.to_csv(OUT_DIR / "invalid_rows_by_site_split_v3.csv", index=False, encoding="utf-8-sig")

    full_site = audit_by_site(full_df, "full50")
    dev_site = audit_by_site(dev_df, "dev10")
    site_audit = pd.concat([full_site, dev_site], ignore_index=True)
    site_audit.to_csv(OUT_DIR / "invalid_rows_by_site_v3.csv", index=False, encoding="utf-8-sig")

    examples = pd.concat(
        [invalid_examples(full_df, "full50"), invalid_examples(dev_df, "dev10")],
        ignore_index=True,
    )
    examples.to_csv(OUT_DIR / "invalid_rows_examples_v3.csv", index=False, encoding="utf-8-sig")

    log("Creating v3 cleaned datasets...")
    full_clean, full_removed = clean_dataset(full_df)
    dev_clean, dev_removed = clean_dataset(dev_df)
    retention = pd.DataFrame(
        [
            retention_summary(full_df, full_clean, full_removed, "full50"),
            retention_summary(dev_df, dev_clean, dev_removed, "dev10"),
        ]
    )
    retention.to_csv(OUT_DIR / "row_retention_summary_v3.csv", index=False, encoding="utf-8-sig")

    full_clean.to_parquet(OUT_DIR / "ph_baseline_ready_v3.parquet", index=False)
    dev_clean.to_parquet(OUT_DIR / "ph_10site_baseline_ready_v3.parquet", index=False)
    full_clean[[SITE_COL, TIME_COL, SPLIT_COL, EVENT_COL, EVENT_LABEL_COL]].to_parquet(
        OUT_DIR / "ph_split_metadata_v3.parquet",
        index=False,
    )
    dev_clean[[SITE_COL, TIME_COL, SPLIT_COL, EVENT_COL, EVENT_LABEL_COL]].to_parquet(
        OUT_DIR / "ph_10site_split_metadata_v3.parquet",
        index=False,
    )
    shutil.copy2(FEATURE_JSON, OUT_DIR / "feature_set_step2_1_v3.json")

    full_summary, full_by_site = split_summary(full_clean, "full50_v3")
    dev_summary, dev_by_site = split_summary(dev_clean, "dev10_v3")
    full_summary.to_csv(OUT_DIR / "ph_split_summary_overall_v3.csv", index=False, encoding="utf-8-sig")
    full_by_site.to_csv(OUT_DIR / "ph_split_summary_by_site_v3.csv", index=False, encoding="utf-8-sig")
    dev_summary.to_csv(OUT_DIR / "ph_10site_split_summary_overall_v3.csv", index=False, encoding="utf-8-sig")
    dev_by_site.to_csv(OUT_DIR / "ph_10site_split_summary_by_site_v3.csv", index=False, encoding="utf-8-sig")

    post_checks = pd.DataFrame(
        [
            *post_clean_validation(full_clean, "full50_v3"),
            *post_clean_validation(dev_clean, "dev10_v3"),
        ]
    )
    post_checks.to_csv(OUT_DIR / "post_clean_validation_checks_v3.csv", index=False, encoding="utf-8-sig")
    v2_comp = v2_comparison(retention)
    if not v2_comp.empty:
        v2_comp.to_csv(OUT_DIR / "v2_vs_v3_retention_comparison.csv", index=False, encoding="utf-8-sig")

    save_figures(audit, site_audit, retention, v2_comp)
    write_report(rules, input_checks, audit, site_split, retention, post_checks, full_summary, dev_summary, v2_comp)

    full_ret = retention[retention["dataset"].eq("full50")].iloc[0]
    dev_ret = retention[retention["dataset"].eq("dev10")].iloc[0]
    affected = audit[
        (audit["dataset"].eq("full50"))
        & (audit["rule"].eq("any_invalid_for_column"))
        & (audit["n_invalid_values"] > 0)
    ][["column", "n_invalid_values"]]

    log(f"full rows before: {int(full_ret['n_rows_before']):,}")
    log(f"full rows removed: {int(full_ret['n_rows_removed']):,}")
    log(f"full rows after: {int(full_ret['n_rows_after']):,}")
    log(f"full sites after: {int(full_ret['n_sites_after'])}")
    log(f"dev10 rows before: {int(dev_ret['n_rows_before']):,}")
    log(f"dev10 rows removed: {int(dev_ret['n_rows_removed']):,}")
    log(f"dev10 rows after: {int(dev_ret['n_rows_after']):,}")
    log(f"dev10 sites after: {int(dev_ret['n_sites_after'])}")
    log("affected full50 columns:")
    log(affected.to_string(index=False) if not affected.empty else "none")
    log(f"post-clean failures: {int(post_checks['status'].eq('FAIL').sum())}")
    log(f"decision: {(OUT_DIR / 'robust_cleaning_v3_decision.txt').read_text(encoding='utf-8').strip()}")
    log(f"output folder path: {OUT_DIR}")


if __name__ == "__main__":
    main()
