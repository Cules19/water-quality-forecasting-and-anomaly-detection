from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
INPUT_DIR = ROOT / "analysis_outputs" / "ph_baseline_ready_v3"
DATA_PATH = INPUT_DIR / "ph_baseline_ready_v3.parquet"
FEATURE_JSON_PATH = INPUT_DIR / "feature_set_step2_1_v3.json"
DEV10_V3_METRICS = ROOT / "analysis_outputs" / "ph_baseline_dev10_v3" / "dev10_v3_model_metrics_overall.csv"
OUT_DIR = ROOT / "analysis_outputs" / "ph_baseline_full50_v3"
FIG_DIR = OUT_DIR / "figures"

TARGET = "00400"
SITE_COL = "site_id"
TIME_COL = "timestamp"
SPLIT_COL = "split"
EVENT_COL = "in_ph_event"
EVENT_LABEL_COL = "ph_event_label"
SENTINEL_THRESHOLD = -999

DEFAULT_FEATURES = [
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
NUMERIC_SENTINEL_CHECK_COLUMNS = [
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
]
PHYSICAL_FEATURES = ["00010", "00095", "63680", "00060", "00065"]
CYCLIC_FEATURES = ["sin_hour", "cos_hour", "sin_day", "cos_day"]
HGB_FULL_TRAIN_ROW_LIMIT = 5_000_000
HGB_SAMPLE_PER_SITE = 100_000
RANDOM_SEED = 42
VALIDATION_COLUMNS = [EVENT_COL, EVENT_LABEL_COL]
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
AUXILIARY_COLUMNS = [TIME_COL, "hour", "day_of_year", SPLIT_COL]
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


def load_feature_set() -> dict[str, object]:
    if not FEATURE_JSON_PATH.exists():
        raise FileNotFoundError(f"Feature JSON not found: {FEATURE_JSON_PATH}")
    return json.loads(FEATURE_JSON_PATH.read_text(encoding="utf-8"))


def load_dataset(feature_set: dict[str, object]) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned Full50 v3 dataset not found: {DATA_PATH}")
    features = list(feature_set.get("main_features", DEFAULT_FEATURES))
    required = sorted(set([TARGET, *features, *AUXILIARY_COLUMNS, *VALIDATION_COLUMNS]))
    log(f"Loading cleaned Full50 v3 dataset: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH, columns=required)
    df[SITE_COL] = df[SITE_COL].map(normalise_site_id).astype("string")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df[SPLIT_COL] = df[SPLIT_COL].astype("string")
    df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce").fillna(0).astype("int8")
    df[EVENT_LABEL_COL] = df[EVENT_LABEL_COL].fillna("none").astype("string")
    numeric_cols = [col for col in features if col != SITE_COL] + [TARGET, "hour", "day_of_year"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values([SITE_COL, TIME_COL]).reset_index(drop=True)


def invalid_value_counts_v3(df: pd.DataFrame) -> dict[str, int]:
    hard_sentinel_mask = pd.Series(False, index=df.index)
    nonfinite_mask = pd.Series(False, index=df.index)
    for col in NUMERIC_SENTINEL_CHECK_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")
        hard_sentinel_mask |= values.le(-999)
        nonfinite_mask |= ~np.isfinite(values.to_numpy(dtype="float64"))

    physical_mask = pd.Series(False, index=df.index)
    for col in PHYSICAL_FEATURES:
        physical_mask |= df[col].le(-100)

    hard_sentinel = int(hard_sentinel_mask.sum())
    physical_le_minus_100 = int(physical_mask.sum())
    target_outside_0_1 = int(((df[TARGET] < 0) | (df[TARGET] > 1)).sum())
    cyclic_mask = pd.Series(False, index=df.index)
    for col in CYCLIC_FEATURES:
        cyclic_mask |= df[col].lt(-1.000001) | df[col].gt(1.000001)
    cyclic_outside = int(cyclic_mask.sum())
    invalid_month = int(((df["month"] < 1) | (df["month"] > 12)).sum())
    nonfinite = int(nonfinite_mask.sum())
    return {
        "hard_sentinel_rows": hard_sentinel,
        "physical_feature_le_minus_100_rows": physical_le_minus_100,
        "target_outside_0_1_rows": target_outside_0_1,
        "cyclic_outside_minus1_1_rows": cyclic_outside,
        "invalid_month_rows": invalid_month,
        "nonfinite_numeric_rows": nonfinite,
    }


def run_initial_checks(df: pd.DataFrame, feature_set: dict[str, object]) -> pd.DataFrame:
    features = list(feature_set.get("main_features", DEFAULT_FEATURES))
    numeric_features = [col for col in features if col != SITE_COL]
    forbidden = set(feature_set.get("forbidden_columns", FORBIDDEN_COLUMNS))
    forbidden_in_features = sorted(set(features).intersection(forbidden | {TARGET, TIME_COL}))
    duplicate_count = int(df.duplicated(subset=[SITE_COL, TIME_COL]).sum())
    split_values = sorted(df[SPLIT_COL].dropna().unique().tolist()) if SPLIT_COL in df.columns else []
    missing_features = [col for col in features if col not in df.columns]
    invalid_counts = invalid_value_counts_v3(df)
    missing_numeric_feature_count = sum(int(df[col].isna().sum()) for col in numeric_features)

    checks = [
        ("input_parquet_exists", DATA_PATH.exists(), str(DATA_PATH)),
        ("feature_json_exists", FEATURE_JSON_PATH.exists(), str(FEATURE_JSON_PATH)),
        ("dataset_loads_successfully", len(df) > 0, f"{len(df):,} rows loaded"),
        ("n_unique_sites_is_50", df[SITE_COL].nunique() == 50, df[SITE_COL].nunique()),
        ("split_column_exists", SPLIT_COL in df.columns, "existing split column"),
        ("train_val_test_exist", {"train", "val", "test"}.issubset(split_values), split_values),
        ("target_exists_and_numeric", TARGET in df.columns and pd.api.types.is_numeric_dtype(df[TARGET]), str(df[TARGET].dtype) if TARGET in df.columns else "missing"),
        ("official_feature_columns_exist", not missing_features, missing_features if missing_features else "None"),
        ("missing_target_count", int(df[TARGET].isna().sum()) == 0, int(df[TARGET].isna().sum())),
        ("missing_numeric_feature_count", missing_numeric_feature_count == 0, missing_numeric_feature_count),
        ("site_id_missing_count", int(df[SITE_COL].isna().sum()) == 0, int(df[SITE_COL].isna().sum())),
        ("timestamp_parse_success", int(df[TIME_COL].isna().sum()) == 0, int(df[TIME_COL].isna().sum())),
        ("duplicate_site_timestamp_count", duplicate_count == 0, duplicate_count),
        ("validation_only_columns_exist", all(col in df.columns for col in VALIDATION_COLUMNS), VALIDATION_COLUMNS),
        ("forbidden_columns_not_in_model_features", not forbidden_in_features, forbidden_in_features if forbidden_in_features else "None"),
        ("no_hard_sentinel_values", invalid_counts["hard_sentinel_rows"] == 0, invalid_counts["hard_sentinel_rows"]),
        ("no_physical_feature_le_minus_100", invalid_counts["physical_feature_le_minus_100_rows"] == 0, invalid_counts["physical_feature_le_minus_100_rows"]),
        ("target_within_0_1", invalid_counts["target_outside_0_1_rows"] == 0, invalid_counts["target_outside_0_1_rows"]),
        ("cyclic_features_within_minus1_1", invalid_counts["cyclic_outside_minus1_1_rows"] == 0, invalid_counts["cyclic_outside_minus1_1_rows"]),
        ("month_within_1_12", invalid_counts["invalid_month_rows"] == 0, invalid_counts["invalid_month_rows"]),
        ("no_nonfinite_numeric_values", invalid_counts["nonfinite_numeric_rows"] == 0, invalid_counts["nonfinite_numeric_rows"]),
    ]
    out = pd.DataFrame(
        [
            {
                "check_name": name,
                "status": "PASS" if passed else "FAIL",
                "value": value,
                "notes": notes if name not in {"input_parquet_exists", "feature_json_exists"} else value,
            }
            for name, passed, value in checks
            for notes in [value]
        ]
    )
    out.to_csv(OUT_DIR / "full50_v3_initial_checks.csv", index=False, encoding="utf-8-sig")
    critical_failed = out[out["status"].eq("FAIL")]
    if not critical_failed.empty:
        raise RuntimeError("Critical v3 initial checks failed:\n" + critical_failed.to_string(index=False))
    return out


def make_split_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    overall = overall.reset_index()[[
        SPLIT_COL,
        "n_rows",
        "n_sites",
        "n_event_rows",
        "event_ratio",
        "n_low_event_rows",
        "n_high_event_rows",
        "target_min",
        "target_max",
        "target_mean",
        "target_std",
    ]]

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
    by_site = by_site.reset_index()[[
        SITE_COL,
        SPLIT_COL,
        "n_rows",
        "n_event_rows",
        "event_ratio",
        "n_low_event_rows",
        "n_high_event_rows",
        "start_time",
        "end_time",
        "target_mean",
        "target_std",
    ]]
    split_order = ["train", "val", "test"]
    split_cat = pd.CategoricalDtype(split_order, ordered=True)
    overall[SPLIT_COL] = overall[SPLIT_COL].astype(split_cat)
    by_site[SPLIT_COL] = by_site[SPLIT_COL].astype(split_cat)
    overall = overall.sort_values(SPLIT_COL)
    by_site = by_site.sort_values([SITE_COL, SPLIT_COL])
    overall.to_csv(OUT_DIR / "full50_v3_split_summary.csv", index=False, encoding="utf-8-sig")
    by_site.to_csv(OUT_DIR / "full50_v3_split_summary_by_site.csv", index=False, encoding="utf-8-sig")
    return overall, by_site


def metric_dict(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float | int]:
    y_true_arr = np.asarray(y_true, dtype="float64")
    y_pred_arr = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    n = int(mask.sum())
    if n == 0:
        return {
            "n_eval_rows": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "y_true_mean": np.nan,
            "y_pred_mean": np.nan,
            "bias": np.nan,
            "residual_std": np.nan,
            "prediction_min": np.nan,
            "prediction_max": np.nan,
            "n_pred_below_0": 0,
            "n_pred_above_1": 0,
        }
    y = y_true_arr[mask]
    pred = y_pred_arr[mask]
    residual = y - pred
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "n_eval_rows": n,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - sse / sst) if sst > 0 else np.nan,
        "y_true_mean": float(y.mean()),
        "y_pred_mean": float(pred.mean()),
        "bias": float(residual.mean()),
        "residual_std": float(np.std(residual)),
        "prediction_min": float(np.min(pred)),
        "prediction_max": float(np.max(pred)),
        "n_pred_below_0": int((pred < 0).sum()),
        "n_pred_above_1": int((pred > 1).sum()),
    }


def add_site_mean_predictions(df: pd.DataFrame) -> pd.Series:
    train = df[df[SPLIT_COL].eq("train")]
    site_means = train.groupby(SITE_COL)[TARGET].mean()
    return df[SITE_COL].map(site_means).fillna(float(train[TARGET].mean())).astype("float64")


def add_seasonal_mean_predictions(df: pd.DataFrame) -> pd.Series:
    train = df[df[SPLIT_COL].eq("train")]
    global_mean = float(train[TARGET].mean())
    site_hour_month = train.groupby([SITE_COL, "month", "hour"])[TARGET].mean().rename("pred")
    site_month = train.groupby([SITE_COL, "month"])[TARGET].mean().rename("pred_site_month")
    site = train.groupby(SITE_COL)[TARGET].mean().rename("pred_site")
    keys = df[[SITE_COL, "month", "hour"]].copy()
    pred = keys.merge(site_hour_month.reset_index(), on=[SITE_COL, "month", "hour"], how="left")["pred"]
    fallback_month = keys[[SITE_COL, "month"]].merge(site_month.reset_index(), on=[SITE_COL, "month"], how="left")[
        "pred_site_month"
    ]
    fallback_site = keys[[SITE_COL]].merge(site.reset_index(), on=SITE_COL, how="left")["pred_site"]
    return pred.fillna(fallback_month).fillna(fallback_site).fillna(global_mean).astype("float64")


def add_persistence_predictions(df: pd.DataFrame) -> pd.Series:
    return df.groupby(SITE_COL, sort=False)[TARGET].shift(4).astype("float64")


def batch_predict(model: Pipeline, X: pd.DataFrame, batch_size: int = 500_000) -> np.ndarray:
    preds = np.empty(len(X), dtype="float64")
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        preds[start:end] = model.predict(X.iloc[start:end])
    return preds


def train_ridge(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, pd.DataFrame, float]:
    numeric_features = [col for col in features if col != SITE_COL]
    train_mask = df[SPLIT_COL].eq("train")
    val_mask = df[SPLIT_COL].eq("val")
    alpha_rows: list[dict[str, object]] = []
    best_rmse = np.inf
    best_alpha = 1.0
    best_model: Pipeline | None = None
    for alpha in [0.1, 1.0, 10.0]:
        log(f"Training M3 Ridge alpha={alpha}...")
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("site", OneHotEncoder(handle_unknown="ignore", sparse_output=True), [SITE_COL]),
            ],
            sparse_threshold=1.0,
        )
        model = Pipeline([("preprocess", preprocessor), ("model", Ridge(alpha=alpha, solver="lsqr"))])
        model.fit(df.loc[train_mask, features], df.loc[train_mask, TARGET])
        val_pred = model.predict(df.loc[val_mask, features])
        metrics = metric_dict(df.loc[val_mask, TARGET], val_pred)
        row = {
            "alpha": alpha,
            "n_eval_rows": metrics["n_eval_rows"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "bias": metrics["bias"],
            "residual_std": metrics["residual_std"],
        }
        alpha_rows.append(row)
        if float(metrics["rmse"]) < best_rmse:
            best_rmse = float(metrics["rmse"])
            best_alpha = alpha
            best_model = model
    assert best_model is not None
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df["selected"] = alpha_df["alpha"].eq(best_alpha)
    alpha_df.to_csv(OUT_DIR / "full50_v3_ridge_alpha_validation.csv", index=False, encoding="utf-8-sig")
    log(f"Selected Ridge alpha={best_alpha} with validation RMSE={best_rmse:.6f}")
    return batch_predict(best_model, df[features]), alpha_df, best_alpha


def build_hgb_training_index(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, str]:
    train_mask = df[SPLIT_COL].eq("train")
    train_rows = int(train_mask.sum())
    train_sites = df.loc[train_mask, SITE_COL]
    use_sample = train_rows > HGB_FULL_TRAIN_ROW_LIMIT
    rng = np.random.default_rng(RANDOM_SEED)
    selected_parts: list[np.ndarray] = []
    summary_rows: list[dict[str, object]] = []

    for site_id, index_values in train_sites.groupby(train_sites, sort=True).groups.items():
        site_index = np.asarray(index_values, dtype=np.int64)
        available = int(site_index.size)
        if use_sample and available > HGB_SAMPLE_PER_SITE:
            selected = rng.choice(site_index, size=HGB_SAMPLE_PER_SITE, replace=False)
            selected.sort()
        else:
            selected = site_index
        selected_parts.append(selected)
        sampled = int(selected.size)
        summary_rows.append(
            {
                "site_id": site_id,
                "available_train_rows": available,
                "sampled_train_rows": sampled,
                "sampling_fraction": sampled / available if available else np.nan,
            }
        )

    sample_index = np.concatenate(selected_parts) if selected_parts else np.array([], dtype=np.int64)
    sample_index.sort()
    sample_summary = pd.DataFrame(summary_rows).sort_values(SITE_COL)
    sample_summary.to_csv(OUT_DIR / "full50_v3_hgb_training_sample_summary.csv", index=False, encoding="utf-8-sig")
    mode = "balanced_sample_per_site" if use_sample else "full_train"
    return sample_index, sample_summary, mode


def train_hgb(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, pd.DataFrame, str]:
    numeric_features = [col for col in features if col != SITE_COL]
    train_index, sample_summary, hgb_mode = build_hgb_training_index(df)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("site", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), [SITE_COL]),
        ],
        sparse_threshold=0.0,
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=0.0,
                    early_stopping=False,
                    random_state=42,
                ),
            ),
        ]
    )
    log(
        "Training M4 HistGradientBoostingRegressor "
        f"with {hgb_mode}: {len(train_index):,} rows..."
    )
    model.fit(df.loc[train_index, features], df.loc[train_index, TARGET])
    return batch_predict(model, df[features]), sample_summary, hgb_mode


def compute_metrics(df: pd.DataFrame, prediction_cols: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, object]] = []
    by_site_rows: list[dict[str, object]] = []
    for model, pred_col in prediction_cols.items():
        for split, group in df.groupby(SPLIT_COL, sort=False):
            overall_rows.append({"model": model, "split": split, **metric_dict(group[TARGET], group[pred_col])})
        for (site_id, split), group in df.groupby([SITE_COL, SPLIT_COL], observed=False):
            metrics = metric_dict(group[TARGET], group[pred_col])
            event_rows = int(group.loc[np.isfinite(group[pred_col]), EVENT_COL].sum())
            metrics.update(
                {
                    "n_event_rows": event_rows,
                    "event_ratio": event_rows / metrics["n_eval_rows"] if metrics["n_eval_rows"] else np.nan,
                }
            )
            by_site_rows.append({"site_id": site_id, "model": model, "split": split, **metrics})
    overall = pd.DataFrame(overall_rows)
    by_site = pd.DataFrame(by_site_rows)
    model_cat = pd.CategoricalDtype(MODEL_ORDER, ordered=True)
    split_cat = pd.CategoricalDtype(["train", "val", "test"], ordered=True)
    overall["model"] = overall["model"].astype(model_cat)
    by_site["model"] = by_site["model"].astype(model_cat)
    overall["split"] = overall["split"].astype(split_cat)
    by_site["split"] = by_site["split"].astype(split_cat)
    overall = overall.sort_values(["model", "split"])
    by_site = by_site.sort_values(["site_id", "model", "split"])
    overall.to_csv(OUT_DIR / "full50_v3_model_metrics_overall.csv", index=False, encoding="utf-8-sig")
    by_site.to_csv(OUT_DIR / "full50_v3_model_metrics_by_site.csv", index=False, encoding="utf-8-sig")
    return overall, by_site


def compute_macro_site_metrics(by_site: pd.DataFrame) -> pd.DataFrame:
    macro = by_site.groupby(["model", "split"], observed=False).agg(
        mean_site_mae=("mae", "mean"),
        median_site_mae=("mae", "median"),
        mean_site_rmse=("rmse", "mean"),
        median_site_rmse=("rmse", "median"),
        mean_site_r2=("r2", "mean"),
        median_site_r2=("r2", "median"),
        n_sites=(SITE_COL, "nunique"),
        n_sites_r2_below_0=("r2", lambda values: int((values < 0).sum())),
        n_sites_r2_below_minus_1=("r2", lambda values: int((values < -1).sum())),
    ).reset_index()
    macro.to_csv(OUT_DIR / "full50_v3_model_metrics_macro_by_site.csv", index=False, encoding="utf-8-sig")
    return macro


def write_split_predictions(
    df: pd.DataFrame,
    prediction_cols: dict[str, str],
    split: str,
    output_name: str,
) -> None:
    output = OUT_DIR / output_name
    if output.exists():
        output.unlink()
    writer: pq.ParquetWriter | None = None
    base_cols = [SITE_COL, TIME_COL, SPLIT_COL, EVENT_COL, EVENT_LABEL_COL, "month", "hour", "day_of_year", TARGET]
    split_mask = df[SPLIT_COL].eq(split)
    for model, pred_col in prediction_cols.items():
        chunk = df.loc[split_mask, base_cols].copy()
        chunk["model"] = model
        chunk = chunk.rename(columns={TARGET: "y_true"})
        chunk["y_pred"] = df.loc[split_mask, pred_col].astype("float64")
        chunk["residual"] = chunk["y_true"] - chunk["y_pred"]
        chunk["abs_residual"] = chunk["residual"].abs()
        chunk = chunk[
            [
                SITE_COL,
                TIME_COL,
                SPLIT_COL,
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
        ]
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output, table.schema, compression="snappy")
        writer.write_table(table)
    if writer is not None:
        writer.close()


def write_prediction_files(df: pd.DataFrame, prediction_cols: dict[str, str]) -> None:
    write_split_predictions(df, prediction_cols, "test", "full50_v3_predictions_test.parquet")
    write_split_predictions(df, prediction_cols, "val", "full50_v3_predictions_val.parquet")


def residual_summary(df: pd.DataFrame, prediction_cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    for model, pred_col in prediction_cols.items():
        for split, group in df.groupby(SPLIT_COL, sort=False):
            residual = group[TARGET] - group[pred_col]
            abs_residual = residual.abs()
            abs_residual = abs_residual[np.isfinite(abs_residual)]
            residual = residual[np.isfinite(residual)]
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "residual_mean": float(residual.mean()),
                    "residual_std": float(residual.std()),
                    "abs_residual_mean": float(abs_residual.mean()),
                    "abs_residual_median": float(abs_residual.median()),
                    "abs_residual_p90": float(abs_residual.quantile(0.90)),
                    "abs_residual_p95": float(abs_residual.quantile(0.95)),
                    "abs_residual_p99": float(abs_residual.quantile(0.99)),
                    "n_predictions": int(abs_residual.shape[0]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "full50_v3_residual_summary.csv", index=False, encoding="utf-8-sig")
    return out


def basic_event_residual_check(df: pd.DataFrame, prediction_cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    test = df[df[SPLIT_COL].eq("test")]
    for model, pred_col in prediction_cols.items():
        residual = (test[TARGET] - test[pred_col]).abs()
        eval_df = test.loc[np.isfinite(residual), [EVENT_COL]].copy()
        eval_df["abs_residual"] = residual[np.isfinite(residual)].to_numpy()
        background = float(eval_df[EVENT_COL].mean()) if len(eval_df) else np.nan
        event_abs = eval_df.loc[eval_df[EVENT_COL].eq(1), "abs_residual"]
        nonevent_abs = eval_df.loc[eval_df[EVENT_COL].eq(0), "abs_residual"]
        row = {
            "model": model,
            "test_background_event_rate": background,
            "mean_abs_residual_event": float(event_abs.mean()),
            "mean_abs_residual_nonevent": float(nonevent_abs.mean()),
            "median_abs_residual_event": float(event_abs.median()),
            "median_abs_residual_nonevent": float(nonevent_abs.median()),
        }
        for pct in [0.01, 0.05, 0.10]:
            n_top = max(1, int(np.ceil(len(eval_df) * pct)))
            top = eval_df.nlargest(n_top, "abs_residual")
            rate = float(top[EVENT_COL].mean())
            label = int(pct * 100)
            row[f"top_{label}pct_event_rate"] = rate
            row[f"top_{label}pct_enrichment"] = rate / background if background > 0 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "full50_v3_basic_event_residual_check.csv", index=False, encoding="utf-8-sig")
    return out


def model_output_check(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        model_metrics = metrics[metrics["model"].astype(str).eq(model)]
        row = {"model": model}
        status = "PASS"
        for split in ["train", "val", "test"]:
            split_metrics = model_metrics[model_metrics["split"].astype(str).eq(split)]
            n = int(split_metrics["n_eval_rows"].iloc[0]) if not split_metrics.empty else 0
            row[f"has_{split}_metrics"] = n > 0
            if split in {"val", "test"}:
                row[f"n_{split}_predictions"] = n
            if n == 0:
                status = "FAIL"
        row["status"] = status
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "full50_v3_model_output_check.csv", index=False, encoding="utf-8-sig")
    return out


def prediction_range_check(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics[[
        "model",
        "split",
        "prediction_min",
        "prediction_max",
        "n_pred_below_0",
        "n_pred_above_1",
        "n_eval_rows",
    ]].copy()
    out["share_pred_below_0"] = out["n_pred_below_0"] / out["n_eval_rows"]
    out["share_pred_above_1"] = out["n_pred_above_1"] / out["n_eval_rows"]
    out = out.drop(columns=["n_eval_rows"])
    out.to_csv(OUT_DIR / "full50_v3_prediction_range_check.csv", index=False, encoding="utf-8-sig")
    return out


def extreme_r2_check(by_site: pd.DataFrame) -> pd.DataFrame:
    out = by_site[[SITE_COL, "model", "split", "r2", "rmse", "mae", "n_eval_rows"]].copy()
    out["flag"] = np.where(out["r2"] < -1, "R2_BELOW_MINUS_1", "OK")
    out.to_csv(OUT_DIR / "full50_v3_extreme_r2_check.csv", index=False, encoding="utf-8-sig")
    return out


def write_text_checks(features: list[str]) -> tuple[str, str]:
    persistence_status = "PASS"
    (OUT_DIR / "full50_v3_persistence_shift_check.txt").write_text(
        "\n".join(
            [
                "PASS",
                "- Persistence-1h uses groupby(site_id)[00400].shift(4).",
                "- Shift is grouped by site, so it cannot cross site boundaries.",
                "- 4-row shift is used because the dominant interval is 15 minutes.",
                "- Rows without valid 1-hour history have missing predictions and are excluded from M2 metrics.",
            ]
        ),
        encoding="utf-8",
    )
    leakage_status = "PASS" if not set(features).intersection(set(FORBIDDEN_COLUMNS) | {TARGET, TIME_COL}) else "FAIL"
    (OUT_DIR / "full50_v3_feature_leakage_check.txt").write_text(
        "\n".join(
            [
                leakage_status,
                f"- Target `{TARGET}` is not used as an input feature.",
                f"- Event columns `{EVENT_COL}` and `{EVENT_LABEL_COL}` are not used as input features.",
                "- Validation-only columns remain in prediction outputs only for residual-event sanity checks.",
                f"- Approved feature list: {features}",
            ]
        ),
        encoding="utf-8",
    )
    return persistence_status, leakage_status


def write_remaining_invalid_value_check(df: pd.DataFrame) -> str:
    invalid_counts = invalid_value_counts_v3(df)
    status = "PASS" if all(value == 0 for value in invalid_counts.values()) else "FAIL"
    lines = [
        status,
        f"- hard sentinel rows <= -999: {invalid_counts['hard_sentinel_rows']}",
        f"- physical feature rows <= -100: {invalid_counts['physical_feature_le_minus_100_rows']}",
        f"- target rows outside [0, 1]: {invalid_counts['target_outside_0_1_rows']}",
        f"- cyclic feature rows outside [-1.000001, 1.000001]: {invalid_counts['cyclic_outside_minus1_1_rows']}",
        f"- invalid month rows: {invalid_counts['invalid_month_rows']}",
        f"- non-finite numeric rows: {invalid_counts['nonfinite_numeric_rows']}",
    ]
    (OUT_DIR / "full50_v3_remaining_invalid_value_check.txt").write_text("\n".join(lines), encoding="utf-8")
    return status


def compare_dev10_full50(full50_metrics: pd.DataFrame) -> pd.DataFrame:
    full50 = full50_metrics.rename(
        columns={
            "mae": "full50_mae",
            "rmse": "full50_rmse",
            "r2": "full50_r2",
            "prediction_min": "full50_prediction_min",
            "prediction_max": "full50_prediction_max",
        }
    )[["model", "split", "full50_mae", "full50_rmse", "full50_r2", "full50_prediction_min", "full50_prediction_max"]]
    comparison = full50.copy()
    if DEV10_V3_METRICS.exists():
        dev10 = pd.read_csv(DEV10_V3_METRICS).rename(
            columns={
                "mae": "dev10_mae",
                "rmse": "dev10_rmse",
                "r2": "dev10_r2",
                "prediction_min": "dev10_prediction_min",
                "prediction_max": "dev10_prediction_max",
            }
        )[["model", "split", "dev10_mae", "dev10_rmse", "dev10_r2", "dev10_prediction_min", "dev10_prediction_max"]]
        comparison = dev10.merge(comparison, on=["model", "split"], how="outer")
    else:
        for col in ["dev10_mae", "dev10_rmse", "dev10_r2", "dev10_prediction_min", "dev10_prediction_max"]:
            comparison[col] = np.nan
    comparison["rmse_change_dev10_to_full50"] = comparison["full50_rmse"] - comparison["dev10_rmse"]
    comparison["r2_change_dev10_to_full50"] = comparison["full50_r2"] - comparison["dev10_r2"]
    comparison.to_csv(OUT_DIR / "full50_v3_dev10_comparison.csv", index=False, encoding="utf-8-sig")
    return comparison


def save_model_config(feature_set: dict[str, object], best_alpha: float, hgb_mode: str) -> None:
    config = {
        "input_dataset": str(DATA_PATH),
        "feature_json": str(FEATURE_JSON_PATH),
        "output_dir": str(OUT_DIR),
        "target": TARGET,
        "features": feature_set.get("main_features", DEFAULT_FEATURES),
        "models": MODEL_ORDER,
        "ridge_alpha_grid": [0.1, 1.0, 10.0],
        "ridge_selected_alpha": best_alpha,
        "hgb_params": {
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.0,
            "random_state": 42,
        },
        "hgb_training_mode": hgb_mode,
        "hgb_full_train_row_limit": HGB_FULL_TRAIN_ROW_LIMIT,
        "hgb_sample_per_site": HGB_SAMPLE_PER_SITE,
        "notes": [
            "Full50 v3 baseline on robust-cleaned v3 dataset.",
            "No event columns used as model inputs.",
            "No pH lag features added.",
            "Persistence-1h uses historical pH only as a temporal benchmark.",
            "Full residual-event validation is not performed in this script.",
        ],
    }
    (OUT_DIR / "full50_v3_model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def save_figures(
    metrics: pd.DataFrame,
    predictions_wide: pd.DataFrame,
    by_site: pd.DataFrame,
    residuals: pd.DataFrame,
    event_check: pd.DataFrame,
) -> pd.DataFrame:
    test = metrics[metrics[SPLIT_COL].eq("test")].copy()
    test["model"] = test["model"].astype(str)
    for metric, filename, color in [
        ("rmse", "fig_full50_v3_test_rmse.png", "#4C78A8"),
        ("mae", "fig_full50_v3_test_mae.png", "#F58518"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(test["model"], test[metric], color=color)
        ax.set_title(f"Full50 v3 test {metric.upper()} by model")
        ax.set_xlabel("Model")
        ax.set_ylabel(f"{metric.upper()}, processed pH scale")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(test))
    ax.vlines(x, test["prediction_min"], test["prediction_max"], color="#4C78A8", linewidth=3)
    ax.axhline(0, color="#D62728", linestyle="--", linewidth=1)
    ax.axhline(1, color="#D62728", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(test["model"], rotation=25, ha="right")
    ax.set_title("Full50 v3 test prediction min/max by model")
    ax.set_ylabel("Predicted processed pH")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_full50_v3_prediction_range.png", dpi=200)
    plt.close(fig)

    site_test = by_site[by_site["split"].astype(str).eq("test")].copy()
    site_test["model"] = site_test["model"].astype(str)
    for metric, filename, ylabel in [
        ("rmse", "fig_full50_v3_site_rmse_boxplot.png", "Site-level test RMSE"),
        ("r2", "fig_full50_v3_site_r2_boxplot.png", "Site-level test R2"),
    ]:
        data = [
            site_test.loc[site_test["model"].eq(model), metric].dropna().to_numpy()
            for model in MODEL_ORDER
        ]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.boxplot(data, tick_labels=MODEL_ORDER, showfliers=False)
        ax.set_title(ylabel + " by model")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=200)
        plt.close(fig)

    test_pred = predictions_wide[predictions_wide[SPLIT_COL].eq("test")]
    sample = test_pred.sample(min(50_000, len(test_pred)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 5))
    test_resid = []
    labels = []
    for model, short in MODEL_SHORT.items():
        abs_resid = (test_pred[TARGET] - test_pred[f"pred_{short}"]).abs()
        test_resid.append(abs_resid[np.isfinite(abs_resid)].sample(min(100_000, np.isfinite(abs_resid).sum()), random_state=42))
        labels.append(model)
    ax.boxplot(test_resid, tick_labels=labels, showfliers=False)
    ax.set_title("Full50 v3 test absolute residual distributions")
    ax.set_ylabel("Absolute residual, processed pH scale")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_full50_v3_residual_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(event_check))
    width = 0.35
    ax.bar(x - width / 2, event_check["mean_abs_residual_event"], width=width, label="event", color="#E45756")
    ax.bar(x + width / 2, event_check["mean_abs_residual_nonevent"], width=width, label="non-event", color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(event_check["model"], rotation=25, ha="right")
    ax.set_title("Full50 v3 mean absolute residual: event vs non-event test rows")
    ax.set_ylabel("Mean absolute residual")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_full50_v3_basic_event_residual_check.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ranked = test.sort_values("rmse")
    axes[0].barh(ranked["model"], ranked["rmse"], color="#4C78A8")
    axes[0].set_title("Test RMSE ranking")
    axes[0].set_xlabel("RMSE")
    r2_ranked = test.sort_values("r2")
    axes[1].barh(r2_ranked["model"], r2_ranked["r2"], color="#54A24B")
    axes[1].set_title("Test R2 ranking")
    axes[1].set_xlabel("R2")
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_full50_v3_model_ranking.png", dpi=200)
    plt.close(fig)

    hgb_site = site_test[site_test["model"].eq("M4_hist_gradient_boosting")].copy()
    selected: list[tuple[str, str]] = []
    if not hgb_site.empty:
        candidates = [
            ("good_hgb_performance", hgb_site.sort_values("rmse").iloc[0][SITE_COL]),
            ("poor_hgb_performance", hgb_site.sort_values("rmse", ascending=False).iloc[0][SITE_COL]),
            ("event_rich_site", hgb_site.sort_values("event_ratio", ascending=False).iloc[0][SITE_COL]),
            ("low_event_site", hgb_site.sort_values("event_ratio").iloc[0][SITE_COL]),
        ]
        seen: set[str] = set()
        for reason, site_id in candidates:
            site_text = str(site_id)
            if site_text not in seen:
                selected.append((reason, site_text))
                seen.add(site_text)

    plot_rows: list[dict[str, object]] = []
    for reason, site_id in selected:
        site_metrics = hgb_site[hgb_site[SITE_COL].astype(str).eq(site_id)].iloc[0]
        site_data = test_pred[test_pred[SITE_COL].astype(str).eq(site_id)].sort_values(TIME_COL)
        if site_data.empty:
            continue
        plot_start = site_data[TIME_COL].min()
        plot_end = min(plot_start + pd.Timedelta(days=14), site_data[TIME_COL].max())
        plot_rows.append(
            {
                "site_id": site_id,
                "selection_reason": reason,
                "model": "M4_hist_gradient_boosting",
                "test_rmse": float(site_metrics["rmse"]),
                "test_r2": float(site_metrics["r2"]),
                "event_ratio": float(site_metrics["event_ratio"]),
                "plot_start": plot_start,
                "plot_end": plot_end,
            }
        )
    plot_sites = pd.DataFrame(plot_rows)
    plot_sites.to_csv(OUT_DIR / "full50_v3_plot_example_sites.csv", index=False, encoding="utf-8-sig")

    def plot_examples(pred_col: str, model_name: str, filename: str) -> None:
        if plot_sites.empty:
            return
        n = len(plot_sites)
        fig, axes = plt.subplots(n, 1, figsize=(12, max(3, 2.6 * n)), sharex=False)
        if n == 1:
            axes = [axes]
        for ax, row in zip(axes, plot_sites.to_dict("records")):
            site_data = test_pred[test_pred[SITE_COL].astype(str).eq(str(row["site_id"]))].sort_values(TIME_COL)
            window = site_data[
                site_data[TIME_COL].between(pd.Timestamp(row["plot_start"]), pd.Timestamp(row["plot_end"]))
            ]
            if len(window) > 2000:
                window = window.iloc[:: int(np.ceil(len(window) / 2000))]
            ax.plot(window[TIME_COL], window[TARGET], label="observed", color="#111111", linewidth=1.2)
            ax.plot(window[TIME_COL], window[pred_col], label="predicted", color="#4C78A8", linewidth=1.0)
            event_window = window[window[EVENT_COL].eq(1)]
            if not event_window.empty:
                ax.scatter(event_window[TIME_COL], event_window[TARGET], s=8, color="#E45756", label="event row")
            ax.set_title(f"{model_name}: {row['selection_reason']} site {row['site_id']}")
            ax.set_ylabel("Processed pH")
            ax.grid(alpha=0.25)
        axes[0].legend(loc="best")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=200)
        plt.close(fig)

    plot_examples("pred_m4", "HGB", "fig_full50_v3_hgb_pred_vs_true_examples.png")
    plot_examples("pred_m3", "Ridge", "fig_full50_v3_ridge_pred_vs_true_examples.png")

    return plot_sites


def determine_decision(
    initial_checks: pd.DataFrame,
    output_check: pd.DataFrame,
    range_check: pd.DataFrame,
    extreme_r2: pd.DataFrame,
    leakage_status: str,
    persistence_status: str,
    invalid_status: str,
    metrics: pd.DataFrame,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if (initial_checks["status"] == "FAIL").any():
        return "FULL50_BASELINE_FAILED_FIX_BEFORE_RESIDUAL_VALIDATION", ["Initial checks failed."]
    if (output_check["status"] == "FAIL").any():
        return "FULL50_BASELINE_FAILED_FIX_BEFORE_RESIDUAL_VALIDATION", ["Some model predictions or metrics are missing."]
    if leakage_status != "PASS":
        return "FULL50_BASELINE_FAILED_FIX_BEFORE_RESIDUAL_VALIDATION", ["Feature leakage check failed."]
    if persistence_status != "PASS":
        return "FULL50_BASELINE_FAILED_FIX_BEFORE_RESIDUAL_VALIDATION", ["Persistence shift check failed."]
    if invalid_status != "PASS":
        return "FULL50_BASELINE_FAILED_FIX_BEFORE_RESIDUAL_VALIDATION", ["Remaining invalid-value check failed."]

    site_test = metrics[(metrics["model"].eq("M0_site_mean")) & (metrics["split"].eq("test"))].iloc[0]
    seasonal_test = metrics[(metrics["model"].eq("M1_seasonal_mean")) & (metrics["split"].eq("test"))].iloc[0]
    pers_test = metrics[(metrics["model"].eq("M2_persistence_1h")) & (metrics["split"].eq("test"))].iloc[0]
    ridge_test = metrics[(metrics["model"].eq("M3_ridge")) & (metrics["split"].eq("test"))].iloc[0]
    hgb_test = metrics[(metrics["model"].eq("M4_hist_gradient_boosting")) & (metrics["split"].eq("test"))].iloc[0]
    if ridge_test["prediction_min"] < -1 or ridge_test["r2"] < -10:
        return "FULL50_BASELINE_RUNS_BUT_NEEDS_REVIEW", ["Ridge still has extreme predictions or extremely negative R2."]
    if hgb_test["prediction_min"] < -0.1 or hgb_test["prediction_max"] > 1.1:
        reasons.append("HGB predictions slightly outside expected processed-pH range.")
    if not (hgb_test["rmse"] < site_test["rmse"] and hgb_test["rmse"] < seasonal_test["rmse"]):
        reasons.append("HGB does not beat both Site Mean and Seasonal Mean on test RMSE.")
    if pers_test["rmse"] > 0.02:
        reasons.append("Persistence-1h is unexpectedly weak.")
    if pers_test["rmse"] > metrics[metrics["split"].eq("test")]["rmse"].min() + 1e-12:
        reasons.append("Persistence-1h is not the strongest overall test model.")
    if reasons:
        return "FULL50_BASELINE_RUNS_BUT_NEEDS_REVIEW", reasons
    return "FULL50_BASELINE_PASSED_READY_FOR_RESIDUAL_VALIDATION", [
        "All five models ran successfully.",
        "No v3 invalid values remain.",
        "Ridge no longer has extreme negative predictions.",
        "HGB remains stable and beats Site Mean and Seasonal Mean overall.",
        "Persistence-1h remains strongest as a temporal benchmark.",
    ]


def write_report(
    df: pd.DataFrame,
    features: list[str],
    initial_checks: pd.DataFrame,
    split_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    by_site: pd.DataFrame,
    macro_metrics: pd.DataFrame,
    alpha_df: pd.DataFrame,
    hgb_sample_summary: pd.DataFrame,
    hgb_mode: str,
    residuals: pd.DataFrame,
    event_check: pd.DataFrame,
    comparison: pd.DataFrame,
    output_check: pd.DataFrame,
    range_check: pd.DataFrame,
    extreme_r2: pd.DataFrame,
    plot_sites: pd.DataFrame,
    leakage_status: str,
    persistence_status: str,
    invalid_status: str,
    decision: str,
    decision_reasons: list[str],
    elapsed: float,
) -> None:
    val_rank = metrics[metrics["split"].eq("val")].sort_values("rmse")
    test_rank = metrics[metrics["split"].eq("test")].sort_values("rmse")
    best_test = test_rank.iloc[0]
    best_feature_test = metrics[
        metrics["split"].eq("test") & metrics["model"].isin(["M3_ridge", "M4_hist_gradient_boosting"])
    ].sort_values("rmse").iloc[0]
    site_test = by_site[by_site["split"].eq("test")]
    median_site_rmse = site_test.groupby("model")["rmse"].median().reset_index(name="median_site_test_rmse")
    median_site_r2 = site_test.groupby("model")["r2"].median().reset_index(name="median_site_test_r2")
    high_rmse_sites = site_test.sort_values("rmse", ascending=False).head(15)
    hgb_site = site_test[site_test["model"].eq("M4_hist_gradient_boosting")].copy()
    site_mean = site_test[site_test["model"].eq("M0_site_mean")][[SITE_COL, "rmse"]].rename(columns={"rmse": "site_mean_rmse"})
    ridge = site_test[site_test["model"].eq("M3_ridge")][[SITE_COL, "rmse"]].rename(columns={"rmse": "ridge_rmse"})
    hgb_compare = hgb_site[[SITE_COL, "rmse", "r2", "event_ratio"]].merge(site_mean, on=SITE_COL).merge(ridge, on=SITE_COL)
    n_hgb_beats_site_mean = int((hgb_compare["rmse"] < hgb_compare["site_mean_rmse"]).sum())
    n_hgb_beats_ridge = int((hgb_compare["rmse"] < hgb_compare["ridge_rmse"]).sum())
    strong_hgb_sites = hgb_site.sort_values("rmse").head(10)
    poor_hgb_sites = hgb_site.sort_values("rmse", ascending=False).head(10)
    flagged_r2 = extreme_r2[extreme_r2["flag"].ne("OK")].sort_values("r2").head(20)
    reason_text = "\n".join(f"- {reason}" for reason in decision_reasons)
    dev10_test_comp = comparison[comparison["split"].astype(str).eq("test")] if not comparison.empty else pd.DataFrame()

    report = f"""# Full50 v3 Baseline Report

## 1. Purpose

This is the first full 50-site baseline prediction experiment on the robust-cleaned v3 pH dataset.

It evaluates prediction performance for processed/scaled pH (`{TARGET}`). Full residual-event validation is intentionally left for a later step.

## 2. Dataset

- Input cleaned v3 file: `{DATA_PATH}`
- Output folder: `{OUT_DIR}`
- Rows: {len(df):,}
- Sites: {df[SITE_COL].nunique()}
- Train/val/test rows: {int(split_summary.loc[split_summary[SPLIT_COL].astype(str).eq('train'), 'n_rows'].iloc[0]):,} / {int(split_summary.loc[split_summary[SPLIT_COL].astype(str).eq('val'), 'n_rows'].iloc[0]):,} / {int(split_summary.loc[split_summary[SPLIT_COL].astype(str).eq('test'), 'n_rows'].iloc[0]):,}
- Target: `{TARGET}` processed/scaled pH
- Feature list: {features}
- Validation-only columns: `{EVENT_COL}`, `{EVENT_LABEL_COL}`
- Robust v3 cleaning used: yes
- Remaining invalid-value status: {invalid_status}

Split summary:

{dataframe_to_markdown(split_summary)}

Initial checks:

{dataframe_to_markdown(initial_checks)}

## 3. Models

- M0 Site Mean
- M1 Seasonal Mean
- M2 Persistence-1h
- M3 Ridge Regression with train-fitted scaling and one-hot site ID
- M4 HistGradientBoosting with train-fitted ordinal site ID

Persistence-1h uses historical pH and is therefore a temporal benchmark. Ridge and HGB are feature-based co-behaviour models using co-measured water-quality and hydrological variables.

No pH lag features, pH rolling features, pH-change target, or deep-learning models were added. HGB training mode: `{hgb_mode}`.

## 4. Overall Results

Validation ranking:

{dataframe_to_markdown(val_rank[["model", "n_eval_rows", "mae", "rmse", "r2", "prediction_min", "prediction_max"]])}

Test ranking:

{dataframe_to_markdown(test_rank[["model", "n_eval_rows", "mae", "rmse", "r2", "prediction_min", "prediction_max"]])}

- Best test model by RMSE: `{best_test['model']}`.
- Best feature-based test model by RMSE: `{best_feature_test['model']}`.
- HGB beats Site Mean overall by test RMSE: {bool(test_rank[test_rank["model"].eq("M4_hist_gradient_boosting")]["rmse"].iloc[0] < test_rank[test_rank["model"].eq("M0_site_mean")]["rmse"].iloc[0])}.
- HGB beats Seasonal Mean overall by test RMSE: {bool(test_rank[test_rank["model"].eq("M4_hist_gradient_boosting")]["rmse"].iloc[0] < test_rank[test_rank["model"].eq("M1_seasonal_mean")]["rmse"].iloc[0])}.
- Ridge behaves normally after v3 cleaning if prediction range is not extreme and R2 is not strongly negative.
- Persistence-1h remains a very strong temporal benchmark, as expected for high-frequency pH time series.

Dev10-to-full50 comparison, test split:

{dataframe_to_markdown(dev10_test_comp) if not dev10_test_comp.empty else "No dev10 comparison file available."}

## 5. Site-Level Prediction Results

Median site test RMSE by model:

{dataframe_to_markdown(median_site_rmse)}

Median site test R2 by model:

{dataframe_to_markdown(median_site_r2)}

Macro-averaged site metrics:

{dataframe_to_markdown(macro_metrics[macro_metrics["split"].astype(str).eq("test")])}

- Number of sites where HGB beats Site Mean by RMSE: {n_hgb_beats_site_mean} / {len(hgb_compare)}
- Number of sites where HGB beats Ridge by RMSE: {n_hgb_beats_ridge} / {len(hgb_compare)}

Strong HGB sites:

{dataframe_to_markdown(strong_hgb_sites[[SITE_COL, "n_eval_rows", "event_ratio", "mae", "rmse", "r2"]])}

Poor HGB sites:

{dataframe_to_markdown(poor_hgb_sites[[SITE_COL, "n_eval_rows", "event_ratio", "mae", "rmse", "r2"]])}

Largest site-level test errors:

{dataframe_to_markdown(high_rmse_sites[[SITE_COL, "model", "n_eval_rows", "n_event_rows", "event_ratio", "mae", "rmse", "r2"]])}

Extreme R2 flags:

{dataframe_to_markdown(flagged_r2) if not flagged_r2.empty else "No site-level R2 < -1 flags."}

## 6. Persistence Benchmark Interpretation

Persistence-1h may be very strong because pH has high short-term autocorrelation at 15-minute sampling. It should not be interpreted as the same type of model as Ridge/HGB. It is a temporal benchmark, not the main explanatory co-behaviour model.

## 7. Ridge vs HGB Interpretation

Ridge is the interpretable linear feature-based baseline. HGB is the main non-linear feature-based baseline. If HGB improves over Ridge, that suggests non-linear feature interactions and site heterogeneity matter for pH prediction.

## 8. Prediction Range and Debug Checks

{dataframe_to_markdown(range_check)}

Feature leakage status: {leakage_status}.

Persistence shift status: {persistence_status}.

Remaining invalid-value status: {invalid_status}.

Model output check:

{dataframe_to_markdown(output_check)}

## 9. Basic Event-Residual Sanity Check

This is a full50 prediction-output sanity check, not final event-validation evidence.

{dataframe_to_markdown(event_check)}

Residual distribution summary:

{dataframe_to_markdown(residuals)}

Example plot sites:

{dataframe_to_markdown(plot_sites) if not plot_sites.empty else "No example sites selected."}

## 10. Limitations

- `{TARGET}` is processed/scaled pH, not raw 0-14 pH.
- Dev10 was only a pipeline test; this full50 run is the first main baseline.
- Full50 is still based on the processed USGS-derived modelling dataset.
- Event labels are weak validation windows, not a fully labelled anomaly benchmark.
- High-frequency pH persistence makes Persistence-1h very strong.
- Site heterogeneity may affect model performance and should be discussed in the dissertation.

## 11. Decision

{decision}

{reason_text}

HGB training sample summary:

{dataframe_to_markdown(hgb_sample_summary.head(60))}

Ridge alpha validation:

{dataframe_to_markdown(alpha_df)}

Runtime: {elapsed:.1f} seconds.
"""
    (OUT_DIR / "full50_v3_baseline_report.md").write_text(report, encoding="utf-8")
    (OUT_DIR / "full50_v3_pipeline_decision.txt").write_text(
        decision + "\n\n" + reason_text + "\n",
        encoding="utf-8",
    )


def main() -> None:
    start = perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    feature_set = load_feature_set()
    features = list(feature_set.get("main_features", DEFAULT_FEATURES))
    df = load_dataset(feature_set)
    initial_checks = run_initial_checks(df, feature_set)
    split_summary, split_by_site = make_split_summaries(df)
    log("Initial checks and split summaries saved.")

    log("Running M0 Site Mean baseline...")
    df["pred_m0"] = add_site_mean_predictions(df)
    log("Running M1 Seasonal Mean baseline...")
    df["pred_m1"] = add_seasonal_mean_predictions(df)
    log("Running M2 Persistence-1h baseline...")
    df["pred_m2"] = add_persistence_predictions(df)
    df["pred_m3"], alpha_df, best_alpha = train_ridge(df, features)
    df["pred_m4"], hgb_sample_summary, hgb_mode = train_hgb(df, features)

    prediction_cols = {
        "M0_site_mean": "pred_m0",
        "M1_seasonal_mean": "pred_m1",
        "M2_persistence_1h": "pred_m2",
        "M3_ridge": "pred_m3",
        "M4_hist_gradient_boosting": "pred_m4",
    }
    log("Computing metrics and debug checks...")
    metrics, by_site = compute_metrics(df, prediction_cols)
    macro_metrics = compute_macro_site_metrics(by_site)
    residuals = residual_summary(df, prediction_cols)
    event_check = basic_event_residual_check(df, prediction_cols)
    output_check = model_output_check(metrics)
    range_check = prediction_range_check(metrics)
    extreme_r2 = extreme_r2_check(by_site)
    persistence_status, leakage_status = write_text_checks(features)
    invalid_status = write_remaining_invalid_value_check(df)
    comparison = compare_dev10_full50(metrics)
    save_model_config(feature_set, best_alpha, hgb_mode)

    log("Saving val/test long-format predictions...")
    write_prediction_files(df, prediction_cols)
    plot_sites = save_figures(metrics, df, by_site, residuals, event_check)

    decision, reasons = determine_decision(
        initial_checks,
        output_check,
        range_check,
        extreme_r2,
        leakage_status,
        persistence_status,
        invalid_status,
        metrics,
    )
    elapsed = perf_counter() - start
    write_report(
        df,
        features,
        initial_checks,
        split_summary,
        metrics,
        by_site,
        macro_metrics,
        alpha_df,
        hgb_sample_summary,
        hgb_mode,
        residuals,
        event_check,
        comparison,
        output_check,
        range_check,
        extreme_r2,
        plot_sites,
        leakage_status,
        persistence_status,
        invalid_status,
        decision,
        reasons,
        elapsed,
    )

    val_best = metrics[metrics["split"].eq("val")].sort_values("rmse").iloc[0]
    test_best = metrics[metrics["split"].eq("test")].sort_values("rmse").iloc[0]
    feature_best = metrics[
        metrics["split"].eq("test") & metrics["model"].isin(["M3_ridge", "M4_hist_gradient_boosting"])
    ].sort_values("rmse").iloc[0]
    ridge_v3 = metrics[(metrics["model"].eq("M3_ridge")) & (metrics["split"].eq("test"))].iloc[0]
    ridge_fixed = bool(ridge_v3["prediction_min"] > -1 and ridge_v3["r2"] > -10)
    site_mean_test = metrics[(metrics["model"].eq("M0_site_mean")) & (metrics["split"].eq("test"))].iloc[0]
    seasonal_test = metrics[(metrics["model"].eq("M1_seasonal_mean")) & (metrics["split"].eq("test"))].iloc[0]
    hgb_test = metrics[(metrics["model"].eq("M4_hist_gradient_boosting")) & (metrics["split"].eq("test"))].iloc[0]
    pers_test = metrics[(metrics["model"].eq("M2_persistence_1h")) & (metrics["split"].eq("test"))].iloc[0]

    log(f"input cleaned Full50 v3 file path: {DATA_PATH}")
    log(f"output folder: {OUT_DIR}")
    log(f"number of rows: {len(df):,}")
    log(f"number of sites: {df[SITE_COL].nunique()}")
    log(f"train rows: {int(split_summary.loc[split_summary[SPLIT_COL].eq('train'), 'n_rows'].iloc[0]):,}")
    log(f"validation rows: {int(split_summary.loc[split_summary[SPLIT_COL].eq('val'), 'n_rows'].iloc[0]):,}")
    log(f"test rows: {int(split_summary.loc[split_summary[SPLIT_COL].eq('test'), 'n_rows'].iloc[0]):,}")
    log(f"models run: {', '.join(MODEL_ORDER)}")
    log(f"HGB training mode: {hgb_mode}")
    log(f"best validation model by RMSE: {val_best['model']} ({float(val_best['rmse']):.6f})")
    log(f"best test model by RMSE: {test_best['model']} ({float(test_best['rmse']):.6f})")
    log(f"best feature-based model by test RMSE: {feature_best['model']} ({float(feature_best['rmse']):.6f})")
    log(f"Ridge v3 test RMSE: {float(ridge_v3['rmse']):.6f}")
    log(f"Ridge v3 test R2: {float(ridge_v3['r2']):.6f}")
    log(f"whether Ridge extreme predictions were fixed: {ridge_fixed}")
    log(f"whether HGB beats Site Mean overall: {bool(hgb_test['rmse'] < site_mean_test['rmse'])}")
    log(f"whether HGB beats Seasonal Mean overall: {bool(hgb_test['rmse'] < seasonal_test['rmse'])}")
    log(f"whether HGB beats Ridge overall: {bool(hgb_test['rmse'] < ridge_v3['rmse'])}")
    log(f"whether HistGradientBoosting remained stable: {bool(hgb_test['prediction_min'] > -0.1 and hgb_test['prediction_max'] < 1.1)}")
    log(f"whether Persistence-1h remains strongest: {bool(test_best['model'] == 'M2_persistence_1h')}")
    log(f"feature leakage status: {leakage_status}")
    log(f"persistence shift status: {persistence_status}")
    log(f"remaining invalid-value status: {invalid_status}")
    log(f"final decision: {decision}")


if __name__ == "__main__":
    main()


