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
BASELINE_DIR = ROOT / "analysis_outputs" / "ph_baseline_full50_v3"
VALIDATION_DIR = ROOT / "analysis_outputs" / "ph_residual_event_validation_full50_v3"
EVENT_WINDOWS_PATH = ROOT / "analysis_outputs" / "ph_dataset" / "ph_event_windows.csv"
OUT_DIR = ROOT / "analysis_outputs" / "ph_nonevent_training_full50_v3"
FIG_DIR = OUT_DIR / "figures"

TARGET = "00400"
SITE_COL = "site_id"
TIME_COL = "timestamp"
SPLIT_COL = "split"
EVENT_COL = "in_ph_event"
EVENT_LABEL_COL = "ph_event_label"
RANDOM_SEED = 42
HGB_SAMPLE_PER_SITE = 100_000
TOP_PCTS = [0.01, 0.05, 0.10]

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
NUMERIC_CHECK_COLUMNS = [
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
FORBIDDEN_COLUMNS = {
    EVENT_COL,
    EVENT_LABEL_COL,
    "incident_start",
    "incident_end",
    "duration_hrs",
    "severity",
    "description",
    "n_points",
    TARGET,
    TIME_COL,
}
MODEL_ORDER = ["M3_ridge_nonevent_train", "M4_hgb_nonevent_train"]
MODEL_SHORT = {
    "M3_ridge_nonevent_train": "ridge_ne",
    "M4_hgb_nonevent_train": "hgb_ne",
}
BASE_MODEL_MAP = {
    "M3_ridge_nonevent_train": "ridge",
    "M4_hgb_nonevent_train": "hgb",
    "M3_ridge": "ridge",
    "M4_hist_gradient_boosting": "hgb",
}


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


def load_feature_set() -> dict[str, object]:
    if not FEATURE_JSON_PATH.exists():
        raise FileNotFoundError(f"Feature JSON not found: {FEATURE_JSON_PATH}")
    return json.loads(FEATURE_JSON_PATH.read_text(encoding="utf-8"))


def load_dataset(feature_set: dict[str, object]) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found: {DATA_PATH}")
    features = list(feature_set.get("main_features", DEFAULT_FEATURES))
    required = sorted(set([TARGET, *features, TIME_COL, SPLIT_COL, EVENT_COL, EVENT_LABEL_COL, "hour", "day_of_year"]))
    log(f"Loading robust-cleaned full50 v3 dataset: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH, columns=required)
    df[SITE_COL] = df[SITE_COL].map(normalise_site_id).astype("string")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df[SPLIT_COL] = df[SPLIT_COL].astype("string")
    df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce").fillna(0).astype("int8")
    df[EVENT_LABEL_COL] = df[EVENT_LABEL_COL].fillna("none").astype("string")
    for col in [TARGET, *[f for f in features if f != SITE_COL], "hour", "day_of_year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values([SITE_COL, TIME_COL]).reset_index(drop=True)


def invalid_value_counts(df: pd.DataFrame) -> dict[str, int]:
    hard = pd.Series(False, index=df.index)
    nonfinite = pd.Series(False, index=df.index)
    for col in NUMERIC_CHECK_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")
        hard |= values.le(-999)
        nonfinite |= ~np.isfinite(values.to_numpy(dtype="float64"))
    physical = pd.Series(False, index=df.index)
    for col in PHYSICAL_FEATURES:
        physical |= df[col].le(-100)
    cyclic = pd.Series(False, index=df.index)
    for col in CYCLIC_FEATURES:
        cyclic |= df[col].lt(-1.000001) | df[col].gt(1.000001)
    return {
        "hard_sentinel_rows": int(hard.sum()),
        "physical_feature_le_minus_100_rows": int(physical.sum()),
        "target_outside_0_1_rows": int(((df[TARGET] < 0) | (df[TARGET] > 1)).sum()),
        "cyclic_outside_minus1_1_rows": int(cyclic.sum()),
        "invalid_month_rows": int(((df["month"] < 1) | (df["month"] > 12)).sum()),
        "nonfinite_numeric_rows": int(nonfinite.sum()),
    }


def run_input_checks(df: pd.DataFrame, feature_set: dict[str, object]) -> pd.DataFrame:
    features = list(feature_set.get("main_features", DEFAULT_FEATURES))
    numeric_features = [col for col in features if col != SITE_COL]
    missing_features = [col for col in features if col not in df.columns]
    forbidden_in_features = sorted(set(features).intersection(FORBIDDEN_COLUMNS))
    duplicate_count = int(df.duplicated(subset=[SITE_COL, TIME_COL]).sum())
    split_values = sorted(df[SPLIT_COL].dropna().astype(str).unique().tolist())
    event_values = sorted(df[EVENT_COL].dropna().astype(int).unique().tolist())
    invalid_counts = invalid_value_counts(df)
    missing_numeric = sum(int(df[col].isna().sum()) for col in numeric_features)
    checks = [
        ("input_parquet_exists", DATA_PATH.exists(), str(DATA_PATH)),
        ("feature_json_exists", FEATURE_JSON_PATH.exists(), str(FEATURE_JSON_PATH)),
        ("dataset_loads_successfully", len(df) > 0, f"{len(df):,} rows"),
        ("n_unique_sites_is_50", df[SITE_COL].nunique() == 50, df[SITE_COL].nunique()),
        ("split_column_exists", SPLIT_COL in df.columns, SPLIT_COL),
        ("train_val_test_exist", {"train", "val", "test"}.issubset(split_values), split_values),
        ("target_exists_and_numeric", TARGET in df.columns and pd.api.types.is_numeric_dtype(df[TARGET]), str(df[TARGET].dtype)),
        ("official_feature_columns_exist", not missing_features, missing_features if missing_features else "None"),
        ("missing_target_count", int(df[TARGET].isna().sum()) == 0, int(df[TARGET].isna().sum())),
        ("missing_numeric_feature_count", missing_numeric == 0, missing_numeric),
        ("site_id_missing_count", int(df[SITE_COL].isna().sum()) == 0, int(df[SITE_COL].isna().sum())),
        ("timestamp_parse_success", int(df[TIME_COL].isna().sum()) == 0, int(df[TIME_COL].isna().sum())),
        ("duplicate_site_timestamp_count", duplicate_count == 0, duplicate_count),
        ("in_ph_event_exists", EVENT_COL in df.columns, EVENT_COL),
        ("ph_event_label_exists", EVENT_LABEL_COL in df.columns, EVENT_LABEL_COL),
        ("in_ph_event_binary", set(event_values).issubset({0, 1}), event_values),
        ("event_columns_not_in_model_features", not forbidden_in_features, forbidden_in_features if forbidden_in_features else "None"),
        ("no_v3_invalid_values", all(value == 0 for value in invalid_counts.values()), invalid_counts),
    ]
    out = pd.DataFrame(
        [{"check_name": name, "status": "PASS" if passed else "FAIL", "value": value, "notes": value} for name, passed, value in checks]
    )
    out.to_csv(OUT_DIR / "nonevent_input_checks.csv", index=False, encoding="utf-8-sig")
    if (out["status"] == "FAIL").any():
        raise RuntimeError("Critical input checks failed:\n" + out.to_string(index=False))
    return out


def train_impact(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df[SPLIT_COL].eq("train")]
    train_events = train[EVENT_COL].astype(int)
    overall = pd.DataFrame(
        [
            {
                "train_rows_all": len(train),
                "train_event_rows_removed": int(train_events.sum()),
                "train_nonevent_rows_remaining": int((train_events == 0).sum()),
                "removed_event_rate": float(train_events.mean()),
                "remaining_train_rate": float((train_events == 0).mean()),
                "n_sites_all": int(train[SITE_COL].nunique()),
                "n_sites_remaining": int(train.loc[train_events.eq(0), SITE_COL].nunique()),
            }
        ]
    )
    by_site = train.groupby(SITE_COL, observed=False).agg(
        train_rows_all=(TARGET, "size"),
        train_event_rows=(EVENT_COL, "sum"),
    ).reset_index()
    by_site["train_nonevent_rows"] = by_site["train_rows_all"] - by_site["train_event_rows"]
    by_site["removed_event_rate"] = by_site["train_event_rows"] / by_site["train_rows_all"]
    by_site["remaining_train_rate"] = by_site["train_nonevent_rows"] / by_site["train_rows_all"]
    by_site["has_enough_nonevent_train"] = by_site["train_nonevent_rows"] >= 10_000
    by_split = df.groupby([SITE_COL, SPLIT_COL], observed=False).agg(
        n_rows=(TARGET, "size"),
        n_event_rows=(EVENT_COL, "sum"),
    ).reset_index()
    by_split["event_ratio"] = by_split["n_event_rows"] / by_split["n_rows"]
    overall.to_csv(OUT_DIR / "nonevent_train_impact_overall.csv", index=False, encoding="utf-8-sig")
    by_site.to_csv(OUT_DIR / "nonevent_train_impact_by_site.csv", index=False, encoding="utf-8-sig")
    by_split.to_csv(OUT_DIR / "nonevent_train_impact_by_site_split.csv", index=False, encoding="utf-8-sig")
    return overall, by_site, by_split


def metric_dict(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(y) & np.isfinite(pred)
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
    y = y[mask]
    pred = pred[mask]
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


def batch_predict(model: Pipeline, x: pd.DataFrame, batch_size: int = 500_000) -> np.ndarray:
    pred = np.empty(len(x), dtype="float64")
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        pred[start:end] = model.predict(x.iloc[start:end])
    return pred


def train_ridge(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    numeric_features = [col for col in features if col != SITE_COL]
    train_mask = df[SPLIT_COL].eq("train") & df[EVENT_COL].eq(0)
    val_mask = df[SPLIT_COL].eq("val")
    test_mask = df[SPLIT_COL].eq("test")
    rows = []
    best_rmse = np.inf
    best_alpha = 1.0
    best_model: Pipeline | None = None
    for alpha in [0.1, 1.0, 10.0]:
        log(f"Training non-event Ridge alpha={alpha}...")
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("site", OneHotEncoder(handle_unknown="ignore", sparse_output=True), [SITE_COL]),
            ],
            sparse_threshold=1.0,
        )
        model = Pipeline([("preprocess", preprocessor), ("model", Ridge(alpha=alpha, solver="lsqr"))])
        model.fit(df.loc[train_mask, features], df.loc[train_mask, TARGET])
        val_pred = batch_predict(model, df.loc[val_mask, features])
        metrics = metric_dict(df.loc[val_mask, TARGET], val_pred)
        rows.append(
            {
                "alpha": alpha,
                "n_eval_rows": metrics["n_eval_rows"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "bias": metrics["bias"],
                "residual_std": metrics["residual_std"],
            }
        )
        if float(metrics["rmse"]) < best_rmse:
            best_rmse = float(metrics["rmse"])
            best_alpha = alpha
            best_model = model
    assert best_model is not None
    alpha_df = pd.DataFrame(rows)
    alpha_df["selected"] = alpha_df["alpha"].eq(best_alpha)
    alpha_df.to_csv(OUT_DIR / "nonevent_ridge_alpha_validation.csv", index=False, encoding="utf-8-sig")
    log(f"Selected non-event Ridge alpha={best_alpha} with validation RMSE={best_rmse:.6f}")
    return (
        batch_predict(best_model, df.loc[val_mask, features]),
        batch_predict(best_model, df.loc[test_mask, features]),
        alpha_df,
        best_alpha,
    )


def build_hgb_sample(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    train_mask = df[SPLIT_COL].eq("train") & df[EVENT_COL].eq(0)
    train_sites = df.loc[train_mask, SITE_COL]
    rng = np.random.default_rng(RANDOM_SEED)
    selected_parts: list[np.ndarray] = []
    rows = []
    for site_id, index_values in train_sites.groupby(train_sites, sort=True).groups.items():
        index_array = np.asarray(index_values, dtype=np.int64)
        available = int(index_array.size)
        if available > HGB_SAMPLE_PER_SITE:
            selected = rng.choice(index_array, size=HGB_SAMPLE_PER_SITE, replace=False)
            selected.sort()
        else:
            selected = index_array
        selected_parts.append(selected)
        sampled = int(selected.size)
        rows.append(
            {
                "site_id": site_id,
                "available_nonevent_train_rows": available,
                "sampled_train_rows": sampled,
                "sampling_fraction": sampled / available if available else np.nan,
            }
        )
    sample_index = np.concatenate(selected_parts) if selected_parts else np.array([], dtype=np.int64)
    sample_index.sort()
    summary = pd.DataFrame(rows).sort_values(SITE_COL)
    summary.to_csv(OUT_DIR / "nonevent_hgb_training_sample_summary.csv", index=False, encoding="utf-8-sig")
    return sample_index, summary


def train_hgb(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    numeric_features = [col for col in features if col != SITE_COL]
    val_mask = df[SPLIT_COL].eq("val")
    test_mask = df[SPLIT_COL].eq("test")
    train_index, sample_summary = build_hgb_sample(df)
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
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    log(f"Training non-event HGB with balanced sample: {len(train_index):,} rows...")
    model.fit(df.loc[train_index, features], df.loc[train_index, TARGET])
    return (
        batch_predict(model, df.loc[val_mask, features]),
        batch_predict(model, df.loc[test_mask, features]),
        sample_summary,
    )


def make_eval_frame(df: pd.DataFrame, split: str, ridge_pred: np.ndarray, hgb_pred: np.ndarray) -> pd.DataFrame:
    split_df = df[df[SPLIT_COL].eq(split)][
        [SITE_COL, TIME_COL, SPLIT_COL, TARGET, EVENT_COL, EVENT_LABEL_COL, "month", "hour", "day_of_year"]
    ].copy()
    split_df = split_df.rename(columns={TARGET: "y_true"})
    frames = []
    for model, pred in [
        ("M3_ridge_nonevent_train", ridge_pred),
        ("M4_hgb_nonevent_train", hgb_pred),
    ]:
        chunk = split_df.copy()
        chunk["model"] = model
        chunk["training_mode"] = "non_event_train"
        chunk["y_pred"] = pred
        chunk["residual"] = chunk["y_true"] - chunk["y_pred"]
        chunk["abs_residual"] = chunk["residual"].abs()
        frames.append(
            chunk[
                [
                    SITE_COL,
                    TIME_COL,
                    SPLIT_COL,
                    "model",
                    "training_mode",
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
        )
    return pd.concat(frames, ignore_index=True)


def write_prediction_file(frame: pd.DataFrame, filename: str) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), OUT_DIR / filename, compression="snappy")


def compute_metrics(eval_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    by_site_rows = []
    for split, frame in eval_frames.items():
        for model, group in frame.groupby("model", observed=False):
            overall_rows.append({"model": model, "training_mode": "non_event_train", "split": split, **metric_dict(group["y_true"], group["y_pred"])})
            for site_id, site_group in group.groupby(SITE_COL, observed=False):
                metrics = metric_dict(site_group["y_true"], site_group["y_pred"])
                n_event = int(site_group[EVENT_COL].sum())
                metrics.update(
                    {
                        "n_event_rows": n_event,
                        "event_ratio": n_event / metrics["n_eval_rows"] if metrics["n_eval_rows"] else np.nan,
                    }
                )
                by_site_rows.append(
                    {
                        "site_id": site_id,
                        "model": model,
                        "training_mode": "non_event_train",
                        "split": split,
                        **metrics,
                    }
                )
    overall = pd.DataFrame(overall_rows)
    by_site = pd.DataFrame(by_site_rows)
    macro = by_site.groupby(["model", "training_mode", "split"], observed=False).agg(
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
    overall.to_csv(OUT_DIR / "nonevent_model_metrics_overall.csv", index=False, encoding="utf-8-sig")
    by_site.to_csv(OUT_DIR / "nonevent_model_metrics_by_site.csv", index=False, encoding="utf-8-sig")
    macro.to_csv(OUT_DIR / "nonevent_model_metrics_macro_by_site.csv", index=False, encoding="utf-8-sig")
    return overall, by_site, macro


def row_level_enrichment(test_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    label_rows = []
    for model, group in test_pred.groupby("model", observed=False):
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
            f1 = 2 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0.0
            rows.append(
                {
                    "model": model,
                    "training_mode": "non_event_train",
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
            flagged_labels = group.loc[flagged, EVENT_LABEL_COL].value_counts()
            for label in ["low", "high"]:
                flagged_count = int(flagged_labels.get(label, 0))
                flagged_rate = flagged_count / n_flagged if n_flagged else np.nan
                base_rate = float(label_background.get(label, 0.0))
                label_rows.append(
                    {
                        "model": model,
                        "training_mode": "non_event_train",
                        "top_pct": pct,
                        "event_label": label,
                        "background_label_rate": base_rate,
                        "flagged_label_rate": flagged_rate,
                        "label_enrichment": flagged_rate / base_rate if base_rate > 0 else np.nan,
                        "n_flagged_label_rows": flagged_count,
                    }
                )
    enrich = pd.DataFrame(rows)
    labels = pd.DataFrame(label_rows)
    enrich.to_csv(OUT_DIR / "nonevent_row_level_enrichment.csv", index=False, encoding="utf-8-sig")
    labels.to_csv(OUT_DIR / "nonevent_low_high_label_enrichment.csv", index=False, encoding="utf-8-sig")
    return enrich, labels


def load_events() -> pd.DataFrame:
    events = pd.read_csv(EVENT_WINDOWS_PATH)
    events[SITE_COL] = events[SITE_COL].map(normalise_site_id).astype("string")
    events["incident_start"] = pd.to_datetime(events["incident_start"], errors="coerce")
    events["incident_end"] = pd.to_datetime(events["incident_end"], errors="coerce")
    events["duration_hrs"] = pd.to_numeric(events["duration_hrs"], errors="coerce")
    events[EVENT_LABEL_COL] = events[EVENT_LABEL_COL].fillna("unknown").astype("string")
    events = events[
        events["label"].isin(["pH_exceedance_low", "pH_exceedance_high"])
        | events[EVENT_LABEL_COL].isin(["low", "high"])
    ].copy()
    events = events.dropna(subset=[SITE_COL, "incident_start", "incident_end"])
    events = events[events["incident_end"] >= events["incident_start"]].copy()
    events = events.sort_values([SITE_COL, "incident_start", "incident_end"]).reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1)
    events["event_type"] = events[EVENT_LABEL_COL].astype(str)
    events["duration_category"] = np.where(events["duration_hrs"] < 24, "short_under_24h", "long_24h_plus")
    return events


def make_wide_test(test_pred: pd.DataFrame) -> pd.DataFrame:
    base_model = MODEL_ORDER[0]
    base = test_pred[test_pred["model"].eq(base_model)].sort_values([SITE_COL, TIME_COL]).reset_index(drop=True)
    base = base[[SITE_COL, TIME_COL, SPLIT_COL, "y_true", EVENT_COL, EVENT_LABEL_COL, "month", "hour", "day_of_year"]].copy()
    for model in MODEL_ORDER:
        short = MODEL_SHORT[model]
        group = test_pred[test_pred["model"].eq(model)].sort_values([SITE_COL, TIME_COL]).reset_index(drop=True)
        if len(group) != len(base):
            raise ValueError(f"Prediction rows do not align for {model}")
        base[f"abs_{short}"] = group["abs_residual"].to_numpy(dtype="float64")
    return base


def event_level_validation(test_pred: pd.DataFrame, enrichment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = load_events()
    wide = make_wide_test(test_pred)
    thresholds = {
        (row["model"], float(row["top_pct"])): float(row["threshold_abs_residual"])
        for row in enrichment.to_dict("records")
    }
    site_groups = {
        str(site_id): site_df.reset_index(drop=True)
        for site_id, site_df in wide.groupby(SITE_COL, observed=False)
    }
    rows = []
    for event in events.to_dict("records"):
        site_df = site_groups.get(str(event[SITE_COL]))
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
        for model in MODEL_ORDER:
            short = MODEL_SHORT[model]
            abs_values = window[f"abs_{short}"].to_numpy(dtype="float64")
            row = {
                "event_id": event["event_id"],
                "site_id": event[SITE_COL],
                "label": event["label"],
                "event_type": event["event_type"],
                "incident_start": event["incident_start"],
                "incident_end": event["incident_end"],
                "duration_hrs": event["duration_hrs"],
                "duration_category": event["duration_category"],
                "model": model,
                "training_mode": "non_event_train",
                "n_prediction_rows_inside_event": len(window),
                "max_abs_residual": float(np.max(abs_values)),
                "mean_abs_residual": float(np.mean(abs_values)),
            }
            for pct in TOP_PCTS:
                hit = bool((abs_values >= thresholds[(model, pct)]).any())
                row[f"hit_top_{int(pct * 100)}pct_abs"] = hit
            rows.append(row)
    event_rows = pd.DataFrame(rows)
    event_rows.to_csv(OUT_DIR / "nonevent_event_level_residual_summary.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    groups = {
        "all": lambda frame: pd.Series(True, index=frame.index),
        "low": lambda frame: frame["event_type"].eq("low"),
        "high": lambda frame: frame["event_type"].eq("high"),
        "short_under_24h": lambda frame: frame["duration_category"].eq("short_under_24h"),
        "long_24h_plus": lambda frame: frame["duration_category"].eq("long_24h_plus"),
    }
    for model in MODEL_ORDER:
        model_rows = event_rows[event_rows["model"].eq(model)]
        for pct in TOP_PCTS:
            hit_col = f"hit_top_{int(pct * 100)}pct_abs"
            for group_name, mask_fn in groups.items():
                group = model_rows[mask_fn(model_rows)]
                n = int(group["event_id"].nunique())
                n_hit = int(group[hit_col].sum()) if n else 0
                summary_rows.append(
                    {
                        "model": model,
                        "training_mode": "non_event_train",
                        "top_pct": pct,
                        "event_group": group_name,
                        "n_events": n,
                        "n_events_with_prediction_rows": n,
                        "n_hit_events": n_hit,
                        "event_hit_rate": n_hit / n if n else np.nan,
                        "median_max_abs_residual": float(group["max_abs_residual"].median()) if n else np.nan,
                        "mean_max_abs_residual": float(group["max_abs_residual"].mean()) if n else np.nan,
                    }
                )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "nonevent_event_level_hit_rate_summary.csv", index=False, encoding="utf-8-sig")
    return event_rows, summary


def model_base_name(model: str) -> str:
    return BASE_MODEL_MAP.get(model, model)


def compare_prediction_metrics(nonevent_metrics: pd.DataFrame) -> pd.DataFrame:
    alltrain = pd.read_csv(BASELINE_DIR / "full50_v3_model_metrics_overall.csv")
    alltrain = alltrain[alltrain["model"].isin(["M3_ridge", "M4_hist_gradient_boosting"])].copy()
    alltrain["model_base"] = alltrain["model"].map(model_base_name)
    nonevent = nonevent_metrics.copy()
    nonevent["model_base"] = nonevent["model"].map(model_base_name)
    merged = alltrain[["model_base", "split", "rmse", "mae", "r2"]].merge(
        nonevent[["model_base", "split", "rmse", "mae", "r2"]],
        on=["model_base", "split"],
        suffixes=("_alltrain", "_nonevent"),
    )
    out = pd.DataFrame(
        {
            "model_base": merged["model_base"],
            "split": merged["split"],
            "alltrain_rmse": merged["rmse_alltrain"],
            "nonevent_rmse": merged["rmse_nonevent"],
            "rmse_change": merged["rmse_nonevent"] - merged["rmse_alltrain"],
            "alltrain_mae": merged["mae_alltrain"],
            "nonevent_mae": merged["mae_nonevent"],
            "mae_change": merged["mae_nonevent"] - merged["mae_alltrain"],
            "alltrain_r2": merged["r2_alltrain"],
            "nonevent_r2": merged["r2_nonevent"],
            "r2_change": merged["r2_nonevent"] - merged["r2_alltrain"],
        }
    )
    out.to_csv(OUT_DIR / "nonevent_vs_alltrain_prediction_comparison.csv", index=False, encoding="utf-8-sig")
    return out


def compare_enrichment(nonevent_enrichment: pd.DataFrame) -> pd.DataFrame:
    alltrain_path = VALIDATION_DIR / "row_level_top_residual_enrichment.csv"
    alltrain = pd.read_csv(alltrain_path)
    alltrain = alltrain[alltrain["model"].isin(["M3_ridge", "M4_hist_gradient_boosting"])].copy()
    alltrain["model_base"] = alltrain["model"].map(model_base_name)
    nonevent = nonevent_enrichment.copy()
    nonevent["model_base"] = nonevent["model"].map(model_base_name)
    merged = alltrain[
        ["model_base", "top_pct", "enrichment", "event_recall", "flagged_event_rate"]
    ].merge(
        nonevent[["model_base", "top_pct", "enrichment", "event_recall", "flagged_event_rate"]],
        on=["model_base", "top_pct"],
        suffixes=("_alltrain", "_nonevent"),
    )
    out = pd.DataFrame(
        {
            "model_base": merged["model_base"],
            "top_pct": merged["top_pct"],
            "alltrain_enrichment": merged["enrichment_alltrain"],
            "nonevent_enrichment": merged["enrichment_nonevent"],
            "enrichment_change": merged["enrichment_nonevent"] - merged["enrichment_alltrain"],
            "alltrain_event_recall": merged["event_recall_alltrain"],
            "nonevent_event_recall": merged["event_recall_nonevent"],
            "event_recall_change": merged["event_recall_nonevent"] - merged["event_recall_alltrain"],
            "alltrain_flagged_event_rate": merged["flagged_event_rate_alltrain"],
            "nonevent_flagged_event_rate": merged["flagged_event_rate_nonevent"],
            "flagged_event_rate_change": merged["flagged_event_rate_nonevent"] - merged["flagged_event_rate_alltrain"],
        }
    )
    out.to_csv(OUT_DIR / "nonevent_vs_alltrain_enrichment_comparison.csv", index=False, encoding="utf-8-sig")
    return out


def summarize_alltrain_event_hits() -> pd.DataFrame:
    hit_path = VALIDATION_DIR / "event_level_residual_hits.csv"
    hits = pd.read_csv(hit_path)
    hits = hits[hits["model"].isin(["M3_ridge", "M4_hist_gradient_boosting"])].copy()
    hits["model_base"] = hits["model"].map(model_base_name)
    hits["event_type"] = hits["ph_event_label"]
    hits["duration_category"] = hits["duration_binary"]
    groups = {
        "all": lambda frame: pd.Series(True, index=frame.index),
        "low": lambda frame: frame["event_type"].eq("low"),
        "high": lambda frame: frame["event_type"].eq("high"),
        "short_under_24h": lambda frame: frame["duration_category"].eq("short_under_24h"),
        "long_24h_plus": lambda frame: frame["duration_category"].eq("long_24h_plus"),
    }
    rows = []
    for model_base, model_rows in hits.groupby("model_base", observed=False):
        for pct in TOP_PCTS:
            hit_col = f"hit_top_{int(pct * 100)}pct"
            for group_name, mask_fn in groups.items():
                group = model_rows[mask_fn(model_rows)]
                n = int(group["event_id"].nunique())
                n_hit = int(group[hit_col].sum()) if n else 0
                rows.append(
                    {
                        "model_base": model_base,
                        "top_pct": pct,
                        "event_group": group_name,
                        "alltrain_event_hit_rate": n_hit / n if n else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def compare_event_hit(nonevent_event_summary: pd.DataFrame) -> pd.DataFrame:
    alltrain = summarize_alltrain_event_hits()
    nonevent = nonevent_event_summary.copy()
    nonevent["model_base"] = nonevent["model"].map(model_base_name)
    merged = alltrain.merge(
        nonevent[["model_base", "top_pct", "event_group", "event_hit_rate"]],
        on=["model_base", "top_pct", "event_group"],
        how="inner",
    ).rename(columns={"event_hit_rate": "nonevent_event_hit_rate"})
    merged["event_hit_rate_change"] = merged["nonevent_event_hit_rate"] - merged["alltrain_event_hit_rate"]
    merged.to_csv(OUT_DIR / "nonevent_vs_alltrain_event_hit_comparison.csv", index=False, encoding="utf-8-sig")
    return merged


def save_figures(
    impact_by_site: pd.DataFrame,
    test_pred: pd.DataFrame,
    prediction_comparison: pd.DataFrame,
    enrichment_comparison: pd.DataFrame,
    event_comparison: pd.DataFrame,
    label_enrichment: pd.DataFrame,
) -> None:
    sorted_impact = impact_by_site.sort_values("remaining_train_rate")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(sorted_impact[SITE_COL], sorted_impact["removed_event_rate"], color="#E45756")
    ax.set_title("Percentage of train rows removed by site")
    ax.set_ylabel("Removed event-row fraction")
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_nonevent_train_impact_by_site.png", dpi=200)
    fig.savefig(FIG_DIR / "fig_nonevent_train_remaining_ratio_by_site.png", dpi=200)
    plt.close(fig)

    test_comp = prediction_comparison[prediction_comparison["split"].eq("test")]
    x = np.arange(len(test_comp))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    ax.bar(x - width / 2, test_comp["alltrain_rmse"], width=width, label="all-train")
    ax.bar(x + width / 2, test_comp["nonevent_rmse"], width=width, label="non-event train")
    ax.set_xticks(x)
    ax.set_xticklabels(test_comp["model_base"])
    ax.set_ylabel("Test RMSE")
    ax.set_title("All-train vs non-event-train test RMSE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_nonevent_vs_alltrain_rmse.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_base in ["ridge", "hgb"]:
        group = enrichment_comparison[enrichment_comparison["model_base"].eq(model_base)]
        ax.plot(group["top_pct"] * 100, group["alltrain_enrichment"], marker="o", linestyle="--", label=f"{model_base} all-train")
        ax.plot(group["top_pct"] * 100, group["nonevent_enrichment"], marker="o", label=f"{model_base} non-event")
    ax.axhline(1, color="#111111", linewidth=1, linestyle="--")
    ax.set_xlabel("Top residual rows (%)")
    ax.set_ylabel("Event enrichment")
    ax.set_title("All-train vs non-event residual-event enrichment")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_nonevent_vs_alltrain_enrichment.png", dpi=200)
    plt.close(fig)

    event_all = event_comparison[event_comparison["event_group"].eq("all") & event_comparison["top_pct"].eq(0.05)]
    x = np.arange(len(event_all))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, event_all["alltrain_event_hit_rate"], width=width, label="all-train")
    ax.bar(x + width / 2, event_all["nonevent_event_hit_rate"], width=width, label="non-event train")
    ax.set_xticks(x)
    ax.set_xticklabels(event_all["model_base"])
    ax.set_ylabel("Top 5% event-level hit rate")
    ax.set_title("All-train vs non-event event hit rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_nonevent_vs_alltrain_event_hit_rate.png", dpi=200)
    plt.close(fig)

    hgb = test_pred[test_pred["model"].eq("M4_hgb_nonevent_train")].copy()
    hgb["event_status"] = np.where(hgb[EVENT_COL].eq(1), "event", "nonevent")
    sample = hgb.sample(min(300_000, len(hgb)), random_state=RANDOM_SEED)
    data = [
        sample.loc[sample["event_status"].eq("nonevent"), "abs_residual"].to_numpy(),
        sample.loc[sample["event_status"].eq("event"), "abs_residual"].to_numpy(),
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(data, tick_labels=["non-event", "event"], showfliers=False)
    ax.set_title("Non-event-trained HGB residual distribution")
    ax.set_ylabel("Absolute residual")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_nonevent_hgb_event_vs_nonevent_residual.png", dpi=200)
    plt.close(fig)

    top5_labels = label_enrichment[label_enrichment["top_pct"].eq(0.05)].copy()
    pivot = top5_labels.pivot_table(index="model", columns="event_label", values="label_enrichment")
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(1, color="#111111", linewidth=1, linestyle="--")
    ax.set_ylabel("Top 5% label enrichment")
    ax.set_title("Low/high enrichment for non-event-trained models")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_nonevent_low_high_enrichment.png", dpi=200)
    plt.close(fig)


def decide(prediction_comparison: pd.DataFrame, enrichment_comparison: pd.DataFrame, event_comparison: pd.DataFrame) -> tuple[str, list[str]]:
    hgb_pred = prediction_comparison[
        prediction_comparison["model_base"].eq("hgb") & prediction_comparison["split"].eq("test")
    ].iloc[0]
    hgb_enrich = enrichment_comparison[
        enrichment_comparison["model_base"].eq("hgb") & enrichment_comparison["top_pct"].eq(0.05)
    ].iloc[0]
    hgb_event = event_comparison[
        event_comparison["model_base"].eq("hgb")
        & event_comparison["top_pct"].eq(0.05)
        & event_comparison["event_group"].eq("all")
    ].iloc[0]
    reasons = [
        f"HGB all-train top 5% enrichment: {hgb_enrich['alltrain_enrichment']:.3f}.",
        f"HGB non-event top 5% enrichment: {hgb_enrich['nonevent_enrichment']:.3f}.",
        f"HGB enrichment change: {hgb_enrich['enrichment_change']:.3f}.",
        f"HGB all-train event top 5% hit rate: {hgb_event['alltrain_event_hit_rate']:.3f}.",
        f"HGB non-event event top 5% hit rate: {hgb_event['nonevent_event_hit_rate']:.3f}.",
        f"HGB test RMSE change: {hgb_pred['rmse_change']:.6f}.",
    ]
    if (
        hgb_enrich["nonevent_enrichment"] >= 1.7
        and hgb_enrich["enrichment_change"] > 0.15
        and hgb_event["event_hit_rate_change"] > 0.02
        and hgb_pred["nonevent_rmse"] < 0.03
    ):
        return "NONEVENT_TRAINING_IMPROVES_ALIGNMENT_USE_AS_SENSITIVITY", reasons
    if hgb_enrich["enrichment_change"] < -0.05 and hgb_event["event_hit_rate_change"] < -0.02:
        return "NONEVENT_TRAINING_WORSE_DO_NOT_USE", reasons
    return "NONEVENT_TRAINING_NO_CLEAR_IMPROVEMENT_KEEP_ALLTRAIN_MAIN", reasons


def write_report(
    df: pd.DataFrame,
    features: list[str],
    checks: pd.DataFrame,
    impact_overall: pd.DataFrame,
    impact_by_site: pd.DataFrame,
    hgb_sample: pd.DataFrame,
    metrics: pd.DataFrame,
    macro: pd.DataFrame,
    enrichment: pd.DataFrame,
    label_enrichment: pd.DataFrame,
    event_summary: pd.DataFrame,
    prediction_comparison: pd.DataFrame,
    enrichment_comparison: pd.DataFrame,
    event_comparison: pd.DataFrame,
    decision: str,
    reasons: list[str],
    elapsed: float,
) -> None:
    hgb_enrich = enrichment_comparison[
        enrichment_comparison["model_base"].eq("hgb") & enrichment_comparison["top_pct"].eq(0.05)
    ].iloc[0]
    ridge_enrich = enrichment_comparison[
        enrichment_comparison["model_base"].eq("ridge") & enrichment_comparison["top_pct"].eq(0.05)
    ].iloc[0]
    reason_text = "\n".join(f"- {reason}" for reason in reasons)
    report = f"""# Non-Event Training Sensitivity Report

## 1. Purpose

This is a sensitivity experiment testing whether excluding known pH event-window rows from training improves residual-event alignment.

The all-train full50 model remains the primary baseline unless non-event training clearly improves residual-event alignment without damaging prediction performance.

## 2. Dataset And Filtering

- Input dataset: `{DATA_PATH}`
- Output folder: `{OUT_DIR}`
- Total rows: {len(df):,}
- Sites: {df[SITE_COL].nunique()}
- Features: {features}

Input checks:

{dataframe_to_markdown(checks)}

Training impact:

{dataframe_to_markdown(impact_overall)}

Sites with lowest remaining train ratio:

{dataframe_to_markdown(impact_by_site.sort_values("remaining_train_rate").head(15))}

Event rows were excluded only from the training subset. Validation and test remained complete. pH values were not modified, and no source files were changed.

## 3. Models

- `M3_ridge_nonevent_train`
- `M4_hgb_nonevent_train`

Site Mean, Seasonal Mean and Persistence-1h were not rerun. HGB sampled up to {HGB_SAMPLE_PER_SITE:,} non-event train rows per site.

HGB training sample summary:

{dataframe_to_markdown(hgb_sample.head(60))}

## 4. Prediction Performance

Non-event metrics:

{dataframe_to_markdown(metrics)}

Macro site metrics:

{dataframe_to_markdown(macro)}

All-train vs non-event prediction comparison:

{dataframe_to_markdown(prediction_comparison)}

## 5. Residual-Event Enrichment

Non-event row-level enrichment:

{dataframe_to_markdown(enrichment)}

All-train vs non-event enrichment comparison:

{dataframe_to_markdown(enrichment_comparison)}

- HGB all-train top 5% enrichment: {hgb_enrich['alltrain_enrichment']:.3f}
- HGB non-event-train top 5% enrichment: {hgb_enrich['nonevent_enrichment']:.3f}
- HGB enrichment change: {hgb_enrich['enrichment_change']:.3f}
- Ridge all-train top 5% enrichment: {ridge_enrich['alltrain_enrichment']:.3f}
- Ridge non-event-train top 5% enrichment: {ridge_enrich['nonevent_enrichment']:.3f}

## 6. Low vs High Events

{dataframe_to_markdown(label_enrichment)}

## 7. Event-Level Hit Rate

Non-event event-level hit summary:

{dataframe_to_markdown(event_summary)}

All-train vs non-event event hit comparison:

{dataframe_to_markdown(event_comparison)}

## 8. Interpretation

Non-event training is useful only if it improves residual-event alignment enough to justify any prediction-performance trade-off. This experiment should be treated as sensitivity evidence rather than replacing the main all-train full50 baseline.

## 9. Limitations

- `in_ph_event == 0` does not guarantee perfectly normal conditions.
- Regulatory events are weak validation windows.
- Removing event rows may disproportionately affect event-heavy sites.
- Long event windows may complicate hit-rate interpretation.
- This experiment does not test pH lag features or pH-change targets.

## 10. Decision

{decision}

{reason_text}

Runtime: {elapsed:.1f} seconds.
"""
    (OUT_DIR / "nonevent_training_sensitivity_report.md").write_text(report, encoding="utf-8")
    (OUT_DIR / "nonevent_training_decision.txt").write_text(decision + "\n\n" + reason_text + "\n", encoding="utf-8")


def main() -> None:
    start = perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    feature_set = load_feature_set()
    features = list(feature_set.get("main_features", DEFAULT_FEATURES))
    df = load_dataset(feature_set)
    checks = run_input_checks(df, feature_set)
    impact_overall, impact_by_site, impact_by_split = train_impact(df)

    val_mask = df[SPLIT_COL].eq("val")
    test_mask = df[SPLIT_COL].eq("test")

    log("Training non-event Ridge...")
    ridge_val, ridge_test, alpha_df, best_alpha = train_ridge(df, features)
    log("Training non-event HGB...")
    hgb_val, hgb_test, hgb_sample = train_hgb(df, features)

    log("Building prediction outputs...")
    val_pred = make_eval_frame(df, "val", ridge_val, hgb_val)
    test_pred = make_eval_frame(df, "test", ridge_test, hgb_test)
    write_prediction_file(test_pred, "nonevent_predictions_test.parquet")
    write_prediction_file(val_pred, "nonevent_predictions_val.parquet")

    metrics, by_site, macro = compute_metrics({"val": val_pred, "test": test_pred})
    enrichment, label_enrichment = row_level_enrichment(test_pred)
    event_rows, event_summary = event_level_validation(test_pred, enrichment)

    prediction_comparison = compare_prediction_metrics(metrics)
    enrichment_comparison = compare_enrichment(enrichment)
    event_comparison = compare_event_hit(event_summary)

    save_figures(impact_by_site, test_pred, prediction_comparison, enrichment_comparison, event_comparison, label_enrichment)
    decision, reasons = decide(prediction_comparison, enrichment_comparison, event_comparison)
    elapsed = perf_counter() - start
    write_report(
        df,
        features,
        checks,
        impact_overall,
        impact_by_site,
        hgb_sample,
        metrics,
        macro,
        enrichment,
        label_enrichment,
        event_summary,
        prediction_comparison,
        enrichment_comparison,
        event_comparison,
        decision,
        reasons,
        elapsed,
    )

    hgb_test_metric = metrics[metrics["model"].eq("M4_hgb_nonevent_train") & metrics["split"].eq("test")].iloc[0]
    ridge_test_metric = metrics[metrics["model"].eq("M3_ridge_nonevent_train") & metrics["split"].eq("test")].iloc[0]
    hgb_enrich = enrichment_comparison[
        enrichment_comparison["model_base"].eq("hgb") & enrichment_comparison["top_pct"].eq(0.05)
    ].iloc[0]
    hgb_event = event_comparison[
        event_comparison["model_base"].eq("hgb")
        & event_comparison["top_pct"].eq(0.05)
        & event_comparison["event_group"].eq("all")
    ].iloc[0]
    log(f"input dataset path: {DATA_PATH}")
    log(f"output folder: {OUT_DIR}")
    log(f"original train rows: {int(impact_overall['train_rows_all'].iloc[0]):,}")
    log(f"train event rows removed: {int(impact_overall['train_event_rows_removed'].iloc[0]):,}")
    log(f"remaining non-event train rows: {int(impact_overall['train_nonevent_rows_remaining'].iloc[0]):,}")
    log(f"number of sites retained: {int(impact_overall['n_sites_remaining'].iloc[0])}")
    log(f"Ridge non-event test RMSE: {float(ridge_test_metric['rmse']):.6f}")
    log(f"HGB non-event test RMSE: {float(hgb_test_metric['rmse']):.6f}")
    log(f"HGB all-train top 5% enrichment: {float(hgb_enrich['alltrain_enrichment']):.6f}")
    log(f"HGB non-event top 5% enrichment: {float(hgb_enrich['nonevent_enrichment']):.6f}")
    log(f"HGB enrichment change: {float(hgb_enrich['enrichment_change']):.6f}")
    log(f"HGB all-train event-level top 5% hit rate: {float(hgb_event['alltrain_event_hit_rate']):.6f}")
    log(f"HGB non-event event-level top 5% hit rate: {float(hgb_event['nonevent_event_hit_rate']):.6f}")
    log(f"final decision: {decision}")


if __name__ == "__main__":
    main()
