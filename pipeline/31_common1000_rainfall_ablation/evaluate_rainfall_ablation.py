"""Extended diagnostics for the common-key teacher-Archive rainfall ablation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPARE_PATH = (
    PROJECT_ROOT
    / "pipeline"
    / "20_common1000_leave_sites_out"
    / "prepare_leave_sites_out.py"
)
SPEC = importlib.util.spec_from_file_location("leave_site_prepare", PREPARE_PATH)
PREPARE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREPARE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    base_path = config.get("base_config")
    if not base_path:
        return config
    base = load_json(resolve(base_path))
    base.update({key: value for key, value in config.items() if key != "base_config"})
    base["base_config"] = str(resolve(base_path))
    return base


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
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


def read_split_predictions(root: Path, run_id: str, split: str) -> pd.DataFrame:
    files = sorted((root / run_id / split).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No predictions: {run_id}/{split}")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    frame["site_id"] = frame["site_id"].map(PREPARE.normalise_site_id)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values(["site_id", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )


def event_intervals(path: Path) -> dict[str, pd.DataFrame]:
    events = pd.read_csv(
        path,
        usecols=["site_id", "parameter", "label", "incident_start", "incident_end"],
        dtype={"site_id": "string"},
    )
    events = events[events["parameter"].astype(str).str.strip().eq("pH (00400)")].copy()
    events["site_id"] = events["site_id"].map(PREPARE.normalise_site_id)
    events["incident_start"] = pd.to_datetime(events["incident_start"], errors="coerce")
    events["incident_end"] = pd.to_datetime(events["incident_end"], errors="coerce")
    events = events.dropna(subset=["site_id", "incident_start", "incident_end"])
    events = events[events["incident_end"].ge(events["incident_start"])].copy()
    events["direction"] = np.where(
        events["label"].astype(str).str.lower().str.contains("low", na=False),
        "low",
        np.where(
            events["label"].astype(str).str.lower().str.contains("high", na=False),
            "high",
            "unknown",
        ),
    )
    return {
        str(site_id): group.sort_values(["incident_start", "incident_end"])
        for site_id, group in events.groupby("site_id", sort=False)
    }


def reconstruct_event_direction(
    timestamps: np.ndarray,
    intervals: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(timestamps)
    low_diff = np.zeros(n + 1, dtype=np.int32)
    high_diff = np.zeros(n + 1, dtype=np.int32)
    unknown_diff = np.zeros(n + 1, dtype=np.int32)
    if intervals is not None:
        times = np.asarray(timestamps, dtype="datetime64[ns]")
        for event in intervals.itertuples(index=False):
            left = int(
                np.searchsorted(
                    times, np.datetime64(event.incident_start, "ns"), side="left"
                )
            )
            right = int(
                np.searchsorted(
                    times, np.datetime64(event.incident_end, "ns"), side="right"
                )
            )
            if right <= left:
                continue
            target = (
                low_diff
                if event.direction == "low"
                else high_diff
                if event.direction == "high"
                else unknown_diff
            )
            target[left] += 1
            target[right] -= 1
    low = np.cumsum(low_diff[:-1]) > 0
    high = np.cumsum(high_diff[:-1]) > 0
    unknown = np.cumsum(unknown_diff[:-1]) > 0
    in_event = low | high | unknown
    direction = np.full(n, "none", dtype=object)
    direction[low & ~high & ~unknown] = "low"
    direction[high & ~low & ~unknown] = "high"
    direction[(low & high) | unknown] = "mixed"
    return in_event, direction


def exact_top_mask(values: np.ndarray, fraction: float, largest: bool) -> np.ndarray:
    count = max(1, int(math.ceil(len(values) * fraction)))
    order = np.argsort(values, kind="mergesort")
    selected = order[-count:] if largest else order[:count]
    mask = np.zeros(len(values), dtype=bool)
    mask[selected] = True
    return mask


def corrected_event_metrics(
    validation: dict[str, pd.DataFrame],
    test: dict[str, pd.DataFrame],
    intervals: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    audit_rows = []
    for run_id, frame in test.items():
        directions = []
        reconstructed_events = []
        original_events = []
        for site_id, group in frame.groupby("site_id", sort=False):
            event, direction = reconstruct_event_direction(
                group["timestamp"].to_numpy(dtype="datetime64[ns]"),
                intervals.get(site_id),
            )
            reconstructed_events.append(event)
            directions.append(direction)
            original_events.append(group["in_event"].to_numpy(dtype=bool))
        reconstructed = np.concatenate(reconstructed_events)
        direction = np.concatenate(directions)
        original = np.concatenate(original_events)
        observed = frame["observed_native"].to_numpy(dtype=np.float64)
        predicted = frame["predicted_native"].to_numpy(dtype=np.float64)
        residual = predicted - observed
        absolute = np.abs(residual)
        low = direction == "low"
        high = direction == "high"
        mixed = direction == "mixed"
        absolute_top5 = exact_top_mask(absolute, 0.05, True)
        negative_top5 = exact_top_mask(residual, 0.05, False)
        positive_top5 = exact_top_mask(residual, 0.05, True)

        validation_residual = (
            validation[run_id]["predicted_native"].to_numpy(dtype=np.float64)
            - validation[run_id]["observed_native"].to_numpy(dtype=np.float64)
        )
        threshold = float(np.quantile(np.abs(validation_residual), 0.95))
        calibrated = absolute >= threshold
        base_event = float(reconstructed.mean())
        base_low = float(low.mean())
        base_high = float(high.mean())
        rows.append(
            {
                "run_id": run_id,
                "test_rows": len(frame),
                "event_rows": int(reconstructed.sum()),
                "low_event_rows": int(low.sum()),
                "high_event_rows": int(high.sum()),
                "mixed_event_rows": int(mixed.sum()),
                "base_event_rate": base_event,
                "exact_test_top5_event_enrichment": (
                    float(reconstructed[absolute_top5].mean() / base_event)
                    if base_event > 0
                    else np.nan
                ),
                "exact_test_negative5_low_enrichment": (
                    float(low[negative_top5].mean() / base_low)
                    if base_low > 0
                    else np.nan
                ),
                "exact_test_positive5_high_enrichment": (
                    float(high[positive_top5].mean() / base_high)
                    if base_high > 0
                    else np.nan
                ),
                "validation_absolute95_threshold": threshold,
                "validation_threshold_selected_test_rows": int(calibrated.sum()),
                "validation_threshold_selected_test_fraction": float(calibrated.mean()),
                "validation_threshold_test_event_enrichment": (
                    float(reconstructed[calibrated].mean() / base_event)
                    if base_event > 0 and calibrated.any()
                    else np.nan
                ),
            }
        )
        audit_rows.append(
            {
                "run_id": run_id,
                "prediction_event_rows": int(original.sum()),
                "reconstructed_event_rows": int(reconstructed.sum()),
                "event_boolean_equal_rows": int((original == reconstructed).sum()),
                "event_boolean_total_rows": len(frame),
                "event_boolean_exact_match": bool(np.array_equal(original, reconstructed)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return np.nan
    if np.nanstd(left[valid]) < 1e-12 or np.nanstd(right[valid]) < 1e-12:
        return np.nan
    return float(
        pd.Series(left[valid]).corr(pd.Series(right[valid]), method="spearman")
    )


def lagged_rain_relationship(
    test: dict[str, pd.DataFrame],
    rainfall_root: Path,
    lags: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    site_rows = []
    drivers = {
        "rain_00045": "00045",
        "precip_api": "precip_api",
        "precip_6h": "precip_6h",
    }
    for run_id, frame in test.items():
        for site_id, prediction in frame.groupby("site_id", sort=False):
            rain = pd.read_parquet(
                rainfall_root / f"{site_id}.parquet",
                columns=["timestamp", *drivers.values()],
            )
            rain["timestamp"] = pd.to_datetime(rain["timestamp"])
            rain = rain.drop_duplicates("timestamp").set_index("timestamp")
            target_time = pd.DatetimeIndex(prediction["timestamp"])
            residual = (
                prediction["predicted_native"].to_numpy(dtype=np.float64)
                - prediction["observed_native"].to_numpy(dtype=np.float64)
            )
            absolute = np.abs(residual)
            for lag in lags:
                lookup = target_time - pd.Timedelta(hours=int(lag))
                for label, column in drivers.items():
                    values = pd.to_numeric(
                        rain[column].reindex(lookup), errors="coerce"
                    ).to_numpy(dtype=np.float64)
                    valid = np.isfinite(values)
                    wet = valid & (values > 0)
                    dry = valid & (values == 0)
                    site_rows.append(
                        {
                            "run_id": run_id,
                            "site_id": site_id,
                            "lag_hours_before_target": lag,
                            "driver": label,
                            "n": int(valid.sum()),
                            "wet_rows": int(wet.sum()),
                            "dry_rows": int(dry.sum()),
                            "spearman_abs_residual": safe_spearman(absolute, values),
                            "spearman_signed_residual": safe_spearman(residual, values),
                            "wet_rmse": (
                                float(np.sqrt(np.mean(np.square(residual[wet]))))
                                if wet.any()
                                else np.nan
                            ),
                            "dry_rmse": (
                                float(np.sqrt(np.mean(np.square(residual[dry]))))
                                if dry.any()
                                else np.nan
                            ),
                        }
                    )
    site_frame = pd.DataFrame(site_rows)
    summary = (
        site_frame.groupby(
            ["run_id", "lag_hours_before_target", "driver"], as_index=False
        )
        .agg(
            sites=("site_id", "nunique"),
            sites_with_finite_abs_correlation=(
                "spearman_abs_residual",
                lambda value: int(np.isfinite(value).sum()),
            ),
            macro_median_spearman_abs_residual=(
                "spearman_abs_residual",
                "median",
            ),
            macro_mean_spearman_abs_residual=("spearman_abs_residual", "mean"),
            macro_median_spearman_signed_residual=(
                "spearman_signed_residual",
                "median",
            ),
            macro_mean_wet_minus_dry_rmse=(
                "wet_rmse",
                lambda value: np.nan,
            ),
        )
    )
    wet_dry = (
        site_frame.assign(
            wet_minus_dry_rmse=site_frame["wet_rmse"] - site_frame["dry_rmse"]
        )
        .groupby(["run_id", "lag_hours_before_target", "driver"])[
            "wet_minus_dry_rmse"
        ]
        .mean()
    )
    summary = summary.drop(columns=["macro_mean_wet_minus_dry_rmse"]).merge(
        wet_dry.rename("macro_mean_wet_minus_dry_rmse").reset_index(),
        on=["run_id", "lag_hours_before_target", "driver"],
        how="left",
        validate="one_to_one",
    )
    return summary, site_frame


def paired_site_results(
    site_metrics: pd.DataFrame,
    random_seed: int,
    baseline: str,
    comparison_models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = site_metrics[site_metrics["split"].eq("test")].copy()
    pivot = test.pivot(index="site_id", columns="run_id", values=["rmse", "nse"])
    comparison_rows = []
    delta_rows = []
    rng = np.random.default_rng(random_seed)
    for model in comparison_models:
        rmse_delta = (
            pivot[("rmse", model)] - pivot[("rmse", baseline)]
        ).dropna()
        nse_delta = (pivot[("nse", model)] - pivot[("nse", baseline)]).dropna()
        matrix = rmse_delta.to_numpy(dtype=np.float64)
        bootstrap = np.empty(10000, dtype=np.float64)
        for start in range(0, len(bootstrap), 500):
            size = min(500, len(bootstrap) - start)
            draw = rng.integers(0, len(matrix), size=(size, len(matrix)))
            bootstrap[start : start + size] = matrix[draw].mean(axis=1)
        comparison_rows.append(
            {
                "comparison": f"{model} minus {baseline}",
                "sites": len(rmse_delta),
                "rmse_improved_sites": int((rmse_delta < 0).sum()),
                "rmse_improved_fraction": float((rmse_delta < 0).mean()),
                "mean_site_rmse_delta": float(rmse_delta.mean()),
                "median_site_rmse_delta": float(rmse_delta.median()),
                "bootstrap_mean_site_rmse_delta_ci_low": float(
                    np.quantile(bootstrap, 0.025)
                ),
                "bootstrap_mean_site_rmse_delta_ci_high": float(
                    np.quantile(bootstrap, 0.975)
                ),
                "nse_improved_sites": int((nse_delta > 0).sum()),
                "nse_improved_fraction": float((nse_delta > 0).mean()),
                "mean_site_nse_delta": float(nse_delta.mean()),
                "median_site_nse_delta": float(nse_delta.median()),
            }
        )
        for site_id in rmse_delta.index:
            delta_rows.append(
                {
                    "site_id": site_id,
                    "comparison_model": model,
                    "baseline_model": baseline,
                    "rmse_delta": float(rmse_delta.loc[site_id]),
                    "nse_delta": float(nse_delta.loc[site_id]),
                }
            )
    return pd.DataFrame(comparison_rows), pd.DataFrame(delta_rows)


def linear_slope(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if len(observed) < 2:
        return np.nan
    denominator = float(np.square(observed - observed.mean()).sum())
    if denominator <= 0:
        return np.nan
    return float(
        np.sum((observed - observed.mean()) * (predicted - predicted.mean()))
        / denominator
    )


def compression_site_metrics(
    test: dict[str, pd.DataFrame],
    site_id: str,
    tail_fraction: float,
) -> pd.DataFrame:
    rows = []
    for run_id, frame in test.items():
        site = frame[frame["site_id"].eq(site_id)]
        observed = site["observed_native"].to_numpy(dtype=np.float64)
        predicted = site["predicted_native"].to_numpy(dtype=np.float64)
        residual = predicted - observed
        low_threshold = float(np.quantile(observed, tail_fraction))
        high_threshold = float(np.quantile(observed, 1 - tail_fraction))
        extreme = (observed <= low_threshold) | (observed >= high_threshold)
        observed_range = float(np.ptp(observed[extreme]))
        predicted_range = float(np.ptp(predicted[extreme]))
        rows.append(
            {
                "site_id": site_id,
                "run_id": run_id,
                "rows": len(site),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "mae": float(np.mean(np.abs(residual))),
                "observed_sd": float(np.std(observed)),
                "predicted_sd": float(np.std(predicted)),
                "predicted_to_observed_sd_ratio": float(
                    np.std(predicted) / np.std(observed)
                ),
                "prediction_vs_observation_slope": linear_slope(observed, predicted),
                "extreme_tail_fraction_each_side": tail_fraction,
                "extreme_observed_range": observed_range,
                "extreme_predicted_range": predicted_range,
                "extreme_range_ratio": (
                    predicted_range / observed_range if observed_range > 0 else np.nan
                ),
                "low_tail_mean_residual": float(
                    residual[observed <= low_threshold].mean()
                ),
                "high_tail_mean_residual": float(
                    residual[observed >= high_threshold].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_config(config_path)
    output = resolve(config["output_root"]) / args.mode
    reports = output / "reports"
    prediction_root = output / "predictions_by_site"
    run_manifest = pd.read_csv(reports / "run_manifest.csv")
    run_ids = run_manifest["run_id"].astype(str).tolist()
    role_rows = run_manifest.assign(
        has_rain=run_manifest["feature_set"].astype(str).str.contains("rain_00045"),
        has_api=run_manifest["feature_set"].astype(str).str.contains("precip_api"),
    )
    baseline_run = str(role_rows.loc[~role_rows["has_rain"], "run_id"].iloc[0])
    rain_run = str(
        role_rows.loc[role_rows["has_rain"] & ~role_rows["has_api"], "run_id"].iloc[0]
    )
    api_run = str(role_rows.loc[role_rows["has_api"], "run_id"].iloc[0])
    validation = {
        run_id: read_split_predictions(prediction_root, run_id, "validation")
        for run_id in run_ids
    }
    test = {
        run_id: read_split_predictions(prediction_root, run_id, "test")
        for run_id in run_ids
    }
    reference_keys = test[run_ids[0]][["site_id", "timestamp"]]
    key_match = all(
        frame[["site_id", "timestamp"]].equals(reference_keys)
        for frame in test.values()
    )

    preparation = load_json(resolve(config["preparation_config"]))
    intervals = event_intervals(resolve(preparation["events_path"]))
    event_metrics, event_audit = corrected_event_metrics(
        validation, test, intervals
    )
    write_csv(reports / "event_enrichment_corrected.csv", event_metrics)
    write_csv(reports / "event_direction_reconstruction_audit.csv", event_audit)

    lag_summary, lag_sites = lagged_rain_relationship(
        test,
        resolve(config["merged_input_root"]),
        [0, 1, 3, 6, 12, 24],
    )
    write_csv(reports / "lagged_rain_residual_summary.csv", lag_summary)
    write_csv(reports / "lagged_rain_residual_by_site.csv", lag_sites)

    site_metrics = pd.read_csv(
        reports / "site_metrics_all.csv", dtype={"site_id": "string"}
    )
    site_metrics["site_id"] = site_metrics["site_id"].map(PREPARE.normalise_site_id)
    paired_summary, paired_sites = paired_site_results(
        site_metrics,
        int(config["random_seed"]),
        baseline_run,
        [rain_run, api_run],
    )
    write_csv(reports / "paired_site_comparison.csv", paired_summary)
    write_csv(reports / "paired_site_deltas.csv", paired_sites)

    compression = compression_site_metrics(test, "01645762", 0.1)
    write_csv(reports / "site_01645762_compression.csv", compression)

    problems = []
    if not key_match:
        problems.append("test prediction keys differ across models")
    if not event_audit["event_boolean_exact_match"].all():
        problems.append("reconstructed event boolean differs from model evaluation boolean")
    required = {
        "event_enrichment_corrected.csv",
        "event_direction_reconstruction_audit.csv",
        "lagged_rain_residual_summary.csv",
        "lagged_rain_residual_by_site.csv",
        "paired_site_comparison.csv",
        "paired_site_deltas.csv",
        "site_01645762_compression.csv",
    }
    missing = [name for name in required if not (reports / name).exists()]
    problems.extend(f"missing output: {name}" for name in missing)
    audit = {
        "status": "pass" if not problems else "fail",
        "mode": args.mode,
        "run_ids": run_ids,
        "model_roles": {
            "baseline": baseline_run,
            "current_rain": rain_run,
            "rain_api_6h": api_run,
        },
        "test_keys_identical_across_models": key_match,
        "event_direction_reconstructed_from_raw_regulatory_intervals": True,
        "event_boolean_matches_training_data": bool(
            event_audit["event_boolean_exact_match"].all()
        ),
        "lag_hours": [0, 1, 3, 6, 12, 24],
        "paired_site_bootstrap_replicates": 10000,
        "compression_site_id": "01645762",
        "problems": problems,
        "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(reports / "extended_diagnostics_audit.json", audit)
    if problems:
        raise RuntimeError(problems)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
