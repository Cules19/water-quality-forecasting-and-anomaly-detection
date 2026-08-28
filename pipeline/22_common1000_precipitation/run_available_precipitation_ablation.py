"""Fair leave-sites-out flow/precipitation screening on the locally available subset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPARE_PATH = PROJECT_ROOT / "pipeline" / "20_common1000_leave_sites_out" / "prepare_leave_sites_out.py"
SPEC = importlib.util.spec_from_file_location("leave_site_prepare", PREPARE_PATH)
PREPARE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREPARE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=lambda item: item.item() if isinstance(item, np.generic) else str(item)), encoding="utf-8")
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_column(path: Path, candidates: list[str]) -> str | None:
    columns = pq.read_schema(path).names if path.suffix.lower() == ".parquet" else list(pd.read_csv(path, nrows=0).columns)
    lowered = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def build_driver_shard(source: Path, target_timestamps: pd.Series, output: Path, fill_hours: int) -> dict[str, Any]:
    timestamp_column = source_column(source, ["timestamp", "TIMESTAMP", "datetime", "date_time"])
    if timestamp_column is None:
        raise ValueError(f"No timestamp in {source}")
    available = set(pq.read_schema(source).names)
    columns = [timestamp_column, "00045", "00060"]
    for code in ("00045", "00060"):
        for candidate in (f"{code}_qc", f"{code}_QC", f"qc_{code}"):
            if candidate in available:
                columns.append(candidate)
                break
    frame = pd.read_parquet(source, columns=list(dict.fromkeys(columns)))
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce")
    frame = frame.dropna(subset=[timestamp_column]).sort_values(timestamp_column, kind="mergesort")
    frame["hour"] = frame[timestamp_column].dt.floor("h")
    hourly = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(target_timestamps)))
    hourly.index.name = "timestamp"
    stats: dict[str, Any] = {}
    for code in ("00045", "00060"):
        values = pd.to_numeric(frame[code], errors="coerce")
        qc_column = next((column for column in frame if str(column).lower() == f"{code}_qc"), None)
        if qc_column is not None:
            qc = pd.to_numeric(frame[qc_column], errors="coerce")
            values = values.where(qc.eq(0))
        values = values.mask(~np.isfinite(values) | values.lt(0) | values.le(-999))
        if code == "00045":
            aggregated = values.groupby(frame["hour"]).sum(min_count=1)
            filled = aggregated.reindex(hourly.index)
        else:
            aggregated = values.groupby(frame["hour"]).mean()
            filled = aggregated.reindex(hourly.index).ffill(limit=fill_hours)
        transformed = np.log1p(filled.where(filled >= 0))
        hourly[f"x_{code}"] = transformed.astype("float32")
        hourly[f"x_{code}_missing"] = transformed.isna().astype("int8")
        stats[f"{code}_nonmissing_hours"] = int(transformed.notna().sum())
        stats[f"{code}_varying"] = bool(transformed.nunique(dropna=True) > 1)
        stats[f"{code}_maximum_native"] = float(filled.max()) if filled.notna().any() else None
    hourly = hourly.reset_index()
    write_parquet(output, hourly)
    return stats


def core_feature_matrix(frame: pd.DataFrame, target_spec: dict[str, Any], preprocessing: dict[str, Any], input_index: np.ndarray) -> tuple[np.ndarray, list[str]]:
    columns = []
    names = []
    for code in target_spec["dynamic_codes"]:
        values = pd.to_numeric(frame[f"x_{code}"], errors="coerce").to_numpy(dtype=np.float64)
        stat = preprocessing["features"][code]
        values = np.where(np.isfinite(values), values, float(stat["mean"]))
        columns.extend([((values[input_index] - float(stat["mean"])) / float(stat["scale"])), frame[f"x_{code}_missing"].to_numpy(dtype=float)[input_index]])
        names.extend([f"x_{code}_standardized", f"x_{code}_missing"])
    if "x_00300_percent_family" in frame:
        columns.append(frame["x_00300_percent_family"].to_numpy(dtype=float)[input_index])
        names.append("x_00300_percent_family")
    for field in PREPARE.CALENDAR_FEATURES:
        columns.append(frame[field].to_numpy(dtype=float)[input_index])
        names.append(field)
    return np.column_stack(columns).astype(np.float32), names


def site_rows(
    site_id: str,
    target_name: str,
    target_spec: dict[str, Any],
    preprocessing: dict[str, Any],
    preparation_config: dict[str, Any],
    prepared: Path,
    drivers: Path,
    horizon: int,
    cap: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], list[str]]:
    frame = pd.read_parquet(prepared / "hourly_native_by_site" / target_name / f"{site_id}.parquet")
    index = PREPARE.valid_window_targets(
        frame,
        target_spec["target_code"],
        list(target_spec["dynamic_codes"]),
        int(preparation_config["sequence_length_hours"]),
        horizon,
        float(preparation_config["maximum_dynamic_missing_fraction_per_window"]),
    )
    if cap > 0 and len(index) > cap:
        index = np.sort(np.random.default_rng(seed).choice(index, cap, replace=False))
    input_index = index - horizon
    core, core_names = core_feature_matrix(frame, target_spec, preprocessing, input_index)
    driver = pd.read_parquet(drivers / f"{site_id}.parquet").set_index("timestamp").reindex(pd.to_datetime(frame["timestamp"])).reset_index()
    extras = {
        code: np.column_stack(
            [
                pd.to_numeric(driver[f"x_{code}"], errors="coerce").to_numpy(dtype=float)[input_index],
                driver[f"x_{code}_missing"].to_numpy(dtype=float)[input_index],
            ]
        ).astype(np.float32)
        for code in ("00045", "00060")
    }
    target = pd.to_numeric(frame[f"target_{target_spec['target_code']}_native"], errors="coerce").to_numpy(dtype=float)[index]
    keys = pd.DataFrame({"site_id": site_id, "timestamp": pd.to_datetime(frame["timestamp"]).to_numpy()[index], "observed_native": target})
    return keys, core, extras, core_names


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return PREPARE.metric_values(observed, predicted)


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = resolve(args.config)
    config = load_json(config_path)
    preparation_config = load_json(resolve(config["preparation_config"]))
    prepared = resolve(config["prepared_root"])
    output = resolve(config["output_root"])
    for directory in ("drivers_by_site", "models", "predictions", "reports"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(resolve(config["cohort_path"]), dtype={"site_id": "string"})
    cohort["site_id"] = cohort["site_id"].map(PREPARE.normalise_site_id)
    cohort = cohort.set_index("site_id")
    schemas = {}
    for site_id, row in cohort.iterrows():
        source = resolve(row["selected_file"])
        columns = set(pq.read_schema(source).names)
        schemas[site_id] = {"source": source, "both": {"00045", "00060"} <= columns}
    common_sites = {site_id for site_id, value in schemas.items() if value["both"]}

    driver_rows = []
    ph_split = pd.read_csv(prepared / "reports" / "site_split_ph.csv", dtype={"site_id": "string"})
    ph_split["site_id"] = ph_split["site_id"].map(PREPARE.normalise_site_id)
    for number, site_id in enumerate(sorted(common_sites & set(ph_split["site_id"])), start=1):
        shard = output / "drivers_by_site" / f"{site_id}.parquet"
        checkpoint = output / "drivers_by_site" / f"{site_id}.json"
        if args.resume and shard.exists() and checkpoint.exists():
            stats = load_json(checkpoint)
        else:
            reference = pd.read_parquet(prepared / "hourly_native_by_site" / "ph" / f"{site_id}.parquet", columns=["timestamp"])
            stats = build_driver_shard(schemas[site_id]["source"], reference["timestamp"], shard, int(config["flow_causal_fill_hours"]))
            stats["site_id"] = site_id
            write_json(checkpoint, stats)
        driver_rows.append(stats)
    driver_status = pd.DataFrame(driver_rows)
    usable = set(driver_status.loc[driver_status["00045_varying"] & driver_status["00060_varying"], "site_id"].astype(str))
    write_csv(output / "reports" / "driver_site_status.csv", driver_status)

    all_metrics = []
    all_predictions = []
    key_reference: dict[tuple[str, str], pd.DataFrame] = {}
    for target_name, target_spec in preparation_config["targets"].items():
        split = pd.read_csv(prepared / "reports" / f"site_split_{target_name}.csv", dtype={"site_id": "string"})
        split["site_id"] = split["site_id"].map(PREPARE.normalise_site_id)
        split = split[split["site_id"].isin(usable)].copy()
        preprocessing = load_json(prepared / "reports" / f"train_only_preprocessing_{target_name}.json")
        payloads = {}
        for split_name in ("train", "validation", "test"):
            rows = []
            cores = []
            extra_values = {"00045": [], "00060": []}
            for number, site_id in enumerate(split.loc[split["split"].eq(split_name), "site_id"].astype(str)):
                cap = int(config["train_cap_per_site"]) if split_name == "train" else 0
                keys, core, extras, core_names = site_rows(
                    site_id, target_name, target_spec, preprocessing, preparation_config, prepared,
                    output / "drivers_by_site", int(config["horizon_hours"]), cap, int(config["random_seed"]) + number,
                )
                rows.append(keys)
                cores.append(core)
                for code in extra_values:
                    extra_values[code].append(extras[code])
            payloads[split_name] = {
                "keys": pd.concat(rows, ignore_index=True),
                "core": np.concatenate(cores),
                "extras": {code: np.concatenate(values) for code, values in extra_values.items()},
                "core_names": core_names,
            }
        # Extra-driver fill/scaling is learned from training sites only.
        extra_stats = {}
        for code in ("00045", "00060"):
            values = payloads["train"]["extras"][code][:, 0].astype(float)
            valid = np.isfinite(values)
            mean = float(values[valid].mean()) if valid.any() else 0.0
            scale = float(values[valid].std()) if valid.any() else 1.0
            if scale < 1e-8:
                scale = 1.0
            extra_stats[code] = {"mean": mean, "scale": scale, "fit_nonmissing_rows": int(valid.sum())}
            for split_name in payloads:
                data = payloads[split_name]["extras"][code]
                raw = data[:, 0].astype(float)
                missing = ~np.isfinite(raw)
                data[:, 0] = np.where(missing, 0.0, (raw - mean) / scale)
                data[:, 1] = missing.astype(float)
        write_json(output / "reports" / f"train_only_driver_preprocessing_{target_name}.json", extra_stats)

        for feature_set, extra_codes in config["feature_sets"].items():
            matrices = {}
            names = list(payloads["train"]["core_names"])
            for split_name, payload in payloads.items():
                parts = [payload["core"]]
                for code in extra_codes:
                    parts.append(payload["extras"][code])
                matrices[split_name] = np.concatenate(parts, axis=1)
            for code in extra_codes:
                names.extend([f"x_{code}_standardized", f"x_{code}_missing"])
            scaler = StandardScaler().fit(matrices["train"])
            train_scaled = scaler.transform(matrices["train"])
            val_scaled = scaler.transform(matrices["validation"])
            test_scaled = scaler.transform(matrices["test"])
            y_train = payloads["train"]["keys"]["observed_native"].to_numpy(dtype=float)
            y_val = payloads["validation"]["keys"]["observed_native"].to_numpy(dtype=float)
            y_test = payloads["test"]["keys"]["observed_native"].to_numpy(dtype=float)

            best_alpha = None
            best_rmse = float("inf")
            best_ridge = None
            for alpha in config["ridge_alphas"]:
                model = Ridge(alpha=float(alpha)).fit(train_scaled, y_train)
                rmse = math.sqrt(mean_squared_error(y_val, model.predict(val_scaled)))
                if rmse < best_rmse:
                    best_alpha, best_rmse, best_ridge = float(alpha), rmse, model
            hgb = HistGradientBoostingRegressor(**config["hgb"]).fit(matrices["train"], y_train)
            for model_name, model, data_by_split in (
                ("ridge", best_ridge, {"validation": val_scaled, "test": test_scaled}),
                ("hgb", hgb, {"validation": matrices["validation"], "test": matrices["test"]}),
            ):
                joblib.dump(model, output / "models" / f"{target_name}_{feature_set}_{model_name}.joblib")
                for split_name in ("validation", "test"):
                    predicted = model.predict(data_by_split[split_name])
                    keys = payloads[split_name]["keys"].copy()
                    keys["target"] = target_name
                    keys["split"] = split_name
                    keys["feature_set"] = feature_set
                    keys["model"] = model_name
                    keys["predicted_native"] = predicted.astype("float32")
                    metric = metrics(keys["observed_native"].to_numpy(), predicted)
                    all_metrics.append(
                        {
                            "target": target_name,
                            "split": split_name,
                            "feature_set": feature_set,
                            "model": model_name,
                            "ridge_alpha": best_alpha if model_name == "ridge" else np.nan,
                            "sites": int(keys["site_id"].nunique()),
                            **metric,
                        }
                    )
                    all_predictions.append(keys)
                    reference_key = (target_name, split_name)
                    current = keys[["site_id", "timestamp", "observed_native"]].reset_index(drop=True)
                    if reference_key not in key_reference:
                        key_reference[reference_key] = current
                    elif not current.equals(key_reference[reference_key]):
                        raise RuntimeError(f"Prediction keys differ for {target_name}/{split_name}/{feature_set}/{model_name}")
            write_json(output / "reports" / f"feature_manifest_{target_name}_{feature_set}.json", {"features": names, "fit_split": "train_sites_only"})

    metric_frame = pd.DataFrame(all_metrics)
    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    write_csv(output / "reports" / "available_precipitation_ablation_metrics.csv", metric_frame)
    write_parquet(output / "predictions" / "available_precipitation_ablation_predictions.parquet", prediction_frame)
    test = metric_frame[metric_frame["split"].eq("test")].sort_values(["target", "model", "rmse"])
    report = "\n".join(
        [
            "# Available precipitation/flow leave-sites-out ablation",
            "",
            f"Common sites with locally varying 00045 and 00060: **{len(usable)}**.",
            "Every feature set uses the same sites, target keys and horizon. External register-linked rainfall files remain unavailable; this is the observed local subset only.",
            "",
            PREPARE.frame_to_markdown(test[["target", "model", "feature_set", "sites", "n", "rmse", "mae", "nse"]]),
        ]
    ) + "\n"
    (output / "reports" / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    success = {
        "status": "pass",
        "common_sites": len(usable),
        "same_keys_all_feature_sets": True,
        "full_external_precipitation_batch_available": False,
        "elapsed_seconds": time.time() - started,
        "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(output / "_SUCCESS", success)
    print(report)


if __name__ == "__main__":
    main()
