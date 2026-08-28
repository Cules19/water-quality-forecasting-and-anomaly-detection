"""Independently validate merged Common1000 teacher-rainfall feature files."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import lfilter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADDED_COLUMNS = [
    "00045",
    "00045_observed_hour",
    "00045_missing",
    "precip_api",
    "precip_api_missing",
    "precip_6h",
    "precip_6h_missing",
    "precip_feature_available",
    "precip_source",
    "precip_unit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def independent_teacher_features(
    rainfall: np.ndarray,
    half_life_hours: float,
    rolling_window_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    decay = float(0.5 ** (1.0 / half_life_hours))
    beta = 1.0 - decay
    if len(rainfall) == 0:
        return rainfall.copy(), rainfall.copy()
    api, _ = lfilter(
        [beta],
        [1.0, -decay],
        rainfall,
        zi=[decay * rainfall[0]],
    )
    rolling = lfilter(
        np.ones(rolling_window_hours, dtype=np.float64),
        [1.0],
        rainfall,
    )
    return api, rolling


def load_expected_archive(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=[
            "TIMESTAMP",
            "00045",
            "00045_observed_hour",
            "00045_source_missing",
            "precip_api",
            "precip_6h",
        ],
    )
    timestamp = pd.to_datetime(frame["TIMESTAMP"], errors="coerce", utc=True)
    return pd.DataFrame(
        {
            "00045": pd.to_numeric(frame["00045"], errors="coerce").to_numpy(
                dtype=np.float64
            ),
            "observed": frame["00045_observed_hour"].astype(bool).to_numpy(),
            "precip_api": pd.to_numeric(
                frame["precip_api"], errors="coerce"
            ).to_numpy(dtype=np.float64),
            "precip_6h": pd.to_numeric(
                frame["precip_6h"], errors="coerce"
            ).to_numpy(dtype=np.float64),
        },
        index=pd.DatetimeIndex(timestamp),
    )


def load_expected_legacy(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    source = pd.read_parquet(path, columns=["TIMESTAMP", "00045"])
    timestamp = pd.to_datetime(source["TIMESTAMP"], errors="coerce", utc=True)
    rainfall = pd.to_numeric(source["00045"], errors="coerce")
    rainfall = rainfall.where(np.isfinite(rainfall) & rainfall.ge(0))
    valid = timestamp.notna()
    hourly = rainfall.loc[valid].groupby(timestamp.loc[valid].dt.floor("h")).sum(
        min_count=1
    )
    hourly = hourly[~hourly.index.duplicated(keep="last")].sort_index()
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h", tz="UTC")
    reindexed = hourly.reindex(full_index)
    observed = reindexed.notna().to_numpy(dtype=bool)
    rainfall_mm = (
        reindexed.fillna(0.0).to_numpy(dtype=np.float64)
        * float(config["legacy_existing_00045_mm_per_inch"])
    )
    api, rolling = independent_teacher_features(
        rainfall_mm,
        float(config["teacher_archive_api_halflife_hours"]),
        int(config["teacher_archive_rolling_window_hours"]),
    )
    return pd.DataFrame(
        {
            "00045": rainfall_mm,
            "observed": observed,
            "precip_api": api,
            "precip_6h": rolling,
        },
        index=full_index,
    )


def equal_numeric(actual: np.ndarray, expected: np.ndarray) -> tuple[bool, float]:
    missing_equal = np.array_equal(np.isnan(actual), np.isnan(expected))
    finite = np.isfinite(expected)
    maximum_error = (
        float(np.max(np.abs(actual[finite] - expected[finite])))
        if finite.any()
        else 0.0
    )
    return (
        bool(
            missing_equal
            and np.allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
        ),
        maximum_error,
    )


def validate_one(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    site_id = str(row["site_id"])
    strategy = str(row["rainfall_strategy"])
    original_path = Path(str(row["prepared_ph_file"]))
    output_path = Path(str(row["output_file"]))
    input_path = Path(str(row["rainfall_input_file"]))
    result: dict[str, Any] = {
        "site_id": site_id,
        "network": row.get("network"),
        "split": row.get("split"),
        "rainfall_strategy": strategy,
        "status": "failed",
        "original_schema_preserved": False,
        "original_values_preserved": False,
        "timestamps_exact": False,
        "site_ids_exact": False,
        "rainfall_exact": False,
        "observation_flags_exact": False,
        "precip_api_exact": False,
        "precip_6h_exact": False,
        "missing_flags_exact": False,
        "source_labels_exact": False,
        "negative_feature_hours": 0,
        "rainfall_max_abs_error": None,
        "precip_api_max_abs_error": None,
        "precip_6h_max_abs_error": None,
        "error_type": "",
        "error_message": "",
    }
    try:
        original = pd.read_parquet(original_path)
        output = pd.read_parquet(output_path)
        expected_schema = list(original.columns) + ADDED_COLUMNS
        if list(output.columns) != expected_schema:
            raise ValueError("Merged output schema differs from expected append-only schema")
        result["original_schema_preserved"] = True
        if not original.equals(output.loc[:, original.columns]):
            raise ValueError("Original prepared pH values or dtypes changed")
        result["original_values_preserved"] = True
        original_time = pd.to_datetime(original["timestamp"], errors="coerce")
        output_time = pd.to_datetime(output["timestamp"], errors="coerce")
        if not original_time.equals(output_time):
            raise ValueError("Merged timestamps differ from original pH timestamps")
        result["timestamps_exact"] = True
        if output["site_id"].astype(str).ne(site_id).any():
            raise ValueError("Merged site identifiers differ")
        result["site_ids_exact"] = True
        utc_keys = pd.DatetimeIndex(original_time).tz_localize("UTC")
        expected = (
            load_expected_archive(input_path)
            if strategy == "canonical_archive"
            else load_expected_legacy(input_path, config)
        ).reindex(utc_keys)
        expected_rain = expected["00045"].to_numpy(dtype=np.float64)
        expected_observed = expected["observed"].fillna(False).to_numpy(dtype=bool)
        expected_api = expected["precip_api"].to_numpy(dtype=np.float64)
        expected_rolling = expected["precip_6h"].to_numpy(dtype=np.float64)

        actual_rain = pd.to_numeric(
            output["00045"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        actual_api = pd.to_numeric(
            output["precip_api"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        actual_rolling = pd.to_numeric(
            output["precip_6h"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        result["rainfall_exact"], result["rainfall_max_abs_error"] = equal_numeric(
            actual_rain, expected_rain
        )
        result["precip_api_exact"], result["precip_api_max_abs_error"] = equal_numeric(
            actual_api, expected_api
        )
        result["precip_6h_exact"], result["precip_6h_max_abs_error"] = equal_numeric(
            actual_rolling, expected_rolling
        )
        actual_observed = output["00045_observed_hour"].astype(bool).to_numpy()
        result["observation_flags_exact"] = bool(
            np.array_equal(actual_observed, expected_observed)
        )
        expected_feature_available = np.isfinite(expected_api) & np.isfinite(
            expected_rolling
        )
        flags_exact = bool(
            np.array_equal(
                output["00045_missing"].to_numpy(dtype=np.int8),
                (~expected_observed).astype(np.int8),
            )
            and np.array_equal(
                output["precip_api_missing"].to_numpy(dtype=np.int8),
                (~np.isfinite(expected_api)).astype(np.int8),
            )
            and np.array_equal(
                output["precip_6h_missing"].to_numpy(dtype=np.int8),
                (~np.isfinite(expected_rolling)).astype(np.int8),
            )
            and np.array_equal(
                output["precip_feature_available"].astype(bool).to_numpy(),
                expected_feature_available,
            )
        )
        result["missing_flags_exact"] = flags_exact
        expected_source = str(row["rainfall_source"])
        result["source_labels_exact"] = bool(
            output["precip_source"].astype(str).eq(expected_source).all()
            and output["precip_unit"].astype(str).eq(config["rainfall_unit"]).all()
        )
        finite_values = np.concatenate(
            [
                actual_rain[np.isfinite(actual_rain)],
                actual_api[np.isfinite(actual_api)],
                actual_rolling[np.isfinite(actual_rolling)],
            ]
        )
        result["negative_feature_hours"] = int(np.less(finite_values, 0).sum())
        checks = [
            result["rainfall_exact"],
            result["observation_flags_exact"],
            result["precip_api_exact"],
            result["precip_6h_exact"],
            result["missing_flags_exact"],
            result["source_labels_exact"],
            result["negative_feature_hours"] == 0,
        ]
        if not all(checks):
            raise ValueError("One or more independent feature checks failed")
        result["status"] = "success"
    except Exception as exc:  # noqa: BLE001
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
    return result


def causality_tests(config: dict[str, Any]) -> dict[str, Any]:
    rainfall = np.zeros(240, dtype=np.float64)
    rainfall[5] = 10.0
    api, rolling = independent_teacher_features(
        rainfall,
        float(config["teacher_archive_api_halflife_hours"]),
        int(config["teacher_archive_rolling_window_hours"]),
    )
    decay = float(
        0.5 ** (1.0 / float(config["teacher_archive_api_halflife_hours"]))
    )
    expected_api = np.zeros_like(rainfall)
    expected_api[5:] = (
        (1.0 - decay) * 10.0 * np.power(decay, np.arange(len(rainfall) - 5))
    )
    expected_rolling = np.zeros_like(rainfall)
    expected_rolling[5:11] = 10.0
    return {
        "ewma_impulse_exact": bool(
            np.allclose(api, expected_api, rtol=1e-12, atol=1e-12)
        ),
        "rolling_6h_impulse_exact": bool(
            np.allclose(rolling, expected_rolling, rtol=1e-12, atol=1e-12)
        ),
        "future_rain_not_used": bool(np.all(api[:5] == 0) and np.all(rolling[:5] == 0)),
        "no_rain_monotonic_decay": bool(np.all(np.diff(api[5:]) <= 0)),
    }


def main() -> int:
    args = parse_args()
    config = load_json(resolve(args.config))
    run_root = resolve(args.run_root)
    qa = pd.read_csv(
        run_root / "reports" / "merge_qa_by_site.csv",
        dtype={"site_id": "string"},
        encoding="utf-8-sig",
    )
    qa = qa.loc[qa["status"].eq("success")].copy()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(config["workers"]))) as executor:
        futures = [
            executor.submit(validate_one, row, config)
            for row in qa.to_dict(orient="records")
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"validated {completed}/{len(futures)}", flush=True)
    validation = pd.DataFrame(results).sort_values("site_id", kind="mergesort")
    reports = run_root / "reports"
    validation.to_csv(
        reports / "independent_validation_by_site.csv",
        index=False,
        encoding="utf-8-sig",
    )
    impulse = causality_tests(config)
    pd.DataFrame([impulse]).to_csv(
        reports / "causality_impulse_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    success = validation["status"].eq("success")
    summary = {
        "validated_sites": int(len(validation)),
        "successful_sites": int(success.sum()),
        "failed_sites": int((~success).sum()),
        "all_original_schemas_preserved": bool(
            validation["original_schema_preserved"].all()
        ),
        "all_original_values_preserved": bool(
            validation["original_values_preserved"].all()
        ),
        "all_timestamps_exact": bool(validation["timestamps_exact"].all()),
        "all_site_ids_exact": bool(validation["site_ids_exact"].all()),
        "all_rainfall_exact": bool(validation["rainfall_exact"].all()),
        "all_observation_flags_exact": bool(
            validation["observation_flags_exact"].all()
        ),
        "all_precip_api_exact": bool(validation["precip_api_exact"].all()),
        "all_precip_6h_exact": bool(validation["precip_6h_exact"].all()),
        "all_missing_flags_exact": bool(validation["missing_flags_exact"].all()),
        "all_source_labels_exact": bool(validation["source_labels_exact"].all()),
        "negative_feature_hours": int(validation["negative_feature_hours"].sum()),
        "causality_impulse_tests": impulse,
    }
    (reports / "independent_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if bool(success.all()) and all(impulse.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
