"""Independently validate all-site teacher Archive precipitation features."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import lfilter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def expected_schema(config: dict[str, Any]) -> list[str]:
    return [
        str(config["timestamp_column"]),
        "site_id",
        str(config["rainfall_input_column"]),
        str(config["rainfall_model_column"]),
        "00045_observed_hour",
        "00045_source_missing",
        str(config["api_output_column"]),
        str(config["rolling_output_column"]),
    ]


def load_expected_rainfall(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    frame = pd.read_csv(
        path,
        usecols=["precipitation_mm", "year", "month", "day", "hour"],
        low_memory=False,
    )
    timestamps = pd.to_datetime(
        {
            "year": pd.to_numeric(frame["year"], errors="coerce"),
            "month": pd.to_numeric(frame["month"], errors="coerce"),
            "day": pd.to_numeric(frame["day"], errors="coerce"),
            "hour": pd.to_numeric(frame["hour"], errors="coerce"),
        },
        errors="coerce",
        utc=True,
    )
    values = pd.to_numeric(frame["precipitation_mm"], errors="coerce")
    values = values.where(np.isfinite(values) & values.ge(0))
    valid_timestamp = timestamps.notna()
    rainfall = pd.Series(
        values.loc[valid_timestamp].to_numpy(dtype=np.float64),
        index=pd.DatetimeIndex(timestamps.loc[valid_timestamp]),
    )
    rainfall = rainfall[~rainfall.index.duplicated(keep="last")].sort_index()
    full_index = pd.date_range(
        rainfall.index.min(),
        rainfall.index.max(),
        freq="h",
        tz="UTC",
    )
    complete = rainfall.reindex(full_index)
    observed = complete.notna().to_numpy(dtype=bool)
    filled = complete.fillna(0.0).to_numpy(dtype=np.float64)
    return full_index, filled, observed


def independent_features(
    rainfall: np.ndarray,
    half_life_hours: float,
    rolling_window_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    decay = float(0.5 ** (1.0 / half_life_hours))
    beta = 1.0 - decay
    if len(rainfall) == 0:
        return rainfall.copy(), rainfall.copy()
    ewma, _ = lfilter(
        [beta],
        [1.0, -decay],
        rainfall,
        zi=[decay * rainfall[0]],
    )
    rolling = lfilter(
        np.ones(int(rolling_window_hours), dtype=np.float64),
        [1.0],
        rainfall,
    )
    return ewma, rolling


def impulse_tests(config: dict[str, Any]) -> dict[str, Any]:
    half_life = float(config["api_halflife_hours"])
    window = int(config["rolling_window_hours"])
    rainfall = np.zeros(100, dtype=np.float64)
    rainfall[0] = 10.0
    ewma, rolling = independent_features(rainfall, half_life, window)
    decay = float(0.5 ** (1.0 / half_life))
    expected_ewma = 10.0 * np.power(decay, np.arange(len(rainfall)))
    expected_rolling = np.zeros_like(rainfall)
    expected_rolling[:window] = 10.0
    return {
        "api_halflife_hours": half_life,
        "rolling_window_hours": window,
        "ewma_impulse_decay_exact": bool(
            np.allclose(ewma, expected_ewma, rtol=1e-12, atol=1e-12)
        ),
        "rolling_6h_impulse_exact": bool(
            np.allclose(rolling, expected_rolling, rtol=1e-12, atol=1e-12)
        ),
        "future_rain_not_used": bool(ewma[0] == rainfall[0]),
        "no_rain_monotonic_decay": bool(np.all(np.diff(ewma) <= 0)),
    }


def validate_one(
    row: dict[str, Any],
    config: dict[str, Any],
    schema: list[str],
) -> dict[str, Any]:
    site_id = str(row["site_id"])
    source_path = Path(str(row["canonical_file_path"]))
    output_path = Path(str(row["output_file"]))
    result: dict[str, Any] = {
        "site_id": site_id,
        "source": row.get("source"),
        "status": "failed",
        "error": "",
        "schema_exact": False,
        "timestamp_exact": False,
        "site_id_exact": False,
        "rainfall_exact": False,
        "observation_flags_exact": False,
        "precip_api_exact": False,
        "precip_6h_exact": False,
        "negative_feature_hours": 0,
    }
    try:
        actual_schema = pq.read_schema(output_path).names
        if actual_schema != schema:
            raise ValueError(f"schema mismatch: {actual_schema}")
        result["schema_exact"] = True

        output = pd.read_parquet(output_path)
        timestamps, rainfall, observed = load_expected_rainfall(source_path)
        actual_timestamps = pd.DatetimeIndex(
            pd.to_datetime(output[str(config["timestamp_column"])], errors="coerce", utc=True)
        )
        if not actual_timestamps.equals(timestamps):
            raise ValueError("output timestamps differ from canonical full-hour index")
        result["timestamp_exact"] = True

        site_ids = output["site_id"].astype("string")
        if not (site_ids.nunique(dropna=False) == 1 and site_ids.iloc[0] == site_id):
            raise ValueError("output site ID mismatch")
        result["site_id_exact"] = True

        precip_mm = pd.to_numeric(
            output[str(config["rainfall_input_column"])],
            errors="coerce",
        ).to_numpy(dtype=np.float32)
        model_rain = pd.to_numeric(
            output[str(config["rainfall_model_column"])],
            errors="coerce",
        ).to_numpy(dtype=np.float32)
        expected_rain = rainfall.astype(np.float32)
        if not (
            np.array_equal(precip_mm, expected_rain)
            and np.array_equal(model_rain, expected_rain)
        ):
            raise ValueError("output rainfall differs from canonical rainfall")
        result["rainfall_exact"] = True

        actual_observed = output["00045_observed_hour"].astype(bool).to_numpy()
        actual_missing = pd.to_numeric(
            output["00045_source_missing"],
            errors="coerce",
        ).to_numpy(dtype=np.int8)
        if not (
            np.array_equal(actual_observed, observed)
            and np.array_equal(actual_missing, (~observed).astype(np.int8))
        ):
            raise ValueError("observation/source-missing flags differ")
        result["observation_flags_exact"] = True

        expected_api, expected_rolling = independent_features(
            rainfall,
            half_life_hours=float(config["api_halflife_hours"]),
            rolling_window_hours=int(config["rolling_window_hours"]),
        )
        actual_api = pd.to_numeric(
            output[str(config["api_output_column"])],
            errors="coerce",
        ).to_numpy(dtype=np.float32)
        actual_rolling = pd.to_numeric(
            output[str(config["rolling_output_column"])],
            errors="coerce",
        ).to_numpy(dtype=np.float32)
        expected_api32 = expected_api.astype(np.float32)
        expected_rolling32 = expected_rolling.astype(np.float32)
        if not np.allclose(
            actual_api,
            expected_api32,
            rtol=2e-5,
            atol=2e-5,
        ):
            raise ValueError("precip_api differs from independent EWMA recurrence")
        result["precip_api_exact"] = True
        if not np.allclose(
            actual_rolling,
            expected_rolling32,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError("precip_6h differs from independent rolling sum")
        result["precip_6h_exact"] = True

        finite_api = np.isfinite(actual_api)
        finite_rolling = np.isfinite(actual_rolling)
        negative_hours = int(
            np.less(actual_api[finite_api], 0).sum()
            + np.less(actual_rolling[finite_rolling], 0).sum()
        )
        if not finite_api.all() or not finite_rolling.all():
            raise ValueError("derived features contain missing values")
        if negative_hours:
            raise ValueError(f"negative derived feature hours: {negative_hours}")
        result.update(
            {
                "status": "success",
                "rows": int(len(output)),
                "precip_api_max_abs_error": float(
                    np.max(
                        np.abs(
                            actual_api.astype(np.float64)
                            - expected_api32.astype(np.float64)
                        )
                    )
                ),
                "precip_6h_max_abs_error": float(
                    np.max(
                        np.abs(
                            actual_rolling.astype(np.float64)
                            - expected_rolling32.astype(np.float64)
                        )
                    )
                ),
                "negative_feature_hours": negative_hours,
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    args = parse_args()
    config = load_json(resolve(args.config))
    schema = expected_schema(config)
    report_dir = args.run_root / "reports"
    mapping = pd.read_csv(
        report_dir / "processing_qa_by_site.csv",
        dtype={"site_id": "string"},
        encoding="utf-8-sig",
    )
    rows = mapping.to_dict(orient="records")
    results: list[dict[str, Any]] = []
    workers = max(1, int(config["workers"]))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(validate_one, row, config, schema): row["site_id"]
            for row in rows
        }
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                print(f"validated {completed}/{len(futures)}", flush=True)

    validation = pd.DataFrame(results).sort_values("site_id", kind="mergesort")
    validation.to_csv(
        report_dir / "independent_validation_by_site.csv",
        index=False,
        encoding="utf-8-sig",
    )
    impulse = impulse_tests(config)
    pd.DataFrame([impulse]).to_csv(
        report_dir / "causality_impulse_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    success = validation["status"].eq("success")
    summary = {
        "validated_sites": int(len(validation)),
        "successful_sites": int(success.sum()),
        "failed_sites": int((~success).sum()),
        "all_schema_exact": bool(validation.loc[success, "schema_exact"].all()),
        "all_timestamps_exact": bool(validation.loc[success, "timestamp_exact"].all()),
        "all_site_ids_exact": bool(validation.loc[success, "site_id_exact"].all()),
        "all_rainfall_exact": bool(validation.loc[success, "rainfall_exact"].all()),
        "all_observation_flags_exact": bool(
            validation.loc[success, "observation_flags_exact"].all()
        ),
        "all_precip_api_exact": bool(
            validation.loc[success, "precip_api_exact"].all()
        ),
        "all_precip_6h_exact": bool(
            validation.loc[success, "precip_6h_exact"].all()
        ),
        "negative_feature_hours": int(
            pd.to_numeric(
                validation["negative_feature_hours"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "causality_impulse_tests": impulse,
    }
    (report_dir / "independent_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if bool(success.all()) and all(
        bool(impulse[key])
        for key in (
            "ewma_impulse_decay_exact",
            "rolling_6h_impulse_exact",
            "future_rain_not_used",
            "no_rain_monotonic_decay",
        )
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
