"""Independently validate API sidecars using a separate recurrence implementation."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


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


def variants_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    warmup_half_lives = float(config["warmup_half_lives"])
    return [
        {
            "name": str(item["name"]),
            "half_life_hours": int(item["half_life_hours"]),
            "alpha": float(0.5 ** (1.0 / int(item["half_life_hours"]))),
            "warmup_hours": int(
                math.ceil(int(item["half_life_hours"]) * warmup_half_lives)
            ),
        }
        for item in config["api_variants"]
    ]


def expected_schema(variants: list[dict[str, Any]]) -> list[str]:
    schema = [
        "timestamp",
        "site_id",
        "00045",
        "00045_observed_hour",
        "00045_missing",
    ]
    for variant in variants:
        name = variant["name"]
        schema.extend(
            [
                name,
                f"{name}_valid",
                f"{name}_missing",
                f"{name}_warmup",
            ]
        )
    return schema


def load_full_rainfall(path: Path) -> pd.Series:
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
    return rainfall.reindex(full_index)


def observed_segments(observed: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], observed.astype(bool), [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[0::2], changes[1::2], strict=True))


def independent_api(
    full_rainfall: pd.Series,
    alpha: float,
    warmup_hours: int,
) -> tuple[pd.Series, pd.Series]:
    """Use pandas EWM with a prepended zero to reproduce the causal recurrence."""
    rainfall = full_rainfall.to_numpy(dtype=np.float64)
    observed = np.isfinite(rainfall)
    output = np.full(len(rainfall), np.nan, dtype=np.float64)
    warmup = np.zeros(len(rainfall), dtype=bool)
    beta = 1.0 - alpha
    for start, end in observed_segments(observed):
        segment = rainfall[start:end]
        extended = np.concatenate(([0.0], segment))
        ewm = (
            pd.Series(extended)
            .ewm(alpha=beta, adjust=False)
            .mean()
            .to_numpy(dtype=np.float64)
        )
        recurrence = ewm[1:] / beta
        positions = np.arange(len(segment))
        segment_warmup = positions < warmup_hours
        warmup[start:end] = segment_warmup
        valid = ~segment_warmup
        output[start:end][valid] = recurrence[valid]
    return (
        pd.Series(output, index=full_rainfall.index),
        pd.Series(warmup, index=full_rainfall.index),
    )


def impulse_test(variant: dict[str, Any]) -> dict[str, Any]:
    alpha = float(variant["alpha"])
    rainfall = np.zeros(50, dtype=np.float64)
    rainfall[0] = 10.0
    recurrence = np.empty_like(rainfall)
    previous = 0.0
    for index, current in enumerate(rainfall):
        previous = current + alpha * previous
        recurrence[index] = previous
    expected = 10.0 * np.power(alpha, np.arange(len(rainfall)))
    return {
        "api_feature": variant["name"],
        "alpha": alpha,
        "impulse_decay_exact": bool(
            np.allclose(recurrence, expected, rtol=1e-12, atol=1e-12)
        ),
        "no_rain_monotonic_decay": bool(np.all(np.diff(recurrence) <= 0)),
        "future_rain_not_used": bool(recurrence[0] == 10.0),
    }


def validate_row(
    row: dict[str, Any],
    variants: list[dict[str, Any]],
    schema: list[str],
) -> dict[str, Any]:
    site_id = str(row["site_id"])
    api_path = Path(str(row["api_sidecar_file"]))
    source_path = Path(str(row["canonical_file_path"]))
    prepared_path = Path(str(row["prepared_file"]))
    base_path = Path(str(row["base_rainfall_sidecar"]))
    result: dict[str, Any] = {
        "site_id": site_id,
        "source": row.get("source"),
        "network": row.get("network"),
        "split": row.get("split"),
        "status": "failed",
        "error": "",
        "timestamp_exact_match": False,
        "base_rainfall_exact_match": False,
        "site_id_exact_match": False,
        "all_api_values_match": False,
        "all_api_flags_match": False,
        "all_missing_semantics_valid": False,
        "negative_api_hours": 0,
    }
    try:
        actual_schema = pq.read_schema(api_path).names
        if actual_schema != schema:
            raise ValueError(f"schema mismatch: {actual_schema}")

        sidecar = pd.read_parquet(api_path)
        prepared = pd.read_parquet(prepared_path, columns=["timestamp"])
        base = pd.read_parquet(
            base_path,
            columns=[
                "timestamp",
                "00045",
                "00045_observed_hour",
                "00045_missing",
            ],
        )
        if len(sidecar) != len(prepared) or len(sidecar) != len(base):
            raise ValueError(
                f"row mismatch: API={len(sidecar)}, prepared={len(prepared)}, base={len(base)}"
            )

        api_timestamps = pd.to_datetime(sidecar["timestamp"], errors="coerce")
        prepared_timestamps = pd.to_datetime(prepared["timestamp"], errors="coerce")
        base_timestamps = pd.to_datetime(base["timestamp"], errors="coerce")
        timestamp_match = bool(
            api_timestamps.equals(prepared_timestamps)
            and api_timestamps.equals(base_timestamps)
        )
        if not timestamp_match:
            raise ValueError("API timestamps differ from prepared/base timestamps")
        result["timestamp_exact_match"] = True

        site_ids = sidecar["site_id"].astype("string")
        site_match = bool(site_ids.nunique(dropna=False) == 1 and site_ids.iloc[0] == site_id)
        if not site_match:
            raise ValueError("API sidecar site ID mismatch")
        result["site_id_exact_match"] = True

        actual_rainfall = pd.to_numeric(sidecar["00045"], errors="coerce").to_numpy(
            dtype=np.float32
        )
        base_rainfall = pd.to_numeric(base["00045"], errors="coerce").to_numpy(
            dtype=np.float32
        )
        if not np.array_equal(np.isnan(actual_rainfall), np.isnan(base_rainfall)):
            raise ValueError("API/base rainfall missingness mismatch")
        finite_rainfall = np.isfinite(actual_rainfall)
        if finite_rainfall.any() and not np.array_equal(
            actual_rainfall[finite_rainfall],
            base_rainfall[finite_rainfall],
        ):
            raise ValueError("API/base rainfall values mismatch")
        actual_observed = sidecar["00045_observed_hour"].astype(bool).to_numpy()
        base_observed = base["00045_observed_hour"].astype(bool).to_numpy()
        actual_missing = pd.to_numeric(
            sidecar["00045_missing"], errors="coerce"
        ).to_numpy(dtype=np.int8)
        base_missing = pd.to_numeric(
            base["00045_missing"], errors="coerce"
        ).to_numpy(dtype=np.int8)
        if not (
            np.array_equal(actual_observed, base_observed)
            and np.array_equal(actual_missing, base_missing)
            and np.array_equal(actual_observed, finite_rainfall)
            and np.array_equal(actual_missing, (~finite_rainfall).astype(np.int8))
        ):
            raise ValueError("API/base rainfall flags mismatch")
        result["base_rainfall_exact_match"] = True

        keys = pd.DatetimeIndex(prepared_timestamps)
        utc_keys = keys.tz_localize("UTC") if keys.tz is None else keys.tz_convert("UTC")
        full_rainfall = load_full_rainfall(source_path)
        all_value_matches = True
        all_flag_matches = True
        all_semantics = True
        total_negative_hours = 0

        for variant in variants:
            name = variant["name"]
            expected_api, expected_warmup = independent_api(
                full_rainfall,
                alpha=float(variant["alpha"]),
                warmup_hours=int(variant["warmup_hours"]),
            )
            expected_values = expected_api.reindex(utc_keys).to_numpy(dtype=np.float32)
            expected_valid = np.isfinite(expected_values)
            expected_warmup_values = (
                expected_warmup.reindex(utc_keys, fill_value=False)
                .to_numpy(dtype=bool)
            )

            actual_values = pd.to_numeric(sidecar[name], errors="coerce").to_numpy(
                dtype=np.float32
            )
            actual_valid = sidecar[f"{name}_valid"].astype(bool).to_numpy()
            actual_missing = pd.to_numeric(
                sidecar[f"{name}_missing"], errors="coerce"
            ).to_numpy(dtype=np.int8)
            actual_warmup = sidecar[f"{name}_warmup"].astype(bool).to_numpy()

            missingness_match = np.array_equal(
                np.isnan(expected_values),
                np.isnan(actual_values),
            )
            finite = np.isfinite(expected_values)
            values_match = bool(
                missingness_match
                and (
                    not finite.any()
                    or np.allclose(
                        expected_values[finite],
                        actual_values[finite],
                        rtol=2e-5,
                        atol=2e-5,
                    )
                )
            )
            max_abs_error = (
                float(
                    np.max(
                        np.abs(
                            expected_values[finite].astype(np.float64)
                            - actual_values[finite].astype(np.float64)
                        )
                    )
                )
                if finite.any()
                else 0.0
            )
            flags_match = bool(
                np.array_equal(actual_valid, expected_valid)
                and np.array_equal(
                    actual_missing,
                    (~expected_valid).astype(np.int8),
                )
                and np.array_equal(actual_warmup, expected_warmup_values)
            )
            semantics_valid = bool(
                np.array_equal(actual_valid, np.isfinite(actual_values))
                and np.array_equal(
                    actual_missing,
                    (~actual_valid).astype(np.int8),
                )
                and not np.any(actual_warmup & ~finite_rainfall)
                and not np.any(actual_warmup & actual_valid)
            )
            negative_hours = int(
                np.less(actual_values[np.isfinite(actual_values)], 0).sum()
            )

            result.update(
                {
                    f"{name}_values_match": values_match,
                    f"{name}_flags_match": flags_match,
                    f"{name}_missing_semantics_valid": semantics_valid,
                    f"{name}_max_abs_error": max_abs_error,
                    f"{name}_valid_hours": int(actual_valid.sum()),
                    f"{name}_warmup_hours": int(actual_warmup.sum()),
                    f"{name}_negative_hours": negative_hours,
                }
            )
            all_value_matches &= values_match
            all_flag_matches &= flags_match
            all_semantics &= semantics_valid
            total_negative_hours += negative_hours

        if not all_value_matches:
            raise ValueError("One or more API value columns differ from independent recurrence")
        if not all_flag_matches:
            raise ValueError("One or more API validity/warm-up flags differ")
        if not all_semantics:
            raise ValueError("One or more API missingness semantics are invalid")
        if total_negative_hours:
            raise ValueError(f"Negative API values found: {total_negative_hours}")

        result.update(
            {
                "status": "success",
                "rows": int(len(sidecar)),
                "all_api_values_match": True,
                "all_api_flags_match": True,
                "all_missing_semantics_valid": True,
                "negative_api_hours": total_negative_hours,
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    args = parse_args()
    config = load_json(resolve(args.config))
    variants = variants_from_config(config)
    schema = expected_schema(variants)
    report_dir = args.run_root / "reports"
    mapping = pd.read_csv(
        report_dir / "api_qa_by_site.csv",
        dtype={"site_id": "string"},
        encoding="utf-8-sig",
    )
    rows = mapping.to_dict(orient="records")
    results: list[dict[str, Any]] = []
    workers = max(1, int(config["workers"]))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(validate_row, row, variants, schema): row["site_id"]
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
    success = validation["status"].eq("success")
    impulse = pd.DataFrame([impulse_test(variant) for variant in variants])
    impulse.to_csv(
        report_dir / "causality_impulse_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "validated_sites": int(len(validation)),
        "success_sites": int(success.sum()),
        "failed_sites": int((~success).sum()),
        "all_timestamp_keys_exact": bool(
            validation.loc[success, "timestamp_exact_match"].fillna(False).all()
        ),
        "all_base_rainfall_exact": bool(
            validation.loc[success, "base_rainfall_exact_match"].fillna(False).all()
        ),
        "all_site_ids_exact": bool(
            validation.loc[success, "site_id_exact_match"].fillna(False).all()
        ),
        "all_api_values_match_independent_recurrence": bool(
            validation.loc[success, "all_api_values_match"].fillna(False).all()
        ),
        "all_api_flags_match": bool(
            validation.loc[success, "all_api_flags_match"].fillna(False).all()
        ),
        "all_missing_semantics_valid": bool(
            validation.loc[success, "all_missing_semantics_valid"].fillna(False).all()
        ),
        "negative_api_hours": int(
            pd.to_numeric(
                validation["negative_api_hours"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "all_impulse_decay_tests_pass": bool(
            impulse[
                [
                    "impulse_decay_exact",
                    "no_rain_monotonic_decay",
                    "future_rain_not_used",
                ]
            ]
            .all(axis=None)
        ),
    }
    (report_dir / "independent_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if bool(success.all()) and summary["all_impulse_decay_tests_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
