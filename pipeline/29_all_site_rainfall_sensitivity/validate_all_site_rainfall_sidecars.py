"""Independently validate all-site prior-format rainfall sidecars."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COLUMNS = [
    "timestamp",
    "site_id",
    "00045",
    "00045_observed_hour",
    "00045_missing",
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


def load_raw(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    frame = pd.read_csv(
        path,
        usecols=["precipitation_mm", "year", "month", "day", "hour"],
        low_memory=False,
    )
    timestamp = pd.to_datetime(
        {
            "year": pd.to_numeric(frame["year"], errors="coerce"),
            "month": pd.to_numeric(frame["month"], errors="coerce"),
            "day": pd.to_numeric(frame["day"], errors="coerce"),
            "hour": pd.to_numeric(frame["hour"], errors="coerce"),
        },
        errors="coerce",
        utc=True,
    )
    rainfall = pd.to_numeric(frame["precipitation_mm"], errors="coerce")
    valid = timestamp.notna()
    series = pd.Series(
        rainfall.loc[valid].to_numpy(dtype=np.float64),
        index=pd.DatetimeIndex(timestamp.loc[valid]),
    )
    series = series[~series.index.duplicated(keep="last")].sort_index()
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="h", tz="UTC")
    complete = series.reindex(full_index)
    return full_index, complete.to_numpy(dtype=np.float64)


def validate_one(row: dict[str, Any]) -> dict[str, Any]:
    site_id = str(row["site_id"])
    result: dict[str, Any] = {
        "site_id": site_id,
        "source": row.get("source"),
        "status": "failed",
        "schema_exact": False,
        "timestamps_exact": False,
        "site_ids_exact": False,
        "rainfall_exact": False,
        "flags_exact": False,
        "negative_rainfall_hours": 0,
        "error_type": "",
        "error_message": "",
    }
    try:
        raw_index, raw_rainfall = load_raw(Path(str(row["canonical_file_path"])))
        sidecar = pd.read_parquet(Path(str(row["sidecar_file"])))
        result["schema_exact"] = list(sidecar.columns) == EXPECTED_COLUMNS
        actual_time = pd.to_datetime(sidecar["timestamp"], errors="coerce", utc=True)
        result["timestamps_exact"] = bool(
            len(actual_time) == len(raw_index)
            and np.array_equal(
                actual_time.astype("int64").to_numpy(),
                raw_index.astype("int64"),
            )
        )
        result["site_ids_exact"] = bool(sidecar["site_id"].astype(str).eq(site_id).all())
        actual_rainfall = pd.to_numeric(
            sidecar["00045"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        result["rainfall_exact"] = bool(
            len(actual_rainfall) == len(raw_rainfall)
            and np.array_equal(np.isnan(actual_rainfall), np.isnan(raw_rainfall))
            and np.allclose(
                actual_rainfall,
                raw_rainfall,
                rtol=0.0,
                atol=1e-6,
                equal_nan=True,
            )
        )
        expected_observed = np.isfinite(raw_rainfall)
        actual_observed = sidecar["00045_observed_hour"].astype(bool).to_numpy()
        actual_missing = (
            pd.to_numeric(sidecar["00045_missing"], errors="coerce")
            .to_numpy(dtype=np.int8)
            .astype(bool)
        )
        result["flags_exact"] = bool(
            np.array_equal(actual_observed, expected_observed)
            and np.array_equal(actual_missing, ~expected_observed)
        )
        result["negative_rainfall_hours"] = int(
            np.less(actual_rainfall[np.isfinite(actual_rainfall)], 0).sum()
        )
        checks = [
            result["schema_exact"],
            result["timestamps_exact"],
            result["site_ids_exact"],
            result["rainfall_exact"],
            result["flags_exact"],
            result["negative_rainfall_hours"] == 0,
        ]
        result["status"] = "success" if all(checks) else "failed"
    except Exception as exc:  # noqa: BLE001
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
    return result


def main() -> int:
    args = parse_args()
    config = load_json(resolve(args.config))
    run_root = resolve(args.run_root)
    qa = pd.read_csv(
        run_root / "reports" / "alignment_qa_by_site.csv",
        dtype={"site_id": "string"},
        encoding="utf-8-sig",
    )
    qa = qa.loc[qa["status"].eq("success")].copy()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(config["workers"]))) as executor:
        futures = [executor.submit(validate_one, row) for row in qa.to_dict("records")]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"validated {completed}/{len(futures)}", flush=True)
    frame = pd.DataFrame(results).sort_values("site_id", kind="mergesort")
    frame.to_csv(
        run_root / "reports" / "independent_validation_by_site.csv",
        index=False,
        encoding="utf-8-sig",
    )
    success = frame["status"].eq("success")
    summary = {
        "validated_sites": int(len(frame)),
        "successful_sites": int(success.sum()),
        "failed_sites": int((~success).sum()),
        "all_schema_exact": bool(frame["schema_exact"].all()),
        "all_timestamps_exact": bool(frame["timestamps_exact"].all()),
        "all_site_ids_exact": bool(frame["site_ids_exact"].all()),
        "all_rainfall_exact": bool(frame["rainfall_exact"].all()),
        "all_flags_exact": bool(frame["flags_exact"].all()),
        "negative_rainfall_hours": int(frame["negative_rainfall_hours"].sum()),
    }
    (run_root / "reports" / "independent_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if bool(success.all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
