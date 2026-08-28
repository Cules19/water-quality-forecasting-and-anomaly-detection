"""Independently validate rainfall sidecars against prepared keys and source CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXPECTED_SCHEMA = [
    "timestamp",
    "site_id",
    "00045",
    "00045_observed_hour",
    "00045_missing",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def source_series(path: Path) -> pd.Series:
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
    valid = timestamps.notna()
    series = pd.Series(
        values.loc[valid].to_numpy(dtype=np.float64),
        index=pd.DatetimeIndex(timestamps.loc[valid]),
    )
    return series[~series.index.duplicated(keep="last")].sort_index()


def validate_row(row: pd.Series) -> dict[str, object]:
    site_id = str(row["site_id"])
    sidecar_path = Path(str(row["sidecar_file"]))
    prepared_path = Path(str(row["prepared_file"]))
    source_path = Path(str(row["canonical_file_path"]))
    result: dict[str, object] = {
        "site_id": site_id,
        "source": row.get("source"),
        "split": row.get("split"),
        "status": "failed",
        "error": "",
    }
    try:
        schema = pq.read_schema(sidecar_path).names
        if schema != EXPECTED_SCHEMA:
            raise ValueError(f"schema mismatch: {schema}")
        sidecar = pd.read_parquet(sidecar_path)
        prepared = pd.read_parquet(prepared_path, columns=["timestamp"])
        if len(sidecar) != len(prepared):
            raise ValueError(f"row mismatch: {len(sidecar)} vs {len(prepared)}")
        sidecar_ts = pd.to_datetime(sidecar["timestamp"], errors="coerce")
        prepared_ts = pd.to_datetime(prepared["timestamp"], errors="coerce")
        if not sidecar_ts.equals(prepared_ts):
            raise ValueError("sidecar timestamps differ from prepared timestamps")
        if sidecar["site_id"].astype("string").nunique(dropna=False) != 1:
            raise ValueError("sidecar contains multiple site IDs")
        if str(sidecar["site_id"].iloc[0]) != site_id:
            raise ValueError("sidecar site ID mismatch")

        values = pd.to_numeric(sidecar["00045"], errors="coerce")
        observed = sidecar["00045_observed_hour"].astype(bool)
        missing = pd.to_numeric(sidecar["00045_missing"], errors="coerce").astype("int8")
        if not observed.equals(values.notna()):
            raise ValueError("observed flag is inconsistent with rainfall nulls")
        if not missing.equals((~observed).astype("int8")):
            raise ValueError("missing flag is inconsistent with observed flag")
        if values.dropna().lt(0).any():
            raise ValueError("negative rainfall appears in sidecar")

        keys = pd.DatetimeIndex(prepared_ts)
        utc_keys = keys.tz_localize("UTC") if keys.tz is None else keys.tz_convert("UTC")
        expected = source_series(source_path).reindex(utc_keys).to_numpy(dtype=np.float32)
        actual = values.to_numpy(dtype=np.float32)
        if not np.array_equal(np.isnan(expected), np.isnan(actual)):
            raise ValueError("source/sidecar missingness mismatch")
        finite = np.isfinite(expected)
        if finite.any() and not np.allclose(
            expected[finite],
            actual[finite],
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError("source/sidecar rainfall values mismatch")

        result.update(
            {
                "status": "success",
                "rows": int(len(sidecar)),
                "observed_hours": int(observed.sum()),
                "missing_hours": int((~observed).sum()),
                "zero_hours": int(values.eq(0).sum()),
                "positive_hours": int(values.gt(0).sum()),
                "negative_hours": int(values.lt(0).sum()),
                "timestamp_exact_match": True,
                "source_value_exact_match_float32": True,
                "missing_semantics_valid": True,
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    args = parse_args()
    report_dir = args.run_root / "reports"
    mapping = pd.read_csv(
        report_dir / "alignment_qa_by_site.csv",
        dtype={"site_id": "string"},
        encoding="utf-8-sig",
    )
    rows = [validate_row(row) for _, row in mapping.iterrows()]
    validation = pd.DataFrame(rows).sort_values("site_id", kind="mergesort")
    validation.to_csv(
        report_dir / "independent_validation_by_site.csv",
        index=False,
        encoding="utf-8-sig",
    )
    success = validation["status"].eq("success")
    summary = {
        "validated_sites": int(len(validation)),
        "success_sites": int(success.sum()),
        "failed_sites": int((~success).sum()),
        "all_timestamp_keys_exact": bool(
            validation.loc[success, "timestamp_exact_match"].fillna(False).all()
        ),
        "all_source_values_exact_at_float32": bool(
            validation.loc[success, "source_value_exact_match_float32"].fillna(False).all()
        ),
        "all_missing_semantics_valid": bool(
            validation.loc[success, "missing_semantics_valid"].fillna(False).all()
        ),
        "negative_rainfall_hours": int(
            pd.to_numeric(validation["negative_hours"], errors="coerce").fillna(0).sum()
        ),
    }
    (report_dir / "independent_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if bool(success.all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
