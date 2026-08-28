#!/usr/bin/env python
"""Post-run duplicate-alias and teacher-compatibility audits for common1000."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis_outputs" / "ph_common1000_processed_v1"
REQUIRED_CODES = ["00400", "00010", "00095", "00300", "63680"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def neon_to_n_alias(site_id: str) -> str | None:
    match = re.fullmatch(r"neon_([A-Za-z0-9]+)_S([0-9]+)", site_id)
    return f"N{match.group(1).upper()}S{match.group(2)}" if match else None


def compare_native_pair(left_path: Path, right_path: Path) -> dict[str, Any]:
    columns = ["timestamp", *REQUIRED_CODES]
    left = pd.read_parquet(left_path, columns=columns)
    right = pd.read_parquet(right_path, columns=columns)
    result: dict[str, Any] = {
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "row_count_equal": len(left) == len(right),
        "timestamps_equal": False,
        "all_required_values_equal": False,
        "maximum_absolute_difference": None,
    }
    if len(left) != len(right):
        return result
    timestamp_equal = np.array_equal(
        pd.to_datetime(left["timestamp"]).to_numpy(dtype="datetime64[ns]"),
        pd.to_datetime(right["timestamp"]).to_numpy(dtype="datetime64[ns]"),
    )
    result["timestamps_equal"] = bool(timestamp_equal)
    all_equal = timestamp_equal
    maximum = 0.0
    for code in REQUIRED_CODES:
        left_values = pd.to_numeric(left[code], errors="coerce").to_numpy(dtype="float64")
        right_values = pd.to_numeric(right[code], errors="coerce").to_numpy(dtype="float64")
        equal = np.allclose(left_values, right_values, rtol=0.0, atol=1e-7, equal_nan=True)
        all_equal &= bool(equal)
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        if finite.any():
            maximum = max(maximum, float(np.max(np.abs(left_values[finite] - right_values[finite]))))
    result["all_required_values_equal"] = bool(all_equal)
    result["maximum_absolute_difference"] = maximum
    return result


def build_alias_audit(status: pd.DataFrame, reports_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    by_site = status.set_index("site_id", drop=False)
    alias_rows: list[dict[str, Any]] = []
    canonical = status.copy()
    canonical["canonical_physical_site_id"] = canonical["site_id"]
    canonical["is_duplicate_alias"] = False
    canonical["preferred_record"] = True
    canonical["duplicate_evidence"] = ""

    for neon_site in sorted(site for site in by_site.index if str(site).startswith("neon_")):
        n_site = neon_to_n_alias(str(neon_site))
        if n_site is None or n_site not in by_site.index:
            alias_rows.append(
                {
                    "neon_site_id": neon_site,
                    "n_series_site_id": n_site or "",
                    "counterpart_exists": False,
                    "duplicate_confirmed": False,
                    "recommendation": "retain; no N-series counterpart found",
                }
            )
            continue
        neon_row = by_site.loc[neon_site]
        n_row = by_site.loc[n_site]
        comparison = compare_native_pair(
            resolve_path(neon_row["native_output"]),
            resolve_path(n_row["native_output"]),
        )
        confirmed = bool(
            comparison["row_count_equal"]
            and comparison["timestamps_equal"]
            and comparison["all_required_values_equal"]
        )
        alias_rows.append(
            {
                "neon_site_id": neon_site,
                "n_series_site_id": n_site,
                "counterpart_exists": True,
                "neon_quality_tier": neon_row["quality_tier"],
                "n_series_quality_tier": n_row["quality_tier"],
                **comparison,
                "duplicate_confirmed": confirmed,
                "recommendation": "use N-series record as preferred physical-site representation" if confirmed else "retain both pending manual review",
            }
        )
        if confirmed:
            mask = canonical["site_id"].eq(neon_site)
            canonical.loc[mask, "canonical_physical_site_id"] = n_site
            canonical.loc[mask, "is_duplicate_alias"] = True
            canonical.loc[mask, "preferred_record"] = False
            canonical.loc[mask, "duplicate_evidence"] = "timestamp and all five cleaned values exactly match N-series counterpart"

    alias_audit = pd.DataFrame(alias_rows)
    preferred = canonical.loc[canonical["preferred_record"]].copy()
    write_csv(alias_audit, reports_dir / "cross_site_alias_audit.csv")
    write_csv(canonical, reports_dir / "canonical_site_manifest.csv")
    write_csv(preferred, reports_dir / "preferred_unique_physical_sites.csv")

    summary = {
        "site_id_records": int(len(canonical)),
        "confirmed_duplicate_alias_records": int(canonical["is_duplicate_alias"].sum()),
        "preferred_unique_physical_sites": int(len(preferred)),
        "primary_strict_site_id_records": int(canonical["quality_tier"].eq("primary_strict").sum()),
        "primary_strict_unique_physical_sites": int(
            preferred["quality_tier"].eq("primary_strict").sum()
        ),
    }
    return alias_audit, canonical, summary


def choose_teacher_artifact(site_id: str) -> tuple[Path | None, Path | None]:
    processed = PROJECT_ROOT / "data" / "processed"
    candidates: list[tuple[Path, Path]] = []
    for parquet in processed.glob(f"{site_id}_*.parquet"):
        if parquet.name.endswith("_processed.parquet"):
            continue
        pipeline = parquet.with_name(f"{parquet.stem}_pipeline.joblib")
        if pipeline.exists():
            candidates.append((parquet, pipeline))
    if not candidates:
        return None, None
    return max(candidates, key=lambda pair: pair[0].stat().st_mtime_ns)


def inverse_teacher_values(values: pd.Series, scaler: Any) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    output = np.full(len(numeric), np.nan, dtype="float64")
    finite = np.isfinite(numeric)
    if finite.any():
        output[finite] = scaler.inverse_transform(numeric[finite].reshape(-1, 1)).ravel()
    return output


def teacher_parity_for_site(site_id: str, status_row: pd.Series) -> list[dict[str, Any]]:
    teacher_path, pipeline_path = choose_teacher_artifact(site_id)
    if teacher_path is None or pipeline_path is None:
        return [{"site_id": site_id, "parameter_code": "", "status": "teacher_artifact_missing"}]
    teacher = pd.read_parquet(teacher_path, columns=REQUIRED_CODES)
    if not isinstance(teacher.index, pd.DatetimeIndex):
        if "timestamp" not in teacher.columns:
            return [{"site_id": site_id, "parameter_code": "", "status": "teacher_timestamp_missing"}]
        teacher = teacher.set_index(pd.to_datetime(teacher.pop("timestamp")))
    teacher.index = pd.to_datetime(teacher.index).tz_localize(None) if teacher.index.tz is not None else pd.to_datetime(teacher.index)
    teacher_duplicate_timestamps = int(teacher.index.duplicated(keep="first").sum())
    if teacher_duplicate_timestamps:
        teacher = teacher.loc[~teacher.index.duplicated(keep="first")].copy()
    native_columns = ["timestamp", *REQUIRED_CODES, *[f"{code}_filled" for code in REQUIRED_CODES[1:]]]
    native = pd.read_parquet(resolve_path(status_row["native_output"]), columns=native_columns)
    sample_count = min(20000, len(native))
    indices = np.unique(np.linspace(0, len(native) - 1, sample_count, dtype="int64")) if sample_count else np.array([], dtype="int64")
    sample = native.iloc[indices].copy()
    sample_timestamp = pd.to_datetime(sample["timestamp"])
    pipeline_object = joblib.load(pipeline_path)
    scalers = pipeline_object.get("scalers", {})
    rows: list[dict[str, Any]] = []
    for code in REQUIRED_CODES:
        if code not in scalers or code not in teacher.columns:
            rows.append(
                {
                    "site_id": site_id,
                    "parameter_code": code,
                    "status": "scaler_or_column_missing",
                }
            )
            continue
        aligned = teacher[code].reindex(sample_timestamp).reset_index(drop=True)
        teacher_native = inverse_teacher_values(aligned, scalers[code])
        new_column = code if code == "00400" else f"{code}_filled"
        new_native = pd.to_numeric(sample[new_column], errors="coerce").to_numpy(dtype="float64")
        comparable = np.isfinite(teacher_native) & np.isfinite(new_native)
        differences = np.abs(teacher_native[comparable] - new_native[comparable])
        rows.append(
            {
                "site_id": site_id,
                "parameter_code": code,
                "status": "compared",
                "teacher_file": str(teacher_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "new_native_file": status_row["native_output"],
                "sampled_timestamps": int(len(sample)),
                "comparable_values": int(comparable.sum()),
                "timestamp_match_rate": float(aligned.notna().mean()) if len(aligned) else 0.0,
                "teacher_duplicate_timestamps_removed_for_audit": teacher_duplicate_timestamps,
                "median_absolute_difference": float(np.median(differences)) if len(differences) else None,
                "maximum_absolute_difference": float(np.max(differences)) if len(differences) else None,
                "within_1e_5_rate": float(np.mean(differences <= 1e-5)) if len(differences) else None,
                "interpretation": "native-value comparison after inverse-transforming teacher output; QC and pH-target policy differences are expected",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_root = resolve_path(args.output_dir)
    reports_dir = output_root / "reports"
    status = pd.read_csv(reports_dir / "site_processing_status.csv", dtype={"site_id": "string"})
    alias_audit, _, alias_summary = build_alias_audit(status, reports_dir)

    by_site = status.set_index("site_id", drop=False)
    parity_rows: list[dict[str, Any]] = []
    for site_id in ["01302020", "01303152", "NPOSES2"]:
        if site_id in by_site.index:
            parity_rows.extend(teacher_parity_for_site(site_id, by_site.loc[site_id]))
        else:
            parity_rows.append(
                {"site_id": site_id, "parameter_code": "", "status": "site_not_in_common1000_manifest"}
            )
    parity = pd.DataFrame(parity_rows)
    write_csv(parity, reports_dir / "teacher_parity_audit.csv")

    compared = parity.loc[parity["status"].eq("compared")].copy()
    summary = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "alias_summary": alias_summary,
        "alias_pairs_checked": int(len(alias_audit)),
        "confirmed_alias_pairs": int(alias_audit.get("duplicate_confirmed", pd.Series(dtype=bool)).fillna(False).sum()),
        "teacher_parity_sites_requested": 3,
        "teacher_parity_parameter_comparisons": int(len(compared)),
        "teacher_parity_comparable_values": int(
            pd.to_numeric(compared.get("comparable_values", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        ),
        "teacher_parity_median_within_1e_5_rate": float(
            pd.to_numeric(compared.get("within_1e_5_rate", pd.Series(dtype=float)), errors="coerce").median()
        ) if len(compared) else None,
        "notes": [
            "Confirmed NEON aliases are retained in the 1204-record processed layer but excluded from the preferred unique-physical-site manifest.",
            "Teacher parity is evaluated in native units after inverse transformation; exact scaled conductance parity is not expected because the new scaler is train-only.",
        ],
    }
    write_json(summary, reports_dir / "postrun_audit_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
