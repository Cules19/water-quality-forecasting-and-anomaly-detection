#!/usr/bin/env python
"""Lightweight processed-only metadata availability check.

This script intentionally avoids heavy data scans. It inspects data/processed
file names, Parquet schemas/metadata, CSV headers, JSON keys, and the registry.
It does not train models and does not build a 300-site dataset.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "ph_processed_metadata_availability_check_v1"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REGISTRY_PATH = PROJECT_ROOT / "data" / "reference" / "sites_register.json"

REQUIRED_TIME_SERIES_FIELDS = [
    "site_id",
    "timestamp",
    "00400",
    "00010",
    "00095",
    "63680",
    "00060",
    "00065",
]
REQUIRED_CODES = ["00400", "00010", "00095", "63680", "00060", "00065"]

STATIC_METADATA_KEYWORDS = {
    "latitude": ["latitude", "lat", "dec_lat_va"],
    "longitude": ["longitude", "lon", "long", "dec_long_va"],
    "site_type": ["site_type", "site_type_cd", "station_type", "site_tp_cd"],
    "catchment": ["catchment", "basin", "huc", "huc_cd", "basin_cd"],
    "drainage_area": ["drainage_area", "drainage_area_km2", "drainage_area_sqmi", "drain_area"],
    "average_discharge": ["average_discharge", "mean_discharge", "official_average_discharge"],
    "elevation": ["elevation", "alt_va", "altitude"],
    "land_cover": ["land_cover", "watershed_land_cover"],
}


def ensure_output(path: Path) -> Path:
    resolved = path.resolve()
    root = OUTPUT_DIR.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"Refusing to write outside output dir: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_csv(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(ensure_output(OUTPUT_DIR / filename), index=False)


def write_text(text: str, filename: str) -> None:
    ensure_output(OUTPUT_DIR / filename).write_text(text, encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def normalise_site_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    text = re.sub(r"\s+", "", text)
    if text.isdigit() and len(text) < 8:
        return text.zfill(8)
    return text


def site_id_from_filename(path: Path) -> str:
    match = re.match(r"^([A-Za-z]?\d{5,}|E\d+[A-Za-z]?)", path.stem)
    return normalise_site_id(match.group(1)) if match else ""


def variables_from_filename(path: Path) -> list[str]:
    return sorted(set(re.findall(r"(?<!\d)(\d{5})(?!\d)", path.stem)))


def flatten_keys(obj: Any, prefix: str = "", limit: int = 120) -> list[str]:
    keys: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            keys.append(full)
            if isinstance(value, dict) and len(keys) < limit:
                keys.extend(flatten_keys(value, full, limit=limit))
            if len(keys) >= limit:
                break
    elif isinstance(obj, list) and obj:
        keys.extend(flatten_keys(obj[0], prefix, limit=limit))
    return keys[:limit]


def inspect_processed_file(path: Path) -> dict[str, Any]:
    ext = path.suffix.lower()
    columns: list[str] = []
    n_rows: int | None = None
    json_keys: list[str] = []
    notes: list[str] = []

    if ext == ".parquet":
        try:
            pf = pq.ParquetFile(path)
            columns = list(pf.schema_arrow.names)
            n_rows = int(pf.metadata.num_rows)
            pandas_meta = pf.metadata.metadata.get(b"pandas") if pf.metadata.metadata else None
            if pandas_meta and b"index_columns" in pandas_meta:
                notes.append("parquet may store pandas index metadata")
        except Exception as exc:
            notes.append(f"parquet schema read failed: {exc}")
    elif ext == ".csv":
        try:
            columns = list(pd.read_csv(path, nrows=0).columns)
            notes.append("CSV header only; row count not computed")
        except Exception as exc:
            notes.append(f"csv header read failed: {exc}")
    elif ext == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            json_keys = flatten_keys(obj)
            columns = json_keys
            if hasattr(obj, "__len__"):
                n_rows = len(obj)
            notes.append("JSON keys only; not treated as time-series table")
        except Exception as exc:
            notes.append(f"json key read failed: {exc}")
    else:
        notes.append("non-tabular/support file; schema not inspected")

    site_id = site_id_from_filename(path)
    col_lower = {c.lower() for c in columns}
    col_raw = set(columns)
    has_site_id = bool({"site_id", "site_id", "site_no", "site", "station_id", "Site_ID"} & col_raw) or any(c.lower() in {"site_id", "site_no", "site", "station_id"} for c in columns)
    has_timestamp = any(c.lower() in {"timestamp", "time", "datetime", "date_time"} for c in columns)
    row = {
        "file_path": rel(path),
        "file_name": path.name,
        "file_type": ext.lstrip("."),
        "size_bytes": path.stat().st_size,
        "site_id_from_filename": site_id,
        "variables_from_filename": ";".join(variables_from_filename(path)),
        "n_rows_from_metadata": n_rows,
        "n_columns_or_keys": len(columns),
        "columns_or_keys_sample": ";".join(columns[:80]),
        "has_site_id_column": has_site_id,
        "has_timestamp_schema": has_timestamp,
        "has_00400_schema": "00400" in col_raw,
        "has_00010_schema": "00010" in col_raw,
        "has_00095_schema": "00095" in col_raw,
        "has_63680_schema": "63680" in col_raw,
        "has_00060_schema": "00060" in col_raw,
        "has_00065_schema": "00065" in col_raw,
        "notes": "; ".join(notes),
    }
    row["has_all_required_timeseries_schema"] = (
        bool(site_id or has_site_id)
        and has_timestamp
        and all(row[f"has_{code}_schema"] for code in REQUIRED_CODES)
    )
    return row


def static_metadata_presence(file_inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field, keywords in STATIC_METADATA_KEYWORDS.items():
        pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
        matches = file_inventory[
            file_inventory["columns_or_keys_sample"].fillna("").str.contains(pattern)
            | file_inventory["file_name"].fillna("").str.contains(pattern)
        ].copy()
        rows.append(
            {
                "metadata_field": field,
                "present_in_processed_schema_or_json_keys": bool(len(matches)),
                "n_processed_files_with_field": int(len(matches)),
                "file_types_with_field": ";".join(sorted(matches["file_type"].dropna().unique().tolist())),
                "example_files": ";".join(matches["file_path"].head(5).tolist()),
                "notes": "Detected by schema/key/name keyword only; no full data scan.",
            }
        )
    return pd.DataFrame(rows)


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def parse_source_description(entry: dict[str, Any]) -> dict[str, Any]:
    raw = entry.get("source_description")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def registry_field_flags(entry: dict[str, Any]) -> dict[str, bool]:
    source_desc = parse_source_description(entry)
    catchment = entry.get("catchment_analysis") if isinstance(entry.get("catchment_analysis"), dict) else {}
    discharge_stats = entry.get("discharge_description", {}).get("statistics", {}) if isinstance(entry.get("discharge_description"), dict) else {}
    drainage = entry.get("drainage_area_km2", source_desc.get("drainage_area_sqkm"))
    return {
        "registry_latitude_available": pd.notna(entry.get("latitude")),
        "registry_longitude_available": pd.notna(entry.get("longitude")),
        "registry_site_type_available": bool(source_desc.get("site_type_cd") or entry.get("site_type")),
        "registry_catchment_available": bool(entry.get("basin_cd") or source_desc.get("huc_cd") or catchment),
        "registry_drainage_area_available": pd.notna(drainage),
        "registry_average_discharge_available": pd.notna(discharge_stats.get("mean_discharge")),
        "registry_elevation_available": pd.notna(entry.get("elevation")),
        "registry_land_cover_available": isinstance(catchment.get("watershed_land_cover"), dict) and bool(catchment.get("watershed_land_cover")),
    }


def build_registry_summary(registry: dict[str, Any], processed_sites: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for site_id in processed_sites:
        entry = registry.get(site_id)
        flags = registry_field_flags(entry) if isinstance(entry, dict) else {
            "registry_latitude_available": False,
            "registry_longitude_available": False,
            "registry_site_type_available": False,
            "registry_catchment_available": False,
            "registry_drainage_area_available": False,
            "registry_average_discharge_available": False,
            "registry_elevation_available": False,
            "registry_land_cover_available": False,
        }
        rows.append({"site_id": site_id, "registry_match": isinstance(entry, dict), **flags})
    match = pd.DataFrame(rows)
    summary_rows = []
    for col in [c for c in match.columns if c.startswith("registry_") and c != "registry_match"]:
        summary_rows.append(
            {
                "registry_metadata_field": col.replace("registry_", "").replace("_available", ""),
                "available_for_processed_site_count": int(match[col].sum()),
                "processed_site_count": int(len(match)),
                "availability_rate_among_processed_sites": float(match[col].mean()) if len(match) else 0.0,
                "source": rel(REGISTRY_PATH),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return summary, match


def write_report(
    inventory: pd.DataFrame,
    schema_summary: pd.DataFrame,
    static_presence: pd.DataFrame,
    registry_summary: pd.DataFrame,
    registry_match: pd.DataFrame,
) -> None:
    processed_sites = sorted([s for s in inventory["site_id_from_filename"].dropna().astype(str).unique().tolist() if s])
    parquet_files = inventory[inventory["file_type"].eq("parquet")]
    csv_files = inventory[inventory["file_type"].eq("csv")]
    json_files = inventory[inventory["file_type"].eq("json")]
    all_required_sites = schema_summary.groupby("site_id_from_filename")["has_all_required_timeseries_schema"].any()
    n_all_required = int(all_required_sites.sum())
    file_type_counts = inventory.groupby("file_type").size().sort_index()
    file_type_lines = "\n".join(f"- `{file_type}`: {count:,}" for file_type, count in file_type_counts.items())
    required_presence_rows = [
        ("site_id / Site_ID", "has_site_id_column"),
        ("timestamp / datetime", "has_timestamp_schema"),
        ("00400", "has_00400_schema"),
        ("00010", "has_00010_schema"),
        ("00095", "has_00095_schema"),
        ("63680", "has_63680_schema"),
        ("00060", "has_00060_schema"),
        ("00065", "has_00065_schema"),
        ("all required fields in same schema", "has_all_required_timeseries_schema"),
    ]
    required_presence_lines = []
    for label, column in required_presence_rows:
        mask = schema_summary[column].fillna(False).astype(bool)
        site_count = schema_summary.loc[mask, "site_id_from_filename"].dropna().astype(str)
        site_count = site_count[site_count.ne("")].nunique()
        required_presence_lines.append(
            f"- `{label}`: {int(mask.sum()):,} tabular files; {site_count:,} processed sites"
        )
    required_presence_text = "\n".join(required_presence_lines)
    registry_match_rate = float(registry_match["registry_match"].mean()) if len(registry_match) else 0.0
    static_absent = static_presence[~static_presence["present_in_processed_schema_or_json_keys"]]["metadata_field"].tolist()
    static_present = static_presence[static_presence["present_in_processed_schema_or_json_keys"]]["metadata_field"].tolist()
    strong_registry_fields = registry_summary[registry_summary["availability_rate_among_processed_sites"] >= 0.5]["registry_metadata_field"].tolist()
    registry_field_lines = "\n".join(
        f"- `{row.registry_metadata_field}`: {int(row.available_for_processed_site_count):,}/{int(row.processed_site_count):,} "
        f"processed sites ({float(row.availability_rate_among_processed_sites):.1%})"
        for row in registry_summary.itertuples(index=False)
    )

    report = f"""# Processed Metadata Availability Check

## 1. What `data/processed` Contains

This lightweight check inspected only file names, Parquet schemas/metadata, CSV headers, and JSON keys. It did not fully scan usable CSV files, rebuild from `data/clean`, train models, build a 300-site dataset, or compute timestamp-level missingness.

- Processed files inspected: {len(inventory):,}
- Parquet files: {len(parquet_files):,}
- CSV files: {len(csv_files):,}
- JSON files: {len(json_files):,}
- Unique site IDs inferred from processed file names: {len(processed_sites):,}
- Sites with processed schema containing `site_id`/filename site, `timestamp`, and all required pH modelling variables (`00400`, `00010`, `00095`, `63680`, `00060`, `00065`): {n_all_required:,}

Full processed file-type inventory:

{file_type_lines}

Processed tabular schemas contain `Site_ID` or an inferable site ID from filename, a time column (`timestamp` or `datetime`), and dynamic water-quality variables. The main schema columns are USGS parameter-code columns such as `00400`, `00010`, `00095`, `63680`, `00060`, and `00065`; some `_processed` files also use readable names such as `Temperature_C`, `Specific_Conductance_uS_cm`, `Dissolved_Oxygen_mg_L`, `pH`, `discharge`, and `Gage_Height_ft`.

Required field presence in processed tabular schemas:

{required_presence_text}

For modelling, `data/processed` is the right primary source to inspect first.

## 2. What Metadata `data/processed` Does Not Contain

Static metadata fields detected in processed schemas/JSON keys: {', '.join(static_present) if static_present else 'none detected'}.

Static metadata fields not detected directly in processed schemas/JSON keys: {', '.join(static_absent) if static_absent else 'none'}.

The Parquet/CSV time-series schemas are primarily measurement tables, not complete site metadata tables. Even where a JSON sidecar contains a few descriptive keys, it should not be treated as the main model-ready metadata source without a separate metadata-building step.

## 3. What Should Come From `sites_register.json`

`sites_register.json` matched {int(registry_match['registry_match'].sum()):,} of {len(registry_match):,} processed site IDs ({registry_match_rate:.1%}).

Registry fields with at least 50% availability among processed sites: {', '.join(strong_registry_fields) if strong_registry_fields else 'none'}.

Registry metadata availability among processed sites:

{registry_field_lines}

The registry should provide official static metadata such as coordinates, basin/catchment identifiers, elevation, drainage area where available, discharge statistics where available, and catchment/land-cover descriptors where available.

## 4. Recommendation for 300-Site Modelling

Yes: the next 300-site modelling pipeline should use `data/processed` as the main time-series source and `data/reference/sites_register.json` as the official static metadata source.

Important caveat: this lightweight check confirms schema and registry availability, not final row-level eligibility. Before training 300-site HGB, build a staged dataset-preparation script that reads only selected processed Parquet files, creates chronological train/val/test splits, and rebuilds train-derived metadata from training rows only.

## 5. Suggested Next Step

Run a lighter staged expansion preparation:

1. Select candidate sites from processed Parquet schemas that contain all required variables.
2. Match registry metadata for those sites.
3. Build 100-site dataset first, then 200, then 300.
4. Only then run the no-metadata vs metadata HGB comparison.
"""
    write_text(report, "processed_metadata_availability_report.md")


def main() -> None:
    files = sorted([p for p in PROCESSED_DIR.iterdir() if p.is_file()])
    inventory = pd.DataFrame([inspect_processed_file(path) for path in files])
    schema_summary = inventory[inventory["file_type"].isin(["parquet", "csv"])].copy()
    static_presence = static_metadata_presence(inventory)
    processed_sites = sorted([s for s in inventory["site_id_from_filename"].dropna().astype(str).unique().tolist() if s])
    registry = load_registry()
    registry_summary, registry_match = build_registry_summary(registry, processed_sites)

    write_csv(inventory, "processed_file_inventory.csv")
    write_csv(schema_summary, "processed_schema_summary.csv")
    write_csv(static_presence, "processed_static_metadata_presence.csv")
    write_csv(registry_summary, "registry_metadata_field_summary.csv")
    write_csv(registry_match, "processed_to_registry_site_match.csv")
    write_report(inventory, schema_summary, static_presence, registry_summary, registry_match)

    all_required_sites = schema_summary.groupby("site_id_from_filename")["has_all_required_timeseries_schema"].any()
    print(
        f"""PROCESSED METADATA AVAILABILITY CHECK COMPLETE

Output folder:
{rel(OUTPUT_DIR)}/

Processed files inspected:
{len(inventory)}

Unique processed site IDs:
{len(processed_sites)}

Sites with all required modelling fields by schema:
{int(all_required_sites.sum())}

Processed sites matched to registry:
{int(registry_match['registry_match'].sum())}

Registry match rate:
{float(registry_match['registry_match'].mean()) if len(registry_match) else 0.0:.3f}

Recommendation:
Use data/processed as the main time-series source and sites_register.json as the static metadata source for staged 100/200/300-site modelling.
""".strip()
    )


if __name__ == "__main__":
    main()
