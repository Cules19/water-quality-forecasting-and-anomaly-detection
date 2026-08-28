#!/usr/bin/env python
"""Build the step-1 site manifest for the common-predictor ~1000-site study.

This is a lightweight inventory-stage script. It does not read observation
values and does not preprocess or train models. It uses the existing clean-file
inventory, deduplicates physical formats and historical snapshots, selects one
representative latest logical dataset per site, and writes auditable candidate,
exclusion, and per-site variable-availability tables.

The proposed common-predictor experiment uses:

* target: pH (00400)
* predictors: water temperature (00010), specific conductance (00095),
  dissolved oxygen (00300), and turbidity (63680)

The 14-day and 30-day thresholds are reported as cohort flags. They are not
used to hide or delete schema-complete candidates at this stage.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_INVENTORY = PROJECT_ROOT / "data" / "work" / "analysis_work" / "inventory.json"
DEFAULT_REGISTRY = PROJECT_ROOT / "data" / "reference" / "sites_register.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "ph_common1000_step1_v1"

TARGET_CODE = "00400"
PREDICTOR_CODES = ["00010", "00095", "00300", "63680"]
REQUIRED_CODES = [TARGET_CODE, *PREDICTOR_CODES]
ALLOWED_EXTENSIONS = {".csv", ".parquet", ".tsv"}
FORMAT_PREFERENCE = {".parquet": 3, ".csv": 2, ".tsv": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the common-predictor candidate and exclusion manifests."
    )
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalise_site_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit() and len(text) < 8:
        text = text.zfill(8)
    return text


def parse_columns(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_date_ordinal(value: Any) -> int:
    timestamp = pd.to_datetime(value, errors="coerce")
    return int(timestamp.toordinal()) if pd.notna(timestamp) else -1


def inspect_header(path: Path) -> list[str]:
    """Inspect only a schema/header when the existing inventory lacks columns."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return list(pq.ParquetFile(path).schema_arrow.names)
    if suffix == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    if suffix == ".tsv":
        return list(pd.read_csv(path, sep="\t", nrows=0).columns)
    return []


def ensure_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def choose_format_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the most efficient physical representation of one logical dataset."""
    return max(
        records,
        key=lambda row: (
            FORMAT_PREFERENCE.get(str(row.get("extension", "")).lower(), 0),
            bool(row.get("columns")),
            safe_int(row.get("row_count")) or -1,
            safe_int(row.get("size_bytes")) or -1,
        ),
    )


def combine_logical_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected = dict(choose_format_record(records))

    schema_source = max(
        records,
        key=lambda row: (
            len(parse_columns(row.get("columns"))),
            FORMAT_PREFERENCE.get(str(row.get("extension", "")).lower(), 0),
        ),
    )
    selected["columns"] = schema_source.get("columns")
    selected["column_count"] = schema_source.get("column_count")
    selected["schema_hash"] = schema_source.get("schema_hash")

    known_rows = [safe_int(row.get("row_count")) for row in records]
    known_rows = [value for value in known_rows if value is not None]
    if known_rows:
        selected["row_count"] = max(known_rows)

    selected["physical_format_count"] = len(records)
    selected["available_formats"] = ";".join(
        sorted({str(row.get("extension", "")).lstrip(".") for row in records})
    )
    return selected


def logical_datasets(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest = [
        row
        for row in inventory
        if row.get("top_folder") == "clean"
        and row.get("is_latest_snapshot") is True
        and str(row.get("extension", "")).lower() in ALLOWED_EXTENSIONS
        and normalise_site_id(row.get("site_id"))
    ]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in latest:
        logical_key = str(row.get("normalized_base") or row.get("relative_path"))
        grouped.setdefault(logical_key, []).append(row)

    output: list[dict[str, Any]] = []
    for logical_key, records in grouped.items():
        combined = combine_logical_dataset(records)
        combined["logical_key"] = logical_key
        combined["site_id"] = normalise_site_id(combined.get("site_id"))

        if not parse_columns(combined.get("columns")):
            source_path = DATA_ROOT / str(combined["relative_path"])
            try:
                inspected = inspect_header(source_path)
                combined["columns"] = ", ".join(inspected)
                combined["column_count"] = len(inspected)
                combined["schema_inspection_note"] = "header/schema inspected on demand"
            except Exception as exc:  # preserve the site with an auditable failure note
                combined["schema_inspection_note"] = f"schema inspection failed: {exc}"
        else:
            combined["schema_inspection_note"] = "columns reused from inventory"

        output.append(combined)
    return output


def dataset_required_codes(row: dict[str, Any]) -> set[str]:
    columns = set(parse_columns(row.get("columns")))
    return {code for code in REQUIRED_CODES if code in columns}


def dataset_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    present = dataset_required_codes(row)
    complete = set(REQUIRED_CODES).issubset(present)
    return (
        int(complete),
        len(present),
        safe_int(row.get("coverage_days")) or -1,
        safe_int(row.get("row_count")) or -1,
        safe_date_ordinal(row.get("end_date")),
        FORMAT_PREFERENCE.get(str(row.get("extension", "")).lower(), 0),
        str(row.get("logical_key", "")),
    )


def select_site_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=dataset_rank)


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def registry_fields(site_id: str, registry: dict[str, Any]) -> dict[str, Any]:
    entry = registry.get(site_id)
    if not isinstance(entry, dict):
        return {
            "registry_match": False,
            "registry_source": "",
            "registry_status": "",
            "registry_still_active": None,
            "country_code": "",
            "latitude": None,
            "longitude": None,
        }
    return {
        "registry_match": True,
        "registry_source": entry.get("source", ""),
        "registry_status": entry.get("status", ""),
        "registry_still_active": entry.get("still_active"),
        "country_code": entry.get("country_cd", ""),
        "latitude": entry.get("latitude"),
        "longitude": entry.get("longitude"),
    }


def availability_row(
    site_id: str,
    selected: dict[str, Any],
    site_dataset_count: int,
    registry: dict[str, Any],
) -> dict[str, Any]:
    columns = set(parse_columns(selected.get("columns")))
    present = [code for code in REQUIRED_CODES if code in columns]
    missing = [code for code in REQUIRED_CODES if code not in columns]
    coverage_days = safe_int(selected.get("coverage_days"))
    complete = not missing

    if complete and coverage_days is not None and coverage_days >= 30:
        duration_tier = "primary_30d_plus"
    elif complete and coverage_days is not None and coverage_days >= 14:
        duration_tier = "supplement_14_to_29d"
    elif complete and coverage_days is not None:
        duration_tier = "short_under_14d"
    elif complete:
        duration_tier = "coverage_unknown"
    else:
        duration_tier = "schema_incomplete"

    row = {
        "site_id": site_id,
        "network": selected.get("network", ""),
        **registry_fields(site_id, registry),
        "selected_file": rel(DATA_ROOT / str(selected.get("relative_path", ""))),
        "file_name": selected.get("file_name", ""),
        "source_format": str(selected.get("extension", "")).lstrip("."),
        "available_formats": selected.get("available_formats", ""),
        "latest_logical_dataset_count": site_dataset_count,
        "start_date": selected.get("start_date"),
        "end_date": selected.get("end_date"),
        "coverage_days": coverage_days,
        "row_count": safe_int(selected.get("row_count")),
        "column_count": safe_int(selected.get("column_count")),
        "schema_hash": selected.get("schema_hash", ""),
        "schema_inspection_note": selected.get("schema_inspection_note", ""),
    }
    for code in REQUIRED_CODES:
        row[f"has_{code}"] = code in columns
    row.update(
        {
            "required_present_count": len(present),
            "required_present_codes": ";".join(present),
            "required_missing_codes": ";".join(missing),
            "common_schema_complete": complete,
            "eligible_min_14_days": complete and coverage_days is not None and coverage_days >= 14,
            "eligible_min_30_days": complete and coverage_days is not None and coverage_days >= 30,
            "duration_tier": duration_tier,
        }
    )
    return row


def write_csv(df: pd.DataFrame, path: Path) -> None:
    # UTF-8 BOM makes leading-zero IDs and Chinese text safer when opened in Excel.
    df.to_csv(path, index=False, encoding="utf-8-sig")


def network_counts(df: pd.DataFrame, mask: Iterable[bool] | None = None) -> dict[str, int]:
    subset = df if mask is None else df.loc[list(mask)]
    counts = Counter(str(value) for value in subset["network"].fillna("Unknown"))
    return dict(sorted(counts.items()))


def build_summary(
    inventory: list[dict[str, Any]],
    logical: list[dict[str, Any]],
    availability: pd.DataFrame,
    candidate: pd.DataFrame,
    excluded: pd.DataFrame,
    inventory_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    individual_availability = {
        code: int(availability[f"has_{code}"].sum()) for code in REQUIRED_CODES
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": rel(SCRIPT_PATH),
        "inventory_source": rel(inventory_path),
        "registry_source": rel(registry_path),
        "target_code": TARGET_CODE,
        "predictor_codes": PREDICTOR_CODES,
        "required_codes": REQUIRED_CODES,
        "counts": {
            "clean_physical_files_all_snapshots": sum(
                row.get("top_folder") == "clean" for row in inventory
            ),
            "clean_latest_physical_tabular_files": sum(
                row.get("top_folder") == "clean"
                and row.get("is_latest_snapshot") is True
                and str(row.get("extension", "")).lower() in ALLOWED_EXTENSIONS
                for row in inventory
            ),
            "clean_latest_logical_datasets": len(logical),
            "clean_latest_unique_sites": len(availability),
            "common_schema_candidate_sites": len(candidate),
            "candidate_sites_min_14_days": int(candidate["eligible_min_14_days"].sum()),
            "candidate_sites_min_30_days": int(candidate["eligible_min_30_days"].sum()),
            "excluded_sites": len(excluded),
            "registry_matches_all_sites": int(availability["registry_match"].sum()),
            "registry_matches_candidates": int(candidate["registry_match"].sum()),
        },
        "individual_variable_site_availability": individual_availability,
        "candidate_network_counts": network_counts(candidate),
        "primary_30d_network_counts": network_counts(
            candidate, candidate["eligible_min_30_days"]
        ),
        "notes": [
            "Physical CSV/Parquet copies of the same normalized dataset are counted once.",
            "Only inventory rows marked is_latest_snapshot=true are in scope.",
            "One representative latest logical dataset is selected per site.",
            "Candidates require 00400, 00010, 00095, 00300 and 63680 in the actual schema.",
            "The 14-day and 30-day thresholds are flags, not hidden deletion rules.",
            "Observation-level missingness and unit harmonisation are deferred to step 2/3.",
        ],
    }


def build_report(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    variable_lines = "\n".join(
        f"- `{code}`: {count:,} sites"
        for code, count in summary["individual_variable_site_availability"].items()
    )
    network_lines = "\n".join(
        f"- {network}: {count:,} candidates"
        for network, count in summary["candidate_network_counts"].items()
    )
    return f"""# Common-predictor ~1000-site step-1 manifest

Generated: {summary['generated_at']}

## Cohort definition

- Target: pH (`00400`)
- Predictors: water temperature (`00010`), specific conductance (`00095`), dissolved oxygen (`00300`), turbidity (`63680`)
- Latest unique clean sites: {counts['clean_latest_unique_sites']:,}
- Schema-complete candidates: {counts['common_schema_candidate_sites']:,}
- Candidates with at least 14 days: {counts['candidate_sites_min_14_days']:,}
- Candidates with at least 30 days: {counts['candidate_sites_min_30_days']:,}
- Schema-incomplete exclusions: {counts['excluded_sites']:,}

## Individual variable availability

{variable_lines}

## Candidate networks

{network_lines}

## Interpretation

This is a schema and time-span audit only. It does not yet validate units,
physical ranges, observation-level missingness, duplicate timestamps, sampling
regularity, or the amount of interpolation that would be required. The 30-day
cohort is therefore preliminary and may shrink during quality control.
"""


def main() -> None:
    args = parse_args()
    inventory_path = args.inventory.resolve()
    registry_path = args.registry.resolve()
    output_dir = ensure_output_dir(args.output_dir)

    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory not found: {inventory_path}")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(inventory, list):
        raise TypeError("Inventory must be a JSON list of file records.")
    registry = load_registry(registry_path)

    logical = logical_datasets(inventory)
    by_site: dict[str, list[dict[str, Any]]] = {}
    for row in logical:
        by_site.setdefault(normalise_site_id(row.get("site_id")), []).append(row)

    availability_rows = []
    for site_id, site_rows in sorted(by_site.items()):
        selected = select_site_dataset(site_rows)
        availability_rows.append(
            availability_row(site_id, selected, len(site_rows), registry)
        )

    availability = pd.DataFrame(availability_rows).sort_values("site_id").reset_index(drop=True)
    candidate = availability.loc[availability["common_schema_complete"]].copy()
    candidate = candidate.sort_values(
        ["eligible_min_30_days", "coverage_days", "row_count", "site_id"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    candidate.insert(0, "candidate_rank", range(1, len(candidate) + 1))
    candidate["selection_reason"] = "latest best-coverage dataset with complete common schema"

    excluded = availability.loc[~availability["common_schema_complete"]].copy()
    excluded["exclusion_reason"] = excluded["required_missing_codes"].map(
        lambda value: "schema unavailable"
        if not value
        else "missing required variables"
    )
    excluded = excluded.sort_values(
        ["required_present_count", "coverage_days", "site_id"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    # Reproducibility and reconciliation checks.
    assert availability["site_id"].is_unique
    assert candidate["site_id"].is_unique
    assert excluded["site_id"].is_unique
    assert set(candidate["site_id"]).isdisjoint(set(excluded["site_id"]))
    assert set(candidate["site_id"]) | set(excluded["site_id"]) == set(availability["site_id"])
    assert candidate["common_schema_complete"].all()
    assert candidate[[f"has_{code}" for code in REQUIRED_CODES]].all(axis=None)
    assert len(candidate) + len(excluded) == len(availability)

    summary = build_summary(
        inventory,
        logical,
        availability,
        candidate,
        excluded,
        inventory_path,
        registry_path,
    )

    write_csv(candidate, output_dir / "candidate_sites_1200.csv")
    write_csv(excluded, output_dir / "excluded_sites.csv")
    write_csv(availability, output_dir / "variable_availability.csv")
    (output_dir / "step1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "STEP1_REPORT.md").write_text(build_report(summary), encoding="utf-8")

    counts = summary["counts"]
    print(f"Output directory: {output_dir}")
    print(f"Latest unique clean sites: {counts['clean_latest_unique_sites']:,}")
    print(f"Schema-complete candidates: {counts['common_schema_candidate_sites']:,}")
    print(f"Candidates >=14 days: {counts['candidate_sites_min_14_days']:,}")
    print(f"Candidates >=30 days: {counts['candidate_sites_min_30_days']:,}")
    print(f"Excluded sites: {counts['excluded_sites']:,}")


if __name__ == "__main__":
    main()
