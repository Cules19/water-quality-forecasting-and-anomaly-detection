from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import threading
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from pyproj import Geod
from shapely.geometry import shape


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "analysis_outputs/ph_h00_upstream1_v1"
SNAPS_PRE = OUT / "site_snap_matches_pre_review.csv"
SNAPS_FINAL = OUT / "site_snap_matches.csv"
CACHE = PROJECT / "data/external/upstream1_primary/upstream_polygon_api_cache"
ENDPOINT = (
    "https://aqua.igb-berlin.de/pygeoapi/processes/"
    "get-upstream-dissolved-cont/execution"
)
WORKERS = 4
GEOD = Geod(ellps="WGS84")
THREAD_LOCAL = threading.local()


def session() -> requests.Session:
    value = getattr(THREAD_LOCAL, "session", None)
    if value is None:
        value = requests.Session()
        value.headers.update({"User-Agent": "Pollution-ID-UPSTREAM1-audit/1.0"})
        THREAD_LOCAL.session = value
    return value


def cache_path(site_id: str) -> Path:
    return CACHE / f"{site_id}.geojson.gz"


def read_cache(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def valid_geometry(value: dict[str, object]) -> bool:
    return str(value.get("type", "")) in {"Polygon", "MultiPolygon"} and bool(
        value.get("coordinates")
    )


def fetch_one(site_id: str, subc_id: int) -> dict[str, object]:
    target = cache_path(site_id)
    if target.exists():
        value = read_cache(target)
        if valid_geometry(value):
            return {
                "site_id": site_id,
                "subcatchment_id": subc_id,
                "status": "EXISTING",
                "response_bytes": target.stat().st_size,
                "attempts": 0,
            }
    body = {
        "inputs": {
            "geometry_only": True,
            "add_upstream_ids": False,
            "subc_id": int(subc_id),
            "comment": "outcome-blind Pollution_ID H00_UPSTREAM1 spatial audit",
        }
    }
    error = ""
    for attempt in range(1, 3):
        try:
            response = session().post(ENDPOINT, json=body, timeout=(60, 300))
            response.raise_for_status()
            value = response.json()
            if not valid_geometry(value):
                raise ValueError(f"unexpected geometry type: {value.get('type')}")
            temporary = target.with_suffix(target.suffix + ".part")
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
                json.dump(value, handle, separators=(",", ":"))
            temporary.replace(target)
            return {
                "site_id": site_id,
                "subcatchment_id": subc_id,
                "status": "DOWNLOADED",
                "response_bytes": target.stat().st_size,
                "attempts": attempt,
            }
        except Exception as exc:  # retries are recorded in the audit table
            error = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(min(60, 2**attempt))
    return {
        "site_id": site_id,
        "subcatchment_id": subc_id,
        "status": "FAILED",
        "response_bytes": 0,
        "attempts": 2,
        "error": error,
    }


def geometry_sha256(value: dict[str, object]) -> str:
    canonical = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    snap_path = SNAPS_FINAL if SNAPS_FINAL.exists() else SNAPS_PRE
    snaps = pd.read_csv(snap_path, dtype={"site_id": "string"})
    columns = ["site_id", "stream_segment_id"]
    if "upstream_area_km2" in snaps:
        columns.append("upstream_area_km2")
    usable = snaps.loc[snaps.stream_segment_id.gt(0), columns].copy()
    if "upstream_area_km2" in usable:
        usable = usable.sort_values(["upstream_area_km2", "site_id"], kind="mergesort")
    tasks = [(str(row.site_id), int(row.stream_segment_id)) for row in usable.itertuples(index=False)]
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(fetch_one, site_id, subc_id): site_id for site_id, subc_id in tasks}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index % 10 == 0 or index == len(futures):
                failed = sum(row["status"] == "FAILED" for row in results)
                print(f"upstream polygons {index}/{len(futures)} complete; failed={failed}", flush=True)

    audit = pd.DataFrame(results).sort_values("site_id", kind="mergesort")
    missing = snaps.loc[~snaps.site_id.isin(audit.site_id), ["site_id", "stream_segment_id"]].copy()
    if not missing.empty:
        missing["subcatchment_id"] = missing["stream_segment_id"]
        missing["status"] = "NOT_REQUESTED_NO_USABLE_SEGMENT"
        missing["response_bytes"] = 0
        missing["attempts"] = 0
        audit = pd.concat(
            [audit, missing[["site_id", "subcatchment_id", "status", "response_bytes", "attempts"]]],
            ignore_index=True,
        ).sort_values("site_id", kind="mergesort")

    output_gpkg = OUT / "upstream_catchments.gpkg"
    if output_gpkg.exists():
        output_gpkg.unlink()
    features: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    successful = audit.loc[audit.status.isin(["EXISTING", "DOWNLOADED"])]
    for row in successful.itertuples(index=False):
        value = read_cache(cache_path(str(row.site_id)))
        geom = shape(value)
        area_m2, _ = GEOD.geometry_area_perimeter(geom)
        digest = geometry_sha256(value)
        features.append(
            {
                "site_id": str(row.site_id),
                "stream_segment_id": int(row.subcatchment_id),
                "api_geodesic_area_km2": abs(float(area_m2)) / 1_000_000.0,
                "geometry_sha256": digest,
                "geometry_source": "GeoFRESH get-upstream-dissolved-cont API",
                "geometry": geom,
            }
        )
        geometry_rows.append(
            {
                "site_id": str(row.site_id),
                "stream_segment_id": int(row.subcatchment_id),
                "api_geodesic_area_km2": abs(float(area_m2)) / 1_000_000.0,
                "geometry_sha256": digest,
                "geometry_type": geom.geom_type,
                "geometry_valid": bool(geom.is_valid),
            }
        )
        if len(features) >= 25:
            frame = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")
            frame.to_file(
                output_gpkg,
                layer="exact_upstream_catchments",
                driver="GPKG",
                mode="a" if output_gpkg.exists() else "w",
            )
            features.clear()
    if features:
        frame = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")
        frame.to_file(
            output_gpkg,
            layer="exact_upstream_catchments",
            driver="GPKG",
            mode="a" if output_gpkg.exists() else "w",
        )

    geometry_audit = pd.DataFrame(geometry_rows).sort_values("site_id", kind="mergesort")
    geometry_audit.to_csv(OUT / "upstream_catchment_geometry_audit.csv", index=False)
    audit.to_csv(OUT / "upstream_polygon_extraction_audit.csv", index=False)
    summary = {
        "status": "PASS" if len(successful) == len(tasks) else "FAIL",
        "endpoint": ENDPOINT,
        "requested": len(tasks),
        "successful": int(len(successful)),
        "failed": int((audit.status == "FAILED").sum()),
        "not_requested_no_usable_segment": int(
            (audit.status == "NOT_REQUESTED_NO_USABLE_SEGMENT").sum()
        ),
        "geometry_layer": "exact_upstream_catchments",
        "geometry_semantics": "GeoFRESH dissolved union of Hydrography90m upstream subcatchments",
        "canonical_area_semantics": "Hydrography90m accumulation maximum in km2; geometry area retained as QA only",
        "workers": WORKERS,
        "outcomes_or_predictions_read": False,
    }
    (OUT / "upstream_polygon_extraction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
