from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import rasterio
import requests


PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / "data/external/upstream1_primary"
DEST = BASE / "downloaded"
OUT = PROJECT / "analysis_outputs/ph_h00_upstream1_v1"
COORDS = OUT / "spatial_inputs/geofresh_910_sites_wgs84.csv"
HYDRO_SHARE = "https://public.igb-berlin.de/index.php/s/agciopgzXjWswF4/download?path=%2F"
ENV_SHARE = "https://public.igb-berlin.de/index.php/s/zw56kEd25NsQqcQ/download?path=%2F"

HYDRO_RASTERS_ALL_REGION_TILES = ["sub_catchment"]
HYDRO_RASTERS_POINT_TILES = ["segment", "basin"]
ENV_SELECTION = {
    "env90m_meritdem_paths_file_sizes.txt": ["elev"],
    "env90m_hydro_paths_file_sizes.txt": [
        "gradient",
        "accumulation",
        "connections",
        "order_strahler",
        "order_shreve",
        "order_horton",
        "order_hack",
        "order_topo",
    ],
    "env90m_observedclimate_paths_file_sizes.txt": [
        "bio12_1981-2010_observed",
        "bio15_1981-2010_observed",
    ],
    "env90m_soil_paths_file_sizes.txt": [
        "phihox",
        "cecsol",
        "orcdrc",
        "clyppt",
        "bdricm",
    ],
    "env90m_landcover_paths_file_sizes.txt": [
        "c190_2020",
        "c10_2020",
        "c20_2020",
        "c30_2020",
        "c50_2020",
        "c60_2020",
        "c70_2020",
        "c80_2020",
        "c90_2020",
        "c100_2020",
    ],
}


def current_remote_size(url: str) -> int:
    """Resolve the current served ZIP size without trusting legacy manifests."""
    with requests.get(
        url,
        headers={"Range": "bytes=0-0"},
        stream=True,
        timeout=(60, 180),
    ) as response:
        response.raise_for_status()
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            return int(content_range.rsplit("/", 1)[1])
        return int(response.headers["Content-Length"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_tile(lon: float, lat: float) -> str:
    horizontal = int(math.floor((lon + 180.0) / 20.0)) * 2
    # Hydrography90m 20-degree tiles have latitude edges at 85, 65, 45,
    # 25, ... degrees (the northern global limit is 85, not 84).
    vertical = int(math.floor((85.0 - lat) / 20.0)) * 2
    return f"h{horizontal:02d}v{vertical:02d}"


def build_plan() -> tuple[pd.DataFrame, pd.DataFrame]:
    coords = pd.read_csv(COORDS, dtype={"id": "string"})
    with rasterio.open(BASE / "regional_unit_ovr.tif") as src:
        coords["regional_unit_id"] = [
            int(value[0]) for value in src.sample(zip(coords.longitude, coords.latitude))
        ]
    coords["point_tile_id"] = [
        point_tile(lon, lat) for lon, lat in zip(coords.longitude, coords.latitude)
    ]
    coords.to_csv(OUT / "site_primary_spatial_index.csv", index=False)

    lookup = pd.read_csv(BASE / "lookup_tile_regunit.txt", sep=r"\s+")
    nonzero_regional_units = sorted(set(coords.loc[coords.regional_unit_id > 0, "regional_unit_id"]))
    region_tiles = sorted(set(lookup.loc[lookup.reg_unit.isin(nonzero_regional_units), "tile"]))
    point_tiles = sorted(set(coords.point_tile_id))

    records: list[dict[str, object]] = []
    hydro_manifest = pd.read_csv(BASE / "hydrography90m_paths_file_sizes.txt", sep=";")
    for variable in HYDRO_RASTERS_ALL_REGION_TILES:
        for tile in region_tiles:
            filename = f"{variable}_{tile}.tif"
            match = hydro_manifest[
                hydro_manifest.file_path.map(lambda value: Path(value).name == filename)
            ]
            if len(match) != 1:
                raise RuntimeError(f"Expected one manifest row for {filename}, found {len(match)}")
            row = match.iloc[0]
            records.append(
                {
                    "family": "Hydrography90m",
                    "variable": variable,
                    "tile": tile,
                    "file_path": row.file_path,
                    "file_size": int(row.file_size),
                    "source_url": HYDRO_SHARE + quote(row.file_path, safe=""),
                }
            )
    for variable in HYDRO_RASTERS_POINT_TILES:
        for tile in point_tiles:
            filename = f"{variable}_{tile}.tif"
            match = hydro_manifest[
                hydro_manifest.file_path.map(lambda value: Path(value).name == filename)
            ]
            if len(match) != 1:
                raise RuntimeError(f"Expected one manifest row for {filename}, found {len(match)}")
            row = match.iloc[0]
            records.append(
                {
                    "family": "Hydrography90m",
                    "variable": variable,
                    "tile": tile,
                    "file_path": row.file_path,
                    "file_size": int(row.file_size),
                    "source_url": HYDRO_SHARE + quote(row.file_path, safe=""),
                }
            )

    for manifest_name, variables in ENV_SELECTION.items():
        manifest = pd.read_csv(BASE / manifest_name, sep=";")
        for variable in variables:
            for tile in region_tiles:
                filename = f"{variable}_{tile}.zip"
                match = manifest[
                    manifest.file_path.map(lambda value: Path(value).name == filename)
                ]
                if len(match) != 1:
                    raise RuntimeError(
                        f"Expected one manifest row for {manifest_name}/{filename}, found {len(match)}"
                    )
                row = match.iloc[0]
                source_url = ENV_SHARE + quote(row.file_path, safe="")
                manifest_size = int(row.file_size)
                # The published MERIT-DEM manifest labels its second column as
                # notneeded_legacy and its byte counts no longer match the
                # current files served by the official share.  Freeze the
                # authoritative HTTP object size for these 21 ZIPs instead.
                resolve_http_size = variable == "elev" or manifest_size < 1024
                file_size = (
                    current_remote_size(source_url) if resolve_http_size else manifest_size
                )
                records.append(
                    {
                        "family": "Environment90m",
                        "variable": variable,
                        "tile": tile,
                        "file_path": row.file_path,
                        "file_size": file_size,
                        "manifest_file_size": manifest_size,
                        "size_source": (
                            "HTTP Content-Range current official object"
                            if resolve_http_size
                            else "official manifest"
                        ),
                        "source_url": source_url,
                    }
                )

    plan = pd.DataFrame(records).sort_values(
        ["family", "variable", "tile"], kind="mergesort"
    )
    if plan.file_path.duplicated().any():
        raise RuntimeError("Duplicate file paths in download plan")
    plan["destination"] = plan.file_path.map(lambda value: str(DEST / value))
    plan.to_csv(OUT / "primary_download_plan.csv", index=False)
    summary = {
        "regional_units": nonzero_regional_units,
        "regional_unit_zero_sites": coords.loc[coords.regional_unit_id == 0, "id"].tolist(),
        "region_tiles": region_tiles,
        "point_tiles": point_tiles,
        "files": len(plan),
        "expected_bytes": int(plan.file_size.sum()),
        "expected_gb_decimal": float(plan.file_size.sum() / 1e9),
        "outcomes_or_predictions_read": False,
        "plan_sha256": hashlib.sha256(plan.to_csv(index=False).encode()).hexdigest(),
    }
    (OUT / "primary_download_plan.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return coords, plan


def download_one(row: dict[str, object]) -> dict[str, object]:
    destination = Path(str(row["destination"]))
    expected = int(row["file_size"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected:
        return {"file_path": row["file_path"], "status": "EXISTING", "bytes": expected}

    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 31):
        existing = partial.stat().st_size if partial.exists() else 0
        if existing == expected:
            break
        if existing > expected:
            partial.unlink()
            existing = 0
        headers: dict[str, str] = {}
        mode = "wb"
        if existing:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"
        before = existing
        try:
            with requests.get(
                str(row["source_url"]), headers=headers, stream=True, timeout=(60, 1800)
            ) as response:
                response.raise_for_status()
                if existing and response.status_code != 206:
                    existing = 0
                    mode = "wb"
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(16 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
        except requests.RequestException:
            if attempt == 30:
                raise
            continue
        actual = partial.stat().st_size
        if actual == expected:
            break
        if actual <= before and attempt == 30:
            raise RuntimeError(
                f"No resumable progress for {row['file_path']}: expected {expected}, got {actual}"
            )
    actual = partial.stat().st_size
    if actual != expected:
        raise RuntimeError(
            f"Size mismatch after retries for {row['file_path']}: expected {expected}, got {actual}"
        )
    os.replace(partial, destination)
    return {"file_path": row["file_path"], "status": "DOWNLOADED", "bytes": actual}


def main() -> None:
    _, plan = build_plan()
    # Preserve the frozen inventory order on disk, but fetch the network identity
    # files first so the outcome-blind snap/delineation audit can start while the
    # larger environmental tables continue downloading.  This changes transfer
    # order only; it does not change the plan contents or their frozen hash.
    priority = {
        "sub_catchment": 0,
        "segment": 1,
        "basin": 2,
        "connections": 3,
        "accumulation": 4,
        "order_strahler": 5,
        "order_shreve": 6,
        "order_horton": 7,
        "order_hack": 8,
        "order_topo": 9,
        "elev": 10,
        "gradient": 11,
    }
    download_queue = plan.assign(
        _priority=plan.variable.map(priority).fillna(100).astype(int)
    ).sort_values(["_priority", "family", "variable", "tile"], kind="mergesort")
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(download_one, row)
            for row in download_queue.drop(columns="_priority").to_dict("records")
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index % 10 == 0 or index == len(futures):
                print(f"{index}/{len(futures)} files complete")

    result_table = pd.DataFrame(results).sort_values("file_path", kind="mergesort")
    result_table.to_csv(OUT / "primary_download_results.csv", index=False)
    audit = {
        "status": "PASS",
        "files": len(result_table),
        "bytes": int(result_table.bytes.sum()),
        "all_expected_sizes_match": True,
        "outcomes_or_predictions_read": False,
    }
    (OUT / "primary_download_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
