from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Geod
from rasterio.windows import Window
from shapely.geometry import Point


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "analysis_outputs/ph_h00_upstream1_v1"
PLAN = OUT / "primary_download_plan.csv"
COORD_AUDIT = PROJECT / "analysis_outputs/ph_h00_solar_dynamic1_v1/coordinate_audit.csv"
SPATIAL_INDEX = OUT / "site_primary_spatial_index.csv"
GEOD = Geod(ellps="WGS84")
SEARCH_RADIUS_M = 2_500.0
CELL_PERTURBATION_DEG = 1.0 / 1200.0


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def point_tile(lon: float, lat: float) -> str:
    h = int(math.floor((lon + 180.0) / 20.0)) * 2
    v = int(math.floor((85.0 - lat) / 20.0)) * 2
    return f"h{h:02d}v{v:02d}"


def plan_paths() -> dict[tuple[str, str], Path]:
    plan = pd.read_csv(PLAN)
    return {
        (str(row.variable), str(row.tile)): Path(str(row.destination))
        for row in plan.itertuples(index=False)
    }


def raster_value(path: Path, lon: float, lat: float) -> int:
    with rasterio.open(path) as src:
        value = next(src.sample([(lon, lat)]))[0]
        if src.nodata is not None and value == src.nodata:
            return 0
        if not np.isfinite(value):
            return 0
        return int(value)


def nearest_stream_pixel(
    path: Path, lon: float, lat: float, preferred_id: int
) -> dict[str, object]:
    with rasterio.open(path) as src:
        row, col = src.index(lon, lat)
        # Longitude width is expanded by latitude so the search covers 2.5 km.
        yres = abs(src.transform.e)
        xres = abs(src.transform.a)
        lat_half = max(3, int(math.ceil((SEARCH_RADIUS_M / 111_320.0) / yres)))
        cos_lat = max(0.08, math.cos(math.radians(lat)))
        lon_half = max(
            3, int(math.ceil((SEARCH_RADIUS_M / (111_320.0 * cos_lat)) / xres))
        )
        r0, r1 = max(0, row - lat_half), min(src.height, row + lat_half + 1)
        c0, c1 = max(0, col - lon_half), min(src.width, col + lon_half + 1)
        window = Window(c0, r0, c1 - c0, r1 - r0)
        array = src.read(1, window=window)
        valid = np.isfinite(array) & (array > 0)
        if src.nodata is not None:
            valid &= array != src.nodata
        preferred = valid & (array == preferred_id) if preferred_id > 0 else np.zeros_like(valid)
        used_preferred_subcatchment = bool(preferred.any())
        candidate_mask = preferred if used_preferred_subcatchment else valid
        rr, cc = np.nonzero(candidate_mask)
        if len(rr) == 0:
            return {
                "snapped_longitude": np.nan,
                "snapped_latitude": np.nan,
                "geodesic_snap_distance_m": np.nan,
                "stream_segment_id": 0,
                "used_containing_subcatchment": False,
                "nearest_competing_segment_id": 0,
                "nearest_competing_distance_m": np.nan,
                "candidate_segment_count": 0,
            }
        global_rows = rr + r0
        global_cols = cc + c0
        xs, ys = rasterio.transform.xy(src.transform, global_rows, global_cols, offset="center")
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        _, _, distances = GEOD.inv(
            np.full(xs.shape, lon), np.full(ys.shape, lat), xs, ys
        )
        within = distances <= SEARCH_RADIUS_M + 1e-6
        if not within.any():
            return {
                "snapped_longitude": np.nan,
                "snapped_latitude": np.nan,
                "geodesic_snap_distance_m": np.nan,
                "stream_segment_id": 0,
                "used_containing_subcatchment": used_preferred_subcatchment,
                "nearest_competing_segment_id": 0,
                "nearest_competing_distance_m": np.nan,
                "candidate_segment_count": 0,
            }
        xs, ys, distances = xs[within], ys[within], distances[within]
        ids = array[rr[within], cc[within]].astype(np.int64)
        best_index = int(np.argmin(distances))
        selected_id = int(ids[best_index])
        unique_ids = np.unique(ids)
        competitor_id = 0
        competitor_distance = np.nan
        # Competitors must be searched across every nearby reach, not only the
        # containing-subcatchment mask used for the canonical selection.
        all_rr, all_cc = np.nonzero(valid)
        all_global_rows = all_rr + r0
        all_global_cols = all_cc + c0
        all_xs, all_ys = rasterio.transform.xy(
            src.transform, all_global_rows, all_global_cols, offset="center"
        )
        all_xs = np.asarray(all_xs, dtype=float)
        all_ys = np.asarray(all_ys, dtype=float)
        _, _, all_distances = GEOD.inv(
            np.full(all_xs.shape, lon), np.full(all_ys.shape, lat), all_xs, all_ys
        )
        all_ids = array[all_rr, all_cc].astype(np.int64)
        all_within = all_distances <= SEARCH_RADIUS_M + 1e-6
        all_ids = all_ids[all_within]
        all_distances = all_distances[all_within]
        alternatives: list[tuple[float, int]] = []
        for segment_id in np.unique(all_ids):
            if int(segment_id) == selected_id:
                continue
            alternatives.append(
                (float(all_distances[all_ids == segment_id].min()), int(segment_id))
            )
        if alternatives:
            competitor_distance, competitor_id = min(alternatives)
        return {
            "snapped_longitude": float(xs[best_index]),
            "snapped_latitude": float(ys[best_index]),
            "geodesic_snap_distance_m": float(distances[best_index]),
            "stream_segment_id": selected_id,
            "used_containing_subcatchment": used_preferred_subcatchment,
            "nearest_competing_segment_id": competitor_id,
            "nearest_competing_distance_m": competitor_distance,
            "candidate_segment_count": int(len(np.unique(all_ids))),
        }


def canonical_table() -> pd.DataFrame:
    paths = plan_paths()
    audit = pd.read_csv(COORD_AUDIT, dtype={"site_id": "string"})
    spatial = pd.read_csv(SPATIAL_INDEX, dtype={"id": "string"}).rename(
        columns={"id": "site_id"}
    )
    keep = ["site_id", "regional_unit_id", "point_tile_id"]
    data = audit.merge(spatial[keep], on="site_id", how="left", validate="one_to_one")
    records: list[dict[str, object]] = []
    for index, row in enumerate(data.itertuples(index=False), start=1):
        lon = float(row.selected_longitude)
        lat = float(row.selected_latitude)
        tile = str(row.point_tile_id)
        sub_path = paths[("sub_catchment", tile)]
        basin_path = paths[("basin", tile)]
        segment_path = paths[("segment", tile)]
        sub_id = raster_value(sub_path, lon, lat)
        basin_id = raster_value(basin_path, lon, lat)
        snapped = nearest_stream_pixel(segment_path, lon, lat, sub_id)
        perturb_ids = []
        for dx, dy in (
            (CELL_PERTURBATION_DEG, 0.0),
            (-CELL_PERTURBATION_DEG, 0.0),
            (0.0, CELL_PERTURBATION_DEG),
            (0.0, -CELL_PERTURBATION_DEG),
        ):
            perturb_tile = point_tile(lon + dx, lat + dy)
            perturb_path = paths.get(("sub_catchment", perturb_tile), sub_path)
            perturb_ids.append(raster_value(perturb_path, lon + dx, lat + dy))
        nonzero_perturb = [value for value in perturb_ids if value > 0]
        perturb_stable = bool(nonzero_perturb) and all(value == sub_id for value in nonzero_perturb)
        distance = float(snapped["geodesic_snap_distance_m"])
        if not np.isfinite(distance) or distance > 2_000 or int(snapped["stream_segment_id"]) == 0:
            match_quality = "UPSTREAM_UNUSABLE"
        elif distance <= 500 and snapped["used_containing_subcatchment"] and perturb_stable:
            match_quality = "HIGH_CONFIDENCE_UPSTREAM"
        else:
            match_quality = "CONTEXTUAL_UPSTREAM"
        records.append(
            {
                "site_id": str(row.site_id),
                "split": str(row.split),
                "country_code": str(row.country_code),
                "us_region": str(row.us_region),
                "network": str(row.network),
                "original_longitude": lon,
                "original_latitude": lat,
                **snapped,
                "hydrography90m_regional_unit_id": int(row.regional_unit_id),
                "drainage_basin_id": basin_id,
                "subcatchment_id": sub_id,
                "perturb_east_subcatchment_id": perturb_ids[0],
                "perturb_west_subcatchment_id": perturb_ids[1],
                "perturb_north_subcatchment_id": perturb_ids[2],
                "perturb_south_subcatchment_id": perturb_ids[3],
                "one_cell_perturbation_stable": perturb_stable,
                "initial_match_quality": match_quality,
                "snap_algorithm": "nearest Hydrography90m segment-raster cell centre within containing subcatchment; 2500 m audit search",
                "snap_search_radius_m": SEARCH_RADIUS_M,
                "snap_cell_perturbation_deg": CELL_PERTURBATION_DEG,
                "outcome_blind": True,
            }
        )
        if index % 100 == 0:
            print(f"snapped {index}/{len(data)}")
    return pd.DataFrame(records).sort_values("site_id", kind="mergesort").reset_index(drop=True)


def stable_hash(table: pd.DataFrame) -> str:
    formatted = table.copy()
    numeric = formatted.select_dtypes(include=["number"]).columns
    for column in numeric:
        formatted[column] = formatted[column].map(
            lambda value: "" if pd.isna(value) else format(float(value), ".12g")
        )
    return sha256_bytes(formatted.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def write_gpkg(table: pd.DataFrame) -> None:
    gpkg = OUT / "site_snap_matches.gpkg"
    original = gpd.GeoDataFrame(
        table.drop(columns=["snapped_longitude", "snapped_latitude"]),
        geometry=[Point(xy) for xy in zip(table.original_longitude, table.original_latitude)],
        crs="EPSG:4326",
    )
    snapped_mask = table.snapped_longitude.notna() & table.snapped_latitude.notna()
    snapped = gpd.GeoDataFrame(
        table.loc[snapped_mask].drop(columns=["original_longitude", "original_latitude"]),
        geometry=[
            Point(xy)
            for xy in zip(
                table.loc[snapped_mask, "snapped_longitude"],
                table.loc[snapped_mask, "snapped_latitude"],
            )
        ],
        crs="EPSG:4326",
    )
    original.to_file(gpkg, layer="original_sites", driver="GPKG")
    snapped.to_file(gpkg, layer="snapped_sites", driver="GPKG", mode="a")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", action="store_true", help="run and compare two complete extractions")
    args = parser.parse_args()
    first = canonical_table()
    first_hash = stable_hash(first)
    repeat_hash = first_hash
    identical = True
    if args.repeat:
        second = canonical_table()
        repeat_hash = stable_hash(second)
        identical = first_hash == repeat_hash and first.equals(second)
    first.to_csv(OUT / "site_snap_matches_pre_review.csv", index=False)
    write_gpkg(first)
    audit = {
        "status": "PASS" if identical else "FAIL",
        "canonical_hash": first_hash,
        "repeat_hash": repeat_hash,
        "exact_table_equality": identical,
        "numeric_format_for_hash": ".12g",
        "floating_tolerance_m": 1e-6,
        "sites": int(len(first)),
        "outcomes_or_predictions_read": False,
    }
    (OUT / "extraction_repeatability_snap.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
