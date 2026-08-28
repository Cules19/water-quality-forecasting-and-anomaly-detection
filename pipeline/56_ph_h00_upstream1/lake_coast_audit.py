from __future__ import annotations

import hashlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "analysis_outputs/ph_h00_upstream1_v1"
SNAPS = OUT / "site_snap_matches_pre_review.csv"
HYDROLAKES_ZIP = PROJECT / "data/external/upstream1_primary/HydroLAKES_polys_v10_shp.zip"
HYDROLAKES_PATH = (
    "zip://C:/Users/xg625/Pollution_ID/data/external/upstream1_primary/"
    "HydroLAKES_polys_v10_shp.zip!HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp"
)
LAND_PATH = (
    "zip://C:/Users/xg625/Pollution_ID/data/external/upstream1_primary/"
    "ne_10m_land.zip!ne_10m_land.shp"
)
REGIONS = {
    "GB": ((-9.0, 49.0, 3.0, 61.0), "EPSG:27700"),
    "CONUS": ((-126.0, 24.0, -66.0, 50.0), "EPSG:5070"),
    "ALASKA": ((-171.0, 50.0, -129.0, 72.0), "EPSG:3338"),
    "PUERTO_RICO": ((-68.0, 17.0, -65.0, 19.0), "EPSG:32161"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lake_matches(points: gpd.GeoDataFrame, lakes: gpd.GeoDataFrame, prefix: str) -> pd.DataFrame:
    joined = gpd.sjoin(points, lakes, how="left", predicate="within")
    joined = joined.sort_values(["site_id", "Lake_area"], ascending=[True, True], kind="mergesort")
    joined = joined.drop_duplicates("site_id", keep="first").set_index("site_id")
    output = pd.DataFrame(index=points.site_id)
    output.index.name = "site_id"
    for column in ["Hylak_id", "Lake_name", "Lake_type", "Lake_area", "Pour_long", "Pour_lat"]:
        output[f"{prefix}_{column.lower()}"] = output.index.map(joined[column])
    output[f"{prefix}_lake_intersection"] = output[f"{prefix}_hylak_id"].notna()
    return output.reset_index()


def main() -> None:
    snaps = pd.read_csv(SNAPS, dtype={"site_id": "string"})
    original = gpd.GeoDataFrame(
        snaps[["site_id", "us_region"]].copy(),
        geometry=gpd.points_from_xy(snaps.original_longitude, snaps.original_latitude),
        crs="EPSG:4326",
    )
    snapped_valid = snaps.snapped_longitude.notna() & snaps.snapped_latitude.notna()
    snapped = gpd.GeoDataFrame(
        snaps.loc[snapped_valid, ["site_id", "us_region"]].copy(),
        geometry=gpd.points_from_xy(
            snaps.loc[snapped_valid, "snapped_longitude"],
            snaps.loc[snapped_valid, "snapped_latitude"],
        ),
        crs="EPSG:4326",
    )
    lake_frames = []
    for bbox, _ in REGIONS.values():
        lake_frames.append(
            pyogrio.read_dataframe(
                HYDROLAKES_PATH,
                bbox=bbox,
                columns=["Hylak_id", "Lake_name", "Lake_type", "Lake_area", "Pour_long", "Pour_lat"],
            )
        )
    lakes = gpd.GeoDataFrame(pd.concat(lake_frames, ignore_index=True), crs="EPSG:4326")
    lakes = lakes.drop_duplicates("Hylak_id")
    original_lake = lake_matches(original, lakes, "original")
    snapped_lake = lake_matches(snapped, lakes, "snapped")
    audit = snaps[
        [
            "site_id",
            "split",
            "country_code",
            "us_region",
            "hydrography90m_regional_unit_id",
            "original_longitude",
            "original_latitude",
            "snapped_longitude",
            "snapped_latitude",
        ]
    ].merge(original_lake, on="site_id", how="left", validate="one_to_one")
    audit = audit.merge(snapped_lake, on="site_id", how="left", validate="one_to_one")

    audit["distance_to_land_water_boundary_m"] = np.nan
    for region, (bbox, crs) in REGIONS.items():
        region_sites = original.loc[original.us_region.eq(region)].copy()
        if region_sites.empty:
            continue
        land = pyogrio.read_dataframe(LAND_PATH, bbox=bbox)
        land = gpd.GeoDataFrame(land, crs="EPSG:4326").to_crs(crs)
        boundaries = gpd.GeoDataFrame(geometry=land.boundary, crs=crs)
        sites_projected = region_sites.to_crs(crs)
        nearest = gpd.sjoin_nearest(
            sites_projected,
            boundaries,
            how="left",
            distance_col="distance_m",
        ).sort_values("site_id").drop_duplicates("site_id")
        distances = nearest.set_index("site_id").distance_m
        mask = audit.us_region.eq(region)
        audit.loc[mask, "distance_to_land_water_boundary_m"] = audit.loc[mask, "site_id"].map(distances)

    audit["lake_intersection_flag"] = audit.original_lake_intersection.fillna(False).astype(bool)
    audit["hydrolakes_id"] = audit.original_hylak_id
    audit["lake_outlet_handling_required"] = audit.lake_intersection_flag
    audit["lake_outlet_handling"] = np.where(
        audit.lake_intersection_flag,
        "flagged for manual review; retained Hydrography90m containing-subcatchment snap; HydroSHEDS pour point used for QA only",
        "not required",
    )
    audit["coastal_review_flag"] = (
        (audit.distance_to_land_water_boundary_m <= 5_000)
        | audit.hydrography90m_regional_unit_id.eq(0)
    )
    audit["estuarine_tidal_review_flag"] = audit.coastal_review_flag
    audit["island_edge_review_flag"] = audit.coastal_review_flag & audit.us_region.isin(["GB", "PUERTO_RICO", "ALASKA"])
    audit["hydrolakes_version"] = "1.0 (2016 technical documentation)"
    audit["hydrolakes_sha256"] = sha256(HYDROLAKES_ZIP)
    audit["hydrolakes_license"] = "CC BY 4.0"
    audit["natural_earth_land_version"] = "5.1.1"
    audit["outcomes_or_predictions_read"] = False
    audit.to_csv(OUT / "lake_coast_estuary_audit.csv", index=False)
    print(
        {
            "lake_intersections": int(audit.lake_intersection_flag.sum()),
            "coastal_review": int(audit.coastal_review_flag.sum()),
            "estuarine_tidal_review": int(audit.estuarine_tidal_review_flag.sum()),
        }
    )


if __name__ == "__main__":
    main()
