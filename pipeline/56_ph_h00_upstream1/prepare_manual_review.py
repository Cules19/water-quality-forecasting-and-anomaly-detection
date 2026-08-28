from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.windows import Window


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "analysis_outputs/ph_h00_upstream1_v1"
SNAPS = OUT / "site_snap_matches.csv"
LAKE = OUT / "lake_coast_estuary_audit.csv"
USGS = OUT / "usgs_site_metadata_qa.csv"
NAMES = OUT / "project_site_name_spatial_qa.csv"
PLAN = OUT / "primary_download_plan.csv"
CAMELS = PROJECT / (
    "analysis_outputs/camels_gb_v2_site_matching_audit_v1/"
    "matching/site_camels_primary_mapping.csv"
)
PAGES = OUT / "manual_review_maps"
RANDOM_SEED = 42
RANDOM_PER_COUNTRY = 25
NEARBY_COMPETITOR_M = 500.0
NAME_PATTERN = re.compile(
    r"LAKE|LOCH|RESERVOIR|POND|ESTUAR|TIDAL|SEA|OCEAN|COAST|CANAL|MARSH|"
    r"BAY|HARBOU?R|DOCK|GROUNDWATER|WELL|SPRING|SLUICE|DRAIN",
    re.IGNORECASE,
)


def truth(value: object) -> bool:
    return bool(value) and not pd.isna(value)


def join_reason(row: pd.Series, columns: list[tuple[str, str]]) -> str:
    return "|".join(label for column, label in columns if truth(row.get(column, False)))


def build_manifest() -> pd.DataFrame:
    snaps = pd.read_csv(SNAPS, dtype={"site_id": "string"})
    lake = pd.read_csv(LAKE, dtype={"site_id": "string"})
    names = pd.read_csv(NAMES, dtype={"site_id": "string"})
    usgs = pd.read_csv(USGS, dtype={"site_id": "string"})
    camels = pd.read_csv(CAMELS, dtype={"site_id": "string"})
    keep_lake = [
        "site_id",
        "lake_intersection_flag",
        "coastal_review_flag",
        "estuarine_tidal_review_flag",
        "island_edge_review_flag",
        "distance_to_land_water_boundary_m",
        "hydrolakes_id",
        "original_pour_long",
        "original_pour_lat",
    ]
    keep_names = ["site_id", "project_source", "project_waterbody_name", "project_river_name"]
    keep_usgs = ["site_id", "station_nm", "site_tp_cd", "drainage_area_km2"]
    table = snaps.merge(lake[keep_lake], on="site_id", how="left", validate="one_to_one")
    table = table.merge(names[keep_names], on="site_id", how="left", validate="one_to_one")
    table = table.merge(usgs[keep_usgs], on="site_id", how="left", validate="one_to_one")
    colocated = set(
        camels.loc[camels.match_category.eq("COLOCATED_GAUGE_MATCH"), "site_id"].astype(str)
    )

    table["review_distance_gt_500m"] = table.geodesic_snap_distance_m.gt(500)
    table["review_lake"] = table.lake_intersection_flag.fillna(False).astype(bool)
    table["review_coast_estuary_tidal_island"] = (
        table[
            [
                "coastal_review_flag",
                "estuarine_tidal_review_flag",
                "island_edge_review_flag",
            ]
        ]
        .fillna(False)
        .astype(bool)
        .any(axis=1)
    )
    table["review_material_competing_reach"] = (
        table.material_alternative_area.fillna(False).astype(bool)
        & table.nearest_competing_distance_m.le(NEARBY_COMPETITOR_M)
    )
    table["review_material_perturbation_path_change"] = table.get(
        "one_cell_perturbation_material_change", False
    )
    table["review_alaska"] = table.us_region.eq("ALASKA")
    usable = table.match_quality.ne("UPSTREAM_UNUSABLE")
    area_low, area_high = table.loc[usable, "upstream_area_km2"].quantile([0.01, 0.99])
    table["review_extreme_area"] = usable & (
        table.upstream_area_km2.le(area_low) | table.upstream_area_km2.ge(area_high)
    )
    if "order_strahler" in table:
        log_area = np.log1p(table.upstream_area_km2)
        expected = table.groupby("order_strahler")["upstream_area_km2"].transform(
            lambda values: np.nanmedian(np.log1p(values))
        )
        mismatch = log_area - expected
        low, high = mismatch[usable & mismatch.notna()].quantile([0.01, 0.99])
        table["area_order_log_residual"] = mismatch
        table["review_implausible_area_order"] = usable & (
            mismatch.le(low) | mismatch.ge(high)
        )
    else:
        table["area_order_log_residual"] = np.nan
        table["review_implausible_area_order"] = False
    text = (
        table.project_waterbody_name.fillna("").astype(str)
        + " | "
        + table.project_river_name.fillna("").astype(str)
        + " | "
        + table.station_nm.fillna("").astype(str)
    )
    table["review_name_or_site_type"] = text.str.contains(NAME_PATTERN) | (
        table.site_tp_cd.notna() & ~table.site_tp_cd.eq("ST")
    )
    table["review_camels_colocated"] = table.site_id.isin(colocated)

    rng = np.random.default_rng(RANDOM_SEED)
    table["review_random_high_confidence"] = False
    for country in ["GB", "US"]:
        candidates = table.index[
            table.country_code.eq(country)
            & table.match_quality.eq("HIGH_CONFIDENCE_UPSTREAM")
        ].to_numpy()
        chosen = rng.choice(candidates, size=RANDOM_PER_COUNTRY, replace=False)
        table.loc[chosen, "review_random_high_confidence"] = True

    reason_columns = [
        ("review_distance_gt_500m", "SNAP_DISTANCE_GT_500M"),
        ("review_lake", "HYDROLAKES_INTERSECTION"),
        ("review_coast_estuary_tidal_island", "COAST_ESTUARY_TIDAL_ISLAND"),
        ("review_material_competing_reach", "MATERIAL_NEARBY_COMPETING_REACH"),
        ("review_material_perturbation_path_change", "ONE_CELL_DIFFERENT_PATH_FACTOR_GT2"),
        ("review_alaska", "ALASKA"),
        ("review_extreme_area", "EXTREME_AREA_1PCT"),
        ("review_implausible_area_order", "EXTREME_AREA_ORDER_COMBINATION"),
        ("review_name_or_site_type", "NAME_OR_SITE_TYPE_QA"),
        ("review_camels_colocated", "CAMELS_COLOCATED_QA"),
        ("review_random_high_confidence", "RANDOM_HIGH_CONFIDENCE_SEED42"),
    ]
    table["review_reason_codes"] = table.apply(lambda row: join_reason(row, reason_columns), axis=1)
    manifest = table.loc[table.review_reason_codes.ne("")].copy()
    manifest["random_seed"] = RANDOM_SEED
    manifest["nearby_competitor_threshold_m"] = NEARBY_COMPETITOR_M
    manifest["review_frozen_before_outcomes"] = True
    manifest["outcomes_or_predictions_read"] = False
    manifest = manifest.sort_values("site_id", kind="mergesort").reset_index(drop=True)
    manifest.to_csv(OUT / "manual_review_manifest.csv", index=False)
    criteria = {
        "random_seed": RANDOM_SEED,
        "random_high_confidence_sites_per_country": RANDOM_PER_COUNTRY,
        "nearby_competing_reach_threshold_m": NEARBY_COMPETITOR_M,
        "extreme_area_quantiles": [0.01, 0.99],
        "area_order_outlier_quantiles": [0.01, 0.99],
        "canonical_selection": "containing-subcatchment Hydrography90m segment",
        "high_confidence_max_distance_m": 500,
        "contextual_max_distance_m": 2000,
        "unusable_above_distance_m": 2000,
        "material_area_ratio": 2.0,
        "outcome_blind": True,
    }
    (OUT / "snap_quality_criteria.json").write_text(
        json.dumps(criteria, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def render_pages(manifest: pd.DataFrame) -> None:
    PAGES.mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv(PLAN)
    segment_paths = {
        str(row.tile): Path(str(row.destination))
        for row in plan.loc[plan.variable.eq("segment")].itertuples(index=False)
    }
    for page_start in range(0, len(manifest), 16):
        page = manifest.iloc[page_start : page_start + 16]
        fig, axes = plt.subplots(4, 4, figsize=(20, 20), constrained_layout=True)
        for ax, row in zip(axes.ravel(), page.itertuples(index=False)):
            lon, lat = float(row.original_longitude), float(row.original_latitude)
            path = segment_paths[str(row.network)] if str(row.network) in segment_paths else None
            if path is None:
                # network records the point tile in the current snap table.
                point_tile = "h%02dv%02d" % (
                    int(math.floor((lon + 180.0) / 20.0)) * 2,
                    int(math.floor((85.0 - lat) / 20.0)) * 2,
                )
                path = segment_paths[point_tile]
            with rasterio.open(path) as src:
                center_row, center_col = src.index(lon, lat)
                radius_m = max(3000.0, float(row.geodesic_snap_distance_m or 0) + 1000.0)
                lat_cells = max(25, int(math.ceil(radius_m / 111_320.0 / abs(src.transform.e))))
                lon_cells = max(
                    25,
                    int(
                        math.ceil(
                            radius_m
                            / (111_320.0 * max(0.08, math.cos(math.radians(lat))))
                            / abs(src.transform.a)
                        )
                    ),
                )
                r0, r1 = max(0, center_row - lat_cells), min(src.height, center_row + lat_cells + 1)
                c0, c1 = max(0, center_col - lon_cells), min(src.width, center_col + lon_cells + 1)
                window = Window(c0, r0, c1 - c0, r1 - r0)
                array = src.read(1, window=window)
                transform = src.window_transform(window)
                extent = (
                    transform.c,
                    transform.c + transform.a * array.shape[1],
                    transform.f + transform.e * array.shape[0],
                    transform.f,
                )
            category = np.zeros_like(array, dtype=np.uint8)
            category[array > 0] = 1
            category[array == int(row.stream_segment_id)] = 2
            if int(row.nearest_competing_segment_id) > 0:
                category[array == int(row.nearest_competing_segment_id)] = 3
            ax.imshow(
                category,
                extent=extent,
                origin="upper",
                interpolation="nearest",
                cmap=ListedColormap(["#f7f7f7", "#9e9e9e", "#006d2c", "#f28e2b"]),
                vmin=0,
                vmax=3,
            )
            ax.scatter(lon, lat, marker="x", s=70, linewidths=2, color="#d62728", label="original")
            if np.isfinite(row.snapped_longitude):
                ax.scatter(
                    row.snapped_longitude,
                    row.snapped_latitude,
                    marker="o",
                    s=45,
                    facecolor="#4e79a7",
                    edgecolor="white",
                    linewidth=0.8,
                    label="snapped",
                )
                ax.plot(
                    [lon, row.snapped_longitude],
                    [lat, row.snapped_latitude],
                    color="#d62728",
                    linewidth=1,
                )
            if np.isfinite(row.original_pour_long) and np.isfinite(row.original_pour_lat):
                ax.scatter(
                    row.original_pour_long,
                    row.original_pour_lat,
                    marker="*",
                    s=90,
                    color="#9467bd",
                    label="HydroLAKES pour",
                )
            name = row.station_nm if isinstance(row.station_nm, str) else row.project_waterbody_name
            name = "" if pd.isna(name) else str(name)
            subtitle = (
                f"{row.site_id} | {row.country_code}/{row.split} | {row.match_quality}\n"
                f"snap={row.geodesic_snap_distance_m:.1f} m; area={row.upstream_area_km2:.2f} km²; "
                f"alt={row.alternative_area_ratio:.2f}x @ {row.nearest_competing_distance_m:.1f} m\n"
                f"{name[:75]}\n{row.review_reason_codes[:110]}"
            )
            ax.set_title(subtitle, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        for ax in axes.ravel()[len(page) :]:
            ax.axis("off")
        page_number = page_start // 16 + 1
        output = PAGES / f"manual_review_page_{page_number:02d}.png"
        fig.savefig(output, dpi=180, facecolor="white")
        plt.close(fig)
        print(f"rendered {output.name}", flush=True)


def main() -> None:
    manifest = build_manifest()
    render_pages(manifest)
    print(f"manual review sites: {len(manifest)}")


if __name__ == "__main__":
    main()
