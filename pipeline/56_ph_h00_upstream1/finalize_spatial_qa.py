from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "analysis_outputs/ph_h00_upstream1_v1"
CAMELS = PROJECT / "analysis_outputs/camels_gb_v2_site_matching_audit_v1"

UNUSABLE = "UPSTREAM_UNUSABLE"
CONTEXTUAL = "CONTEXTUAL_UPSTREAM"
HIGH = "HIGH_CONFIDENCE_UPSTREAM"

# Frozen after inspecting all 29 outcome-blind map pages.  Each station name
# denotes a material river/confluence/weir context that is irreconcilable with
# the sub-0.11 km2 first-order headwater returned by the selected network cell.
# Small explicitly named pools, ponds, minor tributaries, and drains are not in
# this list merely because their derived area is small.
MANUAL_NAME_AREA_CONFLICTS = {
    "E00532A": "Roden at Thistleford Bridge cannot have a 0.083 km2 contributing area",
    "E01572A": "Great Ouse/Ouse Washes at Denver fish pass cannot have a 0.100 km2 contributing area",
    "E02515A": "River Avon at Warleigh Weir cannot have a 0.063 km2 contributing area",
    "E03380A": "River Ver downstream of a confluence cannot have a 0.084 km2 contributing area",
    "E04313A": "Derwent downstream of Tendley Quarry cannot have a 0.091 km2 contributing area",
    "E06073A": "River Tean downstream of Checkley STW cannot have a 0.091 km2 contributing area",
    "E07281A": "River Lark downstream of Fornham STW cannot have a 0.063 km2 contributing area",
    "E08085A": "River Tone at Athelney cannot have a 0.089 km2 contributing area",
}


def discrepancy_factor(derived: pd.Series, reference: pd.Series) -> pd.Series:
    ratio = derived / reference
    return pd.Series(np.maximum(ratio, 1.0 / ratio), index=derived.index)


def build_independent_checks(
    snaps: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    checks: list[dict[str, object]] = []
    conflicts: dict[str, list[str]] = {}

    usgs = pd.read_csv(OUT / "usgs_site_metadata_qa.csv", dtype={"site_id": "string"})
    usgs = usgs.merge(
        snaps[["site_id", "split", "upstream_area_km2"]],
        on="site_id",
        how="inner",
        validate="one_to_one",
    )
    for row in usgs.itertuples(index=False):
        coordinate_difference = float(row.coordinate_difference_m)
        coordinate_status = "PASS" if coordinate_difference <= 500.0 else "REVIEW"
        checks.append(
            {
                "check_family": "USGS_OFFICIAL_COORDINATE",
                "reference_source": "USGS NWIS site service",
                "site_id": row.site_id,
                "split": row.split,
                "metric": "coordinate_difference_m",
                "derived_value": 0.0,
                "reference_value": coordinate_difference,
                "ratio": np.nan,
                "discrepancy_factor": np.nan,
                "pass_or_review_threshold": "PASS if official/project coordinate difference <=500 m",
                "status": coordinate_status,
                "station_or_gauge_name": row.station_nm,
                "qa_only_not_model_input": True,
                "outcomes_or_predictions_read": False,
            }
        )

        reference_area = row.contributing_drainage_area_km2
        if not np.isfinite(reference_area):
            reference_area = row.drainage_area_km2
        if np.isfinite(reference_area) and reference_area > 0 and row.upstream_area_km2 > 0:
            ratio = float(row.upstream_area_km2 / reference_area)
            factor = float(max(ratio, 1.0 / ratio))
            if factor <= 2.0:
                status = "PASS_WITHIN_FACTOR_2"
            elif factor <= 10.0:
                status = "REVIEW_FACTOR_2_TO_10"
            else:
                status = "FAIL_IRRECONCILABLE_FACTOR_GT10"
                conflicts.setdefault(str(row.site_id), []).append(
                    f"USGS official drainage-area discrepancy factor {factor:.3f} >10"
                )
            checks.append(
                {
                    "check_family": "USGS_OFFICIAL_DRAINAGE_AREA",
                    "reference_source": "USGS NWIS site service",
                    "site_id": row.site_id,
                    "split": row.split,
                    "metric": "upstream_area_km2",
                    "derived_value": float(row.upstream_area_km2),
                    "reference_value": float(reference_area),
                    "ratio": ratio,
                    "discrepancy_factor": factor,
                    "pass_or_review_threshold": "pass <=2; review >2 to 10; irreconcilable >10",
                    "status": status,
                    "station_or_gauge_name": row.station_nm,
                    "qa_only_not_model_input": True,
                    "outcomes_or_predictions_read": False,
                }
            )

    mapping = pd.read_csv(
        CAMELS / "matching/site_camels_primary_mapping.csv",
        dtype={"site_id": "string", "primary_gauge_id": "string"},
    )
    mapping = mapping.loc[
        mapping.model_readiness.eq("MODEL_READY_HIGH_CONFIDENCE")
    ].copy()
    attributes = pd.read_csv(
        CAMELS / "attributes/camels_attributes_merged.csv",
        dtype={"gauge_id": "string"},
    )[["gauge_id", "area"]]
    camels = (
        mapping.merge(
            attributes,
            left_on="primary_gauge_id",
            right_on="gauge_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            snaps[["site_id", "split", "upstream_area_km2"]],
            on="site_id",
            how="inner",
            validate="one_to_one",
        )
    )
    if len(camels) != 26:
        raise RuntimeError(f"Expected 26 CAMELS gauge-coincident sites, found {len(camels)}")
    for row in camels.itertuples(index=False):
        ratio = float(row.upstream_area_km2 / row.area)
        factor = float(max(ratio, 1.0 / ratio))
        if factor <= 2.0:
            status = "PASS_WITHIN_FACTOR_2"
        elif factor <= 5.0:
            status = "REVIEW_FACTOR_2_TO_5"
        else:
            status = "FAIL_IRRECONCILABLE_FACTOR_GT5"
            conflicts.setdefault(str(row.site_id), []).append(
                f"CAMELS-GB gauge-coincident drainage-area discrepancy factor {factor:.3f} >5"
            )
        checks.append(
            {
                "check_family": "CAMELS_GB_GAUGE_COINCIDENT_AREA",
                "reference_source": "CAMELS-GB v2 topographic attributes",
                "site_id": row.site_id,
                "split": row.split_x,
                "metric": "upstream_area_km2",
                "derived_value": float(row.upstream_area_km2),
                "reference_value": float(row.area),
                "ratio": ratio,
                "discrepancy_factor": factor,
                "pass_or_review_threshold": "pass <=2; review >2 to 5; irreconcilable >5 for gauge-coincident sites",
                "status": status,
                "station_or_gauge_name": row.primary_gauge_name,
                "qa_only_not_model_input": True,
                "outcomes_or_predictions_read": False,
            }
        )

    return pd.DataFrame(checks), conflicts


def update_snap_geopackage(snaps: pd.DataFrame) -> None:
    source = OUT / "site_snap_matches.gpkg"
    temporary = OUT / "site_snap_matches.final.tmp.gpkg"
    if temporary.exists():
        temporary.unlink()
    attributes = snaps.set_index("site_id")[[
        "pre_final_match_quality",
        "match_quality",
        "final_quality_reason",
        "manual_review_completed",
        "independent_spatial_conflict",
    ]]
    for number, layer in enumerate(["original_sites", "snapped_sites"]):
        frame = gpd.read_file(source, layer=layer)
        frame["site_id"] = frame.site_id.astype("string")
        for column in attributes.columns:
            frame[column] = frame.site_id.map(attributes[column])
        frame.to_file(
            temporary,
            layer=layer,
            driver="GPKG",
            mode="w" if number == 0 else "a",
            index=False,
        )
    os.replace(temporary, source)


def main() -> None:
    snaps = pd.read_csv(
        OUT / "site_snap_matches_pre_review.csv", dtype={"site_id": "string"}
    )
    current = pd.read_csv(OUT / "site_snap_matches.csv", dtype={"site_id": "string"})
    if "pre_final_match_quality" in current.columns:
        automatic_quality = current.set_index("site_id").pre_final_match_quality.astype(
            "string"
        )
    else:
        automatic_quality = current.set_index("site_id").match_quality.astype("string")
    final_columns = [
        column
        for column in current.columns
        if column
        not in {
            "pre_final_match_quality",
            "final_quality_reason",
            "manual_review_completed",
            "independent_spatial_conflict",
        }
    ]
    current = current[final_columns]
    if len(snaps) != 910 or snaps.site_id.nunique() != 910:
        raise RuntimeError("Pre-review snap table must contain 910 unique sites")

    independent_checks, independent_conflicts = build_independent_checks(current)
    for site_id, description in MANUAL_NAME_AREA_CONFLICTS.items():
        independent_conflicts.setdefault(site_id, []).append(
            "outcome-blind manual site-name/derived-area contradiction: " + description
        )

    quality = automatic_quality.copy()
    reasons: dict[str, str] = {}
    for site_id, initial in quality.items():
        reasons[str(site_id)] = (
            "automatic invalid-distance/area/network rule"
            if initial == UNUSABLE
            else "retained after required outcome-blind spatial review"
        )
    for site_id, conflict_reasons in independent_conflicts.items():
        if site_id not in quality.index:
            raise RuntimeError(f"Spatial conflict references unknown site {site_id}")
        quality.loc[site_id] = UNUSABLE
        reasons[site_id] = " | ".join(conflict_reasons)

    # Canal metadata is a known non-natural-waterbody limitation.  It remains
    # usable only as contextual upstream information, never high confidence.
    if "08066850" in quality.index and quality.loc["08066850"] != UNUSABLE:
        quality.loc["08066850"] = CONTEXTUAL
        reasons["08066850"] = "canal/site-type metadata; retained as contextual only"

    manifest = pd.read_csv(
        OUT / "manual_review_manifest.csv", dtype={"site_id": "string"}
    ).reset_index(drop=True)
    if manifest.site_id.duplicated().any():
        raise RuntimeError("Duplicate manual-review site")
    reviewed_sites = set(manifest.site_id)
    if len(manifest) != 463:
        raise RuntimeError(f"Expected 463 required manual reviews, found {len(manifest)}")

    overrides = pd.DataFrame(
        {
            "site_id": current.site_id,
            "final_match_quality": current.site_id.map(quality),
            "final_quality_reason": current.site_id.map(reasons),
            "manual_review_completed": current.site_id.isin(reviewed_sites),
            "independent_spatial_conflict": current.site_id.isin(independent_conflicts),
            "frozen_before_outcomes": True,
            "outcomes_or_predictions_read": False,
        }
    )
    if overrides.final_match_quality.isna().any():
        raise RuntimeError("Missing final match-quality override")
    overrides.to_csv(OUT / "final_match_quality_overrides.csv", index=False)

    final = current.copy()
    final["pre_final_match_quality"] = final.site_id.map(automatic_quality)
    final["match_quality"] = final.site_id.map(quality)
    final["final_quality_reason"] = final.site_id.map(reasons)
    final["manual_review_completed"] = final.site_id.isin(reviewed_sites)
    final["independent_spatial_conflict"] = final.site_id.isin(independent_conflicts)
    final.to_csv(OUT / "site_snap_matches.csv", index=False)

    lake_sites = set(
        manifest.loc[manifest.lake_intersection_flag.fillna(False), "site_id"]
    )
    lake_resolution = {
        "01304057": "RESOLVED_CONTEXTUAL_LAKE_OR_POND_OUTLET_PROXY",
        "04236000": "RESOLVED_NAMED_LAKE_OUTLET_AND_POUR_POINT",
        "E05043A": "RESOLVED_WITH_EXPLICIT_HYDROLAKES_INTERSECTION_FLAG",
    }
    if lake_sites != set(lake_resolution):
        raise RuntimeError(f"Unexpected HydroLAKES review set: {sorted(lake_sites)}")

    results = manifest[[
        "site_id",
        "split",
        "country_code",
        "review_reason_codes",
        "match_quality",
        "project_waterbody_name",
        "project_river_name",
        "station_nm",
        "upstream_area_km2",
        "geodesic_snap_distance_m",
        "lake_intersection_flag",
        "coastal_review_flag",
        "estuarine_tidal_review_flag",
        "island_edge_review_flag",
    ]].copy()
    results = results.rename(columns={"match_quality": "pre_review_match_quality"})
    results["manual_review_page"] = np.arange(len(results)) // 16 + 1
    results["visual_review_completed"] = True
    results["selected_reach_plausible"] = results.site_id.map(quality).ne(UNUSABLE)
    results["site_name_or_reference_area_contradiction"] = results.site_id.isin(
        independent_conflicts
    )
    results["lake_outlet_handling"] = results.site_id.map(lake_resolution).fillna(
        "NOT_APPLICABLE"
    )
    results["manual_decision"] = np.where(
        results.site_id.map(quality).eq(UNUSABLE),
        "DEMOTE_TO_UPSTREAM_UNUSABLE",
        np.where(
            results.site_id.map(quality).eq(CONTEXTUAL),
            "RETAIN_CONTEXTUAL_WITH_FLAGS",
            "RETAIN_HIGH_CONFIDENCE",
        ),
    )
    results["final_match_quality"] = results.site_id.map(quality)
    results["review_notes"] = results.site_id.map(reasons)
    results["review_seed"] = 42
    results["outcome_blind"] = True
    results["outcomes_or_predictions_read"] = False
    results.to_csv(OUT / "manual_review_results.csv", index=False)

    independent_checks["final_match_quality"] = independent_checks.site_id.map(
        quality
    )
    independent_checks.to_csv(OUT / "independent_spatial_checks.csv", index=False)

    for name in ["upstream13_unscaled.csv", "upstream13_model_table.csv"]:
        path = OUT / name
        if not path.exists():
            continue
        table = pd.read_csv(path, dtype={"site_id": "string"})
        table["match_quality"] = table.site_id.map(quality)
        table["usable_upstream"] = table.match_quality.ne(UNUSABLE)
        table.to_csv(path, index=False)

    update_snap_geopackage(final)
    counts = final.match_quality.value_counts().to_dict()
    print(
        {
            "manual_reviews": len(results),
            "independent_checks": len(independent_checks),
            "independent_or_manual_conflicts": len(independent_conflicts),
            "final_match_quality": counts,
        }
    )


if __name__ == "__main__":
    main()
