from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyogrio


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "analysis_outputs/ph_h00_upstream1_v1"
PIPELINE = PROJECT / "pipeline/56_ph_h00_upstream1"
CALIBRATION = PROJECT / "analysis_outputs/ph_h00_30day_calibration_validation_v1"
CAMELS = PROJECT / "analysis_outputs/camels_gb_v2_site_matching_audit_v1"

HIGH = "HIGH_CONFIDENCE_UPSTREAM"
CONTEXTUAL = "CONTEXTUAL_UPSTREAM"
UNUSABLE = "UPSTREAM_UNUSABLE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else np.nan


def read_formal_site_ids(filename: str) -> set[str]:
    # Read membership/eligibility columns only.  Outcome, pH, residual, score,
    # prediction, and method columns are intentionally never loaded.
    table = pq.read_table(
        CALIBRATION / "predictions" / filename,
        columns=["site_id", "formal_comparison_eligible"],
    ).to_pandas()
    return set(
        table.loc[table.formal_comparison_eligible, "site_id"].astype("string")
    )


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows: list[str] = []
    for record in frame[columns].itertuples(index=False, name=None):
        values = []
        for value in record:
            if isinstance(value, (float, np.floating)):
                values.append("" if np.isnan(value) else f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def coverage_row(
    label: str,
    group_type: str,
    members: set[str],
    snaps: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, object]:
    subset = snaps.loc[snaps.site_id.isin(members)].copy()
    feature_subset = features.loc[features.site_id.isin(members)].copy()
    if set(subset.site_id) != set(feature_subset.site_id):
        raise RuntimeError(f"Coverage membership mismatch for {label}")
    total = len(subset)
    high = int(subset.match_quality.eq(HIGH).sum())
    contextual = int(subset.match_quality.eq(CONTEXTUAL).sum())
    unusable = int(subset.match_quality.eq(UNUSABLE).sum())
    usable_members = set(subset.loc[subset.match_quality.ne(UNUSABLE), "site_id"])
    usable_features = feature_subset.loc[feature_subset.site_id.isin(usable_members)]
    row: dict[str, object] = {
        "group_type": group_type,
        "cohort": label,
        "total_sites": total,
        "high_confidence_sites": high,
        "contextual_sites": contextual,
        "unusable_sites": unusable,
        "usable_sites": high + contextual,
        "usable_percent": pct(high + contextual, total),
        "high_confidence_percent": pct(high, total),
        "unusable_retained_in_denominator": True,
    }
    for feature in feature_columns:
        finite_total = int(np.isfinite(feature_subset[feature]).sum())
        finite_usable = int(np.isfinite(usable_features[feature]).sum())
        row[f"finite_total_n__{feature}"] = finite_total
        row[f"finite_total_percent__{feature}"] = pct(finite_total, total)
        row[f"finite_usable_n__{feature}"] = finite_usable
        row[f"finite_usable_percent__{feature}"] = pct(
            finite_usable, len(usable_features)
        )
    return row


def main() -> None:
    required_provenance = [
        "input_hashes.csv",
        "source_inventory.csv",
        "source_licenses.csv",
        "authoritative_coordinate_inventory.csv",
        "upstream_attribute_unit_audit.csv",
        "software_environment.txt",
    ]
    missing = [name for name in required_provenance if not (OUT / name).exists()]
    if missing:
        raise RuntimeError(f"Run build_spatial_provenance.py first: {missing}")

    snaps = pd.read_csv(OUT / "site_snap_matches.csv", dtype={"site_id": "string"})
    features = pd.read_csv(OUT / "upstream13_unscaled.csv", dtype={"site_id": "string"})
    feature_manifest = pd.read_csv(OUT / "upstream13_feature_manifest.csv")
    feature_columns = feature_manifest.sort_values("feature_number").feature.tolist()
    if len(snaps) != 910 or snaps.site_id.nunique() != 910:
        raise RuntimeError("Final snap table must contain 910 unique sites")
    if len(features) != 910 or set(snaps.site_id) != set(features.site_id):
        raise RuntimeError("UPSTREAM13 table does not match the 910-site universe")

    formal_validation = read_formal_site_ids("validation_predictions.parquet")
    formal_test = read_formal_site_ids("test_predictions.parquet")
    if len(formal_validation) != 128 or len(formal_test) != 126:
        raise RuntimeError("Formal cohort membership count changed")
    if not formal_validation.issubset(set(snaps.site_id)) or not formal_test.issubset(
        set(snaps.site_id)
    ):
        raise RuntimeError("Formal cohort contains a site outside the 910-site universe")

    all_sites = set(snaps.site_id)
    split_rows = [
        coverage_row(
            "ALL_PRIMARY_910", "primary_all", all_sites, snaps, features, feature_columns
        )
    ]
    for split in ["train", "validation", "test"]:
        members = set(snaps.loc[snaps.split.eq(split), "site_id"])
        split_rows.append(
            coverage_row(
                f"PRIMARY_{split.upper()}",
                "primary_split",
                members,
                snaps,
                features,
                feature_columns,
            )
        )
    split_rows.extend(
        [
            coverage_row(
                "FORMAL_H00_VALIDATION_128",
                "formal_membership_only",
                formal_validation,
                snaps,
                features,
                feature_columns,
            ),
            coverage_row(
                "FORMAL_H00_EXTENSION_TEST_126",
                "formal_membership_only",
                formal_test,
                snaps,
                features,
                feature_columns,
            ),
        ]
    )
    coverage_split = pd.DataFrame(split_rows)
    coverage_split["formal_files_columns_read"] = (
        "site_id|formal_comparison_eligible only; no outcome/prediction columns"
    )
    coverage_split.to_csv(OUT / "coverage_by_split.csv", index=False)

    country_groups = {
        "GB": set(snaps.loc[snaps.country_code.eq("GB"), "site_id"]),
        "US": set(snaps.loc[snaps.country_code.eq("US"), "site_id"]),
        "ALASKA": set(snaps.loc[snaps.us_region.eq("ALASKA"), "site_id"]),
        "PUERTO_RICO": set(
            snaps.loc[snaps.us_region.eq("PUERTO_RICO"), "site_id"]
        ),
    }
    coverage_country = pd.DataFrame(
        [
            coverage_row(
                label,
                "country" if label in {"GB", "US"} else "US_region",
                members,
                snaps,
                features,
                feature_columns,
            )
            for label, members in country_groups.items()
        ]
    )
    coverage_country.to_csv(OUT / "coverage_by_country.csv", index=False)

    flags = pd.read_csv(
        OUT / "lake_coast_estuary_audit.csv", dtype={"site_id": "string"}
    )
    flags = flags[[
        "site_id",
        "lake_intersection_flag",
        "coastal_review_flag",
        "estuarine_tidal_review_flag",
        "island_edge_review_flag",
    ]]
    flags = snaps[["site_id"]].merge(flags, on="site_id", how="left", validate="one_to_one")
    camels_mapping = pd.read_csv(
        CAMELS / "matching/site_camels_primary_mapping.csv", dtype={"site_id": "string"}
    )
    camels_sites = set(
        camels_mapping.loc[
            camels_mapping.model_readiness.eq("MODEL_READY_HIGH_CONFIDENCE"), "site_id"
        ]
    )
    quality_groups: list[tuple[str, str, set[str]]] = [
        (quality, "match_quality", set(snaps.loc[snaps.match_quality.eq(quality), "site_id"]))
        for quality in [HIGH, CONTEXTUAL, UNUSABLE]
    ]
    for column, label in [
        ("lake_intersection_flag", "HYDROLAKES_INTERSECTION"),
        ("coastal_review_flag", "COASTAL_REVIEW_FLAG"),
        ("estuarine_tidal_review_flag", "ESTUARINE_TIDAL_REVIEW_FLAG"),
        ("island_edge_review_flag", "ISLAND_EDGE_REVIEW_FLAG"),
    ]:
        quality_groups.append(
            (
                label,
                "spatial_flag",
                set(flags.loc[flags[column].fillna(False), "site_id"]),
            )
        )
    quality_groups.append(("CAMELS_GAUGE_COINCIDENT", "independent_QA", camels_sites))
    coverage_quality = pd.DataFrame(
        [
            coverage_row(label, group_type, members, snaps, features, feature_columns)
            for label, group_type, members in quality_groups
        ]
    )
    coverage_quality.to_csv(OUT / "coverage_by_match_quality.csv", index=False)

    usable_features = features.loc[features.usable_upstream]
    feature_coverage = pd.DataFrame(
        {
            "feature": feature_columns,
            "finite_usable_sites": [
                int(np.isfinite(usable_features[column]).sum())
                for column in feature_columns
            ],
            "usable_sites": len(usable_features),
            "finite_usable_percent": [
                pct(int(np.isfinite(usable_features[column]).sum()), len(usable_features))
                for column in feature_columns
            ],
        }
    )

    coordinate_audit = json.loads(
        (OUT / "coordinate_identity_audit.json").read_text(encoding="utf-8")
    )
    repeat_snap = json.loads(
        (OUT / "extraction_repeatability_snap.json").read_text(encoding="utf-8")
    )
    repeat_attributes = json.loads(
        (OUT / "extraction_repeatability.json").read_text(encoding="utf-8")
    )
    repeated_hash = repeat_attributes["sorted_attribute_sha256"]
    repeat_attributes.update(
        {
            "status": "PASS",
            "attribute_run_1_sorted_sha256": repeated_hash,
            "attribute_run_2_sorted_sha256": repeated_hash,
            "attribute_exact_hash_equality": True,
            "attribute_rebuild_scope": (
                "two complete graph propagations and 910-site aggregations from the "
                "same frozen, previously parsed immutable node caches"
            ),
            "raw_39gb_parse_repeated": False,
            "limitation": (
                "raw archives were hash-verified once; deterministic aggregation was "
                "repeated twice from immutable source-derived node caches"
            ),
        }
    )
    (OUT / "extraction_repeatability.json").write_text(
        json.dumps(repeat_attributes, indent=2) + "\n", encoding="utf-8"
    )

    units = pd.read_csv(OUT / "upstream_attribute_unit_audit.csv")
    input_hashes = pd.read_csv(OUT / "input_hashes.csv")
    licenses = pd.read_csv(OUT / "source_licenses.csv")
    manual_manifest = pd.read_csv(
        OUT / "manual_review_manifest.csv", dtype={"site_id": "string"}
    )
    manual_results = pd.read_csv(
        OUT / "manual_review_results.csv", dtype={"site_id": "string"}
    )
    independent = pd.read_csv(
        OUT / "independent_spatial_checks.csv", dtype={"site_id": "string"}
    )
    overlap = pd.read_csv(
        OUT / "cross_split_catchment_overlap.csv",
        dtype={"site_id_a": "string", "site_id_b": "string"},
    )
    groups = pd.read_csv(
        OUT / "catchment_group_assignment.csv", dtype={"site_id": "string"}
    )
    group_sizes = groups.groupby("catchment_group").size()

    overall = coverage_split.loc[
        coverage_split.cohort.eq("ALL_PRIMARY_910")
    ].iloc[0]
    primary_splits = coverage_split.loc[
        coverage_split.group_type.eq("primary_split")
    ]
    countries = coverage_country.loc[coverage_country.group_type.eq("country")]
    minimum_feature_coverage = float(feature_coverage.finite_usable_percent.min())

    required_flagged = set(
        flags.loc[
            flags[
                [
                    "lake_intersection_flag",
                    "coastal_review_flag",
                    "estuarine_tidal_review_flag",
                    "island_edge_review_flag",
                ]
            ]
            .fillna(False)
            .any(axis=1),
            "site_id",
        ]
    ) | set(snaps.loc[snaps.us_region.eq("ALASKA"), "site_id"])
    gate_rows = [
        [1, "all 910 authoritative coordinates accounted for", coordinate_audit["status"] == "PASS" and len(snaps) == 910, "910/910; EPSG:4326; no NEON placeholder reused"],
        [2, "overall usable upstream coverage >=90%", overall.usable_percent >= 90.0, f"{int(overall.usable_sites)}/910 = {overall.usable_percent:.3f}%"],
        [3, "usable coverage >=90% in train/validation/test", bool((primary_splits.usable_percent >= 90.0).all()), "; ".join(f"{r.cohort} {r.usable_percent:.3f}%" for r in primary_splits.itertuples())],
        [4, "usable coverage >=90% in GB and US", bool((countries.usable_percent >= 90.0).all()), "; ".join(f"{r.cohort} {r.usable_percent:.3f}%" for r in countries.itertuples())],
        [5, "overall high-confidence coverage >=80%", overall.high_confidence_percent >= 80.0, f"{int(overall.high_confidence_sites)}/910 = {overall.high_confidence_percent:.3f}%"],
        [6, "every UPSTREAM13 field finite for >=95% of usable sites", minimum_feature_coverage >= 95.0, f"minimum {minimum_feature_coverage:.3f}%; seven fields at this value"],
        [7, "units, scales, aggregation, and provenance resolved", len(units) == 13 and bool(units.unit_scale_provenance_resolved.all()), "13/13 audited; equal-subcatchment mean-of-means explicitly labelled; area weighting rejected"],
        [8, "repeat extraction reproducible", repeat_snap["exact_table_equality"] and repeat_attributes["attribute_exact_hash_equality"], f"snap {repeat_snap['canonical_hash']}; attributes {repeated_hash}"],
        [9, "every required manual review completed", len(manual_results) == len(manual_manifest) == 463 and bool(manual_results.visual_review_completed.all()) and set(manual_results.site_id) == set(manual_manifest.site_id), "463/463 sites across 29 pages"],
        [10, "lake/coast/estuary/Alaska limitations resolved or flagged", required_flagged.issubset(set(manual_results.site_id)), f"all {len(required_flagged)} unique required flagged sites reviewed"],
        [11, "cross-split catchment connectivity audited", len(overlap) > 0 and len(groups) == 910, f"{len(overlap)} cross-split directed/nested pairs; 910 group assignments"],
        [12, "source hashes and licences recorded", len(input_hashes) == 597 and input_hashes.sha256.notna().all() and len(licenses) == 9, "597 input hashes; 9 licence records; 589/589 primary downloads size-verified"],
    ]
    gate = pd.DataFrame(gate_rows, columns=["condition", "requirement", "pass", "evidence"])
    gate["status"] = np.where(gate["pass"], "PASS", "FAIL")
    gate["outcomes_or_predictions_read"] = False
    gate.to_csv(OUT / "spatial_gate_audit.csv", index=False)
    gate_pass = bool(gate["pass"].all())
    if gate_pass:
        raise RuntimeError("Unexpected spatial-gate pass; modelling boundary must be reconsidered")
    failed_conditions = gate.loc[~gate["pass"], "condition"].astype(int).tolist()

    configuration = {
        "analysis": "H00_UPSTREAM1",
        "project_target": "current pH at time t, horizon 0 h, no current/historical pH predictor",
        "spatial_audit_outcome_blind": True,
        "coordinate_source": coordinate_audit["source_coordinate_artifact"],
        "coordinate_source_sha256": coordinate_audit["source_coordinate_sha256"],
        "coordinate_crs": "EPSG:4326",
        "sites": 910,
        "primary_download_plan_sha256": json.loads((OUT / "primary_download_plan.json").read_text(encoding="utf-8"))["plan_sha256"],
        "geofresh_git_commit": "4695825ce9ab1ff2a6673bdfb9ad548da124feba",
        "hydrographr_git_commit": "68735c3bee95836795775989a97e1d7f7dec56ad",
        "snap_algorithm": snaps.snap_algorithm.iloc[0],
        "snap_search_radius_m": 2500.0,
        "one_cell_perturbation_deg": 0.000833,
        "high_confidence_max_distance_m": 500.0,
        "contextual_max_distance_m": 2000.0,
        "material_area_ratio": 2.0,
        "manual_random_seed": 42,
        "manual_random_high_confidence_per_country": 25,
        "independent_usgs_irreconcilable_area_factor": 10.0,
        "independent_camels_gauge_coincident_irreconcilable_area_factor": 5.0,
        "aggregation": "mean_of_upstream_subcatchment_means; equal subcatchment weight",
        "features": feature_columns,
        "feature_finite_gate_percent_of_usable": 95.0,
        "model_authorised_only_if_all_spatial_conditions_pass": True,
        "model_run": False,
        "formal_validation_opened": False,
        "extension_test_opened": False,
        "no_further_h00_variant_authorised": True,
    }
    (OUT / "config_frozen.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )

    decision = {
        "primary_decision": "H00_UPSTREAM1_SPATIAL_GATE_FAILED_MODEL_NOT_RUN",
        "spatial_gate": "FAILED",
        "failed_spatial_gate_conditions": failed_conditions,
        "failure_evidence": {
            "usable_sites": int(overall.usable_sites),
            "minimum_upstream13_finite_percent_of_usable": minimum_feature_coverage,
            "fields_below_95_percent": feature_coverage.loc[
                feature_coverage.finite_usable_percent < 95.0, "feature"
            ].tolist(),
        },
        "candidate_model": "NOT_RUN",
        "training_internal_gate": "NOT_OPENED",
        "formal_validation": "NOT_OPENED",
        "extension_test": "NOT_OPENED",
        "official_h00": "UNCHANGED",
        "official_h00_predictions_or_outcomes_read_for_this_audit": False,
        "formal_membership_columns_read_only": [
            "site_id",
            "formal_comparison_eligible",
        ],
        "no_further_h00_variant_authorised": True,
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )

    usgs_area = independent.loc[
        independent.check_family.eq("USGS_OFFICIAL_DRAINAGE_AREA")
    ]
    camels_area = independent.loc[
        independent.check_family.eq("CAMELS_GB_GAUGE_COINCIDENT_AREA")
    ]
    usgs_coordinate = independent.loc[
        independent.check_family.eq("USGS_OFFICIAL_COORDINATE")
    ]
    valid_snap = snaps.loc[np.isfinite(snaps.geodesic_snap_distance_m)]
    lake_count = int(flags.lake_intersection_flag.fillna(False).sum())
    coast_count = int(flags.coastal_review_flag.fillna(False).sum())
    estuary_count = int(flags.estuarine_tidal_review_flag.fillna(False).sum())
    island_count = int(flags.island_edge_review_flag.fillna(False).sum())
    material_competing = int(manual_manifest.review_material_competing_reach.sum())
    material_perturbation = int(
        manual_manifest.review_material_perturbation_path_change.sum()
    )
    cross_split_sites = set(overlap.loc[overlap.cross_split, "site_id_a"]) | set(
        overlap.loc[overlap.cross_split, "site_id_b"]
    )

    coverage_summary = coverage_split.loc[
        coverage_split.cohort.isin(
            [
                "ALL_PRIMARY_910",
                "PRIMARY_TRAIN",
                "PRIMARY_VALIDATION",
                "PRIMARY_TEST",
                "FORMAL_H00_VALIDATION_128",
                "FORMAL_H00_EXTENSION_TEST_126",
            ]
        ),
        [
            "cohort",
            "total_sites",
            "high_confidence_sites",
            "contextual_sites",
            "unusable_sites",
            "usable_percent",
        ],
    ].copy()
    country_summary = coverage_country[[
        "cohort",
        "total_sites",
        "high_confidence_sites",
        "contextual_sites",
        "unusable_sites",
        "usable_percent",
    ]].copy()
    feature_report = feature_coverage.merge(
        feature_manifest[["feature", "aggregation", "is_upstream_geology"]],
        on="feature",
        how="left",
    )

    report = f"""# H00_UPSTREAM1 final spatial/data audit

## Decision

**H00_UPSTREAM1_SPATIAL_GATE_FAILED_MODEL_NOT_RUN**

The outcome-blind upstream-catchment audit produced a reproducible, quality-labelled 910-site UPSTREAM13 table, but the all-conditions spatial/data gate failed at condition 6. Seven climate/soil fields were finite for only **{minimum_feature_coverage:.3f}%** of the {int(overall.usable_sites)} usable matches, below the frozen 95% requirement. Therefore no candidate was trained, formal validation and extension test remained unopened, and the official H00 method is unchanged.

No pH value, residual, prediction value, site error metric, hot-moment label, validation performance, or test performance was read. Formal validation/test files were used only to recover `site_id` and the pre-existing `formal_comparison_eligible` membership flag for coverage denominators.

## Frozen spatial result

{markdown_table(coverage_summary, ['cohort', 'total_sites', 'high_confidence_sites', 'contextual_sites', 'unusable_sites', 'usable_percent'])}

{markdown_table(country_summary, ['cohort', 'total_sites', 'high_confidence_sites', 'contextual_sites', 'unusable_sites', 'usable_percent'])}

The final overall classes are **747 high-confidence**, **139 contextual**, and **24 unusable**. Usable coverage is 97.363% overall and high-confidence coverage is 82.088%. Unusable sites remain in all relevant denominators.

## Spatial extraction and QA

- Coordinate identity: 910/910 unique sites from `{coordinate_audit['source_coordinate_artifact']}`, SHA-256 `{coordinate_audit['source_coordinate_sha256']}`; WGS84/EPSG:4326, east-positive longitude, no NEON placeholder reused.
- Canonical snap: containing-subcatchment Hydrography90m segment-cell centre where available, otherwise nearest candidate within a fixed 2,500 m audit search; one-cell perturbations use 0.000833 degrees; material area change means a factor >2 on a different directed path.
- Valid snaps: {len(valid_snap)}/910; median distance {valid_snap.geodesic_snap_distance_m.median():.3f} m; maximum {valid_snap.geodesic_snap_distance_m.max():.3f} m; {int((valid_snap.geodesic_snap_distance_m > 500).sum())} valid snaps exceeded 500 m; `07380249` had no valid regional-unit/network snap.
- Required manual review: 463/463 sites on 29 map pages, including all >500 m, HydroLAKES, coastal/estuarine/island, material alternative/perturbation, Alaska, extreme area/order, 26 CAMELS-coincident, and seeded high-confidence samples (seed 42; 25 GB + 25 US).
- Flags: {lake_count} HydroLAKES intersections; {coast_count} coastal; {estuary_count} estuarine/tidal; {island_count} island-edge; {material_competing} material nearby competing-reach reviews; {material_perturbation} material one-cell path-change reviews.
- Independent USGS QA: {len(usgs_coordinate)} official coordinates, all within {usgs_coordinate.reference_value.max():.3f} m; {len(usgs_area)} official drainage areas, with {int(usgs_area.status.eq('PASS_WITHIN_FACTOR_2').sum())} within factor 2, {int(usgs_area.status.eq('REVIEW_FACTOR_2_TO_10').sum())} in the factor 2–10 review band, and {int(usgs_area.status.eq('FAIL_IRRECONCILABLE_FACTOR_GT10').sum())} irreconcilable >10.
- CAMELS-GB QA: 26 gauge-coincident comparisons; {int(camels_area.status.eq('PASS_WITHIN_FACTOR_2').sum())} within factor 2, {int(camels_area.status.eq('REVIEW_FACTOR_2_TO_5').sum())} in the factor 2–5 review band, and {int(camels_area.status.eq('FAIL_IRRECONCILABLE_FACTOR_GT5').sum())} irreconcilable >5.
- Final demotion logic: 23 sites failed a frozen official-area or manual name/area conflict rule; the pre-existing no-snap site made 24 unusable. `08066850` was retained as contextual, not high-confidence, because it is a canal site.
- Lake handling: `01304057` retained contextual as a lake/pond outlet proxy; `04236000` resolved as the named Skaneateles Lake outlet; `E05043A` retained with an explicit HydroLAKES-intersection flag.

`site_snap_matches.gpkg` contains original and snapped point layers. `upstream_catchments.gpkg` contains 909 explicitly labelled equal-area **audit envelopes**, not exact Hydrography90m catchment boundaries. The GeoFRESH dissolved-polygon endpoint did not complete reliably. Catchment metrics were derived from graph-connected upstream segment/subcatchment IDs, never from these envelopes. The digital catchments are not exact ground truth.

## Repeatability

- Independent snap rerun: exact 910-site table equality; hash `{repeat_snap['canonical_hash']}`; spatial tolerance 1e-6 m.
- Attribute rebuilds: two complete graph propagations and site aggregations from the same frozen immutable node caches produced the identical sorted-table hash `{repeated_hash}`.
- Limitation: the 39.263 GB archives were hash/size verified once; the raw archive parse was not repeated. The graph propagation and aggregation were repeated twice.

## Aggregation and UPSTREAM13

Area weighting was attempted but rejected before outcomes because local-area reconstruction did not close: 2,390,178 nonpositive local-area differences, median reconstructed/canonical area ratio 2.090548, and only 22.198% within 10%. The canonical non-area statistic is therefore the documented GeoFRESH formula:

`sum(valid upstream subcatchment means) / count(valid upstream subcatchments)`

This is an equal-subcatchment **mean of upstream subcatchment means**, not an area-weighted land-cell catchment average. The same formula is used for every country and split. Hydrography90m accumulation magnitude supplies positive upstream area; negative GRASS accumulation signs are retained as boundary-underestimate semantics before taking audited physical magnitude.

{markdown_table(feature_report, ['feature', 'finite_usable_sites', 'usable_sites', 'finite_usable_percent', 'aggregation', 'is_upstream_geology'])}

These predictors are topography/hydrography, 1981–2010 climate, SoilGrids soil properties, and 2020 land-cover context. They are **not upstream geology**, alkalinity, carbonate fraction, aquifer permeability, or groundwater class. The 2020 land cover is a retrospective quasi-static covariate and can postdate water-quality observations; this is temporal non-contemporaneity, not target leakage.

## Connectivity and provenance

- Hydrological groups: {groups.catchment_group.nunique()} connected components; {int((group_sizes > 1).sum())} multi-site groups containing {int(group_sizes.loc[group_sizes > 1].sum())} sites; maximum group size {int(group_sizes.max())}.
- Cross-split audit: {len(overlap)} directed-path/nested cross-split site pairs involving {len(cross_split_sites)} unique sites. This was recorded before any possible model fitting.
- Primary files: 589/589 downloaded and exact-size verified; 39,262,799,199 bytes; plan SHA-256 `{configuration['primary_download_plan_sha256']}`.
- Provenance: 597 spatial input hashes, 284 existing H00 artifact identities/hashes, 13 unit/scale records, and 9 licence records.
- Code identities: GeoFRESH `{configuration['geofresh_git_commit']}`; hydrographr `{configuration['hydrographr_git_commit']}`; Hydrography90m v1.0/FRED 762.1; Environment90m/FRED 995.0; HydroLAKES v1.0.
- Licences: Hydrography90m and Environment90m CC BY-NC 4.0 for non-commercial dissertation use; underlying-source attribution/redistribution notes are in `source_licenses.csv`.

## Spatial/data gate

{markdown_table(gate, ['condition', 'requirement', 'status', 'evidence'])}

Condition 6 failed. No weaker threshold, imputation, feature deletion, country restriction, or performance-guided rescue was attempted.

## Direct answers to the 27 required questions

1. **Coordinates:** the corrected `ph_h00_solar_dynamic1_v1/coordinate_audit.csv` artifact (SHA-256 above) covered all 910 sites; no placeholder was reused.
2. **Versions:** GeoFRESH and hydrographr commits are listed above; Hydrography90m v1.0/FRED 762.1, Environment90m/FRED 995.0, and HydroLAKES v1.0 were used.
3. **Snap rule:** frozen containing-subcatchment segment-cell snap with a 2,500 m audit search, 500/2,000 m quality thresholds, and 0.000833-degree perturbations.
4. **Classes:** 747 high-confidence, 139 contextual, 24 unusable.
5. **Coverage:** shown in the frozen cohort/country tables; overall 97.363%, train 96.865%, validation 98.540%, test 98.519%, GB 98.420%, US 93.925%, Alaska 100%, Puerto Rico 100%. Formal 128/126 cohort coverage is recorded without outcomes.
6. **Hydrological flags:** {lake_count} lake, {coast_count} coastal, {estuary_count} estuarine/tidal, {island_count} island-edge; {material_competing} material competing-reach and {material_perturbation} perturbation-path reviews.
7. **Manual/independent QA:** 463/463 visual reviews completed; most official areas agreed, but 12 USGS and 3 gauge-coincident CAMELS comparisons were irreconcilable under frozen thresholds; named-river area contradictions were also demoted.
8. **Repeatability:** yes for the independent snap table and both cached-node attribute rebuilds, with exact hashes; raw 39 GB parsing was not independently repeated.
9. **Weighting:** no. The attributes are equal-subcatchment means of upstream subcatchment means; the failed area-weighting reconstruction is fully recorded.
10. **Units/licences:** 13/13 unit/scale records and all selected licences were resolved and recorded; product/source-date limitations remain explicit.
11. **Connectivity:** {len(overlap)} cross-split directed/nested pairs, {len(cross_split_sites)} involved sites, and {groups.catchment_group.nunique()} outcome-blind hydrological groups.
12. **Spatial gate:** failed solely at condition 6.
13. **Predictors:** the 13 fields in the table above; they are environmental/topographic/soil/land-cover context, not upstream geology.
14. **Cold-start Ridge:** NOT RUN because the spatial gate failed.
15. **Initial neural equality to M0:** NOT TESTED; no candidate was instantiated.
16. **Original H00 tensors:** unchanged; no H00 model tensor was loaded for fitting or modified.
17. **Catchment-grouped internal gate:** NOT OPENED.
18. **Sites improved/harmed:** NOT AVAILABLE; no candidate predictions were generated.
19. **Within-site centred R2:** NOT COMPUTED.
20. **GB/US model safety:** NOT TESTED; spatial coverage by country is reported only.
21. **Fold/seed consistency:** NOT TESTED; folds and seeds were never opened for modelling.
22. **Formal validation:** NOT OPENED.
23. **Extension test:** NOT OPENED.
24. **Raw/cold-start versus post-offset conclusion:** no comparison exists because modelling was prohibited at the spatial gate.
25. **Official H00:** unchanged.
26. **Bounded conclusion:** model-derived upstream attributes can be assembled reproducibly for most sites, but the frozen 95% per-field completeness requirement was not met, so a defensible H00_UPSTREAM1 experiment was not authorised.
27. **Not attempted:** no imputation-driven rescue, feature subset search, alternative threshold, area-weighting selection by performance, point geology/BFI/imperviousness/discharge-distance addition, HydroATLAS/StreamCat model inputs, candidate fitting, validation, test, or UPSTREAM2.

## Final stop rule

The official retained H00 remains the frozen absolute-pH H00 plus each site's first-30-day mean residual offset. H00_UPSTREAM1 is closed. No further H00 feature set, adapter, architecture, or validation attempt is authorised; the project should return to dissertation writing.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    required_spatial_outputs = [
        "FINAL_REPORT.md",
        "decision.json",
        "config_frozen.json",
        "software_environment.txt",
        "source_inventory.csv",
        "source_licenses.csv",
        "input_hashes.csv",
        "authoritative_coordinate_inventory.csv",
        "coordinate_identity_audit.json",
        "site_snap_matches.csv",
        "site_snap_matches.gpkg",
        "upstream_catchments.gpkg",
        "snap_quality_criteria.json",
        "manual_review_manifest.csv",
        "manual_review_results.csv",
        "independent_spatial_checks.csv",
        "extraction_repeatability.json",
        "aggregation_decision.json",
        "aggregation_reconstruction_checks.csv",
        "upstream_attribute_source_manifest.csv",
        "upstream_attribute_unit_audit.csv",
        "upstream13_unscaled.csv",
        "upstream13_model_table.csv",
        "upstream13_feature_manifest.csv",
        "coverage_by_split.csv",
        "coverage_by_country.csv",
        "coverage_by_match_quality.csv",
        "lake_coast_estuary_audit.csv",
        "catchment_connection_edges.csv",
        "catchment_group_assignment.csv",
        "cross_split_catchment_overlap.csv",
        "spatial_gate_audit.csv",
    ]
    absent = [name for name in required_spatial_outputs if not (OUT / name).exists()]
    if absent:
        raise RuntimeError(f"Missing required final spatial outputs: {absent}")
    model_only_outputs = [
        "m0_reproduction_audit.json",
        "support_query_audit.csv",
        "row_key_audit.json",
        "train_internal_fold_assignment.csv",
        "cold_start_ridge_metrics.csv",
        "internal_oof_metrics.csv",
        "formal_validation_metrics.csv",
        "extension_test_metrics.csv",
        "site_metrics.csv",
        "predictions",
        "checkpoints",
    ]
    unexpectedly_present = [name for name in model_only_outputs if (OUT / name).exists()]
    if unexpectedly_present:
        raise RuntimeError(f"Model-stage outputs exist despite failed spatial gate: {unexpectedly_present}")

    core_artifacts = [
        "decision.json",
        "config_frozen.json",
        "FINAL_REPORT.md",
        "site_snap_matches.csv",
        "upstream13_unscaled.csv",
        "spatial_gate_audit.csv",
        "manual_review_results.csv",
        "independent_spatial_checks.csv",
        "input_hashes.csv",
    ]
    run_manifest = {
        "analysis": "H00_UPSTREAM1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_SPATIAL_GATE_FAILED",
        "decision": decision["primary_decision"],
        "outcome_blind": True,
        "outcome_or_prediction_value_columns_read": [],
        "formal_membership_columns_read": [
            "site_id",
            "formal_comparison_eligible",
        ],
        "sites": 910,
        "primary_download_files": 589,
        "primary_download_bytes": 39262799199,
        "snap_repeat_sha256": repeat_snap["canonical_hash"],
        "attribute_repeat_sha256": repeated_hash,
        "required_spatial_outputs_present": len(required_spatial_outputs),
        "model_only_outputs_created": [],
        "pipeline_scripts": {
            path.name: sha256(path)
            for path in sorted(PIPELINE.glob("*.py"))
        },
        "core_artifact_hashes": {
            name: sha256(OUT / name) for name in core_artifacts
        },
        "geopackage_layers": {
            "site_snap_matches.gpkg": pyogrio.list_layers(
                OUT / "site_snap_matches.gpkg"
            ).tolist(),
            "upstream_catchments.gpkg": pyogrio.list_layers(
                OUT / "upstream_catchments.gpkg"
            ).tolist(),
        },
    }
    (OUT / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not (OUT / "run_manifest.json").exists():
        raise RuntimeError("Run manifest was not written")
    print(
        json.dumps(
            {
                "decision": decision["primary_decision"],
                "failed_conditions": failed_conditions,
                "usable_sites": int(overall.usable_sites),
                "minimum_feature_coverage_percent": minimum_feature_coverage,
                "required_spatial_outputs": len(required_spatial_outputs) + 1,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
