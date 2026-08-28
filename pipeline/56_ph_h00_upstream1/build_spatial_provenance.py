from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "analysis_outputs/ph_h00_upstream1_v1"
PRIMARY = PROJECT / "data/external/upstream1_primary"
M0 = PROJECT / "analysis_outputs/ph_common1000_lstm_nohistory_h0_rain_final_v1/full"
CALIBRATION = PROJECT / "analysis_outputs/ph_h00_30day_calibration_validation_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def parquet_identity(path: Path) -> tuple[int, str]:
    parquet = pq.ParquetFile(path)
    return parquet.metadata.num_rows, "|".join(parquet.schema.names)


def csv_identity(path: Path) -> tuple[int, str]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        header = handle.readline().rstrip("\r\n")
        rows = sum(1 for _ in handle)
    return rows, header


def main() -> None:
    plan = pd.read_csv(OUT / "primary_download_plan.csv")
    hash_rows: list[dict[str, object]] = []
    for number, row in enumerate(plan.itertuples(index=False), start=1):
        path = Path(str(row.destination))
        hash_rows.append(
            {
                "role": "primary_spatial_input",
                "family": row.family,
                "variable": row.variable,
                "tile": row.tile,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source_url": row.source_url,
                "values_read_for_provenance": False,
            }
        )
        if number % 50 == 0:
            print(f"hashed primary inputs {number}/{len(plan)}", flush=True)
    supplements = [
        ("hydrography90m_h10v04_accumulation_raster", PRIMARY / "downloaded/r.watershed/accumulation_tiles20d/accumulation_h10v04.tif"),
        ("hydrolakes_v10", PRIMARY / "HydroLAKES_polys_v10_shp.zip"),
        ("natural_earth_10m_land_v5_1_1", PRIMARY / "ne_10m_land.zip"),
        ("hydrography90m_regional_unit_overview", PRIMARY / "regional_unit_ovr.tif"),
        ("authoritative_corrected_coordinates", PROJECT / "analysis_outputs/ph_h00_solar_dynamic1_v1/coordinate_audit.csv"),
        ("geofresh_910_coordinate_input", OUT / "spatial_inputs/geofresh_910_sites_wgs84.csv"),
        ("upstream_connection_graph", OUT / "upstream_connection_graph.npz"),
        ("h10v04_accumulation_recovery", OUT / "h10v04_accumulation_recovered.npz"),
    ]
    for role, path in supplements:
        hash_rows.append(
            {
                "role": role,
                "family": "supplement_or_derived_identity",
                "variable": "",
                "tile": "",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source_url": "",
                "values_read_for_provenance": False,
            }
        )
    pd.DataFrame(hash_rows).to_csv(OUT / "input_hashes.csv", index=False)

    fixed = [
        ("m0_checkpoint", M0 / "models/ph_LSTM_RAIN_H0_H00.pt"),
        ("m0_config", M0 / "reports/effective_config.json"),
        ("m0_preprocessing", M0 / "reports/train_only_preprocessing.json"),
        ("m0_metadata_transform", M0 / "metadata/ph_no_coordinates_train_only_transform.json"),
        ("m0_success_manifest", M0 / "models/ph_LSTM_RAIN_H0_H00_SUCCESS.json"),
        ("m0_exact_reproduction_audit", PROJECT / "analysis_outputs/ph_h00_static3_additive_validation_v1/audit/h00_m0_reproduction_check.json"),
        ("primary_split_manifest", PROJECT / "analysis_outputs/ph_common1000_teacher_archive_features_do_converted_v1/full/reports/primary_archive_experiment_cohort.csv"),
        ("official_support_offsets", CALIBRATION / "calibration/site_calibration_parameters.csv"),
        ("official_validation_postoffset_predictions", CALIBRATION / "predictions/validation_predictions.parquet"),
        ("official_test_postoffset_predictions", CALIBRATION / "predictions/test_predictions.parquet"),
        ("prior_calibration_aware_implementation", PROJECT / "pipeline/54_ph_h00_calibration_aware_residual/run_internal_experiment.py"),
        ("prior_calibration_aware_config", PROJECT / "analysis_outputs/ph_h00_calibration_aware_residual1_v1/config_frozen.json"),
    ]
    inventory_rows: list[dict[str, object]] = []
    for role, path in fixed:
        row_count: int | str = ""
        schema = ""
        if path.suffix == ".parquet":
            row_count, schema = parquet_identity(path)
        elif path.suffix == ".csv":
            row_count, schema = csv_identity(path)
        inventory_rows.append(
            {
                "role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "row_count": row_count,
                "schema": schema,
                "outcome_values_read": False,
            }
        )
    prediction_root = M0 / "predictions_by_site/ph_LSTM_RAIN_H0_H00"
    for path in sorted(prediction_root.rglob("*.parquet")):
        row_count, schema = parquet_identity(path)
        inventory_rows.append(
            {
                "role": "m0_raw_prediction_by_site",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "row_count": row_count,
                "schema": schema,
                "outcome_values_read": False,
            }
        )
    source_inventory = pd.DataFrame(inventory_rows)
    source_inventory.to_csv(OUT / "source_inventory.csv", index=False)

    coords = pd.read_csv(
        OUT / "spatial_inputs/geofresh_910_sites_wgs84.csv", dtype={"id": "string"}
    ).rename(columns={"id": "site_id"})
    coords["crs"] = "EPSG:4326"
    coords["longitude_convention"] = "east-positive"
    coords["authoritative_source"] = str(
        PROJECT / "analysis_outputs/ph_h00_solar_dynamic1_v1/coordinate_audit.csv"
    )
    coords["coordinate_source_sha256"] = sha256(
        PROJECT / "analysis_outputs/ph_h00_solar_dynamic1_v1/coordinate_audit.csv"
    )
    coords["outcome_guided_correction"] = False
    coords.to_csv(OUT / "authoritative_coordinate_inventory.csv", index=False)

    licenses = pd.DataFrame(
        [
            ["Hydrography90m v1.0", "CC BY-NC 4.0", "https://doi.org/10.18728/igb-fred-762.1", "non-commercial dissertation use; attribution required"],
            ["Environment90m 2026", "CC BY-NC 4.0", "https://doi.org/10.18728/igb-fred-995.0", "non-commercial dissertation use; attribution required"],
            ["GeoFRESH source", "GPL-3.0", "https://doi.org/10.1080/17538947.2024.2391033", "code provenance; no bulk database redistribution"],
            ["hydrographr source", "GPL-3.0", "https://doi.org/10.1111/2041-210X.14226", "code attribution and GPL terms"],
            ["HydroLAKES v1.0", "CC BY 4.0", "https://www.hydrosheds.org/products/hydrolakes", "attribution required; do not redistribute complete source archive"],
            ["Natural Earth 10m land v5.1.1", "Public domain", "https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/", "QA coastline only"],
            ["CHELSA v2.1", "underlying-source terms; redistributed through Environment90m", "https://www.chelsa-climate.org/datasets/chelsa_bioclim", "1981-2010 climatology; attribution required"],
            ["SoilGrids250m v2.0", "CC BY 4.0 underlying source", "https://soilgrids.org/", "depth-integrated Environment90m tables"],
            ["ESA CCI Land Cover 2020", "underlying-source terms; redistributed through Environment90m", "https://climate.esa.int/en/projects/land-cover/", "retrospective quasi-static covariate"],
        ],
        columns=["source", "license", "reference", "use_and_redistribution_note"],
    )
    licenses.to_csv(OUT / "source_licenses.csv", index=False)

    unit_rows = [
        [1, "log1p_upstream_area_km2", "Hydrography90m accumulation maximum", "km2", 1.0, "log1p", "positive finite"],
        [2, "upstream_mean_elevation_m", "MERIT DEM v1.0.3 elev mean", "m", 1.0, "none", "finite; global terrestrial elevation"],
        [3, "upstream_mean_gradient", "Hydrography90m gradient", "elevation difference / distance", 1.0, "none", ">=0"],
        [4, "upstream_annual_precipitation", "CHELSA v2.1 bio12 1981-2010", "kg m-2 year-1", 0.1, "none", ">=0"],
        [5, "upstream_precipitation_seasonality", "CHELSA v2.1 bio15 1981-2010", "coefficient of variation percent", 0.1, "none", ">=0"],
        [6, "upstream_soil_ph_h2o", "SoilGrids250m v2.0 phihox", "pH H2O", 0.1, "none", "0 to 14"],
        [7, "upstream_soil_cec", "SoilGrids250m v2.0 cecsol", "cmol(c) kg-1", 1.0, "none", ">=0"],
        [8, "upstream_soil_organic_carbon", "SoilGrids250m v2.0 orcdrc", "g kg-1", 1.0, "none", ">=0"],
        [9, "upstream_soil_clay", "SoilGrids250m v2.0 clyppt", "percent", 1.0, "none", "0 to 100"],
        [10, "upstream_depth_to_bedrock", "SoilGrids250m v2.0 bdricm", "cm; capped at 200 in source", 1.0, "none", "0 to 200"],
        [11, "upstream_urban_fraction", "ESA CCI 2020 c190", "fraction", 1.0, "none", "0 to 1"],
        [12, "upstream_cropland_associated_fraction", "ESA CCI 2020 c10+c20+c30", "fraction", 1.0, "sum selected classes", "0 to 1"],
        [13, "upstream_forest_associated_fraction", "ESA CCI 2020 c50+c60+c70+c80+c90+c100", "fraction", 1.0, "sum selected classes", "0 to 1"],
    ]
    units = pd.DataFrame(
        unit_rows,
        columns=["feature_number", "feature", "source_field", "physical_unit", "source_scale_factor", "model_transform", "valid_range"],
    )
    units["aggregation"] = (
        "mean_of_upstream_subcatchment_means; equal subcatchment weight; "
        "not an area-weighted catchment average"
    )
    units.loc[units.feature_number.eq(1), "aggregation"] = "canonical contributing area"
    units["is_upstream_geology"] = False
    units["unit_scale_provenance_resolved"] = True
    units.to_csv(OUT / "upstream_attribute_unit_audit.csv", index=False)

    environment = [
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"geofresh_git_commit={git_commit(PROJECT / 'data/external/upstream1_sources/geofresh')}",
        f"hydrographr_git_commit={git_commit(PROJECT / 'data/external/upstream1_sources/hydrographr')}",
        "",
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
    ]
    (OUT / "software_environment.txt").write_text("\n".join(environment), encoding="utf-8")
    print(f"source inventory rows: {len(source_inventory)}")


if __name__ == "__main__":
    main()
