from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pandas as pd
import requests
from pyproj import Geod


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "analysis_outputs/ph_h00_upstream1_v1"
SNAPS = OUT / "site_snap_matches_pre_review.csv"
REGISTER = PROJECT / "data/reference/sites_register.json"
USGS_ENDPOINT = "https://waterservices.usgs.gov/nwis/site/"
GEOD = Geod(ellps="WGS84")


def fetch_usgs(site_ids: list[str]) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    http = requests.Session()
    http.headers.update({"User-Agent": "Pollution-ID-UPSTREAM1-audit/1.0"})
    for start in range(0, len(site_ids), 50):
        batch = site_ids[start : start + 50]
        response = http.get(
            USGS_ENDPOINT,
            params={
                "format": "rdb",
                "sites": ",".join(batch),
                "siteOutput": "expanded",
                "siteStatus": "all",
            },
            timeout=180,
        )
        response.raise_for_status()
        lines = [line for line in response.text.splitlines() if not line.startswith("#")]
        if len(lines) < 3:
            continue
        table = pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])), sep="\t", dtype=str)
        tables.append(table)
        print(f"USGS metadata {min(start + 50, len(site_ids))}/{len(site_ids)}", flush=True)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def main() -> None:
    snaps = pd.read_csv(SNAPS, dtype={"site_id": "string"})
    register = json.loads(REGISTER.read_text(encoding="utf-8"))
    local_rows: list[dict[str, object]] = []
    for row in snaps.itertuples(index=False):
        metadata = register.get(str(row.site_id), {})
        local_rows.append(
            {
                "site_id": str(row.site_id),
                "project_source": metadata.get("source"),
                "project_waterbody_name": metadata.get("waterbody_name"),
                "project_river_name": metadata.get("river_name"),
                "project_latitude": metadata.get("latitude"),
                "project_longitude": metadata.get("longitude"),
                "project_site_type": metadata.get("site_type"),
            }
        )
    local = pd.DataFrame(local_rows)
    local.to_csv(OUT / "project_site_name_spatial_qa.csv", index=False)

    source_map = local.set_index("site_id").project_source.astype(str)
    us_ids = snaps.loc[
        snaps.site_id.map(source_map).eq("USGS"), "site_id"
    ].astype(str).tolist()
    usgs = fetch_usgs(us_ids)
    keep = [
        "site_no",
        "station_nm",
        "site_tp_cd",
        "dec_lat_va",
        "dec_long_va",
        "coord_meth_cd",
        "coord_acy_cd",
        "dec_coord_datum_cd",
        "huc_cd",
        "drain_area_va",
        "contrib_drain_area_va",
    ]
    usgs = usgs[[column for column in keep if column in usgs]].rename(columns={"site_no": "site_id"})
    usgs["site_id"] = usgs.site_id.astype("string")
    usgs = snaps.loc[snaps.site_id.isin(us_ids),
        ["site_id", "original_longitude", "original_latitude", "stream_segment_id"]
    ].merge(usgs, on="site_id", how="left", validate="one_to_one")
    lon = pd.to_numeric(usgs.get("dec_long_va"), errors="coerce")
    lat = pd.to_numeric(usgs.get("dec_lat_va"), errors="coerce")
    valid = lon.notna() & lat.notna()
    usgs["coordinate_difference_m"] = math.nan
    if valid.any():
        _, _, distance = GEOD.inv(
            usgs.loc[valid, "original_longitude"].to_numpy(dtype=float),
            usgs.loc[valid, "original_latitude"].to_numpy(dtype=float),
            lon[valid].to_numpy(dtype=float),
            lat[valid].to_numpy(dtype=float),
        )
        usgs.loc[valid, "coordinate_difference_m"] = distance
    usgs["drainage_area_km2"] = pd.to_numeric(usgs.get("drain_area_va"), errors="coerce") * 2.589988110336
    usgs["contributing_drainage_area_km2"] = (
        pd.to_numeric(usgs.get("contrib_drain_area_va"), errors="coerce") * 2.589988110336
    )
    usgs["qa_only_not_model_input"] = True
    usgs["retrieval_endpoint"] = USGS_ENDPOINT
    usgs["outcomes_or_predictions_read"] = False
    usgs.to_csv(OUT / "usgs_site_metadata_qa.csv", index=False)
    summary = {
        "requested_us_sites": len(us_ids),
        "returned_usgs_sites": int(usgs.station_nm.notna().sum()),
        "sites_with_reported_drainage_area": int(usgs.drainage_area_km2.notna().sum()),
        "site_type_counts": usgs.site_tp_cd.fillna("MISSING").value_counts().to_dict(),
        "maximum_coordinate_difference_m": float(usgs.coordinate_difference_m.max()),
        "qa_only_not_model_input": True,
        "outcomes_or_predictions_read": False,
    }
    (OUT / "national_site_qa_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
