from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "analysis_outputs/ph_h00_solar_dynamic1_v1/coordinate_audit.csv"
OUTPUT_ROOT = PROJECT / "analysis_outputs/ph_h00_upstream1_v1"
INPUT_DIR = OUTPUT_ROOT / "spatial_inputs"
CSV_PATH = INPUT_DIR / "geofresh_910_sites_wgs84.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source = pd.read_csv(SOURCE, dtype={"site_id": "string"})
    assert len(source) == 910
    assert source["site_id"].nunique() == 910
    assert source["one_fixed_pair"].astype(bool).all()
    assert source["latitude_in_range"].astype(bool).all()
    assert source["longitude_in_range_east_positive"].astype(bool).all()
    assert not source[["selected_latitude", "selected_longitude"]].isna().any().any()

    coords = source[["site_id", "selected_latitude", "selected_longitude"]].copy()
    coords.columns = ["id", "latitude", "longitude"]
    coords = coords.sort_values("id", kind="mergesort")

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    coords.to_csv(CSV_PATH, index=False, lineterminator="\n", float_format="%.8f")
    assert CSV_PATH.stat().st_size < 1_000_000

    audit = {
        "status": "PASS",
        "source_coordinate_artifact": str(SOURCE),
        "source_coordinate_sha256": sha256(SOURCE),
        "input_csv": str(CSV_PATH),
        "input_csv_sha256": sha256(CSV_PATH),
        "rows": len(coords),
        "unique_site_ids": int(coords["id"].nunique()),
        "crs": "EPSG:4326",
        "longitude_convention": "east-positive",
        "bytes": CSV_PATH.stat().st_size,
        "neon_placeholder_reused": False,
        "outcomes_or_predictions_read": False,
    }
    (OUTPUT_ROOT / "coordinate_identity_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
