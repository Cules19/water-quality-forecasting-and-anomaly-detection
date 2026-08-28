#!/usr/bin/env python3
"""Fetch per-position lat/lon for NEON aquatic sondes.

The site registry stores one lat/lon per NEON site, so S1 (upstream) and S2
(downstream) markers stack on any map. NEON publishes the true per-position
coordinates inside every raw package as `sensor_positions_*.csv`; the
`locationReferenceLatitude` / `Longitude` columns are per-`HOR.VER` and give
the ~500 m S1/S2 separation directly (offsets are cm-scale and can be
ignored for mapping purposes).

Output: `data/reference/neon_sensor_positions.json` — a dict keyed by
NEON site code (`ARIK`, `PRIN`, ...) mapping to a dict of positions:

    {
      "PRIN": {
        "S1": {"lat": 33.378350, "lon": -97.783591, "elev": 208.5,
               "description": "Pringle Creek Water Chemistry and Temperature S1"},
        "S2": {"lat": 33.378358, "lon": -97.781339, "elev": 208.1, ...}
      },
      ...
    }

Usage:
    python core/utilities/fetch_neon_sensor_positions.py

Author: TSLR
Date: 2026-07-03
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NEON_BASE = "https://data.neonscience.org/api/v0"
PRODUCT_WQ = "DP1.20288.001"  # Water quality multisonde — every AIS stream site


def _read_token(env_path: Path) -> str | None:
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("NEON_API_TOKEN="):
            return line.split("=", 1)[1].strip("\"'")
    return None


def _hor_to_position(hor_ver: str) -> str | None:
    """`101.1` -> `S1`, `102.1` -> `S2`, `111.1` -> `S11`, ...
    Returns None for anything that isn't a stream-style HOR.
    """
    hor = str(hor_ver).split(".")[0]
    if not hor.isdigit() or len(hor) != 3 or not hor.startswith("1"):
        return None
    return f"S{int(hor) - 100}"


def _fetch_sensor_positions(session: requests.Session, site: str,
                            month: str) -> pd.DataFrame | None:
    url = f"{NEON_BASE}/data/{PRODUCT_WQ}/{site}/{month}"
    r = session.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    files = r.json().get("data", {}).get("files", [])
    sp = next((f for f in files if "sensor_positions" in f["name"]), None)
    if sp is None:
        return None
    r2 = session.get(sp["url"], timeout=60)
    r2.raise_for_status()
    return pd.read_csv(io.StringIO(r2.content.decode()))


def _extract_current_positions(df: pd.DataFrame) -> dict:
    """Keep only rows whose position window is still open. Dedupe by HOR
    keeping the first sensor description encountered.
    """
    current = df[df["positionEndDateTime"].isna()].copy()
    positions: dict = {}
    for _, row in current.iterrows():
        pos = _hor_to_position(row["HOR.VER"])
        if pos is None or pos in positions:
            continue
        lat = row.get("locationReferenceLatitude")
        lon = row.get("locationReferenceLongitude")
        if pd.isna(lat) or pd.isna(lon):
            continue
        positions[pos] = {
            "lat": float(lat),
            "lon": float(lon),
            "elev": (float(row["locationReferenceElevation"])
                     if pd.notna(row.get("locationReferenceElevation")) else None),
            "description": str(row.get("sensorLocationDescription", "")),
        }
    return positions


def fetch_all(session: requests.Session,
              site_codes: list[str]) -> dict[str, dict]:
    """Fetch per-position coords for every requested NEON site code.

    For each site we pull the most recent available month with a package.
    """
    # One call gives available months for every AIS site
    r = session.get(f"{NEON_BASE}/products/{PRODUCT_WQ}", timeout=60)
    r.raise_for_status()
    site_months = {
        s["siteCode"]: s.get("availableMonths", [])
        for s in r.json()["data"]["siteCodes"]
    }

    out: dict[str, dict] = {}
    for code in site_codes:
        months = site_months.get(code, [])
        if not months:
            logger.warning("No packages listed for %s", code)
            continue
        for month in reversed(months):  # most recent first, fall back if missing
            try:
                df = _fetch_sensor_positions(session, code, month)
            except requests.HTTPError as e:
                logger.warning("HTTP error %s/%s: %s", code, month, e)
                continue
            if df is None or df.empty:
                continue
            positions = _extract_current_positions(df)
            if positions:
                logger.info("%s: %d positions from %s (%s)",
                            code, len(positions), month, list(positions))
                out[code] = {"positions": positions, "sourced_from": month}
                break
        else:
            logger.warning("Exhausted months for %s without positions", code)
    return out


def _default_out_path() -> Path:
    try:
        from config import config
        return Path(config.DATA_DIR) / "reference" / "neon_sensor_positions.json"
    except Exception:
        return Path(__file__).resolve().parents[2] / "data" / "reference" / "neon_sensor_positions.json"


def _neon_site_codes_from_registry() -> list[str]:
    try:
        from config import config
        reg_path = Path(config.REGISTRY_PATH)
    except Exception:
        reg_path = Path(__file__).resolve().parents[2] / "data" / "reference" / "sites_register.json"

    with reg_path.open() as f:
        reg = json.load(f)

    codes = set()
    for sid, s in reg.items():
        if s.get("source") != "NEON":
            continue
        pos = s.get("sonde_position") or ""
        if pos and sid.startswith("N") and sid.endswith(pos):
            codes.add(sid[1:-len(pos)])
    return sorted(codes)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    env_path = Path(__file__).resolve().parents[2] / ".env"
    token = _read_token(env_path) or os.environ.get("NEON_API_TOKEN")
    if not token:
        logger.error("No NEON_API_TOKEN set — cannot proceed")
        return 1

    session = requests.Session()
    session.headers.update({
        "X-API-Token": token,
        "Accept": "application/json",
        "User-Agent": "Pollution_ID/1.0",
    })

    codes = _neon_site_codes_from_registry()
    logger.info("Fetching sensor positions for %d NEON site codes", len(codes))

    result = fetch_all(session, codes)

    out_path = _default_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    logger.info("Wrote %d site codes to %s", len(result), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
