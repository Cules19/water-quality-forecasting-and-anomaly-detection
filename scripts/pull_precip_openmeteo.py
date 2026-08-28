#!/usr/bin/env python3
"""
Download hourly precipitation for sites that lack it, from the Open-Meteo
ERA5-Land reanalysis archive (global, hourly, ~9 km, free, no API key).

Motivated by the precipitation-coverage audit: NEON aquatic sites carry no
native NEON precipitation product (DP1.00006 is terrestrial-only), and a couple
of USGS Alaska gauges fall outside CAMELS-H. Open-Meteo ERA5-Land samples a
single grid cell at each site's exact lat/lon and covers CONUS + Alaska +
Puerto Rico uniformly.

Reads the gap list produced by the audit (site_id, latitude, longitude,
start_date, end_date) and writes one CSV per site into
``data/raw/precipitation_data/`` using the existing schema so the files sit
alongside the CAMELS-H / Met-Office precip already in that directory:

    datetime,precipitation_mm,date,year,month,day,hour

(`datetime` is date-only to match the existing files; the real hourly
resolution lives in the `hour` column.)

Restart-safe: a site whose precip file already exists is skipped unless
``--force`` is passed.

Author: TSLR
Date: 2026-07-15
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# `era5_seamless` uses ERA5-Land (~9 km) where available and falls back to
# ERA5 (~25 km) elsewhere, so it never returns null over coastal/island cells
# (ERA5-Land is land-masked). At time of writing ERA5-Land precipitation is not
# exposed, so this currently resolves to ERA5 hourly everywhere.
OPENMETEO_MODEL = "era5_seamless"
SOURCE_TAG = "OpenMeteo_ERA5seamless"

# ERA5 / ERA5-Land start in 1950 and lag the present by ~5 days.
ERA5_LAND_EARLIEST = "1950-01-01"
ERA5_LAND_LAG_DAYS = 6

DEFAULT_GAP_CSV = REPO_ROOT / "outputs" / "reports" / "precip_gap_sites.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "precipitation_data"


def _setup_logging() -> Path:
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    current = logs_dir / "pull_precip_openmeteo.log"
    dated = logs_dir / f"pull_precip_openmeteo_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(current, mode="w", encoding="utf-8"),
            logging.FileHandler(dated, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return dated


log = logging.getLogger("pull_precip_openmeteo")


def _clamp_dates(start: Optional[str], end: Optional[str]) -> tuple[str, str]:
    """Clamp requested range to ERA5-Land's valid window (YYYY-MM-DD)."""
    def _date_only(s: Optional[str], fallback: str) -> str:
        if not s:
            return fallback
        return str(s)[:10]

    latest_ok = (
        datetime.now(timezone.utc).date()
        - __import__("datetime").timedelta(days=ERA5_LAND_LAG_DAYS)
    ).isoformat()

    lo = max(_date_only(start, "2015-01-01"), ERA5_LAND_EARLIEST)
    hi = min(_date_only(end, latest_ok), latest_ok)
    if hi < lo:
        hi = lo
    return lo, hi


def _fetch_hourly(lat: float, lon: float, start: str, end: str,
                  attempts: int = 4, base_delay: float = 3.0) -> dict:
    """One Open-Meteo archive request for hourly ERA5-Land precipitation."""
    params = {
        "latitude": f"{lat:.5f}",
        "longitude": f"{lon:.5f}",
        "start_date": start,
        "end_date": end,
        "hourly": "precipitation",
        "models": OPENMETEO_MODEL,
        "timezone": "UTC",
    }
    url = f"{ARCHIVE_URL}?{urllib.parse.urlencode(params)}"
    last_exc: Optional[BaseException] = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Pollution_ID/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            # 429 = rate limited; back off. 4xx (other) is fatal.
            if e.code == 429 and i + 1 < attempts:
                wait = base_delay * (2 ** i)
                log.warning("HTTP 429 rate limited; sleeping %.0fs", wait)
                time.sleep(wait)
                last_exc = e
                continue
            body = e.read().decode("utf-8", "replace")[:200]
            raise RuntimeError(f"HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_exc = e
            if i + 1 < attempts:
                wait = base_delay * (2 ** i)
                log.warning("network error (%s); retry %d/%d after %.0fs",
                            e, i + 1, attempts - 1, wait)
                time.sleep(wait)
    raise RuntimeError(f"failed after {attempts} attempts: {last_exc}")


def _write_site_csv(out_dir: Path, site_id: str, times: list[str],
                    precip: list) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"{site_id}_precipitation_from_{SOURCE_TAG}_{ts}.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["datetime", "precipitation_mm", "date", "year", "month", "day", "hour"])
        for t, p in zip(times, precip):
            # t like "2020-01-01T00:00"
            d = t[:10]
            hh = int(t[11:13])
            y, mo, da = d.split("-")
            val = 0.0 if p is None else float(p)
            w.writerow([d, val, d, int(y), int(mo), int(da), hh])
    return out


def _existing_file(out_dir: Path, site_id: str) -> Optional[Path]:
    hits = list(out_dir.glob(f"{site_id}_precipitation_from_{SOURCE_TAG}_*.csv"))
    return hits[0] if hits else None


def _load_gap_rows(gap_csv: Path, only: Optional[list[str]]) -> list[dict]:
    rows: list[dict] = []
    with open(gap_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if only and r["site_id"] not in only:
                continue
            if not r.get("latitude") or not r.get("longitude"):
                log.warning("skip %s: no lat/lon", r["site_id"])
                continue
            rows.append(r)
    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gap-csv", type=Path, default=DEFAULT_GAP_CSV,
                   help=f"Gap-site list (default: {DEFAULT_GAP_CSV})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Output dir (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--sites", nargs="+",
                   help="Restrict to these site_ids (default: all in gap CSV)")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if a precip file already exists")
    p.add_argument("--sleep", type=float, default=1.0,
                   help="Seconds to sleep between sites (politeness)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    dated = _setup_logging()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_gap_rows(args.gap_csv, args.sites)
    log.info("Gap sites to process: %d", len(rows))
    log.info("Source    : %s via Open-Meteo archive", OPENMETEO_MODEL)
    log.info("Out dir   : %s", args.out_dir)
    log.info("Log file  : %s", dated)

    summary = {"ok": [], "skipped": [], "failed": [], "empty": []}
    for i, r in enumerate(rows, 1):
        sid = r["site_id"]
        prefix = f"[{i:>2}/{len(rows)}] {sid}"

        if not args.force:
            existing = _existing_file(args.out_dir, sid)
            if existing:
                log.info("%s SKIP (exists: %s)", prefix, existing.name)
                summary["skipped"].append(sid)
                continue

        start, end = _clamp_dates(r.get("start_date"), r.get("end_date"))
        try:
            lat = float(r["latitude"])
            lon = float(r["longitude"])
            data = _fetch_hourly(lat, lon, start, end)
            hourly = data.get("hourly") or {}
            times = hourly.get("time") or []
            precip = hourly.get("precipitation") or []
            if not times:
                log.warning("%s EMPTY (%s..%s)", prefix, start, end)
                summary["empty"].append(sid)
                continue
            out = _write_site_csv(args.out_dir, sid, times, precip)
            n_nonzero = sum(1 for p in precip if p)
            log.info("%s OK %d hrs (%s..%s) nonzero=%d -> %s",
                     prefix, len(times), start, end, n_nonzero, out.name)
            summary["ok"].append(sid)
        except Exception as exc:  # noqa: BLE001 - keep looping on per-site failures
            log.error("%s FAILED: %s", prefix, exc)
            summary["failed"].append(sid)

        time.sleep(args.sleep)

    log.info("=" * 60)
    log.info("DONE  ok=%d skipped=%d empty=%d failed=%d",
             len(summary["ok"]), len(summary["skipped"]),
             len(summary["empty"]), len(summary["failed"]))
    if summary["failed"]:
        log.warning("failed sites: %s", summary["failed"])
    if summary["empty"]:
        log.warning("empty sites: %s", summary["empty"])
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
