#!/usr/bin/env python3
"""
Merge downloaded hourly precipitation (Open-Meteo / CAMELS-H / Met-Office) into
the per-site parquet shards in data/clean and data/processed as a ``00045``
column (USGS precipitation pcode, mm).

Precip files live in data/raw/precipitation_data/ with schema
``datetime,precipitation_mm,date,year,month,day,hour``. The real hourly
resolution is carried by the year/month/day/hour columns (datetime is
date-only), so timestamps are reconstructed from those.

Site shards are typically 1-min cadence (NEON) with a tz-aware UTC index. Each
row is assigned the hourly precip total for its hour (floor-to-hour join). The
LSTM SiteBank resamples to hourly with ``.mean()``, so broadcasting the hourly
total across the 60 rows in that hour returns the hourly total after resampling
- consistent with how existing 00045 columns behave.

Idempotent: a shard that already carries 00045 is skipped unless ``--force``.
Column is added in place; files are NOT renamed (site discovery keys on the
target pcode + reads dynamic features by column name).

Author: TSLR
Date: 2026-07-15
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import stat
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GAP_CSV = REPO_ROOT / "outputs" / "reports" / "precip_gap_sites.csv"
PRECIP_DIR = REPO_ROOT / "data" / "raw" / "precipitation_data"
DEFAULT_DIRS = [REPO_ROOT / "data" / "processed", REPO_ROOT / "data" / "clean"]

PRECIP_COL = "00045"


def _setup_logging() -> Path:
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    current = logs_dir / "merge_precip_into_sites.log"
    dated = logs_dir / f"merge_precip_into_sites_{ts}.log"
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


log = logging.getLogger("merge_precip_into_sites")


def _load_precip_hourly(site_id: str) -> Optional[pd.Series]:
    """Return an hourly precip Series (UTC index, mm) for a site, or None."""
    hits = sorted(PRECIP_DIR.glob(f"{site_id}_precipitation_from_*.csv"))
    if not hits:
        return None
    # Prefer the most recently written file if several sources exist.
    path = max(hits, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(path, usecols=["precipitation_mm", "year", "month", "day", "hour"])
    ts = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]),
        utc=True,
    )
    s = pd.Series(df["precipitation_mm"].to_numpy(dtype=np.float64), index=ts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = PRECIP_COL
    return s


def _site_shards(site_id: str, data_dir: Path) -> list[Path]:
    out = []
    for p in sorted(data_dir.glob(f"{site_id}_*.parquet")):
        base = p.name
        if "all_sites" in base or base.endswith("_processed.parquet"):
            continue
        out.append(p)
    return out


def _merge_one(shard: Path, precip: pd.Series, force: bool) -> str:
    df = pd.read_parquet(shard)

    if PRECIP_COL in df.columns and not force:
        return "has_precip"

    # Resolve the datetime index.
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        idx_from_col = None
    else:
        idx_from_col = next((c for c in ("TIMESTAMP", "timestamp", "datetime")
                             if c in df.columns), None)
        if idx_from_col is None:
            return "no_timestamp"
        idx = pd.DatetimeIndex(pd.to_datetime(df[idx_from_col], utc=True))

    # Normalise to UTC tz-aware for a clean floor-to-hour join.
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    hour_key = idx.floor("h")
    mapped = precip.reindex(hour_key).to_numpy(dtype=np.float64)
    df[PRECIP_COL] = mapped.astype(np.float32)

    coverage = float(np.isfinite(mapped).mean())
    # DVC links shards to a read-only cache (hardlink/reflink), so write a temp
    # file and atomically replace the directory entry. This breaks the link for
    # this shard without mutating the shared cache blob.
    tmp = shard.with_name(shard.name + ".tmp")
    df.to_parquet(tmp, index=isinstance(df.index, pd.DatetimeIndex))
    try:
        os.chmod(shard, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass
    os.replace(tmp, shard)
    return f"merged(cov={coverage:.2f})"


def _load_sites(gap_csv: Path, only: Optional[list[str]]) -> list[str]:
    with open(gap_csv, newline="", encoding="utf-8") as fh:
        ids = [r["site_id"] for r in csv.DictReader(fh)]
    if only:
        ids = [s for s in ids if s in only]
    return ids


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gap-csv", type=Path, default=DEFAULT_GAP_CSV)
    p.add_argument("--sites", nargs="+", help="Restrict to these site_ids")
    p.add_argument("--dirs", nargs="+", type=Path, default=DEFAULT_DIRS,
                   help="Target dirs (default: data/processed data/clean)")
    p.add_argument("--force", action="store_true",
                   help="Re-merge even if 00045 already present")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    dated = _setup_logging()

    sites = _load_sites(args.gap_csv, args.sites)
    log.info("Sites to merge: %d", len(sites))
    log.info("Target dirs   : %s", [str(d) for d in args.dirs])
    log.info("Log file      : %s", dated)

    counts = {"merged": 0, "skipped_has": 0, "no_precip": 0,
              "no_shard": 0, "no_ts": 0, "failed": 0}

    for i, sid in enumerate(sites, 1):
        precip = _load_precip_hourly(sid)
        if precip is None:
            log.warning("[%d/%d] %s: no precip CSV found", i, len(sites), sid)
            counts["no_precip"] += 1
            continue

        any_shard = False
        for d in args.dirs:
            for shard in _site_shards(sid, d):
                any_shard = True
                try:
                    res = _merge_one(shard, precip, args.force)
                except Exception as exc:  # noqa: BLE001
                    log.error("[%d/%d] %s %s FAILED: %s",
                              i, len(sites), d.name, shard.name, exc)
                    counts["failed"] += 1
                    continue
                if res == "has_precip":
                    counts["skipped_has"] += 1
                    log.info("[%d/%d] %s/%s SKIP (already has 00045)",
                             i, len(sites), d.name, shard.name)
                elif res == "no_timestamp":
                    counts["no_ts"] += 1
                    log.warning("[%d/%d] %s/%s no timestamp column",
                                i, len(sites), d.name, shard.name)
                else:
                    counts["merged"] += 1
                    log.info("[%d/%d] %s/%s %s", i, len(sites), d.name, shard.name, res)
        if not any_shard:
            counts["no_shard"] += 1
            log.warning("[%d/%d] %s: no shards found in target dirs", i, len(sites), sid)

    log.info("=" * 60)
    log.info("DONE  merged=%d already=%d no_precip=%d no_shard=%d no_ts=%d failed=%d",
             counts["merged"], counts["skipped_has"], counts["no_precip"],
             counts["no_shard"], counts["no_ts"], counts["failed"])
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
