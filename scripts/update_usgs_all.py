#!/usr/bin/env python3
"""
Iterate USGS sites currently represented in data/clean/, fetch fresh data
from each site's last-known date to today via core/data/usgs_downloader.py.

For each USGS site, picks the clean file with the latest enddate and parses:
- site_id        (8-digit numeric prefix)
- parameters     (5-digit codes between site and date suffix)
- last_date      (second YYYYMMDD)

Calls usgs_downloader update-current per site, writing fresh raw chunks
to data/raw/ for downstream consolidation.

Skips sites already up to date. Continues on per-site failure; reports
summary at the end.

Author: TSLR
Date: 2026-06-09
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CLEAN_DIR = Path("data/clean")
RAW_DIR = Path("data/raw")
DOWNLOADER = Path("core/data/usgs_downloader.py")
PYTHON = Path(".venv/bin/python")

# USGS clean filename: <8-digit site>_<5-digit param>_..._<YYYYMMDD>_<YYYYMMDD>.csv
USGS_FILE_RE = re.compile(r"^(\d{8})_((?:\d{5}_)+)(\d{8})_(\d{8})\.csv$")


def collect_jobs():
    """Return dict of site_id -> (last_date_str_yyyymmdd, [param_codes]).

    If multiple files per site exist, picks the one with the latest enddate.
    """
    latest = {}
    for f in sorted(CLEAN_DIR.glob("*.csv")):
        m = USGS_FILE_RE.match(f.name)
        if not m:
            continue
        site_id, params_blob, _start, end = m.groups()
        params = [p for p in params_blob.rstrip("_").split("_") if len(p) == 5 and p.isdigit()]
        if not params:
            continue
        prev = latest.get(site_id)
        if prev is None or end > prev[0]:
            latest[site_id] = (end, params)
    return latest


def run_one(site_id, last_yyyymmdd, params, dry_run=False, save_raw=True):
    last_iso = f"{last_yyyymmdd[:4]}-{last_yyyymmdd[4:6]}-{last_yyyymmdd[6:8]}"
    cmd = [
        str(PYTHON),
        str(DOWNLOADER),
        "update-current",
        "--site-id",
        site_id,
        "--last-date",
        last_iso,
        "--parameters",
        *params,
        "--tsv-csv-output",
        str(RAW_DIR),
    ]
    if save_raw:
        cmd.append("--save-raw")
    if dry_run:
        print(f"  DRY-RUN: {' '.join(cmd)}")
        return True, "dry-run"
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            return True, ""
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        return False, " | ".join(tail)[:300]
    except subprocess.TimeoutExpired:
        return False, "timeout (10 min)"
    except Exception as e:
        return False, f"exception: {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="Print commands, don't execute")
    parser.add_argument("--limit", type=int, help="Only process the first N sites (for testing)")
    args = parser.parse_args()

    if not CLEAN_DIR.exists():
        print(f"ERROR: {CLEAN_DIR} does not exist. Run from project root.")
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs()
    if args.limit:
        jobs = dict(list(jobs.items())[: args.limit])

    today_yyyymmdd = datetime.now().strftime("%Y%m%d")
    print(f"USGS update: {len(jobs)} sites detected in {CLEAN_DIR}")
    print(f"Today: {today_yyyymmdd}")
    print()

    ok, skipped, failures = 0, 0, []
    for i, (site_id, (end, params)) in enumerate(sorted(jobs.items()), 1):
        if end >= today_yyyymmdd:
            print(f"[{i}/{len(jobs)}] {site_id}: up to date ({end})")
            skipped += 1
            continue
        print(f"[{i}/{len(jobs)}] {site_id}: {end} → today ({len(params)} params)", flush=True)
        success, err = run_one(site_id, end, params, dry_run=args.dry_run)
        if success:
            ok += 1
        else:
            failures.append((site_id, err))
            print(f"   FAIL: {err}")

    print()
    print(f"Summary: {ok} updated, {skipped} already current, {len(failures)} failed")
    if failures:
        print("First failures:")
        for s, e in failures[:10]:
            print(f"  {s}: {e}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
