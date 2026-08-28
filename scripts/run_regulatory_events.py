#!/usr/bin/env python3
"""
Batch regulatory threshold event detection across all clean data files.

Deduplicates by selecting the latest file per site (by end-date in filename),
then runs detect_regulatory_events on each and writes a combined events CSV.

Usage:
    .venv/bin/python scripts/run_regulatory_events.py
    .venv/bin/python scripts/run_regulatory_events.py --out data/regulatory_events.csv

Author: TSLR
Date: 2026-06-23
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tools.regulatory_threshold_detector import detect_regulatory_events

CLEAN_DIR = Path("/Users/tslr/Projects/work/data/pollution_id/data/clean")
DEFAULT_OUT = Path("/Users/tslr/Projects/work/data/pollution_id/data/processed/regulatory_events.csv")
LOG_DIR = Path("core/logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def latest_file_per_site(clean_dir: Path) -> list[Path]:
    """
    For sites with multiple files, keep only the one with the latest end-date
    encoded in the filename (penultimate underscore-delimited token before .csv).
    Files without a date token are kept as-is.
    """
    from collections import defaultdict
    import re

    groups: dict = defaultdict(list)

    for p in sorted(clean_dir.glob("*.csv")):
        site_id = p.stem.split("_")[0]
        # Extract end-date: last 8-digit token in stem
        dates = re.findall(r"(\d{8})", p.stem)
        end_date = dates[-1] if len(dates) >= 2 else "00000000"
        groups[site_id].append((end_date, p))

    latest = []
    for site_id, entries in groups.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        latest.append(entries[0][1])

    return sorted(latest)


def run(clean_dir: Path, out_path: Path, min_steps: int, gap_tol: int) -> pd.DataFrame:
    files = latest_file_per_site(clean_dir)
    log.info(f"{len(files)} unique sites found (from {len(list(clean_dir.glob('*.csv')))} total files)")

    all_events = []
    skipped = 0
    no_events = 0

    for i, fpath in enumerate(files, 1):
        site_id = fpath.stem.split("_")[0]
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            log.warning(f"[{i}/{len(files)}] {site_id} — could not read: {e}")
            skipped += 1
            continue

        # Resolve timestamp column
        ts_col = "TIMESTAMP" if "TIMESTAMP" in df.columns else (
            "timestamp" if "timestamp" in df.columns else None
        )
        if ts_col is None:
            log.warning(f"[{i}/{len(files)}] {site_id} — no timestamp column, skipping")
            skipped += 1
            continue

        try:
            events = detect_regulatory_events(
                df,
                site_id=site_id,
                timestamp_col=ts_col,
                min_duration_steps=min_steps,
                gap_tolerance_steps=gap_tol,
            )
        except Exception as e:
            log.warning(f"[{i}/{len(files)}] {site_id} — detection error: {e}")
            skipped += 1
            continue

        if events.empty:
            no_events += 1
        else:
            all_events.append(events)
            log.info(f"[{i}/{len(files)}] {site_id}  →  {len(events)} event(s)")

    if not all_events:
        log.info("No events found across any site.")
        return pd.DataFrame()

    combined = pd.concat(all_events, ignore_index=True).sort_values(
        ["site_id", "incident_start"]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    log.info(f"\n{'='*60}")
    log.info(f"  Total events : {len(combined)}")
    log.info(f"  Sites with events: {combined['site_id'].nunique()}")
    log.info(f"  Sites with no events: {no_events}")
    log.info(f"  Skipped (errors): {skipped}")
    log.info(f"  Output: {out_path}")
    log.info(f"{'='*60}\n")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch regulatory event detection over clean data.")
    parser.add_argument("--clean-dir", default=str(CLEAN_DIR), help="Path to clean data directory")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path")
    parser.add_argument("--min-steps", type=int, default=2, help="Min consecutive flagged steps (default 2)")
    parser.add_argument("--gap-tolerance", type=int, default=4, help="Gap tolerance steps (default 4)")
    args = parser.parse_args()

    run(
        clean_dir=Path(args.clean_dir),
        out_path=Path(args.out),
        min_steps=args.min_steps,
        gap_tol=args.gap_tolerance,
    )
