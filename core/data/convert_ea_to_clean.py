#!/usr/bin/env python3
"""
EA → clean conversion

Reads raw EA fetches from data/ea_compatible_format/ and writes USGS-coded
files into data/clean/ following the project naming convention:

    {site_id}_{usgs_codes_sorted}_{start_yyyymmdd}_{end_yyyymmdd}.csv

Header:
    TIMESTAMP, <usgs_code>, ..., Site_ID

Fills the gap noted in scripts/update_data.sh: no surviving script previously
converted EA → clean. The legacy data/clean_ea_temp/ output was last refreshed
2025-10-17 and is left in place untouched.

Usage:
    python core/data/convert_ea_to_clean.py
    python core/data/convert_ea_to_clean.py --site-id E96999A
    python core/data/convert_ea_to_clean.py --dry-run

Author: TSLR
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data.parameters_ea_usgs_toolkit import EA_TO_USGS_PARAMETERS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EA_LOWER_TO_CODE = {ea.lower(): info["usgs_code"] for ea, info in EA_TO_USGS_PARAMETERS.items()}

RAW_DIR = Path("data/ea_compatible_format")
CLEAN_DIR = Path("data/clean")


def latest_raw_per_site(raw_dir: Path) -> dict:
    """Return {site_id: newest_csv_path} based on the end-date suffix in the filename."""
    by_site: dict[str, tuple[str, Path]] = {}
    for p in raw_dir.glob("*.csv"):
        stem = p.stem
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        site_id, end_date = parts[0], parts[-1]
        key = (end_date, p)
        prev = by_site.get(site_id)
        if prev is None or key > prev:
            by_site[site_id] = key
    return {sid: pth for sid, (_, pth) in by_site.items()}


def convert_one(raw_path: Path, clean_dir: Path, dry_run: bool = False) -> tuple[Path, int] | None:
    """Convert one EA raw CSV to a clean CSV. Returns (clean_path, n_rows) or None on skip."""
    df = pd.read_csv(raw_path, parse_dates=["TIMESTAMP"])
    if df.empty:
        logger.warning(f"  empty: {raw_path.name} — skipping")
        return None

    rename = {}
    unmapped = []
    for col in df.columns:
        if col in ("TIMESTAMP", "Site_ID"):
            continue
        code = EA_LOWER_TO_CODE.get(col.lower())
        if code:
            rename[col] = code
        else:
            unmapped.append(col)

    if unmapped:
        logger.warning(f"  {raw_path.name}: unmapped EA params dropped: {unmapped}")

    df = df.rename(columns=rename).drop(columns=unmapped)
    usgs_cols = sorted(c for c in df.columns if c not in ("TIMESTAMP", "Site_ID"))
    if not usgs_cols:
        logger.warning(f"  {raw_path.name}: no mapped parameters — skipping")
        return None

    df = df[["TIMESTAMP"] + usgs_cols + ["Site_ID"]]
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)

    site_id = str(df["Site_ID"].iloc[0])
    start = df["TIMESTAMP"].min().strftime("%Y%m%d")
    end = df["TIMESTAMP"].max().strftime("%Y%m%d")
    codes_str = "_".join(usgs_cols)
    out_name = f"{site_id}_{codes_str}_{start}_{end}.csv"
    out_path = clean_dir / out_name

    if dry_run:
        logger.info(f"  DRY-RUN would write: {out_path}  rows={len(df)}")
        return out_path, len(df)

    df.to_csv(out_path, index=False)
    return out_path, len(df)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--site-id", help="Convert one site only")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Source raw EA dir")
    parser.add_argument("--clean-dir", default=str(CLEAN_DIR), help="Target clean dir")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be written without writing"
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    clean_dir = Path(args.clean_dir)
    if not raw_dir.is_dir():
        logger.error(f"Raw dir not found: {raw_dir}")
        sys.exit(1)
    clean_dir.mkdir(parents=True, exist_ok=True)

    by_site = latest_raw_per_site(raw_dir)
    if args.site_id:
        if args.site_id not in by_site:
            logger.error(f"No raw file for site {args.site_id} in {raw_dir}")
            sys.exit(1)
        by_site = {args.site_id: by_site[args.site_id]}

    logger.info(
        f"Converting {len(by_site)} EA sites: {raw_dir} → {clean_dir} (dry_run={args.dry_run})"
    )
    n_ok = n_skip = total_rows = 0
    for i, (site_id, path) in enumerate(sorted(by_site.items()), 1):
        if i % 100 == 0 or i == 1:
            logger.info(f"  [{i}/{len(by_site)}] {site_id}")
        try:
            result = convert_one(path, clean_dir, dry_run=args.dry_run)
        except Exception as e:
            logger.exception(f"  {site_id}: failed — {e}")
            n_skip += 1
            continue
        if result is None:
            n_skip += 1
        else:
            n_ok += 1
            total_rows += result[1]

    logger.info(f"Done. ok={n_ok}, skipped={n_skip}, total_rows={total_rows:,}")


if __name__ == "__main__":
    main()
