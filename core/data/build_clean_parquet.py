#!/usr/bin/env python3
"""
Build parquet mirrors next to each CSV in data/clean/.

Partners who don't use DVC asked for parquet alongside the canonical CSVs.
Each clean/X.csv gets a sibling clean/X.parquet (snappy compression, TIMESTAMP
parsed as datetime64[ns]). Existing parquets are only rebuilt if older than the
source CSV.

Usage:
    python core/data/build_clean_parquet.py
    python core/data/build_clean_parquet.py --force
    python core/data/build_clean_parquet.py --clean-dir data/clean

Author: TSLR
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_one(csv_path: Path, force: bool) -> tuple[bool, int]:
    """Return (wrote, n_rows). Skips if parquet is newer than csv unless force."""
    parquet_path = csv_path.with_suffix(".parquet")
    if parquet_path.exists() and not force:
        if parquet_path.stat().st_mtime >= csv_path.stat().st_mtime:
            return False, 0

    df = pd.read_csv(csv_path, parse_dates=["TIMESTAMP"])
    df.to_parquet(parquet_path, compression="snappy", index=False)
    return True, len(df)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clean-dir", default="data/clean", help="Directory of clean CSVs")
    parser.add_argument("--force", action="store_true", help="Rebuild parquet even when up-to-date")
    parser.add_argument("--limit", type=int, help="Limit to first N files (testing)")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    if not clean_dir.is_dir():
        logger.error(f"Clean dir not found: {clean_dir}")
        sys.exit(1)

    csvs = sorted(clean_dir.glob("*.csv"))
    if args.limit:
        csvs = csvs[: args.limit]
    logger.info(f"Found {len(csvs)} CSVs in {clean_dir}")

    n_wrote = n_skip = total_rows = 0
    n_err = 0
    for i, csv_path in enumerate(csvs, 1):
        if i % 200 == 0 or i == 1:
            logger.info(f"  [{i}/{len(csvs)}] {csv_path.name}")
        try:
            wrote, n_rows = build_one(csv_path, args.force)
        except Exception as e:
            logger.exception(f"  {csv_path.name}: failed — {e}")
            n_err += 1
            continue
        if wrote:
            n_wrote += 1
            total_rows += n_rows
        else:
            n_skip += 1

    logger.info(
        f"Done. wrote={n_wrote}, up_to_date={n_skip}, errors={n_err}, total_rows={total_rows:,}"
    )


if __name__ == "__main__":
    main()
