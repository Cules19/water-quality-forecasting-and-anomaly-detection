#!/usr/bin/env python3
"""
Pull the full NEON AIS archive for every multisonde site, staged
(download-raw → transform), into data/raw/neon/ and data/clean/.

Per-site loop with skip-if-output-exists, so reruns only pick up missing sites
and the script is restart-safe after network blips.

Writes two logs per the project logging convention:
    logs/pull_neon_archive.log         (current)
    logs/pull_neon_archive_<ts>.log    (archive)

Author: TSLR
Date: 2026-06-25, refactored 2026-06-27
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.data.neon_downloader import NEONDownloader  # noqa: E402

try:
    from config import Config
    DEFAULT_OUTPUT = Path(Config.DATA_DIR) / "clean"
except Exception:
    DEFAULT_OUTPUT = REPO_ROOT / "data" / "clean"


def _setup_logging() -> Path:
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    current = logs_dir / "pull_neon_archive.log"
    dated = logs_dir / f"pull_neon_archive_{ts}.log"
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(current, mode="w"),
            logging.FileHandler(dated, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return dated


def _site_has_clean_output(site: str, clean_dir: Path) -> bool:
    """True if at least one normalised file for this site already exists."""
    return any(clean_dir.glob(f"N{site}S*_*_*_*.csv"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-dir", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Clean output dir (default: {DEFAULT_OUTPUT})")
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="Raw root (default: Config.DATA_DIR/raw/neon)")
    p.add_argument("--start-date", default="2017-01-01",
                   help="Earliest date to ask for (default: 2017-01-01)")
    p.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"),
                   help="Latest date to ask for (default: today)")
    p.add_argument("--products", nargs="+",
                   default=["DP1.20288.001", "DP1.20033.001", "DP1.20053.001"],
                   help="NEON product codes (default: multisonde + SUNA nitrate + surface water temp)")
    p.add_argument("--package", choices=["basic", "expanded"], default="expanded")
    p.add_argument("--sites", nargs="+",
                   help="Restrict to listed sites (default: all 34 multisonde sites)")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if clean output already exists for a site")
    p.add_argument("--skip-transform", action="store_true",
                   help="Stop after raw step (download only, no clean output)")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Sleep seconds between sites to be polite (default: 0.5)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    dated_log = _setup_logging()
    log = logging.getLogger(__name__)

    args.clean_dir.mkdir(parents=True, exist_ok=True)
    log.info("Clean dir : %s", args.clean_dir)
    log.info("Raw dir   : %s", args.raw_dir or "(default Config.DATA_DIR/raw/neon)")
    log.info("Date range: %s .. %s", args.start_date, args.end_date)
    log.info("Products  : %s", args.products)
    log.info("Package   : %s", args.package)
    log.info("Stage     : %s", "download-raw only" if args.skip_transform else "raw + transform")
    log.info("Log file  : %s", dated_log)

    dl = NEONDownloader(package_kind=args.package)

    if args.sites:
        sites = args.sites
        log.info("Site override: %d sites", len(sites))
    else:
        site_meta = dl.list_aquatic_sites()
        sites = sorted(site_meta["siteCode"].tolist())
        log.info("Multisonde sites discovered: %d", len(sites))

    summary = {"ok": [], "skipped": [], "failed": [], "empty": []}
    total_bytes = 0
    started = datetime.now()

    for i, site in enumerate(sites, 1):
        site_start = datetime.now()
        prefix = f"[{i:>2}/{len(sites)}] {site}"

        if not args.force and _site_has_clean_output(site, args.clean_dir):
            log.info("%s SKIP (clean output already present)", prefix)
            summary["skipped"].append(site)
            continue

        try:
            raw_paths = dl.download_raw_site_range(
                site, args.start_date, args.end_date, args.products, args.raw_dir,
            )
            if not raw_paths:
                log.warning("%s EMPTY (no raw files fetched)", prefix)
                summary["empty"].append(site)
                continue

            if args.skip_transform:
                site_bytes = sum(p.stat().st_size for p in raw_paths)
                total_bytes += site_bytes
                log.info("%s RAW OK (%d files, %.1f MB, %.1fs)",
                         prefix, len(raw_paths), site_bytes / 1e6,
                         (datetime.now() - site_start).total_seconds())
                summary["ok"].append(site)
                time.sleep(args.sleep)
                continue

            frames = dl.transform_site_range(site, args.raw_dir, args.package)
            if not frames:
                log.warning("%s TRANSFORM EMPTY", prefix)
                summary["empty"].append(site)
                continue
            clean_paths = dl.save_clean(frames, args.clean_dir)
            site_bytes = sum(p.stat().st_size for p in clean_paths)
            total_bytes += site_bytes
            log.info("%s OK (%d clean files, %.1f MB, %.1fs)",
                     prefix, len(clean_paths), site_bytes / 1e6,
                     (datetime.now() - site_start).total_seconds())
            summary["ok"].append(site)
        except Exception as e:
            log.exception("%s FAILED: %s", prefix, e)
            summary["failed"].append(site)

        time.sleep(args.sleep)

    elapsed = (datetime.now() - started).total_seconds()
    log.info("=" * 70)
    log.info("Done in %.0fs. OK=%d  SKIPPED=%d  EMPTY=%d  FAILED=%d",
             elapsed, len(summary["ok"]), len(summary["skipped"]),
             len(summary["empty"]), len(summary["failed"]))
    log.info("Total bytes written this run: %.1f MB", total_bytes / 1e6)
    if summary["failed"]:
        log.error("Failed sites: %s", ", ".join(summary["failed"]))
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
