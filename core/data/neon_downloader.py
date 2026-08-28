#!/usr/bin/env python3
"""
NEON AIS Downloader — multi-parameter HF water-quality from the NSF NEON network.

Implements the staged-pipeline contract from
`documentation/MULTI_SOURCE_INGEST_DESIGN.md` (revised 2026-06-27):

    Step 1 download-raw  :  API  →  data/raw/neon/<site>/<YYYY-MM>/<orig-filename>.csv
    Step 2 transform     :  data/raw/neon/  →  data/clean/<Site_ID>_<pcodes>_<dates>.csv
    Step 3 preprocess    :  data/clean/  →  data/processed/  (out of scope here)

Site_ID convention (compact, no underscores): ``N<SITE><POS>``
    e.g. NPRINS1, NBLDES1, NBIGCS11 (relocated install), NBLUES12.
    Where SITE is NEON's 4-letter site code and POS is "S" + (HOR - 100).

Clean-file column shape per parameter:
    <pcode>      value, in source units
    <pcode>_qc   FinalQF rollup (NaN where source has no flag)
    <pcode>_unc  ExpUncert (NaN where source doesn't provide it)

Data products covered:
    DP1.20288.001  Water Quality multisonde  — 1-min cadence
                   (SpC, DO, pH, chlorophyll, turbidity, fDOM, depth, …)
    DP1.20033.001  SUNA surface-water nitrate — 15-min cadence
    DP1.20053.001  Surface water temperature  — 5-min cadence

Pseudo-codes (see data/reference/usgs_parameter_mapping.json):
    pid_00301_sealevel   NEON seaLevelDissolvedOxygenSat (variant of USGS 00301)
    pid_32211_rfu        NEON chlaRelativeFluorescence   (variant of USGS 32211)

Raw NEON files are kept verbatim — every column, no renaming. The granular QC
flags, science-review flags, and any other context columns NEON ships in the
expanded package survive in data/raw/neon/.

Author: TSLR
Date: 2026-06-24, refactored 2026-06-27
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import pandas as pd
import requests

logger = logging.getLogger(__name__)


NEON_BASE = "https://data.neonscience.org/api/v0"

# Errors worth retrying — network transients, server-side 5xx, peer resets.
_RETRYABLE_EXC = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    ConnectionResetError,
)

T = TypeVar("T")


def _retry(op: Callable[[], T], attempts: int = 4, base_delay: float = 2.0,
           label: str = "") -> T:
    """Run `op` with exponential backoff. base_delay doubles per attempt
    (2s, 4s, 8s by default). Logs each failed attempt.
    """
    last_exc: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return op()
        except _RETRYABLE_EXC as e:
            last_exc = e
            if i + 1 < attempts:
                wait = base_delay * (2 ** i)
                logger.warning("%s retry %d/%d after %.0fs (%s)",
                               label or "HTTP", i + 1, attempts - 1, wait, e)
                time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            # Only retry 5xx; 4xx is the caller's fault.
            status = getattr(e.response, "status_code", None)
            if status and 500 <= status < 600 and i + 1 < attempts:
                wait = base_delay * (2 ** i)
                logger.warning("%s retry %d/%d after %.0fs (HTTP %d)",
                               label or "HTTP", i + 1, attempts - 1, wait, status)
                time.sleep(wait)
                last_exc = e
            else:
                raise
    raise last_exc  # type: ignore[misc]


@dataclass(frozen=True)
class NeonParameter:
    """One NEON value column → USGS pcode binding."""

    neon_col: str          # value column in the NEON CSV
    usgs_code: str         # USGS pcode used as our column name
    units: str             # source units (kept as-is, per the design)
    description: str
    # NEON's per-param column names. Empty templates mean "fixed name" (e.g. SUNA).
    qc_col_template: str   # e.g. "{n}FinalQF" or "finalQF"
    unc_col_template: str  # e.g. "{n}ExpUncert" or "surfWaterNitrateExpUncert"


# DP1.20288.001 water-quality multisonde — value + ExpUncert + many granular QFs
# + <name>FinalQF rollup. We expose value + FinalQF + ExpUncert.
NEON_MULTISONDE: List[NeonParameter] = [
    NeonParameter("specificConductance",        "00095",            "uS/cm",     "Specific conductance",                 "specificCondFinalQF",     "{n}ExpUncert"),
    NeonParameter("dissolvedOxygen",            "00300",            "mg/L",      "Dissolved oxygen",                     "{n}FinalQF",              "{n}ExpUncert"),
    NeonParameter("localDissolvedOxygenSat",    "00301",            "%",         "Dissolved oxygen saturation (local)",  "localDOSatFinalQF",       "localDOSatExpUncert"),
    NeonParameter("seaLevelDissolvedOxygenSat", "pid_00301_sealevel","%",        "Dissolved oxygen saturation (sea-level corrected)", "seaLevelDOSatFinalQF", "seaLevelDOSatExpUncert"),
    NeonParameter("pH",                         "00400",            "std units", "pH",                                   "{n}FinalQF",              "{n}ExpUncert"),
    NeonParameter("chlorophyll",                "32211",            "ug/L",      "Chlorophyll-a (fluorometric)",         "{n}FinalQF",              "{n}ExpUncert"),
    NeonParameter("chlaRelativeFluorescence",   "pid_32211_rfu",    "RFU",       "Chlorophyll-a relative fluorescence",  "chlaRelFluoroFinalQF",    "chlaRelFluoroExpUncert"),
    NeonParameter("turbidity",                  "63680",            "FNU",       "Turbidity (formazin nephelometric)",   "{n}FinalQF",              "{n}ExpUncert"),
    NeonParameter("fDOM",                       "32295",            "QSU",       "Fluorescent dissolved organic matter", "{n}FinalQF",              "{n}ExpUncert"),
    NeonParameter("sensorDepth",                "00065",            "m",         "Sensor depth (gage height analogue)",  "{n}FinalQF",              "{n}ExpUncert"),
]

# DP1.20033.001 SUNA nitrate — single-parameter table, plain "finalQF" rollup.
NEON_NITRATE: List[NeonParameter] = [
    NeonParameter("surfWaterNitrateMean", "99133", "mg/L as N",
                  "Nitrate, in-situ surface water (SUNA mean)",
                  "finalQF", "surfWaterNitrateExpUncert"),
]

# DP1.20053.001 surface water temperature — 5-min cadence, plain "finalQF".
# Separate from multisonde: the WQ multisonde DP1.20288.001 has no water-temp
# value column (only fDOMTempQF for QC purposes).
NEON_TEMPERATURE: List[NeonParameter] = [
    NeonParameter("surfWaterTempMean", "00010", "degC",
                  "Temperature, water (5-min mean)",
                  "finalQF", "surfWaterTempExpUncert"),
]

# Files per product — drives discovery in NEON packages.
PRODUCT_SPEC = {
    "DP1.20288.001": {"file_token": "waq_instantaneous", "params": NEON_MULTISONDE},
    "DP1.20033.001": {"file_token": "NSW_15_minute",     "params": NEON_NITRATE},
    "DP1.20053.001": {"file_token": "TSW_5min",          "params": NEON_TEMPERATURE},
}


# --------------------------------------------------------------------------- helpers


def _position_label(filename: str) -> str:
    """Map NEON filename position code to S1/S2/S11/...

    NEON filenames look like
    NEON.D11.PRIN.DP1.20288.001.101.100.100.waq_instantaneous.2024-06.basic…csv
    The .101. / .102. / .111. segment is the horizontal position code (HOR).
    """
    parts = filename.split(".")
    for p in parts:
        if p.isdigit() and len(p) == 3 and p.startswith("1"):
            n = int(p) - 100
            return f"S{n}"
    return "S0"


def _site_id(site: str, position: str) -> str:
    """Compact Site_ID per the design note: N<SITE><POS> → e.g. NPRINS1."""
    return f"N{site}{position}"


def _resolve_col(value_col: str, template: str) -> str:
    """Resolve a per-param column name template."""
    return template.format(n=value_col)


def _month_range(start: str, end: str) -> List[str]:
    """List YYYY-MM strings inclusive between two YYYY-MM-DD dates."""
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(day=1)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(day=1)
    out = []
    cur = start_dt
    while cur <= end_dt:
        out.append(cur.strftime("%Y-%m"))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return out


def _default_raw_dir() -> Path:
    try:
        from config import Config
        return Path(Config.DATA_DIR) / "raw" / "neon"
    except Exception:
        return Path(__file__).resolve().parents[2] / "data" / "raw" / "neon"


def _default_clean_dir() -> Path:
    try:
        from config import Config
        return Path(Config.DATA_DIR) / "clean"
    except Exception:
        return Path(__file__).resolve().parents[2] / "data" / "clean"


# ============================================================================ class


class NEONDownloader:
    """NEON AIS adapter implementing the staged-pipeline contract."""

    def __init__(self, timeout: int = 120, user_agent: str = "Pollution_ID/1.0",
                 package_kind: str = "expanded",
                 api_token: Optional[str] = None):
        if package_kind not in ("basic", "expanded"):
            raise ValueError("package_kind must be 'basic' or 'expanded'")
        self.timeout = timeout
        self.package_kind = package_kind
        self.session = requests.Session()
        headers = {"User-Agent": user_agent, "Accept": "application/json"}
        # NEON's /data/ endpoint began rejecting anonymous requests with 403
        # in mid-2026. Token unlocks it and raises the rate limit 200 -> 2000.
        token = api_token or os.environ.get("NEON_API_TOKEN")
        if token:
            headers["X-API-Token"] = token
            logger.info("NEON API token loaded (%s)",
                        "argument" if api_token else "NEON_API_TOKEN env")
        else:
            logger.warning("No NEON API token set — /data/ endpoint will 403. "
                           "Set NEON_API_TOKEN or pass api_token=…")
        self.session.headers.update(headers)

    # --------------------------------------------------------------- discovery

    def list_aquatic_sites(self) -> pd.DataFrame:
        """Return one row per NEON site that carries the WQ multisonde product."""
        r = self.session.get(f"{NEON_BASE}/products/DP1.20288.001", timeout=self.timeout)
        r.raise_for_status()
        site_codes = [s["siteCode"] for s in r.json()["data"]["siteCodes"]]
        rs = self.session.get(f"{NEON_BASE}/sites", timeout=self.timeout)
        rs.raise_for_status()
        meta = {s["siteCode"]: s for s in rs.json()["data"]}
        rows = []
        for code in sorted(site_codes):
            m = meta.get(code, {})
            rows.append({
                "siteCode":      code,
                "siteName":      m.get("siteName"),
                "siteType":      m.get("siteType"),
                "siteLatitude":  m.get("siteLatitude"),
                "siteLongitude": m.get("siteLongitude"),
                "domainCode":    m.get("domainCode"),
                "stateCode":     m.get("stateCode"),
            })
        return pd.DataFrame(rows)

    # ====================================================== STEP 1: download-raw

    def _get_package(self, product: str, site: str, month: str) -> dict:
        url = f"{NEON_BASE}/data/{product}/{site}/{month}"

        def _do() -> dict:
            r = self.session.get(url, timeout=self.timeout)
            if r.status_code == 404:
                return {}
            r.raise_for_status()
            return r.json()["data"]

        return _retry(_do, label=f"package {product}/{site}/{month}")

    def _csv_files_for_product(self, package: dict, file_token: str) -> List[dict]:
        """Filter the package to data CSVs for our product+package_kind."""
        return [
            f for f in package.get("files", [])
            if f["name"].endswith(".csv")
            and file_token in f["name"]
            and self.package_kind in f["name"]
        ]

    def _save_raw_file(self, file_entry: dict, raw_root: Path,
                       site: str, month: str) -> Optional[Path]:
        """Download one NEON CSV verbatim to data/raw/neon/<site>/<month>/."""
        target_dir = raw_root / site / month
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / file_entry["name"]
        if target.exists() and target.stat().st_size > 0:
            return target  # already downloaded — restart-safe

        def _do() -> Path:
            r = self.session.get(file_entry["url"], timeout=self.timeout)
            r.raise_for_status()
            target.write_bytes(r.content)
            return target

        return _retry(_do, label=f"csv {file_entry['name']}")

    def download_raw_site_month(
        self,
        site: str,
        month: str,
        products: Optional[List[str]] = None,
        raw_root: Optional[Path] = None,
    ) -> List[Path]:
        """Step 1: pull every CSV for a site-month verbatim into data/raw/neon/.

        Returns the list of paths written (may include files that already
        existed, in which case they are not re-downloaded).
        """
        products = products or list(PRODUCT_SPEC)
        raw_root = raw_root or _default_raw_dir()
        written: List[Path] = []
        for product in products:
            spec = PRODUCT_SPEC[product]
            pkg = self._get_package(product, site, month)
            if not pkg:
                logger.info("No %s package for %s/%s", product, site, month)
                continue
            entries = self._csv_files_for_product(pkg, spec["file_token"])
            for entry in entries:
                logger.info("RAW pull %s/%s product=%s file=%s",
                            site, month, product, entry["name"])
                p = self._save_raw_file(entry, raw_root, site, month)
                if p:
                    written.append(p)
        return written

    def download_raw_site_range(
        self,
        site: str,
        start_date: str,
        end_date: str,
        products: Optional[List[str]] = None,
        raw_root: Optional[Path] = None,
    ) -> List[Path]:
        """Iterate months in [start, end] and persist every raw NEON CSV."""
        out: List[Path] = []
        for month in _month_range(start_date, end_date):
            out.extend(self.download_raw_site_month(site, month, products, raw_root))
        return out

    # =================================================== STEP 2: transform raw

    def _normalise_one(self, df: pd.DataFrame, params: List[NeonParameter],
                       site: str, position: str) -> pd.DataFrame:
        """One raw NEON CSV → contract-shaped DataFrame.

        Picks startDateTime as TIMESTAMP; for each NEON value column emits
        a value column (renamed to USGS pcode) plus _qc and _unc siblings.
        """
        ts = pd.to_datetime(df["startDateTime"], utc=True)
        out_cols: Dict[str, pd.Series] = {}
        for p in params:
            if p.neon_col not in df.columns:
                logger.warning("Column %s missing for %s/%s — skipped",
                               p.neon_col, site, position)
                continue
            out_cols[p.usgs_code] = pd.to_numeric(df[p.neon_col], errors="coerce")
            qc_name = _resolve_col(p.neon_col, p.qc_col_template)
            if qc_name in df.columns:
                out_cols[f"{p.usgs_code}_qc"] = df[qc_name]
            unc_name = _resolve_col(p.neon_col, p.unc_col_template)
            if unc_name in df.columns:
                out_cols[f"{p.usgs_code}_unc"] = pd.to_numeric(df[unc_name], errors="coerce")
        if not out_cols:
            return pd.DataFrame()
        result = pd.DataFrame({"TIMESTAMP": ts, **out_cols})
        result.insert(0, "Site_ID", _site_id(site, position))
        return result

    @staticmethod
    def _classify_raw_file(path: Path) -> Tuple[Optional[str], str]:
        """Return (product_code, position) for a NEON raw filename."""
        name = path.name
        for product, spec in PRODUCT_SPEC.items():
            if spec["file_token"] in name and "expanded" in name:
                return product, _position_label(name)
            if spec["file_token"] in name and "basic" in name:
                return product, _position_label(name)
        return None, _position_label(name)

    def transform_site_range(
        self,
        site: str,
        raw_root: Optional[Path] = None,
        package_kind: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Step 2: read every raw file for `site`, return one wide DataFrame per
        sonde position (S1, S2, S11, …) keyed by position label.
        """
        raw_root = raw_root or _default_raw_dir()
        pkg = package_kind or self.package_kind
        site_root = raw_root / site
        if not site_root.exists():
            logger.warning("No raw dir for site %s at %s", site, site_root)
            return {}

        per_position: Dict[str, Dict[str, List[pd.DataFrame]]] = {}
        for path in sorted(site_root.rglob("*.csv")):
            if pkg not in path.name:
                continue
            product, position = self._classify_raw_file(path)
            if product is None:
                continue
            spec = PRODUCT_SPEC[product]
            try:
                raw = pd.read_csv(path, low_memory=False)
            except Exception as e:
                logger.error("Could not read %s: %s", path, e)
                continue
            norm = self._normalise_one(raw, spec["params"], site, position)
            if norm.empty:
                continue
            per_position.setdefault(position, {}).setdefault(product, []).append(norm)

        # For each position, concat its per-product frames then outer-merge products.
        out: Dict[str, pd.DataFrame] = {}
        for position, prod_map in per_position.items():
            merged = None
            for product, frames in prod_map.items():
                df = pd.concat(frames, ignore_index=True).sort_values("TIMESTAMP")
                df = df.drop_duplicates(subset=["TIMESTAMP"], keep="first")
                merged = df if merged is None else merged.merge(
                    df, on=["Site_ID", "TIMESTAMP"], how="outer",
                )
            if merged is not None and not merged.empty:
                out[position] = merged.sort_values("TIMESTAMP").reset_index(drop=True)
        return out

    # ------------------------------------------------------------------- save

    def save_clean(self, frames: Dict[str, pd.DataFrame],
                   output_dir: Optional[Path] = None,
                   write_tsv: bool = False) -> List[Path]:
        """Write one CSV per sonde position. Filename:

            N<SITE><POS>_<pcodes>_<startYYYYMMDD>_<endYYYYMMDD>.csv

        TSV mirror is off by default — raw NEON CSV under data/raw/neon/ is
        already the human-friendly source-of-truth.
        """
        output_dir = output_dir or _default_clean_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for position, df in frames.items():
            if df.empty:
                continue
            site_no = df["Site_ID"].iloc[0]      # e.g. "NPRINS1"
            start = df["TIMESTAMP"].min().strftime("%Y%m%d")
            end = df["TIMESTAMP"].max().strftime("%Y%m%d")
            pcodes = sorted(
                c for c in df.columns
                if c not in ("Site_ID", "TIMESTAMP")
                and not c.endswith("_qc")
                and not c.endswith("_unc")
            )
            slug = "_".join(pcodes) if pcodes else "data"

            ordered: List[str] = ["Site_ID", "TIMESTAMP"]
            for code in pcodes:
                ordered.append(code)
                for suffix in ("_qc", "_unc"):
                    col = f"{code}{suffix}"
                    if col in df.columns:
                        ordered.append(col)

            base = f"{site_no}_{slug}_{start}_{end}"
            csv_path = output_dir / f"{base}.csv"
            df[ordered].to_csv(csv_path, index=False)
            if write_tsv:
                df[ordered].to_csv(output_dir / f"{base}.tsv", sep="\t", index=False)
            logger.info("CLEAN write %s (%d rows, %d pcodes)",
                        csv_path.name, len(df), len(pcodes))
            written.append(csv_path)
        return written


# =========================================================================== CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NEON AIS downloader (staged: raw → clean)")
    sub = p.add_subparsers(dest="command", required=True)

    common_products = ["DP1.20288.001", "DP1.20033.001", "DP1.20053.001"]

    raw = sub.add_parser("download-raw", help="Step 1 only: pull raw NEON CSVs into data/raw/neon/")
    raw.add_argument("--site", required=True)
    raw.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    raw.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    raw.add_argument("--products", nargs="+", default=common_products)
    raw.add_argument("--package", choices=["basic", "expanded"], default="expanded")
    raw.add_argument("--raw-dir", type=Path, default=None,
                     help="Override raw root (default: Config.DATA_DIR/raw/neon)")

    xf = sub.add_parser("transform", help="Step 2 only: raw → clean")
    xf.add_argument("--site", required=True)
    xf.add_argument("--raw-dir", type=Path, default=None)
    xf.add_argument("--clean-dir", type=Path, default=None,
                    help="Override clean dir (default: Config.DATA_DIR/clean)")
    xf.add_argument("--package", choices=["basic", "expanded"], default="expanded")

    full = sub.add_parser("download-new",
                          help="Combined: download-raw then transform (default flow)")
    full.add_argument("--site", required=True)
    full.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    full.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    full.add_argument("--products", nargs="+", default=common_products)
    full.add_argument("--package", choices=["basic", "expanded"], default="expanded")
    full.add_argument("--raw-dir", type=Path, default=None)
    full.add_argument("--clean-dir", type=Path, default=None)

    ls = sub.add_parser("list-sites", help="List NEON aquatic sites with the multisonde product")
    ls.add_argument("--output", help="Optional CSV path")

    return p


def _cmd_download_raw(args, dl: NEONDownloader) -> int:
    paths = dl.download_raw_site_range(
        args.site, args.start_date, args.end_date, args.products, args.raw_dir,
    )
    logger.info("Wrote/touched %d raw files for %s", len(paths), args.site)
    return 0 if paths else 1


def _cmd_transform(args, dl: NEONDownloader) -> int:
    frames = dl.transform_site_range(args.site, args.raw_dir, args.package)
    if not frames:
        logger.error("No clean output produced for %s — is raw missing?", args.site)
        return 1
    dl.save_clean(frames, args.clean_dir)
    return 0


def _cmd_download_new(args, dl: NEONDownloader) -> int:
    paths = dl.download_raw_site_range(
        args.site, args.start_date, args.end_date, args.products, args.raw_dir,
    )
    if not paths:
        logger.error("No raw files fetched for %s", args.site)
        return 1
    frames = dl.transform_site_range(args.site, args.raw_dir, args.package)
    if not frames:
        logger.error("Transform yielded nothing for %s", args.site)
        return 1
    dl.save_clean(frames, args.clean_dir)
    return 0


def _cmd_list_sites(args, dl: NEONDownloader) -> int:
    df = dl.list_aquatic_sites()
    if args.output:
        df.to_csv(args.output, index=False)
        logger.info("Wrote %d sites to %s", len(df), args.output)
    else:
        sys.stdout.write(df.to_csv(index=False))
    return 0


_HANDLERS = {
    "download-raw":  _cmd_download_raw,
    "transform":     _cmd_transform,
    "download-new":  _cmd_download_new,
    "list-sites":    _cmd_list_sites,
}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = _build_parser().parse_args()
    dl = NEONDownloader(package_kind=getattr(args, "package", "expanded"))
    return _HANDLERS[args.command](args, dl)


if __name__ == "__main__":
    sys.exit(main())
