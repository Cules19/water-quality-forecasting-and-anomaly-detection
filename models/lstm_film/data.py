"""Data loading and windowing for the FiLM-LSTM sweep.

The expensive work (reading parquet, resampling, computing calendar features and
metadata) happens once in :class:`SiteBank`. Individual experiments then build
cheap :class:`WindowDataset` views over that bank, so varying the sequence
length or model architecture never re-touches disk.

Author: TSLR
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    DYNAMIC_PCODES,
    META_FIELDS,
    N_TIME_FEATURES,
    TARGET_PCODE,
    ExperimentConfig,
)

_PCODE_RE = re.compile(r"_(\d{5})")


# --------------------------------------------------------------------------- #
# Site discovery
# --------------------------------------------------------------------------- #
def _site_id_from_path(path: str) -> str:
    """Filename prefix (before the first underscore) == register key."""
    return os.path.basename(path).split("_", 1)[0]


def _pcodes_in_name(path: str) -> set:
    return set(_PCODE_RE.findall(os.path.basename(path)))


def discover_sites(data_dir: str, require_target: bool = True,
                   target_pcode: str = TARGET_PCODE) -> List[Tuple[str, str]]:
    """Return ``[(site_id, parquet_path), ...]`` for the main pcode shards.

    Skips the ``all_sites_combined`` roll-up and the legacy ``*_processed``
    duplicates. Optionally keeps only shards whose filename advertises the
    target parameter code.
    """
    out: List[Tuple[str, str]] = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.parquet"))):
        base = os.path.basename(path)
        if "all_sites" in base or base.endswith("_processed.parquet"):
            continue
        if require_target and target_pcode not in _pcodes_in_name(path):
            continue
        out.append((_site_id_from_path(path), path))
    return out


def discover_ea_clean(clean_dir: str, target_pcode: str = TARGET_PCODE) -> List[Tuple[str, str]]:
    """Discover EA (``E*``) sites in ``clean_dir`` that carry the target pcode.

    EA sites live only in ``data/clean`` (never built into ``data/processed``)
    and often appear as several date-versioned shards per site; we keep the one
    with the most rows. Values there are in real units (unscaled), which is fine
    because the bank standardises every site independently.
    """
    import pyarrow.parquet as pq

    cand: Dict[str, List[str]] = {}
    for path in glob.glob(os.path.join(clean_dir, "E*.parquet")):
        if target_pcode in _pcodes_in_name(path):
            cand.setdefault(_site_id_from_path(path), []).append(path)

    out: List[Tuple[str, str]] = []
    for sid, paths in cand.items():
        def _rows(p):
            try:
                return pq.ParquetFile(p).metadata.num_rows
            except Exception:
                return 0
        out.append((sid, max(paths, key=_rows)))
    return sorted(out)


# --------------------------------------------------------------------------- #
# Metadata (FiLM conditioning vector)
# --------------------------------------------------------------------------- #
def _safe_log1p(x: float) -> float:
    """log1p guarded against negatives / non-finite sentinels."""
    return float(np.log1p(x)) if np.isfinite(x) and x >= 0 else np.nan


# Land-cover class name -> canonical bucket. ESA WorldCover / register labels.
_LC_BUCKETS = {
    "lc_tree": ("tree", "forest"),
    "lc_crop": ("crop", "cultivated", "agricult"),
    "lc_builtup": ("built", "urban"),
    "lc_grass": ("grass", "herbaceous", "shrub", "rangeland"),
    "lc_water": ("water", "wetland"),
}


def _land_cover_fracs(ca: dict) -> Dict[str, float]:
    """Aggregate the register land-cover dict into canonical fractions (0-1)."""
    out = {k: np.nan for k in _LC_BUCKETS}
    lc = ca.get("watershed_land_cover") if isinstance(ca, dict) else None
    if not isinstance(lc, dict) or not lc:
        return out
    sums = {k: 0.0 for k in _LC_BUCKETS}
    total = 0.0
    for name, pct in lc.items():
        try:
            p = float(pct)
        except (TypeError, ValueError):
            continue
        total += p
        low = str(name).lower()
        for bucket, keys in _LC_BUCKETS.items():
            if any(k in low for k in keys):
                sums[bucket] += p
                break
    if total <= 0:
        return out
    return {k: sums[k] / total for k in _LC_BUCKETS}


def _raw_meta(entry: dict) -> Dict[str, float]:
    """Pull the raw (un-imputed) metadata fields from a register entry."""
    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    dd = entry.get("discharge_description", {})
    stats = dd.get("statistics", {}) if isinstance(dd, dict) else {}
    if not isinstance(stats, dict):
        stats = {}
    ca = entry.get("catchment_analysis", {})
    if not isinstance(ca, dict):
        ca = {}

    has_catchment = 1.0 if ca else 0.0
    has_disc = 1.0 if ("cv_discharge" in stats or "mean_discharge" in stats) else 0.0

    ws_max = _num(ca.get("watershed_elevation_max_m"))
    ws_min = _num(ca.get("watershed_elevation_min_m"))
    relief = ws_max - ws_min if np.isfinite(ws_max) and np.isfinite(ws_min) else np.nan

    precip = _num(ca.get("average_precip_mm_yr"))
    et = _num(ca.get("average_evapo_trans_mm_yr"))
    aridity = et / precip if np.isfinite(et) and np.isfinite(precip) and precip > 0 else np.nan

    lc = _land_cover_fracs(ca)

    return {
        "latitude": _num(entry.get("latitude")),
        "longitude": _num(entry.get("longitude")),
        "elevation": _num(entry.get("elevation")),
        "drainage_area_km2_log": _safe_log1p(_num(entry.get("drainage_area_km2"))),
        "ws_elev_mean": _num(ca.get("watershed_elevation_mean_m")),
        "ws_relief": relief,
        "precip_mm_yr": precip,
        "evapotrans_mm_yr": et,
        "aridity": aridity,
        "cv_discharge": _num(stats.get("cv_discharge")),
        "flashiness_index": _num(stats.get("flashiness_index")),
        "mean_discharge_log": _safe_log1p(_num(stats.get("mean_discharge"))),
        "population_log": _safe_log1p(_num(ca.get("population_est_2020"))),
        "irrigation_pct": _num(ca.get("irrigation_area_pct")),
        "lc_tree": lc["lc_tree"],
        "lc_crop": lc["lc_crop"],
        "lc_builtup": lc["lc_builtup"],
        "lc_grass": lc["lc_grass"],
        "lc_water": lc["lc_water"],
        "has_catchment": has_catchment,
        "has_discharge_stats": has_disc,
    }


def build_meta_matrix(
    site_ids: List[str], register: dict
) -> Tuple[np.ndarray, dict]:
    """Return a standardised ``(n_sites, N_META)`` matrix plus scaler stats.

    Missing fields are median-imputed across the selected sites, then each
    column is z-scored. This keeps the FiLM generator's inputs well-conditioned.
    """
    raw = np.full((len(site_ids), len(META_FIELDS)), np.nan, dtype=np.float64)
    for i, sid in enumerate(site_ids):
        entry = register.get(sid, {})
        m = _raw_meta(entry)
        for j, field in enumerate(META_FIELDS):
            raw[i, j] = m[field]

    medians = np.nanmedian(raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    inds = np.where(np.isnan(raw))
    raw[inds] = np.take(medians, inds[1])

    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    z = (raw - mean) / std
    scaler = {"median": medians.tolist(), "mean": mean.tolist(), "std": std.tolist()}
    return z.astype(np.float32), scaler


# --------------------------------------------------------------------------- #
# Calendar features
# --------------------------------------------------------------------------- #
def _time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """sin/cos of hour-of-day, day-of-year, month -> ``(T, 6)`` float32."""
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    doy = index.dayofyear.to_numpy().astype(np.float64)
    month = index.month.to_numpy().astype(np.float64)
    two_pi = 2.0 * np.pi
    feats = np.stack(
        [
            np.sin(two_pi * hour / 24.0),
            np.cos(two_pi * hour / 24.0),
            np.sin(two_pi * doy / 366.0),
            np.cos(two_pi * doy / 366.0),
            np.sin(two_pi * month / 12.0),
            np.cos(two_pi * month / 12.0),
        ],
        axis=1,
    )
    return feats.astype(np.float32)


# --------------------------------------------------------------------------- #
# Robust per-site cleaning / standardisation
# --------------------------------------------------------------------------- #
def _robust_stats(x: np.ndarray) -> Tuple[float, float]:
    """Return (center, scale) via median / IQR on finite values."""
    finite = x[np.isfinite(x)]
    if finite.size < 8:
        return 0.0, 1.0
    med = float(np.median(finite))
    q1, q3 = np.percentile(finite, [25, 75])
    scale = float((q3 - q1) / 1.349)
    if not np.isfinite(scale) or scale < 1e-6:
        std = float(np.std(finite))
        scale = std if std > 1e-6 else 1.0
    return med, scale


def _clean_outliers(x: np.ndarray, k: float = 12.0) -> np.ndarray:
    """Set gross sentinel outliers (|z| > k on robust scale) to NaN."""
    med, scale = _robust_stats(x)
    out = x.copy()
    bad = np.abs(out - med) > k * scale
    out[bad] = np.nan
    return out


def _do_saturation_mgL(temp_c: np.ndarray) -> np.ndarray:
    """Freshwater DO solubility (mg/L) at 1 atm vs temperature (Benson-Krause).

    Used to convert EA dissolved-oxygen (reported as % saturation) into mg/L so
    it is consistent with the USGS 00300 (mg/L) series.
    """
    T = np.asarray(temp_c, dtype=np.float64) + 273.15
    with np.errstate(invalid="ignore", divide="ignore"):
        ln_cs = (
            -139.34411
            + 1.575701e5 / T
            - 6.642308e7 / T ** 2
            + 1.243800e10 / T ** 3
            - 8.621949e11 / T ** 4
        )
        cs = np.exp(ln_cs)
    return cs


# Hard bound (in robust-std units) applied after per-site standardisation. Any
# value beyond this is treated as a residual sentinel: features are clipped and
# target rows are marked invalid. Guards against sites where >50% of a channel
# is a sentinel (which corrupts even median/IQR statistics).
CLIP = 10.0


# --------------------------------------------------------------------------- #
# Site bank
# --------------------------------------------------------------------------- #
@dataclass
class _Site:
    site_id: str
    idx: int
    path: str                # source parquet (for locating the pipeline scaler)
    dyn: np.ndarray          # (T, n_dynamic) float32, NaN-filled with 0
    row_missing: np.ndarray  # (T,) float32 in {0,1}; 1 == all pcodes missing
    target: np.ndarray       # (T,) float32, per-site standardised
    target_valid: np.ndarray  # (T,) bool
    tgt_center: float        # robust centre used to standardise the target
    tgt_scale: float         # robust scale used to standardise the target
    n: int


class SiteBank:
    """Loads and caches per-site arrays once for reuse across experiments."""

    def __init__(self, cfg: ExperimentConfig, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.sites: List[_Site] = []
        self.site_ids: List[str] = []
        self.meta: np.ndarray | None = None
        self.meta_scaler: dict | None = None
        self._load()

    # -- loading --------------------------------------------------------- #
    def _load(self) -> None:
        cfg = self.cfg
        register = {}
        if os.path.exists(cfg.register_path):
            with open(cfg.register_path, "r", encoding="utf-8") as fh:
                register = json.load(fh)

        found = discover_sites(cfg.data_dir, require_target=True,
                                target_pcode=cfg.target_pcode)
        ea_ids: set = set()
        if cfg.include_ea_clean:
            ea = discover_ea_clean(cfg.clean_dir, target_pcode=cfg.target_pcode)
            # avoid duplicating any site already present in processed/
            seen = {s for s, _ in found}
            ea = [(s, p) for s, p in ea if s not in seen]
            ea_ids = {s for s, _ in ea}
            found = found + ea
            if self.verbose:
                print(f"[SiteBank] +{len(ea)} EA sites from {cfg.clean_dir}")
        if cfg.max_sites is not None and len(found) > cfg.max_sites:
            rng = np.random.default_rng(cfg.site_seed)
            keep = rng.choice(len(found), size=cfg.max_sites, replace=False)
            found = [found[i] for i in sorted(keep)]

        kept_ids: List[str] = []
        for sid, path in found:
            site = self._load_one(sid, path, len(kept_ids), is_ea=sid in ea_ids)
            if site is None:
                continue
            self.sites.append(site)
            kept_ids.append(sid)

        self.site_ids = kept_ids
        self.meta, self.meta_scaler = build_meta_matrix(kept_ids, register)
        if self.verbose:
            total = sum(s.n for s in self.sites)
            print(
                f"[SiteBank] loaded {len(self.sites)} sites, "
                f"{total:,} resampled rows, "
                f"dynamic features = {self.n_dynamic}"
            )

    def _load_one(self, sid: str, path: str, idx: int, is_ea: bool = False) -> _Site | None:
        cfg = self.cfg
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - skip unreadable shards
            if self.verbose:
                print(f"[SiteBank] skip {sid}: {exc}")
            return None

        # Normalise the datetime index (files use lowercase 'timestamp').
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ("timestamp", "TIMESTAMP", "datetime"):
                if col in df.columns:
                    df = df.set_index(pd.to_datetime(df[col]))
                    break
        df = df[~df.index.isna()].sort_index()

        if cfg.target_pcode not in df.columns:
            return None

        # EA dissolved oxygen is reported as % saturation; convert to mg/L using
        # temperature so it is consistent with USGS 00300 (mg/L).
        if is_ea and "00300" in df.columns and "00010" in df.columns:
            cs = _do_saturation_mgL(df["00010"].to_numpy(dtype=np.float64))
            df["00300"] = df["00300"].to_numpy(dtype=np.float64) / 100.0 * cs

        # Restrict to the most recent N years to bound memory / runtime.
        if cfg.years is not None and len(df):
            cutoff = df.index.max() - pd.Timedelta(days=int(cfg.years * 365.25))
            df = df[df.index >= cutoff]

        # Resample to a uniform grid (mean within each bucket).
        num = df.select_dtypes(include=[np.number])
        res = num.resample(cfg.resample).mean()
        if len(res) < cfg.min_rows:
            return None

        # Stats are computed on the training portion only (no leakage).
        # Some sparse sites begin reporting the target late in the uniform
        # calendar grid. For those sites, optionally align the cutoff to the
        # chronological training fraction of actual target observations.
        raw_target = _clean_outliers(
            res[self.cfg.target_pcode].to_numpy(dtype=np.float64)
        )
        train_cut = max(8, int(cfg.train_frac * len(res)))
        if cfg.align_scalers_to_observed_target_split:
            observed_target_rows = np.flatnonzero(np.isfinite(raw_target))
            if observed_target_rows.size >= 8:
                n_target_train = max(
                    8, int(cfg.train_frac * observed_target_rows.size)
                )
                n_target_train = min(n_target_train, observed_target_rows.size)
                train_cut = int(observed_target_rows[n_target_train - 1]) + 1

        # Align dynamic features to the canonical pcode list, cleaning gross
        # sentinel outliers and robustly standardising each channel per site.
        pcodes = list(self.cfg.dynamic_pcodes)
        dyn = np.zeros((len(res), self.n_dynamic), dtype=np.float32)
        present = np.zeros((len(res), len(pcodes)), dtype=bool)
        for j, pc in enumerate(pcodes):
            if pc in res.columns:
                col = _clean_outliers(res[pc].to_numpy(dtype=np.float64))
                present[:, j] = np.isfinite(col)
                center, scale = _robust_stats(col[:train_cut])
                col = (col - center) / scale
                col = np.clip(col, -CLIP, CLIP)
                dyn[:, j] = np.nan_to_num(col, nan=0.0).astype(np.float32)
        dyn[:, len(pcodes):] = _time_features(res.index)

        row_missing = (~present.any(axis=1)).astype(np.float32)

        # Target: clean sentinels, then standardise per site (training stats).
        tgt = raw_target
        center, scale = _robust_stats(tgt[:train_cut])
        tgt = (tgt - center) / scale
        # A row is a valid target only if finite AND within the hard bound;
        # this rejects residual sentinels that survive robust cleaning.
        tgt_valid = np.isfinite(tgt) & (np.abs(tgt) <= CLIP)
        if tgt_valid.sum() < cfg.min_rows // 2:
            return None
        tgt = np.nan_to_num(np.clip(tgt, -CLIP, CLIP), nan=0.0).astype(np.float32)

        return _Site(
            site_id=sid,
            idx=idx,
            path=path,
            dyn=dyn,
            row_missing=row_missing,
            target=tgt,
            target_valid=tgt_valid,
            tgt_center=float(center),
            tgt_scale=float(scale),
            n=len(res),
        )

    # -- properties ------------------------------------------------------ #
    @property
    def n_dynamic(self) -> int:
        return len(self.cfg.dynamic_pcodes) + N_TIME_FEATURES

    @property
    def n_sites(self) -> int:
        return len(self.sites)


# --------------------------------------------------------------------------- #
# Window index + dataset
# --------------------------------------------------------------------------- #
def build_index(
    bank: SiteBank, cfg: ExperimentConfig
) -> Dict[str, np.ndarray]:
    """Build time-ordered per-site train/val/test window indices.

    A window ending at ``t`` predicts the target at ``t + horizon``. It is valid
    when that target is observed and the input span is not mostly missing.
    Returns dict of ``{split: array of shape (K, 2) [site_idx, end_t]}``.
    """
    seq = cfg.seq_len
    hzn = cfg.horizon
    tr, va, te = [], [], []

    for site in bank.sites:
        n = site.n
        last_start = n - hzn  # prediction index t+hzn must be < n
        # valid end positions t in [seq-1, last_start-1]
        starts = np.arange(seq - 1, last_start)
        if len(starts) == 0:
            continue
        pred = starts + hzn
        ok = site.target_valid[pred]
        # rolling missing fraction over the input window
        cum = np.concatenate([[0.0], np.cumsum(site.row_missing)])
        miss_frac = (cum[starts + 1] - cum[starts + 1 - seq]) / seq
        ok &= miss_frac <= cfg.max_nan_frac
        valid = starts[ok]
        if len(valid) == 0:
            continue

        # time-ordered split by position
        n_v = len(valid)
        i_tr = int(cfg.train_frac * n_v)
        i_va = int((cfg.train_frac + cfg.val_frac) * n_v)
        seg_tr = valid[:i_tr][:: cfg.train_stride]
        seg_va = valid[i_tr:i_va][:: cfg.eval_stride]
        seg_te = valid[i_va:][:: cfg.eval_stride]
        for seg, dst in ((seg_tr, tr), (seg_va, va), (seg_te, te)):
            if len(seg):
                col = np.full((len(seg), 2), site.idx, dtype=np.int64)
                col[:, 1] = seg
                dst.append(col)

    def _cat(lst):
        return np.concatenate(lst, axis=0) if lst else np.zeros((0, 2), np.int64)

    train = _cat(tr)
    # subsample training windows to the configured cap
    if cfg.max_train_windows is not None and len(train) > cfg.max_train_windows:
        rng = np.random.default_rng(cfg.seed)
        sel = rng.choice(len(train), size=cfg.max_train_windows, replace=False)
        train = train[np.sort(sel)]

    return {"train": train, "val": _cat(va), "test": _cat(te)}


class WindowDataset(Dataset):
    """A cheap view over :class:`SiteBank` yielding ``(x, site_idx, meta, y)``.

    When ``cfg.use_target_history`` is set, the (standardised) pH series is
    appended as an extra input channel. The window covers times
    ``[t-seq+1 .. t]`` and the label is pH at ``t+horizon``; because the sweep
    forces ``horizon >= 1`` in that mode, the predicted value is never inside
    the input window, so this is leakage-free.
    """

    def __init__(self, bank: SiteBank, index: np.ndarray, cfg: ExperimentConfig):
        self.bank = bank
        self.index = index
        self.seq = cfg.seq_len
        self.hzn = cfg.horizon
        self.use_hist = cfg.use_target_history
        self.meta = torch.from_numpy(bank.meta)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        site_idx, t = self.index[i]
        site = self.bank.sites[site_idx]
        x = site.dyn[t - self.seq + 1 : t + 1]            # (seq, n_dynamic)
        if self.use_hist:
            hist = site.target[t - self.seq + 1 : t + 1, None]  # (seq, 1)
            x = np.concatenate([x, hist], axis=1)
        y = site.target[t + self.hzn]
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.tensor(site_idx, dtype=torch.long),
            self.meta[site_idx],
            torch.tensor(y, dtype=torch.float32),
        )
