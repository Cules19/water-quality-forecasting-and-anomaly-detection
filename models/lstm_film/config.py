"""Experiment configuration and feature definitions for the FiLM-LSTM sweep.

Author: TSLR
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

# --------------------------------------------------------------------------- #
# Feature definitions
# --------------------------------------------------------------------------- #
# Target: pH (USGS parameter code 00400).
TARGET_PCODE = "00400"

# Dynamic (time-varying) exogenous inputs. These are the most widely available
# parameter codes across the processed sites, deliberately EXCLUDING the target
# so the model is not trivially autoregressive.
#   00010 water temperature       00060 discharge        00065 gage height
#   00095 specific conductance     63680 turbidity        00300 dissolved oxygen
#   00045 precipitation (hourly mm; CAMELS-H / Met-Office / ERA5-seamless)
DYNAMIC_PCODES: List[str] = ["00010", "00060", "00065", "00095", "63680", "00300", "00045"]

# Number of engineered calendar features appended to every timestep
# (sin/cos of hour-of-day, day-of-year, month).
N_TIME_FEATURES = 6

# Static per-site metadata used to generate FiLM (gamma, beta) modulation.
# Missing values are median-imputed across the selected sites; the two `has_*`
# flags let the network distinguish imputed rows from observed ones.
META_FIELDS: List[str] = [
    # location / topography
    "latitude",
    "longitude",
    "elevation",
    "drainage_area_km2_log",     # log1p(drainage area) — spans orders of magnitude
    "ws_elev_mean",              # watershed mean elevation
    "ws_relief",                 # watershed elevation range (max - min)
    # climate / hydrology
    "precip_mm_yr",              # annual precipitation
    "evapotrans_mm_yr",          # annual evapotranspiration
    "aridity",                   # ET / precip
    "cv_discharge",              # discharge coefficient of variation
    "flashiness_index",          # Richards-Baker flashiness
    "mean_discharge_log",        # log1p(mean discharge)
    # anthropogenic pressure
    "population_log",            # log1p(catchment population) — urbanisation proxy
    "irrigation_pct",            # irrigated area % — agriculture proxy
    # land cover composition (fractions 0-1)
    "lc_tree",
    "lc_crop",
    "lc_builtup",
    "lc_grass",
    "lc_water",
    # missingness indicators
    "has_catchment",
    "has_discharge_stats",
]

N_DYNAMIC = len(DYNAMIC_PCODES) + N_TIME_FEATURES
N_META = len(META_FIELDS)


# --------------------------------------------------------------------------- #
# Experiment configuration
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    """All knobs for a single training run.

    The sweep mutates a handful of these (``seq_len``, ``hidden``, ``n_layers``,
    ``dropout``, ``use_film``) while holding the data budget fixed.
    """

    # --- data budget -------------------------------------------------------- #
    data_dir: str = "data/processed"
    register_path: str = "data/reference/sites_register.json"
    include_ea_clean: bool = False            # also load EA sites from clean_dir
    clean_dir: str = "data/clean"             # source of EA (E*) parquets
    target_pcode: str = TARGET_PCODE          # parameter code to predict
    dynamic_pcodes: tuple = tuple(DYNAMIC_PCODES)  # exogenous input channels
    resample: str = "1h"           # pandas offset; downsample 10-min -> hourly
    max_sites: int = 60            # cap number of pH sites loaded (None = all)
    years: float | None = 4.0      # keep only the most recent N years per site
    min_rows: int = 2000           # skip sites with fewer resampled rows
    max_nan_frac: float = 0.5      # drop a window if inputs exceed this NaN frac
    site_seed: int = 0             # RNG seed for site sub-sampling
    align_scalers_to_observed_target_split: bool = False
    # When enabled, place the train-statistics cutoff at the chronological 70%
    # point of observed target rows. This avoids empty early calendar periods
    # producing the fallback target centre/scale (0, 1).

    # --- windowing ---------------------------------------------------------- #
    seq_len: int = 24              # input history length (in resampled steps)
    horizon: int = 0               # steps ahead to predict (0 = nowcast)
    use_target_history: bool = False  # feed lagged pH as an input channel
    train_stride: int = 3          # stride between training windows
    eval_stride: int = 6           # stride between val/test windows
    max_train_windows: int | None = 60000   # subsample cap (None = all)

    # --- splits (per-site, time ordered) ------------------------------------ #
    train_frac: float = 0.7
    val_frac: float = 0.15         # test = 1 - train - val

    # --- model -------------------------------------------------------------- #
    hidden: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    site_emb_dim: int = 8
    film_hidden: int = 64
    use_film: bool = True          # False -> identity FiLM (metadata ablation)

    # --- optimisation ------------------------------------------------------- #
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    patience: int = 3              # early-stopping patience on val RMSE
    min_epochs: int = 3            # always train at least this many epochs
    min_delta: float = 1e-4        # min val-RMSE improvement to reset patience
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 0

    # --- bookkeeping -------------------------------------------------------- #
    tag: str = "run"
    out_dir: str = "outputs/reports/lstm_film"

    # derived (filled at build time)
    n_dynamic: int = N_DYNAMIC
    n_meta: int = N_META

    def to_dict(self) -> dict:
        return asdict(self)

    def clone(self, **overrides) -> "ExperimentConfig":
        d = self.to_dict()
        d.update(overrides)
        return ExperimentConfig(**d)

    @property
    def out_path(self) -> Path:
        return Path(self.out_dir)
