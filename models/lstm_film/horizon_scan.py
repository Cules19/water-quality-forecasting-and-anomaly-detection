"""Extended forecast-horizon breakdown for the FiLM-LSTM.

Trains the (history-conditioned) model at a range of lead times using a single
shared :class:`SiteBank`, then writes a CSV and an NSE-vs-horizon decay plot.
This is where genuine forecasting skill (beyond persistence) breaks down.

Author: TSLR
"""
from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import SiteBank
from .train import run_experiment


def run_horizon_scan(
    cfg: ExperimentConfig,
    horizons: List[int],
    device: str = "cpu",
    verbose: bool = True,
) -> pd.DataFrame:
    """Train once per horizon (shared data load) and return a results table."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    if verbose:
        print("Loading data once (shared across all horizons)...")
    bank = SiteBank(cfg, verbose=verbose)

    rows = []
    for h in horizons:
        trial = cfg.clone(use_target_history=True, horizon=h, tag=f"horizon={h}")
        if verbose:
            print("\n" + "=" * 60)
            print(f"HORIZON {h} step(s) ahead")
            print("=" * 60)
        res = run_experiment(bank, trial, device=device, verbose=verbose)
        row = res.row()
        rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(cfg.out_dir, "horizon_scan.csv"), index=False
        )

    df = pd.DataFrame(rows)
    with open(os.path.join(cfg.out_dir, "horizon_scan.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    _plot(cfg.out_dir, df)

    if verbose:
        print("\n" + "=" * 60)
        print("HORIZON SCAN COMPLETE")
        print(df[["horizon", "val_nse", "val_rmse", "test_nse", "test_rmse"]].to_string(index=False))
        print(f"Artifacts written to: {cfg.out_dir}")
    return df


def _plot(out_dir: str, df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.sort_values("horizon")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["horizon"], df["val_nse"], "o-", label="val NSE", color="tab:blue")
    ax1.plot(df["horizon"], df["test_nse"], "s--", label="test NSE", color="tab:green")
    ax1.set_xlabel("forecast horizon (hours ahead)")
    ax1.set_ylabel("NSE")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower left")
    ax1.set_title("pH forecast skill vs lead time (history-conditioned FiLM-LSTM)")

    ax2 = ax1.twinx()
    ax2.plot(df["horizon"], df["test_rmse"], "^:", label="test RMSE", color="tab:red")
    ax2.set_ylabel("test RMSE (standardised)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    out = os.path.join(out_dir, "horizon_decay.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved decay plot to {out}")
