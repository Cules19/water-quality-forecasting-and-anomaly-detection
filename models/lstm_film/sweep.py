"""Staged hyper-parameter sweep for the FiLM-LSTM.

Rather than a full grid (expensive on CPU), this performs a *coordinate* search:
each stage varies one axis while holding the current best fixed, then adopts the
winner before moving on. Stages, in order:

    1. sequence length   (how much history helps)
    2. hidden size       (capacity)
    3. number of layers  (depth)
    4. dropout           (regularisation)
    5. FiLM on/off       (does site metadata conditioning help?)

Every run is appended to a leaderboard CSV/JSON under ``cfg.out_dir``.

Author: TSLR
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd

from .config import ExperimentConfig
from .data import SiteBank
from .train import RunResult, run_experiment


# Default search space per stage. Override via ``stages=`` if desired.
DEFAULT_STAGES: Dict[str, List] = {
    "seq_len": [12, 24, 48, 96],
    "hidden": [32, 64, 128],
    "n_layers": [1, 2, 3],
    "dropout": [0.0, 0.1, 0.3],
    "use_film": [True, False],
}

# Preset that probes autoregression and forecast lead time, starting from a
# known-good architecture. ``use_target_history`` is compared at a fixed 1-step
# horizon so the two runs are directly comparable.
HISTORY_STAGES: Dict[str, List] = {
    "use_target_history": [False, True],
    "horizon": [1, 3, 6, 12],
    "seq_len": [24, 48, 96],
}

STAGE_PRESETS: Dict[str, Dict[str, List]] = {
    "arch": DEFAULT_STAGES,
    "history": HISTORY_STAGES,
}


def _select_metric(res: RunResult) -> float:
    """Lower is better: validation RMSE (NaN guarded)."""
    v = res.val["rmse"]
    return float("inf") if v != v else v  # NaN check


def run_sweep(
    cfg: ExperimentConfig,
    stages: Dict[str, List] | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """Run the staged sweep and return ``{best_config, leaderboard, history}``."""
    stages = stages or DEFAULT_STAGES
    os.makedirs(cfg.out_dir, exist_ok=True)

    if verbose:
        print("=" * 72)
        print("Loading data once (shared across all experiments)...")
    bank = SiteBank(cfg, verbose=verbose)

    best = cfg
    rows: List[dict] = []
    run_id = 0

    for axis, values in stages.items():
        if verbose:
            print("\n" + "=" * 72)
            print(f"STAGE: sweeping '{axis}' over {values}")
            print(f"  (holding best so far: {_summ(best)})")
            print("=" * 72)

        stage_results: List[RunResult] = []
        for val in values:
            trial = best.clone(**{axis: val}, tag=f"{axis}={val}")
            run_id += 1
            if verbose:
                print(f"\n[{run_id}] {axis}={val}")
            try:
                res = run_experiment(bank, trial, device=device, verbose=verbose)
            except RuntimeError as exc:
                if verbose:
                    print(f"  skipped: {exc}")
                continue
            stage_results.append(res)
            row = res.row()
            row["stage"] = axis
            row["run_id"] = run_id
            rows.append(row)
            _flush(cfg, rows)

        if stage_results:
            winner = min(stage_results, key=_select_metric)
            best = best.clone(**{axis: getattr(winner.cfg, axis)})
            if verbose:
                print(
                    f"\n  -> best {axis} = {getattr(winner.cfg, axis)} "
                    f"(val_rmse {winner.val['rmse']:.4f}, "
                    f"val_nse {winner.val['nse']:.4f})"
                )

    leaderboard = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
    _flush(cfg, rows, leaderboard)

    best_cfg = best.clone(tag="best")
    with open(os.path.join(cfg.out_dir, "best_config.json"), "w", encoding="utf-8") as fh:
        json.dump(best_cfg.to_dict(), fh, indent=2)

    if verbose:
        print("\n" + "=" * 72)
        print("SWEEP COMPLETE — leaderboard (top 10 by val_rmse):")
        cols = [
            "stage", "seq_len", "horizon", "use_target_history", "n_layers",
            "hidden", "dropout", "use_film", "n_params", "val_rmse", "val_nse",
            "test_rmse", "test_nse", "train_time_s",
        ]
        print(leaderboard[cols].head(10).to_string(index=False))
        print(f"\nBest configuration: {_summ(best_cfg)}")
        print(f"Artifacts written to: {cfg.out_dir}")

    return {"best_config": best_cfg, "leaderboard": leaderboard, "history": rows}


def _summ(c: ExperimentConfig) -> str:
    return (
        f"seq_len={c.seq_len}, horizon={c.horizon}, "
        f"hist={c.use_target_history}, hidden={c.hidden}, "
        f"n_layers={c.n_layers}, dropout={c.dropout}, use_film={c.use_film}"
    )


def _flush(cfg: ExperimentConfig, rows: List[dict], leaderboard: pd.DataFrame | None = None) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.out_dir, "sweep_results.csv"), index=False)
    with open(os.path.join(cfg.out_dir, "sweep_results.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    if leaderboard is not None:
        leaderboard.to_csv(os.path.join(cfg.out_dir, "leaderboard.csv"), index=False)
