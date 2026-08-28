"""Training loop, metrics, and single-experiment driver.

Author: TSLR
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import SiteBank, WindowDataset, build_index
from .model import FiLMLSTM


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """RMSE, MAE, NSE (Nash-Sutcliffe) and R^2 on the (standardised) target."""
    err = pred - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    var = float(np.var(true))
    nse = float(1.0 - np.mean(err ** 2) / var) if var > 1e-12 else float("nan")
    # R^2 (coefficient of determination) equals NSE for these definitions.
    return {"rmse": rmse, "mae": mae, "nse": nse, "n": int(len(true))}


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    cfg: ExperimentConfig
    val: Dict[str, float]
    test: Dict[str, float]
    n_params: int
    best_epoch: int
    epochs_ran: int
    train_time_s: float
    n_train: int
    n_val: int
    n_test: int

    def row(self) -> dict:
        c = self.cfg
        return {
            "tag": c.tag,
            "seq_len": c.seq_len,
            "horizon": c.horizon,
            "use_target_history": c.use_target_history,
            "n_layers": c.n_layers,
            "hidden": c.hidden,
            "dropout": c.dropout,
            "use_film": c.use_film,
            "n_params": self.n_params,
            "val_rmse": round(self.val["rmse"], 5),
            "val_nse": round(self.val["nse"], 5),
            "test_rmse": round(self.test["rmse"], 5),
            "test_nse": round(self.test["nse"], 5),
            "best_epoch": self.best_epoch,
            "epochs_ran": self.epochs_ran,
            "train_time_s": round(self.train_time_s, 1),
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
        }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    preds, trues = [], []
    for xb, sb, mb, yb in loader:
        xb, sb, mb = xb.to(device), sb.to(device), mb.to(device)
        preds.append(model(xb, sb, mb).cpu().numpy())
        trues.append(yb.numpy())
    if not preds:
        return {"rmse": float("nan"), "mae": float("nan"), "nse": float("nan"), "n": 0}
    return compute_metrics(np.concatenate(preds), np.concatenate(trues))


# --------------------------------------------------------------------------- #
# Single experiment
# --------------------------------------------------------------------------- #
def run_experiment(
    bank: SiteBank, cfg: ExperimentConfig, device: str = "cpu", verbose: bool = True,
    save_dir: str | None = None, make_plot: bool = False,
) -> RunResult:
    """Train one configuration end-to-end and return validation/test metrics.

    If ``save_dir`` is given, the best checkpoint, test predictions and (when
    ``make_plot``) a predicted-vs-actual figure are written there.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Target-history mode must forecast at least one step ahead, otherwise the
    # label would sit inside the input window (leakage).
    horizon = cfg.horizon
    if cfg.use_target_history and horizon < 1:
        horizon = 1
    extra = 1 if cfg.use_target_history else 0
    cfg = cfg.clone(
        n_dynamic=bank.n_dynamic + extra,
        n_meta=bank.meta.shape[1],
        horizon=horizon,
    )
    index = build_index(bank, cfg)

    train_ds = WindowDataset(bank, index["train"], cfg)
    val_ds = WindowDataset(bank, index["val"], cfg)
    test_ds = WindowDataset(bank, index["test"], cfg)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"empty split (train={len(train_ds)}, val={len(val_ds)}); "
            "loosen seq_len / max_nan_frac / data budget"
        )

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=False,
    )
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size * 2, num_workers=cfg.num_workers)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size * 2, num_workers=cfg.num_workers)

    model = FiLMLSTM(cfg, n_sites=bank.n_sites).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2
    )
    loss_fn = nn.MSELoss()

    best_rmse = float("inf")
    best_state = None
    best_epoch = -1
    bad = 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        run_loss = 0.0
        seen = 0
        for xb, sb, mb, yb in train_dl:
            xb, sb, mb, yb = (t.to(device) for t in (xb, sb, mb, yb))
            opt.zero_grad()
            loss = loss_fn(model(xb, sb, mb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            run_loss += loss.item() * len(yb)
            seen += len(yb)

        val = _evaluate(model, val_dl, device)
        sched.step(val["rmse"])
        if verbose:
            print(
                f"  epoch {epoch + 1:2d}/{cfg.epochs}  "
                f"train_mse {run_loss / max(seen, 1):.4f}  "
                f"val_rmse {val['rmse']:.4f}  val_nse {val['nse']:.4f}"
            )

        if val["rmse"] < best_rmse - cfg.min_delta:
            best_rmse = val["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            bad = 0
        else:
            bad += 1
            if epoch + 1 >= cfg.min_epochs and bad >= cfg.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch + 1} "
                          f"(no >{cfg.min_delta:g} val gain for {cfg.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val = _evaluate(model, val_dl, device)
    test = _evaluate(model, test_dl, device)

    result = RunResult(
        cfg=cfg,
        val=val,
        test=test,
        n_params=model.count_params(),
        best_epoch=best_epoch,
        epochs_ran=epoch + 1,
        train_time_s=time.time() - t0,
        n_train=len(train_ds),
        n_val=len(val_ds),
        n_test=len(test_ds),
    )

    if save_dir is not None:
        _save_artifacts(save_dir, model, bank, cfg, index, test_dl, result,
                        device, make_plot, verbose)

    return result


# --------------------------------------------------------------------------- #
# Saving + plotting
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _predict_ordered(model, loader, device):
    """Return (pred, true) arrays over an unshuffled loader."""
    model.eval()
    preds, trues = [], []
    for xb, sb, mb, yb in loader:
        preds.append(model(xb.to(device), sb.to(device), mb.to(device)).cpu().numpy())
        trues.append(yb.numpy())
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(trues)


def _save_artifacts(save_dir, model, bank, cfg, index, test_dl, result,
                    device, make_plot, verbose):
    import json
    import os

    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "meta_scaler": bank.meta_scaler,
        "site_ids": bank.site_ids,
    }
    torch.save(ckpt, os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(result.row(), fh, indent=2)

    pred, true = _predict_ordered(model, test_dl, device)
    np.savez_compressed(
        os.path.join(save_dir, "test_predictions.npz"),
        pred=pred, true=true, index=index["test"],
    )
    if verbose:
        print(f"  saved model + predictions to {save_dir}")

    if make_plot and len(pred):
        _plot(save_dir, pred, true, index["test"], bank, result)


def _plot(save_dir, pred, true, test_index, bank, result):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) predicted vs actual density-ish scatter (subsampled)
    n = len(pred)
    take = np.random.default_rng(0).choice(n, size=min(n, 20000), replace=False)
    lo = float(min(true[take].min(), pred[take].min()))
    hi = float(max(true[take].max(), pred[take].max()))
    axes[0].scatter(true[take], pred[take], s=3, alpha=0.15, edgecolors="none")
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[0].set_xlabel("actual pH (standardised)")
    axes[0].set_ylabel("predicted pH (standardised)")
    axes[0].set_title(
        f"Test: NSE={result.test['nse']:.3f}  RMSE={result.test['rmse']:.3f}"
    )

    # (b) example time-ordered trace for the site with the most test windows
    sids = test_index[:, 0]
    uniq, counts = np.unique(sids, return_counts=True)
    best_site = int(uniq[np.argmax(counts)])
    mask = sids == best_site
    order = np.argsort(test_index[mask, 1])
    seg_true = true[mask][order][:1500]
    seg_pred = pred[mask][order][:1500]
    axes[1].plot(seg_true, label="actual", lw=1.0)
    axes[1].plot(seg_pred, label="predicted", lw=1.0, alpha=0.8)
    axes[1].set_title(f"Site {bank.site_ids[best_site]} — test window (first 1500)")
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("pH (standardised)")
    axes[1].legend()

    fig.tight_layout()
    out = os.path.join(save_dir, "predictions.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved plot to {out}")
