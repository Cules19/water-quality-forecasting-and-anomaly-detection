"""Evaluate a saved FiLM-LSTM run: real-unit metrics + per-site NSE map.

Rebuilds the exact :class:`SiteBank` the checkpoint was trained on, reruns the
test set, then:
  * inverts per-site standardisation and the pipeline ``MinMaxScaler`` to report
    RMSE/MAE in **real pH units**,
  * computes per-site NSE (scale-invariant) and writes a table,
  * plots a geographic NSE map (coloured by skill) and an NSE histogram.

Author: TSLR
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import SiteBank, WindowDataset, build_index
from .model import FiLMLSTM
from .train import compute_metrics


def _load_target_minmax(site_path: str, target_pcode: str):
    """Return (data_min, data_max) of the pipeline MinMaxScaler for the target, or None."""
    jl = site_path[:-len(".parquet")] + "_pipeline.joblib"
    if not os.path.exists(jl):
        return None
    try:
        import joblib
        obj = joblib.load(jl)
        sc = obj.get("scalers", {}).get(target_pcode)
        if sc is None or not hasattr(sc, "data_min_"):
            return None
        return float(sc.data_min_[0]), float(sc.data_max_[0])
    except Exception:
        return None


@torch.no_grad()
def _predict_test(model, loader, device):
    model.eval()
    preds, trues = [], []
    for xb, sb, mb, yb in loader:
        preds.append(model(xb.to(device), sb.to(device), mb.to(device)).cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def evaluate_run(run_dir: str, out_dir: str | None = None,
                 device: str = "cpu", verbose: bool = True) -> Dict:
    """Evaluate the checkpoint in ``run_dir`` and write analysis artifacts."""
    out_dir = out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location=device, weights_only=False)
    cfg = ExperimentConfig(**{k: v for k, v in ckpt["config"].items()
                              if k in ExperimentConfig().to_dict()})
    saved_ids = ckpt["site_ids"]

    if verbose:
        print(f"Rebuilding data for {run_dir} (seq_len={cfg.seq_len}, "
              f"horizon={cfg.horizon}, hist={cfg.use_target_history})...")
    bank = SiteBank(cfg, verbose=verbose)
    if bank.site_ids != saved_ids:
        # Order must match for the site embedding indices to line up.
        raise RuntimeError(
            "site set/order changed since training; cannot align embeddings "
            f"({len(bank.site_ids)} now vs {len(saved_ids)} saved)"
        )

    index = build_index(bank, cfg)
    test_ds = WindowDataset(bank, index["test"], cfg)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size * 2)

    model = FiLMLSTM(cfg, n_sites=bank.n_sites).to(device)
    model.load_state_dict(ckpt["state_dict"])

    pred, true = _predict_test(model, test_dl, device)
    site_idx = index["test"][:, 0]
    overall_std = compute_metrics(pred, true)

    # De-standardise to real pH units and compute per-site metrics.
    register = {}
    if os.path.exists(cfg.register_path):
        with open(cfg.register_path, "r", encoding="utf-8") as fh:
            register = json.load(fh)

    rows = []
    real_res_all = []
    for si, site in enumerate(bank.sites):
        m = site_idx == si
        if not m.any():
            continue
        p_std, t_std = pred[m], true[m]
        nse = compute_metrics(p_std, t_std)["nse"]

        # standardised -> file units -> real pH
        p_file = p_std * site.tgt_scale + site.tgt_center
        t_file = t_std * site.tgt_scale + site.tgt_center
        mm = _load_target_minmax(site.path, cfg.target_pcode)
        if mm is not None:
            dmin, dmax = mm
            span = dmax - dmin
            p_real = p_file * span + dmin
            t_real = t_file * span + dmin
        else:
            p_real, t_real = p_file, t_file  # already file-units fallback

        res = p_real - t_real
        real_res_all.append(res)
        rmse_real = float(np.sqrt(np.mean(res ** 2)))
        mae_real = float(np.mean(np.abs(res)))

        entry = register.get(site.site_id, {})
        rows.append({
            "site_id": site.site_id,
            "n_test": int(m.sum()),
            "nse": round(nse, 4),
            "rmse_real": round(rmse_real, 4),
            "mae_real": round(mae_real, 4),
            "latitude": entry.get("latitude"),
            "longitude": entry.get("longitude"),
            "river_name": entry.get("river_name"),
            "source": entry.get("source"),
        })

    df = pd.DataFrame(rows).sort_values("nse", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "per_site_metrics.csv"), index=False)

    real_res_all = np.concatenate(real_res_all) if real_res_all else np.array([0.0])
    summary = {
        "run_dir": run_dir,
        "n_sites": int(len(df)),
        "n_test_windows": int(len(pred)),
        "overall_nse_std": round(overall_std["nse"], 4),
        "overall_rmse_std": round(overall_std["rmse"], 4),
        "overall_rmse_real": round(float(np.sqrt(np.mean(real_res_all ** 2))), 4),
        "overall_mae_real": round(float(np.mean(np.abs(real_res_all))), 4),
        "median_site_nse": round(float(df["nse"].median()), 4),
        "p25_site_nse": round(float(df["nse"].quantile(0.25)), 4),
        "p10_site_nse": round(float(df["nse"].quantile(0.10)), 4),
        "sites_nse_below_0": int((df["nse"] < 0).sum()),
    }
    with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _plot_map(out_dir, df)
    _plot_hist(out_dir, df)

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print(json.dumps(summary, indent=2))
        print("\nWorst 8 sites by NSE:")
        print(df.tail(8)[["site_id", "n_test", "nse", "rmse_real", "river_name"]].to_string(index=False))
        print(f"\nArtifacts written to: {out_dir}")
    return {"summary": summary, "per_site": df}


def _plot_map(out_dir: str, df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geo = df.dropna(subset=["latitude", "longitude"])
    if geo.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(
        geo["longitude"], geo["latitude"], c=geo["nse"].clip(0, 1),
        cmap="RdYlGn", vmin=0, vmax=1, s=40, edgecolors="k", linewidths=0.3,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("per-site test NSE")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Per-site forecast skill ({len(geo)} sites)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "per_site_nse_map.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved map to {out}")


def _plot_hist(out_dir: str, df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["nse"].clip(-0.5, 1), bins=40, color="tab:blue", edgecolor="k", alpha=0.8)
    med = df["nse"].median()
    ax.axvline(med, color="red", ls="--", label=f"median = {med:.3f}")
    ax.set_xlabel("per-site test NSE")
    ax.set_ylabel("number of sites")
    ax.set_title("Distribution of per-site forecast skill")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(out_dir, "per_site_nse_hist.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved histogram to {out}")
