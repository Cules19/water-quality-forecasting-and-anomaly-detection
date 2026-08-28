"""Command-line entry point for the FiLM-LSTM experiments.

Examples (run from the repository root)::

    # fast plumbing check (few sites, few epochs)
    .venv\\Scripts\\python.exe -m models.lstm_film.run smoke

    # staged architecture sweep over ~60 pH sites (last 4 years, hourly)
    .venv\\Scripts\\python.exe -m models.lstm_film.run sweep

    # bigger sweep across ALL pH sites
    .venv\\Scripts\\python.exe -m models.lstm_film.run sweep --max-sites 0 --years 0

    # train a single best configuration and save the model
    .venv\\Scripts\\python.exe -m models.lstm_film.run train --seq-len 48 --hidden 128 --n-layers 2

Author: TSLR
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from .config import ExperimentConfig
from .data import SiteBank
from .evaluate import evaluate_run
from .horizon_scan import run_horizon_scan
from .sweep import STAGE_PRESETS, run_sweep
from .train import run_experiment


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _apply_common(cfg: ExperimentConfig, args) -> ExperimentConfig:
    over = {}
    if getattr(args, "target", None):
        over["target_pcode"] = args.target
    if getattr(args, "dynamic", None):
        over["dynamic_pcodes"] = tuple(p.strip() for p in args.dynamic.split(","))
    if getattr(args, "include_ea", False):
        over["include_ea_clean"] = True
    if args.max_sites is not None:
        over["max_sites"] = None if args.max_sites == 0 else args.max_sites
    if args.years is not None:
        over["years"] = None if args.years == 0 else args.years
    if args.resample:
        over["resample"] = args.resample
    if args.epochs is not None:
        over["epochs"] = args.epochs
    if args.max_train_windows is not None:
        over["max_train_windows"] = None if args.max_train_windows == 0 else args.max_train_windows
    if args.out_dir:
        over["out_dir"] = args.out_dir
    return cfg.clone(**over)


def cmd_smoke(args) -> None:
    cfg = ExperimentConfig(
        max_sites=6, years=2.0, epochs=3, max_train_windows=4000,
        tag="smoke", out_dir="outputs/reports/lstm_film_smoke",
    )
    cfg = _apply_common(cfg, args)
    device = _device()
    print(f"Device: {device}")
    bank = SiteBank(cfg)
    res = run_experiment(bank, cfg, device=device)
    print("\nSMOKE RESULT")
    print(json.dumps(res.row(), indent=2))


def cmd_sweep(args) -> None:
    cfg = ExperimentConfig(tag="sweep")
    cfg = _apply_common(cfg, args)

    # Optionally seed the base architecture from a previous sweep's winner.
    if args.from_best:
        best_path = args.best_config or os.path.join(cfg.out_dir, "best_config.json")
        if os.path.exists(best_path):
            with open(best_path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            for k in ("seq_len", "hidden", "n_layers", "dropout", "use_film"):
                if k in saved:
                    cfg = cfg.clone(**{k: saved[k]})
            print(f"Seeded base architecture from {best_path}")
        else:
            print(f"--from-best given but {best_path} not found; using defaults")

    stages = dict(STAGE_PRESETS[args.preset])
    if args.preset == "history":
        # Compare nowcast-vs-history fairly at a 1-step horizon.
        cfg = cfg.clone(horizon=1)
    if args.no_film_stage:
        stages.pop("use_film", None)

    device = _device()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"Device: {device}  |  torch threads: {torch.get_num_threads()}  |  preset: {args.preset}")
    result = run_sweep(cfg, stages=stages, device=device)

    if not args.no_train_best:
        best = result["best_config"]
        # Retrain the winning architecture on ALL pH sites and full history.
        full = best.clone(
            max_sites=None,
            years=None,
            epochs=args.full_epochs,
            patience=4,
            max_train_windows=args.full_windows if args.full_windows else None,
            tag="best_full",
            out_dir=os.path.join(cfg.out_dir, "best_full"),
        )
        print("\n" + "=" * 72)
        print("TRAINING BEST CONFIG ON ALL SITES (full history)")
        print(f"  {full.seq_len=} {full.hidden=} {full.n_layers=} "
              f"{full.dropout=} {full.use_film=}")
        print("=" * 72)
        bank = SiteBank(full)
        res = run_experiment(
            bank, full, device=device,
            save_dir=full.out_dir, make_plot=True,
        )
        print("\nFINAL MODEL (all sites):")
        print(json.dumps(res.row(), indent=2))


def cmd_train(args) -> None:
    cfg = ExperimentConfig(tag="train")

    # 1) Seed from a saved config file (architecture + defaults).
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        valid = set(ExperimentConfig().to_dict().keys())
        cfg = cfg.clone(**{k: v for k, v in saved.items() if k in valid})

    # 2) Explicit architecture flags override the file.
    over = {}
    if args.seq_len is not None:
        over["seq_len"] = args.seq_len
    if args.hidden is not None:
        over["hidden"] = args.hidden
    if args.n_layers is not None:
        over["n_layers"] = args.n_layers
    if args.dropout is not None:
        over["dropout"] = args.dropout
    if args.no_film:
        over["use_film"] = False
    cfg = cfg.clone(**over)

    # 3) Common CLI options (data budget, epochs, out_dir) win last.
    cfg = _apply_common(cfg, args)

    device = _device()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"Device: {device}  |  config: seq_len={cfg.seq_len} hidden={cfg.hidden} "
          f"n_layers={cfg.n_layers} dropout={cfg.dropout} use_film={cfg.use_film} "
          f"horizon={cfg.horizon} hist={cfg.use_target_history} "
          f"max_sites={cfg.max_sites} years={cfg.years} epochs={cfg.epochs}")
    bank = SiteBank(cfg)
    res = run_experiment(bank, cfg, device=device, save_dir=cfg.out_dir, make_plot=True)
    print("\nTRAIN RESULT")
    print(json.dumps(res.row(), indent=2))

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "train_result.json"), "w", encoding="utf-8") as fh:
        json.dump({"config": cfg.to_dict(), "result": res.row()}, fh, indent=2)


def cmd_horizon(args) -> None:
    cfg = ExperimentConfig(tag="horizon", out_dir="outputs/reports/lstm_film_horizon")
    # Seed architecture from a saved best config if available (but let the scan
    # control horizon / history itself).
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        valid = set(ExperimentConfig().to_dict().keys())
        skip = {"horizon", "use_target_history", "tag", "out_dir"}
        cfg = cfg.clone(**{k: v for k, v in saved.items() if k in valid and k not in skip})
    cfg = _apply_common(cfg, args)
    horizons = [int(h) for h in args.horizons.split(",")]
    device = _device()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"Device: {device}  |  horizons: {horizons}")
    run_horizon_scan(cfg, horizons, device=device)


def cmd_evaluate(args) -> None:
    device = _device()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"Device: {device}  |  evaluating: {args.run_dir}")
    evaluate_run(args.run_dir, out_dir=args.out_dir, device=device)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FiLM-LSTM water-quality experiments")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--target", type=str, default=None,
                        help="target parameter code (e.g. 00400=pH, 00300=DO)")
        sp.add_argument("--dynamic", type=str, default=None,
                        help="comma-separated dynamic input pcodes")
        sp.add_argument("--include-ea", action="store_true",
                        help="also load EA (E*) sites from the clean dir")
        sp.add_argument("--max-sites", type=int, default=None,
                        help="cap number of pH sites (0 = all)")
        sp.add_argument("--years", type=float, default=None,
                        help="keep last N years per site (0 = all history)")
        sp.add_argument("--resample", type=str, default=None, help="pandas offset, e.g. 1h")
        sp.add_argument("--epochs", type=int, default=None)
        sp.add_argument("--max-train-windows", type=int, default=None,
                        help="subsample cap for training windows (0 = no cap)")
        sp.add_argument("--out-dir", type=str, default=None)

    sp = sub.add_parser("smoke", help="tiny plumbing check")
    add_common(sp)
    sp.set_defaults(func=cmd_smoke)

    sp = sub.add_parser("sweep", help="staged architecture sweep")
    add_common(sp)
    sp.add_argument("--preset", choices=["arch", "history"], default="arch",
                    help="which staged search space to run")
    sp.add_argument("--from-best", action="store_true",
                    help="seed the base architecture from a previous best_config.json")
    sp.add_argument("--best-config", type=str, default=None,
                    help="path to best_config.json to seed from (default: out_dir/best_config.json)")
    sp.add_argument("--no-film-stage", action="store_true",
                    help="skip the FiLM on/off ablation stage")
    sp.add_argument("--no-train-best", action="store_true",
                    help="do not retrain the winning config on all sites afterwards")
    sp.add_argument("--full-epochs", type=int, default=25,
                    help="epochs for the final all-sites training")
    sp.add_argument("--full-windows", type=int, default=200000,
                    help="training-window cap for the final all-sites training (0 = no cap)")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("train", help="train a single configuration")
    add_common(sp)
    sp.add_argument("--seq-len", type=int, default=None)
    sp.add_argument("--hidden", type=int, default=None)
    sp.add_argument("--n-layers", type=int, default=None)
    sp.add_argument("--dropout", type=float, default=None)
    sp.add_argument("--no-film", action="store_true")
    sp.add_argument("--config", type=str, default=None, help="JSON overrides (e.g. best_config.json)")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("horizon", help="forecast-horizon breakdown scan")
    add_common(sp)
    sp.add_argument("--horizons", type=str, default="1,3,6,12,24,48",
                    help="comma-separated lead times (in resampled steps)")
    sp.add_argument("--config", type=str, default=None,
                    help="best_config.json to seed the architecture from")
    sp.set_defaults(func=cmd_horizon)

    sp = sub.add_parser("evaluate", help="real-unit metrics + per-site NSE map for a saved run")
    sp.add_argument("--run-dir", type=str, required=True,
                    help="directory containing model.pt (e.g. .../best_full_long)")
    sp.add_argument("--out-dir", type=str, default=None,
                    help="where to write analysis artifacts (default: run-dir)")
    sp.set_defaults(func=cmd_evaluate)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
