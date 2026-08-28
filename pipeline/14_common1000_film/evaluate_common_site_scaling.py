#!/usr/bin/env python
"""Evaluate trained FiLM scaling models on fixed common N50 and N200 test cohorts."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
MAIN_SCRIPT = SCRIPT_PATH.parent / "run_common1000_film_scaling.py"
SPEC = importlib.util.spec_from_file_location("film_main", MAIN_SCRIPT)
FILM = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FILM)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "common1000_film_scaling_v1.json",
    )
    return parser.parse_args()


def report_text(metrics: pd.DataFrame) -> str:
    lines = [
        "# Fixed-common-site scaling evaluation",
        "",
        "This evaluation separates training-site scale from changing test-cohort difficulty.",
        "",
    ]
    for common_count in (50, 200):
        subset = metrics[metrics["common_test_site_count"].eq(common_count)].sort_values("training_site_count")
        lines.extend(
            [
                f"## Fixed first {common_count} test sites",
                "",
                "| Training sites | Test sites observed | Test rows | RMSE | R2 | Absolute top-5% enrichment |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in subset.itertuples(index=False):
            lines.append(
                f"| {row.training_site_count} | {row.n_sites} | {row.n_rows:,} | {row.rmse:.6f} | "
                f"{row.r2:.6f} | {row.absolute_top5_event_enrichment:.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "- The N200 fixed-site table is the primary 200 -> 500 -> 1009 scaling comparison.",
            "- Every row within a fixed-site table uses exactly the same target timestamps and event labels.",
            "- Models retain their own training-cohort scalers and V2 metadata transforms; only evaluation rows are fixed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = FILM.load_json(args.config.resolve())
    output = FILM.resolve(config["output_root"])
    if not (output / "_SUCCESS").exists():
        raise FileNotFoundError("Complete FiLM training output is required")
    summary = FILM.load_json(output / "reports" / "run_summary.json")
    winner_base = str(summary["winner_base"])
    manifest, arrays, cohort = FILM.load_inputs(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    network_rows: list[dict[str, Any]] = []
    key_audit: dict[int, np.ndarray] = {}

    for training_count in [int(value) for value in config["site_counts"]]:
        model_id = f"{winner_base}_N{training_count}"
        metadata = np.load(output / "metadata" / f"metadata_N{training_count}.npy")
        scaler = FILM.load_json(output / "scalers" / f"dynamic_scaler_N{training_count}.json")
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        scale = np.asarray(scaler["scale"], dtype=np.float32)
        checkpoint = torch.load(
            output / "models" / model_id / "best_model.pt", map_location=device, weights_only=True
        )
        model_spec = checkpoint["model_spec"]
        model = FILM.create_model(model_spec, config, training_count, metadata.shape[1], device)
        model.load_state_dict(checkpoint["model_state_dict"])
        common_counts = [50] + ([200] if training_count >= 200 else [])
        for common_count in common_counts:
            ends = FILM.subset_ends(arrays["test_ends"], arrays, common_count)
            prediction = FILM.predict_scaled(model, ends, arrays, mean, scale, metadata, config, device)
            frame = FILM.prediction_frame(
                model_id,
                "test",
                ends,
                prediction,
                arrays,
                manifest,
                cohort,
                float(config["target_inverse_multiplier"]),
            )
            keys = pd.util.hash_pandas_object(frame[["site_id", "timestamp", "observed_ph"]], index=False).to_numpy()
            if common_count not in key_audit:
                key_audit[common_count] = keys
            elif not np.array_equal(keys, key_audit[common_count]):
                raise ValueError(f"Fixed N{common_count} test keys differ for {model_id}")
            metric = FILM.metric_values(frame["observed_ph"], frame["predicted_ph"])
            tail = pd.DataFrame(FILM.tail_analysis(frame, [0.05]))
            absolute = tail[tail["tail_type"].eq("absolute")].iloc[0]
            rows.append(
                {
                    "model_id": model_id,
                    "training_site_count": training_count,
                    "common_test_site_count": common_count,
                    "n_sites": int(frame["site_id"].nunique()),
                    **metric,
                    "absolute_top5_event_enrichment": absolute.event_enrichment,
                    "negative_top5_event_enrichment": tail[tail["tail_type"].eq("negative")]["event_enrichment"].iloc[0],
                    "positive_top5_event_enrichment": tail[tail["tail_type"].eq("positive")]["event_enrichment"].iloc[0],
                }
            )
            for network, group in frame.groupby("network", sort=True):
                network_rows.append(
                    {
                        "model_id": model_id,
                        "training_site_count": training_count,
                        "common_test_site_count": common_count,
                        "network": network,
                        "n_sites": int(group["site_id"].nunique()),
                        **FILM.metric_values(group["observed_ph"], group["predicted_ph"]),
                    }
                )
        del model
        torch.cuda.empty_cache()

    metrics = pd.DataFrame(rows).sort_values(["common_test_site_count", "training_site_count"])
    FILM.write_csv(output / "reports" / "fixed_common_site_scaling_metrics.csv", metrics)
    FILM.write_csv(output / "reports" / "fixed_common_site_network_metrics.csv", pd.DataFrame(network_rows))
    report = report_text(metrics)
    (output / "reports" / "COMMON_SITE_SCALING_REPORT.md").write_text(report, encoding="utf-8")

    common200 = metrics[metrics["common_test_site_count"].eq(200)].sort_values("training_site_count")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(common200["training_site_count"], common200["rmse"], marker="o", color="#4C78A8")
    ax2.plot(
        common200["training_site_count"],
        common200["absolute_top5_event_enrichment"],
        marker="s",
        color="#E45756",
    )
    ax1.set_xlabel("Training site count")
    ax1.set_ylabel("Fixed-N200 test RMSE", color="#4C78A8")
    ax2.set_ylabel("Fixed-N200 absolute top-5% enrichment", color="#E45756")
    ax1.set_title(f"{winner_base}: fixed-common-site scaling")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "figures" / "fixed_n200_scaling_rmse_enrichment.png", dpi=200)
    plt.close(fig)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
