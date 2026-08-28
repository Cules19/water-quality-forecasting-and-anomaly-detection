"""Freeze CHEAP3 versus CHEAP3_RAIN using validation data only."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (
    PROJECT_ROOT / "analysis_outputs" / "h0_site_transfer_warmup_final_audit_v1"
)
RUN_ROOT = (
    OUTPUT_ROOT
    / "checkpoints"
    / "ph_h01_input_ablation_v1"
    / "full_validation_only"
)
CONFIG_PATH = PROJECT_ROOT / "config" / "h0_final_ph_h01_cheap3_validation_v1.json"
OLD_RAIN_CHECKPOINT = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ph_common1000_lstm_film_ph_sensor_ablation_v1"
    / "full"
    / "models"
    / "ph_LSTM_LOW_COST_H01.pt"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def metrics(run_id: str) -> tuple[dict[str, Any], pd.DataFrame]:
    overall = pd.read_csv(RUN_ROOT / "reports" / "model_metrics_all.csv")
    row = overall.loc[overall["run_id"].eq(run_id)].iloc[0]
    sites = pd.read_csv(RUN_ROOT / "reports" / f"site_metrics_{run_id}.csv")
    if set(sites["split"].astype(str)) != {"validation"}:
        raise RuntimeError(f"Non-validation rows found for {run_id}")
    result = {
        "run_id": run_id,
        "pooled_rmse": float(row["rmse"]),
        "pooled_mae": float(row["mae"]),
        "pooled_nse": float(row["nse"]),
        "macro_site_rmse": float(row["macro_site_rmse"]),
        "macro_site_mae": float(row["macro_site_mae"]),
        "median_site_nse": float(sites["nse"].median()),
        "nse_above_zero_sites": int((sites["nse"] > 0).sum()),
        "sites": int(sites["site_id"].nunique()),
        "rows": int(row["n"]),
    }
    return result, sites


def main() -> None:
    audit = json.loads(
        (RUN_ROOT / "reports" / "independent_audit.json").read_text(encoding="utf-8")
    )
    if audit.get("test_predictions_generated") or audit.get("test_metrics_generated"):
        raise RuntimeError("Selection run is not validation-only")
    cheap, cheap_sites = metrics("ph_LSTM_CHEAP3_H01")
    rain, rain_sites = metrics("ph_LSTM_LOW_COST_H01")
    paired = cheap_sites[["site_id", "rmse", "mae", "nse"]].merge(
        rain_sites[["site_id", "rmse", "mae", "nse"]],
        on="site_id",
        suffixes=("_cheap3", "_cheap3_rain"),
        validate="one_to_one",
    )
    cheap_improved = int((paired["mae_cheap3"] < paired["mae_cheap3_rain"]).sum())
    rain_improved = int((paired["mae_cheap3_rain"] < paired["mae_cheap3"]).sum())
    ties = int((paired["mae_cheap3_rain"] == paired["mae_cheap3"]).sum())
    rain_relative_improvement = (
        (cheap["macro_site_mae"] - rain["macro_site_mae"])
        / cheap["macro_site_mae"]
        * 100.0
    )
    rain_site_improvement_fraction = rain_improved / len(paired)
    relative_gap = (
        abs(cheap["macro_site_mae"] - rain["macro_site_mae"])
        / min(cheap["macro_site_mae"], rain["macro_site_mae"])
        * 100.0
    )
    if relative_gap < 2.0:
        selected = "CHEAP3"
        reason = (
            "Validation macro-site MAE differs by less than 2%; the preregistered "
            "tie rule selects CHEAP3."
        )
    elif rain_relative_improvement >= 2.0 and rain_site_improvement_fraction >= 0.60:
        selected = "CHEAP3_RAIN"
        reason = (
            "CHEAP3_RAIN improves validation macro-site MAE by at least 2% and "
            "improves at least 60% of validation sites."
        )
    else:
        selected = "CHEAP3"
        reason = (
            "CHEAP3_RAIN does not satisfy the preregistered 2% macro-MAE and "
            "60%-of-sites requirements; CHEAP3 is selected."
        )

    checkpoints = {
        "CHEAP3": RUN_ROOT / "models" / "ph_LSTM_CHEAP3_H01.pt",
        "CHEAP3_RAIN": RUN_ROOT / "models" / "ph_LSTM_LOW_COST_H01.pt",
    }
    stored_rain_match = sha256(checkpoints["CHEAP3_RAIN"]) == sha256(OLD_RAIN_CHECKPOINT)
    selected_checkpoint = checkpoints[selected]
    split_path = (
        PROJECT_ROOT
        / "analysis_outputs"
        / "ph_common1000_leave_sites_out_v1"
        / "full"
        / "reports"
        / "site_split_ph.csv"
    )
    payload = {
        "status": "frozen_on_validation",
        "selected_input_set": selected,
        "selection_metric": "validation_macro_mae",
        "validation_results": {"CHEAP3": cheap, "CHEAP3_RAIN": rain},
        "paired_validation_sites": {
            "sites": len(paired),
            "CHEAP3_improved_sites_by_mae": cheap_improved,
            "CHEAP3_RAIN_improved_sites_by_mae": rain_improved,
            "ties": ties,
            "CHEAP3_RAIN_improved_fraction": rain_site_improvement_fraction,
        },
        "decision_rule": (
            "Compare validation macro-site MAE; if relative difference <2%, choose "
            "CHEAP3. Otherwise choose CHEAP3_RAIN only if its macro-site MAE improves "
            "by >=2% and >=60% of validation sites improve; else choose CHEAP3."
        ),
        "decision_reason": reason,
        "cheap3_rain_relative_macro_mae_improvement_percent": rain_relative_improvement,
        "relative_macro_mae_gap_percent": relative_gap,
        "selected_checkpoint_path": str(selected_checkpoint.resolve()),
        "selected_checkpoint_sha256": sha256(selected_checkpoint),
        "checkpoint_sha256": {key: sha256(path) for key, path in checkpoints.items()},
        "stored_rain_checkpoint_path": str(OLD_RAIN_CHECKPOINT.resolve()),
        "stored_rain_checkpoint_sha256": sha256(OLD_RAIN_CHECKPOINT),
        "rain_checkpoint_exactly_matches_stored_warmup": stored_rain_match,
        "selection_run_test_predictions_generated": False,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(CONFIG_PATH.resolve()),
        "config_sha256": sha256(CONFIG_PATH),
        "effective_config_sha256": sha256(RUN_ROOT / "reports" / "effective_config.json"),
        "split_path": str(split_path.resolve()),
        "split_sha256": sha256(split_path),
        "common_key_manifest_sha256": sha256(
            RUN_ROOT / "reports" / "common_key_manifest.csv"
        ),
    }
    frozen = OUTPUT_ROOT / "frozen_selection"
    write_json(frozen / "FINAL_H0_INPUT_SELECTION.json", payload)
    lines = [
        "# Frozen H01 low-cost input selection",
        "",
        "Selection used the 137 validation sites only. No test predictions or test metrics were generated.",
        "",
        "| Input set | Pooled RMSE | Pooled MAE | Macro RMSE | Macro MAE | Median site NSE | NSE > 0 | MAE wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| CHEAP3 | {cheap['pooled_rmse']:.6f} | {cheap['pooled_mae']:.6f} | "
        f"{cheap['macro_site_rmse']:.6f} | {cheap['macro_site_mae']:.6f} | "
        f"{cheap['median_site_nse']:.6f} | {cheap['nse_above_zero_sites']}/{cheap['sites']} | "
        f"{cheap_improved}/{len(paired)} |",
        f"| CHEAP3_RAIN | {rain['pooled_rmse']:.6f} | {rain['pooled_mae']:.6f} | "
        f"{rain['macro_site_rmse']:.6f} | {rain['macro_site_mae']:.6f} | "
        f"{rain['median_site_nse']:.6f} | {rain['nse_above_zero_sites']}/{rain['sites']} | "
        f"{rain_improved}/{len(paired)} |",
        "",
        f"**Frozen selection: `{selected}`.** {reason}",
        "",
        f"CHEAP3_RAIN relative macro-MAE improvement: {rain_relative_improvement:.3f}%.",
        f"CHEAP3_RAIN validation-site improvement fraction: {rain_site_improvement_fraction:.3%}.",
        f"The validation CHEAP3_RAIN checkpoint exactly matches the stored warm-up checkpoint: `{stored_rain_match}`.",
        "",
    ]
    text = "\n".join(lines)
    write_text(frozen / "FINAL_H0_INPUT_SELECTION.md", text)
    write_text(OUTPUT_ROOT / "reports" / "CHEAP3_VS_CHEAP3_RAIN_REPORT.md", text)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
