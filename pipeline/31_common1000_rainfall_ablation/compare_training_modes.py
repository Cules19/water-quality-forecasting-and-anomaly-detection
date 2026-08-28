"""Compare all-data and non-event teacher-Archive rainfall experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NON_EVENT = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ph_common1000_rainfall_ablation_v1"
    / "full"
    / "reports"
)
ALL_DATA = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ph_common1000_rainfall_ablation_all_v1"
    / "full"
    / "reports"
)
OUTPUT = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ph_common1000_rainfall_training_mode_comparison_v1"
)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def role(run_id: str) -> str:
    if "RAIN_API" in run_id:
        return "rain_api_6h"
    if "RAIN" in run_id:
        return "current_rain"
    return "baseline"


def read_test_metrics(root: Path, training_mode: str) -> pd.DataFrame:
    frame = pd.read_csv(root / "model_metrics_all.csv")
    frame = frame[frame["split"].eq("test")].copy()
    frame["role"] = frame["run_id"].astype(str).map(role)
    frame["training_mode"] = training_mode
    return frame


def read_events(root: Path, training_mode: str) -> pd.DataFrame:
    frame = pd.read_csv(root / "event_enrichment_corrected.csv")
    frame["role"] = frame["run_id"].astype(str).map(role)
    frame["training_mode"] = training_mode
    return frame


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6f}"
            )
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_metrics = read_test_metrics(ALL_DATA, "all_data")
    non_metrics = read_test_metrics(NON_EVENT, "non_event")
    metrics = pd.concat([all_metrics, non_metrics], ignore_index=True)
    write_csv(OUTPUT / "test_metrics_both_training_modes.csv", metrics)

    comparison = all_metrics.merge(
        non_metrics,
        on="role",
        suffixes=("_all_data", "_non_event"),
        validate="one_to_one",
    )
    comparison["rmse_all_minus_non_event"] = (
        comparison["rmse_all_data"] - comparison["rmse_non_event"]
    )
    comparison["rmse_all_relative_change_vs_non_event"] = (
        comparison["rmse_all_minus_non_event"] / comparison["rmse_non_event"]
    )
    comparison["macro_site_rmse_all_minus_non_event"] = (
        comparison["macro_site_rmse_all_data"]
        - comparison["macro_site_rmse_non_event"]
    )
    comparison["macro_site_nse_all_minus_non_event"] = (
        comparison["macro_site_nse_all_data"] - comparison["macro_site_nse_non_event"]
    )
    comparison_columns = [
        "role",
        "run_id_all_data",
        "run_id_non_event",
        "rmse_all_data",
        "rmse_non_event",
        "rmse_all_minus_non_event",
        "rmse_all_relative_change_vs_non_event",
        "macro_site_rmse_all_data",
        "macro_site_rmse_non_event",
        "macro_site_rmse_all_minus_non_event",
        "macro_site_nse_all_data",
        "macro_site_nse_non_event",
        "macro_site_nse_all_minus_non_event",
    ]
    comparison = comparison[comparison_columns]
    write_csv(OUTPUT / "all_data_vs_non_event.csv", comparison)

    all_events = read_events(ALL_DATA, "all_data")
    non_events = read_events(NON_EVENT, "non_event")
    event_comparison = all_events.merge(
        non_events,
        on="role",
        suffixes=("_all_data", "_non_event"),
        validate="one_to_one",
    )
    event_comparison["top5_enrichment_all_minus_non_event"] = (
        event_comparison["exact_test_top5_event_enrichment_all_data"]
        - event_comparison["exact_test_top5_event_enrichment_non_event"]
    )
    event_columns = [
        "role",
        "exact_test_top5_event_enrichment_all_data",
        "exact_test_top5_event_enrichment_non_event",
        "top5_enrichment_all_minus_non_event",
        "validation_threshold_test_event_enrichment_all_data",
        "validation_threshold_test_event_enrichment_non_event",
    ]
    event_comparison = event_comparison[event_columns]
    write_csv(OUTPUT / "event_enrichment_training_mode_comparison.csv", event_comparison)

    all_baseline = all_metrics.set_index("role").loc["baseline"]
    all_rain = all_metrics.set_index("role").loc["current_rain"]
    all_api = all_metrics.set_index("role").loc["rain_api_6h"]
    paired = pd.read_csv(ALL_DATA / "paired_site_comparison.csv")
    api_pair = paired[paired["comparison"].str.contains("RAIN_API")].iloc[0]
    common_key_equal = sha256(ALL_DATA / "common_key_manifest.csv") == sha256(
        NON_EVENT / "common_key_manifest.csv"
    )
    all_audit = json.loads((ALL_DATA / "independent_audit.json").read_text())
    non_audit = json.loads((NON_EVENT / "independent_audit.json").read_text())
    all_extended = json.loads((ALL_DATA / "extended_diagnostics_audit.json").read_text())
    non_extended = json.loads(
        (NON_EVENT / "extended_diagnostics_audit.json").read_text()
    )

    report = "\n".join(
        [
            "# All-data versus non-event rainfall experiment",
            "",
            "The site cohort, split, common target keys, architecture, feature transforms,",
            "random seed and evaluation rows are unchanged. The only requested conceptual",
            "change is that all-data training allows event target rows.",
            "",
            "## All-data rainfall ablation",
            "",
            markdown_table(
                all_metrics[
                    [
                        "role",
                        "n",
                        "rmse",
                        "mae",
                        "nse",
                        "macro_site_rmse",
                        "macro_site_nse",
                    ]
                ]
            ),
            "",
            f"- Current rain versus all-data baseline RMSE change: "
            f"{(all_rain['rmse'] / all_baseline['rmse'] - 1) * 100:+.3f}%.",
            f"- Rain + API + 6h versus all-data baseline RMSE change: "
            f"{(all_api['rmse'] / all_baseline['rmse'] - 1) * 100:+.3f}%.",
            f"- Rain + API + 6h improves RMSE at "
            f"{int(api_pair['rmse_improved_sites'])}/135 sites.",
            f"- Its equal-site bootstrap mean RMSE-difference 95% interval is "
            f"[{api_pair['bootstrap_mean_site_rmse_delta_ci_low']:.6f}, "
            f"{api_pair['bootstrap_mean_site_rmse_delta_ci_high']:.6f}].",
            "",
            "## Training-mode comparison",
            "",
            markdown_table(
                comparison[
                    [
                        "role",
                        "rmse_all_data",
                        "rmse_non_event",
                        "rmse_all_relative_change_vs_non_event",
                        "macro_site_rmse_all_data",
                        "macro_site_rmse_non_event",
                        "macro_site_nse_all_data",
                        "macro_site_nse_non_event",
                    ]
                ]
            ),
            "",
            "## Event-enrichment trade-off",
            "",
            markdown_table(event_comparison[event_columns[:4]]),
            "",
            "All-data training lowers ordinary forecasting error, especially for the full",
            "rain/API model. Non-event training retains stronger top-5% residual/event",
            "enrichment in every matched feature configuration. This reproduces the central",
            "trade-off: all-data training is preferable for predictive accuracy, whereas",
            "non-event training is preferable when residuals are intended as event signals.",
            "",
            "## Audit",
            "",
            f"- Common-key manifest identical across training modes: {common_key_equal}.",
            f"- All-data training event windows removed: "
            f"{int(all_metrics['train_event_windows_removed'].sum())} across three reported rows.",
            f"- All-data core/extended audits: {all_audit['status']} / {all_extended['status']}.",
            f"- Non-event core/extended audits: {non_audit['status']} / {non_extended['status']}.",
            "",
        ]
    )
    (OUTPUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    problems = []
    if not common_key_equal:
        problems.append("common-key manifests differ")
    if not (all_metrics["train_event_windows_removed"] == 0).all():
        problems.append("all-data training removed event windows")
    if not all(
        status == "pass"
        for status in (
            all_audit["status"],
            non_audit["status"],
            all_extended["status"],
            non_extended["status"],
        )
    ):
        problems.append("an upstream audit failed")
    audit = {
        "status": "pass" if not problems else "fail",
        "common_key_manifest_identical": common_key_equal,
        "test_rows_per_model": int(all_metrics["n"].iloc[0]),
        "all_data_event_windows_removed": int(
            all_metrics["train_event_windows_removed"].max()
        ),
        "all_data_api_vs_baseline_rmse_change_fraction": float(
            all_api["rmse"] / all_baseline["rmse"] - 1
        ),
        "all_data_api_rmse_improved_sites": int(api_pair["rmse_improved_sites"]),
        "problems": problems,
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(OUTPUT / "audit.json", audit)
    if problems:
        raise RuntimeError(problems)
    print(report)


if __name__ == "__main__":
    main()
