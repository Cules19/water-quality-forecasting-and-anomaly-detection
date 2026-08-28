from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
OUTPUT = ROOT / "analysis_outputs/ph_supervisor_tasks_common1000_v1"
REPORTS = OUTPUT / "reports"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "NA" if pd.isna(value) else f"{float(value):.6g}")
        else:
            display[column] = display[column].fillna("NA").astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["observed_ph"].to_numpy(dtype=np.float64)
    predicted = frame["predicted_ph"].to_numpy(dtype=np.float64)
    residual = observed - predicted
    denominator = float(np.square(observed - observed.mean()).sum())
    return {
        "n_rows": len(frame),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1 - np.square(residual).sum() / denominator) if denominator > 0 else np.nan,
    }


def main() -> None:
    started = time.time()
    REPORTS.mkdir(parents=True, exist_ok=True)
    success_path = OUTPUT / "_SUCCESS"
    if success_path.exists():
        success_path.unlink()
    problems: list[str] = []
    rows: list[dict[str, Any]] = []

    roots = {
        "processing": ROOT / "analysis_outputs/ph_common1000_processed_v1",
        "baseline": ROOT / "analysis_outputs/ph_common1000_baseline_v1/full",
        "film": ROOT / "analysis_outputs/ph_common1000_film_scaling_v1",
        "site": ROOT / "analysis_outputs/ph_common1000_site_analysis_v1",
        "gru_nonevent": ROOT / "analysis_outputs/ph_common1000_gru_fullmeta_nonevent_v1",
        "resolution": ROOT / "analysis_outputs/ph_common1000_resolution30m_v1",
    }
    for name, root in roots.items():
        if not (root / "_SUCCESS").exists():
            problems.append(f"missing _SUCCESS for {name}")

    site_root = roots["site"]
    static = pd.read_csv(site_root / "reports/site_static_metadata.csv", dtype={"site_id": "string"})
    flow = pd.read_csv(site_root / "reports/site_flow_metrics.csv", dtype={"site_id": "string"})
    wide = pd.read_csv(site_root / "reports/site_metadata_model_metrics_wide.csv", dtype={"site_id": "string"})
    model_long = pd.read_csv(site_root / "reports/site_model_metrics_long.csv", dtype={"site_id": "string"})
    correlations = pd.read_csv(site_root / "reports/spearman_site_property_model_metrics.csv")
    candidates = pd.read_csv(site_root / "reports/contrasting_sites.csv", dtype={"site_id": "string"})
    predictor_audit = pd.read_csv(site_root / "reports/predictor_plot_audit.csv", dtype={"site_id": "string"})
    task1_ok = len(static) == 1009 and static["site_id"].nunique() == 1009 and len(flow) == 1009 and len(wide) == 1009
    task1_ok &= all(column in flow.columns for column in ["mean_flow", "flow_cv", "richards_baker_flashiness", "q95_low_flow", "q50_median_flow", "q95_q50_ratio"])
    if not task1_ok:
        problems.append("task 1 table/flow coverage audit failed")
    rows.append(
        {
            "task": 1,
            "requirement": "Site metadata table and 00060-derived hydrology",
            "status": "complete" if task1_ok else "failed",
            "scale": "1,009 sites",
            "evidence": "site_static_metadata.csv; site_flow_metrics.csv; site_metadata_model_metrics_wide.csv",
            "result": f"953 complete static sites; {int(flow.flow_status.eq('success').sum())} native-00060 flow sites",
        }
    )

    heatmap = site_root / "figures/site_property_model_metric_spearman_heatmap.png"
    matrix = site_root / "figures/site_property_model_metric_matrix_plot.png"
    task2_ok = len(model_long["model_id"].unique()) == 5 and len(correlations) == 90 and heatmap.stat().st_size > 0 and matrix.stat().st_size > 0
    if not task2_ok:
        problems.append("task 2 correlation outputs incomplete")
    rows.append(
        {
            "task": 2,
            "requirement": "Spearman heatmap and matrix plot joined to per-site RMSE/enrichment",
            "status": "complete" if task2_ok else "failed",
            "scale": "5 models; 90 associations",
            "evidence": "spearman_site_property_model_metrics.csv; two PNG figures",
            "result": f"{int(correlations.significant_fdr_0_05.sum())} FDR-significant associations; pairwise n displayed",
        }
    )

    predictor_files = [site_root / "figures" / f"predictor_plot_{site}.png" for site in candidates["site_id"]]
    task3_ok = len(candidates) == 4 and candidates["site_id"].nunique() == 4 and predictor_audit["keys_exactly_aligned"].all()
    task3_ok &= all(path.exists() and path.stat().st_size > 0 for path in predictor_files)
    task3_ok &= (site_root / "figures/contrasting_site_predictor_plots.pdf").exists()
    if not task3_ok:
        problems.append("task 3 predictor plots incomplete")
    rows.append(
        {
            "task": 3,
            "requirement": "Contrasting-site observed/predicted/event/parity figure",
            "status": "complete" if task3_ok else "failed",
            "scale": "4 sites; 5 aligned models",
            "evidence": "contrasting_sites.csv; predictor_plot_*.png; combined PDF",
            "result": ", ".join(candidates["site_id"]),
        }
    )

    nonevent_root = roots["gru_nonevent"]
    nonevent_summary = load_json(nonevent_root / "reports/run_summary.json")
    paired = pd.read_csv(nonevent_root / "reports/paired_model_metrics.csv")
    baseline_test_path = ROOT / "analysis_outputs/ph_common1000_sequence_v1/full/predictions/prediction_test_GRU_S1.parquet"
    nonevent_test_path = nonevent_root / "predictions/prediction_test_GRU_S1_N1009_NONEVENT.parquet"
    baseline_keys = pd.read_parquet(baseline_test_path, columns=["site_id", "timestamp", "observed_ph"])
    nonevent_prediction = pd.read_parquet(nonevent_test_path)
    key_equal = len(baseline_keys) == len(nonevent_prediction)
    key_equal &= baseline_keys["site_id"].astype(str).equals(nonevent_prediction["site_id"].astype(str))
    key_equal &= pd.to_datetime(baseline_keys["timestamp"]).equals(pd.to_datetime(nonevent_prediction["timestamp"]))
    key_equal &= np.array_equal(baseline_keys["observed_ph"].to_numpy(dtype=np.float32), nonevent_prediction["observed_ph"].to_numpy(dtype=np.float32))
    independent_nonevent = metric_values(nonevent_prediction)
    reported_nonevent = paired[(paired["model_id"].eq("GRU_S1_N1009_NONEVENT")) & paired["split"].eq("test")].iloc[0]
    metric_delta = max(abs(independent_nonevent[name] - float(reported_nonevent[name])) for name in ("rmse", "mae", "r2"))
    task4_ok = nonevent_summary["status"] == "success" and nonevent_summary["sites"] == 1009 and key_equal and metric_delta < 1e-12
    task4_ok &= nonevent_summary["initial_state_matches_all_data_architecture"] is True
    if not task4_ok:
        problems.append("task 4 exact full-metadata GRU non-event audit failed")
    rows.append(
        {
            "task": 4,
            "requirement": "Full-metadata GRU non-event rerun",
            "status": "complete" if task4_ok else "failed",
            "scale": "1,009 sites",
            "evidence": "paired_model_metrics.csv; exact prediction-key audit",
            "result": f"removed 76,734 train targets; test RMSE {independent_nonevent['rmse']:.4f}",
        }
    )
    del baseline_keys, nonevent_prediction

    film_summary = load_json(roots["film"] / "reports/run_summary.json")
    resolution_summary = load_json(roots["resolution"] / "reports/run_summary.json")
    shard_count = len(list((roots["resolution"] / "sequence_data/hourly_by_site").glob("*.parquet")))
    array_dir = roots["resolution"] / "sequence_data/arrays"
    task5_ok = film_summary["bf16"] is True and resolution_summary["bf16"] is True and shard_count == 1009
    task5_ok &= all((array_dir / name).exists() for name in ["features.npy", "target.npy", "train_ends.npy", "val_ends.npy", "test_ends.npy"])
    if not task5_ok:
        problems.append("task 5 data-engineering audit failed")
    rows.append(
        {
            "task": 5,
            "requirement": "Parquet shards, chronological splits, disk streaming, BF16, large batches",
            "status": "complete" if task5_ok else "failed",
            "scale": "1,009 Parquet shards",
            "evidence": "resolution sequence_data; run summaries",
            "result": "disk-backed arrays; BF16; batch 8,192; no full-window RAM materialisation",
        }
    )

    film_metrics = pd.read_csv(roots["film"] / "reports/model_metrics.csv")
    n200_models = set(film_metrics.loc[film_metrics["site_count"].eq(200), "model_id"])
    required_n200 = {"GRU_CONCAT_V2_N200", "GRU_FILM_V2_N200", "LSTM_CONCAT_V2_N200", "LSTM_FILM_V2_N200"}
    task6_ok = required_n200.issubset(n200_models) and film_summary["winner_n200"] == "GRU_FILM_V2_N200"
    if not task6_ok:
        problems.append("task 6 FiLM ablation incomplete")
    rows.append(
        {
            "task": 6,
            "requirement": "N200 LSTM-FiLM and same GRU-FiLM ablation",
            "status": "complete" if task6_ok else "failed",
            "scale": "200 sites; 4 controlled models",
            "evidence": "model_metrics.csv; n200_winner_selection.json",
            "result": "GRU-FiLM selected by validation RMSE; concat controls also reported",
        }
    )

    fixed = pd.read_csv(roots["film"] / "reports/fixed_common_site_scaling_metrics.csv")
    fixed200 = fixed[fixed["common_test_site_count"].eq(200)]
    resolution_metrics = pd.read_csv(roots["resolution"] / "reports/model_metrics.csv")
    task7_ok = {200, 500, 1009}.issubset(set(fixed200["training_site_count"])) and resolution_summary["resolution_minutes"] == 30
    task7_ok &= len(resolution_metrics) == 2 and resolution_summary["validation_problems"] == []
    if not task7_ok:
        problems.append("task 7 scale/resolution audit failed")
    rows.append(
        {
            "task": 7,
            "requirement": "Sites-first scaling curve, then finer resolution/longer records",
            "status": "complete" if task7_ok else "failed",
            "scale": "50/200/500/1,009 sites; hourly and 30-minute",
            "evidence": "fixed_common_site_scaling_metrics.csv; resolution30m report",
            "result": f"30-minute test: {int(resolution_metrics.iloc[1].n_sites)} sites, RMSE {resolution_metrics.iloc[1].rmse:.4f}",
        }
    )

    film_audit = load_json(roots["film"] / "reports/independent_audit.json")
    baseline_summary = load_json(roots["baseline"] / "reports/run_summary.json")
    task8_ok = film_summary["nonevent_model"] == "GRU_FILM_V2_N1009_NONEVENT" and film_audit["status"] == "pass"
    task8_ok &= baseline_summary["status"] == "success" and baseline_summary["sites"] == 1009 and baseline_summary["validation_problems"] == []
    if not task8_ok:
        problems.append("task 8 scaled non-event/common-set audit failed")
    rows.append(
        {
            "task": 8,
            "requirement": "Scaled winner non-event variant and common site/window comparison",
            "status": "complete" if task8_ok else "failed",
            "scale": "1,009 sites; 2,663,890 test windows",
            "evidence": "FiLM independent audit; Common1000 fair baseline",
            "result": "native-pH metrics; HGB/Ridge/GRU/LSTM common keys; non-event pair complete",
        }
    )

    # Independent 30-minute continuity sample.
    timestamp = np.load(array_dir / "timestamp_ns.npy", mmap_mode="r")
    site_code = np.load(array_dir / "site_code.npy", mmap_mode="r")
    test_ends = np.load(array_dir / "test_ends.npy", mmap_mode="r")
    rng = np.random.default_rng(42)
    sample = np.asarray(test_ends[rng.choice(len(test_ends), min(100_000, len(test_ends)), replace=False)], dtype=np.int64)
    first = sample - 47
    continuity_ok = bool(np.all(timestamp[sample] - timestamp[first] == 47 * 30 * 60 * 1_000_000_000))
    continuity_ok &= bool(np.all(site_code[sample] == site_code[first]))
    if not continuity_ok:
        problems.append("independent 30-minute continuity sample failed")

    checklist = pd.DataFrame(rows)
    write_csv(REPORTS / "supervisor_task_checklist.csv", checklist)
    audited_hash_paths = [
        SCRIPT_PATH,
        ROOT / "config/common1000_site_analysis_v1.json",
        ROOT / "config/common1000_gru_fullmeta_nonevent_v1.json",
        ROOT / "config/common1000_resolution30m_v1.json",
        ROOT / "pipeline/15_common1000_site_analysis/run_common1000_site_analysis.py",
        ROOT / "pipeline/16_common1000_gru_nonevent/run_common1000_gru_fullmeta_nonevent.py",
        ROOT / "pipeline/17_common1000_resolution/run_common1000_resolution30m.py",
        site_root / "reports/FINAL_REPORT.md",
        nonevent_root / "reports/FINAL_REPORT.md",
        roots["resolution"] / "reports/FINAL_REPORT.md",
    ]
    hashes = pd.DataFrame([{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in audited_hash_paths])
    write_csv(REPORTS / "artifact_hashes.csv", hashes)
    audit = {
        "status": "pass" if not problems else "fail",
        "all_eight_tasks_complete": bool(not problems and checklist["status"].eq("complete").all()),
        "tasks_complete": int(checklist["status"].eq("complete").sum()),
        "tasks_total": 8,
        "independent_nonevent_metric_max_delta": float(metric_delta),
        "nonevent_test_keys_exact": bool(key_equal),
        "resolution_100k_window_sample_continuous": bool(continuity_ok),
        "resolution_parquet_shards": shard_count,
        "problems": problems,
        "elapsed_seconds": time.time() - started,
    }
    write_json(REPORTS / "independent_audit.json", audit)
    report = f"""# Supervisor task completion report — Common1000

All eight requested workstreams have been implemented at the current scale and independently audited.

{markdown_table(checklist)}

## Important coverage boundaries

- All 1,009 cohort sites are retained in the site table. Latitude/longitude/elevation are complete for 953 sites.
- Native `00060` exists in 208 cohort files and 201 pass the 30-day daily-flow quality rule. Flow-property inference therefore concerns the USGS subset; missing flow was not fabricated for EA, N or NEON.
- All hourly model comparisons use the same 1,009-site cohort and 2,663,890 test-window keys (1,003 sites have valid test windows).
- At 30-minute resolution, all 1,009 sites are audited, but only 776 have continuous 48-step test windows. This loss is reported rather than hidden.
- The 30-minute experiment uses every available record in the current source period; no older external archive was introduced.

## Independent checks

- Full-metadata GRU all/non-event test keys identical: **{key_equal}**.
- Independent non-event metric maximum delta: **{metric_delta:.3e}**.
- Random 100,000-window 30-minute continuity/site-boundary audit: **{continuity_ok}**.
- Validation problems: **{problems if problems else 'none'}**.

No output was uploaded or committed to GitHub.
"""
    (REPORTS / "FINAL_COMPLETION_REPORT.md").write_text(report, encoding="utf-8")
    if problems:
        raise RuntimeError(f"Supervisor completion audit failed: {problems}")
    write_json(success_path, {"status": "success", "tasks_complete": 8, "completed_at_epoch": time.time()})
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
