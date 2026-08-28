#!/usr/bin/env python
"""Independent post-run audit for Common1000 fair baseline outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
MAIN_MODEL_IDS = [
    "B0_site_mean",
    "B1_seasonal_mean",
    "B2_persistence_1h",
    "B3_R0_ridge",
    "B3_R1_ridge_metadata",
    "B4_H0_hgb",
    "B4_H1_hgb_metadata",
]
NONEVENT_MODEL_IDS = [
    "B3_R0_ridge_nonevent",
    "B3_R1_ridge_metadata_nonevent",
    "B4_H0_hgb_nonevent",
    "B4_H1_hgb_metadata_nonevent",
]
SEQUENCE_MODEL_IDS = ["GRU_S0", "GRU_S1", "LSTM_S1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "common1000_baseline_v1.json",
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    y64 = np.asarray(y, dtype=np.float64)
    p64 = np.asarray(p, dtype=np.float64)
    residual = y64 - p64
    sse = float(np.square(residual).sum())
    denominator = float(np.square(y64 - y64.mean()).sum())
    return (
        math.sqrt(sse / len(y64)),
        float(np.abs(residual).mean()),
        1 - sse / denominator if denominator > 0 else np.nan,
    )


def key_hash(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame[["site_id", "timestamp"]], index=False).to_numpy(dtype=np.uint64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = resolve(config["output_root"]) / args.mode
    sequence = resolve(config["sequence_root"]) / args.mode
    reports = output / "reports"
    arrays_dir = sequence / "sequence_data" / "arrays"
    problems: list[str] = []
    checks: list[dict[str, Any]] = []
    report_metrics = pd.read_csv(reports / "model_comparison_metrics.csv")
    network_report = pd.read_csv(reports / "network_metrics.csv")
    site_report = pd.read_csv(reports / "site_level_metrics.csv", dtype={"site_id": "string"})
    tail_report = pd.read_csv(reports / "residual_tail_analysis.csv")
    reference_hash: dict[str, np.ndarray] = {}
    reference_observed: dict[str, np.ndarray] = {}

    for model_id in [*MAIN_MODEL_IDS, *NONEVENT_MODEL_IDS]:
        for split in ("val", "test"):
            path = output / "predictions" / f"prediction_{split}_{model_id}.parquet"
            frame = pd.read_parquet(path)
            frame["site_id"] = frame["site_id"].astype(str)
            key = f"{model_id}/{split}"
            nulls = int(frame.isna().sum().sum())
            duplicates = int(frame.duplicated(["site_id", "timestamp"]).sum())
            numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64, copy=False)
            nonfinite = int((~np.isfinite(numeric)).sum())
            current_hash = key_hash(frame)
            if model_id == "B0_site_mean":
                reference_hash[split] = current_hash
                reference_observed[split] = frame["observed_ph"].to_numpy(dtype=np.float32)
            else:
                if not np.array_equal(current_hash, reference_hash[split]):
                    problems.append(f"{key}: endpoint keys differ from B0")
                if not np.allclose(frame["observed_ph"], reference_observed[split], atol=1e-6):
                    problems.append(f"{key}: observed pH differs from B0")
            rmse, mae, r2 = metrics(frame["observed_ph"], frame["predicted_ph"])
            expected = report_metrics[
                report_metrics["model_id"].eq(model_id) & report_metrics["split"].eq(split)
            ].iloc[0]
            metric_delta = max(abs(rmse - expected.rmse), abs(mae - expected.mae), abs(r2 - expected.r2))
            network_delta = 0.0
            for network, group in frame.groupby("network", sort=False):
                a, b, c = metrics(group["observed_ph"], group["predicted_ph"])
                expected_network = network_report[
                    network_report["model_id"].eq(model_id)
                    & network_report["split"].eq(split)
                    & network_report["network"].eq(network)
                ].iloc[0]
                network_delta = max(
                    network_delta,
                    abs(a - expected_network.rmse),
                    abs(b - expected_network.mae),
                    abs(c - expected_network.r2),
                    abs(len(group) - expected_network.n_rows),
                )
            site_delta = 0.0
            for site_id, group in frame.groupby("site_id", sort=False):
                a, b, c = metrics(group["observed_ph"], group["predicted_ph"])
                expected_site = site_report[
                    site_report["model_id"].eq(model_id)
                    & site_report["split"].eq(split)
                    & site_report["site_id"].eq(site_id)
                ].iloc[0]
                site_delta = max(site_delta, abs(a - expected_site.rmse), abs(b - expected_site.mae))
                if not (pd.isna(c) and pd.isna(expected_site.r2)):
                    site_delta = max(site_delta, abs(c - expected_site.r2))
            tail_delta = 0.0
            if split == "test":
                residual = frame["residual_observed_minus_predicted"].to_numpy(dtype=np.float64)
                event = frame["in_ph_event"].astype(bool).to_numpy()
                threshold = float(np.quantile(np.abs(residual), 0.95))
                selected = np.abs(residual) >= threshold
                event_rate = float(event[selected].mean())
                base_rate = float(event.mean())
                expected_tail = tail_report[
                    tail_report["model_id"].eq(model_id)
                    & tail_report["scope_type"].eq("overall")
                    & tail_report["tail_fraction_requested"].eq(0.05)
                    & tail_report["tail_type"].eq("absolute")
                ].iloc[0]
                tail_delta = max(
                    abs(threshold - expected_tail.threshold_native_ph),
                    abs(event_rate / base_rate - expected_tail.event_enrichment),
                )
            if nulls or duplicates or nonfinite or metric_delta > 2e-10 or network_delta > 2e-9 or site_delta > 2e-8 or tail_delta > 2e-8:
                problems.append(f"{key}: content/report audit failed")
            checks.append(
                {
                    "key": key,
                    "rows": len(frame),
                    "sites": int(frame["site_id"].nunique()),
                    "time_min": str(frame["timestamp"].min()),
                    "time_max": str(frame["timestamp"].max()),
                    "nulls": nulls,
                    "duplicates": duplicates,
                    "nonfinite": nonfinite,
                    "metric_max_delta": metric_delta,
                    "network_max_delta": network_delta,
                    "site_max_delta": site_delta,
                    "tail_max_delta": tail_delta,
                }
            )
            del frame

    # Recheck sequence references against exactly the same endpoint keys and reported metrics.
    for model_id in SEQUENCE_MODEL_IDS:
        for split in ("val", "test"):
            frame = pd.read_parquet(sequence / "predictions" / f"prediction_{split}_{model_id}.parquet")
            frame["site_id"] = frame["site_id"].astype(str)
            if not np.array_equal(key_hash(frame), reference_hash[split]):
                problems.append(f"{model_id}/{split}: sequence keys differ from baseline")
            rmse, mae, r2 = metrics(frame["observed_ph"], frame["predicted_ph"])
            expected = report_metrics[
                report_metrics["model_id"].eq(model_id) & report_metrics["split"].eq(split)
            ].iloc[0]
            if max(abs(rmse - expected.rmse), abs(mae - expected.mae), abs(r2 - expected.r2)) > 2e-10:
                problems.append(f"{model_id}/{split}: metric mismatch")
            del frame

    # Persistence is independently tied to the prior hourly target.
    target = np.load(arrays_dir / "target.npy", mmap_mode="r")
    for split in ("val", "test"):
        ends = np.load(arrays_dir / f"{split}_ends.npy", mmap_mode="r")
        frame = pd.read_parquet(output / "predictions" / f"prediction_{split}_B2_persistence_1h.parquet")
        if not np.array_equal(frame["predicted_scaled"].to_numpy(dtype=np.float32), target[np.asarray(ends) - 1]):
            problems.append(f"B2/{split}: prediction is not exact previous-hour pH")

    status = pd.read_csv(reports / "site_status.csv", dtype={"site_id": "string"})
    if len(status) != int(config["expected_full_sites"] if args.mode == "full" else len(status)):
        problems.append("site status row count mismatch")
    if args.mode == "full":
        if int(status["test_windows"].eq(0).sum()) != 6:
            problems.append("expected six sites without test windows")
        if int(status["site_mean_global_fallback"].sum()) != 2:
            problems.append("expected two sites with global site-mean fallback")
    audit = {
        "status": "pass" if not problems else "fail",
        "mode": args.mode,
        "prediction_file_checks": checks,
        "problems": problems,
    }
    audit_path = reports / "independent_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    hash_path = reports / "artifact_hashes.csv"
    hashes = pd.read_csv(hash_path)
    additions = []
    for category, path in (("audit_code", SCRIPT_PATH), ("audit_output", audit_path)):
        resolved = str(path.resolve())
        hashes = hashes[~hashes["path"].eq(resolved)]
        additions.append(
            {"category": category, "path": resolved, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    hashes = pd.concat([hashes, pd.DataFrame(additions)], ignore_index=True)
    hashes.to_csv(hash_path, index=False, encoding="utf-8-sig")
    print(json.dumps({"status": audit["status"], "files_checked": len(checks), "problems": problems}, ensure_ascii=False))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
