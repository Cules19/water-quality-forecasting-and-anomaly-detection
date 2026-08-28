from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Common1000 full-metadata GRU without event targets")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


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


def load_sequence_module() -> Any:
    path = ROOT / "pipeline/12_common1000_sequence/run_common1000_sequence_models.py"
    spec = importlib.util.spec_from_file_location("common1000_sequence_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def array_paths(sequence_root: Path) -> dict[str, str]:
    data = sequence_root / "sequence_data" / "arrays"
    names = [
        "features",
        "target",
        "timestamp_ns",
        "site_code",
        "event_direction",
        "event_id",
        "train_ends",
        "val_ends",
        "test_ends",
        "train_sample_ends",
        "val_early_ends",
    ]
    return {name: str(data / f"{name}.npy") for name in names}


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def exact_prediction_keys(a: Path, b: Path) -> bool:
    left = pd.read_parquet(a, columns=["site_id", "timestamp", "observed_ph"])
    right = pd.read_parquet(b, columns=["site_id", "timestamp", "observed_ph"])
    if len(left) != len(right):
        return False
    return bool(
        left["site_id"].astype(str).equals(right["site_id"].astype(str))
        and pd.to_datetime(left["timestamp"]).equals(pd.to_datetime(right["timestamp"]))
        and np.array_equal(left["observed_ph"].to_numpy(dtype=np.float32), right["observed_ph"].to_numpy(dtype=np.float32))
    )


def overall_metrics(frame: pd.DataFrame, model_id: str, split: str) -> dict[str, Any]:
    observed = frame["observed_ph"].to_numpy(dtype=np.float64)
    predicted = frame["predicted_ph"].to_numpy(dtype=np.float64)
    residual = observed - predicted
    denominator = float(np.square(observed - observed.mean()).sum())
    return {
        "model_id": model_id,
        "split": split,
        "n_sites": int(frame["site_id"].nunique()),
        "n_rows": int(len(frame)),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1 - np.square(residual).sum() / denominator) if denominator > 0 else np.nan,
    }


def global_tail_metrics(frame: pd.DataFrame, model_id: str, fractions: list[float]) -> list[dict[str, Any]]:
    absolute = frame["abs_residual"].to_numpy(dtype=np.float64)
    event = frame["in_ph_event"].astype(bool).to_numpy()
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        n_take = max(1, int(math.ceil(len(frame) * fraction)))
        chosen = np.argpartition(absolute, len(absolute) - n_take)[len(absolute) - n_take :]
        selected_rate = float(event[chosen].mean())
        base_rate = float(event.mean())
        rows.append(
            {
                "model_id": model_id,
                "tail_fraction": fraction,
                "n_selected": n_take,
                "threshold_native_ph": float(absolute[chosen].min()),
                "base_event_rate": base_rate,
                "selected_event_rate": selected_rate,
                "event_enrichment": selected_rate / base_rate if base_rate > 0 else np.nan,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    task_config = load_json(config_path)
    output = resolve(task_config["output_root"])
    success = output / "_SUCCESS"
    if args.resume and success.exists():
        print(f"Already complete: {success}")
        return
    if success.exists():
        success.unlink()
    for directory in ("models", "predictions", "reports"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    base = load_sequence_module()
    base_config_path = resolve(task_config["base_sequence_config"])
    config = load_json(base_config_path)
    config["name"] = str(task_config["name"])
    config["random_seed"] = int(task_config["random_seed"])
    sequence_root = resolve(task_config["sequence_input"])
    manifest = load_json(sequence_root / "sequence_data" / "arrays" / "array_manifest.json")
    manifest["paths"] = array_paths(sequence_root)
    arrays = base.load_arrays(manifest)
    metadata_matrix = np.load(sequence_root / "sequence_data" / "metadata_matrix.npy", mmap_mode="r")
    scaler = load_json(sequence_root / "sequence_data" / "dynamic_scaler.json")
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    cohort = pd.read_csv(sequence_root / "reports" / "cohort.csv", dtype={"site_id": "string"})
    spec = {"model_id": str(task_config["model_id"]), "recurrent_type": "gru", "metadata_used": True}
    reference_spec = {"model_id": "GRU_S1", "recurrent_type": "gru", "metadata_used": True}

    base.set_reproducible_seed(int(config["random_seed"]))
    reference_initial = base.create_model(reference_spec, config, len(metadata_matrix), metadata_matrix.shape[1], device)
    reference_initial_hash = state_hash(reference_initial)
    del reference_initial
    base.set_reproducible_seed(int(config["random_seed"]))
    nonevent_initial = base.create_model(spec, config, len(metadata_matrix), metadata_matrix.shape[1], device)
    nonevent_initial_hash = state_hash(nonevent_initial)
    del nonevent_initial
    if reference_initial_hash != nonevent_initial_hash:
        raise RuntimeError("Identical architecture/seed did not produce identical initial state")

    sampled = np.asarray(arrays["train_sample_ends"], dtype=np.int64)
    event_mask = np.asarray(arrays["event_direction"][sampled], dtype=np.int8) != 0
    retained = sampled[~event_mask]
    removed = sampled[event_mask]
    training_arrays = dict(arrays)
    training_arrays["train_sample_ends"] = retained
    site_codes = np.asarray(arrays["site_code"][sampled], dtype=np.int64)
    network_by_code = cohort["network"].astype(str).to_numpy()
    audit = pd.DataFrame({"site_code": site_codes, "is_event_target": event_mask.astype(np.int8)})
    by_site = audit.groupby("site_code", sort=True).agg(original_sampled_train_windows=("is_event_target", "size"), removed_event_target_windows=("is_event_target", "sum")).reset_index()
    by_site["retained_nonevent_train_windows"] = by_site["original_sampled_train_windows"] - by_site["removed_event_target_windows"]
    by_site["site_id"] = by_site["site_code"].map(dict(enumerate(manifest["site_order"])))
    by_site["network"] = by_site["site_code"].map(dict(enumerate(network_by_code)))
    write_csv(output / "reports" / "nonevent_training_by_site.csv", by_site)

    model, history = base.train_model(spec, training_arrays, mean, scale, metadata_matrix, config, output, device, args.resume)
    overall, network, site, tail = base.evaluate_and_save_model(
        model,
        spec,
        arrays,
        mean,
        scale,
        metadata_matrix,
        manifest,
        cohort,
        config,
        output,
        device,
    )
    write_csv(output / "reports" / "nonevent_model_metrics.csv", pd.DataFrame(overall))
    write_csv(output / "reports" / "nonevent_network_metrics.csv", pd.DataFrame(network))
    write_csv(output / "reports" / "nonevent_site_metrics.csv", pd.DataFrame(site))
    write_csv(output / "reports" / "nonevent_residual_tail_metrics.csv", pd.DataFrame(tail))

    comparison_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        baseline_path = resolve(task_config[f"baseline_{split}_prediction"])
        nonevent_path = output / "predictions" / f"prediction_{split}_{spec['model_id']}.parquet"
        if not exact_prediction_keys(baseline_path, nonevent_path):
            raise RuntimeError(f"Baseline/non-event {split} keys or targets differ")
        baseline = pd.read_parquet(baseline_path)
        nonevent = pd.read_parquet(nonevent_path)
        comparison_rows.append(overall_metrics(baseline, "GRU_S1", split))
        comparison_rows.append(overall_metrics(nonevent, spec["model_id"], split))
        if split == "test":
            tail_rows.extend(global_tail_metrics(baseline, "GRU_S1", [0.01, 0.05, 0.10]))
            tail_rows.extend(global_tail_metrics(nonevent, spec["model_id"], [0.01, 0.05, 0.10]))
        del baseline, nonevent
    comparison = pd.DataFrame(comparison_rows)
    tails = pd.DataFrame(tail_rows)
    write_csv(output / "reports" / "paired_model_metrics.csv", comparison)
    write_csv(output / "reports" / "paired_residual_tail_metrics.csv", tails)

    test = comparison[comparison["split"].eq("test")].set_index("model_id")
    base_row = test.loc["GRU_S1"]
    new_row = test.loc[spec["model_id"]]
    top5 = tails[tails["tail_fraction"].eq(0.05)].set_index("model_id")
    report = f"""# Common1000 full-metadata GRU non-event result

## Controlled design

- Sites: {len(cohort):,}.
- Original sampled training windows: {len(sampled):,}.
- Removed event-target windows: {len(removed):,}.
- Retained non-event training windows: {len(retained):,}.
- Validation and test keys are byte-for-byte target-aligned with `GRU_S1`.
- Architecture, 44 safe metadata features, dynamic scaler, seed and initial state are identical.
- Initial state SHA256: `{nonevent_initial_hash}`.

## Test results in native pH units

| Model | RMSE | MAE | R2 | Absolute top-5% event enrichment |
| --- | ---: | ---: | ---: | ---: |
| GRU_S1 all data | {base_row.rmse:.6f} | {base_row.mae:.6f} | {base_row.r2:.6f} | {top5.loc['GRU_S1','event_enrichment']:.3f} |
| GRU_S1 non-event | {new_row.rmse:.6f} | {new_row.mae:.6f} | {new_row.r2:.6f} | {top5.loc[spec['model_id'],'event_enrichment']:.3f} |

RMSE change (non-event minus all-data): {new_row.rmse - base_row.rmse:+.6f}. Event enrichment change: {top5.loc[spec['model_id'],'event_enrichment'] - top5.loc['GRU_S1','event_enrichment']:+.3f}.

Event labels were used only to remove training targets and for post-hoc evaluation; they were never model inputs.
"""
    (output / "reports" / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    problems: list[str] = []
    expected_val = int(manifest["split_window_counts"]["val"])
    expected_test = int(manifest["split_window_counts"]["test"])
    for split, expected in (("val", expected_val), ("test", expected_test)):
        path = output / "predictions" / f"prediction_{split}_{spec['model_id']}.parquet"
        if not path.exists() or pq.ParquetFile(path).metadata.num_rows != expected:
            problems.append(f"{split} prediction row mismatch")
    if len(retained) + len(removed) != len(sampled):
        problems.append("training removal accounting mismatch")
    if (by_site["retained_nonevent_train_windows"] < 0).any():
        problems.append("negative retained site window count")
    if not np.isfinite(comparison[["rmse", "mae", "r2"]].to_numpy(dtype=float)).all():
        problems.append("non-finite paired metrics")
    hashes = [
        {"role": "config", "path": str(config_path), "sha256": sha256_file(config_path)},
        {"role": "script", "path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        {"role": "model", "path": str(output / "models" / f"{spec['model_id']}.pt"), "sha256": sha256_file(output / "models" / f"{spec['model_id']}.pt")},
    ]
    write_csv(output / "reports" / "artifact_hashes.csv", pd.DataFrame(hashes))
    summary = {
        "status": "success" if not problems else "failed_validation",
        "sites": len(cohort),
        "original_sampled_train_windows": len(sampled),
        "removed_event_target_windows": len(removed),
        "retained_nonevent_train_windows": len(retained),
        "validation_windows": expected_val,
        "test_windows": expected_test,
        "initial_state_sha256": nonevent_initial_hash,
        "initial_state_matches_all_data_architecture": reference_initial_hash == nonevent_initial_hash,
        "validation_problems": problems,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "bf16": torch.cuda.is_bf16_supported(),
        "epochs_completed": len(history),
    }
    write_json(output / "reports" / "run_summary.json", summary)
    if problems:
        raise RuntimeError(f"Validation failed: {problems}")
    write_json(success, {"status": "success", "sites": len(cohort), "completed_at_epoch": time.time()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
