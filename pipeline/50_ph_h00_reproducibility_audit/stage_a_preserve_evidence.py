from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "ph_h00_reproducibility_audit_v1"
EVIDENCE_ROOTS = {
    "official": ROOT / "analysis_outputs" / "ph_common1000_lstm_nohistory_h0_rain_final_v1" / "full",
    "hydrological_memory": ROOT / "analysis_outputs" / "ph_h00_hydrological_memory_validation_v1",
    "site_balanced": ROOT / "analysis_outputs" / "ph_h00_site_balanced_validation_v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(path: Path) -> str:
    name = path.name.lower()
    parts = {p.lower() for p in path.parts}
    if path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        return "checkpoint"
    if "config" in name or "configs" in parts:
        return "configuration"
    if "scaler" in name or "scalers" in parts:
        return "scaler"
    if "prediction" in name or "predictions" in parts or "predictions_by_site" in parts:
        return "prediction"
    if "offset" in name or "calibration" in parts:
        return "calibration"
    if "key" in name or "comparison_keys" in parts:
        return "key_manifest"
    if "history" in name or "training_logs" in parts:
        return "training_history"
    if "metric" in name or any(p.endswith("metrics") for p in parts):
        return "metrics"
    if "manifest" in name:
        return "manifest"
    if "completion" in parts or name in {"_success", "completed.json", "sha256sums.txt"}:
        return "completion"
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        return "figure"
    if path.suffix.lower() in {".md", ".docx"}:
        return "report"
    if path.suffix.lower() in {".npy", ".npz", ".parquet", ".csv"}:
        return "data_artifact"
    return "other"


def method_seed(path: Path) -> tuple[str, str]:
    text = str(path).lower()
    seed_match = re.search(r"seed[_-]?(42|43|44)", text)
    seed = seed_match.group(1) if seed_match else ""
    for method in ("f0", "f1", "s0", "s1"):
        if re.search(rf"(?:^|[\\/_-]){method}(?:[\\/_-]|$)", text):
            return method.upper(), seed
    if "ph_lstm_rain_h0_h00" in text:
        return "OFFICIAL_H00", seed
    return "", seed


def main() -> None:
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    for name in ("configs", "hashes", "predictions", "calibration", "metrics", "training_reproduction", "figures", "reports", "summary", "completion"):
        (OUT / name).mkdir(parents=True, exist_ok=True)

    missing = [str(path) for path in EVIDENCE_ROOTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing evidence roots: {missing}")

    files: list[tuple[str, Path]] = []
    before: dict[str, tuple[int, int]] = {}
    for experiment, base in EVIDENCE_ROOTS.items():
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            key = str(path.resolve())
            stat = path.stat()
            before[key] = (stat.st_size, stat.st_mtime_ns)
            files.append((experiment, path))

    rows = []
    for experiment, path in files:
        absolute = path.resolve()
        method, seed = method_seed(path)
        stat = path.stat()
        digest = sha256(path)
        stat_after = path.stat()
        rows.append(
            {
                "absolute_path": str(absolute),
                "relative_path": str(absolute.relative_to(ROOT)),
                "file_size_bytes": stat.st_size,
                "modified_timestamp": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "modified_timestamp_ns": stat.st_mtime_ns,
                "SHA256": digest,
                "artifact_type": classify(path),
                "experiment_name": experiment,
                "seed": seed,
                "method": method,
                "stable_during_hash": bool(stat.st_size == stat_after.st_size and stat.st_mtime_ns == stat_after.st_mtime_ns),
            }
        )

    mutated = []
    for _, path in files:
        stat = path.stat()
        key = str(path.resolve())
        if before[key] != (stat.st_size, stat.st_mtime_ns):
            mutated.append(key)

    fields = list(rows[0])
    inventory = OUT / "audit" / "evidence_inventory.csv"
    with inventory.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    manifest = OUT / "audit" / "evidence_sha256_manifest.txt"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(f"{row['SHA256']}  {row['relative_path']}\n")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["file_size_bytes"]) for row in rows),
        "evidence_roots": {key: str(value) for key, value in EVIDENCE_ROOTS.items()},
        "mutated_files": mutated,
        "decision": "BLOCKED_ARTIFACT_MUTATION" if mutated else "EVIDENCE_PRESERVED",
    }
    with (OUT / "audit" / "evidence_preservation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    if mutated:
        raise RuntimeError("Evidence changed during hashing; audit blocked.")


if __name__ == "__main__":
    main()
