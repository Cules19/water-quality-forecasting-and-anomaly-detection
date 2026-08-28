"""Build test-site rainfall/flow sidecars for error miss attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def select_flow_file(root: Path, site_id: str) -> Path | None:
    candidates: list[Path] = []
    for path in root.glob(f"{site_id}_*.parquet"):
        try:
            columns = set(pq.read_schema(path).names)
        except Exception:
            continue
        if {"timestamp", "00060"}.issubset(columns):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


def hourly_flow(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["timestamp", "00060"])
    if "timestamp" not in frame.columns and frame.index.name == "timestamp":
        frame = frame.reset_index()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    # The current processed-layer 00060 is already a continuous standardized
    # flow signal (negative values are legitimate below-mean anomalies). Keep
    # that signal intact; the attribution stage robustly re-standardizes it
    # within each site.
    flow = pd.to_numeric(frame["00060"], errors="coerce")
    frame = pd.DataFrame(
        {"timestamp": frame["timestamp"].dt.floor("h"), "flow_native": flow}
    ).dropna(subset=["timestamp"])
    hourly = (
        frame.groupby("timestamp", as_index=False, sort=True)["flow_native"].mean()
    )
    hourly["x_00060"] = hourly["flow_native"].astype("float32")
    return hourly[["timestamp", "x_00060"]]


def rainfall_axis(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["timestamp", "00045"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    rain = pd.to_numeric(frame["00045"], errors="coerce").mask(lambda x: x < 0)
    frame["x_00045"] = np.log1p(rain).astype("float32")
    return (
        frame[["timestamp", "x_00045"]]
        .dropna(subset=["timestamp"])
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_json(config_path)
    prediction_root = resolve(config["prediction_root"])
    merged_root = resolve(config["merged_input_root"])
    processed_root = resolve(config["processed_flow_root"])
    output = resolve(config["precipitation_driver_root"])
    output.mkdir(parents=True, exist_ok=True)

    first_run = str(config["models"][0]["run_id"])
    split = str(config["evaluation_split"])
    test_sites = sorted(
        path.stem
        for path in (prediction_root / first_run / split).glob("*.parquet")
    )
    rows: list[dict[str, Any]] = []
    for site_id in test_sites:
        merged_path = merged_root / f"{site_id}.parquet"
        flow_path = select_flow_file(processed_root, site_id)
        if not merged_path.exists():
            rows.append(
                {
                    "site_id": site_id,
                    "status": "excluded",
                    "reason": "missing_merged_rainfall_file",
                }
            )
            continue
        if flow_path is None:
            rows.append(
                {
                    "site_id": site_id,
                    "status": "excluded",
                    "reason": "no_native_00060_flow_file",
                }
            )
            continue
        try:
            rain = rainfall_axis(merged_path)
            flow = hourly_flow(flow_path)
            driver = rain.merge(flow, on="timestamp", how="left", validate="one_to_one")
            driver["x_00045_missing"] = driver["x_00045"].isna().astype("int8")
            driver["x_00060_missing"] = driver["x_00060"].isna().astype("int8")
            destination = output / f"{site_id}.parquet"
            write_parquet(destination, driver)
            rows.append(
                {
                    "site_id": site_id,
                    "status": "built",
                    "reason": "",
                    "source_flow_file": str(flow_path),
                    "model_axis_hours": int(len(driver)),
                    "rain_nonmissing_hours": int(driver["x_00045"].notna().sum()),
                    "flow_nonmissing_hours": int(driver["x_00060"].notna().sum()),
                    "joint_nonmissing_hours": int(
                        driver[["x_00045", "x_00060"]].notna().all(axis=1).sum()
                    ),
                    "output_file": str(destination),
                    "output_sha256": sha256(destination),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "site_id": site_id,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "source_flow_file": str(flow_path),
                }
            )

    status = pd.DataFrame(rows)
    write_csv(output / "driver_site_status.csv", status)
    joint_nonmissing = (
        pd.to_numeric(status["joint_nonmissing_hours"], errors="coerce")
        if "joint_nonmissing_hours" in status
        else pd.Series(dtype="float64")
    )
    summary = {
        "status": "pass"
        if not status["status"].eq("failed").any()
        else "fail",
        "evaluation_sites": len(test_sites),
        "built_driver_sites": int(status["status"].eq("built").sum()),
        "excluded_sites": int(status["status"].eq("excluded").sum()),
        "failed_sites": int(status["status"].eq("failed").sum()),
        "joint_nonmissing_hours": int(joint_nonmissing.fillna(0).sum()),
        "rainfall_source": str(merged_root),
        "flow_source": str(processed_root),
        "flow_parameter": "USGS 00060",
        "flow_transform": "processed standardized 00060 aggregated to hourly mean",
        "config_sha256": sha256(config_path),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
