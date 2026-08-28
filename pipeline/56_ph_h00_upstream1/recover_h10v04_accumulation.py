from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.windows import Window


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "analysis_outputs/ph_h00_upstream1_v1"
BASE = PROJECT / "data/external/upstream1_primary"
MANIFEST = BASE / "hydrography90m_paths_file_sizes.txt"
DEST = BASE / "downloaded/r.watershed/accumulation_tiles20d/accumulation_h10v04.tif"
SUBC = BASE / "downloaded/r.watershed/sub_catchment_tiles20d/sub_catchment_h10v04.tif"
GRAPH = OUT / "upstream_connection_graph.npz"
RECOVERY = OUT / "h10v04_accumulation_recovered.npz"
HYDRO_SHARE = "https://public.igb-berlin.de/index.php/s/agciopgzXjWswF4/download?path=%2F"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download() -> tuple[str, int]:
    manifest = pd.read_csv(MANIFEST, sep=";")
    relative = "r.watershed/accumulation_tiles20d/accumulation_h10v04.tif"
    row = manifest.loc[manifest.file_path.eq(relative)]
    if len(row) != 1:
        raise RuntimeError(f"expected one source row for {relative}, found {len(row)}")
    expected = int(row.iloc[0].file_size)
    url = HYDRO_SHARE + quote(relative, safe="")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    partial = DEST.with_suffix(DEST.suffix + ".part")
    if not DEST.exists() or DEST.stat().st_size != expected:
        for attempt in range(1, 11):
            existing = partial.stat().st_size if partial.exists() else 0
            if existing > expected:
                partial.unlink()
                existing = 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            mode = "ab" if existing else "wb"
            try:
                with requests.get(url, headers=headers, stream=True, timeout=(60, 1800)) as response:
                    response.raise_for_status()
                    if existing and response.status_code != 206:
                        mode = "wb"
                    with partial.open(mode) as handle:
                        for chunk in response.iter_content(16 * 1024 * 1024):
                            if chunk:
                                handle.write(chunk)
            except requests.RequestException:
                if attempt == 10:
                    raise
                continue
            if partial.stat().st_size == expected:
                break
        if partial.stat().st_size != expected:
            raise RuntimeError(
                f"recovery raster size mismatch: {partial.stat().st_size} != {expected}"
            )
        os.replace(partial, DEST)
    return url, expected


def main() -> None:
    url, expected = download()
    with np.load(GRAPH) as graph:
        node_ids = graph["node_ids"]
    recovered = np.full(len(node_ids), -np.inf, dtype=np.float32)
    matched_pixels = 0
    with rasterio.open(SUBC) as subc, rasterio.open(DEST) as accumulation:
        if subc.shape != accumulation.shape or subc.transform != accumulation.transform:
            raise RuntimeError("h10v04 subcatchment and accumulation rasters are not aligned")
        for row_start in range(0, subc.height, 256):
            height = min(256, subc.height - row_start)
            window = Window(0, row_start, subc.width, height)
            ids = subc.read(1, window=window).reshape(-1).astype(np.int64, copy=False)
            values = accumulation.read(1, window=window).reshape(-1).astype(np.float32, copy=False)
            valid = (ids > 0) & np.isfinite(values) & (values > 0)
            if subc.nodata is not None:
                valid &= ids != int(subc.nodata)
            if accumulation.nodata is not None:
                valid &= values != float(accumulation.nodata)
            ids = ids[valid]
            values = values[valid]
            positions = np.searchsorted(node_ids, ids)
            clipped = np.minimum(positions, len(node_ids) - 1)
            matched = (positions < len(node_ids)) & (node_ids[clipped] == ids)
            if matched.any():
                np.maximum.at(recovered, positions[matched], values[matched])
                matched_pixels += int(matched.sum())
            if row_start % 2048 == 0:
                print(
                    f"accumulation recovery rows {row_start}/{subc.height}; "
                    f"nodes={(recovered > -np.inf).sum():,}",
                    flush=True,
                )
    positions = np.flatnonzero(recovered > -np.inf)
    np.savez_compressed(
        RECOVERY,
        node_ids=node_ids[positions],
        accumulation_max_km2=recovered[positions],
    )
    summary = {
        "status": "PASS",
        "reason": "Environment90m accumulation_h10v04 table is header-only; recovered canonical km2 maxima from the aligned Hydrography90m accumulation raster",
        "source_url": url,
        "source_path": str(DEST),
        "source_expected_bytes": expected,
        "source_actual_bytes": DEST.stat().st_size,
        "source_sha256": sha256(DEST),
        "subcatchment_raster": str(SUBC),
        "graph_nodes": len(node_ids),
        "recovered_nodes": int(len(positions)),
        "matched_raster_pixels": matched_pixels,
        "unit": "km2",
        "outcomes_or_predictions_read": False,
    }
    (OUT / "h10v04_accumulation_recovery_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
