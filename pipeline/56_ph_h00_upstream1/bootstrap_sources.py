from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests


PROJECT = Path(__file__).resolve().parents[2]
DEST = PROJECT / "data/external/upstream1_primary"
OUTPUT_ROOT = PROJECT / "analysis_outputs/ph_h00_upstream1_v1"


ENV_BASE = "https://public.igb-berlin.de/index.php/s/zw56kEd25NsQqcQ/download?path=%2F_file_lists/"
FILES = {
    "env90m_soil_paths_file_sizes.txt": ENV_BASE + "env90m_soil_paths_file_sizes.txt",
    "env90m_meritdem_paths_file_sizes.txt": ENV_BASE + "env90m_meritdem_paths_file_sizes.txt",
    "env90m_hydro_paths_file_sizes.txt": ENV_BASE + "env90m_hydro_paths_file_sizes.txt",
    "env90m_observedclimate_paths_file_sizes.txt": ENV_BASE + "env90m_observedclimate_paths_file_sizes.txt",
    "env90m_landcover_paths_file_sizes.txt": ENV_BASE + "env90m_landcover_paths_file_sizes.txt",
    "hydrography90m_paths_file_sizes.txt": "https://drive.usercontent.google.com/download?id=1SEkcgGPutP6ZQPvYtzICh_gcGnVgH_uR&export=download&confirm=t",
    "lookup_tile_regunit.txt": "https://drive.usercontent.google.com/download?id=1deKhOEjGgvUXPwivYyH99hgHlJV7OgUv&export=download&confirm=t",
    "regional_unit_ovr.tif": "https://public.igb-berlin.de/index.php/s/agciopgzXjWswF4/download?path=%2Fglobal&files=regional_unit_ovr.tif",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 10_000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(60, 600)) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if path.stat().st_size < 100:
        raise RuntimeError(f"Downloaded file is unexpectedly small: {path}")


def main() -> None:
    records: list[dict[str, object]] = []
    for name, url in FILES.items():
        path = DEST / name
        download(url, path)
        records.append(
            {
                "name": name,
                "url": url,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        print(name, path.stat().st_size)

    if (DEST / "regional_unit_ovr.tif").stat().st_size < 100_000_000:
        raise RuntimeError("Global regional-unit raster is not the expected data file")

    (OUTPUT_ROOT / "source_bootstrap.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
