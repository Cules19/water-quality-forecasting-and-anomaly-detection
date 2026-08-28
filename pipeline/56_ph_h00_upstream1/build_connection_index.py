from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "analysis_outputs/ph_h00_upstream1_v1"
INDEX_DIR = PROJECT / "data/external/upstream1_primary/connection_binary_index"
PLAN = OUT / "primary_download_plan.csv"
COLUMNS = [
    "subcID",
    "next_stream",
    "prev_str01",
    "prev_str02",
    "prev_str03",
    "prev_str04",
    "prev_str05",
]


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv(PLAN)
    selected = plan.loc[plan.variable.eq("connections")].sort_values("tile")
    manifest: list[dict[str, object]] = []
    for index, row in enumerate(selected.itertuples(index=False), start=1):
        source = Path(str(row.destination))
        if not source.exists():
            raise FileNotFoundError(source)
        ids_path = INDEX_DIR / f"connections_{row.tile}_ids.i32"
        rows_path = INDEX_DIR / f"connections_{row.tile}_links.i32"
        ids_path.unlink(missing_ok=True)
        rows_path.unlink(missing_ok=True)
        with zipfile.ZipFile(source) as archive:
            member = archive.namelist()[0]
            with archive.open(member) as probe:
                first_line = probe.readline().split()
                first_token = first_line[0]
            has_header = first_token == b"subcID"
            present_columns = (
                [token.decode("ascii") for token in first_line]
                if has_header
                else COLUMNS[: len(first_line)]
            )
            with archive.open(member) as handle:
                table = pd.read_csv(
                    handle,
                    sep=r"\s+",
                    dtype={column: "int32" for column in present_columns},
                    header=0 if has_header else None,
                    names=None if has_header else present_columns,
                )
        for missing_column in COLUMNS:
            if missing_column not in table:
                table[missing_column] = np.int32(0)
        values = table[COLUMNS].to_numpy(dtype=np.int32, copy=False)
        del table
        # Environment90m connection text is not guaranteed to be ordered.  Sort
        # each bounded 20-degree tile once, then use deterministic binary search.
        order = np.argsort(values[:, 0], kind="mergesort")
        values = values[order]
        duplicate = values[1:, 0] == values[:-1, 0]
        if duplicate.any():
            duplicate_positions = np.flatnonzero(duplicate) + 1
            if any(
                not np.array_equal(values[position], values[position - 1])
                for position in duplicate_positions
            ):
                raise RuntimeError(f"Conflicting duplicate IDs in {source}")
            values = values[np.r_[True, ~duplicate]]
        ids = values[:, 0]
        count = len(values)
        first_id = int(ids[0])
        last_id = int(ids[-1])
        with ids_path.open("wb") as ids_handle:
            ids.tofile(ids_handle)
        with rows_path.open("wb") as rows_handle:
            values[:, 1:].tofile(rows_handle)
        monotonic = bool(np.all(ids[1:] >= ids[:-1]))
        if ids_path.stat().st_size != count * 4:
            raise RuntimeError(f"ID index size mismatch for {row.tile}")
        if rows_path.stat().st_size != count * 6 * 4:
            raise RuntimeError(f"Link index size mismatch for {row.tile}")
        manifest.append(
            {
                "tile": row.tile,
                "source": str(source),
                "rows": count,
                "first_subcatchment_id": first_id,
                "last_subcatchment_id": last_id,
                "ids_path": str(ids_path),
                "links_path": str(rows_path),
                "sorted_ascending": monotonic,
            }
        )
        print(f"indexed {index}/{len(selected)} {row.tile}: {count:,} rows")
    (INDEX_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"indexed {sum(int(row['rows']) for row in manifest):,} rows")


if __name__ == "__main__":
    main()
