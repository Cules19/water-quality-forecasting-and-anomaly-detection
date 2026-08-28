from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parent / "build_upstream13.py"
SPEC = importlib.util.spec_from_file_location("build_upstream13", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load build_upstream13")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def main() -> None:
    snaps = pd.read_csv(M.SNAPS, dtype={"site_id": "string"})
    index = M.ConnectionIndex()
    root_columns = [
        "stream_segment_id",
        "nearest_competing_segment_id",
        "perturb_east_subcatchment_id",
        "perturb_west_subcatchment_id",
        "perturb_north_subcatchment_id",
        "perturb_south_subcatchment_id",
    ]
    roots = np.unique(
        pd.concat([pd.to_numeric(snaps[column], errors="coerce") for column in root_columns])
        .dropna()
        .to_numpy(dtype=np.int64)
    )
    roots = roots[roots > 0]
    graph_path = M.OUT / "upstream_connection_graph.npz"
    if graph_path.exists():
        with np.load(graph_path) as graph:
            node_ids = graph["node_ids"]
            next_ids = graph["next_ids"]
            prev_ids = graph["prev_ids"]
            topo = graph["topological_order"]
        print(f"reused graph cache: {len(node_ids):,} nodes")
    else:
        rows = M.build_upstream_closure(index, roots)
        node_ids, next_ids, prev_ids, topo = M.graph_arrays(rows)
        np.savez_compressed(
            graph_path,
            node_ids=node_ids,
            next_ids=next_ids,
            prev_ids=prev_ids,
            topological_order=topo,
        )
    accumulation, _ = M.scan_variable("accumulation", node_ids, "max")
    snaps["upstream_area_km2"] = M.site_lookup(accumulation, node_ids, snaps.stream_segment_id)
    snaps["alternative_upstream_area_km2"] = M.site_lookup(
        accumulation, node_ids, snaps.nearest_competing_segment_id
    )
    snaps["alternative_area_ratio"] = np.maximum(
        snaps.alternative_upstream_area_km2 / snaps.upstream_area_km2,
        snaps.upstream_area_km2 / snaps.alternative_upstream_area_km2,
    )
    perturb_ratios = []
    same_path_columns = []
    for direction in ["east", "west", "north", "south"]:
        id_column = f"perturb_{direction}_subcatchment_id"
        area_column = f"perturb_{direction}_upstream_area_km2"
        ratio_column = f"perturb_{direction}_area_ratio"
        snaps[area_column] = M.site_lookup(accumulation, node_ids, snaps[id_column])
        snaps[ratio_column] = np.maximum(
            snaps[area_column] / snaps.upstream_area_km2,
            snaps.upstream_area_km2 / snaps[area_column],
        )
        perturb_ratios.append(snaps[ratio_column].to_numpy(dtype=float))
        same_path_columns.append(
            np.asarray(
                [
                    M.same_directed_flow_path(
                        int(canonical), int(alternative), node_ids, next_ids
                    )
                    for canonical, alternative in zip(
                        snaps.stream_segment_id, snaps[id_column]
                    )
                ],
                dtype=bool,
            )
        )
    ratio_matrix = np.vstack(perturb_ratios).T
    same_path_matrix = np.vstack(same_path_columns).T
    material_area_matrix = np.isfinite(ratio_matrix) & (ratio_matrix > 2.0)
    snaps["maximum_one_cell_perturbation_area_ratio"] = np.nanmax(ratio_matrix, axis=1)
    snaps["one_cell_perturbation_area_factor_gt2"] = material_area_matrix.any(axis=1)
    snaps["one_cell_perturbation_material_change"] = (
        material_area_matrix & ~same_path_matrix
    ).any(axis=1)
    snaps["one_cell_material_change_same_directed_path_only"] = (
        material_area_matrix.any(axis=1)
        & ~(material_area_matrix & ~same_path_matrix).any(axis=1)
    )
    snaps["material_alternative_area"] = snaps.alternative_area_ratio > 2.0
    invalid = ~np.isfinite(snaps.upstream_area_km2) | (snaps.upstream_area_km2 <= 0)
    snaps["precheck_match_quality"] = "HIGH_CONFIDENCE_UPSTREAM"
    contextual = (
        (snaps.geodesic_snap_distance_m > 500)
        | ~snaps.used_containing_subcatchment.astype(bool)
        | snaps.one_cell_perturbation_material_change.fillna(False)
    )
    snaps.loc[contextual, "precheck_match_quality"] = "CONTEXTUAL_UPSTREAM"
    unusable = (
        (snaps.geodesic_snap_distance_m > 2000)
        | ~np.isfinite(snaps.geodesic_snap_distance_m)
        | invalid
        | (snaps.stream_segment_id <= 0)
    )
    snaps.loc[unusable, "precheck_match_quality"] = "UPSTREAM_UNUSABLE"
    snaps.to_csv(M.OUT / "network_spatial_precheck.csv", index=False)
    print(snaps.precheck_match_quality.value_counts().to_string())
    print(f"graph nodes: {len(node_ids):,}")


if __name__ == "__main__":
    main()
