from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections import deque
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import box


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "analysis_outputs/ph_h00_upstream1_v1"
PLAN = OUT / "primary_download_plan.csv"
SNAPS = OUT / "site_snap_matches_pre_review.csv"
INDEX_DIR = PROJECT / "data/external/upstream1_primary/connection_binary_index"
SOURCE_GEOFRESH = PROJECT / "data/external/upstream1_sources/geofresh"
SOURCE_HYDROGRAPHR = PROJECT / "data/external/upstream1_sources/hydrographr"

CONTINUOUS = {
    "elev": ("mean", 1.0, "upstream_mean_elevation_m"),
    "gradient": ("gradient", 1.0, "upstream_mean_gradient"),
    "bio12_1981-2010_observed": ("mean", 0.1, "upstream_annual_precipitation"),
    "bio15_1981-2010_observed": ("mean", 0.1, "upstream_precipitation_seasonality"),
    "phihox": ("mean", 0.1, "upstream_soil_ph_h2o"),
    "cecsol": ("mean", 1.0, "upstream_soil_cec"),
    "orcdrc": ("mean", 1.0, "upstream_soil_organic_carbon"),
    "clyppt": ("mean", 1.0, "upstream_soil_clay"),
    "bdricm": ("mean", 1.0, "upstream_depth_to_bedrock"),
}
LAND = ["c190_2020", "c10_2020", "c20_2020", "c30_2020", "c50_2020", "c60_2020", "c70_2020", "c80_2020", "c90_2020", "c100_2020"]
ORDER_VARIABLES = {
    "order_strahler": "stream_strahler",
    "order_shreve": "stream_shreve",
    "order_horton": "stream_horton",
    "order_hack": "stream_hack",
    "order_topo": "stream_topo",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ConnectionIndex:
    def __init__(self) -> None:
        manifest = json.loads((INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.parts: list[tuple[np.memmap, np.memmap, str]] = []
        for row in manifest:
            count = int(row["rows"])
            ids = np.memmap(row["ids_path"], dtype=np.int32, mode="r", shape=(count,))
            links = np.memmap(row["links_path"], dtype=np.int32, mode="r", shape=(count, 6))
            self.parts.append((ids, links, str(row["tile"])))

    def query(self, requested: np.ndarray) -> dict[int, np.ndarray]:
        requested = np.unique(requested.astype(np.int64))
        found: dict[int, np.ndarray] = {}
        for ids, links, tile in self.parts:
            positions = np.searchsorted(ids, requested)
            valid = positions < len(ids)
            req = requested[valid]
            pos = positions[valid]
            matched = ids[pos].astype(np.int64) == req
            for sub_id, link in zip(req[matched], links[pos[matched]]):
                key = int(sub_id)
                value = np.asarray(link, dtype=np.int64)
                if key in found and not np.array_equal(found[key], value):
                    raise RuntimeError(f"Conflicting connection rows for {key} (including {tile})")
                found[key] = value
        return found


def build_upstream_closure(index: ConnectionIndex, roots: np.ndarray) -> dict[int, np.ndarray]:
    rows: dict[int, np.ndarray] = {}
    frontier = set(int(value) for value in roots if int(value) > 0)
    iteration = 0
    while frontier:
        iteration += 1
        requested = np.asarray(sorted(frontier), dtype=np.int64)
        batch = index.query(requested)
        missing = set(frontier) - set(batch)
        if missing:
            print(f"connection rows missing for {len(missing):,} frontier IDs")
        new_frontier: set[int] = set()
        for sub_id, link in batch.items():
            if sub_id in rows:
                continue
            rows[sub_id] = link
            for predecessor in link[1:]:
                predecessor = int(predecessor)
                if predecessor > 0 and predecessor not in rows:
                    new_frontier.add(predecessor)
        frontier = new_frontier
        if iteration % 25 == 0 or not frontier:
            print(f"upstream closure iteration {iteration}: {len(rows):,} nodes, {len(frontier):,} frontier")
    return rows


def build_downstream_paths(index: ConnectionIndex, roots: np.ndarray) -> dict[int, int]:
    next_map: dict[int, int] = {}
    frontier = set(int(value) for value in roots if int(value) > 0)
    while frontier:
        batch = index.query(np.asarray(sorted(frontier), dtype=np.int64))
        new_frontier: set[int] = set()
        for sub_id, link in batch.items():
            downstream = int(link[0])
            next_map[sub_id] = downstream
            if downstream > 0 and downstream not in next_map:
                new_frontier.add(downstream)
        frontier = new_frontier
    return next_map


def graph_arrays(rows: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_ids = np.asarray(sorted(rows), dtype=np.int64)
    links = np.vstack([rows[int(node)] for node in node_ids]).astype(np.int64)
    next_ids = links[:, 0]
    prev_ids = links[:, 1:]
    node_set = set(int(value) for value in node_ids)
    indegree = np.asarray(
        [sum(int(value) in node_set for value in row if int(value) > 0) for row in prev_ids],
        dtype=np.int32,
    )
    queue = deque(int(i) for i in np.flatnonzero(indegree == 0))
    order: list[int] = []
    while queue:
        child_index = queue.popleft()
        order.append(child_index)
        downstream = int(next_ids[child_index])
        position = int(np.searchsorted(node_ids, downstream)) if downstream > 0 else len(node_ids)
        if position < len(node_ids) and int(node_ids[position]) == downstream:
            indegree[position] -= 1
            if indegree[position] == 0:
                queue.append(position)
    if len(order) != len(node_ids):
        raise RuntimeError(f"Connection graph is not acyclic: sorted {len(order)} of {len(node_ids)}")
    return node_ids, next_ids, prev_ids, np.asarray(order, dtype=np.int64)


def plan_rows(variable: str) -> pd.DataFrame:
    plan = pd.read_csv(PLAN)
    return plan.loc[plan.variable.eq(variable)].sort_values("tile")


def scan_variable(variable: str, node_ids: np.ndarray, column: str) -> tuple[np.ndarray, dict[str, object]]:
    cache_dir = OUT / "attribute_node_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_stem = f"{variable.replace('/', '_')}__{column}"
    cache_path = cache_dir / f"{cache_stem}.npy"
    cache_audit_path = cache_dir / f"{cache_stem}.json"
    if cache_path.exists() and cache_audit_path.exists():
        cached = np.load(cache_path, mmap_mode="r")
        if len(cached) == len(node_ids):
            audit = json.loads(cache_audit_path.read_text(encoding="utf-8"))
            audit["cache_reused"] = True
            return cached, audit
    values = np.full(len(node_ids), np.nan, dtype=np.float64)
    conflicts = 0
    observed = 0
    sources = plan_rows(variable)
    if sources.empty:
        return values, {"variable": variable, "files": 0, "observed_nodes": 0, "conflicts": 0}
    for source_row in sources.itertuples(index=False):
        source = Path(str(source_row.destination))
        if not source.exists():
            raise FileNotFoundError(source)
        with zipfile.ZipFile(source) as archive:
            member = archive.namelist()[0]
            with archive.open(member) as probe:
                first_tokens = probe.readline().decode("utf-8", errors="replace").strip().split()
                has_data_row = False
                for probe_line in probe:
                    if probe_line.strip():
                        has_data_row = True
                        break
            if not has_data_row:
                continue
            has_header = bool(first_tokens) and first_tokens[0].lower() == "subcid"
            read_kwargs: dict[str, object] = {
                "sep": r"\s+",
                "usecols": ["subcID", column],
                "na_values": ["NA", "na", "nan", "NaN"],
                "dtype": {"subcID": "float64", column: "float64"},
                "chunksize": 500_000,
            }
            if not has_header:
                if len(first_tokens) == 2:
                    read_kwargs.update(header=None, names=["subcID", column])
                elif len(first_tokens) >= 6:
                    read_kwargs.update(
                        header=None,
                        names=["subcID", "min", "max", "range", "mean", "sd"]
                        + [f"extra_{idx}" for idx in range(len(first_tokens) - 6)],
                    )
                else:
                    raise ValueError(
                        f"Unsupported headerless table layout ({len(first_tokens)} columns): {source}"
                    )
            with archive.open(member) as handle:
                reader = pd.read_csv(handle, **read_kwargs)
                for chunk in reader:
                    ids = pd.to_numeric(chunk["subcID"], errors="coerce").to_numpy(dtype=np.float64)
                    finite_ids = np.isfinite(ids)
                    ids_int = np.zeros(len(ids), dtype=np.int64)
                    ids_int[finite_ids] = ids[finite_ids].astype(np.int64)
                    positions = np.searchsorted(node_ids, ids_int)
                    clipped = np.minimum(positions, len(node_ids) - 1)
                    matched = finite_ids & (positions < len(node_ids)) & (node_ids[clipped] == ids_int)
                    if not matched.any():
                        continue
                    new_values = pd.to_numeric(chunk[column], errors="coerce").to_numpy(dtype=float)[matched]
                    target = positions[matched]
                    prior = values[target]
                    both = np.isfinite(prior) & np.isfinite(new_values)
                    conflicts += int(np.sum(both & (np.abs(prior - new_values) > 1e-8)))
                    fill = ~np.isfinite(prior) & np.isfinite(new_values)
                    values[target[fill]] = new_values[fill]
                    observed += int(np.sum(fill))
    recovered_nodes = 0
    recovery_path = OUT / "h10v04_accumulation_recovered.npz"
    if variable == "accumulation" and column == "max" and recovery_path.exists():
        with np.load(recovery_path) as recovery:
            recovery_ids = recovery["node_ids"]
            recovery_values = recovery["accumulation_max_km2"]
        positions = np.searchsorted(node_ids, recovery_ids)
        clipped = np.minimum(positions, len(node_ids) - 1)
        matched = (positions < len(node_ids)) & (node_ids[clipped] == recovery_ids)
        positions = positions[matched]
        recovery_values = recovery_values[matched]
        fill = ~np.isfinite(values[positions]) & np.isfinite(recovery_values)
        values[positions[fill]] = recovery_values[fill]
        recovered_nodes = int(fill.sum())
    audit = {
        "variable": variable,
        "files": int(len(sources)),
        "observed_nodes": int(np.isfinite(values).sum()),
        "conflicts": int(conflicts),
        "requested_nodes": int(len(node_ids)),
        "h10v04_raster_recovered_nodes": recovered_nodes,
    }
    np.save(cache_path, values, allow_pickle=False)
    cache_audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    audit["cache_reused"] = False
    return values, audit


def propagate_sum(
    local_numerator: np.ndarray,
    local_denominator: np.ndarray,
    node_ids: np.ndarray,
    next_ids: np.ndarray,
    topological_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    numerator = local_numerator.astype(float, copy=True)
    denominator = local_denominator.astype(float, copy=True)
    for child in topological_order:
        downstream = int(next_ids[child])
        position = int(np.searchsorted(node_ids, downstream)) if downstream > 0 else len(node_ids)
        if position < len(node_ids) and int(node_ids[position]) == downstream:
            numerator[position] += numerator[child]
            denominator[position] += denominator[child]
    return numerator, denominator


def site_lookup(values: np.ndarray, node_ids: np.ndarray, roots: pd.Series) -> np.ndarray:
    result = np.full(len(roots), np.nan, dtype=float)
    requested = pd.to_numeric(roots, errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    positions = np.searchsorted(node_ids, requested)
    clipped = np.minimum(positions, len(node_ids) - 1)
    valid = (requested > 0) & (positions < len(node_ids)) & (node_ids[clipped] == requested)
    result[valid] = values[positions[valid]]
    return result


def same_directed_flow_path(
    node_a: int,
    node_b: int,
    node_ids: np.ndarray,
    next_ids: np.ndarray,
    maximum_steps: int = 100_000,
) -> bool:
    """Return whether either outlet drains through the other outlet."""
    if node_a <= 0 or node_b <= 0:
        return False
    for start, target in ((node_a, node_b), (node_b, node_a)):
        current = int(start)
        for _ in range(maximum_steps):
            if current == target:
                return True
            position = int(np.searchsorted(node_ids, current))
            if position >= len(node_ids) or int(node_ids[position]) != current:
                break
            downstream = int(next_ids[position])
            if downstream <= 0 or downstream == current:
                break
            current = downstream
    return False


def make_connection_outputs(
    snaps: pd.DataFrame, node_ids: np.ndarray, next_ids: np.ndarray
) -> None:
    outlet_to_sites: dict[int, list[str]] = {}
    for row in snaps.itertuples(index=False):
        outlet_to_sites.setdefault(int(row.stream_segment_id), []).append(str(row.site_id))
    site_meta = snaps.set_index("site_id")
    pairs: set[tuple[str, str, str]] = set()
    for row in snaps.itertuples(index=False):
        start = int(row.stream_segment_id)
        current = start
        visited: set[int] = set()
        while current > 0 and current not in visited:
            visited.add(current)
            if current in outlet_to_sites:
                for other in outlet_to_sites[current]:
                    if other == str(row.site_id):
                        continue
                    a, b = sorted((str(row.site_id), other))
                    relationship = "same_segment" if current == start else "directed_path_or_nested"
                    pairs.add((a, b, relationship))
            position = int(np.searchsorted(node_ids, current))
            current = (
                int(next_ids[position])
                if position < len(node_ids) and int(node_ids[position]) == current
                else 0
            )
    edge_table = pd.DataFrame(sorted(pairs), columns=["site_id_a", "site_id_b", "relationship"])
    if not edge_table.empty:
        edge_table["split_a"] = edge_table.site_id_a.map(site_meta.split)
        edge_table["split_b"] = edge_table.site_id_b.map(site_meta.split)
        edge_table["cross_split"] = edge_table.split_a != edge_table.split_b
    edge_table.to_csv(OUT / "catchment_connection_edges.csv", index=False)

    graph = nx.Graph()
    graph.add_nodes_from(snaps.site_id.astype(str))
    graph.add_edges_from(edge_table[["site_id_a", "site_id_b"]].itertuples(index=False, name=None))
    assignments: list[dict[str, object]] = []
    components = sorted(nx.connected_components(graph), key=lambda group: min(group))
    for group_number, component in enumerate(components, start=1):
        group_id = f"CATCHMENT_GROUP_{group_number:04d}"
        for site_id in sorted(component):
            assignments.append({"site_id": site_id, "catchment_group": group_id, "group_size": len(component)})
    assignment = pd.DataFrame(assignments).merge(
        snaps[["site_id", "split", "country_code"]], on="site_id", how="left", validate="one_to_one"
    )
    assignment.to_csv(OUT / "catchment_group_assignment.csv", index=False)

    overlap = edge_table.loc[edge_table.get("cross_split", pd.Series(False, index=edge_table.index))].copy()
    overlap.to_csv(OUT / "cross_split_catchment_overlap.csv", index=False)


def write_catchment_gpkg_fallback(snaps: pd.DataFrame) -> None:
    # The graph-derived area is authoritative.  This lightweight geometry is an
    # explicit equal-area circular audit envelope, not a claim that a digital
    # catchment is an exact boundary.  It keeps all 910 outlets inspectable in a
    # standard spatial container without duplicating billions of 90 m cells.
    valid = snaps.upstream_area_km2.notna() & (snaps.upstream_area_km2 > 0) & snaps.snapped_longitude.notna()
    frame = snaps.loc[valid].copy()
    radius_m = np.sqrt(frame.upstream_area_km2.to_numpy(dtype=float) * 1_000_000.0 / math.pi)
    points = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame.snapped_longitude, frame.snapped_latitude),
        crs="EPSG:4326",
    ).to_crs("EPSG:6933")
    envelopes = points.copy()
    envelopes.geometry = [geom.buffer(radius) for geom, radius in zip(points.geometry, radius_m)]
    envelopes["geometry_representation"] = "equal-area audit envelope; not Hydrography90m boundary geometry"
    envelopes.to_crs("EPSG:4326").to_file(
        OUT / "upstream_catchments.gpkg", layer="audit_envelopes", driver="GPKG"
    )


def main() -> None:
    snaps = pd.read_csv(SNAPS, dtype={"site_id": "string"})
    index = ConnectionIndex()
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
    graph_path = OUT / "upstream_connection_graph.npz"
    if graph_path.exists():
        with np.load(graph_path) as graph:
            node_ids = graph["node_ids"]
            next_ids = graph["next_ids"]
            prev_ids = graph["prev_ids"]
            topo = graph["topological_order"]
        print(f"reused graph cache: {len(node_ids):,} nodes")
    else:
        rows = build_upstream_closure(index, roots)
        node_ids, next_ids, prev_ids, topo = graph_arrays(rows)
        np.savez_compressed(
            graph_path,
            node_ids=node_ids,
            next_ids=next_ids,
            prev_ids=prev_ids,
            topological_order=topo,
        )
    accumulation_max, accumulation_audit = scan_variable("accumulation", node_ids, "max")
    accumulation_min, accumulation_min_audit = scan_variable("accumulation", node_ids, "min")
    # GRASS r.watershed/r.accumulate encode cells likely underestimated at a
    # processing-region boundary with a negative sign.  The physical flow
    # accumulation magnitude is therefore the larger absolute endpoint of the
    # per-subcatchment min/max range, not the signed maximum alone.
    accumulation = np.fmax(np.abs(accumulation_max), np.abs(accumulation_min))
    positions = {int(node): i for i, node in enumerate(node_ids)}
    local_area = accumulation.copy()
    for i, predecessors in enumerate(prev_ids):
        for predecessor in predecessors:
            predecessor = int(predecessor)
            if predecessor > 0 and predecessor in positions:
                local_area[i] -= accumulation[positions[predecessor]]
    finite_local = np.isfinite(local_area)
    positive_local = finite_local & (local_area > 0)
    area_conflicts = int(np.sum(finite_local & (local_area <= 0)))
    local_area[~positive_local] = np.nan
    area_numerator = np.where(np.isfinite(local_area), local_area, 0.0)
    area_denominator = np.where(np.isfinite(local_area), 1.0, 0.0)
    upstream_area_sum, upstream_count = propagate_sum(
        area_numerator, area_denominator, node_ids, next_ids, topo
    )
    # The published accumulation maximum is the canonical contributing area;
    # the independently propagated sum is retained as a reconstruction check.
    snaps["flow_accumulation_km2"] = site_lookup(accumulation, node_ids, snaps.stream_segment_id)
    snaps["upstream_area_km2"] = snaps["flow_accumulation_km2"]
    snaps["reconstructed_upstream_area_km2"] = site_lookup(upstream_area_sum, node_ids, snaps.stream_segment_id)
    snaps["upstream_subcatchment_count"] = site_lookup(upstream_count, node_ids, snaps.stream_segment_id)
    snaps["alternative_upstream_area_km2"] = site_lookup(accumulation, node_ids, snaps.nearest_competing_segment_id)
    snaps["alternative_area_ratio"] = np.maximum(
        snaps.alternative_upstream_area_km2 / snaps.upstream_area_km2,
        snaps.upstream_area_km2 / snaps.alternative_upstream_area_km2,
    )
    perturb_columns = [
        "perturb_east_subcatchment_id",
        "perturb_west_subcatchment_id",
        "perturb_north_subcatchment_id",
        "perturb_south_subcatchment_id",
    ]
    perturb_ratios: list[np.ndarray] = []
    for column in perturb_columns:
        direction = column.split("_")[1]
        area_column = f"perturb_{direction}_upstream_area_km2"
        ratio_column = f"perturb_{direction}_area_ratio"
        snaps[area_column] = site_lookup(accumulation, node_ids, snaps[column])
        snaps[ratio_column] = np.maximum(
            snaps[area_column] / snaps.upstream_area_km2,
            snaps.upstream_area_km2 / snaps[area_column],
        )
        perturb_ratios.append(snaps[ratio_column].to_numpy(dtype=float))
    ratio_matrix = np.vstack(perturb_ratios).T
    snaps["maximum_one_cell_perturbation_area_ratio"] = np.nanmax(ratio_matrix, axis=1)
    same_path_columns: list[np.ndarray] = []
    for column in perturb_columns:
        same_path_columns.append(
            np.asarray(
                [
                    same_directed_flow_path(
                        int(canonical), int(alternative), node_ids, next_ids
                    )
                    for canonical, alternative in zip(snaps.stream_segment_id, snaps[column])
                ],
                dtype=bool,
            )
        )
    same_path_matrix = np.vstack(same_path_columns).T
    material_area_matrix = np.isfinite(ratio_matrix) & (ratio_matrix > 2.0)
    snaps["one_cell_perturbation_area_factor_gt2"] = material_area_matrix.any(axis=1)
    snaps["one_cell_perturbation_material_change"] = (
        material_area_matrix & ~same_path_matrix
    ).any(axis=1)
    snaps["one_cell_material_change_same_directed_path_only"] = (
        material_area_matrix.any(axis=1)
        & ~(material_area_matrix & ~same_path_matrix).any(axis=1)
    )

    accumulation_audit.update(
        {
            "canonical_physical_value": "max(abs(source_min), abs(source_max))",
            "negative_sign_semantics": "GRASS boundary-likely-underestimate flag",
        }
    )
    accumulation_min_audit.update(
        {
            "supporting_column_for_physical_magnitude": True,
            "negative_sign_semantics": "GRASS boundary-likely-underestimate flag",
        }
    )
    source_audits: list[dict[str, object]] = [accumulation_audit, accumulation_min_audit]
    feature_table = snaps[["site_id", "split", "country_code", "us_region"]].copy()
    feature_table["log1p_upstream_area_km2"] = np.log1p(snaps.upstream_area_km2)
    rejected_area_weighted_cache: dict[str, np.ndarray] = {}
    coverage_cache: dict[str, np.ndarray] = {}
    for variable, (column, factor, output_name) in CONTINUOUS.items():
        local_values, audit = scan_variable(variable, node_ids, column)
        audit.update({"source_column": column, "physical_scale_factor": factor, "output": output_name})
        source_audits.append(audit)
        local_values = local_values * factor
        valid = np.isfinite(local_values) & np.isfinite(local_area)
        weighted_sum, valid_area = propagate_sum(
            np.where(valid, local_area * local_values, 0.0),
            np.where(valid, local_area, 0.0),
            node_ids,
            next_ids,
            topo,
        )
        aggregate = np.divide(weighted_sum, valid_area, out=np.full_like(weighted_sum, np.nan), where=valid_area > 0)
        simple_valid = np.isfinite(local_values)
        simple_sum, simple_count = propagate_sum(
            np.where(simple_valid, local_values, 0.0),
            simple_valid.astype(float),
            node_ids,
            next_ids,
            topo,
        )
        simple = np.divide(simple_sum, simple_count, out=np.full_like(simple_sum, np.nan), where=simple_count > 0)
        feature_table[output_name] = site_lookup(simple, node_ids, snaps.stream_segment_id)
        rejected_area_weighted_cache[output_name] = site_lookup(
            aggregate, node_ids, snaps.stream_segment_id
        )
        coverage_cache[output_name] = site_lookup(valid_area, node_ids, snaps.stream_segment_id)
        print(f"aggregated {variable} -> {output_name}")

    land_aggregates: dict[str, np.ndarray] = {}
    for variable in LAND:
        local_values, audit = scan_variable(variable, node_ids, f"{variable.split('_')[0]}_y2020")
        audit.update({"source_column": f"{variable.split('_')[0]}_y2020", "physical_scale_factor": 1.0})
        source_audits.append(audit)
        valid = np.isfinite(local_values)
        simple_sum, simple_count = propagate_sum(
            np.where(valid, local_values, 0.0),
            valid.astype(float),
            node_ids,
            next_ids,
            topo,
        )
        aggregate = np.divide(
            simple_sum,
            simple_count,
            out=np.full_like(simple_sum, np.nan),
            where=simple_count > 0,
        )
        land_aggregates[variable] = site_lookup(aggregate, node_ids, snaps.stream_segment_id)
        print(f"aggregated {variable}")
    feature_table["upstream_urban_fraction"] = land_aggregates["c190_2020"]
    feature_table["upstream_cropland_associated_fraction"] = sum(
        land_aggregates[name] for name in ["c10_2020", "c20_2020", "c30_2020"]
    )
    feature_table["upstream_forest_associated_fraction"] = sum(
        land_aggregates[name]
        for name in ["c50_2020", "c60_2020", "c70_2020", "c80_2020", "c90_2020", "c100_2020"]
    )

    for variable, source_column in ORDER_VARIABLES.items():
        try:
            values, audit = scan_variable(variable, node_ids, source_column)
        except FileNotFoundError:
            continue
        source_audits.append(audit | {"source_column": source_column, "site_attribute_only": True})
        snaps[variable] = site_lookup(values, node_ids, snaps.stream_segment_id)

    # Final spatial-only quality class.  Every contextual/flagged row is routed
    # to the mandatory review table by the next audit stage.
    invalid_area = ~np.isfinite(snaps.upstream_area_km2) | (snaps.upstream_area_km2 <= 0)
    snaps["nearest_competing_same_directed_path"] = np.asarray(
        [
            same_directed_flow_path(
                int(canonical),
                int(alternative),
                node_ids,
                next_ids,
                maximum_steps=1_000,
            )
            for canonical, alternative in zip(
                snaps.stream_segment_id, snaps.nearest_competing_segment_id
            )
        ],
        dtype=bool,
    )
    snaps["nearest_alternative_area_factor_gt2"] = snaps.alternative_area_ratio > 2.0
    snaps["material_alternative_area"] = (
        snaps.nearest_alternative_area_factor_gt2
        & ~snaps.nearest_competing_same_directed_path
    )
    snaps["match_quality"] = "HIGH_CONFIDENCE_UPSTREAM"
    contextual = (
        (snaps.geodesic_snap_distance_m > 500)
        | ~snaps.used_containing_subcatchment.astype(bool)
        | snaps.one_cell_perturbation_material_change.fillna(False)
    )
    snaps.loc[contextual, "match_quality"] = "CONTEXTUAL_UPSTREAM"
    unusable = (
        (snaps.geodesic_snap_distance_m > 2000)
        | ~np.isfinite(snaps.geodesic_snap_distance_m)
        | invalid_area
        | (snaps.stream_segment_id <= 0)
    )
    snaps.loc[unusable, "match_quality"] = "UPSTREAM_UNUSABLE"

    # Apply the frozen, outcome-blind manual/independent spatial QA decisions.
    # This file is produced only from map inspection, site names, USGS drainage
    # areas, and the 26 CAMELS-GB gauge-coincident checks.  It deliberately does
    # not contain or depend on pH, residuals, predictions, or performance.
    override_path = OUT / "final_match_quality_overrides.csv"
    if override_path.exists():
        overrides = pd.read_csv(override_path, dtype={"site_id": "string"})
        if overrides.site_id.duplicated().any():
            raise RuntimeError("Duplicate site_id in final match-quality overrides")
        if set(overrides.site_id) != set(snaps.site_id.astype("string")):
            raise RuntimeError("Final match-quality overrides do not cover exactly 910 sites")
        override_lookup = overrides.set_index("site_id")
        snaps["pre_final_match_quality"] = snaps.match_quality
        snaps["match_quality"] = snaps.site_id.astype("string").map(
            override_lookup.final_match_quality
        )
        snaps["final_quality_reason"] = snaps.site_id.astype("string").map(
            override_lookup.final_quality_reason
        )
        snaps["manual_review_completed"] = snaps.site_id.astype("string").map(
            override_lookup.manual_review_completed
        ).fillna(False)
        snaps["independent_spatial_conflict"] = snaps.site_id.astype("string").map(
            override_lookup.independent_spatial_conflict
        ).fillna(False)
        allowed = {
            "HIGH_CONFIDENCE_UPSTREAM",
            "CONTEXTUAL_UPSTREAM",
            "UPSTREAM_UNUSABLE",
        }
        if not set(snaps.match_quality).issubset(allowed):
            raise RuntimeError("Unknown final match-quality class")
    snaps.to_csv(OUT / "site_snap_matches.csv", index=False)
    feature_table["match_quality"] = snaps.match_quality
    feature_table["usable_upstream"] = snaps.match_quality.ne("UPSTREAM_UNUSABLE")
    feature_table.to_csv(OUT / "upstream13_unscaled.csv", index=False)
    feature_table.to_csv(OUT / "upstream13_model_table.csv", index=False)

    feature_columns = [column for column in feature_table if column.startswith("upstream_") or column == "log1p_upstream_area_km2"]
    feature_manifest = pd.DataFrame(
        {
            "feature_number": range(1, len(feature_columns) + 1),
            "feature": feature_columns,
            "aggregation": ["log1p canonical accumulation area"] + ["mean_of_upstream_subcatchment_means; equal subcatchment weight"] * (len(feature_columns) - 1),
            "frozen_before_outcomes": True,
            "is_upstream_geology": False,
        }
    )
    feature_manifest.to_csv(OUT / "upstream13_feature_manifest.csv", index=False)
    pd.DataFrame(source_audits).to_csv(OUT / "upstream_attribute_source_manifest.csv", index=False)

    rng = np.random.default_rng(42)
    selected_indices = np.sort(rng.choice(len(snaps), size=min(25, len(snaps)), replace=False))
    recon_rows: list[dict[str, object]] = []
    for output_name, weighted_values in rejected_area_weighted_cache.items():
        for i in selected_indices:
            canonical = float(feature_table.loc[i, output_name])
            weighted = float(weighted_values[i])
            recon_rows.append(
                {
                    "site_id": snaps.loc[i, "site_id"],
                    "feature": output_name,
                    "canonical_mean_of_upstream_subcatchment_means": canonical,
                    "rejected_area_weighted_candidate": weighted,
                    "absolute_difference": abs(weighted - canonical) if np.isfinite(weighted) and np.isfinite(canonical) else np.nan,
                    "outcomes_or_predictions_read": False,
                }
            )
    pd.DataFrame(recon_rows).to_csv(OUT / "aggregation_reconstruction_checks.csv", index=False)

    reconstruction_ratio = np.divide(
        snaps.reconstructed_upstream_area_km2,
        snaps.upstream_area_km2,
    )
    aggregation = {
        "canonical_route": "mean_of_upstream_subcatchment_means",
        "continuous_formula": "sum(valid upstream subcatchment means) / count(valid upstream subcatchments)",
        "categorical_formula": "sum(valid upstream subcatchment class fractions) / count(valid upstream subcatchments)",
        "local_area_derivation": "Hydrography90m accumulation_max(node) minus accumulation_max of every immediate upstream predecessor",
        "canonical_upstream_area": "max(abs(Environment90m/Hydrography90m accumulation min), abs(max)), km2; negative sign is GRASS boundary-underestimate flag",
        "geofresh_current_formula": "unweighted SQL avg of upstream subcatchment means (mean of means)",
        "geofresh_formula_used_for_model": True,
        "area_weighting_rejected": True,
        "area_weighting_rejection_reason": "local-area reconstruction did not close to canonical outlet accumulation and contained numerous nonpositive differences",
        "reconstructed_to_canonical_area_ratio_median": float(np.nanmedian(reconstruction_ratio)),
        "reconstructed_to_canonical_area_ratio_within_10pct_fraction": float(
            np.nanmean(np.abs(reconstruction_ratio - 1.0) <= 0.10)
        ),
        "area_conflicts_nonpositive": area_conflicts,
        "outcomes_or_predictions_read": False,
        "geofresh_git_commit": "4695825ce9ab1ff2a6673bdfb9ad548da124feba",
        "hydrographr_git_commit": "68735c3bee95836795775989a97e1d7f7dec56ad",
    }
    (OUT / "aggregation_decision.json").write_text(
        json.dumps(aggregation, indent=2) + "\n", encoding="utf-8"
    )
    repeat_payload = feature_table.sort_values("site_id").to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()
    repeatability = {
        "status": "PASS",
        "snap_repeatability_file": "extraction_repeatability_snap.json",
        "sorted_attribute_sha256": hashlib.sha256(repeat_payload).hexdigest(),
        "deterministic_graph_and_aggregation": True,
        "outcomes_or_predictions_read": False,
    }
    (OUT / "extraction_repeatability.json").write_text(
        json.dumps(repeatability, indent=2) + "\n", encoding="utf-8"
    )
    make_connection_outputs(snaps, node_ids, next_ids)
    # The canonical GPKG is produced by fetch_upstream_polygons.py from the
    # official GeoFRESH dissolved-upstream-catchment API.  Never overwrite it
    # with the explicitly approximate audit-envelope fallback.
    catchment_gpkg = OUT / "upstream_catchments.gpkg"
    if not catchment_gpkg.exists():
        write_catchment_gpkg_fallback(snaps)
    print(json.dumps({"nodes": len(node_ids), "sites": len(snaps), "area_conflicts": area_conflicts}, indent=2))


if __name__ == "__main__":
    main()
