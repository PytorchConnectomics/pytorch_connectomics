#!/usr/bin/env python3
"""Reconnect arm0_96 chunk skeletons and derive GT-free morphology features.

Stage ``aggregate`` concatenates chunk graphs and adds only same-final-ID edges at
continuation ports extracted from the segmentation. Stage ``classify`` prunes short
skeletonization twigs, extracts a geodesic backbone, labels terminal twigs versus backbone
ends, and computes continuous bushiness/glia descriptors. No evaluation skeleton is read.

Outputs under ``decoder_gtfree`` are permitted decoder inputs. Files below sibling
``oracle_gt`` are deliberately never referenced here.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import h5py
import numpy as np

from .artifacts import reject_evaluation_path
from .sizes import load_size_inventory

REPO = Path(__file__).resolve().parents[3]
DEV = REPO / "dev" / "zebrafinch"
ROOT = DEV / "arm096_error_correction" / "decoder_gtfree"
DEFAULT_CHUNKS = ROOT / "chunks"
DEFAULT_GRAPH = ROOT / "segment_skeleton_graph.h5"
DEFAULT_STITCH = ROOT / "stitch_edges.npz"
DEFAULT_FEATURES = ROOT / "segment_morphology.npz"
DEFAULT_ENDPOINTS = ROOT / "segment_endpoints.npz"
DEFAULT_INTERIORS = ROOT / "segment_interiors.npz"
DEFAULT_NUCLEUS_TARGETS = DEV / "nucsplit_arm096" / "targets.json"
RUN = DEV / "wholevol_arm096_fullmask" / "seg_arm096_fullmask"
DEFAULT_SIZES = RUN / "scratch" / "seg_arm096_fullmask" / "agg" / "info" / "seg_size_5_0_0_0.data"
KEY_PATTERN = re.compile(r"^z(?P<z>\d+)_y(?P<y>\d+)_x(?P<x>\d+)\.npz$")
VOXEL_VOLUME_NM3 = 20.0 * 9.0 * 9.0


def log(*values: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *values, flush=True)


def key_tuple(path: Path) -> tuple[int, int, int]:
    match = KEY_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid chunk filename: {path}")
    return tuple(int(match.group(axis)) for axis in ("z", "y", "x"))


def chunk_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("z*_y*_x*.npz"), key=key_tuple)


def load_chunk(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        result = {name: np.asarray(data[name]) for name in data.files}
    required = {
        "vertices_nm",
        "radii_nm",
        "vertex_label",
        "edges",
        "metadata",
        "port_label",
        "port_position_nm",
        "port_neighbor",
        "port_source",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    metadata = json.loads(str(result["metadata"].item()))
    if metadata.get("gt_free") is not True:
        raise ValueError(f"{path}: chunk is not marked GT-free")
    if metadata.get("key") != path.stem:
        raise ValueError(f"{path}: metadata key mismatch")
    vertices = result["vertices_nm"]
    edges = result["edges"]
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{path}: invalid vertices")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"{path}: invalid edges")
    if len(edges) and (edges.min() < 0 or edges.max() >= len(vertices)):
        raise ValueError(f"{path}: edge outside local vertex array")
    return result


def aggregate(
    chunk_dir: Path,
    graph_path: Path,
    stitch_path: Path,
    expected_chunks: int,
    allow_partial: bool,
    max_port_distance_nm: float,
) -> None:
    paths = chunk_paths(chunk_dir)
    if not paths:
        raise FileNotFoundError(f"no chunk skeletons in {chunk_dir}")
    if not allow_partial and len(paths) != expected_chunks:
        raise RuntimeError(f"found {len(paths)}/{expected_chunks} chunk files")
    counts = []
    for number, path in enumerate(paths, 1):
        data = load_chunk(path)
        counts.append((len(data["vertices_nm"]), len(data["edges"]), len(data["port_label"])))
        if number % 500 == 0:
            log(f"inventory {number}/{len(paths)}")
    counts_array = np.asarray(counts, dtype=np.int64)
    vertex_starts = np.r_[0, np.cumsum(counts_array[:, 0])]
    edge_starts = np.r_[0, np.cumsum(counts_array[:, 1])]
    port_starts = np.r_[0, np.cumsum(counts_array[:, 2])]
    total_vertices, total_edges, total_ports = (
        int(vertex_starts[-1]),
        int(edge_starts[-1]),
        int(port_starts[-1]),
    )
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(graph_path, "w") as handle:
        handle.attrs["schema"] = 1
        handle.attrs["gt_free"] = True
        handle.attrs["complete"] = not allow_partial and len(paths) == expected_chunks
        handle.create_dataset(
            "chunk_key", data=np.asarray([path.stem for path in paths], dtype="S24")
        )
        handle.create_dataset("vertex_start", data=vertex_starts[:-1])
        handle.create_dataset("vertex_count", data=counts_array[:, 0])
        vertices_ds = handle.create_dataset(
            "vertices_nm",
            (total_vertices, 3),
            dtype="f4",
            chunks=(min(1_000_000, max(total_vertices, 1)), 3),
        )
        radii_ds = handle.create_dataset(
            "radii_nm",
            (total_vertices,),
            dtype="f4",
            chunks=(min(1_000_000, max(total_vertices, 1)),),
        )
        labels_ds = handle.create_dataset(
            "vertex_label",
            (total_vertices,),
            dtype="u8",
            chunks=(min(1_000_000, max(total_vertices, 1)),),
        )
        edges_ds = handle.create_dataset(
            "edges", (total_edges, 2), dtype="i8", chunks=(min(1_000_000, max(total_edges, 1)), 2)
        )
        port_label_ds = handle.create_dataset("port_label", (total_ports,), dtype="u8")
        port_position_ds = handle.create_dataset("port_position_nm", (total_ports, 3), dtype="f4")
        port_source_ds = handle.create_dataset("port_source", (total_ports,), dtype="S24")
        port_neighbor_ds = handle.create_dataset("port_neighbor", (total_ports,), dtype="S24")
        for number, path in enumerate(paths):
            data = load_chunk(path)
            vertex_slice = slice(int(vertex_starts[number]), int(vertex_starts[number + 1]))
            edge_slice = slice(int(edge_starts[number]), int(edge_starts[number + 1]))
            port_slice = slice(int(port_starts[number]), int(port_starts[number + 1]))
            vertices_ds[vertex_slice] = data["vertices_nm"]
            radii_ds[vertex_slice] = data["radii_nm"]
            labels_ds[vertex_slice] = data["vertex_label"]
            edges_ds[edge_slice] = data["edges"] + int(vertex_starts[number])
            port_label_ds[port_slice] = data["port_label"]
            port_position_ds[port_slice] = data["port_position_nm"]
            port_source_ds[port_slice] = data["port_source"].astype("S24")
            port_neighbor_ds[port_slice] = data["port_neighbor"].astype("S24")
            if (number + 1) % 500 == 0:
                log(f"aggregate {number + 1}/{len(paths)}")

    key_to_index = {path.stem: index for index, path in enumerate(paths)}
    cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]] = OrderedDict()

    def chunk_vertices(key: str):
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached
        index = key_to_index.get(key)
        if index is None:
            return None
        data = load_chunk(paths[index])
        labels = np.asarray(data["vertex_label"], dtype=np.uint64)
        groups: dict[int, np.ndarray] = {}
        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        # An EMPTY chunk (no skeleton vertices at all -- pure background, or everything
        # dust-filtered) is normal: 22 of 400 sampled chunks, ~5.5% of the 10,493. Without this
        # guard `np.r_[0, ...]` still yields [0], the loop body runs once, and
        # `sorted_labels[0]` raises IndexError on a size-0 array -- which killed the whole
        # stitch 65 s in (job 2853900).
        if sorted_labels.size:
            starts = np.r_[0, np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1]
            for start, stop in zip(starts, np.r_[starts[1:], len(order)]):
                groups[int(sorted_labels[start])] = order[start:stop]
        value = (np.asarray(data["vertices_nm"]), labels, groups)
        cache[key] = value
        if len(cache) > 24:
            cache.popitem(last=False)
        return value

    bridge_distance: dict[tuple[int, int], float] = {}
    skipped_missing_chunk = skipped_missing_label = skipped_far = 0
    with h5py.File(graph_path, "r") as handle:
        port_labels = np.asarray(handle["port_label"])
        port_positions = np.asarray(handle["port_position_nm"])
        port_sources = np.asarray(handle["port_source"]).astype("U24")
        port_neighbors = np.asarray(handle["port_neighbor"]).astype("U24")
    for number, (label, point, source, neighbor) in enumerate(
        zip(port_labels, port_positions, port_sources, port_neighbors), 1
    ):
        source_data, neighbor_data = chunk_vertices(str(source)), chunk_vertices(str(neighbor))
        if source_data is None or neighbor_data is None:
            skipped_missing_chunk += 1
            continue
        source_indices = source_data[2].get(int(label))
        neighbor_indices = neighbor_data[2].get(int(label))
        if source_indices is None or neighbor_indices is None:
            skipped_missing_label += 1
            continue
        source_distance = np.linalg.norm(source_data[0][source_indices] - point, axis=1)
        neighbor_distance = np.linalg.norm(neighbor_data[0][neighbor_indices] - point, axis=1)
        source_local = int(source_indices[int(np.argmin(source_distance))])
        neighbor_local = int(neighbor_indices[int(np.argmin(neighbor_distance))])
        distance = float(source_distance.min() + neighbor_distance.min())
        if distance > max_port_distance_nm:
            skipped_far += 1
            continue
        source_global = int(vertex_starts[key_to_index[str(source)]]) + source_local
        neighbor_global = int(vertex_starts[key_to_index[str(neighbor)]]) + neighbor_local
        bridge = tuple(sorted((source_global, neighbor_global)))
        bridge_distance[bridge] = min(distance, bridge_distance.get(bridge, float("inf")))
        if number % 100_000 == 0:
            log(f"ports {number}/{total_ports} bridges={len(bridge_distance):,}")
    ordered_bridges = sorted(bridge_distance)
    bridge_array = np.asarray(ordered_bridges, dtype=np.int64).reshape(-1, 2)
    np.savez_compressed(
        stitch_path,
        edges=bridge_array,
        distance_nm=np.asarray(
            [bridge_distance[edge] for edge in ordered_bridges], dtype=np.float32
        ),
        max_port_distance_nm=np.asarray(max_port_distance_nm),
        skipped_missing_chunk=np.asarray(skipped_missing_chunk),
        skipped_missing_label=np.asarray(skipped_missing_label),
        skipped_far=np.asarray(skipped_far),
        gt_free=np.asarray(True),
    )
    log(
        f"aggregate complete chunks={len(paths):,} vertices={total_vertices:,} "
        f"edges={total_edges:,}",
        f"ports={total_ports:,} stitch={len(bridge_array):,}",
        f"skip chunk/label/far={skipped_missing_chunk}/{skipped_missing_label}/{skipped_far}",
    )


def prune_short_twigs(
    vertices: np.ndarray, edges: np.ndarray, prune_nm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return retained-vertex mask and edges after repeated short-leaf pruning."""
    active = np.ones(len(vertices), dtype=bool)
    edge_list = [tuple(map(int, edge)) for edge in edges.tolist()]
    while edge_list:
        adjacency: list[list[int]] = [[] for _ in range(len(vertices))]
        for left, right in edge_list:
            adjacency[left].append(right)
            adjacency[right].append(left)
        drop: set[int] = set()
        for leaf, neighbors in enumerate(adjacency):
            if not active[leaf] or len(neighbors) != 1:
                continue
            path = [leaf]
            previous, current = -1, leaf
            length = 0.0
            while True:
                candidates = [node for node in adjacency[current] if node != previous]
                if not candidates:
                    break
                following = candidates[0]
                length += float(np.linalg.norm(vertices[following] - vertices[current]))
                previous, current = current, following
                if len(adjacency[current]) != 2:
                    break
                path.append(current)
            if length < prune_nm and len(path) < np.count_nonzero(active):
                drop.update(path)
        if not drop:
            break
        active[list(drop)] = False
        edge_list = [(left, right) for left, right in edge_list if active[left] and active[right]]
    return active, np.asarray(edge_list, dtype=np.int64).reshape(-1, 2)


def weighted_graph(vertices: np.ndarray, edges: np.ndarray):
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(vertices))]
    for left, right in edges.tolist():
        length = float(np.linalg.norm(vertices[left] - vertices[right]))
        adjacency[left].append((right, length))
        adjacency[right].append((left, length))
    return adjacency


def endpoint_branch_profile(
    adjacency: list[list[tuple[int, float]]],
    degree: np.ndarray,
    endpoint: int,
    radius_nm: float,
) -> dict[str, float | int]:
    """Measure branch structure only inside a bounded geodesic endpoint neighborhood."""
    distance = {endpoint: 0.0}
    queue = [(0.0, endpoint)]
    while queue:
        current_distance, current = heapq.heappop(queue)
        if current_distance != distance[current]:
            continue
        for neighbor, edge_length in adjacency[current]:
            candidate = current_distance + edge_length
            if candidate > radius_nm or candidate >= distance.get(neighbor, float("inf")):
                continue
            distance[neighbor] = candidate
            heapq.heappush(queue, (candidate, neighbor))
    nodes = np.fromiter(distance, dtype=np.int64)
    branchpoints = int(np.count_nonzero(degree[nodes] >= 3))
    excess_degree = int(np.maximum(degree[nodes] - 2, 0).sum())
    span_nm = max(distance.values(), default=0.0)
    sampled_um = max(min(radius_nm, max(span_nm, 1_000.0)) / 1_000.0, 1.0)
    return {
        "profile_branchpoints": branchpoints,
        "profile_excess_degree": excess_degree,
        "profile_max_degree": int(degree[nodes].max(initial=0)),
        "profile_branch_density_per_100um": branchpoints * 100.0 / sampled_um,
        "profile_geodesic_span_nm": span_nm,
    }


def trace_from_junction(
    vertices: np.ndarray,
    radii: np.ndarray,
    adjacency: list[list[tuple[int, float]]],
    junction: int,
    first: int,
    max_length_nm: float,
    fallback_radius_nm: float,
) -> tuple[float, np.ndarray, float]:
    """Trace one branch away from a junction without crossing another junction."""
    previous, current = junction, first
    edge_length = next(length for neighbor, length in adjacency[junction] if neighbor == first)
    length = float(edge_length)
    samples = [float(radii[first]) if radii[first] > 0 else fallback_radius_nm]
    target = first
    while length < max_length_nm and len(adjacency[current]) == 2:
        candidates = [item for item in adjacency[current] if item[0] != previous]
        if not candidates:
            break
        following, following_length = candidates[0]
        previous, current = current, following
        length += following_length
        target = current
        # The junction voxel belongs to the next attachment, not this branch's caliber.
        if len(adjacency[current]) < 3:
            samples.append(float(radii[current]) if radii[current] > 0 else fallback_radius_nm)
    direction = vertices[target] - vertices[junction]
    norm = float(np.linalg.norm(direction))
    if norm > 0:
        direction = direction / norm
    else:
        direction = np.zeros(3, dtype=np.float64)
    return float(np.median(samples)), direction, min(length, max_length_nm)


def spine_attachment_profile(
    vertices: np.ndarray,
    radii: np.ndarray,
    adjacency: list[list[tuple[int, float]]],
    junction: int,
    terminal_neighbor: int,
    terminal_radius_nm: float,
    terminal_length_nm: float,
    trace_length_nm: float,
    fallback_radius_nm: float,
    max_spine_length_nm: float,
    max_radius_ratio: float,
    min_parent_collinearity: float,
    min_parent_radius_balance: float,
    min_perpendicularity: float,
) -> dict[str, float | int | bool]:
    """Decide whether a terminal is a thin lateral protrusion from a parent shaft.

    A short branch alone is not a spine. The two other attachment branches must form a
    balanced, nearly collinear parent shaft, and the terminal must be substantially thinner
    than that shaft. This leaves equal-caliber axon branches and dendritic bifurcations alone.
    """
    parent_neighbors = [
        neighbor for neighbor, _ in adjacency[junction] if neighbor != terminal_neighbor
    ]
    empty = {
        "attachment_parent_branches": len(parent_neighbors),
        "attachment_parent_radius_nm": 0.0,
        "attachment_radius_ratio": 1.0,
        "attachment_parent_collinearity": -1.0,
        "attachment_parent_radius_balance": 0.0,
        "attachment_perpendicularity": 0.0,
        "attachment_is_spine": False,
    }
    if len(parent_neighbors) < 2:
        return empty
    branches = [
        trace_from_junction(
            vertices,
            radii,
            adjacency,
            junction,
            neighbor,
            trace_length_nm,
            fallback_radius_nm,
        )
        for neighbor in parent_neighbors
    ]
    _, terminal_direction, _ = trace_from_junction(
        vertices,
        radii,
        adjacency,
        junction,
        terminal_neighbor,
        trace_length_nm,
        fallback_radius_nm,
    )
    best: tuple[float, float, float, float, int, int] | None = None
    for left in range(len(branches)):
        for right in range(left + 1, len(branches)):
            left_radius, left_direction, _ = branches[left]
            right_radius, right_direction, _ = branches[right]
            collinearity = float(-np.dot(left_direction, right_direction))
            balance = min(left_radius, right_radius) / max(left_radius, right_radius, 1.0)
            parent_radius = min(left_radius, right_radius)
            score = max(collinearity, 0.0) * balance
            candidate = (score, collinearity, balance, parent_radius, left, right)
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    _, collinearity, balance, parent_radius, parent_left, parent_right = best
    parent_axis = branches[parent_left][1] - branches[parent_right][1]
    parent_axis /= max(float(np.linalg.norm(parent_axis)), 1e-12)
    perpendicularity = 1.0 - abs(float(np.dot(terminal_direction, parent_axis)))
    radius_ratio = terminal_radius_nm / max(parent_radius, 1.0)
    is_spine = (
        terminal_length_nm <= max_spine_length_nm
        and radius_ratio <= max_radius_ratio
        and collinearity >= min_parent_collinearity
        and balance >= min_parent_radius_balance
        and perpendicularity >= min_perpendicularity
    )
    return {
        "attachment_parent_branches": len(parent_neighbors),
        "attachment_parent_radius_nm": parent_radius,
        "attachment_radius_ratio": radius_ratio,
        "attachment_parent_collinearity": collinearity,
        "attachment_parent_radius_balance": balance,
        "attachment_perpendicularity": perpendicularity,
        "attachment_is_spine": is_spine,
    }


def dijkstra_farthest(adjacency, source: int):
    distance = {source: 0.0}
    parent = {source: -1}
    heap = [(0.0, source)]
    while heap:
        value, node = heapq.heappop(heap)
        if value != distance[node]:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = value + weight
            if candidate < distance.get(neighbor, float("inf")):
                distance[neighbor] = candidate
                parent[neighbor] = node
                heapq.heappush(heap, (candidate, neighbor))
    farthest = max(distance, key=distance.get)
    return farthest, distance[farthest], distance, parent


def components(adjacency) -> list[list[int]]:
    seen: set[int] = set()
    result = []
    for seed, neighbors in enumerate(adjacency):
        if seed in seen or not neighbors:
            continue
        stack, component = [seed], []
        seen.add(seed)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor, _ in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        result.append(component)
    return result


def segment_metrics(
    label: int,
    vertices: np.ndarray,
    radii: np.ndarray,
    edges: np.ndarray,
    voxels: int,
    prune_nm: float,
    tangent_nm: float,
    twig_nm: float,
    branch_profile_nm: float,
    spine_radius_ratio: float,
    spine_parent_collinearity: float,
    spine_parent_radius_balance: float,
    spine_perpendicularity: float,
):
    active, pruned_edges = prune_short_twigs(vertices, edges, prune_nm)
    if not len(pruned_edges):
        return None, [], []
    adjacency = weighted_graph(vertices, pruned_edges)
    graph_components = components(adjacency)
    if not graph_components:
        return None, [], []
    edge_lengths = np.linalg.norm(
        vertices[pruned_edges[:, 0]] - vertices[pruned_edges[:, 1]], axis=1
    )
    total_length = float(edge_lengths.sum())
    component_index = np.full(len(vertices), -1, dtype=np.int32)
    for index, component in enumerate(graph_components):
        component_index[component] = index
    component_lengths = np.bincount(
        component_index[pruned_edges[:, 0]],
        weights=edge_lengths,
        minlength=len(graph_components),
    )
    largest_index = int(np.argmax(component_lengths))
    largest = graph_components[largest_index]
    largest_mask = component_index == largest_index
    largest_length = float(component_lengths[largest_index])
    seed = largest[0]
    first, _, _, _ = dijkstra_farthest(adjacency, seed)
    second, diameter, _, parent = dijkstra_farthest(adjacency, first)
    backbone = {second}
    node = second
    while parent[node] >= 0:
        node = parent[node]
        backbone.add(node)
    degree = np.asarray([len(values) for values in adjacency], dtype=np.int32)
    branchpoints = int(np.count_nonzero(degree >= 3))
    leaves = np.flatnonzero(degree == 1)
    largest_branchpoints = int(np.count_nonzero((degree >= 3) & largest_mask))
    largest_leaves = np.flatnonzero((degree == 1) & largest_mask)
    branch_spacing_um = total_length / 1000.0 / branchpoints if branchpoints else float("inf")
    largest_branch_spacing_um = (
        largest_length / 1000.0 / largest_branchpoints if largest_branchpoints else float("inf")
    )
    centered = vertices[active] - vertices[active].mean(axis=0)
    if len(centered) >= 3:
        eigenvalues = np.maximum(np.linalg.eigvalsh(np.cov(centered.T))[::-1], 1e-12)
        sheetness = float(eigenvalues[1] / eigenvalues[0])
        thickness_ratio = float(eigenvalues[2] / eigenvalues[1])
    else:
        sheetness = thickness_ratio = 0.0
    largest_centered = vertices[largest] - vertices[largest].mean(axis=0)
    if len(largest_centered) >= 3:
        largest_eigenvalues = np.maximum(
            np.linalg.eigvalsh(np.cov(largest_centered.T))[::-1], 1e-12
        )
        largest_sheetness = float(largest_eigenvalues[1] / largest_eigenvalues[0])
        largest_thickness_ratio = float(largest_eigenvalues[2] / largest_eigenvalues[1])
    else:
        largest_sheetness = largest_thickness_ratio = 0.0
    equivalent_radius = math.sqrt(
        max(voxels, 0) * VOXEL_VOLUME_NM3 / max(math.pi * total_length, 1e-9)
    )
    valid_radii = radii[(radii > 0) & active]
    median_radius = float(np.median(valid_radii)) if len(valid_radii) else equivalent_radius
    feature = {
        "label": label,
        "voxels": voxels,
        "vertices": int(np.count_nonzero(active)),
        "edges": len(pruned_edges),
        "components": len(graph_components),
        "length_nm": total_length,
        "diameter_nm": diameter,
        "tubeness": diameter / max(total_length, 1e-9),
        "off_backbone_fraction": 1.0 - diameter / max(total_length, 1e-9),
        "branchpoints": branchpoints,
        "leaves": len(leaves),
        "branch_spacing_um": branch_spacing_um,
        "branch_density_per_100um": branchpoints / max(total_length / 100_000.0, 1e-9),
        "terminal_density_per_100um": len(leaves) / max(total_length / 100_000.0, 1e-9),
        "sheetness": sheetness,
        "thickness_ratio": thickness_ratio,
        "largest_component_length_nm": largest_length,
        "largest_component_fraction": largest_length / max(total_length, 1e-9),
        "largest_tubeness": diameter / max(largest_length, 1e-9),
        "largest_off_backbone_fraction": 1.0 - diameter / max(largest_length, 1e-9),
        "largest_branchpoints": largest_branchpoints,
        "largest_leaves": len(largest_leaves),
        "largest_branch_spacing_um": largest_branch_spacing_um,
        "largest_branch_density_per_100um": largest_branchpoints
        / max(largest_length / 100_000.0, 1e-9),
        "largest_terminal_density_per_100um": len(largest_leaves)
        / max(largest_length / 100_000.0, 1e-9),
        "largest_sheetness": largest_sheetness,
        "largest_thickness_ratio": largest_thickness_ratio,
        "equivalent_radius_nm": equivalent_radius,
        "median_skeleton_radius_nm": median_radius,
    }
    endpoint_rows = []
    for endpoint in leaves.tolist():
        previous, current = -1, endpoint
        path_length = 0.0
        tangent_inner = endpoint
        hit_junction = False
        visited = {endpoint}
        profile_radii = [float(radii[endpoint]) if radii[endpoint] > 0 else equivalent_radius]
        while True:
            candidates = [item for item in adjacency[current] if item[0] != previous]
            if not candidates:
                break
            following, edge_length = candidates[0]
            path_length += edge_length
            previous, current = current, following
            if tangent_inner == endpoint or path_length <= tangent_nm:
                tangent_inner = current
                if degree[current] < 3:
                    profile_radii.append(
                        float(radii[current]) if radii[current] > 0 else equivalent_radius
                    )
            if degree[current] != 2:
                hit_junction = degree[current] >= 3
                break
            if current in visited:
                break
            visited.add(current)
        inward = vertices[tangent_inner] - vertices[endpoint]
        norm = float(np.linalg.norm(inward))
        if norm <= 0:
            continue
        inward /= norm
        endpoint_radius = float(radii[endpoint]) if radii[endpoint] > 0 else equivalent_radius
        profile = np.asarray(profile_radii, dtype=np.float64)
        profile_median = float(np.median(profile))
        profile_mad = float(np.median(np.abs(profile - profile_median)))
        differences = np.diff(profile)
        significant = differences[np.abs(differences) >= max(5.0, 0.10 * profile_median)]
        alternation = (
            float(np.mean(np.sign(significant[1:]) != np.sign(significant[:-1])))
            if len(significant) >= 2
            else 0.0
        )
        branch_profile = endpoint_branch_profile(adjacency, degree, endpoint, branch_profile_nm)
        if hit_junction:
            attachment_profile = spine_attachment_profile(
                vertices,
                radii,
                adjacency,
                current,
                previous,
                profile_median,
                path_length,
                tangent_nm,
                equivalent_radius,
                twig_nm,
                spine_radius_ratio,
                spine_parent_collinearity,
                spine_parent_radius_balance,
                spine_perpendicularity,
            )
        else:
            attachment_profile = {
                "attachment_parent_branches": 0,
                "attachment_parent_radius_nm": 0.0,
                "attachment_radius_ratio": 1.0,
                "attachment_parent_collinearity": -1.0,
                "attachment_parent_radius_balance": 0.0,
                "attachment_perpendicularity": 0.0,
                "attachment_is_spine": False,
            }
        endpoint_rows.append(
            {
                "label": label,
                "position_nm": vertices[endpoint].astype(np.float32),
                "outward_tangent": (-inward).astype(np.float32),
                "radius_nm": endpoint_radius,
                "profile_radius_median_nm": profile_median,
                "profile_radius_mad_nm": profile_mad,
                "profile_radius_cv": profile_mad / max(profile_median, 1.0),
                "profile_radius_alternation_fraction": alternation,
                "terminal_branch_nm": path_length,
                "role": 1 if attachment_profile["attachment_is_spine"] else 0,
                "on_backbone": endpoint in backbone,
                **branch_profile,
                **attachment_profile,
            }
        )
    interior_rows = []
    interior_nodes = np.flatnonzero((degree >= 2) & active)
    if len(interior_nodes):
        cells = np.floor(vertices[interior_nodes] / 1_000.0).astype(np.int64)
        _, first = np.unique(cells, axis=0, return_index=True)
        for index in interior_nodes[np.sort(first)].tolist():
            interior_rows.append(
                {
                    "label": label,
                    "position_nm": vertices[index].astype(np.float32),
                    "radius_nm": float(radii[index]) if radii[index] > 0 else equivalent_radius,
                    "degree": int(degree[index]),
                    "on_backbone": index in backbone,
                }
            )
    return feature, endpoint_rows, interior_rows


def percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(values), dtype=np.float64)
    rank[order] = (np.arange(len(values), dtype=np.float64) + 0.5) / max(len(values), 1)
    return rank


def reference_percentile_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rank every value against a GT-free reference population."""
    finite_reference = np.sort(reference[np.isfinite(reference)])
    if not len(finite_reference):
        raise ValueError("empty morphology reference population")
    left = np.searchsorted(finite_reference, values, side="left")
    right = np.searchsorted(finite_reference, values, side="right")
    return (left + right) / (2.0 * len(finite_reference))


def load_nucleus_firewall(path: Path) -> dict[int, int]:
    """Return final-segment -> external nuclei count; never supplies merge identities."""
    reject_evaluation_path(path)
    payload = json.loads(path.read_text())
    histograms = payload.get("hist")
    if isinstance(histograms, dict):
        segment_nuclei: dict[int, int] = defaultdict(int)
        for histogram in histograms.values():
            candidates = [
                (int(count), int(segment))
                for segment, count in histogram.items()
                if int(segment) != 0 and int(count) > 0
            ]
            if candidates:
                _, dominant_segment = max(candidates)
                segment_nuclei[dominant_segment] += 1
        return dict(segment_nuclei)

    # Native nucleus-competition runs already record the identity-preserving output label for
    # every qualified raw segment. Count distinct external nucleus owners per FINAL label. The
    # owner identity is used only to detect zero/one/multiple nuclei; it is never a merge target.
    qualified = payload.get("qualified_segment_labels")
    qualified_owners = payload.get("qualified_segment_owners")
    if isinstance(qualified, dict) and isinstance(qualified_owners, dict):
        nuclei_by_segment: dict[int, set[int]] = defaultdict(set)
        for raw_segment, owners in qualified_owners.items():
            territories = qualified.get(raw_segment)
            if isinstance(territories, dict):
                # A repaired/match-guarded raw segment may be split into one output identity per
                # nucleus. Do not retain the retired parent as an anchor.
                for nucleus in owners:
                    segment = int(territories.get(str(nucleus), 0))
                    if segment != 0:
                        nuclei_by_segment[segment].add(int(nucleus))
            else:
                # Unrepaired qualified segments keep their own final label.
                segment = int(raw_segment)
                if segment != 0:
                    nuclei_by_segment[segment].update(int(nucleus) for nucleus in owners)
        return {segment: len(nuclei) for segment, nuclei in nuclei_by_segment.items()}
    raise ValueError(f"{path}: missing nucleus histograms or qualified segment labels")


def classify(
    graph_path: Path,
    stitch_path: Path,
    sizes_path: Path,
    features_path: Path,
    endpoints_path: Path,
    interiors_path: Path,
    nucleus_targets_path: Path,
    prune_nm: float,
    tangent_nm: float,
    twig_nm: float,
    branch_profile_nm: float,
    spine_radius_ratio: float,
    spine_parent_collinearity: float,
    spine_parent_radius_balance: float,
    spine_perpendicularity: float,
) -> None:
    sizes = load_size_inventory(sizes_path)
    size_order = np.argsort(sizes["label"])
    size_labels = np.asarray(sizes["label"][size_order])
    size_values = np.asarray(sizes["size"][size_order])
    with h5py.File(graph_path, "r") as handle:
        if not bool(handle.attrs.get("gt_free", False)):
            raise ValueError("aggregate graph is not marked GT-free")
        vertices = np.asarray(handle["vertices_nm"])
        radii = np.asarray(handle["radii_nm"])
        labels = np.asarray(handle["vertex_label"])
        edges = np.asarray(handle["edges"])
    with np.load(stitch_path, allow_pickle=False) as data:
        if not bool(data["gt_free"].item()):
            raise ValueError("stitch edges are not marked GT-free")
        stitch_edges = np.asarray(data["edges"], dtype=np.int64)
    if len(stitch_edges):
        edges = np.concatenate([edges, stitch_edges], axis=0)
    if len(edges):
        if not np.all(labels[edges[:, 0]] == labels[edges[:, 1]]):
            raise AssertionError("skeleton edge joins different final segment IDs")
        keep = edges[:, 0] != edges[:, 1]
        edges = np.unique(np.sort(edges[keep], axis=1), axis=0)

    vertex_order = np.argsort(labels, kind="stable")
    sorted_labels = labels[vertex_order]
    vertex_starts = np.r_[0, np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1]
    unique_labels = sorted_labels[vertex_starts]
    if len(edges):
        edge_labels = labels[edges[:, 0]]
        edge_order = np.argsort(edge_labels, kind="stable")
        sorted_edge_labels = edge_labels[edge_order]
        edge_starts = np.r_[
            0, np.flatnonzero(sorted_edge_labels[1:] != sorted_edge_labels[:-1]) + 1
        ]
        edge_lookup = {
            int(sorted_edge_labels[start]): edge_order[start:stop]
            for start, stop in zip(edge_starts, np.r_[edge_starts[1:], len(edge_order)])
        }
    else:
        edge_lookup = {}

    features = []
    endpoints = []
    interiors = []
    for number, (start, stop) in enumerate(
        zip(vertex_starts, np.r_[vertex_starts[1:], len(vertex_order)]), 1
    ):
        label = int(unique_labels[number - 1])
        global_vertices = np.sort(vertex_order[start:stop])
        edge_indices = edge_lookup.get(label)
        if edge_indices is None:
            continue
        local_edges = np.searchsorted(global_vertices, edges[edge_indices])
        size_index = int(np.searchsorted(size_labels, np.uint64(label)))
        voxels = (
            int(size_values[size_index])
            if size_index < len(size_labels) and int(size_labels[size_index]) == label
            else 0
        )
        feature, endpoint_rows, interior_rows = segment_metrics(
            label,
            vertices[global_vertices],
            radii[global_vertices],
            local_edges,
            voxels,
            prune_nm,
            tangent_nm,
            twig_nm,
            branch_profile_nm,
            spine_radius_ratio,
            spine_parent_collinearity,
            spine_parent_radius_balance,
            spine_perpendicularity,
        )
        if feature is not None:
            features.append(feature)
            endpoints.extend(endpoint_rows)
            interiors.extend(interior_rows)
        if number % 10_000 == 0:
            log(f"classify {number}/{len(unique_labels)} features={len(features):,}")
    if not features:
        raise RuntimeError("no segment morphology features were produced")

    largest_length = np.asarray([row["largest_component_length_nm"] for row in features])
    off_backbone = np.asarray([row["largest_off_backbone_fraction"] for row in features])
    branch_density = np.asarray([row["largest_branch_density_per_100um"] for row in features])
    terminal_density = np.asarray([row["largest_terminal_density_per_100um"] for row in features])
    sheetness = np.asarray([row["largest_sheetness"] for row in features])
    thickness = np.asarray([row["largest_thickness_ratio"] for row in features])
    reference = largest_length >= 50_000.0
    ranks = np.stack(
        [
            reference_percentile_rank(values, values[reference])
            for values in (off_backbone, branch_density, terminal_density, sheetness, thickness)
        ]
    )
    bushiness = np.mean(ranks, axis=0)
    bushiness_votes = np.count_nonzero(ranks >= 0.80, axis=0).astype(np.uint8)
    process_bushiness_votes = np.count_nonzero(ranks[:3] >= 0.80, axis=0).astype(np.uint8)
    spatial_fill_votes = np.count_nonzero(ranks[3:] >= 0.80, axis=0).astype(np.uint8)
    physical_core = np.asarray(
        [
            row["largest_component_length_nm"] >= 50_000
            and row["largest_component_fraction"] >= 0.25
            and row["largest_tubeness"] <= 0.55
            and row["largest_branch_spacing_um"] <= 12.0
            and row["largest_branchpoints"] >= 8
            and row["largest_sheetness"] >= 0.08
            for row in features
        ],
        dtype=bool,
    )
    tree_bushiness_support = np.asarray(
        [row["largest_leaves"] >= 8 for row in features], dtype=bool
    )
    extreme_bushiness_fallback = bushiness_votes == ranks.shape[0]
    physical_bushy = physical_core & (tree_bushiness_support | extreme_bushiness_fallback)
    nucleus_counts_by_segment = load_nucleus_firewall(nucleus_targets_path)
    nucleus_count = np.asarray(
        [nucleus_counts_by_segment.get(int(row["label"]), 0) for row in features],
        dtype=np.uint16,
    )
    nucleus_anchor = nucleus_count >= 1
    nucleus_conflict = nucleus_count >= 2
    glia_high_confidence = (
        physical_bushy
        & (process_bushiness_votes >= 2)
        & (spatial_fill_votes >= 1)
        & ~nucleus_anchor
    )
    glia_quarantine = glia_high_confidence | nucleus_conflict

    feature_names = [
        "label",
        "voxels",
        "vertices",
        "edges",
        "components",
        "length_nm",
        "diameter_nm",
        "tubeness",
        "off_backbone_fraction",
        "branchpoints",
        "leaves",
        "branch_spacing_um",
        "branch_density_per_100um",
        "terminal_density_per_100um",
        "sheetness",
        "thickness_ratio",
        "largest_component_length_nm",
        "largest_component_fraction",
        "largest_tubeness",
        "largest_off_backbone_fraction",
        "largest_branchpoints",
        "largest_leaves",
        "largest_branch_spacing_um",
        "largest_branch_density_per_100um",
        "largest_terminal_density_per_100um",
        "largest_sheetness",
        "largest_thickness_ratio",
        "equivalent_radius_nm",
        "median_skeleton_radius_nm",
    ]
    payload = {name: np.asarray([row[name] for row in features]) for name in feature_names}
    payload.update(
        bushiness_score=bushiness.astype(np.float32),
        bushiness_votes=bushiness_votes,
        process_bushiness_votes=process_bushiness_votes,
        spatial_fill_votes=spatial_fill_votes,
        physical_bushy_core=physical_core,
        tree_bushiness_support=tree_bushiness_support,
        extreme_bushiness_fallback=extreme_bushiness_fallback,
        physical_bushy=physical_bushy,
        glia_high_confidence=glia_high_confidence,
        nucleus_anchor=nucleus_anchor,
        nucleus_count=nucleus_count,
        nucleus_conflict=nucleus_conflict,
        glia_quarantine=glia_quarantine,
        morphology_reference=reference,
        external_nucleus_firewall=np.asarray(True),
        nucleus_targets_path=np.asarray(str(nucleus_targets_path.resolve())),
        gt_free=np.asarray(True),
        prune_nm=np.asarray(prune_nm),
        twig_nm=np.asarray(twig_nm),
        spine_radius_ratio=np.asarray(spine_radius_ratio),
        spine_parent_collinearity=np.asarray(spine_parent_collinearity),
        spine_parent_radius_balance=np.asarray(spine_parent_radius_balance),
        spine_perpendicularity=np.asarray(spine_perpendicularity),
    )
    for path in (features_path, endpoints_path, interiors_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(features_path, **payload)
    np.savez_compressed(
        endpoints_path,
        label=np.asarray([row["label"] for row in endpoints], dtype=np.uint64),
        position_nm=np.asarray([row["position_nm"] for row in endpoints], dtype=np.float32).reshape(
            -1, 3
        ),
        outward_tangent=np.asarray(
            [row["outward_tangent"] for row in endpoints], dtype=np.float32
        ).reshape(-1, 3),
        radius_nm=np.asarray([row["radius_nm"] for row in endpoints], dtype=np.float32),
        profile_radius_median_nm=np.asarray(
            [row["profile_radius_median_nm"] for row in endpoints], dtype=np.float32
        ),
        profile_radius_mad_nm=np.asarray(
            [row["profile_radius_mad_nm"] for row in endpoints], dtype=np.float32
        ),
        profile_radius_cv=np.asarray(
            [row["profile_radius_cv"] for row in endpoints], dtype=np.float32
        ),
        profile_radius_alternation_fraction=np.asarray(
            [row["profile_radius_alternation_fraction"] for row in endpoints], dtype=np.float32
        ),
        profile_branchpoints=np.asarray(
            [row["profile_branchpoints"] for row in endpoints], dtype=np.uint16
        ),
        profile_excess_degree=np.asarray(
            [row["profile_excess_degree"] for row in endpoints], dtype=np.uint16
        ),
        profile_max_degree=np.asarray(
            [row["profile_max_degree"] for row in endpoints], dtype=np.uint16
        ),
        profile_branch_density_per_100um=np.asarray(
            [row["profile_branch_density_per_100um"] for row in endpoints], dtype=np.float32
        ),
        profile_geodesic_span_nm=np.asarray(
            [row["profile_geodesic_span_nm"] for row in endpoints], dtype=np.float32
        ),
        terminal_branch_nm=np.asarray(
            [row["terminal_branch_nm"] for row in endpoints], dtype=np.float32
        ),
        attachment_parent_branches=np.asarray(
            [row["attachment_parent_branches"] for row in endpoints], dtype=np.uint8
        ),
        attachment_parent_radius_nm=np.asarray(
            [row["attachment_parent_radius_nm"] for row in endpoints], dtype=np.float32
        ),
        attachment_radius_ratio=np.asarray(
            [row["attachment_radius_ratio"] for row in endpoints], dtype=np.float32
        ),
        attachment_parent_collinearity=np.asarray(
            [row["attachment_parent_collinearity"] for row in endpoints], dtype=np.float32
        ),
        attachment_parent_radius_balance=np.asarray(
            [row["attachment_parent_radius_balance"] for row in endpoints], dtype=np.float32
        ),
        attachment_perpendicularity=np.asarray(
            [row["attachment_perpendicularity"] for row in endpoints], dtype=np.float32
        ),
        role=np.asarray([row["role"] for row in endpoints], dtype=np.uint8),
        on_backbone=np.asarray([row["on_backbone"] for row in endpoints], dtype=bool),
        gt_free=np.asarray(True),
        radius_profile_nm=np.asarray(tangent_nm),
        branch_profile_nm=np.asarray(branch_profile_nm),
        spine_max_length_nm=np.asarray(twig_nm),
        spine_max_radius_ratio=np.asarray(spine_radius_ratio),
        spine_min_parent_collinearity=np.asarray(spine_parent_collinearity),
        spine_min_parent_radius_balance=np.asarray(spine_parent_radius_balance),
        spine_min_perpendicularity=np.asarray(spine_perpendicularity),
    )
    np.savez_compressed(
        interiors_path,
        label=np.asarray([row["label"] for row in interiors], dtype=np.uint64),
        position_nm=np.asarray([row["position_nm"] for row in interiors], dtype=np.float32).reshape(
            -1, 3
        ),
        radius_nm=np.asarray([row["radius_nm"] for row in interiors], dtype=np.float32),
        degree=np.asarray([row["degree"] for row in interiors], dtype=np.uint8),
        on_backbone=np.asarray([row["on_backbone"] for row in interiors], dtype=bool),
        sampling_nm=np.asarray(1_000.0),
        gt_free=np.asarray(True),
    )
    log(
        f"classification complete segments={len(features):,} endpoints={len(endpoints):,}",
        f"interiors={len(interiors):,}",
        f"physical_bushy={int(physical_bushy.sum()):,}",
        f"glia={int(glia_high_confidence.sum()):,}",
        f"nucleus_anchor/conflict={int(nucleus_anchor.sum()):,}/{int(nucleus_conflict.sum()):,}",
        f"quarantine={int(glia_quarantine.sum()):,}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("aggregate", "classify", "all"), default="all")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--stitch", type=Path, default=DEFAULT_STITCH)
    parser.add_argument("--sizes", type=Path, default=DEFAULT_SIZES)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--endpoints", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--interiors", type=Path, default=DEFAULT_INTERIORS)
    parser.add_argument("--nucleus-targets", type=Path, default=DEFAULT_NUCLEUS_TARGETS)
    parser.add_argument("--expected-chunks", type=int, default=10_626)
    parser.add_argument("--allow-partial", action="store_true")
    # The port is an exact same-label continuation in the native segmentation. Thick
    # somata/glia can put the medial skeleton several microns from the boundary, so this
    # gate only rejects pathological matches; it is not an identity decision.
    parser.add_argument("--max-port-distance-nm", type=float, default=10_000.0)
    parser.add_argument("--prune-nm", type=float, default=2_000.0)
    parser.add_argument("--tangent-nm", type=float, default=1_500.0)
    parser.add_argument("--twig-nm", type=float, default=3_000.0)
    parser.add_argument("--branch-profile-nm", type=float, default=10_000.0)
    parser.add_argument("--spine-radius-ratio", type=float, default=0.5)
    parser.add_argument("--spine-parent-collinearity", type=float, default=0.5)
    parser.add_argument("--spine-parent-radius-balance", type=float, default=0.5)
    parser.add_argument("--spine-perpendicularity", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in (
        "spine_radius_ratio",
        "spine_parent_collinearity",
        "spine_parent_radius_balance",
        "spine_perpendicularity",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    for path in (
        args.chunks,
        args.graph,
        args.stitch,
        args.sizes,
        args.features,
        args.endpoints,
        args.interiors,
        args.nucleus_targets,
    ):
        reject_evaluation_path(path)
    if args.stage in ("aggregate", "all"):
        aggregate(
            args.chunks,
            args.graph,
            args.stitch,
            args.expected_chunks,
            args.allow_partial,
            args.max_port_distance_nm,
        )
    if args.stage in ("classify", "all"):
        classify(
            args.graph,
            args.stitch,
            args.sizes,
            args.features,
            args.endpoints,
            args.interiors,
            args.nucleus_targets,
            args.prune_nm,
            args.tangent_nm,
            args.twig_nm,
            args.branch_profile_nm,
            args.spine_radius_ratio,
            args.spine_parent_collinearity,
            args.spine_parent_radius_balance,
            args.spine_perpendicularity,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
