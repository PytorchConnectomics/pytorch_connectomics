#!/usr/bin/env python3
"""GT-free chunk-wise skeletonization of arm0_96 ABISS final segments.

The expensive volume pass is embarrassingly parallel. Each task reads ABISS-sized cores
with a halo, downsamples only XY to an almost isotropic 20x18x18 nm grid, skeletonizes all
globally non-dust final labels present in the chunk, and writes compact graph fragments.
Positive-face continuation ports are extracted from the segmentation itself so a later
stage can reconnect fragments of the *same final segment ID* without guessing identity.

This module must not import, read, or accept skeleton GT. Its output belongs to
``arm096_error_correction/decoder_gtfree`` and is independent of the oracle directory.

Typical cluster use::

    sbatch dev/zebrafinch/sbatch_arm096_skeletonize.sh

One-chunk smoke test::

    python dev/zebrafinch/arm096_skeletonize_chunks.py --task-id 0 --num-tasks 10626 \
        --max-owned-chunks 1 --output /tmp/arm096_skeleton_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cc3d
import numpy as np

from .artifacts import reject_evaluation_path
from .sizes import load_size_inventory

REPO = Path(__file__).resolve().parents[3]
DEV = REPO / "dev" / "zebrafinch"
RUN = DEV / "wholevol_arm096_fullmask" / "seg_arm096_fullmask"
DEFAULT_SEG = RUN / "precomputed" / "seg" / "seg_arm096_fullmask"
DEFAULT_SIZES = RUN / "scratch" / "seg_arm096_fullmask" / "agg" / "info" / "seg_size_5_0_0_0.data"
DEFAULT_OUTPUT = DEV / "arm096_error_correction" / "decoder_gtfree" / "chunks"
NATIVE_RESOLUTION_ZYX_NM = np.asarray([20.0, 9.0, 9.0], dtype=np.float64)
DEFAULT_CORE_XYZ = np.asarray([512, 512, 256], dtype=np.int64)
DEFAULT_HALO_XYZ = np.asarray([32, 32, 16], dtype=np.int64)
DEFAULT_DOWNSAMPLE_ZYX = np.asarray([1, 2, 2], dtype=np.int64)
TEASAR = {
    "scale": 1.5,
    "const": 300,
    "pdrf_scale": 100000,
    "pdrf_exponent": 4,
    "soma_detection_threshold": 1e9,
    "soma_acceptance_threshold": 1e9,
}


def log(*values: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *values, flush=True)


def cloudvolume(path: Path):
    from cloudvolume import CloudVolume

    return CloudVolume(f"file://{path}", mip=0, fill_missing=True, bounded=False, progress=False)


def grid_chunks(shape_xyz: np.ndarray, core_xyz: np.ndarray):
    grid = np.ceil(shape_xyz / core_xyz).astype(np.int64)
    for z_index in range(int(grid[2])):
        for y_index in range(int(grid[1])):
            for x_index in range(int(grid[0])):
                index_xyz = np.asarray([x_index, y_index, z_index], dtype=np.int64)
                lo = index_xyz * core_xyz
                hi = np.minimum(lo + core_xyz, shape_xyz)
                yield f"z{z_index}_y{y_index}_x{x_index}", index_xyz, lo, hi, grid


def dense_filter(labels_zyx: np.ndarray, keep_labels: np.ndarray):
    present = np.intersect1d(np.unique(labels_zyx), keep_labels, assume_unique=False)
    present = present[present > 0]
    dense = np.zeros(labels_zyx.shape, dtype=np.uint32)
    if not len(present):
        return present, dense
    # Bound peak memory by remapping thin Z slabs.
    for z0 in range(0, labels_zyx.shape[0], 8):
        block = labels_zyx[z0 : z0 + 8]
        indices = np.searchsorted(present, block)
        safe = np.minimum(indices, len(present) - 1)
        found = present[safe] == block
        dense[z0 : z0 + 8] = np.where(found, safe + 1, 0).astype(np.uint32)
    return present, dense


def downsample_segmentation(labels_zyx: np.ndarray, factor_zyx: np.ndarray) -> np.ndarray:
    if np.all(factor_zyx == 1):
        return labels_zyx
    import tinybrain

    return tinybrain.downsample_segmentation(
        np.ascontiguousarray(labels_zyx),
        factor=tuple(int(value) for value in factor_zyx),
        num_mips=1,
    )[0]


def positive_face_ports(
    dense_zyx: np.ndarray,
    present: np.ndarray,
    core_lo_local: np.ndarray,
    core_hi_local: np.ndarray,
    read_lo_zyx: np.ndarray,
    coarse_resolution: np.ndarray,
    key: str,
    index_xyz: np.ndarray,
    grid_xyz: np.ndarray,
) -> dict[str, np.ndarray]:
    labels: list[int] = []
    axes: list[int] = []
    positions: list[np.ndarray] = []
    areas: list[int] = []
    neighbors: list[str] = []
    # ZYX axis -> XYZ grid slot and key formatter order.
    for axis_zyx, grid_slot in ((0, 2), (1, 1), (2, 0)):
        if int(index_xyz[grid_slot]) + 1 >= int(grid_xyz[grid_slot]):
            continue
        inside = int(core_hi_local[axis_zyx]) - 1
        outside = int(core_hi_local[axis_zyx])
        if outside >= dense_zyx.shape[axis_zyx]:
            continue
        left = np.take(dense_zyx, inside, axis=axis_zyx)
        right = np.take(dense_zyx, outside, axis=axis_zyx)
        continuation = np.where((left == right) & (left > 0), left, 0).astype(np.uint32)
        components, count = cc3d.connected_components(
            continuation, connectivity=8, return_N=True, binary_image=False
        )
        neighbor_index = index_xyz.copy()
        neighbor_index[grid_slot] += 1
        neighbor = f"z{int(neighbor_index[2])}_y{int(neighbor_index[1])}_x{int(neighbor_index[0])}"
        plane_axes = [axis for axis in range(3) if axis != axis_zyx]
        for component_id in range(1, int(count) + 1):
            points = np.argwhere(components == component_id)
            if not len(points):
                continue
            first = tuple(points[0].tolist())
            dense_label = int(continuation[first])
            if dense_label <= 0:
                continue
            position_coarse = np.zeros(3, dtype=np.float64)
            position_coarse[axis_zyx] = float(core_hi_local[axis_zyx]) - 0.5
            centroid = points.mean(axis=0)
            position_coarse[plane_axes] = centroid
            position_nm = read_lo_zyx * NATIVE_RESOLUTION_ZYX_NM
            position_nm = position_nm + position_coarse * coarse_resolution
            labels.append(int(present[dense_label - 1]))
            axes.append(axis_zyx)
            positions.append(position_nm)
            areas.append(len(points))
            neighbors.append(neighbor)
    return {
        "port_label": np.asarray(labels, dtype=np.uint64),
        "port_axis_zyx": np.asarray(axes, dtype=np.uint8),
        "port_position_nm": np.asarray(positions, dtype=np.float32).reshape(-1, 3),
        "port_area": np.asarray(areas, dtype=np.uint32),
        "port_neighbor": np.asarray(neighbors, dtype="U24"),
        "port_source": np.full(len(labels), key, dtype="U24"),
    }


def simplify_skeleton(
    vertices: np.ndarray,
    edges: np.ndarray,
    radii: np.ndarray,
    spacing_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse degree-2 chains while retaining samples at a physical spacing."""
    if len(vertices) <= 2 or not len(edges):
        return vertices, edges, radii
    adjacency: list[list[int]] = [[] for _ in range(len(vertices))]
    for left, right in edges.tolist():
        adjacency[left].append(right)
        adjacency[right].append(left)
    keys = {index for index, neighbors in enumerate(adjacency) if len(neighbors) != 2}
    if not keys:
        keys.add(0)
    visited: set[tuple[int, int]] = set()
    selected: list[int] = []
    selected_index: dict[int, int] = {}
    output_edges: list[tuple[int, int]] = []

    def add_vertex(index: int) -> int:
        if index not in selected_index:
            selected_index[index] = len(selected)
            selected.append(index)
        return selected_index[index]

    for start in sorted(keys):
        add_vertex(start)
        for neighbor in adjacency[start]:
            edge_key = tuple(sorted((start, neighbor)))
            if edge_key in visited:
                continue
            path = [start, neighbor]
            visited.add(edge_key)
            previous, current = start, neighbor
            while current not in keys:
                candidates = [node for node in adjacency[current] if node != previous]
                if not candidates:
                    break
                following = candidates[0]
                next_key = tuple(sorted((current, following)))
                if next_key in visited:
                    break
                visited.add(next_key)
                path.append(following)
                previous, current = current, following
            keep_path = [path[0]]
            accumulated = 0.0
            for left, right in zip(path[:-1], path[1:]):
                accumulated += float(np.linalg.norm(vertices[right] - vertices[left]))
                if accumulated >= spacing_nm and right != path[-1]:
                    keep_path.append(right)
                    accumulated = 0.0
            if keep_path[-1] != path[-1]:
                keep_path.append(path[-1])
            for left, right in zip(keep_path[:-1], keep_path[1:]):
                output_edges.append((add_vertex(left), add_vertex(right)))
    if not selected:
        return vertices[:0], np.zeros((0, 2), dtype=np.int64), radii[:0]
    selected_array = np.asarray(selected, dtype=np.int64)
    return (
        vertices[selected_array],
        np.asarray(output_edges, dtype=np.int64).reshape(-1, 2),
        radii[selected_array],
    )


def skeletonize_chunk(
    cv,
    key: str,
    index_xyz: np.ndarray,
    core_lo_xyz: np.ndarray,
    core_hi_xyz: np.ndarray,
    grid_xyz: np.ndarray,
    keep_labels: np.ndarray,
    output: Path,
    halo_xyz: np.ndarray,
    factor_zyx: np.ndarray,
    parallel: int,
    spacing_nm: float,
) -> None:
    import kimimaro

    shape_xyz = np.asarray(cv.shape[:3], dtype=np.int64)
    read_lo_xyz = np.maximum(0, core_lo_xyz - halo_xyz)
    read_hi_xyz = np.minimum(shape_xyz, core_hi_xyz + halo_xyz)
    start = time.time()
    raw_xyz = np.asarray(
        cv[
            int(read_lo_xyz[0]) : int(read_hi_xyz[0]),
            int(read_lo_xyz[1]) : int(read_hi_xyz[1]),
            int(read_lo_xyz[2]) : int(read_hi_xyz[2]),
        ]
    )
    if raw_xyz.ndim == 4:
        raw_xyz = raw_xyz[..., 0]
    raw_zyx = np.transpose(raw_xyz, (2, 1, 0))
    read_lo_zyx = read_lo_xyz[::-1]
    core_lo_zyx = core_lo_xyz[::-1]
    core_hi_zyx = core_hi_xyz[::-1]
    present, dense = dense_filter(raw_zyx, keep_labels)
    del raw_xyz, raw_zyx
    dense = downsample_segmentation(dense, factor_zyx)
    coarse_resolution = NATIVE_RESOLUTION_ZYX_NM * factor_zyx
    core_lo_local = (core_lo_zyx - read_lo_zyx) // factor_zyx
    core_hi_local = (core_hi_zyx - read_lo_zyx) // factor_zyx
    log(
        key,
        f"read/filter/downsample={time.time() - start:.1f}s",
        f"shape={tuple(dense.shape)} labels={len(present):,}",
    )

    skeletons = kimimaro.skeletonize(
        dense,
        teasar_params=TEASAR,
        anisotropy=tuple(float(value) for value in coarse_resolution),
        dust_threshold=1,
        progress=False,
        parallel=parallel,
        fix_branching=True,
        fix_borders=True,
    )

    vertex_parts: list[np.ndarray] = []
    radius_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    vertex_base = 0
    local_low_nm = core_lo_local * coarse_resolution
    local_high_nm = core_hi_local * coarse_resolution
    origin_nm = read_lo_zyx * NATIVE_RESOLUTION_ZYX_NM
    for dense_id, skeleton in skeletons.items():
        dense_index = int(dense_id) - 1
        if dense_index < 0 or dense_index >= len(present):
            continue
        vertices_local = np.asarray(skeleton.vertices, dtype=np.float64)
        edges = np.asarray(skeleton.edges, dtype=np.int64).reshape(-1, 2)
        if not len(vertices_local):
            continue
        keep_vertex = np.all(
            (vertices_local >= local_low_nm) & (vertices_local < local_high_nm), axis=1
        )
        if not np.any(keep_vertex):
            continue
        local_to_kept = np.full(len(vertices_local), -1, dtype=np.int64)
        local_to_kept[keep_vertex] = np.arange(np.count_nonzero(keep_vertex), dtype=np.int64)
        if len(edges):
            keep_edge = keep_vertex[edges[:, 0]] & keep_vertex[edges[:, 1]]
            edges = local_to_kept[edges[keep_edge]]
        radii = np.asarray(getattr(skeleton, "radii", np.zeros(len(vertices_local))))
        if radii.shape != (len(vertices_local),):
            radii = np.zeros(len(vertices_local), dtype=np.float32)
        vertices = vertices_local[keep_vertex] + origin_nm
        radii = radii[keep_vertex].astype(np.float32)
        vertices, edges, radii = simplify_skeleton(vertices, edges, radii, spacing_nm)
        if len(edges):
            edges = edges + vertex_base
        count = len(vertices)
        vertex_parts.append(vertices.astype(np.float32))
        radius_parts.append(radii)
        label_parts.append(np.full(count, present[dense_index], dtype=np.uint64))
        if len(edges):
            edge_parts.append(edges.astype(np.int64))
        vertex_base += count

    ports = positive_face_ports(
        dense,
        present,
        core_lo_local,
        core_hi_local,
        read_lo_zyx,
        coarse_resolution,
        key,
        index_xyz,
        grid_xyz,
    )
    vertices = np.concatenate(vertex_parts) if vertex_parts else np.zeros((0, 3), dtype=np.float32)
    radii = np.concatenate(radius_parts) if radius_parts else np.zeros(0, dtype=np.float32)
    vertex_label = np.concatenate(label_parts) if label_parts else np.zeros(0, dtype=np.uint64)
    edges = np.concatenate(edge_parts) if edge_parts else np.zeros((0, 2), dtype=np.int64)
    metadata = {
        "schema": 1,
        "key": key,
        "core_lo_xyz": core_lo_xyz.tolist(),
        "core_hi_xyz": core_hi_xyz.tolist(),
        "read_lo_xyz": read_lo_xyz.tolist(),
        "read_hi_xyz": read_hi_xyz.tolist(),
        "native_resolution_zyx_nm": NATIVE_RESOLUTION_ZYX_NM.tolist(),
        "downsample_zyx": factor_zyx.tolist(),
        "skeleton_spacing_nm": spacing_nm,
        "gt_free": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{key}.npz"
    temporary = output / f".{key}.{os.getpid()}.npz"
    np.savez_compressed(
        temporary,
        vertices_nm=vertices,
        radii_nm=radii,
        vertex_label=vertex_label,
        edges=edges,
        metadata=np.asarray(json.dumps(metadata)),
        **ports,
    )
    temporary.replace(destination)
    log(
        key,
        f"skeletons={len(skeletons):,} vertices={len(vertices):,} edges={len(edges):,}",
        f"ports={len(ports['port_label']):,} total={time.time() - start:.1f}s -> {destination}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg", type=Path, default=DEFAULT_SEG)
    parser.add_argument("--sizes", type=Path, default=DEFAULT_SIZES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-global-voxels", type=int, default=1_000)
    parser.add_argument("--core-xyz", type=int, nargs=3, default=DEFAULT_CORE_XYZ.tolist())
    parser.add_argument("--halo-xyz", type=int, nargs=3, default=DEFAULT_HALO_XYZ.tolist())
    parser.add_argument(
        "--downsample-zyx", type=int, nargs=3, default=DEFAULT_DOWNSAMPLE_ZYX.tolist()
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument("--num-tasks", type=int, default=80)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--skeleton-spacing-nm", type=float, default=250.0)
    parser.add_argument("--max-owned-chunks", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.seg, args.sizes, args.output):
        reject_evaluation_path(path)
    if args.task_id < 0 or args.task_id >= args.num_tasks:
        raise ValueError("task-id must satisfy 0 <= task-id < num-tasks")
    if args.min_global_voxels < 200:
        raise ValueError("min-global-voxels must be at least ABISS's 200-voxel dust floor")
    core_xyz = np.asarray(args.core_xyz, dtype=np.int64)
    halo_xyz = np.asarray(args.halo_xyz, dtype=np.int64)
    factor_zyx = np.asarray(args.downsample_zyx, dtype=np.int64)
    if np.any(core_xyz <= 0) or np.any(halo_xyz < 0) or np.any(factor_zyx <= 0):
        raise ValueError("core/downsample must be positive and halo nonnegative")
    if args.skeleton_spacing_nm <= 0:
        raise ValueError("skeleton-spacing-nm must be positive")
    sizes = load_size_inventory(args.sizes)
    keep_labels = np.sort(
        np.asarray(sizes["label"][sizes["size"] >= args.min_global_voxels], dtype=np.uint64)
    )
    cv = cloudvolume(args.seg)
    shape_xyz = np.asarray(cv.shape[:3], dtype=np.int64)
    chunks = list(grid_chunks(shape_xyz, core_xyz))
    owned = [item for index, item in enumerate(chunks) if index % args.num_tasks == args.task_id]
    if args.max_owned_chunks is not None:
        owned = owned[: args.max_owned_chunks]
    log(
        f"task={args.task_id}/{args.num_tasks} chunks={len(owned)}/{len(chunks)}",
        f"shape_xyz={tuple(shape_xyz)} keep={len(keep_labels):,} >= {args.min_global_voxels:,}",
    )
    for key, index_xyz, lo, hi, grid in owned:
        destination = args.output / f"{key}.npz"
        if destination.exists() and not args.overwrite:
            log(key, "exists; skip")
            continue
        skeletonize_chunk(
            cv,
            key,
            index_xyz,
            lo,
            hi,
            grid,
            keep_labels,
            args.output,
            halo_xyz,
            factor_zyx,
            args.parallel,
            args.skeleton_spacing_nm,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
