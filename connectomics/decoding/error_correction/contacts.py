#!/usr/bin/env python3
"""Build a GT-free face-contact graph for non-dust arm0_96 final segments.

Each core owns every +Z/+Y/+X voxel face originating inside it, including its positive
chunk boundary. Thus every native-resolution face is visited exactly once. Contact rows
are reduced by unordered final-segment pair within a chunk; no skeleton/evaluation GT is
read or accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .affinity import H5AffinityStore
from .sizes import load_size_inventory
from .skeletonize import (
    DEFAULT_CORE_XYZ,
    DEFAULT_SEG,
    DEFAULT_SIZES,
    NATIVE_RESOLUTION_ZYX_NM,
    cloudvolume,
    dense_filter,
    grid_chunks,
    reject_evaluation_path,
)

REPO = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT = (
    REPO / "dev" / "zebrafinch" / "arm096_error_correction" / "decoder_gtfree_v2" / "contact_chunks"
)
DEFAULT_AFFINITY = (
    REPO
    / "outputs"
    / "nisb_base_banis_plus_zebrafinch_heavy"
    / "20260726_114349"
    / "test_step=00200000"
    / "0"
    / (
        "raw_x1_ch0-1-2_chunked-raw_cs1008x1008x1008_halo72x72x72_"
        "zebrafinch_chunk_raw_grid1008_halo72.h5.chunks"
    )
)
DEFAULT_KEEP_MASK = REPO / "dev" / "zebrafinch" / "tissue_border_keep_mask_full.zarr"
AFFINITY_THRESHOLDS = np.asarray([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)


def log(*values: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *values, flush=True)


def segment_boundary_flags(
    dense_zyx: np.ndarray,
    present: np.ndarray,
    keep_zyx: np.ndarray,
    core_shape_zyx: np.ndarray,
    core_lo_zyx: np.ndarray,
    volume_shape_zyx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return exact volume/keep-boundary flags for labels owned by this core.

    Positive internal faces are owned once, matching the contact graph. Volume faces are
    handled explicitly. ``dense_zyx`` and ``keep_zyx`` include the positive one-voxel halo.
    """
    dense = np.asarray(dense_zyx, dtype=np.uint32)
    keep = np.asarray(keep_zyx, dtype=bool)
    core_shape = np.asarray(core_shape_zyx, dtype=np.int64)
    core_lo = np.asarray(core_lo_zyx, dtype=np.int64)
    volume_shape = np.asarray(volume_shape_zyx, dtype=np.int64)
    if dense.shape != keep.shape:
        raise ValueError("dense segmentation and keep mask must have equal shapes")
    core_slice = tuple(slice(0, int(value)) for value in core_shape)
    core = dense[core_slice]
    core_ids = np.unique(core)
    core_ids = core_ids[core_ids > 0]
    volume_boundary = np.zeros(len(present) + 1, dtype=bool)
    keep_boundary = np.zeros(len(present) + 1, dtype=bool)

    for axis in range(3):
        if core_lo[axis] == 0:
            ids = np.unique(np.take(core, 0, axis=axis))
            volume_boundary[ids[ids > 0]] = True
        if core_lo[axis] + core_shape[axis] == volume_shape[axis]:
            ids = np.unique(np.take(core, int(core_shape[axis]) - 1, axis=axis))
            volume_boundary[ids[ids > 0]] = True

        count = min(int(core_shape[axis]), dense.shape[axis] - 1)
        if count <= 0:
            continue
        low_slice = [slice(0, int(core_shape[index])) for index in range(3)]
        high_slice = list(low_slice)
        low_slice[axis] = slice(0, count)
        high_slice[axis] = slice(1, count + 1)
        keep_low = keep[tuple(low_slice)]
        keep_high = keep[tuple(high_slice)]
        transition = keep_low != keep_high
        if np.any(transition):
            ids = np.unique(
                np.concatenate(
                    (
                        dense[tuple(low_slice)][transition],
                        dense[tuple(high_slice)][transition],
                    )
                )
            )
            keep_boundary[ids[ids > 0]] = True

    # A labelled voxel outside the keep region is itself evidence that the object reaches
    # the algorithmic boundary, even if the transition falls outside this core's + halo.
    outside_ids = np.unique(core[~keep[core_slice]])
    keep_boundary[outside_ids[outside_ids > 0]] = True
    labels = present[core_ids.astype(np.int64) - 1].astype(np.uint64)
    return {
        "segment_label": labels,
        "touches_volume_boundary": volume_boundary[core_ids],
        "touches_keep_boundary": keep_boundary[core_ids],
    }


def face_rows(
    dense_zyx: np.ndarray,
    affinity_core_czyx: np.ndarray,
    present: np.ndarray,
    core_shape_zyx: np.ndarray,
    core_lo_zyx: np.ndarray,
    z_start: int,
    z_stop: int,
    axis: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Reduce one axis/slab to contact geometry and source-indexed affinity evidence."""
    limits = np.minimum(core_shape_zyx, np.asarray(dense_zyx.shape) - 1)
    starts = [z_start, 0, 0]
    stops = [z_stop, int(core_shape_zyx[1]), int(core_shape_zyx[2])]
    stops[axis] = min(stops[axis], int(limits[axis]))
    if any(stop <= start for start, stop in zip(starts, stops)):
        empty = np.zeros(0, dtype=np.uint64)
        return (
            empty,
            np.zeros(0, dtype=np.uint64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.uint64),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, len(AFFINITY_THRESHOLDS)), dtype=np.uint64),
        )
    left_slices = [slice(start, stop) for start, stop in zip(starts, stops)]
    right_slices = list(left_slices)
    right_slices[axis] = slice(starts[axis] + 1, stops[axis] + 1)
    left = dense_zyx[tuple(left_slices)]
    right = dense_zyx[tuple(right_slices)]
    boundary = (left > 0) & (right > 0) & (left != right)
    if not np.any(boundary):
        empty = np.zeros(0, dtype=np.uint64)
        return (
            empty,
            np.zeros(0, dtype=np.uint64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.uint64),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, len(AFFINITY_THRESHOLDS)), dtype=np.uint64),
        )

    coordinates = np.nonzero(boundary)
    left_id = left[boundary].astype(np.uint64)
    right_id = right[boundary].astype(np.uint64)
    lower = np.minimum(left_id, right_id)
    upper = np.maximum(left_id, right_id)
    base = np.uint64(len(present) + 1)
    key = lower * base + upper
    unique_key, inverse = np.unique(key, return_inverse=True)
    count = np.bincount(inverse).astype(np.uint64)

    coordinate_sum_nm = np.zeros((len(unique_key), 3), dtype=np.float64)
    for coordinate_axis in range(3):
        local = coordinates[coordinate_axis].astype(np.float64) + starts[coordinate_axis]
        local += 1.0 if coordinate_axis == axis else 0.5
        absolute_nm = (local + core_lo_zyx[coordinate_axis]) * NATIVE_RESOLUTION_ZYX_NM[
            coordinate_axis
        ]
        coordinate_sum_nm[:, coordinate_axis] = np.bincount(
            inverse, weights=absolute_nm, minlength=len(unique_key)
        )

    axis_count = np.zeros((len(unique_key), 3), dtype=np.uint64)
    axis_count[:, axis] = count
    normal_sum = np.zeros((len(unique_key), 3), dtype=np.int64)
    sign = np.where(left_id < right_id, 1.0, -1.0)
    normal_sum[:, axis] = np.rint(
        np.bincount(inverse, weights=sign, minlength=len(unique_key))
    ).astype(np.int64)
    affinity = np.asarray(affinity_core_czyx[(axis, *left_slices)][boundary], dtype=np.float32)
    affinity_sum = np.bincount(inverse, weights=affinity, minlength=len(unique_key)).astype(
        np.float64
    )
    affinity_sq_sum = np.bincount(
        inverse, weights=np.square(affinity), minlength=len(unique_key)
    ).astype(np.float64)
    affinity_max = np.zeros(len(unique_key), dtype=np.float32)
    np.maximum.at(affinity_max, inverse, affinity)
    affinity_ge = np.stack(
        [
            np.bincount(
                inverse,
                weights=affinity >= threshold,
                minlength=len(unique_key),
            )
            for threshold in AFFINITY_THRESHOLDS
        ],
        axis=1,
    ).astype(np.uint64)
    return (
        unique_key,
        count,
        coordinate_sum_nm,
        axis_count,
        normal_sum,
        affinity_sum,
        affinity_sq_sum,
        affinity_max,
        affinity_ge,
    )


def consolidate_rows(
    keys: list[np.ndarray],
    counts: list[np.ndarray],
    coordinate_sums: list[np.ndarray],
    axis_counts: list[np.ndarray],
    normal_sums: list[np.ndarray],
    affinity_sums: list[np.ndarray],
    affinity_sq_sums: list[np.ndarray],
    affinity_maxima: list[np.ndarray],
    affinity_ge_counts: list[np.ndarray],
    present: np.ndarray,
) -> dict[str, np.ndarray]:
    if not keys:
        return {
            "left": np.zeros(0, dtype=np.uint64),
            "right": np.zeros(0, dtype=np.uint64),
            "count": np.zeros(0, dtype=np.uint64),
            "centroid_nm_zyx": np.zeros((0, 3), dtype=np.float32),
            "axis_count_zyx": np.zeros((0, 3), dtype=np.uint64),
            "normal_sum_zyx": np.zeros((0, 3), dtype=np.int64),
            "area_nm2": np.zeros(0, dtype=np.float64),
            "affinity_sum": np.zeros(0, dtype=np.float64),
            "affinity_sq_sum": np.zeros(0, dtype=np.float64),
            "affinity_max": np.zeros(0, dtype=np.float32),
            "affinity_ge_count": np.zeros((0, len(AFFINITY_THRESHOLDS)), dtype=np.uint64),
            "affinity_thresholds": AFFINITY_THRESHOLDS,
        }
    key = np.concatenate(keys)
    count = np.concatenate(counts)
    coordinate_sum = np.concatenate(coordinate_sums)
    axis_count = np.concatenate(axis_counts)
    normal_sum = np.concatenate(normal_sums)
    affinity_sum = np.concatenate(affinity_sums)
    affinity_sq_sum = np.concatenate(affinity_sq_sums)
    affinity_max = np.concatenate(affinity_maxima)
    affinity_ge_count = np.concatenate(affinity_ge_counts)
    unique_key, inverse = np.unique(key, return_inverse=True)
    total_count = np.bincount(inverse, weights=count, minlength=len(unique_key)).astype(np.uint64)
    total_coordinate_sum = np.stack(
        [
            np.bincount(inverse, weights=coordinate_sum[:, axis], minlength=len(unique_key))
            for axis in range(3)
        ],
        axis=1,
    )
    total_axis_count = np.stack(
        [
            np.bincount(inverse, weights=axis_count[:, axis], minlength=len(unique_key))
            for axis in range(3)
        ],
        axis=1,
    ).astype(np.uint64)
    total_normal_sum = np.stack(
        [
            np.bincount(inverse, weights=normal_sum[:, axis], minlength=len(unique_key))
            for axis in range(3)
        ],
        axis=1,
    ).astype(np.int64)
    total_affinity_sum = np.bincount(inverse, weights=affinity_sum, minlength=len(unique_key))
    total_affinity_sq_sum = np.bincount(inverse, weights=affinity_sq_sum, minlength=len(unique_key))
    total_affinity_max = np.zeros(len(unique_key), dtype=np.float32)
    np.maximum.at(total_affinity_max, inverse, affinity_max)
    total_affinity_ge_count = np.stack(
        [
            np.bincount(
                inverse,
                weights=affinity_ge_count[:, threshold],
                minlength=len(unique_key),
            )
            for threshold in range(len(AFFINITY_THRESHOLDS))
        ],
        axis=1,
    ).astype(np.uint64)
    base = np.uint64(len(present) + 1)
    lower = unique_key // base
    upper = unique_key % base
    face_area_nm2 = np.asarray(
        [
            NATIVE_RESOLUTION_ZYX_NM[1] * NATIVE_RESOLUTION_ZYX_NM[2],
            NATIVE_RESOLUTION_ZYX_NM[0] * NATIVE_RESOLUTION_ZYX_NM[2],
            NATIVE_RESOLUTION_ZYX_NM[0] * NATIVE_RESOLUTION_ZYX_NM[1],
        ]
    )
    return {
        "left": present[lower.astype(np.int64) - 1].astype(np.uint64),
        "right": present[upper.astype(np.int64) - 1].astype(np.uint64),
        "count": total_count,
        "centroid_nm_zyx": (total_coordinate_sum / total_count[:, None]).astype(np.float32),
        "axis_count_zyx": total_axis_count,
        "normal_sum_zyx": total_normal_sum,
        "area_nm2": (total_axis_count * face_area_nm2).sum(axis=1),
        "affinity_sum": total_affinity_sum,
        "affinity_sq_sum": total_affinity_sq_sum,
        "affinity_max": total_affinity_max,
        "affinity_ge_count": total_affinity_ge_count,
        "affinity_thresholds": AFFINITY_THRESHOLDS,
    }


def contact_chunk(
    cv,
    affinity_store: H5AffinityStore,
    key: str,
    core_lo_xyz: np.ndarray,
    core_hi_xyz: np.ndarray,
    keep_labels: np.ndarray,
    output: Path,
    z_slab: int,
    min_global_voxels: int,
) -> None:
    shape_xyz = np.asarray(cv.shape[:3], dtype=np.int64)
    read_hi_xyz = np.minimum(shape_xyz, core_hi_xyz + 1)
    start = time.time()
    raw_xyz = np.asarray(
        cv[
            int(core_lo_xyz[0]) : int(read_hi_xyz[0]),
            int(core_lo_xyz[1]) : int(read_hi_xyz[1]),
            int(core_lo_xyz[2]) : int(read_hi_xyz[2]),
        ]
    )
    if raw_xyz.ndim == 4:
        raw_xyz = raw_xyz[..., 0]
    raw_zyx = np.transpose(raw_xyz, (2, 1, 0))
    present, dense = dense_filter(raw_zyx, keep_labels)
    del raw_xyz, raw_zyx
    core_shape_zyx = (core_hi_xyz - core_lo_xyz)[::-1]
    core_lo_zyx = core_lo_xyz[::-1]
    core_hi_zyx = core_hi_xyz[::-1]
    affinity = affinity_store.read_slab(core_lo_zyx, core_hi_zyx, low_halo=False).values
    expected_affinity_shape = (3, *tuple(int(value) for value in core_shape_zyx))
    if affinity.shape != expected_affinity_shape:
        raise ValueError(f"{key}: affinity shape {affinity.shape} != {expected_affinity_shape}")
    keep_array = affinity_store._keep_array()
    keep_zyx = (
        np.ones(dense.shape, dtype=bool)
        if keep_array is None
        else np.asarray(
            keep_array[
                int(core_lo_zyx[0]) : int(core_lo_zyx[0] + dense.shape[0]),
                int(core_lo_zyx[1]) : int(core_lo_zyx[1] + dense.shape[1]),
                int(core_lo_zyx[2]) : int(core_lo_zyx[2] + dense.shape[2]),
            ],
            dtype=bool,
        )
    )
    boundary = segment_boundary_flags(
        dense,
        present,
        keep_zyx,
        core_shape_zyx,
        core_lo_zyx,
        np.asarray(affinity_store.shape_zyx, dtype=np.int64),
    )

    keys: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    coordinate_sums: list[np.ndarray] = []
    axis_counts: list[np.ndarray] = []
    normal_sums: list[np.ndarray] = []
    affinity_sums: list[np.ndarray] = []
    affinity_sq_sums: list[np.ndarray] = []
    affinity_maxima: list[np.ndarray] = []
    affinity_ge_counts: list[np.ndarray] = []
    for z_start in range(0, int(core_shape_zyx[0]), z_slab):
        z_stop = min(z_start + z_slab, int(core_shape_zyx[0]))
        for axis in range(3):
            rows = face_rows(
                dense,
                affinity,
                present,
                core_shape_zyx,
                core_lo_zyx,
                z_start,
                z_stop,
                axis,
            )
            if len(rows[0]):
                keys.append(rows[0])
                counts.append(rows[1])
                coordinate_sums.append(rows[2])
                axis_counts.append(rows[3])
                normal_sums.append(rows[4])
                affinity_sums.append(rows[5])
                affinity_sq_sums.append(rows[6])
                affinity_maxima.append(rows[7])
                affinity_ge_counts.append(rows[8])
    rows = consolidate_rows(
        keys,
        counts,
        coordinate_sums,
        axis_counts,
        normal_sums,
        affinity_sums,
        affinity_sq_sums,
        affinity_maxima,
        affinity_ge_counts,
        present,
    )
    metadata = {
        "schema": 3,
        "key": key,
        "core_lo_xyz": core_lo_xyz.tolist(),
        "core_hi_xyz": core_hi_xyz.tolist(),
        "native_resolution_zyx_nm": NATIVE_RESOLUTION_ZYX_NM.tolist(),
        "positive_faces_owned": True,
        "min_global_voxels": min_global_voxels,
        "affinity_convention": "BANIS source-indexed R[c,p]=(p,p+e_c)",
        "restore_sigmoid_scale": affinity_store.restore_scale,
        "keep_mask": str(affinity_store.keep_mask_path),
        "gt_free": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{key}.npz"
    temporary = output / f".{key}.{os.getpid()}.npz"
    np.savez_compressed(
        temporary,
        metadata=np.asarray(json.dumps(metadata)),
        **rows,
        **boundary,
    )
    temporary.replace(destination)
    log(
        key,
        f"present={len(present):,} contacts={len(rows['left']):,}",
        f"inventory={len(boundary['segment_label']):,}",
        f"faces={int(rows['count'].sum()):,} total={time.time() - start:.1f}s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg", type=Path, default=DEFAULT_SEG)
    parser.add_argument("--sizes", type=Path, default=DEFAULT_SIZES)
    parser.add_argument("--affinity", type=Path, default=DEFAULT_AFFINITY)
    parser.add_argument("--keep-mask", type=Path, default=DEFAULT_KEEP_MASK)
    parser.add_argument("--restore-sigmoid-scale", type=float, default=0.2)
    parser.add_argument("--volume-shape-zyx", type=int, nargs=3, default=[5700, 10912, 10664])
    parser.add_argument("--affinity-chunk-size", type=int, default=1008)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-global-voxels", type=int, default=1_000)
    parser.add_argument("--core-xyz", type=int, nargs=3, default=DEFAULT_CORE_XYZ.tolist())
    parser.add_argument("--z-slab", type=int, default=8)
    parser.add_argument(
        "--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    parser.add_argument("--num-tasks", type=int, default=80)
    parser.add_argument("--max-owned-chunks", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.seg, args.sizes, args.affinity, args.keep_mask, args.output):
        reject_evaluation_path(path)
    if not 0 <= args.task_id < args.num_tasks:
        raise ValueError("task-id must satisfy 0 <= task-id < num-tasks")
    if args.min_global_voxels < 200:
        raise ValueError("min-global-voxels must be at least ABISS's 200-voxel dust floor")
    if args.z_slab <= 0:
        raise ValueError("z-slab must be positive")
    if args.restore_sigmoid_scale <= 0:
        raise ValueError("restore-sigmoid-scale must be positive")
    sizes = load_size_inventory(args.sizes)
    keep_labels = np.sort(
        np.asarray(sizes["label"][sizes["size"] >= args.min_global_voxels], dtype=np.uint64)
    )
    cv = cloudvolume(args.seg)
    affinity_store = H5AffinityStore(
        root=args.affinity,
        keep_mask=args.keep_mask,
        shape_zyx=args.volume_shape_zyx,
        chunk_size=args.affinity_chunk_size,
        restore_scale=args.restore_sigmoid_scale,
    )
    shape_xyz = np.asarray(cv.shape[:3], dtype=np.int64)
    chunks = list(grid_chunks(shape_xyz, np.asarray(args.core_xyz, dtype=np.int64)))
    owned = [item for index, item in enumerate(chunks) if index % args.num_tasks == args.task_id]
    if args.max_owned_chunks is not None:
        owned = owned[: args.max_owned_chunks]
    log(
        f"task={args.task_id}/{args.num_tasks} chunks={len(owned)}/{len(chunks)}",
        f"shape_xyz={tuple(shape_xyz)} keep={len(keep_labels):,}",
    )
    for key, _, lo, hi, _ in owned:
        destination = args.output / f"{key}.npz"
        if destination.exists() and not args.overwrite:
            log(key, "exists; skip")
            continue
        contact_chunk(
            cv,
            affinity_store,
            key,
            lo,
            hi,
            keep_labels,
            args.output,
            args.z_slab,
            args.min_global_voxels,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
