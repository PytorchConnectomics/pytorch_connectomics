"""NISB segmentation fusion study: split ch0-1-2 false merges with ch3-4-5 seeds.

This is a development script for the seed101 v3-erosion2 artifacts. It keeps
the short-range ch0-1-2 segmentation as the base result, finds large
long-range ch3-4-5 components that look like soma/body seeds, and updates a
copy of the ch0-1-2 H5 by splitting only primary parents that contain multiple
large guide seeds.

Two study modes are supported:

1. oracle: use skeleton/per-GT nERL ownership to choose which ch0-1-2 parents
   are known false-merge suspects. This estimates the upper bound for the
   proposed correction when parent selection is perfect.
2. auto: use only guide-component thresholds and primary/guide overlap. This is
   the production-style path with no GT/skeleton signal.

The implementation reuses the helper primitives from
``connectomics.decoding.decoders.longrange_guided_split`` but avoids loading the
full primary volume into memory. It copies the primary H5 and performs localized
read/modify/write updates on candidate parent bboxes.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import shutil
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.segmentation import watershed

from connectomics.decoding.decoders.longrange_guided_split import (
    _assign_from_markers,
    _format_bbox,
    _GuideSeed,
    _LabelStats,
    _pad_bbox,
    _prepare_markers,
    _select_large_guide_components,
    _union_bboxes,
    _write_csv,
)
from connectomics.decoding.decoders.segmentation_grow import (
    linear_indices_from_coords as _linear_indices_from_coords,
)
from connectomics.decoding.decoders.segmentation_grow import (
    search_sorted_indices,
    sparse_geodesic_grow_labels,
)
from connectomics.metrics.nerl import (
    NerlGraphOptions,
    compute_nerl_score_details,
)

RUN_DIR = Path("outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101")
DEFAULT_PRIMARY = RUN_DIR / "decoded_x1_ch0-1-2_affinity_cc_numba-0-0.66.h5"
DEFAULT_GUIDE = RUN_DIR / "decoded_x1_ch3-4-5_affinity_cc_numba-0-0.75.h5"
DEFAULT_PRIMARY_NERL = (
    RUN_DIR / "eval_decoded_x1_ch0-1-2_affinity_cc_numba-0-0.66_nerl_per_gt_erl.npz"
)
DEFAULT_PRIMARY_GT_SAMPLES = RUN_DIR / "err_analysis/ch0-1-2_cc0.66/gt_segment_samples.csv"
DEFAULT_SKELETON = Path("/projects/weilab/dataset/nisb/base/test/seed101/skeleton.pkl")
DEFAULT_OUT_DIR = RUN_DIR / "seg_fusion"


@dataclass(frozen=True)
class FusionParams:
    min_parent_voxels: int
    min_seed_voxels: int
    min_seed_axis_extent: int
    max_seed_axis_extent: int
    min_seed_overlap_voxels: int
    min_seed_guide_fraction: float
    min_seeds_in_parent: int
    max_splits_per_parent: int
    max_parents: int
    bbox_pad: tuple[int, int, int]
    erosion_iterations: int
    min_marker_voxels: int
    min_child_voxels: int
    max_marker_samples_per_seed: int
    new_label_start: int
    ignore_label: int
    assignment: str


@dataclass(frozen=True)
class FusionCandidate:
    primary_id: int
    primary_voxels: int
    primary_bbox: tuple[slice, slice, slice]
    seeds: tuple[_GuideSeed, ...]
    score: float
    source: str


@dataclass(frozen=True)
class OracleSelection:
    primary_ids: set[int]
    guide_ids: set[int]
    primary_seed_coords: dict[int, np.ndarray]
    guide_seed_coords: dict[int, np.ndarray]


def parse_csv_ints(value: str) -> list[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    ap.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    ap.add_argument("--dataset", default="main")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--guide-bbox-cache",
        type=Path,
        default=None,
        help="NPZ cache for guide label bbox stats; defaults under --out-dir",
    )
    ap.add_argument(
        "--recompute-guide-bboxes",
        action="store_true",
        help="ignore any existing guide bbox cache and rescan the guide H5",
    )
    ap.add_argument(
        "--bbox-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="processes for full-volume bbox scans; oracle usually avoids them",
    )
    ap.add_argument(
        "--bbox-axis",
        type=int,
        default=2,
        help="axis to split into chunk-aligned slabs for full bbox scans",
    )
    ap.add_argument(
        "--bbox-task-chunks",
        type=int,
        default=0,
        help=(
            "H5 chunks along --bbox-axis per task; <=0 makes one "
            "chunk-aligned partition per bbox worker"
        ),
    )
    ap.add_argument(
        "--guide-parent-workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="processes for guide-label to primary-parent overlap scans",
    )
    ap.add_argument(
        "--guide-parent-batch-size",
        type=int,
        default=8,
        help="guide labels handled per guide-parent worker task",
    )
    ap.add_argument(
        "--recompute-guide-parent",
        action="store_true",
        help="ignore existing per-guide parent scan .done files",
    )
    ap.add_argument("--mode", choices=("oracle", "auto", "both"), default="both")
    ap.add_argument("--dry-run", action="store_true", help="write reports but do not write H5s")
    ap.add_argument(
        "--evaluate",
        action="store_true",
        help="compute NERL for fused outputs; loads each fused H5 into memory",
    )

    # Automatic guide filtering.
    ap.add_argument("--min-parent-voxels", type=int, default=200_000)
    ap.add_argument("--min-seed-voxels", type=int, default=50_000)
    ap.add_argument("--min-seed-axis-extent", type=int, default=16)
    ap.add_argument("--max-seed-axis-extent", type=int, default=0)
    ap.add_argument("--min-seed-overlap-voxels", type=int, default=5_000)
    ap.add_argument("--min-seed-guide-fraction", type=float, default=0.25)
    ap.add_argument("--min-seeds-in-parent", type=int, default=2)
    ap.add_argument("--max-splits-per-parent", type=int, default=8)
    ap.add_argument("--max-parents", type=int, default=0)
    ap.add_argument("--bbox-pad", type=int, nargs=3, default=(16, 16, 8))
    ap.add_argument("--erosion-iterations", type=int, default=0)
    ap.add_argument("--min-marker-voxels", type=int, default=1_000)
    ap.add_argument("--min-child-voxels", type=int, default=5_000)
    ap.add_argument(
        "--max-marker-samples-per-seed",
        type=int,
        default=20_000,
        help=(
            "sample cap per guide marker for sampled_nearest assignment; "
            "<=0 keeps all marker voxels"
        ),
    )
    ap.add_argument(
        "--assignment",
        choices=("sparse_geodesic", "sampled_nearest", "watershed", "nearest"),
        default="sparse_geodesic",
        help=(
            "parent-mask assignment from guide markers; sparse_geodesic does marker growth "
            "on parent voxels without loading giant parent bboxes"
        ),
    )
    ap.add_argument(
        "--new-label-start",
        type=int,
        default=4_000_000_000,
        help="large uint32 label band for new children",
    )
    ap.add_argument("--ignore-label", type=int, default=0)

    # Oracle parent selection.
    ap.add_argument("--skeleton", type=Path, default=DEFAULT_SKELETON)
    ap.add_argument("--primary-nerl", type=Path, default=DEFAULT_PRIMARY_NERL)
    ap.add_argument("--oracle-primary-samples", type=Path, default=DEFAULT_PRIMARY_GT_SAMPLES)
    ap.add_argument(
        "--oracle-sample-source",
        choices=("csv", "skeleton"),
        default="csv",
        help="csv reuses err_analysis gt_segment_samples; skeleton resamples H5s",
    )
    ap.add_argument("--oracle-nerl-threshold", type=float, default=0.01)
    ap.add_argument(
        "--oracle-target-mode",
        choices=("low_nerl", "multi_gt"),
        default="low_nerl",
        help=(
            "low_nerl targets any primary owning a low-nERL GT; "
            "multi_gt keeps the older cross-GT-only oracle"
        ),
    )
    ap.add_argument(
        "--oracle-guide-source",
        choices=("global", "skeleton"),
        default="global",
        help=(
            "global uses all large ch3-4-5 components as split seeds; "
            "skeleton uses only guide labels sampled on target GT skeletons"
        ),
    )
    ap.add_argument("--oracle-primary-fraction", type=float, default=0.50)
    ap.add_argument("--oracle-guide-fraction", type=float, default=0.05)
    ap.add_argument("--max-nodes-per-gt", type=int, default=1024)
    ap.add_argument(
        "--target-primary-ids", default="", help="manual comma-separated oracle targets"
    )
    ap.add_argument("--skeleton-id-key", default="id")
    ap.add_argument("--skeleton-position-key", default="index_position")
    ap.add_argument("--skeleton-coord-order", default="xyz")
    ap.add_argument("--dataset-axis-order", default="xyz")

    # Optional NERL eval.
    ap.add_argument("--nerl-num-workers", type=int, default=1)
    ap.add_argument("--nerl-chunk-num", type=int, default=1)
    return ap.parse_args()


def reorder_coords(coords: np.ndarray, source_order: str, target_order: str) -> np.ndarray:
    source_order = source_order.lower()
    target_order = target_order.lower()
    if len(source_order) != len(target_order) or set(source_order) != set(target_order):
        raise ValueError(f"Cannot reorder coordinates from {source_order!r} to {target_order!r}")
    return coords[:, [source_order.index(axis) for axis in target_order]]


def normalize_axis(axis: int, ndim: int) -> int:
    axis = axis + ndim if axis < 0 else axis
    if axis < 0 or axis >= ndim:
        raise ValueError(f"Axis {axis} is invalid for ndim={ndim}")
    return axis


def bbox_to_str(bbox: tuple[slice, slice, slice]) -> tuple[str, str]:
    return _format_bbox(bbox)


def bboxes_intersect(a: tuple[slice, slice, slice], b: tuple[slice, slice, slice]) -> bool:
    return all(
        int(a[axis].start or 0) < int(b[axis].stop or 0)
        and int(b[axis].start or 0) < int(a[axis].stop or 0)
        for axis in range(3)
    )


def guide_bbox_cache_path(args: argparse.Namespace) -> Path:
    if args.guide_bbox_cache is not None:
        return args.guide_bbox_cache
    dataset_tag = args.dataset.strip("/").replace("/", "_") or "main"
    return (
        args.out_dir
        / f"{args.guide.stem}_{dataset_tag}_minvox{args.min_seed_voxels}_label_bboxes.npz"
    )


def selected_bbox_cache_path(
    out_dir: Path,
    volume_path: Path,
    dataset: str,
    source_name: str,
) -> Path:
    dataset_tag = dataset.strip("/").replace("/", "_") or "main"
    return out_dir / f"{volume_path.stem}_{dataset_tag}_{source_name}_selected_label_bboxes.npz"


def _npz_scalar_str(data: np.lib.npyio.NpzFile, key: str) -> str:
    return str(np.asarray(data[key]).item())


RawStats = dict[int, tuple[int, np.ndarray, np.ndarray]]
RawCounts = dict[int, int]


def merge_raw_stats(dst: RawStats, src: RawStats) -> None:
    for label, (count, lo, hi) in src.items():
        if label in dst:
            old_count, old_lo, old_hi = dst[label]
            dst[label] = (
                old_count + int(count),
                np.minimum(old_lo, lo),
                np.maximum(old_hi, hi),
            )
        else:
            dst[label] = (int(count), lo.astype(np.int64), hi.astype(np.int64))


def label_stats_from_raw(raw: RawStats) -> dict[int, _LabelStats]:
    return {
        label: _LabelStats(
            label=label,
            voxels=int(count),
            lo=tuple(int(v) for v in lo),
            hi=tuple(int(v) for v in hi),
        )
        for label, (count, lo, hi) in raw.items()
    }


def slab_cache_dir(final_cache_path: Path, pass_name: str) -> Path:
    return final_cache_path.with_suffix("") / pass_name


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    np.savez(tmp_path, **arrays)
    tmp_path.replace(path)


def raw_stats_to_arrays(raw: RawStats) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(sorted(raw), dtype=np.uint64)
    voxels = np.asarray([raw[int(label)][0] for label in labels], dtype=np.uint64)
    if labels.size:
        lo = np.asarray([raw[int(label)][1] for label in labels], dtype=np.int64)
        hi = np.asarray([raw[int(label)][2] for label in labels], dtype=np.int64)
    else:
        lo = np.empty((0, 3), dtype=np.int64)
        hi = np.empty((0, 3), dtype=np.int64)
    return labels, voxels, lo, hi


def arrays_to_raw_stats(
    labels: np.ndarray,
    voxels: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> RawStats:
    return {
        int(label): (
            int(voxel_count),
            np.asarray(lo_row, dtype=np.int64),
            np.asarray(hi_row, dtype=np.int64),
        )
        for label, voxel_count, lo_row, hi_row in zip(labels, voxels, lo, hi)
    }


def save_raw_stats(path: Path, raw: RawStats) -> None:
    labels, voxels, lo, hi = raw_stats_to_arrays(raw)
    save_npz_atomic(path, labels=labels, voxels=voxels, lo=lo, hi=hi)


def load_raw_stats(path: Path) -> RawStats:
    with np.load(path, allow_pickle=False) as data:
        return arrays_to_raw_stats(
            np.asarray(data["labels"], dtype=np.uint64),
            np.asarray(data["voxels"], dtype=np.uint64),
            np.asarray(data["lo"], dtype=np.int64),
            np.asarray(data["hi"], dtype=np.int64),
        )


def merge_raw_counts(dst: RawCounts, src: RawCounts) -> None:
    for label, count in src.items():
        dst[label] = dst.get(label, 0) + int(count)


def raw_counts_to_arrays(raw: RawCounts) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(sorted(raw), dtype=np.uint64)
    counts = np.asarray([raw[int(label)] for label in labels], dtype=np.uint64)
    return labels, counts


def arrays_to_raw_counts(labels: np.ndarray, counts: np.ndarray) -> RawCounts:
    return {int(label): int(count) for label, count in zip(labels, counts)}


def save_raw_counts(path: Path, raw: RawCounts) -> None:
    labels, counts = raw_counts_to_arrays(raw)
    save_npz_atomic(path, labels=labels, counts=counts)


def load_raw_counts(path: Path) -> RawCounts:
    with np.load(path, allow_pickle=False) as data:
        return arrays_to_raw_counts(
            np.asarray(data["labels"], dtype=np.uint64),
            np.asarray(data["counts"], dtype=np.uint64),
        )


def update_raw_stats_from_block(
    raw: RawStats,
    block: np.ndarray,
    slc: tuple[slice, slice, slice],
    *,
    ignore_label: int,
    label_filter: np.ndarray | None,
) -> None:
    labels, inverse, counts = np.unique(block, return_inverse=True, return_counts=True)
    keep_labels = labels != ignore_label
    if label_filter is not None:
        keep_labels &= np.isin(labels, label_filter, assume_unique=False)
    if not keep_labels.any():
        return

    valid_flat = keep_labels[inverse]
    if not valid_flat.any():
        return

    valid_inverse = inverse[valid_flat]
    used_indices = np.unique(valid_inverse)
    valid_mask = valid_flat.reshape(block.shape)
    coords = np.nonzero(valid_mask)
    start = np.asarray([int(s.start or 0) for s in slc], dtype=np.int64)

    lo = np.full((labels.size, 3), np.iinfo(np.int64).max, dtype=np.int64)
    hi = np.zeros((labels.size, 3), dtype=np.int64)
    for axis, coord in enumerate(coords):
        global_coord = start[axis] + coord.astype(np.int64, copy=False)
        np.minimum.at(lo[:, axis], valid_inverse, global_coord)
        np.maximum.at(hi[:, axis], valid_inverse, global_coord + 1)

    for label_index in used_indices:
        label = int(labels[label_index])
        raw_update = {
            label: (
                int(counts[label_index]),
                lo[label_index].copy(),
                hi[label_index].copy(),
            )
        }
        merge_raw_stats(raw, raw_update)


def update_raw_counts_from_block(
    raw: RawCounts,
    block: np.ndarray,
    *,
    ignore_label: int,
) -> None:
    labels, counts = np.unique(block, return_counts=True)
    keep = labels != ignore_label
    for label_value, count_value in zip(labels[keep], counts[keep]):
        label = int(label_value)
        raw[label] = raw.get(label, 0) + int(count_value)


def selection_from_pairs(pairs: Sequence[tuple[int, int]]) -> tuple[slice, slice, slice]:
    return tuple(slice(int(start), int(stop)) for start, stop in pairs)  # type: ignore[return-value]


def bbox_scan_tasks(
    shape: Sequence[int],
    chunks: Sequence[int],
    axis: int,
    task_chunks: int,
    num_partitions: int,
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    axis = normalize_axis(axis, len(shape))
    axis_chunks = max(1, int(np.ceil(int(shape[axis]) / max(1, int(chunks[axis])))))
    tasks = []

    if task_chunks <= 0:
        num_partitions = max(1, min(int(num_partitions), axis_chunks))
        chunk_edges = np.linspace(0, axis_chunks, num_partitions + 1, dtype=np.int64)
        chunk_ranges = [
            (int(chunk_edges[i]), int(chunk_edges[i + 1]))
            for i in range(num_partitions)
            if chunk_edges[i] < chunk_edges[i + 1]
        ]
    else:
        task_chunks = max(1, int(task_chunks))
        chunk_ranges = [
            (start, min(start + task_chunks, axis_chunks))
            for start in range(0, axis_chunks, task_chunks)
        ]

    for chunk_start, chunk_stop in chunk_ranges:
        start = chunk_start * int(chunks[axis])
        stop = min(chunk_stop * int(chunks[axis]), int(shape[axis]))
        pairs = [(0, int(v)) for v in shape]
        pairs[axis] = (start, stop)
        tasks.append(tuple(pairs))  # type: ignore[arg-type]
    return tasks


def scan_label_stats_h5_selection(
    path: str,
    dataset_name: str,
    selection_pairs: Sequence[tuple[int, int]],
    ignore_label: int,
    label_filter_values: tuple[int, ...] | None,
) -> RawStats:
    label_filter = (
        np.asarray(label_filter_values, dtype=np.uint64)
        if label_filter_values is not None
        else None
    )
    selection = selection_from_pairs(selection_pairs)
    raw: RawStats = {}
    with h5py.File(path, "r") as f:
        dataset = f[dataset_name]
        iterator = dataset.iter_chunks(sel=selection) if dataset.chunks else (selection,)
        for slc in iterator:
            block = np.asarray(dataset[slc])
            update_raw_stats_from_block(
                raw,
                block,
                slc,
                ignore_label=ignore_label,
                label_filter=label_filter,
            )
    return raw


def scan_label_counts_h5_selection(
    path: str,
    dataset_name: str,
    selection_pairs: Sequence[tuple[int, int]],
    ignore_label: int,
) -> RawCounts:
    selection = selection_from_pairs(selection_pairs)
    raw: RawCounts = {}
    with h5py.File(path, "r") as f:
        dataset = f[dataset_name]
        iterator = dataset.iter_chunks(sel=selection) if dataset.chunks else (selection,)
        for slc in iterator:
            block = np.asarray(dataset[slc])
            update_raw_counts_from_block(raw, block, ignore_label=ignore_label)
    return raw


def cached_label_counts_h5_selection(
    path: str,
    dataset_name: str,
    selection_pairs: Sequence[tuple[int, int]],
    ignore_label: int,
    cache_path: str,
    recompute: bool,
) -> RawCounts:
    path_obj = Path(cache_path)
    if path_obj.exists() and not recompute:
        try:
            return load_raw_counts(path_obj)
        except (KeyError, OSError, ValueError) as exc:
            print(f"  ignoring corrupt count slab cache {path_obj}: {exc}")
    raw = scan_label_counts_h5_selection(path, dataset_name, selection_pairs, ignore_label)
    save_raw_counts(path_obj, raw)
    return raw


def cached_label_stats_h5_selection(
    path: str,
    dataset_name: str,
    selection_pairs: Sequence[tuple[int, int]],
    ignore_label: int,
    label_filter_values: tuple[int, ...] | None,
    cache_path: str,
    recompute: bool,
) -> RawStats:
    path_obj = Path(cache_path)
    if path_obj.exists() and not recompute:
        try:
            return load_raw_stats(path_obj)
        except (KeyError, OSError, ValueError) as exc:
            print(f"  ignoring corrupt bbox slab cache {path_obj}: {exc}")
    raw = scan_label_stats_h5_selection(
        path,
        dataset_name,
        selection_pairs,
        ignore_label,
        label_filter_values,
    )
    save_raw_stats(path_obj, raw)
    return raw


def compute_label_counts_h5(
    path: Path,
    dataset_name: str,
    *,
    ignore_label: int,
    bbox_axis: int,
    bbox_workers: int,
    bbox_task_chunks: int,
    cache_dir: Path | None = None,
    recompute_slabs: bool = False,
    source_name: str,
) -> RawCounts:
    with h5py.File(path, "r") as f:
        dataset = f[dataset_name]
        shape = tuple(int(v) for v in dataset.shape)
        chunks = tuple(int(v) for v in (dataset.chunks or dataset.shape))

    tasks = bbox_scan_tasks(
        shape,
        chunks,
        bbox_axis,
        bbox_task_chunks,
        max(1, int(bbox_workers)),
    )
    workers = max(1, min(int(bbox_workers), len(tasks)))
    print(
        f"{source_name}: counting labels in {len(tasks)} slabs along axis "
        f"{normalize_axis(bbox_axis, len(shape))} with {workers} worker(s)"
    )
    if cache_dir is not None:
        print(f"{source_name}: count slab cache dir {cache_dir}")

    raw: RawCounts = {}
    cache_paths = [
        cache_dir / f"count_slab_{i:04d}.npz" if cache_dir is not None else None
        for i in range(len(tasks))
    ]
    if workers == 1:
        for i, selection_pairs in enumerate(tasks):
            cache_path = cache_paths[i]
            merge_raw_counts(
                raw,
                (
                    cached_label_counts_h5_selection(
                        str(path),
                        dataset_name,
                        selection_pairs,
                        ignore_label,
                        str(cache_path),
                        recompute_slabs,
                    )
                    if cache_path is not None
                    else scan_label_counts_h5_selection(
                        str(path),
                        dataset_name,
                        selection_pairs,
                        ignore_label,
                    )
                ),
            )
            print(f"  {source_name}: count slab {i + 1}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i, selection_pairs in enumerate(tasks):
                cache_path = cache_paths[i]
                if cache_path is not None:
                    futures.append(
                        executor.submit(
                            cached_label_counts_h5_selection,
                            str(path),
                            dataset_name,
                            selection_pairs,
                            ignore_label,
                            str(cache_path),
                            recompute_slabs,
                        )
                    )
                else:
                    futures.append(
                        executor.submit(
                            scan_label_counts_h5_selection,
                            str(path),
                            dataset_name,
                            selection_pairs,
                            ignore_label,
                        )
                    )
            for i, future in enumerate(as_completed(futures), start=1):
                merge_raw_counts(raw, future.result())
                print(f"  {source_name}: count slab {i}/{len(tasks)}")
    return raw


def compute_label_stats_h5(
    path: Path,
    dataset_name: str,
    *,
    ignore_label: int,
    bbox_axis: int,
    bbox_workers: int,
    bbox_task_chunks: int,
    label_filter_values: Iterable[int] | None = None,
    min_voxels: int = 0,
    cache_dir: Path | None = None,
    recompute_slabs: bool = False,
    source_name: str,
) -> dict[int, _LabelStats]:
    with h5py.File(path, "r") as f:
        dataset = f[dataset_name]
        shape = tuple(int(v) for v in dataset.shape)
        chunks = tuple(int(v) for v in (dataset.chunks or dataset.shape))

    tasks = bbox_scan_tasks(
        shape,
        chunks,
        bbox_axis,
        bbox_task_chunks,
        max(1, int(bbox_workers)),
    )
    if label_filter_values is not None:
        label_filter_tuple = tuple(
            sorted({int(v) for v in label_filter_values if int(v) != ignore_label})
        )
    elif min_voxels > 0:
        counts = compute_label_counts_h5(
            path,
            dataset_name,
            ignore_label=ignore_label,
            bbox_axis=bbox_axis,
            bbox_workers=bbox_workers,
            bbox_task_chunks=bbox_task_chunks,
            cache_dir=cache_dir / "count" if cache_dir is not None else None,
            recompute_slabs=recompute_slabs,
            source_name=source_name,
        )
        label_filter_tuple = tuple(
            sorted(label for label, count in counts.items() if count >= min_voxels)
        )
        print(
            f"{source_name}: retained {len(label_filter_tuple)}/{len(counts)} "
            f"labels with >= {min_voxels} voxels for bbox scan"
        )
        if not label_filter_tuple:
            return {}
    else:
        label_filter_tuple = None
    workers = max(1, min(int(bbox_workers), len(tasks)))
    stats_cache_dir = cache_dir / "bbox" if cache_dir is not None else None
    print(
        f"{source_name}: scanning bboxes in {len(tasks)} slabs along axis "
        f"{normalize_axis(bbox_axis, len(shape))} with {workers} worker(s)"
    )
    if stats_cache_dir is not None:
        print(f"{source_name}: bbox slab cache dir {stats_cache_dir}")

    raw: RawStats = {}
    cache_paths = [
        stats_cache_dir / f"bbox_slab_{i:04d}.npz" if stats_cache_dir is not None else None
        for i in range(len(tasks))
    ]
    if workers == 1:
        for i, selection_pairs in enumerate(tasks):
            cache_path = cache_paths[i]
            merge_raw_stats(
                raw,
                (
                    cached_label_stats_h5_selection(
                        str(path),
                        dataset_name,
                        selection_pairs,
                        ignore_label,
                        label_filter_tuple,
                        str(cache_path),
                        recompute_slabs,
                    )
                    if cache_path is not None
                    else scan_label_stats_h5_selection(
                        str(path),
                        dataset_name,
                        selection_pairs,
                        ignore_label,
                        label_filter_tuple,
                    )
                ),
            )
            print(f"  {source_name}: bbox slab {i + 1}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i, selection_pairs in enumerate(tasks):
                cache_path = cache_paths[i]
                if cache_path is not None:
                    futures.append(
                        executor.submit(
                            cached_label_stats_h5_selection,
                            str(path),
                            dataset_name,
                            selection_pairs,
                            ignore_label,
                            label_filter_tuple,
                            str(cache_path),
                            recompute_slabs,
                        )
                    )
                else:
                    futures.append(
                        executor.submit(
                            scan_label_stats_h5_selection,
                            str(path),
                            dataset_name,
                            selection_pairs,
                            ignore_label,
                            label_filter_tuple,
                        )
                    )
            for i, future in enumerate(as_completed(futures), start=1):
                merge_raw_stats(raw, future.result())
                print(f"  {source_name}: bbox slab {i}/{len(tasks)}")
    return label_stats_from_raw(raw)


def load_cached_guide_stats(
    path: Path,
    *,
    guide_path: Path,
    dataset: str,
    shape: Sequence[int],
    ignore_label: int,
    min_voxels: int = 0,
) -> dict[int, _LabelStats] | None:
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as data:
            cached_guide = _npz_scalar_str(data, "guide_path")
            cached_dataset = _npz_scalar_str(data, "dataset")
            cached_shape = tuple(int(v) for v in np.asarray(data["shape"], dtype=np.int64))
            cached_ignore_label = int(np.asarray(data["ignore_label"]).item())

            expected_guide = str(guide_path.resolve())
            if (
                cached_guide != expected_guide
                or cached_dataset != dataset
                or cached_shape != tuple(int(v) for v in shape)
                or cached_ignore_label != int(ignore_label)
            ):
                print(f"Ignoring stale guide bbox cache: {path}")
                return None

            voxels_all = np.asarray(data["voxels"], dtype=np.uint64)
            if min_voxels > 0:
                keep = np.flatnonzero(voxels_all >= int(min_voxels))
                labels = np.asarray(data["labels"], dtype=np.uint64)[keep]
                voxels = voxels_all[keep]
                lo = np.asarray(data["lo"], dtype=np.int64)[keep]
                hi = np.asarray(data["hi"], dtype=np.int64)[keep]
            else:
                labels = np.asarray(data["labels"], dtype=np.uint64)
                voxels = voxels_all
                lo = np.asarray(data["lo"], dtype=np.int64)
                hi = np.asarray(data["hi"], dtype=np.int64)
    except (KeyError, OSError, ValueError) as exc:
        print(f"Ignoring unreadable guide bbox cache {path}: {exc}")
        return None

    stats = {
        int(label): _LabelStats(
            label=int(label),
            voxels=int(voxel_count),
            lo=tuple(int(v) for v in lo_row),
            hi=tuple(int(v) for v in hi_row),
        )
        for label, voxel_count, lo_row, hi_row in zip(labels, voxels, lo, hi)
    }
    suffix = f" with >= {min_voxels} voxels" if min_voxels > 0 else ""
    print(f"Loaded {len(stats)} guide label bboxes{suffix} from {path}")
    return stats


def save_cached_guide_stats(
    path: Path,
    stats: dict[int, _LabelStats],
    *,
    guide_path: Path,
    dataset: str,
    shape: Sequence[int],
    ignore_label: int,
) -> None:
    raw = {
        label: (
            stat.voxels,
            np.asarray(stat.lo, dtype=np.int64),
            np.asarray(stat.hi, dtype=np.int64),
        )
        for label, stat in stats.items()
    }
    labels, voxels, lo, hi = raw_stats_to_arrays(raw)

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing final bbox cache for {len(stats)} labels to {path}")
    save_npz_atomic(
        path,
        labels=labels,
        voxels=voxels,
        lo=lo,
        hi=hi,
        shape=np.asarray(shape, dtype=np.int64),
        guide_path=np.asarray(str(guide_path.resolve())),
        dataset=np.asarray(dataset),
        ignore_label=np.asarray(ignore_label, dtype=np.int64),
    )
    print(f"Saved {len(stats)} guide label bboxes to {path}")


def load_or_compute_guide_stats(args: argparse.Namespace) -> dict[int, _LabelStats]:
    cache_path = guide_bbox_cache_path(args)
    with h5py.File(args.guide, "r") as f_guide:
        guide_ds = f_guide[args.dataset]
        shape = tuple(int(v) for v in guide_ds.shape)
        print(f"guide bbox cache: {cache_path}")

        if not args.recompute_guide_bboxes:
            cached = load_cached_guide_stats(
                cache_path,
                guide_path=args.guide,
                dataset=args.dataset,
                shape=shape,
                ignore_label=args.ignore_label,
                min_voxels=args.min_seed_voxels,
            )
            if cached is not None:
                return cached

        print(f"Scanning guide label bboxes: shape={guide_ds.shape} chunks={guide_ds.chunks}")

    stats = compute_label_stats_h5(
        args.guide,
        args.dataset,
        ignore_label=args.ignore_label,
        bbox_axis=args.bbox_axis,
        bbox_workers=args.bbox_workers,
        bbox_task_chunks=args.bbox_task_chunks,
        min_voxels=args.min_seed_voxels,
        cache_dir=slab_cache_dir(cache_path, "slabs"),
        recompute_slabs=args.recompute_guide_bboxes,
        source_name="guide",
    )

    save_cached_guide_stats(
        cache_path,
        stats,
        guide_path=args.guide,
        dataset=args.dataset,
        shape=shape,
        ignore_label=args.ignore_label,
    )
    return stats


def chunk_slices(
    chunk_index: tuple[int, int, int],
    shape: Sequence[int],
    chunks: Sequence[int],
) -> tuple[slice, slice, slice]:
    lo = [chunk_index[axis] * int(chunks[axis]) for axis in range(3)]
    hi = [min(lo[axis] + int(chunks[axis]), int(shape[axis])) for axis in range(3)]
    return tuple(slice(lo[axis], hi[axis]) for axis in range(3))  # type: ignore[return-value]


def trace_label_stats_from_points(
    dataset: h5py.Dataset,
    seed_coords_by_label: dict[int, np.ndarray],
    *,
    ignore_label: int,
    source_name: str,
) -> dict[int, _LabelStats]:
    """Trace connected label bboxes by walking only chunks touched by each label.

    This is much faster than a full-volume bbox scan when oracle skeleton samples
    already tell us one or more coordinates inside each target label.
    """

    shape = np.asarray(dataset.shape, dtype=np.int64)
    chunks = np.asarray(dataset.chunks or dataset.shape, dtype=np.int64)
    grid_shape = (shape + chunks - 1) // chunks

    stats: dict[int, _LabelStats] = {}
    for label, coords in sorted(seed_coords_by_label.items()):
        label = int(label)
        if label == ignore_label:
            continue
        coords = np.asarray(coords, dtype=np.int64)
        valid = np.all((coords >= 0) & (coords < shape), axis=1)
        coords = coords[valid]
        if coords.size == 0:
            continue

        seed_chunks = {tuple(int(v) for v in row) for row in np.unique(coords // chunks, axis=0)}
        queue: deque[tuple[int, int, int]] = deque(seed_chunks)
        visited: set[tuple[int, int, int]] = set()
        voxel_count = 0
        lo = shape.copy()
        hi = np.zeros(3, dtype=np.int64)

        while queue:
            chunk_index = queue.popleft()
            if chunk_index in visited:
                continue
            visited.add(chunk_index)
            slc = chunk_slices(chunk_index, shape, chunks)
            block = np.asarray(dataset[slc])
            mask = block == label
            if not mask.any():
                continue

            local_coords = np.nonzero(mask)
            start = np.asarray([int(s.start or 0) for s in slc], dtype=np.int64)
            lo = np.minimum(
                lo,
                start + np.asarray([int(axis.min()) for axis in local_coords], dtype=np.int64),
            )
            hi = np.maximum(
                hi,
                start + np.asarray([int(axis.max()) + 1 for axis in local_coords], dtype=np.int64),
            )
            voxel_count += int(mask.sum())

            for axis in range(3):
                if chunk_index[axis] > 0 and np.any(np.take(mask, 0, axis=axis)):
                    neighbor = list(chunk_index)
                    neighbor[axis] -= 1
                    queue.append(tuple(neighbor))  # type: ignore[arg-type]
                if chunk_index[axis] + 1 < grid_shape[axis] and np.any(
                    np.take(mask, -1, axis=axis)
                ):
                    neighbor = list(chunk_index)
                    neighbor[axis] += 1
                    queue.append(tuple(neighbor))  # type: ignore[arg-type]

            if len(visited) % 250 == 0:
                print(f"  {source_name}: traced label {label} through " f"{len(visited)} chunks")

        if voxel_count > 0:
            stats[label] = _LabelStats(
                label=label,
                voxels=voxel_count,
                lo=tuple(int(v) for v in lo),
                hi=tuple(int(v) for v in hi),
            )
            print(
                f"  {source_name}: label {label} bbox={_format_bbox(stats[label].bbox)} "
                f"voxels={voxel_count} chunks={len(visited)}"
            )
        else:
            print(f"  {source_name}: label {label} not found from seed chunks")

    return stats


def filter_stats(
    stats: dict[int, _LabelStats],
    label_ids: Iterable[int],
    ignore_label: int,
) -> dict[int, _LabelStats]:
    wanted = {int(v) for v in label_ids if int(v) != ignore_label}
    return {label: stat for label, stat in stats.items() if label in wanted}


def load_or_compute_selected_label_stats(
    volume_path: Path,
    dataset_name: str,
    cache_path: Path,
    label_ids: Iterable[int],
    seed_coords_by_label: dict[int, np.ndarray],
    *,
    ignore_label: int,
    recompute: bool,
    bbox_axis: int,
    bbox_workers: int,
    bbox_task_chunks: int,
    source_name: str,
) -> dict[int, _LabelStats]:
    wanted = {int(v) for v in label_ids if int(v) != ignore_label}
    if not wanted:
        return {}

    with h5py.File(volume_path, "r") as f:
        dataset = f[dataset_name]
        shape = tuple(int(v) for v in dataset.shape)
        cached: dict[int, _LabelStats] | None = None
        if not recompute:
            cached = load_cached_guide_stats(
                cache_path,
                guide_path=volume_path,
                dataset=dataset_name,
                shape=shape,
                ignore_label=ignore_label,
            )
        stats = cached or {}
        missing = wanted.difference(stats)
        if missing:
            print(
                f"{source_name}: computing {len(missing)} selected label bboxes "
                f"without a full-volume scan"
            )
            traceable_coords = {
                label: seed_coords_by_label[label]
                for label in sorted(missing)
                if label in seed_coords_by_label and seed_coords_by_label[label].size
            }
            traced = trace_label_stats_from_points(
                dataset,
                traceable_coords,
                ignore_label=ignore_label,
                source_name=source_name,
            )
            stats.update(traced)

            still_missing = wanted.difference(stats)
            if still_missing:
                print(
                    f"{source_name}: falling back to selected-label parallel scan for "
                    f"{len(still_missing)} labels"
                )
                stats.update(
                    compute_label_stats_h5(
                        volume_path,
                        dataset_name,
                        ignore_label=ignore_label,
                        bbox_axis=bbox_axis,
                        bbox_workers=bbox_workers,
                        bbox_task_chunks=bbox_task_chunks,
                        label_filter_values=still_missing,
                        cache_dir=slab_cache_dir(cache_path, "slabs"),
                        recompute_slabs=recompute,
                        source_name=source_name,
                    )
                )

            save_cached_guide_stats(
                cache_path,
                stats,
                guide_path=volume_path,
                dataset=dataset_name,
                shape=shape,
                ignore_label=ignore_label,
            )

    selected = filter_stats(stats, wanted, ignore_label)
    missing_after = wanted.difference(selected)
    if missing_after:
        print(f"{source_name}: missing bbox stats for labels {sorted(missing_after)}")
    return selected


def candidate_fieldnames() -> list[str]:
    return [
        "primary_id",
        "primary_voxels",
        "bbox_start",
        "bbox_stop",
        "retained_seed_count",
        "retained_seed_ids",
        "retained_seed_overlaps",
        "dominant_primary_fraction",
        "candidate_score",
        "decision",
        "reject_reason",
        "source",
    ]


def decision_fieldnames() -> list[str]:
    return [
        "primary_id",
        "accepted",
        "reject_reason",
        "bbox_start",
        "bbox_stop",
        "guide_seed_ids",
        "output_child_ids",
        "child_voxels",
        "marker_voxels",
        "unassigned_voxels",
        "geodesic_iterations",
        "changed_voxels",
        "assignment",
    ]


def load_nerl_by_gt(path: Path) -> dict[int, float]:
    data = np.load(path, allow_pickle=False)
    gt_ids = np.asarray(data["gt_segment_id"], dtype=np.int64)
    erl = np.asarray(data["erl"], dtype=np.float64)
    nerl = np.divide(
        erl[:, 0],
        erl[:, 1],
        out=np.full(erl.shape[0], np.nan, dtype=np.float64),
        where=erl[:, 1] > 0,
    )
    return {int(gt): float(value) for gt, value in zip(gt_ids, nerl)}


def load_skeleton_coords(
    path: Path,
    *,
    id_key: str,
    position_key: str,
    skeleton_coord_order: str,
    dataset_axis_order: str,
    max_nodes_per_gt: int,
) -> dict[int, np.ndarray]:
    with path.open("rb") as f:
        graph = pickle.load(f)

    coords_by_gt: dict[int, list[np.ndarray]] = defaultdict(list)
    for _node_id, node_data in graph.nodes(data=True):
        gt_id = int(node_data[id_key])
        coords_by_gt[gt_id].append(np.asarray(node_data[position_key], dtype=np.int64))

    sampled: dict[int, np.ndarray] = {}
    for gt_id, coords_list in coords_by_gt.items():
        coords = np.vstack(coords_list)
        coords = reorder_coords(coords, skeleton_coord_order, dataset_axis_order)
        if max_nodes_per_gt > 0 and len(coords) > max_nodes_per_gt:
            idx = np.linspace(0, len(coords) - 1, max_nodes_per_gt, dtype=np.int64)
            coords = coords[idx]
        sampled[gt_id] = coords.astype(np.int64, copy=False)
    return sampled


def sample_dataset_at_coords(dataset: h5py.Dataset, coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return np.empty(0, dtype=dataset.dtype)

    shape = np.asarray(dataset.shape, dtype=np.int64)
    chunks = np.asarray(dataset.chunks or (64, 64, 64), dtype=np.int64)
    valid = np.all((coords >= 0) & (coords < shape), axis=1)
    valid_coords = coords[valid]
    labels = np.full(coords.shape[0], 0, dtype=dataset.dtype)
    if valid_coords.size == 0:
        return labels

    chunk_ids = valid_coords // chunks
    order = np.lexsort((chunk_ids[:, 2], chunk_ids[:, 1], chunk_ids[:, 0]))
    sorted_coords = valid_coords[order]
    sorted_chunks = chunk_ids[order]
    sampled_valid = np.empty(len(valid_coords), dtype=dataset.dtype)

    start = 0
    while start < len(sorted_coords):
        end = start + 1
        while end < len(sorted_coords) and np.array_equal(sorted_chunks[end], sorted_chunks[start]):
            end += 1

        chunk_id = sorted_chunks[start]
        lo = chunk_id * chunks
        hi = np.minimum(lo + chunks, shape)
        block = np.asarray(dataset[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]])
        local = sorted_coords[start:end] - lo
        sampled_valid[order[start:end]] = block[local[:, 0], local[:, 1], local[:, 2]]
        start = end

    labels[np.flatnonzero(valid)] = sampled_valid
    return labels


def dominant_nonzero(labels: np.ndarray) -> tuple[int, int, float, int]:
    labels, counts = np.unique(labels[labels != 0], return_counts=True)
    if labels.size == 0:
        return 0, 0, 0.0, 0
    order = np.argsort(counts)[::-1]
    total = int(counts.sum())
    i = int(order[0])
    return int(labels[i]), int(counts[i]), float(counts[i] / total), int(labels.size)


def parse_top_segment_count(value: str, segment_id: int) -> int:
    for part in value.split(";"):
        if not part:
            continue
        label_text, _, count_text = part.partition(":")
        if label_text and int(label_text) == segment_id and count_text:
            return int(count_text)
    return 0


def oracle_targets_from_gt_sample_csv(
    args: argparse.Namespace,
    out_dir: Path,
) -> OracleSelection | None:
    manual = set(parse_csv_ints(args.target_primary_ids))
    if manual:
        return OracleSelection(
            primary_ids=manual,
            guide_ids=set(),
            primary_seed_coords={},
            guide_seed_coords={},
        )

    if not args.oracle_primary_samples.exists():
        print(
            f"Oracle sample CSV not found, falling back to skeleton sampling: {args.oracle_primary_samples}"
        )
        return None

    rows_by_primary: dict[int, list[dict[str, object]]] = defaultdict(list)
    gt_rows: list[dict[str, object]] = []
    with args.oracle_primary_samples.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            primary_text = (row.get("dominant_segment_id") or "").strip()
            if not primary_text:
                continue
            primary_id = int(primary_text)
            nerl = float(row["nerl"])
            sampled_nodes = int(row["sampled_nodes"])
            primary_fraction = float(row["dominant_fraction"])
            primary_nodes = parse_top_segment_count(row.get("top_pred_segments", ""), primary_id)
            if primary_nodes == 0 and sampled_nodes:
                primary_nodes = int(round(sampled_nodes * primary_fraction))
            out_row = {
                "gt_segment_id": int(row["gt_segment_id"]),
                "nerl": f"{nerl:.8g}",
                "primary_id": primary_id,
                "primary_nodes": primary_nodes,
                "primary_fraction": f"{primary_fraction:.6g}",
                "primary_unique_labels": int(row["num_pred_segments"]),
                "guide_id": 0,
                "guide_nodes": 0,
                "guide_fraction": "0",
                "guide_unique_labels": 0,
            }
            gt_rows.append(out_row)
            if primary_fraction >= args.oracle_primary_fraction:
                rows_by_primary[primary_id].append(out_row)

    _write_csv(
        out_dir / "oracle_gt_ownership.csv",
        gt_rows,
        [
            "gt_segment_id",
            "nerl",
            "primary_id",
            "primary_nodes",
            "primary_fraction",
            "primary_unique_labels",
            "guide_id",
            "guide_nodes",
            "guide_fraction",
            "guide_unique_labels",
        ],
    )

    targets: set[int] = set()
    target_rows: list[dict[str, object]] = []
    for primary_id, rows in sorted(rows_by_primary.items()):
        bad_rows = [row for row in rows if float(row["nerl"]) <= args.oracle_nerl_threshold]
        if args.oracle_target_mode == "multi_gt":
            accepted = len(rows) >= 2 and bool(bad_rows)
        else:
            accepted = bool(bad_rows)
        if not accepted:
            continue
        targets.add(primary_id)
        target_rows.append(
            {
                "primary_id": primary_id,
                "num_gt": len(rows),
                "num_bad_gt": len(bad_rows),
                "gt_ids": ";".join(str(row["gt_segment_id"]) for row in rows),
                "bad_gt_ids": ";".join(str(row["gt_segment_id"]) for row in bad_rows),
                "guide_ids": "",
                "min_nerl": f"{min(float(row['nerl']) for row in rows):.8g}",
            }
        )

    _write_csv(
        out_dir / "oracle_targets.csv",
        target_rows,
        ["primary_id", "num_gt", "num_bad_gt", "gt_ids", "bad_gt_ids", "guide_ids", "min_nerl"],
    )
    print(
        f"Loaded oracle targets from {args.oracle_primary_samples}: "
        f"{len(targets)} primary IDs with nERL <= {args.oracle_nerl_threshold}"
    )
    return OracleSelection(
        primary_ids=targets,
        guide_ids=set(),
        primary_seed_coords={},
        guide_seed_coords={},
    )


def oracle_targets_from_skeleton(
    primary_ds: h5py.Dataset,
    guide_ds: h5py.Dataset,
    args: argparse.Namespace,
    out_dir: Path,
) -> OracleSelection:
    manual = set(parse_csv_ints(args.target_primary_ids))

    nerl_by_gt = load_nerl_by_gt(args.primary_nerl)
    coords_by_gt = load_skeleton_coords(
        args.skeleton,
        id_key=args.skeleton_id_key,
        position_key=args.skeleton_position_key,
        skeleton_coord_order=args.skeleton_coord_order,
        dataset_axis_order=args.dataset_axis_order,
        max_nodes_per_gt=args.max_nodes_per_gt,
    )

    gt_rows: list[dict[str, object]] = []
    owner_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    samples_by_gt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, gt_id in enumerate(sorted(coords_by_gt), start=1):
        coords = coords_by_gt[gt_id]
        primary_labels = sample_dataset_at_coords(primary_ds, coords)
        guide_labels = sample_dataset_at_coords(guide_ds, coords)
        samples_by_gt[gt_id] = (coords, primary_labels, guide_labels)
        primary_id, primary_count, primary_fraction, primary_unique = dominant_nonzero(
            primary_labels
        )
        guide_id, guide_count, guide_fraction, guide_unique = dominant_nonzero(guide_labels)
        nerl = nerl_by_gt.get(gt_id, float("nan"))
        row = {
            "gt_segment_id": gt_id,
            "nerl": f"{nerl:.8g}",
            "primary_id": primary_id,
            "primary_nodes": primary_count,
            "primary_fraction": f"{primary_fraction:.6g}",
            "primary_unique_labels": primary_unique,
            "guide_id": guide_id,
            "guide_nodes": guide_count,
            "guide_fraction": f"{guide_fraction:.6g}",
            "guide_unique_labels": guide_unique,
        }
        gt_rows.append(row)
        if primary_id and primary_fraction >= args.oracle_primary_fraction:
            owner_groups[primary_id].append(row)
        if i == 1 or i == len(coords_by_gt) or i % 100 == 0:
            print(f"  oracle skeleton sampling {i}/{len(coords_by_gt)}")

    _write_csv(
        out_dir / "oracle_gt_ownership.csv",
        gt_rows,
        [
            "gt_segment_id",
            "nerl",
            "primary_id",
            "primary_nodes",
            "primary_fraction",
            "primary_unique_labels",
            "guide_id",
            "guide_nodes",
            "guide_fraction",
            "guide_unique_labels",
        ],
    )

    targets: set[int] = set()
    guide_ids_by_primary: dict[int, set[int]] = defaultdict(set)
    target_rows: list[dict[str, object]] = []
    for primary_id, rows in sorted(owner_groups.items()):
        bad_rows = [row for row in rows if float(row["nerl"]) <= args.oracle_nerl_threshold]
        guide_ids = {
            int(row["guide_id"])
            for row in rows
            if int(row["guide_id"]) != 0
            and float(row["guide_fraction"]) >= args.oracle_guide_fraction
        }
        if manual:
            accepted = primary_id in manual
        elif args.oracle_target_mode == "multi_gt":
            accepted = len(rows) >= 2 and bool(bad_rows) and len(guide_ids) >= 2
        else:
            accepted = bool(bad_rows)
        if accepted:
            targets.add(primary_id)
            guide_ids_by_primary[primary_id].update(guide_ids)
            target_rows.append(
                {
                    "primary_id": primary_id,
                    "num_gt": len(rows),
                    "num_bad_gt": len(bad_rows),
                    "gt_ids": ";".join(str(row["gt_segment_id"]) for row in rows),
                    "bad_gt_ids": ";".join(str(row["gt_segment_id"]) for row in bad_rows),
                    "guide_ids": ";".join(str(v) for v in sorted(guide_ids)),
                    "min_nerl": f"{min(float(row['nerl']) for row in rows):.8g}",
                }
            )

    missing_manual = manual.difference(targets)
    for primary_id in sorted(missing_manual):
        targets.add(primary_id)
        target_rows.append(
            {
                "primary_id": primary_id,
                "num_gt": 0,
                "num_bad_gt": 0,
                "gt_ids": "",
                "bad_gt_ids": "",
                "guide_ids": "",
                "min_nerl": "nan",
            }
        )

    _write_csv(
        out_dir / "oracle_targets.csv",
        target_rows,
        ["primary_id", "num_gt", "num_bad_gt", "gt_ids", "bad_gt_ids", "guide_ids", "min_nerl"],
    )

    primary_seed_lists: dict[int, list[np.ndarray]] = defaultdict(list)
    guide_seed_lists: dict[int, list[np.ndarray]] = defaultdict(list)
    for primary_id in sorted(targets):
        for row in owner_groups.get(primary_id, []):
            gt_id = int(row["gt_segment_id"])
            coords, primary_labels, guide_labels = samples_by_gt[gt_id]
            primary_mask = primary_labels == primary_id
            if primary_mask.any():
                primary_seed_lists[primary_id].append(coords[primary_mask])
            for guide_id in sorted(guide_ids_by_primary.get(primary_id, set())):
                guide_mask = guide_labels == guide_id
                if guide_mask.any():
                    guide_seed_lists[guide_id].append(coords[guide_mask])

    primary_seed_coords = {
        label: np.vstack(parts) for label, parts in primary_seed_lists.items() if parts
    }
    guide_seed_coords = {
        label: np.vstack(parts) for label, parts in guide_seed_lists.items() if parts
    }
    guide_ids = set(guide_seed_coords)
    return OracleSelection(
        primary_ids=targets,
        guide_ids=guide_ids,
        primary_seed_coords=primary_seed_coords,
        guide_seed_coords=guide_seed_coords,
    )


def find_guide_parent_h5(
    primary_ds: h5py.Dataset,
    guide_ds: h5py.Dataset,
    guide_stats: _LabelStats,
    params: FusionParams,
) -> _GuideSeed | None:
    bbox = guide_stats.bbox
    primary_crop = np.asarray(primary_ds[bbox])
    guide_crop = np.asarray(guide_ds[bbox])
    seed_mask = guide_crop == guide_stats.label
    if not seed_mask.any():
        return None

    labels, counts = np.unique(primary_crop[seed_mask], return_counts=True)
    keep = labels != params.ignore_label
    labels = labels[keep]
    counts = counts[keep]
    if labels.size == 0:
        return None

    order = np.argsort(counts)[::-1]
    parent_id = int(labels[order[0]])
    overlap = int(counts[order[0]])
    fraction = overlap / float(guide_stats.voxels)
    if overlap < params.min_seed_overlap_voxels:
        return None
    if fraction < params.min_seed_guide_fraction:
        return None
    return _GuideSeed(
        guide_id=guide_stats.label,
        guide_voxels=guide_stats.voxels,
        bbox=bbox,
        primary_id=parent_id,
        overlap_voxels=overlap,
        dominant_primary_fraction=float(fraction),
    )


def label_stats_record(
    stat: _LabelStats,
) -> tuple[int, int, tuple[int, int, int], tuple[int, int, int]]:
    return (stat.label, stat.voxels, stat.lo, stat.hi)


def label_stats_from_record(
    record: tuple[int, int, tuple[int, int, int], tuple[int, int, int]],
) -> _LabelStats:
    label, voxels, lo, hi = record
    return _LabelStats(label=int(label), voxels=int(voxels), lo=tuple(lo), hi=tuple(hi))


def guide_parent_result_path(result_dir: Path, guide_id: int) -> Path:
    return result_dir / f"{int(guide_id)}.npz"


def guide_parent_done_path(done_dir: Path, guide_id: int) -> Path:
    return done_dir / f"{int(guide_id)}.done"


def write_done(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    tmp_path.write_text("done\n")
    tmp_path.replace(path)


def save_guide_seed_result(path: Path, seed: _GuideSeed | None, stat: _LabelStats) -> None:
    if seed is None:
        save_npz_atomic(
            path,
            valid=np.asarray(0, dtype=np.uint8),
            guide_id=np.asarray(stat.label, dtype=np.uint64),
            guide_voxels=np.asarray(stat.voxels, dtype=np.uint64),
            lo=np.asarray(stat.lo, dtype=np.int64),
            hi=np.asarray(stat.hi, dtype=np.int64),
            primary_id=np.asarray(0, dtype=np.uint64),
            overlap_voxels=np.asarray(0, dtype=np.uint64),
            dominant_primary_fraction=np.asarray(0.0, dtype=np.float64),
        )
        return
    save_npz_atomic(
        path,
        valid=np.asarray(1, dtype=np.uint8),
        guide_id=np.asarray(seed.guide_id, dtype=np.uint64),
        guide_voxels=np.asarray(seed.guide_voxels, dtype=np.uint64),
        lo=np.asarray(stat.lo, dtype=np.int64),
        hi=np.asarray(stat.hi, dtype=np.int64),
        primary_id=np.asarray(seed.primary_id, dtype=np.uint64),
        overlap_voxels=np.asarray(seed.overlap_voxels, dtype=np.uint64),
        dominant_primary_fraction=np.asarray(seed.dominant_primary_fraction, dtype=np.float64),
    )


def load_guide_seed_result(path: Path) -> _GuideSeed | None:
    with np.load(path, allow_pickle=False) as data:
        if int(np.asarray(data["valid"]).item()) == 0:
            return None
        guide_id = int(np.asarray(data["guide_id"]).item())
        guide_voxels = int(np.asarray(data["guide_voxels"]).item())
        lo = tuple(int(v) for v in np.asarray(data["lo"], dtype=np.int64))
        hi = tuple(int(v) for v in np.asarray(data["hi"], dtype=np.int64))
        return _GuideSeed(
            guide_id=guide_id,
            guide_voxels=guide_voxels,
            bbox=tuple(slice(a, b) for a, b in zip(lo, hi)),  # type: ignore[arg-type]
            primary_id=int(np.asarray(data["primary_id"]).item()),
            overlap_voxels=int(np.asarray(data["overlap_voxels"]).item()),
            dominant_primary_fraction=float(np.asarray(data["dominant_primary_fraction"]).item()),
        )


def scan_guide_parent_batch_worker(
    primary_path: str,
    guide_path: str,
    dataset_name: str,
    records: Sequence[tuple[int, int, tuple[int, int, int], tuple[int, int, int]]],
    params: FusionParams,
    result_dir: str,
    done_dir: str,
    recompute: bool,
) -> tuple[int, int]:
    result_dir_path = Path(result_dir)
    done_dir_path = Path(done_dir)
    pending = []
    skipped = 0
    for record in records:
        guide_id = int(record[0])
        result_path = guide_parent_result_path(result_dir_path, guide_id)
        done_path = guide_parent_done_path(done_dir_path, guide_id)
        if done_path.exists() and result_path.exists() and not recompute:
            skipped += 1
            continue
        pending.append(record)

    if not pending:
        return 0, skipped

    computed = 0
    with h5py.File(primary_path, "r") as f_primary, h5py.File(guide_path, "r") as f_guide:
        primary_ds = f_primary[dataset_name]
        guide_ds = f_guide[dataset_name]
        for record in pending:
            stat = label_stats_from_record(record)
            result_path = guide_parent_result_path(result_dir_path, stat.label)
            done_path = guide_parent_done_path(done_dir_path, stat.label)
            seed = find_guide_parent_h5(primary_ds, guide_ds, stat, params)
            save_guide_seed_result(result_path, seed, stat)
            write_done(done_path)
            computed += 1
    return computed, skipped


def scan_guide_parents_checkpointed(
    primary_path: Path,
    guide_path: Path,
    dataset_name: str,
    large_guides: Sequence[_LabelStats],
    params: FusionParams,
    work_dir: Path,
    *,
    workers: int,
    batch_size: int,
    recompute: bool,
    source: str,
) -> list[_GuideSeed]:
    result_dir = work_dir / "results"
    done_dir = work_dir / "done"
    result_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    records = [label_stats_record(stat) for stat in large_guides]
    pending_records = [
        record
        for record in records
        if recompute
        or not (
            guide_parent_done_path(done_dir, int(record[0])).exists()
            and guide_parent_result_path(result_dir, int(record[0])).exists()
        )
    ]
    batch_size = max(1, int(batch_size))
    batches = [
        pending_records[i : i + batch_size] for i in range(0, len(pending_records), batch_size)
    ]
    workers = max(1, min(int(workers), len(batches) or 1))
    print(
        f"{source}: guide-parent cache {work_dir}; "
        f"{len(records) - len(pending_records)} done, {len(pending_records)} pending, "
        f"{workers} worker(s)"
    )

    if batches:
        if workers == 1:
            completed = 0
            for i, batch in enumerate(batches, start=1):
                computed, _skipped = scan_guide_parent_batch_worker(
                    str(primary_path),
                    str(guide_path),
                    dataset_name,
                    batch,
                    params,
                    str(result_dir),
                    str(done_dir),
                    recompute,
                )
                completed += computed
                print(
                    f"  {source}: guide-parent batch {i}/{len(batches)} "
                    f"computed={completed}/{len(pending_records)}"
                )
        else:
            completed = 0
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        scan_guide_parent_batch_worker,
                        str(primary_path),
                        str(guide_path),
                        dataset_name,
                        batch,
                        params,
                        str(result_dir),
                        str(done_dir),
                        recompute,
                    )
                    for batch in batches
                ]
                for i, future in enumerate(as_completed(futures), start=1):
                    computed, _skipped = future.result()
                    completed += computed
                    print(
                        f"  {source}: guide-parent batch {i}/{len(batches)} "
                        f"computed={completed}/{len(pending_records)}"
                    )

    seeds: list[_GuideSeed] = []
    missing_done = 0
    for stat in large_guides:
        result_path = guide_parent_result_path(result_dir, stat.label)
        done_path = guide_parent_done_path(done_dir, stat.label)
        if not done_path.exists() or not result_path.exists():
            missing_done += 1
            continue
        seed = load_guide_seed_result(result_path)
        if seed is not None:
            seeds.append(seed)
    if missing_done:
        print(f"{source}: missing guide-parent results for {missing_done} guide labels")
    return seeds


def save_parent_seed_results(path: Path, primary_id: int, seeds: Sequence[_GuideSeed]) -> None:
    labels = np.asarray([seed.guide_id for seed in seeds], dtype=np.uint64)
    voxels = np.asarray([seed.guide_voxels for seed in seeds], dtype=np.uint64)
    if seeds:
        lo = np.asarray(
            [[int(slc.start or 0) for slc in seed.bbox] for seed in seeds],
            dtype=np.int64,
        )
        hi = np.asarray(
            [[int(slc.stop or 0) for slc in seed.bbox] for seed in seeds],
            dtype=np.int64,
        )
    else:
        lo = np.empty((0, 3), dtype=np.int64)
        hi = np.empty((0, 3), dtype=np.int64)
    save_npz_atomic(
        path,
        primary_id=np.asarray(primary_id, dtype=np.uint64),
        guide_id=labels,
        guide_voxels=voxels,
        lo=lo,
        hi=hi,
        overlap_voxels=np.asarray([seed.overlap_voxels for seed in seeds], dtype=np.uint64),
        dominant_primary_fraction=np.asarray(
            [seed.dominant_primary_fraction for seed in seeds], dtype=np.float64
        ),
    )


def load_parent_seed_results(path: Path) -> list[_GuideSeed]:
    with np.load(path, allow_pickle=False) as data:
        primary_id = int(np.asarray(data["primary_id"]).item())
        labels = np.asarray(data["guide_id"], dtype=np.uint64)
        voxels = np.asarray(data["guide_voxels"], dtype=np.uint64)
        lo = np.asarray(data["lo"], dtype=np.int64)
        hi = np.asarray(data["hi"], dtype=np.int64)
        overlaps = np.asarray(data["overlap_voxels"], dtype=np.uint64)
        fractions = np.asarray(data["dominant_primary_fraction"], dtype=np.float64)
    return [
        _GuideSeed(
            guide_id=int(label),
            guide_voxels=int(voxel_count),
            bbox=tuple(slice(int(a), int(b)) for a, b in zip(lo_row, hi_row)),  # type: ignore[arg-type]
            primary_id=primary_id,
            overlap_voxels=int(overlap),
            dominant_primary_fraction=float(fraction),
        )
        for label, voxel_count, lo_row, hi_row, overlap, fraction in zip(
            labels, voxels, lo, hi, overlaps, fractions
        )
    ]


def parent_seed_result_path(result_dir: Path, primary_id: int) -> Path:
    return result_dir / f"{int(primary_id)}.npz"


def parent_seed_done_path(done_dir: Path, primary_id: int) -> Path:
    return done_dir / f"{int(primary_id)}.done"


def scan_parent_guides_worker(
    primary_path: str,
    guide_path: str,
    dataset_name: str,
    parent_record: tuple[int, int, tuple[int, int, int], tuple[int, int, int]],
    guide_records: dict[int, tuple[int, int, tuple[int, int, int], tuple[int, int, int]]],
    params: FusionParams,
    result_dir: str,
    done_dir: str,
    recompute: bool,
) -> int:
    parent = label_stats_from_record(parent_record)
    result_dir_path = Path(result_dir)
    done_dir_path = Path(done_dir)
    result_path = parent_seed_result_path(result_dir_path, parent.label)
    done_path = parent_seed_done_path(done_dir_path, parent.label)
    if done_path.exists() and result_path.exists() and not recompute:
        return 0

    guide_stats = {
        int(label): label_stats_from_record(record) for label, record in guide_records.items()
    }
    overlap_counts: dict[int, int] = defaultdict(int)
    with h5py.File(primary_path, "r") as f_primary, h5py.File(guide_path, "r") as f_guide:
        primary_ds = f_primary[dataset_name]
        guide_ds = f_guide[dataset_name]
        iterator = primary_ds.iter_chunks(sel=parent.bbox) if primary_ds.chunks else (parent.bbox,)
        for slc in iterator:
            primary_crop = np.asarray(primary_ds[slc])
            parent_mask = primary_crop == parent.label
            if not parent_mask.any():
                continue
            guide_crop = np.asarray(guide_ds[slc])
            labels, counts = np.unique(guide_crop[parent_mask], return_counts=True)
            for label_value, count_value in zip(labels, counts):
                label = int(label_value)
                if label == params.ignore_label or label not in guide_stats:
                    continue
                overlap_counts[label] += int(count_value)

    seeds: list[_GuideSeed] = []
    for label, overlap in sorted(overlap_counts.items()):
        guide_stat = guide_stats[label]
        fraction = overlap / float(guide_stat.voxels)
        if overlap < params.min_seed_overlap_voxels:
            continue
        if fraction < params.min_seed_guide_fraction:
            continue
        seeds.append(
            _GuideSeed(
                guide_id=guide_stat.label,
                guide_voxels=guide_stat.voxels,
                bbox=guide_stat.bbox,
                primary_id=parent.label,
                overlap_voxels=overlap,
                dominant_primary_fraction=float(fraction),
            )
        )

    save_parent_seed_results(result_path, parent.label, seeds)
    write_done(done_path)
    return 1


def scan_parent_guides_checkpointed(
    primary_path: Path,
    guide_path: Path,
    dataset_name: str,
    parent_stats: dict[int, _LabelStats],
    guide_stats: dict[int, _LabelStats],
    params: FusionParams,
    work_dir: Path,
    *,
    workers: int,
    recompute: bool,
    source: str,
) -> list[_GuideSeed]:
    result_dir = work_dir / "results"
    done_dir = work_dir / "done"
    result_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    parent_records = [label_stats_record(stat) for stat in parent_stats.values()]
    guide_records = {label: label_stats_record(stat) for label, stat in guide_stats.items()}
    pending = [
        record
        for record in parent_records
        if recompute
        or not (
            parent_seed_done_path(done_dir, int(record[0])).exists()
            and parent_seed_result_path(result_dir, int(record[0])).exists()
        )
    ]
    workers = max(1, min(int(workers), len(pending) or 1))
    print(
        f"{source}: parent-guide cache {work_dir}; "
        f"{len(parent_records) - len(pending)} done, {len(pending)} pending, "
        f"{workers} worker(s)"
    )

    if pending:
        if workers == 1:
            completed = 0
            for i, record in enumerate(pending, start=1):
                completed += scan_parent_guides_worker(
                    str(primary_path),
                    str(guide_path),
                    dataset_name,
                    record,
                    guide_records,
                    params,
                    str(result_dir),
                    str(done_dir),
                    recompute,
                )
                print(f"  {source}: parent-guide {i}/{len(pending)} completed={completed}")
        else:
            completed = 0
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        scan_parent_guides_worker,
                        str(primary_path),
                        str(guide_path),
                        dataset_name,
                        record,
                        guide_records,
                        params,
                        str(result_dir),
                        str(done_dir),
                        recompute,
                    )
                    for record in pending
                ]
                for i, future in enumerate(as_completed(futures), start=1):
                    completed += future.result()
                    print(f"  {source}: parent-guide {i}/{len(pending)} completed={completed}")

    seeds: list[_GuideSeed] = []
    for record in parent_records:
        primary_id = int(record[0])
        result_path = parent_seed_result_path(result_dir, primary_id)
        done_path = parent_seed_done_path(done_dir, primary_id)
        if done_path.exists() and result_path.exists():
            seeds.extend(load_parent_seed_results(result_path))
    return seeds


def scan_selected_primary_stats(
    primary_ds: h5py.Dataset,
    target_ids: Iterable[int],
    ignore_label: int,
) -> dict[int, _LabelStats]:
    ids = np.asarray(
        sorted({int(v) for v in target_ids if int(v) != ignore_label}), dtype=np.uint64
    )
    if ids.size == 0:
        return {}

    raw: RawStats = {}
    label_filter = ids.astype(np.uint64, copy=False)
    for slc in (
        primary_ds.iter_chunks()
        if primary_ds.chunks
        else (tuple(slice(0, s) for s in primary_ds.shape),)
    ):
        block = np.asarray(primary_ds[slc])
        update_raw_stats_from_block(
            raw,
            block,
            slc,
            ignore_label=ignore_label,
            label_filter=label_filter,
        )

    return label_stats_from_raw(raw)


def discover_candidates(
    primary_ds: h5py.Dataset,
    guide_ds: h5py.Dataset,
    guide_stats: dict[int, _LabelStats],
    params: FusionParams,
    *,
    source: str,
    target_primary_ids: set[int] | None,
    parent_stats_override: dict[int, _LabelStats] | None = None,
    primary_path: Path | None = None,
    guide_path: Path | None = None,
    dataset_name: str = "main",
    work_dir: Path | None = None,
    guide_parent_workers: int = 1,
    guide_parent_batch_size: int = 8,
    recompute_guide_parent: bool = False,
) -> tuple[list[FusionCandidate], list[dict[str, object]]]:
    large_guides = _select_large_guide_components(
        guide_stats,
        min_seed_voxels=params.min_seed_voxels,
        min_seed_axis_extent=params.min_seed_axis_extent,
        max_seed_axis_extent=params.max_seed_axis_extent,
    )
    print(f"{source}: retained {len(large_guides)} large guide components")
    if parent_stats_override is not None and target_primary_ids is not None:
        target_bboxes = [
            stat.bbox
            for label, stat in parent_stats_override.items()
            if label in target_primary_ids
        ]
        if target_bboxes:
            before = len(large_guides)
            large_guides = [
                guide
                for guide in large_guides
                if any(bboxes_intersect(guide.bbox, target_bbox) for target_bbox in target_bboxes)
            ]
            print(
                f"{source}: bbox-intersection filter kept {len(large_guides)}/"
                f"{before} guide components near target parents"
            )

    seed_groups: dict[int, list[_GuideSeed]] = defaultdict(list)
    seed_rows: list[dict[str, object]] = []
    if (
        parent_stats_override is not None
        and target_primary_ids is not None
        and work_dir is not None
        and primary_path is not None
        and guide_path is not None
    ):
        target_parent_stats = {
            label: stat
            for label, stat in parent_stats_override.items()
            if label in target_primary_ids
        }
        target_guide_stats = {stat.label: stat for stat in large_guides}
        seeds = scan_parent_guides_checkpointed(
            primary_path,
            guide_path,
            dataset_name,
            target_parent_stats,
            target_guide_stats,
            params,
            work_dir / "parent_guide_scan",
            workers=guide_parent_workers,
            recompute=recompute_guide_parent,
            source=source,
        )
    elif work_dir is not None and primary_path is not None and guide_path is not None:
        seeds = scan_guide_parents_checkpointed(
            primary_path,
            guide_path,
            dataset_name,
            large_guides,
            params,
            work_dir / "guide_parent_scan",
            workers=guide_parent_workers,
            batch_size=guide_parent_batch_size,
            recompute=recompute_guide_parent,
            source=source,
        )
    else:
        seeds = []
        for i, component in enumerate(large_guides, start=1):
            seed = find_guide_parent_h5(primary_ds, guide_ds, component, params)
            if seed is not None:
                seeds.append(seed)
            if i == 1 or i == len(large_guides) or i % 100 == 0:
                print(f"  {source}: guide-parent scan {i}/{len(large_guides)}")

    for seed in seeds:
        if seed is None:
            continue
        if seed.overlap_voxels < params.min_seed_overlap_voxels:
            continue
        if seed.dominant_primary_fraction < params.min_seed_guide_fraction:
            continue
        if target_primary_ids is not None and seed.primary_id not in target_primary_ids:
            continue
        seed_groups[seed.primary_id].append(seed)
        start, stop = bbox_to_str(seed.bbox)
        seed_rows.append(
            {
                "guide_id": seed.guide_id,
                "guide_voxels": seed.guide_voxels,
                "guide_bbox_start": start,
                "guide_bbox_stop": stop,
                "primary_id": seed.primary_id,
                "overlap_voxels": seed.overlap_voxels,
                "dominant_primary_fraction": f"{seed.dominant_primary_fraction:.6g}",
                "source": source,
            }
        )
    print(f"{source}: retained {len(seed_rows)} guide-parent overlap results")

    if parent_stats_override is None:
        parent_stats = scan_selected_primary_stats(
            primary_ds, seed_groups.keys(), params.ignore_label
        )
    else:
        parent_stats = parent_stats_override

    candidate_rows: list[dict[str, object]] = []
    candidates: list[FusionCandidate] = []
    for primary_id, seeds in sorted(seed_groups.items()):
        seeds = sorted(seeds, key=lambda seed: (-seed.overlap_voxels, seed.guide_id))
        parent = parent_stats.get(primary_id)
        parent_voxels = parent.voxels if parent is not None else 0
        accepted = (
            parent is not None
            and parent_voxels >= params.min_parent_voxels
            and len(seeds) >= params.min_seeds_in_parent
            and len(seeds) <= params.max_splits_per_parent
        )
        reason = ""
        if parent is None:
            reason = "parent_not_found"
        elif parent_voxels < params.min_parent_voxels:
            reason = "parent_too_small"
        elif len(seeds) < params.min_seeds_in_parent:
            reason = "too_few_guide_seeds"
        elif len(seeds) > params.max_splits_per_parent:
            reason = "too_many_guide_seeds"

        score = float(len(seeds) * 1_000_000 + sum(seed.overlap_voxels for seed in seeds))
        bbox = parent.bbox if parent is not None else _union_bboxes([seed.bbox for seed in seeds])
        start, stop = bbox_to_str(bbox)
        candidate_rows.append(
            {
                "primary_id": primary_id,
                "primary_voxels": parent_voxels,
                "bbox_start": start,
                "bbox_stop": stop,
                "retained_seed_count": len(seeds),
                "retained_seed_ids": ";".join(str(seed.guide_id) for seed in seeds),
                "retained_seed_overlaps": ";".join(str(seed.overlap_voxels) for seed in seeds),
                "dominant_primary_fraction": ";".join(
                    f"{seed.dominant_primary_fraction:.6g}" for seed in seeds
                ),
                "candidate_score": f"{score:.6g}",
                "decision": "candidate" if accepted else "reject",
                "reject_reason": reason,
                "source": source,
            }
        )
        if accepted and parent is not None:
            candidates.append(
                FusionCandidate(
                    primary_id=primary_id,
                    primary_voxels=parent_voxels,
                    primary_bbox=parent.bbox,
                    seeds=tuple(seeds),
                    score=score,
                    source=source,
                )
            )

    candidates.sort(key=lambda c: (-c.score, c.primary_id))
    if params.max_parents > 0:
        candidates = candidates[: params.max_parents]
    return candidates, candidate_rows


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def _dataset_copy_kwargs(src: h5py.Dataset) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if src.chunks is not None:
        kwargs["chunks"] = src.chunks
    if src.compression is not None:
        kwargs["compression"] = src.compression
        kwargs["compression_opts"] = src.compression_opts
    if src.shuffle:
        kwargs["shuffle"] = src.shuffle
    if src.fletcher32:
        kwargs["fletcher32"] = src.fletcher32
    return kwargs


def _copy_dataset_cast(
    src: h5py.Dataset,
    dst_group: h5py.Group,
    name: str,
    dtype: np.dtype,
) -> None:
    dst = dst_group.create_dataset(
        name,
        shape=src.shape,
        dtype=dtype,
        **_dataset_copy_kwargs(src),
    )
    _copy_attrs(src.attrs, dst.attrs)
    iterator = src.iter_chunks() if src.chunks else (tuple(slice(0, s) for s in src.shape),)
    for slc in iterator:
        dst[slc] = np.asarray(src[slc], dtype=dtype)


def _copy_group_casting_dataset(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    *,
    dataset_path: str,
    current_path: str = "",
) -> None:
    _copy_attrs(src_group.attrs, dst_group.attrs)
    for key, item in src_group.items():
        item_path = f"{current_path}/{key}" if current_path else key
        if isinstance(item, h5py.Group):
            child = dst_group.create_group(key)
            _copy_group_casting_dataset(
                item,
                child,
                dataset_path=dataset_path,
                current_path=item_path,
            )
        elif isinstance(item, h5py.Dataset):
            dtype = np.dtype(np.uint32) if item_path == dataset_path else item.dtype
            _copy_dataset_cast(item, dst_group, key, dtype)


def copy_primary_to_output(
    primary_path: Path,
    output_path: Path,
    dry_run: bool,
    *,
    dataset: str,
) -> None:
    if dry_run:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    dataset_path = dataset.strip("/")
    target_dtype = np.dtype(np.uint32)
    with h5py.File(primary_path, "r") as src:
        src_dtype = np.dtype(src[dataset].dtype)
    if src_dtype == target_dtype:
        shutil.copy2(primary_path, output_path)
        return
    with h5py.File(primary_path, "r") as src, h5py.File(output_path, "w") as dst:
        src_ds = src[dataset]
        print(f"Creating uint32 output copy for {dataset}: " f"{src_ds.dtype} -> {target_dtype}")
        _copy_group_casting_dataset(src, dst, dataset_path=dataset_path)


def assign_from_markers(
    parent_mask: np.ndarray, markers: np.ndarray, assignment: str
) -> np.ndarray:
    if assignment == "nearest":
        return _assign_from_markers(parent_mask, markers)
    if assignment != "watershed":
        raise ValueError(f"Unknown assignment mode: {assignment}")
    if not np.any(markers):
        return np.zeros_like(markers, dtype=np.uint64)
    distance = ndimage.distance_transform_edt(parent_mask)
    assigned = watershed(-distance, markers.astype(np.int64, copy=False), mask=parent_mask)
    return assigned.astype(np.uint64, copy=False)


def iter_dataset_chunks(
    dataset: h5py.Dataset, bbox: tuple[slice, slice, slice]
) -> Iterable[tuple[slice, slice, slice]]:
    if dataset.chunks:
        yield from dataset.iter_chunks(sel=bbox)
        return
    yield bbox


def slice_origin(slc: tuple[slice, slice, slice]) -> np.ndarray:
    return np.asarray([int(axis.start or 0) for axis in slc], dtype=np.int64)


def append_marker_samples(
    existing: np.ndarray | None,
    coords: np.ndarray,
    max_samples: int,
) -> np.ndarray:
    if coords.size == 0:
        return existing if existing is not None else np.empty((0, 3), dtype=np.int64)
    if max_samples > 0 and coords.shape[0] > max_samples:
        step = int(np.ceil(coords.shape[0] / float(max_samples)))
        coords = coords[::step][:max_samples]
    combined = coords if existing is None else np.concatenate([existing, coords], axis=0)
    if max_samples > 0 and combined.shape[0] > max_samples:
        keep = np.linspace(0, combined.shape[0] - 1, max_samples, dtype=np.int64)
        combined = combined[keep]
    return combined.astype(np.int64, copy=False)


def collect_sampled_markers(
    primary_ds: h5py.Dataset,
    guide_ds: h5py.Dataset,
    bbox: tuple[slice, slice, slice],
    candidate: FusionCandidate,
    params: FusionParams,
) -> tuple[cKDTree | None, np.ndarray, list[int], dict[int, int]]:
    samples_by_seed: dict[int, np.ndarray] = {}
    marker_voxels: dict[int, int] = defaultdict(int)
    seed_ids = [int(seed.guide_id) for seed in candidate.seeds]

    for slc in iter_dataset_chunks(primary_ds, bbox):
        parent_crop = np.asarray(primary_ds[slc])
        parent_mask = parent_crop == candidate.primary_id
        if not parent_mask.any():
            continue
        guide_crop = np.asarray(guide_ds[slc])
        origin = slice_origin(slc)
        for seed_id in seed_ids:
            marker_mask = parent_mask & (guide_crop == seed_id)
            count = int(marker_mask.sum())
            if count == 0:
                continue
            marker_voxels[seed_id] += count
            coords = np.column_stack(np.nonzero(marker_mask)).astype(np.int64, copy=False)
            coords += origin
            samples_by_seed[seed_id] = append_marker_samples(
                samples_by_seed.get(seed_id),
                coords,
                params.max_marker_samples_per_seed,
            )

    points: list[np.ndarray] = []
    point_marker_ids: list[np.ndarray] = []
    retained_guide_ids: list[int] = []
    for marker_id, seed_id in enumerate(seed_ids, start=1):
        samples = samples_by_seed.get(seed_id)
        if samples is None or samples.size == 0:
            continue
        if marker_voxels[seed_id] < params.min_marker_voxels:
            continue
        retained_guide_ids.append(seed_id)
        points.append(samples.astype(np.float32, copy=False))
        point_marker_ids.append(np.full(samples.shape[0], marker_id, dtype=np.uint16))

    if len(retained_guide_ids) < 2 or not points:
        return None, np.empty(0, dtype=np.uint16), retained_guide_ids, marker_voxels

    all_points = np.concatenate(points, axis=0)
    all_marker_ids = np.concatenate(point_marker_ids, axis=0)
    return cKDTree(all_points), all_marker_ids, retained_guide_ids, marker_voxels


def query_marker_assignments(
    tree: cKDTree,
    point_marker_ids: np.ndarray,
    slc: tuple[slice, slice, slice],
    parent_mask: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    coords = np.nonzero(parent_mask)
    if coords[0].size == 0:
        return coords, np.empty(0, dtype=np.uint16)
    points = np.column_stack(coords).astype(np.float32, copy=False)
    points += slice_origin(slc).astype(np.float32, copy=False)
    _distance, nearest = tree.query(points, workers=1)
    return coords, point_marker_ids[np.asarray(nearest, dtype=np.int64)]


def linear_indices_from_coords(
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    slc: tuple[slice, slice, slice],
    shape: Sequence[int],
) -> np.ndarray:
    return _linear_indices_from_coords(
        coords,
        origin=[int(axis.start or 0) for axis in slc],
        shape=shape,
    )


def sort_aligned(indices: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if indices.size == 0:
        return indices.astype(np.uint64, copy=False), values
    order = np.argsort(indices, kind="mergesort")
    return indices[order].astype(np.uint64, copy=False), values[order]


def search_sorted_members(
    sorted_values: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return search_sorted_indices(sorted_values, query)


def collect_sparse_marker_graph(
    primary_ds: h5py.Dataset,
    guide_ds: h5py.Dataset,
    bbox: tuple[slice, slice, slice],
    candidate: FusionCandidate,
    params: FusionParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], dict[int, int]]:
    retained = [
        seed
        for seed in candidate.seeds
        if seed.overlap_voxels >= params.min_marker_voxels
        and seed.overlap_voxels >= params.min_seed_overlap_voxels
    ]
    retained_guide_ids = [int(seed.guide_id) for seed in retained]
    seed_to_marker = {int(seed.guide_id): idx for idx, seed in enumerate(retained, start=1)}
    if len(retained_guide_ids) < 2:
        empty_u64 = np.empty(0, dtype=np.uint64)
        empty_u16 = np.empty(0, dtype=np.uint16)
        return empty_u64, empty_u16, empty_u64, empty_u16, retained_guide_ids, {}

    marker_index_parts: list[np.ndarray] = []
    marker_label_parts: list[np.ndarray] = []
    residual_index_parts: list[np.ndarray] = []
    marker_voxels: dict[int, int] = defaultdict(int)

    for slc in iter_dataset_chunks(primary_ds, bbox):
        parent_crop = np.asarray(primary_ds[slc])
        parent_mask = parent_crop == candidate.primary_id
        if not parent_mask.any():
            continue
        guide_crop = np.asarray(guide_ds[slc])
        marker_local = np.zeros(parent_mask.shape, dtype=np.uint16)
        for guide_id, marker_id in seed_to_marker.items():
            guide_mask = parent_mask & (guide_crop == guide_id)
            count = int(guide_mask.sum())
            if count == 0:
                continue
            marker_voxels[guide_id] += count
            marker_local[guide_mask] = np.uint16(marker_id)

        marker_mask = marker_local != 0
        if marker_mask.any():
            coords = np.nonzero(marker_mask)
            marker_index_parts.append(linear_indices_from_coords(coords, slc, primary_ds.shape))
            marker_label_parts.append(marker_local[coords].astype(np.uint16, copy=True))

        residual_mask = parent_mask & ~marker_mask
        if residual_mask.any():
            residual_index_parts.append(
                linear_indices_from_coords(np.nonzero(residual_mask), slc, primary_ds.shape)
            )

    marker_indices = (
        np.concatenate(marker_index_parts).astype(np.uint64, copy=False)
        if marker_index_parts
        else np.empty(0, dtype=np.uint64)
    )
    marker_labels = (
        np.concatenate(marker_label_parts).astype(np.uint16, copy=False)
        if marker_label_parts
        else np.empty(0, dtype=np.uint16)
    )
    residual_indices = (
        np.concatenate(residual_index_parts).astype(np.uint64, copy=False)
        if residual_index_parts
        else np.empty(0, dtype=np.uint64)
    )

    marker_indices, marker_labels = sort_aligned(marker_indices, marker_labels)
    residual_indices = np.sort(residual_indices, kind="mergesort")
    residual_labels = np.zeros(residual_indices.shape, dtype=np.uint16)
    return (
        marker_indices,
        marker_labels,
        residual_indices,
        residual_labels,
        retained_guide_ids,
        marker_voxels,
    )


def geodesic_grow_residual(
    residual_indices: np.ndarray,
    residual_labels: np.ndarray,
    marker_indices: np.ndarray,
    marker_labels: np.ndarray,
    shape: Sequence[int],
    *,
    source: str,
    primary_id: int,
) -> int:
    def log_progress(iteration: int, assigned: int, total: int, frontier: int) -> None:
        if iteration == 0:
            print(
                f"  {source} parent={primary_id}: geodesic init assigned "
                f"{assigned}/{total} residual voxels"
            )
        elif iteration == 1 or iteration % 50 == 0:
            print(
                f"  {source} parent={primary_id}: geodesic iter {iteration}, "
                f"assigned {assigned}/{total}, frontier={frontier}"
            )

    result = sparse_geodesic_grow_labels(
        marker_indices,
        marker_labels,
        residual_indices,
        shape=shape,
        progress=log_progress,
    )
    residual_labels[:] = result.labels
    return result.iterations


def sparse_geodesic_split_candidate(
    primary_ds: h5py.Dataset,
    guide_ds: h5py.Dataset,
    candidate: FusionCandidate,
    params: FusionParams,
    next_label: int,
    idx: int,
    total: int,
) -> tuple[dict[str, object], int]:
    bbox = _pad_bbox(candidate.primary_bbox, primary_ds.shape, params.bbox_pad)
    start, stop = bbox_to_str(bbox)
    decision: dict[str, object] = {
        "primary_id": candidate.primary_id,
        "accepted": False,
        "reject_reason": "",
        "bbox_start": start,
        "bbox_stop": stop,
        "guide_seed_ids": ";".join(str(seed.guide_id) for seed in candidate.seeds),
        "output_child_ids": "",
        "child_voxels": "",
        "marker_voxels": "",
        "unassigned_voxels": "",
        "geodesic_iterations": "",
        "changed_voxels": 0,
        "assignment": params.assignment,
    }

    print(
        f"  split {idx}/{total} parent={candidate.primary_id}: "
        f"collecting sparse geodesic markers"
    )
    (
        marker_indices,
        marker_labels,
        residual_indices,
        residual_labels,
        retained_guide_ids,
        marker_voxels,
    ) = collect_sparse_marker_graph(primary_ds, guide_ds, bbox, candidate, params)
    if len(retained_guide_ids) < 2 or marker_indices.size == 0:
        decision["reject_reason"] = "too_few_markers_after_filter"
        decision["guide_seed_ids"] = ";".join(str(v) for v in retained_guide_ids)
        return decision, next_label

    print(
        f"  split {idx}/{total} parent={candidate.primary_id}: "
        f"marker_voxels={marker_indices.size}, residual_voxels={residual_indices.size}"
    )
    iterations = geodesic_grow_residual(
        residual_indices,
        residual_labels,
        marker_indices,
        marker_labels,
        primary_ds.shape,
        source=f"split {idx}/{total}",
        primary_id=candidate.primary_id,
    )
    del marker_indices, marker_labels

    marker_count_array = np.zeros(len(retained_guide_ids) + 1, dtype=np.uint64)
    for seed_idx, guide_id in enumerate(retained_guide_ids, start=1):
        marker_count_array[seed_idx] = np.uint64(marker_voxels.get(int(guide_id), 0))
    if residual_labels.size:
        residual_counts = np.bincount(
            residual_labels.astype(np.int64, copy=False),
            minlength=len(retained_guide_ids) + 1,
        ).astype(np.uint64, copy=False)
        marker_count_array[: residual_counts.size] += residual_counts

    marker_ids = np.arange(1, len(retained_guide_ids) + 1, dtype=np.uint16)
    child_counts = marker_count_array[1:]
    valid = child_counts >= np.uint64(params.min_child_voxels)
    marker_ids = marker_ids[valid]
    child_counts = child_counts[valid]
    if marker_ids.size < 2:
        decision["reject_reason"] = "too_few_children_after_assignment"
        decision["guide_seed_ids"] = ";".join(str(v) for v in retained_guide_ids)
        decision["marker_voxels"] = ";".join(
            str(int(marker_voxels.get(int(v), 0))) for v in retained_guide_ids
        )
        decision["unassigned_voxels"] = int(np.sum(residual_labels == 0))
        decision["geodesic_iterations"] = iterations
        return decision, next_label

    order = np.lexsort((marker_ids, -child_counts))
    marker_ids = marker_ids[order]
    child_counts = child_counts[order]

    marker_to_output: dict[int, int] = {}
    output_ids: list[int] = []
    for child_idx, marker_id in enumerate(marker_ids):
        if child_idx == 0:
            output_id = candidate.primary_id
        else:
            if next_label > np.iinfo(np.uint32).max:
                raise ValueError("new label allocation exceeded uint32 range")
            output_id = next_label
            next_label += 1
        marker_to_output[int(marker_id)] = output_id
        output_ids.append(output_id)

    seed_to_marker = {
        int(guide_id): idx for idx, guide_id in enumerate(retained_guide_ids, start=1)
    }
    changed = 0
    for slc in iter_dataset_chunks(primary_ds, bbox):
        parent_crop = np.asarray(primary_ds[slc])
        parent_mask = parent_crop == candidate.primary_id
        if not parent_mask.any():
            continue
        guide_crop = np.asarray(guide_ds[slc])
        coords = np.nonzero(parent_mask)
        replacement = np.full(coords[0].shape, candidate.primary_id, dtype=parent_crop.dtype)
        marker_for_voxel = np.zeros(coords[0].shape, dtype=np.uint16)
        guide_values = guide_crop[coords]
        for guide_id, marker_id in seed_to_marker.items():
            marker_for_voxel[guide_values == guide_id] = np.uint16(marker_id)

        residual_mask = marker_for_voxel == 0
        if np.any(residual_mask) and residual_indices.size:
            parent_lin = linear_indices_from_coords(coords, slc, primary_ds.shape)
            residual_positions = np.flatnonzero(residual_mask)
            pos, valid_residual = search_sorted_members(
                residual_indices,
                parent_lin[residual_positions],
            )
            if np.any(valid_residual):
                marker_for_voxel[residual_positions[valid_residual]] = residual_labels[
                    pos[valid_residual]
                ]

        for marker_id, output_id in marker_to_output.items():
            replacement[marker_for_voxel == marker_id] = np.asarray(
                output_id, dtype=parent_crop.dtype
            )
        changed += int(np.sum(replacement != candidate.primary_id))
        if np.any(replacement != candidate.primary_id):
            parent_crop[coords] = replacement
            primary_ds[slc] = parent_crop

    decision.update(
        {
            "accepted": True,
            "guide_seed_ids": ";".join(str(v) for v in retained_guide_ids),
            "output_child_ids": ";".join(str(v) for v in output_ids),
            "child_voxels": ";".join(str(int(v)) for v in child_counts),
            "marker_voxels": ";".join(
                str(int(marker_voxels.get(int(v), 0))) for v in retained_guide_ids
            ),
            "unassigned_voxels": int(np.sum(residual_labels == 0)),
            "geodesic_iterations": iterations,
            "changed_voxels": changed,
        }
    )
    print(
        f"  split {idx}/{total} parent={candidate.primary_id} "
        f"children={output_ids} changed={changed}"
    )
    return decision, next_label


def sampled_nearest_split_candidate(
    primary_ds: h5py.Dataset,
    guide_ds: h5py.Dataset,
    candidate: FusionCandidate,
    params: FusionParams,
    next_label: int,
    idx: int,
    total: int,
) -> tuple[dict[str, object], int]:
    bbox = _pad_bbox(candidate.primary_bbox, primary_ds.shape, params.bbox_pad)
    start, stop = bbox_to_str(bbox)
    decision: dict[str, object] = {
        "primary_id": candidate.primary_id,
        "accepted": False,
        "reject_reason": "",
        "bbox_start": start,
        "bbox_stop": stop,
        "guide_seed_ids": ";".join(str(seed.guide_id) for seed in candidate.seeds),
        "output_child_ids": "",
        "child_voxels": "",
        "changed_voxels": 0,
        "assignment": params.assignment,
    }

    print(
        f"  split {idx}/{total} parent={candidate.primary_id}: " f"collecting sampled guide markers"
    )
    tree, point_marker_ids, retained_guide_ids, marker_voxels = collect_sampled_markers(
        primary_ds,
        guide_ds,
        bbox,
        candidate,
        params,
    )
    if tree is None:
        decision["reject_reason"] = "too_few_markers_after_filter"
        decision["guide_seed_ids"] = ";".join(str(v) for v in retained_guide_ids)
        return decision, next_label

    print(
        f"  split {idx}/{total} parent={candidate.primary_id}: "
        f"retained {len(retained_guide_ids)} markers, "
        f"samples={int(point_marker_ids.size)}; counting assignments"
    )
    child_count_map: dict[int, int] = defaultdict(int)
    parent_voxels_seen = 0
    for slc in iter_dataset_chunks(primary_ds, bbox):
        parent_crop = np.asarray(primary_ds[slc])
        parent_mask = parent_crop == candidate.primary_id
        if not parent_mask.any():
            continue
        _coords, assigned = query_marker_assignments(tree, point_marker_ids, slc, parent_mask)
        parent_voxels_seen += int(assigned.size)
        marker_ids, counts = np.unique(assigned, return_counts=True)
        for marker_id, count in zip(marker_ids, counts):
            child_count_map[int(marker_id)] += int(count)

    if parent_voxels_seen == 0:
        decision["reject_reason"] = "parent_not_found"
        return decision, next_label

    marker_ids = np.asarray(sorted(child_count_map), dtype=np.uint16)
    child_counts = np.asarray([child_count_map[int(marker_id)] for marker_id in marker_ids])
    valid = child_counts >= params.min_child_voxels
    marker_ids = marker_ids[valid]
    child_counts = child_counts[valid]
    if marker_ids.size < 2:
        decision["reject_reason"] = "too_few_children_after_assignment"
        decision["guide_seed_ids"] = ";".join(str(v) for v in retained_guide_ids)
        return decision, next_label

    print(
        f"  split {idx}/{total} parent={candidate.primary_id}: "
        f"counted {parent_voxels_seen} parent voxels; writing output chunks"
    )
    order = np.lexsort((marker_ids, -child_counts))
    marker_ids = marker_ids[order]
    child_counts = child_counts[order]

    output_ids: list[int] = []
    marker_to_output: dict[int, int] = {}
    for child_idx, marker_id in enumerate(marker_ids):
        if child_idx == 0:
            output_id = candidate.primary_id
        else:
            if next_label > np.iinfo(np.uint32).max:
                raise ValueError("new label allocation exceeded uint32 range")
            output_id = next_label
            next_label += 1
        output_ids.append(output_id)
        marker_to_output[int(marker_id)] = output_id

    changed = 0
    fallback_id = candidate.primary_id
    for slc in iter_dataset_chunks(primary_ds, bbox):
        parent_crop = np.asarray(primary_ds[slc])
        parent_mask = parent_crop == candidate.primary_id
        if not parent_mask.any():
            continue
        coords, assigned = query_marker_assignments(tree, point_marker_ids, slc, parent_mask)
        replacement = np.full(assigned.shape, fallback_id, dtype=parent_crop.dtype)
        for marker_id, output_id in marker_to_output.items():
            replacement[assigned == marker_id] = np.asarray(output_id, dtype=parent_crop.dtype)
        changed += int(np.sum(replacement != fallback_id))
        if np.any(replacement != fallback_id):
            parent_crop[coords] = replacement
            primary_ds[slc] = parent_crop

    retained_marker_voxels = [
        marker_voxels.get(int(seed_id), 0)
        for seed_id in retained_guide_ids
        if marker_voxels.get(int(seed_id), 0) >= params.min_marker_voxels
    ]
    decision.update(
        {
            "accepted": True,
            "guide_seed_ids": ";".join(str(v) for v in retained_guide_ids),
            "output_child_ids": ";".join(str(v) for v in output_ids),
            "child_voxels": ";".join(str(int(v)) for v in child_counts),
            "changed_voxels": changed,
            "marker_voxels": ";".join(str(int(v)) for v in retained_marker_voxels),
        }
    )
    print(
        f"  split {idx}/{total} parent={candidate.primary_id} "
        f"children={output_ids} changed={changed}"
    )
    return decision, next_label


def apply_fusion(
    primary_path: Path,
    guide_path: Path,
    output_path: Path,
    candidates: Sequence[FusionCandidate],
    params: FusionParams,
    *,
    dataset: str,
    dry_run: bool,
) -> list[dict[str, object]]:
    copy_primary_to_output(primary_path, output_path, dry_run, dataset=dataset)
    decisions: list[dict[str, object]] = []
    next_label = int(params.new_label_start)
    if next_label <= 0 or next_label > np.iinfo(np.uint32).max:
        raise ValueError(f"new_label_start={next_label} is outside uint32 range")

    if dry_run:
        with h5py.File(primary_path, "r") as f_primary:
            shape = f_primary[dataset].shape
        for candidate in candidates:
            bbox = _pad_bbox(candidate.primary_bbox, shape, params.bbox_pad)
            start, stop = bbox_to_str(bbox)
            decisions.append(
                {
                    "primary_id": candidate.primary_id,
                    "accepted": False,
                    "reject_reason": "dry_run",
                    "bbox_start": start,
                    "bbox_stop": stop,
                    "guide_seed_ids": ";".join(str(seed.guide_id) for seed in candidate.seeds),
                    "output_child_ids": "",
                    "child_voxels": "",
                    "marker_voxels": "",
                    "unassigned_voxels": "",
                    "geodesic_iterations": "",
                    "changed_voxels": 0,
                    "assignment": params.assignment,
                }
            )
        print(f"dry-run: skipped split assignment for {len(candidates)} candidates")
        return decisions

    primary_open_mode = "r" if dry_run else "r+"
    primary_target = primary_path if dry_run else output_path
    with h5py.File(primary_target, primary_open_mode) as f_primary, h5py.File(
        guide_path, "r"
    ) as f_guide:
        primary_ds = f_primary[dataset]
        guide_ds = f_guide[dataset]
        if params.assignment == "sparse_geodesic":
            for idx, candidate in enumerate(candidates, start=1):
                decision, next_label = sparse_geodesic_split_candidate(
                    primary_ds,
                    guide_ds,
                    candidate,
                    params,
                    next_label,
                    idx,
                    len(candidates),
                )
                decisions.append(decision)
            return decisions

        if params.assignment == "sampled_nearest":
            for idx, candidate in enumerate(candidates, start=1):
                decision, next_label = sampled_nearest_split_candidate(
                    primary_ds,
                    guide_ds,
                    candidate,
                    params,
                    next_label,
                    idx,
                    len(candidates),
                )
                decisions.append(decision)
            return decisions

        for idx, candidate in enumerate(candidates, start=1):
            bbox = _pad_bbox(candidate.primary_bbox, primary_ds.shape, params.bbox_pad)
            start, stop = bbox_to_str(bbox)
            decision: dict[str, object] = {
                "primary_id": candidate.primary_id,
                "accepted": False,
                "reject_reason": "",
                "bbox_start": start,
                "bbox_stop": stop,
                "guide_seed_ids": ";".join(str(seed.guide_id) for seed in candidate.seeds),
                "output_child_ids": "",
                "child_voxels": "",
                "marker_voxels": "",
                "unassigned_voxels": "",
                "geodesic_iterations": "",
                "changed_voxels": 0,
                "assignment": params.assignment,
            }

            parent_crop = np.asarray(primary_ds[bbox])
            guide_crop = np.asarray(guide_ds[bbox])
            parent_mask = parent_crop == candidate.primary_id
            if not parent_mask.any():
                decision["reject_reason"] = "parent_not_found"
                decisions.append(decision)
                continue

            markers, retained_guide_ids = _prepare_markers(
                parent_mask,
                guide_crop,
                candidate.seeds,
                erosion_iterations=params.erosion_iterations,
                min_marker_voxels=params.min_marker_voxels,
            )
            if len(retained_guide_ids) < 2:
                decision["reject_reason"] = "too_few_markers_after_filter"
                decisions.append(decision)
                continue

            assigned = assign_from_markers(parent_mask, markers, params.assignment)
            marker_ids, child_counts = np.unique(assigned[assigned != 0], return_counts=True)
            valid = child_counts >= params.min_child_voxels
            marker_ids = marker_ids[valid]
            child_counts = child_counts[valid]
            if marker_ids.size < 2:
                decision["reject_reason"] = "too_few_children_after_assignment"
                decisions.append(decision)
                continue

            order = np.lexsort((marker_ids, -child_counts))
            marker_ids = marker_ids[order]
            child_counts = child_counts[order]

            output_ids: list[int] = []
            for child_idx, _marker_id in enumerate(marker_ids):
                if child_idx == 0:
                    output_ids.append(candidate.primary_id)
                else:
                    if next_label > np.iinfo(np.uint32).max:
                        raise ValueError("new label allocation exceeded uint32 range")
                    output_ids.append(next_label)
                    next_label += 1

            updated = parent_crop.copy()
            for marker_id, output_id in zip(marker_ids, output_ids):
                updated[parent_mask & (assigned == marker_id)] = np.uint32(output_id)

            changed = int(np.sum((updated != parent_crop) & parent_mask))
            if not dry_run:
                primary_ds[bbox] = updated

            decision.update(
                {
                    "accepted": True,
                    "guide_seed_ids": ";".join(str(v) for v in retained_guide_ids),
                    "output_child_ids": ";".join(str(v) for v in output_ids),
                    "child_voxels": ";".join(str(int(v)) for v in child_counts),
                    "marker_voxels": "",
                    "changed_voxels": changed,
                }
            )
            decisions.append(decision)
            print(
                f"  split {idx}/{len(candidates)} parent={candidate.primary_id} "
                f"children={output_ids} changed={changed}"
            )
    return decisions


def evaluate_nerl(path: Path, args: argparse.Namespace, out_npz: Path) -> None:
    print(f"Evaluating NERL for {path}; this loads the full fused volume into memory.")
    with h5py.File(path, "r") as f:
        seg = f[args.dataset][...]
    result = compute_nerl_score_details(
        seg,
        args.skeleton,
        graph_options=NerlGraphOptions(
            skeleton_id_attribute=args.skeleton_id_key,
            skeleton_position_attribute=args.skeleton_position_key,
            skeleton_position_order=args.skeleton_coord_order,
            prediction_position_order=args.dataset_axis_order,
        ),
        chunk_num=args.nerl_chunk_num,
        num_workers=args.nerl_num_workers,
    )
    print(
        f"  NERL={result.nerl:.6f} pred_erl={result.pred_erl:.6f} "
        f"gt_erl={result.gt_erl:.6f} skeletons={result.num_skeletons}"
    )
    np.savez_compressed(
        out_npz,
        gt_segment_id=np.asarray(result.graph.skeleton_id),
        erl=result.per_gt_erl,
    )


def params_from_args(args: argparse.Namespace) -> FusionParams:
    return FusionParams(
        min_parent_voxels=args.min_parent_voxels,
        min_seed_voxels=args.min_seed_voxels,
        min_seed_axis_extent=args.min_seed_axis_extent,
        max_seed_axis_extent=args.max_seed_axis_extent,
        min_seed_overlap_voxels=args.min_seed_overlap_voxels,
        min_seed_guide_fraction=args.min_seed_guide_fraction,
        min_seeds_in_parent=args.min_seeds_in_parent,
        max_splits_per_parent=args.max_splits_per_parent,
        max_parents=args.max_parents,
        bbox_pad=tuple(int(v) for v in args.bbox_pad),
        erosion_iterations=args.erosion_iterations,
        min_marker_voxels=args.min_marker_voxels,
        min_child_voxels=args.min_child_voxels,
        max_marker_samples_per_seed=args.max_marker_samples_per_seed,
        new_label_start=args.new_label_start,
        ignore_label=args.ignore_label,
        assignment=args.assignment,
    )


def run_one(
    name: str,
    primary_path: Path,
    guide_path: Path,
    out_dir: Path,
    dataset: str,
    params: FusionParams,
    *,
    target_primary_ids: set[int] | None,
    dry_run: bool,
    evaluate: bool,
    args: argparse.Namespace,
    guide_stats: dict[int, _LabelStats],
    parent_stats: dict[int, _LabelStats] | None = None,
) -> None:
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"decoded_x1_ch0-1-2_affinity_cc_numba-0-0.66_{name}.h5"

    with h5py.File(primary_path, "r") as f_primary, h5py.File(guide_path, "r") as f_guide:
        primary_ds = f_primary[dataset]
        guide_ds = f_guide[dataset]
        candidates, candidate_rows = discover_candidates(
            primary_ds,
            guide_ds,
            guide_stats,
            params,
            source=name,
            target_primary_ids=target_primary_ids,
            parent_stats_override=parent_stats,
            primary_path=primary_path,
            guide_path=guide_path,
            dataset_name=dataset,
            work_dir=run_dir,
            guide_parent_workers=args.guide_parent_workers,
            guide_parent_batch_size=args.guide_parent_batch_size,
            recompute_guide_parent=args.recompute_guide_parent,
        )

    _write_csv(run_dir / "guided_split_candidates.csv", candidate_rows, candidate_fieldnames())
    print(f"{name}: {len(candidates)} accepted candidates")
    decisions = apply_fusion(
        primary_path,
        guide_path,
        output_path,
        candidates,
        params,
        dataset=dataset,
        dry_run=dry_run,
    )
    _write_csv(run_dir / "guided_split_decisions.csv", decisions, decision_fieldnames())

    if evaluate and not dry_run:
        evaluate_nerl(output_path, args, run_dir / f"eval_{output_path.stem}_nerl_per_gt_erl.npz")


def main() -> None:
    args = parse_args()
    params = params_from_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"primary: {args.primary}")
    print(f"guide:   {args.guide}")
    print(f"out:     {args.out_dir}")

    if args.mode in {"oracle", "both"}:
        oracle_dir = args.out_dir / "oracle"
        oracle_selection = (
            oracle_targets_from_gt_sample_csv(args, oracle_dir)
            if args.oracle_sample_source == "csv"
            else None
        )
        if oracle_selection is None:
            with h5py.File(args.primary, "r") as f_primary, h5py.File(args.guide, "r") as f_guide:
                oracle_selection = oracle_targets_from_skeleton(
                    f_primary[args.dataset],
                    f_guide[args.dataset],
                    args,
                    oracle_dir,
                )
        print(f"oracle target primary IDs: {sorted(oracle_selection.primary_ids)}")
        print(f"oracle guide seed IDs: {sorted(oracle_selection.guide_ids)}")

        oracle_primary_stats = load_or_compute_selected_label_stats(
            args.primary,
            args.dataset,
            selected_bbox_cache_path(oracle_dir, args.primary, args.dataset, "primary"),
            oracle_selection.primary_ids,
            oracle_selection.primary_seed_coords,
            ignore_label=args.ignore_label,
            recompute=args.recompute_guide_bboxes,
            bbox_axis=args.bbox_axis,
            bbox_workers=args.bbox_workers,
            bbox_task_chunks=args.bbox_task_chunks,
            source_name="oracle-primary",
        )
        if args.oracle_guide_source == "global":
            oracle_guide_stats = load_or_compute_guide_stats(args)
        else:
            oracle_guide_stats = load_or_compute_selected_label_stats(
                args.guide,
                args.dataset,
                selected_bbox_cache_path(oracle_dir, args.guide, args.dataset, "guide"),
                oracle_selection.guide_ids,
                oracle_selection.guide_seed_coords,
                ignore_label=args.ignore_label,
                recompute=args.recompute_guide_bboxes,
                bbox_axis=args.bbox_axis,
                bbox_workers=args.bbox_workers,
                bbox_task_chunks=args.bbox_task_chunks,
                source_name="oracle-guide",
            )
        print(
            f"oracle bbox stats: primary={len(oracle_primary_stats)} "
            f"guide={len(oracle_guide_stats)}"
        )
        run_one(
            "oracle",
            args.primary,
            args.guide,
            args.out_dir,
            args.dataset,
            params,
            target_primary_ids=oracle_selection.primary_ids,
            dry_run=args.dry_run,
            evaluate=args.evaluate,
            args=args,
            guide_stats=oracle_guide_stats,
            parent_stats=oracle_primary_stats,
        )

    if args.mode in {"auto", "both"}:
        guide_stats = load_or_compute_guide_stats(args)
        print(f"Guide labels with bbox stats: {len(guide_stats)}")
        run_one(
            "auto",
            args.primary,
            args.guide,
            args.out_dir,
            args.dataset,
            params,
            target_primary_ids=None,
            dry_run=args.dry_run,
            evaluate=args.evaluate,
            args=args,
            guide_stats=guide_stats,
        )

    print("done")


if __name__ == "__main__":
    main()
