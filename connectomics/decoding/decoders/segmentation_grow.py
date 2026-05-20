"""Marker-based segmentation growth utilities and decoder."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import h5py
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from ..utils import cast2dtype

__all__ = [
    "affinity_foreground_score",
    "binary_geodesic_grow",
    "linear_indices_from_coords",
    "search_sorted_indices",
    "SparseGeodesicGrowResult",
    "sparse_geodesic_grow_labels",
    "watershed_grow_labels",
    "grow_segmentation_from_affinity",
    "segmentation_grow",
]

logger = logging.getLogger(__name__)

ChannelReduction = Literal["max", "min", "mean"]
SparseGrowProgress = Callable[[int, int, int, int], None]


@dataclass(frozen=True)
class SparseGeodesicGrowResult:
    labels: np.ndarray
    iterations: int
    initial_assigned: int
    final_assigned: int


def _as_3d_segmentation(seg: np.ndarray) -> np.ndarray:
    arr = np.asarray(seg)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(
            "segmentation_grow expects a 3D segmentation or singleton-channel "
            f"4D segmentation, got shape {arr.shape}."
        )
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"segmentation_grow expects integer labels, got dtype {arr.dtype}.")
    return arr


def _load_h5_dataset(path: str | Path, dataset: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if dataset in f:
            return f[dataset][...]
        keys = [key for key in f.keys() if hasattr(f[key], "shape")]
        if len(keys) == 1:
            return f[keys[0]][...]
        raise ValueError(
            f"{path}: expected dataset {dataset!r} or exactly one dataset, got {list(f.keys())}"
        )


def affinity_foreground_score(
    affinities: np.ndarray,
    *,
    channel_reduction: ChannelReduction = "max",
) -> np.ndarray:
    """Reduce affinity channels to one foreground score volume.

    ``max`` matches the NISB fill probe: any local affinity above the low fill
    threshold can admit a voxel into the growable foreground.
    """
    arr = np.asarray(affinities)
    if arr.ndim == 3:
        return arr.astype(np.float32, copy=False)
    if arr.ndim != 4:
        raise ValueError(f"Expected 3D or channel-first 4D affinity array, got {arr.shape}.")

    if channel_reduction == "max":
        return arr.max(axis=0).astype(np.float32, copy=False)
    if channel_reduction == "min":
        return arr.min(axis=0).astype(np.float32, copy=False)
    if channel_reduction == "mean":
        return arr.mean(axis=0, dtype=np.float32)
    raise ValueError(
        f"Unknown channel_reduction {channel_reduction!r}; expected 'max', 'min', or 'mean'."
    )


def binary_geodesic_grow(
    mask: np.ndarray,
    seed_mask: np.ndarray,
    *,
    max_steps: int | None,
    connectivity: int,
) -> np.ndarray:
    """Grow a binary seed mask inside ``mask`` for a bounded geodesic radius."""
    if max_steps is None or max_steps < 0:
        return np.asarray(mask, dtype=bool)
    if max_steps == 0:
        return np.asarray(seed_mask, dtype=bool) & np.asarray(mask, dtype=bool)
    structure = ndimage.generate_binary_structure(mask.ndim, connectivity)
    return ndimage.binary_dilation(
        seed_mask & mask,
        structure=structure,
        iterations=int(max_steps),
        mask=mask,
    )


def linear_indices_from_coords(
    coords: tuple[np.ndarray, ...],
    *,
    origin: Sequence[int],
    shape: Sequence[int],
) -> np.ndarray:
    """Convert N-dimensional coordinates to C-order linear indices."""
    if len(coords) != len(shape) or len(origin) != len(shape):
        raise ValueError("coords, origin, and shape must have the same rank.")
    if len(shape) != 3:
        raise ValueError("linear_indices_from_coords currently supports 3D volumes.")
    x = coords[0].astype(np.uint64, copy=False) + np.uint64(int(origin[0]))
    y = coords[1].astype(np.uint64, copy=False) + np.uint64(int(origin[1]))
    z = coords[2].astype(np.uint64, copy=False) + np.uint64(int(origin[2]))
    return (x * np.uint64(int(shape[1])) + y) * np.uint64(int(shape[2])) + z


def search_sorted_indices(
    sorted_values: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return insertion positions and exact-match mask for sorted uint indices."""
    pos = np.searchsorted(sorted_values, query)
    valid = pos < sorted_values.size
    if np.any(valid):
        valid_values = sorted_values[pos[valid]] == query[valid]
        exact = np.zeros(valid.shape, dtype=bool)
        exact[np.flatnonzero(valid)] = valid_values
        valid = exact
    return pos, valid


def _sort_aligned(indices: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if indices.size == 0:
        return indices.astype(np.uint64, copy=False), values
    order = np.argsort(indices, kind="mergesort")
    return indices[order].astype(np.uint64, copy=False), values[order]


def _axis_stride(axis: int, shape: Sequence[int]) -> np.uint64:
    if axis == 0:
        return np.uint64(int(shape[1]) * int(shape[2]))
    if axis == 1:
        return np.uint64(int(shape[2]))
    return np.uint64(1)


def _neighbor_valid_mask(
    indices: np.ndarray,
    axis: int,
    sign: int,
    shape: Sequence[int],
) -> np.ndarray:
    if axis == 0:
        stride = int(shape[1]) * int(shape[2])
        coord = indices // np.uint64(stride)
        return coord < np.uint64(int(shape[0]) - 1) if sign > 0 else coord > 0
    if axis == 1:
        stride = int(shape[2])
        coord = (indices // np.uint64(stride)) % np.uint64(int(shape[1]))
        return coord < np.uint64(int(shape[1]) - 1) if sign > 0 else coord > 0
    coord = indices % np.uint64(int(shape[2]))
    return coord < np.uint64(int(shape[2]) - 1) if sign > 0 else coord > 0


def _seed_sparse_targets_from_markers(
    target_indices: np.ndarray,
    target_labels: np.ndarray,
    marker_indices: np.ndarray,
    marker_labels: np.ndarray,
    shape: Sequence[int],
) -> np.ndarray:
    dst_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    tie_parts: list[np.ndarray] = []
    if target_indices.size == 0 or marker_indices.size == 0:
        return np.empty(0, dtype=np.int64)
    all_positions = np.arange(target_indices.size, dtype=np.int64)
    for axis in range(3):
        stride = _axis_stride(axis, shape)
        for sign in (-1, 1):
            valid_boundary = _neighbor_valid_mask(target_indices, axis, sign, shape)
            if not np.any(valid_boundary):
                continue
            src_positions = all_positions[valid_boundary]
            query = (
                target_indices[valid_boundary] + stride
                if sign > 0
                else target_indices[valid_boundary] - stride
            )
            pos, valid = search_sorted_indices(marker_indices, query)
            if not np.any(valid):
                continue
            dst_positions = src_positions[valid]
            unassigned = target_labels[dst_positions] == 0
            if not np.any(unassigned):
                continue
            dst_positions = dst_positions[unassigned]
            marker_positions = pos[valid][unassigned]
            dst_parts.append(dst_positions)
            label_parts.append(marker_labels[marker_positions])
            tie_parts.append(marker_indices[marker_positions])
    return _apply_sparse_assignments(target_labels, dst_parts, label_parts, tie_parts)


def _apply_sparse_assignments(
    target_labels: np.ndarray,
    dst_parts: list[np.ndarray],
    label_parts: list[np.ndarray],
    tie_parts: list[np.ndarray],
) -> np.ndarray:
    """Apply sparse growth proposals with deterministic per-target tie breaking."""
    if not dst_parts:
        return np.empty(0, dtype=np.int64)

    dst_positions = np.concatenate(dst_parts).astype(np.int64, copy=False)
    labels = np.concatenate(label_parts)
    tie_values = np.concatenate(tie_parts).astype(np.uint64, copy=False)
    unassigned = target_labels[dst_positions] == 0
    if not np.any(unassigned):
        return np.empty(0, dtype=np.int64)

    dst_positions = dst_positions[unassigned]
    labels = labels[unassigned]
    tie_values = tie_values[unassigned]
    order = np.lexsort((tie_values, dst_positions))
    dst_sorted = dst_positions[order]
    first = np.unique(dst_sorted, return_index=True)[1]
    chosen_dst = dst_sorted[first]
    chosen_labels = labels[order][first]

    still_unassigned = target_labels[chosen_dst] == 0
    if not np.any(still_unassigned):
        return np.empty(0, dtype=np.int64)
    chosen_dst = chosen_dst[still_unassigned]
    target_labels[chosen_dst] = chosen_labels[still_unassigned]
    return chosen_dst.astype(np.int64, copy=False)


def sparse_geodesic_grow_labels(
    marker_indices: np.ndarray,
    marker_labels: np.ndarray,
    target_indices: np.ndarray,
    *,
    shape: Sequence[int],
    progress: SparseGrowProgress | None = None,
) -> SparseGeodesicGrowResult:
    """Grow marker labels through sparse target voxels by 6-neighbor geodesic distance.

    ``marker_indices`` and ``target_indices`` are C-order linear indices in the
    full volume. Returned labels are aligned to sorted ``target_indices``.
    """
    marker_indices, marker_labels = _sort_aligned(
        np.asarray(marker_indices, dtype=np.uint64),
        np.asarray(marker_labels),
    )
    target_indices = np.sort(np.asarray(target_indices, dtype=np.uint64), kind="mergesort")
    target_labels = np.zeros(target_indices.shape, dtype=marker_labels.dtype)
    frontier = _seed_sparse_targets_from_markers(
        target_indices,
        target_labels,
        marker_indices,
        marker_labels,
        shape,
    )
    initial_assigned = int(np.count_nonzero(target_labels))
    total = int(target_labels.size)
    if progress is not None:
        progress(0, initial_assigned, total, int(frontier.size))

    iteration = 0
    while frontier.size:
        iteration += 1
        dst_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        tie_parts: list[np.ndarray] = []
        source_indices = target_indices[frontier]
        for axis in range(3):
            stride = _axis_stride(axis, shape)
            for sign in (-1, 1):
                valid_boundary = _neighbor_valid_mask(source_indices, axis, sign, shape)
                if not np.any(valid_boundary):
                    continue
                src_positions = frontier[valid_boundary]
                query = (
                    source_indices[valid_boundary] + stride
                    if sign > 0
                    else source_indices[valid_boundary] - stride
                )
                pos, valid = search_sorted_indices(target_indices, query)
                if not np.any(valid):
                    continue
                dst_positions = pos[valid]
                unassigned = target_labels[dst_positions] == 0
                if not np.any(unassigned):
                    continue
                dst_positions = dst_positions[unassigned]
                source_positions = src_positions[valid][unassigned]
                dst_parts.append(dst_positions.astype(np.int64, copy=False))
                label_parts.append(target_labels[source_positions])
                tie_parts.append(source_indices[valid][unassigned])
        if not dst_parts:
            break
        frontier = _apply_sparse_assignments(target_labels, dst_parts, label_parts, tie_parts)
        if frontier.size == 0:
            break
        if progress is not None:
            progress(
                iteration,
                int(np.count_nonzero(target_labels)),
                total,
                int(frontier.size),
            )

    return SparseGeodesicGrowResult(
        labels=target_labels,
        iterations=iteration,
        initial_assigned=initial_assigned,
        final_assigned=int(np.count_nonzero(target_labels)),
    )


def watershed_grow_labels(
    seed_seg: np.ndarray,
    grow_mask: np.ndarray,
    *,
    cost: np.ndarray | None = None,
    fill_mask: np.ndarray | None = None,
    max_grow_steps: int | None = None,
    connectivity: int = 1,
    ignore_label: int = 0,
) -> np.ndarray:
    """Assign growable voxels to marker labels by constrained watershed.

    ``seed_seg`` supplies marker labels. ``grow_mask`` is the binary domain that
    labels are allowed to enter. If ``cost`` is omitted, the watershed uses a
    flat cost, which is equivalent to geodesic marker growth with deterministic
    tie handling from scikit-image.
    """
    seg = _as_3d_segmentation(seed_seg)
    grow = np.asarray(grow_mask, dtype=bool)
    if grow.shape != seg.shape:
        raise ValueError(f"grow_mask shape {grow.shape} does not match segmentation {seg.shape}.")
    if not (1 <= int(connectivity) <= seg.ndim):
        raise ValueError(f"connectivity must be in [1, {seg.ndim}], got {connectivity}.")

    marker_mask = seg != int(ignore_label)
    if not np.any(marker_mask):
        return seg.copy()
    if int(seg.max()) > np.iinfo(np.int64).max:
        raise ValueError("watershed markers exceed int64 range.")

    domain = grow | marker_mask
    domain &= binary_geodesic_grow(
        domain,
        marker_mask,
        max_steps=max_grow_steps,
        connectivity=int(connectivity),
    )

    if fill_mask is None:
        target = domain
    else:
        target = np.asarray(fill_mask, dtype=bool)
        if target.shape != seg.shape:
            raise ValueError(
                f"fill_mask shape {target.shape} does not match segmentation {seg.shape}."
            )
        target &= domain
    if not np.any(target):
        return seg.copy()

    if cost is None:
        cost_arr = np.zeros(seg.shape, dtype=np.float32)
    else:
        cost_arr = np.asarray(cost, dtype=np.float32)
        if cost_arr.shape != seg.shape:
            raise ValueError(
                f"cost shape {cost_arr.shape} does not match segmentation {seg.shape}."
            )

    assigned = watershed(
        cost_arr,
        markers=seg.astype(np.int64, copy=False),
        mask=domain,
        connectivity=int(connectivity),
    )
    result = seg.copy()
    valid_fill = target & (assigned != int(ignore_label))
    result[valid_fill] = assigned[valid_fill].astype(result.dtype, copy=False)
    return result


def grow_segmentation_from_affinity(
    seed_seg: np.ndarray,
    affinities: np.ndarray,
    *,
    foreground_threshold: float = 0.3,
    channel_reduction: ChannelReduction = "max",
    max_fill_steps: int | None = 64,
    connectivity: int = 1,
    cost_power: float = 1.0,
    fill_only_zero: bool = True,
    ignore_label: int = 0,
) -> np.ndarray:
    """Grow existing labels through low-threshold affinity foreground.

    The seed segmentation provides markers, typically a high-threshold
    connected-component decode such as cc=0.66. Affinity foreground provides the
    lower-threshold region that may be filled. Marker watershed assigns only
    voxels connected through foreground, and ``max_fill_steps`` bounds the local
    geodesic growth radius for chunkable behavior.
    """
    seg = _as_3d_segmentation(seed_seg)
    score = affinity_foreground_score(
        affinities,
        channel_reduction=channel_reduction,
    )
    if score.shape != seg.shape:
        raise ValueError(
            f"Affinity foreground shape {score.shape} does not match seed segmentation {seg.shape}."
        )
    if not (1 <= int(connectivity) <= seg.ndim):
        raise ValueError(f"connectivity must be in [1, {seg.ndim}], got {connectivity}.")

    foreground = score > float(foreground_threshold)
    cost = 1.0 - np.clip(score.astype(np.float32, copy=False), 0.0, 1.0)
    if cost_power != 1.0:
        cost = np.power(cost, float(cost_power)).astype(np.float32, copy=False)

    marker_mask = seg != int(ignore_label)
    fill_mask = foreground if not fill_only_zero else foreground & ~marker_mask
    return watershed_grow_labels(
        seg,
        foreground,
        cost=cost,
        fill_mask=fill_mask,
        max_grow_steps=max_fill_steps,
        connectivity=int(connectivity),
        ignore_label=ignore_label,
    )


def segmentation_grow(
    seg: np.ndarray,
    affinities: np.ndarray | None = None,
    *,
    affinity_path: str = "",
    affinity_dataset: str = "main",
    foreground_threshold: float = 0.3,
    channel_reduction: ChannelReduction = "max",
    max_fill_steps: int | None = 64,
    connectivity: int = 1,
    cost_power: float = 1.0,
    fill_only_zero: bool = True,
    ignore_label: int = 0,
) -> np.ndarray:
    """Decoder wrapper for marker-based segmentation growth.

    In a decode pipeline, run this after a high-threshold segmentation decoder
    and pass original affinities with ``use_original_input``.
    """
    if affinities is None and affinity_path:
        affinities = _load_h5_dataset(affinity_path, affinity_dataset)
    if affinities is None:
        logger.info("segmentation_grow: no affinities provided; returning input.")
        return _as_3d_segmentation(seg).copy()

    filled = grow_segmentation_from_affinity(
        seg,
        affinities,
        foreground_threshold=foreground_threshold,
        channel_reduction=channel_reduction,
        max_fill_steps=max_fill_steps,
        connectivity=connectivity,
        cost_power=cost_power,
        fill_only_zero=fill_only_zero,
        ignore_label=ignore_label,
    )
    return cast2dtype(filled)
