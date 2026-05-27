"""Seeded branch-split postprocessing for instance segmentations."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import cc3d
import numpy as np
from scipy import ndimage

from ..utils import cast2dtype

__all__ = ["branch_split"]

logger = logging.getLogger(__name__)


def _load_h5_dataset(path: str, dataset: str = "main") -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        if dataset in f:
            return f[dataset][...]
        keys = [key for key in f.keys() if hasattr(f[key], "shape")]
        if len(keys) == 1:
            return f[keys[0]][...]
        raise ValueError(
            f"{path}: expected dataset '{dataset}' or exactly one dataset, got {list(f.keys())}"
        )


def _bbox_from_mask(mask: np.ndarray) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _target_id_set(target_ids: Optional[Sequence[int]]) -> Optional[set[int]]:
    if target_ids is None:
        return None
    return {int(label) for label in target_ids}


def _erosion_seed_components(
    parent_mask: np.ndarray,
    *,
    erosion_iterations: int,
    connectivity: int,
) -> np.ndarray:
    eroded = np.zeros_like(parent_mask, dtype=bool)
    structure_2d = ndimage.generate_binary_structure(2, 1)
    for z in range(parent_mask.shape[0]):
        eroded[z] = ndimage.binary_erosion(
            parent_mask[z],
            structure=structure_2d,
            iterations=erosion_iterations,
            border_value=0,
        )
    return cc3d.connected_components(eroded, connectivity=connectivity)


def _candidate_seed_labels(
    seed_values: np.ndarray,
    *,
    parent_size: int,
    min_seed_size: int,
    min_seed_fraction: float,
    max_seed_fraction: float,
    max_splits_per_parent: int,
) -> list[int]:
    labels, counts = np.unique(seed_values, return_counts=True)
    keep = labels != 0
    labels = labels[keep]
    counts = counts[keep]
    if len(labels) < 2:
        return []

    order = np.lexsort((labels, -counts))
    labels = labels[order]
    counts = counts[order]

    candidates: list[int] = []
    for idx, (label, count) in enumerate(zip(labels, counts)):
        count = int(count)
        fraction = count / float(parent_size)
        if idx == 0:
            candidates.append(int(label))
            continue
        if count < min_seed_size:
            continue
        if fraction < min_seed_fraction:
            continue
        if max_seed_fraction > 0 and fraction > max_seed_fraction:
            continue
        candidates.append(int(label))
        if len(candidates) >= max_splits_per_parent:
            break

    return candidates if len(candidates) >= 2 else []


def _assign_parent_from_markers(parent_mask: np.ndarray, markers: np.ndarray) -> np.ndarray:
    seed_mask = markers != 0
    if not seed_mask.any():
        return np.zeros_like(markers, dtype=np.uint64)
    _, nearest = ndimage.distance_transform_edt(~seed_mask, return_indices=True)
    assigned = markers[tuple(nearest)]
    assigned = assigned.astype(np.uint64, copy=False)
    assigned[~parent_mask] = 0
    return assigned


def _affinity_seed_segmentation(
    affinities: np.ndarray,
    *,
    threshold: float,
    backend: str,
    edge_offset: int,
) -> np.ndarray:
    from .segmentation import decode_affinity_cc

    return decode_affinity_cc(
        np.asarray(affinities),
        threshold=threshold,
        backend=backend,
        edge_offset=edge_offset,
    )


def branch_split(
    seg: np.ndarray,
    affinities: Optional[np.ndarray] = None,
    *,
    seed_seg: Optional[np.ndarray] = None,
    seed_seg_path: str = "",
    seed_dataset: str = "main",
    affinity_path: str = "",
    affinity_dataset: str = "main",
    seed_affinity_threshold: float = 0.75,
    seed_backend: str = "auto",
    edge_offset: int = 0,
    target_ids: Optional[Sequence[int]] = None,
    min_parent_size: int = 10_000,
    min_seed_size: int = 1_000,
    min_seed_fraction: float = 0.001,
    max_seed_fraction: float = 0.75,
    max_splits_per_parent: int = 8,
    preserve_largest: bool = True,
    assignment: str = "nearest",
    fallback_2d_erosion: bool = False,
    erosion_iterations: int = 1,
    connectivity: int = 6,
) -> np.ndarray:
    """Split large labels using conservative branch seed components.

    This is an error-correction decoder: it expects an instance segmentation as
    input and optionally uses affinities or an external seed segmentation to
    propose branch splits. The most common use is a chained decode pipeline:
    ``decode_waterz`` followed by ``branch_split`` with ``use_original_input``.
    """
    seg = np.asarray(seg)
    if seg.ndim != 3:
        raise ValueError(f"branch_split expects a 3D segmentation, got shape {seg.shape}.")

    if seed_seg is None and seed_seg_path:
        seed_seg = _load_h5_dataset(seed_seg_path, seed_dataset)
    if affinities is None and affinity_path:
        affinities = _load_h5_dataset(affinity_path, affinity_dataset)
    if seed_seg is None and affinities is not None:
        seed_seg = _affinity_seed_segmentation(
            affinities,
            threshold=seed_affinity_threshold,
            backend=seed_backend,
            edge_offset=edge_offset,
        )

    if seed_seg is not None:
        seed_seg = np.asarray(seed_seg)
        if seed_seg.shape != seg.shape:
            raise ValueError(
                f"branch_split seed segmentation shape {seed_seg.shape} does not match "
                f"segmentation shape {seg.shape}."
            )
    elif not fallback_2d_erosion:
        logger.info("branch_split: no seed segmentation or affinities; returning input.")
        return seg.copy()

    target_set = _target_id_set(target_ids)
    labels, counts = np.unique(seg, return_counts=True)
    parent_sizes = {
        int(label): int(count)
        for label, count in zip(labels, counts)
        if int(label) != 0 and int(count) >= min_parent_size
    }
    if target_set is not None:
        parent_sizes = {
            label: count for label, count in parent_sizes.items() if label in target_set
        }

    result = seg.astype(np.uint64, copy=True)
    next_label = int(result.max()) + 1
    n_split = 0

    for parent_id, parent_size in parent_sizes.items():
        parent_mask_global = seg == parent_id
        if not parent_mask_global.any():
            continue
        bbox = _bbox_from_mask(parent_mask_global)
        parent_mask = parent_mask_global[bbox]

        if seed_seg is not None:
            seed_crop = np.asarray(seed_seg[bbox])
            seed_values = seed_crop[parent_mask]
        else:
            seed_crop = _erosion_seed_components(
                parent_mask,
                erosion_iterations=erosion_iterations,
                connectivity=connectivity,
            )
            seed_values = seed_crop[parent_mask]

        candidates = _candidate_seed_labels(
            seed_values,
            parent_size=parent_size,
            min_seed_size=min_seed_size,
            min_seed_fraction=min_seed_fraction,
            max_seed_fraction=max_seed_fraction,
            max_splits_per_parent=max_splits_per_parent,
        )
        if not candidates:
            continue

        markers = np.zeros(parent_mask.shape, dtype=np.uint64)
        for idx, seed_label in enumerate(candidates):
            output_label = parent_id if preserve_largest and idx == 0 else next_label
            if not (preserve_largest and idx == 0):
                next_label += 1
            markers[parent_mask & (seed_crop == seed_label)] = output_label

        if assignment == "nearest":
            assigned = _assign_parent_from_markers(parent_mask, markers)
        elif assignment == "seeds":
            assigned = markers
        else:
            raise ValueError("branch_split assignment must be 'nearest' or 'seeds'.")

        result_crop = result[bbox]
        changed = parent_mask & (assigned != 0) & (assigned != parent_id)
        if not changed.any():
            continue
        result_crop[parent_mask & (assigned != 0)] = assigned[parent_mask & (assigned != 0)]
        result[bbox] = result_crop
        n_split += 1

    logger.info("branch_split: split %d parent segments", n_split)
    return cast2dtype(result)
