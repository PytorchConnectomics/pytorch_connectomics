from __future__ import annotations

import cc3d
import fastremap
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
from skimage.morphology import binary_dilation, disk

from .affinity import (
    affinity_deepem_crop_enabled,
    compute_affinity_crop_pad,
    crop_spatial_by_offsets,
    crop_spatial_by_pad,
    parse_affinity_offsets,
    resolve_affinity_channel_groups_from_cfg,
    resolve_affinity_offsets_for_channel_slice,
    seg_to_affinity,
)
from .flow import seg2d_to_flows

__all__ = [
    "seg_to_flows",
    "seg_to_affinity",
    "seg_to_polarity",
    "seg_to_small_seg",
    "seg_to_binary",
    "seg_to_instance_bd",
    "seg_erosion_dilation",
]


def seg_to_flows(label: np.ndarray) -> np.ndarray:
    """Convert segmentation to flow fields.

    Args:
        label: (y, x) for 2D or (z, y, x) for 3D

    Returns:
        (2, y, x) for 2D or (2, z, y, x) for 3D (channel first)
    """
    masks = label.squeeze().astype(np.int32)

    if masks.ndim == 3:
        nz, ny, nx = masks.shape
        mu = np.zeros((2, nz, ny, nx), np.float32)
        for zi in range(nz):
            mu0 = seg2d_to_flows(masks[zi])[0]
            mu[:, zi] = mu0
        return mu
    elif masks.ndim == 2:
        mu, _, _ = seg2d_to_flows(masks)
        return mu.astype(np.float32)
    else:
        raise ValueError(f"expecting 2D or 3D labels but received {masks.ndim}D input!")


def _boundary_shift_3d(seg, bd_temp, edge_mode):
    """Optimized 3D boundary detection using shift-and-compare.

    Computes the diff once per axis and marks both neighbors, halving comparisons.
    """
    if edge_mode == "all":
        for axis in range(3):
            s_lo = [slice(None)] * 3
            s_hi = [slice(None)] * 3
            s_lo[axis] = slice(None, -1)
            s_hi[axis] = slice(1, None)
            diff = seg[tuple(s_lo)] != seg[tuple(s_hi)]
            bd_temp[tuple(s_lo)] |= diff
            bd_temp[tuple(s_hi)] |= diff
    elif edge_mode == "seg-all":
        for axis in range(3):
            s_lo = [slice(None)] * 3
            s_hi = [slice(None)] * 3
            s_lo[axis] = slice(None, -1)
            s_hi[axis] = slice(1, None)
            a, b = seg[tuple(s_lo)], seg[tuple(s_hi)]
            diff = (a != b) & ((a > 0) | (b > 0))
            bd_temp[tuple(s_lo)] |= diff
            bd_temp[tuple(s_hi)] |= diff
    elif edge_mode == "seg-no-bg":
        for axis in range(3):
            s_lo = [slice(None)] * 3
            s_hi = [slice(None)] * 3
            s_lo[axis] = slice(None, -1)
            s_hi[axis] = slice(1, None)
            a, b = seg[tuple(s_lo)], seg[tuple(s_hi)]
            diff = (a != b) & (a > 0) & (b > 0)
            bd_temp[tuple(s_lo)] |= diff
            bd_temp[tuple(s_hi)] |= diff


def _boundary_shift_2d(seg_2d, bd_2d, edge_mode):
    """Optimized 2D boundary detection using shift-and-compare."""
    if edge_mode == "all":
        for axis in range(2):
            s_lo = [slice(None)] * 2
            s_hi = [slice(None)] * 2
            s_lo[axis] = slice(None, -1)
            s_hi[axis] = slice(1, None)
            diff = seg_2d[tuple(s_lo)] != seg_2d[tuple(s_hi)]
            bd_2d[tuple(s_lo)] |= diff
            bd_2d[tuple(s_hi)] |= diff
    elif edge_mode == "seg-all":
        for axis in range(2):
            s_lo = [slice(None)] * 2
            s_hi = [slice(None)] * 2
            s_lo[axis] = slice(None, -1)
            s_hi[axis] = slice(1, None)
            a, b = seg_2d[tuple(s_lo)], seg_2d[tuple(s_hi)]
            diff = (a != b) & ((a > 0) | (b > 0))
            bd_2d[tuple(s_lo)] |= diff
            bd_2d[tuple(s_hi)] |= diff
    elif edge_mode == "seg-no-bg":
        for axis in range(2):
            s_lo = [slice(None)] * 2
            s_hi = [slice(None)] * 2
            s_lo[axis] = slice(None, -1)
            s_hi[axis] = slice(1, None)
            a, b = seg_2d[tuple(s_lo)], seg_2d[tuple(s_hi)]
            diff = (a != b) & (a > 0) & (b > 0)
            bd_2d[tuple(s_lo)] |= diff
            bd_2d[tuple(s_hi)] |= diff


def seg_to_instance_bd(
    seg: np.ndarray, thickness: int = 1, edge_mode: str = "seg-all", mode: str = "3d"
) -> np.ndarray:
    """Generate instance boundary/contour maps from segmentation masks.

    Args:
        seg: Input segmentation map (Z, Y, X).
        thickness: Boundary thickness in pixels. 1 uses optimized shift comparison.
        edge_mode: "all" | "seg-all" | "seg-no-bg"
        mode: "3d" for full 3D or "2d" for slice-by-slice

    Returns:
        Binary boundary map (uint8) of same shape as seg.
    """
    sz = seg.shape
    bd = np.zeros(sz, np.uint8)

    if mode == "3d":
        if thickness == 1:
            bd_temp = np.zeros(sz, dtype=bool)
            _boundary_shift_3d(seg, bd_temp, edge_mode)
            bd = bd_temp.astype(np.uint8)
        else:
            if edge_mode == "all":
                seg_eroded = grey_erosion(seg, thickness, mode="reflect")
                bd = (seg != seg_eroded).astype(np.uint8)
            elif edge_mode == "seg-all":
                seg_eroded = grey_erosion(seg, thickness, mode="reflect")
                bd = ((seg > 0) & (seg != seg_eroded)).astype(np.uint8)
            elif edge_mode == "seg-no-bg":
                seg_dilated = grey_dilation(seg, thickness, mode="reflect")
                seg_mask = np.where(seg > 0, seg_dilated, np.inf)
                seg_eroded = grey_erosion(seg_mask, thickness, mode="reflect")
                bd = ((seg > 0) & (seg_dilated != seg_eroded)).astype(np.uint8)
    else:  # mode == '2d'
        if thickness == 1:
            for z in range(sz[0]):
                bd_slice = np.zeros(seg[z].shape, dtype=bool)
                _boundary_shift_2d(seg[z], bd_slice, edge_mode)
                bd[z] = bd_slice.astype(np.uint8)
        else:
            for z in range(sz[0]):
                slice_2d = seg[z]
                if edge_mode == "all":
                    eroded = grey_erosion(slice_2d, thickness, mode="reflect")
                    bd[z] = (slice_2d != eroded).astype(np.uint8)
                elif edge_mode == "seg-all":
                    eroded = grey_erosion(slice_2d, thickness, mode="reflect")
                    bd[z] = ((slice_2d > 0) & (slice_2d != eroded)).astype(np.uint8)
                elif edge_mode == "seg-no-bg":
                    dilated = grey_dilation(slice_2d, thickness, mode="reflect")
                    masked = np.where(slice_2d > 0, dilated, np.inf)
                    eroded = grey_erosion(masked, thickness, mode="reflect")
                    bd[z] = ((slice_2d > 0) & (dilated != eroded)).astype(np.uint8)
    return bd


def seg_to_binary(label, segment_id=[]):
    """Convert segmentation to binary mask.

    Args:
        label: Segmentation array
        segment_id: List of segment IDs to include as foreground.
                   Empty list [] means all non-zero labels.
    """
    if not segment_id:
        return label > 0

    fg_mask = np.zeros_like(label, dtype=bool)
    for seg_id in segment_id:
        fg_mask |= (label == int(seg_id))
    return fg_mask


def seg_to_polarity(label: np.ndarray, exclusive: bool = False) -> np.ndarray:
    """Convert the label to synaptic polarity target.

    Args:
        label: Segmentation array where odd labels are pre-synaptic, even are post-synaptic
        exclusive: If False, returns 3-channel non-exclusive masks (for BCE loss).
                  If True, returns single-channel exclusive classes (for CE loss).
    """
    pos = np.logical_and((label % 2) == 1, label > 0)
    neg = np.logical_and((label % 2) == 0, label > 0)

    if not exclusive:
        return np.stack([pos, neg, (label > 0)], 0).astype(np.float32)

    return np.maximum(pos.astype(np.int64), 2 * neg.astype(np.int64))


def seg_to_synapse_instance(label: np.ndarray):
    """Convert semantic polarity annotation to instance annotation."""
    indices = np.unique(label)
    if not np.array_equal(indices, [0, 1, 2]):
        raise ValueError(f"Expected labels [0, 1, 2], got {indices}")

    fg = (label != 0).astype(bool)
    struct = disk(2, dtype=bool)[np.newaxis, :, :]
    fg = binary_dilation(fg, struct)
    segm = cc3d.connected_components(fg).astype(int)

    seg_pos = (label == 1).astype(segm.dtype)
    seg_neg = (label == 2).astype(segm.dtype)

    seg_pos = seg_pos * (segm * 2 - 1)
    seg_neg = seg_neg * (segm * 2)
    instance_label = np.maximum(seg_pos, seg_neg)

    return fastremap.refit(instance_label)


def seg_to_small_seg(seg: np.ndarray, threshold: int = 100) -> np.ndarray:
    """Convert segmentation to small object mask (vectorized).

    Args:
        seg: Input segmentation array
        threshold: Maximum voxel count for small objects (default: 100)

    Returns:
        Small object mask (1.0 for small objects, 0.0 otherwise)
    """
    labeled_seg = cc3d.connected_components(seg)
    unique_labels, counts = np.unique(labeled_seg, return_counts=True)

    # Vectorized: build lookup table
    lut = np.zeros(unique_labels.max() + 1, dtype=np.float32)
    mask = (counts <= threshold) & (unique_labels > 0)
    lut[unique_labels[mask]] = 1.0

    return lut[labeled_seg]


def seg_erosion_dilation(
    seg: np.ndarray, operation: str = "erosion", kernel_size: int = 1
) -> np.ndarray:
    """Apply erosion and/or dilation to segmentation.

    Args:
        seg: Input segmentation array
        operation: 'erosion', 'dilation', or 'both'
        kernel_size: Kernel size for morphological operation
    """
    from skimage.morphology import erosion, dilation

    struct_elem = disk(kernel_size, dtype=bool)
    footprint_2d = struct_elem

    result = seg.copy()

    if operation in ("erosion", "both"):
        for z in range(seg.shape[0]):
            result[z] = erosion(seg[z] if operation == "erosion" else result[z], footprint_2d)
    if operation in ("dilation", "both"):
        for z in range(result.shape[0]):
            result[z] = dilation(result[z], footprint_2d)
    if operation not in ("erosion", "dilation", "both"):
        raise ValueError(f"Unknown operation: {operation}. Use 'erosion', 'dilation', or 'both'")

    return result
