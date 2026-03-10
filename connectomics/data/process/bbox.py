from __future__ import annotations

import itertools
from collections import OrderedDict
from typing import Optional, Tuple, Union

import numpy as np
from scipy.ndimage import find_objects

__all__ = [
    "bbox_ND",
    "bbox_relax",
    "adjust_bbox",
    "index2bbox",
    "crop_ND",
    "replace_ND",
    "rand_window",
    "compute_bbox_all",
]


def bbox_ND(img: np.ndarray, relax: int = 0) -> tuple:
    """Calculate the bounding box of an object in a N-dimensional numpy array.
    All non-zero elements are treated as foregounrd. Please note that the
    calculated bounding-box coordinates are inclusive.
    Reference: https://stackoverflow.com/a/31402351

    Args:
        img (np.ndarray): a N-dimensional array with zero as background.
        relax (int): relax the bbox by n pixels for each side of each axis.

    Returns:
        tuple: N-dimensional bounding box coordinates.
    """
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])

    return bbox_relax(out, img.shape, relax)


def bbox_relax(coord: Union[tuple, list], shape: tuple, relax: int = 0) -> tuple:
    if len(coord) != len(shape) * 2:
        raise ValueError(
            f"Expected {len(shape) * 2} coordinates for {len(shape)}D shape, got {len(coord)}"
        )
    coord = list(coord)
    for i in range(len(shape)):
        coord[2 * i] = max(0, coord[2 * i] - relax)
        coord[2 * i + 1] = min(shape[i], coord[2 * i + 1] + relax)

    return tuple(coord)


def adjust_bbox(low, high, sz):
    if high < low:
        raise ValueError(f"high ({high}) must be >= low ({low})")
    bbox_sz = high - low
    diff = abs(sz - bbox_sz) // 2
    if bbox_sz >= sz:
        return low + diff, low + diff + sz

    return low - diff, low - diff + sz


def index2bbox(seg: np.ndarray, indices: list, relax: int = 0, iterative: bool = False) -> dict:
    """Calculate the bounding boxes associated with the given mask indices.
    For a small number of indices, the iterative approach may be preferred.

    Note:
        Since labels with value 0 are ignored in ``scipy.ndimage.find_objects``,
        the first tuple in the output list is associated with label index 1.
    """
    bbox_dict = OrderedDict()

    if iterative:
        # calculate the bounding boxes of each segment iteratively
        for idx in indices:
            temp = seg == idx  # binary mask of the current seg
            bbox = bbox_ND(temp, relax=relax)
            bbox_dict[idx] = bbox
        return bbox_dict

    # calculate the bounding boxes using scipy.ndimage.find_objects
    loc = find_objects(seg)
    seg_shape = seg.shape
    for idx, item in enumerate(loc):
        if item is None:
            # For scipy.ndimage.find_objects, if a number is
            # missing, None is returned instead of a slice.
            continue

        object_idx = idx + 1  # 0 is ignored in find_objects
        if object_idx not in indices:
            continue

        bbox = []
        for x in item:  # slice() object
            bbox.append(x.start)
            bbox.append(x.stop - 1)  # bbox is inclusive by definition
        bbox_dict[object_idx] = bbox_relax(bbox, seg_shape, relax)
    return bbox_dict


def _coord2slice(coord: Tuple[int], ndim: int, end_included: bool = False):
    if len(coord) != ndim * 2:
        raise ValueError(f"Expected {ndim * 2} coordinates for {ndim}D array, got {len(coord)}")
    slicing = []
    for i in range(ndim):
        start = coord[2 * i]
        end = coord[2 * i + 1] + 1 if end_included else coord[2 * i + 1]
        slicing.append(slice(start, end))
    slicing = tuple(slicing)
    return slicing


def crop_ND(img: np.ndarray, coord: Tuple[int], end_included: bool = False) -> np.ndarray:
    """Crop a chunk from a N-dimensional array based on the
    bounding box coordinates.
    """
    slicing = _coord2slice(coord, img.ndim, end_included)
    return img[slicing].copy()


def replace_ND(
    img: np.ndarray,
    replacement: np.ndarray,
    coord: Tuple[int],
    end_included: bool = False,
    overwrite_bg: bool = False,
) -> np.ndarray:
    """Replace a chunk from a N-dimensional array based on the
    bounding box coordinates.
    """
    slicing = _coord2slice(coord, img.ndim, end_included)

    if not overwrite_bg:  # only overwrite foreground pixels
        temp = img[slicing].copy()
        mask_fg = (replacement != 0).astype(temp.dtype)
        mask_bg = (replacement == 0).astype(temp.dtype)
        replacement = replacement * mask_fg + temp * mask_bg

    img[slicing] = replacement
    return img.copy()


def rand_window(w0, w1, sz, rand_shift: int = 0):
    if w1 < w0:
        raise ValueError(f"w1 ({w1}) must be >= w0 ({w0})")
    diff = np.abs((w1 - w0) - sz)
    if (w1 - w0) <= sz:
        if rand_shift > 0:  # random shift augmentation
            start_l = max(w0 - diff // 2 - rand_shift, w1 - sz)
            start_r = min(w0, w0 - diff // 2 + rand_shift)
            low = np.random.randint(start_l, start_r)
        else:
            low = w0 - diff // 2
    else:
        if rand_shift > 0:  # random shift augmentation
            start_l = max(w0, w0 + diff // 2 - rand_shift)
            start_r = min(w0 + diff // 2 + rand_shift, w1 - sz)
            low = np.random.randint(start_l, start_r)
        else:
            low = w0 + diff // 2
    high = low + sz
    return low, high


def compute_bbox_all(
    seg: np.ndarray, do_count: bool = False, uid: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """Compute bounding boxes for all instances in a 2D or 3D segmentation.

    Uses scipy.ndimage.find_objects for a single-pass C-level computation.

    Args:
        seg: 2D or 3D instance segmentation.
        do_count: Whether to include voxel counts.
        uid: Restrict to these instance IDs.

    Returns:
        Array with columns [id, min0, max0, min1, max1, ...] (inclusive coords),
        or None if no instances found.
    """
    return _compute_bbox_all_find_objects(seg, do_count, uid)


def _compute_bbox_all_find_objects(
    seg: np.ndarray, do_count: bool = False, uid: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """Compute bounding boxes for all instances using scipy.ndimage.find_objects.

    Works for both 2D and 3D segmentation. Uses a single C-level pass over the
    volume instead of iterating slices/rows/cols in Python.

    Args:
        seg: 2D or 3D instance segmentation (H, W) or (D, H, W).
        do_count: Whether to include voxel counts.
        uid: Restrict to these instance IDs. Default: all non-zero IDs.

    Returns:
        Array of shape (N, 2*ndim+1 [+1 if do_count]) where each row is:
          2D: [id, ymin, ymax, xmin, xmax, (count)]
          3D: [id, zmin, zmax, ymin, ymax, xmin, xmax, (count)]
        Coordinates are inclusive. Returns None if no instances found.
    """
    ndim = seg.ndim
    if ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D input, got {ndim}D")

    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid > 0]
    if len(uid) == 0:
        return None

    # find_objects returns list indexed by (label - 1)
    loc = find_objects(seg)

    # Count voxels if requested (single pass)
    counts_dict = {}
    if do_count:
        seg_ui, seg_uc = np.unique(seg, return_counts=True)
        counts_dict = dict(zip(seg_ui, seg_uc))

    rows = []
    for label_id in uid:
        label_id = int(label_id)
        idx = label_id - 1  # find_objects is 0-indexed
        if idx < 0 or idx >= len(loc) or loc[idx] is None:
            continue
        slices = loc[idx]
        row = [label_id]
        for s in slices:
            row.append(s.start)
            row.append(s.stop - 1)  # inclusive
        if do_count:
            row.append(counts_dict.get(label_id, 0))
        rows.append(row)

    if not rows:
        return None

    return np.array(rows, dtype=int)
