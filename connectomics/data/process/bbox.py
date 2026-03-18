from __future__ import annotations

import itertools
from collections import OrderedDict
from typing import Optional, Tuple, Union

import numpy as np

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


def index2bbox(seg: np.ndarray, indices: list, relax: int = 0, iterative: bool = True) -> dict:
    """Calculate the bounding boxes associated with the given mask indices."""
    bbox_dict = OrderedDict()
    for idx in indices:
        temp = seg == idx
        bbox = bbox_ND(temp, relax=relax)
        bbox_dict[idx] = bbox
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

    Scans along each axis to find min/max extents per segment ID.
    Ported from em_util.

    Args:
        seg: 2D or 3D instance segmentation.
        do_count: Whether to include voxel counts.
        uid: Restrict to these instance IDs.

    Returns:
        2D: ``[id, ymin, ymax, xmin, xmax, (count)]``
        3D: ``[id, zmin, zmax, ymin, ymax, xmin, xmax, (count)]``
        None if no instances found.
    """
    if seg.ndim == 2:
        return _compute_bbox_all_2d(seg, do_count, uid)
    elif seg.ndim == 3:
        return _compute_bbox_all_3d(seg, do_count, uid)
    else:
        raise ValueError(f"Expected 2D or 3D input, got {seg.ndim}D")


def _compute_bbox_all_2d(
    seg: np.ndarray, do_count: bool = False, uid: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """2D bounding boxes via row/column scan (from em_util)."""
    H, W = seg.shape
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid > 0]
    if len(uid) == 0:
        return None

    uid_max = int(uid.max())
    sid_dict = {int(u): i for i, u in enumerate(uid)}
    ncols = 6 if do_count else 5
    out = np.zeros((len(uid), ncols), dtype=np.int64)
    out[:, 0] = uid
    out[:, 1] = H
    out[:, 3] = W

    rids = np.where((seg > 0).any(axis=1))[0]
    for rid in rids:
        sid = np.unique(seg[rid])
        sid = sid[(sid > 0) & (sid <= uid_max)]
        sid_ind = [sid_dict[int(x)] for x in sid if int(x) in sid_dict]
        out[sid_ind, 1] = np.minimum(out[sid_ind, 1], rid)
        out[sid_ind, 2] = np.maximum(out[sid_ind, 2], rid)

    cids = np.where((seg > 0).any(axis=0))[0]
    for cid in cids:
        sid = np.unique(seg[:, cid])
        sid = sid[(sid > 0) & (sid <= uid_max)]
        sid_ind = [sid_dict[int(x)] for x in sid if int(x) in sid_dict]
        out[sid_ind, 3] = np.minimum(out[sid_ind, 3], cid)
        out[sid_ind, 4] = np.maximum(out[sid_ind, 4], cid)

    if do_count:
        seg_ui, seg_uc = np.unique(seg, return_counts=True)
        for i, j in zip(seg_ui, seg_uc):
            if int(i) in sid_dict:
                out[sid_dict[int(i)], -1] = j

    return out


def _compute_bbox_all_3d(
    seg: np.ndarray, do_count: bool = False, uid: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """3D bounding boxes via slice/row/column scan (from em_util)."""
    D, H, W = seg.shape
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid > 0]
    if len(uid) == 0:
        return None

    uid_max = int(uid.max())
    sid_dict = {int(u): i for i, u in enumerate(uid)}
    ncols = 8 if do_count else 7
    out = np.zeros((len(uid), ncols), dtype=np.int64)
    out[:, 0] = uid
    out[:, 1] = D
    out[:, 2] = -1
    out[:, 3] = H
    out[:, 4] = -1
    out[:, 5] = W
    out[:, 6] = -1

    zids = np.where((seg > 0).reshape(D, -1).any(axis=1))[0]
    for zid in zids:
        sid = np.unique(seg[zid])
        sid = sid[(sid > 0) & (sid <= uid_max)]
        sid_ind = [sid_dict[int(x)] for x in sid if int(x) in sid_dict]
        out[sid_ind, 1] = np.minimum(out[sid_ind, 1], zid)
        out[sid_ind, 2] = np.maximum(out[sid_ind, 2], zid)

    rids = np.where((seg > 0).sum(axis=0).sum(axis=1) > 0)[0]
    for rid in rids:
        sid = np.unique(seg[:, rid])
        sid = sid[(sid > 0) & (sid <= uid_max)]
        sid_ind = [sid_dict[int(x)] for x in sid if int(x) in sid_dict]
        out[sid_ind, 3] = np.minimum(out[sid_ind, 3], rid)
        out[sid_ind, 4] = np.maximum(out[sid_ind, 4], rid)

    cids = np.where((seg > 0).sum(axis=0).sum(axis=0) > 0)[0]
    for cid in cids:
        sid = np.unique(seg[:, :, cid])
        sid = sid[(sid > 0) & (sid <= uid_max)]
        sid_ind = [sid_dict[int(x)] for x in sid if int(x) in sid_dict]
        out[sid_ind, 5] = np.minimum(out[sid_ind, 5], cid)
        out[sid_ind, 6] = np.maximum(out[sid_ind, 6], cid)

    if do_count:
        seg_ui, seg_uc = np.unique(seg, return_counts=True)
        for i, j in zip(seg_ui, seg_uc):
            if int(i) in sid_dict:
                out[sid_dict[int(i)], -1] = j

    return out
