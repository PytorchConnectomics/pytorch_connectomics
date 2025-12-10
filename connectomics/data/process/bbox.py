from __future__ import division, print_function
from collections import OrderedDict
from typing import Optional, Union, Tuple

import itertools
import numpy as np
from scipy.ndimage import find_objects

__all__ = [
    "bbox_ND",
    "bbox_relax",
    "adjust_bbox",
    "index2bbox",
    "crop_ND",
    "replace_ND",
    "crop_pad_data",
    "rand_window",
    "compute_bbox_all",
    "compute_bbox_all_2d",
    "compute_bbox_all_3d",
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
    assert len(coord) == len(shape) * 2
    coord = list(coord)
    for i in range(len(shape)):
        coord[2 * i] = max(0, coord[2 * i] - relax)
        coord[2 * i + 1] = min(shape[i], coord[2 * i + 1] + relax)

    return tuple(coord)


def adjust_bbox(low, high, sz):
    assert high >= low
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
    assert len(coord) == ndim * 2
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


def crop_pad_data(data, z, bbox_2d, pad_val=0, mask=None, return_box=False):
    """Crop a 2D patch from 3D volume given the z index and 2d bbox."""
    sz = data.shape[1:]
    y1o, y2o, x1o, x2o = bbox_2d  # region to crop
    y1m, y2m, x1m, x2m = 0, sz[0], 0, sz[1]
    y1, x1 = max(y1o, y1m), max(x1o, x1m)
    y2, x2 = min(y2o, y2m), min(x2o, x2m)
    cropped = data[z, y1:y2, x1:x2]

    if mask is not None:
        mask_2d = mask[z, y1:y2, x1:x2]
        cropped = cropped * (mask_2d != 0).astype(cropped.dtype)

    pad = ((y1 - y1o, y2o - y2), (x1 - x1o, x2o - x2))
    if not all(v == 0 for v in pad):
        cropped = np.pad(cropped, pad, mode="constant", constant_values=pad_val)

    if not return_box:
        return cropped

    return cropped, [y1, y2, x1, x2], pad


def rand_window(w0, w1, sz, rand_shift: int = 0):
    assert w1 >= w0
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
    """
    Compute the bounding boxes of segments in a segmentation map.

    Args:
        seg (numpy.ndarray): The input segmentation map (2D or 3D).
        do_count (bool, optional): Whether to compute the segment counts. Defaults to False.
        uid (numpy.ndarray, optional): The segment IDs to compute the bounding
            boxes for. Defaults to None.

    Returns:
        numpy.ndarray: An array containing the bounding boxes of the segments.

    Raises:
        ValueError: If the input volume is not 2D or 3D.

    Notes:
        - The function computes the bounding boxes of segments in a segmentation map.
        - The bounding boxes represent the minimum and maximum coordinates of
          each segment in the map.
        - The function can compute the segment counts if `do_count` is set to True.
        - The bounding boxes are returned as an array.
    """
    if seg.ndim == 2:
        return compute_bbox_all_2d(seg, do_count, uid)
    elif seg.ndim == 3:
        return compute_bbox_all_3d(seg, do_count, uid)
    else:
        raise ValueError("Input volume should be either 2D or 3D")


def compute_bbox_all_2d(
    seg: np.ndarray, do_count: bool = False, uid: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """
    Compute the bounding boxes of 2D instance segmentation.

    Args:
        seg (numpy.ndarray): The 2D instance segmentation.
        do_count (bool, optional): Whether to compute the count of each instance. Defaults to False.
        uid (numpy.ndarray, optional): The unique identifier for each instance. Defaults to None.

    Returns:
        numpy.ndarray: The computed bounding boxes of the instances.

    Notes:
        - The input segmentation should have dimensions HxW, where H is the
          height and W is the width.
        - Each row in the output represents an instance and contains the following information:
            - seg id: The ID of the instance.
            - bounding box: The coordinates of the bounding box in the format
              [ymin, ymax, xmin, xmax].
            - count (optional): The count of pixels belonging to the instance.
        - If the `uid` argument is not provided, the unique identifiers are
          automatically determined from the segmentation.
        - Instances with no pixels are excluded from the output.
    """
    sz = seg.shape
    assert len(sz) == 2
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid > 0]
    if len(uid) == 0:
        return None
    # memory efficient
    uid_max = int(uid.max())
    sid_dict = dict(zip(uid, range(len(uid))))
    out = np.zeros((len(uid), 5 + do_count), dtype=int)

    out[:, 0] = uid
    out[:, 1] = sz[0]
    out[:, 3] = sz[1]
    # for each row
    rids = np.where((seg > 0).sum(axis=1) > 0)[0]
    for rid in rids:
        sid = np.unique(seg[rid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        sid_ind = [sid_dict[x] for x in sid]
        out[sid_ind, 1] = np.minimum(out[sid_ind, 1], rid)
        out[sid_ind, 2] = np.maximum(out[sid_ind, 2], rid)
    cids = np.where((seg > 0).sum(axis=0) > 0)[0]
    for cid in cids:
        sid = np.unique(seg[:, cid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        sid_ind = [sid_dict[x] for x in sid]
        out[sid_ind, 3] = np.minimum(out[sid_ind, 3], cid)
        out[sid_ind, 4] = np.maximum(out[sid_ind, 4], cid)

    if do_count:
        seg_ui, seg_uc = np.unique(seg, return_counts=True)
        for i, j in zip(seg_ui, seg_uc):
            if i in sid_dict:
                out[sid_dict[i], -1] = j
    return out


def compute_bbox_all_3d(
    seg: np.ndarray, do_count: bool = False, uid: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """
    Compute the bounding boxes of 3D instance segmentation.

    Args:
        seg (numpy.ndarray): The 3D instance segmentation.
        do_count (bool, optional): Whether to compute the count of each instance. Defaults to False.
        uid (numpy.ndarray, optional): The unique identifier for each instance. Defaults to None.

    Returns:
        numpy.ndarray: The computed bounding boxes of the instances.

    Notes:
        - Each row in the output represents an instance and contains the following information:
            - seg id: The ID of the instance.
            - bounding box: The coordinates of the bounding box in the format
              [zmin, zmax, ymin, ymax, xmin, xmax].
            - count (optional): The count of voxels belonging to the instance.
        - The output only includes instances with valid bounding boxes.
    """

    sz = seg.shape
    assert len(sz) == 3, "Input segment should have 3 dimensions"
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid > 0]
    if len(uid) == 0:
        return None
    uid_max = int(uid.max())

    sid_dict = dict(zip(uid, range(len(uid))))
    out = np.zeros((len(uid), 7 + do_count), dtype=int)
    out[:, 0] = uid
    out[:, 1] = sz[0]
    out[:, 2] = -1
    out[:, 3] = sz[1]
    out[:, 4] = -1
    out[:, 5] = sz[2]
    out[:, 6] = -1

    # for each slice
    zids = np.where((seg > 0).sum(axis=1).sum(axis=1) > 0)[0]
    for zid in zids:
        sid = np.unique(seg[zid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        sid_ind = [sid_dict[x] for x in sid]
        out[sid_ind, 1] = np.minimum(out[sid_ind, 1], zid)
        out[sid_ind, 2] = np.maximum(out[sid_ind, 2], zid)

    # for each row
    rids = np.where((seg > 0).sum(axis=0).sum(axis=1) > 0)[0]
    for rid in rids:
        sid = np.unique(seg[:, rid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        sid_ind = [sid_dict[x] for x in sid]
        out[sid_ind, 3] = np.minimum(out[sid_ind, 3], rid)
        out[sid_ind, 4] = np.maximum(out[sid_ind, 4], rid)

    # for each col
    cids = np.where((seg > 0).sum(axis=0).sum(axis=0) > 0)[0]
    for cid in cids:
        sid = np.unique(seg[:, :, cid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        sid_ind = [sid_dict[x] for x in sid]
        out[sid_ind, 5] = np.minimum(out[sid_ind, 5], cid)
        out[sid_ind, 6] = np.maximum(out[sid_ind, 6], cid)

    if do_count:
        seg_ui, seg_uc = np.unique(seg, return_counts=True)
        for i, j in zip(seg_ui, seg_uc):
            if i in sid_dict:
                out[sid_dict[i], -1] = j
    return out
