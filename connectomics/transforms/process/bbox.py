from __future__ import division, print_function
from collections import OrderedDict
from typing import Optional, Union, List, Tuple

import itertools
import numpy as np
from scipy.ndimage import find_objects


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


def bbox_relax(coord: Union[tuple, list], 
               shape: tuple, 
               relax: int = 0) -> tuple:
    assert len(coord) == len(shape) * 2
    coord = list(coord)
    for i in range(len(shape)):
        coord[2*i] = max(0, coord[2*i]-relax)
        coord[2*i+1] = min(shape[i], coord[2*i+1]+relax)

    return tuple(coord)


def adjust_bbox(low, high, sz):
    assert high >= low
    bbox_sz = high - low
    diff = abs(sz - bbox_sz) // 2
    if bbox_sz >= sz:
        return low + diff, low + diff + sz

    return low - diff, low - diff + sz


def index2bbox(seg: np.ndarray, indices: list, relax: int = 0,
               iterative: bool = False) -> dict:
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
            temp = (seg == idx) # binary mask of the current seg
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

        object_idx = idx + 1 # 0 is ignored in find_objects
        if object_idx not in indices:
            continue

        bbox = []
        for x in item: # slice() object
            bbox.append(x.start)
            bbox.append(x.stop-1) # bbox is inclusive by definition
        bbox_dict[object_idx] = bbox_relax(bbox, seg_shape, relax)
    return bbox_dict


def _coord2slice(coord: Tuple[int], ndim: int, end_included: bool = False):
    assert len(coord) == ndim * 2
    slicing = []
    for i in range(ndim):
        start = coord[2*i]
        end = coord[2*i+1] + 1 if end_included else coord[2*i+1]
        slicing.append(slice(start, end))
    slicing = tuple(slicing)
    return slicing


def crop_ND(img: np.ndarray, coord: Tuple[int], 
            end_included: bool = False) -> np.ndarray:
    """Crop a chunk from a N-dimensional array based on the 
    bounding box coordinates.
    """
    slicing = _coord2slice(coord, img.ndim, end_included)
    return img[slicing].copy()


def replace_ND(img: np.ndarray, replacement: np.ndarray, coord: Tuple[int], 
               end_included: bool = False, overwrite_bg: bool = False) -> np.ndarray:
    """Replace a chunk from a N-dimensional array based on the 
    bounding box coordinates.
    """
    slicing = _coord2slice(coord, img.ndim, end_included)

    if not overwrite_bg: # only overwrite foreground pixels
        temp = img[slicing].copy()
        mask_fg = (replacement!=0).astype(temp.dtype)
        mask_bg = (replacement==0).astype(temp.dtype)
        replacement = replacement * mask_fg + temp * mask_bg

    img[slicing] = replacement
    return img.copy()


def crop_pad_data(data, z, bbox_2d, pad_val=0, mask=None, return_box=False):
    """Crop a 2D patch from 3D volume given the z index and 2d bbox.
    """
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
        cropped = np.pad(cropped, pad, mode='constant',
                         constant_values=pad_val)

    if not return_box:
        return cropped

    return cropped, [y1, y2, x1, x2], pad


def rand_window(w0, w1, sz, rand_shift: int = 0):
    assert (w1 >= w0)
    diff = np.abs((w1-w0)-sz)
    if (w1-w0) <= sz:
        if rand_shift > 0: # random shift augmentation
            start_l = max(w0-diff//2-rand_shift, w1-sz)
            start_r = min(w0, w0-diff//2+rand_shift)
            low = np.random.randint(start_l, start_r)
        else:
            low = w0 - diff//2 
    else:
        if rand_shift > 0: # random shift augmentation
            start_l = max(w0, w0+diff//2-rand_shift)
            start_r = min(w0+diff//2+rand_shift, w1-sz)
            low = np.random.randint(start_l, start_r)
        else:
            low = w0 + diff//2 
    high = low + sz    
    return low, high
