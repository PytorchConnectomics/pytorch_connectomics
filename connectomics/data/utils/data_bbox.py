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


def crop_ND(img: np.ndarray, coord: Tuple[int], 
            end_included: bool = False) -> np.ndarray:
    """Crop a chunk from a N-dimensional array based on the 
    bounding box coordinates.
    """
    N = img.ndim
    assert len(coord) == N * 2
    slicing = []
    for i in range(N):
        start = coord[2*i]
        end = coord[2*i+1] + 1 if end_included else coord[2*i+1]
        slicing.append(slice(start, end))
    slicing = tuple(slicing)
    return img[slicing].copy()


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
