from __future__ import print_function, division
from typing import Optional, Union, List, Tuple

import itertools
import numpy as np
from tqdm import tqdm
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


def crop_ND(img: np.ndarray, coord: Tuple[int]) -> np.ndarray:
    N = img.ndim
    assert len(coord) == N * 2
    slicing = []
    for i in range(N):
        slicing.append(slice(coord[2*i], coord[2*i+1]))
    slicing = tuple(slicing)
    return img[slicing].copy()


def index2bbox(seg: np.ndarray, indices: list, relax: int = 0,
               progress: bool = False, iterative: bool = False) -> dict:
    bbox_dict = {}

    # calculate the bounding boxes using scipy.ndimage.find_objects
    if not iterative:
        loc = find_objects(seg)
        seg_shape = seg.shape
        for idx, item in enumerate(loc):
            if item is None:
                # For scipy.ndimage.find_objects, if a number is 
                # missing, None is returned instead of a slice.
                continue
            bbox = []
            for x in item: # slice() object
                bbox.append(x.start)
                bbox.append(x.stop-1) # bbox is inclusive by definition
            object_idx = idx + 1 # 0 is ignored
            bbox_dict[object_idx] = bbox_relax(bbox, seg_shape, relax)
        return bbox_dict

    # calculate the bounding boxes of each segment iteratively
    for idx in (tqdm(indices) if progress else indices):
        temp = (seg == idx) # binary mask of the current seg
        bbox = bbox_ND(temp, relax=relax)
        bbox_dict[idx] = bbox
    return bbox_dict


def diff_segm(seg1: np.ndarray, seg2: np.ndarray, iou_thres: float = 0.75, 
              progress: bool = False) -> dict:
    """Check the differences between two 3D instance segmentation maps. The 
    background pixels (value=0) are ignored.

    Args:
        seg1 (np.ndarray): the first segmentation map.
        seg2 (np.ndarray): the second segmentation map.
        iou_thres (float): the threshold of intersection-over-union. Default: 0.75
        progress (bool): show progress bar. Default: False

    Returns:
        dict: a dict contains lists of shared and unique indicies

    Note:
        The shared segments in two segmentation maps can have different indices,
        therefore they are saved separately in the output dict.
    """
    def _get_indices_counts(seg: np.ndarray):
        # return indices and counts while ignoring the background
        indices, counts = np.unique(seg, return_counts=True)
        if indices[0] == 0:
            return indices[1:], counts[1:]
        else:
            return indices, counts

    results ={
        "seg1_unique": [],
        "seg2_unique": [],
        "shared1": [],
        "shared2": [],
    }

    indices1, counts1 = _get_indices_counts(seg1)
    indices2, counts2 = _get_indices_counts(seg2)
    if len(indices1) == 0: # no non-background objects
        results["seg2_unique"] = list(indices2)
        return results
    if len(indices2) == 0:
        results["seg1_unique"] = list(indices1)
        return results
    
    counts_dict1 = dict(zip(indices1, counts1))
    counts_dict2 = dict(zip(indices2, counts2))
    bbox_dict1 = index2bbox(seg1, indices1, relax=1, progress=progress)

    for idx1 in (tqdm(indices1) if progress else indices1):
        bbox = bbox_dict1[idx1]
        crop_seg1, crop_seg2 = crop_ND(seg1, bbox), crop_ND(seg2, bbox)
        temp1 = (crop_seg1==idx1).astype(int)

        best_iou = 0.0
        crop_indices = np.unique(crop_seg2)
        for idx2 in crop_indices:
            if idx2 == 0: # ignore background
                continue 
            temp2 = (crop_seg2==idx2).astype(int)
            overlap = (temp1*temp2).sum()
            union = counts_dict1[idx1] + counts_dict2[idx2] - overlap
            iou = overlap / float(union)
            if iou > best_iou:
                best_iou = iou
                matched_idx2 = idx2

        if best_iou < iou_thres:
            results["seg1_unique"].append(idx1)
        else: # the segment is shared in both segmentation maps
            results["shared1"].append(idx1)
            results["shared2"].append(matched_idx2)

    # "seg2_unique" contains elements in indices2 but not in "shared2"
    results["seg2_unique"] = list(set(indices2) - set(results["shared2"]))
    return results
