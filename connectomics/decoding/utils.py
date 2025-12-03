"""
Shared utility functions for decoding operations.

This module provides common utilities used across different decoding functions:
    - cast2dtype: Optimal dtype casting for segmentation masks
    - remove_small_instances: Remove small spurious instances
    - remove_large_instances: Remove large instances
    - merge_small_objects: Merge small objects with neighbors
"""

from __future__ import print_function, division
import numpy as np

from skimage.morphology import dilation, remove_small_objects

from connectomics.data.process import get_seg_type, bbox_ND, crop_ND


__all__ = [
    "cast2dtype",
    "remove_small_instances",
    "remove_large_instances",
    "merge_small_objects",
]


def cast2dtype(segm: np.ndarray) -> np.ndarray:
    """Cast the segmentation mask to the best dtype to save storage.

    Args:
        segm (numpy.ndarray): Segmentation mask.

    Returns:
        numpy.ndarray: Segmentation mask with optimal dtype.
    """
    max_id = np.amax(np.unique(segm))
    m_type = get_seg_type(int(max_id))
    return segm.astype(m_type)


def remove_small_instances(
    segm: np.ndarray, thres_small: int = 128, mode: str = "background"
) -> np.ndarray:
    """Remove small spurious instances.

    Args:
        segm (numpy.ndarray): Segmentation mask.
        thres_small (int): Size threshold for small objects. Default: 128
        mode (str): Removal mode. Options: 'none', 'background', 'background_2d',
                    'neighbor', 'neighbor_2d'. Default: 'background'

    Returns:
        numpy.ndarray: Segmentation mask with small instances removed.

    Note:
        The function remove_small_objects expects ar to be an array with labeled objects, and
        removes objects smaller than min_size. If ar is bool, the image is first labeled. This
        leads to potentially different behavior for bool and 0-and-1 arrays. Reference:
        https://scikit-image.org/docs/stable/api/skimage.morphology.html#remove-small-objects
    """
    assert mode in ["none", "background", "background_2d", "neighbor", "neighbor_2d"]

    if mode == "none":
        return segm

    if mode == "background":
        return remove_small_objects(segm, thres_small)
    elif mode == "background_2d":
        temp = [remove_small_objects(segm[i], thres_small) for i in range(segm.shape[0])]
        return np.stack(temp, axis=0)

    if mode == "neighbor":
        return merge_small_objects(segm, thres_small, do_3d=True)
    elif mode == "neighbor_2d":
        temp = [merge_small_objects(segm[i], thres_small) for i in range(segm.shape[0])]
        return np.stack(temp, axis=0)


def merge_small_objects(segm: np.ndarray, thres_small: int, do_3d: bool = False) -> np.ndarray:
    """Merge small objects with their neighbors.

    Args:
        segm (numpy.ndarray): Segmentation mask.
        thres_small (int): Size threshold for small objects.
        do_3d (bool): Whether to use 3D structuring element. Default: False

    Returns:
        numpy.ndarray: Segmentation mask with small objects merged.
    """
    struct = np.ones((1, 3, 3)) if do_3d else np.ones((3, 3))
    indices, counts = np.unique(segm, return_counts=True)

    for i in range(len(indices)):
        idx = indices[i]
        if counts[i] < thres_small:
            temp = (segm == idx).astype(np.uint8)
            coord = bbox_ND(temp, relax=2)
            cropped = crop_ND(temp, coord)

            diff = dilation(cropped, struct) - cropped
            diff_segm = crop_ND(segm, coord)
            diff_segm[np.where(diff == 0)] = 0

            u, ct = np.unique(diff_segm, return_counts=True)
            if len(u) > 1 and u[0] == 0:
                u, ct = u[1:], ct[1:]

            segm[np.where(segm == idx)] = u[np.argmax(ct)]

    return segm


def remove_large_instances(segm: np.ndarray, max_size: int = 2000) -> np.ndarray:
    """Remove large instances given a maximum size threshold.

    Args:
        segm (numpy.ndarray): Segmentation mask.
        max_size (int): Maximum size threshold. Default: 2000

    Returns:
        numpy.ndarray: Segmentation mask with large instances removed.
    """
    out = np.copy(segm)
    component_sizes = np.bincount(segm.ravel())
    too_large = component_sizes > max_size
    too_large_mask = too_large[segm]
    out[too_large_mask] = 0
    return out
