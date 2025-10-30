from __future__ import print_function, division
from typing import Optional, Tuple

import torch
import scipy
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.morphology import (
    remove_small_holes,
    skeletonize,
    binary_erosion,
    disk,
    ball,
)
from skimage.measure import label as label_cc  # avoid namespace conflict
from skimage.filters import gaussian

from .bbox import compute_bbox_all, bbox_relax
from .misc import get_padsize, array_unpad
from .quantize import energy_quantize

__all__ = [
    "edt_semantic",
    "edt_instance",
    "distance_transform",
    "skeleton_aware_distance_transform",
    "smooth_edge",
]


def edt_semantic(
    label: np.ndarray,
    mode: str = "2d",
    alpha_fore: float = 8.0,
    alpha_back: float = 50.0,
):
    """Euclidean distance transform (DT or EDT) for binary semantic mask.

    Optimizations:
    - Preallocate arrays for 2D mode
    - Avoid list append + stack pattern
    - Compute masks once
    """
    assert mode in ["2d", "3d"]
    do_2d = label.ndim == 2

    resolution = (6.0, 1.0, 1.0)  # anisotropic data
    if mode == "2d" or do_2d:
        resolution = (1.0, 1.0)

    # Compute masks once (avoid redundant comparisons)
    fore = (label != 0).astype(np.uint8)
    back = (label == 0).astype(np.uint8)

    if mode == "3d" or do_2d:
        fore_edt = _edt_binary_mask(fore, resolution, alpha_fore)
        back_edt = _edt_binary_mask(back, resolution, alpha_back)
    else:
        # Optimized 2D mode: preallocate instead of list comprehension
        n_slices = label.shape[0]
        fore_edt = np.zeros_like(fore, dtype=np.float32)
        back_edt = np.zeros_like(back, dtype=np.float32)

        for i in range(n_slices):
            fore_edt[i] = _edt_binary_mask(fore[i], resolution, alpha_fore)
            back_edt[i] = _edt_binary_mask(back[i], resolution, alpha_back)

    distance = fore_edt - back_edt
    return np.tanh(distance)


def _edt_binary_mask(mask, resolution, alpha):
    if (mask == 1).all():  # tanh(5) = 0.99991
        return np.ones_like(mask).astype(float) * 5

    return distance_transform_edt(mask, resolution) / alpha


def edt_instance(
    label: np.ndarray,
    mode: str = "2d",
    quantize: bool = True,
    resolution: Tuple[float] = (1.0, 1.0, 1.0),
    padding: bool = False,
    erosion: int = 0,
):
    assert mode in ["2d", "3d"]
    if mode == "3d":
        # calculate 3d distance transform for instances
        vol_distance = distance_transform(
            label, resolution=resolution, padding=padding, erosion=erosion
        )
        if quantize:
            vol_distance = energy_quantize(vol_distance)
        return vol_distance

    # Optimized 2D mode: preallocate arrays instead of lists
    vol_distance = np.zeros(label.shape, dtype=np.float32)

    # Process slices without copying (use view instead)
    for i in range(label.shape[0]):
        distance = distance_transform(
            label[i],  # No .copy() - distance_transform doesn't modify input
            padding=padding,
            erosion=erosion,
        )
        vol_distance[i] = distance

    if quantize:
        vol_distance = energy_quantize(vol_distance)

    return vol_distance


def distance_transform(
    label: np.ndarray,
    bg_value: float = -1.0,
    relabel: bool = True,
    padding: bool = False,
    resolution: Tuple[float] = (1.0, 1.0),
    erosion: int = 0,
):
    """Euclidean distance transform (DT or EDT) for instance masks.

    Optimizations:
    - Preallocate arrays with correct dtype
    - Avoid repeated dtype conversions
    - Cache footprint creation outside loop
    - Use in-place operations where possible
    """
    eps = 1e-6
    pad_size = 2

    if relabel:
        label = label_cc(label)

    if padding:
        label = np.pad(label, pad_size, mode="constant", constant_values=0)

    label_shape = label.shape

    # Preallocate with correct dtypes
    distance = np.zeros(label_shape, dtype=np.float32)

    if label.max() > 0:
        # Cache footprint creation (expensive operation)
        footprint = None
        if erosion > 0:
            footprint = disk(erosion) if label.ndim == 2 else ball(erosion)

        # Compute all bounding boxes at once and process each instance
        bbox_array = compute_bbox_all(label, do_count=False)

        # Process each instance with EDT computation
        if bbox_array is not None:
            for i in range(bbox_array.shape[0]):
                idx = int(bbox_array[i, 0])

                # Extract bbox coords and apply 1-pixel relaxation (for EDT smoothness)
                if label.ndim == 2:
                    # 2D: [id, y_min, y_max, x_min, x_max]
                    bbox_coords = [
                        bbox_array[i, 1],
                        bbox_array[i, 2] + 1,  # +1 for exclusive max
                        bbox_array[i, 3],
                        bbox_array[i, 4] + 1,
                    ]
                    relaxed = bbox_relax(bbox_coords, label_shape, relax=1)
                    bbox = (
                        slice(relaxed[0], relaxed[1]),
                        slice(relaxed[2], relaxed[3]),
                    )
                else:  # 3D
                    # 3D: [id, z_min, z_max, y_min, y_max, x_min, x_max]
                    bbox_coords = [
                        bbox_array[i, 1],
                        bbox_array[i, 2] + 1,  # +1 for exclusive max
                        bbox_array[i, 3],
                        bbox_array[i, 4] + 1,
                        bbox_array[i, 5],
                        bbox_array[i, 6] + 1,
                    ]
                    relaxed = bbox_relax(bbox_coords, label_shape, relax=1)
                    bbox = (
                        slice(relaxed[0], relaxed[1]),
                        slice(relaxed[2], relaxed[3]),
                        slice(relaxed[4], relaxed[5]),
                    )

                # Extract instance mask within bounding box
                temp2 = binary_fill_holes((label[bbox] == idx))

                if erosion > 0:
                    temp2 = binary_erosion(temp2, footprint)

                # Skip empty masks
                if not temp2.any():
                    continue

                # Compute EDT only within bounding box (MAJOR SPEEDUP)
                boundary_edt = distance_transform_edt(temp2, resolution)
                edt_max = boundary_edt.max()

                if edt_max > eps:  # Avoid division by zero
                    # Normalize and update distance map (only within bbox)
                    energy = boundary_edt / (edt_max + eps)
                    # Direct multiplication (avoid intermediate array)
                    distance[bbox] = np.maximum(distance[bbox], energy * temp2)

    # Apply background value
    if bg_value != 0:
        distance[distance == 0] = bg_value

    if padding:
        pad_tuple = get_padsize(pad_size, ndim=distance.ndim)
        distance = array_unpad(distance, pad_tuple)

    return distance


def smooth_edge(binary, smooth_sigma: float = 2.0, smooth_threshold: float = 0.5):
    """Smooth the object contour."""
    for _ in range(2):
        binary = gaussian(binary, sigma=smooth_sigma, preserve_range=True)
        binary = (binary > smooth_threshold).astype(np.uint8)

    return binary


def skeleton_aware_distance_transform(
    label: np.ndarray,
    bg_value: float = -1.0,
    relabel: bool = True,
    padding: bool = False,
    resolution: Tuple[float] = (1.0, 1.0, 1.0),
    alpha: float = 0.8,
    smooth: bool = True,
    smooth_skeleton_only: bool = True,
):
    """Skeleton-based distance transform (SDT).

    Lin, Zudi, et al. "Structure-Preserving Instance Segmentation via Skeleton-Aware
    Distance Transform." International Conference on Medical Image Computing and
    Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2023.
    """
    eps = 1e-6
    pad_size = 2

    if relabel:
        label = label_cc(label)

    if padding:
        # The distance_transform_edt function does not treat image border
        # as background. If image border needs to be considered as background
        # in distance calculation, set padding to True.
        label = np.pad(label, pad_size, mode="constant", constant_values=0)

    label_shape = label.shape

    skeleton = np.zeros(label_shape, dtype=np.uint8)
    distance = np.zeros(label_shape, dtype=np.float32)

    indices = np.unique(label)
    if len(indices) > 1:
        for idx in indices[indices > 0]:
            temp2 = remove_small_holes(label == idx, 16, connectivity=1)
            binary = temp2.copy()

            if smooth:
                binary = smooth_edge(binary)
                if binary.astype(int).sum() <= 32:
                    # Reverse the smoothing operation if it makes
                    # the output mask empty (or very small).
                    binary = temp2.copy()
                else:
                    if smooth_skeleton_only:
                        binary = binary * temp2
                    else:
                        temp2 = binary.copy()

            skeleton_mask = skeletonize(binary)
            skeleton_mask = (skeleton_mask != 0).astype(np.uint8)
            skeleton += skeleton_mask

            skeleton_edt = distance_transform_edt(1 - skeleton_mask, resolution)
            boundary_edt = distance_transform_edt(temp2, resolution)

            energy = boundary_edt / (skeleton_edt + boundary_edt + eps)  # normalize
            energy = energy**alpha
            distance = np.maximum(distance, energy * temp2.astype(np.float32))

    if bg_value != 0:
        distance[distance == 0] = bg_value

    if padding:
        # Unpad the output array to preserve original shape.
        distance = array_unpad(distance, get_padsize(pad_size, ndim=distance.ndim))

    return distance
