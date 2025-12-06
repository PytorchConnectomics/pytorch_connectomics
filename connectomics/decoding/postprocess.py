"""
General post-processing utilities for segmentation refinement.

This module provides utilities for:
    - Filtering (binarize_and_median)
    - Instance editing (remove/add/merge masks, watershed split)
    - 2D to 3D stitching (stitch_3d)
    - IoU calculations (intersection_over_union)
"""

from __future__ import print_function, division, annotations
from typing import List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from connectomics.config import BinaryPostprocessingConfig

from scipy import ndimage

# from skimage.measure import label  # Replaced with cc3d.connected_components
import cc3d
from skimage.feature import peak_local_max

# from skimage.segmentation import watershed  # Replaced with mahotas for better performance
import mahotas

from connectomics.data.process import bbox_ND, crop_ND, replace_ND


__all__ = [
    "binarize_and_median",
    "remove_masks",
    "add_masks",
    "merge_masks",
    "watershed_split",
    "stitch_3d",
    "intersection_over_union",
    "apply_binary_postprocessing",
]


def binarize_and_median(
    pred: np.ndarray, size: tuple = (7, 7, 7), thres: float = 0.8
) -> np.ndarray:
    """First binarize the prediction with a given threshold, and
    then conduct median filtering to reduce noise.

    Args:
        pred (numpy.ndarray): predicted foreground probability within (0,1).
        size (tuple): kernel size of filtering. Default: (7,7,7)
        thres (float): threshold for binarizing the prediction. Default: 0.8

    Returns:
        numpy.ndarray: Filtered binary mask.
    """
    pred = (pred > thres).astype(np.uint8)
    pred = ndimage.median_filter(pred, size=size)
    return pred


def remove_masks(vol: np.ndarray, indices: List[int]) -> np.ndarray:
    """Remove objects by indices from a segmentation volume.

    Args:
        vol (numpy.ndarray): Segmentation volume.
        indices (list): List of object IDs to remove.

    Returns:
        numpy.ndarray: Volume with specified objects removed.
    """
    for idx in indices:
        vol[np.where(vol == idx)] = 0
    return vol


def add_masks(vol_base: np.ndarray, vol: np.ndarray, indices: List[int]) -> np.ndarray:
    """Add the instances in a new segmentation volume to the
    original one. A new instance can overwrite existing object
    pixels if the corresponding region contains non-background.

    Args:
        vol_base (numpy.ndarray): Base segmentation volume.
        vol (numpy.ndarray): Volume containing new objects to add.
        indices (list): List of object IDs from vol to add.

    Returns:
        numpy.ndarray: Combined segmentation volume.
    """
    max_idx = max(np.unique(vol_base))
    for i, idx in enumerate(indices):
        vol_base[np.where(vol == idx)] = max_idx + i + 1
    return vol_base


def merge_masks(vol: np.ndarray, indices: List[List[int]]) -> np.ndarray:
    """Merge two or more masks into a single one.

    Args:
        vol (numpy.ndarray): Segmentation volume.
        indices (list of lists): Each inner list contains IDs to merge together.

    Returns:
        numpy.ndarray: Volume with merged objects.
    """
    for merges in indices:
        temp = np.zeros_like(vol)
        for i, idx in enumerate(merges):
            if i == 0:
                main_idx = idx
            temp = temp + (vol == idx).astype(temp.dtype)
        vol[np.where(temp != 0)] = main_idx
    return vol


def watershed_split(
    vol: np.ndarray, index: int, show_id: bool = False, min_distance: int = 5
) -> np.ndarray:
    """Apply watershed transform to split a 3D object into two or more
    parts based on the given index.

    Args:
        vol (numpy.ndarray): 3D label array.
        index (int): ID of the object to split.
        show_id (bool): Whether to print the new IDs. Default: False
        min_distance (int): Minimum distance between peaks. Default: 5

    Returns:
        numpy.ndarray: Volume with split object.

    References:
        https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
    """
    assert vol.ndim == 3  # 3D label array
    max_idx = max(np.unique(vol))
    binary = vol == index
    bbox = bbox_ND(binary, relax=1)  # avoid cropped object touching borders
    cropped = crop_ND(binary, bbox, end_included=True)

    # see https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
    distance = ndimage.distance_transform_edt(cropped)
    coords = peak_local_max(distance, min_distance=min_distance, labels=cropped)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = cc3d.connected_components(mask)
    split_objects = mahotas.cwatershed(-distance, markers)
    split_objects[~cropped] = (
        0  # Apply mask manually (mahotas 1.4.18 doesn't support mask parameter)
    )

    seg_id = np.unique(split_objects)
    new_id = []
    if seg_id[0] == 0:
        seg_id = seg_id[1:]  # ignore background pixels
    for i, idx in enumerate(seg_id):
        split_objects[np.where(split_objects == idx)] = max_idx + i + 1
        new_id.append(max_idx + i + 1)
    if show_id:
        print(new_id)

    vol = replace_ND(vol, split_objects, bbox, end_included=True)
    return vol


def stitch_3d(masks: np.ndarray, stitch_threshold: float = 0.25) -> np.ndarray:
    r"""Takes a volume stack of 2D annotations and stitches into 3D annotations using IOU.

    Args:
        masks (numpy.ndarray): 3D volume comprised of a 2D annotations stack
            of shape:math:`(Z, Y, X)`.
        stitch_threshold (float): threshold for joining 2D annotations via IOU. Default: 0.25

    Returns:
        numpy.ndarray: 3D stitched segmentation.
    """
    mmax = masks[0].max()
    empty = 0

    for i in range(len(masks) - 1):
        # retrieve all intersecting pairs, discard background
        iou = intersection_over_union(masks[i + 1], masks[i])[1:, 1:]
        if not iou.size and empty == 0:
            mmax = masks[i + 1].max()
        elif not iou.size and not empty == 0:
            icount = masks[i + 1].max()
            istitch = np.arange(mmax + 1, mmax + icount + 1, 1, int)
            mmax += icount
            istitch = np.append(np.array(0), istitch)
            masks[i + 1] = istitch[masks[i + 1]]
        else:
            # set all iou value that did not breach the threshold to zero
            iou[iou < stitch_threshold] = 0.0
            # we calculated the IoU for each possible masks pair
            # for each mask only consider the pairing with the greatest IoU
            iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1) == 0.0)[0]
            istitch[ino] = np.arange(mmax + 1, mmax + len(ino) + 1, 1, int)
            mmax += len(ino)
            istitch = np.append(np.array(0), istitch)
            masks[i + 1] = istitch[masks[i + 1]]
            empty = 1

    return masks


def intersection_over_union(masks_true: np.ndarray, masks_pred: np.ndarray) -> np.ndarray:
    """Calculates the intersection over union for all mask pairs.

    Abducted from the cellpose repository
    (https://github.com/MouseLand/cellpose/blob/master/cellpose/metrics.py).

    Args:
        masks_true (numpy.ndarray): 2D label array where 0=NO masks; 1,2... are
            mask labels, shape:math:`(Y, X)`.
        masks_pred (numpy.ndarray): 2D label array where 0=NO masks; 1,2... are
            mask labels, shape:math:`(Y, X)`.

    Returns:
        numpy.ndarray: A ND-array recording the IoU score (float) for each label pair,
                       size [masks_true.max()+1, masks_pred.max()+1]
    """
    overlap = _label_overlap(masks_true, masks_pred)

    # index wise encoding of how often a predicted label coincides with true labels
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    # index wise encoding of how often a true label coincides with predicted labels
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)

    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def _label_overlap(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Creates a look up table that records the pixel overlap
    between two 2D label arrays.

    Args:
        x (numpy.ndarray): 2D label array where 0=NO masks; 1,2... are mask
            labels, shape:math:`(Y, X)`.
        y (numpy.ndarray): 2D label array where 0=NO masks; 1,2... are mask
            labels, shape:math:`(Y, X)`.

    Returns:
        numpy.ndarray: A ND-array matrix recording the pixel overlaps,
            size:math:`[x.max()+1, y.max()+1]`
    """
    # flatten the 2D label arrays
    x = x.ravel()
    y = y.ravel()

    assert len(x) == len(y), "The label masks must have the same shape"

    # initialize the lookup table
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)

    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def apply_binary_postprocessing(
    pred: np.ndarray, config: "BinaryPostprocessingConfig"
) -> np.ndarray:
    """Apply binary segmentation postprocessing pipeline.

    Pipeline order:
        1. Ensure input is binary (convert if needed)
        2. Apply median filter (optional)
        3. Apply morphological opening (erosion + dilation)
        4. Apply morphological closing (dilation + erosion)
        5. Extract connected components and filter by size/keep top-k

    Args:
        pred (numpy.ndarray): Binary mask (values 0 or 1) or predicted
            probabilities in range [0, 1]. Shape can be 2D (H, W) or 3D (D, H, W).
        config (BinaryPostprocessingConfig): Configuration for postprocessing pipeline.

    Returns:
        numpy.ndarray: Postprocessed binary mask (same shape as input).
                       Values: 0 (background) or 1 (foreground).

    Example:
        >>> from connectomics.config import BinaryPostprocessingConfig, ConnectedComponentsConfig
        >>> config = BinaryPostprocessingConfig(
        ...     enabled=True,
        ...     opening_iterations=2,
        ...     connected_components=ConnectedComponentsConfig(top_k=1)
        ... )
        >>> pred = np.random.rand(128, 128)  # Random probabilities
        >>> binary_mask = apply_binary_postprocessing(pred, config)
    """
    if not config or not config.enabled:
        # If no postprocessing, ensure binary output
        if pred.max() <= 1:
            return (pred > 0.5).astype(np.uint8)
        else:
            return (pred > 0).astype(np.uint8)

    # Step 1: Ensure input is binary
    # Check if input is already binary (0/1) or needs thresholding
    if np.all((pred == 0) | (pred == 1)):
        # Already binary
        binary = pred.astype(np.uint8)
    else:
        # Determine threshold value based on value range
        if pred.max() <= 1.0:
            # Probability values in [0, 1], default threshold at 0.5
            threshold = 0.5
        else:
            # Values > 1, default threshold at 0
            threshold = 0.0

        # Apply threshold
        binary = (pred >= threshold).astype(np.uint8)

    # Step 2: Apply median filter (optional noise reduction)
    if config.median_filter_size is not None:
        binary = ndimage.median_filter(binary, size=config.median_filter_size)

    # Step 3: Morphological opening (erosion + dilation) - removes small objects
    if config.opening_iterations > 0:
        binary = ndimage.binary_opening(binary, iterations=config.opening_iterations).astype(
            np.uint8
        )

    # Step 4: Morphological closing (dilation + erosion) - fills small holes
    if config.closing_iterations > 0:
        binary = ndimage.binary_closing(binary, iterations=config.closing_iterations).astype(
            np.uint8
        )

    # Step 5: Connected components filtering
    if config.connected_components is not None and config.connected_components.enabled:
        cc_config = config.connected_components

        # Extract connected components
        connectivity = cc_config.connectivity
        labels = cc3d.connected_components(binary, connectivity=connectivity)

        # Get component sizes
        component_sizes = np.bincount(labels.ravel())
        # Skip background (label 0)
        component_sizes[0] = 0

        # Filter by minimum size
        if cc_config.min_size > 0:
            small_components = np.where(component_sizes < cc_config.min_size)[0]
            for label_id in small_components:
                labels[labels == label_id] = 0

        # Keep only top-k largest components
        if cc_config.top_k is not None and cc_config.top_k > 0:
            # Get sizes (excluding background)
            sizes = component_sizes[1:]  # Skip background
            label_ids = np.arange(1, len(component_sizes))

            if len(sizes) > cc_config.top_k:
                # Get indices of top-k largest components
                top_k_indices = np.argsort(sizes)[-cc_config.top_k :]
                top_k_labels = label_ids[top_k_indices]

                # Create mask keeping only top-k
                keep_mask = np.zeros_like(labels, dtype=bool)
                for label_id in top_k_labels:
                    keep_mask |= labels == label_id

                labels = keep_mask.astype(np.uint8)
            else:
                # Convert labels back to binary (0/1)
                labels = (labels > 0).astype(np.uint8)
        else:
            # Convert labels back to binary (0/1)
            labels = (labels > 0).astype(np.uint8)

        binary = labels

    return binary
