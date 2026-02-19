from __future__ import print_function, division
from typing import Optional, Tuple, Dict

import numpy as np
import kimimaro
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.morphology import (
    remove_small_holes,
    binary_erosion,
    disk,
    ball,
)
from skimage.filters import gaussian

from .bbox_processor import BBoxProcessorConfig, BBoxInstanceProcessor
from .quantize import energy_quantize

__all__ = [
    "edt_semantic",
    "edt_instance",
    "distance_transform",
    "skeleton_aware_distance_transform",
    "smooth_edge",
    "signed_distance_transform",
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
    """Euclidean distance transform (EDT) for instance masks.

    Refactored to use BBoxInstanceProcessor for cleaner code and consistency.

    Args:
        label: Instance segmentation (H, W) or (D, H, W)
        bg_value: Background value for non-instance regions
        relabel: Whether to relabel connected components
        padding: Whether to pad before computing EDT
        resolution: Pixel/voxel resolution for anisotropic data
        erosion: Erosion kernel size (0 = no erosion)

    Returns:
        Normalized distance map with same shape as input
    """
    eps = 1e-6

    # Configure bbox processor
    config = BBoxProcessorConfig(
        bg_value=bg_value,
        relabel=relabel,
        padding=padding,
        pad_size=2,
        bbox_relax=1,
        combine_mode="max",
    )

    # Precompute erosion footprint (shared across instances)
    footprint = None
    if erosion > 0:
        footprint = disk(erosion) if label.ndim == 2 else ball(erosion)

    # Define per-instance EDT computation
    def compute_instance_edt(
        label_crop: np.ndarray, instance_id: int, bbox: Tuple[slice, ...], context: Dict
    ) -> Optional[np.ndarray]:
        """Compute normalized EDT for a single instance within bbox."""
        # Extract instance mask
        mask = binary_fill_holes((label_crop == instance_id))

        # Apply erosion if requested
        if context["footprint"] is not None:
            mask = binary_erosion(mask, context["footprint"])

        # Skip empty masks
        if not mask.any():
            return None

        # Compute EDT only within bbox
        boundary_edt = distance_transform_edt(mask, context["resolution"])
        edt_max = boundary_edt.max()

        if edt_max < eps:
            return None

        # Normalize and return
        energy = boundary_edt / (edt_max + eps)
        return energy * mask

    # Process all instances with bbox optimization
    processor = BBoxInstanceProcessor(config)
    return processor.process(
        label, compute_instance_edt, resolution=resolution, footprint=footprint
    )


def smooth_edge(binary, smooth_sigma: float = 2.0, smooth_threshold: float = 0.5):
    """Smooth the object contour."""
    for _ in range(2):
        binary = gaussian(binary, sigma=smooth_sigma, preserve_range=True)
        binary = (binary > smooth_threshold).astype(np.uint8)

    return binary


def signed_distance_transform(
    label: np.ndarray,
    resolution: Tuple[float] = (1.0, 1.0, 1.0),
    alpha: float = 8.0,
) -> np.ndarray:
    """Compute smooth signed distance transform for instance segmentation.
    
    This function produces a true signed distance transform where both foreground
    and background have meaningful gradient information, solving the class imbalance
    problem of traditional EDT approaches.
    
    Returns SDT in range [-1, 1]:
    - Positive values: inside instances (distance from boundary)
    - Negative values: outside instances (distance to nearest instance)
    - Zero: at instance boundaries
    
    Args:
        label: Instance segmentation (H, W) or (D, H, W)
        resolution: Pixel/voxel resolution for anisotropic data (z, y, x)
                   For 2D: (y, x)
                   For 3D: (z, y, x)
        alpha: Smoothness parameter for tanh normalization (default: 8.0)
               Higher values = sharper transitions at boundaries
               Lower values = smoother, more gradual transitions
    
    Returns:
        Signed distance transform in range [-1, 1] with same shape as input
        
    Example:
        >>> # 2D mitochondria segmentation
        >>> label_2d = np.array([[0, 0, 1, 1, 0],
        ...                       [0, 1, 1, 1, 0],
        ...                       [0, 0, 1, 0, 0]])
        >>> sdt_2d = signed_distance_transform(label_2d, resolution=(1.0, 1.0))
        >>> # sdt_2d will have positive values inside instance 1, negative outside
        
        >>> # 3D volume with anisotropic resolution
        >>> sdt_3d = signed_distance_transform(label_3d, resolution=(40, 16, 16), alpha=8.0)
    
    Notes:
        - This approach eliminates class imbalance by ensuring both foreground
          and background contribute meaningful gradients
        - The tanh normalization ensures smooth, bounded values suitable for
          regression with MSE loss
        - For mitochondria segmentation, typical alpha values are 6-10
    """
    # Adjust resolution for 2D vs 3D
    if label.ndim == 2:
        resolution = resolution[-2:] if len(resolution) > 2 else resolution
    elif label.ndim == 3:
        resolution = resolution if len(resolution) == 3 else (1.0, 1.0, 1.0)
    
    # Create binary masks
    foreground_mask = (label > 0).astype(np.uint8)
    background_mask = (label == 0).astype(np.uint8)
    
    # Compute EDT for both foreground and background
    if foreground_mask.any():
        foreground_edt = distance_transform_edt(foreground_mask, sampling=resolution)
    else:
        # No foreground - return all negative
        return -np.ones_like(label, dtype=np.float32)
    
    if background_mask.any():
        background_edt = distance_transform_edt(background_mask, sampling=resolution)
    else:
        # No background - return all positive
        return np.ones_like(label, dtype=np.float32)
    
    # Combine into signed distance
    # Positive inside instances, negative outside
    sdt = np.where(foreground_mask, foreground_edt, -background_edt)
    
    # Normalize to [-1, 1] using tanh
    # This provides smooth, bounded values suitable for regression
    sdt_normalized = np.tanh(sdt / alpha)
    
    return sdt_normalized.astype(np.float32)


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

    Refactored to use BBoxInstanceProcessor for cleaner code and consistency.
    Uses kimimaro for fast skeletonization (10-100x faster than scikit-image).

    Args:
        label: Instance segmentation (H, W) or (D, H, W)
        bg_value: Background value for non-instance regions
        relabel: Whether to relabel connected components
        padding: Whether to pad before computing distance
        resolution: Voxel resolution for anisotropic data (z, y, x)
        alpha: Skeleton influence exponent (higher = stronger skeleton influence)
        smooth: Whether to smooth edges before skeletonization
        smooth_skeleton_only: Only smooth skeleton mask (not entire object)

    Returns:
        Skeleton-aware distance map with same shape as input
    """
    eps = 1e-6

    # Fast-path: empty label should produce all background energy.
    if np.sum(label > 0) == 0:
        return np.full(label.shape, bg_value, dtype=np.float32)

    # Configure bbox processor
    config = BBoxProcessorConfig(
        bg_value=bg_value,
        relabel=relabel,
        padding=padding,
        pad_size=2,
        bbox_relax=2,
        combine_mode="max",
    )

    # Define per-instance skeleton EDT computation
    def compute_skeleton_edt(
        label_crop: np.ndarray, instance_id: int, bbox: Tuple[slice, ...], context: Dict
    ) -> Optional[np.ndarray]:
        """Compute skeleton-aware EDT for a single instance within bbox."""
        # Extract and clean mask
        temp2 = remove_small_holes(label_crop == instance_id, 16, connectivity=1)

        if not temp2.any():
            return None

        binary = temp2

        # Smooth if requested
        if context["smooth"]:
            binary_smooth = smooth_edge(binary.astype(np.uint8))
            if binary_smooth.astype(int).sum() > 32:
                if context["smooth_skeleton_only"]:
                    binary = binary_smooth.astype(bool) & temp2
                else:
                    binary = binary_smooth.astype(bool)
                    temp2 = binary

        # Skeletonize using kimimaro
        skeleton_mask = _skeletonize_instance(label_crop, instance_id, context["resolution"])

        # Fallback to regular EDT if skeletonization fails
        if skeleton_mask is None or not skeleton_mask.any():
            boundary_edt = distance_transform_edt(temp2, context["resolution"])
            edt_max = boundary_edt.max()
            if edt_max > eps:
                energy = (boundary_edt / (edt_max + eps)) ** context["alpha"]
                return energy * temp2.astype(np.float32)
            return None

        # Compute skeleton-aware EDT
        skeleton_edt = distance_transform_edt(~skeleton_mask, context["resolution"])
        boundary_edt = distance_transform_edt(temp2, context["resolution"])

        # Normalized energy
        energy = boundary_edt / (skeleton_edt + boundary_edt + eps)
        energy = energy ** context["alpha"]

        return energy * temp2.astype(np.float32)

    # Process all instances
    processor = BBoxInstanceProcessor(config)
    return processor.process(
        label,
        compute_skeleton_edt,
        resolution=resolution,
        alpha=alpha,
        smooth=smooth,
        smooth_skeleton_only=smooth_skeleton_only,
    )


def _skeletonize_instance(
    label_crop: np.ndarray, instance_id: int, resolution: Tuple[float, ...]
) -> Optional[np.ndarray]:
    """Helper function to skeletonize a single instance using kimimaro.

    Args:
        label_crop: Cropped label array containing the instance
        instance_id: ID of the instance to skeletonize
        resolution: Voxel resolution for anisotropic data

    Returns:
        Binary skeleton mask, or None if skeletonization fails
    """
    instance_label = np.where(label_crop == instance_id, 1, 0).astype(np.uint32)

    try:
        skeletons = kimimaro.skeletonize(
            instance_label,
            anisotropy=resolution,
            fix_branching=False,
            fix_borders=False,
            dust_threshold=5,
            parallel=1,
            progress=False,
        )

        if 1 in skeletons and len(skeletons[1].vertices) > 0:
            skeleton_mask = np.zeros(label_crop.shape, dtype=bool)
            vertices = skeletons[1].vertices.astype(int)

            # Filter valid vertices
            valid_mask = np.all(
                (vertices >= 0) & (vertices < np.array(skeleton_mask.shape)), axis=1
            )
            valid_vertices = vertices[valid_mask]

            if len(valid_vertices) > 0:
                if label_crop.ndim == 3:
                    skeleton_mask[
                        valid_vertices[:, 0], valid_vertices[:, 1], valid_vertices[:, 2]
                    ] = True
                else:
                    skeleton_mask[valid_vertices[:, 0], valid_vertices[:, 1]] = True
                return skeleton_mask

    except Exception:
        pass

    return None
