from __future__ import annotations

from typing import Dict, Optional, Tuple

import kimimaro
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.filters import gaussian
from skimage.morphology import (
    ball,
    binary_erosion,
    disk,
    remove_small_holes,
)

import cc3d

from .bbox_processor import BBoxInstanceProcessor, BBoxProcessorConfig
from .quantize import energy_quantize

__all__ = [
    "edt_semantic",
    "edt_instance",
    "distance_transform",
    "skeleton_aware_distance_transform",
    "precompute_sdt_volume",
    "smooth_edge",
    "signed_distance_transform",
]


def edt_semantic(
    label: np.ndarray,
    mode: str = "2d",
    alpha_fore: float = 8.0,
    alpha_back: float = 50.0,
    resolution: Tuple[float, ...] = None,
):
    """Euclidean distance transform (DT or EDT) for binary semantic mask.

    Args:
        label: Binary or instance segmentation array
        mode: '2d' for slice-by-slice or '3d' for full volume
        alpha_fore: Foreground distance normalization factor
        alpha_back: Background distance normalization factor
        resolution: Voxel resolution. Default: (1.0, 1.0) for 2D, (1.0, 1.0, 1.0) for 3D
    """
    if mode not in ("2d", "3d"):
        raise ValueError(f"mode must be '2d' or '3d', got {mode!r}")
    do_2d = label.ndim == 2

    if resolution is None:
        resolution = (1.0, 1.0) if (mode == "2d" or do_2d) else (1.0, 1.0, 1.0)
    elif mode == "2d" and not do_2d and len(resolution) == 3:
        resolution = resolution[-2:]

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
    quantize: bool = False,
    resolution: Tuple[float] = (1.0, 1.0, 1.0),
    padding: bool = False,
    erosion: int = 0,
    bg_value: float = -1.0,
):
    if mode not in ("2d", "3d"):
        raise ValueError(f"mode must be '2d' or '3d', got {mode!r}")
    if mode == "3d":
        # calculate 3d distance transform for instances
        vol_distance = distance_transform(
            label, resolution=resolution, padding=padding, erosion=erosion
        )
        if quantize:
            vol_distance = energy_quantize(vol_distance)
        return vol_distance

    # Optimized 2D mode: preallocate arrays instead of lists
    vol_distance = np.full(label.shape, bg_value, dtype=np.float32)

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
        mask = binary_fill_holes(label_crop == instance_id)

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
    relabel: bool = False,
    padding: bool = False,
    resolution: Tuple[float] = (1.0, 1.0, 1.0),
    alpha: float = 0.8,
    smooth: bool = False,
    smooth_skeleton_only: bool = True,
):
    """Skeleton-based distance transform (SDT).

    Lin, Zudi, et al. "Structure-Preserving Instance Segmentation via Skeleton-Aware
    Distance Transform." International Conference on Medical Image Computing and
    Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2023.

    Uses batch kimimaro skeletonization: all instances are skeletonized in a single
    call with automatic parallelism, then per-instance EDT is computed via
    BBoxInstanceProcessor.

    Args:
        label: Instance segmentation (H, W) or (D, H, W)
        bg_value: Background value for non-instance regions
        relabel: Whether to relabel connected components
        padding: Whether to pad before computing distance
        resolution: Voxel resolution for anisotropic data (z, y, x)
        alpha: Skeleton influence exponent (higher = stronger skeleton influence)
        smooth: Whether to smooth edges before skeletonization (default False;
                adds ~20% overhead with marginal quality impact when using kimimaro)
        smooth_skeleton_only: Only smooth skeleton mask (not entire object)

    Returns:
        Skeleton-aware distance map with same shape as input
    """
    eps = 1e-6

    # Fast-path: empty label should produce all background energy.
    if np.sum(label > 0) == 0:
        return np.full(label.shape, bg_value, dtype=np.float32)

    # 1. Relabel outside processor so we can batch-skeletonize.
    if relabel:
        label = cc3d.connected_components(label, connectivity=6)

    # 2. Batch skeletonize all instances in one call (parallel across instances).
    skeleton_vertices = _batch_skeletonize(label, resolution)

    # 3. Per-instance EDT using BBoxProcessor (skeletons already computed).
    #    Padding coordinate offset: if padding is enabled, the processor pads the
    #    label internally, shifting coordinates by pad_size. We account for this
    #    when translating skeleton vertices to bbox-local coordinates.
    pad_offset = 2 if padding else 0

    config = BBoxProcessorConfig(
        bg_value=bg_value,
        relabel=False,  # already relabeled above
        padding=padding,
        pad_size=2,
        bbox_relax=2,
        combine_mode="max",
    )

    def compute_skeleton_edt(
        label_crop: np.ndarray, instance_id: int, bbox: Tuple[slice, ...], context: Dict
    ) -> Optional[np.ndarray]:
        """Compute skeleton-aware EDT for a single instance within bbox."""
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

        # Look up pre-computed skeleton and translate to bbox-local coordinates.
        skeleton_mask = _skeleton_vertices_to_mask(
            context["skeleton_vertices"].get(instance_id),
            label_crop.shape,
            bbox,
            context["pad_offset"],
        )

        # Fallback to regular EDT if skeletonization failed for this instance.
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

        energy = boundary_edt / (skeleton_edt + boundary_edt + eps)
        energy = energy ** context["alpha"]

        return energy * temp2.astype(np.float32)

    processor = BBoxInstanceProcessor(config)
    return processor.process(
        label,
        compute_skeleton_edt,
        skeleton_vertices=skeleton_vertices,
        pad_offset=pad_offset,
        resolution=resolution,
        alpha=alpha,
        smooth=smooth,
        smooth_skeleton_only=smooth_skeleton_only,
    )


def _batch_skeletonize(
    label: np.ndarray, resolution: Tuple[float, ...]
) -> Dict[int, np.ndarray]:
    """Skeletonize all instances in one kimimaro call.

    Returns:
        Dict mapping instance_id → (N, ndim) int array of vertex coordinates
        in the input label's coordinate system.
    """
    try:
        skeletons = kimimaro.skeletonize(
            label.astype(np.uint32),
            anisotropy=resolution,
            fix_branching=False,
            fix_borders=False,
            dust_threshold=5,
            parallel=0,  # auto-detect cores
            progress=False,
        )
    except Exception:
        return {}

    result = {}
    for inst_id, skel in skeletons.items():
        if len(skel.vertices) > 0:
            result[inst_id] = skel.vertices.astype(int)
    return result


def _skeleton_vertices_to_mask(
    vertices: Optional[np.ndarray],
    crop_shape: Tuple[int, ...],
    bbox: Tuple[slice, ...],
    pad_offset: int,
) -> Optional[np.ndarray]:
    """Convert skeleton vertices (full-volume coords) to a binary mask in bbox-local coords.

    Args:
        vertices: (N, ndim) vertex coordinates in the original (unpadded) label space,
                  or None if this instance had no skeleton.
        crop_shape: Shape of the bbox crop.
        bbox: Tuple of slices defining the bbox in the (possibly padded) label.
        pad_offset: Coordinate offset added by padding (0 if no padding).
    """
    if vertices is None or len(vertices) == 0:
        return None

    # Translate: original-label coords → padded-label coords → bbox-local coords.
    bbox_origin = np.array([s.start for s in bbox])
    local_verts = vertices + pad_offset - bbox_origin

    # Filter to valid range.
    valid = np.all((local_verts >= 0) & (local_verts < np.array(crop_shape)), axis=1)
    local_verts = local_verts[valid]

    if len(local_verts) == 0:
        return None

    mask = np.zeros(crop_shape, dtype=bool)
    if len(crop_shape) == 3:
        mask[local_verts[:, 0], local_verts[:, 1], local_verts[:, 2]] = True
    else:
        mask[local_verts[:, 0], local_verts[:, 1]] = True
    return mask


def precompute_sdt_volume(
    label_path: str,
    output_path: str,
    resolution: Tuple[float, ...] = (1.0, 1.0, 1.0),
    alpha: float = 0.8,
    bg_value: float = -1.0,
) -> str:
    """Precompute skeleton-aware distance transform on a full label volume.

    Computes the SDT once on the entire volume and saves to HDF5.
    Subsequent training runs load the precomputed result, avoiding
    the expensive per-crop skeletonization.

    Args:
        label_path: Path to the instance segmentation label volume.
        output_path: Path to save the precomputed SDT (HDF5).
        resolution: Voxel resolution (z, y, x) for anisotropic data.
        alpha: Skeleton influence exponent.
        bg_value: Background value for non-instance regions.

    Returns:
        The output_path (for chaining).
    """
    import logging
    import time

    from ..io.io import read_volume, save_volume

    logger = logging.getLogger(__name__)
    logger.info(f"Precomputing SDT: {label_path} → {output_path}")

    label = read_volume(label_path)
    logger.info(f"  Label shape: {label.shape}, unique instances: {len(np.unique(label)) - 1}")

    t0 = time.time()
    sdt = skeleton_aware_distance_transform(
        label, resolution=resolution, alpha=alpha, bg_value=bg_value
    )
    elapsed = time.time() - t0
    logger.info(f"  SDT computed in {elapsed:.1f}s, range: [{sdt.min():.3f}, {sdt.max():.3f}]")

    save_volume(output_path, sdt)
    logger.info(f"  Saved to {output_path}")

    return output_path


def sdt_path_for_label(label_path: str) -> str:
    """Derive the SDT cache path from a label file path.

    Example: ``datasets/SNEMI/train-labels.tif`` → ``datasets/SNEMI/train-labels_sdt.h5``
    """
    import os

    base, _ = os.path.splitext(label_path)
    return base + "_sdt.h5"
