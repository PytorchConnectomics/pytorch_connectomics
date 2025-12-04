"""
Segmentation decoding functions for mitochondria and other organelles.

Post-processing functions for mitochondria instance segmentation model outputs
as described in "MitoEM Dataset: Large-scale 3D Mitochondria Instance Segmentation
from EM Images" (MICCAI 2020, https://donglaiw.github.io/page/mitoEM/index.html).

Functions:
    - decode_instance_binary_contour_distance: Binary + contour + distance → instances via watershed
    - decode_affinity_cc: Affinity predictions → instances via fast connected
      components (Numba-accelerated)
"""

import warnings
from typing import List, Optional, Tuple

import cc3d
import fastremap
import mahotas
import numpy as np
from scipy.ndimage import zoom
from skimage.morphology import remove_small_objects

from connectomics.data.process.target import seg_to_semantic_edt

from .utils import cast2dtype, remove_small_instances

try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Define dummy jit decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


__all__ = [
    "decode_instance_binary_contour_distance",
    "decode_affinity_cc",
    "affinity_cc3d",
]


def decode_instance_binary_contour_distance(
    predictions: np.ndarray,
    mode: str = "watershed",  # "watershed" or "cc"
    binary_channels: Optional[List[int]] = [0],
    contour_channels: Optional[List[int]] = [1],
    distance_channels: Optional[List[int]] = [2],
    binary_threshold: Tuple[float, float] = (0.9, 0.85),
    contour_threshold: Optional[Tuple[float, float]] = (0.8, 1.1),
    distance_threshold: Tuple[float, float] = (0.5, 0),
    precomputed_seed: Optional[np.ndarray] = None,
    min_seed_size: int = 32,
    return_seed: bool = False,
):
    r"""Convert binary foreground probability maps, instance contours and signed distance
    transform to instance masks via watershed or connected components.

    This unified function supports both watershed and connected components approaches for
    instance segmentation from multi-channel predictions (binary, contour, distance).

    Note:
        When using watershed mode, this function uses `mahotas.cwatershed` which converts
        the input into ``np.float64`` data type for processing. Please ensure enough memory
        is allocated when handling large arrays.

    Args:
        predictions (numpy.ndarray): Multi-channel prediction map of shape :math:`(C, Z, Y, X)`.
            Typically contains binary foreground, instance contours, and signed distance transform.
        mode (str): Decoding algorithm to use. Options:
            - 'watershed': Use watershed segmentation with distance transform (more accurate)
            - 'cc': Use connected components (faster, simpler)
            Default: 'watershed'
        binary_channels (list of int, optional): Channel indices for binary foreground mask.
            If multiple channels provided, they are averaged. Default: [0]
        contour_channels (list of int, optional): Channel indices for instance contours.
            If multiple channels provided, they are averaged. Set to None to disable contour
            constraints (for BANIS-style binary+distance only). Default: [1]
        distance_channels (list of int, optional): Channel indices for signed distance transform.
            If multiple channels provided, they are averaged. Default: [2]
        binary_threshold (tuple): Tuple of two floats (seed_threshold,
            foreground_threshold) for binary mask. The first value is used for
            seed generation, the second for foreground mask. Default: (0.9, 0.85)
        contour_threshold (tuple or None): Tuple of two floats (seed_threshold,
            foreground_threshold) for instance contours. The first value is used
            for seed generation, the second for foreground mask. Set to None to
            disable contour constraints. Default: (0.8, 1.1)
        distance_threshold (tuple): Tuple of two floats (seed_threshold,
            foreground_threshold) for signed distance. The first value is used
            for seed generation, the second for foreground mask. Default: (0.5, 0)
        precomputed_seed (numpy.ndarray, optional): Precomputed seed map to use instead of
            computing seeds from thresholds. Default: None
        min_seed_size (int): Minimum size of seed objects in pixels. Seeds smaller than this
            are removed before watershed. Only used in watershed mode. Default: 32
        return_seed (bool): Whether to return the seed map along with the segmentation.
            If True, returns (segmentation, seed). Only applicable in watershed mode. Default: False

    Returns:
        numpy.ndarray or tuple: Instance segmentation mask of shape :math:`(Z, Y, X)`.
            If return_seed=True (watershed mode only), returns tuple (segmentation, seed).

    Examples:
        >>> # Standard 3-channel watershed (binary, contour, distance)
        >>> seg = decode_instance_binary_contour_distance(predictions, mode='watershed')

        >>> # BANIS-style 2-channel (binary, distance) - no contour
        >>> seg = decode_instance_binary_contour_distance(
        ...     predictions,  # shape (2, Z, Y, X)
        ...     mode='watershed',
        ...     binary_threshold=(0.5, 0.5),
        ...     contour_channels=None,  # Disable contour
        ...     contour_threshold=None,
        ...     distance_threshold=(0.0, -1.0),
        ... )

        >>> # Fast connected components mode
        >>> seg = decode_instance_binary_contour_distance(
        ...     predictions,
        ...     mode='cc',
        ...     binary_threshold=(0.9, 0.85),
        ...     contour_threshold=(0.8, 1.1),
        ... )

        >>> # Explicit channel selection with averaging
        >>> seg = decode_instance_binary_contour_distance(
        ...     predictions,  # shape (3, Z, Y, X) with channels [aff_x, aff_y, SDT]
        ...     binary_channels=[0, 1],  # Average channels 0 and 1 for binary
        ...     contour_channels=None,   # No contour
        ...     distance_channels=[2],   # Channel 2 for distance
        ...     contour_threshold=None,
        ... )

        >>> # Return seed map for debugging
        >>> seg, seed = decode_instance_binary_contour_distance(
        ...     predictions, mode='watershed', return_seed=True
        ... )
    """

    binary, contour, distance = None, None, None
    if binary_channels is not None:
        if len(binary_channels) > 1:  # average multiple channels
            binary = predictions[binary_channels].mean(axis=0)
        else:
            binary = predictions[binary_channels[0]]  # use first channel

    if contour_channels is not None:
        if len(contour_channels) > 1:
            contour = predictions[contour_channels].mean(axis=0)
        else:
            contour = predictions[contour_channels[0]]

    if distance_channels is not None:
        if len(distance_channels) > 1:
            distance = predictions[distance_channels].mean(axis=0)
        else:
            distance = predictions[distance_channels[0]]

    # step 1: compute the foreground mask
    foreground = None
    if binary is not None:
        foreground = binary > binary_threshold[1]
    if contour is not None:
        foreground = (
            foreground * (contour < contour_threshold[1])
            if foreground is not None
            else (contour < contour_threshold[1])
        )
    if distance is not None:
        foreground = (
            foreground * (distance > distance_threshold[1])
            if foreground is not None
            else (distance > distance_threshold[1])
        )

    if mode == "cc":
        segmentation = cc3d.connected_components(foreground)
    elif mode == "watershed":
        # Watershed mode requires distance channel
        if distance is None:
            distance = seg_to_semantic_edt(foreground, mode="3d")
        # step 2: compute the instance seeds
        if precomputed_seed is not None:
            seed = precomputed_seed
        else:  # compute the instance seeds
            seed_map = None
            if binary is not None:
                seed_map = binary > binary_threshold[0]
            if contour is not None:
                seed_map = (
                    seed_map * (contour < contour_threshold[0])
                    if seed_map is not None
                    else (contour < contour_threshold[0])
                )
            if distance is not None:
                seed_map = (
                    seed_map * (distance > distance_threshold[0])
                    if seed_map is not None
                    else (distance > distance_threshold[0])
                )
            seed = cc3d.connected_components(seed_map)
            seed = remove_small_objects(seed, min_seed_size)

        # step 3: compute the segmentation mask
        distance[distance < 0] = 0
        segmentation = mahotas.cwatershed(-distance.astype(np.float64), seed)
        segmentation[~foreground] = (
            0  # Apply mask manually (mahotas 1.4.18 doesn't support mask parameter)
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    segmentation = fastremap.refit(segmentation)

    if return_seed:
        return segmentation, seed
    else:
        return segmentation


# ==============================================================================
# Affinity-based Segmentation (BANIS-inspired)
# ==============================================================================


@jit(nopython=True)
def _connected_components_affinity_3d_numba(hard_aff: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated connected components from 3D affinities.

    Uses flood-fill algorithm with 6-connectivity (face neighbors only).
    Provides 10-100x speedup over pure Python implementations.

    Args:
        hard_aff: Boolean affinities, shape (3, D, H, W)
                 - Channel 0: x-direction connections
                 - Channel 1: y-direction connections
                 - Channel 2: z-direction connections

    Returns:
        segmentation: Instance segmentation, shape (D, H, W)
                     Each component gets unique ID >= 1, background is 0

    Note:
        This function is JIT-compiled with Numba for performance.
        Reference: BANIS baseline (inference.py)
    """
    visited = np.zeros(hard_aff.shape[1:], dtype=np.uint8)
    seg = np.zeros(hard_aff.shape[1:], dtype=np.uint32)
    cur_id = 1

    # Flood-fill from each foreground voxel
    for i in range(visited.shape[0]):
        for j in range(visited.shape[1]):
            for k in range(visited.shape[2]):
                # Check if foreground and unvisited
                if hard_aff[:, i, j, k].any() and not visited[i, j, k]:
                    # Start new component - use arrays for stack (Numba compatible)
                    stack_size = 0
                    max_stack = visited.shape[0] * visited.shape[1] * visited.shape[2]
                    stack_x = np.zeros(max_stack, dtype=np.int32)
                    stack_y = np.zeros(max_stack, dtype=np.int32)
                    stack_z = np.zeros(max_stack, dtype=np.int32)

                    # Push initial voxel
                    stack_x[stack_size] = i
                    stack_y[stack_size] = j
                    stack_z[stack_size] = k
                    stack_size += 1
                    visited[i, j, k] = True

                    # Flood-fill
                    while stack_size > 0:
                        # Pop from stack
                        stack_size -= 1
                        x = stack_x[stack_size]
                        y = stack_y[stack_size]
                        z = stack_z[stack_size]

                        seg[x, y, z] = cur_id

                        # Check 6-connected neighbors
                        # Positive x
                        if (
                            x + 1 < visited.shape[0]
                            and hard_aff[0, x, y, z]
                            and not visited[x + 1, y, z]
                        ):
                            stack_x[stack_size] = x + 1
                            stack_y[stack_size] = y
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x + 1, y, z] = True

                        # Positive y
                        if (
                            y + 1 < visited.shape[1]
                            and hard_aff[1, x, y, z]
                            and not visited[x, y + 1, z]
                        ):
                            stack_x[stack_size] = x
                            stack_y[stack_size] = y + 1
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x, y + 1, z] = True

                        # Positive z
                        if (
                            z + 1 < visited.shape[2]
                            and hard_aff[2, x, y, z]
                            and not visited[x, y, z + 1]
                        ):
                            stack_x[stack_size] = x
                            stack_y[stack_size] = y
                            stack_z[stack_size] = z + 1
                            stack_size += 1
                            visited[x, y, z + 1] = True

                        # Negative x
                        if x - 1 >= 0 and hard_aff[0, x - 1, y, z] and not visited[x - 1, y, z]:
                            stack_x[stack_size] = x - 1
                            stack_y[stack_size] = y
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x - 1, y, z] = True

                        # Negative y
                        if y - 1 >= 0 and hard_aff[1, x, y - 1, z] and not visited[x, y - 1, z]:
                            stack_x[stack_size] = x
                            stack_y[stack_size] = y - 1
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x, y - 1, z] = True

                        # Negative z
                        if z - 1 >= 0 and hard_aff[2, x, y, z - 1] and not visited[x, y, z - 1]:
                            stack_x[stack_size] = x
                            stack_y[stack_size] = y
                            stack_z[stack_size] = z - 1
                            stack_size += 1
                            visited[x, y, z - 1] = True

                    cur_id += 1

    return seg


def decode_affinity_cc(
    affinities: np.ndarray,
    threshold: float = 0.5,
    use_numba: bool = True,
    min_instance_size: Optional[int] = None,
    thres_small: int = 0,
    remove_small_mode: str = "background",
    scale_factors: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    r"""Convert affinity predictions to instance segmentation via connected components.

    This function implements fast connected component labeling on affinity graphs,
    providing 10-100x speedup when Numba is available compared to standard methods.

    The algorithm uses only **short-range affinities** (first 3 channels) to build
    a connectivity graph, then performs flood-fill to identify connected components.
    Each component receives a unique instance ID.

    Args:
        affinities (numpy.ndarray): Affinity predictions of shape :math:`(C, Z, Y, X)` where:

            - C >= 3 (first 3 channels are short-range affinities)
            - Channel 0: x-direction (left-right) connections
            - Channel 1: y-direction (top-bottom) connections
            - Channel 2: z-direction (front-back) connections
            - Channels 3+: long-range affinities (ignored)

        threshold (float): Threshold for binarizing affinities. Affinities > threshold
            indicate connected voxels. Default: 0.5
        use_numba (bool): Use Numba JIT acceleration if available. Provides 10-100x speedup.
            Falls back to skimage if Numba not available. Default: True
        min_instance_size (int): minimum size threshold for instances to keep. Objects with fewer
            voxels are removed. Set to 0 to keep all objects. Default: 0
        remove_small_mode (str): Method for removing small objects:

            - ``'background'``: Replace with background (0)
            - ``'neighbor'``: Merge with nearest neighbor
            - ``'background_2d'`` or ``'neighbor_2d'``: Apply slice-wise
            - ``'none'``: Keep all objects

            Default: ``'background'``
        scale_factors (tuple or None): Optional anisotropic scaling factors (z, y, x).
            When provided, the segmentation is resized with nearest-neighbor interpolation.
            Useful for matching voxel resolutions. Default: None

    Returns:
        numpy.ndarray: Instance segmentation mask of shape :math:`(Z, Y, X)` with
            dtype uint32. Each connected component has a unique ID >= 1, background is 0.

    Examples:
        >>> # Basic usage with affinity predictions
        >>> affinities = model(image)  # Shape: (6, 128, 128, 128)
        >>> segmentation = decode_affinity_cc(affinities, threshold=0.5)
        >>> print(segmentation.shape)  # (128, 128, 128)
        >>> print(segmentation.max())  # Number of instances

        >>> # Remove small objects
        >>> segmentation = decode_affinity_cc(
        ...     affinities,
        ...     threshold=0.5,
        ...     min_instance_size=100  # Remove objects < 100 voxels
        ... )
        >>> # Resize to target voxel spacing
        >>> segmentation = decode_affinity_cc(
        ...     affinities,
        ...     threshold=0.5,
        ...     scale_factors=(2.0, 1.0, 0.5)
        ... )

    Note:
        - **Numba acceleration**: Install numba for 10-100x speedup:
          ``pip install numba>=0.60.0``
        - **6-connectivity**: Uses face neighbors only (not edges/corners)
        - **Short-range only**: Only first 3 channels used, long-range ignored
        - **Memory efficient**: Processes in-place when possible

    Reference:
        BANIS baseline (https://github.com/kreshuklab/BANIS)
        Fast connected components for neuron instance segmentation.

    See Also:
        - :func:`decode_binary_cc`: Connected components on binary masks
        - :func:`decode_binary_contour_watershed`: Watershed on binary + contour predictions
    """
    affinities = np.asarray(affinities)
    if affinities.ndim != 4:
        raise ValueError(f"Expected affinities with shape (C, Z, Y, X), got {affinities.ndim}D")
    if affinities.shape[0] < 3:
        raise ValueError(f"Expected >= 3 channels, got {affinities.shape[0]}")

    # Extract short-range affinities (first 3 channels)
    short_range_aff = affinities[:3]
    foreground_mask = (short_range_aff > 0).any(axis=0)

    # Binarize affinities
    hard_aff = short_range_aff > threshold

    # Connected components
    if use_numba and NUMBA_AVAILABLE:
        # Fast Numba implementation (10-100x speedup)
        segmentation = _connected_components_affinity_3d_numba(hard_aff)
    else:
        # Fallback to skimage (slower but always available)
        if use_numba and not NUMBA_AVAILABLE:
            warnings.warn(
                "Numba not available. Using skimage (slower). "
                "Install numba for 10-100x speedup: pip install numba>=0.60.0",
                UserWarning,
            )

        # Create foreground mask (any affinity > 0)
        foreground = hard_aff.any(axis=0)
        segmentation = cc3d.connected_components(foreground)

    # Fill any foreground voxels that lost all connections at high thresholds
    if foreground_mask.any():
        missing_mask = foreground_mask & (segmentation == 0)
        if missing_mask.any():
            missing_labels = cc3d.connected_components(missing_mask)
            if segmentation.max() == 0:
                segmentation = missing_labels
            else:
                segmentation = segmentation.astype(np.int64, copy=False)
                segmentation[missing_mask] = missing_labels[missing_mask] + segmentation.max()

    # Remove small instances
    min_size = min_instance_size if min_instance_size is not None else thres_small
    if min_size is None:
        min_size = 0
    if min_size > 0:
        segmentation = remove_small_instances(segmentation, min_size, remove_small_mode)

    segmentation = fastremap.refit(segmentation)

    if scale_factors is not None:
        if len(scale_factors) != 3:
            raise ValueError("scale_factors must be a tuple/list of three values (z, y, x)")
        segmentation = zoom(segmentation, zoom=scale_factors, order=0, mode="nearest")
        segmentation = fastremap.refit(segmentation.astype(np.int64, copy=False))

    # Ensure background label (0) is present
    if segmentation.size > 0 and not np.any(segmentation == 0):
        segmentation = segmentation.copy()
        segmentation[0, 0, 0] = 0

    # Cast to compact integer dtype
    return cast2dtype(segmentation)


# Public alias used across tutorials/tests
affinity_cc3d = decode_affinity_cc
