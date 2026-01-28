"""
Segmentation decoding functions for mitochondria and other organelles.

Post-processing functions for mitochondria instance segmentation model outputs
as described in "MitoEM Dataset: Large-scale 3D Mitochondria Instance Segmentation
from EM Images" (MICCAI 2020, https://donglaiw.github.io/page/mitoEM/index.html).

Functions:
    - decode_instance_binary_contour_distance: Binary + contour + distance → instances via watershed
    - decode_distance_watershed: SDT → instances via watershed with recomputed EDT
    - decode_affinity_cc: Affinity predictions → instances via fast connected
      components (Numba-accelerated)
"""

import warnings
from typing import List, Optional, Tuple

import cc3d
import fastremap
import mahotas
import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from connectomics.data.process.target import seg_to_semantic_edt

from .utils import cast2dtype

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

try:
    import edt

    EDT_AVAILABLE = True
except ImportError:
    EDT_AVAILABLE = False


__all__ = [
    "decode_instance_binary_contour_distance",
    "decode_affinity_cc",
    "decode_distance_watershed",
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
    min_instance_size: int = 0,
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
        min_instance_size (int): Minimum size of final instance objects in voxels. Instances
            smaller than this are removed after watershed/cc segmentation. Set to 0 to disable.
            Default: 0
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

    # Remove small instances if min_instance_size is specified
    if min_instance_size > 0:
        from .utils import remove_small_instances
        segmentation = remove_small_instances(segmentation, thres_small=min_instance_size, mode='background')

    if return_seed:
        return segmentation, seed
    else:
        return segmentation


# ==============================================================================
# SDT-based Watershed with Recomputed EDT
# ==============================================================================


def decode_distance_watershed(
    predictions: np.ndarray,
    distance_channels: Optional[List[int]] = [0],
    distance_threshold: Tuple[float, float] = (0.5, 0),
    min_seed_size: int = 50,
    min_instance_size: int = 0,
    use_fast_edt: bool = True,
    edt_parallel: int = 4,
    edt_anisotropy: Optional[Tuple[float, ...]] = None,
    edt_downsample_factor: int = 1,
    return_seed: bool = False,
    **kwargs  # Accept but ignore unused parameters for compatibility
) -> np.ndarray:
    """
    Convert signed distance transform (SDT) predictions to instance segmentation
    via watershed with recomputed Euclidean Distance Transform (EDT).
    
    This function implements the SDT-only approach where:
    1. Predicted SDT determines foreground mask (SDT > 0)
    2. Precise EDT is recomputed on the foreground mask
    3. Watershed uses the recomputed EDT (not predicted SDT)
    
    Key Difference from decode_instance_binary_contour_distance:
    - This function RECOMPUTES precise geometric EDT from foreground mask
    - decode_instance_binary_contour_distance uses predicted distance directly
    - Generally more accurate but slower due to EDT computation
    
    The recomputed EDT provides ground-truth Euclidean distances within the
    foreground mask, which can be more reliable for watershed flooding compared
    to using the network's predicted distance values directly.
    
    Args:
        predictions (np.ndarray): Predictions of shape :math:`(C, Z, Y, X)`.
            Typically contains SDT predictions in one or more channels.
        distance_channels (list of int, optional): Channel indices for SDT.
            If multiple channels provided, they are averaged. Default: [0]
        distance_threshold (tuple): Tuple of two floats (seed_threshold, foreground_threshold).
            - threshold[0]: SDT threshold for seeds (instance centers). Default: 0.5
            - threshold[1]: SDT threshold for foreground (instance regions). Default: 0
        min_seed_size (int): Minimum seed size in pixels. Seeds smaller than this
            are removed before watershed. Default: 50
        min_instance_size (int): Minimum size of final instance objects in voxels.
            Instances smaller than this are removed after watershed. Set to 0 to disable.
            Default: 0
        use_fast_edt (bool): Use fast edt library if available for acceleration.
            Falls back to scipy if library not installed. Default: True
        edt_parallel (int): Number of parallel threads for fast edt library.
            Only used if use_fast_edt=True and edt library is available. Default: 4
        edt_anisotropy (tuple or None): Anisotropy values for EDT computation
            (e.g., (3.75, 1.0, 1.0) for anisotropic EM data with 30nm z vs 8nm xy).
            If None, assumes isotropic (all 1.0). Default: None
        edt_downsample_factor (int): Downsample factor for EDT computation to save
            memory/time. If > 1, foreground mask is downsampled, EDT computed, then
            upsampled and scaled. Default: 1 (no downsampling)
        return_seed (bool): Whether to return the seed map along with the segmentation.
            If True, returns (segmentation, seed). Default: False
        **kwargs: Additional parameters for compatibility with YAML configs.
            Unused parameters (binary_channels, contour_channels, binary_threshold,
            contour_threshold, mode, etc.) are silently ignored.
        
    Returns:
        np.ndarray or tuple: Instance segmentation mask of shape :math:`(Z, Y, X)`.
            If return_seed=True, returns tuple (segmentation, seed).
            
    Examples:
        >>> # Basic usage with default parameters
        >>> seg = decode_distance_watershed(predictions, distance_channels=[0])
        
        >>> # With custom thresholds and seed size
        >>> seg = decode_distance_watershed(
        ...     predictions,
        ...     distance_threshold=(0.3, 0),
        ...     min_seed_size=100,
        ...     min_instance_size=500
        ... )
        
        >>> # With anisotropic data (30nm z, 8nm xy)
        >>> seg = decode_distance_watershed(
        ...     predictions,
        ...     distance_channels=[0],
        ...     edt_anisotropy=(3.75, 1.0, 1.0),
        ...     use_fast_edt=True,
        ...     edt_parallel=8
        ... )
        
        >>> # With downsampling for large volumes
        >>> seg = decode_distance_watershed(
        ...     predictions,
        ...     edt_downsample_factor=2,  # 2x downsampling
        ...     use_fast_edt=True
        ... )
        
        >>> # Return seed map for debugging
        >>> seg, seed = decode_distance_watershed(
        ...     predictions,
        ...     distance_channels=[0],
        ...     return_seed=True
        ... )
        
    Note:
        - This function uses 26-connectivity for seed generation (vs 6-connectivity default)
        - EDT recomputation can be slow for large volumes; use edt_downsample_factor
          or enable use_fast_edt for acceleration
        - The fast edt library (pip install edt) provides significant speedup (10-50x)
    """
    
    # DEBUG: Print input to decoding
    try:
        from ..utils.debug_utils import print_tensor_stats
        print_tensor_stats(
            predictions,
            stage_name="STAGE 9: INPUT TO DECODING (watershed)",
            tensor_name="predictions",
            print_once=True,
            extra_info={
                "decoding_method": "decode_distance_watershed",
                "distance_channels": distance_channels,
                "distance_threshold": distance_threshold,
                "expected_range": "[-1, 1] for SDT after tanh"
            }
        )
    except:
        pass  # Silently skip if debug utils not available
    
    # Stage 1: Extract SDT channel
    if distance_channels is None or len(distance_channels) == 0:
        raise ValueError("distance_channels must be specified and non-empty")
    
    if len(distance_channels) > 1:
        # Average multiple channels
        distance = predictions[distance_channels].mean(axis=0)
    else:
        # Use single channel
        distance = predictions[distance_channels[0]]
    
    # Stage 2: Generate foreground mask
    # Use distance_threshold[1] (default: 0) - SDT > 0 indicates foreground
    foreground_mask = distance > distance_threshold[1]
    
    if not foreground_mask.any():
        warnings.warn(
            f"No foreground voxels found with threshold {distance_threshold[1]}. "
            "Returning empty segmentation.",
            UserWarning
        )
        empty_seg = np.zeros(distance.shape, dtype=np.uint32)
        if return_seed:
            return empty_seg, empty_seg
        return empty_seg
    
    # Stage 3: Generate seeds
    # Use distance_threshold[0] (default: 0.5) - high SDT indicates instance centers
    seed_mask = distance > distance_threshold[0]
    
    if not seed_mask.any():
        warnings.warn(
            f"No seed voxels found with threshold {distance_threshold[0]}. "
            "Returning empty segmentation.",
            UserWarning
        )
        empty_seg = np.zeros(distance.shape, dtype=np.uint32)
        if return_seed:
            return empty_seg, empty_seg
        return empty_seg
    
    # Connected components with 26-connectivity (captures diagonal connections)
    seed = cc3d.connected_components(seed_mask.astype(np.uint8), connectivity=26)
    
    # Remove small seeds
    if min_seed_size > 0:
        seed = remove_small_objects(seed, min_size=min_seed_size)
    
    if seed.max() == 0:
        warnings.warn(
            f"No seeds remain after removing small objects (min_size={min_seed_size}). "
            "Returning empty segmentation.",
            UserWarning
        )
        empty_seg = np.zeros(distance.shape, dtype=np.uint32)
        if return_seed:
            return empty_seg, empty_seg
        return empty_seg
    
    # Stage 4: Recompute precise EDT on foreground mask
    # KEY DIFFERENCE: We do NOT use predicted distance directly
    # Instead, we compute ground-truth Euclidean distance within foreground
    
    if edt_downsample_factor > 1:
        # Downsample for speed/memory efficiency
        zoom_factors = tuple([1.0 / edt_downsample_factor] * foreground_mask.ndim)
        foreground_downsampled = ndimage.zoom(
            foreground_mask.astype(np.float32),
            zoom_factors,
            order=0  # Nearest neighbor for binary mask
        ).astype(bool)
        
        # Compute EDT on downsampled mask
        if use_fast_edt and EDT_AVAILABLE:
            if edt_anisotropy is None:
                edt_anisotropy_scaled = tuple([1.0] * foreground_downsampled.ndim)
            else:
                # Scale anisotropy for downsampled resolution
                edt_anisotropy_scaled = tuple([a / edt_downsample_factor for a in edt_anisotropy])
            
            distance_fg_downsampled = edt.edt(
                foreground_downsampled.astype(np.uint8),
                anisotropy=edt_anisotropy_scaled,
                black_border=True,
                parallel=edt_parallel
            )
        else:
            if not use_fast_edt and EDT_AVAILABLE:
                pass  # User explicitly disabled fast edt
            elif use_fast_edt and not EDT_AVAILABLE:
                warnings.warn(
                    "Fast edt library not available. Using scipy.ndimage (slower). "
                    "Install edt for 10-50x speedup: pip install edt",
                    UserWarning
                )
            
            distance_fg_downsampled = ndimage.distance_transform_edt(
                foreground_downsampled
            )
        
        # Upsample EDT and scale distances
        distance_fg = ndimage.zoom(
            distance_fg_downsampled,
            tuple([edt_downsample_factor] * distance_fg_downsampled.ndim),
            order=1  # Linear interpolation for continuous distance field
        )
        distance_fg *= edt_downsample_factor  # Scale distances back to original resolution
        
        # Ensure output shape matches input (zoom can introduce +/-1 voxel differences)
        if distance_fg.shape != foreground_mask.shape:
            # Crop or pad to match
            slices = tuple([slice(0, min(s1, s2)) for s1, s2 in zip(distance_fg.shape, foreground_mask.shape)])
            distance_fg_corrected = np.zeros(foreground_mask.shape, dtype=distance_fg.dtype)
            distance_fg_corrected[slices] = distance_fg[slices]
            distance_fg = distance_fg_corrected
    
    else:
        # No downsampling - compute EDT at full resolution
        if use_fast_edt and EDT_AVAILABLE:
            if edt_anisotropy is None:
                edt_anisotropy = tuple([1.0] * foreground_mask.ndim)
            
            distance_fg = edt.edt(
                foreground_mask.astype(np.uint8),
                anisotropy=edt_anisotropy,
                black_border=True,
                parallel=edt_parallel
            )
        else:
            if not use_fast_edt and EDT_AVAILABLE:
                pass  # User explicitly disabled fast edt
            elif use_fast_edt and not EDT_AVAILABLE:
                warnings.warn(
                    "Fast edt library not available. Using scipy.ndimage (slower). "
                    "Install edt for 10-50x speedup: pip install edt",
                    UserWarning
                )
            
            if edt_anisotropy is not None:
                # scipy supports anisotropy via sampling parameter
                distance_fg = ndimage.distance_transform_edt(
                    foreground_mask,
                    sampling=edt_anisotropy
                )
            else:
                distance_fg = ndimage.distance_transform_edt(foreground_mask)
    
    # Stage 5: Watershed segmentation
    # Use skimage watershed (supports mask parameter directly)
    segmentation = watershed(
        -distance_fg,  # Invert: peaks (high distance) become valleys for watershed
        seed,
        mask=foreground_mask
    )
    
    # Stage 6: Post-processing
    # Refit labels to be consecutive [0, 1, 2, ..., N]
    segmentation = fastremap.refit(segmentation)
    
    # Remove small instances if min_instance_size is specified
    if min_instance_size > 0:
        from .utils import remove_small_instances
        segmentation = remove_small_instances(
            segmentation,
            thres_small=min_instance_size,
            mode='background'
        )
    
    # Return segmentation (and optionally seed map)
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

    Returns:
        numpy.ndarray: Instance segmentation mask of shape :math:`(Z, Y, X)` with
            dtype uint32. Each connected component has a unique ID >= 1, background is 0.

    Examples:
        >>> # Basic usage with affinity predictions
        >>> affinities = model(image)  # Shape: (6, 128, 128, 128)
        >>> segmentation = decode_affinity_cc(affinities, threshold=0.5)
        >>> print(segmentation.shape)  # (128, 128, 128)
        >>> print(segmentation.max())  # Number of instances

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
    if NUMBA_AVAILABLE:
        # Fast Numba implementation (10-100x speedup)
        segmentation = _connected_components_affinity_3d_numba(hard_aff)
    else:
        # Fallback to skimage (slower but always available)
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

    segmentation = fastremap.refit(segmentation)

    # Ensure background label (0) is present
    if segmentation.size > 0 and not np.any(segmentation == 0):
        segmentation = segmentation.copy()
        segmentation[0, 0, 0] = 0

    # Cast to compact integer dtype
    return cast2dtype(segmentation)
