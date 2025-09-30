"""
Segmentation decoding functions for mitochondria and other organelles.

Post-processing functions for mitochondria instance segmentation model outputs
as described in "MitoEM Dataset: Large-scale 3D Mitochondria Instance Segmentation
from EM Images" (MICCAI 2020, https://donglaiw.github.io/page/mitoEM/index.html).

Functions:
    - binary_connected: Binary foreground → instances via connected components
    - binary_watershed: Binary foreground → instances via watershed
    - bc_connected: Binary + contour → instances via connected components
    - bc_watershed: Binary + contour → instances via watershed
    - bcd_watershed: Binary + contour + distance → instances via watershed
    - affinity_cc3d: Affinity predictions → instances via fast connected components (Numba-accelerated)
"""

from __future__ import print_function, division
from typing import Optional, Tuple, Union
import numpy as np

from skimage.measure import label
from skimage.transform import resize
from skimage.morphology import dilation, remove_small_objects
from skimage.segmentation import watershed

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
    'binary_connected',
    'binary_watershed',
    'bc_connected',
    'bc_watershed',
    'bcd_watershed',
    'affinity_cc3d',
]


def binary_connected(
    volume: np.ndarray,
    thres: float = 0.8,
    thres_small: int = 128,
    scale_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    remove_small_mode: str = 'background'
) -> np.ndarray:
    r"""Convert binary foreground probability maps to instance masks via
    connected-component labeling.

    Args:
        volume (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        thres (float): threshold of foreground. Default: 0.8
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
        
    Returns:
        numpy.ndarray: Instance segmentation mask.
    """
    semantic = volume[0]
    foreground = (semantic > int(255 * thres))
    segm = label(foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(semantic.shape[0] * scale_factors[0]),
            int(semantic.shape[1] * scale_factors[1]),
            int(semantic.shape[2] * scale_factors[2])
        )
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    return cast2dtype(segm)


def binary_watershed(
    volume: np.ndarray,
    thres1: float = 0.98,
    thres2: float = 0.85,
    thres_small: int = 128,
    scale_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    remove_small_mode: str = 'background',
    seed_thres: int = 32
) -> np.ndarray:
    r"""Convert binary foreground probability maps to instance masks via
    watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
        function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

    Args:
        volume (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of seeds. Default: 0.98
        thres2 (float): threshold of foreground. Default: 0.85
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
        seed_thres (int): minimum size of seed objects. Default: 32
        
    Returns:
        numpy.ndarray: Instance segmentation mask.
    """
    semantic = volume[0]
    seed_map = semantic > int(255 * thres1)
    foreground = semantic > int(255 * thres2)
    seed = label(seed_map)
    seed = remove_small_objects(seed, seed_thres)
    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(semantic.shape[0] * scale_factors[0]),
            int(semantic.shape[1] * scale_factors[1]),
            int(semantic.shape[2] * scale_factors[2])
        )
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    return cast2dtype(segm)


def bc_connected(
    volume: np.ndarray,
    thres1: float = 0.8,
    thres2: float = 0.5,
    thres_small: int = 128,
    scale_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    dilation_struct: Tuple[int, int, int] = (1, 5, 5),
    remove_small_mode: str = 'background'
) -> np.ndarray:
    r"""Convert binary foreground probability maps and instance contours to
    instance masks via connected-component labeling.

    Note:
        The instance contour provides additional supervision to distinguish closely touching
        objects. However, the decoding algorithm only keep the intersection of foreground and
        non-contour regions, which will systematically result in incomplete instance masks.
        Therefore we apply morphological dilation (check :attr:`dilation_struct`) to enlarge
        the object masks.

    Args:
        volume (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of foreground. Default: 0.8
        thres2 (float): threshold of instance contours. Default: 0.5
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        dilation_struct (tuple): the shape of the structure for morphological dilation. Default: (1, 5, 5)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
        
    Returns:
        numpy.ndarray: Instance segmentation mask.
    """
    semantic = volume[0]
    boundary = volume[1]
    foreground = (semantic > int(255 * thres1)) * (boundary < int(255 * thres2))

    segm = label(foreground)
    struct = np.ones(dilation_struct)
    segm = dilation(segm, struct)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(semantic.shape[0] * scale_factors[0]),
            int(semantic.shape[1] * scale_factors[1]),
            int(semantic.shape[2] * scale_factors[2])
        )
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    return cast2dtype(segm)


def bc_watershed(
    volume: np.ndarray,
    thres1: float = 0.9,
    thres2: float = 0.8,
    thres3: float = 0.85,
    thres_small: int = 128,
    scale_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    remove_small_mode: str = 'background',
    seed_thres: int = 32,
    return_seed: bool = False,
    precomputed_seed: Optional[np.ndarray] = None
):
    r"""Convert binary foreground probability maps and instance contours to
    instance masks via watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
        function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

    Args:
        volume (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of seeds. Default: 0.9
        thres2 (float): threshold of instance contours. Default: 0.8
        thres3 (float): threshold of foreground. Default: 0.85
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
        seed_thres (int): minimum size of seed objects. Default: 32
        return_seed (bool): whether to return the seed map. Default: False
        precomputed_seed (numpy.ndarray, optional): precomputed seed map. Default: None
        
    Returns:
        numpy.ndarray or tuple: Instance segmentation mask, or (mask, seed) if return_seed=True.
    """
    assert volume.shape[0] == 2
    semantic = volume[0]
    boundary = volume[1]
    foreground = (semantic > int(255 * thres3))

    if precomputed_seed is not None:
        seed = precomputed_seed
    else:  # compute the instance seeds
        seed_map = (semantic > int(255 * thres1)) * (boundary < int(255 * thres2))
        seed = label(seed_map)
        seed = remove_small_objects(seed, seed_thres)

    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(semantic.shape[0] * scale_factors[0]),
            int(semantic.shape[1] * scale_factors[1]),
            int(semantic.shape[2] * scale_factors[2])
        )
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    if not return_seed:
        return cast2dtype(segm)

    return cast2dtype(segm), seed


def bcd_watershed(
    volume: np.ndarray,
    thres1: float = 0.9,
    thres2: float = 0.8,
    thres3: float = 0.85,
    thres4: float = 0.5,
    thres5: float = 0.0,
    thres_small: int = 128,
    scale_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    remove_small_mode: str = 'background',
    seed_thres: int = 32,
    return_seed: bool = False,
    precomputed_seed: Optional[np.ndarray] = None
):
    r"""Convert binary foreground probability maps, instance contours and signed distance
    transform to instance masks via watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
        function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

    Args:
        volume (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of seeds. Default: 0.9
        thres2 (float): threshold of instance contours. Default: 0.8
        thres3 (float): threshold of foreground. Default: 0.85
        thres4 (float): threshold of signed distance for locating seeds. Default: 0.5
        thres5 (float): threshold of signed distance for foreground. Default: 0.0
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
        seed_thres (int): minimum size of seed objects. Default: 32
        return_seed (bool): whether to return the seed map. Default: False
        precomputed_seed (numpy.ndarray, optional): precomputed seed map. Default: None
        
    Returns:
        numpy.ndarray or tuple: Instance segmentation mask, or (mask, seed) if return_seed=True.
    """
    assert volume.shape[0] == 3
    semantic, boundary, distance = volume[0], volume[1], volume[2]
    distance = (distance / 255.0) * 2.0 - 1.0
    foreground = (semantic > int(255 * thres3)) * (distance > thres5)

    if precomputed_seed is not None:
        seed = precomputed_seed
    else:  # compute the instance seeds
        seed_map = (semantic > int(255 * thres1)) * (boundary < int(255 * thres2)) * (distance > thres4)
        seed = label(seed_map)
        seed = remove_small_objects(seed, seed_thres)

    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(semantic.shape[0] * scale_factors[0]),
            int(semantic.shape[1] * scale_factors[1]),
            int(semantic.shape[2] * scale_factors[2])
        )
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    if not return_seed:
        return cast2dtype(segm)

    return cast2dtype(segm), seed


# ==============================================================================
# Affinity-based Segmentation (BANIS-inspired)
# ==============================================================================


@jit(nopython=True)
def _connected_components_3d_numba(hard_aff: np.ndarray) -> np.ndarray:
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
                        if x + 1 < visited.shape[0] and hard_aff[0, x, y, z] and not visited[x + 1, y, z]:
                            stack_x[stack_size] = x + 1
                            stack_y[stack_size] = y
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x + 1, y, z] = True

                        # Positive y
                        if y + 1 < visited.shape[1] and hard_aff[1, x, y, z] and not visited[x, y + 1, z]:
                            stack_x[stack_size] = x
                            stack_y[stack_size] = y + 1
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x, y + 1, z] = True

                        # Positive z
                        if z + 1 < visited.shape[2] and hard_aff[2, x, y, z] and not visited[x, y, z + 1]:
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


def affinity_cc3d(
    affinities: np.ndarray,
    threshold: float = 0.5,
    use_numba: bool = True,
    thres_small: int = 0,
    scale_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    remove_small_mode: str = 'background'
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
        thres_small (int): Size threshold for removing small objects. Objects with fewer
            voxels are removed. Set to 0 to keep all objects. Default: 0
        scale_factors (tuple): Scale factors for resizing output in :math:`(Z, Y, X)` order.
            Use (1.0, 1.0, 1.0) for no resizing. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): Method for removing small objects:

            - ``'background'``: Replace with background (0)
            - ``'neighbor'``: Merge with nearest neighbor
            - ``'none'``: Keep all objects

            Default: ``'background'``

    Returns:
        numpy.ndarray: Instance segmentation mask of shape :math:`(Z, Y, X)` with
            dtype uint32. Each connected component has a unique ID >= 1, background is 0.

    Examples:
        >>> # Basic usage with affinity predictions
        >>> affinities = model(image)  # Shape: (6, 128, 128, 128)
        >>> segmentation = affinity_cc3d(affinities, threshold=0.5)
        >>> print(segmentation.shape)  # (128, 128, 128)
        >>> print(segmentation.max())  # Number of instances

        >>> # Remove small objects
        >>> segmentation = affinity_cc3d(
        ...     affinities,
        ...     threshold=0.5,
        ...     thres_small=100  # Remove objects < 100 voxels
        ... )

        >>> # Resize output
        >>> segmentation = affinity_cc3d(
        ...     affinities,
        ...     threshold=0.5,
        ...     scale_factors=(2.0, 2.0, 2.0)  # Upsample 2x
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
        - :func:`binary_connected`: Connected components on binary masks
        - :func:`bc_watershed`: Watershed on binary + contour predictions
    """
    assert affinities.ndim == 4, f"Expected 4D array, got {affinities.ndim}D"
    assert affinities.shape[0] >= 3, f"Expected >= 3 channels, got {affinities.shape[0]}"

    # Extract short-range affinities (first 3 channels)
    short_range_aff = affinities[:3]

    # Binarize affinities
    hard_aff = short_range_aff > threshold

    # Connected components
    if use_numba and NUMBA_AVAILABLE:
        # Fast Numba implementation (10-100x speedup)
        segm = _connected_components_3d_numba(hard_aff)
    else:
        # Fallback to skimage (slower but always available)
        if use_numba and not NUMBA_AVAILABLE:
            import warnings
            warnings.warn(
                "Numba not available. Using skimage (slower). "
                "Install numba for 10-100x speedup: pip install numba>=0.60.0",
                UserWarning
            )

        # Create foreground mask (any affinity > 0)
        foreground = hard_aff.any(axis=0)
        segm = label(foreground)

    # Remove small instances
    if thres_small > 0:
        segm = remove_small_instances(segm, thres_small, remove_small_mode)

    # Resize if requested
    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(segm.shape[0] * scale_factors[0]),
            int(segm.shape[1] * scale_factors[1]),
            int(segm.shape[2] * scale_factors[2])
        )
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    return cast2dtype(segm)
