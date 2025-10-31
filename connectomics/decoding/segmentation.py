"""
Segmentation decoding functions for mitochondria and other organelles.

Post-processing functions for mitochondria instance segmentation model outputs
as described in "MitoEM Dataset: Large-scale 3D Mitochondria Instance Segmentation
from EM Images" (MICCAI 2020, https://donglaiw.github.io/page/mitoEM/index.html).

Functions:
    - decode_binary_cc: Binary foreground → instances via connected components
    - decode_binary_watershed: Binary foreground → instances via watershed
    - decode_binary_contour_cc: Binary + contour → instances via connected components
    - decode_binary_contour_watershed: Binary + contour → instances via watershed
    - decode_binary_contour_distance_watershed: Binary + contour + distance → instances via watershed
    - decode_affinity_cc: Affinity predictions → instances via fast connected components (Numba-accelerated)
"""

from __future__ import print_function, division
from typing import Optional, Tuple
import numpy as np
import cc3d
import fastremap

from skimage.morphology import dilation, remove_small_objects
from scipy.ndimage import zoom
import mahotas


from .utils import remove_small_instances

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
    "decode_binary_thresholding",
    "decode_binary_cc",
    "decode_binary_watershed",
    "decode_binary_contour_cc",
    "decode_binary_contour_watershed",
    "decode_binary_contour_distance_watershed",
    "decode_affinity_cc",
]


def decode_binary_thresholding(
    predictions: np.ndarray,
    threshold_range: Tuple[float, float] = (0.8, 1.0),
) -> np.ndarray:
    r"""Convert binary foreground probability maps to binary mask via simple thresholding.

    This is a lightweight decoding function that applies thresholding to convert
    probability predictions to binary segmentation masks. Unlike instance segmentation
    methods, this produces a semantic segmentation (no individual instance IDs).

    The function uses the minimum threshold from threshold_range to binarize predictions.
    This is useful for simple binary segmentation tasks where instance separation is not needed.

    Args:
        predictions (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)` or :math:`(C, Y, X)`.
            The first channel (predictions[0]) is used as the foreground probability.
            Values should be in range [0, 1] (normalized) or [0, 255] (uint8).
        threshold_range (tuple): Tuple of (min_threshold, max_threshold) for binarization.
            Only the minimum threshold is used. Values >= min_threshold become foreground (1).
            Default: (0.8, 1.0)

    Returns:
        numpy.ndarray: Binary segmentation mask with shape matching input spatial dimensions.
            Values: 0 (background) or 1 (foreground).
            For 3D input: shape :math:`(Z, Y, X)`
            For 2D input: shape :math:`(Y, X)`

    Examples:
        >>> # 3D predictions (normalized [0, 1])
        >>> predictions = np.random.rand(2, 64, 128, 128)  # (C, Z, Y, X)
        >>> binary_mask = decode_binary_thresholding(predictions, threshold_range=(0.8, 1.0))
        >>> print(binary_mask.shape)  # (64, 128, 128)
        >>> print(np.unique(binary_mask))  # [0, 1]

        >>> # 3D predictions (uint8 [0, 255])
        >>> predictions = np.random.randint(0, 256, (2, 64, 128, 128), dtype=np.uint8)
        >>> binary_mask = decode_binary_thresholding(predictions, threshold_range=(0.8, 1.0))

        >>> # 2D predictions
        >>> predictions = np.random.rand(2, 512, 512)  # (C, Y, X)
        >>> binary_mask = decode_binary_thresholding(predictions, threshold_range=(0.5, 1.0))
        >>> print(binary_mask.shape)  # (512, 512)

    Note:
        - **Auto-detection of value range**: Automatically handles both normalized [0, 1]
          and uint8 [0, 255] predictions
        - **2D/3D support**: Works with both 2D (C, Y, X) and 3D (C, Z, Y, X) inputs
        - **Channel 0 usage**: Uses first channel (predictions[0]) as foreground probability
        - **Simple thresholding**: No morphological operations or connected components
        - **Post-processing**: Use binary postprocessing config for refinement (opening/closing/CC filtering)

    See Also:
        - :func:`decode_binary_cc`: Binary threshold + connected components (instance segmentation)
        - :func:`decode_binary_watershed`: Binary threshold + watershed (instance segmentation)
        - :class:`connectomics.config.BinaryPostprocessingConfig`: For morphological refinement
    """
    # Extract foreground probability (first channel)
    semantic = predictions[0]

    # Auto-detect if predictions are in [0, 1] or [0, 255] range
    max_value = np.max(semantic)
    if max_value > 1.0:
        # Assume uint8 range [0, 255]
        threshold = threshold_range[0] * 255
    else:
        # Assume normalized range [0, 1]
        threshold = threshold_range[0]

    # Apply thresholding
    binary_mask = (semantic > threshold).astype(np.uint8)

    return binary_mask


def decode_binary_cc(
    predictions: np.ndarray,
    foreground_threshold: float = 0.8,
    min_instance_size: int = 128,
    scale_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    remove_small_mode: str = "background",
) -> np.ndarray:
    r"""Convert binary foreground probability maps to instance masks via
    connected-component labeling.

    Args:
        predictions (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        foreground_threshold (float): threshold of foreground. Default: 0.8
        min_instance_size (int): minimum size threshold for instances to keep. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``

    Returns:
        numpy.ndarray: Instance segmentation mask.
    """
    semantic = predictions[0]
    foreground = semantic > int(255 * foreground_threshold)
    segmentation = cc3d.connected_components(foreground)
    segmentation = remove_small_instances(
        segmentation, min_instance_size, remove_small_mode
    )

    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(semantic.shape[0] * scale_factors[0]),
            int(semantic.shape[1] * scale_factors[1]),
            int(semantic.shape[2] * scale_factors[2]),
        )
        # Calculate zoom factors for target size
        zoom_factors = [
            out_size / in_size
            for out_size, in_size in zip(target_size, segmentation.shape)
        ]
        segmentation = zoom(segmentation, zoom_factors, order=0, mode="nearest")

    return fastremap.refit(segmentation)


def decode_binary_watershed(
    predictions: np.ndarray,
    seed_threshold: float = 0.98,
    foreground_threshold: float = 0.85,
    min_instance_size: int = 128,
    scale_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    remove_small_mode: str = "background",
    min_seed_size: int = 32,
) -> np.ndarray:
    r"""Convert binary foreground probability maps to instance masks via
    watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
        function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

    Args:
        predictions (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        seed_threshold (float): threshold for identifying seed points. Default: 0.98
        foreground_threshold (float): threshold for foreground mask. Default: 0.85
        min_instance_size (int): minimum size threshold for instances to keep. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
        min_seed_size (int): minimum size of seed objects. Default: 32

    Returns:
        numpy.ndarray: Instance segmentation mask.
    """
    semantic = predictions[0]
    seed_map = semantic > int(255 * seed_threshold)
    foreground = semantic > int(255 * foreground_threshold)
    seed = cc3d.connected_components(seed_map)
    seed = remove_small_objects(seed, min_seed_size)
    segmentation = mahotas.cwatershed(
        -semantic.astype(np.float64), seed, mask=foreground
    )
    segmentation = remove_small_instances(
        segmentation, min_instance_size, remove_small_mode
    )

    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(semantic.shape[0] * scale_factors[0]),
            int(semantic.shape[1] * scale_factors[1]),
            int(semantic.shape[2] * scale_factors[2]),
        )
        # Calculate zoom factors for target size
        zoom_factors = [
            out_size / in_size
            for out_size, in_size in zip(target_size, segmentation.shape)
        ]
        segmentation = zoom(segmentation, zoom_factors, order=0, mode="nearest")
    return fastremap.refit(segmentation)


def decode_binary_contour_cc(
    predictions: np.ndarray,
    foreground_threshold: float = 0.8,
    contour_threshold: float = 0.5,
    min_instance_size: int = 128,
    scale_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    dilation_struct: Tuple[int, int, int] = (1, 5, 5),
    remove_small_mode: str = "background",
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
        predictions (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        foreground_threshold (float): threshold for foreground. Default: 0.8
        contour_threshold (float): threshold for instance contours. Default: 0.5
        min_instance_size (int): minimum size threshold for instances to keep. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        dilation_struct (tuple): the shape of the structure for morphological dilation. Default: (1, 5, 5)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``

    Returns:
        numpy.ndarray: Instance segmentation mask.
    """
    semantic = predictions[0]
    boundary = predictions[1]
    foreground = (semantic > int(255 * foreground_threshold)) * (
        boundary < int(255 * contour_threshold)
    )

    segmentation = cc3d.connected_components(foreground)
    struct = np.ones(dilation_struct)
    segmentation = dilation(segmentation, struct)
    segmentation = remove_small_instances(
        segmentation, min_instance_size, remove_small_mode
    )

    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(semantic.shape[0] * scale_factors[0]),
            int(semantic.shape[1] * scale_factors[1]),
            int(semantic.shape[2] * scale_factors[2]),
        )
        # Calculate zoom factors for target size
        zoom_factors = [
            out_size / in_size
            for out_size, in_size in zip(target_size, segmentation.shape)
        ]
        segmentation = zoom(segmentation, zoom_factors, order=0, mode="nearest")

    return fastremap.refit(segmentation)


def decode_binary_contour_watershed(
    predictions: np.ndarray,
    seed_threshold: float = 0.9,
    contour_threshold: float = 0.8,
    foreground_threshold: float = 0.85,
    min_instance_size: int = 128,
    scale_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    remove_small_mode: str = "background",
    min_seed_size: int = 32,
    return_seed: bool = False,
    precomputed_seed: Optional[np.ndarray] = None,
    prediction_scale: int = 255,
):
    r"""Convert binary foreground probability maps and instance contours to
    instance masks via watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
        function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

    Args:
        predictions (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        seed_threshold (float): threshold for identifying seed points. Default: 0.9
        contour_threshold (float): threshold for instance contours. Default: 0.8
        foreground_threshold (float): threshold for foreground mask. Default: 0.85
        min_instance_size (int): minimum size threshold for instances to keep. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
        min_seed_size (int): minimum size of seed objects. Default: 32
        return_seed (bool): whether to return the seed map. Default: False
        precomputed_seed (numpy.ndarray, optional): precomputed seed map. Default: None
        prediction_scale (int): scale of input predictions (255 for uint8 range). Default: 255

    Returns:
        numpy.ndarray or tuple: Instance segmentation mask, or (mask, seed) if return_seed=True.
    """
    assert predictions.shape[0] == 2
    semantic, boundary = predictions[0], predictions[1]
    if prediction_scale == 255:
        seed_threshold = seed_threshold * prediction_scale
        contour_threshold = contour_threshold * prediction_scale
        foreground_threshold = foreground_threshold * prediction_scale

    foreground = semantic > foreground_threshold

    if precomputed_seed is not None:
        seed = precomputed_seed
    else:  # compute the instance seeds
        seed_map = (semantic > seed_threshold) * (boundary < contour_threshold)
        seed = cc3d.connected_components(seed_map)
        seed = remove_small_objects(seed, min_seed_size)

    segmentation = mahotas.cwatershed(
        -semantic.astype(np.float64), seed, mask=foreground
    )
    segmentation = remove_small_instances(
        segmentation, min_instance_size, remove_small_mode
    )

    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(semantic.shape[0] * scale_factors[0]),
            int(semantic.shape[1] * scale_factors[1]),
            int(semantic.shape[2] * scale_factors[2]),
        )
        # Calculate zoom factors for target size
        zoom_factors = [
            out_size / in_size
            for out_size, in_size in zip(target_size, segmentation.shape)
        ]
        segmentation = zoom(segmentation, zoom_factors, order=0, mode="nearest")

    segmentation = fastremap.refit(segmentation)

    if not return_seed:
        return segmentation

    return segmentation, seed


def decode_binary_contour_distance_watershed(
    predictions: np.ndarray,
    binary_threshold: Tuple[float, float] = (0.9, 0.85),
    contour_threshold: Tuple[float, float] = (0.8, 1.1),
    distance_threshold: Tuple[float, float] = (0.5, 0),
    min_instance_size: int = 128,
    scale_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    remove_small_mode: str = "background",
    min_seed_size: int = 32,
    return_seed: bool = False,
    precomputed_seed: Optional[np.ndarray] = None,
    prediction_scale: int = 255,
):
    r"""Convert binary foreground probability maps, instance contours and signed distance
    transform to instance masks via watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
        function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

    Args:
        predictions (numpy.ndarray): foreground, contour, and distance probability of shape :math:`(3, Z, Y, X)`.
        binary_threshold (tuple): tuple of two floats (seed_threshold, foreground_threshold) for binary mask.
            The first value is used for seed generation, the second for foreground mask. Default: (0.9, 0.85)
        contour_threshold (tuple): tuple of two floats (seed_threshold, foreground_threshold) for instance contours.
            The first value is used for seed generation, the second for foreground mask. Default: (0.8, 1.1)
        distance_threshold (tuple): tuple of two floats (seed_threshold, foreground_threshold) for signed distance.
            The first value is used for seed generation, the second for foreground mask. Default: (0.5, -0.5)
        min_instance_size (int): minimum size threshold for instances to keep. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
        min_seed_size (int): minimum size of seed objects. Default: 32
        return_seed (bool): whether to return the seed map. Default: False
        precomputed_seed (numpy.ndarray, optional): precomputed seed map. Default: None
        prediction_scale (int): scale of input predictions (255 for uint8 range). Default: 255

    Returns:
        numpy.ndarray or tuple: Instance segmentation mask, or (mask, seed) if return_seed=True.
    """
    assert predictions.shape[0] == 3
    binary, contour, distance = predictions[0], predictions[1], predictions[2]

    if prediction_scale == 255:
        distance = (distance / prediction_scale) * 2.0 - 1.0
        binary_threshold = binary_threshold * prediction_scale
        contour_threshold = contour_threshold * prediction_scale
        distance_threshold = distance_threshold * prediction_scale

    if precomputed_seed is not None:
        seed = precomputed_seed
    else:  # compute the instance seeds
        seed_map = (
            (binary > binary_threshold[0])
            * (contour < contour_threshold[0])
            * (distance > distance_threshold[0])
        )
        seed = cc3d.connected_components(seed_map)
        seed = remove_small_objects(seed, min_seed_size)

    foreground = (
        (binary > binary_threshold[1])
        * (contour < contour_threshold[1])
        * (distance > distance_threshold[1])
    )

    segmentation = mahotas.cwatershed(
        -distance.astype(np.float64), seed, mask=foreground
    )
    segmentation = remove_small_instances(
        segmentation, min_instance_size, remove_small_mode
    )

    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(binary.shape[0] * scale_factors[0]),
            int(binary.shape[1] * scale_factors[1]),
            int(binary.shape[2] * scale_factors[2]),
        )
        # Calculate zoom factors for target size
        zoom_factors = [
            out_size / in_size
            for out_size, in_size in zip(target_size, segmentation.shape)
        ]
        segmentation = zoom(segmentation, zoom_factors, order=0, mode="nearest")

    segmentation = fastremap.refit(segmentation)
    if not return_seed:
        return segmentation

    return segmentation, seed


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
                        if (
                            x - 1 >= 0
                            and hard_aff[0, x - 1, y, z]
                            and not visited[x - 1, y, z]
                        ):
                            stack_x[stack_size] = x - 1
                            stack_y[stack_size] = y
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x - 1, y, z] = True

                        # Negative y
                        if (
                            y - 1 >= 0
                            and hard_aff[1, x, y - 1, z]
                            and not visited[x, y - 1, z]
                        ):
                            stack_x[stack_size] = x
                            stack_y[stack_size] = y - 1
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x, y - 1, z] = True

                        # Negative z
                        if (
                            z - 1 >= 0
                            and hard_aff[2, x, y, z - 1]
                            and not visited[x, y, z - 1]
                        ):
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
    min_instance_size: int = 0,
    scale_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    remove_small_mode: str = "background",
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
        >>> segmentation = decode_affinity_cc(affinities, threshold=0.5)
        >>> print(segmentation.shape)  # (128, 128, 128)
        >>> print(segmentation.max())  # Number of instances

        >>> # Remove small objects
        >>> segmentation = decode_affinity_cc(
        ...     affinities,
        ...     threshold=0.5,
        ...     min_instance_size=100  # Remove objects < 100 voxels
        ... )

        >>> # Resize output
        >>> segmentation = decode_affinity_cc(
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
        - :func:`decode_binary_cc`: Connected components on binary masks
        - :func:`decode_binary_contour_watershed`: Watershed on binary + contour predictions
    """
    assert affinities.ndim == 4, f"Expected 4D array, got {affinities.ndim}D"
    assert (
        affinities.shape[0] >= 3
    ), f"Expected >= 3 channels, got {affinities.shape[0]}"

    # Extract short-range affinities (first 3 channels)
    short_range_aff = affinities[:3]

    # Binarize affinities
    hard_aff = short_range_aff > threshold

    # Connected components
    if use_numba and NUMBA_AVAILABLE:
        # Fast Numba implementation (10-100x speedup)
        segmentation = _connected_components_3d_numba(hard_aff)
    else:
        # Fallback to skimage (slower but always available)
        if use_numba and not NUMBA_AVAILABLE:
            import warnings

            warnings.warn(
                "Numba not available. Using skimage (slower). "
                "Install numba for 10-100x speedup: pip install numba>=0.60.0",
                UserWarning,
            )

        # Create foreground mask (any affinity > 0)
        foreground = hard_aff.any(axis=0)
        segmentation = cc3d.connected_components(foreground)

    # Remove small instances
    if min_instance_size > 0:
        segmentation = remove_small_instances(
            segmentation, min_instance_size, remove_small_mode
        )

    # Resize if requested
    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(segmentation.shape[0] * scale_factors[0]),
            int(segmentation.shape[1] * scale_factors[1]),
            int(segmentation.shape[2] * scale_factors[2]),
        )
        # Calculate zoom factors for target size
        zoom_factors = [
            out_size / in_size
            for out_size, in_size in zip(target_size, segmentation.shape)
        ]
        segmentation = zoom(segmentation, zoom_factors, order=0, mode="nearest")

    return fastremap.refit(segmentation)
