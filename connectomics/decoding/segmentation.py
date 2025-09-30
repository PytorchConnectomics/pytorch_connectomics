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
"""

from __future__ import print_function, division
from typing import Optional, Tuple
import numpy as np

from skimage.measure import label
from skimage.transform import resize
from skimage.morphology import dilation, remove_small_objects
from skimage.segmentation import watershed

from .utils import cast2dtype, remove_small_instances


__all__ = [
    'binary_connected',
    'binary_watershed',
    'bc_connected',
    'bc_watershed',
    'bcd_watershed',
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
