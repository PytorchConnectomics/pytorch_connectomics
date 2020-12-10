# Post-processing functions of mitochondria instance segmentation model outputs
# as described in "MitoEM Dataset: Large-scale 3D Mitochondria Instance Segmentation 
# from EM Images (MICCAI 2020, https://donglaiw.github.io/page/mitoEM/index.html)".
import numpy as np

from skimage.measure import label
from skimage.transform import resize
from skimage.morphology import dilation
from skimage.segmentation import watershed

from .utils import remove_small_instances

def binary_connected(volume, thres=0.8, thres_small=128, scale_factors=(1.0, 1.0, 1.0),
                     remove_small_mode='background'):
    """From binary foreground probability map to instance masks via
    connected-component labeling.

    Args: 
        volume (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        thres (float): threshold of foreground. Default: 0.8
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: :math:`(1.0, 1.0, 1.0)`
        remove_small_mode (str): ``'background'`` or ``'neighbor'``. Default: ``'background'``
    """
    semantic = volume[0]
    foreground = (semantic > int(255*thres))
    segm = label(foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]), 
                       int(semantic.shape[1]*scale_factors[1]), 
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
    return segm.astype(np.uint32)

def binary_watershed(volume, thres1=0.98, thres2=0.85, thres_small=128, scale_factors=(1.0, 1.0, 1.0),
                     remove_small_mode='background'):
    """From binary foreground probability map to instance masks via
    watershed segmentation algorithm.

    Args: 
        volume (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of seeds. Default: 0.98
        thres2 (float): threshold of foreground. Default: 0.85
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: :math:`(1.0, 1.0, 1.0)`
        remove_small_mode (str): ``'background'`` or ``'neighbor'``. Default: ``'background'``
    """
    semantic = volume[0]
    seed_map = semantic > int(255*thres1)
    foreground = semantic > int(255*thres2)
    seed = label(seed_map)
    segm = watershed(-semantic, seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]), 
                       int(semantic.shape[1]*scale_factors[1]), 
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
    return segm.astype(np.uint32)

def bc_connected(volume, thres1=0.8, thres2=0.5, thres_small=128, scale_factors=(1.0, 1.0, 1.0), 
                 dilation_struct=(1,5,5), remove_small_mode='background'):
    """From binary foreground probability map and instance contours to 
    instance masks via connected-component labeling.

    Note:
        The instance contour provides additional supervision to distinguish closely touching
        objects. However, the decoding algorithm only keep the intersection of foreground and 
        non-contour regions, which will systematically result in imcomplete instance masks.
        Therefore we apply morphological dilation (check :attr:`dilation_struct`) to enlarge 
        the object masks.

    Args: 
        volume (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of foreground. Default: 0.8
        thres2 (float): threshold of instance contours. Default: 0.5
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: :math:`(1.0, 1.0, 1.0)`
        dilation_struct (tuple): the shape of the structure for morphological dilation. Default: :math:`(1, 5, 5)`
        remove_small_mode (str): ``'background'`` or ``'neighbor'``. Default: ``'background'``
    """
    semantic = volume[0]
    boundary = volume[1]
    foreground = (semantic > int(255*thres1)) * (boundary < int(255*thres2))

    segm = label(foreground)
    struct = np.ones(dilation_struct)
    segm = dilation(segm, struct)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]), 
                       int(semantic.shape[1]*scale_factors[1]), 
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
    return segm.astype(np.uint32)

def bc_watershed(volume, thres1=0.9, thres2=0.8, thres3=0.85, thres_small=128, scale_factors=(1.0, 1.0, 1.0),
                 remove_small_mode='background'):
    """From binary foreground probability map and instance contours to 
    instance masks via watershed segmentation algorithm.

    Args: 
        volume (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of seeds. Default: 0.9
        thres2 (float): threshold of instance contours. Default: 0.8
        thres3 (float): threshold of foreground. Default: 0.85
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: :math:`(1.0, 1.0, 1.0)`
        remove_small_mode (str): ``'background'`` or ``'neighbor'``. Default: ``'background'``
    """
    semantic = volume[0]
    boundary = volume[1]
    seed_map = (semantic > int(255*thres1)) * (boundary < int(255*thres2))
    foreground = (semantic > int(255*thres3))
    seed = label(seed_map)
    segm = watershed(-semantic, seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]), 
                       int(semantic.shape[1]*scale_factors[1]), 
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
    return segm.astype(np.uint32)
