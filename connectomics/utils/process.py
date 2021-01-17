import numpy as np

from scipy import ndimage
from skimage.measure import label
from skimage.transform import resize
from skimage.morphology import dilation
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects

from ..data.utils import getSegType

__all__ = ['binary_connected',
           'binary_watershed',
           'bc_connected',
           'bc_watershed',
           'polarity2instance']

# Post-processing functions of mitochondria instance segmentation model outputs
# as described in "MitoEM Dataset: Large-scale 3D Mitochondria Instance Segmentation 
# from EM Images (MICCAI 2020, https://donglaiw.github.io/page/mitoEM/index.html)".
def binary_connected(volume, thres=0.8, thres_small=128, scale_factors=(1.0, 1.0, 1.0),
                     remove_small_mode='background'):
    r"""Convert binary foreground probability maps to instance masks via
    connected-component labeling.

    Args: 
        volume (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        thres (float): threshold of foreground. Default: 0.8
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
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

    return cast2dtype(segm)

def binary_watershed(volume, thres1=0.98, thres2=0.85, thres_small=128, scale_factors=(1.0, 1.0, 1.0),
                     remove_small_mode='background'):
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
    """
    semantic = volume[0]
    seed_map = semantic > int(255*thres1)
    foreground = semantic > int(255*thres2)
    seed = label(seed_map)
    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]), 
                       int(semantic.shape[1]*scale_factors[1]), 
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    return cast2dtype(segm)

def bc_connected(volume, thres1=0.8, thres2=0.5, thres_small=128, scale_factors=(1.0, 1.0, 1.0), 
                 dilation_struct=(1,5,5), remove_small_mode='background'):
    r"""Convert binary foreground probability maps and instance contours to 
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
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        dilation_struct (tuple): the shape of the structure for morphological dilation. Default: (1, 5, 5)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
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

    return cast2dtype(segm)

def bc_watershed(volume, thres1=0.9, thres2=0.8, thres3=0.85, thres_small=128, scale_factors=(1.0, 1.0, 1.0),
                 remove_small_mode='background'):
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
    """
    semantic = volume[0]
    boundary = volume[1]
    seed_map = (semantic > int(255*thres1)) * (boundary < int(255*thres2))
    foreground = (semantic > int(255*thres3))
    seed = label(seed_map)
    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]), 
                       int(semantic.shape[1]*scale_factors[1]), 
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
        
    return cast2dtype(segm)

# Post-processing functions for synaptic polarity model outputs as described
# in "Two-Stream Active Query Suggestion for Active Learning in Connectomics 
# (ECCV 2020, https://zudi-lin.github.io/projects/#two_stream_active)".
def polarity2instance(volume, thres=0.5, thres_small=128, 
                      scale_factors=(1.0, 1.0, 1.0), semantic=False):
    r"""From synaptic polarity prediction to instance masks via connected-component 
    labeling. The input volume should be a 3-channel probability map of shape :math:`(C, Z, Y, X)`
    where :math:`C=3`, representing pre-synaptic region, post-synaptic region and their
    union, respectively.

    Note:
        For each pair of pre- and post-synaptic segmentation, the decoding function will
        annotate pre-synaptic region as :math:`2n-1` and post-synaptic region as :math:`2n`,
        for :math:`n>0`. If :attr:`semantic=True`, all pre-synaptic pixels are labeled with
        while all post-synaptic pixels are labeled with 2. Both kinds of annotation are compatible
        with the ``TARGET_OPT: ['1']`` configuration in training. 

    Note:
        The number of pre- and post-synaptic segments will be reported when setting :attr:`semantic=False`.
        Note that the numbers can be different due to either incomplete syanpses touching the volume borders,
        or errors in the prediction. We thus make a conservative estimate of the total number of synapses
        by using the relatively small number among the two.

    Args: 
        volume (numpy.ndarray): 3-channel probability map of shape :math:`(3, Z, Y, X)`.
        thres (float): probability threshold of foreground. Default: 0.5
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing the output volume in :math:`(Z, Y, X)` order. Default: :math:`(1.0, 1.0, 1.0)`
        semantic (bool): return only the semantic mask of pre- and post-synaptic regions. Default: False

    Examples::
        >>> from connectomics.data.utils import readvol, savevol
        >>> from connectomics.utils.processing import polarity2instance
        >>> volume = readvol(input_name)
        >>> instances = polarity2instance(volume)
        >>> savevol(output_name, instances)
    """
    thres = int(255.0 * thres)
    temp = (volume > thres).astype(np.uint8)

    syn_pre = temp[0] * temp[2]
    syn_pre = remove_small_objects(syn_pre, 
                min_size=thres_small, connectivity=1)
    syn_post = temp[1] * temp[2]
    syn_post = remove_small_objects(syn_post, 
                min_size=thres_small, connectivity=1)

    if semantic: 
        # Generate only the semantic mask. The pre-synaptic region is labeled
        # with 1, while the post-synaptic region is labeled with 2.
        segm = np.maximum(syn_pre.astype(np.uint8), 
                          syn_post.astype(np.uint8) * 2)

    else:
        # Generate the instance mask.
        foreground = dilation(temp[2].copy(), np.ones((1,5,5)))
        foreground = label(foreground)

        # Since non-zero pixels in seg_pos and seg_neg are subsets of temp[2], 
        # they are naturally subsets of non-zero pixels in foreground.
        seg_pre = (foreground*2 - 1) * syn_pre.astype(foreground.dtype)
        seg_post = (foreground*2) * syn_post.astype(foreground.dtype)
        segm = np.maximum(seg_pre, seg_post)

        # Report the number of synapses
        num_syn_pre = len(np.unique(seg_pre))-1
        num_syn_post = len(np.unique(seg_post))-1
        num_syn = min(num_syn_pre, num_syn_post) # a conservative estimate
        print("Stats: found %d pre- and %d post-" % (num_syn_pre, num_syn_post) + 
              "synaptic segments in the volume")
        print("There are %d synapses under a conservative estimate." % num_syn)

    # resize the segmentation based on specified scale factors
    if not all(x==1.0 for x in scale_factors):
        target_size = (int(segm.shape[0]*scale_factors[0]), 
                       int(segm.shape[1]*scale_factors[1]), 
                       int(segm.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
    
    return cast2dtype(segm)

# utils for post-processing
def binarize_and_median(pred, size=(7,7,7), thres=0.8):
    """First binarize the prediction with a given threshold, and
    then conduct median filtering to reduce noise.

    pred (numpy.ndarray): predicted foreground probability within (0,1).
    size (tuple): kernal size of filtering. Default: (7,7,7)
    thres (float): threshold for binarizing the prediction. Default: 0.8
    """
    pred = (pred > thres).astype(np.uint8)
    pred = ndimage.median_filter(pred, size=size)
    return pred

def remove_small_instances(segm, thres_small=128, mode='background'):
    """Remove small spurious instances. 
    """
    assert mode in ['none', 'background', 'neighbor']
    if mode == 'none':
        return segm
    elif mode == 'background':
        return remove_small_objects(segm, thres_small)

    seg_idx = np.unique(segm)[1:]
    for idx in seg_idx:
        temp = (segm == idx).astype(np.uint8)
        if temp.sum() < thres_small:
            temp_dilated = dilation(temp, np.ones((1,3,3)))
            diff = temp_dilated - temp
            diff_mask = segm.copy()
            diff_mask[np.where(diff==0)]=0
            touch_idx, counts = np.unique(diff_mask, return_counts=True)

            if len(touch_idx) > 1 and touch_idx[0] == 0:
                touch_idx = touch_idx[1:]
                counts = counts[1:]

            segm[np.where(segm==idx)] = touch_idx[np.argmax(counts)]

    return segm

def cast2dtype(segm):
    """Cast the segmentation mask to the best dtype to save storage.
    """
    max_id = np.amax(np.unique(segm))
    m_type = getSegType(int(max_id))
    return segm.astype(m_type)