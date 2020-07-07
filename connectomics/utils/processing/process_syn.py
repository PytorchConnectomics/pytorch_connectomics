# Post-processing functions for synaptic polarity model outputs as described
# in "Two-Stream Active Query Suggestion for Large-Scale Object Detection in 
# Connectomics (ECCV 2020)".
import numpy as np

from skimage.measure import label
from skimage.transform import resize
from skimage.morphology import remove_small_objects, dilation

from connectomics.data.utils import getSegType

def polarity_to_instance(volume, thres=0.5, thres_small=128, 
                         scale_factors=(1.0, 1.0, 1.0), semantic=False):
    """From synaptic polarity prediction to instance masks via connected-component 
    labeling. The input volume should be a 3-channel probability map of shape :math:`(C, Z, Y, X)`
    where :math:`C=3`, representing pre-synaptic region, post-synaptic region and their
    union, respectively.

    Note:
        For each pair of pre- and post-synaptic segmentation, the decoding function will
        annotate pre-synaptic region as :math:`2n-1` and post-synaptic region as :math:`2n`,
        for :math:`n>0`. If :attr:`semantic=True`, all pre-synaptic pixels are labeled with
        while all post-synaptic pixels are labeled with 2. Both kinds of annotation are compatible
        with the ``TARGET_OPT: ['1']`` configuration in training. 

    Args: 
        volume (numpy.ndarray): 3-channel probability map of shape :math:`(3, Z, Y, X)`.
        thres (float): threshold of foreground. Default: 0.5
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: :math:`(1.0, 1.0, 1.0)`
        semantic (bool): only return the semantic mask of synaptic polarity. Default: False
    """
    thres = int(255.0 * thres)
    temp = (volume > thres).astype(np.uint8)

    syn_pos = temp[0] * temp[2]
    syn_pos = remove_small_objects(syn_pos, 
                min_size=thres_small, connectivity=1)
    syn_neg = temp[1] * temp[2]
    syn_neg = remove_small_objects(syn_neg, 
                min_size=thres_small, connectivity=1)

    if semantic: 
        # Generate only the semantic mask. The pre-synaptic region is labeled
        # with 1, while the post-synaptic region is labeled with 2.
        segm = np.maximum(syn_pos.astype(np.uint8), 
                          syn_neg.astype(np.uint8) * 2)

    else:
        # Generate the instance mask.
        foreground = dilation(temp[2].copy(), np.ones((1,5,5)))
        foreground = label(foreground)

        # Since non-zero pixels in seg_pos and seg_neg are subsets of temp[2], 
        # they are naturally subsets of non-zero pixels in foreground.
        seg_pos = (foreground*2 - 1) * syn_pos.astype(foreground.dtype)
        seg_neg = (foreground*2) * syn_neg.astype(foreground.dtype)
        segm = np.maximum(seg_pos, seg_neg)

        # Cast the segmentation to the best dtype to save memory.
        max_id = np.maximum(np.unique(segm))
        m_type = getSegType(max_id)
        segm = segm.astype(m_type)

    # resize the segmentation based on specified scale factors
    if not all(x==1.0 for x in scale_factors):
        target_size = (int(segm.shape[0]*scale_factors[0]), 
                       int(segm.shape[1]*scale_factors[1]), 
                       int(segm.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
    return segm