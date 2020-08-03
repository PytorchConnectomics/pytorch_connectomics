# Post-processing functions for synaptic polarity model outputs as described
# in "Two-Stream Active Query Suggestion for Active Learning in Connectomics 
# (ECCV 2020, https://zudi-lin.github.io/projects/#two_stream_active)".
import numpy as np

from skimage.measure import label
from skimage.transform import resize
from skimage.morphology import remove_small_objects, dilation

from connectomics.data.utils import getSegType

def polarity2instance(volume, thres=0.5, thres_small=128, 
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
    
    # Cast the segmentation to the best dtype to save memory.
    max_id = np.amax(np.unique(segm))
    m_type = getSegType(int(max_id))
    segm = segm.astype(m_type)
    return segm

