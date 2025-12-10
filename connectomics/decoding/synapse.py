"""
Synaptic polarity decoding functions.

Post-processing functions for synaptic polarity model outputs as described
in "Two-Stream Active Query Suggestion for Active Learning in Connectomics"
(ECCV 2020, https://zudi-lin.github.io/projects/#two_stream_active).

Functions:
    - polarity2instance: Convert synaptic polarity predictions to instance masks
"""

from __future__ import print_function, division
from typing import Tuple
import numpy as np

from skimage.morphology import binary_dilation, remove_small_objects
from scipy.ndimage import zoom
import cc3d

from .utils import cast2dtype


__all__ = ["polarity2instance"]


def polarity2instance(
    volume: np.ndarray,
    thres: float = 0.5,
    thres_small: int = 128,
    scale_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    semantic: bool = False,
    dilate_sz: int = 5,
    exclusive: bool = False,
) -> np.ndarray:
    r"""From synaptic polarity prediction to instance masks via connected-component
    labeling. The input volume should be a 3-channel probability map of shape :math:`(C, Z, Y, X)`
    where :math:`C=3`, representing pre-synaptic region, post-synaptic region and their
    union, respectively. The function also handles the case where the pre- and post-synaptic masks
    are exclusive (applied a softmax function before post-processing).

    Note:
        For each pair of pre- and post-synaptic segmentation, the decoding function will
        annotate pre-synaptic region as :math:`2n-1` and post-synaptic region as :math:`2n`,
        for :math:`n>0`. If :attr:`semantic=True`, all pre-synaptic pixels are labeled with 1
        while all post-synaptic pixels are labeled with 2. Both kinds of annotation are compatible
        with the ``TARGET_OPT: ['1']`` configuration in training.

    Note:
        The number of pre- and post-synaptic segments will be reported when
        setting :attr:`semantic=False`. Note that the numbers can be different
        due to either incomplete synapses touching the volume borders, or errors
        in the prediction. We thus make a conservative estimate of the total
        number of synapses by using the relatively small number among the two.

    Args:
        volume (numpy.ndarray): 3-channel probability map of shape :math:`(3, Z, Y, X)`.
        thres (float): probability threshold of foreground. Default: 0.5
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing the output volume in
            :math:`(Z, Y, X)` order. Default: :math:`(1.0, 1.0, 1.0)`
        semantic (bool): return only the semantic mask of pre- and post-synaptic
            regions. Default: False
        dilate_sz (int): define a struct of size (1, dilate_sz, dilate_sz) to
            dilate the masks. Default: 5
        exclusive (bool): whether the synaptic masks are exclusive (with softmax)
            or not. Default: False

    Returns:
        numpy.ndarray: Instance or semantic segmentation mask.

    Examples::
        >>> from connectomics.data.io import read_volume, save_volume
        >>> from connectomics.decoding import polarity2instance
        >>> volume = read_volume(input_name)
        >>> instances = polarity2instance(volume)
        >>> save_volume(output_name, instances)
    """
    if exclusive:
        idx_arr = np.argmax(volume, axis=0)
        temp = np.stack(
            [
                idx_arr == 1,
                idx_arr == 2,
                idx_arr != 0,  # union of pre- and post-synaptic masks
            ],
            axis=0,
        )
    else:
        thres = int(255.0 * thres)
        temp = volume > thres  # boolean array

    del volume
    syn_pre = temp[0]
    syn_pre &= temp[2]
    syn_pre = remove_small_objects(syn_pre, min_size=thres_small, connectivity=1)

    syn_post = temp[1]
    syn_post &= temp[2]
    syn_post = remove_small_objects(syn_post, min_size=thres_small, connectivity=1)

    if semantic:
        # Generate only the semantic mask. The pre-synaptic region is labeled
        # with 1, while the post-synaptic region is labeled with 2.
        segm = np.maximum(syn_pre.astype(np.uint8), syn_post.astype(np.uint8) * 2)

    else:  # Generate the instance mask.
        # The pre- and post-synaptic masks may not touch each other. Dilating the
        # union masks to define each synapse instance.
        foreground = binary_dilation(temp[2], np.ones((1, dilate_sz, dilate_sz), bool))
        del temp

        foreground = cast2dtype(cc3d.connected_components(foreground, connectivity=6))

        # Since non-zero pixels in seg_pos and seg_neg are subsets of temp[2],
        # they are naturally subsets of non-zero pixels in foreground.
        seg_pre = (foreground * 2 - 1) * syn_pre.astype(foreground.dtype)
        del syn_pre
        seg_post = (foreground * 2) * syn_post.astype(foreground.dtype)
        del syn_post
        segm = np.maximum(seg_pre, seg_post)

        # Report the number of synapses
        num_pre = len(np.unique(seg_pre)) - 1
        num_post = len(np.unique(seg_post)) - 1
        num_syn = min(num_pre, num_post)  # a conservative estimate
        print(f"Stats: found {num_pre} pre- and {num_post} post-synaptic segments.")
        print(f"There are {num_syn} synapses under a conservative estimate.")
        del seg_pre, seg_post

    # resize the segmentation based on specified scale factors
    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(segm.shape[0] * scale_factors[0]),
            int(segm.shape[1] * scale_factors[1]),
            int(segm.shape[2] * scale_factors[2]),
        )
        # Calculate zoom factors for target size
        zoom_factors = [out_size / in_size for out_size, in_size in zip(target_size, segm.shape)]
        segm = zoom(segm, zoom_factors, order=0, mode="nearest")

    return cast2dtype(segm)
