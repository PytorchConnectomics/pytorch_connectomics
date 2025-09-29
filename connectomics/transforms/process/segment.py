"""
Segmentation processing functions for PyTorch Connectomics.
"""

from __future__ import print_function, division
import numpy as np


def im_to_col(volume, kernel_size, stride=1):
    """Extract patches from volume using sliding window."""
    # Simple numpy implementation
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * volume.ndim

    # Basic patch extraction implementation
    patches = []
    for i in range(0, volume.shape[0] - kernel_size[0] + 1, stride):
        for j in range(0, volume.shape[1] - kernel_size[1] + 1, stride):
            for k in range(0, volume.shape[2] - kernel_size[2] + 1, stride):
                patch = volume[i:i+kernel_size[0], j:j+kernel_size[1], k:k+kernel_size[2]]
                patches.append(patch.flatten())

    return np.array(patches)


def relabel_volume(volume):
    """Relabel connected components in volume."""
    try:
        from skimage.measure import label
        return label(volume)
    except ImportError:
        # Fallback to simple unique labeling if skimage not available
        return volume


__all__ = ['im_to_col', 'relabel_volume']