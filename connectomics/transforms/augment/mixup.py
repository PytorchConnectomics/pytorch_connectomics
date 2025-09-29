import random
import numpy as np
from numpy.core.numeric import indices
import torch
from itertools import combinations

class MixupAugmentor(object):
    r"""Mixup augmentor (experimental). Conduct linear interpolation between two image samples. 
    The segmentation mask of the sample with higher weight should be used with the augmented output. 

    The input can be a `numpy.ndarray` or `torch.Tensor` of shape :math:`(B, C, Z, Y, X)`.
    
    Args:
        min_ratio (float): minimal interpolation ratio of the target volume. Default: 0.7
        max_ratio (float): maximal interpolation ratio of the target volume. Default: 0.9
        num_aug (int): number of volumes to be augmented in a batch. Default: 2

    Examples::
        >>> from connectomics.data.augmentation import MixupAugmentor
        >>> mixup_augmentor = MixupAugmentor(num_aug=2)
        >>> volume = mixup_augmentor(volume)
        >>> pred = model(volume)
    """  
    def __init__(self, 
                 min_ratio: float = 0.7, 
                 max_ratio: float = 0.9, 
                 num_aug: int = 2):

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.num_aug = num_aug

    def __call__(self, volume):
        if isinstance(volume, torch.Tensor):
            num_vol = volume.size(0)
        elif isinstance(volume, np.ndarray):
            num_vol = volume.shape[0]
        else:
            raise TypeError("Type {} is not supported in MixupAugmentor".format(type(volume)))

        num_aug = self.num_aug if self.num_aug <= num_vol else num_vol
        indices = list(range(num_vol))

        # random sampling without replacement 
        major_idx = random.sample(indices, num_aug) 

        minor_idx = []
        for x in major_idx:
            temp = indices.copy()
            temp.remove(x)
            minor_idx.append(random.sample(temp, 1)[0])

        for i in range(len(major_idx)):
            ratio = random.uniform(self.min_ratio, self.max_ratio)
            volume[major_idx[i]] = volume[major_idx[i]] * ratio + volume[minor_idx[i]] * (1-ratio)

        return volume
