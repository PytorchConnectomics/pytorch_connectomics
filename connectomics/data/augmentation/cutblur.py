from __future__ import print_function, division
from typing import Optional

import numpy as np
from .augmentor import DataAugment
from skimage.transform import resize

class CutBlur(DataAugment):
    r"""3D CutBlur data augmentation, adapted from https://arxiv.org/abs/2004.00448.

    Randomly downsample a cuboid region in the volume to force the model
    to learn super-resolution when making predictions. This augmentation
    is only applied to images.

    Args:
        length_ratio (float): the ratio of the cuboid length compared with volume length.
        down_ratio_min (float): minimal downsample ratio to generate low-res region.
        down_ratio_max (float): maximal downsample ratio to generate low-res region.
        downsample_z (bool): downsample along the z axis (default: False).
        p (float): probability of applying the augmentation. Default: 0.5
        additional_targets(dict, optional): additional targets to augment. Default: None
    """

    def __init__(self,
                 length_ratio: float = 0.25,
                 down_ratio_min: float = 2.0,
                 down_ratio_max: float = 8.0,
                 downsample_z: bool = False,
                 p: float = 0.5,
                 additional_targets: Optional[dict] = None,
                 skip_targets: list = []):
        super(CutBlur, self).__init__(p, additional_targets, skip_targets)
        self.length_ratio = length_ratio
        self.down_ratio_min = down_ratio_min
        self.down_ratio_max = down_ratio_max
        self.downsample_z = downsample_z

    def set_params(self):
        r"""There is no change in sample size.
        """
        pass

    def cut_blur(self, images, zl, zh, yl, yh, xl, xh, down_ratio):
        zdim = images.shape[0]
        if zdim == 1:
            temp = images[:, yl:yh, xl:xh].copy()
        else:
            temp = images[zl:zh, yl:yh, xl:xh].copy()

        if zdim > 1 and self.downsample_z:
            out_shape = np.array(temp.shape) /  down_ratio
        else:
            out_shape = np.array(temp.shape) /  np.array([1, down_ratio, down_ratio])

        out_shape = out_shape.astype(int)
        downsampled = resize(temp, out_shape, order=1, mode='reflect',
                             clip=True, preserve_range=True, anti_aliasing=True)
        upsampled = resize(downsampled, temp.shape, order=0, mode='reflect',
                             clip=True, preserve_range=True, anti_aliasing=False)

        if zdim == 1:
            images[:, yl:yh, xl:xh] = upsampled
        else:
            images[zl:zh, yl:yh, xl:xh] = upsampled

        return images

    def random_region(self, vol_len, random_state):
        cuboid_len = int(self.length_ratio * vol_len)
        low = random_state.randint(0, vol_len-cuboid_len)
        high = low + cuboid_len
        return low, high

    def get_random_params(self, images, random_state):
        zdim = images.shape[0]
        zl, zh = None, None
        if zdim > 1:
            zl, zh = self.random_region(images.shape[0], random_state)
        yl, yh = self.random_region(images.shape[1], random_state)
        xl, xh = self.random_region(images.shape[2], random_state)

        down_ratio = random_state.uniform(self.down_ratio_min, self.down_ratio_max)
        return zl, zh, yl, yh, xl, xh, down_ratio

    def __call__(self, sample, random_state=np.random.RandomState()):
        images = sample['image'].copy()
        random_params = self.get_random_params(images, random_state)

        sample['image'] = self.cut_blur(images, *random_params)
        for key in self.additional_targets.keys():
            if key not in self.skip_targets and self.additional_targets[key] == 'img':
                sample[key] = self.cut_blur(sample[key].copy(), *random_params)
        return sample
