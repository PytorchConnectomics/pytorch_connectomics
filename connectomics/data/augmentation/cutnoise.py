from __future__ import print_function, division
from typing import Optional

import numpy as np
from .augmentor import DataAugment

class CutNoise(DataAugment):
    r"""3D CutNoise data augmentation.

    Randomly add noise to a cuboid region in the volume to force the model
    to learn denoising when making predictions. This augmentation is only
    applied to images.

    Args:
        length_ratio (float): the ratio of the cuboid length compared with volume length.
        mode (string): the distribution of the noise pattern. Default: ``'uniform'``.
        scale (float): scale of the random noise. Default: 0.2.
        p (float): probability of applying the augmentation. Default: 0.5
        additional_targets(dict, optional): additional targets to augment. Default: None
    """

    def __init__(self,
                 length_ratio: float = 0.25,
                 mode: str = 'uniform',
                 scale: float = 0.2,
                 p: float = 0.5,
                 additional_targets: Optional[dict] = None,
                 skip_targets: list = []):

        super(CutNoise, self).__init__(p, additional_targets, skip_targets)
        self.length_ratio = length_ratio
        self.mode = mode
        self.scale = scale

    def set_params(self):
        r"""There is no change in sample size.
        """
        pass

    def cut_noise(self, images, zl, zh, yl, yh, xl, xh, noise):
        zdim = images.shape[0]
        if zdim == 1:
            temp = images[:, yl:yh, xl:xh].copy()
        else:
            temp = images[zl:zh, yl:yh, xl:xh].copy()
        temp = temp + noise
        temp = np.clip(temp, 0, 1)

        if zdim == 1:
            images[:, yl:yh, xl:xh] = temp
        else:
            images[zl:zh, yl:yh, xl:xh] = temp
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

        z_len = zh - zl if zdim > 1 else 1
        noise_shape = (z_len, yh-yl, xh-xl)
        noise = random_state.uniform(-self.scale, self.scale, noise_shape)
        return zl, zh, yl, yh, xl, xh, noise

    def __call__(self, sample, random_state=np.random.RandomState()):
        images = sample['image'].copy()
        random_params = self.get_random_params(images, random_state)

        sample['image'] = self.cut_noise(images, *random_params)
        for key in self.additional_targets.keys():
            if key not in self.skip_targets and self.additional_targets[key] == 'img':
                sample[key] = self.cut_noise(sample[key].copy(), *random_params)
        return sample
