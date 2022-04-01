from __future__ import print_function, division
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from .augmentor import DataAugment

class Elastic(DataAugment):
    r"""Elastic deformation of images as described in [Simard2003]_ (with modifications).
    The implementation is based on https://gist.github.com/erniejunior/601cdf56d2b424757de5.
    This augmentation is applied to both images and masks.

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.

    Args:
        alpha (float): maximum pixel-moving distance of elastic deformation. Default: 10.0
        sigma (float): standard deviation of the Gaussian filter. Default: 4.0
        p (float): probability of applying the augmentation. Default: 0.5
        additional_targets(dict, optional): additional targets to augment. Default: None
    """

    interpolation = {'img': cv2.INTER_LINEAR,
                     'mask': cv2.INTER_NEAREST}
    border_mode = cv2.BORDER_CONSTANT

    def __init__(self,
                 alpha: float = 16.0,
                 sigma: float = 4.0,
                 p: float = 0.5,
                 additional_targets: Optional[dict] = None,
                 skip_targets: list = []):

        super(Elastic, self).__init__(p, additional_targets, skip_targets)
        self.alpha = alpha
        self.sigma = sigma
        self.set_params()

    def set_params(self):
        r"""The rescale augmentation is only applied to the `xy`-plane. The required
        sample size before transformation need to be larger as decided by the maximum
        pixel-moving distance (:attr:`self.alpha`).
        """
        max_margin = int(self.alpha) + 1
        self.sample_params['add'] = [0, max_margin, max_margin]

    def elastic_wrap(self, images, mapx, mapy, target_type='img'):
        transformed_images = []

        assert images.ndim in [3, 4]
        for i in range(images.shape[-3]):
            if images.ndim == 3:
                transformed_images.append(cv2.remap(images[i], mapx, mapy,
                    self.interpolation[target_type], borderMode=self.border_mode))
            else: # multi-channel images in (c,z,y,x) format
                temp = [cv2.remap(images[channel, i], mapx, mapy, self.interpolation[target_type],
                        borderMode=self.border_mode) for channel in range(images.shape[0])]
                transformed_images.append(np.stack(temp, 0))

        axis = 0 if images.ndim == 3 else 1
        transformed_images = np.stack(transformed_images, axis)

        return transformed_images

    def get_random_params(self, images, random_state):
        height, width = images.shape[-2:] # (c, z, y, x) or (z, y, x)

        dx = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), self.sigma) * self.alpha)
        dy = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), self.sigma) * self.alpha)

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        mapx, mapy = np.float32(x + dx), np.float32(y + dy)
        return mapx, mapy

    def __call__(self, sample, random_state=np.random.RandomState()):
        images = sample['image'].copy()
        mapx, mapy = self.get_random_params(images, random_state)
        sample['image'] = self.elastic_wrap(images, mapx, mapy, 'img')

        for key in self.additional_targets.keys():
            if key not in self.skip_targets:
                sample[key] = self.elastic_wrap(sample[key].copy(), mapx, mapy,
                    target_type = self.additional_targets[key])

        return sample
