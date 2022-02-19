from __future__ import print_function, division
from typing import Optional

import numpy as np
from .augmentor import DataAugment

class Grayscale(DataAugment):
    r"""Grayscale intensity augmentation, adapted from ELEKTRONN (http://elektronn.org/).

    Randomly adjust contrast/brightness, randomly invert the color space
    and apply gamma correction. This augmentation is only applied to images.

    Args:
        contrast_factor (float): intensity of contrast change. Default: 0.3
        brightness_factor (float): intensity of brightness change. Default: 0.3
        mode (string): one of ``'2D'``, ``'3D'`` or ``'mix'``. Default: ``'mix'``
        invert (bool): whether to invert the images. Default: False
        invert_p (float): probability of inverting the images. Default: 0.0
        p (float): probability of applying the augmentation. Default: 0.5
        additional_targets(dict, optional): additional targets to augment. Default: None
    """

    def __init__(self,
                 contrast_factor: float = 0.3,
                 brightness_factor: float = 0.3,
                 mode: str = 'mix',
                 invert: bool = False,
                 invert_p: float = 0.0,
                 p: float = 0.5,
                 additional_targets: Optional[dict] = None,
                 skip_targets: list = []):

        super(Grayscale, self).__init__(p, additional_targets, skip_targets)
        self._set_mode(mode)
        self.invert = invert
        self.invert_p = invert_p
        self.CONTRAST_FACTOR   = contrast_factor
        self.BRIGHTNESS_FACTOR = brightness_factor

    def set_params(self):
        r"""There is no change in sample size.
        """
        pass

    def __call__(self, sample, random_state=np.random.RandomState()):

        if self.mode == 'mix':
            mode = '3D' if random_state.rand() > 0.5 else '2D'
        else:
            mode = self.mode

        images = sample['image'].copy()
        assert mode in ['2D', '3D']
        if self.mode == '2D':
            aug_func = self._augment2D
            ran = random_state.rand(images.shape[-3]*3)
        else:
            aug_func = self._augment3D
            ran = random_state.rand(3)

        do_invert = self.invert and random_state.rand() < self.invert_p
        sample['image'] = aug_func(images, ran, do_invert)
        for key in self.additional_targets.keys():
            if key not in self.skip_targets and self.additional_targets[key] == 'img':
                sample[key] = aug_func(sample[key].copy(), ran, do_invert)

        return sample

    def _augment2D(self, imgs, ran, do_invert=False):
        r"""
        Adapted from ELEKTRONN (http://elektronn.org/).
        """
        transformedimgs = np.copy(imgs)
        for z in range(transformedimgs.shape[-3]):
            img = transformedimgs[z, :, :]
            img *= 1 + (ran[z*3] - 0.5)*self.CONTRAST_FACTOR
            img += (ran[z*3+1] - 0.5)*self.BRIGHTNESS_FACTOR
            img = np.clip(img, 0, 1)
            img **= 2.0**(ran[z*3+2]*2 - 1)
            transformedimgs[z, :, :] = img

        if do_invert:
            return self._invert(transformedimgs)
        return transformedimgs

    def _augment3D(self, imgs, ran, do_invert=False):
        r"""
        Adapted from ELEKTRONN (http://elektronn.org/).
        """
        transformedimgs = np.copy(imgs)
        transformedimgs *= 1 + (ran[0] - 0.5)*self.CONTRAST_FACTOR
        transformedimgs += (ran[1] - 0.5)*self.BRIGHTNESS_FACTOR
        transformedimgs = np.clip(transformedimgs, 0, 1)
        transformedimgs **= 2.0**(ran[2]*2 - 1)

        if do_invert:
            return self._invert(transformedimgs)
        return transformedimgs

    def _invert(self, imgs):
        r"""
        Invert input images.
        """
        transformedimgs = np.copy(imgs)
        transformedimgs = 1.0-transformedimgs
        transformedimgs = np.clip(transformedimgs, 0, 1)

        return transformedimgs

    ####################################################################
    ## Setters.
    ####################################################################

    def _set_mode(self, mode):
        r"""Set 2D/3D/mix greyscale value augmentation mode."""
        assert mode=='2D' or mode=='3D' or mode=='mix'
        self.mode = mode
