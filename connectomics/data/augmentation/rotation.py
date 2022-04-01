from __future__ import print_function, division
from typing import Optional

import cv2
import numpy as np
from .augmentor import DataAugment

class Rotate(DataAugment):
    r"""
    Continuous rotatation of the `xy`-plane.

    If the rotation degree is arbitrary, the sample size for `x`- and `y`-axes should be at
    least :math:`\sqrt{2}` times larger than the input size to ensure there is no non-valid region
    after center-crop. This augmentation is applied to both images and masks.

    Args:
        rot90 (bool): rotate the sample by only 90 degrees. Default: True
        p (float): probability of applying the augmentation. Default: 0.5
        additional_targets(dict, optional): additional targets to augment. Default: None
    """

    interpolation = {'img': cv2.INTER_LINEAR,
                     'mask': cv2.INTER_NEAREST}
    border_mode = cv2.BORDER_CONSTANT

    def __init__(self,
                 rot90: bool = True,
                 p: float = 0.5,
                 additional_targets: Optional[dict] = None,
                 skip_targets: list = []):

        super(Rotate, self).__init__(p, additional_targets, skip_targets)
        self.rot90 = rot90
        self.set_params()

    def set_params(self):
        r"""The rescale augmentation is only applied to the `xy`-plane. If :attr:`self.rot90=True`,
        then there is no change in sample size. For arbitrary rotation degree, the required
        sample size before transformation need to be :math:`\sqrt{2}` times larger.
        """
        if not self.rot90: # arbitrary rotation degree
            self.sample_params['ratio'] = [1.0, 1.42, 1.42] # sqrt(2)

    def rotate(self, imgs, M, target_type='img'):
        height, width = imgs.shape[-2:]
        transformedimgs = np.copy(imgs)
        for z in range(transformedimgs.shape[-3]):
            img = transformedimgs[z, :, :]
            dst = cv2.warpAffine(img, M ,(height,width), 1.0,
                flags=self.interpolation[target_type], borderMode=self.border_mode)
            transformedimgs[z, :, :] = dst

        return transformedimgs

    def __call__(self, sample, random_state=np.random.RandomState()):
        images = sample['image'].copy()

        if self.rot90:
            k = random_state.randint(0, 4)
            sample['image'] = np.rot90(images, k, axes=(1, 2))

            for key in self.additional_targets.keys():
                if key not in self.skip_targets:
                    sample[key] = np.rot90(sample[key].copy(), k, axes=(1, 2))

        else: # rotate the array by arbitrary degree
            height, width = images.shape[-2:]
            M = cv2.getRotationMatrix2D((height/2, width/2), random_state.rand()*360.0, 1)
            sample['image'] = self.rotate(images, M, target_type='img')

            for key in self.additional_targets.keys():
                if key not in self.skip_targets:
                    sample[key] = self.rotate(sample[key].copy(), M,
                        target_type = self.additional_targets[key])

        return sample
