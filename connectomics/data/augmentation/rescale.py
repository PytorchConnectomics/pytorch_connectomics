from __future__ import print_function, division
from typing import Optional

import numpy as np
from .augmentor import DataAugment
from skimage.transform import resize

class Rescale(DataAugment):
    r"""
    Rescale augmentation. This augmentation is applied to both images and masks.

    Args:
        low (float): lower bound of the random scale factor. Default: 0.8
        high (float): higher bound of the random scale factor. Default: 1.2
        fix_aspect (bool): fix aspect ratio or not. Default: False
        p (float): probability of applying the augmentation. Default: 0.5
        additional_targets(dict, optional): additional targets to augment. Default: None
    """

    interpolation = {'img': 1, 'mask': 0}
    anti_aliasing = {'img': True, 'mask': False}

    def __init__(self,
                 low: float = 0.8,
                 high: float = 1.25,
                 fix_aspect: bool = False,
                 p: float = 0.5,
                 additional_targets: Optional[dict] = None,
                 skip_targets: list = []):

        super(Rescale, self).__init__(p, additional_targets, skip_targets)
        self.low = low
        self.high = high
        self.fix_aspect = fix_aspect

        self.set_params()

    def set_params(self):
        r"""The rescale augmentation is only applied to the `xy`-plane. The required
        sample size before transformation need to be larger as decided by the lowest
        scaling factor (:attr:`self.low`).
        """
        assert (self.low >= 0.5)
        assert (self.low <= 1.0)
        ratio = 1.0 / self.low
        self.sample_params['ratio'] = [1.0, ratio, ratio]

    def random_scale(self, random_state):
        rand_scale = random_state.rand() * (self.high - self.low) + self.low
        rand_scale = 1.0 / rand_scale
        return rand_scale

    def _get_coord(self, sf, images, axis, random_state):
        length = int(sf * images.shape[axis])
        if length <= images.shape[axis]:
            start = random_state.randint(0, images.shape[axis]-length+1)
            end = start + length
            mode = 'upscale'
        else:
            start = int(np.floor((length - images.shape[axis]) / 2))
            end = int(np.ceil((length - images.shape[axis]) / 2))
            mode = 'downscale'
        return start, end, mode

    def get_random_params(self, images, random_state):
        if self.fix_aspect:
            sf_x = self.random_scale(random_state)
            sf_y = sf_x
        else:
            sf_x = self.random_scale(random_state)
            sf_y = self.random_scale(random_state)

        y0, y1, y_mode = self._get_coord(sf_y, images, 1, random_state)
        x0, x1, x_mode = self._get_coord(sf_x, images, 2, random_state)

        x_params = (x0, x1, x_mode)
        y_params = (y0, y1, y_mode)

        return x_params, y_params

    def apply_rescale(self, image, x_params, y_params, target_type='img'):
        x0, x1, x_mode = x_params
        y0, y1, y_mode = y_params
        transformed_image = image.copy()

        # process y-axis
        if y_mode == 'upscale':
            transformed_image = transformed_image[:, y0:y1, :]
        else:
            transformed_image = np.pad(transformed_image, ((0, 0),(y0, y1),(0, 0)),
                                       mode='constant')

        # process x-axis
        if x_mode == 'upscale':
            transformed_image = transformed_image[:, :, x0:x1]
        else:
            transformed_image = np.pad(transformed_image, ((0, 0),(0, 0),(x0, x1)),
                                       mode='constant')

        output_image = resize(transformed_image, image.shape, order=self.interpolation[target_type],
                              mode='constant', cval=0, clip=True, preserve_range=True,
                              anti_aliasing=self.anti_aliasing[target_type])
        return output_image

    def __call__(self, sample, random_state=np.random.RandomState()):
        images = sample['image'].copy()
        x_params, y_params = self.get_random_params(images, random_state)
        sample['image'] = self.apply_rescale(images, x_params, y_params, 'img')

        for key in self.additional_targets.keys():
            if key not in self.skip_targets:
                sample[key] = self.apply_rescale(sample[key].copy(), x_params, y_params, 
                    target_type = self.additional_targets[key])

        return sample
