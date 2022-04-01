from __future__ import print_function, division
from typing import Optional

import cv2
import math
import numpy as np
from .augmentor import DataAugment

class MisAlignment(DataAugment):
    r"""Mis-alignment data augmentation of image stacks. This augmentation is
    applied to both images and masks.

    Args:
        displacement (int): maximum pixel displacement in `xy`-plane. Default: 16
        rotate_ratio (float): ratio of rotation-based mis-alignment. Default: 0.0
        p (float): probability of applying the augmentation. Default: 0.5
        additional_targets(dict, optional): additional targets to augment. Default: None
    """
    def __init__(self,
                 displacement: int = 16,
                 rotate_ratio: float = 0.0,
                 p: float = 0.5,
                 additional_targets: Optional[dict] = None,
                 skip_targets: list = []):
        super(MisAlignment, self).__init__(p, additional_targets, skip_targets)
        self.displacement = displacement
        self.rotate_ratio = rotate_ratio
        self.set_params()

    def set_params(self):
        r"""The mis-alignment augmentation is only applied to the `xy`-plane. The required
        sample size before transformation need to be larger as decided by :attr:`self.displacement`.
        """
        self.sample_params['add'] = [0,
                                     int(math.ceil(self.displacement / 2.0)),
                                     int(math.ceil(self.displacement / 2.0))]

    def _apply_misalign(self, input, out_shape, x0, y0,
                        x1, y1, idx, mode='slip'):

        output = np.zeros(out_shape, input.dtype)
        assert mode in ['slip', 'translation']
        if mode == 'slip':
            output = input[:, y0:y0+out_shape[1], x0:x0+out_shape[2]]
            output[idx] = input[idx, y1:y1+out_shape[1], x1:x1+out_shape[2]]
        else:
            output[:idx] = input[:idx, y0:y0+out_shape[1], x0:x0+out_shape[2]]
            output[idx:] = input[idx:, y1:y1+out_shape[1], x1:x1+out_shape[2]]
        return output

    def misalignment(self, sample, random_state):
        images = sample['image'].copy()
        kwargs = {}
        out_shape = (images.shape[0],
                     images.shape[1]-self.displacement,
                     images.shape[2]-self.displacement)
        kwargs['out_shape'] = out_shape

        kwargs['x0'] = random_state.randint(self.displacement)
        kwargs['y0'] = random_state.randint(self.displacement)
        kwargs['x1'] = random_state.randint(self.displacement)
        kwargs['y1'] = random_state.randint(self.displacement)
        kwargs['idx'] = random_state.choice(np.array(range(1, out_shape[0]-1)), 1)[0]
        kwargs['mode'] = 'slip' if random_state.rand() < 0.5 else 'translation'

        sample['image'] = self._apply_misalign(images, **kwargs)
        for key in self.additional_targets.keys():
            if key not in self.skip_targets:
                sample[key] = self._apply_misalign(sample[key].copy(), **kwargs)

        return sample

    def _apply_misalign_rot(self, input, idx, M, H, W, target_type='img', mode='slip'):
        if target_type=='img':
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_NEAREST
        assert mode in ['slip', 'translation']
        if mode == 'slip':
            input[idx] = cv2.warpAffine(input[idx], M, (H,W), 1.0,
                flags=interpolation, borderMode=cv2.BORDER_CONSTANT)
        else:
            for i in range(idx, input.shape[0]):
                input[i] = cv2.warpAffine(input[i], M, (H,W), 1.0,
                    flags=interpolation, borderMode=cv2.BORDER_CONSTANT)

        return input

    def misalignment_rot(self, sample, random_state):
        images = sample['image'].copy()

        height, width = images.shape[-2:]
        assert height == width
        M = self.random_rotate_matrix(height, random_state)
        idx = random_state.choice(np.array(range(1, images.shape[0]-1)), 1)[0]
        mode = 'slip' if random_state.rand() < 0.5 else 'translation'

        sample['image'] = self._apply_misalign_rot(images, idx, M,
            height, width, target_type='img', mode=mode)
        for key in self.additional_targets.keys():
            if key not in self.skip_targets:
                target_type = self.additional_targets[key]
                sample[key] = self._apply_misalign_rot(sample[key].copy(), idx, M,
                    height, width, target_type=target_type, mode=mode)

        return sample

    def random_rotate_matrix(self, height, random_state):
        x = (self.displacement / 2.0)
        y = ((height - self.displacement) / 2.0) * 1.42
        angle = math.asin(x/y) * 2.0 * 57.2958 # convert radians to degrees
        rand_angle = (random_state.rand() - 0.5) * 2.0 * angle
        M = cv2.getRotationMatrix2D((height/2, height/2), rand_angle, 1)
        return M

    def __call__(self, sample, random_state=np.random.RandomState()):
        if random_state.rand() < self.rotate_ratio:
            sample = self.misalignment_rot(sample, random_state)
        else:
            sample = self.misalignment(sample, random_state)
        return sample
