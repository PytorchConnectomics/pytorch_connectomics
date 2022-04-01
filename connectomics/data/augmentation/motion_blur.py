from __future__ import print_function, division
from typing import Optional

import cv2
import random
import numpy as np
from .augmentor import DataAugment

class MotionBlur(DataAugment):
    r"""Motion blur data augmentation of image stacks. This augmentation is only
    applied to images.

    Args:
        sections (int): number of sections along z dimension to apply motion blur. Default: 2
        kernel_size (int): kernel size for motion blur. Default: 11
        p (float): probability of applying the augmentation. Default: 0.5
        additional_targets(dict, optional): additional targets to augment. Default: None
    """
    def __init__(self,
                 sections: int = 2,
                 kernel_size: int = 11,
                 p: float = 0.5,
                 additional_targets: Optional[dict] = None,
                 skip_targets: list = []):

        super(MotionBlur, self).__init__(p, additional_targets, skip_targets)
        self.size = kernel_size
        self.sections = sections
        self.set_params()

    def set_params(self):
        r"""There is no change in sample size.
        """
        pass

    def motion_blur(self, images, kernel_motion_blur, selected_idx):
        for idx in selected_idx:
            # applying the kernel to the input image
            images[idx] = cv2.filter2D(images[idx], -1, kernel_motion_blur)

        return images

    def get_random_params(self, images, random_state):
        # generating the kernel
        kernel_motion_blur = np.zeros((self.size, self.size))
        if random.random() > 0.5: # horizontal kernel
            kernel_motion_blur[int((self.size-1)/2), :] = np.ones(self.size)
        else: # vertical kernel
            kernel_motion_blur[:, int((self.size-1)/2)] = np.ones(self.size)
        kernel_motion_blur = kernel_motion_blur / self.size

        k = min(self.sections, images.shape[0])
        selected_idx = random_state.choice(images.shape[0], k, replace=False)
        return kernel_motion_blur, selected_idx

    def __call__(self, sample, random_state=np.random.RandomState()):
        images = sample['image'].copy()
        kernel_motion_blur, selected_idx = self.get_random_params(images, random_state)

        sample['image'] = self.motion_blur(images, kernel_motion_blur, selected_idx)
        for key in self.additional_targets.keys():
            if key not in self.skip_targets and self.additional_targets[key] == 'img':
                sample[key] = self.motion_blur(sample[key].copy(),
                        kernel_motion_blur, selected_idx)
        return sample
