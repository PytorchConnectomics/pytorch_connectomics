from __future__ import print_function, division
from typing import Optional

import numpy as np
from .augmentor import DataAugment

from skimage.draw import line
from scipy.ndimage.morphology import binary_dilation

class MissingParts(DataAugment):
    r"""Missing-parts augmentation of image stacks. This augmentation is only
    applied to images.

    Args:
        iterations (int): number of iterations in binary dilation. Default: 64
        p (float): probability of applying the augmentation. Default: 0.5
        additional_targets(dict, optional): additional targets to augment. Default: None
    """
    def __init__(self,
                 iterations: int = 64,
                 p: float = 0.5,
                 additional_targets: Optional[dict] = None,
                 skip_targets: list = []):
        super(MissingParts, self).__init__(p, additional_targets, skip_targets)
        self.iterations = iterations
        self.set_params()

    def set_params(self):
        r"""There is no change in sample size.
        """
        pass

    def prepare_slice_mask(self, shape, random_state):
        # randomly choose fixed x or fixed y with p = 1/2
        fixed_x = random_state.rand() < 0.5
        if fixed_x:
            x0, y0 = 0, np.random.randint(1, shape[1] - 2)
            x1, y1 = shape[0] - 1, np.random.randint(1, shape[1] - 2)
        else:
            x0, y0 = random_state.randint(1, shape[0] - 2), 0
            x1, y1 = random_state.randint(1, shape[0] - 2), shape[1] - 1

        # generate the mask of the line that should be blacked out
        line_mask = np.zeros(shape, dtype='bool')
        rr, cc = line(x0, y0, x1, y1)
        line_mask[rr, cc] = 1

        # dilate the line mask
        line_mask = binary_dilation(line_mask, iterations=self.iterations)

        return line_mask

    def deform_2d(self, image2d, transform_params):
        line_mask = transform_params
        section = image2d.squeeze()
        mean = section.mean()

        section[line_mask] = mean
        return section

    def apply_deform(self, images, transforms):
        transformedimgs = np.copy(images)
        num_section = images.shape[0]

        for i in range(num_section):
            if i in transforms.keys():
                transformedimgs[i] = self.deform_2d(images[i], transforms[i])
        return transformedimgs

    def get_random_params(self, images, random_state):
        num_section = images.shape[0]
        slice_shape = images.shape[1:]
        transforms = {}

        i=0
        while i < num_section:
            if random_state.rand() < self.p:
                transforms[i] = self.prepare_slice_mask(slice_shape, random_state)
                i += 1 # at most one deformed image in any consecutive 2 images
            i += 1
        return transforms

    def __call__(self, sample, random_state=np.random.RandomState()):
        images = sample['image'].copy()
        transforms = self.get_random_params(images, random_state)
        sample['image'] = self.apply_deform(images, transforms)

        # apply the same augmentation to other images
        for key in self.additional_targets.keys():
            if key not in self.skip_targets and self.additional_targets[key] == 'img':
                sample[key] = self.apply_deform(sample[key].copy(), transforms)

        return sample
