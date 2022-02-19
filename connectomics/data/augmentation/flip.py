from __future__ import print_function, division
from typing import Optional

import numpy as np
from .augmentor import DataAugment

class Flip(DataAugment):
    r"""
    Randomly flip along `z`-, `y`- and `x`-axes as well as swap `y`- and `x`-axes
    for anisotropic image volumes. For learning on isotropic image volumes set
    :attr:`do_ztrans` to 1 to swap `z`- and `x`-axes (the inputs need to be cubic).
    This augmentation is applied to both images and masks.

    Args:
        do_ztrans (int): set to 1 to swap z- and x-axes for isotropic data. Default: 0
        p (float): probability of applying the augmentation. Default: 0.5
        additional_targets(dict, optional): additional targets to augment. Default: None
    """
    def __init__(self,
                 do_ztrans: int = 0,
                 p: float = 0.5,
                 additional_targets: Optional[dict] = None,
                 skip_targets: list = []):

        super(Flip, self).__init__(p, additional_targets, skip_targets)
        self.do_ztrans = do_ztrans

    def set_params(self):
        r"""There is no change in sample size.
        """
        pass

    def flip_and_swap(self, data, rule):
        assert data.ndim==3 or data.ndim==4
        if data.ndim == 3: # 3-channel input in z,y,x
            # z reflection.
            if rule[0]:
                data = data[::-1, :, :]
            # y reflection.
            if rule[1]:
                data = data[:, ::-1, :]
            # x reflection.
            if rule[2]:
                data = data[:, :, ::-1]
            # Transpose in xy.
            if rule[3]:
                data = data.transpose(0, 2, 1)
            # Transpose in xz.
            if self.do_ztrans==1 and rule[4]:
                data = data.transpose(2, 1, 0)
        else: # 4-channel input in c,z,y,x
            # z reflection.
            if rule[0]:
                data = data[:, ::-1, :, :]
            # y reflection.
            if rule[1]:
                data = data[:, :, ::-1, :]
            # x reflection.
            if rule[2]:
                data = data[:, :, :, ::-1]
            # Transpose in xy.
            if rule[3]:
                data = data.transpose(0, 1, 3, 2)
            # Transpose in xz.
            if self.do_ztrans==1 and rule[4]:
                data = data.transpose(0, 3, 2, 1)
        return data

    def __call__(self, sample, random_state=np.random.RandomState()):
        rule = random_state.randint(2, size=4+self.do_ztrans)
        sample['image'] = self.flip_and_swap(sample['image'].copy(), rule)
        for key in self.additional_targets.keys():
            if key not in self.skip_targets:
                sample[key] = self.flip_and_swap(sample[key].copy(), rule)

        return sample
