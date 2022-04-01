from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
from typing import Optional
import numpy as np

class DataAugment(object, metaclass=ABCMeta):
    r"""
    DataAugment interface. A data augmentor needs to conduct the following steps:

    1. Set :attr:`sample_params` at initialization to compute required sample size.
    2. Randomly generate augmentation parameters for the current transform.
    3. Apply the transform to a pair of images and corresponding labels.

    All the real data augmentations (except mix-up augmentor and test-time augmentor)
    should be a subclass of this class.

    Args:
        p (float): probability of applying the augmentation. Default: 0.5
        additional_targets(dict, optional): additional targets to augment. Default: None
    """
    def __init__(self,
                 p: float = 0.5,
                 additional_targets: Optional[dict] = None,
                 skip_targets: list = []):
        super().__init__()

        assert p >= 0.0 and p <=1.0
        self.p = p
        self.sample_params = {
            'ratio': np.array([1.0, 1.0, 1.0]),
            'add': np.array([0, 0, 0])}

        if additional_targets is not None:
            self.additional_targets = additional_targets
        else: # initialize as an empty dictionary
            self.additional_targets = {}

        self.skip_targets = skip_targets

    @abstractmethod
    def set_params(self):
        r"""
        Calculate the appropriate sample size with data augmentation.

        Some data augmentations (wrap, misalignment, etc.) require a larger sample
        size than the original, depending on the augmentation parameters that are
        randomly chosen. This function takes the data augmentation
        parameters and returns an updated data sampling size accordingly.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, sample, random_state=None):
        r"""
        Apply the data augmentation.

        For a multi-CPU dataloader, one may need to use a unique index to generate
        the random seed (:attr:`random_state`), otherwise different workers may generate
        the same pseudo-random number for augmentation and sampling.

        The only required key in :attr:`sample` is ``'image'``. The keys that are not
        specified in :attr:`additional_targets` will be ignored.
        """
        raise NotImplementedError
