import numpy as np

class DataAugment(object):
    r"""
    DataAugment interface.

    1. Randomly generate parameters for current augmentation methods.
    2. Apply the paramters to the valid_mask and generate a new one.
    3. Calculate new input size and sample from the volume.
    4. Apply the augmentation method to the new sample and crop to input_size.
    """
    def __init__(self, input_size):
        self.input_size = input_size
        self.valid_mask = np.ones(self.input_size, dtype=np.uint8)

    def calculate(self, param):
        r"""Calculate appropriate input wize with data augmentation.

        Some data augmentation (wrap, mis-alignment etc.) require larger 
	sample size than the original, depending on the augmentation parameters
	that are randomly chosen. For such cases, here we determine random augmentation 
	parameters and return an updated input size accordingly.
        """
        raise NotImplementedError

    def random_param(self, seed):
        r"""Generate random parameters for augmentation
        """
        raise NotImplementedError

    def __call__(self, sample, param):
        r"""Apply data augmentation."""
        raise NotImplementedError
