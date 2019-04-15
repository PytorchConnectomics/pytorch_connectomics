import numpy as np
__all__ = ['DataAugment']


class DataAugment(object):
    """
    DataAugment interface.

    1. Randomly generate parameters for current augmentation methods.
    2. Apply the paramters to the valid_mask and generate a new one.
    3. Calculate new input size and sample from the volume.
    4. Apply the augmentation method to the new sample and crop to input_size.
    """
    def __init__(self, p=0.5):
        assert p >= 0.0 and p <=1.0
        self.p = p
        self.sample_params = {
            'ratio': np.array([1.0, 1.0, 1.0]),
            'add': np.array([0, 0, 0])}

    def set_params(self):
        """
        Calculate appropriate input wize with data augmentation.
        
        Some data augmentation (wrap, mis-alignment etc.) require larger 
        sample size than the original, depending on the augmentation parameters
	    that are randomly chosen. For such cases, here we determine random augmentation 
	    parameters and return an updated input size accordingly.
        """
        raise NotImplementedError

    def __call__(self, data, random_state):
        """Apply data augmentation
        """