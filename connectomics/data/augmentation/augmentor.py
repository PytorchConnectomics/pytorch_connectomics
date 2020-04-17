import numpy as np

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
        Calculate appropriate sample wize with data augmentation.
        
        Some data augmentations (wrap, misalignment, etc.) require a larger sample 
        size than the original, depending on the augmentation parameters that are 
        randomly chosen. This function takes parameters for random data augmentation 
        parameters and returns an updated input size accordingly.
        """
        raise NotImplementedError

    def __call__(self, data, random_state=None):
        """
        Apply data augmentation

        For a multi-CPU dataloader, may need to use a unique index to generate 
        the random seed (random_state), otherwise different workers may generate
        the same pseudo-random number for augmentation and sampling.
        """
        raise NotImplementedError
