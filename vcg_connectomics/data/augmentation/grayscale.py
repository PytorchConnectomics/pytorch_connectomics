import numpy as np
from .augmentor import DataAugment

class Greyscale(DataAugment):
    """
    Greyscale value augmentation.

    Randomly adjust contrast/brightness, randomly invert
    and apply random gamma correction.
    """

    def __init__(self, input_size, mode='mix', skip_ratio=0.3):
        """Initialize parameters.

        Args:
            input_size: (z, y, x)
            mode: '2D', '3D', 'mix'
            skip_ratio: Probability of skipping augmentation.
        """
        super.__init__(input_size)
        self.set_mode(mode)
        self.set_skip_ratio(skip_ratio)
        self.CONTRAST_FACTOR   = 0.3
        self.BRIGHTNESS_FACTOR = 0.3

    def calculate(self):
        # No change in input size
        return self.input_size

    def __call__(self, sample, **kwargs):
        if not self.skip:
            if self.mode == 'mix':
                mode = '3D' if np.random.rand() > 0.5 else '2D'
            else:
                mode = self.mode
            if mode is '2D': self.augment2D(sample, **kwargs)
            if mode is '3D': self.augment3D(sample, **kwargs)
        return sample

    def augment2D(self, sample, **kwargs):
        """
        Adapted from ELEKTRONN (http://elektronn.org/).
        """
        imgs = kwargs['imgs']
        for key in imgs:
            for z in xrange(sample[key].shape[-3]):
                img = sample[key][...,z,:,:]
                img *= 1 + (np.random.rand() - 0.5)*self.CONTRAST_FACTOR
                img += (np.random.rand() - 0.5)*self.BRIGHTNESS_FACTOR
                img = np.clip(img, 0, 1)
                img **= 2.0**(np.random.rand()*2 - 1)
                sample[key][...,z,:,:] = img
        return sample

    def augment3D(self, sample, **kwargs):
        """
        Adapted from ELEKTRONN (http://elektronn.org/).
        """
        imgs = kwargs['imgs']
        for key in imgs:
            sample[key] *= 1 + (np.random.rand() - 0.5)*self.CONTRAST_FACTOR
            sample[key] += (np.random.rand() - 0.5)*self.BRIGHTNESS_FACTOR
            sample[key] = np.clip(sample[key], 0, 1)
            sample[key] **= 2.0**(np.random.rand()*2 - 1)
        return sample

    def invert(self, sample)
        """
        Invert input images
        """
        pass
        
    ####################################################################
    ## Setters.
    ####################################################################

    def set_mode(self, mode):
        """Set 2D/3D/mix greyscale value augmentation mode."""
        assert mode=='2D' or mode=='3D' or mode=='mix'
        self.mode = mode

    def set_skip_ratio(self, ratio):
        """Set the probability of skipping augmentation."""
        assert ratio >= 0.0 and ratio <= 1.0
        self.skip_ratio = ratio
