import numpy as np
from .augmentor import DataAugment

class Grayscale(DataAugment):
    """
    Grayscale value augmentation.

    Randomly adjust contrast/brightness, randomly invert
    and apply random gamma correction.
    """

    def __init__(self, contrast_factor=0.3, brightness_factor=0.3, mode='mix', p=0.5):
        """Initialize parameters.

        Args:
            contrast_factor (float): intensity of contrast change.
            brightness_factor (float): intensity of brightness change.
            mode (string): '2D', '3D' or 'mix'.
            p (float): probability of applying the augmentation.
        """
        super(Grayscale, self).__init__(p=p)
        self.set_mode(mode)
        self.CONTRAST_FACTOR   = contrast_factor
        self.BRIGHTNESS_FACTOR = brightness_factor

    def set_params(self):
        # No change in sample size
        pass

    def __call__(self, data, random_state):
        if random_state is None:
            random_state = np.random.RandomState(1234)

        if self.mode == 'mix':
            mode = '3D' if random_state.rand() > 0.5 else '2D'
        else:
            mode = self.mode

        # apply augmentations  
        if mode is '2D': data = self.augment2D(data, random_state)
        if mode is '3D': data = self.augment3D(data, random_state)
        return data

    def augment2D(self, data, random_state=None):
        """
        Adapted from ELEKTRONN (http://elektronn.org/).
        """
        imgs = data['image']
        transformedimgs = np.copy(imgs)
        for z in range(transformedimgs.shape[-3]):
            img = transformedimgs[z, :, :]
            img *= 1 + (np.random.rand() - 0.5)*self.CONTRAST_FACTOR
            img += (np.random.rand() - 0.5)*self.BRIGHTNESS_FACTOR
            img = np.clip(img, 0, 1)
            img **= 2.0**(np.random.rand()*2 - 1)
            transformedimgs[z, :, :] = img

        data['image'] = transformedimgs
        return data    

    def augment3D(self, data, random_state=None):
        """
        Adapted from ELEKTRONN (http://elektronn.org/).
        """
        imgs = data['image']
        transformedimgs = np.copy(imgs)
        transformedimgs *= 1 + (np.random.rand() - 0.5)*self.CONTRAST_FACTOR
        transformedimgs += (np.random.rand() - 0.5)*self.BRIGHTNESS_FACTOR
        transformedimgs = np.clip(transformedimgs, 0, 1)
        transformedimgs **= 2.0**(np.random.rand()*2 - 1)
        
        data['image'] = transformedimgs
        return data

    def invert(self, data, random_state=None):
        """
        Invert input images
        """
        imgs = data['image']
        transformedimgs = np.copy(imgs)
        transformedimgs = 1.0-transformedimgs
        transformedimgs = np.clip(transformedimgs, 0, 1)

        data['image'] = transformedimgs
        return data

    ####################################################################
    ## Setters.
    ####################################################################

    def set_mode(self, mode):
        """Set 2D/3D/mix greyscale value augmentation mode."""
        assert mode=='2D' or mode=='3D' or mode=='mix'
        self.mode = mode
