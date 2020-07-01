import numpy as np
from .augmentor import DataAugment

class CutNoise(DataAugment):
    """3D CutNoise data augmentation.

    Randomly add noise to a cuboid region in the volume to force the model
    to learn denoising when making predictions.

    Args:
        length_ratio (float): the ratio of the cuboid length compared with volume length.
        mode (string): the distribution of the noise pattern. Default: ``'uniform'``.
        scale (float): scale of the random noise. Default: 0.2.
        p (float): probability of applying the augmentation.
    """

    def __init__(self, 
                 length_ratio=0.25, 
                 mode='uniform',
                 scale=0.2,
                 p=0.5):
        super(CutNoise, self).__init__(p=p)
        self.length_ratio = length_ratio
        self.mode = mode
        self.scale = scale

    def set_params(self):
        # No change in sample size
        pass

    def cut_noise(self, data, random_state):
        images = data['image'].copy()
        labels = data['label'].copy()

        zl, zh = self.random_region(images.shape[0], random_state)
        yl, yh = self.random_region(images.shape[1], random_state)
        xl, xh = self.random_region(images.shape[2], random_state)
        
        temp = images[zl:zh, yl:yh, xl:xh].copy()
        noise = random_state.uniform(-self.scale, self.scale, temp.shape)
        temp = temp + noise
        temp = np.clip(temp, 0, 1)

        images[zl:zh, yl:yh, xl:xh] = temp
        return images, labels

    def random_region(self, vol_len, random_state):
        cuboid_len = int(self.length_ratio * vol_len)
        low = random_state.randint(0, vol_len-cuboid_len)
        high = low + cuboid_len
        return low, high

    def __call__(self, data, random_state=np.random):
        new_images, new_labels = self.cut_noise(data, random_state)
        return {'image': new_images, 'label': new_labels}