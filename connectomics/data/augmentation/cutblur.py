import numpy as np
from .augmentor import DataAugment
from skimage.transform import resize

class CutBlur(DataAugment):
    """3D CutBlur data augmentation, adapted from https://arxiv.org/abs/2004.00448.

    Randomly downsample a cuboid region in the volume to force the model
    to learn super-resolution when making predictions.

    Args:
        length_ratio (float): the ratio of the cuboid length compared with volume length.
        down_ratio_min (float): minimal downsample ratio to generate low-res region.
        down_ratio_max (float): maximal downsample ratio to generate low-res region.
        downsample_z (bool): downsample along the z axis (default: False).
        p (float): probability of applying the augmentation.
    """

    def __init__(self, 
                 length_ratio=0.25, 
                 down_ratio_min=2.0,
                 down_ratio_max=8.0,
                 downsample_z=False,
                 p=0.5):
        super(CutBlur, self).__init__(p=p)
        self.length_ratio = length_ratio
        self.down_ratio_min = down_ratio_min
        self.down_ratio_max = down_ratio_max
        self.downsample_z = downsample_z

    def set_params(self):
        # No change in sample size
        pass

    def cut_blur(self, data, random_state):
        images = data['image'].copy()
        labels = data['label'].copy()

        zdim = images.shape[0]
        
        if zdim > 1:
            zl, zh = self.random_region(images.shape[0], random_state)
        yl, yh = self.random_region(images.shape[1], random_state)
        xl, xh = self.random_region(images.shape[2], random_state)
        
        if zdim == 1:
            temp = images[:, yl:yh, xl:xh].copy()
        else:
            temp = images[zl:zh, yl:yh, xl:xh].copy()

        down_ratio = random_state.uniform(self.down_ratio_min, self.down_ratio_max)
        if zdim > 1 and self.downsample_z:
            out_shape = np.array(temp.shape) /  down_ratio
        else:
            out_shape = np.array(temp.shape) /  np.array([1, down_ratio, down_ratio])

        out_shape = out_shape.astype(int)
        downsampled = resize(temp, out_shape, order=1, mode='reflect', 
                             clip=True, preserve_range=True, anti_aliasing=True)
        upsampled = resize(downsampled, temp.shape, order=0, mode='reflect', 
                             clip=True, preserve_range=True, anti_aliasing=False)

        if zdim == 1:
            images[:, yl:yh, xl:xh] = upsampled
        else:
            images[zl:zh, yl:yh, xl:xh] = upsampled
        return images, labels


    def random_region(self, vol_len, random_state):
        cuboid_len = int(self.length_ratio * vol_len)
        low = random_state.randint(0, vol_len-cuboid_len)
        high = low + cuboid_len
        return low, high

    def __call__(self, data, random_state=np.random):
        new_images, new_labels = self.cut_blur(data, random_state)
        return {'image': new_images, 'label': new_labels}
