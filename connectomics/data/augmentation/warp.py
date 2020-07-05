import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from .augmentor import DataAugment

class Elastic(DataAugment):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    The implementation is based on https://gist.github.com/erniejunior/601cdf56d2b424757de5.

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.

    Args:
        alpha (float): maximum pixel-moving distance of elastic deformation. Default: 10.0
        sigma (float): standard deviation of the Gaussian filter. Default: 4.0
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(self,
                 alpha=10.0,
                 sigma=4.0,
                 p=0.5):
        
        super(Elastic, self).__init__(p)
        self.alpha = alpha
        self.sigma = sigma
        self.image_interpolation = cv2.INTER_LINEAR
        self.label_interpolation = cv2.INTER_NEAREST
        self.border_mode = cv2.BORDER_CONSTANT
        self.set_params()

    def set_params(self):
        max_margin = int(self.alpha) + 1
        self.sample_params['add'] = [0, max_margin, max_margin]

    def __call__(self, data, random_state=np.random):
        if 'label' in data and data['label'] is not None:
            image, label = data['image'], data['label']
        else:
            image = data['image']  

        height, width = image.shape[-2:] # (c, z, y, x)

        dx = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), self.sigma) * self.alpha)
        dy = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), self.sigma) * self.alpha)

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        mapx, mapy = np.float32(x + dx), np.float32(y + dy)

        output = {}
        transformed_image = []
        transformed_label = []

        for i in range(image.shape[-3]):
            if image.ndim == 3:
                transformed_image.append(cv2.remap(image[i], mapx, mapy, 
                                    self.image_interpolation, borderMode=self.border_mode))     
            else:
                temp = [cv2.remap(image[channel, i], mapx, mapy, self.image_interpolation, 
                        borderMode=self.border_mode) for channel in range(image.shape[0])]     
                transformed_image.append(np.stack(temp, 0))          
            if 'label' in data and data['label'] is not None:
                transformed_label.append(cv2.remap(label[i], mapx, mapy, self.label_interpolation, borderMode=self.border_mode))

        if image.ndim == 3: # (z,y,x)
            transformed_image = np.stack(transformed_image, 0)
        else: # (c,z,y,x)
            transformed_image = np.stack(transformed_image, 1)

        transformed_label = np.stack(transformed_label, 0)
        output['image'] = transformed_image
        output['label'] = transformed_label

        return output
