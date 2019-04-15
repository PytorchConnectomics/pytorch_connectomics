from __future__ import division

import random
import warnings
import numpy as np

class Compose(object):
    """Compose transforms

    Args:
        transforms (list): list of transformations to compose.
        input_size (tuple): input size of model in (z, y, x).
        keep_uncropped (bool): keep uncropped images and labels (default: False)
    """
    def __init__(self, 
                 transforms, 
                 input_size = (8,196,196),
                 keep_uncropped = False):
        self.transforms = transforms
        self.input_size = np.array(input_size)
        self.sample_size = self.input_size.copy()
        self.set_sample_params()
        self.keep_uncropped = keep_uncropped

    def set_sample_params(self):
        for _, t in enumerate(self.transforms):
            self.sample_size = np.ceil(self.sample_size * t.sample_params['ratio']).astype(int)
            self.sample_size = self.sample_size + (2 * np.array(t.sample_params['add']))
        print('Sample size required for the augmentor:', self.sample_size)

    def crop(self, data):
        image, label = data['image'], data['label']

        assert image.shape[-3:] == label.shape
        assert image.ndim == 3 or image.ndim == 4
        margin = (label.shape[1] - self.input_size[1]) // 2
        if margin==0:
            return {'image': image, 'mask': label}
        else:    
            low = margin
            high = margin + self.input_size[1]
            if image.ndim == 3:
                if self.keep_uncropped == True:
                    return {'image': image[:, low:high, low:high],
                            'label': label[:, low:high, low:high],
                            'image_uncropped': image,
                            'label_uncropped': label}               
                else:
                    return {'image': image[:, low:high, low:high],
                            'label': label[:, low:high, low:high]}
            else:
                if self.keep_uncropped == True:
                    return {'image': image[:, :, low:high, low:high],
                            'label': label[:, low:high, low:high],
                            'image_uncropped': image,
                            'label_uncropped': label}
                else:
                    return {'image': image[:, :, low:high, low:high],
                            'label': label[:, low:high, low:high]}                                        

    def __call__(self, data, random_state=None):
        data['image'] = data['image'].astype(np.float32)
        for _, t in enumerate(self.transforms):
            if random.random() < t.p:
                data = t(data, random_state)

        # crop the data to input size
        data = self.crop(data)
        return data