import cv2
import math
import random
import numpy as np
from .augmentor import DataAugment

class MotionBlur(DataAugment):
    """Motion blur data augmentation of image stacks.
    
    Args:
        sections (int): number of sections along z dimension to apply motion blur. Default: 2
        kernel_size (int): kernel size for motion blur. Default: 11
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(self, sections=2, kernel_size=11, p=0.5):
        super(MotionBlur, self).__init__(p=p)
        self.size = kernel_size
        self.sections = sections
        self.set_params()

    def set_params(self):
        # No change in sample size
        pass

    def motion_blur(self, data, random_state):
        images = data['image'].copy()
        labels = data['label'].copy()

        # generating the kernel
        kernel_motion_blur = np.zeros((self.size, self.size))
        if random.random() > 0.5: # horizontal kernel
            kernel_motion_blur[int((self.size-1)/2), :] = np.ones(self.size)
        else: # vertical kernel
            kernel_motion_blur[:, int((self.size-1)/2)] = np.ones(self.size)
        kernel_motion_blur = kernel_motion_blur / self.size

        k = min(self.sections, images.shape[0])
        selected_idx = np.random.choice(images.shape[0], k, replace=True)

        for idx in selected_idx:
            # applying the kernel to the input image
            images[idx] = cv2.filter2D(images[idx], -1, kernel_motion_blur)
    
        return images, labels

    def __call__(self, data, random_state=np.random):
        new_images, new_labels = self.motion_blur(data, random_state)
        return {'image': new_images, 'label': new_labels}
