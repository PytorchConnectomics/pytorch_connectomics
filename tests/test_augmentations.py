import unittest
import torch
import numpy as np

from connectomics.data.augmentation import *

class TestModelBlock(unittest.TestCase):

    def test_mixup(self):
        """Test mixup for numpy.ndarray and torch.Tensor.
        """
        mixup_augmentor = MixupAugmentor(num_aug=2)
        volume = np.ones((4,1,8,32,32))
        volume = mixup_augmentor(volume)

        volume = torch.ones(4,1,8,32,32)
        volume = mixup_augmentor(volume) 

if __name__ == '__main__':
    unittest.main()
