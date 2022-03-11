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

    def test_copypaste(self):
        """Test copypaste augment for numpy.ndarray and torch.Tensor.
        """
        np.random.seed(42)
        cp_augmentor = CopyPasteAugmentor()
        volume, label = np.random.randn(8,32,32), np.zeros((8,32,32))
        label[2:4, 10:20, 10:20] = 1
        volume_np = cp_augmentor({'image': volume, 'label':label})

        volume, label = torch.from_numpy(volume), torch.from_numpy(label)
        volume_torch = cp_augmentor({'image': volume, 'label':label})

        self.assertTrue(torch.allclose(volume_torch, torch.from_numpy(volume_np), atol=1e-6))

if __name__ == '__main__':
    unittest.main()
