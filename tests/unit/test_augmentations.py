import unittest
import torch
import numpy as np

from connectomics.data.augment.monai_transforms import RandMixupd, RandCopyPasted

class TestModelBlock(unittest.TestCase):

    def test_mixup(self):
        """Test mixup for numpy.ndarray and torch.Tensor."""
        mixup_augmentor = RandMixupd(keys=['image'], prob=1.0, alpha_range=(0.3, 0.7))

        # Test with numpy
        volume_np = np.ones((4, 1, 8, 32, 32))
        data = {'image': volume_np}
        result = mixup_augmentor(data)
        self.assertIsInstance(result['image'], np.ndarray)

        # Test with torch
        volume_torch = torch.ones(4, 1, 8, 32, 32)
        data = {'image': volume_torch}
        result = mixup_augmentor(data)
        self.assertIsInstance(result['image'], torch.Tensor)

    def test_copypaste(self):
        """Test copypaste augment for numpy.ndarray and torch.Tensor."""
        np.random.seed(42)
        cp_augmentor = RandCopyPasted(keys=['image'], label_key='label', prob=1.0, max_obj_ratio=0.5)

        # Test with numpy
        volume_np = np.random.randn(8, 32, 32).astype(np.float32)
        label_np = np.zeros((8, 32, 32), dtype=np.uint8)
        label_np[2:4, 10:20, 10:20] = 1
        label_np[5:7, 15:25, 15:25] = 2
        data_np = cp_augmentor({'image': volume_np, 'label': label_np})
        self.assertIsInstance(data_np['image'], np.ndarray)
        self.assertIsInstance(data_np['label'], np.ndarray)

        # Test with torch
        volume_torch = torch.from_numpy(volume_np)
        label_torch = torch.from_numpy(label_np)
        data_torch = cp_augmentor({'image': volume_torch, 'label': label_torch})
        self.assertIsInstance(data_torch['image'], torch.Tensor)
        self.assertIsInstance(data_torch['label'], torch.Tensor)

if __name__ == '__main__':
    unittest.main()
