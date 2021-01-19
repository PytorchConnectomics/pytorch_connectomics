import unittest
import torch

from connectomics.model.unet import UNet3D

class TestModelBlock(unittest.TestCase):

    def test_unet_3d(self):
        """Test UNet 3D model.
        """
        b, d, h, w = 4, 8, 64, 64
        in_channel, out_channel = 1, 3
        x = torch.rand(b, in_channel, d, h, w)
        model = UNet3D('residual', in_channel, out_channel, pooling=True)
        out = model(x)
        self.assertTupleEqual(tuple(out.shape), (b, out_channel, d, h, w))

        b, d, h, w = 4, 9, 65, 65
        in_channel, out_channel = 1, 2
        x = torch.rand(b, in_channel, d, h, w)
        model = UNet3D('residual', in_channel, out_channel, pooling=False)
        out = model(x)
        self.assertTupleEqual(tuple(out.shape), (b, out_channel, d, h, w))

if __name__ == '__main__':
    unittest.main()
