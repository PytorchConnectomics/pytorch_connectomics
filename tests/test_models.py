import unittest
import torch

from connectomics.model import build_model
from connectomics.model.unet import UNet3D
from connectomics.model.fpn import FPN3D

from connectomics.config import get_cfg_defaults

class TestModelBlock(unittest.TestCase):

    def test_unet_3d(self):
        """Tested UNet3D model with odd and even input sizes.
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
        model = UNet3D('residual_se', in_channel, out_channel, pooling=False)
        out = model(x)
        self.assertTupleEqual(tuple(out.shape), (b, out_channel, d, h, w))

        b, d, h, w = 1, 65, 65, 65
        in_channel, out_channel = 1, 2
        x = torch.rand(b, in_channel, d, h, w)
        model = UNet3D('residual_se', in_channel, out_channel, 
                       pooling=False, is_isotropic=True)
        out = model(x)
        self.assertTupleEqual(tuple(out.shape), (b, out_channel, d, h, w))

    def test_fpn_3d(self):
        b, d, h, w = 1, 65, 65, 65
        in_channel, out_channel = 1, 2
        x = torch.rand(b, in_channel, d, h, w)
        model = FPN3D('resnet', 'residual', in_channel=in_channel, out_channel=out_channel)
        out = model(x)
        self.assertTupleEqual(tuple(out.shape), (b, out_channel, d, h, w))

    def test_build_model(self):
        """Test building model from configs.
        """
        cfg = get_cfg_defaults()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _ = build_model(cfg, device)

if __name__ == '__main__':
    unittest.main()
