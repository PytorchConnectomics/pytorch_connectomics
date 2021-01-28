import unittest
import torch

from connectomics.model import build_model
from connectomics.model.unet import UNet3D
from connectomics.model.fpn import FPN3D
from connectomics.model.backbone import RepVGG3D
from connectomics.model.utils.misc import IntermediateLayerGetter

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

    def test_rep_vgg_3d(self):
        feat_keys=['y0', 'y1', 'y2', 'y3', 'y4']
        return_layers = {'layer0': feat_keys[0],
                        'layer1': feat_keys[1], 
                        'layer2': feat_keys[2],
                        'layer3': feat_keys[3],
                        'layer4': feat_keys[4]}   

        model_train = RepVGG3D()
        converted_weights = model_train.repvgg_convert_model()
        model_deploy = RepVGG3D(deploy=True)
        model_deploy.load_reparam_model(converted_weights)

        model_train = IntermediateLayerGetter(model_train, return_layers).eval()
        model_deploy = IntermediateLayerGetter(model_deploy, return_layers).eval()

        x = torch.rand(2, 1, 9, 65, 65)
        z1, z2 = model_train(x), model_deploy(x)
        for key in feat_keys:
            self.assertTrue(torch.allclose(z1[key], z2[key], atol=1e-6))

    def test_build_model(self):
        """Test building model from configs.
        """
        cfg = get_cfg_defaults()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg, device)
        self.assertTrue(isinstance(model, (torch.nn.Module, 
                                           torch.nn.DataParallel, 
                                           torch.nn.parallel.DistributedDataParallel)))

if __name__ == '__main__':
    unittest.main()
