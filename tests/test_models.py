import unittest

import torch
from collections import OrderedDict

from connectomics.model import build_model
from connectomics.model.arch import UNet3D, UNet2D, FPN3D, UNetPlus3D
from connectomics.model.backbone import RepVGG3D, RepVGGBlock3D
from connectomics.model.utils.misc import IntermediateLayerGetter

from connectomics.config import get_cfg_defaults


class TestModelBlock(unittest.TestCase):

    def test_unet_3d(self):
        """Tested UNet3D model with odd and even input sizes.
        """
        for model_class in [UNet3D, UNetPlus3D]:
            b, d, h, w = 4, 8, 64, 64
            in_channel, out_channel = 1, 3
            x = torch.rand(b, in_channel, d, h, w)
            model = model_class(block_type='residual', in_channel=in_channel,
                                out_channel=out_channel, pooling=True)
            out = model(x)
            self.assertTupleEqual(tuple(out.shape), (b, out_channel, d, h, w))

            b, d, h, w = 4, 9, 65, 65
            in_channel, out_channel = 1, 2
            x = torch.rand(b, in_channel, d, h, w)
            model = model_class(block_type='residual_se', in_channel=in_channel,
                                out_channel=out_channel, pooling=False)
            out = model(x)
            self.assertTupleEqual(tuple(out.shape), (b, out_channel, d, h, w))

            b, d, h, w = 1, 65, 65, 65
            in_channel, out_channel = 1, 2
            x = torch.rand(b, in_channel, d, h, w)
            model = model_class(block_type='residual_se', in_channel=in_channel,
                                out_channel=out_channel, pooling=False, is_isotropic=True)
            out = model(x)
            self.assertTupleEqual(tuple(out.shape), (b, out_channel, d, h, w))

    def test_unet_2d(self):
        """Tested UNet2D model with odd and even input sizes.
        """
        b, h, w = 4, 64, 64
        in_channel, out_channel = 1, 3
        x = torch.rand(b, in_channel, h, w)
        model = UNet2D('residual', in_channel, out_channel, pooling=True)
        out = model(x)
        self.assertTupleEqual(tuple(out.shape), (b, out_channel, h, w))

        b, h, w = 4, 65, 65
        in_channel, out_channel = 1, 2
        x = torch.rand(b, in_channel, h, w)
        model = UNet2D('residual_se', in_channel, out_channel, pooling=False)
        out = model(x)
        self.assertTupleEqual(tuple(out.shape), (b, out_channel, h, w))

    def test_fpn_3d(self):
        b, d, h, w = 1, 65, 65, 65
        in_channel, out_channel = 1, 2
        x = torch.rand(b, in_channel, d, h, w)
        model = FPN3D('resnet', 'residual', in_channel=in_channel,
                      out_channel=out_channel)
        out = model(x)
        self.assertTupleEqual(tuple(out.shape), (b, out_channel, d, h, w))

    def test_rep_vgg_3d(self):
        r"""Test the 3D RepVGG model. Making sure the outputs of model in train and deploy
        modes are the same.
        """
        feat_keys = ['y0', 'y1', 'y2', 'y3', 'y4']
        return_layers = {'layer0': feat_keys[0],
                         'layer1': feat_keys[1],
                         'layer2': feat_keys[2],
                         'layer3': feat_keys[3],
                         'layer4': feat_keys[4]}

        model_train = RepVGG3D()
        converted_weights = model_train.repvgg_convert_model()
        model_deploy = RepVGG3D(deploy=True)
        model_deploy.load_reparam_model(converted_weights)

        model_train = IntermediateLayerGetter(
            model_train, return_layers).eval()
        model_deploy = IntermediateLayerGetter(
            model_deploy, return_layers).eval()

        x = torch.rand(2, 1, 9, 65, 65)
        z1, z2 = model_train(x), model_deploy(x)
        for key in feat_keys:
            # The default eps value added to the denominator for numerical stability
            # in PyTorch batchnormalization layer is 1e-5.
            self.assertTrue(torch.allclose(z1[key], z2[key], atol=1e-4))

    def test_build_default_model(self):
        r"""Test building model from configs.
        """
        cfg = get_cfg_defaults()
        cfg.freeze()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg, device)
        self.assertTrue(isinstance(model, (torch.nn.Module,
                                           torch.nn.DataParallel,
                                           torch.nn.parallel.DistributedDataParallel)))

    def test_build_fpn_with_repvgg(self):
        r"""Test building a 3D FPN model with RepVGG backbone from configs.
        """
        cfg = get_cfg_defaults()
        cfg.MODEL.ARCHITECTURE = 'fpn_3d'
        cfg.MODEL.BACKBONE = 'repvgg'
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg, device).eval()
        message = "Get unexpected model architecture!"

        arch_name = model.module.__class__.__name__
        self.assertEqual(arch_name, "FPN3D", message)

        message = "No RepVGG block in the backbone!"
        count = 0
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock3D):
                count += 1
        self.assertGreater(count, 0)

        # test the weight conversion when using RepVGG as backbone
        model.eval()
        train_dict = model.module.state_dict()
        deploy_dict = RepVGG3D.repvgg_convert_as_backbone(train_dict)

        cfg.MODEL.DEPLOY_MODE = True
        deploy_model = build_model(cfg, device).eval()
        deploy_model.module.load_state_dict(deploy_dict, strict=True)

        x = torch.rand(2, 1, 9, 65, 65)
        y1 = model(x)
        y2 = deploy_model(x)
        self.assertTrue(torch.allclose(y1, y2, atol=1e-4))

    def test_build_fpn_with_botnet(self):
        r"""Test building a 3D FPN model with BotNet3D backbone from configs.
        """
        cfg = get_cfg_defaults()
        cfg.MODEL.ARCHITECTURE = 'fpn_3d'
        cfg.MODEL.BACKBONE = 'botnet'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg, device).eval()

        d, h, w = cfg.MODEL.INPUT_SIZE
        x = torch.rand(2, 1, d, h, w)
        y1 = model(x)
        self.assertTupleEqual(tuple(y1.shape), (2, 1, d, h, w))

    def test_build_fpn_with_efficientnet(self):
        r"""Test building a 3D FPN model with EfficientNet3D backbone from configs.
        """
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        cfg = get_cfg_defaults()
        cfg.MODEL.ARCHITECTURE = 'fpn_3d'
        cfg.MODEL.BACKBONE = 'efficientnet'

        d, h, w = 9, 65, 65
        x = torch.rand(2, 1, d, h, w).to(device)

        # inverted residual blocks
        cfg.MODEL.BLOCK_TYPE = 'inverted_res'
        model = build_model(cfg, device).eval()
        y1 = model(x)
        self.assertTupleEqual(tuple(y1.shape), (2, 1, d, h, w))

        # inverted residual blocks with dilation
        cfg.MODEL.BLOCK_TYPE = 'inverted_res_dilated'
        model = build_model(cfg, device).eval()
        y1 = model(x)
        self.assertTupleEqual(tuple(y1.shape), (2, 1, d, h, w))

    def test_build_deeplab_with_resnet(self):
        r"""Test building 2D deeplabv3 model with resnet backbone.
        """
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        cfg = get_cfg_defaults()
        cfg.MODEL.BACKBONE = 'resnet101'
        cfg.MODEL.AUX_OUT = True

        c_i = cfg.MODEL.IN_PLANES
        c_o = cfg.MODEL.OUT_PLANES
        b, h, w = 2, 65, 65
        x = torch.rand(b, c_i, h, w).to(device)

        for arch in ['deeplabv3a', 'deeplabv3b', 'deeplabv3c']:
            cfg.MODEL.ARCHITECTURE = arch
            model = build_model(cfg, device).eval()
            y = model(x)
            self.assertTrue(isinstance(y, OrderedDict))
            self.assertTrue("aux" in y.keys())
            for key in y.keys():
                self.assertTupleEqual(
                    tuple(y[key].shape),
                    (b, c_o, h, w))


if __name__ == '__main__':
    unittest.main()
