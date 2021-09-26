import unittest

import torch

from connectomics.model.block import *
from connectomics.model.backbone.repvgg import RepVGGBlock2D, RepVGGBlock3D
from connectomics.model.backbone.botnet import BottleBlock


class TestModelBlock(unittest.TestCase):

    def test_basic_blocks(self):
        """Test basic blocks.
        """
        b, d, h, w = 2, 8, 32, 32
        c_in, c_out = 16, 32

        # test 2d basic block
        block_2d = conv2d_norm_act(c_in, c_out)
        x = torch.rand(b, c_in, h, w)
        out = block_2d(x)
        out_shape = tuple(out.shape)
        self.assertTupleEqual(out_shape, (b, c_out, h, w))

        # test 3d basic block
        block_3d = conv3d_norm_act(c_in, c_out)
        x = torch.rand(b, c_in, d, h, w)
        out = block_3d(x)
        out_shape = tuple(out.shape)
        self.assertTupleEqual(out_shape, (b, c_out, d, h, w))

    def _test_residual(self, x, block, target_shape, **kwargs):
        model = block(**kwargs)
        out = model(x)
        out_shape = tuple(out.shape)
        self.assertTupleEqual(out_shape, target_shape)

    def test_residual_blocks(self):
        """
        Test 2D and 3D residual blocks and squeeze-and-excitation residual blocks.
        """
        b, d, h, w = 2, 9, 33, 33
        c_in = 16  # input channels

        for c_out in [c_in, c_in*2]:  # output channels
            x = torch.rand(b, c_in, h, w)  # 2d residual blocks
            target_shape = (b, c_out, h, w)
            for block in [BasicBlock2d, BasicBlock2dSE]:
                self._test_residual(
                    x, block, target_shape, in_planes=c_in,
                    planes=c_out, dilation=4)

            x = torch.rand(b, c_in, d, h, w)  # 3d residual blocks
            target_shape = (b, c_out, d, h, w)
            for block in [BasicBlock3d, BasicBlock3dSE,
                          BasicBlock3dPA, BasicBlock3dPASE]:
                self._test_residual(
                    x, block, target_shape, in_planes=c_in,
                    planes=c_out, dilation=4, isotropic=False)

    def _test_non_local(self, block, x):
        out = block(x)
        self.assertTupleEqual(tuple(out.shape), tuple(x.shape))

    def test_non_local_blocks(self):
        """Test 1D, 2D and 3D non-local blocks.
        """
        b, c, d, h, w = 2, 32, 17, 33, 33

        self. _test_non_local(
            NonLocalBlock1D(c), torch.rand(b, c, w))
        self. _test_non_local(
            NonLocalBlock2D(c), torch.rand(b, c, h, w))
        self. _test_non_local(
            NonLocalBlock3D(c), torch.rand(b, c, d, h, w))

    def test_repvgg_block_2d(self):
        """Test 2D RepVGG blocks.
        """
        b, h, w = 2, 32, 32
        c_in = 8

        for c_out in [8, 16]:
            for dilation in [1, 4]:
                train_block = RepVGGBlock2D(
                    c_in, c_out, dilation=dilation, deploy=False)
                train_block.eval()
                kernel, bias = train_block.repvgg_convert()
                deploy_block = RepVGGBlock2D(
                    c_in, c_out, dilation=dilation, deploy=True)
                deploy_block.load_reparam_kernel(kernel, bias)
                deploy_block.eval()

                x = torch.rand(b, c_in, h, w)
                out1 = train_block(x)
                out2 = deploy_block(x)
                self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

    def test_repvgg_block_3d(self):
        """Test 3D RepVGG blocks.
        """
        b, d, h, w = 2, 8, 32, 32
        c_in = 8

        for c_out in [8, 16]:
            for isotropic in [True, False]:
                for dilation in [1, 4]:
                    train_block = RepVGGBlock3D(c_in, c_out, dilation=dilation,
                                                isotropic=isotropic, deploy=False)
                    train_block.eval()
                    kernel, bias = train_block.repvgg_convert()
                    deploy_block = RepVGGBlock3D(c_in, c_out, dilation=dilation,
                                                 isotropic=isotropic, deploy=True)
                    deploy_block.load_reparam_kernel(kernel, bias)
                    deploy_block.eval()

                    x = torch.rand(b, c_in, d, h, w)
                    out1 = train_block(x)
                    out2 = deploy_block(x)
                    self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

    def test_bottleneck_attention_block(self):
        # AbsPosEmb
        block3d = BottleBlock(dim=16, fmap_size=(
            8, 8, 8), dim_out=16, proj_factor=4, downsample=False, dim_head=16, )
        tensor = torch.randn(2, 16, 8, 8, 8)
        self.assertTupleEqual(tuple(block3d(tensor).shape), (2, 16, 8, 8, 8))

        # RelPosEmb
        block3d = BottleBlock(dim=16, fmap_size=(8, 16, 18), dim_out=16,
                              proj_factor=4, downsample=False, dim_head=16, rel_pos_emb=True)
        tensor = torch.randn(2, 16, 8, 16, 18)
        self.assertTupleEqual(tuple(block3d(tensor).shape), (2, 16, 8, 16, 18))

    def test_blurpool_block(self):

        # Test 3-D Blurpool (isotropic)
        blurpool = BlurPool3D(channels=10, stride=4)
        # 10 channels, batch_size = 1
        input_tensor = torch.Tensor(1, 10, 128, 128, 128)
        output_tensor = blurpool(input_tensor)
        self.assertTupleEqual(tuple(output_tensor.shape), (1, 10, 32, 32, 32))

        # Test 3-D Blurpool (anisotropic)
        blurpool = BlurPool3D(channels=10, stride=(1, 2, 2))
        # 10 channels, batch_size = 1
        input_tensor = torch.Tensor(1, 10, 32, 64, 64)
        output_tensor = blurpool(input_tensor)
        self.assertTupleEqual(tuple(output_tensor.shape), (1, 10, 32, 32, 32))

        # Test 2-D blurpool
        blurpool = BlurPool2D(channels=10, stride=2)
        # 10 channels, batch_size = 1
        input_tensor = torch.Tensor(1, 10, 64, 64)
        output_tensor = blurpool(input_tensor)
        self.assertTupleEqual(tuple(output_tensor.shape), (1, 10, 32, 32))

        # Test 1-D blurpool
        length, stride, c = 16, 2, 4
        blurpool_1d = BlurPool1D(channels=c, stride=stride)
        # Input needs to be of form (batch_size, channels, dim)
        input_tensor = torch.Tensor(1, c, length)
        output_tensor = blurpool_1d(input_tensor)
        self.assertTupleEqual(tuple(output_tensor.shape),
                              (1, c, length // 2))


if __name__ == '__main__':
    unittest.main()
