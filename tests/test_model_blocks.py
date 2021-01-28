import unittest
import torch

from connectomics.model.block import *
from connectomics.model.backbone.repvgg import RepVGGBlock2D, RepVGGBlock3D

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

    def test_residual_blocks(self):
        """
        Test 2D and 3D residual blocks and squeeze-and-excitation residual blocks.
        """
        b, d, h, w = 2, 8, 32, 32
        c_in = 16 # input channels

        for c_out in [c_in, c_in*2]: # output channels
            x = torch.rand(b, c_in, h, w) # 2d residual blocks
            residual_2d = BasicBlock2d(c_in, c_out, dilation=4)
            residual_se_2d = BasicBlock2dSE(c_in, c_out, dilation=4)

            out = residual_2d(x)
            out_shape = tuple(out.shape)
            self.assertTupleEqual(out_shape, (b, c_out, h, w))
            out = residual_se_2d(x)
            out_shape = tuple(out.shape)
            self.assertTupleEqual(out_shape, (b, c_out, h, w))

            x = torch.rand(b, c_in, d, h, w) # 3d residual blocks
            residual_3d = BasicBlock3d(c_in, c_out, dilation=4, isotropic=False)
            residual_se_3d = BasicBlock3dSE(c_in, c_out, dilation=4, isotropic=False)

            out = residual_3d(x)
            out_shape = tuple(out.shape)
            self.assertTupleEqual(out_shape, (b, c_out, d, h, w))
            out = residual_se_3d(x)
            out_shape = tuple(out.shape)
            self.assertTupleEqual(out_shape, (b, c_out, d, h, w))

    def test_repvgg_block_2d(self):
        """Test 2D RepVGG blocks.
        """
        b, h, w = 2, 32, 32
        c_in = 8

        for c_out in [8, 16]:
            for dilation in [1, 4]:
                train_block = RepVGGBlock2D(c_in, c_out, dilation=dilation, deploy=False)
                train_block.eval()
                kernel, bias = train_block.repvgg_convert()
                deploy_block = RepVGGBlock2D(c_in, c_out, dilation=dilation, deploy=True)
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

if __name__ == '__main__':
    unittest.main()
