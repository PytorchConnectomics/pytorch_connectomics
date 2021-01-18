import unittest
import torch

from connectomics.model.block import *

class TestModelBlock(unittest.TestCase):

    def test_basic_blocks(self):
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
            residual_2d = residual_block_2d(c_in, c_out, dilation=4)
            residual_se_2d = residual_se_block_2d(c_in, c_out, dilation=4)

            out = residual_2d(x)
            out_shape = tuple(out.shape)
            self.assertTupleEqual(out_shape, (b, c_out, h, w))
            out = residual_se_2d(x)
            out_shape = tuple(out.shape)
            self.assertTupleEqual(out_shape, (b, c_out, h, w))

            x = torch.rand(b, c_in, d, h, w) # 3d residual blocks
            residual_3d = residual_block_3d(c_in, c_out, dilation=4, isotropy=False)
            residual_se_3d = residual_se_block_3d(c_in, c_out, dilation=4, isotropy=False)

            out = residual_3d(x)
            out_shape = tuple(out.shape)
            self.assertTupleEqual(out_shape, (b, c_out, d, h, w))
            out = residual_se_3d(x)
            out_shape = tuple(out.shape)
            self.assertTupleEqual(out_shape, (b, c_out, d, h, w))

if __name__ == '__main__':
    unittest.main()
