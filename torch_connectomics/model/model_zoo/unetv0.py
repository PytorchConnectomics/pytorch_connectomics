import os,sys

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch_connectomics.model.blocks import *
from torch_connectomics.libs.sync import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

class unetv0(nn.Module):
    """U-net with residual blocks.

    Args:
        in_channel (int): number of input channels.
        out_channel (int): number of output channels.
        filters (list): number of filters at each u-net stage.
    """
    def __init__(self, in_channel=1, out_channel=3, filters=[32,64,128,256,256], act = 'sigmoid'):
        super().__init__()

        # encoding path
        self.layer1_E = nn.Sequential(
            residual_block_2d(in_channel, filters[0], projection=True),
            squeeze_excitation_3d(channel=filters[0], channel_reduction=2, spatial_reduction=16)
        )
        self.layer2_E = nn.Sequential(
            residual_block_2d(filters[0], filters[1], projection=True),
            squeeze_excitation_3d(channel=filters[1], channel_reduction=4, spatial_reduction=8)
        )
        self.layer3_E = nn.Sequential(
            residual_block_3d(filters[1], filters[2], projection=True),
            squeeze_excitation_3d(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )
        self.layer4_E = nn.Sequential(
            bottleneck_dilated_3d(filters[2], filters[3], projection=True),
            squeeze_excitation_3d(channel=filters[3], channel_reduction=16, spatial_reduction=2, z_reduction=2)
        )

        # center block
        self.center = nn.Sequential(
            bottleneck_dilated_3d(filters[3], filters[4], projection=True),
            squeeze_excitation_3d(channel=filters[4], channel_reduction=16, spatial_reduction=2, z_reduction=2)
        )

        # decoding path
        self.layer1_D = nn.Sequential(
            residual_block_2d(filters[0], filters[0], projection=True),
            squeeze_excitation_3d(channel=filters[0], channel_reduction=2, spatial_reduction=16)
        )
        self.layer2_D = nn.Sequential(
            residual_block_2d(filters[1], filters[1], projection=True),
            squeeze_excitation_3d(channel=filters[1], channel_reduction=4, spatial_reduction=8)
        )
        self.layer3_D = nn.Sequential(
            residual_block_3d(filters[2], filters[2], projection=True),
            squeeze_excitation_3d(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )
        self.layer4_D = nn.Sequential(
            bottleneck_dilated_3d(filters[3], filters[3], projection=True),
            squeeze_excitation_3d(channel=filters[3], channel_reduction=16, spatial_reduction=2, z_reduction=2)
        )

        # pooling & upsample
        self.down = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)

        # conv + upsample
        self.conv1 = conv3d_bn_elu(filters[1], filters[0], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv2 = conv3d_bn_elu(filters[2], filters[1], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv3 = conv3d_bn_elu(filters[3], filters[2], kernel_size=(1,1,1), padding=(0,0,0)) 
        self.conv4 = conv3d_bn_elu(filters[4], filters[3], kernel_size=(1,1,1), padding=(0,0,0))

        # convert to probability
        self.fconv = conv3d_bn_non(filters[0], out_channel, kernel_size=(3,3,3), padding=(1,1,1))

        #final layer activation
        if act == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Sigmoid()

    def forward(self, x):

        # encoding path
        z1 = self.layer1_E(x)
        x = self.down(z1)
        z2 = self.layer2_E(x)
        x = self.down(z2)
        z3 = self.layer3_E(x)
        x = self.down(z3)
        z4 = self.layer4_E(x)
        x = self.down(z4)

        x = self.center(x)

        # decoding path
        x = self.up(self.conv4(x))
        x = x + z4
        x = self.layer4_D(x)

        x = self.up(self.conv3(x))
        x = x + z3
        x = self.layer3_D(x)

        x = self.up(self.conv2(x))
        x = x + z2
        x = self.layer2_D(x)

        x = self.up(self.conv1(x))
        x = x + z1
        x = self.layer1_D(x)

        x = self.fconv(x)
        x = self.act(x)
        return x

def test():
    model = unetv0()
    print('model type: ', model.__class__.__name__)
    num_params = sum([p.data.nelement() for p in model.parameters()])
    print('number of trainable parameters: ', num_params)
    x = torch.randn(8, 1, 4, 64, 64)
    y = model(x)
    print(x.size(), y.size())

if __name__ == '__main__':
    test()