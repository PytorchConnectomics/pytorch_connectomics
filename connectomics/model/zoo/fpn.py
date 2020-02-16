import os,sys
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from ..block import *
from ..utils import *

class fpn(nn.Module):
    def __init__(self, in_channel=1, out_channel=3, filters=[32,64,128,256,256]):
        super().__init__()
        
        # layers
        self.layer1 = nn.Sequential(
            residual_block_2d(in_channel, filters[0], projection=True),
            residual_block_2d(filters[0], filters[0], projection=False),
            residual_block_2d(filters[0], filters[0], projection=True),
            squeeze_excitation_3d(channel=filters[0], channel_reduction=2, spatial_reduction=16)
        )
        self.layer2 = nn.Sequential(
            residual_block_2d(filters[0], filters[1], projection=True),
            residual_block_2d(filters[1], filters[1], projection=False),
            residual_block_2d(filters[1], filters[1], projection=True),
            squeeze_excitation_3d(channel=filters[1], channel_reduction=4, spatial_reduction=8)
        )
        self.layer3 = nn.Sequential(
            residual_block_3d(filters[1], filters[2], projection=True),
            residual_block_3d(filters[2], filters[2], projection=False),
            residual_block_3d(filters[2], filters[2], projection=True),
            squeeze_excitation_3d(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )
        self.layer4 = nn.Sequential(
            bottleneck_dilated_3d(filters[2], filters[3], projection=True),
            bottleneck_dilated_3d(filters[3], filters[3], projection=False),
            bottleneck_dilated_3d(filters[3], filters[3], projection=True),
            squeeze_excitation_3d(channel=filters[3], channel_reduction=16, spatial_reduction=2, z_reduction=2)
        )
        self.layer5 = nn.Sequential(
            bottleneck_dilated_3d(filters[3], filters[4], projection=True),
            bottleneck_dilated_3d(filters[4], filters[4], projection=False),
            bottleneck_dilated_3d(filters[4], filters[4], projection=True),
            squeeze_excitation_3d(channel=filters[3], channel_reduction=16, spatial_reduction=2, z_reduction=2)
        )

        # pooling & upsample
        self.down = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)

        # conv + upsample
        self.conv2 = conv3d_bn_elu(filters[1], filters[0], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv3 = conv3d_bn_elu(filters[2], filters[0], kernel_size=(1,1,1), padding=(0,0,0)) 
        self.conv4 = conv3d_bn_elu(filters[3], filters[0], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv5 = conv3d_bn_elu(filters[3], filters[0], kernel_size=(1,1,1), padding=(0,0,0))

        # convert to probability
        self.fconv = conv3d_bn_non(filters[0], out_channel, kernel_size=(3,3,3), padding=(1,1,1))

    def forward(self, x):

        z1 = self.layer1(x)
        x = self.down(z1)
        z2 = self.layer2(x)
        x = self.down(z2)
        z3 = self.layer3(x)
        x = self.down(z3)
        z4 = self.layer4(x)
        x = self.down(z4)
        z5 = self.layer5(x)

        y5 = self.conv5(z5)
        y4 = self.conv4(z4)
        y3 = self.conv3(z3)
        y2 = self.conv2(z2)
        y1 = z1

        out = self.up(self.up(self.up(self.up(y5)+y4)+y3)+y2)+y1
        #out = self.up(self.up(y5+y4+y3)+y2)+y1
        out = self.fconv(out)
        out = torch.sigmoid(out)
        return out

def test():
    model = fpn()
    print('model type: ', model.__class__.__name__)
    num_params = sum([p.data.nelement() for p in model.parameters()])
    print('number of trainable parameters: ', num_params)
    x = torch.randn(8, 1, 4, 64, 64)
    y = model(x)
    print(x.size(), y.size())

if __name__ == '__main__':
    test()
