import os,sys

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch_connectomics.model.blocks import *
from torch_connectomics.model.utils import *
from torch_connectomics.libs.sync import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d


class unetv3(nn.Module):
    """Lightweight U-net with residual blocks (based on [Lee2017]_ with modifications).

    .. [Lee2017] Lee, Kisuk, Jonathan Zung, Peter Li, Viren Jain, and 
        H. Sebastian Seung. "Superhuman accuracy on the SNEMI3D connectomics 
        challenge." arXiv preprint arXiv:1706.00120, 2017.
        
    Args:
        in_channel (int): number of input channels.
        out_channel (int): number of output channels.
        filters (list): number of filters at each u-net stage.
    """
    def __init__(self, in_channel=1, out_channel=3, filters=[8, 12, 16, 20, 24]):
    #def __init__(self, in_channel=1, out_channel=3, filters=[28, 36, 48, 64, 80]):
        super().__init__()

        # encoding path
        self.layer1_E = nn.Sequential(
            conv3d_bn_elu(in_planes=in_channel, out_planes=filters[0], 
                          kernel_size=(1,5,5), stride=1, padding=(0,2,2)),
            conv3d_bn_elu(in_planes=filters[0], out_planes=filters[0], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_2d(filters[0], filters[0], projection=False)  
        )
        self.layer2_E = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[0], out_planes=filters[1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_3d(filters[1], filters[1], projection=False)
        )
        self.layer3_E = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[1], out_planes=filters[2], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_3d(filters[2], filters[2], projection=False)
        )
        self.layer4_E = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[2], out_planes=filters[3], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_3d(filters[3], filters[3], projection=False)
        )

        # center block
        self.center = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[3], out_planes=filters[4], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_3d(filters[4], filters[4], projection=True)
        )

        # decoding path
        self.layer1_D = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[0], out_planes=filters[0], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_2d(filters[0], filters[0], projection=False),
            conv3d_bn_non(in_planes=filters[0], out_planes=out_channel, 
                          kernel_size=(1,5,5), stride=1, padding=(0,2,2))
        )
        self.layer2_D = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[1], out_planes=filters[1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_3d(filters[1], filters[1], projection=False)
        )
        self.layer3_D = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[2], out_planes=filters[2], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_3d(filters[2], filters[2], projection=False)
        )
        self.layer4_D = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[3], out_planes=filters[3], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_3d(filters[3], filters[3], projection=False)
        )

        # pooling & upsample
        self.down = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)

        # conv + upsample
        self.conv1 = conv3d_bn_elu(filters[1], filters[0], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv2 = conv3d_bn_elu(filters[2], filters[1], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv3 = conv3d_bn_elu(filters[3], filters[2], kernel_size=(1,1,1), padding=(0,0,0)) 
        self.conv4 = conv3d_bn_elu(filters[4], filters[3], kernel_size=(1,1,1), padding=(0,0,0))

        #initialization
        ortho_init(self)

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

        x = torch.sigmoid(x)
        return x

def test():
    model = unetv3()
    print('model type: ', model.__class__.__name__)
    num_params = sum([p.data.nelement() for p in model.parameters()])
    print('number of trainable parameters: ', num_params)
    x = torch.randn(8, 1, 4, 128, 128)
    y = model(x)
    print(x.size(), y.size())

if __name__ == '__main__':
    test()
