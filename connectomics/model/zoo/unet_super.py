import os,sys
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from ..block import *
from ..utils import *
from .unet_residual import unet_residual_3d

class Unet_super(unet_residual_3d):
    
    def __init__(self, in_channel=1, out_channel=3, filters=[1, 64, 1, 28, 36, 48, 64, 80], pad_mode='rep',\
                 norm_mode='bn', act_mode='elu', do_embedding=True, head_depth=1):
        super().__init__(in_channel=in_channel, out_channel=out_channel,\
                                              filters=filters[3:], pad_mode=pad_mode, norm_mode=norm_mode,\
                                              act_mode=act_mode, do_embedding=do_embedding,\
                                              head_depth=head_depth)
        # Upsampling specific
        self.deconv1 = nn.ConvTranspose3d(filters[0],filters[1],5,stride=(2,3,3))
        self.conv1 = nn.Conv3d(filters[1], filters[2], kernel_size=(3,3,3))

    def forward(self,x):
        x = F.relu(self.deconv1(x))# Use upscale bilinear
        # x [nsamples, nchannels, z, y, x]
        x = x[:,:,:x.size()[2]-1,:,:]
        x = self.conv1(x)
        x = torch.sigmoid(x)
        return super().forward(x)


def test():
    model = Unet_super()
    print('model type: ', model.__class__.__name__)
    num_params = sum([p.data.nelement() for p in model.parameters()])
    print('number of trainable parameters: ', num_params)
    x = torch.randn(1, 1, 18, 64, 64)
    y = model(x)
    print(x.size(), y.size())

if __name__ == '__main__':
    test()
    
