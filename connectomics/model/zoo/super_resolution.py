import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..block import *
from ..utils import *

class SuperResolution (nn.Module):

    def __init__(self, in_channel=1, out_channel=1, filters=[1,64,1]):
        super(SuperResolution,self).__init__()        
        filters=[1,64,1]
        # Use a transposed convolution layer with learnable params, stride and kernel size to get the output size -> pytoch doc
        self.deconv1 = nn.ConvTranspose3d(filters[0],filters[1],5,stride=(2,3,3))
#         self.conv1 = conv3d_bn_elu(filters[1], filters[2], kernel_size=(3,3,3), padding=(0,0,0)) # A conv layer to get our output
        self.conv1 = nn.Conv3d(filters[1], filters[2], kernel_size=(3,3,3))
    
    def forward(self, x):
        x = F.relu(self.deconv1(x))
        # To have upscale of 2 -> get rid of one depth column *******************ask Donglai*************
        # x [nsamples, nchannels, z, y, x]
        x = x[:,:,:x.size()[2]-1,:,:]
        x = self.conv1(x)
        x = torch.sigmoid(x)
        return x
    
def test():
    model = SuperResolution()
    print('model type: ', model.__class__.__name__)
    num_params = sum([p.data.nelement() for p in model.parameters()])
    print('number of trainable parameters: ', num_params)
    x = torch.randn(1, 1, 4, 64, 64)
    y = model(x)
    print(x.size(), y.size())

if __name__ == '__main__':
    test()
    
