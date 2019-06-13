import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch_connectomics.model.blocks import *
from torch_connectomics.libs.sync import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

class unet_2d(nn.Module):
    def __init__(self, in_num=1, out_num=1, filters=[32,64,128,256], activation='sigmoid'):
        super(unet_2d, self).__init__()
        self.activation = activation
        print('final activation function: '+self.activation)

        # Encoding Path
        self.layer1_E = nn.Sequential(
            residual_block_2d_c2(in_num, filters[0], projection=True),
            residual_block_2d_c2(filters[0], filters[0], projection=False),
            residual_block_2d_c2(filters[0], filters[0], projection=True)
            #SELayer(channel=filters[0], channel_reduction=2, spatial_reduction=16)
        )

        self.layer2_E = nn.Sequential(
            residual_block_2d_c2(filters[0], filters[1], projection=True),
            residual_block_2d_c2(filters[1], filters[1], projection=False),
            residual_block_2d_c2(filters[1], filters[1], projection=True)
            #SELayer(channel=filters[1], channel_reduction=4, spatial_reduction=8)
        )
        self.layer3_E = nn.Sequential(
            residual_block_2d_c2(filters[1], filters[2], projection=True),
            residual_block_2d_c2(filters[2], filters[2], projection=False),
            residual_block_2d_c2(filters[2], filters[2], projection=True)
            #SELayer(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )

        # Center Block
        self.center = nn.Sequential(
            bottleneck_dilated_2d(filters[2], filters[3], projection=True),
            bottleneck_dilated_2d(filters[3], filters[3], projection=False),
            bottleneck_dilated_2d(filters[3], filters[3], projection=True)
            #SELayer(channel=filters[3], channel_reduction=16, spatial_reduction=2, z_reduction=2)
        )

        # Decoding Path
        self.layer1_D = nn.Sequential(
            residual_block_2d_c2(filters[0], filters[0], projection=True),
            residual_block_2d_c2(filters[0], filters[0], projection=False),
            residual_block_2d_c2(filters[0], filters[0], projection=True)
            #SELayer(channel=filters[0], channel_reduction=2, spatial_reduction=16)
        )

        self.layer2_D = nn.Sequential(
            residual_block_2d_c2(filters[1], filters[1], projection=True),
            residual_block_2d_c2(filters[1], filters[1], projection=False),
            residual_block_2d_c2(filters[1], filters[1], projection=True)
            #SELayer(channel=filters[1], channel_reduction=4, spatial_reduction=8)
        )
        self.layer3_D = nn.Sequential(
            residual_block_2d_c2(filters[2], filters[2], projection=True),
            residual_block_2d_c2(filters[2], filters[2], projection=False),
            residual_block_2d_c2(filters[2], filters[2], projection=True)
            #SELayer(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )

        # down & up sampling
        self.down = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.up = nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=True)

        # convert to probability
        self.conv1 = conv2d_bn_elu(filters[1], filters[0], kernel_size=(1,1), padding=(0,0))
        self.conv2 = conv2d_bn_elu(filters[2], filters[1], kernel_size=(1,1), padding=(0,0))
        self.conv3 = conv2d_bn_elu(filters[3], filters[2], kernel_size=(1,1), padding=(0,0))
        self.fconv = conv2d_bn_non(filters[0], out_num, kernel_size=(3,3), padding=(1,1))

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):

        z1 = self.layer1_E(x)
        x = self.down(z1)
        z2 = self.layer2_E(x)
        x = self.down(z2)
        z3 = self.layer3_E(x)
        x = self.down(z3)

        x = self.center(x)

        # Decoding Path
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
        if self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.activation == 'tanh':
            x = torch.tanh(x) 
        return x

class unet_2d_ds(unet_2d):
    # unet_2d with deep supervision
    def __init__(self, in_num=1, out_num=1, filters=[32,64,128,256], activation='sigmoid'):
        super().__init__(in_num, out_num, filters, activation)
        print('final activation function: ' + self.activation)   

        self.so1_conv1 = conv2d_bn_elu(filters[1], filters[0], kernel_size=(3,3), padding=(1,1))
        self.so1_fconv = conv2d_bn_non(filters[0], out_num, kernel_size=(3,3), padding=(1,1))

        self.so2_conv1 = conv2d_bn_elu(filters[2], filters[0], kernel_size=(3,3), padding=(1,1))
        self.so2_fconv = conv2d_bn_non(filters[0], out_num, kernel_size=(3,3), padding=(1,1))

        self.so3_conv1 = conv2d_bn_elu(filters[3], filters[0], kernel_size=(3,3), padding=(1,1))
        self.so3_fconv = conv2d_bn_non(filters[0], out_num, kernel_size=(3,3), padding=(1,1))

    def forward(self, x):

        z1 = self.layer1_E(x)
        x = self.down(z1)
        z2 = self.layer2_E(x)
        x = self.down(z2)
        z3 = self.layer3_E(x)
        x = self.down(z3)

        x = self.center(x)

        # side output 3
        so3_add =  self.so3_conv1(x)
        so3 = self.so3_fconv(so3_add)
        so3 = torch.sigmoid(so3)

        # Decoding Path
        x = self.up(self.conv3(x))
        x = x + z3
        x = self.layer3_D(x)
    
        # side output 2
        so2_add =  self.so2_conv1(x) + self.up(so3_add)
        so2 = self.so2_fconv(so2_add)
        so2 = torch.sigmoid(so2)

        x = self.up(self.conv2(x))
        x = x + z2
        x = self.layer2_D(x)

        # side output 1
        so1_add =  self.so1_conv1(x) + self.up(so2_add)
        so1 = self.so1_fconv(so1_add)
        so1 = torch.sigmoid(so1)

        x = self.up(self.conv1(x))
        x = x + z1
        x = self.layer1_D(x)

        x = x + self.up(so1_add)

        x = self.fconv(x)
        if self.activation == 'sigmoid':
            x = 2.0 * (torch.sigmoid(x) - 0.5)
        elif self.activation == 'tanh':
            x = torch.tanh(x) 
        return x, so1, so2, so3 

class unet_2d_sk(unet_2d_ds):
    def __init__(self, in_num=1, out_num=1, filters=[32,64,128,256], activation='sigmoid'):
        super().__init__(in_num, out_num, filters, activation)
        self.map_conv1 = conv2d_bn_elu(filters[0], filters[0], kernel_size=(3,3), padding=(1,1))
        self.map_fconv = conv2d_bn_non(filters[0], out_num, kernel_size=(3,3), padding=(1,1))

    def forward(self, x):
        z1 = self.layer1_E(x)
        x = self.down(z1)
        z2 = self.layer2_E(x)
        x = self.down(z2)
        z3 = self.layer3_E(x)
        x = self.down(z3)

        x = self.center(x)

        # side output 3
        so3_add =  self.so3_conv1(x)
        so3 = self.so3_fconv(so3_add)
        so3 = torch.sigmoid(so3)

        # Decoding Path
        x = self.up(self.conv3(x))
        x = x + z3
        x = self.layer3_D(x)
    
        # side output 2
        so2_add =  self.so2_conv1(x) + self.up(so3_add)
        so2 = self.so2_fconv(so2_add)
        so2 = torch.sigmoid(so2)

        x = self.up(self.conv2(x))
        x = x + z2
        x = self.layer2_D(x)

        # side output 1
        so1_add =  self.so1_conv1(x) + self.up(so2_add)
        so1 = self.so1_fconv(so1_add)
        so1 = torch.sigmoid(so1)

        x = self.up(self.conv1(x))
        x = x + z1
        x = self.layer1_D(x)

        x = x + self.up(so1_add)

        # side output 0
        so0 = self.fconv(x)
        so0 = torch.sigmoid(so0)

        # energy map
        x = self.map_conv1(x)
        x = self.map_fconv(x)
        if self.activation == 'sigmoid':
            x = 2.0 * (torch.sigmoid(x) - 0.5)
        elif self.activation == 'tanh':
            x = torch.tanh(x) 

        # print('x', x.size() )
        # print('0',so0.size())
        # print('1',so1.size())
        # print('2',so2.size())
        # print('3',so3.size())
        return x, so0, so1, so2, so3     


class unet_2d_so1(unet_2d):
    def __init__(self, in_num=1, out_num=1, filters=[32,64,128,256], activation='sigmoid'):
        super().__init__(in_num, out_num, filters, activation)
        print('final activation function: ' + self.activation)

        self.side_out1 = nn.Sequential(
            conv2d_bn_elu(filters[1], filters[0], kernel_size=(3,3), padding=(1,1)),
            conv2d_bn_non(filters[0], out_num, kernel_size=(3,3), padding=(1,1)))

    def forward(self, x):

        z1 = self.layer1_E(x)
        x = self.down(z1)
        z2 = self.layer2_E(x)
        x = self.down(z2)
        z3 = self.layer3_E(x)
        x = self.down(z3)

        x = self.center(x)

        # Decoding Path
        x = self.up(self.conv3(x))
        x = x + z3
        x = self.layer3_D(x)

        x = self.up(self.conv2(x))
        x = x + z2
        x = self.layer2_D(x)

        # side output
        so1 = self.side_out1(x)
        so1 = torch.sigmoid(so1)

        x = self.up(self.conv1(x))
        x = x + z1
        x = self.layer1_D(x)

        x = self.fconv(x)
        if self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.activation == 'tanh':
            x = torch.tanh(x) 
        return x, so1

class unet_2d_so2(unet_2d_so1):
    def __init__(self, in_num=1, out_num=1, filters=[32,64,128,256], activation='sigmoid'):
        super().__init__(in_num, out_num, filters, activation)
        print('final activation function: ' + self.activation)

        self.side_out2 = nn.Sequential(
            conv2d_bn_elu(filters[2], filters[0], kernel_size=(3,3), padding=(1,1)),
            conv2d_bn_non(filters[0], out_num, kernel_size=(3,3), padding=(1,1)))    

    def forward(self, x):

        z1 = self.layer1_E(x)
        x = self.down(z1)
        z2 = self.layer2_E(x)
        x = self.down(z2)
        z3 = self.layer3_E(x)
        x = self.down(z3)

        x = self.center(x)

        # Decoding Path
        x = self.up(self.conv3(x))
        x = x + z3
        x = self.layer3_D(x)

        # side output 2
        so2 = self.side_out2(x)
        so2 = torch.sigmoid(so2)

        x = self.up(self.conv2(x))
        x = x + z2
        x = self.layer2_D(x)

        # side output 1
        so1 = self.side_out1(x)
        so1 = torch.sigmoid(so1)

        x = self.up(self.conv1(x))
        x = x + z1
        x = self.layer1_D(x)

        x = self.fconv(x)
        if self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.activation == 'tanh':
            x = torch.tanh(x) 
        return x, so1, so2

class unet_2d_so3(unet_2d_so2):
    def __init__(self, in_num=1, out_num=1, filters=[32,64,128,256], activation='sigmoid'):
        super().__init__(in_num, out_num, filters, activation)
        print('final activation function: ' + self.activation)

        self.side_out3 = nn.Sequential(
            conv2d_bn_elu(filters[3], filters[0], kernel_size=(3,3), padding=(1,1)),
            conv2d_bn_non(filters[0], out_num, kernel_size=(3,3), padding=(1,1)))    

    def forward(self, x):

        z1 = self.layer1_E(x)
        x = self.down(z1)
        z2 = self.layer2_E(x)
        x = self.down(z2)
        z3 = self.layer3_E(x)
        x = self.down(z3)

        x = self.center(x)

        # side output 3
        so3 = self.side_out3(x)
        so3 = torch.sigmoid(so3)

        # Decoding Path
        x = self.up(self.conv3(x))
        x = x + z3
        x = self.layer3_D(x)

        # side output 2
        so2 = self.side_out2(x)
        so2 = torch.sigmoid(so2)

        x = self.up(self.conv2(x))
        x = x + z2
        x = self.layer2_D(x)

        # side output 1
        so1 = self.side_out1(x)
        so1 = torch.sigmoid(so1)

        x = self.up(self.conv1(x))
        x = x + z1
        x = self.layer1_D(x)

        x = self.fconv(x)
        if self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.activation == 'tanh':
            x = torch.tanh(x) 
        return x, so1, so2, so3       

class SELayer(nn.Module):
    # Squeeze-and-excitation layer
    def __init__(self, channel, channel_reduction=4, spatial_reduction=4, z_reduction=1):
        super(SELayer, self).__init__()
        self.pool_size = (z_reduction, spatial_reduction, spatial_reduction)
        self.se = nn.Sequential(
                nn.AvgPool3d(kernel_size=self.pool_size, stride=self.pool_size),
                nn.Conv3d(channel, channel // channel_reduction, kernel_size=1),
                SynchronizedBatchNorm3d(channel // channel_reduction),
                nn.ELU(inplace=True),
                nn.Conv3d(channel // channel_reduction, channel, kernel_size=1),
                SynchronizedBatchNorm3d(channel),
                nn.Sigmoid(),
                nn.Upsample(scale_factor=self.pool_size, mode='trilinear', align_corners=True),
                )     

    def forward(self, x):
        y = self.se(x)
        z = x + y*x
        return z 

def test():
    model = unet_2d_ds(in_num=1, out_num=1, filters=[32,64,128,256], activation='sigmoid')
    x = torch.rand(8, 1, 512, 512)
    out = model(x)
    print(out.size())

if __name__ == '__main__':
    test()
