import torch
import torch.nn as nn
from ..block import *
from ..utils import *

class unet_residual_2d(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, filters=[32,64,128,256], pad_mode='rep', norm_mode='bn', act_mode='elu', head_depth=1):
        super().__init__()
        
        self.depth = len(filters) - 1
        filters = [in_channel] + filters

        # Encoding Path
        self.downC = nn.ModuleList(
            [nn.Sequential(
            conv2d_norm_act(in_planes=filters[x], out_planes=filters[x+1], 
                          kernel_size=(3,3), stride=1, padding=(1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_2d_c2(filters[x+1], filters[x+1], projection=True, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_2d_c2(filters[x+1], filters[x+1], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_2d_c2(filters[x+1], filters[x+1], projection=True, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            ) for x in range(self.depth)])

        # Center Block
        self.center = nn.Sequential(
            bottleneck_dilated_2d(filters[-2], filters[-1], projection=True),
            bottleneck_dilated_2d(filters[-1], filters[-1], projection=False),
            bottleneck_dilated_2d(filters[-1], filters[-1], projection=True)
        )

        # Decoding Path
        self.upC = nn.ModuleList(
            [nn.Sequential(
                conv2d_norm_act(in_planes=filters[x+1], out_planes=filters[x+1], 
                          kernel_size=(3,3), stride=1, padding=(1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_2d_c2(filters[x+1], filters[x+1], projection=True, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_2d_c2(filters[x+1], filters[x+1], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_2d_c2(filters[x+1], filters[x+1], projection=True, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            ) for x in range(self.depth)])

        # down & up sampling
        self.downS = nn.ModuleList([nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) for x in range(self.depth)])
        head_pred = [residual_block_2d_c2(filters[1], filters[1], projection=False)
                                for x in range(head_depth-1)] + \
                    [conv2d_norm_act(filters[1], out_channel, kernel_size=(1, 1), padding=0, norm_mode=norm_mode)]
        
        self.upS = nn.ModuleList( [nn.Sequential(*head_pred)] + \
                             [nn.Sequential(
                    conv2d_norm_act(filters[x+1], filters[x], kernel_size=(1, 1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                    nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)) for x in range(1, 1 + self.depth)])
  

        # initialization
        ortho_init(self)

    def forward(self, x):
       # encoding path
        down_u = [None] * (self.depth)
        for i in range(self.depth):
            down_u[i] = self.downC[i](x)
            x = self.downS[i](down_u[i])

        x = self.center(x)

        # decoding path
        for i in range(self.depth-1,-1,-1):
            x = down_u[i] + self.upS[i + 1](x)
            x = self.upC[i](x)

        x = self.upS[0](x)

        x = torch.sigmoid(x)
        return x

class unet_2d_ds(nn.Module):
    # unet_2d with deep supervision
    def __init__(self, in_channel=1, out_channel=1, filters=[32,64,128,256], activation='sigmoid'):
        super().__init__(in_channel, out_channel, filters, activation)
        print('final activation function: ' + self.activation)   

        self.so1_conv1 = conv2d_bn_elu(filters[1], filters[0], kernel_size=(3,3), padding=(1,1))
        self.so1_fconv = conv2d_bn_non(filters[0], out_channel, kernel_size=(3,3), padding=(1,1))

        self.so2_conv1 = conv2d_bn_elu(filters[2], filters[0], kernel_size=(3,3), padding=(1,1))
        self.so2_fconv = conv2d_bn_non(filters[0], out_channel, kernel_size=(3,3), padding=(1,1))

        self.so3_conv1 = conv2d_bn_elu(filters[3], filters[0], kernel_size=(3,3), padding=(1,1))
        self.so3_fconv = conv2d_bn_non(filters[0], out_channel, kernel_size=(3,3), padding=(1,1))

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

class unet_2d_sk(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, filters=[32,64,128,256], activation='sigmoid'):
        super().__init__(in_channel, out_channel, filters, activation)
        self.map_conv1 = conv2d_bn_elu(filters[0], filters[0], kernel_size=(3,3), padding=(1,1))
        self.map_fconv = conv2d_bn_non(filters[0], out_channel, kernel_size=(3,3), padding=(1,1))

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


class unet_2d_so1(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, filters=[32,64,128,256], activation='sigmoid'):
        super().__init__(in_channel, out_channel, filters, activation)
        print('final activation function: ' + self.activation)

        self.side_out1 = nn.Sequential(
            conv2d_bn_elu(filters[1], filters[0], kernel_size=(3,3), padding=(1,1)),
            conv2d_bn_non(filters[0], out_channel, kernel_size=(3,3), padding=(1,1)))

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

class unet_2d_so2(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, filters=[32,64,128,256], activation='sigmoid'):
        super().__init__(in_channel, out_channel, filters, activation)
        print('final activation function: ' + self.activation)

        self.side_out2 = nn.Sequential(
            conv2d_bn_elu(filters[2], filters[0], kernel_size=(3,3), padding=(1,1)),
            conv2d_bn_non(filters[0], out_channel, kernel_size=(3,3), padding=(1,1)))    

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

class unet_2d_so3(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, filters=[32,64,128,256], activation='sigmoid'):
        super().__init__(in_channel, out_channel, filters, activation)
        print('final activation function: ' + self.activation)

        self.side_out3 = nn.Sequential(
            conv2d_bn_elu(filters[3], filters[0], kernel_size=(3,3), padding=(1,1)),
            conv2d_bn_non(filters[0], out_channel, kernel_size=(3,3), padding=(1,1)))    

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


