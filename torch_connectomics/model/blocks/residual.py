import os,sys
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .basic import *

# 1. Residual blocks
# implemented with 2D conv
class residual_block_2d_c2(nn.Module):
    def __init__(self, in_planes, out_planes, projection=True, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(residual_block_2d_c2, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv2d_norm_act( in_planes, out_planes, kernel_size=(3,3), padding=(1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv2d_norm_act(out_planes, out_planes, kernel_size=(3,3), padding=(1,1), pad_mode=pad_mode, norm_mode=norm_mode)
        )
        self.projector = conv2d_norm_act(in_planes, out_planes, kernel_size=(1,1), padding=(0,0), norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y  

# implemented with 3D conv
class residual_block_2d(nn.Module):
    """
    Residual Block 2D

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
    """
    def __init__(self, in_planes, out_planes, projection=True, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(residual_block_2d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_norm_act( in_planes, out_planes, kernel_size=(1,3,3), padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(1,3,3), padding=(0,1,1), pad_mode=pad_mode,norm_mode=norm_mode)
        )
        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0), norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y  

class residual_block_3d(nn.Module):
    """Residual Block 3D

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
    """
    def __init__(self, in_planes, out_planes, projection=False, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(residual_block_3d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_norm_act( in_planes, out_planes, kernel_size=(3,3,3), padding=(1,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(3,3,3), padding=(1,1,1), pad_mode=pad_mode, norm_mode=norm_mode)
        )
        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0), norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y       

class bottleneck_dilated_2d(nn.Module):
    """Bottleneck Residual Block 2D with Dilated Convolution

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
        dilate (int): dilation rate of conv filters.
    """
    def __init__(self, in_planes, out_planes, projection=False, dilate=2, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(bottleneck_dilated_2d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv2d_norm_act(in_planes, out_planes, kernel_size=(1, 1), padding=(0, 0), norm_mode=norm_mode, act_mode=act_mode),
            conv2d_norm_act(out_planes, out_planes, kernel_size=(3, 3), dilation=(dilate, dilate), padding=(dilate, dilate), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv2d_norm_act(out_planes, out_planes, kernel_size=(1, 1), padding=(0, 0), norm_mode=norm_mode)
        )
        self.projector = conv2d_norm_act(in_planes, out_planes, kernel_size=(1, 1), padding=(0, 0), norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.elu(y)
        return y

class bottleneck_dilated_3d(nn.Module):
    """Bottleneck Residual Block 3D with Dilated Convolution

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
        dilate (int): dilation rate of conv filters.
    """
    def __init__(self, in_planes, out_planes, projection=False, dilate=2, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(bottleneck_dilated_3d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_norm_act( in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0), norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(3,3,3), 
                          dilation=(1, dilate, dilate), padding=(1, dilate, dilate), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0), norm_mode=norm_mode)
        )
        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0), norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y        

class dilated_fusion_block(nn.Module):
    """Dilated Conv & Fusion Block

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        channel_reduction (int): channel squeezing factor.
        spatial_reduction (int): pooling factor for x,y axes.
        z_reduction (int): pooling factor for z axis.
    """
    def __init__(self, in_planes, out_planes, channel_reduction=16, spatial_reduction=2, z_reduction=1, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(dilated_fusion_block, self).__init__() 
        self.se_layer = squeeze_excitation_3d(channel=out_planes, channel_reduction=channel_reduction, 
                                spatial_reduction=spatial_reduction, z_reduction=z_reduction)

        self.inconv = conv3d_norm_act(in_planes,  out_planes, kernel_size=(3,3,3), 
                                    stride=1, padding=(1,1,1), bias=True, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)

        self.block1 = conv3d_norm_act(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,1,1), padding=(0,1,1), bias=False, pad_mode=pad_mode, norm_mode=norm_mode)
        self.block2 = conv3d_norm_act(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,2,2), padding=(0,2,2), bias=False, pad_mode=pad_mode, norm_mode=norm_mode)
        self.block3 = conv3d_norm_act(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,4,4), padding=(0,4,4), bias=False, pad_mode=pad_mode, norm_mode=norm_mode)
        self.block4 = conv3d_norm_act(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,8,8), padding=(0,8,8), bias=False, pad_mode=pad_mode, norm_mode=norm_mode)
        self.act_mode = act_mode

    def forward(self, x):
        residual  = self.inconv(x)

        x1 = self.block1(residual)
        x2 = self.block2(get_functional_act(self.act_mode)(x1))
        x3 = self.block3(get_functional_act(self.act_mode)(x2))
        x4 = self.block4(get_functional_act(self.act_mode)(x3))

        out = residual + x1 + x2 + x3 + x4
        out = self.se_layer(out)
        out = get_functional_act(self.act_mode)(out)
        return out 

class squeeze_excitation_2d(nn.Module):
    """Squeeze-and-Excitation Block 2D

    Args:
        channel (int): number of input channels.
        channel_reduction (int): channel squeezing factor.
        spatial_reduction (int): pooling factor for x,y axes.
    """
    def __init__(self, channel, channel_reduction=4, spatial_reduction=4, norm_mode='bn', act_mode='elu'):
        super(squeeze_excitation_2d, self).__init__()
        self.pool_size = (spatial_reduction, spatial_reduction)
        layers = [nn.AvgPool2d(kernel_size=self.pool_size, stride=self.pool_size)]
        layers += conv2d_norm_act(channel, channel // channel_reduction, kernel_size=1, padding=0, norm_mode=norm_mode, act_mode=act_mode, return_list=True)
        layers += conv2d_norm_act(channel // channel_reduction, channel, kernel_size=1, padding=0, norm_mode=norm_mode, return_list=True)
        layers = [nn.Sigmoid(),
                nn.Upsample(scale_factor=self.pool_size, mode='trilinear', align_corners=False)]    
        self.se = nn.Sequential(*layers)

    def forward(self, x):
        y = self.se(x)
        z = x + y*x
        return z

class squeeze_excitation_3d(nn.Module):
    """Squeeze-and-Excitation Block 3D

    Args:
        channel (int): number of input channels.
        channel_reduction (int): channel squeezing factor.
        spatial_reduction (int): pooling factor for x,y axes.
        z_reduction (int): pooling factor for z axis.
    """
    def __init__(self, channel, channel_reduction=4, spatial_reduction=4, z_reduction=1, norm_mode='bn', act_mode='elu'):
        super(squeeze_excitation_3d, self).__init__()
        self.pool_size = (z_reduction, spatial_reduction, spatial_reduction)

        layers = [nn.AvgPool3d(kernel_size=self.pool_size, stride=self.pool_size)]
        layers += conv3d_norm_act(channel,  channel//channel_reduction, kernel_size=1, padding=0, norm_mode=norm_mode, act_mode=act_mode, return_list=True)
        layers += conv3d_norm_act(channel//channel_reduction, channel,  kernel_size=1, padding=0, norm_mode=norm_mode, return_list=True)
        layers += [nn.Sigmoid(),
                nn.Upsample(scale_factor=self.pool_size, mode='trilinear', align_corners=False)]
        self.se = nn.Sequential(*layers) 

    def forward(self, x):
        y = self.se(x)
        z = x + y*x
        return z
