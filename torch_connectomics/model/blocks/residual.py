import os,sys
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .basic import *
from torch_connectomics.libs.sync import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

# 1. Residual blocks
# implemented with 2D conv
class residual_block_2d_c2(nn.Module):
    def __init__(self, in_planes, out_planes, projection=True):
        super(residual_block_2d_c2, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv2d_bn_elu( in_planes, out_planes, kernel_size=(3,3), padding=(1,1)),
            conv2d_bn_non(out_planes, out_planes, kernel_size=(3,3), padding=(1,1))
        )
        self.projector = conv2d_bn_non(in_planes, out_planes, kernel_size=(1,1), padding=(0,0))
        self.elu = nn.ELU(inplace=True)
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.elu(y)
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
    def __init__(self, in_planes, out_planes, projection=True):
        super(residual_block_2d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_bn_elu( in_planes, out_planes, kernel_size=(1,3,3), padding=(0,1,1)),
            conv3d_bn_non(out_planes, out_planes, kernel_size=(1,3,3), padding=(0,1,1))
        )
        self.projector = conv3d_bn_non(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0))
        self.elu = nn.ELU(inplace=True)
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.elu(y)
        return y  

class residual_block_3d(nn.Module):
    """Residual Block 3D

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
    """
    def __init__(self, in_planes, out_planes, projection=False):
        super(residual_block_3d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_bn_elu( in_planes, out_planes, kernel_size=(3,3,3), padding=(1,1,1)),
            conv3d_bn_non(out_planes, out_planes, kernel_size=(3,3,3), padding=(1,1,1))
        )
        self.projector = conv3d_bn_non(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0))
        self.elu = nn.ELU(inplace=True)
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.elu(y)
        return y       

class bottleneck_dilated_2d(nn.Module):
    """Bottleneck Residual Block 2D with Dilated Convolution

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
        dilate (int): dilation rate of conv filters.
    """
    def __init__(self, in_planes, out_planes, projection=False, dilate=2):
        super(bottleneck_dilated_2d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv2d_bn_elu(in_planes, out_planes,
                          kernel_size=(1, 1), padding=(0, 0)),
            conv2d_bn_elu(out_planes, out_planes, 
                          kernel_size=(3, 3), dilation=(dilate, dilate), padding=(dilate, dilate)),
            conv2d_bn_non(out_planes, out_planes,
                          kernel_size=(1, 1), padding=(0, 0))
        )
        self.projector = conv2d_bn_non(
            in_planes, out_planes, kernel_size=(1, 1), padding=(0, 0))
        self.elu = nn.ELU(inplace=True)

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
    def __init__(self, in_planes, out_planes, projection=False, dilate=2):
        super(bottleneck_dilated_3d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_bn_elu( in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0)),
            conv3d_bn_elu(out_planes, out_planes, kernel_size=(3,3,3), 
                          dilation=(1, dilate, dilate), padding=(1, dilate, dilate)),
            conv3d_bn_non(out_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0))
        )
        self.projector = conv3d_bn_non(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0))
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.elu(y)
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
    def __init__(self, in_planes, out_planes, channel_reduction=16, spatial_reduction=2, z_reduction=1):
        super(dilated_fusion_block, self).__init__() 
        self.se_layer = squeeze_excitation_3d(channel=out_planes, channel_reduction=channel_reduction, 
                                spatial_reduction=spatial_reduction, z_reduction=z_reduction)

        self.inconv = conv3d_bn_elu(in_planes,  out_planes, kernel_size=(3,3,3), 
                                    stride=1, padding=(1,1,1), bias=True)

        self.block1 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,1,1), padding=(0,1,1), bias=False)
        self.block2 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,2,2), padding=(0,2,2), bias=False)
        self.block3 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,4,4), padding=(0,4,4), bias=False)
        self.block4 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,8,8), padding=(0,8,8), bias=False)                                                                                     

    def forward(self, x):
        residual  = self.inconv(x)

        x1 = self.block1(residual)
        x2 = self.block2(F.elu(x1, inplace=True))
        x3 = self.block3(F.elu(x2, inplace=True))
        x4 = self.block4(F.elu(x3, inplace=True))

        out = residual + x1 + x2 + x3 + x4
        out = self.se_layer(out)
        out = F.elu(out, inplace=True)
        return out 

class squeeze_excitation_2d(nn.Module):
    """Squeeze-and-Excitation Block 2D

    Args:
        channel (int): number of input channels.
        channel_reduction (int): channel squeezing factor.
        spatial_reduction (int): pooling factor for x,y axes.
    """
    def __init__(self, channel, channel_reduction=4, spatial_reduction=4):
        super(squeeze_excitation_2d, self).__init__()
        self.pool_size = (spatial_reduction, spatial_reduction)
        self.se = nn.Sequential(
                nn.AvgPool2d(kernel_size=self.pool_size, stride=self.pool_size),
                nn.Conv2d(channel, channel // channel_reduction, kernel_size=1),
                SynchronizedBatchNorm2d(channel // channel_reduction),
                nn.ELU(inplace=True),
                nn.Conv2d(channel // channel_reduction, channel, kernel_size=1),
                SynchronizedBatchNorm3d(channel),
                nn.Sigmoid(),
                nn.Upsample(scale_factor=self.pool_size, mode='trilinear', align_corners=False),
                )     

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
    def __init__(self, channel, channel_reduction=4, spatial_reduction=4, z_reduction=1):
        super(squeeze_excitation_3d, self).__init__()
        self.pool_size = (z_reduction, spatial_reduction, spatial_reduction)
        self.se = nn.Sequential(
                nn.AvgPool3d(kernel_size=self.pool_size, stride=self.pool_size),
                nn.Conv3d(channel, channel // channel_reduction, kernel_size=1),
                SynchronizedBatchNorm3d(channel // channel_reduction),
                nn.ELU(inplace=True),
                nn.Conv3d(channel // channel_reduction, channel, kernel_size=1),
                SynchronizedBatchNorm3d(channel),
                nn.Sigmoid(),
                nn.Upsample(scale_factor=self.pool_size, mode='trilinear', align_corners=False),
                )     

    def forward(self, x):
        y = self.se(x)
        z = x + y*x
        return z
