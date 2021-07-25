from __future__ import print_function, division
from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import *
from .att_layer import make_att_3d, SELayer2d, SELayer3d
from ..utils import get_activation, get_norm_3d

# ---------------------------
# Basic Residual Blocks
# ---------------------------


class BasicBlock2d(nn.Module):
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 projection: bool = False,
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn'):
        super(BasicBlock2d, self).__init__()
        self.conv = nn.Sequential(
            conv2d_norm_act(in_planes, planes, kernel_size=3, dilation=dilation,
                            stride=stride, groups=groups, padding=dilation,
                            pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv2d_norm_act(planes, planes, kernel_size=3, dilation=dilation,
                            stride=1, groups=groups, padding=dilation,
                            pad_mode=pad_mode, norm_mode=norm_mode, act_mode='none')
        )

        self.projector = nn.Identity()
        if in_planes != planes or stride != 1 or projection == True:
            self.projector = conv2d_norm_act(
                in_planes, planes, kernel_size=1, padding=0,
                stride=stride, norm_mode=norm_mode, act_mode='none')
        self.act = get_activation(act_mode)

    def forward(self, x):
        y = self.conv(x)
        y = y + self.projector(x)
        return self.act(y)


class BasicBlock3d(nn.Module):
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: Union[int, tuple] = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 projection: bool = False,
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 isotropic: bool = False):
        super(BasicBlock3d, self).__init__()
        if isotropic:
            kernel_size, padding = 3, dilation
        else:
            kernel_size, padding = (1, 3, 3), (0, dilation, dilation)
        self.conv = nn.Sequential(
            conv3d_norm_act(in_planes, planes, kernel_size=kernel_size, dilation=dilation,
                            stride=stride, groups=groups, padding=padding,
                            pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(planes, planes, kernel_size=kernel_size, dilation=dilation,
                            stride=1, groups=groups, padding=padding,
                            pad_mode=pad_mode, norm_mode=norm_mode, act_mode='none')
        )

        self.projector = nn.Identity()
        if in_planes != planes or stride != 1 or projection == True:
            self.projector = conv3d_norm_act(
                in_planes, planes, kernel_size=1, padding=0,
                stride=stride, norm_mode=norm_mode, act_mode='none')
        self.act = get_activation(act_mode)

    def forward(self, x):
        y = self.conv(x)
        y = y + self.projector(x)
        return self.act(y)


class BasicBlock3dPA(nn.Module):
    """Pre-activation 3D basic residual block.
    """

    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: Union[int, tuple] = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 projection: bool = False,
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 isotropic: bool = False):
        super(BasicBlock3dPA, self).__init__()
        if isotropic:
            kernel_size, padding = 3, dilation
        else:
            kernel_size, padding = (1, 3, 3), (0, dilation, dilation)

        self.conv = nn.Sequential(
            norm_act_conv3d(in_planes, planes, kernel_size=kernel_size, dilation=dilation,
                            stride=stride, groups=groups, padding=padding,
                            pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            norm_act_conv3d(planes, planes, kernel_size=kernel_size, dilation=dilation,
                            stride=1, groups=groups, padding=padding,
                            pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
        )

        self.projector = nn.Identity()
        if in_planes != planes or stride != 1 or projection == True:
            self.projector = conv3d_norm_act(
                in_planes, planes, kernel_size=1, padding=0,
                stride=stride, norm_mode=norm_mode, act_mode='none')

    def forward(self, x):
        y = self.conv(x)
        x = y + self.projector(x)
        return x

# ---------------------------
# SE Residual Blocks
# ---------------------------


class BasicBlock2dSE(BasicBlock2d):
    def __init__(self, in_planes, planes, act_mode='relu', **kwargs):
        super().__init__(in_planes=in_planes,
                         planes=planes,
                         act_mode=act_mode,
                         **kwargs)
        self.conv = nn.Sequential(
            self.conv,
            SELayer2d(planes, act_mode=act_mode))


class BasicBlock3dSE(BasicBlock3d):
    def __init__(self, in_planes, planes, act_mode='relu', **kwargs):
        super().__init__(in_planes=in_planes,
                         planes=planes,
                         act_mode=act_mode,
                         **kwargs)
        self.conv = nn.Sequential(
            self.conv,
            SELayer3d(planes, act_mode=act_mode))


class BasicBlock3dPASE(BasicBlock3dPA):
    def __init__(self, in_planes, planes, act_mode='relu', **kwargs):
        super().__init__(in_planes=in_planes,
                         planes=planes,
                         act_mode=act_mode,
                         **kwargs)
        self.conv = nn.Sequential(
            self.conv,
            SELayer3d(planes, act_mode=act_mode))

# ---------------------------
# Inverted Residual Blocks
# ---------------------------


class InvertedResidual(nn.Module):
    """3D Inverted Residual Block with Depth-wise Convolution"""

    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 stride,
                 expansion_factor=1,
                 attention=None,
                 conv_type='standard',
                 bn_momentum=0.1,
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 isotropic: bool = False,
                 bias: bool = False):
        super(InvertedResidual, self).__init__()

        # assert stride in [1, 2, (1, 2, 2)]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        # self.apply_residual = (in_ch == out_ch and stride == 1)

        conv_layer = dwconvkxkxk if isotropic else dwconv1xkxk
        DWConv = conv_layer(mid_ch, kernel_size, stride,
                            conv_type=conv_type, padding_mode=pad_mode)

        self.layers1 = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, 1, bias=False),
            get_norm_3d(norm_mode, mid_ch, bn_momentum),
            get_activation(act_mode),
            DWConv,  # Depthwise
            get_norm_3d(norm_mode, mid_ch, bn_momentum),
            get_activation(act_mode))

        self.layers2 = nn.Sequential(
            nn.Conv3d(mid_ch, out_ch, 1, bias=bias),
            get_norm_3d(norm_mode, out_ch, bn_momentum))

        self.attention = make_att_3d(attention, mid_ch)
            
        self.projector = nn.Identity()
        if DWConv.stride != (1,1,1):
            self.projector = nn.Sequential(
                nn.AvgPool3d(DWConv.stride, DWConv.stride),
                conv3d_norm_act(
                in_ch, out_ch, kernel_size=1, padding=0,
                stride=1, norm_mode=norm_mode, act_mode='none')
            )
        elif in_ch != out_ch: 
             self.projector = conv3d_norm_act(
                in_ch, out_ch, kernel_size=1, padding=0,
                stride=1, norm_mode=norm_mode, act_mode='none')
        else:
            self.projector = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.layers1(x)
        out = self.attention(out)
        out = self.layers2(out)

        if any([out.shape[i]!=identity.shape[i] for i in range(2,5)]):
            pad=[]
            for i in range(2,5):
                if out.shape[i] != identity.shape[i] and identity.shape[i]%2==1:
                    pad.extend([1,1]) 
                else:
                    pad.extend([0,0])
            identity = F.pad(identity, pad[::-1], mode='replicate')
        out += self.projector(identity)

        return out


class InvertedResidualDilated(nn.Module):
    """3D Inverted Residual Block with Dilated Depth-wise Convolution"""
    dilation_factors = [1, 2, 4, 8]

    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 stride,
                 expansion_factor=1,
                 attention=None,
                 conv_type='standard',
                 bn_momentum=0.1,
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 isotropic: bool = True,
                 bias: bool = False):
        super(InvertedResidualDilated, self).__init__()

        # assert stride in [1, 2, (1, 2, 2)]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        # self.apply_residual = (in_ch == out_ch and stride == 1)


        self.DWConv = get_dilated_dw_convs(
            mid_ch,
            self.dilation_factors,
            kernel_size,
            stride,
            conv_type,
            pad_mode,
            isotropic,
        )

        self.layers1_a = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, 1, bias=False),
            get_norm_3d(norm_mode, mid_ch, bn_momentum),
            get_activation(act_mode))

        self.layers1_b = nn.Sequential(
            get_norm_3d(norm_mode, mid_ch, bn_momentum),
            get_activation(act_mode))

        self.layers2 = nn.Sequential(
            # Linear pointwise. Note that there's no activation.
            nn.Conv3d(mid_ch, out_ch, 1, bias=bias),
            get_norm_3d(norm_mode, out_ch, bn_momentum))

        self.attention = make_att_3d(attention, mid_ch)

        self.projector = nn.Identity()
        if self.DWConv[0].stride != (1,1,1):
            self.projector = nn.Sequential(
                nn.AvgPool3d(self.DWConv[0].stride, self.DWConv[0].stride),
                conv3d_norm_act(
                in_ch, out_ch, kernel_size=1, padding=0,
                stride=1, norm_mode=norm_mode, act_mode='none')
            )
        elif in_ch != out_ch: 
             self.projector = conv3d_norm_act(
                in_ch, out_ch, kernel_size=1, padding=0,
                stride=1, norm_mode=norm_mode, act_mode='none')
        else:
            self.projector = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.layers1_a(x)
        out = self._split_conv_cat(out, self.DWConv)
        out = self.layers1_b(out)

        out = self.attention(out)
        out = self.layers2(out)

        if any([out.shape[i]!=identity.shape[i] for i in range(2,5)]):
            pad=[]
            for i in range(2,5):
                if out.shape[i] != identity.shape[i] and identity.shape[i]%2==1:
                    pad.extend([1,1]) 
                else:
                    pad.extend([0,0])
            identity = F.pad(identity, pad[::-1], mode='replicate')
        out += self.projector(identity)

        return out

    def _split_conv_cat(self, x, conv_layers):
        _, c, _, _, _ = x.size()
        z = []
        y = torch.split(x, c // len(self.dilation_factors), dim=1)
        for i in range(len(self.dilation_factors)):
            z.append(conv_layers[i](y[i]))
        return torch.cat(z, dim=1)
