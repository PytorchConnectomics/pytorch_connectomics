from __future__ import print_function, division
from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import *
from ..utils import get_activation

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
        if in_planes != planes or stride != 1 or projection == True:
            self.projector = conv2d_norm_act(in_planes, planes, kernel_size=1, 
                padding=0, stride=stride, norm_mode=norm_mode, act_mode='none')
        else:
            self.projector = nn.Identity()
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
            kernel_size, padding = (1,3,3), (0,dilation,dilation)
        self.conv = nn.Sequential(
            conv3d_norm_act(in_planes, planes, kernel_size=kernel_size, dilation=dilation,
                            stride=stride, groups=groups, padding=padding, 
                            pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(planes, planes, kernel_size=kernel_size, dilation=dilation,
                            stride=1, groups=groups, padding=padding,
                            pad_mode=pad_mode, norm_mode=norm_mode, act_mode='none')
        )
        if in_planes != planes or stride != 1 or projection == True:
            self.projector = conv3d_norm_act(in_planes, planes, kernel_size=1, 
                padding=0, stride=stride, norm_mode=norm_mode, act_mode='none')
        else:
            self.projector = nn.Identity()
        self.act = get_activation(act_mode)
        
    def forward(self, x):
        y = self.conv(x)
        y = y + self.projector(x)
        return self.act(y)
