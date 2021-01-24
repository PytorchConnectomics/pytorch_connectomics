import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual import *
from ..utils import get_activation

class SELayer2d(nn.Module):
    def __init__(self, channel, reduction=4, act_mode='relu'):
        super(SELayer2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            get_activation(act_mode),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELayer3d(nn.Module):
    def __init__(self, channel, reduction=4, act_mode='relu'):
        super(SELayer3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            get_activation(act_mode),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

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
