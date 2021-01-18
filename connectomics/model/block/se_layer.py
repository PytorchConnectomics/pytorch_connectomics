import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual import *
from ..utils import get_activation

class se_layer_2d(nn.Module):
    def __init__(self, channel, reduction=4, act_mode='relu'):
        super(se_layer_2d, self).__init__()
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

class se_layer_3d(nn.Module):
    def __init__(self, channel, reduction=4, act_mode='relu'):
        super(se_layer_3d, self).__init__()
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

class residual_se_block_2d(residual_block_2d):
    def __init__(self, in_planes, out_planes, projection=False, dilation=1,
                 pad_mode='replicate', norm_mode='bn', act_mode='relu'):
        super().__init__(in_planes, out_planes, projection, dilation,
                         pad_mode, norm_mode, act_mode)
        self.conv = nn.Sequential(
            self.conv,
            se_layer_2d(out_planes, act_mode=act_mode))

class residual_se_block_3d(residual_block_3d):
    def __init__(self, in_planes, out_planes, projection=False, dilation=1,
                 pad_mode='replicate', norm_mode='bn', act_mode='relu',
                 isotropy=True):
        super().__init__(in_planes, out_planes, projection,
                         dilation, pad_mode, norm_mode, 
                         act_mode, isotropy)
        self.conv = nn.Sequential(
            self.conv,
            se_layer_3d(out_planes, act_mode=act_mode))
