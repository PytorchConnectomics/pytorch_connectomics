import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import *
from ..utils import get_norm, get_activation

class residual_block_2d(nn.Module):
    def __init__(self, in_planes, out_planes, projection=False, 
                 pad_mode='replicate', norm_mode='bn', act_mode='relu'):
        super(residual_block_2d, self).__init__()
        self.conv = nn.Sequential(
            conv2d_norm_act( in_planes, out_planes, kernel_size=(3,3), 
                            padding=(1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv2d_norm_act(out_planes, out_planes, kernel_size=(3,3), 
                            padding=(1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode='none')
        )
        if in_planes != out_planes or projection == True:
            self.projector = conv2d_norm_act(in_planes, out_planes, kernel_size=1, 
                padding=0, norm_mode=norm_mode, act_mode='none')
        else:
            self.projector = nn.Identity()
        self.act = get_activation(act_mode)
        
    def forward(self, x):
        y = self.conv(x)
        y = y + self.projector(x)
        return self.act(y)

class residual_block_3d(nn.Module):
    def __init__(self, in_planes, out_planes, projection=False, 
                 pad_mode='replicate', norm_mode='bn', act_mode='relu',
                 isotropy=True):
        super(residual_block_3d, self).__init__()
        if isotropy:
            kernel_size, padding = (3,3,3), (1,1,1)
        else:
            kernel_size, padding = (1,3,3), (0,1,1)
        self.conv = nn.Sequential(
            conv3d_norm_act( in_planes, out_planes, kernel_size=kernel_size, 
                            padding=padding, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=kernel_size, 
                            padding=padding, pad_mode=pad_mode, norm_mode=norm_mode, act_mode='none')
        )
        if in_planes != out_planes or projection == True:
            self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=1, 
                padding=0, norm_mode=norm_mode, act_mode='none')
        self.act = get_activation(act_mode)
        
    def forward(self, x):
        y = self.conv(x)
        y = y + self.projector(x)
        return self.act(y)
