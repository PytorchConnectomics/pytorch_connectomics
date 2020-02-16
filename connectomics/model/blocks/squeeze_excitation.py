import torch.nn as nn
from .basic import *

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
