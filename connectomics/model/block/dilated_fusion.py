import torch
import torch.nn as nn

from .squeeze_excitation import squeeze_excitation_3d
from .basic import *

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
