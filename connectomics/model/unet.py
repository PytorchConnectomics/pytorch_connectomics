from __future__ import print_function, division
from typing import Optional, List

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .block import *
from .utils.initialize import *

class UNet3D(nn.Module):
    """3D U-Net architecture.

    Args:
        block_type (str): the block type at each U-Net stage. Default: ``'residual'``
        in_channel (int): number of input channels. Default: 1
        out_channel (int): number of output channels. Default: 3
        filters (List[int]): number of filters at each U-Net stage. Default: [28, 36, 48, 64, 80]
        isotropy (List[bool]): specify each U-Net stage is isotropic or anisotropic. 
            Default: [False, False, True, True, True]
        pad_mode (str): one of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'replicate'``
        norm_mode (str): one of ``'bn'``, ``'sync_bn'`` ``'in'`` or ``'gn'``. Default: ``'bn'``
        act_mode (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``, 
            ``'swish'``, ``'efficient_swish'`` or ``'none'``. Default: ``'relu'``
        do_embedding (bool): embed the input to high-dimensional feature space. Default: `True`
        head_depth (int): depth of the U-Net head. Default: 1
        output_act (str): activation function for the output layer. Default: ``'sigmoid'``
    """
    def __init__(self, 
                 block_type = 'residual',
                 in_channel: int = 1, 
                 out_channel: int = 3, 
                 filters: List[int] = [28, 36, 48, 64, 80], 
                 isotropy: List[bool] = [False, False, True, True, True],
                 pad_mode: str = 'replicate', 
                 norm_mode: str = 'BN', 
                 act_mode: str = 'elu', 
                 do_embedding: bool = True, 
                 head_depth: int = 1, 
                 output_act: str = 'sigmoid'):
        super().__init__()

        self.depth = len(filters)-2
        self.do_embedding = do_embedding
        self.output_act = output_act # activation function for the output layer

        # encoding path
        if self.do_embedding: 
            num_out = filters[1]
            self.downE = nn.Sequential(
                # anisotropic embedding
                conv3d_norm_act(in_planes=in_channel, out_planes=filters[0], 
                              kernel_size=(1,5,5), stride=1, padding=(0,2,2), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                # 2d residual module
                conv3d_norm_act(in_planes=filters[0], out_planes=filters[0], 
                              kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_2d(filters[0], filters[0], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            )
        else:
            filters[0] = in_channel
            num_out = out_channel
        
        self.downC = nn.ModuleList(
            [nn.Sequential(
            conv3d_norm_act(in_planes=filters[x], out_planes=filters[x+1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_3d(filters[x+1], filters[x+1], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            ) for x in range(self.depth)])

        # pooling downsample
        self.downS = nn.ModuleList([nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) for x in range(self.depth+1)])

        # center block
        self.center = nn.Sequential(conv3d_norm_act(in_planes=filters[-2], out_planes=filters[-1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_3d(filters[-1], filters[-1], projection=True)
        )

        self.upC = nn.ModuleList(
            [nn.Sequential(
                conv3d_norm_act(in_planes=filters[x+1], out_planes=filters[x+1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_3d(filters[x+1], filters[x+1], projection=False)
            ) for x in range(self.depth)])

        if self.do_embedding: 
            # decoding path
            self.upE = nn.Sequential(
                conv3d_norm_act(in_planes=filters[0], out_planes=filters[0], 
                              kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_2d(filters[0], filters[0], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                conv3d_norm_act(in_planes=filters[0], out_planes=out_channel, 
                              kernel_size=(1,5,5), stride=1, padding=(0,2,2), pad_mode=pad_mode, norm_mode=norm_mode)
            )
            # conv + upsample
            self.upS = nn.ModuleList([nn.Sequential(
                            conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                            nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)) for x in range(self.depth+1)])
        else:
            # new
            head_pred = [residual_block_3d(filters[1], filters[1], projection=False)
                                    for x in range(head_depth-1)] + \
                              [conv3d_norm_act(filters[1], out_channel, kernel_size=(1,1,1), padding=0, norm_mode=norm_mode)]
            self.upS = nn.ModuleList( [nn.Sequential(*head_pred)] + \
                                 [nn.Sequential(
                        conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                        nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)) for x in range(1,self.depth+1)])

        #initialization
        ortho_init(self)

    def forward(self, x):
        # encoding path
        if self.do_embedding: 
            z = self.downE(x)
            x = self.downS[0](z)

        down_u = [None] * (self.depth)
        for i in range(self.depth):
            down_u[i] = self.downC[i](x)
            x = self.downS[i+1](down_u[i])

        x = self.center(x)

        # decoding path
        for i in range(self.depth-1,-1,-1):
            x = down_u[i] + self.upS[i+1](x)
            x = self.upC[i](x)

        if self.do_embedding: 
            x = z + self.upS[0](x)
            x = self.upE(x)
        else:
            x = self.upS[0](x)

        x = get_functional_act(self.output_act)(x)
        return x

class UNet2D(nn.Module):
    """2D U-Net architecture.

    Args:
        block_type (str): the block type at each U-Net stage. Default: ``'residual'``
        in_channel (int): number of input channels. Default: 1
        out_channel (int): number of output channels. Default: 3
        filters (List[int]): number of filters at each U-Net stage. Default: [28, 36, 48, 64, 80]
        isotropy (List[bool]): specify each U-Net stage is isotropic or anisotropic. 
            Default: [False, False, True, True, True]
        pad_mode (str): one of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'replicate'``
        norm_mode (str): one of ``'bn'``, ``'sync_bn'`` ``'in'`` or ``'gn'``. Default: ``'bn'``
        act_mode (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``, 
            ``'swish'`` and ``'efficient_swish'``. Default: ``'relu'``
        do_embedding (bool): embed the input to high-dimensional feature space. Default: `True`
        head_depth (int): depth of the U-Net head. Default: 1
        output_act (str): activation function for the output layer. Default: ``'sigmoid'``
    """
    def __init__(self):
        pass
