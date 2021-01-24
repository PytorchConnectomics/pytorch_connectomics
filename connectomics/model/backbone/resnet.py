from __future__ import print_function, division
from typing import Optional, List

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from ..block import *
from ..utils import get_functional_act

class ResNet3D(nn.Module):
    """ResNet backbone for 3D semantic/instance segmentation. 
       The global average pooling and fully-connected layer are removed.
    """

    block_dict = {
        'residual': BasicBlock3d,
        'residual_se': BasicBlock3dSE,
    }
    num_stages = 5

    def __init__(self, 
                 block_type = 'residual',
                 in_channel: int = 1, 
                 filters: List[int] = [28, 36, 48, 64, 80],
                 blocks: List[int] = [2, 2, 2, 2],
                 isotropy: List[bool] = [False, False, False, True, True],
                 pad_mode: str = 'replicate', 
                 act_mode: str = 'elu', 
                 norm_mode: str = 'bn', 
                 init_mode: str = 'orthogonal',
                 **kwargs):
        super().__init__()
        assert len(filters) == self.num_stages
        self.block = self.block_dict[block_type]
        shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        if isotropy[0]:
            kernel_size, padding = 5, 2
        else:
            kernel_size, padding = (1,5,5), (0,2,2)
        self.conv = conv3d_norm_act(in_channel, filters[0], 
            kernel_size=kernel_size, padding=padding, **shared_kwargs)

        self.layer1 = self._make_layer(filters[0], filters[1], blocks[0], 2, isotropy[1])
        self.layer2 = self._make_layer(filters[1], filters[2], blocks[1], 2, isotropy[2])
        self.layer3 = self._make_layer(filters[2], filters[3], blocks[2], 2, isotropy[3])
        self.layer4 = self._make_layer(filters[3], filters[4], blocks[3], 2, isotropy[4])

    def _make_layer(self, in_planes: int, planes: int, blocks: int, 
                    stride: int = 1, isotropic: bool = False, **kwargs):
        if stride == 2 and not isotropic: stride = (1, 2, 2)
        layers = []
        layers.append(self.block(in_planes, planes, stride=stride, 
            isotropic=isotropic, **kwargs))
        for _ in range(1, blocks):
            layers.append(self.block(planes, planes, stride=1, 
                isotropic=isotropic, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
        