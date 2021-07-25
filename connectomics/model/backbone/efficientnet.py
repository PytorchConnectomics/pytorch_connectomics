from __future__ import print_function, division
from typing import Optional, Union, List

import torch
import torch.nn as nn
from ..block.basic import get_conv
from ..block.residual import InvertedResidual, InvertedResidualDilated
from ..utils import get_activation, get_norm_3d

class DilatedBlock(nn.Module):
    def __init__(self, conv_type, in_channel, inplanes, dilation_factors, pad_mode):
        super().__init__()
        self.conv = nn.ModuleList([get_conv(conv_type)(in_channel, inplanes, kernel_size=3, 
                bias=False, stride=1, dilation=dilation_factors[i], padding=dilation_factors[i], 
                padding_mode=pad_mode) for i in range(4)])

    def forward(self, x):
        return self._conv_and_cat(x, self.conv)
    
    def _conv_and_cat(self, x, conv_layers):
        y = [conv(x) for conv in conv_layers]
        return torch.cat(y, dim=1)
        
class EfficientNet3D(nn.Module):
    """EfficientNet backbone for 3D semantic and instance segmentation.
    """
    expansion_factor = 1
    dilation_factors = [1, 2, 4, 8]
    num_stages = 5

    block_dict = {
        'inverted_res': InvertedResidual,
        'inverted_res_dilated': InvertedResidualDilated,
    }

    def __init__(self,
                 block_type: str = 'inverted_res',
                 in_channel: int = 1,
                 filters: List[int] = [32, 64, 96, 128, 160],
                 blocks: List[int] = [1, 2, 2, 2, 4],
                 ks: List[int] = [3, 3, 5, 3, 3],
                 isotropy: List[bool] = [False, False, False, True, True],
                 attention: str = 'squeeze_excitation',
                 bn_momentum: float = 0.01,
                 conv_type: str = 'standard',
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 **_):
        super(EfficientNet3D, self).__init__()
        block = self.block_dict[block_type]

        self.inplanes = filters[0]

        if block == InvertedResidualDilated:
            self.all_dilated = True
            num_conv = len(self.dilation_factors)
            self.conv1 = DilatedBlock(conv_type, 
                            in_channel, 
                            self.inplanes//num_conv, 
                            self.dilation_factors, 
                            pad_mode)
        else:
            self.all_dilated = False
            self.conv1 = get_conv(conv_type)(
                in_channel,
                self.inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=pad_mode,
                bias=False)

        self.bn1 = get_norm_3d(norm_mode, self.inplanes, bn_momentum)
        self.relu = get_activation(act_mode)

        shared_kwargs = {
            'expansion_factor': self.expansion_factor,
            'bn_momentum': bn_momentum,
            'norm_mode': norm_mode,
            'attention': attention,
            'pad_mode': pad_mode,
            'act_mode': act_mode,
        }

        self.layer0 = dw_stack(block, filters[0], filters[0], kernel_size=ks[0], stride=1,
                               repeats=blocks[0], isotropic=isotropy[0], shared=shared_kwargs)
        self.layer1 = dw_stack(block, filters[0], filters[1], kernel_size=ks[1], stride=2,
                               repeats=blocks[1], isotropic=isotropy[1], shared=shared_kwargs)
        self.layer2 = dw_stack(block, filters[1], filters[2], kernel_size=ks[2], stride=2,
                               repeats=blocks[2], isotropic=isotropy[2], shared=shared_kwargs)
        self.layer3 = dw_stack(block, filters[2], filters[3], kernel_size=ks[3], stride=(1, 2, 2),
                               repeats=blocks[3], isotropic=isotropy[3], shared=shared_kwargs)
        self.layer4 = dw_stack(block, filters[3], filters[4], kernel_size=ks[4], stride=2,
                               repeats=blocks[4], isotropic=isotropy[4], shared=shared_kwargs)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def dw_stack(block, in_ch, out_ch, kernel_size, stride,
             repeats, isotropic, shared):
    """ Creates a stack of inverted residual blocks. 
    """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = block(in_ch, out_ch, kernel_size, stride,
                  isotropic=isotropic, **shared)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            block(out_ch, out_ch, kernel_size, 1,
                  isotropic=isotropic, **shared))
    return nn.Sequential(first, *remaining)
