from typing import Optional, List
import torch
import torch.nn as nn

from ..utils import *

class Discriminator3D(nn.Module):
    """3D PatchGAN discriminator

    Args:
        in_channel (int): number of input channels. Default: 1
        filters (List[int]): number of filters at each U-Net stage. Default: [32, 64, 96, 96, 96]
        pad_mode (str): one of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'replicate'`
        act_mode (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``,
            ``'swish'``, ``'efficient_swish'`` or ``'none'``. Default: ``'elu'``
        norm_mode (str): one of ``'bn'``, ``'sync_bn'`` ``'in'`` or ``'gn'``. Default: ``'in'``
        dilation (int): dilation rate of the conv kernels. Default: 1
        is_isotropic (bool): whether the whole model is isotropic. Default: False
        isotropy (List[bool]): specify each discriminator layer is isotropic or anisotropic. All elements will
            be `True` if :attr:`is_isotropic` is `True`. Default: [False, False, False, True, True]
        stride_list (List[int]): list of strides for each conv layer. Default: [2, 2, 2, 2, 1]
    """

    def __init__(self,
                 in_channel: int = 1,
                 filters: List[int] = [64, 64, 128, 128, 256],
                 pad_mode: str = 'replicate',
                 act_mode: str = 'leaky_relu',
                 norm_mode: str = 'in',
                 dilation: int = 1,
                 is_isotropic: bool = False,
                 isotropy: List[bool] = [False, False, False, True, True],
                 stride_list: List[int] = [2, 2, 2, 2, 1]
        ) -> None:
        super().__init__()
        self.depth = len(filters)
        if is_isotropic:
            isotropy = [True] * self.depth
        assert len(filters) == len(isotropy)

        for i in range(self.depth):
            if not isotropy[i] and stride_list[i] == 2:
                # do not downsample z axis
                stride_list[i] = (1,2,2)

        # no need to use bias as norm layers have affine parameters
        use_bias = True if norm_mode == 'none' else False

        dilation_base = dilation
        ks, padding, dilation = self._get_kernal_size(5, isotropy[0], dilation_base)
        sequence = [
            nn.Conv3d(in_channel, filters[0], kernel_size=ks, stride=stride_list[0],
                      padding=padding, padding_mode=pad_mode, dilation=dilation, bias=use_bias),
            get_norm_3d(norm_mode, filters[0]),
            get_activation(act_mode)]

        for n in range(1, self.depth):
            ks, padding, dilation = self._get_kernal_size(3, isotropy[n], dilation_base)
            sequence += [
                nn.Conv3d(filters[n-1], filters[n], kernel_size=ks, stride=stride_list[n],
                          padding=padding, padding_mode=pad_mode, dilation=dilation, bias=use_bias),
                get_norm_3d(norm_mode, filters[n]),
                get_activation(act_mode)]

        ks, padding, _ = self._get_kernal_size(3, True, 1)
        sequence += [nn.Conv3d(filters[-1], 1, kernel_size=ks, stride=1, padding=padding,
                               padding_mode=pad_mode, bias=True)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

    def _get_kernal_size(self, ks: int, is_isotropic: bool, dilation: int=1):
        assert ks >= 3
        padding = (ks + (ks-1) * (dilation-1)) // 2
        if is_isotropic:
            return ks, padding, dilation

        return (1, ks, ks), (0, padding, padding), (1, dilation, dilation)
