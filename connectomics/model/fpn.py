from __future__ import print_function, division
from typing import Optional, List

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .block import conv3d_norm_act
from .utils import model_init


class FPN3D(nn.Module):
    """3D feature pyramid network (FPN). This design is flexible in handling both isotropic data and anisotropic data.

    Args:
        backbone_type (str): the block type at each U-Net stage. Default: ``'resnet'``
        block_type (str): the block type in the backbone. Default: ``'residual'``
        in_channel (int): number of input channels. Default: 1
        out_channel (int): number of output channels. Default: 3
        filters (List[int]): number of filters at each FPN stage. Default: [28, 36, 48, 64, 80]
        is_isotropic (bool): whether the whole model is isotropic. Default: False
        isotropy (List[bool]): specify each U-Net stage is isotropic or anisotropic. All elements will
            be `True` if :attr:`is_isotropic` is `True`. Default: [False, False, True, True, True]
        pad_mode (str): one of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'replicate'``
        act_mode (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``, 
            ``'swish'``, ``'efficient_swish'`` or ``'none'``. Default: ``'relu'``
        norm_mode (str): one of ``'bn'``, ``'sync_bn'`` ``'in'`` or ``'gn'``. Default: ``'bn'``
        init_mode (str): one of ``'xavier'``, ``'kaiming'``, ``'selu'`` or ``'orthogonal'``. Default: ``'orthogonal'``
        deploy (bool): build backbone in deploy mode (exclusive for RepVGG backbone). Default: False
    """

    def __init__(self,
                 backbone_type: str = 'resnet',
                 block_type: str = 'residual',
                 feature_keys: List[str] = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5'],
                 in_channel: int = 1,
                 out_channel: int = 3,
                 filters: List[int] = [28, 36, 48, 64, 80],
                 is_isotropic: bool = False,
                 isotropy: List[bool] = [False, False, False, True, True],
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 init_mode: str = 'orthogonal',
                 deploy: bool = False,
                 fmap_size=[17, 129, 129],
                 **kwargs):
        super().__init__()
        self.depth = len(filters)
        assert len(isotropy) == self.depth
        if is_isotropic:
            isotropy = [True] * self.depth

        shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode
        }

        backbone_kwargs = {
            'block_type': block_type,
            'in_channel': in_channel,
            'filters': filters,
            'isotropy': isotropy,
            'deploy': deploy,
            'fmap_size': fmap_size,
        }
        backbone_kwargs.update(shared_kwargs)

        self.backbone = build_backbone(
            backbone_type, feature_keys, **backbone_kwargs)
        self.feature_keys = feature_keys

        latplanes = filters[0]
        self.latlayers = nn.ModuleList(
            [conv3d_norm_act(x, latplanes, kernel_size=1,
                             padding=0, **shared_kwargs) for x in filters])

        self.smooth = nn.ModuleList()
        for i in range(self.depth):
            kernel_size, padding = self._get_kernel_size(isotropy[i])
            self.smooth.append(conv3d_norm_act(latplanes, latplanes,
                                               kernel_size=kernel_size, padding=padding, **shared_kwargs))

        kernel_size_io, padding_io = self._get_kernel_size(
            is_isotropic, io_layer=True)
        self.conv_out = conv3d_norm_act(filters[0], out_channel, kernel_size_io,
                                        padding=padding_io, pad_mode=pad_mode, bias=True,
                                        act_mode='none', norm_mode='none')

        # initialization
        model_init(self, init_mode)

    def forward(self, x):
        z = self.backbone(x)
        features = [self.latlayers[i](z[self.feature_keys[i]])
                    for i in range(self.depth)]

        out = features[self.depth-1]
        for j in range(self.depth-1):
            i = self.depth-1-j
            out = self._up_smooth_add(out, features[i-1], self.smooth[i])
        out = self.smooth[0](out)
        return self.conv_out(out)

    def _up_smooth_add(self, x, y, smooth):
        """Upsample, smooth and add two feature maps.
        """
        x = F.interpolate(x, size=y.shape[2:], mode='trilinear',
                          align_corners=True)
        return smooth(x) + y

    def _get_kernel_size(self, is_isotropic, io_layer=False):
        if io_layer:  # kernel and padding size of I/O layers
            if is_isotropic:
                return (5, 5, 5), (2, 2, 2)
            return (1, 5, 5), (0, 2, 2)

        if is_isotropic:
            return (3, 3, 3), (1, 1, 1)
        return (1, 3, 3), (0, 1, 1)
