from __future__ import print_function, division
from typing import Optional, List

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from ..block import *
from ..utils import model_init


class UNet3D(nn.Module):
    """3D residual U-Net architecture. This design is flexible in handling both isotropic data and anisotropic data.
    Args:
        block_type (str): the block type at each U-Net stage. Default: ``'residual'``
        in_channel (int): number of input channels. Default: 1
        out_channel (int): number of output channels. Default: 3
        filters (List[int]): number of filters at each U-Net stage. Default: [28, 36, 48, 64, 80]
        is_isotropic (bool): whether the whole model is isotropic. Default: False
        isotropy (List[bool]): specify each U-Net stage is isotropic or anisotropic. All elements will
            be `True` if :attr:`is_isotropic` is `True`. Default: [False, False, False, True, True]
        pad_mode (str): one of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'replicate'``
        act_mode (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``, 
            ``'swish'``, ``'efficient_swish'`` or ``'none'``. Default: ``'relu'``
        norm_mode (str): one of ``'bn'``, ``'sync_bn'`` ``'in'`` or ``'gn'``. Default: ``'bn'``
        init_mode (str): one of ``'xavier'``, ``'kaiming'``, ``'selu'`` or ``'orthogonal'``. Default: ``'orthogonal'``
        pooling (bool): downsample by max-pooling if `True` else using stride. Default: `False`
    """

    block_dict = {
        'residual': BasicBlock3d,
        'residual_pa': BasicBlock3dPA,
        'residual_se': BasicBlock3dSE,
        'residual_se_pa': BasicBlock3dPASE,
    }

    def __init__(self,
                 block_type='residual',
                 in_channel: int = 1,
                 out_channel: int = 3,
                 filters: List[int] = [28, 36, 48, 64, 80],
                 is_isotropic: bool = False,
                 isotropy: List[bool] = [False, False, False, True, True],
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 init_mode: str = 'orthogonal',
                 pooling: bool = False,
                 **kwargs):
        super().__init__()
        assert len(filters) == len(isotropy)
        self.depth = len(filters)
        if is_isotropic:
            isotropy = [True] * self.depth

        self.pooling = pooling
        block = self.block_dict[block_type]

        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        # input and output layers
        kernel_size_io, padding_io = self._get_kernal_size(
            is_isotropic, io_layer=True)
        self.conv_in = conv3d_norm_act(in_channel, filters[0], kernel_size_io,
                                       padding=padding_io, **self.shared_kwargs)
        self.conv_out = conv3d_norm_act(filters[0], out_channel, kernel_size_io, bias=True,
                                        padding=padding_io, pad_mode=pad_mode, act_mode='none', norm_mode='none')

        # encoding path
        self.down_layers = nn.ModuleList()
        for i in range(self.depth):
            kernel_size, padding = self._get_kernal_size(isotropy[i])
            previous = max(0, i-1)
            stride = self._get_stride(isotropy[i], previous, i)
            layer = nn.Sequential(
                self._make_pooling_layer(isotropy[i], previous, i),
                conv3d_norm_act(filters[previous], filters[i], kernel_size,
                                stride=stride, padding=padding, **self.shared_kwargs),
                block(filters[i], filters[i], **self.shared_kwargs))
            self.down_layers.append(layer)

        # decoding path
        self.up_layers = nn.ModuleList()
        for j in range(1, self.depth):
            kernel_size, padding = self._get_kernal_size(isotropy[j])
            layer = nn.ModuleList([
                conv3d_norm_act(filters[j], filters[j-1], kernel_size,
                                padding=padding, **self.shared_kwargs),
                block(filters[j-1], filters[j-1], **self.shared_kwargs)])
            self.up_layers.append(layer)

        # initialization
        model_init(self, mode=init_mode)

    def forward(self, x):
        x = self.conv_in(x)

        down_x = [None] * (self.depth-1)
        for i in range(self.depth-1):
            x = self.down_layers[i](x)
            down_x[i] = x

        x = self.down_layers[-1](x)

        for j in range(self.depth-1):
            i = self.depth-2-j
            x = self.up_layers[i][0](x)
            x = self._upsample_add(x, down_x[i])
            x = self.up_layers[i][1](x)

        x = self.conv_out(x)
        return x

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        When pooling layer is used, the input size is assumed to be even, 
        therefore :attr:`align_corners` is set to `False` to avoid feature 
        mis-match. When downsampling by stride, the input size is assumed 
        to be 2n+1, and :attr:`align_corners` is set to `True`.
        """
        align_corners = False if self.pooling else True
        x = F.interpolate(x, size=y.shape[2:], mode='trilinear',
                          align_corners=align_corners)
        return x + y

    def _get_kernal_size(self, is_isotropic, io_layer=False):
        if io_layer:  # kernel and padding size of I/O layers
            if is_isotropic:
                return (5, 5, 5), (2, 2, 2)
            return (1, 5, 5), (0, 2, 2)

        if is_isotropic:
            return (3, 3, 3), (1, 1, 1)
        return (1, 3, 3), (0, 1, 1)

    def _get_stride(self, is_isotropic, previous, i):
        if self.pooling or previous == i:
            return 1

        return self._get_downsample(is_isotropic)

    def _get_downsample(self, is_isotropic):
        if not is_isotropic:
            return (1, 2, 2)
        return 2

    def _make_pooling_layer(self, is_isotropic, previous, i):
        if self.pooling and previous != i:
            kernel_size = stride = self._get_downsample(is_isotropic)
            return nn.MaxPool3d(kernel_size, stride)

        return nn.Identity()


class UNetPlus3D(UNet3D):
    def __init__(self,
                 filters: List[int] = [28, 36, 48, 64, 80],
                 norm_mode: str = 'bn',
                 **kwargs):

        super().__init__(filters=filters, norm_mode=norm_mode, **kwargs)
        self.feat_layers = nn.ModuleList(
            [conv3d_norm_act(filters[-1], filters[k-1], 1, **self.shared_kwargs)
             for k in range(1, self.depth)]
        )
        self.non_local = NonLocalBlock3D(
            filters[-1], sub_sample=False, norm_mode=norm_mode)

    def forward(self, x):
        x = self.conv_in(x)

        down_x = [None] * (self.depth-1)
        for i in range(self.depth-1):
            x = self.down_layers[i](x)
            down_x[i] = x

        x = self.down_layers[-1](x)
        x = self.non_local(x)
        feat = x  # lowest-res feature map

        for j in range(self.depth-1):
            i = self.depth-2-j
            x = self.up_layers[i][0](x)
            x = self._upsample_add(x, down_x[i])
            x = self._upsample_add(self.feat_layers[i](feat), x)
            x = self.up_layers[i][1](x)

        x = self.conv_out(x)
        return x


class UNet2D(nn.Module):
    """2D residual U-Net architecture.
    Args:
        block_type (str): the block type at each U-Net stage. Default: ``'residual'``
        in_channel (int): number of input channels. Default: 1
        out_channel (int): number of output channels. Default: 3
        filters (List[int]): number of filters at each U-Net stage. Default: [28, 36, 48, 64, 80]
        pad_mode (str): one of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'replicate'``
        act_mode (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``, 
            ``'swish'``, ``'efficient_swish'`` or ``'none'``. Default: ``'relu'``
        norm_mode (str): one of ``'bn'``, ``'sync_bn'`` ``'in'`` or ``'gn'``. Default: ``'bn'``
        init_mode (str): one of ``'xavier'``, ``'kaiming'``, ``'selu'`` or ``'orthogonal'``. Default: ``'orthogonal'``
        pooling (bool): downsample by max-pooling if `True` else using stride. Default: `False`
    """

    block_dict = {
        'residual': BasicBlock2d,
        'residual_se': BasicBlock2dSE,
    }

    def __init__(self,
                 block_type='residual',
                 in_channel: int = 1,
                 out_channel: int = 3,
                 filters: List[int] = [32, 64, 128, 256, 512],
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 init_mode: str = 'orthogonal',
                 pooling: bool = False,
                 **kwargs):
        super().__init__()
        self.depth = len(filters)
        self.pooling = pooling
        block = self.block_dict[block_type]

        shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        # input and output layers
        self.conv_in = conv2d_norm_act(
            in_channel, filters[0], 5, padding=2, **shared_kwargs)
        self.conv_out = conv2d_norm_act(filters[0], out_channel, 5, padding=2,
                                        bias=True, pad_mode=pad_mode, act_mode='none', norm_mode='none')

        # encoding path
        self.down_layers = nn.ModuleList()
        for i in range(self.depth):
            kernel_size, padding = 3, 1
            previous = max(0, i-1)
            stride = self._get_stride(previous, i)
            layer = nn.Sequential(
                self._make_pooling_layer(previous, i),
                conv2d_norm_act(filters[previous], filters[i], kernel_size,
                                stride=stride, padding=padding, **shared_kwargs),
                block(filters[i], filters[i], **shared_kwargs))
            self.down_layers.append(layer)

        # decoding path
        self.up_layers = nn.ModuleList()
        for j in range(1, self.depth):
            kernel_size, padding = 3, 1
            layer = nn.ModuleList([
                conv2d_norm_act(filters[j], filters[j-1], kernel_size,
                                padding=padding, **shared_kwargs),
                block(filters[j-1], filters[j-1], **shared_kwargs)])
            self.up_layers.append(layer)

        # initialization
        model_init(self, mode=init_mode)

    def forward(self, x):
        x = self.conv_in(x)

        down_x = [None] * (self.depth-1)
        for i in range(self.depth-1):
            x = self.down_layers[i](x)
            down_x[i] = x

        x = self.down_layers[-1](x)

        for j in range(self.depth-1):
            i = self.depth-2-j
            x = self.up_layers[i][0](x)
            x = self._upsample_add(x, down_x[i])
            x = self.up_layers[i][1](x)

        x = self.conv_out(x)
        return x

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        When pooling layer is used, the input size is assumed to be even, 
        therefore :attr:`align_corners` is set to `False` to avoid feature 
        mis-match. When downsampling by stride, the input size is assumed 
        to be 2n+1, and :attr:`align_corners` is set to `False`.
        """
        align_corners = False if self.pooling else True
        x = F.interpolate(x, size=y.shape[2:], mode='bilinear',
                          align_corners=align_corners)
        return x + y

    def _get_stride(self, previous, i):
        if self.pooling or previous == i:
            return 1
        return 2

    def _make_pooling_layer(self, previous, i):
        if self.pooling and previous != i:
            kernel_size = stride = 2
            return nn.MaxPool2d(kernel_size, stride)

        return nn.Identity()
