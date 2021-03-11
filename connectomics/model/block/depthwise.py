import torch
import torch.nn as nn
from ..utils import get_activation, get_norm_3d, get_conv
from .partialconv3d import PartialConv3d
from .pooling_attention import make_attention_module

def dwconv1xkxk(planes, kernel_size=3, stride=1, 
                dilation=1, conv_type='standard', padding_mode='zeros'):
    """1xkxk depthwise convolution with padding"""
    padding = (kernel_size - 1) * dilation // 2
    dilation = (1, dilation, dilation)
    padding = (0, padding, padding)
    stride = (1, stride, stride)
    return get_conv(conv_type)(
        planes,
        planes,
        kernel_size=(1, kernel_size, kernel_size),
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        groups=planes,
        bias=False,
        dilation=dilation)

def dwconvkxkxk(planes, kernel_size=3, stride=1, 
                dilation=1, conv_type='standard', padding_mode='zeros'):
    """kxkxk depthwise convolution with padding"""
    padding = (kernel_size - 1) * dilation // 2
    return get_conv(conv_type)(
        planes,
        planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        groups=planes,
        bias=False,
        dilation=dilation)

###########################
# Inverted Residual Blocks
###########################

class InvertedResidual(nn.Module):
    """3D Inverted Residual Block with Depth-wise Convolution"""
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 stride,
                 isotropic=True,
                 expansion_factor=1,
                 bn_momentum=0.1,
                 padding_mode='zeros',
                 norm_layer='bn',
                 attention=None,
                 conv_type='standard',
                 activation='relu'):
        super(InvertedResidual, self).__init__()

        assert stride in [1, 2, (1,2,2)]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)

        if isotropic:
            DWConv = dwconvkxkxk(mid_ch, kernel_size, stride, 
                conv_type=conv_type, padding_mode=padding_mode)
        else:
            DWConv = dwconv1xkxk(mid_ch, kernel_size, stride, 
                conv_type=conv_type, padding_mode=padding_mode)

        self.layers1 = nn.Sequential(
            # Pointwise
            nn.Conv3d(in_ch, mid_ch, 1, bias=False),
            get_norm_3d(norm_layer, mid_ch, bn_momentum),
            get_activation(activation),
            # Depthwise
            DWConv,
            get_norm_3d(norm_layer, mid_ch, bn_momentum),
            get_activation(activation))

        self.layers2 = nn.Sequential(
            # Linear pointwise. Note that there's no activation.
            nn.Conv3d(mid_ch, out_ch, 1, bias=False),
            get_norm_3d(norm_layer, out_ch, bn_momentum))

        if attention is not None:
            self.attention = make_attention_module(attention, mid_ch)
        else:
            self.attention = None

    def forward(self, x):
        identity = x

        out = self.layers1(x)
        if self.attention is not None:
            out = self.attention(out)
        out = self.layers2(out)

        if self.apply_residual:
            out += identity

        return out

class InvertedResidualDilated(nn.Module):
    """3D Inverted Residual Block with Dilated Depth-wise Convolution"""
    dilation_factors = [1, 2, 4, 8]

    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 stride,
                 isotropic=True,
                 expansion_factor=1,
                 bn_momentum=0.1,
                 padding_mode='zeros',
                 norm_layer=None,
                 attention=None,
                 conv_type='standard',
                 activation='relu'):
        super(InvertedResidualDilated, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        assert stride in [1, 2, (1,2,2)]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)

        # Depthwise convolution
        if isotropic:
            self.DWConv = nn.ModuleList([
                dwconvkxkxk(
                    mid_ch // len(self.dilation_factors),
                    kernel_size,
                    stride,
                    conv_type=conv_type,
                    padding_mode=padding_mode,
                    dilation=self.dilation_factors[i]) for i in range(4)
            ])
        else:
            self.DWConv = nn.ModuleList([
                dwconv1xkxk(
                    mid_ch // len(self.dilation_factors),
                    kernel_size,
                    stride,
                    conv_type=conv_type,
                    padding_mode=padding_mode,
                    dilation=self.dilation_factors[i]) for i in range(4)
            ])

        self.layers1_a = nn.Sequential(
            # Pointwise
            nn.Conv3d(in_ch, mid_ch, 1, bias=False),
            get_norm_3d(norm_layer, mid_ch, bn_momentum),
            get_activation(activation))

        self.layers1_b = nn.Sequential(
            get_norm_3d(norm_layer, mid_ch, bn_momentum),
            get_activation(activation))

        self.layers2 = nn.Sequential(
            # Linear pointwise. Note that there's no activation.
            nn.Conv3d(mid_ch, out_ch, 1, bias=False),
            get_norm_3d(norm_layer, out_ch, bn_momentum))

        if attention is not None:
            self.attention = make_attention_module(attention, mid_ch)
        else:
            self.attention = None

    def forward(self, x):
        identity = x

        out = self.layers1_a(x)
        out = self._split_conv_cat(out, self.DWConv)
        out = self.layers1_b(out)

        if self.attention is not None:
            out = self.attention(out)
        out = self.layers2(out)

        if self.apply_residual:
            out += identity

        return out

    def _split_conv_cat(self, x, conv_layers):
        _, c, _, _, _ = x.size()
        z = []
        y = torch.split(x, c // len(self.dilation_factors), dim=1)
        for i in range(len(self.dilation_factors)):
            z.append(conv_layers[i](y[i]))
        return torch.cat(z, dim=1)


def dw_stack(block, in_ch, out_ch, kernel_size, stride, repeats, isotropic, shared_args):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = block(in_ch, out_ch, kernel_size, stride, isotropic, **shared_args)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            block(out_ch, out_ch, kernel_size, 1, isotropic, **shared_args))
    return nn.Sequential(first, *remaining)