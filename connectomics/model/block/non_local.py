# Adapted from https://github.com/AlexHex7/Non-local_pytorch
import torch
from torch import nn
from torch.nn import functional as F
from ..utils import get_norm_1d, get_norm_2d, get_norm_3d

__all__ = [
    'NonLocalBlock1D',
    'NonLocalBlock2D',
    'NonLocalBlock3D',
]


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3,
                 sub_sample=True, norm_layer=True, norm_mode='bn'):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=3, stride=2)
            get_norm_func = get_norm_3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=3, stride=2)
            get_norm_func = get_norm_2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=3, stride=2)
            get_norm_func = get_norm_1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if norm_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                get_norm_func(norm_mode, self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NonLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None,
                 sub_sample=True, norm_layer=True, norm_mode='bn'):
        super(NonLocalBlock1D, self).__init__(in_channels, inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              norm_layer=norm_layer, norm_mode=norm_mode)


class NonLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None,
                 sub_sample=True, norm_layer=True, norm_mode='bn'):
        super(NonLocalBlock2D, self).__init__(in_channels, inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              norm_layer=norm_layer, norm_mode=norm_mode)


class NonLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None,
                 sub_sample=True, norm_layer=True, norm_mode='bn'):
        super(NonLocalBlock3D, self).__init__(in_channels, inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              norm_layer=norm_layer, norm_mode=norm_mode)
