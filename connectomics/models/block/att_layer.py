import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_activation


def make_att_3d(attention, channel, act_mode='relu'):
    if attention == 'strip_pool':
        return StripPoolingAttention3D(channel, act_mode)
    elif attention == 'plane_pool':
        return PlanePoolingAttention3D(channel, act_mode)
    elif attention == 'squeeze_excitation':
        return SELayer3d(channel, reduction=8, act_mode=act_mode)
    else:
        return nn.Identity()

# ------------------------------
# Squeeze-and-Excitation Layers
# ------------------------------


class SELayer2d(nn.Module):
    def __init__(self, channel, reduction=16, act_mode='relu'):
        super(SELayer2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            get_activation(act_mode),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SELayer3d(nn.Module):
    def __init__(self, channel, reduction=4, act_mode='relu'):
        super(SELayer3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            get_activation(act_mode),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


# ---------------------------------------
# Strip-Pooling and Plane-Pooling Layers
# ---------------------------------------


class StripPoolingAttention3D(nn.Module):
    """
    """
    reduction = 4

    def __init__(self, channel, act_mode='relu'):
        super(StripPoolingAttention3D, self).__init__()

        self.channel = channel

        self.pool_z = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_y = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_x = nn.AdaptiveAvgPool3d((1, 1, None))

        self.conv_z = nn.Conv3d(
            channel, channel // self.reduction, (3, 1, 1), padding=(1, 0, 0))
        self.conv_y = nn.Conv3d(
            channel, channel // self.reduction, (1, 3, 1), padding=(0, 1, 0))
        self.conv_x = nn.Conv3d(
            channel, channel // self.reduction, (1, 1, 3), padding=(0, 0, 1))

        self.relu = get_activation(act_mode)
        self.conv1x1x1 = nn.Conv3d(
            channel // self.reduction, channel, 1, bias=False)

    def forward(self, x):
        _, _, l, h, w = x.size()

        x1 = self.conv_z(self.pool_z(x)).expand(-1, -1, l, h, w)
        x2 = self.conv_y(self.pool_y(x)).expand(-1, -1, l, h, w)
        x3 = self.conv_x(self.pool_x(x)).expand(-1, -1, l, h, w)

        # scale the input tensor
        # There is a weird behavior: fusion = self.relu(x1 + x2 + x3) causes
        # RuntimeError: CUDA error: an illegal memory access was encountered.
        fusion = self.relu(x1) + self.relu(x2) + self.relu(x3)
        fusion = self.conv1x1x1(fusion).sigmoid()
        return x * fusion


class PlanePoolingAttention3D(nn.Module):
    """
    """
    reduction = 4

    def __init__(self, channel, act_mode='relu'):
        super(PlanePoolingAttention3D, self).__init__()

        self.channel = channel

        self.pool_zy = nn.AdaptiveAvgPool3d((None, None, 1))
        self.pool_yx = nn.AdaptiveAvgPool3d((1, None, None))
        self.pool_xz = nn.AdaptiveAvgPool3d((None, 1, None))

        self.conv_zy = nn.Conv3d(
            channel, channel // self.reduction, (3, 3, 1), padding=(1, 1, 0))
        self.conv_yx = nn.Conv3d(
            channel, channel // self.reduction, (1, 3, 3), padding=(0, 1, 1))
        self.conv_xz = nn.Conv3d(
            channel, channel // self.reduction, (3, 1, 3), padding=(1, 0, 1))

        self.relu = get_activation(act_mode)
        self.conv1x1x1 = nn.Conv3d(
            channel // self.reduction, channel, 1, bias=False)

    def forward(self, x):
        _, _, l, h, w = x.size()

        x1 = self.conv_zy(self.pool_zy(x)).expand(-1, -1, l, h, w)
        x2 = self.conv_yx(self.pool_yx(x)).expand(-1, -1, l, h, w)
        x3 = self.conv_xz(self.pool_xz(x)).expand(-1, -1, l, h, w)

        # scale the input tensor
        # There is a weird behavior: fusion = self.relu(x1 + x2 + x3) causes
        # RuntimeError: CUDA error: an illegal memory access was encountered.
        fusion = self.relu(x1) + self.relu(x2) + self.relu(x3)
        fusion = self.conv1x1x1(fusion).sigmoid()
        return x * fusion
