# 2D DeepLabV3 model in PyTorch, adapted from
# https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
from __future__ import print_function, division
from typing import Type, Any, Callable, Union, List, Optional

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..utils.misc import get_norm_2d, get_activation
from ..utils.misc import IntermediateLayerGetter
from ..backbone import resnet


class DeepLabV3(nn.Module):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_. This implementation only
    supports 2D inputs. Pretrained ResNet weights on the ImgeaNet
    dataset is loaded by default. 

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    def __init__(self,
                 name: str,
                 backbone_type: str,
                 out_channel: int = 1,
                 aux_out: bool = False,
                 **kwargs):
        super().__init__()
        assert name in ['deeplabv3a', 'deeplabv3b', 'deeplabv3c']
        # 1. build resnet backbone (also load pretrained weights)
        backbone = resnet.__dict__[backbone_type](
            pretrained=True,
            replace_stride_with_dilation=[False, True, True],
            **kwargs)

        return_layers = {'layer4': 'out'}
        if aux_out:
            return_layers['layer3'] = 'aux'
        if name == 'deeplabv3c':
            return_layers['layer1'] = 'low_level_feat'
        self.backbone = IntermediateLayerGetter(backbone, return_layers)

        # 2. build auxiliary classifier (optional)
        self.aux_classifier = None
        if aux_out:
            inplanes = 1024
            self.aux_classifier = FCNHead(1024, out_channel, **kwargs)

        # 3. build deeplab classifier
        head_map = {
            'deeplabv3a': DeepLabHeadA,
            'deeplabv3b': DeepLabHeadB,
            'deeplabv3c': DeepLabHeadC,
        }
        inplanes = 2048
        self.classifier = head_map[name](inplanes, out_channel, **kwargs)

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        if "low_level_feat" in features.keys():
            feat = features["low_level_feat"]
            x = self.classifier(x, feat)
        else:
            x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear',
                          align_corners=True)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear',
                              align_corners=True)
            result["aux"] = x

        return result

# ---------------------------
# DeepLab Heads
# ---------------------------


class DeepLabHeadA(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 **_):
        conv3x3 = nn.Conv2d(256, 256, 3, padding=1,
                            padding_mode=pad_mode, bias=False)
        super(DeepLabHeadA, self).__init__(
            ASPP(in_channels, [12, 24, 36], 256,
                 pad_mode, act_mode, norm_mode),
            conv3x3,
            get_norm_2d(norm_mode, 256),
            get_activation(act_mode),
            nn.Conv2d(256, num_classes, 1)
        )


class DeepLabHeadB(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 **_):
        super(DeepLabHeadB, self).__init__()
        self.aspp = ASPP(in_channels, [12, 24, 36], 256,
                         pad_mode, act_mode, norm_mode)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1,
                      padding_mode=pad_mode, bias=False),
            get_norm_2d(norm_mode, 128),
            get_activation(act_mode)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1,
                      padding_mode=pad_mode, bias=False),
            get_norm_2d(norm_mode, 128),
            get_activation(act_mode),
            nn.Conv2d(128, num_classes, 3, padding=1,
                      padding_mode=pad_mode)
        )

    def forward(self, x):
        x = self.aspp(x)
        H, W = self._interp_shape(x)

        x = self.conv1(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear',
                          align_corners=True)
        x = self.conv2(x)
        return x

    def _interp_shape(self, x):
        H, W = x.shape[-2:]
        H = 2*H-1 if H % 2 == 1 else 2*H
        W = 2*W-1 if W % 2 == 1 else 2*W
        return H, W


class DeepLabHeadC(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 **_):
        super(DeepLabHeadC, self).__init__()
        self.aspp = ASPP(in_channels, [12, 24, 36], 256,
                         pad_mode, act_mode, norm_mode)
        self.conv = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            get_norm_2d(norm_mode, 32),
            get_activation(act_mode),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(288, 256, 3, padding=1,
                      padding_mode=pad_mode, bias=False),
            get_norm_2d(norm_mode, 256),
            get_activation(act_mode),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x, low_level_feat):
        feat_shape = low_level_feat.shape[-2:]
        x = self.aspp(x)
        x = F.interpolate(x, size=feat_shape, mode='bilinear',
                          align_corners=True)
        low_level_feat = self.conv(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.classifier(x)
        return x

# ---------------------------
# ASPP Modules
# ---------------------------


class ASPPConv(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dilation: int,
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn'):
        conv3x3 = nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                            dilation=dilation, padding_mode=pad_mode, bias=False)
        modules = [
            conv3x3,
            get_norm_2d(norm_mode, out_channels),
            get_activation(act_mode),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn'):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            get_norm_2d(norm_mode, out_channels),
            get_activation(act_mode),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear',
                             align_corners=False)


class ASPP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 atrous_rates: List[int],
                 out_channels: int = 256,
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn'):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            get_norm_2d(norm_mode, out_channels),
            get_activation(act_mode)))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, pad_mode=pad_mode,
                                    act_mode=act_mode, norm_mode=norm_mode))

        modules.append(ASPPPooling(in_channels, out_channels,
                                   act_mode=act_mode, norm_mode=norm_mode))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            get_norm_2d(norm_mode, out_channels),
            get_activation(act_mode))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

# ---------------------------
# FCN (auxiliary classifier)
# ---------------------------


class FCNHead(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 **_):
        inter_channels = in_channels // 4
        conv3x3 = nn.Conv2d(in_channels, inter_channels, 3, padding=1,
                            padding_mode=pad_mode, bias=False)
        layers = [
            conv3x3,
            get_norm_2d(norm_mode, inter_channels),
            get_activation(act_mode),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)
