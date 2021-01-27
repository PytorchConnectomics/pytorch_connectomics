from __future__ import print_function, division
from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_activation

def conv_bn_2d(in_planes, planes, kernel_size, stride, padding, groups=1, pad_mode='zeros'):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_planes, planes, kernel_size, stride=stride, 
                                        padding=padding, padding_mode=pad_mode,
                                        groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(planes))
    return result

class RepVGGBlock2D(nn.Module):
    """ 2D RepVGG Block, adapted from:
    https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, 
                 padding=1, dilation=1, groups=1, pad_mode='zeros', 
                 act_mode='relu', deploy=False):
        super(RepVGGBlock2D, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_planes = in_planes
        self.act = get_activation(act_mode)

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_planes, planes, kernel_size, stride=stride, padding=padding, 
                                         groups=groups, dilation=dilation, padding_mode=pad_mode, bias=True)

        else:
            self.rbr_identity = nn.BatchNorm2d(in_planes) if planes == in_planes and stride == 1 else nn.Identity()
            self.rbr_dense = conv_bn_2d(in_planes, planes, kernel_size, stride, padding, groups=groups, pad_mode=pad_mode)
            self.rbr_1x1 = conv_bn_2d(in_planes, planes, 1, stride, padding=0, groups=groups)

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(x))

        x = self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x)
        return self.act(x)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernel_id, bias3x3 + bias1x1 + bias_id

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if isinstance(branch, nn.Identity):
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_planes // self.groups
                kernel_value = torch.zeros((self.in_planes, input_dim, 3, 3), 
                                           dtype=torch.float, device=branch.weight.device)
                for i in range(self.in_planes):
                    kernel_value[i, i % input_dim, 1, 1] = 1.0
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel, bias

    def load_reparam_kernel(self, kernel, bias):
        assert hasattr(self, 'rbr_reparam')
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
