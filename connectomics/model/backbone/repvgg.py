from __future__ import print_function, division
from typing import Optional, Union, List

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_activation

def conv_bn_2d(in_planes, planes, kernel_size, stride, padding, groups=1, 
               dilation=1, pad_mode='zeros'):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_planes, planes, kernel_size, stride=stride, 
                                        padding=padding, padding_mode=pad_mode,
                                        groups=groups, dilation=dilation, bias=False))
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
        padding = dilation

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_planes, planes, kernel_size, stride=stride, padding=padding, 
                                         groups=groups, dilation=dilation, padding_mode=pad_mode, bias=True)

        else:
            self.rbr_identity = nn.BatchNorm2d(in_planes) if planes == in_planes and stride == 1 else None
            self.rbr_dense = conv_bn_2d(in_planes, planes, kernel_size, stride, padding, groups=groups, 
                                        dilation=dilation, pad_mode=pad_mode)
            self.rbr_1x1 = conv_bn_2d(in_planes, planes, 1, stride, padding=0, groups=groups)

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(x))

        identity = 0
        if self.rbr_identity is not None:
            identity = self.rbr_identity(x)

        x = self.rbr_dense(x) + self.rbr_1x1(x) + identity
        return self.act(x)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernel_id, bias3x3 + bias1x1 + bias_id

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
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


def conv_bn_3d(in_planes, planes, kernel_size, stride, padding, groups=1, 
               dilation=1, pad_mode='zeros'):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv3d(in_planes, planes, kernel_size, stride=stride, 
                                        padding=padding, padding_mode=pad_mode,
                                        groups=groups, dilation=dilation, bias=False))
    result.add_module('bn', nn.BatchNorm3d(planes))
    return result

class RepVGGBlock3D(nn.Module):
    """ 3D RepVGG Block, adapted from:
    https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, 
                 padding=1, dilation=1, groups=1, pad_mode='zeros', 
                 act_mode='relu', isotropic=False, deploy=False):
        super(RepVGGBlock3D, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_planes = in_planes
        self.act = get_activation(act_mode)

        self.isotropic = isotropic
        padding = dilation
        if not self.isotropic:
            dilation = (1, dilation, dilation)
            kernel_size, padding = (1, 3, 3), (0, padding, padding)

        if deploy:
            self.rbr_reparam = nn.Conv3d(in_planes, planes, kernel_size, stride=stride, padding=padding, 
                                         groups=groups, dilation=dilation, padding_mode=pad_mode, bias=True)

        else:
            self.rbr_identity = nn.BatchNorm3d(in_planes) if planes == in_planes and stride == 1 else None
            self.rbr_dense = conv_bn_3d(in_planes, planes, kernel_size, stride, padding, groups=groups, 
                                        dilation=dilation, pad_mode=pad_mode)
            self.rbr_1x1 = conv_bn_3d(in_planes, planes, 1, stride, padding=0, groups=groups)

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(x))

        identity = 0
        if self.rbr_identity is not None:
            identity = self.rbr_identity(x)

        x = self.rbr_dense(x) + self.rbr_1x1(x) + identity
        return self.act(x)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernel_id, bias3x3 + bias1x1 + bias_id

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        # The padding size by which to pad some dimensions of input are described 
        # starting from the last dimension and moving forward.
        pad_size = [1,1,1,1,1,1] if self.isotropic else [1,1,1,1,0,0]
        return torch.nn.functional.pad(kernel1x1, pad_size)

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm3d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_planes // self.groups
                z_dim = 3 if self.isotropic else 1
                z_idx = 1 if self.isotropic else 0
                kernel_value = torch.zeros((self.in_planes, input_dim, z_dim, 3, 3), 
                                           dtype=torch.float, device=branch.weight.device)
                for i in range(self.in_planes):
                    kernel_value[i, i % input_dim, z_idx, 1, 1] = 1.0
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel, bias

    def load_reparam_kernel(self, kernel, bias):
        assert hasattr(self, 'rbr_reparam')
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias


class RepVGG3D(nn.Module):
    """RepVGG backbone for 3D semantic/instance segmentation. 
       The global average pooling and fully-connected layer are removed.
    """
    num_stages = 5
    block = RepVGGBlock3D

    def __init__(self, 
                 in_channel: int = 1, 
                 filters: List[int] = [28, 36, 48, 64, 80],
                 blocks: List[int] = [4, 4, 4, 4],
                 isotropy: List[bool] = [False, False, False, True, True],
                 pad_mode: str = 'replicate', 
                 act_mode: str = 'elu', 
                 deploy: bool = False,
                 **_):
        super().__init__()
        assert len(filters) == self.num_stages

        self.shared_kwargs = {
            'deploy': deploy,
            'pad_mode': pad_mode,
            'act_mode': act_mode}

        self.layer0 = self._make_layer(in_channel, filters[0], 1, 1, isotropy[0])
        self.layer1 = self._make_layer(filters[0], filters[1], blocks[0], 2, isotropy[1])
        self.layer2 = self._make_layer(filters[1], filters[2], blocks[1], 2, isotropy[2])
        self.layer3 = self._make_layer(filters[2], filters[3], blocks[2], 2, isotropy[3])
        self.layer4 = self._make_layer(filters[3], filters[4], blocks[3], 2, isotropy[4])

    def _make_layer(self, in_planes: int, planes: int, blocks: int, 
                    stride: int = 1, isotropic: bool = False):
        if stride == 2 and not isotropic: stride = (1, 2, 2)
        layers = []
        layers.append(self.block(in_planes, planes, stride=stride, 
            isotropic=isotropic, **self.shared_kwargs))
        for _ in range(1, blocks):
            layers.append(self.block(planes, planes, stride=1, 
                isotropic=isotropic, **self.shared_kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
        
    def repvgg_convert_model(self):
        converted_weights = {}
        for name, module in self.named_modules():
            if hasattr(module, 'repvgg_convert'):
                kernel, bias = module.repvgg_convert()
                converted_weights[name + '.rbr_reparam.weight'] = kernel
                converted_weights[name + '.rbr_reparam.bias'] = bias
        return converted_weights

    def load_reparam_model(self, converted_weights):
        for name, param in self.named_parameters():
            if name in converted_weights.keys():
                param.data = converted_weights[name]

    @staticmethod
    def repvgg_convert_as_backbone(train_dict):
        # state_dict key format: backbone.layer0.0.rbr_dense.conv.weight
        deploy_dict = copy.deepcopy(train_dict)

        for name in train_dict.keys(): # keys will be deleted in deploy_dict
            name_split = name.split('.')
            if name in deploy_dict and name_split[0]=='backbone' and name_split[3]=='rbr_dense':
                sz = deploy_dict[name].shape
                in_planes, planes, isotropic = sz[1], sz[0], sz[2]==3
                repvgg_block = RepVGGBlock3D(in_planes, planes, isotropic=isotropic)

                prefix = ".".join(name_split[:3])
                temp_dict = {}
                for key in repvgg_block.state_dict().keys():
                    w_name = prefix + '.' + key
                    temp_dict[key] = deploy_dict[w_name]
                    del deploy_dict[w_name]

                repvgg_block.load_state_dict(temp_dict)
                kernel, bias = repvgg_block.repvgg_convert()
                deploy_dict[prefix + '.rbr_reparam.weight'] = kernel
                deploy_dict[prefix + '.rbr_reparam.bias'] = bias
        
        return deploy_dict
