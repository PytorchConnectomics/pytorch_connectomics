import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_norm_2d, get_norm_3d, get_activation

def conv2d_norm_act(in_planes, planes, kernel_size=(3, 3), stride=1, groups=1,
                    dilation=(1, 1), padding=(1, 1), bias=False, pad_mode='replicate', 
                    norm_mode='bn', act_mode='relu', return_list=False):

    layers = [nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, 
                        groups=groups, dilation=dilation, padding=padding, 
                        padding_mode=pad_mode, bias=bias)] 
    layers += [get_norm_2d(norm_mode, planes)]
    layers += [get_activation(act_mode)]

    if return_list: # return a list of layers
        return layers

    return nn.Sequential(*layers)

def conv3d_norm_act(in_planes, planes, kernel_size=(3,3,3), stride=1, groups=1,
                    dilation=(1,1,1), padding=(1,1,1), bias=False, pad_mode='replicate', 
                    norm_mode='bn', act_mode='relu', return_list=False):

    layers = [nn.Conv3d(in_planes, planes, kernel_size=kernel_size, stride=stride, 
                        groups=groups, padding=padding, padding_mode=pad_mode, 
                        dilation=dilation, bias=bias)] 
    layers += [get_norm_3d(norm_mode, planes)]
    layers += [get_activation(act_mode)]

    if return_list: # return a list of layers
        return layers

    return nn.Sequential(*layers)
