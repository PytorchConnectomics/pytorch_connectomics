import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_norm, get_activation

def conv2d_norm_act(in_planes, out_planes, kernel_size=(3, 3), stride=1, 
                    dilation=(1, 1), padding=(1, 1), bias=False, pad_mode='replicate', 
                    norm_mode='bn', act_mode='relu', return_list=False):

    layers = [nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                        stride=stride, padding=padding, padding_mode=pad_mode, 
                        dilation=dilation, bias=bias)] 
    layers += get_norm(norm_mode, out_planes)
    layers += get_activation(act_mode)

    if return_list:
        return layers 

    return nn.Sequential(*layers)

def conv3d_norm_act(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
                    dilation=(1,1,1), padding=(1,1,1), bias=False, pad_mode='replicate', 
                    norm_mode='bn', act_mode='relu', return_list=False):

    layers = [nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                        stride=stride, padding=padding, padding_mode=pad_mode, 
                        dilation=dilation, bias=bias)] 
    layers += get_norm(norm_mode, out_planes)
    layers += get_activation(act_mode)

    if return_list:
        return layers

    return nn.Sequential(*layers)
