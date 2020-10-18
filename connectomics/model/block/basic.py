import os,sys
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from ..norm import *

# common layers
def get_functional_act(mode='relu'):
    activation_dict = {
        'relu': F.relu_,
        'tanh': torch.tanh,
        'elu': F.elu_,
        'sigmoid': torch.sigmoid,
        'softmax': lambda x: F.softmax(x, dim=1),
        'none': lambda x: x,
    }
    return activation_dict[mode]

def get_layer_act(mode=''):
    if mode == '':
        return []
    elif mode == 'relu':
        return [nn.ReLU(inplace=True)]
    elif mode == 'elu':
        return [nn.ELU(inplace=True)]
    elif mode[:5] == 'leaky':
        # 'leaky0.2' 
        return [nn.LeakyReLU(inplace=True, negative_slope=float(mode[5:]))]
    raise ValueError('Unknown activation layer option {}'.format(mode))

def get_layer_norm(out_planes, norm_mode='', dim=2):
    if norm_mode=='':
        return []
    elif norm_mode=='bn':
        if dim==1:
            return [SynchronizedBatchNorm1d(out_planes)]
        elif dim==2:
            return [SynchronizedBatchNorm2d(out_planes)]
        elif dim==3:
            return [SynchronizedBatchNorm3d(out_planes)]
    elif norm_mode=='abn':
        if dim==1:
            return [nn.BatchNorm1d(out_planes)]
        elif dim==2:
            return [nn.BatchNorm2d(out_planes)]
        elif dim==3:
            return [nn.BatchNorm3d(out_planes)]
    elif norm_mode=='in':
        if dim==1:
            return [nn.InstanceNorm1d(out_planes)]
        elif dim==2:
            return [nn.InstanceNorm2d(out_planes)]
        elif dim==3:
            return [nn.InstanceNorm3d(out_planes)]
    elif norm_mode=='bin':
        if dim==1:
            return [BatchInstanceNorm1d(out_planes)]
        elif dim==2:
            return [BatchInstanceNorm2d(out_planes)]
        elif dim==3:
            return [BatchInstanceNorm3d(out_planes)]
    raise ValueError('Unknown normalization norm option {}'.format(mode))


# conv basic blocks
def conv2d_norm_act(in_planes, out_planes, kernel_size=(3,3), stride=1, 
                  dilation=(1,1), padding=(1,1), bias=True, pad_mode='rep', norm_mode='', act_mode='', return_list=False):

    if isinstance(padding,int):
        pad_mode = pad_mode if padding!=0 else 'zeros'
    else:
        pad_mode = pad_mode if max(padding)!=0 else 'zeros'

    if pad_mode in ['zeros','circular']:
        layers = [nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=padding, padding_mode=pad_mode, dilation=dilation, bias=bias)] 
    elif pad_mode=='rep':
        # the size of the padding should be a 6-tuple        
        padding = tuple([x for x in padding for _ in range(2)][::-1])
        layers = [nn.ReplicationPad2d(padding),
                  nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=0, dilation=dilation, bias=bias)]
    else:
        raise ValueError('Unknown padding option {}'.format(mode))
    layers += get_layer_norm(out_planes, norm_mode)
    layers += get_layer_act(act_mode)
    if return_list:
        return layers
    else:
        return nn.Sequential(*layers)

def conv3d_norm_act(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
                  dilation=(1,1,1), padding=(1,1,1), bias=True, pad_mode='rep', norm_mode='', act_mode='', return_list=False):
    
    if isinstance(padding,int):
        pad_mode = pad_mode if padding!=0 else 'zeros'
    else:
        pad_mode = pad_mode if max(padding)!=0 else 'zeros'

    if pad_mode in ['zeros','circular']:
        layers = [nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=padding, padding_mode=pad_mode, dilation=dilation, bias=bias)] 
    elif pad_mode=='rep':
        # the size of the padding should be a 6-tuple        
        padding = tuple([x for x in padding for _ in range(2)][::-1])
        layers = [nn.ReplicationPad3d(padding),
                  nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=0, dilation=dilation, bias=bias)]
    else:
        raise ValueError('Unknown padding option {}'.format(mode))

    layers += get_layer_norm(out_planes, norm_mode, 3)
    layers += get_layer_act(act_mode)
    if return_list:
        return layers
    else:
        return nn.Sequential(*layers)
