# Adapted from https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
from typing import Union, List

import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

filt_dict = {
    1: np.array([1., ]),
    2: np.array([1., 1.]),
    3: np.array([1., 2., 1.]),
    4: np.array([1., 3., 3., 1.]),
    5: np.array([1., 4., 6., 4., 1.]),
    6: np.array([1., 5., 10., 10., 5., 1.]),
    7: np.array([1., 6., 15., 20., 15., 6., 1.]),
}


class BlurPool1D(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(BlurPool1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2),
                          int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        a = filt_dict[self.filt_size]
        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer(
            'filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class BlurPool2D(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool2D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)),
                          int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        a = filt_dict[self.filt_size]
        filt = torch.Tensor(a[:, None]*a[None, :])
        filt = filt/torch.sum(filt)
        self.register_buffer(
            'filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer_2d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class BlurPool3D(nn.Module):
    def __init__(self,
                 channels: int,
                 pad_type: str = 'zero',
                 filt_size: Union[int, List[int]] = 3,
                 stride: Union[int, List[int]] = 2,
                 pad_off=0):
        super(BlurPool3D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.stride = stride
        self.channels = channels

        if isinstance(self.filt_size, int):
            a = filt_dict[self.filt_size]
            filt = torch.Tensor(a[:, None, None] *
                                a[None, :, None] *
                                a[None, None, :])
            self.pad_sizes = [
                int(1. * (filt_size - 1) / 2),
                int(np.ceil(1. * (filt_size - 1) / 2))] * 3
        else:  # different filter size for z, y, x
            assert len(self.filt_size) == 3
            z = filt_dict[self.filt_size[0]]
            y = filt_dict[self.filt_size[1]]
            x = filt_dict[self.filt_size[2]]
            filt = torch.Tensor(z[:, None, None] *
                                y[None, :, None] *
                                x[None, None, :])
            self.pad_sizes = []
            for i in range(3):
                self.pad_sizes += [
                    int(1. * (filt_size[i] - 1) / 2),
                    int(np.ceil(1. * (filt_size[i] - 1) / 2))]

        filt = filt / torch.sum(filt)  # normalize
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]

        self.register_buffer(
            'filt', filt[None, None, :, :, :].repeat((channels, 1, 1, 1, 1)))
        self.pad = get_pad_layer_3d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off != 0:
                inp = self.pad(inp)
            return inp[:, :, ::self.stride[0], ::self.stride[1], ::self.stride[2]]

        return F.conv3d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class ZeroPad1d(torch.nn.modules.padding.ConstantPad1d):
    def __init__(self, padding):
        super(ZeroPad1d, self).__init__(padding, 0.)


class ZeroPad3d(torch.nn.modules.padding.ConstantPad3d):
    def __init__(self, padding):
        super(ZeroPad3d, self).__init__(padding, 0.)


def get_pad_layer_3d(pad_type):
    PadLayer = None
    if pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad3d
    elif pad_type == 'zero':
        PadLayer = ZeroPad3d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


def get_pad_layer_2d(pad_type):
    PadLayer = None
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


def get_pad_layer_1d(pad_type):
    PadLayer = None
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer
