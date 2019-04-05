import os,sys
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from vcg_connectomics.libs.sync import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

# 2D basic blocks
def conv2d_pad(in_planes, out_planes, kernel_size=(3, 3), stride=1,
               dilation=(1, 1), padding=(1, 1), bias=False):
    # the size of the padding should be a 6-tuple
    padding = tuple([x for x in padding for _ in range(2)][::-1])
    return nn.Sequential(
        nn.ReplicationPad2d(padding),
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=0, dilation=dilation, bias=bias))

def conv2d_bn_non(in_planes, out_planes, kernel_size=(3, 3), stride=1,
                  dilation=(1, 1), padding=(1, 1), bias=False):
    return nn.Sequential(
        conv2d_pad(in_planes, out_planes, kernel_size, stride,
                   dilation, padding, bias),
        SynchronizedBatchNorm2d(out_planes))

def conv2d_bn_elu(in_planes, out_planes, kernel_size=(3, 3), stride=1,
                  dilation=(1, 1), padding=(1, 1), bias=False):
    return nn.Sequential(
        conv2d_pad(in_planes, out_planes, kernel_size, stride,
                   dilation, padding, bias),
        SynchronizedBatchNorm2d(out_planes),
        nn.ELU(inplace=True))

# 3D basic blocks
def conv3d_pad(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
               dilation=(1,1,1), padding=(1,1,1), bias=False):
    # the size of the padding should be a 6-tuple        
    padding = tuple([x for x in padding for _ in range(2)][::-1])
    return  nn.Sequential(
                nn.ReplicationPad3d(padding),
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=0, dilation=dilation, bias=bias))     

def conv3d_bn_non(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
                  dilation=(1,1,1), padding=(1,1,1), bias=False):
    return nn.Sequential(
            conv3d_pad(in_planes, out_planes, kernel_size, stride, 
                       dilation, padding, bias),
            SynchronizedBatchNorm3d(out_planes))              

def conv3d_bn_elu(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
                  dilation=(1,1,1), padding=(1,1,1), bias=False):
    return nn.Sequential(
            conv3d_pad(in_planes, out_planes, kernel_size, stride, 
                       dilation, padding, bias),
            SynchronizedBatchNorm3d(out_planes),
            nn.ELU(inplace=True))