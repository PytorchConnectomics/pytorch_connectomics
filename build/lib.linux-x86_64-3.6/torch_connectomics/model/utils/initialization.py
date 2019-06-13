import torch
import torch.nn as nn
from math import sqrt

def xavier_init(model):
    # default xavier initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform(
                m.weight(), gain=nn.init.calculate_gain('relu'))

def he_init(model):
    # he initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal(m.weight, mode='fan_in')

def selu_init(model):
    # selu init
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            nn.init.normal(m.weight, 0, sqrt(1. / fan_in))
        elif isinstance(m, nn.Linear):
            fan_in = m.in_features
            nn.init.normal(m.weight, 0, sqrt(1. / fan_in))

def ortho_init(model):
    # orthogonal initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
