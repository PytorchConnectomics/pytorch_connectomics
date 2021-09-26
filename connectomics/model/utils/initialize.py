import torch
import torch.nn as nn
from math import sqrt

def model_init(model, mode='orthogonal'):
    """Initialization of model weights.
    """
    model_init_dict = {
        'xavier': xavier_init,
        'kaiming': kaiming_init,
        'selu': selu_init,
        'orthogonal': ortho_init,
    }
    # Applies fn recursively to every submodule (as returned by .children()) as well 
    # as self. See https://pytorch.org/docs/stable/generated/torch.nn.Module.html.
    model.apply(model_init_dict[mode])

def xavier_init(model):
    # sxavier initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform_(
                m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def kaiming_init(model):
    # he initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in')

def selu_init(model):
    # selu init
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            nn.init.normal(m.weight, 0, sqrt(1. / fan_in))
        elif isinstance(m, nn.Linear):
            fan_in = m.in_features
            nn.init.normal(m.weight, 0, sqrt(1. / fan_in))

def ortho_init(model):
    # orthogonal initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
