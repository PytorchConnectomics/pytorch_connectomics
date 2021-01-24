from __future__ import print_function, division
from typing import Optional, List

import torch
import torch.nn as nn
from .resnet import ResNet3D
from ..utils.misc import IntermediateLayerGetter

def build_backbone(backbone_type: str, 
                   feat_keys: List[str],
                   **kwargs):
    if backbone_type == 'resnet':
        backbone = ResNet3D(**kwargs)
        assert len(feat_keys) == backbone.num_stages
        return_layers = { 'conv' : feat_keys[0],
                         'layer1': feat_keys[1], 
                         'layer2': feat_keys[2],
                         'layer3': feat_keys[3],
                         'layer4': feat_keys[4]}
    return IntermediateLayerGetter(backbone, return_layers)
