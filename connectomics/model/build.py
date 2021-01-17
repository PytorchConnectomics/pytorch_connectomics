import torch
import torch.nn as nn
import numpy as np

from .zoo import *
from .norm import patch_replication_callback

def build_model(cfg, device):
    MODEL_MAP = {'unet_residual_3d': unet_residual_3d,
                 'unet_residual_2d': unet_residual_2d,
                 'fpn': fpn,
                 'super':SuperResolution,
                 'unet_super':Unet_super}

    assert cfg.MODEL.ARCHITECTURE in MODEL_MAP.keys()
    if cfg.MODEL.ARCHITECTURE == 'super':
        model = MODEL_MAP[cfg.MODEL.ARCHITECTURE](in_channel=cfg.MODEL.IN_PLANES, out_channel=cfg.MODEL.OUT_PLANES, filters=cfg.MODEL.FILTERS)
    elif cfg.MODEL.ARCHITECTURE == 'unet_residual_2d':
        model = MODEL_MAP[cfg.MODEL.ARCHITECTURE](in_channel=cfg.MODEL.IN_PLANES, out_channel=cfg.MODEL.OUT_PLANES, filters=cfg.MODEL.FILTERS, \
                                             pad_mode=cfg.MODEL.PAD_MODE, norm_mode=cfg.MODEL.NORM_MODE, act_mode=cfg.MODEL.ACT_MODE,
                                             head_depth=cfg.MODEL.HEAD_DEPTH)
    else:
        model = MODEL_MAP[cfg.MODEL.ARCHITECTURE](in_channel=cfg.MODEL.IN_PLANES, out_channel=cfg.MODEL.OUT_PLANES, filters=cfg.MODEL.FILTERS, \
                                             pad_mode=cfg.MODEL.PAD_MODE, norm_mode=cfg.MODEL.NORM_MODE, act_mode=cfg.MODEL.ACT_MODE,
                                             do_embedding=(cfg.MODEL.EMBEDDING==1), head_depth=cfg.MODEL.HEAD_DEPTH, output_act=cfg.MODEL.OUTPUT_ACT)

    print('model: ', model.__class__.__name__)
    model = nn.DataParallel(model, device_ids=range(cfg.SYSTEM.NUM_GPUS))
    patch_replication_callback(model)
    model = model.to(device)
    
    return model