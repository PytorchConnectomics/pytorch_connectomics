import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from .zoo import *
from .norm import patch_replication_callback

def build_model(cfg, device, rank=None):
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
    return make_parallel(model, cfg, device, rank)

def make_parallel(model, cfg, device, rank=None):
    if cfg.SYSTEM.PARALLEL == 'DDP':
        print('Parallelism with DistributedDataParallel.')
        # Currently SyncBatchNorm only supports DistributedDataParallel (DDP) 
        # with single GPU per process. Use torch.nn.SyncBatchNorm.convert_sync_batchnorm() 
        # to convert BatchNorm*D layer to SyncBatchNorm before wrapping Network with DDP.
        if cfg.MODEL.NORM_MODE == "SyncBN":
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = model.to(device)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        assert rank is not None
        model = DistributedDataParallel(model, device_ids=[rank],
                                        output_device=rank)

    elif cfg.SYSTEM.PARALLEL == 'DP':
        gpu_device_ids = list(range(cfg.SYSTEM.NUM_GPUS))
        print('Parallelism with DataParallel.')
        model = nn.DataParallel(model, device_ids=gpu_device_ids)
        model = model.to(device)

    else:
        print('No parallelism across multiple GPUs.')
        model = model.to(device)

    return model
