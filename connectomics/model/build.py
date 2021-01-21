import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from .unet import UNet3D

def build_model(cfg, device, rank=None):
    MODEL_MAP = {'unet_3d': UNet3D}

    model_arch = cfg.MODEL.ARCHITECTURE
    assert model_arch in MODEL_MAP.keys()
    kwargs = {
        'in_channel': cfg.MODEL.IN_PLANES,
        'out_channel': cfg.MODEL.OUT_PLANES,
        'filters': cfg.MODEL.FILTERS,
        'is_isotropic': cfg.DATASET.ISOTROPIC,
        'isotropy': cfg.MODEL.ISOTROPY,
        'pad_mode': cfg.MODEL.PAD_MODE,
        'act_mode': cfg.MODEL.ACT_MODE,
        'norm_mode': cfg.MODEL.NORM_MODE,
        'pooling': cfg.MODEL.POOING_LAYER,
        'output_act': cfg.MODEL.OUTPUT_ACT,
    }
    if 'fpn' in model_arch:
        kwargs['backbone'] = cfg.MODEL.BACKBONE
    if 'unet' in model_arch:
        kwargs['block_type'] = cfg.MODEL.BLOCK_TYPE

    model = MODEL_MAP[cfg.MODEL.ARCHITECTURE](**kwargs)
    print('model: ', model.__class__.__name__)    
    return make_parallel(model, cfg, device, rank)

def make_parallel(model, cfg, device, rank=None):
    if cfg.SYSTEM.PARALLEL == 'DDP':
        print('Parallelism with DistributedDataParallel.')
        # Currently SyncBatchNorm only supports DistributedDataParallel (DDP) 
        # with single GPU per process. Use torch.nn.SyncBatchNorm.convert_sync_batchnorm() 
        # to convert BatchNorm*D layer to SyncBatchNorm before wrapping Network with DDP.
        if cfg.MODEL.NORM_MODE == "sync_bn":
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
