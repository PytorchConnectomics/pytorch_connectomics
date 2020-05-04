import os,datetime
import torch
import torch.nn as nn
import numpy as np

from .zoo import *
from .norm import patch_replication_callback
from .utils import Monitor, Criterion

def build_model(cfg, device, checkpoint):
    MODEL_MAP = {'unet_residual_3d': unet_residual_3d,
                 'fpn': fpn,
                'super':SuperResolution,
                'unet_super':Unet_super}

    assert cfg.MODEL.ARCHITECTURE in MODEL_MAP.keys()
    if cfg.MODEL.ARCHITECTURE == 'super':
        model = MODEL_MAP[cfg.MODEL.ARCHITECTURE](in_channel=cfg.MODEL.IN_PLANES, out_channel=cfg.MODEL.OUT_PLANES, filters=cfg.MODEL.FILTERS)
    else:
        model = MODEL_MAP[cfg.MODEL.ARCHITECTURE](in_channel=cfg.MODEL.IN_PLANES, out_channel=cfg.MODEL.OUT_PLANES, filters=cfg.MODEL.FILTERS, \
                                             pad_mode=cfg.MODEL.PAD_MODE, norm_mode=cfg.MODEL.NORM_MODE, act_mode=cfg.MODEL.ACT_MODE,
                                             do_embedding=(cfg.MODEL.EMBEDDING==1), head_depth=cfg.MODEL.HEAD_DEPTH)

    print('model: ', model.__class__.__name__)
    model = nn.DataParallel(model, device_ids=range(cfg.SYSTEM.NUM_GPUS))
    patch_replication_callback(model)
    model = model.to(device)

    if checkpoint is not None:
        print('Load pretrained model: ', checkpoint)
        if cfg.MODEL.EXACT: 
            # exact matching: the weights shape in pretrain model and current model are identical
            weight = torch.load(checkpoint)
            # change channels if needed
            if cfg.MODEL.PRE_MODEL_LAYER[0] != '':
                if cfg.MODEL.PRE_MODEL_LAYER_SELECT[0]==-1: # replicate channels
                    for kk in cfg.MODEL.PRE_MODEL_LAYER:
                        sz = list(np.ones(weight[kk][0:1].ndim,int))
                        sz[0] = cfg.MODEL.MODEL.OUT_PLANES
                        weight[kk] = weight[kk][0:1].repeat(sz)
                else: # select channels
                    for kk in cfg.MODEL.PRE_MODEL_LAYER:
                        weight[kk] = weight[kk][cfg.MODEL.PRE_MODEL_LAYER_SELECT]
            model.load_state_dict(weight)
        else:
            pretrained_dict = torch.load(cfg.MODEL.PRE_MODEL)
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict 
            if cfg.MODEL.SIZE_MATCH:
                model_dict.update(pretrained_dict) 
            else:
                for param_tensor in pretrained_dict:
                    if model_dict[param_tensor].size() == pretrained_dict[param_tensor].size():
                        model_dict[param_tensor] = pretrained_dict[param_tensor]       
            # 3. load the new state dict
            model.load_state_dict(model_dict)     
    
    return model

def build_monitor(cfg):
    time_now = str(datetime.datetime.now()).split(' ')
    date = time_now[0]
    time = time_now[1].split('.')[0].replace(':','-')
    log_path = os.path.join(cfg.DATASET.OUTPUT_PATH, 'log'+date+'_'+time)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    return Monitor(log_path, cfg.MONITOR.LOG_OPT+[cfg.MODEL.BATCH_SIZE],\
                   cfg.MONITOR.VIS_OPT, cfg.MONITOR.ITERATION_NUM)

def build_criterion(cfg, device):
    return Criterion(device, cfg.MODEL.TARGET_OPT, cfg.MODEL.LOSS_OPTION, cfg.MODEL.LOSS_WEIGHT,\
                     cfg.MODEL.REGU_OPT, cfg.MODEL.REGU_WEIGHT)
