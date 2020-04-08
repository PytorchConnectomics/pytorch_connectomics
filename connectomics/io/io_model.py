import os,datetime
import torch
import torch.nn as nn
import numpy as np

from ..model.zoo import *
from ..model.norm import patch_replication_callback
from ..model.utils import Monitor, Criterion


def get_model(args, exact=True, size_match=True):
    MODEL_MAP = {'unet_residual_3d': unet_residual_3d,
                 'fpn': fpn,
                'super':SuperResolution}

    assert args.architecture in MODEL_MAP.keys()
    if args.architecture == 'super':
        model = MODEL_MAP[args.architecture](in_channel=1, out_channel=args.model_out_channel, filters=args.filters)
    else:
        model = MODEL_MAP[args.architecture](in_channel=1, out_channel=args.model_out_channel, filters=args.filters, \
                                             pad_mode=args.model_pad_mode, norm_mode=args.model_norm_mode, act_mode=args.model_act_mode,                                             do_embedding=(args.model_embedding==1), head_depth=args.model_head_depth)

    print('model: ', model.__class__.__name__)
    model = nn.DataParallel(model, device_ids=range(args.num_gpu))
    patch_replication_callback(model)
    model = model.to(args.device)

    if args.pre_model!='':
        print('Load pretrained model:',args.pre_model)
        if exact: 
            # exact matching: the weights shape in pretrain model and current model are identical
            weight = torch.load(args.pre_model)
            # change channels if needed
            if args.pre_model_layer[0] != '':
                if args.pre_model_layer_select[0]==-1: # replicate channels
                    for kk in args.pre_model_layer:
                        sz = list(np.ones(weight[kk][0:1].ndim,int))
                        sz[0] = args.model_out_channel
                        weight[kk] = weight[kk][0:1].repeat(sz)
                else: # select channels
                    for kk in args.pre_model_layer:
                        weight[kk] = weight[kk][args.pre_model_layer_select]
            model.load_state_dict(weight)
        else:
            pretrained_dict = torch.load(args.pre_model)
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict 
            if size_match:
                model_dict.update(pretrained_dict) 
            else:
                for param_tensor in pretrained_dict:
                    if model_dict[param_tensor].size() == pretrained_dict[param_tensor].size():
                        model_dict[param_tensor] = pretrained_dict[param_tensor]       
            # 3. load the new state dict
            model.load_state_dict(model_dict)     
    
    return model

def get_monitor(args):
    time_now = str(datetime.datetime.now()).split(' ')
    date = time_now[0]
    time = time_now[1].split('.')[0].replace(':','-')
    log_path = os.path.join(args.output_path, 'log'+date+'_'+time)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    return Monitor(log_path, args.mon_log_opt+[args.batch_size],\
                   args.mon_vis_opt, args.mon_iter_num)

def get_criterion(args):
    return Criterion(args.device, args.target_opt, args.loss_opt, args.loss_weight,\
                     args.regu_opt, args.regu_weight)
