import torch
import torch.nn as nn

from torch_connectomics.model.model_zoo import *
from torch_connectomics.libs.sync import patch_replication_callback

def get_model(args, exact=True, size_match=True):
    MODEL_MAP = {'unetv0': unetv0,
                 'unetv1': unetv1,
                 'unetv2': unetv2,
                 'unetv3': unetv3,
                 'fpn': fpn}

    assert args.architecture in MODEL_MAP.keys()
    if args.task == 2:
        model = MODEL_MAP[args.architecture](in_channel=1, out_channel=args.out_channel, act='tanh',filters=args.filters)
    else:        
        model = MODEL_MAP[args.architecture](in_channel=1, out_channel=args.out_channel, filters=args.filters)
    print('model: ', model.__class__.__name__)
    model = nn.DataParallel(model, device_ids=range(args.num_gpu))
    patch_replication_callback(model)
    model = model.to(args.device)

    if args.pre_model!='':
        print('Load pretrained model:',args.pre_model)
        if exact: 
            # exact matching: the weights shape in pretrain model and current model are identical
            model.load_state_dict(torch.load(args.pre_model))
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


