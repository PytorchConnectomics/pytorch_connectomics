import os,sys
import numpy as np
import h5py, time, argparse, itertools, datetime
from scipy import ndimage

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from torch_connectomics.model.model_zoo import *
from torch_connectomics.libs.sync import DataParallelWithCallback

# tensorboardX
from tensorboardX import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# functions
def init(args):
    sn = args.output+'/'
    if not os.path.isdir(sn):
        os.makedirs(sn)
    # I/O size in (z,y,x), no specified channel number
    model_io_size = np.array([int(x) for x in args.model_input.split(',')])

    # select training machine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("output path: ", sn)
    print("device: ", device)

    return model_io_size, device

def get_logger(args):
    log_name = args.output+'/log'
    date = str(datetime.datetime.now()).split(' ')[0]
    time = str(datetime.datetime.now()).split(' ')[1].split('.')[0]
    log_name += '_approx_'+date+'_'+time
    logger = open(log_name+'.txt','w') # unbuffered, write instantly

    # tensorboardX
    writer = SummaryWriter('runs/'+log_name)
    return logger, writer

def setup_model(args, device, exact=True, size_match=True,filters=[8, 12, 16, 20, 24]):

    MODEL_MAP = {'unetv0': unetv0,
                 'unetv1': unetv1,
                 'unetv2': unetv2,
                 'unetv3': unetv3,
                 'fpn': fpn}

    assert args.architecture in MODEL_MAP.keys()
    if args.task == 2:
        model = MODEL_MAP[args.architecture](in_channel=1, out_channel=args.out_channel, act='tanh',filters=filters)
    else:        
        model = MODEL_MAP[args.architecture](in_channel=1, out_channel=args.out_channel, filters=filters)
    print('model: ', model.__class__.__name__)
    model = DataParallelWithCallback(model, device_ids=range(args.num_gpu))
    model = model.to(device)

    if bool(args.load_model):
        print('Load pretrained model:')
        print(args.pre_model)
        if exact:
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

def blend(sz, sigma=1, mu=0.0):  
    """
    Gaussian blending
    """
    zz, yy, xx = np.meshgrid(np.linspace(-1,1,sz[0], dtype=np.float32), 
                                np.linspace(-1,1,sz[1], dtype=np.float32),
                                np.linspace(-1,1,sz[2], dtype=np.float32), indexing='ij')

    dd = np.sqrt(zz*zz + yy*yy + xx*xx)
    ww = 1e-4 + np.exp(-( (dd-mu)**2 / ( 2.0 * sigma**2 )))
    print('weight shape:', ww.shape)

    return ww
