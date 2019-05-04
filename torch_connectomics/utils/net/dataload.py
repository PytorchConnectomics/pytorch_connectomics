import os,sys
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from torch_connectomics.data.dataset import AffinityDataset
from torch_connectomics.data.utils import collate_fn, collate_fn_test
from torch_connectomics.data.augmentation import *

def get_input(args, model_io_size, mode='train'):
    """
    Prepare dataloader for training and inference.
    """
    assert mode in ['train', 'test']

    if mode=='test':
        pad_size = model_io_size // 2
    else:
        pad_size = (0,0,0)
    volume_shape = []

    dir_name = args.train.split('@')
    img_name = args.img_name.split('@')
    img_name = [dir_name[0] + x for x in img_name]
    if mode=='train':
        seg_name = args.seg_name.split('@')
        seg_name = [dir_name[0] + x for x in seg_name]
    
    # 1. load data
    model_input = [None]*len(img_name)
    if mode=='train':
        assert len(img_name)==len(seg_name)
        model_label = [None]*len(seg_name)

    for i in range(len(img_name)):
        model_input[i] = np.array(h5py.File(img_name[i], 'r')['main'])/255.0
        model_input[i] = np.pad(model_input[i], ((pad_size[0],pad_size[0]), 
                                                 (pad_size[1],pad_size[1]), 
                                                 (pad_size[2],pad_size[2])), 'reflect')
        print("volume shape: ", model_input[i].shape)
        volume_shape.append(model_input[i].shape)
        model_input[i] = model_input[i].astype(np.float32)

        if mode=='train':
            model_label[i] = np.array(h5py.File(seg_name[i], 'r')['main'])
            model_label[i] = model_label[i].astype(np.float32)
            print("label shape: ", model_label[i].shape)
   
    if mode=='train':
        # setup augmentor
        augmentor = Compose([Rotate(p=1.0),
                             Rescale(p=0.5),
                             Flip(p=1.0),
                             Elastic(alpha=12.0, p=0.75),
                             Grayscale(p=0.75),
                             MissingParts(p=0.9),
                             MissingSection(p=0.5),
                             MisAlignment(p=1.0, displacement=16)], 
                             input_size = model_io_size)
        # augmentor = None # debug
    else:
        augmentor = None

    print('data augmentation: ', augmentor is not None)
    SHUFFLE = (mode=='train')
    print('batch size: ', args.batch_size)

    if mode=='train':
        if augmentor is None:
            sample_input_size = model_io_size
        else:
            sample_input_size = augmentor.sample_size
        dataset = AffinityDataset(volume=model_input, label=model_label, sample_input_size=sample_input_size,
                                  sample_label_size=sample_input_size, augmentor=augmentor, mode = 'train')    
        img_loader =  torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = collate_fn,
                num_workers=args.num_cpu, pin_memory=True)
        return img_loader

    else:
        dataset = AffinityDataset(volume=model_input, label=None, sample_input_size=model_io_size, \
                                  sample_label_size=None, sample_stride=model_io_size // 2, \
                                  augmentor=None, mode='test')      
        img_loader =  torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = collate_fn_test,
                num_workers=args.num_cpu, pin_memory=True)                  
        return img_loader, volume_shape, pad_size