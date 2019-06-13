import os,sys
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from torch_connectomics.data.dataset import AffinityDataset, SynapseDataset, SynapsePolarityDataset, MitoDataset, MitoSkeletonDataset
from torch_connectomics.data.utils import collate_fn, collate_fn_test, collate_fn_skel
from torch_connectomics.data.augmentation import *

TASK_MAP = {0: 'neuron segmentation',
            1: 'synapse detection',
            11: 'synapse polarity detection',
            2: 'mitochondria segmentation',
            22:'mitochondira segmentation with skeleton transform'}
 

def get_input_data(args, pad_size):
    dir_name = args.train.split('@')
    img_name = args.img_name.split('@')
    img_name = [dir_name[0] + x for x in img_name]
    if mode=='train':
        seg_name = args.seg_name.split('@')
        seg_name = [dir_name[0] + x for x in seg_name]
        if args.valid_mask is not None:
            mask_names = args.valid_mask.split('@')
            mask_locations = [dir_name[0] + x for x in mask_names]

    model_input = [None]*len(img_name)
    if mode=='train':
        assert len(img_name)==len(seg_name)
        model_label = [None]*len(seg_name)
        if args.valid_mask is not None:
            assert len(img_name) == len(mask_locations)
            model_mask = [None] * len(mask_locations)


    for i in range(len(img_name)):
        model_input[i] = np.array(h5py.File(img_name[i], 'r')['main'])/255.0
        model_input[i] = np.pad(model_input[i], ((pad_size[0],pad_size[0]), 
                                                 (pad_size[1],pad_size[1]), 
                                                 (pad_size[2],pad_size[2])), 'reflect')
        print(f"volume shape: {model_input[i].shape}")
        model_input[i] = model_input[i].astype(np.float32)

        if mode=='train':
            model_label[i] = np.array(h5py.File(seg_name[i], 'r')['main'])
            model_label[i] = model_label[i].astype(np.float32)
            model_label[i] = np.pad(model_label[i], ((pad_size[0],pad_size[0]), 
                                                     (pad_size[1],pad_size[1]), 
                                                     (pad_size[2],pad_size[2])), 'reflect')
            print(f"label shape: {model_label[i].shape}")
            
            assert model_input[i].shape == model_label[i].shape
            
            if args.valid_mask is not None:
                model_mask[i] = np.array(h5py.File(mask_locations[i], 'r')['main'])
                model_mask[i] = model_mask[i].astype(np.float32)
                model_mask[i] = np.pad(model_mask[i], ((pad_size[0],pad_size[0]),
                                                         (pad_size[1],pad_size[1]),
                                                         (pad_size[2],pad_size[2])), 'reflect')
                
                print(f"mask shape: {model_mask[i].shape}")
                assert model_input[i].shape == model_mask[i].shape
    return model_input, model_mask, model_label


def get_input(args, model_io_size, mode='train', preload_data=[None,None,None]):
    """Prepare dataloader for training and inference.
    """
    print('Task: ', TASK_MAP[args.task])
    assert mode in ['train', 'test']

    pad_size = [int(x) for x in args.model_pad.split(',')]

    # 1. load data
    if preload_data[0] is None: # load from command line args
        model_input, model_mask, model_label = get_input_data(args, pad_size)
    else:
        model_input, model_mask, model_label = preload_data


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

        if args.task == 0: # affininty prediction
            dataset = AffinityDataset(volume=model_input, label=model_label, sample_input_size=sample_input_size,
                                      sample_label_size=sample_input_size, augmentor=augmentor, mode = 'train')
        if args.task == 1: # synapse detection
            dataset = SynapseDataset(volume=model_input, label=model_label, sample_input_size=sample_input_size,
                                     sample_label_size=sample_input_size, augmentor=augmentor, mode = 'train')
        if args.task == 11: # synapse polarity detection
            dataset = SynapsePolarityDataset(volume=model_input, label=model_label, sample_input_size=sample_input_size,
                                     sample_label_size=sample_input_size, augmentor=augmentor, mode = 'train')
        if args.task == 2: # mitochondira segmentation
            dataset = MitoDataset(volume=model_input, label=model_label, sample_input_size=sample_input_size,
                                  sample_label_size=sample_input_size, augmentor=augmentor, mode = 'train')
        if args.task == 22: # mitochondira segmentation with skeleton transform
            dataset = MitoSkeletonDataset(volume=model_input, label=model_label, sample_input_size=sample_input_size,
                                  sample_label_size=sample_input_size, augmentor=augmentor, valid_mask=model_mask, mode='train')
            img_loader =  torch.utils.data.DataLoader(
                  dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = collate_fn_skel,
                  num_workers=args.num_cpu, pin_memory=True)
            return img_loader

        img_loader =  torch.utils.data.DataLoader(
              dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = collate_fn,
              num_workers=args.num_cpu, pin_memory=True)
        return img_loader

    else:
        # test stride
        if len(args.test_stride)==0: # not defined, do default 50%
            test_stride = np.maximum(1,model_io_size // 2)
        else:
            test_stride = [int(x) for x in args.test_stride.split(',')]
        if args.task == 0:
            dataset = AffinityDataset(volume=model_input, label=None, sample_input_size=model_io_size, \
                                      sample_label_size=None, sample_stride=test_stride, \
                                      augmentor=None, mode='test')
        elif args.task == 1: 
            dataset = SynapseDataset(volume=model_input, label=None, sample_input_size=model_io_size, \
                                     sample_label_size=None, sample_stride=test_stride, \
                                     augmentor=None, mode='test')
        elif args.task == 11:
        	dataset = SynapsePolarityDataset(volume=model_input, label=None, sample_input_size=model_io_size,
                                     sample_label_size=None, sample_stride=test_stride, \
                                     augmentor=None, mode = 'test')
        elif args.task == 2:
            dataset = MitoDataset(volume=model_input, label=None, sample_input_size=model_io_size, \
                                  sample_label_size=None, sample_stride=test_stride, \
                                  augmentor=None, mode='test')
        elif args.task == 22: 
        	dataset = MitoSkeletonDataset(volume=model_input, label=None, sample_input_size=model_io_size, \
                                  sample_label_size=None, sample_stride=test_stride, \
                                  augmentor=None, mode='test')

        img_loader =  torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = collate_fn_test,
                num_workers=args.num_cpu, pin_memory=True)                  
        return img_loader

