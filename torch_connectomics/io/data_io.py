import os,sys
import numpy as np
import h5py
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from torch_connectomics.data.dataset import *
from torch_connectomics.data.utils import collate_fn, collate_fn_test, collate_fn_skel
from torch_connectomics.data.augmentation import *
from torch_connectomics.libs.seg.seg_util import widen_border3

TASK_MAP = {0: 'neuron segmentation',
            1: 'synapse detection',
            11: 'synapse polarity detection',
            2: 'mitochondria segmentation',
            22:'mitochondira segmentation with skeleton transform'}
 

def get_data(args, mode='train'):
    dir_name = args.train.split('@')
    img_name = args.img_name.split('@')
    img_name = [dir_name[0] + x for x in img_name]
    model_mask = None
    model_label = None

    if mode=='train':
        label_name = args.label_name.split('@')
        label_name = [dir_name[0] + x for x in label_name]
        if args.valid_mask is not None:
            mask_names = args.valid_mask.split('@')
            mask_locations = [dir_name[0] + x for x in mask_names]

    model_input = [None]*len(img_name)
    if mode=='train':
        assert len(img_name)==len(label_name)
        model_label = [None]*len(label_name)
        if args.valid_mask is not None:
            assert len(img_name) == len(mask_locations)
            model_mask = [None] * len(mask_locations)


    for i in range(len(img_name)):
        model_input[i] = np.array(h5py.File(img_name[i], 'r')['main'])/255.0
        model_input[i] = model_input[i].astype(np.float32)
        if (args.data_scale!=1).any():
            model_input[i] = zoom(model_input[i], args.data_scale, order=1) 
        model_input[i] = np.pad(model_input[i], ((args.pad_size[0],args.pad_size[0]), 
                                                 (args.pad_size[1],args.pad_size[1]), 
                                                 (args.pad_size[2],args.pad_size[2])), 'reflect')
        print(f"volume shape: {model_input[i].shape}")

        if mode=='train':
            model_label[i] = np.array(h5py.File(label_name[i], 'r')['main'])
            model_label[i] = model_label[i].astype(np.float32)
            if (args.data_scale!=1).any():
                model_label[i] = zoom(model_label[i], args.data_scale, order=0) 
            if args.label_erosion!=0:
                model_label[i] = widen_border3(model_label[i],args.label_erosion)
            model_label[i] = np.pad(model_label[i], ((args.pad_size[0],args.pad_size[0]), 
                                                     (args.pad_size[1],args.pad_size[1]), 
                                                     (args.pad_size[2],args.pad_size[2])), 'reflect')
            print(f"label shape: {model_label[i].shape}")
            
            assert model_input[i].shape == model_label[i].shape
            
            if args.valid_mask is not None:
                model_mask[i] = np.array(h5py.File(mask_locations[i], 'r')['main'])
                model_mask[i] = model_mask[i].astype(np.float32)
                if (args.data_scale!=1).any():
                    model_mask[i] = zoom(model_mask[i], args.data_scale, order=0) 
                model_mask[i] = np.pad(model_mask[i], ((args.pad_size[0],args.pad_size[0]),
                                                       (args.pad_size[1],args.pad_size[1]),
                                                       (args.pad_size[2],args.pad_size[2])), 'reflect')
                
                print(f"mask shape: {model_mask[i].shape}")
                assert model_input[i].shape == model_mask[i].shape
                
    return model_input, model_mask, model_label


def get_dataloader(args, mode='train', preload_data=[None,None,None], dataset=None):
    """Prepare dataloader for training and inference.
    """
    print('Task: ', TASK_MAP[args.task], end='\t')
    print('Mode: ', mode)
    assert mode in ['train', 'test']
    SHUFFLE = (mode == 'train')

    if mode=='train':
        if args.task == 22:
            cf = collate_fn_skel
        else:
            cf = collate_fn
    else:
        cf = collate_fn_test 
    # given dataset
    if dataset is not None:
        img_loader =  torch.utils.data.DataLoader(
              dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = cf,
              num_workers=args.num_cpu, pin_memory=True)
        return img_loader

    # 1. load data
    if args.do_tile==0:
        if preload_data[0] is None: # load from command line args
            model_input, model_mask, model_label = get_data(args, mode=mode)
        else:
            model_input, model_mask, model_label = preload_data


    # setup augmentor
    if mode=='train':
        augmentor = Compose([Rotate(p=1.0),
                             Rescale(p=0.5),
                             Flip(p=1.0),
                             Elastic(alpha=12.0, p=0.75),
                             Grayscale(p=0.75),
                             MissingParts(p=0.9),
                             MissingSection(p=0.5),
                             MisAlignment(p=1.0, displacement=16)], 
                             input_size = args.model_io_size)
    else:
        augmentor = None

    print('data augmentation: ', augmentor is not None)
    print('batch size: ', args.batch_size)

    if mode=='train':
        if augmentor is None:
            sample_input_size = args.model_io_size
        else:
            sample_input_size = augmentor.sample_size
        sample_label_size=sample_input_size
        sample_stride = (1,1,1)
    else:
        sample_input_size = args.test_size
        sample_label_size=None
        sample_stride = args.test_stride

    # dataset
    if args.do_tile==1:
        dataset = TileDataset(dataset_type=args.task, chunk_num=args.data_chunk_num, chunk_iter=args.data_chunk_iter, chunk_stride=args.data_chunk_stride,
                              volume_json=args.train+args.img_name, label_json=args.train+args.label_name,
                              sample_input_size=sample_input_size, sample_label_size=sample_label_size,sample_stride=sample_stride,
                              augmentor=augmentor, mode = mode)
    else:
        # print('sample crop size: ', sample_input_size)
        if args.task == 0: # affininty prediction
            dataset = AffinityDataset(volume=model_input, label=model_label, 
                                      sample_input_size=sample_input_size, sample_label_size=sample_label_size,sample_stride=sample_stride, 
                                      augmentor=augmentor, mode = mode)
        if args.task == 1: # synapse detection
            dataset = SynapseDataset(volume=model_input, label=model_label, 
                                     sample_input_size=sample_input_size,sample_label_size=sample_label_size,sample_stride=sample_stride, 
                                     augmentor=augmentor, mode = mode)
        if args.task == 11: # synapse polarity detection
            dataset = SynapsePolarityDataset(volume=model_input, label=model_label, 
                                             sample_input_size=sample_input_size,sample_label_size=sample_label_size,sample_stride=sample_stride, 
                                             augmentor=augmentor, mode = mode)
        if args.task == 2: # mitochondira segmentation
            dataset = MitoDataset(volume=model_input, label=model_label,
                                  sample_input_size=sample_input_size,sample_label_size=sample_label_size,sample_stride=sample_stride,
                                  augmentor=augmentor, mode = mode)
        if args.task == 22: # mitochondira segmentation with skeleton transform
            dataset = MitoSkeletonDataset(volume=model_input, label=model_label,
                                          sample_input_size=sample_input_size,sample_label_size=sample_label_size,sample_stride=sample_stride,
                                          augmentor=augmentor, valid_mask=model_mask, mode=mode)

    if args.do_tile==1:
        return dataset
    else:
        img_loader =  torch.utils.data.DataLoader(
              dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = cf,
              num_workers=args.num_cpu, pin_memory=True)

        return img_loader
