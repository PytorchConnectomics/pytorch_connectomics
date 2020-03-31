import os,sys
import numpy as np
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from ..data.dataset import *
from ..data.utils import collate_fn_target, collate_fn_test, seg_widen_border
from ..data.augmentation import *
from .io_file import readvol


def _get_input(args, mode='train'):
    dir_name = args.input_path.split('@')
    img_name = args.img_name.split('@')
    img_name = [dir_name[0] + x for x in img_name]

    label = None
    volume = [None]*len(img_name)
    if mode=='train':
        label_name = args.label_name.split('@')
        label_name = [dir_name[0] + x for x in label_name]
        assert len(img_name)==len(label_name)
        label = [None]*len(label_name)

    for i in range(len(img_name)):
        volume[i] = readvol(img_name[i])
        if (args.data_scale!=1).any():
            volume[i] = zoom(volume[i], args.data_scale, order=1) 
        volume[i] = np.pad(volume[i], ((args.pad_size[0],args.pad_size[0]), 
                                                 (args.pad_size[1],args.pad_size[1]), 
                                                 (args.pad_size[2],args.pad_size[2])), 'reflect')
        print(f"volume shape: {volume[i].shape}")

        if mode=='train':
            label[i] = readvol(label_name[i])
            if (args.data_scale!=1).any():
                label[i] = zoom(label[i], args.data_scale, order=0) 
            if args.label_erosion!=0:
                label[i] = seg_widen_border(label[i],args.label_erosion)
            if args.label_binary and label[i].max()>1:
                label[i] = label[i]//255
            if args.label_mag !=0:
                label[i] = (label[i]/args.label_mag).astype(np.float32)
                
            label[i] = np.pad(label[i], ((args.pad_size[0],args.pad_size[0]), 
                                                     (args.pad_size[1],args.pad_size[1]), 
                                                     (args.pad_size[2],args.pad_size[2])), 'reflect')
            print(f"label shape: {label[i].shape}")
            
            #assert volume[i].shape == label[i].shape !MB
            
                
    return volume, label


def get_dataset(args, mode='train', preload_data=[None,None]):
    """Prepare dataset for training and inference.
    """
    assert mode in ['train', 'test']

    label_erosion = 0
    sample_label_size = args.model_output_size
    sample_invalid_thres = args.data_invalid_thres
    augmentor = None
    topt,wopt = -1,-1
    if mode=='train':
        sample_input_size = args.model_input_size
        if args.data_aug_mode==1:
            augmentor = Compose([Flip(p=1.0, do_ztrans=args.data_aug_ztrans),
                             Grayscale(p=0.75),
                             MissingParts(p=0.9),
                             MissingSection(p=0.5),
                             MisAlignment(p=1.0, displacement=16)], 
                             input_size = args.model_input_size)
            sample_input_size = augmentor.sample_size
        elif args.data_aug_mode==2:
            augmentor = Compose([Rotate(p=1.0),
                             Rescale(p=0.5),
                             Flip(p=1.0, do_ztrans=args.data_aug_ztrans),
                             Elastic(alpha=12.0, p=0.75),
                             Grayscale(p=0.75),
                             MissingParts(p=0.9),
                             MissingSection(p=0.5),
                             MisAlignment(p=1.0, displacement=16)], 
                             input_size = args.model_input_size)
            sample_input_size = augmentor.sample_size
        label_erosion = args.label_erosion
        sample_stride = (1,1,1)
        topt, wopt = args.target_opt, args.weight_opt
    elif mode=='test':
        sample_stride = args.test_stride
        sample_input_size = args.model_input_size
      
    # dataset
    if args.do_chunk_tile==1:
        label_json = args.input_path+args.label_name if mode=='train' else ''
        dataset = TileDataset(chunk_num=args.data_chunk_num, chunk_num_ind=args.data_chunk_num_ind, chunk_iter=args.data_chunk_iter, chunk_stride=args.data_chunk_stride,
                              volume_json=args.input_path+args.img_name, label_json=label_json,
                              sample_input_size=sample_input_size, sample_label_size=sample_label_size,
                              sample_stride=sample_stride, sample_invalid_thres = sample_invalid_thres,
                              augmentor=augmentor, target_opt = topt, weight_opt = wopt, mode = mode, 
                              label_erosion = label_erosion, pad_size=args.pad_size)
    else:
        if preload_data[0] is None: # load from command line args
            volume, label = _get_input(args, mode=mode)
        else:
            volume, label = preload_data
        dataset = VolumeDataset(volume=volume, label=label, 
                              sample_volume_size=sample_input_size, sample_label_size=sample_label_size,
                              sample_stride=sample_stride, sample_invalid_thres=sample_invalid_thres, 
                              augmentor=augmentor, target_opt = topt, weight_opt = wopt, mode = mode)

    return dataset

def get_dataloader(args, mode='train', dataset=None, preload_data=[None,None,None]):
    """Prepare dataloader for training and inference.
    """
    print('Mode: ', mode)
    assert mode in ['train', 'test']
    SHUFFLE = (mode == 'train')
    cf = collate_fn_test 
    if mode=='train':
        cf = collate_fn_target

    if dataset == None:
        dataset = get_dataset(args, mode, preload_data)
    
    img_loader =  torch.utils.data.DataLoader(
          dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = cf,
          num_workers=args.num_cpu, pin_memory=True)
    return img_loader
