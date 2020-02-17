import os,sys
import numpy as np
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from ..data.dataset import *
from ..data.utils import collate_fn, collate_fn_plus, collate_fn_test, collate_fn_skel,seg_widen_border
from ..data.augmentation import *
from .io_file import readvol


def _get_input(args, mode='train'):
    dir_name = args.input_path.split('@')
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
        model_input[i] = readvol(img_name[i])
        if (args.data_scale!=1).any():
            model_input[i] = zoom(model_input[i], args.data_scale, order=1) 
        model_input[i] = np.pad(model_input[i], ((args.pad_size[0],args.pad_size[0]), 
                                                 (args.pad_size[1],args.pad_size[1]), 
                                                 (args.pad_size[2],args.pad_size[2])), 'reflect')
        print(f"volume shape: {model_input[i].shape}")

        if mode=='train':
            model_label[i] = readvol(label_name[i])
            if (args.data_scale!=1).any():
                model_label[i] = zoom(model_label[i], args.data_scale, order=0) 
            if args.label_erosion!=0:
                model_label[i] = widen_border3(model_label[i],args.label_erosion)
            if args.label_binary and model_label[i].max()>1:
                model_label[i] = model_label[i]//255
            model_label[i] = np.pad(model_label[i], ((args.pad_size[0],args.pad_size[0]), 
                                                     (args.pad_size[1],args.pad_size[1]), 
                                                     (args.pad_size[2],args.pad_size[2])), 'reflect')
            print(f"label shape: {model_label[i].shape}")
            
            assert model_input[i].shape == model_label[i].shape
            
            if args.valid_mask is not None:
                model_mask[i] = readvol(mask_locations[i])
                if (args.data_scale!=1).any():
                    model_mask[i] = zoom(model_mask[i], args.data_scale, order=0) 
                model_mask[i] = np.pad(model_mask[i], ((args.pad_size[0],args.pad_size[0]),
                                                       (args.pad_size[1],args.pad_size[1]),
                                                       (args.pad_size[2],args.pad_size[2])), 'reflect')
                
                print(f"mask shape: {model_mask[i].shape}")
                assert model_input[i].shape == model_mask[i].shape
    
    return model_input, model_mask, model_label


def get_dataset(args, mode='train', preload_data=[None,None,None]):
    """Prepare dataset for training and inference.
    """
    assert mode in ['train', 'test']

    label_erosion = 0
    sample_label_size=None
    sample_invalid_thres=args.data_invalid_thres
    if mode=='train':
        augmentor = Compose([Rotate(p=1.0),
                             Rescale(p=0.5),
                             Flip(p=1.0, do_ztrans=args.data_aug_ztrans),
                             Elastic(alpha=12.0, p=0.75),
                             Grayscale(p=0.75),
                             MissingParts(p=0.9),
                             MissingSection(p=0.5),
                             MisAlignment(p=1.0, displacement=16)], 
                             input_size = args.model_io_size)
        label_erosion = args.label_erosion
        if augmentor is None:
            sample_input_size = args.model_io_size
        else:
            sample_input_size = augmentor.sample_size
        sample_label_size=sample_input_size
        sample_stride = (1,1,1)
    elif mode=='test':
        sample_input_size = args.test_size
        sample_stride = args.test_stride
       
    # dataset
    if args.do_chunk_tile==1:
        label_json = args.input_path+args.label_name if mode=='train' else ''
        dataset = TileDataset(chunk_num=args.data_chunk_num, chunk_num_ind=args.data_chunk_num_ind, chunk_iter=args.data_chunk_iter, chunk_stride=args.data_chunk_stride,
                              volume_json=args.input_path+args.img_name, label_json=label_json,
                              sample_input_size=sample_input_size, sample_label_size=sample_label_size,
                              sample_stride=sample_stride, sample_invalid_thres = sample_invalid_thres,
                              augmentor=augmentor, mode = mode,
                              label_erosion = label_erosion, pad_size=args.pad_size)
    else:
        if preload_data[0] is None: # load from command line args
            model_input, model_mask, model_label = _get_input(args, mode=mode)
        else:
            model_input, model_mask, model_label = preload_data
        dataset = VolumeDataset(volume=model_input, label=model_label, 
                              sample_input_size=sample_input_size, sample_label_size=sample_label_size,
                              sample_stride=sample_stride, sample_invalid_thres=sample_invalid_thres, 
                              augmentor=augmentor, mode = mode)

    return dataset

def get_dataloader(args, mode='train', dataset=None, preload_data=[None,None,None]):
    """Prepare dataloader for training and inference.
    """
    print('Mode: ', mode)
    assert mode in ['train', 'test']
    SHUFFLE = (mode == 'train')
    cf = collate_fn_test 
    if mode=='train':
        cf = collate_fn

    if dataset == None:
        dataset = get_dataset(args, mode, preload_data)

    img_loader =  torch.utils.data.DataLoader(
          dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = cf,
          num_workers=args.num_cpu, pin_memory=True)
    return img_loader
