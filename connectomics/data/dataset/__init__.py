import os,sys
import numpy as np
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from .dataset_volume import VolumeDataset
from .dataset_tile import TileDataset
from ..utils import collate_fn_target, collate_fn_test, seg_widen_border, readvol
from ..augmentation import *

__all__ = ['VolumeDataset',
           'TileDataset']

def _get_input(cfg, mode='train'):
    dir_name = cfg.DATASET.INPUT_PATH.split('@')
    img_name = cfg.DATASET.IMAGE_NAME.split('@')
    img_name = [dir_name[0] + x for x in img_name]

    label = None
    volume = [None]*len(img_name)
    if mode=='train':
        label_name = cfg.DATASET.LABEL_NAME.split('@')
        label_name = [dir_name[0] + x for x in label_name]
        assert len(img_name)==len(label_name)
        label = [None]*len(label_name)

    for i in range(len(img_name)):
        volume[i] = readvol(img_name[i])
        print(f"volume shape (original): {volume[i].shape}")
        if (np.array(cfg.DATASET.DATA_SCALE)!=1).any():
            volume[i] = zoom(volume[i], cfg.DATASET.DATA_SCALE, order=1) 
        volume[i] = np.pad(volume[i], ((cfg.DATASET.PAD_SIZE[0],cfg.DATASET.PAD_SIZE[0]), 
                                       (cfg.DATASET.PAD_SIZE[1],cfg.DATASET.PAD_SIZE[1]), 
                                       (cfg.DATASET.PAD_SIZE[2],cfg.DATASET.PAD_SIZE[2])), 'reflect')
        print(f"volume shape (after scale and padding): {volume[i].shape}")

        if mode=='train':
            label[i] = readvol(label_name[i])
            if (np.array(cfg.DATASET.DATA_SCALE)!=1).any():
                label[i] = zoom(label[i], cfg.DATASET.DATA_SCALE, order=0) 
            if cfg.DATASET.LABEL_EROSION!=0:
                label[i] = seg_widen_border(label[i], cfg.DATASET.LABEL_EROSION)
            if cfg.DATASET.LABEL_BINARY and label[i].max()>1:
                label[i] = label[i] // 255
            if cfg.DATASET.LABEL_MAG !=0:
                label[i] = (label[i]/cfg.DATASET.LABEL_MAG).astype(np.float32)
                
            label[i] = np.pad(label[i], ((cfg.DATASET.PAD_SIZE[0],cfg.DATASET.PAD_SIZE[0]), 
                                         (cfg.DATASET.PAD_SIZE[1],cfg.DATASET.PAD_SIZE[1]), 
                                         (cfg.DATASET.PAD_SIZE[2],cfg.DATASET.PAD_SIZE[2])), 'reflect')
            print(f"label shape: {label[i].shape}")
            
            #assert volume[i].shape == label[i].shape !MB
                 
    return volume, label


def get_dataset(cfg, augmentor, mode='train'):
    """Prepare dataset for training and inference.
    """
    assert mode in ['train', 'test']

    label_erosion = 0
    sample_label_size = cfg.MODEL.OUTPUT_SIZE
    sample_invalid_thres = cfg.DATASET.DATA_INVALID_THRES
    augmentor = augmentor
    topt,wopt = -1,-1
    if mode == 'train':
        sample_volume_size = cfg.MODEL.INPUT_SIZE
        sample_volume_size = augmentor.sample_size
        sample_label_size = sample_volume_size
        label_erosion = cfg.DATASET.LABEL_EROSION
        sample_stride = (1,1,1)
        topt, wopt = cfg.MODEL.TARGET_OPT, cfg.MODEL.WEIGHT_OPT
    elif mode == 'test':
        sample_stride = cfg.INFERENCE.STRIDE
        sample_volume_size = cfg.MODEL.INPUT_SIZE
      
    # dataset
    if cfg.DATASET.DO_CHUNK_TITLE==1:
        label_json = cfg.DATASET.INPUT_PATH+cfg.DATASET.LABEL_NAME if mode=='train' else ''
        dataset = TileDataset(chunk_num=cfg.DATASET.DATA_CHUNK_NUM, 
                              chunk_num_ind=cfg.DATASET.DATA_CHUNK_NUM_IND, 
                              chunk_iter=cfg.DATASET.DATA_CHUNK_ITER, 
                              chunk_stride=cfg.DATASET.DATA_CHUNK_STRIDE,
                              volume_json=cfg.DATASET.INPUT_PATH+cfg.DATASET.IMAGE_NAME, 
                              label_json=label_json,
                              sample_volume_size=sample_volume_size, 
                              sample_label_size=sample_label_size,
                              sample_stride=sample_stride, 
                              sample_invalid_thres = sample_invalid_thres,
                              augmentor=augmentor, 
                              target_opt = topt, 
                              weight_opt = wopt, 
                              mode = mode, 
                              label_erosion = label_erosion, 
                              pad_size=cfg.DATASET.PAD_SIZE)

    else:
        if cfg.DATASET.PRE_LOAD_DATA[0] is None: # load from cfg
            volume, label = _get_input(cfg, mode=mode)
        else:
            volume, label = cfg.DATASET.PRE_LOAD_DATA
        dataset = VolumeDataset(volume=volume, 
                                label=label, 
                                sample_volume_size=sample_volume_size, 
                                sample_label_size=sample_label_size,
                                sample_stride=sample_stride, 
                                sample_invalid_thres=sample_invalid_thres, 
                                augmentor=augmentor, 
                                target_opt=topt, 
                                weight_opt=wopt, 
                                mode=mode,
                                # Specify options for rejection samping:
                                reject_size_thres=cfg.DATASET.REJECT_SAMPLING.SIZE_THRES, 
                                reject_after_aug=cfg.DATASET.REJECT_SAMPLING.AFTER_AUG,
                                reject_p=cfg.DATASET.REJECT_SAMPLING.P)

    return dataset

def build_dataloader(cfg, augmentor, mode='train', dataset=None):
    """Prepare dataloader for training and inference.
    """
    print('Mode: ', mode)
    assert mode in ['train', 'test']

    SHUFFLE = (mode == 'train')

    if mode ==  'train':
        cf = collate_fn_target
        batch_size = cfg.SOLVER.SAMPLES_PER_BATCH
    else:
        cf = collate_fn_test
        batch_size = cfg.INFERENCE.SAMPLES_PER_BATCH

    if dataset == None:
        dataset = get_dataset(cfg, augmentor, mode)
    
    img_loader =  torch.utils.data.DataLoader(
          dataset, batch_size=batch_size, shuffle=SHUFFLE, collate_fn = cf,
          num_workers=cfg.SYSTEM.NUM_CPUS, pin_memory=True)

    return img_loader
