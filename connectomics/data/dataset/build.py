import os
import math
import numpy as np
from scipy.ndimage import zoom

import torch
import torch.utils.data

from .dataset_volume import VolumeDataset
from .dataset_tile import TileDataset
from ..utils import *

def _make_path_list(cfg, dir_name, file_name, rank=None):
    r"""Concatenate directory path(s) and filenames and return
    the complete file paths. 
    """
    if not cfg.DATASET.IS_ABSOLUTE_PATH:
        assert len(dir_name) == 1 or len(dir_name) == len(file_name)
        if len(dir_name) == 1:
            file_name = [os.path.join(dir_name[0], x) for x in file_name]
        else:
            file_name = [os.path.join(dir_name[i], file_name[i]) 
                        for i in range(len(file_name))]

    file_name = _distribute_data(cfg, file_name, rank)
    return file_name

def _distribute_data(cfg, file_name, rank=None):
    r"""Distribute the data (files) equally for multiprocessing.
    """
    if rank is None or cfg.DATASET.DISTRIBUTED == False:
        return file_name

    world_size = cfg.SYSTEM.NUM_GPUS
    num_files = len(file_name)
    print('total number of files: ', num_files)
    ratio = num_files / float(world_size)
    ratio = int(math.ceil(ratio-1) + 1) # 1.0 -> 1, 1.1 -> 2

    extended = [file_name[i % num_files] for i in range(world_size*ratio)]
    splited = [extended[i:i+ratio] for i in range(0, len(extended), ratio)]

    return splited[rank]

def _get_file_list(name):
    suffix = name.split('.')[-1]
    if suffix == 'txt':
        filelist = [line.rstrip('\n') for line in open(name)]
        return filelist

    return name.split('@')

def _get_input(cfg, mode='train', rank=None):
    r"""Load the inputs specified by the configuration options.
    """
    dir_name = _get_file_list(cfg.DATASET.INPUT_PATH)
    img_name = _get_file_list(cfg.DATASET.IMAGE_NAME)
    img_name = _make_path_list(cfg, dir_name, img_name, rank)
    print(rank, len(img_name), list(map(os.path.basename, img_name)))

    label = None
    if mode=='train' and cfg.DATASET.LABEL_NAME is not None:
        label_name = _get_file_list(cfg.DATASET.LABEL_NAME)
        label_name = _make_path_list(cfg, dir_name, label_name, rank)
        assert len(label_name) == len(img_name)
        label = [None]*len(label_name)

    valid_mask = None
    if mode=='train' and cfg.DATASET.VALID_MASK_NAME is not None:
        valid_mask_name = _get_file_list(cfg.DATASET.VALID_MASK_NAME)
        valid_mask_name = _make_path_list(cfg, dir_name, valid_mask_name, rank)
        assert len(valid_mask_name) == len(img_name)
        valid_mask = [None]*len(valid_mask_name)

    volume = [None] * len(img_name)
    for i in range(len(img_name)):
        volume[i] = readvol(img_name[i])
        print(f"volume shape (original): {volume[i].shape}")
        if cfg.DATASET.NORMALIZE:
            volume[i] = normalize_image(volume[i])
        if (np.array(cfg.DATASET.DATA_SCALE)!=1).any():
            volume[i] = zoom(volume[i], cfg.DATASET.DATA_SCALE, order=1)
        volume[i] = np.pad(volume[i], get_padsize(cfg.DATASET.PAD_SIZE), 'reflect')
        print(f"volume shape (after scaling and padding): {volume[i].shape}")

        if mode=='train' and label is not None:
            label[i] = readvol(label_name[i])
            if cfg.DATASET.LABEL_VAST:
                label[i] = vast2Seg(label[i])
            if label[i].ndim == 2: # make it into 3D volume
                label[i] = label[i][None,:]
            if (np.array(cfg.DATASET.DATA_SCALE)!=1).any():
                label[i] = zoom(label[i], cfg.DATASET.DATA_SCALE, order=0) 
            if cfg.DATASET.LABEL_EROSION!=0:
                label[i] = seg_widen_border(label[i], cfg.DATASET.LABEL_EROSION)
            if cfg.DATASET.LABEL_BINARY and label[i].max()>1:
                label[i] = label[i] // 255
            if cfg.DATASET.LABEL_MAG !=0:
                label[i] = (label[i]/cfg.DATASET.LABEL_MAG).astype(np.float32)
                
            label[i] = np.pad(label[i], get_padsize(cfg.DATASET.PAD_SIZE), 'reflect')
            print(f"label shape: {label[i].shape}")

        if mode=='train' and valid_mask is not None:
            valid_mask[i] = readvol(valid_mask_name[i])
            if (np.array(cfg.DATASET.DATA_SCALE)!=1).any():
                valid_mask[i] = zoom(valid_mask[i], cfg.DATASET.DATA_SCALE, order=0) 

            valid_mask[i] = np.pad(valid_mask[i], get_padsize(cfg.DATASET.PAD_SIZE), 'reflect')
            print(f"valid_mask shape: {label[i].shape}")
                 
    return volume, label, valid_mask


def get_dataset(cfg, augmentor, mode='train', rank=None):
    r"""Prepare dataset for training and inference.
    """
    assert mode in ['train', 'test']

    label_erosion = 0
    sample_label_size = cfg.MODEL.OUTPUT_SIZE
    topt, wopt = ['0'], [['0']]
    if mode == 'train':
        sample_volume_size = augmentor.sample_size if augmentor is not None else cfg.MODEL.INPUT_SIZE
        sample_label_size = sample_volume_size
        label_erosion = cfg.DATASET.LABEL_EROSION
        sample_stride = (1, 1, 1)
        topt, wopt = cfg.MODEL.TARGET_OPT, cfg.MODEL.WEIGHT_OPT
        iter_num = cfg.SOLVER.ITERATION_TOTAL * cfg.SOLVER.SAMPLES_PER_BATCH 
    elif mode == 'test':
        sample_stride = cfg.INFERENCE.STRIDE
        sample_volume_size = cfg.MODEL.INPUT_SIZE
        iter_num = -1

    shared_kwargs = {
        "sample_volume_size": sample_volume_size,
        "sample_label_size": sample_label_size,
        "sample_stride": sample_stride,
        "augmentor": augmentor,
        "target_opt": topt,
        "weight_opt": wopt,
        "mode": mode,
        "do_2d": cfg.DATASET.DO_2D,
        "reject_size_thres": cfg.DATASET.REJECT_SAMPLING.SIZE_THRES,
        "reject_p": cfg.DATASET.REJECT_SAMPLING.P,
    }
      
    if cfg.DATASET.DO_CHUNK_TITLE==1: # build TileDataset
        label_json, valid_mask_json = None, None
        if mode == 'train':
            if cfg.DATASET.LABEL_NAME is not None:
                label_json = cfg.DATASET.INPUT_PATH + cfg.DATASET.LABEL_NAME
            if cfg.DATASET.VALID_MASK_NAME is not None:
                valid_mask_json = cfg.DATASET.INPUT_PATH + cfg.DATASET.VALID_MASK_NAME

        dataset = TileDataset(chunk_num=cfg.DATASET.DATA_CHUNK_NUM, 
                              chunk_num_ind=cfg.DATASET.DATA_CHUNK_NUM_IND, 
                              chunk_iter=cfg.DATASET.DATA_CHUNK_ITER, 
                              chunk_stride=cfg.DATASET.DATA_CHUNK_STRIDE,
                              volume_json=cfg.DATASET.INPUT_PATH+cfg.DATASET.IMAGE_NAME, 
                              label_json=label_json,
                              valid_mask_json=valid_mask_json,
                              label_erosion=label_erosion, 
                              pad_size=cfg.DATASET.PAD_SIZE,
                              **shared_kwargs)

    else: # build VolumeDataset
        volume, label, valid_mask = _get_input(cfg, mode=mode, rank=rank)
        dataset = VolumeDataset(volume=volume, label=label, valid_mask=valid_mask,
                                iter_num=iter_num, **shared_kwargs)

    return dataset

def build_dataloader(cfg, augmentor, mode='train', dataset=None, rank=None):
    r"""Prepare dataloader for training and inference.
    """
    print('Mode: ', mode)
    assert mode in ['train', 'test']

    if mode ==  'train':
        cf = collate_fn_train
        batch_size = cfg.SOLVER.SAMPLES_PER_BATCH
    else:
        cf = collate_fn_test
        batch_size = cfg.INFERENCE.SAMPLES_PER_BATCH * cfg.SYSTEM.NUM_GPUS

    if dataset == None:
        dataset = get_dataset(cfg, augmentor, mode, rank)

    sampler = None
    num_workers = cfg.SYSTEM.NUM_CPUS
    if cfg.SYSTEM.DISTRIBUTED:
        num_workers = cfg.SYSTEM.NUM_CPUS // cfg.SYSTEM.NUM_GPUS
        if cfg.DATASET.DISTRIBUTED == False:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    SHUFFLE = (mode == 'train') and (sampler is None)

    # In PyTorch, each worker will create a copy of the Dataset, so if the data 
    # is preload the data, the memory usage should increase a lot.
    # https://discuss.pytorch.org/t/define-iterator-on-dataloader-is-very-slow/52238/2
    img_loader =  torch.utils.data.DataLoader(
          dataset, batch_size=batch_size, shuffle=SHUFFLE, collate_fn = cf,
          sampler=sampler, num_workers=num_workers, pin_memory=True)

    return img_loader
