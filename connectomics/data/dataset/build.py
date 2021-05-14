from __future__ import print_function, division
from typing import Union, List

import os
import math
import glob
import copy
import numpy as np
from scipy.ndimage import zoom

import torch
import torch.utils.data

from .dataset_volume import VolumeDataset
from .dataset_tile import TileDataset
from .collate import *
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

        if cfg.DATASET.LOAD_2D:
            temp_list = copy.deepcopy(file_name)
            file_name = []
            for x in temp_list:
                suffix = x.split('/')[-1]
                if suffix in ['*.png', '*.tif']:
                    file_name += sorted(glob.glob(x, recursive=True))

    file_name = _distribute_data(cfg, file_name, rank)
    return file_name


def _distribute_data(cfg, file_name, rank=None):
    r"""Distribute the data (files) equally for multiprocessing.
    """
    if rank is None or cfg.DATASET.DISTRIBUTED == False:
        return file_name

    world_size = cfg.SYSTEM.NUM_GPUS
    num_files = len(file_name)
    ratio = num_files / float(world_size)
    ratio = int(math.ceil(ratio-1) + 1)  # 1.0 -> 1, 1.1 -> 2

    extended = [file_name[i % num_files] for i in range(world_size*ratio)]
    splited = [extended[i:i+ratio] for i in range(0, len(extended), ratio)]

    return splited[rank]


def _get_file_list(name: Union[str, List[str]]) -> list:
    if isinstance(name, list):
        return name

    suffix = name.split('.')[-1]
    if suffix == 'txt':  # a text file saving the absolute path
        filelist = [line.rstrip('\n') for line in open(name)]
        return filelist

    return name.split('@')


def _get_input(cfg,
               mode='train',
               rank=None,
               dir_name_init: Optional[list] = None,
               img_name_init: Optional[list] = None):
    r"""Load the inputs specified by the configuration options.
    """
    assert mode in ['train', 'val', 'test']
    if dir_name_init is not None:
        dir_name = dir_name_init
    else:
        dir_name = _get_file_list(cfg.DATASET.INPUT_PATH)

    if mode == 'val':
        img_name = cfg.DATASET.VAL_IMAGE_NAME
        label_name = cfg.DATASET.VAL_LABEL_NAME
        valid_mask_name = cfg.DATASET.VAL_VALID_MASK_NAME
        pad_size = cfg.DATASET.VAL_PAD_SIZE
    else:
        img_name = cfg.DATASET.IMAGE_NAME
        label_name = cfg.DATASET.LABEL_NAME
        valid_mask_name = cfg.DATASET.VALID_MASK_NAME
        pad_size = cfg.DATASET.PAD_SIZE

    if img_name_init is not None:
        img_name = img_name_init
    else:
        img_name = _get_file_list(img_name)
    img_name = _make_path_list(cfg, dir_name, img_name, rank)
    print(rank, len(img_name), list(map(os.path.basename, img_name)))

    label = None
    if mode in ['val', 'train'] and label_name is not None:
        label_name = _get_file_list(label_name)
        label_name = _make_path_list(cfg, dir_name, label_name, rank)
        assert len(label_name) == len(img_name)
        label = [None]*len(label_name)

    valid_mask = None
    if mode in ['val', 'train'] and valid_mask_name is not None:
        valid_mask_name = _get_file_list(valid_mask_name)
        valid_mask_name = _make_path_list(cfg, dir_name, valid_mask_name, rank)
        assert len(valid_mask_name) == len(img_name)
        valid_mask = [None]*len(valid_mask_name)

    pad_mode = cfg.DATASET.PAD_MODE
    volume = [None] * len(img_name)
    read_fn = readvol if not cfg.DATASET.LOAD_2D else readimg_as_vol
    for i in range(len(img_name)):
        volume[i] = read_fn(img_name[i])
        print(f"volume shape (original): {volume[i].shape}")
        if cfg.DATASET.NORMALIZE_RANGE:
            volume[i] = normalize_range(volume[i])
        if (np.array(cfg.DATASET.DATA_SCALE) != 1).any():
            volume[i] = zoom(volume[i], cfg.DATASET.DATA_SCALE, order=1)
        volume[i] = np.pad(volume[i], get_padsize(pad_size), pad_mode)
        print(f"volume shape (after scaling and padding): {volume[i].shape}")

        if mode in ['val', 'train'] and label is not None:
            label[i] = read_fn(label_name[i])
            if cfg.DATASET.LABEL_VAST:
                label[i] = vast2Seg(label[i])
            if label[i].ndim == 2:  # make it into 3D volume
                label[i] = label[i][None, :]
            if (np.array(cfg.DATASET.DATA_SCALE) != 1).any():
                label[i] = zoom(label[i], cfg.DATASET.DATA_SCALE, order=0)
            if cfg.DATASET.LABEL_BINARY and label[i].max() > 1:
                label[i] = label[i] // 255
            if cfg.DATASET.LABEL_MAG != 0:
                label[i] = (label[i]/cfg.DATASET.LABEL_MAG).astype(np.float32)

            label[i] = np.pad(label[i], get_padsize(pad_size), pad_mode)
            print(f"label shape: {label[i].shape}")

        if mode in ['val', 'train'] and valid_mask is not None:
            valid_mask[i] = read_fn(valid_mask_name[i])
            if (np.array(cfg.DATASET.DATA_SCALE) != 1).any():
                valid_mask[i] = zoom(
                    valid_mask[i], cfg.DATASET.DATA_SCALE, order=0)

            valid_mask[i] = np.pad(
                valid_mask[i], get_padsize(pad_size), pad_mode)
            print(f"valid_mask shape: {label[i].shape}")

    return volume, label, valid_mask


def get_dataset(cfg,
                augmentor,
                mode='train',
                rank=None,
                dir_name_init: Optional[list] = None,
                img_name_init: Optional[list] = None):
    r"""Prepare dataset for training and inference.
    """
    assert mode in ['train', 'val', 'test']

    sample_label_size = cfg.MODEL.OUTPUT_SIZE
    topt, wopt = ['0'], [['0']]
    if mode == 'train':
        sample_volume_size = augmentor.sample_size if augmentor is not None else cfg.MODEL.INPUT_SIZE
        sample_label_size = sample_volume_size
        sample_stride = (1, 1, 1)
        topt, wopt = cfg.MODEL.TARGET_OPT, cfg.MODEL.WEIGHT_OPT
        iter_num = cfg.SOLVER.ITERATION_TOTAL * cfg.SOLVER.SAMPLES_PER_BATCH
        if cfg.SOLVER.SWA.ENABLED:
            iter_num += cfg.SOLVER.SWA.BN_UPDATE_ITER

    elif mode == 'val':
        sample_volume_size = cfg.MODEL.INPUT_SIZE
        sample_label_size = sample_volume_size
        sample_stride = [x//2 for x in sample_volume_size]
        topt, wopt = cfg.MODEL.TARGET_OPT, cfg.MODEL.WEIGHT_OPT
        iter_num = -1

    elif mode == 'test':
        sample_volume_size = cfg.MODEL.INPUT_SIZE
        sample_stride = cfg.INFERENCE.STRIDE
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
        "reject_diversity": cfg.DATASET.REJECT_SAMPLING.DIVERSITY,
        "reject_p": cfg.DATASET.REJECT_SAMPLING.P,
        "data_mean": cfg.DATASET.MEAN,
        "data_std": cfg.DATASET.STD,
        "erosion_rates": cfg.MODEL.LABEL_EROSION,
        "dilation_rates": cfg.MODEL.LABEL_DILATION,
    }

    if cfg.DATASET.DO_CHUNK_TITLE == 1:  # build TileDataset
        label_json, valid_mask_json = None, None
        if mode == 'train':
            if cfg.DATASET.LABEL_NAME is not None:
                label_json = cfg.DATASET.INPUT_PATH + cfg.DATASET.LABEL_NAME
            if cfg.DATASET.VALID_MASK_NAME is not None:
                valid_mask_json = cfg.DATASET.INPUT_PATH + cfg.DATASET.VALID_MASK_NAME

        dataset = TileDataset(chunk_num=cfg.DATASET.DATA_CHUNK_NUM,
                              chunk_ind=cfg.DATASET.DATA_CHUNK_IND,
                              chunk_ind_split=cfg.DATASET.CHUNK_IND_SPLIT,
                              chunk_iter=cfg.DATASET.DATA_CHUNK_ITER,
                              chunk_stride=cfg.DATASET.DATA_CHUNK_STRIDE,
                              volume_json=cfg.DATASET.INPUT_PATH+cfg.DATASET.IMAGE_NAME,
                              label_json=label_json,
                              valid_mask_json=valid_mask_json,
                              pad_size=cfg.DATASET.PAD_SIZE,
                              **shared_kwargs)

    else:  # build VolumeDataset
        volume, label, valid_mask = _get_input(
            cfg, mode, rank, dir_name_init, img_name_init)
        dataset = VolumeDataset(volume=volume, label=label, valid_mask=valid_mask,
                                iter_num=iter_num, **shared_kwargs)

    return dataset


def build_dataloader(cfg, augmentor, mode='train', dataset=None, rank=None):
    r"""Prepare dataloader for training and inference.
    """
    assert mode in ['train', 'val', 'test']
    print('Mode: ', mode)

    if mode == 'train':
        cf = collate_fn_train
        batch_size = cfg.SOLVER.SAMPLES_PER_BATCH
    elif mode == 'val':
        cf = collate_fn_train
        batch_size = cfg.SOLVER.SAMPLES_PER_BATCH * 4
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

    # In PyTorch, each worker will create a copy of the Dataset, so if the data
    # is preload the data, the memory usage should increase a lot.
    # https://discuss.pytorch.org/t/define-iterator-on-dataloader-is-very-slow/52238/2
    img_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=cf,
        sampler=sampler, num_workers=num_workers, pin_memory=True)

    return img_loader
