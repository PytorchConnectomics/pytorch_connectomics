from __future__ import print_function, division
from typing import Union, List

import os
import math
import glob
import copy
import numpy as np

import torch
import torch.utils.data

from connectomics.data.dataset.dataset_cond import VolumeDatasetCond
from connectomics.data.dataset.collate import *
from connectomics.data.utils import *
from connectomics.data.dataset.build import _get_file_list, _make_path_list, _rescale, _pad

from .collate import collate_fn_test_cond

def _get_input(cfg,
               mode='train',
               rank=None,
               dir_name_init: Optional[list] = None,
               img_name_init: Optional[list] = None):
    r"""Load the inputs specified by the configuration options.
    """
    assert mode in ['train', 'test']
    dir_path = cfg.DATASET.INPUT_PATH
    if dir_name_init is not None:
        dir_name = dir_name_init
    else:
        dir_name = _get_file_list(dir_path)

    if mode == 'test':
        img_name = cfg.DATASET.IMAGE_NAME
        # path to the 3D volume with the 2D seed annotations
        label_name = cfg.CONDITIONAL.INFERENCE_CONDITIONAL 
        pad_size = cfg.DATASET.PAD_SIZE
    else:
        img_name = cfg.DATASET.IMAGE_NAME
        label_name = cfg.DATASET.LABEL_NAME
        pad_size = cfg.DATASET.PAD_SIZE

    if img_name_init is not None:
        img_name = img_name_init
    else:
        img_name = _get_file_list(img_name, prefix=dir_path)
    img_name = _make_path_list(cfg, dir_name, img_name, rank)
    print(rank, len(img_name), list(map(os.path.basename, img_name)))

    label = None
    # we also require the label/seed volume during testing
    if mode in ['train', 'test'] and label_name is not None:
        label_name = _get_file_list(label_name, prefix=dir_path)
        label_name = _make_path_list(cfg, dir_name, label_name, rank)
        assert len(label_name) == len(img_name)
        label = [None]*len(label_name)

    pad_mode = cfg.DATASET.PAD_MODE
    volume = [None] * len(img_name)
    read_fn = readvol if not cfg.DATASET.LOAD_2D else readimg_as_vol

    for i in range(len(img_name)):
        volume[i] = read_fn(img_name[i], drop_channel=cfg.DATASET.DROP_CHANNEL)
        print(f"volume shape (original): {volume[i].shape}")
        if cfg.DATASET.NORMALIZE_RANGE:
            volume[i] = normalize_range(volume[i])
        volume[i] = _rescale(volume[i], cfg.DATASET.IMAGE_SCALE, order=3)
        volume[i] = _pad(volume[i], pad_size, pad_mode)
        print(f"volume shape (after scaling and padding): {volume[i].shape}")

        # we also require the label/seed volume during testing
        if mode in ['train', 'test'] and label is not None:
            label[i] = read_fn(label_name[i], drop_channel=cfg.DATASET.DROP_CHANNEL)            
            label[i] = _pad(label[i], pad_size, pad_mode)

            print(f"label shape (after scaling and padding): {label[i].shape}")
            if cfg.DATASET.LOAD_2D:
                assert volume[i].shape[1:] == label[i].shape[1:]
            else:
                assert volume[i].shape == label[i].shape[-3:]

    return volume, label


def get_dataset(cfg,
                augmentor,
                mode='train',
                rank=None,
                dataset_class=VolumeDatasetCond,
                dir_name_init: Optional[list] = None,
                img_name_init: Optional[list] = None):
    r"""Prepare dataset for training and inference.
    """
    assert mode in ['train', 'test']

    sample_label_size = cfg.MODEL.OUTPUT_SIZE
    topt, wopt = ['0'], [['0']]
    if mode == 'train':
        sample_volume_size = augmentor.sample_size if augmentor is not None else cfg.MODEL.INPUT_SIZE
        sample_label_size = sample_volume_size
        topt, wopt = cfg.MODEL.TARGET_OPT, cfg.MODEL.WEIGHT_OPT
        iter_num = cfg.SOLVER.ITERATION_TOTAL * cfg.SOLVER.SAMPLES_PER_BATCH
        if cfg.SOLVER.SWA.ENABLED:
            iter_num += cfg.SOLVER.SWA.BN_UPDATE_ITER
    elif mode == 'test':
        sample_volume_size = cfg.MODEL.INPUT_SIZE
        iter_num = -1

    volume, label = _get_input(
        cfg, mode, rank, dir_name_init, img_name_init)

    # dataset kwargs
    kwargs = {
        "label_type": cfg.CONDITIONAL.LABEL_TYPE,
        "sample_size": sample_volume_size,
        "augmentor": augmentor,
        "weight_opt": wopt,
        "mode": mode,
        "data_mean": cfg.DATASET.MEAN,
        "data_std": cfg.DATASET.STD
        }

    dataset = dataset_class(label=[syn_sem2inst(ll) for ll in label],
                            volume=volume, iter_num=iter_num, **kwargs)

    return dataset

def build_dataloader(cfg, augmentor=None, mode='train', dataset=None, rank=None,
                     dataset_class=VolumeDatasetCond, cf=collate_fn_train):
    r"""Prepare dataloader for training and inference.
    """
    assert mode in ['train', 'test']

    if mode == 'train':
        batch_size = cfg.SOLVER.SAMPLES_PER_BATCH
    else:
        cf = collate_fn_test_cond # update the collate function
        batch_size = cfg.INFERENCE.SAMPLES_PER_BATCH * cfg.SYSTEM.NUM_GPUS

    if dataset is None: # no pre-defined dataset instance
        dataset = get_dataset(cfg, augmentor, mode, rank, dataset_class)

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
