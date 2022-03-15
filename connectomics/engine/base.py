from __future__ import print_function, division
from typing import Optional
import warnings

import numpy as np
from yacs.config import CfgNode

import torch

class TrainerBase(object):
    def __init__(self,
                 cfg: CfgNode,
                 device: torch.device,
                 mode: str = 'train',
                 rank: Optional[int] = None):

        assert mode in ['train', 'test']
        self.cfg = cfg
        self.device = device
        self.output_dir = cfg.DATASET.OUTPUT_PATH
        self.mode = mode
        self.rank = rank
        self.is_main_process = rank is None or rank == 0
        self.inference_singly = (mode == 'test') and cfg.INFERENCE.DO_SINGLY

    def to_device(self, *args):
        if len(args) == 1:
            return args[0].to(self.device, non_blocking=True)
        return [x.to(self.device, non_blocking=True) for x in args]
