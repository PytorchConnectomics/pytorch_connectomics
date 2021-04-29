from __future__ import print_function, division
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from .loss import *
from .regularization import *
from ..utils import get_functional_act, SplitActivation

class Criterion(object):
    """Calculating losses and regularizations given the prediction, target and weight mask.

    Args:
        device (torch.device): model running device. GPUs are recommended for model training and inference.
        target_opt (List[str], optional): target options. Defaults to ['1'].
        loss_opt (List[List[str]], optional): loss options for the specified targets. Defaults to [['WeightedBCE']].
        output_act (List[List[str]], optional): activation functions for each loss option. Defaults to [['none']].
        loss_weight (List[List[float]], optional): the scalar weight of each loss. Defaults to [[1.]].
        regu_opt (Optional[List[str]], optional): regularization options. Defaults to None.
        regu_target (Optional[List[List[int]]], optional): indicies of predictions for applying regularization. Defaults to None.
        regu_weight (Optional[List[float]], optional): the scalar weight of each regularization. Defaults to None.
        do_2d (bool, optional): whether to conduct 2D training. Defaults to False.
    """
    loss_dict = {
        'WeightedMSE': WeightedMSE,
        'WeightedMAE': WeightedMAE,
        'WeightedBCE': WeightedBCE,
        'DiceLoss': DiceLoss,
        'WeightedCE': WeightedCE,
        'WeightedBCEWithLogitsLoss': WeightedBCEWithLogitsLoss,
        'WeightedLSBCEWithLogitsLoss': WeightedLSBCEWithLogitsLoss,
        'WeightedLSBCEFocalLoss': WeightedLSBCEFocalLoss
    }

    regu_dict = {
        'Binary': BinaryReg,
        'FgContour': FgContourConsistency,
        'ContourDT': ContourDTConsistency,
        'FgDT': ForegroundDTConsistency,
    }

    def __init__(self, 
                 device: torch.device, 
                 target_opt: List[str] = ['1'], 
                 loss_opt: List[List[str]] = [['WeightedBCE']], 
                 output_act: List[List[str]] = [['none']], 
                 loss_weight: List[List[float]] = [[1.]], 
                 regu_opt: Optional[List[str]] = None,
                 regu_target: Optional[List[List[int]]] = None,
                 regu_weight: Optional[List[float]] = None,
                 do_2d: bool = False):

        self.device = device
        self.target_opt = target_opt
        self.splitter = SplitActivation(target_opt, split_only=True, do_2d=do_2d)

        self.num_target = len(target_opt)
        self.num_regu = 0 if regu_opt is None else len(regu_opt)

        self.loss_opt = loss_opt
        self.loss_fn = self.get_loss(loss_opt)
        self.loss_w = loss_weight

        self.regu_opt = regu_opt
        self.regu_fn = self.get_regu(regu_opt)
        self.regu_t = regu_target
        self.regu_w = regu_weight

        self.act = self.get_act(output_act)

    def get_regu(self, regu_opt):
        regu = None
        if regu_opt is not None:
            regu = [None]*len(regu_opt)
            for i, ropt in enumerate(regu_opt):
                assert ropt in self.regu_dict
                regu[i] = self.regu_dict[ropt]()
        return regu

    def get_loss(self, loss_opt):
        out = [None]*self.num_target
        for i in range(self.num_target):
            out[i] = [None]*len(loss_opt[i])
            for j, lopt in enumerate(loss_opt[i]):
                assert lopt in self.loss_dict
                out[i][j] = self.loss_dict[lopt]()
        return out

    def get_act(self, output_act):
        out = [None]*self.num_target
        for i in range(self.num_target):
            out[i] = [None]*len(output_act[i])
            for j, act in enumerate(output_act[i]):
                out[i][j] = get_functional_act(act)
        return out

    def to_torch(self, data):
        if type(data) == torch.Tensor:
            return data.to(self.device, non_blocking=True)
        return torch.from_numpy(data).to(self.device)

    def __call__(self, pred, target, weight):
        # target, weight: torch.Tensor or numpy.ndarray
        # pred: torch.Tensor
        x = self.splitter(pred)

        loss = 0.0
        # Record individual losses and regularizations for
        # visualization in tensorboardX.
        losses_vis = {}
        for i in range(self.num_target): # iterate over tasks
            target_t = self.to_torch(target[i])
            for j in range(len(self.loss_fn[i])): # iterate over losses for a task
                w_mask = None if weight[i][j].shape[-1] == 1 else self.to_torch(weight[i][j])
                loss_temp = self.loss_w[i][j] * self.loss_fn[i][j](
                            self.act[i][j](x[i]), 
                            target=target_t, 
                            weight_mask=w_mask)
                loss += loss_temp
                loss_tag = self.target_opt[i] + '_' + self.loss_opt[i][j]
                losses_vis[loss_tag] = loss_temp

        for i in range(self.num_regu):
            targets = [x[j] for j in self.regu_t[i]]
            regu_temp = self.regu_w[i]*self.regu_fn[i](*targets)
            loss += regu_temp
            targets_name = [self.target_opt[j] for j in self.regu_t[i]]
            regu_tag = '_'.join(targets_name) + '_' + self.regu_opt[i] 
            losses_vis[regu_tag] = regu_temp
        
        return loss, losses_vis

    @classmethod
    def build_from_cfg(cls, cfg, device):
        """Build a Criterion class based on the config options.

        Args:
            cfg (yacs.config.CfgNode): YACS configuration options.
            device (torch.device): model running device type. GPUs are recommended for model training and inference.
        """
        return cls(device, cfg.MODEL.TARGET_OPT, cfg.MODEL.LOSS_OPTION, cfg.MODEL.OUTPUT_ACT, 
                   cfg.MODEL.LOSS_WEIGHT, cfg.MODEL.REGU_OPT, cfg.MODEL.REGU_TARGET, 
                   cfg.MODEL.REGU_WEIGHT, do_2d=cfg.DATASET.DO_2D)
