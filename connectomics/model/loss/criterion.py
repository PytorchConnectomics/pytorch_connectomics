from __future__ import print_function, division
from typing import Optional, List, Union, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

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
        loss_kwargs Optional[List[List[dict]]]: a list of kwargs given to the loss functions. Defaults to None.
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
                 loss_kwargs: Optional[List[List[dict]]] = None,
                 regu_opt: Optional[List[str]] = None,
                 regu_target: Optional[List[List[int]]] = None,
                 regu_weight: Optional[List[float]] = None,
                 do_2d: bool = False):

        self.device = device
        self.target_opt = target_opt
        self.splitter = SplitActivation(
            target_opt, split_only=True, do_2d=do_2d)

        self.num_target = len(target_opt)
        self.num_regu = 0 if regu_opt is None else len(regu_opt)

        self.loss_opt = loss_opt
        self.loss_fn = self.get_loss(loss_opt, loss_kwargs)
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

    def get_loss(self, loss_opt, loss_kwargs=None):
        out = [None]*self.num_target
        for i in range(self.num_target):
            out[i] = [None]*len(loss_opt[i])
            for j, lopt in enumerate(loss_opt[i]):
                params = None
                if loss_kwargs is not None:
                    params = loss_kwargs[i][j]
                out[i][j] = self.get_one_loss(lopt, params)
        return out

    def get_one_loss(self, lopt, params):
        assert lopt in self.loss_dict
        if params is None:
            return self.loss_dict[lopt]()

        # pass the kwargs to the corresponding loss function
        return self.loss_dict[lopt](**params)

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

    def evaluate(self,
                 pred: Tensor,
                 target: Union[List[Tensor], List[np.ndarray]],
                 weight: Union[List[Tensor], List[np.ndarray]],
                 key: Optional[str] = None,
                 losses_vis: dict = {},  # visualizing individual losses
                 ) -> Tuple[Tensor, dict]:
        # split the prediction for each target
        x = self.splitter(pred)

        loss = 0.0
        for i in range(self.num_target):
            target_t = self.to_torch(target[i])
            for j in range(len(self.loss_fn[i])):
                w_mask = None if weight[i][j].shape[-1] == 1 else self.to_torch(
                    weight[i][j])
                loss_temp = self.loss_w[i][j] * self.loss_fn[i][j](
                    self.act[i][j](x[i]),
                    target=target_t,
                    weight_mask=w_mask)
                loss += loss_temp
                loss_tag = self.target_opt[i] + '_' + \
                    self.loss_opt[i][j] + '_' + str(i)
                if key is not None:
                    loss_tag += '_' + key
                assert loss_tag not in losses_vis.keys()
                losses_vis[loss_tag] = loss_temp

        for i in range(self.num_regu):
            targets = [x[j] for j in self.regu_t[i]]
            regu_temp = self.regu_w[i]*self.regu_fn[i](*targets)
            loss += regu_temp
            targets_name = [self.target_opt[j] for j in self.regu_t[i]]
            regu_tag = '_'.join(targets_name) + '_' + \
                self.regu_opt[i] + '_' + str(i)
            if key is not None:
                regu_tag += '_' + key
            assert regu_tag not in losses_vis.keys()
            losses_vis[regu_tag] = regu_temp

        return loss, losses_vis

    def __call__(self,
                 pred: Union[Tensor, OrderedDict],
                 target: Union[List[Tensor], List[np.ndarray]],
                 weight: Union[List[Tensor], List[np.ndarray]],
                 ) -> Tuple[Tensor, dict]:

        losses_vis = {}
        if isinstance(pred, Tensor):
            # Python’s default arguments are evaluated once when the function is defined, not each time the function is
            # called (like it is in say, Ruby). This means that if you use a mutable default argument and mutate it, you
            # will and have mutated that object for all future calls to the function as well.
            # (According to https://docs.python-guide.org/writing/gotchas/)
            return self.evaluate(pred, target, weight, losses_vis=losses_vis)

        # evaluate OrderedDict predicted by DeepLab
        loss = 0.0
        for key in pred.keys():
            temp_loss, losses_vis = self.evaluate(
                pred[key], target, weight, key, losses_vis)
            loss += temp_loss

        return loss, losses_vis

    @classmethod
    def build_from_cfg(cls, cfg, device):
        """Build a Criterion class based on the config options.

        Args:
            cfg (yacs.config.CfgNode): YACS configuration options.
            device (torch.device): model running device type. GPUs are recommended for model training and inference.
        """
        loss_kwargs = None
        if cfg.MODEL.LOSS_KWARGS_KEY is not None:
            keys = cfg.MODEL.LOSS_KWARGS_KEY
            vals = cfg.MODEL.LOSS_KWARGS_VAL
            assert len(keys) == len(vals)
            loss_kwargs = [None] * len(keys)
            for i in range(len(keys)):
                assert len(keys[i]) == len(vals[i])
                loss_kwargs[i] = [None] * len(keys[i])
                for j in range(len(keys[i])):
                    if keys[i][j] is not None:
                        loss_kwargs[i][j] = dict(zip(keys[i][j], vals[i][j]))

        return cls(device, cfg.MODEL.TARGET_OPT, cfg.MODEL.LOSS_OPTION, cfg.MODEL.OUTPUT_ACT,
                   cfg.MODEL.LOSS_WEIGHT, loss_kwargs, cfg.MODEL.REGU_OPT, cfg.MODEL.REGU_TARGET,
                   cfg.MODEL.REGU_WEIGHT, do_2d=cfg.DATASET.DO_2D)
