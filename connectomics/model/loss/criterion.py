import numpy as np
import torch
import torch.nn as nn
from .loss import *
from ..utils import get_functional_act

def build_criterion(cfg, device):
    """Build a Criterion class based on the config options.

    Args:
        cfg (yacs.config.CfgNode): YACS configuration options.
        device (torch.device): model running device type. GPUs are recommended for model training and inference.
    """
    return Criterion(device, cfg.MODEL.TARGET_OPT, cfg.MODEL.LOSS_OPTION, cfg.MODEL.OUTPUT_ACT, 
                     cfg.MODEL.LOSS_WEIGHT, cfg.MODEL.REGU_OPT, cfg.MODEL.REGU_WEIGHT)

class Criterion(object):
    loss_dict = {
        'WeightedMSE': WeightedMSE,
        'WeightedBCE': WeightedBCE,
        'JaccardLoss': JaccardLoss,
        'DiceLoss': DiceLoss,
        'WeightedCE': WeightedCE,
        'WeightedBCEWithLogitsLoss': WeightedBCEWithLogitsLoss,
    }

    def __init__(self, device=0, target_opt=['1'], loss_opt=[['WeightedBCE']], 
                 output_act=[['none']], loss_weight=[[1.]], regu_opt=[], regu_weight=[]):
        self.device = device
        self.target_opt = target_opt
        self.loss_weight = loss_weight
        self.num_target = len(target_opt)
        self.num_regu = len(regu_opt)

        self.loss  = self.get_loss(loss_opt)
        self.loss_w = loss_weight
        self.regu = self.get_regu(regu_opt)
        self.regu_w = regu_weight

        self.act = self.get_act(output_act)

    def get_regu(self, regu_opt=[]):
        regu = None
        if len(regu_opt)>0:
            regu = [None]*len(regu_opt)
            for i in range(len(regu_opt)):
                if regu_opt[i] == 0:
                    regu[i] = nn.L1Loss()
                elif regu_opt[i] == 'BinaryReg':
                    regu[i] = BinaryReg()
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

    def eval(self, pred, target, weight):
        # target, weight: numpy.ndarray
        # pred: torch.Tensor
        loss = 0.0
        cid = 0 # channel index for prediction
        for i in range(self.num_target):
            # for each target
            numC = self.get_num_channel(i, target)
            target_t = self.to_torch(target[i])
            for j in range(len(self.loss[i])):
                if weight[i][j].shape[-1] == 1: # placeholder for no weight
                    loss += self.loss_weight[i][j]*self.loss_w[i][j]*self.loss[i][j](
                        self.act[i][j](pred[:,cid:cid+numC]), target_t)
                else:
                    loss += self.loss_weight[i][j]*self.loss_w[i][j]*self.loss[i][j](
                        self.act[i][j](pred[:,cid:cid+numC]), target_t, self.to_torch(weight[i][j]))
            cid += numC
        for i in range(self.num_regu):
            loss += self.regu[i](pred)*self.regu_w[i]
        return loss

    def get_num_channel(self, i, target):
        topt = self.target_opt[i]
        if topt[0] == '9': # generic segmantic segmentation
            numC = topt.split('-')[1]
            numC = int(numC)
        elif topt[0] == '5': # quantized distance transform
            numC = 11
        else:
            numC = target[i].shape[1]
        return numC
