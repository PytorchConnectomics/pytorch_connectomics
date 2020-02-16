import numpy as np
import torch
import torch.nn as nn

from ...data.utils import label_to_target, label_to_weight
from ..loss import *

class Criterion(object):
    def __init__(self, device=0, target_opt=[1], loss_opt=[0], loss_weight=[1.], regu_opt=[], regu_weight=[]):
        self.device = device
        self.target_opt = target_opt
        self.loss_opt = loss_opt
        self.loss_weight = loss_weight
        self.num_loss = len(loss_opt)
        self.num_regu = len(regu_opt)

        self.loss, self.loss_w = self.get_loss()
        self.regu = self.get_regu(regu_opt)
        self.regu_w = regu_weight


    def get_regu(self, regu_opt=[]):
        regu = None
        if len(regu_opt)>0:
            regu = [None]*len(regu_opt)
            for i in range(len(regu_opt)):
                if regu_opt == 0:
                    regu[i] = nn.L1Loss()
        return regu

    def get_loss(self):
        out = [None]*self.num_loss
        out_w = [None]*self.num_loss
        for lid,lopt in enumerate(self.loss_opt):
            if lopt in [0, 0.1]: # 0: weight, 0.1 no weighted
                out[lid] = [WeightedMSE()]
                out_w[lid] = [1.0]
            elif lopt in [1, 1.1]: # 1: weight, 1.1 no weighted
                out[lid] = [WeightedBCE()]
                out_w[lid] = [1.0]
            elif int(lopt) == 2:
                if lopt==2:
                    out[lid] = [JaccardLoss()]
                    out_w[lid] = [1.0]
                else:
                    ww = lopt-int(lopt)
                    out[lid] = [WeightedBCE(),JaccardLoss()]
                    out_w[lid] = [ww, 1.0-ww]
            elif lopt == 3:
                if lopt==3:
                    out[lid] = [DiceLoss()]
                    out_w[lid] = [1.0]
                else:
                    ww = lopt-int(lopt)
                    out[lid] = [WeightedBCE(),DiceLoss()]
                    out_w[lid] = [ww, 1.0-ww]
        return out, out_w


    def to_torch(self, data):
        return torch.from_numpy(data).to(self.device)

    def eval(self, label, pred, mask=None):
        # label: numpy
        # pred: torch
        # mask: torch
        # compute loss
        loss = 0
        pred_st = 0
        for i in range(self.num_loss):
            label_target = self.to_torch(label_to_target(self.target_opt[i], label))
            numC = label_target.shape[0]
            weight = label_to_weight(self.loss_opt[i], label, mask)
            for j in range(len(self.loss[i])):
                if weight[j] is None:
                    loss += self.loss_weight[i]*self.loss_w[i][j]*self.loss[i][j](pred[pred_st:pred_st+numC], label_target)
                else:
                    loss += self.loss_weight[i]*self.loss_w[i][j]*self.loss[i][j](pred[pred_st:pred_st+numC], label_target, self.to_torch(weight[j]))
            pred_st += numC

        for i in range(self.num_regu):
            loss += self.regu[i](pred)*self.regu_w[i]
        return loss
