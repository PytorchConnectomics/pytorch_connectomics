import numpy as np
import torch
import torch.nn as nn

from ..loss import *

class Criterion(object):
    def __init__(self, device=0, target_opt=['1'], loss_opt=[['WeightedBCE']], loss_weight=[[1.]], regu_opt=[], regu_weight=[]):
        self.device = device
        self.target_opt = target_opt
        self.loss_opt = loss_opt
        self.loss_weight = loss_weight
        self.num_target = len(target_opt)
        self.num_regu = len(regu_opt)

        self.loss  = self.get_loss()
        self.loss_w = loss_weight
        self.regu = self.get_regu(regu_opt)
        self.regu_w = regu_weight

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

    def get_loss(self):
        out = [None]*self.num_target
        for i in range(self.num_target):
            out[i] = [None]*len(self.loss_opt[i])
            for j,lopt in enumerate(self.loss_opt[i]):
                if lopt == 'WeightedMSE':
                    out[i][j] = WeightedMSE()
                elif lopt == 'WeightedBCE':
                    out[i][j] = WeightedBCE()
                elif lopt == 'JaccardLoss':
                    out[i][j] = JaccardLoss()
                elif lopt == 'DiceLoss':
                    out[i][j] = DiceLoss()
                elif lopt == 'WeightedCE':
                    out[i][j] = WeightedCE()
                else:
                    print('Unknown loss option {}'.format(lopt))
        return out

    def to_torch(self, data):
        return torch.from_numpy(data).to(self.device)

    def eval(self, pred, target, weight):
        # target, weight: numpy
        # pred: torch
        # compute loss
        loss = 0
        cid = 0 # channel index for prediction
        for i in range(self.num_target):
            # for each target
            numC = self.get_num_channel(i, target)
            target_t = self.to_torch(target[i])
            for j in range(len(self.loss[i])):
                if weight[i][j].shape[-1] == 1: # placeholder for no weight
                    loss += self.loss_weight[i][j]*self.loss_w[i][j]*self.loss[i][j](pred[:,cid:cid+numC], target_t)
                else:
                    loss += self.loss_weight[i][j]*self.loss_w[i][j]*self.loss[i][j](pred[:,cid:cid+numC], target_t, self.to_torch(weight[i][j]))
            cid += numC
        for i in range(self.num_regu):
            loss += self.regu[i](pred)*self.regu_w[i]
        return loss

    def get_num_channel(self, i, target):
        topt = self.target_opt[i]
        if topt[0] == '9': # generic segmantic segmentation
            numC = topt.split('-')[1]
            numC = int(numC)
        else:
            numC = target[i].shape[1]
        return numC
