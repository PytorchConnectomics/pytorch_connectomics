from __future__ import print_function, division
import numpy as np
import random
import torch

####################################################################
## Collate Functions
####################################################################

def collate_fn(batch):
    """
    Puts each data field into a tensor with outer dimension batch size
    :param batch:
    :return:
    """
    pos, out_input, out_label, weights, weight_factor = zip(*batch)
    out_input = torch.stack(out_input, 0)
    out_label = torch.stack(out_label, 0)
    weights = torch.stack(weights, 0)
    weight_factor = np.stack(weight_factor, 0)
    return pos, out_input, out_label, weights, weight_factor

def collate_fn_test(batch):
    pos, out_input = zip(*batch)
    out_input = torch.stack(out_input, 0)
    return pos, out_input

def collate_fn_plus(batch):
    """
    Puts each data field into a tensor with outer dimension batch size
    :param batch:
    :return:
    """
    pos, out_input, out_label, weights, weight_factor, others = zip(*batch)
    out_input = torch.stack(out_input, 0)
    out_label = torch.stack(out_label, 0)
    weights = torch.stack(weights, 0)
    weight_factor = np.stack(weight_factor, 0)

    extra = [None]*len(others)
    for i in range(len(others)):
        extra[i] = torch.stack(others[i], 0)

    return pos, out_input, out_label, weights, weight_factor, extra

def collate_fn_skel(batch):
    """
    Puts each data field into a tensor with outer dimension batch size
    :param batch:
    :return:
    """
    pos, out_input, out_label, weights, weight_factor, out_distance, out_skeleton = zip(*batch)
    out_input = torch.stack(out_input, 0)
    out_label = torch.stack(out_label, 0)
    weights = torch.stack(weights, 0)
    weight_factor = np.stack(weight_factor, 0)
    out_distance = np.stack(out_distance, 0)
    out_skeleton = np.stack(out_skeleton, 0)

    return pos, out_input, out_label, weights, weight_factor, out_distance, out_skeleton
