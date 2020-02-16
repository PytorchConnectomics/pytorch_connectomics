from __future__ import print_function, division
import numpy as np
import random
import torch

####################################################################
## Collate Functions
####################################################################

def collate_fn_test(batch):
    pos, out_input = zip(*batch)
    out_input = np.stack(out_input, 0)
    return pos, out_input

def collate_fn(batch):
    """
    Puts each data field into a tensor with outer dimension batch size
    :param batch:
    :return:
    """
    pos, out_input, out_label, out_mask = zip(*batch)
    out_input = np.stack(out_input, 0)
    out_label = np.stack(out_label, 0)
    if out_mask[0].ndim==1:
        out_mask = None
    else:
        out_mask = np.stack(out_mask, 0)
    return pos, out_input, out_label, out_mask


def collate_fn_plus(batch):
    """
    Puts each data field into a tensor with outer dimension batch size
    :param batch:
    :return:
    """
    pos, out_input, out_label, others = zip(*batch)
    out_input = np.stack(out_input, 0)
    out_label = np.stack(out_label, 0)
    
    if len(others[0]) >1:
        extra = [None]*len(others[0])
        for i in range(len(others[0])):
            extra[i] = np.stack([others[x][i] for x in range(len(others))], 0)
    else: # just one more var
        extra = np.stack(others[0], 0)

    return pos, out_input, out_label, extra

def collate_fn_skel(batch):
    """
    Puts each data field into a tensor with outer dimension batch size
    :param batch:
    :return:
    """
    pos, out_input, out_label, weights, out_distance, out_skeleton = zip(*batch)
    out_input = np.stack(out_input, 0)
    out_label = np.stack(out_label, 0)
    weights = np.stack(weights, 0)
    out_distance = np.stack(out_distance, 0)
    out_skeleton = np.stack(out_skeleton, 0)

    return pos, out_input, out_label, weights, out_distance, out_skeleton

def collate_fn_long_range(batch):
    """
    Puts each data field into a tensor with outer dimension batch size
    :param batch:
    :return:
    """
    pos, out_input, out_label = zip(*batch)

    out_input = np.stack(out_input, 0)
    out_label = np.stack(out_label, 0)
    return pos
