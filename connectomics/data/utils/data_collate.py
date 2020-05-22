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

def collate_fn_target(batch):
    """
    Puts each data field into a tensor with outer dimension batch size
    :param batch:
    :return:
    """
    pos, out_input, out_target, out_weight = zip(*batch)
    out_input = np.stack(out_input, 0)
    out_target_l = [None]*len(out_target[0]) 
    out_weight_l = [[None]*len(out_weight[0][x]) for x in range(len(out_weight[0]))] 
   
    for i in range(len(out_target[0])):
        out_target_l[i] = np.stack([out_target[x][i] for x in range(len(out_target))], 0)

    # each target can have multiple loss/weights
    for i in range(len(out_weight[0])):
        for j in range(len(out_weight[0][i])):
            out_weight_l[i][j] = np.stack([out_weight[x][i][j] for x in range(len(out_weight))], 0)

    return pos, out_input, out_target_l, out_weight_l

# def collate_fn(batch):
#     """
#     Puts each data field into a tensor with outer dimension batch size
#     :param batch:
#     :return:
#     """
#     pos, out_input, out_label, out_mask = zip(*batch)
#     out_input = np.stack(out_input, 0)
#     out_label = np.stack(out_label, 0)
#     if out_mask[0].ndim==1:
#         out_mask = None
#     else:
#         out_mask = np.stack(out_mask, 0)
#     return pos, out_input, out_label, out_mask

# def collate_fn_skel(batch):
#     """
#     Puts each data field into a tensor with outer dimension batch size
#     :param batch:
#     :return:
#     """
#     pos, out_input, out_label, weights, out_distance, out_skeleton = zip(*batch)
#     out_input = np.stack(out_input, 0)
#     out_label = np.stack(out_label, 0)
#     weights = np.stack(weights, 0)
#     out_distance = np.stack(out_distance, 0)
#     out_skeleton = np.stack(out_skeleton, 0)

#     return pos, out_input, out_label, weights, out_distance, out_skeleton

# def collate_fn_long_range(batch):
#     """
#     Puts each data field into a tensor with outer dimension batch size
#     :param batch:
#     :return:
#     """
#     pos, out_input, out_label = zip(*batch)

#     out_input = np.stack(out_input, 0)
#     out_label = np.stack(out_label, 0)
#     return pos
