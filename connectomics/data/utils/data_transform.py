from __future__ import print_function, division

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_holes
from skimage.measure import label as label_cc # avoid namespace conflict

def distance_transform_vol(label, quantize=True, mode='2d'):
    if mode == '3d':
        # calculate 3d distance transform
        vol_distance, vol_semantic = distance_transform(
            label, resolution=(1.0, 1.0, 1.0))
        if quantize:
            vol_distance = energy_quantize(vol_distance)
        return vol_distance

    vol_distance = []
    vol_semantic = []
    for i in range(label.shape[0]):
        label_img = label[i].copy()
        distance, semantic = distance_transform(label_img)
        vol_distance.append(distance)
        vol_semantic.append(semantic)

    vol_distance = np.stack(vol_distance, 0)
    vol_semantic = np.stack(vol_semantic, 0)
    if quantize:
        vol_distance = energy_quantize(vol_distance)
        
    return vol_distance

def distance_transform(label, bg_value=-1.0, relabel=True, resolution=(1.0, 1.0)):
    """Euclidean distance transform (DT or EDT).
    """
    eps = 1e-6

    if relabel:
        label = label_cc(label)

    label_shape = label.shape
    distance = np.zeros(label_shape, dtype=np.float32) + bg_value
    semantic = np.zeros(label_shape, dtype=np.uint8)

    indices = np.unique(label)
    if indices[0] == 0:
        if len(indices) > 1: # exclude background
            indices = indices[1:]
        else: # all-background sample
            return distance, semantic

    for idx in indices:
        temp1 = label.copy() == idx
        temp2 = remove_small_holes(temp1, 16, connectivity=1)

        semantic += temp2.astype(np.uint8)
        boundary_edt = distance_transform_edt(temp2, resolution)
        energy = boundary_edt / (boundary_edt.max() + eps) # normalize
        distance = np.maximum(distance, energy * temp2.astype(np.float32))

    return distance, semantic   

def energy_quantize(energy, levels=10):
    """Convert the continuous energy map into the quantized version.
    """
    # np.digitize returns the indices of the bins to which each 
    # value in input array belongs. The default behavior is bins[i-1] <= x < bins[i].
    bins = [-1.0]
    for i in range(levels):
        bins.append(float(i) / float(levels))
    bins.append(1.1)
    bins = np.array(bins)
    quantized = np.digitize(energy, bins) - 1
    return quantized.astype(np.int64)

def decode_quantize(output, mode='max'):
    # output: torch tensor of size (B, C, *)
    if mode == 'max':
        pred = torch.argmax(output, axis=1)
        max_value = output.size()[1]
        energy = pred / float(max_value)
    elif mode == 'average':
        out_shape = output.shape
        bins = np.array([0.1 * float(x-1) for x in range(11)])
        bins = torch.from_numpy(bins.astype(np.float32))
        bins = bins.view(1, -1, 1)
        bins = bins.to(output.device)

        output = output.view(out_shape[0], out_shape[1], -1) # (B, C, *)
        pred = torch.softmax(output, axis=1)
        energy = (pred*bins).view(out_shape).sum(1)

    return energy
