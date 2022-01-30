from __future__ import print_function, division
from typing import Optional, Tuple

import torch
import scipy
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_holes
from skimage.measure import label as label_cc  # avoid namespace conflict

from .data_misc import get_padsize, array_unpad

__all__ = [
    'edt_semantic',
    'edt_instance',
    'decode_quantize',
]


def edt_semantic(
        label: np.ndarray,
        mode: str = '2d',
        alpha_fore: float = 8.0,
        alpha_back: float = 50.0):
    """Euclidean distance transform (DT or EDT) for binary semantic mask.
    """
    assert mode in ['2d', '3d']
    do_2d = (label.ndim == 2)

    resolution = (6.0, 1.0, 1.0)  # anisotropic data
    if mode == '2d' or do_2d:
        resolution = (1.0, 1.0)

    fore = (label != 0).astype(np.uint8)
    back = (label == 0).astype(np.uint8)

    if mode == '3d' or do_2d:
        fore_edt = _edt_binary_mask(fore, resolution, alpha_fore)
        back_edt = _edt_binary_mask(back, resolution, alpha_back)
    else:
        fore_edt = [_edt_binary_mask(fore[i], resolution, alpha_fore)
                    for i in range(label.shape[0])]
        back_edt = [_edt_binary_mask(back[i], resolution, alpha_back)
                    for i in range(label.shape[0])]
        fore_edt, back_edt = np.stack(fore_edt, 0), np.stack(back_edt, 0)
    distance = fore_edt - back_edt
    return np.tanh(distance)


def _edt_binary_mask(mask, resolution, alpha):
    if (mask == 1).all():  # tanh(5) = 0.99991
        return np.ones_like(mask).astype(float) * 5

    return distance_transform_edt(mask, resolution) / alpha


def edt_instance(label: np.ndarray,
                 mode: str = '2d',
                 quantize: bool = True,
                 resolution: Tuple[float] = (1.0, 1.0, 1.0),
                 padding: bool = False):
    assert mode in ['2d', '3d']
    if mode == '3d':
        # calculate 3d distance transform for instances
        vol_distance, vol_semantic = distance_transform(
            label, resolution=resolution, padding=padding)
        if quantize:
            vol_distance = energy_quantize(vol_distance)
        return vol_distance

    vol_distance = []
    vol_semantic = []
    for i in range(label.shape[0]):
        label_img = label[i].copy()
        distance, semantic = distance_transform(label_img, padding=padding)
        vol_distance.append(distance)
        vol_semantic.append(semantic)

    vol_distance = np.stack(vol_distance, 0)
    vol_semantic = np.stack(vol_semantic, 0)
    if quantize:
        vol_distance = energy_quantize(vol_distance)

    return vol_distance


def distance_transform(label: np.ndarray,
                       bg_value: float = -1.0,
                       relabel: bool = True,
                       padding: bool = False,
                       resolution: Tuple[float] = (1.0, 1.0)):
    """Euclidean distance transform (DT or EDT) for instance masks.
    """
    eps = 1e-6
    pad_size = 2

    if relabel:
        label = label_cc(label)

    if padding:
        # The distance_transform_edt function does not treat image border
        # as background. If image border needs to be considered as background
        # in distance calculation, set padding to True.
        label = np.pad(label, pad_size, mode='constant', constant_values=0)

    label_shape = label.shape
    all_bg_sample = False
    distance = np.zeros(label_shape, dtype=np.float32) + bg_value
    semantic = np.zeros(label_shape, dtype=np.uint8)

    indices = np.unique(label)
    if indices[0] == 0:
        if len(indices) > 1:  # exclude background
            indices = indices[1:]
        else:  # all-background sample
            all_bg_sample = True

    if not all_bg_sample:
        for idx in indices:
            temp1 = label.copy() == idx
            temp2 = remove_small_holes(temp1, 16, connectivity=1)

            semantic += temp2.astype(np.uint8)
            boundary_edt = distance_transform_edt(temp2, resolution)
            energy = boundary_edt / (boundary_edt.max() + eps)  # normalize
            distance = np.maximum(distance, energy * temp2.astype(np.float32))

    if padding:
        # Unpad the output array to preserve original shape.
        distance = array_unpad(distance, get_padsize(
            pad_size, ndim=distance.ndim))
        semantic = array_unpad(semantic, get_padsize(
            pad_size, ndim=distance.ndim))

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
    assert type(output) in [torch.Tensor, np.ndarray]
    assert mode in ['max', 'mean']
    if type(output) == torch.Tensor:
        return _decode_quant_torch(output, mode)
    else:
        return _decode_quant_numpy(output, mode)


def _decode_quant_torch(output, mode='max'):
    # output: torch tensor of size (B, C, *)
    if mode == 'max':
        pred = torch.argmax(output, axis=1)
        max_value = output.size()[1]
        energy = pred / float(max_value)
    elif mode == 'mean':
        out_shape = output.shape
        bins = np.array([0.1 * float(x-1) for x in range(11)])
        bins = torch.from_numpy(bins.astype(np.float32))
        bins = bins.view(1, -1, 1)
        bins = bins.to(output.device)

        output = output.view(out_shape[0], out_shape[1], -1)  # (B, C, *)
        pred = torch.softmax(output, axis=1)
        energy = (pred*bins).view(out_shape).sum(1)

    return energy


def _decode_quant_numpy(output, mode='max'):
    # output: numpy array of shape (C, *)
    if mode == 'max':
        pred = np.argmax(output, axis=0)
        max_value = output.shape[0]
        energy = pred / float(max_value)
    elif mode == 'mean':
        out_shape = output.shape
        bins = np.array([0.1 * float(x-1) for x in range(11)])
        bins = bins.reshape(-1, 1)

        output = output.reshape(out_shape[0], -1)  # (C, *)
        pred = scipy.special.softmax(output, axis=0)
        energy = (pred*bins).reshape(out_shape).sum(0)

    return energy
