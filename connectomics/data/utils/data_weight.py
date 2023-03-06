from __future__ import print_function, division
from typing import Optional, Union, List
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_dilation
from .data_misc import split_masks


def seg_to_weights(targets, wopts, mask=None, seg=None):
    # input: list of targets
    out = [None]*len(wopts)
    for wid, wopt in enumerate(wopts):
        out[wid] = seg_to_weight(targets[wid], wopt, mask, seg)
    return out


def seg_to_weight(target, wopts, mask=None, seg=None):
    out = [None]*len(wopts)
    foo = np.zeros((1), int)
    for wid, wopt in enumerate(wopts):
        if wopt[0] == '1':  # 1: by gt-target ratio
            dilate = (wopt == '1-1')
            out[wid] = weight_binary_ratio(target.copy(), None if mask is None else mask.copy(), dilate)
        elif wopt[0] == '2':  # 2: unet weight
            assert seg is not None
            _, w0, w1 = wopt.split('-')
            out[wid] = weight_unet3d(seg, float(w0), float(w1))
        elif mask is not None:  # valid region only
            out[wid] = (mask!=0).astype(np.float32)[np.newaxis, :]
        else:  # no weight map
            out[wid] = foo
    return out


def weight_binary_ratio(label, mask=None, dilate=False):
    if label.max() == label.min():
        # uniform weights for single-label volume
        if mask is not None:
            return (mask!=0).astype(np.float32)[np.newaxis, :]
        return np.ones_like(label, np.float32)

    # Generate weight map by balancing the foreground and background.
    min_ratio = 5e-2
    label = (label!=0).astype(np.float64)  # foreground
    if mask is not None:
        mask = mask.astype(label.dtype)[np.newaxis, :]
        ww = (label*mask).sum() / mask.sum()
    else:
        ww = label.sum() / np.prod(label.shape)
    ww = np.clip(ww, a_min=min_ratio, a_max=1-min_ratio)
    weight_factor = max(ww, 1-ww)/min(ww, 1-ww)

    if dilate:  # Use higher weights for regions close to foreground.
        N = label.ndim
        assert N in [3, 4]
        struct = np.ones([1]*(N-2) + [3, 3])

        label = (label != 0)
        label = binary_dilation(label, struct).astype(np.float64)

    # Case 1 -- Affinity Map
    # In that case, ww is large (i.e., ww > 1 - ww), which means the high weight
    # factor should be applied to background pixels.

    # Case 2 -- Contour Map
    # In that case, ww is small (i.e., ww < 1 - ww), which means the high weight
    # factor should be applied to foreground pixels.

    if ww > 1-ww:
        # Switch when foreground is the dominate class.
        label = 1 - label
    weight = weight_factor*label + (1-label)

    if mask is not None:
        weight = weight * mask

    return weight.astype(np.float32)


def weight_unet3d(seg, w0=10.0, w1=5.0, sigma=5):
    out = np.ones_like(seg).astype(np.float32)
    zid = np.where((seg > 0).max(axis=1).max(axis=1) > 0)[0]
    for z in zid:
        out[z] = weight_unet2d(seg[z], w0, w1, sigma)
    return out[np.newaxis]


def weight_unet2d(seg, w0=10.0, w1=5.0, sigma=5):
    min_val = 1.0
    max_val = max(w0, w1)

    masks = split_masks(seg)
    N, H, W = masks.shape
    if N < 2:  # Number of foreground segments is smaller than 2.
        weight_map = (seg != 0).astype(np.float32) * w1
        return np.clip(weight_map, min_val, max_val)

    distance = []
    foreground = np.zeros((H, W), dtype=np.uint8)
    for i in range(N):
        binary = (masks[i] != 0).astype(np.uint8)
        foreground = np.maximum(foreground, binary)
        dist = distance_transform_edt(1-binary)
        distance.append(dist)

    distance = np.stack(distance, 0)
    distance = np.partition(distance, 1, axis=0)
    d1 = distance[0, :, :]
    d2 = distance[1, :, :]
    weight_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))
    weight_map = weight_map * (1-foreground).astype(np.float32)
    weight_map += foreground.astype(np.float32) * w1

    return np.clip(weight_map, min_val, max_val)
