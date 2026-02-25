"""
Segmentation processing functions for PyTorch Connectomics.
"""

from typing import List, Optional, Union

import cc3d
import numpy as np
import torch
from skimage.morphology import binary_dilation, dilation, erosion

RATES_TYPE = Optional[Union[List[int], int]]


def im_to_col(volume, kernel_size, stride=1):
    """Extract patches from volume using sliding window."""
    # Parameters
    M, N = volume.shape
    # Get Starting block indices
    start_idx = np.arange(0, M - kernel_size[0] + 1, stride)[:, None] * N + np.arange(
        0, N - kernel_size[1] + 1, stride
    )
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(kernel_size[0])[:, None] * N + np.arange(kernel_size[1])
    # Get all actual indices & index into input array for final output
    return np.take(volume, start_idx.ravel()[:, None] + offset_idx.ravel())


def seg_erosion_instance(seg, tsz_h=1):
    # Kisuk Lee's thesis (A.1.4):
    # "we preprocessed the ground truth seg such that any voxel centered on a
    # 3 × 3 × 1 window containing more than one positive segment ID (zero is
    # reserved for background) is marked as background."
    # seg=0: background
    tsz = 2 * tsz_h + 1
    sz = seg.shape
    is_tensor = torch.is_tensor(seg)

    def _to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _apply_mask(x, mask):
        if torch.is_tensor(x):
            mask_t = torch.as_tensor(mask, device=x.device, dtype=x.dtype)
            return x * mask_t
        return x * mask

    if len(sz) == 3:
        for z in range(sz[0]):
            seg_z = _to_numpy(seg[z]) if is_tensor else seg[z]
            mm = seg_z.max()
            patch = im_to_col(
                np.pad(seg_z, ((tsz_h, tsz_h), (tsz_h, tsz_h)), "reflect"), [tsz, tsz]
            )
            p0 = patch.max(axis=1)
            patch[patch == 0] = mm + 1
            p1 = patch.min(axis=1)
            seg[z] = _apply_mask(seg[z], (p0 == p1).reshape(sz[1:]))
    else:
        seg_np = _to_numpy(seg) if is_tensor else seg
        mm = seg_np.max()
        patch = im_to_col(np.pad(seg_np, ((tsz_h, tsz_h), (tsz_h, tsz_h)), "reflect"), [tsz, tsz])
        p0 = patch.max(axis=1)
        patch[patch == 0] = mm + 1
        p1 = patch.min(axis=1)
        seg = _apply_mask(seg, (p0 == p1).reshape(sz))
    return seg


def seg_to_small_seg(seg, thres=25, rr=2):
    # rr: z/x-y resolution ratio
    sz = seg.shape
    mask = np.zeros(sz, np.uint8)
    for z in np.where(seg.max(axis=1).max(axis=1) > 0)[0]:
        tmp = cc3d.connected_components(seg[z])
        ui, uc = np.unique(tmp, return_counts=True)
        rl = np.zeros(ui[-1] + 1, np.uint8)
        rl[ui[uc < thres]] = 1
        rl[0] = 0
        mask[z] += rl[tmp]
    for y in np.where(seg.max(axis=2).max(axis=0) > 0)[0]:
        tmp = cc3d.connected_components(seg[:, y])
        ui, uc = np.unique(tmp, return_counts=True)
        rl = np.zeros(ui[-1] + 1, np.uint8)
        rl[ui[uc < thres // rr]] = 1
        rl[0] = 0
        mask[:, y] += rl[tmp]
    for x in np.where(seg.max(axis=0).max(axis=0) > 0)[0]:
        tmp = cc3d.connected_components(seg[:, :, x])
        ui, uc = np.unique(tmp, return_counts=True)
        rl = np.zeros(ui[-1] + 1, np.uint8)
        rl[ui[uc < thres // rr]] = 1
        rl[0] = 0
        mask[:, :, x] += rl[tmp]
    return mask


def seg_markInvalid(seg, iter_num=2, do_2d=True):
    # find invalid
    # if do erosion(seg==0), then miss the border
    if do_2d:
        stel = np.array([[1, 1, 1], [1, 1, 1]]).astype(bool)
        if len(seg.shape) == 2:
            out = binary_dilation(seg > 0, structure=stel, iterations=iter_num)
            seg[out == 0] = -1
        else:  # save memory
            for z in range(seg.shape[0]):
                tmp = seg[z]  # by reference
                out = binary_dilation(tmp > 0, structure=stel, iterations=iter_num)
                tmp[out == 0] = -1
    else:
        stel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(bool)
        out = binary_dilation(seg > 0, structure=stel, iterations=iter_num)
        seg[out == 0] = -1
    return seg


def seg_erosion(label: np.ndarray, index: int, erosion_rates: RATES_TYPE = None):
    if erosion_rates is None:
        return label

    label_erosion = erosion_rates
    if isinstance(label_erosion, list):
        label_erosion = label_erosion[index]
    return erosion(label, label_erosion)


def seg_dilation(label: np.ndarray, index: int, dilation_rates: RATES_TYPE = None):
    if dilation_rates is None:
        return label

    label_dilation = dilation_rates
    if isinstance(label_dilation, list):
        label_dilation = label_dilation[index]

    tsz = 2 * label_dilation + 1
    assert label.ndim in [2, 3]
    shape = (1, tsz, tsz) if label.ndim == 3 else (tsz, tsz)
    return dilation(label, np.ones(shape, dtype=label.dtype))


def seg_selection(label, indices):
    mid = label.max() + 1
    relabel = np.zeros(mid + 1, label.dtype)
    relabel[indices] = np.arange(1, len(indices) + 1)
    return relabel[label]


__all__ = [
    "im_to_col",
    "seg_erosion_instance",
    "seg_markInvalid",
    "seg_erosion",
    "seg_dilation",
    "seg_selection",
]
