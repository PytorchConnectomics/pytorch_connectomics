import numpy as np
from scipy.ndimage import convolve
import torch
from torch.nn.functional import conv2d, conv3d

# Process image stacks.


def count_volume(data_sz, vol_sz, stride):
    return 1 + np.ceil((data_sz - vol_sz) / stride.astype(float)).astype(int)


def crop_volume(data, sz, st=(0, 0, 0)):
    # must be (z, y, x) or (c, z, y, x) format
    assert data.ndim in [3, 4]
    st = np.array(st).astype(np.int32)

    if data.ndim == 3:
        return data[st[0]:st[0] + sz[0], st[1]:st[1] + sz[1], st[2]:st[2] + sz[2]]
    else:  # crop spatial dimensions
        return data[:, st[0]:st[0] + sz[0], st[1]:st[1] + sz[1], st[2]:st[2] + sz[2]]


def get_valid_pos_torch(mask, vol_sz, valid_ratio):
    # torch version
    # bug: out of memory
    valid_thres = valid_ratio * np.prod(vol_sz)
    mask.shape
    if len(vol_sz) == 3:
        mask_sum = (
            conv3d(
                torch.from_numpy(mask[None, None].astype(int)),
                torch.ones(tuple(vol_sz))[None, None],
                padding="valid",
            )[0, 0].numpy()
            >= valid_thres
        )
        zz, yy, xx = np.meshgrid(
            np.arange(mask_sum.shape[0]), np.arange(mask_sum.shape[1]), np.arange(mask_sum.shape[2])
        )
        valid_pos = np.stack([zz.T[mask_sum], yy.T[mask_sum], xx.T[mask_sum]], axis=1)
    else:
        mask_sum = (
            conv2d(
                torch.from_numpy(mask[None, None].astype(int)),
                torch.ones(tuple(vol_sz))[None, None],
                padding="valid",
            )[0, 0].numpy()
            >= valid_thres
        )
        yy, xx = np.meshgrid(np.arange(mask_sum.shape[0]), np.arange(mask_sum.shape[1]))
        valid_pos = np.stack([yy.T[mask_sum], xx.T[mask_sum]], axis=1)
    return valid_pos


def get_valid_pos(mask, vol_sz, valid_ratio):
    # scipy version
    valid_thres = valid_ratio * np.prod(vol_sz)
    data_sz = mask.shape
    mask_sum = convolve(mask.astype(int), np.ones(vol_sz), mode="constant", cval=0)
    pad_sz_pre = (np.array(vol_sz) - 1) // 2
    pad_sz_post = data_sz - (vol_sz - pad_sz_pre - 1)
    valid_pos = np.zeros([0, 3])
    if len(vol_sz) == 3:
        mask_sum = (
            mask_sum[
                pad_sz_pre[0]:pad_sz_post[0],
                pad_sz_pre[1]:pad_sz_post[1],
                pad_sz_pre[2]:pad_sz_post[2],
            ]
            >= valid_thres
        )
        if mask_sum.max() > 0:
            zz, yy, xx = np.meshgrid(
                np.arange(mask_sum.shape[0]),
                np.arange(mask_sum.shape[1]),
                np.arange(mask_sum.shape[2]),
            )
            valid_pos = np.stack(
                [
                    zz.transpose([1, 0, 2])[mask_sum],
                    yy.transpose([1, 0, 2])[mask_sum],
                    xx.transpose([1, 0, 2])[mask_sum],
                ],
                axis=1,
            )
    else:
        mask_sum = (
            mask_sum[pad_sz_pre[0]:pad_sz_post[0], pad_sz_pre[1]:pad_sz_post[1]] >= valid_thres
        )
        if mask_sum.max() > 0:
            yy, xx = np.meshgrid(np.arange(mask_sum.shape[0]), np.arange(mask_sum.shape[1]))
            valid_pos = np.stack([yy.T[mask_sum], xx.T[mask_sum]], axis=1)
    return valid_pos
