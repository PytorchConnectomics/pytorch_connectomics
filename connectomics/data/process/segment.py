"""
Segmentation processing functions for PyTorch Connectomics.
"""


from __future__ import annotations
import numpy as np
import torch


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


def seg_selection(label, indices):
    mid = label.max() + 1
    relabel = np.zeros(mid + 1, label.dtype)
    relabel[indices] = np.arange(1, len(indices) + 1)
    return relabel[label]


__all__ = [
    "im_to_col",
    "seg_erosion_instance",
    "seg_selection",
]
