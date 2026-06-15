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
    # seg=0: background.
    #
    # Separable max/min box filters instead of an im_to_col gather: a voxel is
    # kept iff its window holds a single positive ID, i.e. the window max equals
    # the window min taken over positive IDs only. Identical output to the gather
    # form but without materializing the (N, window^2) patch matrix (the cost
    # blows up for large tsz_h).
    #
    # tsz_h controls the window half-size:
    #   * scalar  -> Kisuk XY-only erosion (window 2*tsz_h+1 over the last two
    #                axes; axis 0 is the slice axis for 3D input). No Z erosion.
    #   * sequence (one half-size per array axis) -> full anisotropic N-D erosion.
    #     For the (9,9,20) nm NISB layout, a physically isotropic erosion uses a
    #     smaller Z half-size, e.g. (r, r, round(r*9/20)) for (X, Y, Z) arrays.
    from scipy.ndimage import maximum_filter, minimum_filter

    is_tensor = torch.is_tensor(seg)
    seg_np = seg.detach().cpu().numpy() if is_tensor else np.asarray(seg)
    if np.isscalar(tsz_h):
        tsz = 2 * tsz_h + 1
        size = (1, tsz, tsz) if seg_np.ndim == 3 else (tsz, tsz)
    else:
        tsz_h = tuple(tsz_h)
        if len(tsz_h) != seg_np.ndim:
            raise ValueError(
                f"tsz_h sequence length {len(tsz_h)} != seg ndim {seg_np.ndim}"
            )
        size = tuple(2 * t + 1 for t in tsz_h)

    big = seg_np.max() + 1  # sentinel > any positive ID, so the min ignores background
    win_max = maximum_filter(seg_np, size=size, mode="reflect")
    win_min_pos = minimum_filter(np.where(seg_np > 0, seg_np, big), size=size, mode="reflect")
    keep = win_max == win_min_pos  # window holds exactly one positive ID

    if is_tensor:
        return seg * torch.as_tensor(keep, device=seg.device, dtype=seg.dtype)
    return seg_np * keep


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
