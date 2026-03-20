"""Intersection-over-Union (IoU) computation for segmentations.

Ported from ``em_util.seg.iou``. Computes per-segment IoU between two
segmentation maps using bounding-box-accelerated overlap counting — only
scans pixels within each segment's bbox, making it fast for sparse
segmentations.

Two main functions:

* :func:`seg_to_iou` — IoU between two co-registered 2D/3D segmentations.
* :func:`segs_to_iou` — Track segments across consecutive z-slices via IoU.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from .bbox import compute_bbox_all

__all__ = ["seg_to_iou", "segs_to_iou"]


def seg_to_iou(
    seg0: np.ndarray,
    seg1: np.ndarray,
    uid0: Optional[np.ndarray] = None,
    bb0: Optional[np.ndarray] = None,
    uid1: Optional[np.ndarray] = None,
    uc1: Optional[np.ndarray] = None,
    th_iou: float = 0,
) -> np.ndarray:
    """Compute per-segment IoU between two segmentation maps.

    For each segment in *seg0* (or the subset *uid0*), finds the
    best-matching segment in *seg1* by overlap count within the bbox.

    Args:
        seg0: First segmentation, 2D ``(Y, X)`` or 3D ``(Z, Y, X)``.
        seg1: Second segmentation, same shape as *seg0*.
        uid0: Subset of segment IDs in *seg0* to compute IoU for.
            If None, uses all non-zero IDs.
        bb0: Pre-computed bounding boxes for *seg0* from
            :func:`compute_bbox_all` with ``do_count=True``.
        uid1: Unique segment IDs in *seg1* (for size lookup).
        uc1: Voxel counts for *uid1*.
        th_iou: If > 0, filter output to pairs with IoU > *th_iou*.

    Returns:
        ``(N, 5)`` int64 array, one row per segment in *uid0*::

            [seg_id, best_match_id, count0, count1, overlap_count]

        - ``seg_id``: ID in *seg0*
        - ``best_match_id``: best-matching ID in *seg1* (0 if no overlap)
        - ``count0``: voxel count of seg_id in *seg0*
        - ``count1``: voxel count of best_match_id in *seg1*
        - ``overlap_count``: number of overlapping voxels
    """
    if seg0.shape != seg1.shape:
        raise ValueError(
            f"seg0 and seg1 must have the same shape, "
            f"got {seg0.shape} and {seg1.shape}"
        )

    # Prepare seg0 info: uid0, bb0 (with counts)
    if uid0 is None:
        if bb0 is None:
            bb0 = compute_bbox_all(seg0, do_count=True)
        uid0 = bb0[:, 0] if bb0 is not None else np.array([], dtype=np.int64)
    elif bb0 is None:
        bb0 = compute_bbox_all(seg0, do_count=True, uid=uid0)
    else:
        # Filter bb0 to uid0
        mask = np.isin(bb0[:, 0], uid0)
        bb0 = bb0[mask]
        uid0 = bb0[:, 0]

    if len(uid0) == 0:
        return np.zeros((0, 5), dtype=np.int64)
    uc0 = bb0[:, -1]

    # Prepare seg1 info: uid1, uc1
    if uid1 is None or uc1 is None:
        uid1, uc1 = np.unique(seg1, return_counts=True)

    if len(uid1) == 0:
        return np.zeros((0, 5), dtype=np.int64)

    # Build uid1 -> count lookup
    uc1_map = dict(zip(uid1.tolist(), uc1.tolist()))

    out = np.zeros((len(uid0), 5), dtype=np.int64)
    out[:, 0] = uid0
    out[:, 2] = uc0

    ndim = seg0.ndim
    for j, sid in enumerate(uid0):
        coords = bb0[j, 1:-1]  # bbox without seg_id and count
        if ndim == 2:
            y0, y1, x0, x1 = coords
            roi0 = seg0[y0 : y1 + 1, x0 : x1 + 1]
            roi1 = seg1[y0 : y1 + 1, x0 : x1 + 1]
        else:
            z0, z1, y0, y1, x0, x1 = coords
            roi0 = seg0[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
            roi1 = seg1[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]

        # Overlap: seg1 IDs where seg0 == sid
        masked = roi1 * (roi0 == sid)
        ui, uc = np.unique(masked, return_counts=True)
        uc[ui == 0] = 0  # ignore background overlap

        if (ui > 0).any():
            best_id = ui[np.argmax(uc)]
            out[j, 1] = best_id
            out[j, 3] = uc1_map.get(int(best_id), 0)
            out[j, 4] = uc.max()

    if th_iou > 0:
        denom = (out[:, 2] + out[:, 3] - out[:, 4]).astype(np.float64)
        denom[denom == 0] = 1
        score = out[:, 4].astype(np.float64) / denom
        return out[score > th_iou]

    return out


def segs_to_iou(
    get_seg: Callable[[int], np.ndarray],
    index: Sequence[int],
    th_iou: float = 0,
) -> List[np.ndarray]:
    """Track segments across consecutive z-slices via IoU.

    Iterates consecutive pairs ``(index[i], index[i+1])`` and computes
    :func:`seg_to_iou` with bbox acceleration.

    Args:
        get_seg: Function ``get_seg(i)`` returning the 2D segmentation
            at slice index *i*.
        index: Sequence of slice indices to process.
        th_iou: If > 0, only return pairs with IoU above this threshold
            (returns ``(N, 2)`` match arrays instead of full ``(N, 5)``).

    Returns:
        List of ``len(index) - 1`` arrays, one per consecutive boundary.
        Each is either ``(N, 5)`` (full IoU info) or ``(N, 2)`` (matched
        ID pairs when *th_iou* > 0).
    """
    if len(index) < 2:
        return []

    out: List[np.ndarray] = [np.empty(0)] * (len(index) - 1)
    seg0 = get_seg(index[0])
    bb0 = compute_bbox_all(seg0, do_count=True)

    for i, z in enumerate(index[1:]):
        seg1 = get_seg(z)
        bb1 = compute_bbox_all(seg1, do_count=True)

        if bb1 is not None and bb0 is not None:
            iou = seg_to_iou(
                seg0, seg1,
                bb0=bb0,
                uid1=bb1[:, 0],
                uc1=bb1[:, -1],
            )
            if th_iou == 0:
                out[i] = iou
            else:
                # Filter and return only matched pairs
                iou = iou[iou[:, 1] != 0]
                if len(iou) > 0:
                    denom = (iou[:, 2] + iou[:, 3] - iou[:, 4]).astype(np.float64)
                    denom[denom == 0] = 1
                    score = iou[:, 4].astype(np.float64) / denom
                    out[i] = iou[score > th_iou, :2]
                else:
                    out[i] = np.zeros((0, 2), dtype=np.int64)
        else:
            # Empty slice — propagate previous bbox info
            if bb0 is not None:
                empty = np.zeros((bb0.shape[0], 5), dtype=np.int64)
                empty[:, 0] = bb0[:, 0]
                empty[:, 2] = bb0[:, -1]
                out[i] = empty

        bb0 = bb1
        seg0 = seg1

    return out
