"""Oracle transformations used to report attainable segmentation ceilings."""

from __future__ import annotations

import numpy as np

__all__ = ["oracle_merge_segmentation"]


def oracle_merge_segmentation(seg: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Relabel each predicted fragment to its majority-overlap ground-truth ID."""
    seg = np.asarray(seg)
    gt = np.asarray(gt)
    if seg.shape != gt.shape:
        raise ValueError(f"prediction/ground-truth shape mismatch: {seg.shape} vs {gt.shape}")

    npred = int(seg.max()) + 1
    key = gt.ravel().astype(np.int64) * npred + seg.ravel().astype(np.int64)
    unique, counts = np.unique(key, return_counts=True)
    gt_ids = unique // npred
    pred_ids = unique % npred
    best: dict[int, tuple[int, int]] = {}
    for gt_id, pred_id, count in zip(
        gt_ids.tolist(),
        pred_ids.tolist(),
        counts.tolist(),
    ):
        if pred_id == 0 or gt_id == 0:
            continue
        if pred_id not in best or count > best[pred_id][1]:
            best[pred_id] = (gt_id, count)
    lut: np.ndarray = np.zeros(npred, np.int64)
    next_id = int(gt.max()) + 1
    for pred_id in range(1, npred):
        lut[pred_id] = best[pred_id][0] if pred_id in best else (next_id := next_id + 1)
    return lut[seg].astype(np.uint32)
