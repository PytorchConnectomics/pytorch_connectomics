"""Slice-based branch merge for waterz segmentation postprocessing.

Resolves false splits in waterz segmentations by analyzing segment
continuity across consecutive z-slices.  Inspired by the branch
resolution algorithm in em_pipeline (em_pipeline/tasks/branch.py)
developed for zebrafish connectomics.

**Critical design constraint** (bbox-touch filter): merges are only
considered at z-boundaries where at least one segment *ends* or
*begins*.  Without this, adjacent neurons whose 2D cross-sections
happen to overlap at interior boundaries get catastrophically chained
together via the union-find (ARE 0.05 → 0.92 on SNEMI).

The algorithm operates in three stages:

1. **IOU merge** — Pairs where segment A ends at slice z and segment B
   begins at slice z+1 with high 2D Jaccard overlap.

2. **Best-buddy merge** — Mutual best-match at bbox endpoints: A's last
   slice matches B's first slice, and they are each other's best
   overlap partner in both directions.

3. **One-sided IOU merge** — Small segments almost entirely contained
   in the overlap region with a larger neighbor at a bbox boundary.

Each stage optionally validates against the mean z-boundary affinity.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import fastremap
import numpy as np

from ..utils import cast2dtype

__all__ = ["branch_merge"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------


class _UnionFind:
    """Union-find with path compression for segment merging."""

    __slots__ = ("_parent",)

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        while self._parent.get(x, x) != x:
            px = self._parent[x]
            self._parent[x] = self._parent.get(px, px)
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def apply(self, seg: np.ndarray) -> np.ndarray:
        """Relabel *seg* so that merged IDs share the same label."""
        max_id = int(seg.max())
        if max_id == 0:
            return seg.copy()
        mapping = np.arange(max_id + 1, dtype=np.int64)
        for i in range(1, max_id + 1):
            mapping[i] = self.find(i)
        result = mapping[seg.astype(np.intp, copy=False)]
        return fastremap.renumber(result.astype(np.uint64))[0]


# ---------------------------------------------------------------------------
# Z-extent computation
# ---------------------------------------------------------------------------


def _compute_z_extents(seg: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Compute first and last z-slice for each segment ID.

    Returns (z_first, z_last) dicts mapping segment ID → z-index.
    """
    z_first: Dict[int, int] = {}
    z_last: Dict[int, int] = {}
    for z in range(seg.shape[0]):
        for sid in np.unique(seg[z]):
            sid = int(sid)
            if sid == 0:
                continue
            if sid not in z_first:
                z_first[sid] = z
            z_last[sid] = z
    return z_first, z_last


# ---------------------------------------------------------------------------
# Slice overlap computation
# ---------------------------------------------------------------------------


def _slice_overlaps(
    s0: np.ndarray,
    s1: np.ndarray,
    z_aff: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Overlap statistics between two consecutive 2D label maps.

    Returns (N, K) float64 array where each row is::

        [id0, id1, size0, size1, overlap, mean_z_affinity?]
    """
    fg = (s0 > 0) & (s1 > 0)
    ncols = 6 if z_aff is not None else 5
    if not fg.any():
        return np.empty((0, ncols), dtype=np.float64)

    a = s0[fg].astype(np.int64)
    b = s1[fg].astype(np.int64)

    u0, c0 = np.unique(s0[s0 > 0], return_counts=True)
    u1, c1 = np.unique(s1[s1 > 0], return_counts=True)
    size0_map = dict(zip(u0.tolist(), c0.tolist()))
    size1_map = dict(zip(u1.tolist(), c1.tolist()))

    pairs = np.stack([a, b], axis=1)
    unique_pairs, inverse, counts = np.unique(
        pairs, axis=0, return_inverse=True, return_counts=True,
    )

    n = len(unique_pairs)
    result = np.zeros((n, ncols), dtype=np.float64)
    result[:, 0] = unique_pairs[:, 0]
    result[:, 1] = unique_pairs[:, 1]
    result[:, 2] = np.array([size0_map[int(i)] for i in unique_pairs[:, 0]])
    result[:, 3] = np.array([size1_map[int(i)] for i in unique_pairs[:, 1]])
    result[:, 4] = counts

    if z_aff is not None:
        aff_vals = z_aff[fg].astype(np.float64)
        aff_sums = np.zeros(n, dtype=np.float64)
        np.add.at(aff_sums, inverse, aff_vals)
        result[:, 5] = aff_sums / counts

    return result


# ---------------------------------------------------------------------------
# Bbox-touch filter
# ---------------------------------------------------------------------------


def _bbox_touch_mask(
    overlaps: np.ndarray,
    z: int,
    z_first: Dict[int, int],
    z_last: Dict[int, int],
    seg_sizes: Dict[int, int],
    min_segment_size: int,
    strict: bool = True,
) -> np.ndarray:
    """Return boolean mask: True for pairs eligible for merging.

    Applies two filters:

    1. **Bbox-touch**: at least one segment must end/begin at this boundary.
       *strict* requires BOTH (A ends at z AND B starts at z+1);
       relaxed requires EITHER.

    2. **Minimum size**: both segments must have >= *min_segment_size* total
       voxels.  This prevents dust fragments from chaining unrelated neurons
       through the union-find.
    """
    if len(overlaps) == 0:
        return np.empty(0, dtype=bool)

    id0_ends = np.array([z_last.get(int(x), -1) == z for x in overlaps[:, 0]])
    id1_starts = np.array([z_first.get(int(x), -1) == z + 1 for x in overlaps[:, 1]])

    if strict:
        bbox_ok = id0_ends & id1_starts
    else:
        bbox_ok = id0_ends | id1_starts

    if min_segment_size > 0:
        size_ok = np.array([
            seg_sizes.get(int(overlaps[i, 0]), 0) >= min_segment_size
            and seg_sizes.get(int(overlaps[i, 1]), 0) >= min_segment_size
            for i in range(len(overlaps))
        ])
        bbox_ok &= size_ok

    return bbox_ok


# ---------------------------------------------------------------------------
# Per-stage merge logic
# ---------------------------------------------------------------------------


def _stage1_iou(
    overlaps: np.ndarray,
    uf: _UnionFind,
    iou_threshold: float,
    aff_threshold: float,
    has_aff: bool,
    bbox_mask: np.ndarray,
    singleton_size_ratio: float = 0.0,
) -> int:
    """Stage 1: merge pairs with high full IOU at bbox boundaries."""
    if len(overlaps) == 0:
        return 0

    denom = (overlaps[:, 2] + overlaps[:, 3] - overlaps[:, 4]).astype(np.float64)
    denom[denom == 0] = 1
    iou = overlaps[:, 4] / denom
    candidates = (iou >= iou_threshold) & bbox_mask

    if has_aff and aff_threshold > 0:
        candidates &= overlaps[:, 5] >= aff_threshold

    merged_ids: set[int] = set()
    count = 0
    for idx in np.where(candidates)[0]:
        a, b = int(overlaps[idx, 0]), int(overlaps[idx, 1])
        if uf.find(a) != uf.find(b):
            uf.union(a, b)
            merged_ids.add(a)
            merged_ids.add(b)
            count += 1

    # Singleton merge: size-ratio pairs at bbox boundaries not already merged
    if singleton_size_ratio > 0 and len(overlaps) > 0:
        sz0 = overlaps[:, 2].astype(np.float64)
        sz1 = overlaps[:, 3].astype(np.float64)
        sz_ratio = sz0 / np.maximum(sz1, 1)
        ratio_ok = (
            (sz_ratio >= singleton_size_ratio)
            & (sz_ratio <= 1.0 / singleton_size_ratio)
            & bbox_mask
        )
        if has_aff and aff_threshold > 0:
            ratio_ok &= overlaps[:, 5] >= aff_threshold

        for idx in np.where(ratio_ok)[0]:
            a, b = int(overlaps[idx, 0]), int(overlaps[idx, 1])
            if a in merged_ids and b in merged_ids:
                continue
            if uf.find(a) != uf.find(b):
                uf.union(a, b)
                count += 1

    return count


def _stage2_best_buddy(
    overlaps: np.ndarray,
    uf: _UnionFind,
    aff_threshold: float,
    has_aff: bool,
    bbox_mask: np.ndarray,
) -> int:
    """Stage 2: merge mutual best-match pairs at bbox boundaries."""
    if len(overlaps) == 0:
        return 0

    mapped_0 = np.array([uf.find(int(x)) for x in overlaps[:, 0]])
    mapped_1 = np.array([uf.find(int(x)) for x in overlaps[:, 1]])
    diff = (mapped_0 != mapped_1) & bbox_mask
    if not diff.any():
        return 0

    ovl = overlaps[diff]
    m0 = mapped_0[diff]
    m1 = mapped_1[diff]

    fwd_best: dict[int, tuple[int, float, int]] = {}
    for i in range(len(ovl)):
        r0, r1, ov = int(m0[i]), int(m1[i]), ovl[i, 4]
        if r0 not in fwd_best or ov > fwd_best[r0][1]:
            fwd_best[r0] = (r1, ov, i)

    bwd_best: dict[int, tuple[int, float, int]] = {}
    for i in range(len(ovl)):
        r0, r1, ov = int(m0[i]), int(m1[i]), ovl[i, 4]
        if r1 not in bwd_best or ov > bwd_best[r1][1]:
            bwd_best[r1] = (r0, ov, i)

    count = 0
    for r0, (r1, _, idx) in fwd_best.items():
        if r1 in bwd_best and bwd_best[r1][0] == r0:
            if has_aff and aff_threshold > 0 and ovl[idx, 5] < aff_threshold:
                continue
            if uf.find(r0) != uf.find(r1):
                uf.union(r0, r1)
                count += 1
    return count


def _stage3_one_sided(
    overlaps: np.ndarray,
    uf: _UnionFind,
    threshold: float,
    min_size: int,
    aff_threshold: float,
    has_aff: bool,
    bbox_mask: np.ndarray,
) -> int:
    """Stage 3: merge segments with high one-sided IOU at bbox boundaries."""
    if len(overlaps) == 0 or threshold <= 0:
        return 0

    mapped_0 = np.array([uf.find(int(x)) for x in overlaps[:, 0]])
    mapped_1 = np.array([uf.find(int(x)) for x in overlaps[:, 1]])
    diff = (mapped_0 != mapped_1) & bbox_mask
    if not diff.any():
        return 0

    ovl = overlaps[diff]
    min_sz = np.minimum(ovl[:, 2], ovl[:, 3])
    one_sided = ovl[:, 4] / np.maximum(min_sz, 1)
    candidates = (one_sided >= threshold) & (min_sz >= min_size)

    if has_aff and aff_threshold > 0:
        candidates &= ovl[:, 5] >= aff_threshold

    count = 0
    for idx in np.where(candidates)[0]:
        a, b = int(ovl[idx, 0]), int(ovl[idx, 1])
        ra, rb = uf.find(a), uf.find(b)
        if ra != rb:
            uf.union(ra, rb)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def branch_merge(
    seg: np.ndarray,
    affinities: Optional[np.ndarray] = None,
    *,
    iou_threshold: float = 0.5,
    singleton_size_ratio: float = 0.5,
    best_buddy: bool = True,
    one_sided_threshold: float = 0.8,
    one_sided_min_size: int = 100,
    affinity_threshold: float = 0.0,
    channel_order: str = "zyx",
) -> np.ndarray:
    """Resolve false splits in a waterz segmentation via z-slice IOU analysis.

    Only merges at **bbox boundaries** — where one segment ends and another
    begins.  This prevents catastrophic false merges between adjacent neurons
    whose cross-sections happen to overlap at interior z-slices.

    Parameters
    ----------
    seg : ndarray, shape (Z, Y, X)
        Instance segmentation from waterz (or any watershed method).
    affinities : ndarray, shape (C, Z, Y, X), optional
        Raw affinity predictions with C >= 3 short-range channels.
        Used for boundary affinity validation when *affinity_threshold* > 0.
    iou_threshold : float
        Stage 1 threshold.  Segment pairs with full Jaccard IOU above
        this value at bbox boundaries are merged.  Default: 0.5
    singleton_size_ratio : float
        Stage 1 singleton threshold.  Segments not merged by IoU but
        with size ratio in ``[ratio, 1/ratio]`` at bbox boundaries.
        Set to 0 to disable.  Default: 0.5
    best_buddy : bool
        Enable Stage 2 (mutual best-match merge).  Default: True
    one_sided_threshold : float
        Stage 3 threshold.  Set to 0 to disable.  Default: 0.8
    one_sided_min_size : int
        Stage 3 minimum segment size in the slice.  Default: 100
    affinity_threshold : float
        Minimum mean z-boundary affinity for a merge.  Default: 0.0
    channel_order : str
        Channel order of *affinities*: ``"zyx"`` or ``"xyz"``.  Default: ``"zyx"``

    Returns
    -------
    ndarray, shape (Z, Y, X)
        Segmentation with false splits resolved.
    """
    seg = np.asarray(seg)
    if seg.ndim != 3:
        raise ValueError(f"Expected 3D segmentation (Z,Y,X), got {seg.ndim}D")

    n_slices = seg.shape[0]
    if n_slices < 2:
        return seg.copy()

    # Compute z-extent for each segment (first/last z-slice)
    logger.info("Computing z-extents for bbox-touch filter...")
    z_first, z_last = _compute_z_extents(seg)

    # Prepare z-affinity slices if available
    z_affs: list[Optional[np.ndarray]] = [None] * (n_slices - 1)
    has_aff = False
    if affinities is not None and affinity_threshold > 0:
        affinities = np.asarray(affinities, dtype=np.float32)
        if affinities.ndim != 4 or affinities.shape[0] < 3:
            raise ValueError(
                f"Expected affinities (C>=3, Z, Y, X), got {affinities.shape}"
            )
        if channel_order.lower() == "xyz":
            z_ch = affinities[2]
        else:
            z_ch = affinities[0]
        for z in range(n_slices - 1):
            z_affs[z] = z_ch[z + 1]
        has_aff = True

    # Compute slice overlaps and bbox-touch masks
    logger.info("Computing slice overlaps for %d boundaries...", n_slices - 1)
    all_overlaps = []
    bbox_masks_strict = []
    bbox_masks_relaxed = []
    for z in range(n_slices - 1):
        ovl = _slice_overlaps(seg[z], seg[z + 1], z_affs[z])
        all_overlaps.append(ovl)
        bbox_masks_strict.append(_bbox_touch_mask(ovl, z, z_first, z_last, strict=True))
        bbox_masks_relaxed.append(_bbox_touch_mask(ovl, z, z_first, z_last, strict=False))

    uf = _UnionFind()

    # Stage 1: High IOU merge at bbox boundaries (relaxed: either end)
    s1_count = 0
    for ovl, bmask in zip(all_overlaps, bbox_masks_relaxed):
        s1_count += _stage1_iou(
            ovl, uf, iou_threshold, affinity_threshold, has_aff,
            bbox_mask=bmask, singleton_size_ratio=singleton_size_ratio,
        )
    logger.info("Stage 1 (IOU >= %.2f, bbox-touch): %d merges", iou_threshold, s1_count)

    # Stage 2: Best-buddy merge at strict bbox boundaries
    s2_count = 0
    if best_buddy:
        for ovl, bmask in zip(all_overlaps, bbox_masks_strict):
            s2_count += _stage2_best_buddy(ovl, uf, affinity_threshold, has_aff, bmask)
        logger.info("Stage 2 (best-buddy, bbox-touch): %d merges", s2_count)

    # Stage 3: One-sided IOU at relaxed bbox boundaries
    s3_count = 0
    if one_sided_threshold > 0:
        for ovl, bmask in zip(all_overlaps, bbox_masks_relaxed):
            s3_count += _stage3_one_sided(
                ovl, uf, one_sided_threshold, one_sided_min_size,
                affinity_threshold, has_aff, bmask,
            )
        logger.info("Stage 3 (one-sided >= %.2f, bbox-touch): %d merges",
                     one_sided_threshold, s3_count)

    total = s1_count + s2_count + s3_count
    if total == 0:
        logger.info("No merges found, returning original segmentation")
        return seg.copy()

    logger.info("Applying %d total merges...", total)
    result = uf.apply(seg)

    n_before = len(np.unique(seg)) - (1 if 0 in seg else 0)
    n_after = len(np.unique(result)) - (1 if 0 in result else 0)
    logger.info("Segments: %d -> %d (-%d)", n_before, n_after, n_before - n_after)

    return cast2dtype(result)
