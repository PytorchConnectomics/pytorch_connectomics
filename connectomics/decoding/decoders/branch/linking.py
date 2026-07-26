"""Link volume-unique 2D sections into conservative 3D tracklets.

This ports the exact path reached by
``decode_v2.decode_sections(..., no_force_split=True)``: consecutive-slice
best-overlap rows, conservative IoU links, relaxed mutual best buddies, then
one union-find relabel.  Force-split detection and relinking are unreachable
on this path and are intentionally not vendored here.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from connectomics.data.processing.bbox import apply_lut
from connectomics.data.processing.iou import segs_to_iou

__all__ = ["branch_link"]


LINK_IOU = 0.2
BB_IOU = 0.3


def _section_index(seg2d: np.ndarray) -> np.ndarray:
    """Return each label's z-slice, asserting volume-unique positive labels."""
    assert seg2d.ndim == 3, f"sections must be ZYX, got {seg2d.shape}"
    max_id = int(seg2d.max())
    z_of: np.ndarray = np.full(max_id + 1, -1, np.int32)
    for z in range(seg2d.shape[0]):
        labels = np.unique(seg2d[z])
        labels = labels[labels > 0].astype(np.int64, copy=False)
        repeated = labels[z_of[labels] >= 0]
        assert (
            not repeated.size
        ), f"nonzero section label {int(repeated[0])} appears on multiple z-slices"
        z_of[labels] = z
    return z_of


def _pair_array(*arrays: np.ndarray) -> np.ndarray:
    present = [np.asarray(array) for array in arrays if np.asarray(array).size]
    if not present:
        return np.zeros((0, 2), np.uint64)
    return np.vstack(present).astype(np.uint64, copy=False)


def _raw_iou_table(
    seg2d: np.ndarray, z_of: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the established ``segs_to_iou`` best-successor table."""
    tables = segs_to_iou(lambda z: seg2d[z], range(seg2d.shape[0]))
    tables = [table for table in tables if len(table)]
    table = np.vstack(tables) if tables else np.zeros((0, 5), np.int64)
    assert table.ndim == 2 and table.shape[1] >= 5, f"invalid IoU table shape {table.shape}"
    table = table[(table[:, :2] > 0).all(1)]
    if not len(table):
        empty_i: np.ndarray = np.zeros(0, np.int64)
        empty_f: np.ndarray = np.zeros(0, float)
        return empty_i, empty_i.copy(), empty_f, empty_f.copy(), empty_f.copy()
    a = table[:, 0].astype(np.int64)
    b = table[:, 1].astype(np.int64)
    aa = table[:, 2].astype(float)
    ab = table[:, 3].astype(float)
    inter = table[:, 4].astype(float)
    consecutive = z_of[b] == z_of[a] + 1
    return a[consecutive], b[consecutive], aa[consecutive], ab[consecutive], inter[consecutive]


def _conservative_pairs(
    a: np.ndarray,
    b: np.ndarray,
    aa: np.ndarray,
    ab: np.ndarray,
    inter: np.ndarray,
    min_iou: float = 0.2,
) -> tuple[np.ndarray, int]:
    iou = inter / (aa + ab - inter)
    order = np.argsort(-iou)
    fwd = {}
    bwd = {}
    for i in order:
        ai, bi = int(a[i]), int(b[i])
        if ai not in fwd:
            fwd[ai] = bi
        if bi not in bwd:
            bwd[bi] = ai
    indeg_fwd = Counter(fwd.values())
    indeg_bwd = Counter(bwd.values())
    keep = np.array(
        [
            iou[i] >= min_iou
            and fwd.get(int(a[i])) == int(b[i])
            and bwd.get(int(b[i])) == int(a[i])
            and indeg_fwd[int(b[i])] == 1
            and indeg_bwd[int(a[i])] == 1
            for i in range(len(a))
        ]
    )
    return np.stack([a[keep], b[keep]], 1).astype(np.uint32), int(keep.sum())


def _bb_pairs(
    a: np.ndarray,
    b: np.ndarray,
    aa: np.ndarray,
    ab: np.ndarray,
    inter: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Mutual-best relaxed: keep reciprocal argmax-IoU partners."""
    iou = inter / (aa + ab - inter)
    bestf: dict[int, tuple[float, int]] = {}
    bestb: dict[int, tuple[float, int]] = {}
    for ai, bi, value in zip(a, b, iou):
        if value < threshold:
            continue
        if value > bestf.get(int(ai), (-1,))[0]:
            bestf[int(ai)] = (value, int(bi))
        if value > bestb.get(int(bi), (-1,))[0]:
            bestb[int(bi)] = (value, int(ai))
    out = [(ai, value[1]) for ai, value in bestf.items() if bestb.get(value[1], (0, -1))[1] == ai]
    return np.array(out, np.uint64) if out else np.zeros((0, 2), np.uint64)


def _link_recipe(
    a: np.ndarray,
    b: np.ndarray,
    aa: np.ndarray,
    ab: np.ndarray,
    inter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the winning conservative(0.2) + best-buddy(0.3) recipe."""
    spine, _ = _conservative_pairs(a, b, aa, ab, inter, min_iou=LINK_IOU)
    bb = _bb_pairs(a, b, aa, ab, inter, BB_IOU)
    return spine.astype(np.uint64), bb.astype(np.uint64)


def _apply(
    seg: np.ndarray,
    pairs: np.ndarray,
    *,
    inplace: bool = False,
) -> np.ndarray:
    out = seg if inplace else seg.copy()
    if not len(pairs):
        return out.astype(np.uint32, copy=False)
    try:
        from waterz import merge_id
    except ImportError as exc:  # pragma: no cover - environment/dependency error.
        raise ImportError(
            "branch_link requires the repository's waterz package to be installed"
        ) from exc

    relabel = np.asarray(merge_id(pairs[:, 0], pairs[:, 1]))
    lut = np.arange(int(out.max()) + 1, dtype=out.dtype)
    prefix = min(len(relabel), len(lut))
    lut[:prefix] = relabel[:prefix]
    apply_lut(out, lut)
    return out.astype(np.uint32, copy=False)


def _apply_links(
    seg2d: np.ndarray,
    spine: np.ndarray,
    bb: np.ndarray,
    *,
    inplace: bool = False,
) -> np.ndarray:
    return _apply(
        seg2d,
        _pair_array(spine, bb),
        inplace=inplace,
    )


def branch_link(
    aff: np.ndarray,
    sections: np.ndarray,
    *,
    inplace: bool = False,
) -> np.ndarray:
    """Conservatively link globally unique sections into v0 tracklets.

    ``aff`` is accepted for the graph-op ``[raw, seg]`` contract and validated,
    but the linking itself is geometric: the research path
    (``decode_v2.decode_sections(..., no_force_split=True)``) returns before
    affinity is consumed, so no in-plane affinity is computed here.
    """
    aff = np.asarray(aff)
    if aff.ndim != 4 or aff.shape[0] != 3:
        raise ValueError(f"affinity must be CZYX with 3 channels, got {aff.shape}")
    seg2d = np.asarray(sections)
    if seg2d.shape != aff.shape[1:]:
        raise ValueError(f"affinity/sections shape mismatch: {aff.shape[1:]} vs {seg2d.shape}")

    z_of = _section_index(seg2d)
    a, b, aa, ab, inter = _raw_iou_table(seg2d, z_of)
    spine, bb = _link_recipe(a, b, aa, ab, inter)
    return _apply_links(seg2d, spine, bb, inplace=inplace)
