"""GT-free completeness objective for axon segmentations.

A decent-size axon is complete when it touches the volume border at least
twice (the same face twice is allowed, for example for a U-turn) and spans at
least 25% of Z. Small segments are ignored. This is a ranker, not a quota.
"""

from __future__ import annotations

import numpy as np

from connectomics.data.processing.bbox import seg_stats

MIN_SPAN_FRAC, MIN_SIZE, BORDER = 0.25, 20000, 2


def completeness_report(
    seg: np.ndarray,
    verbose_top: int = 8,
    stats=None,
) -> tuple[int, int]:
    """Report the count of complete decent-size axons.

    Args:
        seg: Three-dimensional instance segmentation.
        verbose_top: Maximum number of largest incomplete segments to print.
        stats: Optional cached ``(bounds, sizes)`` or full
            :func:`seg_stats` result.

    Returns:
        ``(complete_count, decent_axon_count)``.
    """
    Z, Y, X = seg.shape
    if stats is None:
        zr, sizes, _ = seg_stats(seg)
    else:
        zr, sizes = stats[:2]
    big = [L for L in zr if sizes[L] >= MIN_SIZE and (zr[L][1] - zr[L][0] + 1) >= MIN_SPAN_FRAC * Z]
    # Count distinct border contacts per segment: each of the six faces, and
    # each end of the segment counts separately.
    inc: list[int]
    comp: list[int]
    inc, comp, rows = [], [], []
    for L in big:
        z0, z1, y0, y1, x0, x1 = zr[L]
        ends = 0
        for zt in (z0, z1):
            m = seg[zt] == L
            if not m.any():
                continue
            ys, xs = np.where(m)
            at_face = (
                zt <= BORDER
                or zt >= Z - 1 - BORDER
                or ys.min() <= BORDER
                or ys.max() >= Y - 1 - BORDER
                or xs.min() <= BORDER
                or xs.max() >= X - 1 - BORDER
            )
            if at_face:
                ends += 1
        (comp if ends >= 2 else inc).append(L)
        rows.append((sizes[L], L, z0, z1, ends))
    rows.sort(reverse=True)
    n = len(big)
    print(
        f"\n=== segmentation: {n} decent axons "
        f"(size>={MIN_SIZE}, span>={int(MIN_SPAN_FRAC * Z)}z) ===",
        flush=True,
    )
    print(
        f"  COMPLETE (>=2 border ends): {len(comp)} "
        f"({100 * len(comp) / max(n, 1):.0f}%)  |  "
        f"INCOMPLETE: {len(inc)} ({100 * len(inc) / max(n, 1):.0f}%)",
        flush=True,
    )
    print("  largest INCOMPLETE (these need a merge):", flush=True)
    k = 0
    for sz, L, z0, z1, ends in rows:
        if ends >= 2:
            continue
        print(
            f"    seg {L}: sz{sz} z{z0}-{z1} ({z1 - z0 + 1}sl) border-ends {ends}",
            flush=True,
        )
        k += 1
        if k >= verbose_top:
            break
    return len(comp), n


__all__ = [
    "BORDER",
    "MIN_SIZE",
    "MIN_SPAN_FRAC",
    "completeness_report",
]
