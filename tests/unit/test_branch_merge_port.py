from __future__ import annotations

import numpy as np

from connectomics.data.processing.bbox import seg_stats
from connectomics.decoding.decoders.branch.merge import (
    bridge_weak_gaps,
    complete_sections,
    merge_sections,
)


def test_complete_sections_absorbs_lateral_and_z_isolated_fragments():
    seg = np.zeros((4, 12, 12), np.uint32)
    seg[:3, 2:8, 2:8] = 1

    # A two-slice fragment moves laterally, so its aggregate bbox contains an
    # in-slice neighbor belonging to the large section.
    seg[1, 3:5, 6] = 2
    seg[2, 3:5, 7] = 2

    # This isolated one-slice fragment has no lateral neighbor, but its mask
    # overlaps label 1 on the preceding z-slice.
    seg[3, 3:7, 3:7] = 3

    completed = complete_sections(
        seg.copy(),
        min_size=20,
        stats=seg_stats(seg),
    )

    assert not np.any(completed == 2)
    assert not np.any(completed == 3)
    assert np.all(completed[seg == 2] == 1)
    assert np.all(completed[seg == 3] == 1)


def test_merge_sections_rejects_ambiguous_iou_runner_up():
    seg = np.zeros((6, 16, 16), np.uint32)
    seg[:3, 3:13, 3:13] = 1
    seg[3:, 3:13, 3:8] = 2
    seg[3:, 3:13, 8:13] = 3
    afz = np.ones_like(seg, dtype=np.float32)

    merged, count = merge_sections(
        seg,
        afz,
        aff_lo=0.4,
        merge_iou=0.2,
        min_ov=1,
        min_size=1,
        margin=0.15,
        rounds=1,
    )

    assert count == 0
    np.testing.assert_array_equal(merged, seg)


def test_bridge_weak_gaps_uses_projected_mask_mutual_match():
    seg = np.zeros((8, 16, 16), np.uint32)
    seg[:3, 4:8, 4:8] = 1
    seg[5:, 4:8, 4:8] = 2
    foreground = np.ones_like(seg, dtype=np.float32)

    merged, count = bridge_weak_gaps(
        seg,
        foreground,
        max_gap=5,
        cal_ratio=1.6,
        min_iou=0.35,
        weak_lo=0.3,
        min_size=1,
        dim_tol=3,
        margin=0.15,
        rounds=1,
    )

    assert count == 1
    assert len(np.unique(merged[merged > 0])) == 1
