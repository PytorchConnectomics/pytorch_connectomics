from __future__ import annotations

import numpy as np
import pytest

from connectomics.decoding.decoders.branch import branch_extend
from connectomics.decoding.registry import get_decoder


def _tube_volume(z_size: int = 24, size: int = 40) -> np.ndarray:
    """A single 10x10 tube running the full z extent, in three broken pieces."""
    seg = np.zeros((z_size, size, size), np.uint32)
    seg[0:8, 15:25, 15:25] = 1
    seg[8:16, 15:25, 15:25] = 2
    seg[16:z_size, 15:25, 15:25] = 3
    return seg


def _affinity(seg: np.ndarray, value: float = 0.9) -> np.ndarray:
    return np.full((3, *seg.shape), value, np.float32)


def test_extends_a_broken_tube_to_both_faces():
    seg = _tube_volume()
    aff = _affinity(seg)

    out = branch_extend(aff, seg, min_size=100, min_span_frac=0.25, max_gap=1)

    # The three pieces become one label spanning z0..z-1.
    assert len(np.unique(out[out > 0])) == 1
    zs = np.where(out.any(axis=(1, 2)))[0]
    assert zs.min() == 0 and zs.max() == seg.shape[0] - 1


def test_skips_ends_that_already_reach_a_border():
    # One label already spanning the volume: both ends are border ends, so the
    # stage must leave the neighbouring fragment alone.
    seg = np.zeros((24, 40, 40), np.uint32)
    seg[:, 15:25, 15:25] = 1
    seg[10:14, 26:30, 15:25] = 2
    aff = _affinity(seg)

    out = branch_extend(aff, seg, min_size=100, min_span_frac=0.25)

    assert np.array_equal(out, seg)


def test_does_not_join_laterally_separated_tubes():
    # Two tubes side by side, each stopping mid-volume with no overlap: there
    # is no cross-section continuity, so neither may claim the other.
    seg = np.zeros((24, 40, 40), np.uint32)
    seg[0:12, 5:15, 5:15] = 1
    seg[12:24, 25:35, 25:35] = 2
    aff = _affinity(seg)

    out = branch_extend(aff, seg, min_size=100, min_span_frac=0.25)

    assert len(np.unique(out[out > 0])) == 2


def test_low_seam_affinity_vetoes_the_step():
    seg = _tube_volume()
    aff = _affinity(seg, value=0.05)

    out = branch_extend(aff, seg, min_size=100, min_span_frac=0.25, max_gap=1)

    assert len(np.unique(out[out > 0])) == 3


def test_max_gap_reaches_across_a_dropped_slice():
    seg = _tube_volume()
    seg[8:10] = 0  # two slices with no label at all
    aff = _affinity(seg)

    # Pieces 2 and 3 still touch, so only the gap at z8-9 blocks a single tube.
    touching = branch_extend(aff, seg, min_size=100, min_span_frac=0.25, max_gap=1)
    assert len(np.unique(touching[touching > 0])) == 2

    bridged = branch_extend(aff, seg, min_size=100, min_span_frac=0.25, max_gap=4)
    assert len(np.unique(bridged[bridged > 0])) == 1


def test_absorb_tubes_false_keeps_decent_tubes_apart():
    seg = np.zeros((24, 40, 40), np.uint32)
    seg[0:12, 15:25, 15:25] = 1
    seg[12:24, 15:25, 15:25] = 2
    aff = _affinity(seg)

    kept = branch_extend(aff, seg, min_size=100, min_span_frac=0.25, absorb_tubes=False)
    assert len(np.unique(kept[kept > 0])) == 2

    joined = branch_extend(aff, seg, min_size=100, min_span_frac=0.25, absorb_tubes=True)
    assert len(np.unique(joined[joined > 0])) == 1


def test_registered_as_a_binary_graph_op():
    seg = _tube_volume()
    aff = _affinity(seg)
    op = get_decoder("branch_extend")

    out = op([aff, seg], min_size=100, min_span_frac=0.25, max_gap=1)

    assert len(np.unique(out[out > 0])) == 1


def test_rejects_malformed_inputs():
    seg = _tube_volume()
    with pytest.raises(ValueError, match="CZYX"):
        branch_extend(np.zeros((2, *seg.shape), np.float32), seg)
    with pytest.raises(ValueError, match="spatial shapes differ"):
        branch_extend(np.zeros((3, 4, 4, 4), np.float32), seg)
