"""Greedy mean-affinity agglomeration: boundary means are recomputed after each merge."""

import numpy as np

from connectomics.decoding.decoders.affinity_agglomeration import agglomerate_mean_affinity


def test_recomputed_mean_blocks_a_thin_strong_contact():
    # x ->   A C      A-B: one y edge at 0.9; A-C: one x edge at 0.6; B-C: three x edges at 0.1.
    #        B C      Single linkage at 0.5 would join all three through A-C; after A+B merge the
    #        B C      boundary to C has mean (0.6 + 3 * 0.1) / 4 < 0.5, so C stays separate.
    #        B C
    seg = np.array([[[5, 9], [7, 9], [7, 9], [7, 9]]], dtype=np.uint32)
    aff = np.zeros((3, 1, 4, 2), dtype=np.float32)
    aff[1, 0, 1, 0] = 0.9
    aff[0, 0, 0, 1] = 0.6
    aff[0, 0, 1:, 1] = 0.1
    out = agglomerate_mean_affinity(seg, aff, 0.5)
    assert out.dtype == seg.dtype
    assert out[0, 0, 0] == out[0, 1, 0] and out[0, 0, 0] in (5, 7)
    assert (out[0, :, 1] == 9).all()
    np.testing.assert_array_equal(agglomerate_mean_affinity(seg, aff, 0.95), seg)
    assert len(np.unique(agglomerate_mean_affinity(seg, aff, 0.2))) == 1


def test_background_is_never_merged():
    seg = np.array([[[1, 0, 2]]], dtype=np.uint64)
    aff = np.ones((3, 1, 1, 3), dtype=np.float32)
    np.testing.assert_array_equal(agglomerate_mean_affinity(seg, aff, 0.1), seg)
