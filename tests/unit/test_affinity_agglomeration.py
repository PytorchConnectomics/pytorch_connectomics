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


def _bridge_case():
    # Two 4x8x8 blocks (labels 1, 2) side by side in x, separated by a membrane plane at
    # x = 8 whose affinities are 0.05, except a 1-voxel bridge (one x edge at 0.9). Inside each
    # block every affinity is 0.95; block 3 continues block 1 across a full-face contact at 0.8.
    seg = np.zeros((4, 8, 24), dtype=np.uint32)
    seg[..., :8] = 1
    seg[..., 8:16] = 2
    seg[..., 16:] = 3
    aff = np.full((3, 4, 8, 24), 0.95, dtype=np.float32)
    aff[0, :, :, 8] = 0.05  # membrane between 1 and 2 (stored at the second voxel)
    aff[0, 2, 4, 8] = 0.9  # the bridge
    aff[0, :, :, 16] = 0.8  # wide, strong contact 2 - 3
    return seg, aff


def test_contact_prior_blocks_a_bridge_between_large_segments():
    seg, aff = _bridge_case()
    # Only the bridge edge and the membrane between 1 and 2: plain mean is low anyway, so
    # isolate the bridge by making the whole 1-2 boundary just the bridge.
    seg[..., 8] = 0  # background column; the bridge voxel joins 1 and 2 directly
    seg[2, 4, 8] = 1
    aff[0, 2, 4, 9] = 0.9
    plain = agglomerate_mean_affinity(seg, aff, 0.5)
    assert plain[2, 4, 0] == plain[2, 4, 12]  # plain mean joins 1 and 2 through one edge
    prior = agglomerate_mean_affinity(seg, aff, 0.5, contact_prior=1.0)
    assert prior[2, 4, 0] != prior[2, 4, 12]  # 0.9 / (1 + 1 * 129 ** (1/3)) < 0.5
    assert prior[2, 4, 12] == prior[2, 4, 20]  # the 32-edge contact at 0.8 still merges


def test_contact_prior_still_absorbs_small_fragments():
    seg = np.ones((1, 4, 8), dtype=np.uint32)
    seg[0, 1, 3] = 2  # a one-voxel fragment with four strong edges to segment 1
    aff = np.full((3, 1, 4, 8), 0.9, dtype=np.float32)
    out = agglomerate_mean_affinity(seg, aff, 0.5, contact_prior=1.0)
    assert len(np.unique(out)) == 1  # 3.6 / (4 + 1) = 0.72


def test_zero_prior_matches_plain_mean_on_random_basins():
    from scipy import ndimage

    rng = np.random.default_rng(0)
    seg = ndimage.label(rng.random((6, 20, 20)) > 0.2)[0].astype(np.uint32)
    seg = (rng.integers(1, 40, size=seg.max() + 1)[seg] * (seg > 0)).astype(np.uint32)
    aff = rng.random((3, 6, 20, 20)).astype(np.float32)
    for t in (0.3, 0.5):
        a = agglomerate_mean_affinity(seg, aff, t)
        b = agglomerate_mean_affinity(seg, aff, t, contact_prior=0.0)
        np.testing.assert_array_equal(a, b)
        c = agglomerate_mean_affinity(seg, aff, t, contact_prior=2.0)
        assert len(np.unique(c)) >= len(np.unique(a))  # a prior never merges more
