from __future__ import annotations

import numpy as np

from connectomics.decoding.decoders.segmentation_grow import (
    affinity_foreground_score,
    binary_geodesic_grow,
    grow_segmentation_from_affinity,
    segmentation_grow,
    sparse_geodesic_grow_labels,
    watershed_grow_labels,
)


def test_segmentation_grow_grows_through_connected_foreground_only():
    seg = np.zeros((1, 7, 7), dtype=np.uint16)
    seg[0, 3, 1] = 11
    seg[0, 3, 5] = 22

    aff = np.zeros((3, 1, 7, 7), dtype=np.float32)
    aff[:, 0, 3, 1:3] = 0.9
    aff[:, 0, 3, 4:6] = 0.9

    filled = grow_segmentation_from_affinity(
        seg,
        aff,
        foreground_threshold=0.3,
        max_fill_steps=8,
    )

    assert filled[0, 3, 2] == 11
    assert filled[0, 3, 4] == 22
    assert filled[0, 3, 3] == 0


def test_segmentation_grow_respects_geodesic_step_limit():
    seg = np.zeros((1, 1, 7), dtype=np.uint16)
    seg[0, 0, 1] = 5
    aff = np.ones((3, 1, 1, 7), dtype=np.float32)

    filled = grow_segmentation_from_affinity(
        seg,
        aff,
        foreground_threshold=0.3,
        max_fill_steps=2,
    )

    np.testing.assert_array_equal(filled[0, 0], [5, 5, 5, 5, 0, 0, 0])


def test_segmentation_grow_decoder_uses_original_affinities():
    seg = np.zeros((1, 5, 5), dtype=np.uint16)
    seg[0, 2, 1] = 7
    aff = np.zeros((3, 1, 5, 5), dtype=np.float32)
    aff[:, 0, 2, 1:4] = 0.8

    filled = segmentation_grow(
        seg,
        affinities=aff,
        foreground_threshold=0.3,
        max_fill_steps=4,
    )

    assert filled.dtype == np.uint8
    assert np.all(filled[0, 2, 1:4] == 7)


def test_affinity_foreground_score_reductions():
    aff = np.stack(
        [
            np.full((2, 2, 2), 0.2, dtype=np.float32),
            np.full((2, 2, 2), 0.8, dtype=np.float32),
        ]
    )

    assert np.all(affinity_foreground_score(aff, channel_reduction="max") == 0.8)
    assert np.all(affinity_foreground_score(aff, channel_reduction="min") == 0.2)
    assert np.allclose(affinity_foreground_score(aff, channel_reduction="mean"), 0.5)


def test_binary_geodesic_grow_stays_inside_mask():
    mask = np.zeros((1, 1, 7), dtype=bool)
    mask[0, 0, 1:6] = True
    seed = np.zeros_like(mask)
    seed[0, 0, 2] = True

    grown = binary_geodesic_grow(mask, seed, max_steps=2, connectivity=1)

    np.testing.assert_array_equal(grown[0, 0], [False, True, True, True, True, False, False])


def test_watershed_grow_labels_assigns_multiple_markers_geodesically():
    seg = np.zeros((1, 1, 7), dtype=np.uint16)
    seg[0, 0, 1] = 11
    seg[0, 0, 5] = 22
    mask = np.ones_like(seg, dtype=bool)

    grown = watershed_grow_labels(seg, mask, connectivity=1)

    assert grown[0, 0, 2] == 11
    assert grown[0, 0, 4] == 22


def test_sparse_geodesic_grow_labels_assigns_target_indices():
    # Full shape is 1x1x7. Markers at x=1 and x=5 grow through target voxels.
    marker_indices = np.asarray([1, 5], dtype=np.uint64)
    marker_labels = np.asarray([11, 22], dtype=np.uint16)
    target_indices = np.asarray([2, 3, 4], dtype=np.uint64)

    result = sparse_geodesic_grow_labels(
        marker_indices,
        marker_labels,
        target_indices,
        shape=(1, 1, 7),
    )

    np.testing.assert_array_equal(result.labels, [11, 11, 22])
