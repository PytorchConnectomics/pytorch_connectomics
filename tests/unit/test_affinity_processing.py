import numpy as np
import torch

from connectomics.data.processing.affinity import (
    AffinityTarget,
    compute_affinity_crop_pad,
    compute_affinity_valid_mask,
    resolve_affinity_offsets_from_kwargs,
    seg_to_affinity,
)
from connectomics.data.processing.build import count_stacked_label_transform_channels


def _banis_comp_affinities_reference(seg: np.ndarray, long_range: int = 10):
    """Reference copied from lib/banis/data.py::comp_affinities semantics.

    Returns ``(values, mask)`` both bool, mirroring the new explicit-mask
    contract emitted by ``seg_to_affinity``.
    """
    labeled_mask = seg != -1
    affinities = np.zeros((6, *seg.shape), dtype=bool)

    affinities[0, :-1] = seg[:-1] == seg[1:]
    affinities[1, :, :-1] = seg[:, :-1] == seg[:, 1:]
    affinities[2, :, :, :-1] = seg[:, :, :-1] == seg[:, :, 1:]

    affinities[3, :-long_range] = seg[:-long_range] == seg[long_range:]
    affinities[4, :, :-long_range] = seg[:, :-long_range] == seg[:, long_range:]
    affinities[5, :, :, :-long_range] = seg[:, :, :-long_range] == seg[:, :, long_range:]

    affinities[:, seg == 0] = 0

    loss_mask = np.zeros_like(affinities, dtype=bool)
    loss_mask[0, :-1] = labeled_mask[:-1] & labeled_mask[1:]
    loss_mask[1, :, :-1] = labeled_mask[:, :-1] & labeled_mask[:, 1:]
    loss_mask[2, :, :, :-1] = labeled_mask[:, :, :-1] & labeled_mask[:, :, 1:]
    loss_mask[3, :-long_range] = labeled_mask[:-long_range] & labeled_mask[long_range:]
    loss_mask[4, :, :-long_range] = labeled_mask[:, :-long_range] & labeled_mask[:, long_range:]
    loss_mask[5, :, :, :-long_range] = (
        labeled_mask[:, :, :-long_range] & labeled_mask[:, :, long_range:]
    )

    return affinities, loss_mask


def test_affinity_mode_selects_storage_voxel_for_positive_offset():
    seg = np.ones((1, 1, 4), dtype=np.uint32)

    deepem = seg_to_affinity(seg, offsets=["0-0-1"], affinity_mode="deepem")
    banis = seg_to_affinity(seg, offsets=["0-0-1"], affinity_mode="banis")

    # Mask is False at the no-pair boundary voxel:
    # ``deepem`` stores at the destination voxel, so index 0 is invalid;
    # ``banis`` stores at the source voxel, so index N-1 is invalid.
    np.testing.assert_array_equal(deepem.mask[0, 0, 0], np.array([False, True, True, True]))
    np.testing.assert_array_equal(banis.mask[0, 0, 0], np.array([True, True, True, False]))
    # Values are the equality predicate (no sentinel encoding).
    np.testing.assert_array_equal(deepem.values[0, 0, 0], np.array([False, True, True, True]))
    np.testing.assert_array_equal(banis.values[0, 0, 0], np.array([True, True, True, False]))


def test_banis_affinity_marks_unlabeled_and_trailing_edges_invalid():
    seg = np.array([[[1, 1, -1, 1, 1]]], dtype=np.int32)

    aff = seg_to_affinity(seg, offsets=["0-0-1"], affinity_mode="banis")

    # Edge 0 is valid/positive, edges touching seg == -1 are invalid, edge 3 is
    # valid/positive, and the trailing border has no destination voxel.
    np.testing.assert_array_equal(
        aff.values[0, 0, 0],
        np.array([True, False, False, True, False]),
    )
    np.testing.assert_array_equal(
        aff.mask[0, 0, 0],
        np.array([True, False, False, True, False]),
    )


def test_affinity_mode_selects_valid_mask_and_crop_side():
    offsets = [(0, 0, 1)]

    deepem_mask = compute_affinity_valid_mask(
        offsets,
        (1, 1, 4),
        affinity_mode="deepem",
    )
    banis_mask = compute_affinity_valid_mask(
        offsets,
        (1, 1, 4),
        affinity_mode="banis",
    )

    assert deepem_mask.dtype == torch.bool
    assert banis_mask.dtype == torch.bool
    assert torch.equal(deepem_mask[0, 0, 0], torch.tensor([False, True, True, True]))
    assert torch.equal(banis_mask[0, 0, 0], torch.tensor([True, True, True, False]))
    assert compute_affinity_crop_pad(offsets, affinity_mode="deepem") == ((0, 0), (0, 0), (1, 0))
    assert compute_affinity_crop_pad(offsets, affinity_mode="banis") == ((0, 0), (0, 0), (0, 1))


def test_long_range_affinity_resolves_six_channels():
    kwargs = {"long_range": 10, "affinity_mode": "banis"}

    assert resolve_affinity_offsets_from_kwargs(kwargs) == [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (10, 0, 0),
        (0, 10, 0),
        (0, 0, 10),
    ]
    assert (
        count_stacked_label_transform_channels(
            {"targets": [{"name": "affinity", "kwargs": kwargs}]}
        )
        == 6
    )


def test_banis_affinity_matches_reference_for_range_1_and_10():
    seg = np.ones((12, 13, 14), dtype=np.int32)
    seg[:4, 2:9, 3:12] = 2
    seg[6:, 6:, 7:] = 3
    seg[:, 5:7, :] = 0
    seg[3:9, :, 4] = -1
    seg[10, 11, 12] = -1

    expected_values, expected_mask = _banis_comp_affinities_reference(seg, long_range=10)
    actual = seg_to_affinity(seg, long_range=10, affinity_mode="banis")

    # Mask must match exactly.
    np.testing.assert_array_equal(actual.mask, expected_mask)
    # Values at masked-out positions are don't-care (the loss skips them);
    # only compare where the mask is True.
    np.testing.assert_array_equal(actual.values & actual.mask, expected_values & expected_mask)


def test_seg_to_affinity_returns_explicit_target_and_mask_for_both_modes():
    seg = np.array(
        [
            [[1, 1, -1], [1, 0, -1], [-1, -1, -1]],
            [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
        ],
        dtype=np.int32,
    )

    banis = seg_to_affinity(seg, offsets=["0-0-1"], affinity_mode="banis")
    assert isinstance(banis, AffinityTarget)
    assert banis.affinity_mode == "banis"
    assert banis.values.dtype == np.bool_
    assert banis.mask.dtype == np.bool_
    assert banis.values.shape == (1, 2, 3, 3)
    assert banis.mask.shape == banis.values.shape
    assert banis.shape == banis.values.shape

    deepem = seg_to_affinity(seg, offsets=["0-0-1"], affinity_mode="deepem")
    assert isinstance(deepem, AffinityTarget)
    assert deepem.affinity_mode == "deepem"
    assert deepem.values.dtype == np.bool_
    assert deepem.mask.dtype == np.bool_

    # Each mode masks any voxel whose source or destination is unlabeled,
    # plus the no-pair border. The two modes differ only in which side of
    # the array holds the storage; the *set* of valid (mask=True) edges
    # should agree between the two modes per offset.
    assert int(banis.mask.sum()) == int(deepem.mask.sum())
    # Storage convention: banis stores at source, deepem at destination.
    # Shifting deepem's mask one voxel back along the offset axis must
    # recover banis's mask (and vice versa) for the (0,0,1) offset.
    expected_banis_from_deepem = np.zeros_like(deepem.mask)
    expected_banis_from_deepem[..., :-1] = deepem.mask[..., 1:]
    np.testing.assert_array_equal(banis.mask, expected_banis_from_deepem)
