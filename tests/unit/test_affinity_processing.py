import numpy as np
import torch

from connectomics.data.processing.affinity import (
    compute_affinity_crop_pad,
    compute_affinity_valid_mask,
    resolve_affinity_offsets_from_kwargs,
    seg_to_affinity,
)
from connectomics.data.processing.build import count_stacked_label_transform_channels


def test_affinity_mode_selects_storage_voxel_for_positive_offset():
    seg = np.ones((1, 1, 4), dtype=np.uint32)

    deepem = seg_to_affinity(seg, offsets=["0-0-1"], affinity_mode="deepem")
    banis = seg_to_affinity(seg, offsets=["0-0-1"], affinity_mode="banis")

    np.testing.assert_array_equal(deepem[0, 0, 0], np.array([0, 1, 1, 1], dtype=np.float32))
    np.testing.assert_array_equal(banis[0, 0, 0], np.array([1, 1, 1, -1], dtype=np.float32))


def test_banis_affinity_marks_unlabeled_and_trailing_edges_invalid():
    seg = np.array([[[1, 1, -1, 1, 1]]], dtype=np.int32)

    aff = seg_to_affinity(seg, offsets=["0-0-1"], affinity_mode="banis")

    # BANIS stores at the source voxel and uses aff == -1 to skip invalid edges:
    # edge 0 is valid/positive, edges touching seg == -1 are invalid, edge 3 is
    # valid/positive, and the trailing border has no destination voxel.
    np.testing.assert_array_equal(
        aff[0, 0, 0],
        np.array([1, -1, -1, 1, -1], dtype=np.float32),
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

    assert torch.equal(deepem_mask[0, 0, 0], torch.tensor([0.0, 1.0, 1.0, 1.0]))
    assert torch.equal(banis_mask[0, 0, 0], torch.tensor([1.0, 1.0, 1.0, 0.0]))
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
