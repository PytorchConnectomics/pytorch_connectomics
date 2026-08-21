"""Tests for the caliber-weighted affinity loss weight target."""

import numpy as np
import pytest

from connectomics.data.processing.affinity import (
    seg_to_affinity,
    seg_to_thin_affinity_weight,
)
from connectomics.data.processing.build import count_stacked_label_transform_channels
from connectomics.data.processing.transforms import MultiTaskLabelTransformd

OFFSETS = ["1-0-0", "0-1-0", "0-0-1"]


def _two_bars():
    """A thick bar and a thin bar of the same instance-free geometry.

    Instance 1 is 7 voxels wide in axis 1, instance 2 is 1 voxel wide, with a
    background gap between them so the boundary EDT differs strongly.
    """
    seg = np.zeros((16, 16, 4), np.int32)
    seg[:, 2:9, :] = 1  # thick
    seg[:, 13:14, :] = 2  # thin
    return seg


def test_weight_is_higher_on_the_thin_instance():
    seg = _two_bars()
    w = seg_to_thin_affinity_weight(
        seg, offsets=OFFSETS, affinity_mode="banis", radius_ref=4.0, max_weight=5.0
    )
    aff = seg_to_affinity(seg, offsets=OFFSETS, affinity_mode="banis")
    same = aff.values & aff.mask
    thin_w = w[same & (seg[None] == 2)]
    thick_w = w[same & (seg[None] == 1)]
    assert thin_w.size and thick_w.size
    assert thin_w.mean() > thick_w.mean()
    # the 1-voxel-wide bar has caliber ~0.5 voxel -> nearly the full weight
    assert thin_w.min() >= 4.0
    # the core of the 7-wide bar is >= radius_ref away from the boundary
    assert thick_w.min() == pytest.approx(1.0)


def test_cross_instance_and_background_edges_keep_weight_one():
    seg = _two_bars()
    w = seg_to_thin_affinity_weight(
        seg, offsets=OFFSETS, affinity_mode="banis", radius_ref=4.0, max_weight=5.0
    )
    aff = seg_to_affinity(seg, offsets=OFFSETS, affinity_mode="banis")
    not_same = ~aff.values
    assert np.all(w[not_same] == 1.0)
    assert w.min() >= 1.0 and w.max() <= 5.0


def test_include_negative_weights_every_edge():
    seg = _two_bars()
    w = seg_to_thin_affinity_weight(
        seg, offsets=OFFSETS, affinity_mode="banis", radius_ref=4.0,
        max_weight=5.0, include_negative=True,
    )
    aff = seg_to_affinity(seg, offsets=OFFSETS, affinity_mode="banis")
    # boundary voxels of the thick bar are cross-instance/background edges but
    # still sit at a small caliber, so they now carry weight > 1
    assert w[~aff.values].max() > 1.0


def test_resolution_makes_an_anisotropic_axis_count_more():
    seg = np.zeros((16, 16, 16), np.int32)
    seg[:, 6:10, :] = 1  # 4 voxels wide along axis 1
    iso = seg_to_thin_affinity_weight(
        seg, offsets=OFFSETS, affinity_mode="banis",
        resolution=(1.0, 1.0, 1.0), radius_ref=8.0, max_weight=5.0,
    )
    aniso = seg_to_thin_affinity_weight(
        seg, offsets=OFFSETS, affinity_mode="banis",
        resolution=(1.0, 4.0, 1.0), radius_ref=8.0, max_weight=5.0,
    )
    # with axis 1 stretched 4x the bar is physically thicker, so it is weighted less
    assert aniso.mean() < iso.mean()


def test_shape_matches_the_paired_affinity_target():
    seg = _two_bars()
    for offsets, long_range in ([OFFSETS, None], [None, 10]):
        w = seg_to_thin_affinity_weight(
            seg, offsets=offsets, long_range=long_range, affinity_mode="banis"
        )
        aff = seg_to_affinity(seg, offsets=offsets, long_range=long_range,
                              affinity_mode="banis")
        assert w.shape == aff.values.shape
        assert w.dtype == np.float32


def test_deepem_mode_stores_at_the_destination_voxel():
    seg = np.zeros((8, 8, 8), np.int32)
    seg[:, 3:5, :] = 1
    banis = seg_to_thin_affinity_weight(
        seg, offsets=["0-1-0"], affinity_mode="banis", radius_ref=4.0, max_weight=5.0
    )
    deepem = seg_to_thin_affinity_weight(
        seg, offsets=["0-1-0"], affinity_mode="deepem", radius_ref=4.0, max_weight=5.0
    )
    # same edge set, shifted one voxel along axis 1
    assert np.array_equal(banis[0, :, :-1, :], deepem[0, :, 1:, :])


def test_rejects_bad_parameters():
    seg = _two_bars()
    with pytest.raises(ValueError, match="max_weight"):
        seg_to_thin_affinity_weight(seg, offsets=OFFSETS, max_weight=0.5)
    with pytest.raises(ValueError, match="radius_ref"):
        seg_to_thin_affinity_weight(seg, offsets=OFFSETS, radius_ref=0.0)
    with pytest.raises(ValueError, match="resolution length"):
        seg_to_thin_affinity_weight(seg, offsets=OFFSETS, resolution=(1.0, 1.0))


def test_registered_in_the_label_transform_and_channel_count():
    tasks = [
        {"name": "affinity", "kwargs": {"offsets": OFFSETS, "affinity_mode": "banis"}},
        {
            "name": "thin_affinity_weight",
            "kwargs": {"offsets": OFFSETS, "affinity_mode": "banis",
                       "radius_ref": 4.0, "max_weight": 5.0},
        },
    ]
    assert count_stacked_label_transform_channels({"targets": tasks}) == 6

    transform = MultiTaskLabelTransformd(keys=["label"], tasks=tasks)
    out = transform({"label": _two_bars()[None]})
    stacked = out["label"]
    assert tuple(stacked.shape) == (6, 16, 16, 4)
    weights = stacked[3:6].numpy()
    assert weights.min() >= 1.0 and weights.max() > 1.0
    # the affinity valid mask still covers the affinity channels only
    assert tuple(out["label_mask"].shape) == (6, 16, 16, 4)
