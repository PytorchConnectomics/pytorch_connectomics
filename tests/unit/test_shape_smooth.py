from __future__ import annotations

import numpy as np
import pytest

from connectomics.decoding.decoders.shape_smooth import (
    label_opening,
    shape_smooth,
    split_area_outliers,
)
from connectomics.decoding.registry import get_decoder


def _tube(z_size: int = 32, size: int = 48) -> np.ndarray:
    seg = np.zeros((z_size, size, size), np.uint32)
    seg[:, 18:28, 18:28] = 1
    return seg


def test_opening_removes_a_hairline_neck_and_cc3d_separates():
    # Two blocks joined by a one-slice bridge, all under one label.
    seg = np.zeros((16, 32, 32), np.uint32)
    seg[2:14, 5:13, 5:13] = 1
    seg[2:14, 19:27, 19:27] = 1
    seg[8, 12:20, 12:20] = 1  # the neck

    out = shape_smooth(seg, split=False)

    labels = np.unique(out[out > 0])
    assert len(labels) == 2, f"expected the neck to be opened away, got {labels}"


def test_opening_never_adds_voxels():
    seg = _tube()
    out = label_opening(seg)
    assert set(np.unique(out)) <= {0, 1}
    assert (out > 0).sum() <= (seg > 0).sum()
    assert not ((out > 0) & (seg == 0)).any()


def test_subvoxel_spherical_radius_is_rejected_not_silently_ignored():
    # Measured: spherical_open(radius=1.0) erodes zero voxels on this lattice.
    with pytest.raises(ValueError, match="erodes nothing"):
        label_opening(_tube(), radius=1.0)


def test_split_carves_the_extra_region_at_an_area_jump():
    # A tube that abruptly gains a second lobe for a run of slices.
    seg = _tube()
    seg[12:20, 18:28, 30:40] = 1

    out, splits = split_area_outliers(seg, min_size=100, min_extra=50)

    assert splits >= 1
    # The original tube column keeps its id; the added lobe becomes a new one.
    assert out[16, 20, 20] == 1
    assert out[16, 20, 35] != 1
    assert out[16, 20, 35] != 0
    # Every slice of the lobe is carved away from label 1.
    assert not (out[12:20, 18:28, 30:40] == 1).any()


def test_split_leaves_a_uniform_tube_alone():
    seg = _tube()
    out, splits = split_area_outliers(seg, min_size=100, min_extra=50)
    assert splits == 0
    assert np.array_equal(out, seg)


def test_split_anchors_above_when_the_run_starts_at_the_first_slice():
    seg = _tube()
    seg[0:6, 18:28, 30:40] = 1

    out, splits = split_area_outliers(seg, min_size=100, min_extra=50)

    assert splits == 1
    assert out[2, 20, 20] == 1
    assert out[2, 20, 35] != 1


def test_registered_as_a_unary_decoder():
    seg = _tube()
    seg[12:20, 18:28, 30:40] = 1
    op = get_decoder("shape_smooth")

    out = op([seg], open=False, bump_min_extra=50, split_min_size=100)

    assert out.dtype == np.uint32
    assert len(np.unique(out[out > 0])) >= 2


def test_rejects_malformed_inputs():
    with pytest.raises(ValueError, match="ZYX"):
        shape_smooth(np.zeros((4, 4), np.uint32))
    with pytest.raises(TypeError, match="integer"):
        shape_smooth(np.zeros((4, 4, 4), np.float32))


def test_2d_and_3d_opening_both_remove_an_in_plane_spur():
    seg = np.zeros((12, 32, 32), np.uint32)
    seg[:, 10:22, 10:22] = 1
    seg[5, 22:26, 15] = 1  # thinner than the structuring element

    for plane in ("2d", "3d"):
        out = label_opening(seg, plane=plane)
        assert not (out[5, 22:26, 15] > 0).any(), f"{plane} should open the spur away"


def test_3d_opening_destroys_a_z_thin_label_that_2d_keeps():
    # A label wedged between two others in z has no background to regrow from
    # after a 3D erosion, so the 3D pass deletes it outright. This is why a
    # densely packed segmentation loses whole thin objects to a 3D opening.
    seg = np.zeros((12, 32, 32), np.uint32)
    seg[0:5, 10:22, 10:22] = 1
    seg[5:7, 10:22, 10:22] = 2
    seg[7:12, 10:22, 10:22] = 3

    assert int((label_opening(seg, plane="2d") == 2).sum()) == int((seg == 2).sum())
    assert int((label_opening(seg, plane="3d") == 2).sum()) == 0


def test_invalid_plane_is_rejected():
    with pytest.raises(ValueError, match="plane must be"):
        label_opening(_tube(), plane="xy")
