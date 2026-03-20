"""Test that SDT via precomputed skeleton matches direct computation."""

import numpy as np
import pytest

from connectomics.data.processing.distance import (
    _batch_skeletonize,
    skeleton_aware_distance_transform,
    skeleton_aware_edt_from_skeleton_vol,
)


def _make_label_volume(shape=(64, 64, 64), n_objects=5, seed=42):
    """Create a synthetic label volume with non-overlapping spheres."""
    rng = np.random.RandomState(seed)
    label = np.zeros(shape, dtype=np.int32)
    inst_id = 1
    for _ in range(n_objects):
        radius = rng.randint(6, 12)
        center = [rng.randint(radius + 1, s - radius - 1) for s in shape]
        zz, yy, xx = np.ogrid[
            -center[0] : shape[0] - center[0],
            -center[1] : shape[1] - center[1],
            -center[2] : shape[2] - center[2],
        ]
        mask = (zz**2 + yy**2 + xx**2) <= radius**2
        # Only place if no overlap
        if not np.any(label[mask] > 0):
            label[mask] = inst_id
            inst_id += 1
    return label


class TestBatchSkeletonizeCoordinates:
    """Verify _batch_skeletonize returns voxel-space coordinates."""

    def test_vertices_within_bounds(self):
        """Skeleton vertices must be valid voxel indices."""
        label = _make_label_volume()
        resolution = (30.0, 8.0, 8.0)
        vertices = _batch_skeletonize(label, resolution)

        assert len(vertices) > 0, "Expected at least one skeleton"
        for inst_id, verts in vertices.items():
            assert verts.shape[1] == 3
            for dim in range(3):
                assert np.all(verts[:, dim] >= 0), (
                    f"Instance {inst_id}: negative coordinate in dim {dim}"
                )
                assert np.all(verts[:, dim] < label.shape[dim]), (
                    f"Instance {inst_id}: coordinate >= shape in dim {dim}"
                )

    def test_vertices_hit_foreground(self):
        """Skeleton vertices should land on their own instance."""
        label = _make_label_volume()
        resolution = (8.0, 8.0, 8.0)
        vertices = _batch_skeletonize(label, resolution)

        for inst_id, verts in vertices.items():
            labels_at_verts = label[verts[:, 0], verts[:, 1], verts[:, 2]]
            hit_rate = np.mean(labels_at_verts == inst_id)
            assert hit_rate > 0.5, (
                f"Instance {inst_id}: only {hit_rate:.0%} of skeleton voxels "
                f"land on the correct instance"
            )


class TestSDTPrecomputedMatchesDirect:
    """Compare direct SDT with precomputed-skeleton SDT."""

    @pytest.fixture()
    def label_and_params(self):
        label = _make_label_volume(shape=(48, 48, 48), n_objects=4, seed=123)
        resolution = (8.0, 8.0, 8.0)
        alpha = 0.8
        bg_value = -1.0
        return label, resolution, alpha, bg_value

    def test_sdt_values_match(self, label_and_params):
        """Direct SDT and precomputed-skeleton SDT should produce identical results."""
        label, resolution, alpha, bg_value = label_and_params

        # Path 1: direct (skeletonize + EDT in one call)
        sdt_direct = skeleton_aware_distance_transform(
            label, resolution=resolution, alpha=alpha, bg_value=bg_value,
        )

        # Path 2: precomputed skeleton volume, then EDT from it
        skeleton_vertices = _batch_skeletonize(label, resolution)
        skel_vol = np.zeros(label.shape, dtype=label.dtype)
        for inst_id, verts in skeleton_vertices.items():
            valid = np.all((verts >= 0) & (verts < np.array(label.shape)), axis=1)
            verts = verts[valid]
            if len(verts) > 0:
                skel_vol[verts[:, 0], verts[:, 1], verts[:, 2]] = inst_id

        sdt_precomputed = skeleton_aware_edt_from_skeleton_vol(
            label, skel_vol, resolution=resolution, alpha=alpha, bg_value=bg_value,
        )

        # Both should have the same shape and close values
        assert sdt_direct.shape == sdt_precomputed.shape

        # Background regions should match exactly
        bg_mask = label == 0
        np.testing.assert_array_equal(
            sdt_direct[bg_mask], sdt_precomputed[bg_mask],
            err_msg="Background values differ",
        )

        # Foreground regions should be close (direct path uses padding, precomputed doesn't,
        # but with padding=False they should match)
        fg_mask = label > 0
        np.testing.assert_allclose(
            sdt_direct[fg_mask], sdt_precomputed[fg_mask],
            atol=1e-5, rtol=1e-5,
            err_msg="Foreground SDT values differ between direct and precomputed paths",
        )

    def test_skeleton_volume_nonzero(self, label_and_params):
        """Precomputed skeleton volume should have nonzero voxels."""
        label, resolution, _, _ = label_and_params
        skeleton_vertices = _batch_skeletonize(label, resolution)

        skel_vol = np.zeros(label.shape, dtype=label.dtype)
        for inst_id, verts in skeleton_vertices.items():
            valid = np.all((verts >= 0) & (verts < np.array(label.shape)), axis=1)
            verts = verts[valid]
            if len(verts) > 0:
                skel_vol[verts[:, 0], verts[:, 1], verts[:, 2]] = inst_id

        n_skel = int((skel_vol > 0).sum())
        n_fg = int((label > 0).sum())
        assert n_skel > 0, "Skeleton volume has 0 voxels — coordinate conversion bug?"
        assert n_skel < n_fg, "Skeleton should be sparser than the full segmentation"
