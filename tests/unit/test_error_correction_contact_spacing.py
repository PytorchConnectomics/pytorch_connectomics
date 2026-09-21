"""Physical contact geometry and source-edge ownership for anisotropic volumes."""

import numpy as np
import pytest

from connectomics.decoding.error_correction.affinity import restore_sigmoid
from connectomics.decoding.error_correction.contacts import consolidate_rows, face_rows


@pytest.mark.parametrize("axis", range(3))
def test_contact_uses_source_affinity_channel_and_physical_face_center(axis):
    labels = np.ones((3, 3, 3), dtype=np.uint32)
    destination = [slice(None)] * 3
    destination[axis] = 2
    labels[tuple(destination)] = 2
    stored = np.full((3, 3, 3, 3), 0.5, dtype=np.float32)
    source = [slice(None)] * 3
    source[axis] = 1
    expected_affinity = 0.83
    stored[(axis, *source)] = 1 / (
        1 + np.exp(-0.2 * np.log(expected_affinity / (1 - expected_affinity)))
    )
    restored = restore_sigmoid(stored)
    spacing = np.asarray((24.0, 18.0, 18.0))
    origin = np.asarray((10, 20, 30))
    rows = face_rows(
        labels,
        restored,
        np.asarray((1, 2)),
        np.asarray(labels.shape),
        origin,
        0,
        3,
        axis,
        spacing_zyx_nm=spacing,
    )
    graph = consolidate_rows(
        *[[value] for value in rows], np.asarray((1, 2)), spacing_zyx_nm=spacing
    )
    np.testing.assert_array_equal(graph["count"], [9])
    np.testing.assert_allclose(graph["affinity_sum"] / graph["count"], expected_affinity, atol=1e-6)
    expected = origin.astype(float) + 1.5
    expected[axis] = origin[axis] + 2
    np.testing.assert_allclose(graph["centroid_nm_zyx"], (expected * spacing)[None])
    np.testing.assert_allclose(graph["area_nm2"], [9 * np.prod(np.delete(spacing, axis))])
    np.testing.assert_array_equal(graph["axis_count_zyx"][0], np.eye(3, dtype=int)[axis] * 9)


@pytest.mark.parametrize("slab_depth", (1, 2, 3))
def test_positive_halo_owns_every_slab_seam_once(slab_depth):
    rng = np.random.default_rng(73)
    labels = rng.integers(0, 4, size=(5, 4, 3), dtype=np.uint32)
    affinity = rng.random((3, *labels.shape), dtype=np.float32)
    present = np.asarray((1, 2, 3))
    spacing = np.asarray((24.0, 18.0, 18.0))

    def extract(depth):
        parts = [[] for _ in range(9)]
        for lo in range(0, len(labels), depth):
            hi = min(lo + depth, len(labels))
            for axis in range(3):
                rows = face_rows(
                    labels[lo : hi + 1],
                    affinity[:, lo:hi],
                    present,
                    np.asarray((hi - lo, *labels.shape[1:])),
                    np.asarray((lo, 0, 0)),
                    0,
                    hi - lo,
                    axis,
                    spacing_zyx_nm=spacing,
                )
                if len(rows[0]):
                    for part, row in zip(parts, rows):
                        part.append(row)
        return consolidate_rows(*parts, present, spacing_zyx_nm=spacing)

    actual, expected = extract(slab_depth), extract(len(labels))
    for field in actual:
        np.testing.assert_allclose(actual[field], expected[field], err_msg=field)


@pytest.mark.parametrize("spacing", ((1, 2), (1, 0, 3), (1, -1, 3), (1, np.nan, 3)))
def test_contact_rejects_invalid_spacing(spacing):
    with pytest.raises(ValueError, match="three finite positive"):
        face_rows(
            np.ones((2, 2, 2)),
            np.ones((3, 2, 2, 2)),
            np.asarray((1,)),
            np.asarray((2, 2, 2)),
            np.zeros(3),
            0,
            2,
            0,
            spacing_zyx_nm=spacing,
        )
