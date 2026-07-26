import numpy as np
import pytest

from connectomics.decoding.decoders.branch import linking, sections


def test_seg_2d_keeps_sections_and_makes_ids_volume_unique(monkeypatch):
    local_sections = np.array([[[1, 1], [0, 2]]], dtype=np.uint64)

    monkeypatch.setattr(
        sections,
        "_watershed_concat",
        lambda _aff: local_sections.copy(),
    )

    import waterz

    def fake_agglomerate(_aff, **kwargs):
        assert kwargs["thresholds"] == [0.3]
        assert kwargs["aff_threshold_low"] == sections.AFF_LOW
        assert kwargs["aff_threshold_high"] == 0.98
        yield kwargs["fragments"]

    monkeypatch.setattr(waterz, "agglomerate", fake_agglomerate)

    affinity = np.ones((3, 2, 2, 2), dtype=np.float32)
    result = sections.seg_2d(affinity)

    assert result.dtype == np.uint32
    assert set(np.unique(result[0])) == {0, 1, 2}
    assert set(np.unique(result[1])) == {0, 3, 4}
    assert not (set(np.unique(result[0])) - {0}) & (set(np.unique(result[1])) - {0})


def test_branch_link_joins_mutual_consecutive_sections():
    sections_2d = np.zeros((3, 5, 8), dtype=np.uint32)
    sections_2d[0, 1:4, 1:3] = 1
    sections_2d[0, 1:4, 5:7] = 2
    sections_2d[1, 1:4, 1:3] = 3
    sections_2d[1, 1:4, 5:7] = 4
    sections_2d[2, 1:4, 1:3] = 5
    sections_2d[2, 1:4, 5:7] = 6

    affinity = np.zeros((3,) + sections_2d.shape, dtype=np.float32)
    result = linking.branch_link(affinity, sections_2d)

    assert result.dtype == np.uint32
    left = [int(result[z, 2, 1]) for z in range(3)]
    right = [int(result[z, 2, 5]) for z in range(3)]
    assert len(set(left)) == 1
    assert len(set(right)) == 1
    assert left[0] != right[0]


def test_branch_link_rejects_non_unique_cross_slice_section_ids():
    sections_2d = np.zeros((2, 3, 3), dtype=np.uint32)
    sections_2d[0, 1, 1] = 1
    sections_2d[1, 1, 1] = 1
    affinity = np.zeros((3,) + sections_2d.shape, dtype=np.float32)

    with pytest.raises(AssertionError, match="appears on multiple z-slices"):
        linking.branch_link(affinity, sections_2d)


def test_branch_link_lut_preserves_sparse_high_ids_and_supports_inplace():
    segmentation = np.array([[[1, 2, 99]]], dtype=np.uint32)
    pairs = np.array([[1, 2]], dtype=np.uint64)

    result = linking._apply(segmentation, pairs, inplace=True)

    assert result is segmentation
    assert result[0, 0, 0] == result[0, 0, 1]
    assert result[0, 0, 2] == 99
