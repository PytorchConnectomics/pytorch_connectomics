import numpy as np
import pytest

import connectomics.data.processing.bbox as bbox_module
import connectomics.metrics.completeness as completeness_module
from connectomics.data.processing.bbox import apply_lut, compute_bbox_all, seg_stats
from connectomics.metrics.completeness import (
    BORDER,
    MIN_SIZE,
    MIN_SPAN_FRAC,
    completeness_report,
)


def test_seg_stats_matches_compute_bbox_all_and_collects_centroids(monkeypatch):
    seg = np.zeros((4, 5, 6), dtype=np.uint32)
    seg[0:2, 1:4, 2:5] = 1
    seg[3, 0, 5] = 4

    statistics = bbox_module.cc3d.statistics
    calls = 0

    def counted_statistics(labels):
        nonlocal calls
        calls += 1
        return statistics(labels)

    monkeypatch.setattr(bbox_module.cc3d, "statistics", counted_statistics)
    bounds, sizes, centroids = seg_stats(seg, want_centroids=True)

    expected_rows = compute_bbox_all(seg, do_count=True)
    expected = {int(row[0]): tuple(int(value) for value in row[1:7]) for row in expected_rows}
    assert bounds == expected
    assert calls == 1
    assert int(sizes[1]) == 18
    assert int(sizes[4]) == 1
    assert 2 not in bounds
    assert centroids[1] == pytest.approx((0.5, 2.0, 3.0))
    assert centroids[4] == pytest.approx((3.0, 0.0, 5.0))


def test_apply_lut_matches_fancy_indexing_and_mutates_in_place():
    seg = np.array(
        [
            [[0, 1], [2, 3]],
            [[3, 2], [1, 0]],
            [[1, 3], [0, 2]],
        ],
        dtype=np.uint16,
    )
    original = seg.copy()
    lut = np.array([0, 11, 7, 2], dtype=np.int64)

    result = apply_lut(seg, lut, chunk=2)

    assert result is seg
    np.testing.assert_array_equal(result, lut[original])
    assert result.dtype == np.uint16


def _mixed_completeness_segmentation():
    seg = np.zeros((16, 128, 256), dtype=np.uint32)
    # Complete: both z ends touch a border, and size exceeds MIN_SIZE.
    seg[:, 10:26, 10:90] = 1
    # Incomplete: exactly MIN_SIZE and exactly MIN_SPAN_FRAC * Z, wholly interior.
    seg[5:9, 50:100, 130:230] = 2
    return seg


def test_completeness_report_preserves_size_span_and_border_gates(capsys):
    assert (MIN_SPAN_FRAC, MIN_SIZE, BORDER) == (0.25, 20000, 2)
    seg = _mixed_completeness_segmentation()

    assert completeness_report(seg) == (1, 2)

    output = capsys.readouterr().out
    assert "2 decent axons" in output
    assert "COMPLETE (>=2 border ends): 1 (50%)" in output
    assert "seg 2: sz20000 z5-8 (4sl) border-ends 0" in output


def test_completeness_accepts_two_contacts_on_the_same_face(capsys):
    seg = np.zeros((16, 128, 128), dtype=np.uint32)
    seg[5:11, 30:90, 0:60] = 1

    assert completeness_report(seg) == (1, 1)
    assert "INCOMPLETE: 0 (0%)" in capsys.readouterr().out


def test_completeness_reuses_cached_stats(monkeypatch, capsys):
    seg = _mixed_completeness_segmentation()
    bounds, sizes, _ = seg_stats(seg)

    def unexpected_statistics(_seg):
        raise AssertionError("cached stats should avoid recomputation")

    monkeypatch.setattr(completeness_module, "seg_stats", unexpected_statistics)

    assert completeness_report(seg, stats=(bounds, sizes)) == (1, 2)
    capsys.readouterr()
