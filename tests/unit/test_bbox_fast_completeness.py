import numpy as np
import pytest

import connectomics.data.processing.bbox as bbox_module
import connectomics.metrics.tube as tube_module
from connectomics.data.processing.bbox import apply_lut, compute_bbox_all, seg_stats
from connectomics.metrics.tube import (
    BORDER,
    MIN_SIZE,
    MIN_SPAN_FRAC,
    TubeAnalysisConfig,
    analyze_tubes,
    completeness_report,
    format_tube_analysis,
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
    assert "decent (long and voxels>=20000): 2" in output
    assert "COMPLETE (>=2 border ends): 1/2 (50.0%" in output
    assert "seg 2: voxels 20000, z5-8 (4 span), border ends 0" in output


def test_completeness_accepts_two_contacts_on_the_same_face(capsys):
    seg = np.zeros((16, 128, 128), dtype=np.uint32)
    seg[5:11, 30:90, 0:60] = 1

    assert completeness_report(seg) == (1, 1)
    assert "INCOMPLETE: 0" in capsys.readouterr().out


def test_completeness_reuses_cached_stats(monkeypatch, capsys):
    seg = _mixed_completeness_segmentation()
    bounds, sizes, _ = seg_stats(seg)

    def unexpected_statistics(_seg):
        raise AssertionError("cached stats should avoid recomputation")

    monkeypatch.setattr(tube_module, "seg_stats", unexpected_statistics)

    assert completeness_report(seg, stats=(bounds, sizes)) == (1, 2)
    capsys.readouterr()


def _tube_quality_segmentation():
    seg = np.zeros((12, 48, 64), dtype=np.uint16)

    # A complete U-like tube: both z ends touch x0 while the body moves inward.
    seg[0:3, 8:13, 0:6] = 1
    seg[3:9, 8:13, 4:10] = 1
    seg[9:12, 8:13, 0:6] = 1

    # A connected parallel merge: two persistent strands joined on one slice.
    seg[:, 22:27, 20:25] = 2
    seg[:, 22:27, 34:39] = 2
    seg[5, 22:27, 25:34] = 2

    # Two significant disconnected pieces with the same label.
    seg[2:5, 34:39, 4:9] = 3
    seg[7:10, 34:39, 50:55] = 3
    return seg


def test_analyze_tubes_reports_completeness_parallelism_and_disconnection():
    seg = _tube_quality_segmentation()
    config = TubeAnalysisConfig(
        substantial_min_z_slices=4,
        substantial_min_voxels=20,
        long_span_fraction=0.25,
        decent_min_voxels=40,
        border_margin=1,
        border_patch_min_voxels=5,
        multi_component_min_voxels=10,
        multi_component_slice_step=1,
        parallel_min_slices=4,
        parallel_fraction_threshold=0.3,
        disconnected_component_min_voxels=20,
        bump_min_slices=4,
        bump_relative_excess=0.2,
        bump_absolute_excess=10,
        bump_max_slices=2,
        bump_median_window=3,
    )

    result = analyze_tubes(seg, config)
    records = {tube.label: tube for tube in result.tubes}

    assert records[1].is_complete is True
    assert records[1].face_contacts == ("x0", "z0", "zmax")
    assert records[1].border_patch_count == 2
    assert records[1].is_valid_tube is True

    assert records[2].is_parallel is True
    assert records[2].is_disconnected is False
    assert records[2].bump_count == 1
    assert records[2].is_valid_tube is False

    assert records[3].is_disconnected is True
    assert records[3].significant_component_count_3d == 2

    summary = result.summary
    assert summary.total_label_count == 3
    assert summary.substantial_count == 3
    assert summary.parallel_count == 1
    assert summary.disconnected_count == 1
    assert summary.complete_count == 2
    assert summary.valid_count == 1


def test_tube_analysis_distinguishes_face_count_from_border_ends():
    seg = np.zeros((8, 24, 24), dtype=np.uint16)
    # One open endpoint at a corner touches two faces but is only one border end.
    seg[2:4, 0:4, 0:4] = 1
    seg[4:7, 4:8, 4:8] = 1
    config = TubeAnalysisConfig(
        substantial_min_z_slices=2,
        substantial_min_voxels=10,
        long_span_fraction=0.25,
        decent_min_voxels=10,
        border_margin=0,
        border_patch_min_voxels=1,
        multi_component_min_voxels=2,
        parallel_min_slices=2,
        disconnected_component_min_voxels=2,
        bump_min_slices=3,
        bump_absolute_excess=1,
        bump_max_slices=2,
        bump_median_window=3,
    )

    tube = analyze_tubes(seg, config).tubes[0]

    assert tube.face_count == 2
    assert tube.border_end_count == 1
    assert tube.border_patch_count == 1
    assert tube.is_complete is False


def test_format_tube_analysis_exposes_count_and_volume_rankers():
    result = analyze_tubes(
        _tube_quality_segmentation(),
        TubeAnalysisConfig(
            substantial_min_z_slices=4,
            substantial_min_voxels=20,
            long_span_fraction=0.25,
            decent_min_voxels=40,
            border_margin=1,
            border_patch_min_voxels=5,
            multi_component_min_voxels=10,
            multi_component_slice_step=1,
            parallel_min_slices=4,
            disconnected_component_min_voxels=20,
            bump_min_slices=4,
            bump_absolute_excess=10,
            bump_max_slices=2,
            bump_median_window=3,
        ),
    )

    report = format_tube_analysis(result, top_incomplete=1)

    assert "3 substantial" in report
    assert "VALID (complete and single): 1/3" in report
    assert "parallel 1, disconnected 1" in report


@pytest.mark.parametrize(
    "seg, error",
    [
        (np.zeros((3, 3), dtype=np.uint8), ValueError),
        (np.zeros((3, 3, 3), dtype=np.float32), TypeError),
        (-np.ones((3, 3, 3), dtype=np.int8), ValueError),
    ],
)
def test_analyze_tubes_rejects_invalid_segmentations(seg, error):
    with pytest.raises(error):
        analyze_tubes(seg)


def test_tube_analysis_has_an_explicit_package_api():
    import connectomics.metrics as metrics

    expected = {
        "TubeAnalysis",
        "TubeAnalysisConfig",
        "TubeAnalysisSummary",
        "TubeRecord",
        "analyze_tubes",
        "completeness_report",
        "format_tube_analysis",
    }

    assert expected <= set(metrics.__all__)
