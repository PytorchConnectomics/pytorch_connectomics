from __future__ import annotations

import json

import numpy as np
import pytest

from connectomics.metrics.unsupervised.morphology import (
    MORPHOLOGY_CLASSES,
    MorphologyConfig,
    analyze_morphology,
)


def _config(**kwargs):
    values = {
        "voxel_size_um": (0.1, 0.1, 0.1),
        "min_voxels": 1,
        "min_axon_length_um": 1.0,
        "component_min_voxels": 2,
        "border_margin_um": 0.0,
        "profile_smoothing_um": 0.0,
    }
    values.update(kwargs)
    return MorphologyConfig(**values)


def test_morphology_public_api_snapshot():
    import connectomics.metrics.unsupervised.morphology as morphology

    assert set(morphology.__all__) == {
        "MORPHOLOGY_CLASSES",
        "MorphologyConfig",
        "MorphologyRecord",
        "MorphologyAnalysis",
        "analyze_morphology",
    }


@pytest.mark.parametrize("axis", [0, 2])
def test_straight_tubes_have_physical_radius_length_and_opposite_terminal_faces(axis):
    shape = [12, 12, 12]
    shape[axis] = 40
    seg = np.zeros(shape, dtype=np.uint32)
    selection = [slice(5, 7)] * 3
    selection[axis] = slice(None)
    seg[tuple(selection)] = 17

    result = analyze_morphology(seg, _config())
    record = result.records[0]

    assert record.label == 17
    assert record.dominant_axis == axis
    assert record.axis_angles_deg_zyx[axis] == pytest.approx(0)
    assert record.radius_um == pytest.approx(np.sqrt(0.04 / np.pi))
    assert record.length_um == pytest.approx(4.0)
    assert record.chord_length_um == pytest.approx(4.0)
    assert record.continuity_proxy == pytest.approx(1)
    assert record.tortuosity == pytest.approx(1)
    assert record.boundary_end_count == 2
    assert record.boundary_end_faces == (("z0", "zmax") if axis == 0 else ("x0", "xmax"))
    assert record.morphology_class == "good_axon_candidate"
    assert result.config == _config()
    assert result.volume_shape == tuple(shape)


def test_oblique_tube_uses_physical_pca_and_corrected_cross_section():
    spacing = np.array((0.04, 0.02, 0.02))
    z, y, x = np.ogrid[:50, :64, :140]
    perpendicular = ((x + 0.5) * spacing[2] - (z + 0.5) * spacing[0] - 0.4) / np.sqrt(2)
    distance_squared = perpendicular**2 + ((y + 0.5) * spacing[1] - 0.64) ** 2
    seg = (distance_squared <= 0.08**2).astype(np.uint8)

    record = analyze_morphology(
        seg, _config(voxel_size_um=tuple(spacing), border_margin_um=0.04)
    ).records[0]

    assert record.axis_angles_deg_zyx[0] == pytest.approx(45, abs=1)
    assert record.axis_angles_deg_zyx[2] == pytest.approx(45, abs=1)
    assert record.radius_um == pytest.approx(0.08, abs=0.008)
    assert record.length_um == pytest.approx(2 * np.sqrt(2), rel=0.06)
    assert 0.9 < record.continuity_proxy <= 1.0


def test_disconnected_same_id_never_bridges_the_gap_in_length_or_continuity():
    seg = np.zeros((20, 12, 12), dtype=np.uint16)
    seg[:8, 5:7, 5:7] = 1
    seg[12:, 5:7, 5:7] = 1

    result = analyze_morphology(seg, _config(min_axon_length_um=0.5))
    record = result.records[0]

    assert record.component_count == record.significant_component_count == 2
    assert record.component_lengths_um == pytest.approx((0.8, 0.8))
    assert record.length_um == pytest.approx(1.6)
    assert record.chord_length_um == pytest.approx(0.8)
    assert record.continuity_proxy == pytest.approx(0.4)
    assert record.boundary_end_count == 1
    assert record.suspicious_geometry
    assert record.morphology_class == "suspicious_axon_candidate"
    populated_bin = next(item for item in result.summary["radius_bins"] if item["count"])
    assert populated_bin["length_weighted_segment_length_um"] == pytest.approx(0.8)


def test_one_side_wall_contact_is_not_two_terminal_contacts():
    seg = np.zeros((40, 12, 12), dtype=np.uint16)
    seg[8:32, :2, 5:7] = 1

    record = analyze_morphology(seg, _config()).records[0]

    assert record.touches_boundary
    assert record.boundary_end_count == 0
    assert record.morphology_class == "broken_axon_candidate"


def test_profile_smoothing_reduces_voxel_jitter_without_shortening_axial_span():
    seg = np.zeros((60, 20, 20), dtype=np.uint8)
    for z in range(60):
        x = 8 + 2 * int(z % 4 >= 2)
        seg[z, 8:10, x : x + 2] = 1

    raw = analyze_morphology(seg, _config()).records[0]
    smooth = analyze_morphology(seg, _config(profile_smoothing_um=0.2)).records[0]

    assert smooth.length_um < raw.length_um
    assert smooth.length_um == pytest.approx(6, rel=0.1)
    assert smooth.tortuosity >= 1
    assert smooth.boundary_end_count == 2


def test_short_side_cropped_object_is_not_an_interior_fragment():
    seg = np.zeros((20, 20, 20), dtype=np.uint8)
    seg[8:11, :2, 8:10] = 1

    record = analyze_morphology(seg, _config()).records[0]

    assert record.touches_boundary
    assert record.boundary_end_count == 0
    assert record.morphology_class != "small_interior_fragment_candidate"


def test_single_corner_patch_does_not_count_as_two_ends():
    seg = np.zeros((20, 20, 20), dtype=np.uint8)
    seg[:2, :2, :2] = 1

    record = analyze_morphology(seg, _config(border_margin_um=0.1)).records[0]

    assert record.boundary_end_count < 2


def test_sparse_large_ids_are_retained_exactly():
    seg = np.zeros((20, 12, 12), dtype=np.uint64)
    label = 2**63 + 7
    seg[:, 5:7, 5:7] = label

    result = analyze_morphology(seg, _config())

    assert result.records[0].label == label
    assert result.summary["coverage"]["total_label_count"] == 1


def test_sparse_label_without_background_keeps_original_id_and_exact_coverage():
    label = 2**64 - 1
    seg = np.full((20, 3, 2), label, dtype=np.uint64)

    result = analyze_morphology(seg, _config())

    assert result.records[0].label == label
    assert result.records[0].voxel_count == seg.size
    assert result.summary["coverage"]["total_label_count"] == 1
    assert result.summary["coverage"]["analyzed_foreground_fraction"] == 1
    json.dumps(result.summary, allow_nan=False)


@pytest.mark.parametrize("permutation", [(1, 2, 0), (2, 0, 1)])
def test_axis_permutation_with_anisotropic_spacing_preserves_physical_metrics(permutation):
    seg = np.zeros((30, 12, 10), dtype=np.uint16)
    seg[:, 5:7, 4:6] = 3
    spacing = np.array((0.2, 0.1, 0.05))
    reference = analyze_morphology(seg, _config(voxel_size_um=tuple(spacing))).records[0]

    rotated = analyze_morphology(
        np.transpose(seg, permutation),
        _config(voxel_size_um=tuple(spacing[list(permutation)])),
    ).records[0]

    for attribute in ("length_um", "radius_um", "volume_um3", "continuity_proxy", "elongation"):
        assert getattr(rotated, attribute) == pytest.approx(getattr(reference, attribute))
    assert rotated.centroid_um_zyx == pytest.approx(
        np.asarray(reference.centroid_um_zyx)[list(permutation)]
    )
    assert rotated.morphology_class == reference.morphology_class == "good_axon_candidate"
    assert rotated.boundary_end_count == 2


@pytest.mark.parametrize("shape", [(0, 3, 4), (3, 4, 5)])
def test_empty_arrays_have_json_safe_empty_measurements(shape):
    result = analyze_morphology(np.zeros(shape, dtype=np.uint16), _config())

    assert result.records == ()
    assert result.summary["coverage"]["total_label_count"] == 0
    assert result.summary["nerl"] is None
    assert result.summary["radius_bins"][-1]["upper_um"] is None
    json.dumps(result.summary, allow_nan=False)


def test_size_filter_reports_count_and_voxel_coverage():
    seg = np.zeros((20, 12, 12), dtype=np.uint16)
    seg[:, 5:7, 5:7] = 1
    seg[5:7, 1:3, 1:3] = 2

    result = analyze_morphology(seg, _config(min_voxels=10))

    coverage = result.summary["coverage"]
    assert coverage["total_label_count"] == 2
    assert coverage["analyzed_label_count"] == 1
    assert coverage["excluded_voxels"] == 8
    assert coverage["analyzed_foreground_fraction"] == pytest.approx(80 / 88)
    excluded = analyze_morphology(seg, _config(min_voxels=100))
    assert excluded.records == ()
    assert excluded.summary["coverage"]["excluded_label_count"] == 2


def test_radius_bins_are_disjoint_and_weight_observed_component_lengths():
    seg = np.zeros((40, 24, 24), dtype=np.uint16)
    seg[:, 4:6, 4:6] = 1
    seg[10:30, 14:16, 14:16] = 2

    result = analyze_morphology(seg, _config(radius_bins_um=(0.0, 0.1, 0.2)))
    radius_bin = result.summary["radius_bins"][1]

    assert radius_bin["count"] == 2
    assert radius_bin["total_length_um"] == pytest.approx(6)
    assert radius_bin["length_weighted_segment_length_um"] == pytest.approx(20 / 6)
    assert radius_bin["length_weighted_continuity"] == pytest.approx(5 / 6)
    assert sum(item["count"] for item in result.summary["radius_bins"]) == 2
    assert result.summary["nerl"] is None
    assert "reference skeletons" in result.summary["nerl_unavailable_reason"]


def test_connected_parallel_strands_are_suspicious_without_claiming_a_true_merge():
    seg = np.zeros((50, 30, 30), dtype=np.uint8)
    seg[:, 12:14, 8:10] = 1
    seg[:, 12:14, 17:19] = 1
    seg[24:26, 12:14, 8:19] = 1

    record = analyze_morphology(seg, _config()).records[0]

    assert record.component_count == 1
    assert record.multistrand_fraction > 0.8
    assert record.suspicious_geometry
    assert record.morphology_class == "suspicious_axon_candidate"


def test_branching_is_semantically_neutral_and_thick_shafts_are_only_candidates():
    ring = np.zeros((30, 30, 10), dtype=np.uint8)
    ring[4:26, 4:6, 4:6] = 1
    ring[4:26, 24:26, 4:6] = 1
    ring[4:6, 4:26, 4:6] = 1
    ring[24:26, 4:26, 4:6] = 1
    bushy = analyze_morphology(ring, _config()).records[0]
    assert bushy.elongation < 3
    assert bushy.multistrand_fraction > 0.5
    assert bushy.morphology_class == "branched_process_candidate"
    assert "branched_process_candidate" in MORPHOLOGY_CLASSES
    assert "bushy_glia_like_candidate" not in MORPHOLOGY_CLASSES

    thick = np.zeros((70, 30, 30), dtype=np.uint8)
    thick[:, 10:20, 10:20] = 1
    dendrite = analyze_morphology(thick, _config()).records[0]
    assert dendrite.morphology_class == "dendrite_like_candidate"

    fragment = np.zeros((30, 30, 30), dtype=np.uint8)
    fragment[10:14, 14:16, 14:16] = 1
    spine_candidate = analyze_morphology(fragment, _config()).records[0]
    assert spine_candidate.morphology_class == "small_interior_fragment_candidate"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voxel_size_um": (1, 2)},
        {"voxel_size_um": (1, 0, 1)},
        {"voxel_size_um": (1, float("nan"), 1)},
        {"min_voxels": 0},
        {"min_voxels": 1.5},
        {"component_min_voxels": True},
        {"min_elongation": 0.5},
        {"multi_component_fraction": 0},
        {"multi_component_fraction": 1.1},
        {"profile_smoothing_um": -1},
        {"radius_bins_um": (0, 0.1, float("inf"))},
        {"radius_bins_um": (0.1, 0.2)},
        {"radius_bins_um": (0, 0.2, 0.1)},
    ],
)
def test_invalid_configuration_raises(kwargs):
    with pytest.raises(ValueError):
        _config(**kwargs)


@pytest.mark.parametrize(
    "seg",
    [
        np.ones((3, 3), dtype=np.uint8),
        np.zeros((3, 3, 3), dtype=float),
        np.zeros((3, 3, 3), dtype=bool),
        -np.ones((3, 3, 3), dtype=np.int16),
    ],
)
def test_invalid_label_arrays_raise(seg):
    with pytest.raises(ValueError):
        analyze_morphology(seg, _config())
