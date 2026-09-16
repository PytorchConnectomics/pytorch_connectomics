"""Contracts for reusable class decisions from independent morphology evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from connectomics.metrics.unsupervised.arbor import BackboneConfig, analyze_arbor
from connectomics.metrics.unsupervised.classification import (
    SegmentClassification,
    SegmentClassificationConfig,
    classify_segment,
)
from connectomics.metrics.unsupervised.morphology import MORPHOLOGY_CLASSES


@pytest.fixture(scope="module")
def thick_backbone():
    return analyze_arbor(
        np.array([[0, 0, 0], [4, 0, 0], [8, 0, 0]]),
        np.array([[0, 1], [1, 2]]),
        np.full(3, 0.2),
    )


def test_explicit_public_api_and_json_safe_decision(thick_backbone):
    import connectomics.metrics.unsupervised.classification as classification

    assert set(classification.__all__) == {
        "SegmentClassificationConfig",
        "SegmentClassification",
        "classify_segment",
    }
    result = classify_segment("branched_process_candidate", 2000, arbor=thick_backbone)
    assert isinstance(result, SegmentClassification)
    serialized = json.loads(json.dumps(asdict(result), allow_nan=False))
    assert serialized["backbone_candidate"] is True
    assert serialized["display_class"] == "dendrite_like_candidate"
    assert set(serialized) == {
        "morphology_class",
        "automatic_class",
        "display_class",
        "automatic_basis",
        "backbone_assessed",
        "backbone_candidate",
        "display_basis",
    }


@pytest.mark.parametrize("morphology_class", MORPHOLOGY_CLASSES)
def test_without_arbor_or_review_every_canonical_profile_class_is_preserved(morphology_class):
    result = classify_segment(morphology_class, 2000)

    assert result.morphology_class == result.automatic_class == result.display_class
    assert result.display_class == morphology_class
    assert result.automatic_basis == "profile_geometry"
    assert result.display_basis == "automatic"
    assert not result.backbone_assessed
    assert not result.backbone_candidate


def test_backbone_refinement_preserves_independent_profile_evidence(thick_backbone):
    result = classify_segment("branched_process_candidate", 2000, arbor=thick_backbone)

    assert result.morphology_class == "branched_process_candidate"
    assert result.automatic_class == result.display_class == "dendrite_like_candidate"
    assert result.automatic_basis == "backbone_geometry"
    assert result.backbone_assessed
    assert result.backbone_candidate


@pytest.mark.parametrize(
    "changes",
    [
        {"diameter_length_um": None},
        {"diameter_median_radius_um": None},
        {"diameter_length_um": np.nan},
        {"diameter_median_radius_um": np.inf},
        {"has_cycles": True},
        {"component_count": 2},
        {"diameter_length_um": 3.0},
        {"diameter_median_radius_um": 0.05},
        {"diameter_median_radius_um": 0.8},
    ],
)
def test_failed_or_unknown_backbone_keeps_neutral_profile_class(changes, thick_backbone):
    result = classify_segment(
        "branched_process_candidate", 2000, arbor=replace(thick_backbone, **changes)
    )

    assert result.automatic_class == result.display_class == "branched_process_candidate"
    assert result.automatic_basis == "profile_geometry"
    assert result.backbone_assessed
    assert not result.backbone_candidate
    json.dumps(asdict(result), allow_nan=False)


def test_profile_dendrite_evidence_survives_a_failed_backbone_gate(thick_backbone):
    result = classify_segment(
        "dendrite_like_candidate",
        2000,
        arbor=replace(thick_backbone, diameter_median_radius_um=0.05),
    )

    assert result.morphology_class == result.automatic_class == "dendrite_like_candidate"
    assert result.automatic_basis == "profile_geometry"
    assert not result.backbone_candidate


@pytest.mark.parametrize("reviewed_class", ["glia", "purkinje_dendrite"])
def test_semantic_review_changes_only_display_and_never_automatic_evidence(
    reviewed_class, thick_backbone
):
    automatic = classify_segment("branched_process_candidate", 2000, arbor=thick_backbone)
    reviewed = classify_segment(
        "branched_process_candidate", 2000, arbor=thick_backbone, reviewed_class=reviewed_class
    )

    assert reviewed == replace(
        automatic, display_class=reviewed_class, display_basis="user_annotation"
    )
    assert reviewed.automatic_class == "dendrite_like_candidate"


@pytest.mark.parametrize("voxel_count", [1, 999, 1000])
@pytest.mark.parametrize("reviewed_class", [None, "glia"])
def test_crumbs_size_precedence_over_arbor_and_semantic_review(
    voxel_count, reviewed_class, thick_backbone
):
    result = classify_segment(
        "branched_process_candidate",
        voxel_count,
        arbor=thick_backbone,
        reviewed_class=reviewed_class,
    )

    assert result.automatic_class == "dendrite_like_candidate"
    if voxel_count < 1000:
        assert result.display_class == "crumbs"
        assert result.display_basis == "size"
    else:
        assert result.display_class == (reviewed_class or "dendrite_like_candidate")
        assert result.display_basis == ("user_annotation" if reviewed_class else "automatic")


def test_custom_size_threshold_and_one_voxel_disable_default_crumbs_cutoff():
    custom = SegmentClassificationConfig(crumbs_max_voxels_exclusive=50)
    assert classify_segment("unclassified", 49, config=custom).display_class == "crumbs"
    assert classify_segment("unclassified", 50, config=custom).display_class == "unclassified"
    disabled = SegmentClassificationConfig(crumbs_max_voxels_exclusive=1)
    assert classify_segment("unclassified", 1, config=disabled).display_class == "unclassified"


def test_custom_physical_backbone_gate_is_independent_of_size_threshold(thick_backbone):
    strict = SegmentClassificationConfig(
        crumbs_max_voxels_exclusive=1,
        backbone=BackboneConfig(min_diameter_length_um=10),
    )
    default = classify_segment("branched_process_candidate", 2000, arbor=thick_backbone)
    result = classify_segment(
        "branched_process_candidate", 2000, arbor=thick_backbone, config=strict
    )

    assert default.backbone_candidate
    assert not result.backbone_candidate
    assert result.display_class == "branched_process_candidate"


def test_numpy_integer_counts_and_config_are_normalized_without_precision_loss():
    config = SegmentClassificationConfig(crumbs_max_voxels_exclusive=np.int64(1000))
    assert type(config.crumbs_max_voxels_exclusive) is int
    assert classify_segment("unclassified", np.uint64(2**63), config=config).display_class == (
        "unclassified"
    )


@pytest.mark.parametrize("value", [0, -1, 1.0, np.float64(1000), True, np.bool_(True), "1000"])
def test_invalid_voxel_counts_and_size_thresholds_are_rejected(value):
    with pytest.raises(ValueError, match="voxel_count must be a positive integer"):
        classify_segment("unclassified", value)
    with pytest.raises(ValueError, match="crumbs_max_voxels_exclusive must be a positive integer"):
        SegmentClassificationConfig(crumbs_max_voxels_exclusive=value)


@pytest.mark.parametrize("value", ["crumbs", "glia", "bushy_glia_like_candidate", "", None])
def test_noncanonical_profile_classes_are_rejected(value):
    with pytest.raises(ValueError, match="Unknown morphology_class"):
        classify_segment(value, 2000)


@pytest.mark.parametrize("value", ["crumbs", "", " ", " glia", "glia ", 42])
def test_invalid_review_labels_are_rejected_even_for_crumbs(value):
    with pytest.raises(ValueError, match="reviewed_class"):
        classify_segment("unclassified", 1, reviewed_class=value)


def test_invalid_typed_configuration_and_arbor_inputs_are_rejected():
    with pytest.raises(TypeError, match="backbone must be a BackboneConfig"):
        SegmentClassificationConfig(backbone={})
    with pytest.raises(TypeError, match="config must be a SegmentClassificationConfig"):
        classify_segment("unclassified", 2000, config={})
    with pytest.raises(TypeError, match="arbor must be ArborMetrics or None"):
        classify_segment("unclassified", 2000, arbor={})
