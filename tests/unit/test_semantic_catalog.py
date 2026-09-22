"""Contracts for reusable semantic classification and evaluation artifacts."""

import hashlib
import json

import h5py
import numpy as np
import pytest

from connectomics.evaluation.semantic import (
    build_semantic_catalog,
    summarize_semantic_records,
    write_semantic_artifacts,
)
from connectomics.metrics.unsupervised.classification import classify_semantic_candidate


@pytest.mark.parametrize(
    "kind,radius,shaft,expected",
    [
        ("axon_like", 0.08, 0, ("axon", "local_axon_caliber")),
        ("dendrite_like", 0.1, 4, ("axon", "thin_shaft_with_branches_or_swellings")),
        ("dendrite_like", 0.1, 0, ("unclassified", "branch_density_without_thick_backbone")),
        ("dendrite_like", 0.3, 0, ("dendrite", "thick_backbone_candidate")),
        ("ambiguous_caliber", 0.18, 1, ("unclassified", "ambiguous_caliber")),
        ("unmeasured", None, 0, ("unclassified", "no_local_caliber")),
    ],
)
def test_semantic_candidates_from_measurements(kind, radius, shaft, expected):
    assert (
        classify_semantic_candidate(
            kind, radius, shaft, axon_max_radius_um=0.15, dendrite_min_radius_um=0.2
        )
        == expected
    )


@pytest.mark.parametrize("gates", [(0, 0.2), (0.3, 0.2), (0.15, float("nan"))])
def test_invalid_caliber_gates_rejected(gates):
    with pytest.raises(ValueError, match="Caliber gates"):
        classify_semantic_candidate(
            "axon_like", 0.1, 1, axon_max_radius_um=gates[0], dendrite_min_radius_um=gates[1]
        )


def test_volume_denominators_include_unclassified_and_background():
    records = [
        {"semantic_class": "axon", "voxel_count": 60},
        {"semantic_class": "unclassified", "voxel_count": 20},
    ]
    summary = summarize_semantic_records(records, (1, 10, 10), 0.5)
    assert summary["foreground_percent"] == 80
    assert summary["background_voxels"] == 20
    assert summary["foreground_volume_um3"] == 40
    categories = {r["class"]: r for r in summary["categories"]}
    assert categories["axon"]["percent_foreground"] == 75
    assert categories["axon"]["percent_roi"] == 60
    assert categories["blood_vessel"]["assessment_status"] == "not_assessed"
    assert sum(r["percent_foreground"] for r in categories.values()) == pytest.approx(100)


@pytest.fixture
def sources(tmp_path):
    segmentation = tmp_path / "labels.h5"
    with h5py.File(segmentation, "w") as handle:
        handle["main"] = np.array([[[0, 1, 1, 2], [3, 3, 3, 3]]], dtype=np.uint32)
    sizes = tmp_path / "sizes.npz"
    np.savez(sizes, ids=[1, 2, 3], counts=[2, 1, 4])
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "metadata": {
                    "segmentation_sha256": hashlib.sha256(segmentation.read_bytes()).hexdigest(),
                    "shape_zyx": [1, 2, 4],
                    "spacing_nm_zyx": [1000, 1000, 1000],
                    "coordinate_convention": "voxel centers, ZYX micrometers",
                    "continuity_config": {
                        "axon_max_radius_um": 0.15,
                        "dendrite_min_radius_um": 0.2,
                    },
                },
                "segments": [
                    {
                        "id": "1",
                        "voxel_count": 2,
                        "semantic_type": "axon_like",
                        "continuity": {"caliber_radius_um": 0.1},
                    },
                    {"id": "3", "voxel_count": 4, "semantic_type": "unmeasured"},
                ],
            }
        )
    )
    return analysis, segmentation, sizes


def test_catalog_roundtrip_preserves_all_labels_and_large_unknowns(sources, tmp_path):
    catalog = build_semantic_catalog(*sources, layer_uri="gs://example/labels", large_volume_um3=3)
    assert [r["id"] for r in catalog["segments"]] == ["3", "1", "2"]
    assert catalog["segments"][0]["classification_basis"] == "no_local_caliber"
    assert catalog["segments"][2]["classification_basis"] == "below_analysis_cutoff"
    assert catalog["summary"]["foreground_percent"] == 87.5
    paths = write_semantic_artifacts(catalog, tmp_path / "artifacts", title="Synthetic dataset")
    assert json.loads(paths["semantic_segmentation.json"].read_text()) == catalog
    assert json.loads(paths["semantic_summary.json"].read_text()) == catalog["summary"]
    queue = json.loads(paths["semantic_review_queue.json"].read_text())["segments"]
    assert [r["id"] for r in queue] == ["3"]
    assert queue[0]["percent_foreground"] == pytest.approx(400 / 7)
    assert paths["semantic_composition.png"].read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("corrupt", ["hash", "counts"])
def test_catalog_rejects_mismatched_sources(sources, corrupt):
    analysis, segmentation, sizes = sources
    if corrupt == "hash":
        with h5py.File(segmentation, "r+") as handle:
            handle["main"][0, 0, 0] = 1
        message = "Segmentation differs"
    else:
        np.savez(sizes, ids=[1, 2, 3], counts=[1, 2, 4])
        message = "Cached label histogram differs"
    with pytest.raises(ValueError, match=message):
        build_semantic_catalog(*sources, layer_uri="gs://example/labels")


def test_semantic_evaluation_api_is_explicit():
    from connectomics.evaluation import semantic

    assert set(semantic.__all__) == {
        "build_semantic_catalog",
        "summarize_semantic_records",
        "write_semantic_artifacts",
    }
