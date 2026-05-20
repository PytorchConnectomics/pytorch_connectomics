from __future__ import annotations

import csv

import h5py
import numpy as np

from connectomics.decoding.decoders.longrange_guided_split import longrange_guided_split


def _merged_parent_and_two_guides() -> tuple[np.ndarray, np.ndarray]:
    primary = np.zeros((6, 12, 12), dtype=np.uint32)
    primary[:, 1:11, 1:11] = 1

    guide = np.zeros_like(primary)
    guide[:, 1:6, 1:5] = 10
    guide[:, 7:10, 7:10] = 20
    return primary, guide


def test_longrange_guided_split_splits_parent_with_two_large_guide_components(tmp_path):
    primary, guide = _merged_parent_and_two_guides()

    corrected = longrange_guided_split(
        primary,
        guide_seg=guide,
        min_parent_voxels=1,
        min_seed_voxels=1,
        min_seed_axis_extent=1,
        min_seed_overlap_voxels=1,
        min_seed_guide_fraction=0.0,
        min_seeds_in_parent=2,
        bbox_pad=0,
        assignment="nearest",
        report_dir=str(tmp_path),
    )

    assert set(np.unique(corrected)) == {0, 1, 21}
    assert np.all(corrected[guide == 10] == 1)
    assert np.all(corrected[guide == 20] == 21)

    with (tmp_path / "guided_split_candidates.csv").open() as f:
        candidate_rows = list(csv.DictReader(f))
    assert candidate_rows[0]["primary_id"] == "1"
    assert candidate_rows[0]["decision"] == "candidate"
    assert candidate_rows[0]["retained_seed_ids"] == "10;20"

    with (tmp_path / "guided_split_decisions.csv").open() as f:
        decision_rows = list(csv.DictReader(f))
    assert decision_rows[0]["accepted"] == "True"
    assert decision_rows[0]["output_child_ids"] == "1;21"


def test_longrange_guided_split_leaves_single_seed_parent_unchanged(tmp_path):
    primary = np.zeros((4, 8, 8), dtype=np.uint32)
    primary[:, 1:7, 1:7] = 1
    guide = np.zeros_like(primary)
    guide[:, 2:5, 2:5] = 10

    corrected = longrange_guided_split(
        primary,
        guide_seg=guide,
        min_parent_voxels=1,
        min_seed_voxels=1,
        min_seed_axis_extent=1,
        min_seed_overlap_voxels=1,
        min_seed_guide_fraction=0.0,
        min_seeds_in_parent=2,
        report_dir=str(tmp_path),
    )

    np.testing.assert_array_equal(corrected, primary)
    with (tmp_path / "guided_split_candidates.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["decision"] == "reject"
    assert rows[0]["reject_reason"] == "too_few_guide_seeds"


def test_longrange_guided_split_loads_guide_h5_and_defaults_report_dir(tmp_path):
    primary, guide = _merged_parent_and_two_guides()
    guide_path = tmp_path / "guide.h5"
    with h5py.File(guide_path, "w") as f:
        f.create_dataset("main", data=guide)

    corrected = longrange_guided_split(
        primary,
        guide_seg_path=str(guide_path),
        min_parent_voxels=1,
        min_seed_voxels=1,
        min_seed_axis_extent=1,
        min_seed_overlap_voxels=1,
        min_seed_guide_fraction=0.0,
        min_seeds_in_parent=2,
        bbox_pad=0,
        assignment="seeded_watershed",
    )

    assert set(np.unique(corrected)) == {0, 1, 21}
    assert (tmp_path / "guided_split_candidates.csv").exists()
    assert (tmp_path / "guided_split_decisions.csv").exists()
