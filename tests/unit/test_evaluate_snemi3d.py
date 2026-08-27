from __future__ import annotations

import csv

import h5py
import numpy as np

from scripts.evaluate_snemi3d import (
    evaluate_candidate,
    full_volume_crop,
    snemi3d_gc_crop,
    write_results,
)


def _write_h5(path, array):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("main", data=array)


def test_snemi3d_gc_crop_matches_the_published_container():
    bounds = snemi3d_gc_crop((100, 1024, 1024))

    assert (bounds.z0, bounds.z1) == (25, 75)
    assert (bounds.y0, bounds.y1) == (97, 928)
    assert (bounds.x0, bounds.x1) == (97, 928)
    assert bounds.shape == (50, 831, 831)
    assert np.prod(bounds.shape) == 34_528_050


def test_evaluate_candidate_scores_only_compatible_integer_segmentations(tmp_path):
    gt = np.zeros((4, 5, 6), dtype=np.uint16)
    gt[:, :3] = 1
    gt[:, 3:] = 2
    gt_path = tmp_path / "gt.h5"
    pred_path = tmp_path / "pred.h5"
    float_path = tmp_path / "affinity.h5"
    _write_h5(gt_path, gt)
    _write_h5(pred_path, gt.astype(np.uint32))
    _write_h5(float_path, gt.astype(np.float32))

    bounds = full_volume_crop(gt.shape)
    perfect = evaluate_candidate(
        pred_path,
        dataset="main",
        expected_shape=gt.shape,
        bounds=bounds,
        ground_truth_crop=gt,
    )
    skipped = evaluate_candidate(
        float_path,
        dataset="main",
        expected_shape=gt.shape,
        bounds=bounds,
        ground_truth_crop=gt,
    )

    assert perfect["status"] == "ok"
    assert float(perfect["adapted_rand_error"]) == 0.0
    assert float(perfect["precision"]) == 1.0
    assert float(perfect["recall"]) == 1.0
    assert perfect["foreground_instances"] == 2
    assert skipped["status"] == "skipped"
    assert "not an integer segmentation" in skipped["reason"]


def test_write_results_orders_scores_before_skips(tmp_path):
    output = tmp_path / "scores.tsv"
    rows = [
        {"status": "skipped", "adapted_rand_error": "", "path": "skip"},
        {"status": "ok", "adapted_rand_error": "0.2", "path": "worse"},
        {"status": "ok", "adapted_rand_error": "0.1", "path": "best"},
    ]
    complete_rows = []
    for row in rows:
        complete = {
            "precision": "",
            "recall": "",
            "foreground_instances": "",
            "foreground_fraction": "",
            "dataset": "main",
            "shape": "",
            "dtype": "",
            "crop": "",
            "crop_shape": "",
            "crop_voxels": "",
            "reason": "",
            **row,
        }
        complete_rows.append(complete)

    write_results(output, complete_rows)

    with output.open(newline="") as handle:
        saved = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["path"] for row in saved] == ["best", "worse", "skip"]
