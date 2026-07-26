from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from connectomics.config import Config
from connectomics.evaluation.context import EvaluationContext
from connectomics.evaluation.nerl import compute_nerl_metrics
from connectomics.evaluation.report import compute_test_metrics, save_metrics_to_file
from connectomics.metrics.oracle import oracle_merge_segmentation


def test_oracle_merge_relabels_each_fragment_to_majority_gt():
    prediction = np.array(
        [
            [[1, 1, 1, 2], [1, 2, 2, 2]],
            [[3, 3, 0, 4], [3, 0, 4, 4]],
        ],
        dtype=np.uint32,
    )
    ground_truth = np.array(
        [
            [[10, 10, 20, 20], [10, 20, 20, 20]],
            [[30, 30, 0, 0], [30, 0, 0, 0]],
        ],
        dtype=np.uint32,
    )

    result = oracle_merge_segmentation(prediction, ground_truth)

    assert np.all(result[prediction == 1] == 10)
    assert np.all(result[prediction == 2] == 20)
    assert np.all(result[prediction == 3] == 30)
    assert np.all(result[prediction == 4] > ground_truth.max())
    assert np.all(result[prediction == 0] == 0)


def test_oracle_merge_rejects_shape_mismatch():
    prediction = np.zeros((2, 2, 2), dtype=np.uint32)
    ground_truth = np.zeros((1, 2, 2), dtype=np.uint32)

    try:
        oracle_merge_segmentation(prediction, ground_truth)
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("shape mismatch must raise")


def test_nerl_reporting_computes_requested_oracle_merge(monkeypatch):
    test_cfg = SimpleNamespace(
        skeleton="graph.npz",
        skeleton_mask=None,
        resolution=[25, 9, 9],
    )
    evaluation_cfg = SimpleNamespace(
        metrics=["nerl", "nerl_oracle_merge"],
        nerl_merge_threshold=10,
        nerl_chunk_num=1,
        nerl_num_workers=1,
    )
    context = EvaluationContext(
        cfg=SimpleNamespace(data=SimpleNamespace(test=test_cfg)),
        evaluation_cfg=evaluation_cfg,
    )
    prediction = np.array([[[1, 1, 2, 2]]], dtype=np.uint32)
    labels = np.array([[[[10, 10, 10, 10]]]], dtype=np.uint32)
    calls = []

    def fake_score(segmentation, graph_value, **kwargs):
        calls.append((np.asarray(segmentation).copy(), graph_value, kwargs))
        return SimpleNamespace(
            nerl=0.5 if len(calls) == 1 else 0.9,
            pred_erl=5.0 if len(calls) == 1 else 9.0,
            gt_erl=10.0,
            num_skeletons=1,
            graph=SimpleNamespace(skeleton_id=np.array([10], dtype=np.uint64)),
            per_gt_erl=np.array([[5.0, 10.0]], dtype=np.float64),
        )

    monkeypatch.setattr(
        "connectomics.evaluation.nerl.compute_nerl_score_details",
        fake_score,
    )
    metrics = {}

    compute_nerl_metrics(
        context,
        prediction,
        "",
        metrics,
        "sample",
        dense_labels=labels,
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], prediction)
    np.testing.assert_array_equal(
        calls[1][0],
        np.full(prediction.shape, 10, dtype=np.uint32),
    )
    assert calls[0][1] == calls[1][1] == "graph.npz"
    assert calls[0][2]["merge_threshold"] == calls[1][2]["merge_threshold"] == 10
    assert metrics["nerl"] == 0.5
    assert metrics["nerl_oracle_merge"] == 0.9


def test_nerl_reporting_oracle_only_skips_base_score(monkeypatch):
    test_cfg = SimpleNamespace(
        skeleton="graph.npz",
        skeleton_mask=None,
        resolution=[25, 9, 9],
    )
    evaluation_cfg = SimpleNamespace(
        metrics=["nerl_oracle_merge"],
        nerl_merge_threshold=10,
        nerl_chunk_num=1,
        nerl_num_workers=1,
    )
    context = EvaluationContext(
        cfg=SimpleNamespace(data=SimpleNamespace(test=test_cfg)),
        evaluation_cfg=evaluation_cfg,
    )
    prediction = np.array([[[1, 1, 2, 2]]], dtype=np.uint32)
    labels = np.array([[[[10, 10, 10, 10]]]], dtype=np.uint32)
    calls = []

    def fake_score(segmentation, graph_value, **kwargs):
        calls.append((np.asarray(segmentation).copy(), graph_value, kwargs))
        return SimpleNamespace(
            nerl=0.9,
            pred_erl=9.0,
            gt_erl=10.0,
            num_skeletons=1,
        )

    monkeypatch.setattr(
        "connectomics.evaluation.nerl.compute_nerl_score_details",
        fake_score,
    )
    metrics = {}

    compute_nerl_metrics(
        context,
        prediction,
        "",
        metrics,
        "sample",
        dense_labels=labels,
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(
        calls[0][0],
        np.full(prediction.shape, 10, dtype=np.uint32),
    )
    assert "nerl" not in metrics
    assert metrics["nerl_oracle_merge"] == 0.9


def test_evaluation_output_follows_decoded_artifact_path():
    cfg = SimpleNamespace(
        decoding=SimpleNamespace(save_path="outputs/decoded"),
    )
    context = EvaluationContext(
        cfg=cfg,
        inference_cfg=SimpleNamespace(save_path="outputs/inference"),
    )

    assert context.resolved_output_path() == "outputs/decoded"


def test_oracle_merge_can_be_requested_without_base_nerl(monkeypatch):
    cfg = Config()
    cfg.evaluation.enabled = True
    cfg.evaluation.metrics = ["nerl_oracle_merge"]
    captured = {}
    context = EvaluationContext(
        cfg=cfg,
        evaluation_cfg=cfg.evaluation,
        inference_cfg=cfg.inference,
        enabled=True,
        metrics_sink=lambda metrics: captured.update(metrics),
    )

    def fake_compute_nerl(
        _context,
        _predictions,
        _prefix,
        metrics_dict,
        _volume,
        **_kwargs,
    ):
        metrics_dict["nerl_oracle_merge"] = 0.9

    monkeypatch.setattr(
        "connectomics.evaluation.report.compute_nerl_metrics",
        fake_compute_nerl,
    )

    compute_test_metrics(
        context,
        np.zeros((2, 2), dtype=np.uint32),
        torch.zeros((2, 2), dtype=torch.uint32),
        volume_name="sample",
    )

    assert captured["nerl_oracle_merge"] == 0.9
    assert "nerl" not in captured


def test_oracle_only_score_is_written_to_metrics_report(tmp_path):
    cfg = Config()
    cfg.inference.save_path = str(tmp_path)
    context = EvaluationContext(
        cfg=cfg,
        evaluation_cfg=cfg.evaluation,
        inference_cfg=cfg.inference,
    )

    save_metrics_to_file(
        context,
        {
            "volume_name": "sample",
            "nerl_oracle_merge": 0.9,
            "nerl_oracle_merge_pred_erl": 9.0,
        },
    )

    report = (tmp_path / "sample" / "eval_prediction_x1.txt").read_text()
    assert "NERL Oracle-Merge:            0.900000" in report
    assert "Oracle-Merge Pred ERL:        9.000000" in report
