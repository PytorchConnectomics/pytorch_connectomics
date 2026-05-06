from __future__ import annotations

import numpy as np
import pytest
import torch

from connectomics.config import Config
from connectomics.evaluation import EvaluationContext, compute_test_metrics
from connectomics.evaluation.metrics import compute_binary_metrics, is_instance_segmentation


class _NoMetricContext:
    def metric(self, _name):
        return None

    def metric_requested(self, _name):
        return False


def test_uint16_predictions_are_instance_segmentations():
    pred = torch.tensor([[0, 1], [2, 3]], dtype=torch.uint16)

    assert is_instance_segmentation(pred) is True


def test_binary_metrics_accept_uint16_tensors_for_dtype_safe_reductions():
    metrics = {}

    compute_binary_metrics(
        _NoMetricContext(),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.uint16),
        torch.tensor([[0, 1], [0, 1]], dtype=torch.uint16),
        volume_prefix="",
        metrics_dict=metrics,
        prediction_threshold=0.5,
    )

    assert metrics == {}


def test_nerl_only_evaluation_skips_unrequested_label_metric_groups(monkeypatch):
    cfg = Config()
    cfg.evaluation.enabled = True
    cfg.evaluation.metrics = ["nerl"]
    captured = {}
    context = EvaluationContext(
        cfg=cfg,
        evaluation_cfg=cfg.evaluation,
        inference_cfg=cfg.inference,
        enabled=True,
        metrics_sink=lambda metrics: captured.update(metrics),
    )

    def _fake_compute_nerl_metrics(_context, _predictions, _prefix, metrics_dict, _volume):
        metrics_dict["nerl"] = 1.0

    monkeypatch.setattr(
        "connectomics.evaluation.report.compute_nerl_metrics",
        _fake_compute_nerl_metrics,
    )
    monkeypatch.setattr(
        "connectomics.evaluation.report.compute_binary_metrics",
        lambda *_args, **_kwargs: pytest.fail("binary metrics should be skipped"),
    )
    monkeypatch.setattr(
        "connectomics.evaluation.report.compute_instance_metrics",
        lambda *_args, **_kwargs: pytest.fail("instance metrics should be skipped"),
    )

    compute_test_metrics(
        context,
        np.zeros((2, 2), dtype=np.uint16),
        torch.zeros((2, 2), dtype=torch.uint16),
        volume_name="vol0",
    )

    assert captured["nerl"] == 1.0
    assert captured["volume_name"] == "vol0"


def test_requested_adapted_rand_runs_instance_metrics_for_float_cached_segmentation(
    monkeypatch,
):
    cfg = Config()
    cfg.evaluation.enabled = True
    cfg.evaluation.metrics = ["adapted_rand"]
    captured = {}
    context = EvaluationContext(
        cfg=cfg,
        evaluation_cfg=cfg.evaluation,
        inference_cfg=cfg.inference,
        enabled=True,
        metrics_sink=lambda metrics: captured.update(metrics),
    )

    def _fake_compute_instance_metrics(
        _context,
        pred_tensor,
        labels_tensor,
        _volume_prefix,
        metrics_dict,
        _instance_iou_threshold,
    ):
        metrics_dict["adapted_rand_error"] = 0.0
        metrics_dict["pred_dtype"] = str(pred_tensor.dtype)
        metrics_dict["label_dtype"] = str(labels_tensor.dtype)

    monkeypatch.setattr(
        "connectomics.evaluation.report.compute_instance_metrics",
        _fake_compute_instance_metrics,
    )
    monkeypatch.setattr(
        "connectomics.evaluation.report.compute_binary_metrics",
        lambda *_args, **_kwargs: pytest.fail("binary metrics should be skipped"),
    )

    compute_test_metrics(
        context,
        np.zeros((2, 2), dtype=np.float32),
        torch.zeros((2, 2), dtype=torch.uint16),
        volume_name="vol0",
    )

    assert captured["adapted_rand_error"] == 0.0
    assert captured["pred_dtype"] == "torch.float32"
    assert captured["label_dtype"] == "torch.uint16"
