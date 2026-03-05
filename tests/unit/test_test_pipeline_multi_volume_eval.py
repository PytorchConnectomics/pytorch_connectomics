"""Regression tests for multi-volume evaluation in test pipeline."""

from __future__ import annotations

import numpy as np
import torch

from connectomics.training.lightning.test_pipeline import run_test_step


class _DummyModule:
    def __init__(self):
        self.device = torch.device("cpu")

    def _resolve_test_output_config(self, _batch):
        return "test", "/tmp/results", "_prediction.h5", ["train-input", "test-input_z29"]

    def _load_cached_predictions(self, _output_dir, _filenames, _cache_suffix, _mode):
        pred = np.zeros((2, 1, 4, 4, 4), dtype=np.uint16)
        return pred, True, "_prediction.h5"

    def _is_test_evaluation_enabled(self):
        return True


def test_run_test_step_evaluates_each_volume_in_multi_volume_batch(monkeypatch):
    module = _DummyModule()
    batch = {
        "image": torch.zeros((2, 1, 4, 4, 4), dtype=torch.float32),
        "label": torch.zeros((2, 1, 4, 4, 4), dtype=torch.float32),
    }

    called_names = []

    def _fake_compute_test_metrics(_module, _decoded_predictions, _labels, volume_name=None):
        called_names.append(volume_name)

    monkeypatch.setattr(
        "connectomics.training.lightning.test_pipeline.compute_test_metrics",
        _fake_compute_test_metrics,
    )

    out = run_test_step(module, batch, batch_idx=0)
    assert isinstance(out, torch.Tensor)
    assert called_names == ["train-input", "test-input_z29"]
