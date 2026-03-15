"""Regression tests for multi-volume evaluation in test pipeline."""

from __future__ import annotations

import numpy as np
import torch

from connectomics.config import Config
from connectomics.inference.output import resolve_output_filenames
from connectomics.training.lightning.test_pipeline import run_test_step


class _DummyInferenceManager:
    def predict_with_tta(self, images, mask=None, mask_align_to_image=False):
        return torch.ones_like(images)

    def is_distributed_tta_sharding_enabled(self):
        return False

    def should_skip_postprocess_on_rank(self):
        return False


class _DummyModule:
    def __init__(self):
        self.device = torch.device("cpu")
        self.cfg = Config()
        self.inference_manager = _DummyInferenceManager()

    def _get_runtime_inference_config(self):
        return self.cfg.inference

    def _get_test_evaluation_config(self):
        return None

    def _resolve_test_output_config(self, _batch):
        return "test", "/tmp/results", "_x1_prediction.h5", ["train-input", "test-input_z29"]

    def _load_cached_predictions(self, _output_dir, _filenames, _cache_suffix, _mode):
        pred = np.zeros((2, 1, 4, 4, 4), dtype=np.uint16)
        return pred, True, "_x1_prediction.h5"

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


class _CroppingModule:
    def __init__(self):
        self.device = torch.device("cpu")
        self.cfg = Config()
        self.cfg.inference.postprocessing.crop_pad = [1, 2, 3]
        self.inference_manager = _DummyInferenceManager()

    def _get_runtime_inference_config(self):
        return self.cfg.inference

    def _get_test_evaluation_config(self):
        return None

    def _resolve_test_output_config(self, _batch):
        return "test", "/tmp/results", "_x1_prediction.h5", ["sample"]

    def _load_cached_predictions(self, _output_dir, _filenames, _cache_suffix, _mode):
        return None, False, ""

    def _summarize_tta_plan(self, _image_ndim):
        return "disabled"

    def _is_test_evaluation_enabled(self):
        return True


def test_run_test_step_crops_predictions_by_postprocessing_crop_pad(monkeypatch):
    module = _CroppingModule()
    batch = {
        "image": torch.zeros((1, 1, 6, 8, 10), dtype=torch.float32),
        "label": torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32),
    }

    captured = {}

    def _fake_process_decoding_postprocessing(
        _module,
        predictions_np,
        *,
        filenames,
        mode,
        batch_meta,
        save_final_predictions,
    ):
        captured["predictions_shape"] = tuple(predictions_np.shape)
        return predictions_np

    def _fake_evaluate_decoded_predictions(
        _module,
        decoded_predictions,
        labels,
        *,
        filenames,
        batch_idx,
    ):
        captured["decoded_shape"] = tuple(decoded_predictions.shape)
        captured["label_shape"] = tuple(labels.shape)

    monkeypatch.setattr(
        "connectomics.training.lightning.test_pipeline._process_decoding_postprocessing",
        _fake_process_decoding_postprocessing,
    )
    monkeypatch.setattr(
        "connectomics.training.lightning.test_pipeline._evaluate_decoded_predictions",
        _fake_evaluate_decoded_predictions,
    )

    out = run_test_step(module, batch, batch_idx=0)

    assert isinstance(out, torch.Tensor)
    assert captured["predictions_shape"] == (1, 1, 4, 4, 4)
    assert captured["decoded_shape"] == (1, 1, 4, 4, 4)
    assert captured["label_shape"] == (1, 1, 4, 4, 4)


class _AffinityCroppingModule:
    def __init__(self):
        self.device = torch.device("cpu")
        self.cfg = Config()
        self.cfg.data.label_transform.targets = [
            {
                "name": "affinity",
                "kwargs": {
                    "offsets": ["1-0-0", "0-1-0", "0-0-1"],
                    "deepem_crop": True,
                },
            }
        ]
        self.inference_manager = _DummyInferenceManager()

    def _get_runtime_inference_config(self):
        return self.cfg.inference

    def _get_test_evaluation_config(self):
        return None

    def _resolve_test_output_config(self, _batch):
        return "test", "/tmp/results", "_x1_prediction.h5", ["sample"]

    def _load_cached_predictions(self, _output_dir, _filenames, _cache_suffix, _mode):
        return None, False, ""

    def _summarize_tta_plan(self, _image_ndim):
        return "disabled"

    def _is_test_evaluation_enabled(self):
        return True


def test_run_test_step_crops_affinity_predictions_and_labels(monkeypatch):
    module = _AffinityCroppingModule()
    batch = {
        "image": torch.zeros((1, 3, 5, 5, 5), dtype=torch.float32),
        "label": torch.zeros((1, 3, 5, 5, 5), dtype=torch.float32),
    }

    captured = {}

    def _fake_process_decoding_postprocessing(
        _module,
        predictions_np,
        *,
        filenames,
        mode,
        batch_meta,
        save_final_predictions,
    ):
        captured["predictions_shape"] = tuple(predictions_np.shape)
        return predictions_np

    def _fake_evaluate_decoded_predictions(
        _module,
        decoded_predictions,
        labels,
        *,
        filenames,
        batch_idx,
    ):
        captured["decoded_shape"] = tuple(decoded_predictions.shape)
        captured["label_shape"] = tuple(labels.shape)

    monkeypatch.setattr(
        "connectomics.training.lightning.test_pipeline._process_decoding_postprocessing",
        _fake_process_decoding_postprocessing,
    )
    monkeypatch.setattr(
        "connectomics.training.lightning.test_pipeline._evaluate_decoded_predictions",
        _fake_evaluate_decoded_predictions,
    )

    out = run_test_step(module, batch, batch_idx=0)

    assert isinstance(out, torch.Tensor)
    assert captured["predictions_shape"] == (1, 3, 4, 4, 4)
    assert captured["decoded_shape"] == (1, 3, 4, 4, 4)
    assert captured["label_shape"] == (1, 3, 4, 4, 4)


class _AsymmetricPostprocessAffinityModule:
    def __init__(self):
        self.device = torch.device("cpu")
        self.cfg = Config()
        self.cfg.inference.postprocessing.crop_pad = [0, 1, 1, 2, 1, 2]
        self.cfg.data.label_transform.targets = [
            {
                "name": "affinity",
                "kwargs": {
                    "offsets": ["1-0-0", "0-1-0", "0-0-1"],
                    "deepem_crop": True,
                },
            }
        ]
        self.inference_manager = _DummyInferenceManager()

    def _get_runtime_inference_config(self):
        return self.cfg.inference

    def _get_test_evaluation_config(self):
        return None

    def _resolve_test_output_config(self, _batch):
        return "test", "/tmp/results", "_x1_prediction.h5", ["sample"]

    def _load_cached_predictions(self, _output_dir, _filenames, _cache_suffix, _mode):
        return None, False, ""

    def _summarize_tta_plan(self, _image_ndim):
        return "disabled"

    def _is_test_evaluation_enabled(self):
        return True


def test_run_test_step_combines_asymmetric_crop_pad_with_affinity_crop(monkeypatch):
    module = _AsymmetricPostprocessAffinityModule()
    batch = {
        "image": torch.zeros((1, 3, 6, 8, 8), dtype=torch.float32),
        "label": torch.zeros((1, 3, 4, 4, 4), dtype=torch.float32),
    }

    captured = {}

    def _fake_process_decoding_postprocessing(
        _module,
        predictions_np,
        *,
        filenames,
        mode,
        batch_meta,
        save_final_predictions,
    ):
        captured["predictions_shape"] = tuple(predictions_np.shape)
        return predictions_np

    def _fake_evaluate_decoded_predictions(
        _module,
        decoded_predictions,
        labels,
        *,
        filenames,
        batch_idx,
    ):
        captured["decoded_shape"] = tuple(decoded_predictions.shape)
        captured["label_shape"] = tuple(labels.shape)

    monkeypatch.setattr(
        "connectomics.training.lightning.test_pipeline._process_decoding_postprocessing",
        _fake_process_decoding_postprocessing,
    )
    monkeypatch.setattr(
        "connectomics.training.lightning.test_pipeline._evaluate_decoded_predictions",
        _fake_evaluate_decoded_predictions,
    )

    out = run_test_step(module, batch, batch_idx=0)

    assert isinstance(out, torch.Tensor)
    assert captured["predictions_shape"] == (1, 3, 4, 4, 4)
    assert captured["decoded_shape"] == (1, 3, 4, 4, 4)
    assert captured["label_shape"] == (1, 3, 4, 4, 4)


class _ListBatchModule:
    def __init__(self):
        self.device = torch.device("cpu")
        self.cfg = Config()
        self.inference_manager = _DummyInferenceManager()

    def _get_runtime_inference_config(self):
        return self.cfg.inference

    def _get_test_evaluation_config(self):
        return None

    def _resolve_test_output_config(self, batch):
        filenames = resolve_output_filenames(self.cfg, batch, global_step=0)
        return "test", "/tmp/results", "_x1_prediction.h5", filenames

    def _load_cached_predictions(self, _output_dir, _filenames, _cache_suffix, _mode):
        return None, False, ""

    def _summarize_tta_plan(self, _image_ndim):
        return "disabled"

    def _is_test_evaluation_enabled(self):
        return False


def test_run_test_step_handles_unstacked_list_batches(monkeypatch):
    module = _ListBatchModule()
    batch = {
        "image": [
            torch.zeros((1, 4, 4, 4), dtype=torch.float32),
            torch.zeros((1, 5, 6, 6), dtype=torch.float32),
        ],
        "label": [
            torch.zeros((1, 4, 4, 4), dtype=torch.float32),
            torch.zeros((1, 5, 6, 6), dtype=torch.float32),
        ],
        "image_meta_dict": [
            {"filename_or_obj": "/tmp/vol_a.h5"},
            {"filename_or_obj": "/tmp/vol_b.h5"},
        ],
    }

    seen = []

    def _fake_process_decoding_postprocessing(
        _module,
        predictions_np,
        *,
        filenames,
        mode,
        batch_meta,
        save_final_predictions,
    ):
        assert mode == "test"
        assert batch_meta and len(batch_meta) == 1
        seen.append(("decode", filenames[0], tuple(predictions_np.shape)))
        return predictions_np

    def _fake_evaluate_decoded_predictions(
        _module,
        decoded_predictions,
        labels,
        *,
        filenames,
        batch_idx,
    ):
        seen.append(
            ("eval", filenames[0], tuple(decoded_predictions.shape), tuple(labels.shape), batch_idx)
        )

    monkeypatch.setattr(
        "connectomics.training.lightning.test_pipeline._process_decoding_postprocessing",
        _fake_process_decoding_postprocessing,
    )
    monkeypatch.setattr(
        "connectomics.training.lightning.test_pipeline._evaluate_decoded_predictions",
        _fake_evaluate_decoded_predictions,
    )

    out = run_test_step(module, batch, batch_idx=3)

    assert isinstance(out, torch.Tensor)
    assert seen == [
        ("decode", "vol_a", (1, 1, 4, 4, 4)),
        ("eval", "vol_a", (1, 1, 4, 4, 4), (1, 1, 4, 4, 4), 6),
        ("decode", "vol_b", (1, 1, 5, 6, 6)),
        ("eval", "vol_b", (1, 1, 5, 6, 6), (1, 1, 5, 6, 6), 7),
    ]


class _MaskCheckingInferenceManager(_DummyInferenceManager):
    def predict_with_tta(self, images, mask=None, mask_align_to_image=False):
        assert torch.is_tensor(mask)
        assert mask.ndim in (images.ndim - 1, images.ndim)
        return torch.ones_like(images)


class _MaskListBatchModule(_ListBatchModule):
    def __init__(self):
        super().__init__()
        self.inference_manager = _MaskCheckingInferenceManager()


def test_run_test_step_coerces_singleton_list_masks(monkeypatch):
    module = _MaskListBatchModule()
    batch = {
        "image": [torch.zeros((1, 4, 4, 4), dtype=torch.float32)],
        "mask": [[torch.ones((1, 4, 4, 4), dtype=torch.float32)]],
        "image_meta_dict": [{"filename_or_obj": "/tmp/vol_mask.h5"}],
    }

    monkeypatch.setattr(
        "connectomics.training.lightning.test_pipeline._process_decoding_postprocessing",
        lambda _module, predictions_np, **kwargs: predictions_np,
    )
    monkeypatch.setattr(
        "connectomics.training.lightning.test_pipeline._evaluate_decoded_predictions",
        lambda *_args, **_kwargs: None,
    )

    out = run_test_step(module, batch, batch_idx=0)
    assert isinstance(out, torch.Tensor)
