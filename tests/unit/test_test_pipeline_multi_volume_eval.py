"""Regression tests for multi-volume evaluation in test pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from connectomics.config import Config
from connectomics.evaluation import compute_test_metrics
from connectomics.evaluation.nerl import import_em_erl
from connectomics.inference.output import resolve_output_filenames
from connectomics.training.lightning.test_pipeline import (
    _apply_predecode_prediction_crops,
    run_test_step,
)


class _DummyInferenceManager:
    def predict_with_tta(
        self,
        images,
        mask=None,
        mask_align_to_image=False,
        requested_head=None,
    ):
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
        "connectomics.evaluation.stage.compute_test_metrics",
        _fake_compute_test_metrics,
    )

    out = run_test_step(module, batch, batch_idx=0)
    assert isinstance(out, torch.Tensor)
    assert called_names == ["train-input", "test-input_z29"]


class _NerlModule:
    def __init__(self, graph_path):
        self.device = torch.device("cpu")
        self.cfg = Config()
        self.cfg.evaluation.enabled = True
        self.cfg.evaluation.metrics = ["nerl"]
        self.cfg.evaluation.nerl_graph = str(graph_path)
        self.inference_manager = _DummyInferenceManager()
        self.saved_metrics = None

    def _get_runtime_inference_config(self):
        return self.cfg.inference

    def _get_test_evaluation_config(self):
        return self.cfg.evaluation

    def _is_test_evaluation_enabled(self):
        return True

    def _cfg_value(self, cfg, key, default=None):
        return getattr(cfg, key, default)

    def _cfg_float(self, cfg, key, default):
        return float(getattr(cfg, key, default))

    def _save_metrics_to_file(self, metrics_dict):
        self.saved_metrics = dict(metrics_dict)

    def log(self, *args, **kwargs):
        return None


def test_compute_test_metrics_supports_nerl_without_dense_labels(tmp_path):
    ERLGraph, _, _ = import_em_erl()
    graph_path = tmp_path / "gt_graph.npz"
    graph = ERLGraph(
        skeleton_id=np.array([10, 20], dtype=np.uint64),
        skeleton_len=np.array([1.0, 1.0], dtype=np.float64),
        node_skeleton_index=np.array([0, 0, 1, 1], dtype=np.uint32),
        node_coords_zyx=np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
            ],
            dtype=np.float32,
        ),
        edge_u=np.array([0, 2], dtype=np.uint32),
        edge_v=np.array([1, 3], dtype=np.uint32),
        edge_len=np.array([1.0, 1.0], dtype=np.float32),
        edge_ptr=np.array([0, 1, 2], dtype=np.uint64),
    )
    graph.save_npz(graph_path)

    module = _NerlModule(graph_path)
    decoded = np.zeros((1, 2, 2), dtype=np.uint32)
    decoded[0, 0, :] = 1
    decoded[0, 1, :] = 2

    compute_test_metrics(module, decoded, labels=None, volume_name="sample")

    assert module.saved_metrics is not None
    assert module.saved_metrics["volume_name"] == "sample"
    assert module.saved_metrics["nerl"] == 1.0
    assert module.saved_metrics["nerl_erl"] == 1.0
    assert module.saved_metrics["nerl_max_erl"] == 1.0
    assert module.saved_metrics["nerl_pred_erl"] == 1.0
    assert module.saved_metrics["nerl_gt_erl"] == 1.0
    np.testing.assert_allclose(
        module.saved_metrics["nerl_per_gt_erl"],
        np.array([[1.0, 1.0], [1.0, 1.0]]),
    )
    np.testing.assert_array_equal(
        module.saved_metrics["nerl_gt_segment_ids"],
        np.array([10, 20], dtype=np.uint64),
    )


def test_compute_test_metrics_supports_banis_skeleton_pickle(tmp_path):
    nx = pytest.importorskip("networkx")
    import pickle

    skeleton_path = tmp_path / "skeleton.pkl"
    skeleton = nx.Graph()
    skeleton.add_node(0, id=10, index_position=(0, 0, 0))
    skeleton.add_node(1, id=10, index_position=(0, 0, 1))
    skeleton.add_node(2, id=20, index_position=(0, 1, 0))
    skeleton.add_node(3, id=20, index_position=(0, 1, 1))
    skeleton.add_edge(0, 1, edge_length=1.0)
    skeleton.add_edge(2, 3, edge_length=1.0)
    with open(skeleton_path, "wb") as f:
        pickle.dump(skeleton, f)

    module = _NerlModule(skeleton_path)
    decoded = np.zeros((1, 2, 2), dtype=np.uint32)
    decoded[0, 0, :] = 1
    decoded[0, 1, :] = 2

    compute_test_metrics(module, decoded, labels=None, volume_name="sample")

    assert module.saved_metrics is not None
    assert module.saved_metrics["nerl"] == 1.0
    np.testing.assert_allclose(
        module.saved_metrics["nerl_per_gt_erl"],
        np.array([[1.0, 1.0], [1.0, 1.0]]),
    )


class _CroppingModule:
    def __init__(self):
        self.device = torch.device("cpu")
        self.cfg = Config()
        self.cfg.inference.crop_pad = [1, 2, 3]
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


def test_predecode_crop_compacts_prediction_storage():
    module = _CroppingModule()
    predictions = np.arange(1 * 1 * 6 * 8 * 10, dtype=np.float32).reshape(1, 1, 6, 8, 10)

    cropped, reference_shape = _apply_predecode_prediction_crops(
        module,
        predictions,
        reference_image_shape=(1, 1, 6, 8, 10),
        item_name="predictions",
    )

    assert tuple(cropped.shape) == (1, 1, 4, 4, 4)
    assert reference_shape == (4, 4, 4)
    assert not np.shares_memory(cropped, predictions)
    assert cropped.flags.c_contiguous


class _AffinityCroppingModule:
    def __init__(self):
        self.device = torch.device("cpu")
        self.cfg = Config()
        self.cfg.data.label_transform.targets = [
            {
                "name": "affinity",
                "kwargs": {
                    "offsets": ["1-0-0", "0-1-0", "0-0-1"],
                    "affinity_mode": "deepem",
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


class _SelectedBanisAffinityCroppingModule(_AffinityCroppingModule):
    def __init__(self):
        super().__init__()
        self.cfg.inference.select_channel = [2, 1, 0]
        self.cfg.data.label_transform.targets = [
            {
                "name": "affinity",
                "kwargs": {
                    "offsets": [
                        "0-0-1",
                        "0-1-0",
                        "1-0-0",
                        "0-0-10",
                        "0-10-0",
                        "10-0-0",
                    ],
                    "affinity_mode": "banis",
                },
            }
        ]


def test_run_test_step_affinity_crop_uses_selected_channels(monkeypatch):
    module = _SelectedBanisAffinityCroppingModule()
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
    assert captured["label_shape"] == (1, 3, 4, 4, 4)


class _AsymmetricPostprocessAffinityModule:
    def __init__(self):
        self.device = torch.device("cpu")
        self.cfg = Config()
        self.cfg.inference.crop_pad = [0, 1, 1, 2, 1, 2]
        self.cfg.data.label_transform.targets = [
            {
                "name": "affinity",
                "kwargs": {
                    "offsets": ["1-0-0", "0-1-0", "0-0-1"],
                    "affinity_mode": "deepem",
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


class _SavedPredictionPathCroppingModule:
    def __init__(self):
        self.device = torch.device("cpu")
        self.cfg = Config()
        self.cfg.inference.crop_pad = [1, 2, 3]
        self.cfg.inference.saved_prediction_path = "/tmp/raw_affinity_prediction.h5"
        self.inference_manager = _DummyInferenceManager()

    def _get_runtime_inference_config(self):
        return self.cfg.inference

    def _get_test_evaluation_config(self):
        return None

    def _resolve_test_output_config(self, _batch):
        return "test", "/tmp/results", "_x1_prediction.h5", ["sample"]

    def _load_cached_predictions(self, _output_dir, _filenames, _cache_suffix, _mode):
        pred = np.zeros((1, 1, 6, 8, 10), dtype=np.uint16)
        return pred, True, "_prediction.h5"

    def _is_test_evaluation_enabled(self):
        return True


def test_run_test_step_crops_saved_prediction_path_before_decode(monkeypatch):
    module = _SavedPredictionPathCroppingModule()
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
    def predict_with_tta(
        self,
        images,
        mask=None,
        mask_align_to_image=False,
        requested_head=None,
    ):
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


class _SaveAllHeadsInferenceManager:
    def __init__(self):
        self.requested_heads = []

    def predict_with_tta(
        self,
        images,
        mask=None,
        mask_align_to_image=False,
        requested_head=None,
    ):
        self.requested_heads.append(requested_head)
        if requested_head == "sdt":
            return torch.full((images.shape[0], 1, *images.shape[2:]), 7.0, dtype=images.dtype)
        return torch.full((images.shape[0], 2, *images.shape[2:]), 3.0, dtype=images.dtype)

    def is_distributed_tta_sharding_enabled(self):
        return False

    def should_skip_postprocess_on_rank(self):
        return False


class _SaveAllHeadsModule:
    def __init__(self):
        self.device = torch.device("cpu")
        self.cfg = Config()
        self.cfg.model.out_channels = 3
        self.cfg.model.primary_head = "affinity"
        self.cfg.model.heads = {
            "affinity": {"out_channels": 2, "num_blocks": 0},
            "sdt": {"out_channels": 1, "num_blocks": 0},
        }
        self.cfg.inference.head = "affinity"
        self.cfg.inference.save_prediction.enabled = True
        self.cfg.inference.save_prediction.save_all_heads = True
        self.inference_manager = _SaveAllHeadsInferenceManager()

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
        return False

    def _get_prediction_checkpoint_path(self):
        return None


def test_run_test_step_saves_all_named_output_heads(monkeypatch):
    module = _SaveAllHeadsModule()
    batch = {
        "image": torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32),
    }

    saved = []

    def _fake_write_outputs(_cfg, predictions_np, filenames, *, suffix, mode, batch_meta):
        saved.append((tuple(predictions_np.shape), tuple(filenames), suffix, mode))

    monkeypatch.setattr(
        "connectomics.training.lightning.test_pipeline.write_outputs",
        _fake_write_outputs,
    )
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
    assert module.inference_manager.requested_heads == ["affinity", "sdt"]
    assert saved == [
        ((1, 2, 4, 4, 4), ("sample",), "tta_x1_head-affinity_prediction", "test"),
        ((1, 1, 4, 4, 4), ("sample",), "tta_x1_head-sdt_prediction", "test"),
    ]
