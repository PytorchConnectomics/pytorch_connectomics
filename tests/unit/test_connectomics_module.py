from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from connectomics.config import Config
from connectomics.config.schema.stages import TestConfig as HydraTestConfig
from connectomics.training.lightning import ConnectomicsModule
from connectomics.training.lightning.test_pipeline import log_test_epoch_metrics


def _stub_logging(module: ConnectomicsModule, sink: Optional[List[str]] = None) -> None:
    """Replace Lightning logging with lightweight callables for unit tests."""

    def log_dict_override(*args, **kwargs):
        return None

    def log_override(name, *args, **kwargs):
        if sink is not None:
            sink.append(name)
        return None

    module.log_dict = log_dict_override
    module.log = log_override


class SimpleModel(nn.Module):
    """Tiny 3D conv model with optional deep supervision branch."""

    def __init__(self, deep_supervision: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.conv = nn.Conv3d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        if self.deep_supervision:
            return {"output": output, "ds_1": F.avg_pool3d(output, kernel_size=2)}
        return output


class SimpleMultiHeadModel(nn.Module):
    """Tiny model that returns named task heads at the main output scale."""

    def __init__(self):
        super().__init__()
        self.affinity = nn.Conv3d(1, 2, kernel_size=3, padding=1)
        self.sdt = nn.Conv3d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        return {
            "output": {
                "affinity": self.affinity(x),
                "sdt": self.sdt(x),
            }
        }


def _base_config() -> Config:
    cfg = Config()
    cfg.model.loss.losses = [
        {"function": "DiceLoss", "weight": 1.0, "pred_slice": "0:1", "target_slice": "0:1"}
    ]
    cfg.model.out_channels = 1
    return cfg


def test_training_step_single_output_returns_loss():
    """Training step should run with a plain tensor output."""
    cfg = _base_config()
    module = ConnectomicsModule(cfg, model=SimpleModel())
    _stub_logging(module)

    batch = {
        "image": torch.rand(2, 1, 8, 8, 8),
        "label": torch.rand(2, 1, 8, 8, 8),
    }

    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_training_step_uses_deep_supervision_branch():
    """When the model returns auxiliary outputs, the deep supervision path is used."""
    cfg = _base_config()
    module = ConnectomicsModule(cfg, model=SimpleModel(deep_supervision=True))
    _stub_logging(module)

    branch_called = {"used": False}

    def fake_deep_supervision(outputs, labels, stage="train", mask=None):
        branch_called["used"] = True
        return torch.tensor(0.0, requires_grad=True), {"train_loss_total": 0.0}

    module.loss_orchestrator.compute_deep_supervision_loss = fake_deep_supervision

    batch = {
        "image": torch.rand(1, 1, 6, 6, 6),
        "label": torch.rand(1, 1, 6, 6, 6),
    }

    loss = module.training_step(batch, 0)

    assert branch_called["used"]
    assert torch.isfinite(loss)


def test_training_step_accepts_named_multi_head_outputs():
    """Training should route named-head outputs through the standard loss path."""
    cfg = Config()
    cfg.model.out_channels = 3
    cfg.model.primary_head = "affinity"
    cfg.model.heads = {
        "affinity": {"out_channels": 2, "num_blocks": 0},
        "sdt": {"out_channels": 1, "num_blocks": 0},
    }
    cfg.model.loss.losses = [
        {
            "function": "DiceLoss",
            "weight": 1.0,
            "pred_head": "affinity",
            "target_slice": "0:2",
        },
        {
            "function": "WeightedMSELoss",
            "weight": 1.0,
            "pred_head": "sdt",
            "target_slice": "2:3",
        },
    ]
    module = ConnectomicsModule(cfg, model=SimpleMultiHeadModel())
    _stub_logging(module)

    batch = {
        "image": torch.rand(2, 1, 8, 8, 8),
        "label": torch.rand(2, 3, 8, 8, 8),
    }

    loss = module.training_step(batch, 0)

    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_validation_step_logs_metrics_for_named_head_output():
    """Validation metrics should select the configured named head and target slice."""
    cfg = Config()
    cfg.model.out_channels = 3
    cfg.model.primary_head = "affinity"
    cfg.model.heads = {
        "affinity": {"out_channels": 2, "num_blocks": 0},
        "sdt": {"out_channels": 1, "num_blocks": 0},
    }
    cfg.inference.head = "sdt"
    cfg.inference.evaluation.enabled = True
    cfg.inference.evaluation.metrics = ["accuracy"]
    cfg.model.loss.losses = [
        {
            "function": "DiceLoss",
            "weight": 1.0,
            "pred_head": "affinity",
            "target_slice": "0:2",
        },
        {
            "function": "WeightedMSELoss",
            "weight": 1.0,
            "pred_head": "sdt",
            "target_slice": "2:3",
        },
    ]
    module = ConnectomicsModule(cfg, model=SimpleMultiHeadModel())
    logged_names: list[str] = []
    _stub_logging(module, sink=logged_names)
    module.on_validation_start()

    batch = {
        "image": torch.rand(1, 1, 6, 6, 6),
        "label": torch.cat(
            [
                torch.rand(1, 2, 6, 6, 6),
                torch.randint(0, 2, (1, 1, 6, 6, 6)).float(),
            ],
            dim=1,
        ),
    }

    loss = module.validation_step(batch, 0)

    assert torch.isfinite(loss)
    assert "val_accuracy" in logged_names


def test_validation_step_uses_head_target_slice_mapping_without_loss_target_slice():
    """Validation should honor model.heads.*.target_slice for the selected head."""
    cfg = Config()
    cfg.model.out_channels = 3
    cfg.model.primary_head = "affinity"
    cfg.model.heads = {
        "affinity": {"out_channels": 2, "num_blocks": 0, "target_slice": "0:2"},
        "sdt": {"out_channels": 1, "num_blocks": 0, "target_slice": "2:3"},
    }
    cfg.inference.head = "sdt"
    cfg.inference.evaluation.enabled = True
    cfg.inference.evaluation.metrics = ["accuracy"]
    cfg.model.loss.losses = [
        {
            "function": "DiceLoss",
            "weight": 1.0,
            "pred_head": "affinity",
        },
        {
            "function": "WeightedMSELoss",
            "weight": 1.0,
            "pred_head": "sdt",
        },
    ]
    module = ConnectomicsModule(cfg, model=SimpleMultiHeadModel())
    logged_names: list[str] = []
    _stub_logging(module, sink=logged_names)
    module.on_validation_start()

    batch = {
        "image": torch.rand(1, 1, 6, 6, 6),
        "label": torch.cat(
            [
                torch.rand(1, 2, 6, 6, 6),
                torch.randint(0, 2, (1, 1, 6, 6, 6)).float(),
            ],
            dim=1,
        ),
    }

    loss = module.validation_step(batch, 0)

    assert torch.isfinite(loss)
    assert "val_accuracy" in logged_names


def test_validation_step_logs_metrics_when_enabled():
    """Validation step should compute metrics when enabled in the config."""
    cfg = _base_config()
    cfg.inference.evaluation.enabled = True
    cfg.inference.evaluation.metrics = ["accuracy"]

    module = ConnectomicsModule(cfg, model=SimpleModel())
    logged_names: list[str] = []
    _stub_logging(module, sink=logged_names)

    batch = {
        "image": torch.rand(1, 1, 6, 6, 6),
        "label": torch.randint(0, 2, (1, 1, 6, 6, 6)).float(),
    }

    loss = module.validation_step(batch, 0)

    assert torch.isfinite(loss)
    assert any(name.startswith("val_") for name in logged_names)


def test_configure_optimizers_includes_uncertainty_weighter_parameters():
    """Adaptive loss-weighter parameters should be included in optimization."""
    cfg = Config()
    cfg.model.out_channels = 1
    cfg.model.loss.losses = [
        {"function": "DiceLoss", "weight": 1.0, "pred_slice": "0:1", "target_slice": "0:1"},
        {
            "function": "BCEWithLogitsLoss",
            "weight": 1.0,
            "pred_slice": "0:1",
            "target_slice": "0:1",
        },
    ]
    cfg.model.loss.loss_balancing.strategy = "uncertainty"

    module = ConnectomicsModule(cfg, model=SimpleModel())
    opt_config = module.configure_optimizers()
    optimizer = opt_config["optimizer"]

    optimized_param_ids = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }

    assert module.loss_weighter is not None
    assert id(module.loss_weighter.log_vars) in optimized_param_ids


def test_uncertainty_weighter_log_vars_update_after_optimizer_step():
    """Uncertainty log-variance parameters should receive gradients and update."""
    torch.manual_seed(0)

    cfg = Config()
    cfg.model.out_channels = 1
    cfg.optimization.optimizer.lr = 1e-2
    cfg.model.loss.losses = [
        {
            "function": "BCEWithLogitsLoss",
            "weight": 1.0,
            "pred_slice": "0:1",
            "target_slice": "0:1",
        },
        {"function": "MSELoss", "weight": 1.0, "pred_slice": "0:1", "target_slice": "0:1"},
    ]
    cfg.model.loss.loss_balancing.strategy = "uncertainty"

    module = ConnectomicsModule(cfg, model=SimpleModel())
    _stub_logging(module)

    optimizer = module.configure_optimizers()["optimizer"]
    batch = {
        "image": torch.ones(2, 1, 6, 6, 6),
        "label": torch.ones(2, 1, 6, 6, 6),
    }

    before = module.loss_weighter.log_vars.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = module.training_step(batch, 0)
    loss.backward()

    grad = module.loss_weighter.log_vars.grad
    assert grad is not None
    assert torch.any(torch.abs(grad) > 0)

    optimizer.step()
    after = module.loss_weighter.log_vars.detach().clone()

    assert not torch.allclose(after, before)


def test_load_state_dict_ignores_stale_loss_function_keys():
    """Strict loading should ignore obsolete loss-function-only checkpoint keys."""
    cfg = _base_config()
    module = ConnectomicsModule(cfg, model=SimpleModel())

    state = dict(module.state_dict())
    state["loss_functions.0.pos_weight"] = torch.tensor([10.0])

    result = module.load_state_dict(state, strict=True)

    assert "loss_functions.0.pos_weight" not in result.unexpected_keys


def test_runtime_inference_config_uses_merged_root_config():
    """Runtime inference config should come from cfg.inference only."""
    cfg = _base_config()
    cfg.test = HydraTestConfig()
    cfg.inference.save_prediction.enabled = True
    cfg.test.inference.save_prediction.enabled = False

    module = ConnectomicsModule(cfg, model=SimpleModel())

    runtime_cfg = module._get_runtime_inference_config()
    assert runtime_cfg.save_prediction.enabled is True


def test_resolve_test_output_config_uses_runtime_inference_output_path(monkeypatch):
    """Output path should come from cfg.inference.save_prediction."""
    cfg = _base_config()
    cfg.inference.save_prediction.output_path = "/tmp/test_results"
    cfg.inference.save_prediction.cache_suffix = "_x1_prediction.h5"
    module = ConnectomicsModule(cfg, model=SimpleModel())

    monkeypatch.setattr(
        "connectomics.training.lightning.model.resolve_output_filenames",
        lambda _cfg, _batch, global_step=0: ["sample_a"],
    )

    mode, output_dir, cache_suffix, filenames = ConnectomicsModule._resolve_test_output_config(
        module, batch={}
    )

    assert mode == "test"
    assert output_dir == "/tmp/test_results"
    assert cache_suffix == "_x1_prediction.h5"
    assert filenames == ["sample_a"]


def test_resolve_test_output_config_uses_current_tta_suffix_when_tta_is_enabled(monkeypatch):
    cfg = _base_config()
    cfg.inference.save_prediction.output_path = "/tmp/test_results"
    cfg.inference.save_prediction.cache_suffix = "_x1_prediction.h5"
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.flip_axes = [0, 1, 2]
    cfg.inference.test_time_augmentation.rotation90_axes = [[1, 2]]
    module = ConnectomicsModule(cfg, model=SimpleModel())

    monkeypatch.setattr(
        "connectomics.training.lightning.model.resolve_output_filenames",
        lambda _cfg, _batch, global_step=0: ["sample_a"],
    )

    mode, output_dir, cache_suffix, filenames = ConnectomicsModule._resolve_test_output_config(
        module, batch={}
    )

    assert mode == "test"
    assert output_dir == "/tmp/test_results"
    assert cache_suffix == "_tta_x12_prediction.h5"
    assert filenames == ["sample_a"]


def test_resolve_test_output_config_includes_checkpoint_name_in_tta_suffix(monkeypatch):
    cfg = _base_config()
    cfg.inference.save_prediction.output_path = "/tmp/test_results"
    cfg.inference.save_prediction.cache_suffix = "_x1_prediction.h5"
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.flip_axes = [0, 1, 2]
    cfg.inference.test_time_augmentation.rotation90_axes = [[1, 2]]
    module = ConnectomicsModule(cfg, model=SimpleModel())
    module._prediction_checkpoint_path = "/tmp/checkpoints/best.ckpt"

    monkeypatch.setattr(
        "connectomics.training.lightning.model.resolve_output_filenames",
        lambda _cfg, _batch, global_step=0: ["sample_a"],
    )

    mode, output_dir, cache_suffix, filenames = ConnectomicsModule._resolve_test_output_config(
        module, batch={}
    )

    assert mode == "test"
    assert output_dir == "/tmp/test_results"
    assert cache_suffix == "_tta_x12_ckpt-best_prediction.h5"
    assert filenames == ["sample_a"]


def test_resolve_test_output_config_includes_output_head_in_tta_suffix(monkeypatch):
    cfg = _base_config()
    cfg.model.out_channels = 3
    cfg.model.primary_head = "affinity"
    cfg.model.heads = {
        "affinity": {"out_channels": 2, "num_blocks": 0},
        "sdt": {"out_channels": 1, "num_blocks": 0},
    }
    cfg.inference.head = "sdt"
    cfg.inference.save_prediction.output_path = "/tmp/test_results"
    cfg.inference.save_prediction.cache_suffix = "_x1_prediction.h5"
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.flip_axes = None
    cfg.inference.test_time_augmentation.rotation90_axes = None
    module = ConnectomicsModule(cfg, model=SimpleMultiHeadModel())

    monkeypatch.setattr(
        "connectomics.training.lightning.model.resolve_output_filenames",
        lambda _cfg, _batch, global_step=0: ["sample_a"],
    )

    mode, output_dir, cache_suffix, filenames = ConnectomicsModule._resolve_test_output_config(
        module, batch={}
    )

    assert mode == "test"
    assert output_dir == "/tmp/test_results"
    assert cache_suffix == "_tta_x1_head-sdt_prediction.h5"
    assert filenames == ["sample_a"]


def test_save_metrics_to_file_uses_runtime_inference_output_path(tmp_path):
    """Metrics should be written under cfg.inference.save_prediction.output_path."""
    cfg = _base_config()
    cfg.inference.save_prediction.output_path = str(tmp_path)
    module = ConnectomicsModule(cfg, model=SimpleModel())

    ConnectomicsModule._save_metrics_to_file(
        module,
        {
            "volume_name": "vol0",
            "jaccard": 0.5,
        },
    )

    assert (tmp_path / "evaluation_metrics_vol0_x1_prediction.txt").exists()


def test_save_metrics_to_file_matches_final_prediction_tag(tmp_path):
    cfg = _base_config()
    cfg.inference.save_prediction.output_path = str(tmp_path)
    cfg.inference.select_channel = [0, 1, 2]
    cfg.inference.decoding = [
        {
            "name": "decode_waterz",
            "kwargs": {
                "merge_function": "aff50_his256",
                "thresholds": [0.4],
            },
        }
    ]
    module = ConnectomicsModule(cfg, model=SimpleModel())
    module._prediction_checkpoint_path = "/tmp/checkpoints/best.ckpt"

    ConnectomicsModule._save_metrics_to_file(
        module,
        {
            "volume_name": "test-input",
            "jaccard": 0.5,
        },
    )

    assert (
        tmp_path
        / "evaluation_metrics_test-input_x1_ch0-1-2_ckpt-best_prediction_waterz_aff50_his256-0.4.txt"
    ).exists()

    tsv_path = tmp_path / "decode_experiments.tsv"
    assert tsv_path.exists()
    lines = tsv_path.read_text().strip().splitlines()
    assert "input_tta_prediction_name" in lines[0]
    assert "test-input_tta_x1_ch0-1-2_ckpt-best_prediction.h5" in lines[1]


def test_load_cached_predictions_reads_existing_prediction_files(tmp_path, monkeypatch):
    """Existing cached predictions should load without falling back to inference."""
    cfg = _base_config()
    module = ConnectomicsModule(cfg, model=SimpleModel())
    pred_file = tmp_path / "sample_x1_prediction.h5"
    pred_file.write_text("stub")

    expected = np.ones((1, 4, 4, 4), dtype=np.float32)
    monkeypatch.setattr(
        "connectomics.training.lightning.model.read_volume", lambda *_args, **_kwargs: expected
    )

    predictions, loaded, suffix = module._load_cached_predictions(
        str(tmp_path),
        ["sample"],
        "_x1_prediction.h5",
        "test",
    )

    assert loaded is True
    assert suffix == "_x1_prediction.h5"
    assert predictions.shape == (1, 4, 4, 4)


def test_load_cached_predictions_prefers_final_prediction_for_checkpoint_tagged_tta_cache(
    tmp_path, monkeypatch
):
    cfg = _base_config()
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.flip_axes = [0, 1, 2]
    cfg.inference.test_time_augmentation.rotation90_axes = [[1, 2]]
    module = ConnectomicsModule(cfg, model=SimpleModel())
    module._prediction_checkpoint_path = "/tmp/checkpoints/best.ckpt"
    pred_file = tmp_path / "sample_x12_ckpt-best_prediction.h5"
    pred_file.write_text("stub")

    expected = np.ones((1, 4, 4, 4), dtype=np.float32)
    monkeypatch.setattr(
        "connectomics.training.lightning.model.read_volume", lambda *_args, **_kwargs: expected
    )

    predictions, loaded, suffix = module._load_cached_predictions(
        str(tmp_path),
        ["sample"],
        "_tta_x12_ckpt-best_prediction.h5",
        "test",
    )

    assert loaded is True
    assert suffix == "_x12_ckpt-best_prediction.h5"
    assert predictions.shape == (1, 4, 4, 4)


def test_load_cached_predictions_does_not_pick_unrelated_tta_channel_cache(tmp_path, monkeypatch):
    cfg = _base_config()
    cfg.inference.save_prediction.cache_suffix = "_x1_ch4-6-9_prediction_waterz_t0.4.h5"
    cfg.inference.select_channel = [4, 6, 9]
    module = ConnectomicsModule(cfg, model=SimpleModel())
    unrelated_tta_file = tmp_path / "sample_tta_x1_ch0-1-2_prediction.h5"
    unrelated_tta_file.write_text("stub")

    monkeypatch.setattr(
        "connectomics.training.lightning.model.read_volume",
        lambda *_args, **_kwargs: np.ones((1, 4, 4, 4), dtype=np.float32),
    )

    predictions, loaded, suffix = module._load_cached_predictions(
        str(tmp_path),
        ["sample"],
        "_x1_ch4-6-9_prediction_waterz_t0.4.h5",
        "test",
    )

    assert predictions is None
    assert loaded is False
    assert suffix == "_x1_ch4-6-9_prediction_waterz_t0.4.h5"


def test_load_cached_predictions_does_not_pick_legacy_tta_cache_when_checkpoint_tag_is_expected(
    tmp_path, monkeypatch
):
    cfg = _base_config()
    cfg.inference.test_time_augmentation.enabled = True
    module = ConnectomicsModule(cfg, model=SimpleModel())
    module._prediction_checkpoint_path = (
        "/tmp/checkpoints/epoch=074-train_loss_total_epoch=1.0462.ckpt"
    )

    legacy_tta_file = tmp_path / "sample_tta_x1_prediction.h5"
    legacy_tta_file.write_text("stub")

    monkeypatch.setattr(
        "connectomics.training.lightning.model.read_volume",
        lambda *_args, **_kwargs: np.ones((1, 4, 4, 4), dtype=np.float32),
    )

    predictions, loaded, suffix = module._load_cached_predictions(
        str(tmp_path),
        ["sample"],
        "_x1_prediction.h5",
        "test",
    )

    assert predictions is None
    assert loaded is False
    assert suffix == "_x1_prediction.h5"


def test_on_test_epoch_end_logs_aggregated_metrics_once():
    cfg = _base_config()
    cfg.inference.evaluation.enabled = True
    cfg.inference.evaluation.metrics = ["accuracy"]

    module = ConnectomicsModule(cfg, model=SimpleModel())
    logged_names: list[str] = []
    _stub_logging(module, sink=logged_names)
    module.test_accuracy = torchmetrics.Accuracy(task="binary")
    module.test_accuracy.update(torch.tensor([1, 0]), torch.tensor([1, 0]))

    module.on_test_epoch_end()

    assert logged_names == ["test_accuracy"]


def test_log_test_epoch_metrics_uses_rank_zero_only_logging_for_distributed_tta_sharding(
    monkeypatch,
):
    cfg = _base_config()
    cfg.inference.evaluation.enabled = True
    cfg.inference.evaluation.metrics = ["accuracy"]
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.distributed_sharding = True

    module = ConnectomicsModule(cfg, model=SimpleModel())
    calls: list[tuple[str, bool]] = []

    def log_override(name, *_args, **kwargs):
        calls.append((name, kwargs["sync_dist"]))
        return None

    module.log = log_override
    module.test_accuracy = torchmetrics.Accuracy(task="binary")
    module.test_accuracy.update(torch.tensor([1, 0]), torch.tensor([1, 0]))

    monkeypatch.setattr(
        "connectomics.training.lightning.test_pipeline._is_distributed_tta_sharding_active",
        lambda _module: True,
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    log_test_epoch_metrics(module)

    assert calls == [("test_accuracy", False)]


def test_log_test_epoch_metrics_skips_nonzero_ranks_for_distributed_tta_sharding(monkeypatch):
    cfg = _base_config()
    cfg.inference.evaluation.enabled = True
    cfg.inference.evaluation.metrics = ["accuracy"]
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.distributed_sharding = True

    module = ConnectomicsModule(cfg, model=SimpleModel())
    calls: list[str] = []

    def log_override(name, *_args, **_kwargs):
        calls.append(name)
        return None

    module.log = log_override
    module.test_accuracy = torchmetrics.Accuracy(task="binary")
    module.test_accuracy.update(torch.tensor([1, 0]), torch.tensor([1, 0]))

    monkeypatch.setattr(
        "connectomics.training.lightning.test_pipeline._is_distributed_tta_sharding_active",
        lambda _module: True,
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    log_test_epoch_metrics(module)

    assert calls == []
