from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from connectomics.config import Config
from connectomics.config.schema.stages import TestConfig as HydraTestConfig
from connectomics.training.lightning import ConnectomicsModule


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
    cfg.inference.save_prediction.cache_suffix = "_prediction.h5"
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
    assert cache_suffix == "_prediction.h5"
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

    assert (tmp_path / "evaluation_metrics_vol0.txt").exists()


def test_load_cached_predictions_reads_existing_prediction_files(tmp_path, monkeypatch):
    """Existing cached predictions should load without falling back to inference."""
    cfg = _base_config()
    module = ConnectomicsModule(cfg, model=SimpleModel())
    pred_file = tmp_path / "sample_prediction.h5"
    pred_file.write_text("stub")

    expected = np.ones((1, 4, 4, 4), dtype=np.float32)
    monkeypatch.setattr("connectomics.training.lightning.model.read_volume", lambda *_args, **_kwargs: expected)

    predictions, loaded, suffix = module._load_cached_predictions(
        str(tmp_path),
        ["sample"],
        "_prediction.h5",
        "test",
    )

    assert loaded is True
    assert suffix == "_prediction.h5"
    assert predictions.shape == (1, 4, 4, 4)
