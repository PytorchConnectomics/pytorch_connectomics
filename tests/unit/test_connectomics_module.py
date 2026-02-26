from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from connectomics.config import Config
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
    cfg.model.loss_functions = ["DiceLoss"]
    cfg.model.loss_weights = [1.0]
    cfg.model.loss_terms = [
        {
            "name": "seg",
            "loss_index": 0,
            "pred_slice": [0, 1],
            "target_slice": [0, 1],
            "task_name": "seg",
        }
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

    # Track that the deep supervision handler is invoked
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
    # At least one of the requested metrics should be logged
    assert any(name.startswith("val_") for name in logged_names)
