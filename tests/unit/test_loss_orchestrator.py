from types import SimpleNamespace

import torch
import torch.nn as nn

from connectomics.training.deep_supervision import DeepSupervisionHandler as LegacyDeepSupervisionHandler
from connectomics.training.deep_supervision import match_target_to_output as legacy_match_target
from connectomics.training.loss_orchestrator import (
    DeepSupervisionHandler,
    LossOrchestrator,
    match_target_to_output,
)


def _cfg(multi_task_config=None):
    model = SimpleNamespace(
        deep_supervision_clamp_min=-20.0,
        deep_supervision_clamp_max=20.0,
        multi_task_config=multi_task_config,
        deep_supervision_weights=None,
    )
    return SimpleNamespace(model=model)


class WeightAwareSpyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None):
        self.calls.append(
            {
                "pred_shape": tuple(pred.shape),
                "target_shape": tuple(target.shape),
                "weight": None if weight is None else weight.detach().clone(),
                "target": target.detach().clone(),
            }
        )
        loss = (pred - target).abs()
        if weight is not None:
            loss = loss * weight
        return loss.mean()


class NoWeightSpyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        self.call_count += 1
        return (pred - target).abs().mean()


class _CrossEntropySpyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_shapes = []
        self.targets = []

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        self.target_shapes.append(tuple(target.shape))
        self.targets.append(target.detach().clone())
        # Keep a differentiable scalar tied to the input.
        return input.mean() * 0.0


CrossEntropyLossWrapperSpy = type("CrossEntropyLossWrapper", (_CrossEntropySpyBase,), {})


def _expected_foreground_weight(target: torch.Tensor) -> torch.Tensor:
    out = torch.ones_like(target)
    out[target > 0] = 2.0
    return out


def test_deep_supervision_shim_reexports_new_symbols():
    assert LegacyDeepSupervisionHandler is DeepSupervisionHandler
    assert DeepSupervisionHandler is LossOrchestrator
    assert legacy_match_target is match_target_to_output


def test_standard_loss_uses_foreground_weight_only_for_weight_aware_losses():
    weighted_loss = WeightAwareSpyLoss()
    no_weight_loss = NoWeightSpyLoss()
    orchestrator = LossOrchestrator(
        cfg=_cfg(),
        loss_functions=nn.ModuleList([weighted_loss, no_weight_loss]),
        loss_weights=[1.0, 1.0],
        enable_nan_detection=False,
        debug_on_nan=False,
    )

    outputs = torch.zeros(1, 1, 2, 2, 2)
    labels = torch.tensor(
        [[[[[-1.0, 0.0], [1.0, 2.0]], [[-0.5, 0.2], [0.0, 3.0]]]]]
    )

    total_loss, loss_dict = orchestrator.compute_standard_loss(outputs, labels, stage="train")

    assert torch.isfinite(total_loss)
    assert "train_loss_total" in loss_dict
    assert no_weight_loss.call_count == 1
    assert len(weighted_loss.calls) == 1
    received_weight = weighted_loss.calls[0]["weight"]
    assert received_weight is not None
    assert torch.equal(received_weight, _expected_foreground_weight(labels))


def test_multitask_single_scale_routes_class_index_and_dense_targets():
    ce_loss = CrossEntropyLossWrapperSpy()
    reg_loss = WeightAwareSpyLoss()
    cfg = _cfg(
        multi_task_config=[
            [0, 3, "seg", [0]],
            [3, 4, "sdt", [1]],
        ]
    )
    orchestrator = LossOrchestrator(
        cfg=cfg,
        loss_functions=nn.ModuleList([ce_loss, reg_loss]),
        loss_weights=[1.0, 1.0],
        enable_nan_detection=False,
        debug_on_nan=False,
    )

    outputs = torch.randn(1, 4, 4, 4, 4)
    class_labels = torch.randint(0, 3, (1, 1, 4, 4, 4)).float()
    dense_labels = torch.randn(1, 1, 4, 4, 4)
    labels = torch.cat([class_labels, dense_labels], dim=1)

    total_loss, loss_dict = orchestrator.compute_multitask_loss(outputs, labels, stage="train")

    assert torch.isfinite(total_loss)
    assert "train_loss_seg_weight" in loss_dict
    assert "train_loss_sdt_weight" in loss_dict
    assert ce_loss.target_shapes == [(1, 1, 4, 4, 4)]
    assert len(reg_loss.calls) == 1
    assert reg_loss.calls[0]["weight"] is not None


def test_deep_supervision_multitask_resizes_targets_per_task_and_applies_foreground_weight():
    ce_loss = CrossEntropyLossWrapperSpy()
    reg_loss = WeightAwareSpyLoss()
    cfg = _cfg(
        multi_task_config=[
            [0, 3, "seg", [0]],
            [3, 4, "sdt", [1]],
        ]
    )
    orchestrator = LossOrchestrator(
        cfg=cfg,
        loss_functions=nn.ModuleList([ce_loss, reg_loss]),
        loss_weights=[1.0, 1.0],
        enable_nan_detection=False,
        debug_on_nan=False,
    )

    main_output = torch.randn(1, 4, 6, 6, 6)
    ds_output = torch.randn(1, 4, 3, 3, 3)
    outputs = {"output": main_output, "ds_1": ds_output}

    class_labels = torch.randint(0, 3, (1, 1, 6, 6, 6)).float()
    dense_labels = torch.randn(1, 1, 6, 6, 6)
    labels = torch.cat([class_labels, dense_labels], dim=1)

    total_loss, loss_dict = orchestrator.compute_deep_supervision_loss(outputs, labels, stage="train")

    assert torch.isfinite(total_loss)
    assert "train_loss_scale_0" in loss_dict
    assert "train_loss_scale_1" in loss_dict

    # CE target routing should stay 1-channel at both scales.
    assert ce_loss.target_shapes == [(1, 1, 6, 6, 6), (1, 1, 3, 3, 3)]

    # Coarse-scale class labels must be nearest-resized (remain integer-valued).
    coarse_ce_target = ce_loss.targets[1]
    assert torch.allclose(coarse_ce_target, coarse_ce_target.round())

    # Regression task should receive foreground weighting on both scales (regression labels >0).
    assert len(reg_loss.calls) == 2
    assert reg_loss.calls[0]["target_shape"] == (1, 1, 6, 6, 6)
    assert reg_loss.calls[1]["target_shape"] == (1, 1, 3, 3, 3)
    assert reg_loss.calls[0]["weight"] is not None
    assert reg_loss.calls[1]["weight"] is not None
