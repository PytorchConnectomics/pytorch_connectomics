from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from connectomics.models.loss import LossMetadata, create_loss
from connectomics.training.loss import LossOrchestrator


def _cfg(losses=None):
    model = SimpleNamespace(
        deep_supervision_clamp_min=-20.0,
        deep_supervision_clamp_max=20.0,
        deep_supervision_weights=None,
        losses=losses,
    )
    return SimpleNamespace(model=model)


class WeightAwareSpyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self._connectomics_loss_metadata = LossMetadata(
            name="WeightAwareSpyLoss",
            call_kind="pred_target",
            target_kind="dense",
            spatial_weight_arg="weight",
        )

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


class MaskPredOnlySpyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, pred: torch.Tensor, mask: torch.Tensor = None):
        self.calls.append(
            {
                "pred_shape": tuple(pred.shape),
                "mask_shape": None if mask is None else tuple(mask.shape),
                "mask": None if mask is None else mask.detach().clone(),
            }
        )
        loss = pred.abs()
        if mask is not None:
            loss = loss * mask
        return loss.mean()


class MaskPredPredSpyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, pred1: torch.Tensor, pred2: torch.Tensor, mask: torch.Tensor = None):
        self.calls.append(
            {
                "pred1_shape": tuple(pred1.shape),
                "pred2_shape": tuple(pred2.shape),
                "mask_shape": None if mask is None else tuple(mask.shape),
                "mask": None if mask is None else mask.detach().clone(),
            }
        )
        loss = (pred1 - pred2).abs()
        if mask is not None:
            loss = loss * mask
        return loss.mean()


def _expected_foreground_weight(target: torch.Tensor) -> torch.Tensor:
    out = torch.ones_like(target)
    out[target > 0] = 2.0
    return out


def test_create_loss_attaches_metadata_for_supervised_and_regularization_losses():
    weighted_mse = create_loss("WeightedMSELoss")
    binary_reg = create_loss("BinaryRegularization")

    weighted_meta = weighted_mse._connectomics_loss_metadata
    reg_meta = binary_reg._connectomics_loss_metadata

    assert weighted_meta.name == "WeightedMSELoss"
    assert weighted_meta.call_kind == "pred_target"
    assert weighted_meta.spatial_weight_arg == "weight"

    assert reg_meta.name == "BinaryRegularization"
    assert reg_meta.call_kind == "pred_only"
    assert reg_meta.spatial_weight_arg == "mask"


def test_loss_orchestrator_requires_explicit_losses():
    with pytest.raises(ValueError, match="model\\.losses is required"):
        LossOrchestrator(
            cfg=_cfg(),
            loss_functions=nn.ModuleList([NoWeightSpyLoss()]),
            loss_weights=[1.0],
            enable_nan_detection=False,
            debug_on_nan=False,
        )


def test_standard_loss_uses_foreground_weight_only_for_weight_aware_losses():
    weighted_loss = WeightAwareSpyLoss()
    no_weight_loss = NoWeightSpyLoss()
    orchestrator = LossOrchestrator(
        cfg=_cfg(
            losses=[
                {"pred_slice": [0, 1], "target_slice": [0, 1], "weight": 1.0},
                {"pred_slice": [0, 1], "target_slice": [0, 1], "weight": 1.0},
            ]
        ),
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
        losses=[
            {
                "pred_slice": [0, 3],
                "target_slice": [0, 1],
                "target_kind": "class_index",
                "weight": 1.0,
            },
            {
                "pred_slice": [3, 4],
                "target_slice": [1, 2],
                "weight": 1.0,
            },
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

    total_loss, loss_dict = orchestrator.compute_standard_loss(outputs, labels, stage="train")

    assert torch.isfinite(total_loss)
    assert "train_loss_task_loss_0_weight" in loss_dict
    assert "train_loss_task_loss_1_weight" in loss_dict
    assert ce_loss.target_shapes == [(1, 1, 4, 4, 4)]
    assert len(reg_loss.calls) == 1
    assert reg_loss.calls[0]["weight"] is not None


def test_deep_supervision_multitask_resizes_targets_per_task_and_applies_foreground_weight():
    ce_loss = CrossEntropyLossWrapperSpy()
    reg_loss = WeightAwareSpyLoss()
    cfg = _cfg(
        losses=[
            {
                "pred_slice": [0, 3],
                "target_slice": [0, 1],
                "target_kind": "class_index",
                "weight": 1.0,
            },
            {
                "pred_slice": [3, 4],
                "target_slice": [1, 2],
                "weight": 1.0,
            },
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


def test_explicit_loss_terms_support_pred_only_pred_pred_and_mask_dispatch():
    pred_only_loss = MaskPredOnlySpyLoss()
    pred_only_loss._connectomics_loss_metadata = LossMetadata(
        name="MaskPredOnlySpy",
        call_kind="pred_only",
        target_kind="none",
        spatial_weight_arg="mask",
    )

    pred_pred_loss = MaskPredPredSpyLoss()
    pred_pred_loss._connectomics_loss_metadata = LossMetadata(
        name="MaskPredPredSpy",
        call_kind="pred_pred",
        target_kind="none",
        spatial_weight_arg="mask",
    )

    cfg = _cfg(
        losses=[
            {
                "pred_slice": [0, 1],
                "mask_slice": [0, 1],
                "weight": 0.25,
            },
            {
                "pred_slice": [1, 2],
                "pred2_slice": [2, 3],
                "mask_slice": [1, 2],
                "weight": 0.5,
            },
        ]
    )

    orchestrator = LossOrchestrator(
        cfg=cfg,
        loss_functions=nn.ModuleList([pred_only_loss, pred_pred_loss]),
        loss_weights=[0.25, 0.5],
        enable_nan_detection=False,
        debug_on_nan=False,
    )

    outputs = torch.randn(1, 3, 4, 4, 4)
    labels = torch.rand(1, 2, 4, 4, 4)

    total_loss, loss_dict = orchestrator.compute_standard_loss(outputs, labels, stage="train")

    assert torch.isfinite(total_loss)
    assert len(pred_only_loss.calls) == 1
    assert len(pred_pred_loss.calls) == 1
    assert pred_only_loss.calls[0]["mask_shape"] == (1, 1, 4, 4, 4)
    assert pred_pred_loss.calls[0]["mask_shape"] == (1, 1, 4, 4, 4)
    assert "train_loss_term_loss_0_raw" in loss_dict
    assert "train_loss_term_loss_1_raw" in loss_dict
    assert "train_loss_task_loss_0_weight" in loss_dict
    assert "train_loss_task_loss_1_weight" in loss_dict


def test_explicit_loss_terms_deep_supervision_resizes_masks_and_respects_main_only_terms():
    pred_only_loss = MaskPredOnlySpyLoss()
    pred_only_loss._connectomics_loss_metadata = LossMetadata(
        name="MaskPredOnlySpy",
        call_kind="pred_only",
        target_kind="none",
        spatial_weight_arg="mask",
    )

    pred_pred_loss = MaskPredPredSpyLoss()
    pred_pred_loss._connectomics_loss_metadata = LossMetadata(
        name="MaskPredPredSpy",
        call_kind="pred_pred",
        target_kind="none",
        spatial_weight_arg="mask",
    )

    cfg = _cfg(
        losses=[
            {
                "pred_slice": [0, 1],
                "mask_slice": [0, 1],
                "apply_deep_supervision": False,
                "weight": 1.0,
            },
            {
                "pred_slice": [1, 2],
                "pred2_slice": [2, 3],
                "mask_slice": [1, 2],
                "weight": 1.0,
            },
        ]
    )

    orchestrator = LossOrchestrator(
        cfg=cfg,
        loss_functions=nn.ModuleList([pred_only_loss, pred_pred_loss]),
        loss_weights=[1.0, 1.0],
        enable_nan_detection=False,
        debug_on_nan=False,
    )

    outputs = {
        "output": torch.randn(1, 3, 6, 6, 6),
        "ds_1": torch.randn(1, 3, 3, 3, 3),
    }
    labels = torch.rand(1, 2, 6, 6, 6)

    total_loss, loss_dict = orchestrator.compute_deep_supervision_loss(outputs, labels, stage="train")

    assert torch.isfinite(total_loss)
    assert len(pred_only_loss.calls) == 1  # main scale only
    assert len(pred_pred_loss.calls) == 2  # main + ds_1
    assert pred_pred_loss.calls[0]["mask_shape"] == (1, 1, 6, 6, 6)
    assert pred_pred_loss.calls[1]["mask_shape"] == (1, 1, 3, 3, 3)
    assert "train_loss_scale_0_term_loss_0_raw" in loss_dict
    assert "train_loss_scale_1_term_loss_1_raw" in loss_dict
