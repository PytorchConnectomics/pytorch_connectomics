"""Loss metadata describing how PyTorch loss modules are invoked."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn


LossCallKind = str
TargetKind = str


@dataclass(frozen=True)
class LossMetadata:
    """Static metadata describing how to invoke a loss module."""

    name: str
    call_kind: LossCallKind = "pred_target"  # pred_target | pred_only | pred_pred | unsupported
    target_kind: TargetKind = "dense"  # dense | class_index | none
    spatial_weight_arg: Optional[str] = None  # weight | mask | None


_LOSS_METADATA_BY_NAME = {
    # Standard supervised segmentation losses (dense targets unless noted)
    "DiceLoss": LossMetadata("DiceLoss"),
    "DiceCELoss": LossMetadata("DiceCELoss"),
    "DiceFocalLoss": LossMetadata("DiceFocalLoss"),
    "GeneralizedDiceLoss": LossMetadata("GeneralizedDiceLoss"),
    "FocalLoss": LossMetadata("FocalLoss"),
    "TverskyLoss": LossMetadata("TverskyLoss"),
    "BCEWithLogitsLoss": LossMetadata("BCEWithLogitsLoss"),
    "CrossEntropyLoss": LossMetadata("CrossEntropyLoss", target_kind="class_index"),
    "MSELoss": LossMetadata("MSELoss"),
    "L1Loss": LossMetadata("L1Loss"),
    # Custom supervised losses
    "SmoothL1Loss": LossMetadata("SmoothL1Loss", spatial_weight_arg="weight"),
    "WeightedBCEWithLogitsLoss": LossMetadata(
        "WeightedBCEWithLogitsLoss", spatial_weight_arg="weight"
    ),
    "WeightedMSELoss": LossMetadata("WeightedMSELoss", spatial_weight_arg="weight"),
    "WeightedMAELoss": LossMetadata("WeightedMAELoss", spatial_weight_arg="weight"),
    # GAN is not compatible with the generic supervised orchestrator path
    "GANLoss": LossMetadata("GANLoss", call_kind="unsupported", target_kind="none"),
    # Regularization losses
    "BinaryRegularization": LossMetadata(
        "BinaryRegularization", call_kind="pred_only", target_kind="none", spatial_weight_arg="mask"
    ),
    "ForegroundDistanceConsistency": LossMetadata(
        "ForegroundDistanceConsistency",
        call_kind="pred_pred",
        target_kind="none",
        spatial_weight_arg="mask",
    ),
    "ContourDistanceConsistency": LossMetadata(
        "ContourDistanceConsistency",
        call_kind="pred_pred",
        target_kind="none",
        spatial_weight_arg="mask",
    ),
    "ForegroundContourConsistency": LossMetadata(
        "ForegroundContourConsistency",
        call_kind="pred_pred",
        target_kind="none",
        spatial_weight_arg="mask",
    ),
    "NonOverlapRegularization": LossMetadata(
        "NonOverlapRegularization", call_kind="pred_only", target_kind="none"
    ),
}


_CLASSNAME_TO_METADATA_NAME = {
    # Torch / MONAI / custom classes
    "DiceLoss": "DiceLoss",
    "DiceCELoss": "DiceCELoss",
    "DiceFocalLoss": "DiceFocalLoss",
    "GeneralizedDiceLoss": "GeneralizedDiceLoss",
    "FocalLoss": "FocalLoss",
    "TverskyLoss": "TverskyLoss",
    "BCEWithLogitsLoss": "BCEWithLogitsLoss",
    "CrossEntropyLoss": "CrossEntropyLoss",
    "CrossEntropyLossWrapper": "CrossEntropyLoss",
    "MSELoss": "MSELoss",
    "L1Loss": "L1Loss",
    "SmoothL1Loss": "SmoothL1Loss",
    "WeightedBCEWithLogitsLoss": "WeightedBCEWithLogitsLoss",
    "WeightedMSELoss": "WeightedMSELoss",
    "WeightedMAELoss": "WeightedMAELoss",
    "GANLoss": "GANLoss",
    "BinaryRegularization": "BinaryRegularization",
    "ForegroundDistanceConsistency": "ForegroundDistanceConsistency",
    "ContourDistanceConsistency": "ContourDistanceConsistency",
    "ForegroundContourConsistency": "ForegroundContourConsistency",
    "NonOverlapRegularization": "NonOverlapRegularization",
}


def get_loss_metadata(loss_name: str) -> LossMetadata:
    """Return registered metadata for a known loss name."""
    if loss_name not in _LOSS_METADATA_BY_NAME:
        raise ValueError(f"No metadata registered for loss: {loss_name}")
    return _LOSS_METADATA_BY_NAME[loss_name]


def attach_loss_metadata(loss_fn: nn.Module, loss_name: str) -> nn.Module:
    """Attach registered loss metadata to a module instance for downstream dispatch."""
    setattr(loss_fn, "_connectomics_loss_metadata", get_loss_metadata(loss_name))
    return loss_fn


def get_loss_metadata_for_module(loss_fn: nn.Module) -> LossMetadata:
    """Fetch metadata from an annotated module, or infer a safe fallback."""
    meta = getattr(loss_fn, "_connectomics_loss_metadata", None)
    if isinstance(meta, LossMetadata):
        return meta

    class_name = loss_fn.__class__.__name__
    mapped_name = _CLASSNAME_TO_METADATA_NAME.get(class_name)
    if mapped_name is not None:
        return _LOSS_METADATA_BY_NAME[mapped_name]

    # Conservative fallback: supervised dense target without optional spatial weighting.
    return LossMetadata(name=class_name)


__all__ = [
    "LossCallKind",
    "TargetKind",
    "LossMetadata",
    "attach_loss_metadata",
    "get_loss_metadata",
    "get_loss_metadata_for_module",
]
