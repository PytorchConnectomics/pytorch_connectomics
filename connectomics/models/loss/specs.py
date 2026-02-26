"""Loss metadata and explicit loss-term specifications.

This module provides:
- Metadata describing how each loss should be called (pred/target vs pred-only, etc.)
- Parsing/validation for explicit `model.loss_terms` configs
- Small helpers for task counting and metadata lookup
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class LossTermSpec:
    """Compiled explicit loss term used by the training loss orchestrator."""

    name: str
    loss_index: int
    coefficient: float
    call_kind: LossCallKind
    target_kind: TargetKind
    pred_slice: Optional[Tuple[int, int]] = None
    target_slice: Optional[Tuple[int, int]] = None
    pred2_slice: Optional[Tuple[int, int]] = None
    mask_slice: Optional[Tuple[int, int]] = None
    task_name: Optional[str] = None
    apply_deep_supervision: bool = True
    spatial_weight_arg: Optional[str] = None


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
    "WeightedBCEWithLogitsLoss": LossMetadata("WeightedBCEWithLogitsLoss"),
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


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a key/attribute from dict-like or attribute-like config objects."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    # OmegaConf DictConfig implements get(); SimpleNamespace does not.
    get_fn = getattr(obj, "get", None)
    if callable(get_fn):
        try:
            return get_fn(key, default)
        except TypeError:
            pass
    return getattr(obj, key, default)


def _coerce_slice(value: Any, field_name: str) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"{field_name} must be [start, end], got: {value!r}"
        )
    start, end = int(value[0]), int(value[1])
    if end <= start:
        raise ValueError(f"{field_name} must satisfy end > start, got: {value!r}")
    return (start, end)


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


def compile_loss_terms_from_config(
    cfg: Any,
    loss_functions: Sequence[nn.Module],
    loss_weights: Sequence[float],
    *,
    loss_metadata: Optional[Sequence[LossMetadata]] = None,
) -> List[LossTermSpec]:
    """Compile explicit `model.loss_terms` config to validated loss term specs."""
    model_cfg = _cfg_get(cfg, "model", None)
    terms_cfg = _cfg_get(model_cfg, "loss_terms", None)
    metas = list(loss_metadata) if loss_metadata is not None else [
        get_loss_metadata_for_module(loss_fn) for loss_fn in loss_functions
    ]
    compiled: List[LossTermSpec] = []

    if terms_cfg is None:
        raise ValueError(
            "model.loss_terms is required. Configure explicit loss terms "
            "(pred/target slices, task names, and optional masks)."
        )

    terms_list = list(terms_cfg)

    def _resolve_loss_index(term_cfg: Any, term_idx: int) -> int:
        raw = _cfg_get(term_cfg, "loss_index", None)
        if raw is None:
            raw = _cfg_get(term_cfg, "loss", None)
        if raw is None:
            raise ValueError(
                f"loss_terms[{term_idx}] must define 'loss_index' (or integer 'loss')"
            )
        if isinstance(raw, str):
            matches = [i for i, m in enumerate(metas) if m.name == raw]
            if len(matches) == 1:
                return matches[0]
            if len(matches) == 0:
                raise ValueError(
                    f"loss_terms[{term_idx}] references unknown loss name {raw!r}. "
                    "Use loss_index or include the loss in model.loss_functions."
                )
            raise ValueError(
                f"loss_terms[{term_idx}] loss name {raw!r} matches multiple entries in "
                "model.loss_functions. Use loss_index to disambiguate."
            )
        idx = int(raw)
        if idx < 0 or idx >= len(loss_functions):
            raise ValueError(
                f"loss_terms[{term_idx}] loss_index {idx} out of range for "
                f"{len(loss_functions)} configured losses"
            )
        return idx

    for term_idx, term_cfg in enumerate(terms_list):
        loss_index = _resolve_loss_index(term_cfg, term_idx)
        base_meta = metas[loss_index]

        call_kind = str(_cfg_get(term_cfg, "call_kind", _cfg_get(term_cfg, "call", base_meta.call_kind)))
        target_kind = str(_cfg_get(term_cfg, "target_kind", base_meta.target_kind))
        spatial_weight_arg = _cfg_get(term_cfg, "spatial_weight_arg", base_meta.spatial_weight_arg)

        pred_slice = _coerce_slice(_cfg_get(term_cfg, "pred_slice", _cfg_get(term_cfg, "pred", None)), "pred_slice")
        target_slice = _coerce_slice(_cfg_get(term_cfg, "target_slice", _cfg_get(term_cfg, "target", None)), "target_slice")
        pred2_slice = _coerce_slice(_cfg_get(term_cfg, "pred2_slice", _cfg_get(term_cfg, "pred2", None)), "pred2_slice")
        mask_slice = _coerce_slice(_cfg_get(term_cfg, "mask_slice", _cfg_get(term_cfg, "mask", None)), "mask_slice")

        if call_kind == "unsupported":
            raise ValueError(
                f"loss_terms[{term_idx}] uses unsupported generic loss '{base_meta.name}'. "
                "Wrap it in a custom training step path or custom loss term executor."
            )
        if call_kind not in {"pred_target", "pred_only", "pred_pred"}:
            raise ValueError(f"Unsupported call_kind {call_kind!r} in loss_terms[{term_idx}]")

        if call_kind == "pred_target":
            if pred_slice is None or target_slice is None:
                raise ValueError(
                    f"loss_terms[{term_idx}] pred_target terms require pred_slice and target_slice"
                )
        elif call_kind == "pred_only":
            if pred_slice is None:
                raise ValueError(
                    f"loss_terms[{term_idx}] pred_only terms require pred_slice"
                )
            if target_kind != "none":
                target_kind = "none"
        elif call_kind == "pred_pred":
            if pred_slice is None or pred2_slice is None:
                raise ValueError(
                    f"loss_terms[{term_idx}] pred_pred terms require pred_slice and pred2_slice"
                )
            if target_kind != "none":
                target_kind = "none"

        if target_kind not in {"dense", "class_index", "none"}:
            raise ValueError(f"Unsupported target_kind {target_kind!r} in loss_terms[{term_idx}]")

        coefficient = float(
            _cfg_get(term_cfg, "coefficient", _cfg_get(term_cfg, "weight", loss_weights[loss_index]))
        )
        task_name = _cfg_get(term_cfg, "task_name", _cfg_get(term_cfg, "task", None))
        name = _cfg_get(term_cfg, "name", f"term_{term_idx}")
        apply_deep_supervision = bool(_cfg_get(term_cfg, "apply_deep_supervision", True))

        compiled.append(
            LossTermSpec(
                name=str(name),
                loss_index=loss_index,
                coefficient=coefficient,
                call_kind=call_kind,
                target_kind=target_kind,
                pred_slice=pred_slice,
                target_slice=target_slice,
                pred2_slice=pred2_slice,
                mask_slice=mask_slice,
                task_name=None if task_name is None else str(task_name),
                apply_deep_supervision=apply_deep_supervision,
                spatial_weight_arg=None if spatial_weight_arg is None else str(spatial_weight_arg),
            )
        )

    return compiled


def infer_num_loss_tasks_from_config(cfg: Any) -> int:
    """Infer task count for adaptive loss weighting from explicit loss terms."""
    model_cfg = _cfg_get(cfg, "model", None)
    terms_cfg = _cfg_get(model_cfg, "loss_terms", None)
    if terms_cfg:
        task_names = []
        for term_idx, term_cfg in enumerate(list(terms_cfg)):
            task_name = _cfg_get(term_cfg, "task_name", _cfg_get(term_cfg, "task", None))
            if task_name is None:
                task_name = _cfg_get(term_cfg, "name", f"term_{term_idx}")
            task_names.append(str(task_name))
        return max(1, len(set(task_names)))

    return 1


__all__ = [
    "LossMetadata",
    "LossTermSpec",
    "attach_loss_metadata",
    "compile_loss_terms_from_config",
    "get_loss_metadata",
    "get_loss_metadata_for_module",
    "infer_num_loss_tasks_from_config",
]
