"""Training loss-plan compilation from unified losses config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch.nn as nn

from ...models.loss.metadata import LossCallKind, LossMetadata, TargetKind, get_loss_metadata_for_module


@dataclass(frozen=True)
class LossTermSpec:
    """Compiled loss term used by the training loss orchestrator."""

    name: str
    loss_index: int
    coefficient: float
    call_kind: LossCallKind
    target_kind: TargetKind
    pred_slice: Optional[Tuple[int, int]] = None
    target_slice: Optional[Tuple[int, int]] = None
    pred2_slice: Optional[Tuple[int, int]] = None
    mask_slice: Optional[Tuple[int, int]] = None
    apply_deep_supervision: bool = True
    spatial_weight_arg: Optional[str] = None
    foreground_weight: Optional[Union[float, str]] = None


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
        raise ValueError(f"{field_name} must be [start, end], got: {value!r}")
    start, end = int(value[0]), int(value[1])
    if end <= start:
        raise ValueError(f"{field_name} must satisfy end > start, got: {value!r}")
    return (start, end)


def compile_loss_terms_from_config(
    cfg: Any,
    loss_functions: Sequence[nn.Module],
    loss_weights: Sequence[float],
    *,
    loss_metadata: Optional[Sequence[LossMetadata]] = None,
) -> List[LossTermSpec]:
    """Compile unified ``model.losses`` config to validated loss term specs.

    Each entry in ``model.losses`` maps 1:1 to a loss function (by index)
    and carries its own weight, channel slices, and routing info.
    """
    model_cfg = _cfg_get(cfg, "model", None)
    losses_cfg = _cfg_get(model_cfg, "losses", None)
    metas = list(loss_metadata) if loss_metadata is not None else [
        get_loss_metadata_for_module(loss_fn) for loss_fn in loss_functions
    ]
    compiled: List[LossTermSpec] = []

    if losses_cfg is None:
        raise ValueError(
            "model.losses is required. Configure a list of loss entries "
            "with function, weight, pred_slice, and target_slice."
        )

    losses_list = list(losses_cfg)
    if len(losses_list) != len(loss_functions):
        raise ValueError(
            f"model.losses has {len(losses_list)} entries but "
            f"{len(loss_functions)} loss functions were built. These must match."
        )

    def _parse_foreground_weight(term_cfg: Any, term_idx: int) -> Optional[Union[float, str]]:
        raw = _cfg_get(term_cfg, "foreground_weight", None)
        if raw is None:
            return None
        if isinstance(raw, str):
            mode = raw.strip().lower()
            if mode != "ratio":
                raise ValueError(
                    f"losses[{term_idx}] foreground_weight must be a positive number "
                    f"or 'ratio', got {raw!r}"
                )
            return "ratio"
        value = float(raw)
        if value <= 0:
            raise ValueError(
                f"losses[{term_idx}] foreground_weight must be > 0, got {value}"
            )
        return value

    for term_idx, term_cfg in enumerate(losses_list):
        loss_index = term_idx  # 1:1 mapping
        base_meta = metas[loss_index]

        call_kind = str(
            _cfg_get(term_cfg, "call_kind", _cfg_get(term_cfg, "call", base_meta.call_kind))
        )
        target_kind = str(_cfg_get(term_cfg, "target_kind", base_meta.target_kind))
        spatial_weight_arg = _cfg_get(term_cfg, "spatial_weight_arg", base_meta.spatial_weight_arg)

        pred_slice = _coerce_slice(
            _cfg_get(term_cfg, "pred_slice", _cfg_get(term_cfg, "pred", None)),
            "pred_slice",
        )
        target_slice = _coerce_slice(
            _cfg_get(term_cfg, "target_slice", _cfg_get(term_cfg, "target", None)),
            "target_slice",
        )
        pred2_slice = _coerce_slice(
            _cfg_get(term_cfg, "pred2_slice", _cfg_get(term_cfg, "pred2", None)),
            "pred2_slice",
        )
        mask_slice = _coerce_slice(
            _cfg_get(term_cfg, "mask_slice", _cfg_get(term_cfg, "mask", None)),
            "mask_slice",
        )

        if call_kind == "unsupported":
            raise ValueError(
                f"losses[{term_idx}] uses unsupported generic loss '{base_meta.name}'. "
                "Wrap it in a custom training step path or custom loss term executor."
            )
        if call_kind not in {"pred_target", "pred_only", "pred_pred"}:
            raise ValueError(f"Unsupported call_kind {call_kind!r} in losses[{term_idx}]")

        if call_kind == "pred_target":
            if pred_slice is None or target_slice is None:
                raise ValueError(
                    f"losses[{term_idx}] pred_target terms require pred_slice and target_slice"
                )
        elif call_kind == "pred_only":
            if pred_slice is None:
                raise ValueError(f"losses[{term_idx}] pred_only terms require pred_slice")
            if target_kind != "none":
                target_kind = "none"
        elif call_kind == "pred_pred":
            if pred_slice is None or pred2_slice is None:
                raise ValueError(
                    f"losses[{term_idx}] pred_pred terms require pred_slice and pred2_slice"
                )
            if target_kind != "none":
                target_kind = "none"

        if target_kind not in {"dense", "class_index", "none"}:
            raise ValueError(f"Unsupported target_kind {target_kind!r} in losses[{term_idx}]")

        coefficient = float(
            _cfg_get(term_cfg, "coefficient", _cfg_get(term_cfg, "weight", loss_weights[loss_index]))
        )
        apply_deep_supervision = bool(_cfg_get(term_cfg, "apply_deep_supervision", True))
        foreground_weight = _parse_foreground_weight(term_cfg, term_idx)
        if foreground_weight is not None and spatial_weight_arg != "weight":
            raise ValueError(
                f"losses[{term_idx}] foreground_weight is only supported for losses "
                f"with spatial_weight_arg='weight' (got {base_meta.name})"
            )

        compiled.append(
            LossTermSpec(
                name=f"term_{term_idx}",
                loss_index=loss_index,
                coefficient=coefficient,
                call_kind=call_kind,
                target_kind=target_kind,
                pred_slice=pred_slice,
                target_slice=target_slice,
                pred2_slice=pred2_slice,
                mask_slice=mask_slice,
                apply_deep_supervision=apply_deep_supervision,
                spatial_weight_arg=None if spatial_weight_arg is None else str(spatial_weight_arg),
                foreground_weight=foreground_weight,
            )
        )

    return compiled


def infer_num_loss_tasks_from_config(cfg: Any) -> int:
    """Infer task count for adaptive loss weighting from unified losses config.

    Each loss entry is its own task (no task_name grouping).
    """
    model_cfg = _cfg_get(cfg, "model", None)
    losses_cfg = _cfg_get(model_cfg, "losses", None)
    if losses_cfg:
        return max(1, len(list(losses_cfg)))
    return 1


__all__ = [
    "LossTermSpec",
    "compile_loss_terms_from_config",
    "infer_num_loss_tasks_from_config",
]
