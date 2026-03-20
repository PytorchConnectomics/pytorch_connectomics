"""Shared helpers for selecting tensors from model outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch


def _cfg_value(cfg_obj: Any, key: str, default: Any = None) -> Any:
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, Mapping):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


def _get_model_heads(cfg: Any) -> Mapping[str, Any]:
    model_cfg = _cfg_value(cfg, "model", None)
    model_heads = _cfg_value(model_cfg, "heads", None) or {}
    return model_heads if isinstance(model_heads, Mapping) else {}


def get_model_head_names(cfg: Any) -> list[str]:
    """Return configured named model heads in declaration order."""
    return list(_get_model_heads(cfg).keys())


def get_total_model_head_channels(cfg: Any) -> int:
    """Return the sum of configured named head channel counts."""
    total = 0
    for head_cfg in _get_model_heads(cfg).values():
        total += int(_cfg_value(head_cfg, "out_channels", 0))
    return total


def resolve_output_head(
    cfg: Any,
    *,
    requested_head: Optional[str] = None,
    purpose: str = "output selection",
    allow_none: bool = True,
) -> Optional[str]:
    """Resolve the named output head requested explicitly or via config."""
    model_cfg = _cfg_value(cfg, "model", None)
    model_heads = _get_model_heads(cfg)
    if not model_heads:
        return None

    if requested_head is not None:
        if not isinstance(requested_head, str) or not requested_head.strip():
            raise ValueError(f"Requested output head for {purpose} must be a non-empty string.")
        requested_head = requested_head.strip()
        if requested_head not in model_heads:
            raise ValueError(
                f"Requested output head '{requested_head}' for {purpose} is not present in "
                f"model.heads ({sorted(model_heads.keys())})."
            )
        return requested_head

    inference_cfg = _cfg_value(cfg, "inference", None)
    configured_head = _cfg_value(inference_cfg, "head", None)
    if configured_head is not None:
        return resolve_output_head(
            cfg,
            requested_head=configured_head,
            purpose=purpose,
            allow_none=allow_none,
        )

    primary_head = _cfg_value(model_cfg, "primary_head", None)
    if primary_head is not None:
        if not isinstance(primary_head, str) or not primary_head.strip():
            raise ValueError(f"model.primary_head for {purpose} must be a non-empty string.")
        primary_head = primary_head.strip()
        if primary_head not in model_heads:
            raise ValueError(
                f"model.primary_head='{primary_head}' for {purpose} is not present in "
                f"model.heads ({sorted(model_heads.keys())})."
            )
        return primary_head

    if len(model_heads) == 1:
        return next(iter(model_heads.keys()))

    if allow_none:
        return None

    raise ValueError(
        f"{purpose} requires inference.head or model.primary_head when model.heads has "
        f"multiple entries ({sorted(model_heads.keys())})."
    )


def resolve_configured_output_head(
    cfg: Any,
    *,
    purpose: str = "output selection",
    allow_none: bool = True,
) -> Optional[str]:
    """Resolve the named output head requested by config, if any."""
    return resolve_output_head(
        cfg,
        requested_head=None,
        purpose=purpose,
        allow_none=allow_none,
    )


def resolve_output_channels(
    cfg: Any,
    *,
    requested_head: Optional[str] = None,
    purpose: str = "output selection",
    allow_ambiguous: bool = True,
) -> Optional[int]:
    """Resolve the number of channels produced by a selected output head."""
    model_heads = _get_model_heads(cfg)
    if model_heads:
        selected_head = resolve_output_head(
            cfg,
            requested_head=requested_head,
            purpose=purpose,
            allow_none=allow_ambiguous,
        )
        if selected_head is None:
            return None
        return int(_cfg_value(model_heads[selected_head], "out_channels", 0))

    model_cfg = _cfg_value(cfg, "model", None)
    out_channels = _cfg_value(model_cfg, "out_channels", None)
    if out_channels is None:
        return None
    return int(out_channels)


def resolve_configured_output_channels(
    cfg: Any,
    *,
    purpose: str = "output selection",
    allow_ambiguous: bool = True,
) -> Optional[int]:
    """Resolve the number of channels produced by the selected output head."""
    return resolve_output_channels(
        cfg,
        requested_head=None,
        purpose=purpose,
        allow_ambiguous=allow_ambiguous,
    )


def resolve_head_target_slice(cfg: Any, head_name: str):
    """Return the configured label target slice for a named head, if provided."""
    model_heads = _get_model_heads(cfg)
    if head_name not in model_heads:
        return None
    return _cfg_value(model_heads[head_name], "target_slice", None)


def unwrap_main_output(outputs: Any) -> Any:
    """Return the main output tensor or named-head mapping."""
    if isinstance(outputs, Mapping) and "output" in outputs:
        return outputs["output"]
    return outputs


def select_output_tensor(
    outputs: Any,
    *,
    requested_head: Optional[str] = None,
    primary_head: Optional[str] = None,
    purpose: str = "output selection",
) -> tuple[torch.Tensor, Optional[str]]:
    """Select one tensor from a tensor, deep-supervision dict, or named-head mapping."""
    normalized_output = unwrap_main_output(outputs)

    if isinstance(normalized_output, torch.Tensor):
        if requested_head is not None:
            raise ValueError(
                f"{purpose} requested head '{requested_head}', but the model output is a single "
                "tensor."
            )
        return normalized_output, None

    if not isinstance(normalized_output, Mapping):
        raise TypeError(
            f"{purpose} expected a tensor or mapping, got {type(normalized_output).__name__}."
        )
    if not normalized_output:
        raise ValueError(f"{purpose} received an empty output mapping.")

    resolved_head = requested_head
    if resolved_head is None:
        if primary_head is not None and primary_head in normalized_output:
            resolved_head = primary_head
        elif len(normalized_output) == 1:
            resolved_head = next(iter(normalized_output.keys()))
        else:
            raise ValueError(
                f"{purpose} requires an explicit head because available output heads are "
                f"{sorted(normalized_output.keys())}."
            )

    if resolved_head not in normalized_output:
        raise ValueError(
            f"{purpose} requested head '{resolved_head}', but available output heads are "
            f"{sorted(normalized_output.keys())}."
        )

    selected = normalized_output[resolved_head]
    if not isinstance(selected, torch.Tensor):
        raise TypeError(
            f"{purpose} requires head '{resolved_head}' to be a tensor, got "
            f"{type(selected).__name__}."
        )

    return selected, resolved_head
