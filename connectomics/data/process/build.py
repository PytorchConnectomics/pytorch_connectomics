"""Factory for label transform pipelines in PyTorch Connectomics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from monai.transforms import Compose, MapTransform

from ...config.pipeline.dict_utils import to_plain
from .transforms import MultiTaskLabelTransformd


def _resolve_dtype(dtype_value: Optional[Any]) -> Optional[torch.dtype]:
    """Convert config-provided dtype spec into a torch dtype."""
    if dtype_value is None:
        return None
    if isinstance(dtype_value, torch.dtype):
        return dtype_value
    if isinstance(dtype_value, str):
        attr = dtype_value.lower().strip()
        if hasattr(torch, attr):
            resolved = getattr(torch, attr)
            if isinstance(resolved, torch.dtype):
                return resolved
    raise ValueError(f"Unsupported torch dtype specification: {dtype_value!r}")


def create_label_transform_pipeline(cfg: Any = None, **kwargs: Any) -> MapTransform:
    """Create a label transformation pipeline from config.

    Primary entry point for label processing. Uses MultiTaskLabelTransformd
    to generate multiple target types from instance segmentation labels.

    Args:
        cfg: Config object (OmegaConf DictConfig, plain dict) with fields:
            - keys: Input key(s) (default: ["label"])
            - targets: List of task specs (null/[] for identity)
            - stack_outputs, retain_original, output_key_format, output_dtype,
              allow_missing_keys: See MultiTaskLabelTransformd.

    Returns:
        MONAI transform (identity Compose if targets is null/empty,
        single MultiTaskLabelTransformd otherwise).
    """
    if cfg is None:
        cfg = {}
    cfg = to_plain(cfg)
    if not isinstance(cfg, dict):
        raise TypeError(
            "Expected OmegaConf DictConfig, structured dataclass config, or plain dict "
            f"for cfg, got {type(cfg).__name__}"
        )
    cfg = {**cfg, **kwargs}

    # Keys configuration
    keys_raw = cfg.get("keys", None)
    if keys_raw is None or callable(keys_raw):
        keys_option = [cfg.get("input_key", "label")]
    elif isinstance(keys_raw, str):
        keys_option = [keys_raw]
    else:
        keys_option = list(keys_raw)

    stack_outputs = cfg.get("stack_outputs", True)
    retain_original = cfg.get("retain_original", False)
    output_key_format = cfg.get("output_key_format", "{key}_{task}")
    allow_missing_keys = cfg.get("allow_missing_keys", False)
    output_dtype_setting = cfg.get("output_dtype", "float32")
    output_dtype = _resolve_dtype(output_dtype_setting) if output_dtype_setting else None

    target_cfg = cfg.get("targets", None)

    # Identity transform for null/empty targets
    if target_cfg is None or (isinstance(target_cfg, (list, tuple)) and len(target_cfg) == 0):
        return Compose([])

    converted = to_plain(target_cfg)
    raw_tasks = list(converted) if isinstance(converted, (list, tuple)) else [converted]

    # Normalize task entries
    tasks: List[Any] = []
    for entry in raw_tasks:
        if isinstance(entry, str):
            tasks.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("name") or entry.get("task") or entry.get("type")
            if name is None:
                raise ValueError(f"Task entry {entry} missing 'name'/'task'/'type'.")
            processed: Dict[str, Any] = {"name": name, "kwargs": dict(entry.get("kwargs", {}))}
            if "output_key" in entry:
                processed["output_key"] = entry["output_key"]
            tasks.append(processed)
        else:
            raise TypeError(f"Unsupported task specification: {entry!r}")

    if not tasks:
        raise ValueError("At least one task must be specified in 'targets'.")

    return MultiTaskLabelTransformd(
        keys=list(keys_option),
        tasks=tasks,
        stack_outputs=stack_outputs,
        output_dtype=output_dtype,
        retain_original=retain_original,
        output_key_format=output_key_format,
        allow_missing_keys=allow_missing_keys,
    )


__all__ = [
    "create_label_transform_pipeline",
]
