"""Factory for label transform pipelines in PyTorch Connectomics."""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import torch
from monai.transforms import Compose
from omegaconf import DictConfig, ListConfig, OmegaConf

from .monai_transforms import MultiTaskLabelTransformd


def _to_plain(obj: Any) -> Any:
    """Convert OmegaConf containers to native Python types."""
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


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


def create_label_transform_pipeline(cfg: Any = None, **kwargs: Any) -> Compose:
    """Create a label transformation pipeline from config.

    Primary entry point for label processing. Uses MultiTaskLabelTransformd
    to generate multiple target types from instance segmentation labels.

    Args:
        cfg: Config object (OmegaConf DictConfig or plain dict) with fields:
            - keys: Input key(s) (default: ["label"])
            - targets: List of task specs (null/[] for identity)
            - stack_outputs, retain_original, output_key_format, output_dtype,
              allow_missing_keys: See MultiTaskLabelTransformd.

    Returns:
        MONAI Compose pipeline (identity if targets is null/empty).
    """
    if cfg is None:
        cfg = {}
    if isinstance(cfg, (DictConfig, ListConfig)):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, dict):
        cfg = {**cfg, **kwargs}
    else:
        # Attribute-style config object
        for key, value in kwargs.items():
            setattr(cfg, key, value)

    def _get(key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    # Keys configuration
    keys_raw = _get("keys", None)
    if keys_raw is None or callable(keys_raw):
        keys_option = [_get("input_key", "label")]
    elif isinstance(keys_raw, str):
        keys_option = [keys_raw]
    else:
        keys_option = list(keys_raw)

    stack_outputs = _get("stack_outputs", True)
    retain_original = _get("retain_original", False)
    output_key_format = _get("output_key_format", "{key}_{task}")
    allow_missing_keys = _get("allow_missing_keys", False)
    output_dtype_setting = _get("output_dtype", "float32")
    output_dtype = _resolve_dtype(output_dtype_setting) if output_dtype_setting else None

    target_cfg = _get("targets", None)

    # Identity transform for null/empty targets
    if target_cfg is None or (isinstance(target_cfg, (list, tuple)) and len(target_cfg) == 0):
        return Compose([])

    converted = _to_plain(target_cfg)
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

    transform = MultiTaskLabelTransformd(
        keys=list(keys_option),
        tasks=tasks,
        stack_outputs=stack_outputs,
        output_dtype=output_dtype,
        retain_original=retain_original,
        output_key_format=output_key_format,
        allow_missing_keys=allow_missing_keys,
    )

    return Compose([transform])


__all__ = [
    "create_label_transform_pipeline",
]
