"""
Modern MONAI Compose factory functions for PyTorch Connectomics.

This module provides clean, modern factory functions to create MONAI Compose pipelines
for connectomics workflows using configuration objects.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from types import SimpleNamespace
import torch
from monai.transforms import Compose

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
except ImportError:  # pragma: no cover - OmegaConf is expected but keep fallback
    DictConfig = tuple()  # type: ignore
    ListConfig = tuple()  # type: ignore
    OmegaConf = None  # type: ignore


def _to_plain(obj: Any) -> Any:
    """Recursively convert OmegaConf containers to native Python types."""
    if OmegaConf is not None and isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, DictConfig):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, ListConfig):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
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


def _coerce_config(cfg: Any = None, extra_kwargs: Optional[Dict[str, Any]] = None) -> Any:
    """Return an object with attribute-style access for transform configuration."""
    if isinstance(cfg, dict):
        namespace = SimpleNamespace(**cfg)
    elif cfg is None:
        namespace = SimpleNamespace()
    else:
        namespace = cfg
    if extra_kwargs:
        for key, value in extra_kwargs.items():
            setattr(namespace, key, value)
    return namespace


from .monai_transforms import MultiTaskLabelTransformd


def create_label_transform_pipeline(cfg: Any = None, **kwargs: Any) -> Compose:
    """Create a label transformation pipeline from config.

    This is the primary entry point for all label processing in PyTorch Connectomics.
    It uses the multi-task transform system to generate multiple target types from
    a single instance segmentation label.

    Args:
        cfg: Configuration object with fields:
            - keys: Key(s) for input segmentation (default: ["label"])
            - targets: List of task specifications (required, or null for no transform)
            - stack_outputs: Whether to stack outputs (default: True)
            - retain_original: Keep original label (default: False)
            - output_key_format: Format for output keys (default: "{key}_{task}")
            - output_dtype: Output data type (default: "float32")
            - allow_missing_keys: Allow missing keys (default: False)

        Note:
            Setting targets to null or [] returns an identity transform (no-op)
            that passes labels through unchanged. Useful for pre-processed labels.

            All task-specific parameters (segment_id, boundary thickness, etc.)
            should be specified directly in the task kwargs, not at the top level.

    Returns:
        MONAI Compose pipeline for label transformations

    Output Shape:
        All outputs are in channel-first format [C, D, H, W]:
        - Single task: [1, D, H, W]
        - Multi-task (stacked): [C, D, H, W] where C = sum of task channels
        - Multi-task (non-stacked): Multiple keys, each [C_i, D, H, W]

    Example configs:
        # Multi-task transformation
        label_transform:
          targets:
            - name: binary                    # Produces 1 channel
            - name: instance_boundary         # Produces 1 channel
              kwargs:
                thickness: 1
                do_bg_edges: false
            - name: instance_edt              # Produces 1 channel
              kwargs:
                mode: "2d"
                quantize: false
          # Output shape: [3, D, H, W] for stacked mode

        # No transformation (identity - use raw labels)
        label_transform:
          targets: null                       # or targets: []
          # Output: Original labels unchanged
    """
    cfg = _coerce_config(cfg, kwargs)

    # Keys configuration
    # Note: Must check if 'keys' exists in config to avoid getting dict.keys() method
    if hasattr(cfg, "__dict__") and "keys" in cfg.__dict__:
        keys_attr = cfg.keys
    elif hasattr(cfg, "__contains__") and "keys" in cfg:
        keys_attr = cfg["keys"] if isinstance(cfg, dict) else getattr(cfg, "keys")
    else:
        keys_attr = None

    if keys_attr is None or callable(keys_attr):  # Protect against dict.keys() method
        keys_option = [
            (
                getattr(cfg, "input_key", None) or cfg.get("input_key", "label")
                if isinstance(cfg, dict)
                else "label"
            )
        ]
    elif isinstance(keys_attr, str):
        keys_option = [keys_attr]
    else:
        keys_option = list(keys_attr)

    # Transform configuration
    stack_outputs = getattr(cfg, "stack_outputs", True)
    retain_original = getattr(cfg, "retain_original", False)
    output_key_format = getattr(cfg, "output_key_format", "{key}_{task}")
    allow_missing_keys = getattr(cfg, "allow_missing_keys", False)
    output_dtype_setting = getattr(cfg, "output_dtype", "float32")
    output_dtype = (
        _resolve_dtype(output_dtype_setting) if output_dtype_setting is not None else None
    )

    # Get targets configuration
    target_cfg = getattr(cfg, "targets", None)

    # Allow null/empty targets for no transformation (identity transform)
    if target_cfg is None or (isinstance(target_cfg, (list, tuple)) and len(target_cfg) == 0):
        # Return identity transform (no-op pipeline)
        return Compose([])

    # Convert targets to plain Python types
    converted = _to_plain(target_cfg)
    if isinstance(converted, (list, tuple)):
        raw_tasks = list(converted)
    else:
        raw_tasks = [converted]

    # Process tasks
    tasks: List[Any] = []
    for entry in raw_tasks:
        if isinstance(entry, str):
            tasks.append(entry)
            continue
        if isinstance(entry, dict):
            entry_dict = dict(entry)
            name = entry_dict.get("name") or entry_dict.get("task") or entry_dict.get("type")
            if name is None:
                raise ValueError(f"Task entry {entry_dict} missing 'name'/'task'/'type'.")
            kwargs_dict = dict(entry_dict.get("kwargs", {}))
            processed: Dict[str, Any] = {
                "name": name,
                "kwargs": kwargs_dict,
            }
            if "output_key" in entry_dict:
                processed["output_key"] = entry_dict["output_key"]
            tasks.append(processed)
            continue
        raise TypeError(f"Unsupported task specification: {entry!r}")

    if not tasks:
        raise ValueError("At least one task must be specified in 'targets'.")

    # Create the multi-task transform
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
