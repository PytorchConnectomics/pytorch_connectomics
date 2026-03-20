"""Factory for label transform pipelines in PyTorch Connectomics."""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Dict, List, Optional

import torch
from monai.transforms import Compose, MapTransform
from omegaconf import DictConfig, ListConfig, OmegaConf

from .affinity import parse_affinity_offsets
from .transforms import MultiTaskLabelTransformd


def _to_plain(obj: Any) -> Any:
    """Convert OmegaConf containers and dataclasses to native Python types."""
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    if is_dataclass(obj):
        return OmegaConf.to_container(OmegaConf.structured(obj), resolve=True)
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


def _normalize_target_tasks(cfg: Dict[str, Any]) -> List[Any]:
    """Normalize configured label targets into task specs."""
    target_cfg = cfg.get("targets", None)
    if target_cfg is None or (isinstance(target_cfg, (list, tuple)) and len(target_cfg) == 0):
        return []

    converted = _to_plain(target_cfg)
    raw_tasks = list(converted) if isinstance(converted, (list, tuple)) else [converted]

    inheritable_kwargs = {}
    for inheritable_key in ("resolution",):
        inheritable_value = cfg.get(inheritable_key)
        if inheritable_value is not None:
            inheritable_kwargs[inheritable_key] = inheritable_value

    tasks: List[Any] = []
    for entry in raw_tasks:
        if isinstance(entry, str):
            tasks.append(entry)
            continue

        if not isinstance(entry, dict):
            raise TypeError(f"Unsupported task specification: {entry!r}")

        name = entry.get("name") or entry.get("task") or entry.get("type")
        if name is None:
            raise ValueError(f"Task entry {entry} missing name/task/type.")

        task_kwargs = dict(entry.get("kwargs", {}))
        task_defaults = MultiTaskLabelTransformd._TASK_DEFAULTS.get(name, {})
        for inheritable_key, inheritable_value in inheritable_kwargs.items():
            if inheritable_key in task_defaults and inheritable_key not in task_kwargs:
                task_kwargs[inheritable_key] = inheritable_value

        processed: Dict[str, Any] = {"name": name, "kwargs": task_kwargs}
        if "output_key" in entry:
            processed["output_key"] = entry["output_key"]
        tasks.append(processed)

    return tasks


def _task_output_channels(task: Any) -> int:
    """Return how many stacked channels a normalized task contributes."""
    if isinstance(task, str):
        name = task
        task_kwargs: Dict[str, Any] = {}
    elif isinstance(task, dict):
        name = task.get("name") or task.get("task") or task.get("type")
        task_kwargs = dict(task.get("kwargs", {}))
    else:
        raise TypeError(f"Unsupported task specification: {task!r}")

    if name is None:
        raise ValueError(f"Task entry {task!r} missing name/task/type.")

    defaults = MultiTaskLabelTransformd._TASK_DEFAULTS.get(name, {})
    resolved_kwargs = {**defaults, **task_kwargs}

    if name == "affinity":
        return len(parse_affinity_offsets(resolved_kwargs.get("offsets", [])))
    if name == "polarity":
        return 1 if bool(resolved_kwargs.get("exclusive", False)) else 3
    if name == "flow":
        return 2

    return 1


def count_stacked_label_transform_channels(cfg: Any = None, **kwargs: Any) -> Optional[int]:
    """Return the stacked output channel count for label_transform.targets."""
    if cfg is None:
        cfg = {}
    cfg = _to_plain(cfg)
    if not isinstance(cfg, dict):
        raise TypeError(
            "Expected OmegaConf DictConfig, structured dataclass config, or plain dict "
            f"for cfg, got {type(cfg).__name__}"
        )
    cfg = {**cfg, **kwargs}

    if not cfg.get("stack_outputs", True):
        return None

    tasks = _normalize_target_tasks(cfg)
    if not tasks:
        return None

    return sum(_task_output_channels(task) for task in tasks)


def create_label_transform_pipeline(cfg: Any = None, **kwargs: Any) -> MapTransform:
    """Create a label transformation pipeline from config."""
    if cfg is None:
        cfg = {}
    cfg = _to_plain(cfg)
    if not isinstance(cfg, dict):
        raise TypeError(
            "Expected OmegaConf DictConfig, structured dataclass config, or plain dict "
            f"for cfg, got {type(cfg).__name__}"
        )
    cfg = {**cfg, **kwargs}

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

    tasks = _normalize_target_tasks(cfg)
    if not tasks:
        return Compose([])

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
    "count_stacked_label_transform_channels",
    "create_label_transform_pipeline",
]
