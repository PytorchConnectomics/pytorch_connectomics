"""PyTorch checkpoint safe-global registration."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import is_dataclass
from typing import Iterable

SCHEMA_MODULES = (
    "connectomics.config.schema.system",
    "connectomics.config.schema.model",
    "connectomics.config.schema.model_monai",
    "connectomics.config.schema.model_mednext",
    "connectomics.config.schema.model_rsunet",
    "connectomics.config.schema.model_nnunet",
    "connectomics.config.schema.data",
    "connectomics.config.schema.optimization",
    "connectomics.config.schema.monitor",
    "connectomics.config.schema.inference",
    "connectomics.config.schema.decoding",
    "connectomics.config.schema.evaluation",
    "connectomics.config.schema.stages",
    "connectomics.config.schema.root",
)


def iter_config_dataclasses() -> list[type]:
    """Return all dataclass config classes that may appear in checkpoints."""
    safe_dataclasses: list[type] = []
    for module_name in SCHEMA_MODULES:
        module = importlib.import_module(module_name)
        safe_dataclasses.extend(
            obj for obj in module.__dict__.values() if inspect.isclass(obj) and is_dataclass(obj)
        )
    return list(dict.fromkeys(safe_dataclasses))


def register_torch_safe_globals(extra_classes: Iterable[type] = ()) -> None:
    """Register config dataclasses for torch 2.6+ weights-only checkpoint loading."""
    try:
        import torch

        if not (
            hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals")
        ):
            return
        torch.serialization.add_safe_globals(
            list(dict.fromkeys([*iter_config_dataclasses(), *extra_classes]))
        )
    except Exception:
        pass


__all__ = ["iter_config_dataclasses", "register_torch_safe_globals"]
