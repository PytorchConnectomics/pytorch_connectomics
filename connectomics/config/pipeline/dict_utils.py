"""Shared dictionary-conversion helpers for config modules."""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Dict

from omegaconf import DictConfig, ListConfig, OmegaConf


def to_plain(obj: Any) -> Any:
    """Convert OmegaConf containers and dataclasses to native Python types.

    Returns the native Python equivalent (dict, list, or the original object).
    """
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    if is_dataclass(obj):
        return OmegaConf.to_container(OmegaConf.structured(obj), resolve=True)
    return obj


def as_plain_dict(value: Any) -> Dict[str, Any]:
    """Convert config-like values into a plain Python dict."""
    if value is None or isinstance(value, str):
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, DictConfig):
        container = OmegaConf.to_container(value, resolve=True)
        return container if isinstance(container, dict) else {}

    try:
        node = OmegaConf.structured(value) if is_dataclass(value) else OmegaConf.create(value)
        container = OmegaConf.to_container(node, resolve=True)
    except Exception:
        return {}
    return container if isinstance(container, dict) else {}


def cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a key from dict-like or attribute-like config objects.

    Handles plain dicts, OmegaConf DictConfig, and dataclass-style objects.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


__all__ = ["to_plain", "as_plain_dict", "cfg_get"]
