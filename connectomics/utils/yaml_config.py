"""Small YAML inheritance loader for standalone workflow scripts.

The main PyTC configuration loader uses OmegaConf.  A few tutorial workflows
are intentionally lightweight standalone scripts, but still need the same
``_base_`` and ``${params.foo}`` conveniences for portable recipes.
"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]


_PARAMETER = re.compile(r"\$\{params\.([A-Za-z0-9_.-]+)\}")


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _lookup(params: Mapping[str, Any], path: str) -> Any:
    value: Any = params
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"undefined parameter 'params.{path}'")
        value = value[part]
    return value


def _interpolate(value: Any, params: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        return {key: _interpolate(item, params) for key, item in value.items()}
    if isinstance(value, list):
        return [_interpolate(item, params) for item in value]
    if not isinstance(value, str):
        return value

    full = _PARAMETER.fullmatch(value)
    if full:
        return _interpolate(_lookup(params, full.group(1)), params)
    return _PARAMETER.sub(lambda match: str(_lookup(params, match.group(1))), value)


def load_yaml_with_bases_and_params(path: Path) -> dict[str, Any]:
    """Load YAML recursively, then resolve tutorial-local ``params`` values.

    ``params`` is intentionally removed from the returned mapping so standalone
    workflow schemas remain strict about their actual input keys.
    """

    def load(current: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
        current = current.resolve()
        if current in stack:
            cycle = " -> ".join(str(item) for item in (*stack, current))
            raise ValueError(f"cyclic YAML inheritance: {cycle}")
        raw = yaml.safe_load(current.read_text()) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"YAML root must be a mapping: {current}")
        bases = raw.get("_base_", [])
        if isinstance(bases, (str, Path)):
            bases = [bases]
        if not isinstance(bases, list):
            raise ValueError(f"_base_ must be a path or list of paths: {current}")
        merged: dict[str, Any] = {}
        for base in bases:
            if not isinstance(base, (str, Path)):
                raise ValueError(f"_base_ entries must be paths: {current}")
            base_path = Path(base)
            if not base_path.is_absolute():
                base_path = current.parent / base_path
            merged = _merge(merged, load(base_path, (*stack, current)))
        return _merge(merged, {key: value for key, value in raw.items() if key != "_base_"})

    payload = load(path)
    params = payload.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping")
    resolved_params = _interpolate(params, params)
    if not isinstance(resolved_params, Mapping):  # defensive; params starts as a mapping
        raise ValueError("params must resolve to a mapping")
    resolved = _interpolate(payload, resolved_params)
    resolved.pop("params", None)
    return resolved
