"""
Stage resolution for Hydra configuration system.

Contains the logic for resolving default/stage-specific config sections
into runtime config with explicit precedence merging.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Callable, Dict

from omegaconf import DictConfig, ListConfig, OmegaConf

from ..schema import Config
from ..schema.root import MergeContext
from .dict_utils import as_plain_dict


def _collect_explicit_paths(yaml_node: Any, path: str = "") -> set[str]:
    """Collect all explicit key paths present in YAML."""
    paths: set[str] = set()

    if isinstance(yaml_node, ListConfig):
        for idx, item in enumerate(yaml_node):
            item_path = f"{path}[{idx}]" if path else f"[{idx}]"
            paths.update(_collect_explicit_paths(item, item_path))
        return paths

    if not isinstance(yaml_node, DictConfig):
        return paths

    for key in yaml_node.keys():
        child_path = f"{path}.{key}" if path else str(key)
        paths.add(child_path)
        paths.update(_collect_explicit_paths(yaml_node[key], child_path))

    return paths


# ---------------------------------------------------------------------------
# OmegaConf conversion helpers
# ---------------------------------------------------------------------------


def _to_omegaconf(value: Any) -> DictConfig:
    """Convert supported config objects into a mutable OmegaConf node."""
    if value is None:
        return OmegaConf.create({})
    if isinstance(value, DictConfig):
        return value
    if isinstance(value, dict):
        return OmegaConf.create(value)
    if is_dataclass(value):
        return OmegaConf.structured(value)
    return OmegaConf.create(value)


def _merge_dataclass(base_obj: Any, *overrides: Any) -> Any:
    """Merge one or more override objects into a dataclass instance."""
    merged = _to_omegaconf(base_obj)
    for override in overrides:
        if override is None:
            continue
        merged = OmegaConf.merge(merged, _to_omegaconf(override))
    return OmegaConf.to_object(merged)


# ---------------------------------------------------------------------------
# Stage resolution helpers
# ---------------------------------------------------------------------------


def _normalize_mode(mode: str) -> str:
    mode_normalized = mode.strip().lower()
    if mode_normalized == "tune-test":
        # tune-test first executes tuning, then test; prefer tune profile at setup stage.
        return "tune"
    return mode_normalized


def _as_path_set(value: Any) -> set[str]:
    if isinstance(value, set):
        return value
    if value is None:
        return set()
    return set(value)


def _get_merge_context(cfg: Config) -> MergeContext:
    """Get typed merge context from config, creating one if needed."""
    merge_context = getattr(cfg, "_merge_context", None)
    if not isinstance(merge_context, MergeContext):
        merge_context = MergeContext()
        setattr(cfg, "_merge_context", merge_context)
    return merge_context


def _build_explicit_path_checker(explicit_field_paths: set[str]) -> Callable[[str], bool]:
    # Pre-compute a prefix set so each lookup is O(1) instead of O(n).
    prefix_set: set[str] = set()
    for p in explicit_field_paths:
        prefix_set.add(p)
        parts = p.replace("[", ".").replace("]", "").split(".")
        for i in range(1, len(parts)):
            prefix_set.add(".".join(parts[:i]))

    def _has_explicit_path(base_path: str) -> bool:
        return base_path in prefix_set

    return _has_explicit_path


def _drop_none_values(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys with None values from a flat mapping."""
    return {k: v for k, v in mapping.items() if v is not None}


def _extract_explicit_patch(
    section_obj: Any,
    section_path: str,
    has_explicit_path: Callable[[str], bool],
) -> Dict[str, Any]:
    """Extract only explicitly provided YAML keys under a section path."""
    section_map = as_plain_dict(section_obj)
    if not section_map:
        return {}

    _MISSING = object()

    def _walk(node: Any, path: str) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for key, value in node.items():
                child_path = f"{path}.{key}" if path else str(key)
                child = _walk(value, child_path)
                if child is not _MISSING:
                    out[key] = child
            return out if out else _MISSING

        if isinstance(node, list):
            return node if has_explicit_path(path) else _MISSING

        return node if has_explicit_path(path) else _MISSING

    extracted = _walk(section_map, section_path)
    return extracted if isinstance(extracted, dict) else {}


def _has_section_patch(value: Any) -> bool:
    """Return whether a section override should participate in runtime merging."""
    return value != {}


def _resolve_stage_key(mode: str) -> str:
    if mode == "train":
        return "train"
    if mode == "test":
        return "test"
    return "tune"


_MODE_SECTIONS: Dict[str, tuple[str, ...]] = {
    "train": ("system", "model", "data", "optimization", "monitor"),
    "test": ("system", "model", "data", "inference", "decoding", "evaluation"),
    "tune": ("system", "model", "data", "inference", "decoding", "evaluation"),
}

_RUNTIME_SECTIONS = (
    "system",
    "model",
    "data",
    "optimization",
    "monitor",
    "inference",
    "decoding",
    "evaluation",
)


def _collect_default_overrides(
    cfg: Config,
    has_explicit_path: Callable[[str], bool],
) -> Dict[str, Any]:
    """Collect explicit default-stage patches for runtime merges."""
    default_stage = getattr(cfg, "default", None)
    if default_stage is None:
        return {name: {} for name in _RUNTIME_SECTIONS}

    return {
        "system": _extract_explicit_patch(
            getattr(default_stage, "system", None),
            "default.system",
            has_explicit_path,
        ),
        "model": _extract_explicit_patch(
            getattr(default_stage, "model", None),
            "default.model",
            has_explicit_path,
        ),
        "data": _extract_explicit_patch(
            getattr(default_stage, "data", None),
            "default.data",
            has_explicit_path,
        ),
        "optimization": _extract_explicit_patch(
            getattr(default_stage, "optimization", None),
            "default.optimization",
            has_explicit_path,
        ),
        "monitor": _extract_explicit_patch(
            getattr(default_stage, "monitor", None),
            "default.monitor",
            has_explicit_path,
        ),
        "inference": _extract_explicit_patch(
            getattr(default_stage, "inference", None),
            "default.inference",
            has_explicit_path,
        ),
        "decoding": _extract_explicit_patch(
            getattr(default_stage, "decoding", None),
            "default.decoding",
            has_explicit_path,
        ),
        "evaluation": _extract_explicit_patch(
            getattr(default_stage, "evaluation", None),
            "default.evaluation",
            has_explicit_path,
        ),
    }


def _collect_stage_overrides(
    cfg: Config,
    mode: str,
    has_explicit_path: Callable[[str], bool],
) -> Dict[str, Any]:
    """Collect mode-specific explicit patches for runtime merges."""
    stage_overrides: Dict[str, Any] = {name: {} for name in _RUNTIME_SECTIONS}

    stage_key = _resolve_stage_key(mode)
    stage_cfg = getattr(cfg, stage_key, None)
    if stage_cfg is None:
        return stage_overrides

    for section_name in _MODE_SECTIONS[stage_key]:
        section_obj = getattr(stage_cfg, section_name, None)
        section_path = f"{stage_key}.{section_name}"
        stage_overrides[section_name] = _extract_explicit_patch(
            section_obj,
            section_path,
            has_explicit_path,
        )

    return stage_overrides


def _merge_runtime_sections(
    cfg: Config,
    default_overrides: Dict[str, Any],
    stage_overrides: Dict[str, Any],
) -> None:
    """Merge runtime sections with precedence: schema-defaults < default < mode-specific."""
    for section_name in _RUNTIME_SECTIONS:
        default_section = default_overrides.get(section_name, {})
        stage_section = stage_overrides.get(section_name, {})
        if section_name == "system":
            default_section = _drop_none_values(default_section)
            stage_section = _drop_none_values(stage_section)
        if not _has_section_patch(default_section) and not _has_section_patch(stage_section):
            continue

        target_section = getattr(cfg, section_name)
        if not _has_section_patch(default_section):
            default_section = {}
        if not _has_section_patch(stage_section):
            stage_section = {}
        setattr(cfg, section_name, _merge_dataclass(target_section, default_section, stage_section))


def _is_selector_key(key: str) -> bool:
    return key == "profile" or key == "transform_profile" or key.endswith("_profile")


def _collect_unresolved_selector_paths(node: Any, path: str = "") -> list[str]:
    unresolved: list[str] = []

    if isinstance(node, (list, ListConfig)):
        for idx, item in enumerate(node):
            item_path = f"{path}[{idx}]" if path else f"[{idx}]"
            unresolved.extend(_collect_unresolved_selector_paths(item, item_path))
        return unresolved

    # Iterate key-value pairs from dataclass fields or dict entries
    if is_dataclass(node):
        items = (
            (f.name, getattr(node, f.name)) for f in fields(node) if not f.name.startswith("_")
        )
    elif isinstance(node, (dict, DictConfig)):
        items = ((str(k), node[k]) for k in node.keys())
    else:
        return unresolved

    for key_str, child_value in items:
        child_path = f"{path}.{key_str}" if path else key_str
        if _is_selector_key(key_str) and child_value not in (None, ""):
            unresolved.append(child_path)
        unresolved.extend(_collect_unresolved_selector_paths(child_value, child_path))

    return unresolved


def _assert_no_runtime_profile_selectors(cfg: Config) -> None:
    """Enforce single-phase profile resolution (all selectors resolved pre-conversion)."""
    unresolved = _collect_unresolved_selector_paths(cfg)
    if not unresolved:
        return

    displayed = ", ".join(sorted(unresolved)[:10])
    if len(unresolved) > 10:
        displayed += f", ... (+{len(unresolved) - 10} more)"

    raise ValueError(
        "Unresolved profile selectors found after YAML profile expansion. "
        "All profile selectors must be resolved pre-conversion. "
        f"Unresolved selector path(s): {displayed}"
    )


# ---------------------------------------------------------------------------
# Main stage resolver
# ---------------------------------------------------------------------------


def resolve_default_profiles(cfg: Config, mode: str = "train") -> Config:
    """Resolve default stage profiles into runtime sections."""
    if getattr(cfg, "default", None) is None:
        return cfg

    _assert_no_runtime_profile_selectors(cfg)

    mode_normalized = _normalize_mode(mode)
    merge_context = _get_merge_context(cfg)
    explicit_field_paths = _as_path_set(merge_context.explicit_field_paths)
    has_explicit_path = _build_explicit_path_checker(explicit_field_paths)

    default_overrides = _collect_default_overrides(cfg, has_explicit_path)
    stage_overrides = _collect_stage_overrides(
        cfg,
        mode_normalized,
        has_explicit_path,
    )
    _merge_runtime_sections(cfg, default_overrides, stage_overrides)

    return cfg
