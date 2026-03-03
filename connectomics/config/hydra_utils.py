"""
Utility functions for Hydra configuration system.

Provides helpers for loading, saving, validating, and manipulating configs.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, ListConfig, OmegaConf

from .hydra_config import Config
from .model_arch import resolve_arch_profile_model_patch


def _normalize_base_paths(base_field: Any, config_path: Path) -> List[Path]:
    """Normalize `_base_` field to an ordered list of absolute paths."""
    if base_field is None:
        return []

    if isinstance(base_field, (str, Path)):
        base_entries = [str(base_field)]
    elif isinstance(base_field, (list, tuple, ListConfig)):
        base_entries = [str(item) for item in base_field]
    else:
        raise TypeError(
            f"Invalid _base_ value in {config_path}: expected string or list, "
            f"got {type(base_field)}"
        )

    resolved_paths: List[Path] = []
    for base_entry in base_entries:
        base_path = Path(base_entry)
        if not base_path.is_absolute():
            base_path = (config_path.parent / base_path).resolve()
        if not base_path.exists():
            raise FileNotFoundError(
                f"Base config not found: {base_entry} (resolved to {base_path}) in {config_path}"
            )
        resolved_paths.append(base_path)

    return resolved_paths


def _load_config_with_bases(config_path: Path, loading_stack: Tuple[Path, ...] = ()) -> DictConfig:
    """Load YAML config recursively with `_base_` inheritance."""
    config_path = config_path.resolve()
    if config_path in loading_stack:
        cycle = " -> ".join(str(p) for p in (*loading_stack, config_path))
        raise ValueError(f"Detected cyclic _base_ config inheritance: {cycle}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    yaml_conf = OmegaConf.load(config_path)
    if yaml_conf is None:
        yaml_conf = OmegaConf.create({})
    if not isinstance(yaml_conf, DictConfig):
        raise TypeError(
            f"Config root must be a mapping in {config_path}, got {type(yaml_conf)} instead"
        )

    base_field = yaml_conf.get("_base_", None)
    if "_base_" in yaml_conf:
        del yaml_conf["_base_"]

    merged_base = OmegaConf.create({})
    for base_path in _normalize_base_paths(base_field, config_path):
        base_conf = _load_config_with_bases(base_path, (*loading_stack, config_path))
        merged_base = OmegaConf.merge(merged_base, base_conf)

    return OmegaConf.merge(merged_base, yaml_conf)


class YamlProfileApplier:
    """Base strategy for applying a profile family to a merged YAML config."""

    cleanup_keys: Tuple[str, ...] = ()

    def apply(self, yaml_conf: DictConfig) -> None:
        raise NotImplementedError


class PathMoveApplier(YamlProfileApplier):
    """Move/merge a top-level key into a target config path."""

    def __init__(
        self,
        source_key: str,
        target_path: str,
        merge_into_existing: bool = True,
    ) -> None:
        self.source_key = source_key
        self.target_path = target_path
        self.merge_into_existing = merge_into_existing
        self.cleanup_keys = (source_key,)

    def apply(self, yaml_conf: DictConfig) -> None:
        value = yaml_conf.get(self.source_key, None)
        if value is None:
            return

        if self.merge_into_existing:
            _merge_into_path(yaml_conf, self.target_path, value)
        else:
            OmegaConf.update(yaml_conf, self.target_path, value, force_add=True)


def _merge_into_path(yaml_conf: DictConfig, target_path: str, profile_value: Any) -> None:
    """Merge profile payload into target path, keeping explicit config overrides."""
    existing = OmegaConf.select(yaml_conf, target_path)
    if existing is None:
        OmegaConf.update(yaml_conf, target_path, profile_value, force_add=True)
        return

    merged = OmegaConf.merge(profile_value, existing)
    OmegaConf.update(yaml_conf, target_path, merged, force_add=True)


def _auto_enable_sections(default_node: Any, yaml_node: Any) -> None:
    """Auto-set `enabled: true` when a section has user-provided keys.

    Rule:
    - if a schema node has an `enabled` field whose default is False or None,
    - and YAML provides any sibling key under that section without explicitly
      setting `enabled`,
    - then set `enabled: true`.
    """
    if isinstance(yaml_node, ListConfig):
        default_item = None
        if isinstance(default_node, ListConfig) and len(default_node) > 0:
            default_item = default_node[0]
        for item in yaml_node:
            _auto_enable_sections(default_item, item)
        return

    if not isinstance(yaml_node, DictConfig):
        return

    for key in list(yaml_node.keys()):
        child_default = None
        if isinstance(default_node, DictConfig) and key in default_node:
            child_default = default_node[key]
        _auto_enable_sections(child_default, yaml_node[key])

    if not isinstance(default_node, DictConfig):
        return

    if "enabled" in default_node and "enabled" not in yaml_node:
        has_other_keys = any(key != "enabled" for key in yaml_node.keys())
        if not has_other_keys:
            return

        default_enabled = default_node.get("enabled")
        if default_enabled is False or default_enabled is None:
            yaml_node["enabled"] = True


def _collect_explicit_enabled_paths(yaml_node: Any, path: str = "") -> set[str]:
    """Collect paths where `enabled` is explicitly provided by YAML."""
    paths: set[str] = set()

    if isinstance(yaml_node, ListConfig):
        for idx, item in enumerate(yaml_node):
            item_path = f"{path}[{idx}]" if path else f"[{idx}]"
            paths.update(_collect_explicit_enabled_paths(item, item_path))
        return paths

    if not isinstance(yaml_node, DictConfig):
        return paths

    if "enabled" in yaml_node:
        enabled_path = f"{path}.enabled" if path else "enabled"
        paths.add(enabled_path)

    for key in yaml_node.keys():
        child_path = f"{path}.{key}" if path else str(key)
        paths.update(_collect_explicit_enabled_paths(yaml_node[key], child_path))

    return paths


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


def _auto_enable_changed_sections(
    default_node: Any,
    runtime_node: Any,
    explicit_enabled_paths: set[str],
    explicit_field_paths: set[str],
    path: str = "",
) -> None:
    """Auto-enable sections only when values differ from defaults and not explicitly set."""

    def _to_plain(node: Any) -> Any:
        if isinstance(node, (DictConfig, ListConfig)):
            return OmegaConf.to_container(node, resolve=True)
        return node

    if isinstance(runtime_node, ListConfig):
        for idx, item in enumerate(runtime_node):
            default_item = None
            if isinstance(default_node, ListConfig):
                if idx < len(default_node):
                    default_item = default_node[idx]
                elif len(default_node) > 0:
                    default_item = default_node[0]
            item_path = f"{path}[{idx}]" if path else f"[{idx}]"
            _auto_enable_changed_sections(
                default_item,
                item,
                explicit_enabled_paths,
                explicit_field_paths,
                item_path,
            )
        return

    if not isinstance(runtime_node, DictConfig):
        return

    for key in runtime_node.keys():
        child_default = None
        if isinstance(default_node, DictConfig) and key in default_node:
            child_default = default_node[key]
        child_path = f"{path}.{key}" if path else str(key)
        _auto_enable_changed_sections(
            child_default,
            runtime_node[key],
            explicit_enabled_paths,
            explicit_field_paths,
            child_path,
        )

    if "enabled" not in runtime_node:
        return

    current_enabled = runtime_node.get("enabled")
    if current_enabled not in (False, None):
        return

    enabled_path = f"{path}.enabled" if path else "enabled"
    if enabled_path in explicit_enabled_paths:
        return

    has_explicit_sibling = False
    for key in runtime_node.keys():
        if key == "enabled":
            continue
        sibling_path = f"{path}.{key}" if path else str(key)
        if sibling_path in explicit_field_paths:
            has_explicit_sibling = True
            break

    has_changed_sibling = False
    for key in runtime_node.keys():
        if key == "enabled":
            continue

        runtime_child = runtime_node[key]
        default_child = None
        if isinstance(default_node, DictConfig) and key in default_node:
            default_child = default_node[key]

        if _to_plain(runtime_child) != _to_plain(default_child):
            has_changed_sibling = True
            break

    if has_explicit_sibling or has_changed_sibling:
        runtime_node["enabled"] = True


class MappingProfileApplier(YamlProfileApplier):
    """Apply selected profile mapping into one or more target config paths."""

    def __init__(
        self,
        selector_paths: Union[str, List[str], Tuple[str, ...]],
        profiles_key: str,
        section_to_target: Dict[str, str],
        strict_profile_sections: bool = False,
        cleanup_keys: Optional[Tuple[str, ...]] = None,
    ) -> None:
        if isinstance(selector_paths, str):
            selector_paths = [selector_paths]
        self.selector_paths = list(selector_paths)
        self.profiles_key = profiles_key
        self.section_to_target = section_to_target
        self.strict_profile_sections = strict_profile_sections
        self.cleanup_keys = cleanup_keys if cleanup_keys is not None else (profiles_key,)

    def apply(self, yaml_conf: DictConfig) -> None:
        selected = None
        selected_path = None
        for selector_path in self.selector_paths:
            selected = OmegaConf.select(yaml_conf, selector_path)
            if selected is not None:
                selected_path = selector_path
                break

        profiles = yaml_conf.get(self.profiles_key, None)
        if selected is None or profiles is None:
            return

        if selected not in profiles:
            available = ", ".join(sorted(str(k) for k in profiles.keys()))
            raise ValueError(
                f"Unknown selector '{selected}' at '{selected_path}'. "
                f"Available profiles: [{available}]"
            )

        selected_profile = profiles[selected]
        if not isinstance(selected_profile, (dict, DictConfig)):
            raise ValueError(
                f"Profile '{selected}' in '{self.profiles_key}' must be a mapping, "
                f"got {type(selected_profile)}"
            )

        if self.strict_profile_sections:
            allowed_sections = set(self.section_to_target.keys())
            selected_sections = {str(k) for k in selected_profile.keys()}
            unsupported_sections = sorted(selected_sections - allowed_sections)
            if unsupported_sections:
                allowed_display = ", ".join(sorted(allowed_sections))
                raise ValueError(
                    f"Profile '{selected}' in '{self.profiles_key}' has unsupported sections: "
                    f"{unsupported_sections}. Allowed sections: [{allowed_display}]"
                )

        for profile_section, target_path in self.section_to_target.items():
            if profile_section in selected_profile:
                _merge_into_path(yaml_conf, target_path, selected_profile[profile_section])


class ArchProfileApplier(YamlProfileApplier):
    """Resolve selected arch profile via OOP registry and merge into `model`."""

    def __init__(
        self,
        selector_paths: Union[str, List[str], Tuple[str, ...]],
        profiles_key: str = "arch_profiles",
        target_path: str = "model",
        cleanup_keys: Optional[Tuple[str, ...]] = None,
    ) -> None:
        if isinstance(selector_paths, str):
            selector_paths = [selector_paths]
        self.selector_paths = list(selector_paths)
        self.profiles_key = profiles_key
        self.target_path = target_path
        self.cleanup_keys = cleanup_keys if cleanup_keys is not None else ("arch_profile", profiles_key)

    def apply(self, yaml_conf: DictConfig) -> None:
        selected = None
        selected_path = None
        for selector_path in self.selector_paths:
            selected = OmegaConf.select(yaml_conf, selector_path)
            if selected is not None:
                selected_path = selector_path
                break

        profiles = yaml_conf.get(self.profiles_key, None)
        if selected is None or profiles is None:
            return

        if selected not in profiles:
            available = ", ".join(sorted(str(k) for k in profiles.keys()))
            raise ValueError(
                f"Unknown selector '{selected}' at '{selected_path}'. "
                f"Available profiles: [{available}]"
            )

        model_patch = resolve_arch_profile_model_patch(
            profile_name=str(selected),
            profile_value=profiles[selected],
        )
        _merge_into_path(yaml_conf, self.target_path, model_patch)


class ReferenceProfileApplier(YamlProfileApplier):
    """Resolve `${profiles_key.<name>}` references for specific target paths."""

    def __init__(self, profiles_key: str, target_paths: List[str]) -> None:
        self.profiles_key = profiles_key
        self.target_paths = target_paths
        self.cleanup_keys = (profiles_key,)

    def apply(self, yaml_conf: DictConfig) -> None:
        profiles = yaml_conf.get(self.profiles_key, None)
        if profiles is None:
            return

        pattern = re.compile(rf"\$\{{{re.escape(self.profiles_key)}\.([A-Za-z0-9_\-]+)\}}")

        for target_path in self.target_paths:
            value = OmegaConf.select(yaml_conf, target_path)
            if value is None:
                continue

            # Inline already-resolved interpolation payloads before helper cleanup.
            try:
                resolved_value = OmegaConf.to_container(value, resolve=True)
                OmegaConf.update(yaml_conf, target_path, resolved_value, force_add=True)
            except Exception:
                pass

            value = OmegaConf.select(yaml_conf, target_path)
            if not isinstance(value, str):
                continue

            match = pattern.fullmatch(value)
            if not match:
                continue

            profile_name = match.group(1)
            if profile_name not in profiles:
                available = ", ".join(sorted(str(k) for k in profiles.keys()))
                raise ValueError(
                    f"Unknown profile '{profile_name}' in {self.profiles_key}. "
                    f"Available profiles: [{available}]"
                )

            OmegaConf.update(yaml_conf, target_path, profiles[profile_name], force_add=True)


class ListProfileReferenceApplier(YamlProfileApplier):
    """Resolve list entries like `- profile: name` using profile registries."""

    def __init__(
        self,
        profiles_key: str,
        target_paths: List[str],
        profile_key: str = "profile",
    ) -> None:
        self.profiles_key = profiles_key
        self.target_paths = target_paths
        self.profile_key = profile_key
        self.cleanup_keys = (profiles_key,)

    def apply(self, yaml_conf: DictConfig) -> None:
        profiles = yaml_conf.get(self.profiles_key, None)
        if profiles is None:
            return

        for target_path in self.target_paths:
            value = OmegaConf.select(yaml_conf, target_path)
            if not isinstance(value, ListConfig):
                continue

            expanded_items: list[Any] = []
            changed = False

            for item in value:
                if not isinstance(item, (DictConfig, dict)) or self.profile_key not in item:
                    expanded_items.append(item)
                    continue

                profile_name = item[self.profile_key]
                if profile_name not in profiles:
                    available = ", ".join(sorted(str(k) for k in profiles.keys()))
                    raise ValueError(
                        f"Unknown profile '{profile_name}' in {self.profiles_key}. "
                        f"Available profiles: [{available}]"
                    )

                profile_payload = profiles[profile_name]
                overrides = {k: v for k, v in item.items() if k != self.profile_key}

                if not overrides:
                    # Pure profile reference: expand entire profile list inline
                    if not isinstance(profile_payload, (ListConfig, list)):
                        raise ValueError(
                            f"Profile '{profile_name}' in {self.profiles_key} must resolve to a list "
                            f"for target '{target_path}', got {type(profile_payload)}"
                        )
                    expanded_items.extend(list(profile_payload))
                else:
                    # Profile + overrides: use first profile item as base, merge overrides on top
                    if isinstance(profile_payload, (ListConfig, list)):
                        if not profile_payload:
                            raise ValueError(
                                f"Profile '{profile_name}' in {self.profiles_key} is empty."
                            )
                        base = OmegaConf.create(
                            OmegaConf.to_container(profile_payload[0], resolve=False)
                        )
                    elif isinstance(profile_payload, (DictConfig, dict)):
                        base = OmegaConf.create(
                            OmegaConf.to_container(profile_payload, resolve=False)
                        )
                    else:
                        raise ValueError(
                            f"Profile '{profile_name}' in {self.profiles_key} must be a list or dict, "
                            f"got {type(profile_payload)}"
                        )
                    merged = OmegaConf.merge(base, OmegaConf.create(overrides))
                    expanded_items.append(merged)

                changed = True

            if changed:
                OmegaConf.update(yaml_conf, target_path, expanded_items, force_add=True)


class ValueProfileApplier(YamlProfileApplier):
    """Apply a selected profile value directly into a target path."""

    def __init__(
        self,
        selector_paths: Union[str, List[str], Tuple[str, ...]],
        profiles_key: str,
        target_path: str,
        merge_into_existing: bool = True,
        cleanup_keys: Optional[Tuple[str, ...]] = None,
    ) -> None:
        if isinstance(selector_paths, str):
            selector_paths = [selector_paths]
        self.selector_paths = list(selector_paths)
        self.profiles_key = profiles_key
        self.target_path = target_path
        self.merge_into_existing = merge_into_existing
        self.cleanup_keys = cleanup_keys if cleanup_keys is not None else (profiles_key,)

    def apply(self, yaml_conf: DictConfig) -> None:
        selected = None
        selected_path = None
        for selector_path in self.selector_paths:
            selected = OmegaConf.select(yaml_conf, selector_path)
            if selected is not None:
                selected_path = selector_path
                break

        profiles = yaml_conf.get(self.profiles_key, None)
        if selected is None or profiles is None:
            return

        if selected not in profiles:
            available = ", ".join(sorted(str(k) for k in profiles.keys()))
            raise ValueError(
                f"Unknown selector '{selected}' at '{selected_path}'. "
                f"Available profiles: [{available}]"
            )

        if self.merge_into_existing:
            _merge_into_path(yaml_conf, self.target_path, profiles[selected])
        else:
            OmegaConf.update(yaml_conf, self.target_path, profiles[selected], force_add=True)


class YamlProfileEngine:
    """Run a set of profile appliers and cleanup helper keys afterward."""

    def __init__(self, appliers: List[YamlProfileApplier]) -> None:
        self.appliers = appliers

    def apply(self, yaml_conf: DictConfig) -> DictConfig:
        if not isinstance(yaml_conf, DictConfig):
            return yaml_conf

        cleanup_keys: set[str] = set()
        for applier in self.appliers:
            applier.apply(yaml_conf)
            cleanup_keys.update(applier.cleanup_keys)

        for key in cleanup_keys:
            if key in yaml_conf:
                del yaml_conf[key]

        return yaml_conf


_YAML_PROFILE_ENGINE = YamlProfileEngine(
    [
        PathMoveApplier(
            source_key="system_profiles",
            target_path="shared.system_profiles",
            merge_into_existing=True,
        ),
        ValueProfileApplier(
            selector_paths=["shared.pipeline_profile", "pipeline_profile"],
            profiles_key="pipeline_profiles",
            target_path="shared",
            merge_into_existing=True,
            cleanup_keys=("pipeline_profile", "pipeline_profiles"),
        ),
        ArchProfileApplier(
            selector_paths=["shared.arch_profile", "arch_profile"],
            cleanup_keys=("arch_profile", "arch_profiles"),
        ),
        MappingProfileApplier(
            selector_paths=["shared.data_transform_profile", "data_transform_profile"],
            profiles_key="data_transform_profiles",
            section_to_target={
                "data_transform": "data.data_transform",
                "image_transform": "data.image_transform",
                "nnunet_preprocessing": "data.nnunet_preprocessing",
            },
            cleanup_keys=("data_transform_profile", "data_transform_profiles"),
        ),
        ValueProfileApplier(
            selector_paths=["shared.augmentation_profile", "augmentation_profile"],
            profiles_key="augmentation_profiles",
            target_path="data.augmentation",
            merge_into_existing=True,
            cleanup_keys=("augmentation_profile", "augmentation_profiles"),
        ),
        ValueProfileApplier(
            selector_paths=["shared.dataloader_profile", "dataloader_profile"],
            profiles_key="dataloader_profiles",
            target_path="data.dataloader",
            merge_into_existing=True,
            cleanup_keys=("dataloader_profile", "dataloader_profiles"),
        ),
        ValueProfileApplier(
            selector_paths=["shared.optimizer_profile", "optimizer_profile"],
            profiles_key="optimizer_profiles",
            target_path="optimization",
            merge_into_existing=True,
            cleanup_keys=("optimizer_profile", "optimizer_profiles"),
        ),
        ValueProfileApplier(
            selector_paths=["shared.loss_profile", "loss_profile"],
            profiles_key="loss_profiles",
            target_path="model.loss.losses",
            merge_into_existing=False,
            cleanup_keys=("loss_profile", "loss_profiles"),
        ),
        ValueProfileApplier(
            selector_paths=["shared.label_profile", "label_profile"],
            profiles_key="label_profiles",
            target_path="data.label_transform",
            merge_into_existing=False,
            cleanup_keys=("label_profile", "label_profiles"),
        ),
        ValueProfileApplier(
            selector_paths=["shared.decoding_profile", "decoding_profile"],
            profiles_key="decoding_profiles",
            target_path="inference.decoding",
            merge_into_existing=False,
            cleanup_keys=("decoding_profile", "decoding_profiles"),
        ),
        ValueProfileApplier(
            selector_paths=["shared.activation_profile", "activation_profile"],
            profiles_key="activation_profiles",
            target_path="inference.test_time_augmentation.channel_activations",
            merge_into_existing=False,
            cleanup_keys=("activation_profile", "activation_profiles"),
        ),
        ReferenceProfileApplier(
            profiles_key="loss_profiles",
            target_paths=["model.loss.losses"],
        ),
        ReferenceProfileApplier(
            profiles_key="label_profiles",
            target_paths=["data.label_transform"],
        ),
        ReferenceProfileApplier(
            profiles_key="decoding_profiles",
            target_paths=["inference.decoding"],
        ),
        ReferenceProfileApplier(
            profiles_key="activation_profiles",
            target_paths=[
                "inference.test_time_augmentation.channel_activations",
                "shared.inference.test_time_augmentation.channel_activations",
                "test.inference.test_time_augmentation.channel_activations",
                "tune.inference.test_time_augmentation.channel_activations",
            ],
        ),
        ListProfileReferenceApplier(
            profiles_key="decoding_profiles",
            target_paths=[
                "inference.decoding",
                "shared.inference.decoding",
                "test.decoding",
                "test.inference.decoding",
                "tune.inference.decoding",
            ],
        ),
        ReferenceProfileApplier(
            profiles_key="augmentation_profiles",
            target_paths=["data.augmentation"],
        ),
        ReferenceProfileApplier(
            profiles_key="dataloader_profiles",
            target_paths=["data.dataloader"],
        ),
        ReferenceProfileApplier(
            profiles_key="optimizer_profiles",
            target_paths=["optimization"],
        ),
    ]
)


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with defaults merged
    """
    config_path = Path(config_path).resolve()
    yaml_conf = _load_config_with_bases(config_path)
    yaml_conf = _YAML_PROFILE_ENGINE.apply(yaml_conf)
    explicit_enabled_paths = _collect_explicit_enabled_paths(yaml_conf)
    explicit_field_paths = _collect_explicit_paths(yaml_conf)

    # Merge with structured config defaults
    default_conf = OmegaConf.structured(Config)
    merged = OmegaConf.merge(default_conf, yaml_conf)

    # Convert to dataclass instance
    cfg = OmegaConf.to_object(merged)
    setattr(cfg, "_explicit_enabled_paths", explicit_enabled_paths)
    setattr(cfg, "_explicit_field_paths", explicit_field_paths)
    return cfg


def save_config(cfg: Config, save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        cfg: Config object to save
        save_path: Path where to save the YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    omega_conf = OmegaConf.structured(cfg)
    OmegaConf.save(omega_conf, save_path)


def merge_configs(base_cfg: Config, *override_cfgs: Union[Config, Dict, str, Path]) -> Config:
    """
    Merge multiple configurations together.

    Args:
        base_cfg: Base configuration
        *override_cfgs: One or more override configs (Config, dict, or path to YAML)

    Returns:
        Merged Config object
    """
    result = OmegaConf.structured(base_cfg)

    for override_cfg in override_cfgs:
        if isinstance(override_cfg, (str, Path)):
            override_omega = OmegaConf.load(override_cfg)
        elif isinstance(override_cfg, Config):
            override_omega = OmegaConf.structured(override_cfg)
        elif isinstance(override_cfg, (dict, DictConfig)):
            override_omega = OmegaConf.create(override_cfg)
        else:
            raise TypeError(f"Unsupported config type: {type(override_cfg)}")

        result = OmegaConf.merge(result, override_omega)

    return OmegaConf.to_object(result)


def update_from_cli(cfg: Config, overrides: List[str]) -> Config:
    """
    Update config from command-line overrides.

    Supports dot notation: ['data.dataloader.batch_size=4', 'model.arch.type=unet3d']

    Args:
        cfg: Base Config object
        overrides: List of 'key=value' strings

    Returns:
        Updated Config object
    """
    cfg_omega = OmegaConf.structured(cfg)
    cli_conf = OmegaConf.from_dotlist(overrides)
    merged = OmegaConf.merge(cfg_omega, cli_conf)
    return OmegaConf.to_object(merged)


def to_dict(cfg: Config, resolve: bool = True) -> Dict[str, Any]:
    """
    Convert Config to dictionary.

    Args:
        cfg: Config object
        resolve: Whether to resolve variable interpolations

    Returns:
        Dictionary representation
    """
    omega_conf = OmegaConf.structured(cfg)
    return OmegaConf.to_container(omega_conf, resolve=resolve)


def from_dict(d: Dict[str, Any]) -> Config:
    """
    Create Config from dictionary.

    Args:
        d: Dictionary with configuration values

    Returns:
        Config object
    """
    default_conf = OmegaConf.structured(Config)
    dict_conf = OmegaConf.create(d)
    merged = OmegaConf.merge(default_conf, dict_conf)
    return OmegaConf.to_object(merged)


def print_config(cfg: Config, resolve: bool = True) -> None:
    """
    Pretty print configuration.

    Args:
        cfg: Config to print
        resolve: Whether to resolve variable interpolations
    """
    omega_conf = OmegaConf.structured(cfg)
    print(OmegaConf.to_yaml(omega_conf, resolve=resolve))


def validate_config(cfg: Config) -> None:
    """
    Validate configuration values.

    Args:
        cfg: Config object to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Model validation
    if cfg.model.in_channels <= 0:
        raise ValueError("model.in_channels must be positive")
    if cfg.model.out_channels <= 0:
        raise ValueError("model.out_channels must be positive")
    if len(cfg.model.input_size) not in [2, 3]:
        raise ValueError(
            "model.input_size must be 2D or 3D (got length {})".format(len(cfg.model.input_size))
        )

    # System validation
    if cfg.system.num_workers < 0:
        raise ValueError("system.num_workers must be non-negative")

    # Data validation
    if len(cfg.data.dataloader.patch_size) not in [2, 3]:
        raise ValueError(
            "data.dataloader.patch_size must be 2D or 3D (got length {})".format(
                len(cfg.data.dataloader.patch_size)
            )
        )
    if cfg.data.dataloader.batch_size <= 0:
        raise ValueError("data.dataloader.batch_size must be positive")

    # Optimizer validation
    if cfg.optimization.optimizer.lr <= 0:
        raise ValueError("optimization.optimizer.lr must be positive")
    if cfg.optimization.optimizer.weight_decay < 0:
        raise ValueError("optimization.optimizer.weight_decay must be non-negative")

    # Training validation
    # [FIX 2] Allow max_epochs to be 0 or negative when using step-based training
    max_steps_cfg = getattr(cfg.optimization, "max_steps", None)
    if max_steps_cfg is None or max_steps_cfg <= 0:
        # Epoch-based training: max_epochs must be positive
        if cfg.optimization.max_epochs <= 0:
            raise ValueError("optimization.max_epochs must be positive when max_steps is not set")
    # If max_steps is set, max_epochs can be anything (will be overridden to -1 in trainer)

    if cfg.optimization.gradient_clip_val < 0:
        raise ValueError("optimization.gradient_clip_val must be non-negative")
    if cfg.optimization.accumulate_grad_batches <= 0:
        raise ValueError("optimization.accumulate_grad_batches must be positive")
    if hasattr(cfg.optimization, "ema") and getattr(cfg.optimization.ema, "enabled", False):
        if cfg.optimization.ema.decay <= 0 or cfg.optimization.ema.decay >= 1:
            raise ValueError("optimization.ema.decay must be in (0, 1)")
        if cfg.optimization.ema.warmup_steps < 0:
            raise ValueError("optimization.ema.warmup_steps must be non-negative")

    # Loss validation
    model_loss_cfg = getattr(cfg.model, "loss", None)
    losses_cfg = getattr(model_loss_cfg, "losses", None)
    if losses_cfg is not None:
        for i, entry in enumerate(losses_cfg):
            if not isinstance(entry, dict):
                raise ValueError(f"model.loss.losses[{i}] must be a dict")
            if "function" not in entry:
                raise ValueError(f"model.loss.losses[{i}] must have a 'function' key")
            w = entry.get("weight", 1.0)
            if w < 0:
                raise ValueError(f"model.loss.losses[{i}].weight must be non-negative")


def get_config_hash(cfg: Config) -> str:
    """
    Generate a hash string for the configuration.

    Useful for experiment tracking and reproducibility.

    Args:
        cfg: Config object

    Returns:
        Hash string
    """
    import hashlib

    omega_conf = OmegaConf.structured(cfg)
    yaml_str = OmegaConf.to_yaml(omega_conf, resolve=True)
    return hashlib.md5(yaml_str.encode()).hexdigest()[:8]


def create_experiment_name(cfg: Config) -> str:
    """
    Create a descriptive experiment name from config.

    Args:
        cfg: Config object

    Returns:
        Experiment name string
    """
    parts = [
        cfg.model.arch.type,
        f"bs{cfg.data.dataloader.batch_size}",
        f"lr{cfg.optimization.optimizer.lr:.0e}",
        get_config_hash(cfg),
    ]
    return "_".join(parts)


def resolve_data_paths(cfg: Config) -> Config:
    """
    Resolve data paths by combining base paths (train_path, val_path)
    with relative file paths (train_image, train_label, etc.).

    This function modifies the config in-place by:
    1. Prepending base paths to relative file paths
    2. Expanding glob patterns to actual file lists
    3. Flattening nested lists from glob expansion

    Supported paths:
    - Training: cfg.data.train.path + cfg.data.train.image/label/mask
    - Validation: cfg.data.val.path + cfg.data.val.image/label/mask
    - Testing: cfg.test.data.val.path + cfg.test.data.val.image/label/mask
    - Tuning: cfg.tune.data.val.path + cfg.tune.data.val.image/label/mask

    Note: Test data belongs in cfg.test.data, tuning data in cfg.tune.data

    Args:
        cfg: Config object to resolve paths for

    Returns:
        Config object with resolved paths (same object, modified in-place)

    Example:
        >>> cfg.data.train.path = "/data/barcode/"
        >>> cfg.data.train.image = ["PT37/*_raw.tif", "file.tif"]
        >>> resolve_data_paths(cfg)
        >>> print(cfg.data.train.image)
        [
            '/data/barcode/PT37/img1_raw.tif',
            '/data/barcode/PT37/img2_raw.tif',
            '/data/barcode/file.tif'
        ]

        >>> cfg.test.data.val.path = "/data/test/"
        >>> cfg.test.data.val.image = ["volume_*.tif"]
        >>> resolve_data_paths(cfg)
        >>> print(cfg.test.data.val.image)
        ['/data/test/volume_1.tif', '/data/test/volume_2.tif']
    """
    import os
    from glob import glob

    def _combine_path(
        base_path: str, file_path: Optional[Union[str, List[str]]]
    ) -> Optional[Union[str, List[str]]]:
        """Helper to combine base path with file path(s) and expand globs."""
        if file_path is None:
            return file_path

        # Handle list of paths
        if isinstance(file_path, list):
            result: List[str] = []
            for p in file_path:
                resolved = _combine_path(base_path, p)
                if resolved is None:
                    continue
                # If resolved is a list (from glob expansion), extend
                if isinstance(resolved, list):
                    result.extend(resolved)
                else:
                    result.append(resolved)
            return result

        # Handle string path
        # Combine with base path if relative
        if base_path and not os.path.isabs(file_path):
            file_path = os.path.join(base_path, file_path)

        # Expand glob patterns with optional selector support
        # Format: path/*.tiff[0] or path/*.tiff[filename]
        import re

        selector_match = re.match(r"^(.+)\[(.+)\]$", file_path)

        if selector_match:
            # Has selector - extract glob pattern and selector
            glob_pattern = selector_match.group(1)
            selector = selector_match.group(2)

            expanded = sorted(glob(glob_pattern))
            if not expanded:
                return file_path  # No matches - return original

            # Select file based on selector
            try:
                # Try numeric index
                index = int(selector)
                if index < -len(expanded) or index >= len(expanded):
                    print(
                        f"Warning: Index {index} out of range for {len(expanded)} files, "
                        f"using first"
                    )
                    return expanded[0]
                return expanded[index]
            except ValueError:
                # Not a number, try filename match
                from pathlib import Path

                matching = [
                    f for f in expanded if Path(f).name == selector or Path(f).stem == selector
                ]
                if not matching:
                    # Try partial match
                    matching = [f for f in expanded if selector in Path(f).name]
                if matching:
                    return matching[0]
                else:
                    print(
                        f"Warning: No file matches selector '{selector}', "
                        f"using first of {len(expanded)} files"
                    )
                    return expanded[0]

        elif "*" in file_path or "?" in file_path:
            # Standard glob without selector
            expanded = sorted(glob(file_path))
            if expanded:
                return expanded
            else:
                # No matches - return original pattern (will be caught by validation)
                return file_path

        return file_path

    # Resolve training paths (always expand globs, use train_path as base if available)
    train_base = cfg.data.train.path if cfg.data.train.path else ""
    cfg.data.train.image = _combine_path(train_base, cfg.data.train.image)
    cfg.data.train.label = _combine_path(train_base, cfg.data.train.label)
    cfg.data.train.mask = _combine_path(train_base, cfg.data.train.mask)
    train_json_resolved = _combine_path(train_base, cfg.data.train.json)
    if isinstance(train_json_resolved, list):
        cfg.data.train.json = train_json_resolved[0] if train_json_resolved else None
    else:
        cfg.data.train.json = train_json_resolved

    # Resolve validation paths (always expand globs, use val_path as base if available)
    val_base = cfg.data.val.path if cfg.data.val.path else ""
    cfg.data.val.image = _combine_path(val_base, cfg.data.val.image)
    cfg.data.val.label = _combine_path(val_base, cfg.data.val.label)
    cfg.data.val.mask = _combine_path(val_base, cfg.data.val.mask)
    val_json_resolved = _combine_path(val_base, cfg.data.val.json)
    if isinstance(val_json_resolved, list):
        cfg.data.val.json = val_json_resolved[0] if val_json_resolved else None
    else:
        cfg.data.val.json = val_json_resolved

    # Resolve test data paths (cfg.test.data.val.*)
    if cfg.test is not None:
        test_data = cfg.test.data
        test_path_value = getattr(test_data.val, "path", "")
        test_base = test_path_value if isinstance(test_path_value, str) else ""
        test_data.val.image = _combine_path(test_base, test_data.val.image)
        test_data.val.label = _combine_path(test_base, test_data.val.label)
        test_data.val.mask = _combine_path(test_base, test_data.val.mask)

    # Resolve tuning data paths (cfg.tune.data.val.*)
    if cfg.tune is not None:
        tune_data = cfg.tune.data
        tune_path_value = getattr(tune_data.val, "path", "")
        tune_base = tune_path_value if isinstance(tune_path_value, str) else ""
        tune_data.val.image = _combine_path(tune_base, tune_data.val.image)
        tune_data.val.label = _combine_path(tune_base, tune_data.val.label)
        tune_data.val.mask = _combine_path(tune_base, tune_data.val.mask)

    return cfg


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


def resolve_shared_profiles(cfg: Config, mode: str = "train") -> Config:
    """Resolve shared stage profiles into runtime sections.

    This keeps the existing runtime access pattern (cfg.system/cfg.data/cfg.inference)
    while allowing YAMLs to define reusable profiles under ``shared``.
    """
    shared = getattr(cfg, "shared", None)
    if shared is None:
        return cfg

    mode_normalized = mode.strip().lower()
    if mode_normalized == "tune-test":
        # tune-test first executes tuning, then test; prefer tune profile at setup stage.
        mode_normalized = "tune"

    def _as_dict(value: Any) -> Dict[str, Any]:
        """Convert a config value into a plain dict, or {} if not mapping-like."""
        if value is None:
            return {}

        if isinstance(value, str):
            return {}

        try:
            container = OmegaConf.to_container(_to_omegaconf(value), resolve=True)
        except Exception:
            return {}

        return container if isinstance(container, dict) else {}

    def _extract_system_selector(selector_obj: Any) -> tuple[Optional[str], Dict[str, Any]]:
        """Extract (profile, direct_overrides) from SystemConfig-like objects."""
        if selector_obj is None:
            return None, {}

        # Allow shorthand: shared.system: profile_name
        if isinstance(selector_obj, str):
            return selector_obj, {}

        selector_map = _as_dict(selector_obj)
        profile_name = selector_map.get("profile")

        # Direct stage overrides
        direct_overrides: Dict[str, Any] = {}
        for key in ("num_gpus", "num_workers", "seed"):
            value = selector_map.get(key, None)
            if value is not None:
                direct_overrides[key] = value

        return profile_name, direct_overrides

    def _extract_inference_selector(selector_obj: Any) -> tuple[Optional[str], Dict[str, Any], Dict[str, Any]]:
        """Extract (profile, profile_overrides, direct_overrides) from inference selectors."""
        if selector_obj is None:
            return None, {}, {}

        if isinstance(selector_obj, str):
            return selector_obj, {}, {}

        selector_map = _as_dict(selector_obj)
        profile_name = selector_map.get("profile")
        profile_overrides = selector_map.get("overrides", {}) or {}
        direct_overrides = {
            k: v for k, v in selector_map.items() if k not in {"profile", "overrides"}
        }
        return profile_name, _as_dict(profile_overrides), direct_overrides

    data_default_cfg = OmegaConf.structured(Config).data
    data_allowed_keys = set(data_default_cfg.keys())
    explicit_field_paths = getattr(cfg, "_explicit_field_paths", set())
    if not isinstance(explicit_field_paths, set):
        explicit_field_paths = set(explicit_field_paths)

    def _has_explicit_path(base_path: str) -> bool:
        return any(
            p == base_path or p.startswith(f"{base_path}.") or p.startswith(f"{base_path}[")
            for p in explicit_field_paths
        )

    def _extract_mode_data_overrides(stage_data_obj: Any, stage_prefix: str) -> Dict[str, Any]:
        """Extract only explicit runtime-data keys from mode-specific stage data."""
        stage_data_map = _as_dict(stage_data_obj)
        if not stage_data_map:
            return {}

        overrides: Dict[str, Any] = {}
        for key, value in stage_data_map.items():
            if key not in data_allowed_keys:
                continue

            full_path = f"{stage_prefix}.{key}"
            if not _has_explicit_path(full_path):
                continue
            overrides[key] = value

        # Stage data carries `batch_size` at top level for test/tune data sections.
        # Map it into runtime `data.dataloader.batch_size`.
        batch_size_path = f"{stage_prefix}.batch_size"
        if "batch_size" in stage_data_map and _has_explicit_path(batch_size_path):
            dataloader_overrides = overrides.get("dataloader", {})
            if not isinstance(dataloader_overrides, dict):
                dataloader_overrides = _as_dict(dataloader_overrides)
            dataloader_overrides["batch_size"] = stage_data_map["batch_size"]
            overrides["dataloader"] = dataloader_overrides

        return overrides

    # 1) System profile resolution
    system_profiles = getattr(shared, "system_profiles", {}) or {}
    shared_selector = getattr(shared, "system", None)
    shared_profile_name, shared_overrides = _extract_system_selector(shared_selector)

    if mode_normalized == "train":
        stage_selector = getattr(getattr(cfg, "train", None), "system", None)
    elif mode_normalized == "test":
        stage_selector = (
            getattr(cfg.test, "system", None)
            if hasattr(cfg, "test") and cfg.test is not None
            else None
        )
    else:  # tune
        stage_selector = (
            getattr(cfg.tune, "system", None)
            if hasattr(cfg, "tune") and cfg.tune is not None
            else None
        )

    stage_profile_name, stage_overrides = _extract_system_selector(stage_selector)
    profile_name = stage_profile_name or shared_profile_name

    if profile_name is not None and profile_name not in system_profiles:
        available = ", ".join(sorted(str(k) for k in system_profiles.keys()))
        raise ValueError(
            f"Unknown system profile '{profile_name}' for mode='{mode_normalized}'. "
            f"Available system_profiles: [{available}]"
        )

    profile_cfg = system_profiles.get(profile_name) if profile_name else None

    if profile_cfg is not None or shared_overrides or stage_overrides:
        merged_profile = _to_omegaconf(profile_cfg)
        merged_profile = OmegaConf.merge(
            merged_profile,
            _to_omegaconf(shared_overrides),
            _to_omegaconf(stage_overrides),
        )
        profile_payload = OmegaConf.to_container(merged_profile, resolve=True)
        if not isinstance(profile_payload, dict):
            profile_payload = {}
        profile_payload = {k: v for k, v in profile_payload.items() if v is not None}
        cfg.system = _merge_dataclass(cfg.system, profile_payload)

    # 2) Generic shared + mode-specific runtime section merge:
    # defaults (already on cfg) < shared.<section> < <mode>.<section>
    _, _, shared_inference_overrides = _extract_inference_selector(
        getattr(shared, "inference", None)
    )

    shared_overrides: Dict[str, Dict[str, Any]] = {
        "model": _as_dict(getattr(shared, "model", None)),
        "data": _as_dict(getattr(shared, "data", None)),
        "optimization": _as_dict(getattr(shared, "optimization", None)),
        "monitor": _as_dict(getattr(shared, "monitor", None)),
        "inference": shared_inference_overrides,
    }

    stage_overrides: Dict[str, Dict[str, Any]] = {
        "model": {},
        "data": {},
        "optimization": {},
        "monitor": {},
        "inference": {},
    }
    if mode_normalized == "train":
        train_cfg = getattr(cfg, "train", None)
        if train_cfg is not None:
            stage_overrides = {
                "model": _as_dict(getattr(train_cfg, "model", None)),
                "data": _as_dict(getattr(train_cfg, "data", None)),
                "optimization": _as_dict(getattr(train_cfg, "optimization", None)),
                "monitor": _as_dict(getattr(train_cfg, "monitor", None)),
                "inference": {},
            }
    elif mode_normalized == "test":
        test_cfg = getattr(cfg, "test", None)
        if test_cfg is not None:
            _, _, test_inference_overrides = _extract_inference_selector(
                getattr(test_cfg, "inference", None)
            )
            stage_overrides = {
                "model": _as_dict(getattr(test_cfg, "model", None)),
                "data": _extract_mode_data_overrides(getattr(test_cfg, "data", None), "test.data"),
                "optimization": {},
                "monitor": {},
                "inference": test_inference_overrides,
            }
    else:  # tune
        tune_cfg = getattr(cfg, "tune", None)
        if tune_cfg is not None:
            _, _, tune_inference_overrides = _extract_inference_selector(
                getattr(tune_cfg, "inference", None)
            )
            stage_overrides = {
                "model": _as_dict(getattr(tune_cfg, "model", None)),
                "data": _extract_mode_data_overrides(getattr(tune_cfg, "data", None), "tune.data"),
                "optimization": {},
                "monitor": {},
                "inference": tune_inference_overrides,
            }

    for section_name in ("model", "data", "optimization", "monitor", "inference"):
        shared_section = shared_overrides.get(section_name, {})
        stage_section = stage_overrides.get(section_name, {})
        if not shared_section and not stage_section:
            continue
        target_section = getattr(cfg, section_name)
        setattr(
            cfg,
            section_name,
            _merge_dataclass(target_section, shared_section, stage_section),
        )

    # 3) Data transform profile resolution (stage-specific)
    data_profiles = getattr(shared, "data_transform_profiles", {}) or {}
    if data_profiles:
        default_cfg = Config()

        def _apply_data_profile(stage_data_cfg: Any, stage_prefix: str, default_stage_data_cfg: Any) -> None:
            if stage_data_cfg is None:
                return
            if default_stage_data_cfg is None and is_dataclass(stage_data_cfg):
                try:
                    default_stage_data_cfg = type(stage_data_cfg)()
                except Exception:
                    default_stage_data_cfg = None

            image_transform_cfg = getattr(stage_data_cfg, "image_transform", None)
            profile_name = (
                getattr(image_transform_cfg, "transform_profile", None)
                if image_transform_cfg is not None
                else None
            ) or ("default" if "default" in data_profiles else None)
            profile_cfg = data_profiles.get(profile_name) if profile_name else None

            if profile_cfg is None:
                return

            profile_conf = _to_omegaconf(profile_cfg) if profile_cfg is not None else OmegaConf.create({})
            profile_data_transform = profile_conf.get("data_transform", None)
            profile_image_transform = profile_conf.get("image_transform", None)
            profile_mask_transform = profile_conf.get("mask_transform", None)
            profile_nnunet_preprocessing = profile_conf.get("nnunet_preprocessing", None)

            def _collect_section_overrides(section_name: str) -> Dict[str, Any]:
                section_obj = getattr(stage_data_cfg, section_name, None)
                default_section_obj = (
                    getattr(default_stage_data_cfg, section_name, None)
                    if default_stage_data_cfg is not None
                    else None
                )
                if section_obj is None:
                    return {}

                section_map = _as_dict(section_obj)
                default_section_map = _as_dict(default_section_obj)

                # If explicit path tracking is unavailable, fall back to non-default values.
                has_explicit_tracking = bool(explicit_field_paths)

                overrides: Dict[str, Any] = {}
                for key, value in section_map.items():
                    stage_key_path = f"{stage_prefix}.{section_name}.{key}"
                    shared_key_path = f"shared.data.{section_name}.{key}"
                    is_explicit = _has_explicit_path(stage_key_path) or _has_explicit_path(shared_key_path)
                    is_nondefault = value != default_section_map.get(key)
                    if is_explicit or (not has_explicit_tracking and is_nondefault) or (
                        has_explicit_tracking and is_nondefault
                    ):
                        overrides[key] = value
                return overrides

            data_transform_override = _collect_section_overrides("data_transform")
            image_transform_override = _collect_section_overrides("image_transform")
            mask_transform_override = _collect_section_overrides("mask_transform")
            nnunet_pre_override = _collect_section_overrides("nnunet_preprocessing")

            if hasattr(stage_data_cfg, "data_transform") and profile_data_transform is not None:
                stage_data_cfg.data_transform = _merge_dataclass(
                    stage_data_cfg.data_transform,
                    profile_data_transform,
                    data_transform_override,
                )

            if hasattr(stage_data_cfg, "image_transform") and profile_image_transform is not None:
                stage_data_cfg.image_transform = _merge_dataclass(
                    stage_data_cfg.image_transform,
                    profile_image_transform,
                    image_transform_override,
                )

            if hasattr(stage_data_cfg, "mask_transform") and profile_mask_transform is not None:
                stage_data_cfg.mask_transform = _merge_dataclass(
                    stage_data_cfg.mask_transform,
                    profile_mask_transform,
                    mask_transform_override,
                )

            if (
                hasattr(stage_data_cfg, "nnunet_preprocessing")
                and profile_nnunet_preprocessing is not None
            ):
                stage_data_cfg.nnunet_preprocessing = _merge_dataclass(
                    stage_data_cfg.nnunet_preprocessing,
                    profile_nnunet_preprocessing,
                    nnunet_pre_override,
                )

        _apply_data_profile(
            getattr(cfg, "data", None),
            "data",
            getattr(default_cfg, "data", None),
        )
        _apply_data_profile(
            getattr(getattr(cfg, "test", None), "data", None),
            "test.data",
            getattr(getattr(default_cfg, "test", None), "data", None),
        )
        _apply_data_profile(
            getattr(getattr(cfg, "tune", None), "data", None),
            "tune.data",
            getattr(getattr(default_cfg, "tune", None), "data", None),
        )

    # 4) Inference profile resolution (test/tune only)
    inference_profiles = getattr(shared, "inference_profiles", {}) or {}
    if mode_normalized in {"test", "tune"}:
        (
            shared_profile_name,
            shared_profile_overrides,
            shared_direct_overrides,
        ) = _extract_inference_selector(
            getattr(shared, "inference", None)
        )
        if mode_normalized == "test":
            selector = (
                getattr(cfg.test, "inference", None)
                if hasattr(cfg, "test") and cfg.test is not None
                else None
            )
        else:  # tune
            selector = (
                getattr(cfg.tune, "inference", None)
                if hasattr(cfg, "tune") and cfg.tune is not None
                else None
            )

        (
            stage_profile_name,
            stage_profile_overrides,
            stage_direct_overrides,
        ) = _extract_inference_selector(selector)
        profile_name = (
            stage_profile_name
            or shared_profile_name
            or ("default" if "default" in inference_profiles else None)
        )
        merged_profile_overrides = _to_omegaconf(shared_direct_overrides)
        merged_profile_overrides = OmegaConf.merge(
            merged_profile_overrides,
            _to_omegaconf(shared_profile_overrides),
            _to_omegaconf(stage_profile_overrides),
            _to_omegaconf(stage_direct_overrides),
        )
        merged_profile_overrides_dict = OmegaConf.to_container(
            merged_profile_overrides, resolve=True
        )
        if not isinstance(merged_profile_overrides_dict, dict):
            merged_profile_overrides_dict = {}

        if profile_name is not None and profile_name not in inference_profiles:
            available = ", ".join(sorted(str(k) for k in inference_profiles.keys()))
            raise ValueError(
                f"Unknown inference profile '{profile_name}' for mode='{mode_normalized}'. "
                f"Available inference_profiles: [{available}]"
            )

        profile_cfg = inference_profiles.get(profile_name) if profile_name else None

        if profile_cfg is not None or merged_profile_overrides_dict:
            cfg.inference = _merge_dataclass(
                cfg.inference,
                profile_cfg,
                merged_profile_overrides_dict,
            )

        # Apply inference.system resource overrides onto runtime system in test/tune modes.
        # Only apply explicit values from selected inference profile and/or explicit
        # shared/mode inference overrides. Do not apply dataclass defaults implicitly.
        inference_system_payload: Dict[str, Any] = {}
        system_keys = ("num_gpus", "num_workers", "seed")

        # 1) Selected inference profile defaults (explicit keys only)
        if profile_name and profile_cfg is not None:
            profile_system_map = _as_dict(_as_dict(profile_cfg).get("system", {}))
            for key in system_keys:
                profile_key_path = f"shared.inference_profiles.{profile_name}.system.{key}"
                if _has_explicit_path(profile_key_path):
                    value = profile_system_map.get(key)
                    if value is not None:
                        inference_system_payload[key] = value

        # 2) Explicit shared/mode inference overrides (highest precedence)
        override_system_map = _as_dict(merged_profile_overrides_dict.get("system", {}))
        for key in system_keys:
            if key in override_system_map and override_system_map.get(key) is not None:
                inference_system_payload[key] = override_system_map[key]

        if inference_system_payload:
            cfg.system = _merge_dataclass(cfg.system, inference_system_payload)

    explicit_enabled_paths = getattr(cfg, "_explicit_enabled_paths", set())
    explicit_field_paths = getattr(cfg, "_explicit_field_paths", set())
    if not isinstance(explicit_enabled_paths, set):
        explicit_enabled_paths = set(explicit_enabled_paths)
    if not isinstance(explicit_field_paths, set):
        explicit_field_paths = set(explicit_field_paths)

    default_conf = OmegaConf.structured(Config)
    auto_enable_paths: List[tuple[str, Any]] = [
        ("data", getattr(cfg, "data", None)),
        ("optimization", getattr(cfg, "optimization", None)),
        ("monitor", getattr(cfg, "monitor", None)),
        ("inference", getattr(cfg, "inference", None)),
    ]
    if getattr(cfg, "test", None) is not None:
        auto_enable_paths.append(("test.data", getattr(cfg.test, "data", None)))
    if getattr(cfg, "tune", None) is not None:
        auto_enable_paths.append(("tune", getattr(cfg, "tune", None)))
        auto_enable_paths.append(("tune.data", getattr(cfg.tune, "data", None)))

    for target_path, section_obj in auto_enable_paths:
        if section_obj is None:
            continue

        default_section = OmegaConf.select(default_conf, target_path)
        if default_section is None:
            continue

        runtime_section = OmegaConf.structured(section_obj)
        _auto_enable_changed_sections(
            default_section,
            runtime_section,
            explicit_enabled_paths=explicit_enabled_paths,
            explicit_field_paths=explicit_field_paths,
            path=target_path,
        )

        if "." in target_path:
            parts = target_path.split(".")
            parent_obj = cfg
            for part in parts[:-1]:
                parent_obj = getattr(parent_obj, part)
            child_key = parts[-1]
            setattr(parent_obj, child_key, OmegaConf.to_object(runtime_section))
        else:
            setattr(cfg, target_path, OmegaConf.to_object(runtime_section))

    setattr(cfg, "_explicit_enabled_paths", explicit_enabled_paths)
    setattr(cfg, "_explicit_field_paths", explicit_field_paths)
    return cfg


__all__ = [
    "load_config",
    "save_config",
    "merge_configs",
    "update_from_cli",
    "to_dict",
    "from_dict",
    "print_config",
    "validate_config",
    "get_config_hash",
    "create_experiment_name",
    "resolve_data_paths",
    "resolve_shared_profiles",
]
