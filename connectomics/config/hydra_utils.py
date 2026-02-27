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


def _merge_into_path(yaml_conf: DictConfig, target_path: str, profile_value: Any) -> None:
    """Merge profile payload into target path, keeping explicit config overrides."""
    existing = OmegaConf.select(yaml_conf, target_path)
    if existing is None:
        OmegaConf.update(yaml_conf, target_path, profile_value, force_add=True)
        return

    merged = OmegaConf.merge(profile_value, existing)
    OmegaConf.update(yaml_conf, target_path, merged, force_add=True)


class MappingProfileApplier(YamlProfileApplier):
    """Apply selected profile mapping into one or more target config paths."""

    def __init__(
        self,
        selector_paths: Union[str, List[str], Tuple[str, ...]],
        profiles_key: str,
        section_to_target: Dict[str, str],
        cleanup_keys: Optional[Tuple[str, ...]] = None,
    ) -> None:
        if isinstance(selector_paths, str):
            selector_paths = [selector_paths]
        self.selector_paths = list(selector_paths)
        self.profiles_key = profiles_key
        self.section_to_target = section_to_target
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
        for profile_section, target_path in self.section_to_target.items():
            if profile_section in selected_profile:
                _merge_into_path(yaml_conf, target_path, selected_profile[profile_section])


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
        MappingProfileApplier(
            selector_paths=["shared.arch_profile", "arch_profile"],
            profiles_key="arch_profiles",
            section_to_target={
                "shared": "shared",
                "system": "system",
                "model": "model",
                "data": "data",
                "optimization": "optimization",
                "monitor": "monitor",
                "inference": "inference",
            },
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
            selector_paths=["shared.loss_profile", "loss_profile"],
            profiles_key="loss_profiles",
            target_path="model.losses",
            merge_into_existing=False,
            cleanup_keys=("loss_profile", "loss_profiles"),
        ),
        ReferenceProfileApplier(
            profiles_key="loss_profiles",
            target_paths=["model.losses"],
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

    # Merge with structured config defaults
    default_conf = OmegaConf.structured(Config)
    merged = OmegaConf.merge(default_conf, yaml_conf)

    # Convert to dataclass instance
    cfg = OmegaConf.to_object(merged)

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

    Supports dot notation: ['system.training.batch_size=4', 'model.architecture=unet3d']

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
    if cfg.system.training.batch_size <= 0:
        raise ValueError("system.training.batch_size must be positive")
    if cfg.system.training.num_workers < 0:
        raise ValueError("system.training.num_workers must be non-negative")

    # Data validation
    if len(cfg.data.patch_size) not in [2, 3]:
        raise ValueError(
            "data.patch_size must be 2D or 3D (got length {})".format(len(cfg.data.patch_size))
        )

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
    if cfg.model.losses is not None:
        for i, entry in enumerate(cfg.model.losses):
            if not isinstance(entry, dict):
                raise ValueError(f"model.losses[{i}] must be a dict")
            if "function" not in entry:
                raise ValueError(f"model.losses[{i}] must have a 'function' key")
            w = entry.get("weight", 1.0)
            if w < 0:
                raise ValueError(f"model.losses[{i}].weight must be non-negative")


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
        cfg.model.architecture,
        f"bs{cfg.system.training.batch_size}",
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
    - Training: cfg.data.train_path + cfg.data.train_image/train_label/train_mask
    - Validation: cfg.data.val_path + cfg.data.val_image/val_label/val_mask
    - Testing: cfg.test.data.test_path + cfg.test.data.test_image/test_label/test_mask
    - Tuning: cfg.tune.data.test_path + cfg.tune.data.tune_image/tune_label/tune_mask

    Note: Test data belongs in cfg.test.data, tuning data in cfg.tune.data

    Args:
        cfg: Config object to resolve paths for

    Returns:
        Config object with resolved paths (same object, modified in-place)

    Example:
        >>> cfg.data.train_path = "/data/barcode/"
        >>> cfg.data.train_image = ["PT37/*_raw.tif", "file.tif"]
        >>> resolve_data_paths(cfg)
        >>> print(cfg.data.train_image)
        [
            '/data/barcode/PT37/img1_raw.tif',
            '/data/barcode/PT37/img2_raw.tif',
            '/data/barcode/file.tif'
        ]

        >>> cfg.test.data.test_path = "/data/test/"
        >>> cfg.test.data.test_image = ["volume_*.tif"]
        >>> resolve_data_paths(cfg)
        >>> print(cfg.test.data.test_image)
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
    train_base = cfg.data.train_path if cfg.data.train_path else ""
    cfg.data.train_image = _combine_path(train_base, cfg.data.train_image)
    cfg.data.train_label = _combine_path(train_base, cfg.data.train_label)
    cfg.data.train_mask = _combine_path(train_base, cfg.data.train_mask)
    train_json_resolved = _combine_path(train_base, cfg.data.train_json)
    if isinstance(train_json_resolved, list):
        cfg.data.train_json = train_json_resolved[0] if train_json_resolved else None
    else:
        cfg.data.train_json = train_json_resolved

    # Resolve validation paths (always expand globs, use val_path as base if available)
    val_base = cfg.data.val_path if cfg.data.val_path else ""
    cfg.data.val_image = _combine_path(val_base, cfg.data.val_image)
    cfg.data.val_label = _combine_path(val_base, cfg.data.val_label)
    cfg.data.val_mask = _combine_path(val_base, cfg.data.val_mask)
    val_json_resolved = _combine_path(val_base, cfg.data.val_json)
    if isinstance(val_json_resolved, list):
        cfg.data.val_json = val_json_resolved[0] if val_json_resolved else None
    else:
        cfg.data.val_json = val_json_resolved

    # Resolve test data paths (cfg.test.data.test_*)
    if cfg.test is not None:
        test_data = cfg.test.data
        test_path_value = getattr(test_data, "test_path", "")
        test_base = test_path_value if isinstance(test_path_value, str) else ""
        test_data.test_image = _combine_path(test_base, test_data.test_image)
        test_data.test_label = _combine_path(test_base, test_data.test_label)
        test_data.test_mask = _combine_path(test_base, test_data.test_mask)

    # Resolve tuning data paths (cfg.tune.data.tune_*)
    if cfg.tune is not None:
        tune_data = cfg.tune.data
        tune_path_value = getattr(tune_data, "test_path", "")
        tune_base = tune_path_value if isinstance(tune_path_value, str) else ""
        tune_data.tune_image = _combine_path(tune_base, tune_data.tune_image)
        tune_data.tune_label = _combine_path(tune_base, tune_data.tune_label)
        tune_data.tune_mask = _combine_path(tune_base, tune_data.tune_mask)

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

    # 1) System profile resolution
    system_profiles = getattr(shared, "system_profiles", {}) or {}
    if system_profiles:
        if mode_normalized == "train":
            selector = getattr(getattr(cfg, "train", None), "system", None)
            target_attr = "training"
            default_profile = "train_default"
        elif mode_normalized == "test":
            selector = (
                getattr(cfg.test, "system", None)
                if hasattr(cfg, "test") and cfg.test is not None
                else None
            )
            target_attr = "inference"
            default_profile = "infer_default"
        else:  # tune
            selector = (
                getattr(cfg.tune, "system", None)
                if hasattr(cfg, "tune") and cfg.tune is not None
                else None
            )
            target_attr = "inference"
            default_profile = "infer_default"

        profile_name = getattr(selector, "profile", None) or (
            default_profile if default_profile in system_profiles else None
        )
        profile_cfg = system_profiles.get(profile_name) if profile_name else None
        profile_overrides = getattr(selector, "overrides", {}) if selector else {}

        if profile_cfg is not None or profile_overrides:
            merged_profile = _to_omegaconf(profile_cfg)
            merged_profile = OmegaConf.merge(merged_profile, _to_omegaconf(profile_overrides))
            profile_payload = OmegaConf.to_container(merged_profile, resolve=True)
            if not isinstance(profile_payload, dict):
                profile_payload = {}
            profile_seed = profile_payload.pop("seed", None)

            target_cfg = getattr(cfg.system, target_attr)
            setattr(
                cfg.system,
                target_attr,
                _merge_dataclass(target_cfg, profile_payload),
            )
            if profile_seed is not None:
                cfg.system.seed = int(profile_seed)

    # 2) Data transform profile resolution (stage-specific)
    data_profiles = getattr(shared, "data_transform_profiles", {}) or {}
    if data_profiles:
        def _apply_data_profile(stage_data_cfg: Any) -> None:
            if stage_data_cfg is None:
                return

            profile_name = getattr(stage_data_cfg, "transform_profile", None) or (
                "default" if "default" in data_profiles else None
            )
            profile_cfg = data_profiles.get(profile_name) if profile_name else None
            profile_overrides = getattr(stage_data_cfg, "transform_overrides", {}) or {}

            if profile_cfg is None and not profile_overrides:
                return

            profile_conf = _to_omegaconf(profile_cfg) if profile_cfg is not None else OmegaConf.create({})
            profile_data_transform = profile_conf.get("data_transform", None)
            profile_image_transform = profile_conf.get("image_transform", None)
            profile_mask_transform = profile_conf.get("mask_transform", None)
            profile_nnunet_preprocessing = profile_conf.get("nnunet_preprocessing", None)

            data_transform_override = profile_overrides.get("data_transform", {})
            image_transform_override = profile_overrides.get("image_transform", {})
            mask_transform_override = profile_overrides.get("mask_transform", {})
            nnunet_pre_override = profile_overrides.get("nnunet_preprocessing", {})

            if (
                hasattr(stage_data_cfg, "data_transform")
                and profile_data_transform is not None
            ):
                stage_data_cfg.data_transform = _merge_dataclass(
                    stage_data_cfg.data_transform,
                    profile_data_transform,
                    data_transform_override,
                )
            elif hasattr(stage_data_cfg, "data_transform") and data_transform_override:
                stage_data_cfg.data_transform = _merge_dataclass(
                    stage_data_cfg.data_transform,
                    data_transform_override,
                )

            if (
                hasattr(stage_data_cfg, "image_transform")
                and profile_image_transform is not None
            ):
                stage_data_cfg.image_transform = _merge_dataclass(
                    stage_data_cfg.image_transform,
                    profile_image_transform,
                    image_transform_override,
                )
            elif hasattr(stage_data_cfg, "image_transform") and image_transform_override:
                stage_data_cfg.image_transform = _merge_dataclass(
                    stage_data_cfg.image_transform,
                    image_transform_override,
                )

            if (
                hasattr(stage_data_cfg, "mask_transform")
                and profile_mask_transform is not None
            ):
                stage_data_cfg.mask_transform = _merge_dataclass(
                    stage_data_cfg.mask_transform,
                    profile_mask_transform,
                    mask_transform_override,
                )
            elif hasattr(stage_data_cfg, "mask_transform") and mask_transform_override:
                stage_data_cfg.mask_transform = _merge_dataclass(
                    stage_data_cfg.mask_transform,
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
            elif hasattr(stage_data_cfg, "nnunet_preprocessing") and nnunet_pre_override:
                stage_data_cfg.nnunet_preprocessing = _merge_dataclass(
                    stage_data_cfg.nnunet_preprocessing,
                    nnunet_pre_override,
                )

        _apply_data_profile(getattr(cfg, "data", None))
        _apply_data_profile(getattr(getattr(cfg, "test", None), "data", None))
        _apply_data_profile(getattr(getattr(cfg, "tune", None), "data", None))

    # 3) Inference profile resolution (test/tune only)
    inference_profiles = getattr(shared, "inference_profiles", {}) or {}
    if inference_profiles and mode_normalized in {"test", "tune"}:
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

        profile_name = getattr(selector, "profile", None) or (
            "default" if "default" in inference_profiles else None
        )
        profile_cfg = inference_profiles.get(profile_name) if profile_name else None
        profile_overrides = getattr(selector, "overrides", {}) if selector else {}

        if profile_cfg is not None or profile_overrides:
            cfg.inference = _merge_dataclass(cfg.inference, profile_cfg, profile_overrides)

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
