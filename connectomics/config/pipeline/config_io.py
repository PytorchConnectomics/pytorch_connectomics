"""
Configuration I/O, validation, and path resolution for Hydra configuration system.

Provides helpers for loading, saving, validating, and manipulating configs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import warnings
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, ListConfig, OmegaConf

from ...data.processing.build import count_stacked_label_transform_channels
from ...models.architectures.registry import get_architecture_info
from ...utils.channel_slices import infer_min_required_channels
from ...utils.model_outputs import resolve_configured_output_head
from ..schema import Config
from ..schema.root import MergeContext
from .profile_engine import _YAML_PROFILE_ENGINE
from .stage_resolver import _collect_explicit_paths

# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Unconsumed-key warnings (Recommendation 3)
# ---------------------------------------------------------------------------


def _warn_unconsumed_keys(yaml_conf: DictConfig) -> None:
    """Warn about top-level YAML keys that don't match any Config dataclass field.

    Called after profile engine cleanup so only unconsumed keys remain.
    Catches typos like ``mode`` instead of ``model``.
    """
    known_fields = {f.name for f in dataclasses.fields(Config)}
    # Also allow internal keys that are used by the config system
    known_fields.update({"_base_"})

    for key in yaml_conf.keys():
        if str(key) not in known_fields:
            warnings.warn(
                f"Unconsumed top-level config key '{key}' is not a known Config field. "
                f"Known fields: {sorted(known_fields - {'_base_'})}. "
                f"This key will be ignored. Did you mean one of the known fields?",
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


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

    _warn_unconsumed_keys(yaml_conf)

    explicit_field_paths = _collect_explicit_paths(yaml_conf)

    # Merge with structured config defaults
    default_conf = OmegaConf.structured(Config)
    merged = OmegaConf.merge(default_conf, yaml_conf)

    # Convert to dataclass instance
    cfg = OmegaConf.to_object(merged)

    merge_context = getattr(cfg, "_merge_context", None)
    if not isinstance(merge_context, MergeContext):
        merge_context = MergeContext()
        setattr(cfg, "_merge_context", merge_context)
    merge_context.explicit_field_paths = explicit_field_paths
    return cfg


# ---------------------------------------------------------------------------
# Config save / merge / CLI
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Config serialization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Config validation (with cross-section warnings from Recommendation 3)
# ---------------------------------------------------------------------------


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
    model_heads = getattr(cfg.model, "heads", None) or {}
    inference_cfg = getattr(cfg, "inference", None)
    inference_head = getattr(inference_cfg, "head", None) if inference_cfg is not None else None
    images_cfg = getattr(getattr(getattr(cfg, "monitor", None), "logging", None), "images", None)
    visualization_head = getattr(images_cfg, "head", None) if images_cfg is not None else None
    if model_heads:
        arch_type = getattr(cfg.model.arch, "type", "")
        if arch_type not in {"mednext", "mednext_custom"}:
            raise ValueError(
                "model.heads is currently only supported for architecture "
                f"'mednext' or 'mednext_custom' (got '{arch_type}')."
            )

        for head_name, head_cfg in model_heads.items():
            head_out_channels = int(getattr(head_cfg, "out_channels", 0))
            head_num_blocks = int(getattr(head_cfg, "num_blocks", 0))
            head_hidden_channels = getattr(head_cfg, "hidden_channels", None)
            if head_hidden_channels is not None:
                head_hidden_channels = int(head_hidden_channels)
            if head_out_channels <= 0:
                raise ValueError(
                    f"model.heads.{head_name}.out_channels must be positive "
                    f"(got {head_out_channels})"
                )
            if head_num_blocks < 0:
                raise ValueError(
                    f"model.heads.{head_name}.num_blocks must be non-negative "
                    f"(got {head_num_blocks})"
                )
            if head_hidden_channels is not None and head_hidden_channels <= 0:
                raise ValueError(
                    f"model.heads.{head_name}.hidden_channels must be positive "
                    f"(got {head_hidden_channels})"
                )

        primary_head = getattr(cfg.model, "primary_head", None)
        if primary_head is not None and primary_head not in model_heads:
            raise ValueError(
                f"model.primary_head='{primary_head}' is not present in model.heads "
                f"({sorted(model_heads.keys())})."
            )
        if inference_head is not None and inference_head not in model_heads:
            raise ValueError(
                f"inference.head='{inference_head}' is not present in model.heads "
                f"({sorted(model_heads.keys())})."
            )
        if (
            visualization_head is not None
            and visualization_head != "all"
            and visualization_head not in model_heads
        ):
            raise ValueError(
                f"monitor.logging.images.head='{visualization_head}' is not present in "
                f"model.heads ({sorted(model_heads.keys())})."
            )
    elif inference_head is not None:
        raise ValueError("inference.head requires model.heads to be configured")
    elif visualization_head is not None:
        raise ValueError("monitor.logging.images.head requires model.heads to be configured")
    if len(cfg.model.input_size) not in [2, 3]:
        raise ValueError(
            f"model.input_size must be 2D or 3D (got length {len(cfg.model.input_size)})"
        )

    # System validation
    if cfg.system.num_workers < 0:
        raise ValueError("system.num_workers must be non-negative")

    # Data validation
    if len(cfg.data.dataloader.patch_size) not in [2, 3]:
        raise ValueError(
            "data.dataloader.patch_size must be 2D or 3D "
            f"(got length {len(cfg.data.dataloader.patch_size)})"
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
    if (
        model_heads
        and model_loss_cfg is not None
        and getattr(model_loss_cfg, "deep_supervision", False)
    ):
        raise ValueError(
            "model.heads is not yet compatible with model.loss.deep_supervision=True. "
            "Disable deep supervision for MedNeXt multi-head models."
        )
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
            for head_key in ("pred_head", "pred2_head"):
                head_name = entry.get(head_key)
                if head_name is None:
                    continue
                if not isinstance(head_name, str) or not head_name.strip():
                    raise ValueError(
                        f"model.loss.losses[{i}].{head_key} must be a non-empty string"
                    )
                if not model_heads:
                    raise ValueError(
                        f"model.loss.losses[{i}].{head_key} requires model.heads to be configured"
                    )
                if head_name not in model_heads:
                    raise ValueError(
                        f"model.loss.losses[{i}].{head_key}='{head_name}' is not present in "
                        f"model.heads ({sorted(model_heads.keys())})"
                    )
            if model_heads and len(model_heads) > 1:
                resolved_pred_head = entry.get(
                    "pred_head", getattr(cfg.model, "primary_head", None)
                )
                if resolved_pred_head is None:
                    raise ValueError(
                        f"model.loss.losses[{i}] must define pred_head or model.primary_head "
                        f"when model.heads has multiple entries ({sorted(model_heads.keys())})"
                    )

    # --- Cross-section coherence validation (UX 2.4 #3) ---
    _validate_cross_section_coherence(cfg)


def _architecture_supports_deep_supervision(arch_type: str) -> bool:
    """Infer deep-supervision support from architecture registry metadata."""
    arch_info = get_architecture_info().get(arch_type)
    if arch_info is None:
        return True

    module_name = arch_info.get("module", "")
    # Current MONAI wrappers are single-scale and do not expose deep supervision.
    return not module_name.endswith("monai_models")


def _validate_cross_section_coherence(cfg: Config) -> None:
    """Validate resolved cross-section coherence and raise clear errors."""
    # 1) model.input_size vs data.dataloader.patch_size mismatch
    model_input = list(cfg.model.input_size)
    patch_size = list(cfg.data.dataloader.patch_size)
    if model_input != patch_size:
        raise ValueError(
            "Cross-section validation failed: model.input_size "
            f"{model_input} must match data.dataloader.patch_size {patch_size}."
        )

    # 2) output-channel coherence vs loss/label/decoding/activation channel usage
    out_channels = cfg.model.out_channels
    model_heads = getattr(cfg.model, "heads", None) or {}
    primary_head = getattr(cfg.model, "primary_head", None)
    sole_head = next(iter(model_heads.keys())) if len(model_heads) == 1 else None
    required_output_channels: List[tuple[str, int]] = []

    label_cfg = getattr(cfg.data, "label_transform", None)
    stacked_label_channels = (
        count_stacked_label_transform_channels(label_cfg) if label_cfg is not None else None
    )

    def _resolve_selector_head(entry: Any, *, selector_key: str) -> Optional[str]:
        if selector_key == "pred2_slice":
            selector_head = entry.get("pred2_head", entry.get("pred_head", None))
        else:
            selector_head = entry.get("pred_head", None)

        if selector_head is None:
            selector_head = primary_head or sole_head
        return selector_head

    def _validate_head_channel_capacity(
        *,
        selector_key: str,
        selector_head: str,
        min_channels: int,
        loss_idx: int,
    ) -> bool:
        if selector_head not in model_heads:
            return False

        head_channels = int(getattr(model_heads[selector_head], "out_channels", 0))
        if min_channels > head_channels:
            raise ValueError(
                "Cross-section validation failed: "
                f"model.loss.losses[{loss_idx}].{selector_key} requires at least "
                f"{min_channels} channels in head '{selector_head}', but "
                f"model.heads.{selector_head}.out_channels is {head_channels}."
            )
        return True

    def _validate_label_channel_capacity(selector_value: Any, *, path: str) -> None:
        min_channels = infer_min_required_channels(selector_value, context=path)
        if min_channels is None:
            return

        if stacked_label_channels is not None:
            if min_channels > stacked_label_channels:
                raise ValueError(
                    "Cross-section validation failed: "
                    f"{path} requires at least {min_channels} stacked label channels, but "
                    f"data.label_transform.targets produces {stacked_label_channels}."
                )
            return

        if not model_heads:
            required_output_channels.append((path, min_channels))

    # 2a) Loss channel selectors
    model_loss_cfg = getattr(cfg.model, "loss", None)
    losses_cfg = getattr(model_loss_cfg, "losses", None) if model_loss_cfg else None
    if losses_cfg is not None:
        for i, entry in enumerate(losses_cfg):
            if not isinstance(entry, dict):
                continue
            selector_keys = ("pred_slice", "target_slice", "mask_slice", "pred2_slice")
            for selector_key in selector_keys:
                min_channels = infer_min_required_channels(
                    entry.get(selector_key),
                    context=f"model.loss.losses[{i}].{selector_key}",
                )
                if min_channels is not None:
                    path = f"model.loss.losses[{i}].{selector_key}"
                    if selector_key in {"pred_slice", "pred2_slice"} and model_heads:
                        selector_head = _resolve_selector_head(entry, selector_key=selector_key)
                        if selector_head is not None and _validate_head_channel_capacity(
                            selector_key=selector_key,
                            selector_head=selector_head,
                            min_channels=min_channels,
                            loss_idx=i,
                        ):
                            continue
                    if selector_key in {"target_slice", "mask_slice"}:
                        _validate_label_channel_capacity(entry.get(selector_key), path=path)
                        continue
                    required_output_channels.append((path, min_channels))

    # 2b) Label transform targets (legacy lower-bound expectation for flat outputs)
    if not model_heads and stacked_label_channels:
        required_output_channels.append(("data.label_transform.targets", stacked_label_channels))

    # 2c) Explicit head-to-label routing
    if model_heads:
        for head_name, head_cfg in model_heads.items():
            target_slice = getattr(head_cfg, "target_slice", None)
            if target_slice is None:
                continue
            _validate_label_channel_capacity(
                target_slice,
                path=f"model.heads.{head_name}.target_slice",
            )

    # 2d) Decoding kwargs channel selectors (*_channels)
    decoding_cfg = getattr(cfg.inference, "decoding", None)
    decode_has_channel_selection = False
    decode_output_head = None
    decode_available_channels = out_channels
    decode_channel_scope = "model output"
    if isinstance(decoding_cfg, list):
        for i, decode_step in enumerate(decoding_cfg):
            kwargs = getattr(decode_step, "kwargs", None)
            if not isinstance(kwargs, dict):
                continue
            if any(key.endswith("_channels") for key in kwargs):
                decode_has_channel_selection = True
                break

        if model_heads and decode_has_channel_selection:
            decode_output_head = resolve_configured_output_head(
                cfg,
                purpose="decode channel selection",
                allow_none=True,
            )
            if len(model_heads) > 1 and decode_output_head is None:
                raise ValueError(
                    "Cross-section validation failed: decode channel selectors require "
                    "inference.head or model.primary_head when model.heads has multiple "
                    f"entries ({sorted(model_heads.keys())})."
                )
            if decode_output_head in model_heads:
                decode_available_channels = int(
                    getattr(model_heads[decode_output_head], "out_channels", out_channels)
                )
                decode_channel_scope = f"head '{decode_output_head}'"

        for i, decode_step in enumerate(decoding_cfg):
            kwargs = getattr(decode_step, "kwargs", None)
            if not isinstance(kwargs, dict):
                continue
            for key, value in kwargs.items():
                if not key.endswith("_channels"):
                    continue
                min_channels = infer_min_required_channels(
                    value,
                    context=f"inference.decoding[{i}].kwargs.{key}",
                )
                if min_channels is not None:
                    path = f"inference.decoding[{i}].kwargs.{key}"
                    if model_heads and decode_has_channel_selection:
                        if min_channels > decode_available_channels:
                            raise ValueError(
                                "Cross-section validation failed: "
                                f"{path} requires at least {min_channels} channels in "
                                f"{decode_channel_scope}, but only "
                                f"{decode_available_channels} are available."
                            )
                        continue
                    required_output_channels.append((path, min_channels))

    # 2e) TTA channel selectors
    tta_cfg = getattr(cfg.inference, "test_time_augmentation", None)
    channel_activations = getattr(tta_cfg, "channel_activations", None) if tta_cfg else None
    select_channel = getattr(tta_cfg, "select_channel", None) if tta_cfg else None
    tta_has_channel_selection = bool(channel_activations) or select_channel is not None
    tta_output_head = (
        resolve_configured_output_head(
            cfg,
            purpose="TTA channel selection",
            allow_none=True,
        )
        if model_heads
        else None
    )
    if (
        model_heads
        and len(model_heads) > 1
        and tta_has_channel_selection
        and tta_output_head is None
    ):
        raise ValueError(
            "Cross-section validation failed: TTA channel selectors require inference.head "
            "or model.primary_head when model.heads has multiple entries "
            f"({sorted(model_heads.keys())})."
        )
    tta_available_channels = (
        int(getattr(model_heads[tta_output_head], "out_channels", out_channels))
        if tta_output_head in model_heads
        else out_channels
    )
    tta_channel_scope = (
        f"head '{tta_output_head}'" if tta_output_head in model_heads else "model output"
    )

    def _validate_tta_channel_capacity(selector_value: Any, *, path: str) -> None:
        min_selector_channels = infer_min_required_channels(
            selector_value,
            context=path,
        )
        if min_selector_channels is None:
            return
        if min_selector_channels > tta_available_channels:
            raise ValueError(
                "Cross-section validation failed: "
                f"{path} requires at least {min_selector_channels} channels in {tta_channel_scope}, "
                f"but only {tta_available_channels} are available."
            )

    if isinstance(channel_activations, list):
        for i, spec in enumerate(channel_activations):
            if not isinstance(spec, dict):
                raise ValueError(
                    "Cross-section validation failed: "
                    f"inference.test_time_augmentation.channel_activations[{i}] "
                    "must be a mapping with 'channels' and 'activation'."
                )
            if "channels" not in spec or "activation" not in spec:
                raise ValueError(
                    "Cross-section validation failed: "
                    f"inference.test_time_augmentation.channel_activations[{i}] "
                    "must define both 'channels' and 'activation'."
                )
            _validate_tta_channel_capacity(
                spec["channels"],
                path=f"inference.test_time_augmentation.channel_activations[{i}].channels",
            )
    _validate_tta_channel_capacity(
        select_channel,
        path="inference.test_time_augmentation.select_channel",
    )

    if required_output_channels:
        required_max = max(req for _, req in required_output_channels)
        if required_max > out_channels:
            details = ", ".join(
                f"{path} needs >= {req}"
                for path, req in sorted(required_output_channels, key=lambda x: x[1], reverse=True)
            )
            raise ValueError(
                "Cross-section validation failed: model.out_channels is "
                f"{out_channels}, but resolved pipeline components require at least "
                f"{required_max} channels ({details})."
            )

    # 3) deep_supervision=True with architectures that don't support it
    deep_supervision = (
        getattr(model_loss_cfg, "deep_supervision", False) if model_loss_cfg else False
    )
    if deep_supervision:
        arch_type = getattr(cfg.model.arch, "type", "")
        if not _architecture_supports_deep_supervision(arch_type):
            raise ValueError(
                "Cross-section validation failed: model.loss.deep_supervision=True but "
                f"architecture '{arch_type}' does not support deep supervision. "
                "Use MedNeXt/RSUNet or disable deep supervision."
            )


# ---------------------------------------------------------------------------
# Config hashing and naming
# ---------------------------------------------------------------------------


def get_config_hash(cfg: Config) -> str:
    """
    Generate a hash string for the configuration.

    Useful for experiment tracking and reproducibility.

    Args:
        cfg: Config object

    Returns:
        Hash string
    """
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


# ---------------------------------------------------------------------------
# Data path resolution
# ---------------------------------------------------------------------------


def resolve_data_paths(cfg: Config) -> Config:
    """
    Resolve data paths by combining split base paths with relative file paths.

    This function modifies the config in-place by:
    1. Prepending base paths to relative file paths
    2. Expanding glob patterns to actual file lists
    3. Flattening nested lists from glob expansion

    Supported split paths:
    - Training: cfg.data.train.path + cfg.data.train.image/label/mask
    - Validation: cfg.data.val.path + cfg.data.val.image/label/mask
    - Inference/Test: cfg.data.test.path + cfg.data.test.image/label/mask

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

        >>> cfg.data.test.path = "/data/test/"
        >>> cfg.data.test.image = ["volume_*.tif"]
        >>> resolve_data_paths(cfg)
        >>> print(cfg.data.test.image)
        ['/data/test/volume_1.tif', '/data/test/volume_2.tif']
    """

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
            except ValueError:
                # Not a number, try filename match
                matching = [
                    f for f in expanded if Path(f).name == selector or Path(f).stem == selector
                ]
                if not matching:
                    # Try partial match
                    matching = [f for f in expanded if selector in Path(f).name]
                if matching:
                    return matching[0]
                else:
                    raise ValueError(
                        f"Glob selector '{selector}' did not match any file in pattern "
                        f"'{glob_pattern}' ({len(expanded)} candidates)."
                    )
            if index < -len(expanded) or index >= len(expanded):
                raise ValueError(
                    f"Glob selector index out of range: '{file_path}' resolved to "
                    f"{len(expanded)} files but requested index {index}."
                )
            return expanded[index]

        elif "*" in file_path or "?" in file_path:
            # Standard glob without selector
            expanded = sorted(glob(file_path))
            if expanded:
                return expanded
            else:
                # No matches - return original pattern (will be caught by validation)
                return file_path

        return file_path

    # Prepend root_path to each split's path when the split path is relative
    root_path = getattr(cfg.data, "root_path", "") or ""
    if root_path:
        for split_attr in ("train", "val", "test"):
            split_cfg = getattr(cfg.data, split_attr, None)
            if split_cfg is None:
                continue
            sp = getattr(split_cfg, "path", "") or ""
            if not sp:
                split_cfg.path = root_path
            elif not os.path.isabs(sp):
                split_cfg.path = os.path.join(root_path, sp)

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

    def _resolve_split_paths(split_cfg):
        split_path_value = getattr(split_cfg, "path", "")
        split_base = split_path_value if isinstance(split_path_value, str) else ""
        split_cfg.image = _combine_path(split_base, split_cfg.image)
        split_cfg.label = _combine_path(split_base, split_cfg.label)
        split_cfg.mask = _combine_path(split_base, split_cfg.mask)

    # Resolve inference/test paths from merged runtime cfg.data.
    if getattr(cfg.data, "test", None) is not None:
        _resolve_split_paths(cfg.data.test)

    return cfg
