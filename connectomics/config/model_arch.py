"""
Architecture profile resolution for ModelConfig.

This module provides an extensible OOP registry for architecture families.
Each family owns:
- default mapping to runtime ModelConfig keys
- optional variant handling
- validation of profile keys
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Any, Dict, Optional, Type

from omegaconf import DictConfig, OmegaConf

from .hydra_config import ModelConfig


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, DictConfig):
        container = OmegaConf.to_container(value, resolve=True)
        return container if isinstance(container, dict) else {}
    if isinstance(value, dict):
        return dict(value)
    return {}


_MODEL_CONFIG_KEYS = {f.name for f in fields(ModelConfig)}
_PROFILE_RESERVED_KEYS = {"type", "variant", "params", "description"}


class BaseArchSpec(ABC):
    """Base class for architecture family specs."""

    type_name: str = ""
    aliases: tuple[str, ...] = ()
    key_aliases: Dict[str, str] = {}

    @classmethod
    def supported_types(cls) -> tuple[str, ...]:
        return (cls.type_name, *cls.aliases)

    @abstractmethod
    def default_arch_type(self, profile: Dict[str, Any]) -> str:
        """Return runtime ModelConfig.arch.type default for this family."""

    def apply_variant(self, profile: Dict[str, Any], model_patch: Dict[str, Any]) -> None:
        """Optional variant handling for a family."""
        _ = profile
        _ = model_patch

    def remap_keys(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        remapped: Dict[str, Any] = {}
        for key, value in overrides.items():
            remapped_key = self.key_aliases.get(key, key)
            remapped[remapped_key] = value
        return remapped

    def validate_keys(self, profile_name: str, overrides: Dict[str, Any]) -> None:
        invalid = sorted(k for k in overrides.keys() if k not in _MODEL_CONFIG_KEYS)
        if invalid:
            raise ValueError(
                f"Arch profile '{profile_name}' has invalid model keys: {invalid}. "
                "Only ModelConfig keys are allowed."
            )

    def build_model_patch(self, profile_name: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        model_patch: Dict[str, Any] = {"arch": {"type": self.default_arch_type(profile)}}
        self.apply_variant(profile, model_patch)

        params = profile.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(
                f"Arch profile '{profile_name}' has non-mapping 'params' ({type(params)})."
            )

        direct_overrides = {k: v for k, v in profile.items() if k not in _PROFILE_RESERVED_KEYS}

        merged_overrides = {**direct_overrides, **params}
        merged_overrides = self.remap_keys(merged_overrides)
        self.validate_keys(profile_name, merged_overrides)
        model_patch.update(merged_overrides)
        return model_patch


class MedNeXtSpec(BaseArchSpec):
    type_name = "mednext"
    aliases = ("mednext_custom",)
    key_aliases = {}
    _allowed_variants = {"S", "B", "M", "L"}

    def default_arch_type(self, profile: Dict[str, Any]) -> str:
        return "mednext"

    def apply_variant(self, profile: Dict[str, Any], model_patch: Dict[str, Any]) -> None:
        variant = profile.get("variant")
        if variant is None:
            return
        variant_str = str(variant)
        if variant_str not in self._allowed_variants:
            raise ValueError(
                f"Invalid MedNeXt variant '{variant_str}'. "
                f"Expected one of {sorted(self._allowed_variants)}."
            )
        mednext_patch = model_patch.setdefault("mednext", {})
        mednext_patch["size"] = variant_str

    def remap_keys(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        remapped = dict(overrides)
        mednext_patch = remapped.setdefault("mednext", {})
        if "kernel_size" in remapped:
            mednext_patch["kernel_size"] = remapped.pop("kernel_size")
        if "size" in remapped:
            mednext_patch["size"] = remapped.pop("size")
        return remapped

    def validate_keys(self, profile_name: str, overrides: Dict[str, Any]) -> None:
        top_level_invalid = sorted(k for k in overrides.keys() if k not in _MODEL_CONFIG_KEYS)
        if top_level_invalid:
            raise ValueError(
                f"Arch profile '{profile_name}' has invalid model keys: {top_level_invalid}. "
                "Only ModelConfig keys are allowed."
            )
        if "mednext" in overrides and not isinstance(overrides["mednext"], dict):
            raise ValueError(f"Arch profile '{profile_name}' key 'mednext' must be a mapping.")


class RSUNetSpec(BaseArchSpec):
    type_name = "rsunet"

    def default_arch_type(self, profile: Dict[str, Any]) -> str:
        return "rsunet"

    def remap_keys(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        remapped = dict(overrides)
        rsunet_patch = remapped.setdefault("rsunet", {})
        if "filters" in remapped:
            rsunet_patch["width"] = remapped.pop("filters")
        for old_key, new_key in (
            ("rsunet_norm", "norm"),
            ("rsunet_activation", "activation"),
            ("rsunet_num_groups", "num_groups"),
            ("rsunet_down_factors", "down_factors"),
            ("rsunet_depth_2d", "depth_2d"),
            ("rsunet_kernel_2d", "kernel_2d"),
            ("rsunet_act_negative_slope", "act_negative_slope"),
            ("rsunet_act_init", "act_init"),
        ):
            if old_key in remapped:
                rsunet_patch[new_key] = remapped.pop(old_key)
        return remapped

    def validate_keys(self, profile_name: str, overrides: Dict[str, Any]) -> None:
        top_level_invalid = sorted(k for k in overrides.keys() if k not in _MODEL_CONFIG_KEYS)
        if top_level_invalid:
            raise ValueError(
                f"Arch profile '{profile_name}' has invalid model keys: {top_level_invalid}. "
                "Only ModelConfig keys are allowed."
            )
        if "rsunet" in overrides and not isinstance(overrides["rsunet"], dict):
            raise ValueError(f"Arch profile '{profile_name}' key 'rsunet' must be a mapping.")


class MonaiUNetSpec(BaseArchSpec):
    type_name = "monai_unet"

    def default_arch_type(self, profile: Dict[str, Any]) -> str:
        return "monai_unet"

    def remap_keys(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        remapped = dict(overrides)
        monai_patch = remapped.setdefault("monai", {})
        for key in (
            "filters",
            "dropout",
            "norm",
            "num_groups",
            "activation",
            "spatial_dims",
            "num_res_units",
            "kernel_size",
            "strides",
            "act",
            "upsample",
            "upsample_mode",
            "upsample_interp_mode",
            "upsample_align_corners",
        ):
            if key in remapped:
                monai_patch[key] = remapped.pop(key)
        return remapped

    def validate_keys(self, profile_name: str, overrides: Dict[str, Any]) -> None:
        top_level_invalid = sorted(k for k in overrides.keys() if k not in _MODEL_CONFIG_KEYS)
        if top_level_invalid:
            raise ValueError(
                f"Arch profile '{profile_name}' has invalid model keys: {top_level_invalid}. "
                "Only ModelConfig keys are allowed."
            )
        if "monai" in overrides and not isinstance(overrides["monai"], dict):
            raise ValueError(f"Arch profile '{profile_name}' key 'monai' must be a mapping.")


class MonaiBasicUNet3DSpec(BaseArchSpec):
    type_name = "monai_basic_unet3d"

    def default_arch_type(self, profile: Dict[str, Any]) -> str:
        return "monai_basic_unet3d"

    def remap_keys(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        return MonaiUNetSpec().remap_keys(overrides)

    def validate_keys(self, profile_name: str, overrides: Dict[str, Any]) -> None:
        MonaiUNetSpec().validate_keys(profile_name, overrides)


class ArchSpecRegistry:
    """Registry mapping architecture type -> spec class."""

    def __init__(self) -> None:
        self._specs: Dict[str, Type[BaseArchSpec]] = {}

    def register(self, spec_cls: Type[BaseArchSpec]) -> None:
        for key in spec_cls.supported_types():
            key_normalized = key.strip().lower()
            if not key_normalized:
                continue
            self._specs[key_normalized] = spec_cls

    def create(self, type_name: str) -> Optional[BaseArchSpec]:
        key = type_name.strip().lower()
        spec_cls = self._specs.get(key)
        return spec_cls() if spec_cls is not None else None

    def available_types(self) -> list[str]:
        return sorted(self._specs.keys())


ARCH_SPEC_REGISTRY = ArchSpecRegistry()
for _spec in (MedNeXtSpec, RSUNetSpec, MonaiUNetSpec, MonaiBasicUNet3DSpec):
    ARCH_SPEC_REGISTRY.register(_spec)


def resolve_arch_profile_model_patch(profile_name: str, profile_value: Any) -> Dict[str, Any]:
    """Resolve a single arch profile to a ModelConfig patch dict."""
    profile = _to_plain_dict(profile_value)
    if not profile:
        raise ValueError(f"Arch profile '{profile_name}' must be a mapping.")

    arch_type = profile.get("type")
    if arch_type is None:
        raise ValueError(
            f"Arch profile '{profile_name}' must define 'type'."
        )

    spec = ARCH_SPEC_REGISTRY.create(str(arch_type))
    if spec is None:
        available = ", ".join(ARCH_SPEC_REGISTRY.available_types())
        raise ValueError(
            f"Unknown arch profile type '{arch_type}' in '{profile_name}'. "
            f"Available types: [{available}]"
        )

    return spec.build_model_patch(profile_name, profile)

