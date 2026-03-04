"""
YAML profile engine for Hydra configuration system.

Contains profile applier classes, declarative registration tables,
and the engine that resolves profile references in YAML configs.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, ListConfig, OmegaConf

from .model_arch import resolve_arch_profile_model_patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_into_path(yaml_conf: DictConfig, target_path: str, profile_value: Any) -> None:
    """Merge profile payload into target path, keeping explicit config overrides."""
    existing = OmegaConf.select(yaml_conf, target_path)
    if existing is None:
        OmegaConf.update(yaml_conf, target_path, profile_value, force_add=True)
        return

    merged = OmegaConf.merge(profile_value, existing)
    OmegaConf.update(yaml_conf, target_path, merged, force_add=True)


def _delete_config_path(yaml_conf: DictConfig, path: str) -> None:
    """Delete a top-level or dotted key path from an OmegaConf mapping if present."""
    if not path:
        return

    if "." not in path:
        if path in yaml_conf:
            del yaml_conf[path]
        return

    parts = [p for p in path.split(".") if p]
    if not parts:
        return

    parent: Any = yaml_conf
    for part in parts[:-1]:
        if not isinstance(parent, DictConfig) or part not in parent:
            return
        parent = parent[part]

    leaf = parts[-1]
    if isinstance(parent, DictConfig) and leaf in parent:
        del parent[leaf]


_CANONICAL_STAGE_KEYS: set[str] = {"default", "train", "test", "tune"}
_ALLOWED_SELECTOR_PATHS: set[str] = set()
_PROFILE_PARENT_RELS: set[str] = set()


def _normalize_selector_path(path: str) -> str:
    return re.sub(r"\[\d+\]", "", path)


def _strip_stage_prefix(path: str) -> str:
    if not path:
        return path
    head, sep, tail = path.partition(".")
    if head in _CANONICAL_STAGE_KEYS:
        return tail if sep else ""
    return path


def _is_selector_candidate(path: str, key: str) -> bool:
    if key.endswith("_profile"):
        return True
    if key != "profile":
        return False

    # Treat only known `*.profile` parent sections as selector candidates.
    parent_path = path.rsplit(".", 1)[0] if "." in path else ""
    parent_rel = _strip_stage_prefix(_normalize_selector_path(parent_path))
    return parent_rel in _PROFILE_PARENT_RELS


def _collect_selector_candidate_paths(
    yaml_node: Any,
    path: str = "",
    in_profiles_registry: bool = False,
) -> set[str]:
    paths: set[str] = set()

    if isinstance(yaml_node, ListConfig):
        for idx in range(len(yaml_node)):
            child_path = f"{path}[{idx}]" if path else f"[{idx}]"
            # Use raw node access to avoid resolving interpolations while scanning.
            item = yaml_node._get_node(idx)
            paths.update(_collect_selector_candidate_paths(item, child_path, in_profiles_registry))
        return paths

    if not isinstance(yaml_node, (DictConfig, dict)):
        return paths

    for key in yaml_node.keys():
        key_str = str(key)
        child_path = f"{path}.{key_str}" if path else key_str
        child_value = (
            yaml_node._get_node(key) if isinstance(yaml_node, DictConfig) else yaml_node[key]
        )

        # Ignore selector-like keys inside profile registries (e.g. arch_profiles.* payloads).
        child_in_profiles_registry = in_profiles_registry or (path == "" and key_str.endswith("_profiles"))
        if not child_in_profiles_registry and _is_selector_candidate(child_path, key_str):
            if child_value not in (None, ""):
                paths.add(_normalize_selector_path(child_path))

        paths.update(
            _collect_selector_candidate_paths(
                child_value,
                child_path,
                child_in_profiles_registry,
            )
        )

    return paths


def _reject_noncanonical_selector_paths(yaml_conf: DictConfig) -> None:
    selector_candidates = _collect_selector_candidate_paths(yaml_conf)
    invalid_paths = sorted(
        path for path in selector_candidates if path not in _ALLOWED_SELECTOR_PATHS
    )
    if not invalid_paths:
        return

    allowed_display = ", ".join(sorted(_ALLOWED_SELECTOR_PATHS))
    invalid_display = ", ".join(invalid_paths)
    raise ValueError(
        "Non-canonical profile selector path(s) detected: "
        f"{invalid_display}. Allowed selector paths: [{allowed_display}]"
    )


def _reject_removed_stage_keys(yaml_conf: DictConfig) -> None:
    if "shared" not in yaml_conf:
        return
    raise ValueError(
        "Top-level 'shared' config section has been removed. "
        "Use top-level 'default' instead."
    )


# ---------------------------------------------------------------------------
# Profile applier classes
# ---------------------------------------------------------------------------


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

        if selected is None:
            return
        profiles = yaml_conf.get(self.profiles_key, None)
        if profiles is None:
            raise ValueError(
                f"Selector '{selected}' at '{selected_path}' requires "
                f"'{self.profiles_key}' to be defined in YAML."
            )

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

        # Pipeline profiles are selected via `default.pipeline_profile`, but some
        # profile payloads interpolate `${model.out_channels}` during early YAML
        # profile expansion. Materialize this key at root as soon as pipeline
        # profile is applied so later interpolation has a stable anchor.
        if selected_path == "default.pipeline_profile":
            selected_profile = profiles[selected]
            pipeline_out_ch = OmegaConf.select(selected_profile, "model.out_channels")
            if pipeline_out_ch is not None and OmegaConf.select(yaml_conf, "model.out_channels") is None:
                OmegaConf.update(
                    yaml_conf,
                    "model.out_channels",
                    pipeline_out_ch,
                    force_add=True,
                )


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

        if selected is None:
            return
        profiles = yaml_conf.get(self.profiles_key, None)
        if profiles is None:
            raise ValueError(
                f"Selector '{selected}' at '{selected_path}' requires "
                f"'{self.profiles_key}' to be defined in YAML."
            )

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

        if selected is None:
            return
        profiles = yaml_conf.get(self.profiles_key, None)
        if profiles is None:
            raise ValueError(
                f"Selector '{selected}' at '{selected_path}' requires "
                f"'{self.profiles_key}' to be defined in YAML."
            )

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


# ---------------------------------------------------------------------------
# Profile engine
# ---------------------------------------------------------------------------


class YamlProfileEngine:
    """Run a set of profile appliers and cleanup helper keys afterward."""

    def __init__(self, appliers: List[YamlProfileApplier]) -> None:
        self.appliers = appliers

    def apply(self, yaml_conf: DictConfig) -> DictConfig:
        if not isinstance(yaml_conf, DictConfig):
            return yaml_conf

        _reject_removed_stage_keys(yaml_conf)
        _reject_noncanonical_selector_paths(yaml_conf)

        cleanup_keys: set[str] = set()
        for applier in self.appliers:
            applier.apply(yaml_conf)
            cleanup_keys.update(applier.cleanup_keys)

        for key in cleanup_keys:
            _delete_config_path(yaml_conf, key)

        return yaml_conf


# ---------------------------------------------------------------------------
# Declarative profile registration tables
# ---------------------------------------------------------------------------

_STAGE_DEFAULT = "default"
_STAGE_TRAIN = "train"
_STAGE_TEST = "test"
_STAGE_TUNE = "tune"


def _stage_path(stage: str, rel_path: str) -> str:
    if stage:
        return f"{stage}.{rel_path}" if rel_path else stage
    return rel_path


# Each tuple:
# (phase, profiles_key, stages, selector_rel, target_rel, merge_into_existing)
_VALUE_PROFILE_FAMILIES: List[Tuple[str, str, Tuple[str, ...], str, str, bool]] = [
    ("before_arch", "pipeline_profiles", (_STAGE_DEFAULT,), "pipeline_profile", "", True),
    ("before_arch", "system_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN, _STAGE_TEST, _STAGE_TUNE), "system.profile", "system", True),
    ("after_arch", "augmentation_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN), "data.augmentation.profile", "data.augmentation", True),
    ("after_arch", "dataloader_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN, _STAGE_TEST, _STAGE_TUNE), "data.dataloader.profile", "data.dataloader", True),
    ("after_arch", "optimizer_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN), "optimization.profile", "optimization", True),
    ("after_arch", "loss_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN), "model.loss.profile", "model.loss.losses", False),
    ("after_arch", "label_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN), "data.label_transform.profile", "data.label_transform", False),
    ("after_arch", "decoding_profiles", (_STAGE_DEFAULT, _STAGE_TEST, _STAGE_TUNE), "inference.decoding_profile", "inference.decoding", False),
    ("after_arch", "activation_profiles", (_STAGE_DEFAULT, _STAGE_TEST, _STAGE_TUNE), "inference.test_time_augmentation.activation_profile", "inference.test_time_augmentation.channel_activations", False),
]


def _build_value_profile_specs() -> Dict[str, List[Tuple[List[str], str, str, bool, Tuple[str, ...]]]]:
    specs_by_phase: Dict[str, List[Tuple[List[str], str, str, bool, Tuple[str, ...]]]] = {
        "before_arch": [],
        "after_arch": [],
    }

    for phase, profiles_key, stages, selector_rel, target_rel, merge in _VALUE_PROFILE_FAMILIES:
        for stage in stages:
            selector_path = _stage_path(stage, selector_rel)
            target_path = _stage_path(stage, target_rel)
            specs_by_phase[phase].append(
                (
                    [selector_path],
                    profiles_key,
                    target_path,
                    merge,
                    (profiles_key, selector_path),
                )
            )

    return specs_by_phase


# Each tuple: (stages, selector_rel, target_rel)
_ARCH_PROFILE_FAMILY: Tuple[Tuple[str, ...], str, str] = (
    (_STAGE_DEFAULT, _STAGE_TRAIN, _STAGE_TEST, _STAGE_TUNE),
    "model.arch.profile",
    "model",
)


def _build_arch_profile_specs() -> List[Tuple[List[str], str, Tuple[str, ...]]]:
    stages, selector_rel, target_rel = _ARCH_PROFILE_FAMILY
    specs: List[Tuple[List[str], str, Tuple[str, ...]]] = []
    for stage in stages:
        selector_path = _stage_path(stage, selector_rel)
        target_path = _stage_path(stage, target_rel)
        specs.append(([selector_path], target_path, ("arch_profiles", selector_path)))
    return specs


# Each tuple: (profiles_key, stages, target_rel)
_REFERENCE_PROFILE_FAMILIES: List[Tuple[str, Tuple[str, ...], str]] = [
    ("loss_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN), "model.loss.losses"),
    ("label_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN), "data.label_transform"),
    ("decoding_profiles", (_STAGE_DEFAULT, _STAGE_TEST, _STAGE_TUNE), "inference.decoding"),
    ("activation_profiles", (_STAGE_DEFAULT, _STAGE_TEST, _STAGE_TUNE), "inference.test_time_augmentation.channel_activations"),
    ("augmentation_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN), "data.augmentation"),
    ("dataloader_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN, _STAGE_TEST, _STAGE_TUNE), "data.dataloader"),
    ("optimizer_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN), "optimization"),
    ("system_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN, _STAGE_TEST, _STAGE_TUNE), "system"),
]


def _build_reference_profile_specs() -> List[Tuple[str, List[str]]]:
    specs: List[Tuple[str, List[str]]] = []
    for profiles_key, stages, target_rel in _REFERENCE_PROFILE_FAMILIES:
        target_paths = [_stage_path(stage, target_rel) for stage in stages]
        specs.append((profiles_key, target_paths))
    return specs


# Each tuple: (profiles_key, stages, target_rel)
_LIST_REFERENCE_FAMILIES: List[Tuple[str, Tuple[str, ...], str]] = [
    ("decoding_profiles", (_STAGE_DEFAULT, _STAGE_TUNE, _STAGE_TEST), "inference.decoding"),
]


def _build_list_reference_specs() -> List[Tuple[str, List[str]]]:
    specs: List[Tuple[str, List[str]]] = []
    for profiles_key, stages, target_rel in _LIST_REFERENCE_FAMILIES:
        target_paths = [_stage_path(stage, target_rel) for stage in stages]
        specs.append((profiles_key, target_paths))
    return specs


def _build_allowed_selector_paths() -> set[str]:
    paths: set[str] = set()

    for _, _, stages, selector_rel, _, _ in _VALUE_PROFILE_FAMILIES:
        for stage in stages:
            paths.add(_normalize_selector_path(_stage_path(stage, selector_rel)))

    arch_stages, arch_selector_rel, _ = _ARCH_PROFILE_FAMILY
    for stage in arch_stages:
        paths.add(_normalize_selector_path(_stage_path(stage, arch_selector_rel)))

    return paths


def _build_profile_parent_rels(allowed_selector_paths: set[str]) -> set[str]:
    rels: set[str] = set()
    for path in allowed_selector_paths:
        if not path.endswith(".profile"):
            continue
        parent_path = path[: -len(".profile")]
        rels.add(_strip_stage_prefix(parent_path))
    return rels


_ALLOWED_SELECTOR_PATHS = _build_allowed_selector_paths()
_PROFILE_PARENT_RELS = _build_profile_parent_rels(_ALLOWED_SELECTOR_PATHS)
_VALUE_PROFILE_SPECS_BY_PHASE = _build_value_profile_specs()
_ARCH_PROFILE_SPECS = _build_arch_profile_specs()
_REFERENCE_PROFILE_SPECS = _build_reference_profile_specs()
_LIST_REFERENCE_SPECS = _build_list_reference_specs()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_profile_engine() -> YamlProfileEngine:
    """Build the YAML profile engine from declarative tables.

    The applier ordering is:
    1. Pipeline + system ValueProfileAppliers
    2. ArchProfileAppliers
    3. Remaining ValueProfileAppliers (augmentation through activation)
    4. ReferenceProfileAppliers
    5. ListProfileReferenceAppliers
    """
    appliers: List[YamlProfileApplier] = []

    before_arch = _VALUE_PROFILE_SPECS_BY_PHASE["before_arch"]
    after_arch = _VALUE_PROFILE_SPECS_BY_PHASE["after_arch"]

    # 1) Pipeline + system value profiles
    for selector_paths, profiles_key, target_path, merge, cleanup_keys in before_arch:
        appliers.append(ValueProfileApplier(
            selector_paths=selector_paths,
            profiles_key=profiles_key,
            target_path=target_path,
            merge_into_existing=merge,
            cleanup_keys=cleanup_keys,
        ))

    # 2) Architecture profiles
    for selector_paths, target_path, cleanup_keys in _ARCH_PROFILE_SPECS:
        appliers.append(ArchProfileApplier(
            selector_paths=selector_paths,
            target_path=target_path,
            cleanup_keys=cleanup_keys,
        ))

    # 3) Remaining value profiles (augmentation through activation)
    for selector_paths, profiles_key, target_path, merge, cleanup_keys in after_arch:
        appliers.append(ValueProfileApplier(
            selector_paths=selector_paths,
            profiles_key=profiles_key,
            target_path=target_path,
            merge_into_existing=merge,
            cleanup_keys=cleanup_keys,
        ))

    # 4) Reference profile appliers
    for profiles_key, target_paths in _REFERENCE_PROFILE_SPECS:
        appliers.append(ReferenceProfileApplier(
            profiles_key=profiles_key,
            target_paths=target_paths,
        ))

    # 5) List profile reference appliers
    for profiles_key, target_paths in _LIST_REFERENCE_SPECS:
        appliers.append(ListProfileReferenceApplier(
            profiles_key=profiles_key,
            target_paths=target_paths,
        ))

    return YamlProfileEngine(appliers)


_YAML_PROFILE_ENGINE = _build_profile_engine()
