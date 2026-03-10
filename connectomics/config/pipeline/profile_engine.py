"""
YAML profile engine for Hydra configuration system.

Contains profile applier classes, declarative registration tables,
and the engine that resolves profile references in YAML configs.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple, Union

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

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


def _safe_get_child(node: Any, key: Any) -> Any:
    """Get child value without resolving interpolations, returning None on failure."""
    if isinstance(node, DictConfig):
        if OmegaConf.is_interpolation(node, key):
            return None
        try:
            return node[key]
        except OmegaConfBaseException:
            return None
    if isinstance(node, ListConfig):
        if OmegaConf.is_interpolation(node, key):
            return None
        try:
            return node[key]
        except OmegaConfBaseException:
            return None
    return node[key]


def _collect_selector_candidate_paths(
    yaml_node: Any,
    path: str = "",
    in_profiles_registry: bool = False,
) -> set[str]:
    paths: set[str] = set()

    if isinstance(yaml_node, ListConfig):
        for idx in range(len(yaml_node)):
            child_path = f"{path}[{idx}]" if path else f"[{idx}]"
            item = _safe_get_child(yaml_node, idx)
            if item is not None:
                paths.update(
                    _collect_selector_candidate_paths(item, child_path, in_profiles_registry)
                )
        return paths

    if not isinstance(yaml_node, (DictConfig, dict)):
        return paths

    for key in yaml_node.keys():
        key_str = str(key)
        child_path = f"{path}.{key_str}" if path else key_str
        child_value = _safe_get_child(yaml_node, key)

        # Ignore selector-like keys inside profile registries (e.g. arch_profiles.* payloads).
        child_in_profiles_registry = in_profiles_registry or (
            path == "" and key_str.endswith("_profiles")
        )
        if not child_in_profiles_registry and _is_selector_candidate(child_path, key_str):
            if child_value not in (None, ""):
                paths.add(_normalize_selector_path(child_path))

        if child_value is not None:
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
        "Top-level 'shared' config section has been removed. " "Use top-level 'default' instead."
    )


# ---------------------------------------------------------------------------
# Profile applier classes
# ---------------------------------------------------------------------------


class YamlProfileApplier:
    """Base strategy for applying a profile family to a merged YAML config."""

    cleanup_keys: Tuple[str, ...] = ()

    def apply(self, yaml_conf: DictConfig) -> None:
        raise NotImplementedError


class ValueProfileApplier(YamlProfileApplier):
    """Apply a selected profile value directly into a target path."""

    def __init__(
        self,
        selector_paths: Union[str, List[str], Tuple[str, ...]],
        profiles_key: str,
        target_path: str,
        merge_into_existing: bool = True,
        cleanup_keys: Optional[Tuple[str, ...]] = None,
        list_key: Optional[str] = None,
    ) -> None:
        if isinstance(selector_paths, str):
            selector_paths = [selector_paths]
        self.selector_paths = list(selector_paths)
        self.profiles_key = profiles_key
        self.target_path = target_path
        self.merge_into_existing = merge_into_existing
        self.cleanup_keys = cleanup_keys if cleanup_keys is not None else (profiles_key,)
        self.list_key = list_key

    def _apply_list_overrides(self, yaml_conf: DictConfig, selector_path: str) -> None:
        """Apply per-index overrides to a list-valued profile.

        After a profile expands into a list (e.g. ``model.loss.losses``),
        an ``overrides`` dict at the selector's parent path can patch
        individual entries by position::

            model:
              loss:
                profile: loss_binary
                overrides:
                  0: {pos_weight: auto}
                  1: {weight: 0.5}
        """
        parent_path = selector_path.rsplit(".", 1)[0] if "." in selector_path else ""
        overrides_path = f"{parent_path}.overrides" if parent_path else "overrides"

        overrides = OmegaConf.select(yaml_conf, overrides_path)
        if overrides is None or not isinstance(overrides, (DictConfig, dict)):
            return

        list_path = f"{self.target_path}.{self.list_key}" if self.list_key else self.target_path
        target_list = OmegaConf.select(yaml_conf, list_path)
        if not isinstance(target_list, (ListConfig, list)):
            return

        for idx_key, patch in overrides.items():
            idx = int(idx_key)
            if idx < 0 or idx >= len(target_list):
                raise ValueError(
                    f"Override index {idx} at '{overrides_path}' is out of range "
                    f"for profile list at '{self.target_path}' (length {len(target_list)})."
                )
            if isinstance(patch, (DictConfig, dict)):
                merged = OmegaConf.merge(target_list[idx], patch)
                target_list[idx] = merged

        _delete_config_path(yaml_conf, overrides_path)

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

        self._apply_list_overrides(yaml_conf, selected_path)

        # Pipeline profiles are selected via `default.pipeline_profile`, but some
        # profile payloads interpolate `${model.out_channels}` during early YAML
        # profile expansion. Materialize this key at root as soon as pipeline
        # profile is applied so later interpolation has a stable anchor.
        if selected_path == "default.pipeline_profile":
            selected_profile = profiles[selected]
            pipeline_out_ch = OmegaConf.select(selected_profile, "model.out_channels")
            if (
                pipeline_out_ch is not None
                and OmegaConf.select(yaml_conf, "model.out_channels") is None
            ):
                OmegaConf.update(
                    yaml_conf,
                    "model.out_channels",
                    pipeline_out_ch,
                    force_add=True,
                )


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
            except (OmegaConfBaseException, TypeError, ValueError):
                continue

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
        list_key: Optional[str] = None,
    ) -> None:
        self.profiles_key = profiles_key
        self.target_paths = target_paths
        self.profile_key = profile_key
        self.list_key = list_key
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

                # Extract nested list from dict-valued profiles.
                profile_list = profile_payload
                if self.list_key and isinstance(profile_payload, (DictConfig, dict)):
                    profile_list = profile_payload.get(self.list_key, profile_payload)

                overrides = {k: v for k, v in item.items() if k != self.profile_key}

                if not overrides:
                    # Pure profile reference: expand entire profile list inline
                    if not isinstance(profile_list, (ListConfig, list)):
                        raise ValueError(
                            f"Profile '{profile_name}' in {self.profiles_key} "
                            "must resolve to a list "
                            f"for target '{target_path}', got {type(profile_list)}"
                        )
                    expanded_items.extend(list(profile_list))
                else:
                    # Profile + overrides: use first profile item as base, merge overrides on top
                    if isinstance(profile_list, (ListConfig, list)):
                        if not profile_list:
                            raise ValueError(
                                f"Profile '{profile_name}' in {self.profiles_key} is empty."
                            )
                        base = OmegaConf.create(
                            OmegaConf.to_container(profile_list[0], resolve=False)
                        )
                    elif isinstance(profile_list, (DictConfig, dict)):
                        base = OmegaConf.create(OmegaConf.to_container(profile_list, resolve=False))
                    else:
                        raise ValueError(
                            f"Profile '{profile_name}' in {self.profiles_key} "
                            "must be a list or dict, "
                            f"got {type(profile_list)}"
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
# (profiles_key, stages, selector_rel, target_rel, merge_into_existing, list_key)
# Order matters: pipeline/system first, then arch, then the rest.
# All profiles are dict-valued and use merge semantics. ``list_key`` identifies the
# nested list within dict-valued profiles (used for positional overrides).
_VALUE_PROFILE_FAMILIES: List[Tuple[str, Tuple[str, ...], str, str, bool, Optional[str]]] = [
    ("pipeline_profiles", (_STAGE_DEFAULT,), "pipeline_profile", "", True, None),
    (
        "system_profiles",
        (_STAGE_DEFAULT, _STAGE_TRAIN, _STAGE_TEST, _STAGE_TUNE),
        "system.profile",
        "system",
        True,
        None,
    ),
    (
        "arch_profiles",
        (_STAGE_DEFAULT, _STAGE_TRAIN, _STAGE_TEST, _STAGE_TUNE),
        "model.arch.profile",
        "model",
        True,
        None,
    ),
    (
        "augmentation_profiles",
        (_STAGE_DEFAULT, _STAGE_TRAIN),
        "data.augmentation.profile",
        "data.augmentation",
        True,
        None,
    ),
    (
        "dataloader_profiles",
        (_STAGE_DEFAULT, _STAGE_TRAIN, _STAGE_TEST, _STAGE_TUNE),
        "data.dataloader.profile",
        "data.dataloader",
        True,
        None,
    ),
    (
        "optimizer_profiles",
        (_STAGE_DEFAULT, _STAGE_TRAIN),
        "optimization.profile",
        "optimization",
        True,
        None,
    ),
    (
        "loss_profiles",
        (_STAGE_DEFAULT, _STAGE_TRAIN),
        "model.loss.profile",
        "model.loss",
        True,
        "losses",
    ),
    (
        "label_profiles",
        (_STAGE_DEFAULT, _STAGE_TRAIN),
        "data.label_transform.profile",
        "data.label_transform",
        True,
        None,
    ),
    (
        "decoding_profiles",
        (_STAGE_DEFAULT, _STAGE_TEST, _STAGE_TUNE),
        "inference.decoding_profile",
        "inference",
        True,
        "decoding",
    ),
    (
        "activation_profiles",
        (_STAGE_DEFAULT, _STAGE_TEST, _STAGE_TUNE),
        "inference.test_time_augmentation.activation_profile",
        "inference.test_time_augmentation",
        True,
        "channel_activations",
    ),
    ("tune_profiles", (_STAGE_TUNE,), "profile", "", True, None),
]


def _build_value_profile_specs() -> (
    List[Tuple[List[str], str, str, bool, Tuple[str, ...], Optional[str]]]
):
    specs: List[Tuple[List[str], str, str, bool, Tuple[str, ...], Optional[str]]] = []

    for profiles_key, stages, selector_rel, target_rel, merge, list_key in _VALUE_PROFILE_FAMILIES:
        for stage in stages:
            selector_path = _stage_path(stage, selector_rel)
            target_path = _stage_path(stage, target_rel)
            specs.append(
                (
                    [selector_path],
                    profiles_key,
                    target_path,
                    merge,
                    (profiles_key, selector_path),
                    list_key,
                )
            )

    return specs


# Each tuple: (profiles_key, stages, target_rel)
_REFERENCE_PROFILE_FAMILIES: List[Tuple[str, Tuple[str, ...], str]] = [
    ("loss_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN), "model.loss"),
    ("label_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN), "data.label_transform"),
    ("decoding_profiles", (_STAGE_DEFAULT, _STAGE_TEST, _STAGE_TUNE), "inference"),
    (
        "activation_profiles",
        (_STAGE_DEFAULT, _STAGE_TEST, _STAGE_TUNE),
        "inference.test_time_augmentation",
    ),
    ("augmentation_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN), "data.augmentation"),
    (
        "dataloader_profiles",
        (_STAGE_DEFAULT, _STAGE_TRAIN, _STAGE_TEST, _STAGE_TUNE),
        "data.dataloader",
    ),
    ("optimizer_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN), "optimization"),
    ("system_profiles", (_STAGE_DEFAULT, _STAGE_TRAIN, _STAGE_TEST, _STAGE_TUNE), "system"),
]


def _build_reference_profile_specs() -> List[Tuple[str, List[str]]]:
    specs: List[Tuple[str, List[str]]] = []
    for profiles_key, stages, target_rel in _REFERENCE_PROFILE_FAMILIES:
        target_paths = [_stage_path(stage, target_rel) for stage in stages]
        specs.append((profiles_key, target_paths))
    return specs


# Each tuple: (profiles_key, stages, target_rel, list_key)
# ``list_key`` identifies the nested list within dict-valued profiles.
_LIST_REFERENCE_FAMILIES: List[Tuple[str, Tuple[str, ...], str, str]] = [
    (
        "decoding_profiles",
        (_STAGE_DEFAULT, _STAGE_TUNE, _STAGE_TEST),
        "inference.decoding",
        "decoding",
    ),
]


def _build_list_reference_specs() -> List[Tuple[str, List[str], str]]:
    specs: List[Tuple[str, List[str], str]] = []
    for profiles_key, stages, target_rel, list_key in _LIST_REFERENCE_FAMILIES:
        target_paths = [_stage_path(stage, target_rel) for stage in stages]
        specs.append((profiles_key, target_paths, list_key))
    return specs


def _build_allowed_selector_paths() -> set[str]:
    paths: set[str] = set()

    for _, stages, selector_rel, _, _, _ in _VALUE_PROFILE_FAMILIES:
        for stage in stages:
            paths.add(_normalize_selector_path(_stage_path(stage, selector_rel)))

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
_VALUE_PROFILE_SPECS = _build_value_profile_specs()
_REFERENCE_PROFILE_SPECS = _build_reference_profile_specs()
_LIST_REFERENCE_SPECS = _build_list_reference_specs()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_profile_engine() -> YamlProfileEngine:
    """Build the YAML profile engine from declarative tables.

    The applier ordering is:
    1. ValueProfileAppliers (pipeline, system, arch, augmentation, ..., activation)
    2. ReferenceProfileAppliers
    3. ListProfileReferenceAppliers
    """
    appliers: List[YamlProfileApplier] = []

    # 1) Value profile appliers (order defined by _VALUE_PROFILE_FAMILIES table)
    for (
        selector_paths,
        profiles_key,
        target_path,
        merge,
        cleanup_keys,
        list_key,
    ) in _VALUE_PROFILE_SPECS:
        appliers.append(
            ValueProfileApplier(
                selector_paths=selector_paths,
                profiles_key=profiles_key,
                target_path=target_path,
                merge_into_existing=merge,
                cleanup_keys=cleanup_keys,
                list_key=list_key,
            )
        )

    # 2) Reference profile appliers
    for profiles_key, target_paths in _REFERENCE_PROFILE_SPECS:
        appliers.append(
            ReferenceProfileApplier(
                profiles_key=profiles_key,
                target_paths=target_paths,
            )
        )

    # 3) List profile reference appliers
    for profiles_key, target_paths, list_key in _LIST_REFERENCE_SPECS:
        appliers.append(
            ListProfileReferenceApplier(
                profiles_key=profiles_key,
                target_paths=target_paths,
                list_key=list_key,
            )
        )

    return YamlProfileEngine(appliers)


_YAML_PROFILE_ENGINE = _build_profile_engine()
