"""Core config loading, profile resolution, and stage merging."""

from .config_io import (
    load_config,
    save_config,
    merge_configs,
    update_from_cli,
    to_dict,
    from_dict,
    print_config,
    validate_config,
    get_config_hash,
    create_experiment_name,
    resolve_data_paths,
)
from .stage_resolver import resolve_default_profiles
from .dict_utils import to_plain, as_plain_dict, cfg_get

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
    "resolve_default_profiles",
    "to_plain",
    "as_plain_dict",
    "cfg_get",
]
