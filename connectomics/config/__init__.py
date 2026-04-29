"""
Modern Hydra-based configuration system for PyTorch Connectomics.
"""

from .pipeline.config_io import (
    create_experiment_name,
    from_dict,
    get_config_hash,
    load_config,
    merge_configs,
    print_config,
    resolve_data_paths,
    save_config,
    to_dict,
    update_from_cli,
    validate_config,
)
from .pipeline.dict_utils import as_plain_dict, cfg_get, to_plain
from .pipeline.stage_resolver import resolve_default_profiles

# New Hydra config system (primary)
from .schema import Config

__all__ = [
    # Hydra config system
    "Config",
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
    # Config utilities
    "to_plain",
    "as_plain_dict",
    "cfg_get",
]
