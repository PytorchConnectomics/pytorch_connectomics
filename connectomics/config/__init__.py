"""
Modern Hydra-based configuration system for PyTorch Connectomics.
"""

# New Hydra config system (primary)
from .hydra_config import Config
from .hydra_utils import (
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
)


__all__ = [
    # Hydra config system
    'Config',
    'load_config',
    'save_config',
    'merge_configs',
    'update_from_cli',
    'to_dict',
    'from_dict',
    'print_config',
    'validate_config',
    'get_config_hash',
    'create_experiment_name',
]