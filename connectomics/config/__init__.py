"""
Modern Hydra-based configuration system for PyTorch Connectomics.
"""

# Auto-configuration system
from .hardware.auto_config import (
    AutoConfigPlanner,
    AutoPlanResult,
    auto_plan_config,
    resolve_runtime_resource_sentinels,
)

# GPU utilities
from .hardware.gpu_utils import (
    estimate_gpu_memory_required,
    get_gpu_info,
    get_optimal_num_workers,
    print_gpu_info,
    suggest_batch_size,
)
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
    # Auto-configuration
    "auto_plan_config",
    "AutoConfigPlanner",
    "AutoPlanResult",
    "resolve_runtime_resource_sentinels",
    # GPU utilities
    "get_gpu_info",
    "print_gpu_info",
    "suggest_batch_size",
    "estimate_gpu_memory_required",
    "get_optimal_num_workers",
]
