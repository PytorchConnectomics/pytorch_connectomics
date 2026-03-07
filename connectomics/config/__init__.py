"""
Modern Hydra-based configuration system for PyTorch Connectomics.
"""

# New Hydra config system (primary)
from .schema import Config
from .pipeline.config_io import (
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
from .pipeline.stage_resolver import resolve_default_profiles
from .pipeline.dict_utils import to_plain, as_plain_dict, cfg_get

# Auto-configuration system
from .hardware.auto_config import (
    auto_plan_config,
    AutoConfigPlanner,
    AutoPlanResult,
    resolve_runtime_resource_sentinels,
)

# GPU utilities
from .hardware.gpu_utils import (
    get_gpu_info,
    print_gpu_info,
    suggest_batch_size,
    estimate_gpu_memory_required,
    get_optimal_num_workers,
)


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
