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
    resolve_data_paths,
)

# Auto-configuration system
from .auto_config import (
    auto_plan_config,
    AutoConfigPlanner,
    AutoPlanResult,
    resolve_runtime_resource_sentinels,
)

# GPU utilities
from .gpu_utils import (
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
