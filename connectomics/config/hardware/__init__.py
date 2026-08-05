"""GPU-aware auto-planning and SLURM cluster utilities."""

from . import slurm_utils
from .auto_config import (
    AutoConfigPlanner,
    AutoPlanResult,
    auto_plan_config,
    resolve_runtime_resource_sentinels,
)
from .gpu_utils import (
    SUPPORTED_ACCELERATORS,
    empty_accelerator_cache,
    estimate_gpu_memory_required,
    get_accelerator_device_count,
    get_gpu_info,
    get_optimal_num_workers,
    is_mps_available,
    print_gpu_info,
    resolve_accelerator_type,
    suggest_batch_size,
)

__all__ = [
    "auto_plan_config",
    "AutoConfigPlanner",
    "AutoPlanResult",
    "resolve_runtime_resource_sentinels",
    "SUPPORTED_ACCELERATORS",
    "resolve_accelerator_type",
    "get_accelerator_device_count",
    "is_mps_available",
    "empty_accelerator_cache",
    "get_gpu_info",
    "print_gpu_info",
    "suggest_batch_size",
    "estimate_gpu_memory_required",
    "get_optimal_num_workers",
    "slurm_utils",
]
