"""GPU-aware auto-planning and SLURM cluster utilities."""

from . import slurm_utils
from .auto_config import (
    AutoConfigPlanner,
    AutoPlanResult,
    auto_plan_config,
    resolve_runtime_resource_sentinels,
)
from .gpu_utils import (
    estimate_gpu_memory_required,
    get_gpu_info,
    get_optimal_num_workers,
    print_gpu_info,
    suggest_batch_size,
)

__all__ = [
    "auto_plan_config",
    "AutoConfigPlanner",
    "AutoPlanResult",
    "resolve_runtime_resource_sentinels",
    "get_gpu_info",
    "print_gpu_info",
    "suggest_batch_size",
    "estimate_gpu_memory_required",
    "get_optimal_num_workers",
    "slurm_utils",
]
