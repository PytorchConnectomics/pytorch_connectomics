"""PyTorch-only training loss components (framework-agnostic)."""

from .balancing import (
    BaseLossWeighter,
    GradNormLossWeighter,
    UncertaintyLossWeighter,
    build_loss_weighter,
)
from .orchestrator import LossOrchestrator, match_target_to_output
from .plan import LossTermSpec, compile_loss_terms_from_config, infer_num_loss_tasks_from_config

__all__ = [
    "LossOrchestrator",
    "match_target_to_output",
    "LossTermSpec",
    "compile_loss_terms_from_config",
    "infer_num_loss_tasks_from_config",
    "BaseLossWeighter",
    "UncertaintyLossWeighter",
    "GradNormLossWeighter",
    "build_loss_weighter",
]
