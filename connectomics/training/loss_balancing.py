"""
Adaptive loss balancing strategies for multi-task learning.

Supports:
- Uncertainty weighting (Kendall et al. 2018): learns log-variance per task
- GradNorm (Chen et al. 2018): balances tasks by matching gradient norms
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _select_shared_parameters(model: nn.Module, strategy: str = "last") -> List[nn.Parameter]:
    """
    Select a small, shared parameter set for GradNorm gradient measurement.

    Args:
        model: Model containing learnable parameters.
        strategy: Selection strategy ("last", "first", "all").

    Returns:
        List of parameters to use for gradient norm computation.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return []

    strategy = (strategy or "last").lower()
    if strategy == "first":
        return params[:1]
    if strategy == "all":
        return params
    # Default: last layer (smallest subset, cheapest to differentiate)
    return params[-1:]


class BaseLossWeighter(nn.Module):
    """Interface for loss weighting strategies."""

    def combine(
        self, losses: Sequence[torch.Tensor], names: Sequence[str], stage: str
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Combine task losses into a single scalar.

        Args:
            losses: Iterable of task losses (unweighted).
            names: Task names (same order as losses) for logging.
            stage: "train" or "val" (used for conditional logic/logging).

        Returns:
            total_loss: Scalar loss used for backprop.
            weights: Tensor of weights applied per task (for logging).
            log_dict: Additional logs related to weighting.
        """
        raise NotImplementedError


class UncertaintyLossWeighter(BaseLossWeighter):
    """Learned uncertainty weighting (Kendall et al., 2018)."""

    def __init__(self, num_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def combine(
        self, losses: Sequence[torch.Tensor], names: Sequence[str], stage: str
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        loss_tensor = torch.stack([loss for loss in losses])
        weights = torch.exp(-self.log_vars)

        # Standard homoscedastic uncertainty formulation
        weighted_losses = 0.5 * weights * loss_tensor
        reg_term = 0.5 * self.log_vars
        total_loss = weighted_losses.sum() + reg_term.sum()

        log_dict = {
            f"{stage}_loss_uncertainty/{name}_weight": w.item()
            for name, w in zip(names, weights.detach())
        }
        log_dict[f"{stage}_loss_uncertainty/reg"] = reg_term.sum().item()
        return total_loss, weights.detach(), log_dict


class GradNormLossWeighter(BaseLossWeighter):
    """GradNorm (Chen et al., 2018) for balancing task training rates."""

    def __init__(
        self,
        num_tasks: int,
        alpha: float = 0.5,
        gradnorm_lambda: float = 1.0,
        shared_parameters: Optional[Iterable[nn.Parameter]] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gradnorm_lambda = gradnorm_lambda
        self.task_weights = nn.Parameter(torch.ones(num_tasks))
        self.register_buffer("initial_losses", None)
        self.shared_parameters = list(shared_parameters) if shared_parameters is not None else []

    def _normalized_weights(self) -> torch.Tensor:
        # Keep weights positive and with constant sum for stability
        raw = torch.relu(self.task_weights)
        return raw * (len(raw) / (raw.sum() + 1e-6))

    def combine(
        self, losses: Sequence[torch.Tensor], names: Sequence[str], stage: str
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        losses_tensor = torch.stack([loss for loss in losses])
        weights = self._normalized_weights()

        weighted_losses = weights * losses_tensor
        total_weighted_loss = weighted_losses.sum()

        # No GradNorm updates during validation/inference or if we lack parameters
        if (
            not self.training
            or stage != "train"
            or not self.shared_parameters
            or len(losses_tensor) == 0
        ):
            log_dict = {
                f"{stage}_loss_gradnorm/{name}_weight": w.item()
                for name, w in zip(names, weights.detach())
            }
            return total_weighted_loss, weights.detach(), log_dict

        # Track initial losses for relative training rates
        if self.initial_losses is None:
            self.initial_losses = losses_tensor.detach()

        # Compute gradient norms for each task on a small shared parameter set
        base_grad_norms = []
        for loss in losses:
            grads = torch.autograd.grad(
                loss,
                self.shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            grad_values = [g.norm() for g in grads if g is not None]
            if len(grad_values) == 0:
                grad_norm = torch.tensor(0.0, device=loss.device)
            else:
                grad_norm = torch.stack(grad_values).mean()
            base_grad_norms.append(grad_norm.detach())

        base_grad_norms = torch.stack(base_grad_norms)
        weighted_grad_norms = weights * base_grad_norms
        grad_norm_mean = weighted_grad_norms.mean()

        # Relative inverse training rates
        loss_ratios = losses_tensor.detach() / (self.initial_losses + 1e-12)
        loss_ratio_mean = loss_ratios.mean()
        target_grad_norms = grad_norm_mean * (loss_ratios / loss_ratio_mean) ** self.alpha

        grad_norm_loss = F.l1_loss(weighted_grad_norms, target_grad_norms)
        total_loss = total_weighted_loss + self.gradnorm_lambda * grad_norm_loss

        log_dict = {
            f"{stage}_loss_gradnorm/{name}_weight": w.item()
            for name, w in zip(names, weights.detach())
        }
        log_dict[f"{stage}_loss_gradnorm/reg"] = grad_norm_loss.item()
        return total_loss, weights.detach(), log_dict


def build_loss_weighter(
    cfg, num_tasks: int, model: Optional[nn.Module] = None
) -> Optional[BaseLossWeighter]:
    """
    Build a loss weighting strategy from config.

    Args:
        cfg: Hydra/OmegaConf config.
        num_tasks: Number of task losses to balance.
        model: Model (for GradNorm shared-parameter selection).

    Returns:
        Loss weighter module or None (for static weights).
    """
    if not hasattr(cfg, "model") or not hasattr(cfg.model, "loss_balancing"):
        return None

    lb_cfg = cfg.model.loss_balancing
    strategy = getattr(lb_cfg, "strategy", None)
    if strategy is None:
        return None

    strategy = strategy.lower()
    if strategy == "uncertainty":
        return UncertaintyLossWeighter(num_tasks)

    if strategy == "gradnorm":
        shared_params = (
            _select_shared_parameters(model, getattr(lb_cfg, "gradnorm_parameter_strategy", "last"))
            if model is not None
            else []
        )
        return GradNormLossWeighter(
            num_tasks=num_tasks,
            alpha=getattr(lb_cfg, "gradnorm_alpha", 0.5),
            gradnorm_lambda=getattr(lb_cfg, "gradnorm_lambda", 1.0),
            shared_parameters=shared_params,
        )

    raise ValueError(f"Unknown loss balancing strategy: {strategy}")
