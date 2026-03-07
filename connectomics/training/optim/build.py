"""
Optimizer and learning rate scheduler builder.

Supports Hydra/OmegaConf configurations.
Code adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
"""


from __future__ import annotations
import logging
from collections.abc import Mapping
from typing import Any, Dict, List, Set
import torch

logger = logging.getLogger(__name__)
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR

from .lr_scheduler import WarmupCosineLR


__all__ = ["build_optimizer", "build_lr_scheduler"]


def _scheduler_param(sched_cfg: Any, key: str, default: Any) -> Any:
    """Resolve scheduler parameters from `scheduler.params` first, then direct fields."""
    params = getattr(sched_cfg, "params", None)
    if isinstance(params, Mapping) and key in params and params[key] is not None:
        return params[key]
    value = getattr(sched_cfg, key, default)
    return default if value is None else value


def _scheduler_specific_param(sched_cfg: Any, key: str, default: Any) -> Any:
    """Resolve scheduler-specific parameters from `scheduler.params` only."""
    params = getattr(sched_cfg, "params", None)
    if isinstance(params, Mapping) and key in params and params[key] is not None:
        return params[key]
    return default


def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from Hydra configuration.

    Args:
        cfg: Hydra configuration object
        model: PyTorch model

    Returns:
        Configured optimizer

    Examples:
        >>> optimizer = build_optimizer(cfg, model)
    """
    # Get optimizer config
    if hasattr(cfg, "optimization") and hasattr(cfg.optimization, "optimizer"):
        opt_cfg = cfg.optimization.optimizer
    else:
        raise ValueError("Config must have 'optimization.optimizer' section")

    # Extract parameters
    optimizer_name = opt_cfg.name.lower() if hasattr(opt_cfg, "name") else "adamw"
    lr = opt_cfg.lr if hasattr(opt_cfg, "lr") else 1e-4
    weight_decay = opt_cfg.weight_decay if hasattr(opt_cfg, "weight_decay") else 1e-4

    # Build parameter groups with differential learning rates and weight decay
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()

    # Get optional parameters
    weight_decay_norm = getattr(opt_cfg, "weight_decay_norm", 0.0)
    weight_decay_bias = getattr(opt_cfg, "weight_decay_bias", weight_decay)
    bias_lr_factor = getattr(opt_cfg, "bias_lr_factor", 1.0)

    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            param_lr = lr
            param_weight_decay = weight_decay

            if isinstance(module, norm_module_types):
                param_weight_decay = weight_decay_norm
            elif key == "bias":
                param_lr = lr * bias_lr_factor
                param_weight_decay = weight_decay_bias

            params.append({"params": [value], "lr": param_lr, "weight_decay": param_weight_decay})

    # Create optimizer
    if optimizer_name == "adamw":
        betas = tuple(opt_cfg.betas) if hasattr(opt_cfg, "betas") else (0.9, 0.999)
        eps = opt_cfg.eps if hasattr(opt_cfg, "eps") else 1e-8
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adam":
        betas = tuple(opt_cfg.betas) if hasattr(opt_cfg, "betas") else (0.9, 0.999)
        eps = opt_cfg.eps if hasattr(opt_cfg, "eps") else 1e-8
        optimizer = torch.optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        momentum = opt_cfg.momentum if hasattr(opt_cfg, "momentum") else 0.9
        nesterov = getattr(opt_cfg, "nesterov", False)
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    else:
        raise ValueError(
            f"Unknown optimizer: '{optimizer_name}'. "
            f"Supported optimizers: adamw, adam, sgd"
        )

    logger.info(f"Optimizer: {optimizer.__class__.__name__} (lr={lr}, wd={weight_decay})")
    return optimizer


def build_lr_scheduler(
    cfg, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a learning rate scheduler from Hydra configuration.

    Args:
        cfg: Hydra configuration object
        optimizer: PyTorch optimizer

    Returns:
        Configured learning rate scheduler

    Examples:
        >>> scheduler = build_lr_scheduler(cfg, optimizer)
    """
    return _build_lr_scheduler_hydra(cfg, optimizer)


def _build_lr_scheduler_hydra(
    cfg, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build LR scheduler from Hydra config."""
    # Get scheduler config
    if hasattr(cfg, "optimization") and hasattr(cfg.optimization, "scheduler"):
        sched_cfg = cfg.optimization.scheduler
    else:
        raise ValueError("Config must have 'optimization.scheduler' section")

    # Extract scheduler name
    scheduler_name = _scheduler_param(sched_cfg, "name", "cosineannealinglr").lower()

    if scheduler_name == "cosineannealinglr":
        # Get max epochs from training config or default
        default_t_max = (
            cfg.optimization.max_epochs
            if hasattr(cfg, "optimization") and hasattr(cfg.optimization, "max_epochs")
            else 100
        )
        t_max = _scheduler_specific_param(sched_cfg, "t_max", default_t_max)
        eta_min = _scheduler_param(sched_cfg, "min_lr", 1e-6)

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )

    elif scheduler_name in ("cosineannealingwarmrestarts", "cosinewarmrestarts"):
        T_0 = _scheduler_specific_param(sched_cfg, "T_0", 200)
        T_mult = _scheduler_specific_param(sched_cfg, "T_mult", 1)
        eta_min = _scheduler_param(sched_cfg, "min_lr", 1e-5)

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
        )

    elif scheduler_name == "steplr":
        step_size = _scheduler_specific_param(sched_cfg, "step_size", 30)
        gamma = _scheduler_specific_param(sched_cfg, "gamma", 0.1)

        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    elif scheduler_name == "multisteplr":
        milestones = list(_scheduler_specific_param(sched_cfg, "milestones", [30, 60, 90]))
        gamma = _scheduler_specific_param(sched_cfg, "gamma", 0.1)

        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )

    elif scheduler_name == "reducelronplateau":
        mode = _scheduler_param(sched_cfg, "mode", "min")
        factor = _scheduler_param(sched_cfg, "factor", 0.1)
        patience = _scheduler_param(sched_cfg, "patience", 10)
        threshold = _scheduler_param(sched_cfg, "threshold", 1e-4)
        cooldown = _scheduler_param(sched_cfg, "cooldown", 0)
        eps = _scheduler_param(sched_cfg, "eps", 1e-8)
        min_lr = _scheduler_param(sched_cfg, "min_lr", 1e-6)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            cooldown=cooldown,
            eps=eps,
            min_lr=min_lr,
        )

    elif scheduler_name == "warmupcosine" or scheduler_name == "warmupcosinelr":
        # Get max iterations
        default_max_iter = (
            cfg.optimization.max_epochs
            if hasattr(cfg, "optimization") and hasattr(cfg.optimization, "max_epochs")
            else 100
        )
        max_iter = _scheduler_specific_param(sched_cfg, "max_iter", default_max_iter)
        warmup_iters = _scheduler_param(sched_cfg, "warmup_epochs", 5)
        warmup_factor = _scheduler_param(sched_cfg, "warmup_start_lr", 0.001)
        eta_min = _scheduler_param(sched_cfg, "min_lr", 0.0)

        scheduler = WarmupCosineLR(
            optimizer,
            max_iter,
            warmup_factor=warmup_factor,
            warmup_iters=warmup_iters,
            warmup_method="linear",
            eta_min=eta_min,
        )

    elif scheduler_name == "constant" or scheduler_name == "constantlr":
        # Constant learning rate (no decay)
        # Recommended for MedNeXt architecture
        from torch.optim.lr_scheduler import LambdaLR

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0
        )
        logger.info("  Using constant learning rate (no decay)")

    else:
        # Default to CosineAnnealingLR
        t_max = (
            cfg.optimization.max_epochs
            if hasattr(cfg, "optimization") and hasattr(cfg.optimization, "max_epochs")
            else 100
        )
        t_max = _scheduler_specific_param(sched_cfg, "t_max", t_max)
        eta_min = _scheduler_param(sched_cfg, "min_lr", 1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    logger.info(f"LR Scheduler: {scheduler.__class__.__name__}")
    return scheduler
