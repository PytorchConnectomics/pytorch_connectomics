"""
Optimizer and learning rate scheduler builder.

Supports both YACS (legacy) and Hydra/OmegaConf (modern) configurations.
Code adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
"""

from typing import Any, Dict, List, Set, Union
import torch
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR, StepLR

from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR

try:
    from yacs.config import CfgNode
    YACS_AVAILABLE = True
except ImportError:
    CfgNode = None
    YACS_AVAILABLE = False

try:
    from omegaconf import DictConfig
    OMEGACONF_AVAILABLE = True
except ImportError:
    DictConfig = None
    OMEGACONF_AVAILABLE = False


__all__ = ['build_optimizer', 'build_lr_scheduler']


def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from configuration.
    
    Supports both YACS (cfg.SOLVER.*) and Hydra (cfg.optimizer.*) configs.
    
    Args:
        cfg: Configuration object (CfgNode or Hydra Config)
        model: PyTorch model
        
    Returns:
        Configured optimizer
        
    Examples:
        >>> # Hydra config
        >>> optimizer = build_optimizer(cfg, model)
        >>> # YACS config
        >>> optimizer = build_optimizer(cfg, model)
    """
    # Check if this is YACS or Hydra config
    is_yacs = YACS_AVAILABLE and isinstance(cfg, CfgNode) if CfgNode else hasattr(cfg, 'SOLVER')
    
    if is_yacs:
        return _build_optimizer_yacs(cfg, model)
    else:
        return _build_optimizer_hydra(cfg, model)


def _build_optimizer_yacs(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build optimizer from YACS config."""
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
    
    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            
            if isinstance(module, norm_module_types):
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
            elif key == "bias":
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            
            params.append({
                "params": [value],
                "lr": lr,
                "weight_decay": weight_decay
            })
    
    name = cfg.SOLVER.NAME
    assert name in ["SGD", "Adam", "AdamW"]
    
    if name == "SGD":
        optimizer = torch.optim.SGD(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    else:
        optimizer = getattr(torch.optim, name)(
            params, cfg.SOLVER.BASE_LR, betas=cfg.SOLVER.BETAS
        )
    
    print(f'Optimizer: {optimizer.__class__.__name__}')
    return optimizer


def _build_optimizer_hydra(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build optimizer from Hydra config."""
    # Get optimizer config
    opt_cfg = cfg.optimizer if hasattr(cfg, 'optimizer') else cfg
    
    # Extract parameters
    optimizer_name = opt_cfg.name.lower() if hasattr(opt_cfg, 'name') else 'adamw'
    lr = opt_cfg.lr if hasattr(opt_cfg, 'lr') else 1e-4
    weight_decay = opt_cfg.weight_decay if hasattr(opt_cfg, 'weight_decay') else 1e-4
    
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
    weight_decay_norm = getattr(opt_cfg, 'weight_decay_norm', 0.0)
    weight_decay_bias = getattr(opt_cfg, 'weight_decay_bias', weight_decay)
    bias_lr_factor = getattr(opt_cfg, 'bias_lr_factor', 1.0)
    
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
            
            params.append({
                "params": [value],
                "lr": param_lr,
                "weight_decay": param_weight_decay
            })
    
    # Create optimizer
    if optimizer_name == 'adamw':
        betas = tuple(opt_cfg.betas) if hasattr(opt_cfg, 'betas') else (0.9, 0.999)
        eps = opt_cfg.eps if hasattr(opt_cfg, 'eps') else 1e-8
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optimizer_name == 'adam':
        betas = tuple(opt_cfg.betas) if hasattr(opt_cfg, 'betas') else (0.9, 0.999)
        eps = opt_cfg.eps if hasattr(opt_cfg, 'eps') else 1e-8
        optimizer = torch.optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optimizer_name == 'sgd':
        momentum = opt_cfg.momentum if hasattr(opt_cfg, 'momentum') else 0.9
        nesterov = getattr(opt_cfg, 'nesterov', False)
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    else:
        # Default to AdamW
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    print(f'Optimizer: {optimizer.__class__.__name__} (lr={lr}, wd={weight_decay})')
    return optimizer


def build_lr_scheduler(
    cfg,
    optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a learning rate scheduler from configuration.
    
    Supports both YACS (cfg.SOLVER.*) and Hydra (cfg.scheduler.*) configs.
    
    Args:
        cfg: Configuration object (CfgNode or Hydra Config)
        optimizer: PyTorch optimizer
        
    Returns:
        Configured learning rate scheduler
        
    Examples:
        >>> scheduler = build_lr_scheduler(cfg, optimizer)
    """
    # Check if this is YACS or Hydra config
    is_yacs = YACS_AVAILABLE and isinstance(cfg, CfgNode) if CfgNode else hasattr(cfg, 'SOLVER')
    
    if is_yacs:
        return _build_lr_scheduler_yacs(cfg, optimizer)
    else:
        return _build_lr_scheduler_hydra(cfg, optimizer)


def _build_lr_scheduler_yacs(
    cfg: CfgNode,
    optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build LR scheduler from YACS config."""
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "MultiStepLR":
        return MultiStepLR(
            optimizer,
            milestones=cfg.SOLVER.STEPS,
            gamma=cfg.SOLVER.GAMMA,
        )
    elif name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.SOLVER.GAMMA,
            patience=10,
        )
    else:
        raise ValueError(f"Unknown LR scheduler: {name}")


def _build_lr_scheduler_hydra(
    cfg,
    optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build LR scheduler from Hydra config."""
    # Get scheduler config
    sched_cfg = cfg.scheduler if hasattr(cfg, 'scheduler') else cfg
    
    # Extract scheduler name
    scheduler_name = sched_cfg.name.lower() if hasattr(sched_cfg, 'name') else 'cosineannealinglr'
    
    if scheduler_name == 'cosineannealinglr':
        # Get max epochs from training config or default
        if hasattr(cfg, 'training'):
            t_max = cfg.training.max_epochs
        elif hasattr(sched_cfg, 't_max'):
            t_max = sched_cfg.t_max
        else:
            t_max = 100
        
        eta_min = sched_cfg.min_lr if hasattr(sched_cfg, 'min_lr') else 1e-6
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
    
    elif scheduler_name == 'steplr':
        step_size = sched_cfg.step_size if hasattr(sched_cfg, 'step_size') else 30
        gamma = sched_cfg.gamma if hasattr(sched_cfg, 'gamma') else 0.1
        
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    
    elif scheduler_name == 'multisteplr':
        milestones = list(sched_cfg.milestones) if hasattr(sched_cfg, 'milestones') else [30, 60, 90]
        gamma = sched_cfg.gamma if hasattr(sched_cfg, 'gamma') else 0.1
        
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )
    
    elif scheduler_name == 'reducelronplateau':
        mode = sched_cfg.mode if hasattr(sched_cfg, 'mode') else 'min'
        factor = sched_cfg.factor if hasattr(sched_cfg, 'factor') else 0.1
        patience = sched_cfg.patience if hasattr(sched_cfg, 'patience') else 10
        min_lr = sched_cfg.min_lr if hasattr(sched_cfg, 'min_lr') else 1e-6
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
    
    elif scheduler_name == 'warmupcosine' or scheduler_name == 'warmupcosinelr':
        # Get max iterations
        if hasattr(cfg, 'training'):
            max_iter = cfg.training.max_epochs
        elif hasattr(sched_cfg, 'max_iter'):
            max_iter = sched_cfg.max_iter
        else:
            max_iter = 100
        
        warmup_iters = sched_cfg.warmup_epochs if hasattr(sched_cfg, 'warmup_epochs') else 5
        warmup_factor = sched_cfg.warmup_start_lr if hasattr(sched_cfg, 'warmup_start_lr') else 0.001
        
        scheduler = WarmupCosineLR(
            optimizer,
            max_iter,
            warmup_factor=warmup_factor,
            warmup_iters=warmup_iters,
            warmup_method='linear',
        )
    
    else:
        # Default to CosineAnnealingLR
        t_max = cfg.training.max_epochs if hasattr(cfg, 'training') else 100
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)
    
    print(f'LR Scheduler: {scheduler.__class__.__name__}')
    return scheduler