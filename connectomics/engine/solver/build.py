# Code adapted from Detectron2(https://github.com/facebookresearch/detectron2)
from typing import Any, Dict, List, Set
import torch
from torch.optim.lr_scheduler import MultiStepLR

from yacs.config import CfgNode

from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR

from connectomics.model.norm import BatchInstanceNorm1d
from connectomics.model.norm import BatchInstanceNorm2d
from connectomics.model.norm import BatchInstanceNorm3d

from connectomics.model.norm import SynchronizedBatchNorm1d
from connectomics.model.norm import SynchronizedBatchNorm2d
from connectomics.model.norm import SynchronizedBatchNorm3d


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # Inherits from connectomics.model
        BatchInstanceNorm1d,
        BatchInstanceNorm2d,
        BatchInstanceNorm3d,
        SynchronizedBatchNorm1d,
        SynchronizedBatchNorm2d,
        SynchronizedBatchNorm3d,
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
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
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
            milestones = cfg.SOLVER.MILESTONES,
            gamma = cfg.SOLVER.GAMMA)
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
