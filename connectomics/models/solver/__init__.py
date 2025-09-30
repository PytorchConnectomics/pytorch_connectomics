"""
Optimizer and LR scheduler builders for PyTorch Connectomics.

Supports both YACS (legacy) and Hydra/OmegaConf (modern) configurations.
Code adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
"""

from .build import build_lr_scheduler, build_optimizer
from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR

__all__ = [
    'build_optimizer',
    'build_lr_scheduler',
    'WarmupCosineLR',
    'WarmupMultiStepLR',
]