# Code adapted from Detectron2(https://github.com/facebookresearch/detectron2)
from .build import build_lr_scheduler, build_optimizer, build_swa_model
from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR