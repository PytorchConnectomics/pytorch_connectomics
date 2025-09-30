"""
Modern model builder using MONAI native models.

Uses MONAI's built-in architectures with automatic hyperparameter setting.
No custom architectures - all models are from MONAI or nnU-Net.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


def build_model(cfg, device=None, rank=None):
    """
    Build model from configuration using MONAI native models.
    
    Supports both YACS (cfg.MODEL.ARCHITECTURE) and Hydra (cfg.model.architecture) configs.
    
    Supported architectures:
    - monai_basic_unet3d: MONAI BasicUNet (simple, fast)
    - monai_unet: MONAI UNet with residual units
    - monai_unetr: MONAI UNETR (transformer-based)
    - monai_swin_unetr: MONAI Swin UNETR
    - mednext: MedNeXt (if available)
    """
    # Support both YACS (cfg.MODEL.ARCHITECTURE) and Hydra (cfg.model.architecture)
    if hasattr(cfg, 'MODEL'):
        model_arch = cfg.MODEL.ARCHITECTURE
    else:
        model_arch = cfg.model.architecture
    
    # Build model based on architecture
    if model_arch == 'monai_basic_unet3d':
        model = _build_basic_unet(cfg)
    elif model_arch == 'monai_unet':
        model = _build_monai_unet(cfg)
    elif model_arch == 'monai_unetr':
        model = _build_unetr(cfg)
    elif model_arch == 'monai_swin_unetr':
        model = _build_swin_unetr(cfg)
    elif model_arch == 'mednext':
        model = _build_mednext(cfg)
    else:
        raise ValueError(
            f"Unknown architecture: {model_arch}. "
            f"Available: monai_basic_unet3d, monai_unet, monai_unetr, monai_swin_unetr, mednext"
        )
    
    print(f'Model: {model.__class__.__name__} ({model_arch})')
    return make_parallel(model, cfg, device, rank)


def _build_basic_unet(cfg):
    """Build MONAI BasicUNet - simple and fast."""
    from monai.networks.nets import BasicUNet
    
    # Support both YACS and Hydra config
    if hasattr(cfg, 'MODEL'):
        in_channels = cfg.MODEL.IN_PLANES
        out_channels = cfg.MODEL.OUT_PLANES
        features = cfg.MODEL.FILTERS[:6] if len(cfg.MODEL.FILTERS) >= 6 else (32, 64, 128, 256, 512, 1024)
        dropout = getattr(cfg.MODEL, 'DROPOUT', 0.0)
        act = getattr(cfg.MODEL, 'ACTIVATION', 'relu')
        norm = getattr(cfg.MODEL, 'NORM_MODE', 'batch')
    else:
        in_channels = cfg.model.in_channels
        out_channels = cfg.model.out_channels
        # BasicUNet requires exactly 6 feature levels
        base_features = list(cfg.model.filters) if hasattr(cfg.model, 'filters') else [32, 64, 128, 256, 512]
        # Extend to 6 levels if needed
        while len(base_features) < 6:
            base_features.append(base_features[-1] * 2)
        features = tuple(base_features[:6])
        dropout = getattr(cfg.model, 'dropout', 0.0)
        act = getattr(cfg.model, 'activation', 'relu')
        norm = getattr(cfg.model, 'norm', 'batch')
    
    return BasicUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        dropout=dropout,
        act=act,
        norm=norm,
    )


def _build_monai_unet(cfg):
    """Build MONAI UNet with residual units."""
    from monai.networks.nets import UNet
    
    features = cfg.MODEL.FILTERS if len(cfg.MODEL.FILTERS) >= 4 else (32, 64, 128, 256, 512)
    channels = list(features)[:5]  # Limit to 5 levels
    strides = [2] * (len(channels) - 1)  # Downsample 2x each level
    
    return UNet(
        spatial_dims=3,
        in_channels=cfg.MODEL.IN_PLANES,
        out_channels=cfg.MODEL.OUT_PLANES,
        channels=channels,
        strides=strides,
        num_res_units=getattr(cfg.MODEL, 'NUM_RES_UNITS', 2),
        kernel_size=getattr(cfg.MODEL, 'KERNEL_SIZE', 3),
        norm=getattr(cfg.MODEL, 'NORM_MODE', 'batch'),
        dropout=getattr(cfg.MODEL, 'DROPOUT', 0.0),
    )


def _build_unetr(cfg):
    """Build MONAI UNETR (transformer-based)."""
    from monai.networks.nets import UNETR
    
    return UNETR(
        in_channels=cfg.MODEL.IN_PLANES,
        out_channels=cfg.MODEL.OUT_PLANES,
        img_size=cfg.MODEL.INPUT_SIZE,
        feature_size=getattr(cfg.MODEL, 'UNETR_FEATURE_SIZE', 16),
        hidden_size=getattr(cfg.MODEL, 'HIDDEN_SIZE', 768),
        mlp_dim=getattr(cfg.MODEL, 'MLP_DIM', 3072),
        num_heads=getattr(cfg.MODEL, 'UNETR_NUM_HEADS', 12),
        pos_embed=getattr(cfg.MODEL, 'POS_EMBED', 'perceptron'),
        norm_name=getattr(cfg.MODEL, 'NORM_MODE', 'instance'),
        dropout_rate=getattr(cfg.MODEL, 'UNETR_DROPOUT_RATE', 0.0),
    )


def _build_swin_unetr(cfg):
    """Build MONAI Swin UNETR."""
    from monai.networks.nets import SwinUNETR
    
    return SwinUNETR(
        img_size=cfg.MODEL.INPUT_SIZE,
        in_channels=cfg.MODEL.IN_PLANES,
        out_channels=cfg.MODEL.OUT_PLANES,
        feature_size=getattr(cfg.MODEL, 'FEATURE_SIZE', 48),
        use_checkpoint=getattr(cfg.MODEL, 'USE_CHECKPOINT', False),
        drop_rate=getattr(cfg.MODEL, 'DROPOUT', 0.0),
        attn_drop_rate=getattr(cfg.MODEL, 'ATTN_DROP_RATE', 0.0),
        dropout_path_rate=getattr(cfg.MODEL, 'DROPOUT_PATH_RATE', 0.0),
    )


def _build_mednext(cfg):
    """Build MedNeXt model (if available)."""
    try:
        from mednextv1.mednext import MedNeXt
    except ImportError:
        raise ImportError(
            "MedNeXt not installed. Install with: pip install mednextv1"
        )
    
    # MedNeXt model sizes
    model_size = getattr(cfg.MODEL, 'MEDNEXT_SIZE', 'S')  # S, B, M, L
    
    return MedNeXt(
        in_channels=cfg.MODEL.IN_PLANES,
        n_channels=32 if model_size == 'S' else 64,
        n_classes=cfg.MODEL.OUT_PLANES,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2] if model_size != 'S' else [2, 2, 2, 2, 2, 2, 2, 2, 2],
        kernel_size=getattr(cfg.MODEL, 'KERNEL_SIZE', 3),
        deep_supervision=getattr(cfg.MODEL, 'DEEP_SUPERVISION', False),
        do_res=True,
        do_res_up_down=True,
    )


def make_parallel(model, cfg, device, rank=None, find_unused_parameters=False):
    """Wrap model for parallel training."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Support both YACS and Hydra config
    if hasattr(cfg, 'SYSTEM'):
        parallel_mode = getattr(cfg.SYSTEM, 'PARALLEL', 'NONE')
    else:
        parallel_mode = 'NONE'  # Lightning handles parallelism
    
    if parallel_mode == 'DDP':
        print('Parallelism with DistributedDataParallel.')
        
        # Convert to SyncBatchNorm if specified
        if getattr(cfg.MODEL, 'NORM_MODE', 'batch') == 'sync_bn':
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        model = model.to(device)
        
        if rank is None:
            raise ValueError("rank must be specified for DDP")
        
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[rank], 
            output_device=rank, 
            find_unused_parameters=find_unused_parameters
        )
    
    elif parallel_mode == 'DP':
        gpu_device_ids = list(range(cfg.SYSTEM.NUM_GPUS))
        print(f'Parallelism with DataParallel on GPUs: {gpu_device_ids}')
        model = nn.DataParallel(model, device_ids=gpu_device_ids)
        model = model.to(device)
    
    else:
        print('No parallelism across multiple GPUs.')
        model = model.to(device)
    
    return model


def update_state_dict(cfg, model_dict: Dict[str, Any], mode: str = 'train') -> Dict[str, Any]:
    """
    Process state dict for loading checkpoints.
    
    Handles:
    - SWA (Stochastic Weight Averaging) models
    - Parallel wrapper removal
    """
    if 'n_averaged' in model_dict.keys():
        print(f"Number of models averaged in SWA: {model_dict['n_averaged']}")
    
    # Remove 'module.' prefix from DataParallel/DDP if needed
    new_dict = {}
    for key, value in model_dict.items():
        if key.startswith('module.'):
            new_dict[key[7:]] = value
        else:
            new_dict[key] = value
    
    return new_dict


__all__ = ['build_model', 'make_parallel', 'update_state_dict']