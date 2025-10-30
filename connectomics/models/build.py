"""
Modern model builder using architecture registry.

Uses MONAI and MedNeXt native models with automatic configuration.
All models are registered in the architecture registry.

Uses Hydra/OmegaConf configuration.
"""

import torch

from .arch import (
    get_architecture_builder,
    print_available_architectures,
)


def build_model(cfg, device=None, rank=None):
    """
    Build model from configuration using architecture registry.

    Args:
        cfg: Hydra config object with model configuration
        device: torch.device (optional, auto-detected if None)
        rank: Rank for DDP (optional, unused - Lightning handles DDP)

    Returns:
        Model ready for training

    Available architectures:
        - MONAI models: monai_basic_unet3d, monai_unet, monai_unetr, monai_swin_unetr
        - MedNeXt models: mednext, mednext_custom

    Example:
        cfg = OmegaConf.create({
            'model': {
                'architecture': 'mednext',
                'in_channels': 1,
                'out_channels': 2,
                'mednext_size': 'S',
                'kernel_size': 3,
                'deep_supervision': True,
            }
        })
        model = build_model(cfg)

    To see all available architectures:
        from connectomics.models.arch import print_available_architectures
        print_available_architectures()
    """
    # Get architecture name
    model_arch = cfg.model.architecture

    # Get builder from registry
    try:
        builder = get_architecture_builder(model_arch)
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nAvailable architectures:")
        print_available_architectures()
        raise

    # Build model
    model = builder(cfg)

    # Print model info
    print(f'\nModel: {model.__class__.__name__} (architecture: {model_arch})')
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print(f'  Parameters: {info["parameters"]:,}')
        print(f'  Trainable: {info["trainable_parameters"]:,}')
        print(f'  Deep Supervision: {info["deep_supervision"]}')
        if info["deep_supervision"]:
            print(f'  Output Scales: {info["output_scales"]}')

    # Move to device
    # Note: PyTorch Lightning handles DDP/DP automatically, so we just move to device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    print(f'  Device: {device}\n')

    return model


def update_state_dict(cfg, model_dict: dict, mode: str = 'train') -> dict:
    """
    Process state dict for loading checkpoints.

    Handles:
    - SWA (Stochastic Weight Averaging) models
    - Parallel wrapper removal (DataParallel/DDP)

    Args:
        cfg: Config object (unused, kept for compatibility)
        model_dict: State dict from checkpoint
        mode: 'train' or 'test' (unused)

    Returns:
        Processed state dict

    Note:
        PyTorch Lightning handles DDP state dict processing automatically.
        This function is mainly for legacy checkpoint compatibility.
    """
    if 'n_averaged' in model_dict.keys():
        print(f"Loading SWA model (averaged {model_dict['n_averaged']} checkpoints)")

    # Remove 'module.' prefix from DataParallel/DDP if present
    new_dict = {}
    for key, value in model_dict.items():
        if key.startswith('module.'):
            new_dict[key[7:]] = value  # Remove 'module.' prefix
        else:
            new_dict[key] = value

    return new_dict


__all__ = [
    'build_model',
    'update_state_dict',
]