"""
Modern model builder using architecture registry.

Uses MONAI and MedNeXt native models with automatic configuration.
All models are registered in the architecture registry.

Uses Hydra/OmegaConf configuration.
"""

import torch

from pathlib import Path

from .arch import (
    get_architecture_builder,
    print_available_architectures,
)


def load_external_weights(model, cfg):
    """
    Load model weights from an external checkpoint file.

    Supports loading from:
    - PyTorch Lightning checkpoints (state_dict under 'state_dict' key)
    - Raw PyTorch checkpoints (direct state_dict)
    - BANIS/nnUNet style checkpoints

    Args:
        model: The model to load weights into
        cfg: Config object with external_weights_path and external_weights_key_prefix

    Returns:
        Model with loaded weights
    """
    weights_path = cfg.model.external_weights_path
    key_prefix = getattr(cfg.model, 'external_weights_key_prefix', 'model.')

    if not Path(weights_path).exists():
        raise FileNotFoundError(f"External weights file not found: {weights_path}")

    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

    # Extract state_dict based on checkpoint format
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            # PyTorch Lightning checkpoint
            state_dict = checkpoint['state_dict']
            print(f'    Loaded Lightning checkpoint (epoch={checkpoint.get("epoch", "?")})')
        elif 'model_state_dict' in checkpoint:
            # Some training frameworks use this key
            state_dict = checkpoint['model_state_dict']
        else:
            # Assume the dict is the state_dict itself
            state_dict = checkpoint
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

    # Strip key prefix if specified
    # This handles cases where weights are saved as "model.conv1.weight"
    # but the model expects "conv1.weight"
    if key_prefix:
        new_state_dict = {}
        stripped_count = 0
        for key, value in state_dict.items():
            if key.startswith(key_prefix):
                new_key = key[len(key_prefix):]
                new_state_dict[new_key] = value
                stripped_count += 1
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
        if stripped_count > 0:
            print(f'    Stripped "{key_prefix}" prefix from {stripped_count} keys')

    # Handle torch.compile() models which have "_orig_mod." prefix
    # Check if keys have this prefix
    has_orig_mod = any(k.startswith('_orig_mod.') for k in state_dict.keys())
    if has_orig_mod:
        new_state_dict = {}
        compile_stripped = 0
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[10:]  # len('_orig_mod.') = 10
                new_state_dict[new_key] = value
                compile_stripped += 1
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
        if compile_stripped > 0:
            print(f'    Stripped "_orig_mod." prefix from {compile_stripped} keys (torch.compile model)')

    # Handle wrapped models (e.g., MedNeXtWrapper has model.model)
    target_module = model
    if hasattr(model, 'model'):
        target_module = model.model
        print(f'    Loading into wrapped model: {target_module.__class__.__name__}')

    # Load state dict
    missing_keys, unexpected_keys = target_module.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f'    Warning: {len(missing_keys)} missing keys')
        if len(missing_keys) <= 5:
            for key in missing_keys:
                print(f'      - {key}')
        else:
            print(f'      First 5: {missing_keys[:5]}')

    if unexpected_keys:
        print(f'    Warning: {len(unexpected_keys)} unexpected keys')
        if len(unexpected_keys) <= 5:
            for key in unexpected_keys:
                print(f'      - {key}')
        else:
            print(f'      First 5: {unexpected_keys[:5]}')

    if not missing_keys and not unexpected_keys:
        print(f'    Successfully loaded all weights')

    return model


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

    # Load external weights if specified
    external_weights_path = getattr(cfg.model, 'external_weights_path', None)
    if external_weights_path:
        print(f'\n  Loading external weights from: {external_weights_path}')
        model = load_external_weights(model, cfg)

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
    'load_external_weights',
    'update_state_dict',
]