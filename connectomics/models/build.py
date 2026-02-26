"""
Modern model builder using architecture registry.

Uses MONAI and MedNeXt native models with automatic configuration.
All models are registered in the architecture registry.

This module is intentionally model-only:
- selects an architecture builder
- instantiates the model
- prints model structure info

It does not load checkpoints or move the model to a device.
"""

from .arch import (
    get_architecture_builder,
    print_available_architectures,
)

def build_model(cfg):
    """
    Build model from configuration using architecture registry.

    Args:
        cfg: Hydra config object with model configuration
    Returns:
        Instantiated model (left on default device)

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
    print(f"\nModel: {model.__class__.__name__} (architecture: {model_arch})")
    if hasattr(model, "get_model_info"):
        info = model.get_model_info()
        print(f"  Parameters: {info['parameters']:,}")
        print(f"  Trainable: {info['trainable_parameters']:,}")
        print(f"  Deep Supervision: {info['deep_supervision']}")
        if info["deep_supervision"]:
            print(f"  Output Scales: {info['output_scales']}")

    print("")

    return model


__all__ = [
    "build_model",
]
