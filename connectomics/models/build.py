"""
Modern model builder using architecture registry.

Uses MONAI and MedNeXt native models with automatic configuration.
All models are registered in the architecture registry.

This module is intentionally model-only:
- selects an architecture builder
- instantiates the model
- logs model structure info

It does not load checkpoints or move the model to a device.
"""


from __future__ import annotations
import logging

from .arch import get_architecture_builder

logger = logging.getLogger(__name__)


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
                'arch': {'type': 'mednext'},
                'in_channels': 1,
                'out_channels': 2,
                'mednext': {'size': 'S', 'kernel_size': 3},
                'deep_supervision': True,
            }
        })
        model = build_model(cfg)

    To see all available architectures:
        from connectomics.models.arch import print_available_architectures
        print_available_architectures()
    """
    # Get architecture name
    model_arch = cfg.model.arch.type

    # Get builder from registry (raises ValueError with available archs on failure)
    builder = get_architecture_builder(model_arch)

    # Build model
    model = builder(cfg)

    # Log model info
    logger.info("Model: %s (architecture: %s)", model.__class__.__name__, model_arch)
    if hasattr(model, "get_model_info"):
        info = model.get_model_info()
        logger.info("  Parameters: %s", f"{info['parameters']:,}")
        logger.info("  Trainable: %s", f"{info['trainable_parameters']:,}")
        logger.info("  Deep Supervision: %s", info['deep_supervision'])
        if info["deep_supervision"]:
            logger.info("  Output Scales: %s", info['output_scales'])

    return model


__all__ = [
    "build_model",
]
