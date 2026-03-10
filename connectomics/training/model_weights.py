"""PyTorch-only helpers for loading external model weights into a built model."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_external_weights(model: nn.Module, cfg) -> nn.Module:
    """
    Load model weights from an external checkpoint file into an existing model.

    Supports loading from:
    - PyTorch Lightning checkpoints (state_dict under 'state_dict')
    - Raw PyTorch checkpoints (direct state_dict)
    - Checkpoints with 'model_state_dict'
    """
    weights_path = getattr(cfg.model, "external_weights_path", None)
    if not weights_path:
        raise ValueError("cfg.model.external_weights_path must be set to load external weights")

    key_prefix = getattr(cfg.model, "external_weights_key_prefix", "model.")
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"External weights file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            logger.info("Loaded Lightning checkpoint (epoch=%s)", checkpoint.get("epoch", "?"))
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

    if key_prefix:
        stripped_state_dict = {}
        stripped_count = 0
        for key, value in state_dict.items():
            if key.startswith(key_prefix):
                stripped_state_dict[key[len(key_prefix) :]] = value
                stripped_count += 1
            else:
                stripped_state_dict[key] = value
        state_dict = stripped_state_dict
        if stripped_count > 0:
            logger.info('Stripped "%s" prefix from %d keys', key_prefix, stripped_count)

    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        compile_stripped_state_dict = {}
        compile_stripped = 0
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                compile_stripped_state_dict[key[10:]] = value
                compile_stripped += 1
            else:
                compile_stripped_state_dict[key] = value
        state_dict = compile_stripped_state_dict
        if compile_stripped > 0:
            logger.info(
                'Stripped "_orig_mod." prefix from %d keys (torch.compile model)',
                compile_stripped,
            )

    target_module = model.model if hasattr(model, "model") else model
    if target_module is not model:
        logger.info("Loading into wrapped model: %s", target_module.__class__.__name__)

    missing_keys, unexpected_keys = target_module.load_state_dict(state_dict, strict=False)

    if missing_keys:
        logger.warning("%d missing keys", len(missing_keys))
        for key in missing_keys[:5]:
            logger.warning("  - %s", key)
        if len(missing_keys) > 5:
            logger.warning("  ... (%d more)", len(missing_keys) - 5)

    if unexpected_keys:
        logger.warning("%d unexpected keys", len(unexpected_keys))
        for key in unexpected_keys[:5]:
            logger.warning("  - %s", key)
        if len(unexpected_keys) > 5:
            logger.warning("  ... (%d more)", len(unexpected_keys) - 5)

    if not missing_keys and not unexpected_keys:
        logger.info("Successfully loaded all weights")

    return model


__all__ = ["load_external_weights"]
