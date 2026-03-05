"""
Configuration and factory functions for PyTorch Lightning training.

This module contains all "create" functions that build training components:
- create_datamodule(): Build Lightning DataModule from config
- setup_seed_everything(): Expose Lightning's seed_everything helper
- Helper utilities used by main training scripts
"""

from __future__ import annotations

from .data_factory import create_datamodule
from .runtime import cleanup_run_directory, modify_checkpoint_state, setup_run_directory


def setup_seed_everything():
    """
    Return Lightning's canonical seed helper.

    Returns:
        seed_everything function
    """
    from pytorch_lightning import seed_everything
    return seed_everything


__all__ = [
    "setup_seed_everything",
    "create_datamodule",
    "setup_run_directory",
    "cleanup_run_directory",
    "modify_checkpoint_state",
]
