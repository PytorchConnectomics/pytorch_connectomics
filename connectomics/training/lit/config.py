"""
Configuration and factory functions for PyTorch Lightning training.

This module contains all "create" functions that build training components:
- create_datamodule(): Build Lightning DataModule from config
- setup_seed_everything(): Handle seed_everything across Lightning versions
- Helper utilities used by main training scripts
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch

from .data_factory import create_datamodule
from .path_utils import expand_file_paths as _expand_file_paths
from .runtime import cleanup_run_directory as _cleanup_run_directory
from .runtime import modify_checkpoint_state as _modify_checkpoint_state
from .runtime import setup_run_directory as _setup_run_directory


def setup_seed_everything():
    """
    Setup seed_everything function with fallback for older Lightning versions.

    Returns:
        seed_everything function
    """
    try:
        from pytorch_lightning.utilities.seed import seed_everything

        return seed_everything
    except ImportError:
        try:
            from pytorch_lightning import seed_everything

            return seed_everything
        except ImportError:
            # Fallback for older versions
            def seed_everything(seed, workers=True):
                import random

                import numpy as np

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

            return seed_everything


def expand_file_paths(path_or_pattern) -> List[str]:
    """
    Backward-compatible wrapper for shared path expansion helper.

    Args:
        path_or_pattern: Single file path, glob pattern, or list of paths/patterns

    Returns:
        List of expanded file paths, sorted alphabetically
    """
    return _expand_file_paths(path_or_pattern)


def setup_run_directory(mode: str, cfg, checkpoint_dirpath: str):
    """Backward-compatible wrapper for runtime directory setup."""
    return _setup_run_directory(mode, cfg, checkpoint_dirpath)


def cleanup_run_directory(output_base: Path):
    """Backward-compatible wrapper for runtime directory cleanup."""
    return _cleanup_run_directory(output_base)


def modify_checkpoint_state(
    checkpoint_path: Optional[str],
    run_dir: Path,
    reset_optimizer: bool = False,
    reset_scheduler: bool = False,
    reset_epoch: bool = False,
    reset_early_stopping: bool = False,
) -> Optional[str]:
    """Backward-compatible wrapper for runtime checkpoint mutation."""
    return _modify_checkpoint_state(
        checkpoint_path,
        run_dir,
        reset_optimizer=reset_optimizer,
        reset_scheduler=reset_scheduler,
        reset_epoch=reset_epoch,
        reset_early_stopping=reset_early_stopping,
    )


__all__ = [
    "setup_seed_everything",
    "expand_file_paths",
    "create_datamodule",
    "setup_run_directory",
    "cleanup_run_directory",
    "modify_checkpoint_state",
]
