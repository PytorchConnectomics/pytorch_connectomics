"""
Lightning training package.

Exports Lightning-specific modules (model, data, trainer, callbacks, config/utils).
"""

from .model import ConnectomicsModule
from .data import ConnectomicsDataModule, VolumeDataModule, TileDataModule
from .trainer import create_trainer
from .callbacks import VisualizationCallback, NaNDetectionCallback, EMAWeightsCallback, create_callbacks
from .config import (
    create_datamodule,
    expand_file_paths,
    setup_seed_everything,
    setup_run_directory,
    cleanup_run_directory,
    modify_checkpoint_state,
)
from .utils import (
    parse_args,
    setup_config,
    extract_best_score_from_checkpoint,
)

__all__ = [
    "ConnectomicsModule",
    "ConnectomicsDataModule",
    "VolumeDataModule",
    "TileDataModule",
    "create_trainer",
    "VisualizationCallback",
    "NaNDetectionCallback",
    "EMAWeightsCallback",
    "create_callbacks",
    "create_datamodule",
    "expand_file_paths",
    "setup_run_directory",
    "cleanup_run_directory",
    "modify_checkpoint_state",
    "setup_seed_everything",
    "parse_args",
    "setup_config",
    "extract_best_score_from_checkpoint",
]
