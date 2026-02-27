"""
Lightning training package.

Exports Lightning-specific modules (model, data, trainer, callbacks, config/utils).
"""

from .callbacks import (
    EMAWeightsCallback,
    NaNDetectionCallback,
    VisualizationCallback,
)
from .config import (
    expand_file_paths,
    setup_seed_everything,
)
from .data import ConnectomicsDataModule, TileDataModule, VolumeDataModule
from .data_factory import create_datamodule
from .model import ConnectomicsModule
from .runtime import cleanup_run_directory, modify_checkpoint_state, setup_run_directory
from .trainer import create_trainer
from .utils import (
    extract_best_score_from_checkpoint,
    parse_args,
    setup_config,
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
