"""
Lightning training package.

Exports Lightning-specific modules (model, data, trainer, callbacks, config/utils).
"""

from .callbacks import (
    EMAWeightsCallback,
    NaNDetectionCallback,
    ValidationReseedingCallback,
    VisualizationCallback,
)
from .data import ConnectomicsDataModule, SimpleDataModule
from .data_factory import create_datamodule
from .model import ConnectomicsModule
from .path_utils import expand_file_paths
from .runtime import cleanup_run_directory, modify_checkpoint_state, setup_run_directory
from .trainer import create_trainer
from .utils import (
    compute_tta_passes,
    extract_best_score_from_checkpoint,
    final_prediction_output_tag,
    is_tta_cache_suffix,
    parse_args,
    resolve_prediction_cache_suffix,
    setup_config,
    setup_seed_everything,
    tta_cache_suffix,
    tta_cache_suffix_candidates,
)

__all__ = [
    "ConnectomicsModule",
    "ConnectomicsDataModule",
    "SimpleDataModule",
    "create_trainer",
    "VisualizationCallback",
    "NaNDetectionCallback",
    "EMAWeightsCallback",
    "ValidationReseedingCallback",
    "create_datamodule",
    "expand_file_paths",
    "setup_run_directory",
    "cleanup_run_directory",
    "modify_checkpoint_state",
    "setup_seed_everything",
    "parse_args",
    "setup_config",
    "extract_best_score_from_checkpoint",
    "compute_tta_passes",
    "final_prediction_output_tag",
    "tta_cache_suffix",
    "tta_cache_suffix_candidates",
    "resolve_prediction_cache_suffix",
    "is_tta_cache_suffix",
]
