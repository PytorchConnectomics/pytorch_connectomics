"""
Utility functions for PyTorch Lightning training scripts.

This module provides helper functions for:
- Command-line argument parsing
- Configuration setup and validation
- File path expansion
- Checkpoint utilities
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import torch

from ...config import (
    Config,
    load_config,
    print_config,
    resolve_data_paths,
    resolve_default_profiles,
    resolve_runtime_resource_sentinels,
    update_from_cli,
    validate_config,
)
from .path_utils import expand_file_paths

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PyTorch Connectomics Training with Hydra Config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        required=False,
        type=str,
        help="Path to Hydra YAML config file",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick demo with tutorials/minimal.yaml (auto fast-dev-run=1)",
    )
    parser.add_argument(
        "--debug-config",
        action="store_true",
        help="Print fully resolved runtime config (after default/mode/profile merging)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "tune", "tune-test"],
        default="train",
        help=(
            "Mode: train, test (with optional labels for metrics), tune, or "
            "tune-test (default: train)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Path to checkpoint for resuming/testing/prediction",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Reset optimizer state when loading checkpoint (useful for changing learning rate)",
    )
    parser.add_argument(
        "--reset-scheduler",
        action="store_true",
        help="Reset scheduler state when loading checkpoint",
    )
    parser.add_argument(
        "--reset-epoch",
        action="store_true",
        help="Reset epoch counter when loading checkpoint (start from epoch 0)",
    )
    parser.add_argument(
        "--reset-early-stopping",
        action="store_true",
        help="Reset early stopping patience counter when loading checkpoint",
    )
    parser.add_argument(
        "--reset-max-epochs",
        type=int,
        default=None,
        help=(
            "Override max_epochs from config (useful when resuming training "
            "with different epoch count)"
        ),
    )
    parser.add_argument(
        "--fast-dev-run",
        type=int,
        nargs="?",
        const=1,
        default=0,
        help="Run N batches for quick debugging (default: 0, no argument defaults to 1)",
    )
    parser.add_argument(
        "--nnunet-preprocess",
        action="store_true",
        help=(
            "Enable nnU-Net-style preprocessing (foreground crop, spacing-aware "
            "resampling, normalization) for this run"
        ),
    )
    parser.add_argument(
        "--external-prefix",
        type=str,
        default=None,
        help="Prefix to strip from external checkpoint keys (e.g., 'model.' for BANIS checkpoints)",
    )
    # Test arguments
    test_group = parser.add_argument_group("test", "Test/inference arguments")
    test_group.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Shard index for distributing test volumes across machines (0-indexed)",
    )
    test_group.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards for distributing test volumes across machines",
    )

    # Parameter tuning arguments
    tune_group = parser.add_argument_group("tune", "Parameter tuning arguments")
    tune_group.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to parameter file (overrides config parameter_source)",
    )
    tune_group.add_argument(
        "--param-source",
        choices=["fixed", "tuned", "optuna"],
        default=None,
        help="Parameter source: fixed, tuned, or optuna (overrides config)",
    )
    tune_group.add_argument(
        "--tune-trials",
        type=int,
        default=None,
        help="Number of Optuna trials (overrides config, use with --mode tune or tune-test)",
    )
    tune_group.add_argument(
        "--tune-timeout",
        type=int,
        default=None,
        help="Whole-study Optuna timeout in seconds (overrides tune.timeout)",
    )
    tune_group.add_argument(
        "--tune-trial-timeout",
        type=int,
        default=None,
        help="Per-trial tuning timeout in seconds (overrides tune.trial_timeout)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in key=value format (e.g., data.dataloader.batch_size=8)",
    )

    return parser.parse_args()


def setup_config(args) -> Config:
    """
    Setup configuration from YAML file and CLI overrides.

    Args:
        args: Command line arguments

    Returns:
        Validated Config object
    """
    # Load base config from YAML
    logger.info(f"Loading config: {args.config}")
    cfg = load_config(args.config)

    # Extract config file name and set output folder
    # Use config name only (e.g., mito_lucchi++).
    config_path = Path(args.config)
    config_name = config_path.stem  # Get filename without extension
    output_folder = f"outputs/{config_name}/"

    # Update checkpoint dirpath only if not provided by the user
    if not getattr(cfg.monitor.checkpoint, "dirpath", None):
        cfg.monitor.checkpoint.dirpath = str(Path(output_folder) / "checkpoints")
    else:
        cfg.monitor.checkpoint.dirpath = str(Path(cfg.monitor.checkpoint.dirpath))

    # Update prediction output path only if not provided.
    save_pred_cfg = cfg.inference.save_prediction
    if not getattr(save_pred_cfg, "output_path", None):
        save_pred_cfg.output_path = str(Path(output_folder) / "results")
    else:
        save_pred_cfg.output_path = str(Path(save_pred_cfg.output_path))
    logger.info(f"Prediction output directory: {save_pred_cfg.output_path}")

    # Note: We handle timestamping manually in main() to create run directories
    # Set this to False to prevent PyTorch Lightning from adding its own timestamp
    cfg.monitor.checkpoint.use_timestamp = False

    logger.info(f"Checkpoints base directory: {cfg.monitor.checkpoint.dirpath}")

    # Apply CLI overrides
    if args.overrides:
        logger.info(f"Applying {len(args.overrides)} CLI overrides")
        cfg = update_from_cli(cfg, args.overrides)

    # Resolve default-stage profiles into runtime sections (system/data/inference)
    cfg = resolve_default_profiles(cfg, mode=args.mode)

    # Resolve data paths on merged runtime data section.
    cfg = resolve_data_paths(cfg)

    if cfg.tune is not None:
        if args.tune_trials is not None:
            logger.info("Overriding tune.n_trials: %s -> %s", cfg.tune.n_trials, args.tune_trials)
            cfg.tune.n_trials = args.tune_trials

        if args.tune_timeout is not None:
            logger.info("Overriding tune.timeout: %s -> %s", cfg.tune.timeout, args.tune_timeout)
            cfg.tune.timeout = args.tune_timeout

        if args.tune_trial_timeout is not None:
            logger.info(
                "Overriding tune.trial_timeout: %s -> %s",
                cfg.tune.trial_timeout,
                args.tune_trial_timeout,
            )
            cfg.tune.trial_timeout = args.tune_trial_timeout
    elif any(
        value is not None
        for value in (args.tune_trials, args.tune_timeout, args.tune_trial_timeout)
    ):
        logger.warning("Ignoring --tune-* CLI overrides because the config has no tune section")

    # Override max_epochs if --reset-max-epochs is specified
    if args.reset_max_epochs is not None:
        logger.info(
            f"Overriding max_epochs: {cfg.optimization.max_epochs} -> {args.reset_max_epochs}"
        )
        cfg.optimization.max_epochs = args.reset_max_epochs

    # Handle external weights loading (when --external-prefix is specified with --checkpoint)
    if args.external_prefix is not None and args.checkpoint:
        logger.info(
            f"Loading external weights from checkpoint with prefix '{args.external_prefix}'"
        )
        cfg.model.external_weights_path = args.checkpoint
        cfg.model.external_weights_key_prefix = args.external_prefix

    # Override config for fast-dev-run mode
    if args.fast_dev_run:
        fast_dev_num_gpus = 1 if torch.cuda.is_available() else 0
        logger.info("Fast-dev-run mode: Overriding config for debugging")
        logger.info(f"   - num_gpus: {cfg.system.num_gpus} -> {fast_dev_num_gpus}")
        logger.info(
            f"   - num_workers: {cfg.system.num_workers} -> 0 "
            "(avoid multiprocessing in debug mode)"
        )
        logger.info(
            f"   - batch_size: Controlled by PyTorch Lightning (--fast-dev-run={args.fast_dev_run})"
        )
        logger.info("   - input patch: 64^3 for lightweight debug")
        logger.info("   - MedNeXt size: S for lightweight debug")
        cfg.system.num_gpus = fast_dev_num_gpus
        cfg.system.num_workers = 0
        cfg.model.input_size = [64, 64, 64]
        cfg.model.output_size = [64, 64, 64]
        cfg.data.dataloader.patch_size = [64, 64, 64]
        cfg.model.mednext.size = "S"
        # Keep CellMap shapes in sync with the smaller debug patch
        if getattr(cfg.data, "cellmap", None):
            cfg.data.cellmap["input_array_info"]["shape"] = [64, 64, 64]
            cfg.data.cellmap["target_array_info"]["shape"] = [64, 64, 64]

    # Resolve -1 sentinels (auto-max resources for current runtime allocation).
    cfg = resolve_runtime_resource_sentinels(cfg, print_results=True)

    # CPU-only fallback after all overrides: ensure no CUDA-only settings remain.
    if not torch.cuda.is_available():
        if cfg.system.num_gpus > 0:
            logger.info("CUDA not available, setting num_gpus=0")
            cfg.system.num_gpus = 0
        if cfg.system.num_workers > 0:
            logger.info("CUDA not available, setting num_workers=0 to avoid dataloader crashes")
            cfg.system.num_workers = 0

    # Optional convenience toggle to enable nnU-Net preprocessing via CLI
    if getattr(args, "nnunet_preprocess", False):
        logger.info("Enabling nnU-Net preprocessing from CLI flag")
        cfg.data.nnunet_preprocessing.enabled = True

    # Validate configuration
    logger.info("Validating configuration...")
    validate_config(cfg)

    if getattr(args, "debug_config", False):
        logger.info("\n========================")
        logger.info("RESOLVED CONFIG (DEBUG)")
        logger.info("========================")
        print_config(cfg, resolve=True)

    # Note: Output directory will be created later in main() with timestamp
    # (see lines around "Create run directory only for training mode")

    return cfg


def extract_best_score_from_checkpoint(ckpt_path: str, monitor_metric: str) -> Optional[float]:
    """
    Extract best score from checkpoint filename.

    Args:
        ckpt_path: Path to checkpoint file
        monitor_metric: Metric name to extract (e.g., 'train_loss_total_epoch', 'val/loss')

    Returns:
        Extracted score or None if not found
    """
    if not ckpt_path:
        return None

    filename = Path(ckpt_path).stem  # Get filename without extension

    # Replace '/' with underscore for metric name (e.g., 'val/loss' -> 'val_loss')
    metric_pattern = monitor_metric.replace("/", "_")

    # Try multiple patterns to extract the metric value:
    # 1. Full metric name: "train_loss_total_epoch=0.1234"
    # 2. Abbreviated in filename: "loss=0.1234" (when metric is "train_loss_total_epoch")
    # 3. Other common abbreviations

    patterns = [
        rf"{metric_pattern}=([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",  # Full name
    ]

    # Add abbreviated patterns by extracting the last part after '_' or '/'
    if "_" in monitor_metric or "/" in monitor_metric:
        short_name = monitor_metric.split("_")[-1].split("/")[-1]
        patterns.append(rf"{short_name}=([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def setup_seed_everything():
    """
    Return Lightning's canonical seed helper.

    Returns:
        seed_everything function
    """
    from pytorch_lightning import seed_everything

    return seed_everything


def compute_tta_passes(cfg: Config, spatial_dims: int = 3) -> int:
    """Return the total number of TTA inference passes from config.

    This determines the multiplier in the cached prediction filename
    (e.g. ``_tta_x16_prediction.h5``).  When TTA is disabled the count is 1.
    """
    inference_cfg = getattr(cfg, "inference", None)
    if inference_cfg is None:
        return 1
    tta_cfg = getattr(inference_cfg, "test_time_augmentation", None)
    if tta_cfg is None or not bool(getattr(tta_cfg, "enabled", False)):
        return 1
    from ...inference.tta import resolve_tta_augmentation_combinations

    return len(
        resolve_tta_augmentation_combinations(
            tta_cfg,
            spatial_dims=spatial_dims,
        )
    )


def format_select_channel_tag(cfg: Config) -> str:
    """Return a compact channel-selection tag for prediction filenames.

    When ``select_channel`` is set in the TTA config, the tag disambiguates
    cached predictions produced with different channel selections, e.g.
    ``"_ch4-6-9"`` for ``select_channel: [4, 6, 9]``.

    Returns an empty string when no channel selection is active (all channels
    are kept).
    """
    inference_cfg = getattr(cfg, "inference", None)
    if inference_cfg is None:
        return ""
    tta_cfg = getattr(inference_cfg, "test_time_augmentation", None)
    if tta_cfg is None:
        return ""

    sel = getattr(tta_cfg, "select_channel", None)
    if sel is None:
        sel = getattr(inference_cfg, "output_channel", None)
    if sel is None:
        return ""

    # Coerce to a list of ints when possible
    if isinstance(sel, (list, tuple)):
        indices = [int(x) for x in sel]
    elif isinstance(sel, int):
        indices = [sel]
    elif isinstance(sel, str):
        s = sel.strip()
        if s == ":" or s == "":
            return ""  # all channels
        return f"_ch{s.replace(':', '-')}"
    else:
        return ""

    return "_ch" + "-".join(str(i) for i in indices)


def format_decode_tag(cfg: Config) -> str:
    """Return a compact decoding-parameter tag for final prediction filenames.

    Encodes every decode step and all decode-kwarg values so that different
    decoding configurations produce distinct output files without repeating
    verbose kwarg names. Returns an empty string when no decoding is configured.
    """

    def _sanitize_decode_component(text: str) -> str:
        safe_text = re.sub(r"[^A-Za-z0-9._=]+", "-", text)
        safe_text = re.sub(r"-{2,}", "-", safe_text)
        return safe_text.strip("-")

    def _flatten_decode_values(value) -> list[str]:
        if hasattr(value, "items"):
            result: list[str] = []
            for _key, nested_value in sorted(dict(value).items()):
                result.extend(_flatten_decode_values(nested_value))
            return result
        if isinstance(value, (list, tuple)):
            result: list[str] = []
            for nested_value in value:
                result.extend(_flatten_decode_values(nested_value))
            return result
        if isinstance(value, bool):
            return ["true" if value else "false"]
        if value is None:
            return ["none"]
        if isinstance(value, float):
            return [format(value, "g")]
        return [str(value)]

    inference_cfg = getattr(cfg, "inference", None)
    if inference_cfg is None:
        return ""
    decoding = getattr(inference_cfg, "decoding", None)
    if not decoding:
        return ""

    try:
        steps = list(decoding)
    except TypeError:
        steps = [decoding]

    parts = []
    for step in steps:
        name = getattr(step, "name", None) or (step.get("name") if isinstance(step, dict) else None)
        if not name:
            continue

        short = name.replace("decode_", "")
        kwargs = getattr(step, "kwargs", None)
        if kwargs is None and isinstance(step, dict):
            kwargs = step.get("kwargs", {})

        value_tokens = _flatten_decode_values(kwargs) if kwargs is not None else []
        if not value_tokens:
            parts.append(short)
            continue

        kwargs_tag = _sanitize_decode_component("-".join(value_tokens))
        parts.append(f"{short}_{kwargs_tag}" if kwargs_tag else short)

    if not parts:
        return ""
    return "_" + "__".join(parts)


def format_checkpoint_name_tag(checkpoint_path: Optional[str | Path]) -> str:
    """Return a compact checkpoint tag for prediction cache filenames."""
    if checkpoint_path is None:
        return ""

    path_value = str(checkpoint_path).strip()
    if not path_value:
        return ""

    stem = Path(path_value).expanduser().stem.strip()
    if not stem:
        return ""

    safe_stem = re.sub(r"[^A-Za-z0-9._=-]+", "-", stem).strip("-")
    if not safe_stem:
        return ""

    return f"_ckpt-{safe_stem}"


def final_prediction_output_tag(
    cfg: Config, spatial_dims: int = 3, checkpoint_path: Optional[str | Path] = None
) -> str:
    """Return the final decoded prediction tag used in output filenames."""
    n = compute_tta_passes(cfg, spatial_dims=spatial_dims)
    ch = format_select_channel_tag(cfg)
    ckpt = format_checkpoint_name_tag(checkpoint_path)
    dec = format_decode_tag(cfg)
    return f"x{n}{ch}{ckpt}_prediction{dec}"


def tta_cache_suffix(
    cfg: Config, spatial_dims: int = 3, checkpoint_path: Optional[str | Path] = None
) -> str:
    """Return the TTA prediction cache suffix, e.g. ``_tta_x16_ch4-6-9_ckpt-last_prediction.h5``."""
    n = compute_tta_passes(cfg, spatial_dims=spatial_dims)
    ch = format_select_channel_tag(cfg)
    ckpt = format_checkpoint_name_tag(checkpoint_path)
    return f"_tta_x{n}{ch}{ckpt}_prediction.h5"


def tta_cache_suffix_candidates(
    cfg: Config, spatial_dims: int = 3, checkpoint_path: Optional[str | Path] = None
) -> list[str]:
    """Return exact TTA cache suffix candidates ordered from most to least specific."""
    candidates = [tta_cache_suffix(cfg, spatial_dims=spatial_dims, checkpoint_path=checkpoint_path)]
    if checkpoint_path is not None:
        return candidates

    legacy_suffix = tta_cache_suffix(cfg, spatial_dims=spatial_dims)
    if legacy_suffix not in candidates:
        candidates.append(legacy_suffix)
    return candidates


def tuning_artifact_tag(
    cfg: Config, spatial_dims: int = 3, checkpoint_path: Optional[str | Path] = None
) -> str:
    """Return the cache-style tuning tag without leading underscore or file extension."""
    return Path(
        tta_cache_suffix(cfg, spatial_dims=spatial_dims, checkpoint_path=checkpoint_path)
    ).stem.lstrip("_")


def tuning_best_params_filename(
    cfg: Config, spatial_dims: int = 3, checkpoint_path: Optional[str | Path] = None
) -> str:
    """Return the checkpoint/channel-aware best-params filename for tune outputs."""
    return f"best_params_{tuning_artifact_tag(cfg, spatial_dims=spatial_dims, checkpoint_path=checkpoint_path)}.yaml"


def tuning_best_params_filename_candidates(
    cfg: Config, spatial_dims: int = 3, checkpoint_path: Optional[str | Path] = None
) -> list[str]:
    """Return ordered candidate best-params filenames, including legacy fallback."""
    candidates = [
        tuning_best_params_filename(cfg, spatial_dims=spatial_dims, checkpoint_path=checkpoint_path)
    ]
    legacy_name = "best_params.yaml"
    if legacy_name not in candidates:
        candidates.append(legacy_name)
    return candidates


def tuning_study_db_filename(
    cfg: Config,
    study_name: str,
    spatial_dims: int = 3,
    checkpoint_path: Optional[str | Path] = None,
) -> str:
    """Return a checkpoint/channel-aware SQLite filename for saved Optuna studies."""
    safe_study = re.sub(r"[^A-Za-z0-9._=-]+", "-", str(study_name)).strip("-")
    if not safe_study:
        safe_study = "parameter_optimization"
    return (
        f"{safe_study}_"
        f"{tuning_artifact_tag(cfg, spatial_dims=spatial_dims, checkpoint_path=checkpoint_path)}.db"
    )


def resolve_prediction_cache_suffix(
    cfg: Config, mode: str, checkpoint_path: Optional[str | Path] = None
) -> str:
    """Return the expected prediction cache suffix for the current runtime mode."""
    inference_cfg = getattr(cfg, "inference", None)
    save_prediction_cfg = getattr(inference_cfg, "save_prediction", None)
    configured_suffix = getattr(save_prediction_cfg, "cache_suffix", "_x1_prediction.h5")

    if mode in ("tune", "tune-test"):
        return tta_cache_suffix(cfg, checkpoint_path=checkpoint_path)

    if mode == "test":
        tta_cfg = getattr(inference_cfg, "test_time_augmentation", None)
        if tta_cfg is not None and bool(getattr(tta_cfg, "enabled", False)):
            return tta_cache_suffix(cfg, checkpoint_path=checkpoint_path)

    return configured_suffix


def is_tta_cache_suffix(suffix: str | None) -> bool:
    """Return True for any TTA intermediate prediction suffix (``_tta_x*_prediction.h5``)."""
    if not suffix:
        return False
    return suffix.startswith("_tta_x") and suffix.endswith("_prediction.h5")


__all__ = [
    "parse_args",
    "setup_config",
    "expand_file_paths",
    "extract_best_score_from_checkpoint",
    "setup_seed_everything",
    "compute_tta_passes",
    "format_select_channel_tag",
    "format_decode_tag",
    "format_checkpoint_name_tag",
    "final_prediction_output_tag",
    "tta_cache_suffix",
    "tta_cache_suffix_candidates",
    "tuning_artifact_tag",
    "tuning_best_params_filename",
    "tuning_best_params_filename_candidates",
    "tuning_study_db_filename",
    "resolve_prediction_cache_suffix",
    "is_tta_cache_suffix",
]
