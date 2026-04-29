"""Command-line parsing and runtime configuration setup."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from ..config import (
    Config,
    load_config,
    resolve_default_profiles,
    validate_config,
)
from ..config.hardware import resolve_runtime_resource_sentinels
from ..config.pipeline import print_config, resolve_data_paths, update_from_cli
from .preflight import validate_runtime_coherence

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
    """Load, merge, validate, and return the runtime config for the requested mode."""
    logger.info("Loading config: %s", args.config)
    cfg = load_config(args.config)

    config_path = Path(args.config)
    output_folder = f"outputs/{config_path.stem}/"

    if not getattr(cfg.monitor.checkpoint, "dirpath", None):
        cfg.monitor.checkpoint.dirpath = str(Path(output_folder) / "checkpoints")
    else:
        cfg.monitor.checkpoint.dirpath = str(Path(cfg.monitor.checkpoint.dirpath))

    save_pred_cfg = cfg.inference.save_prediction
    if not getattr(save_pred_cfg, "output_path", None):
        save_pred_cfg.output_path = str(Path(output_folder) / "results")
    else:
        save_pred_cfg.output_path = str(Path(save_pred_cfg.output_path))
    logger.info("Prediction output directory: %s", save_pred_cfg.output_path)

    cfg.monitor.checkpoint.use_timestamp = False
    logger.info("Checkpoints base directory: %s", cfg.monitor.checkpoint.dirpath)

    if args.overrides:
        logger.info("Applying %d CLI overrides", len(args.overrides))
        cfg = update_from_cli(cfg, args.overrides)

    cfg = resolve_default_profiles(cfg, mode=args.mode)
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

    if args.reset_max_epochs is not None:
        logger.info(
            "Overriding max_epochs: %s -> %s",
            cfg.optimization.max_epochs,
            args.reset_max_epochs,
        )
        cfg.optimization.max_epochs = args.reset_max_epochs

    if args.external_prefix is not None and args.checkpoint:
        logger.info("Loading external weights from checkpoint with prefix %r", args.external_prefix)
        cfg.model.external_weights_path = args.checkpoint
        cfg.model.external_weights_key_prefix = args.external_prefix

    if args.fast_dev_run:
        fast_dev_num_gpus = 1 if torch.cuda.is_available() else 0
        logger.info("Fast-dev-run mode: Overriding config for debugging")
        logger.info("   - num_gpus: %s -> %s", cfg.system.num_gpus, fast_dev_num_gpus)
        logger.info(
            "   - num_workers: %s -> 0 (avoid multiprocessing in debug mode)",
            cfg.system.num_workers,
        )
        logger.info(
            "   - batch_size: Controlled by PyTorch Lightning (--fast-dev-run=%s)",
            args.fast_dev_run,
        )
        logger.info("   - input patch: 64^3 for lightweight debug")
        logger.info("   - MedNeXt size: S for lightweight debug")
        cfg.system.num_gpus = fast_dev_num_gpus
        cfg.system.num_workers = 0
        cfg.model.input_size = [64, 64, 64]
        cfg.model.output_size = [64, 64, 64]
        cfg.data.dataloader.patch_size = [64, 64, 64]
        cfg.model.mednext.size = "S"
        if getattr(cfg.data, "cellmap", None):
            cfg.data.cellmap["input_array_info"]["shape"] = [64, 64, 64]
            cfg.data.cellmap["target_array_info"]["shape"] = [64, 64, 64]

    cfg = resolve_runtime_resource_sentinels(cfg, print_results=True)

    if not torch.cuda.is_available():
        if cfg.system.num_gpus > 0:
            logger.info("CUDA not available, setting num_gpus=0")
            cfg.system.num_gpus = 0
        if cfg.system.num_workers > 0:
            logger.info("CUDA not available, setting num_workers=0 to avoid dataloader crashes")
            cfg.system.num_workers = 0

    if getattr(args, "nnunet_preprocess", False):
        logger.info("Enabling nnU-Net preprocessing from CLI flag")
        cfg.data.nnunet_preprocessing.enabled = True

    logger.info("Validating configuration...")
    validate_config(cfg)
    validate_runtime_coherence(cfg)

    if getattr(args, "debug_config", False):
        logger.info("\n========================")
        logger.info("RESOLVED CONFIG (DEBUG)")
        logger.info("========================")
        print_config(cfg, resolve=True)

    return cfg


__all__ = ["parse_args", "setup_config"]
