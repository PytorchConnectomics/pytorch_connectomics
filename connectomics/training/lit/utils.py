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
import re
from glob import glob
from pathlib import Path
from typing import List, Optional

import torch

from ...config import (
    Config,
    load_config,
    update_from_cli,
    validate_config,
    resolve_data_paths,
)


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
        help="Run quick demo with synthetic data (30 seconds, no config needed)",
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
        "--external-prefix",
        type=str,
        default=None,
        help="Prefix to strip from external checkpoint keys (e.g., 'model.' for BANIS checkpoints)",
    )
    # Parameter tuning arguments
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to parameter file (overrides config parameter_source)",
    )
    parser.add_argument(
        "--param-source",
        choices=["fixed", "tuned", "optuna"],
        default=None,
        help="Parameter source: fixed, tuned, or optuna (overrides config)",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=None,
        help="Number of Optuna trials (overrides config, use with --mode tune or tune-test)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in key=value format (e.g., data.batch_size=8)",
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
    print(f"📄 Loading config: {args.config}")
    cfg = load_config(args.config)

    # Resolve data paths (combine train_path with train_image, etc.)
    cfg = resolve_data_paths(cfg)

    # Extract config file name and set output folder
    # Include architecture name for unified configs (e.g., lucchi++_rsunet)
    config_path = Path(args.config)
    config_name = config_path.stem  # Get filename without extension
    arch_name = cfg.model.architecture
    output_folder = f"outputs/{config_name}_{arch_name}/"

    # Update checkpoint dirpath only if not provided by the user
    if not getattr(cfg.monitor.checkpoint, "dirpath", None):
        cfg.monitor.checkpoint.dirpath = str(Path(output_folder) / "checkpoints")
    else:
        cfg.monitor.checkpoint.dirpath = str(Path(cfg.monitor.checkpoint.dirpath))

    # Update test output path only if test section exists and output_path not provided
    if hasattr(cfg, "test") and hasattr(cfg.test, "data"):
        if not getattr(cfg.test.data, "output_path", None):
            cfg.test.data.output_path = str(Path(output_folder) / "results")
        else:
            cfg.test.data.output_path = str(Path(cfg.test.data.output_path))
        print(f"📂 Test output directory: {cfg.test.data.output_path}")

    # Note: We handle timestamping manually in main() to create run directories
    # Set this to False to prevent PyTorch Lightning from adding its own timestamp
    cfg.monitor.checkpoint.use_timestamp = False

    print(f"📁 Checkpoints base directory: {cfg.monitor.checkpoint.dirpath}")

    # Apply CLI overrides
    if args.overrides:
        print(f"⚙️  Applying {len(args.overrides)} CLI overrides")
        cfg = update_from_cli(cfg, args.overrides)

    # Override max_epochs if --reset-max-epochs is specified
    if args.reset_max_epochs is not None:
        print(f"⚙️  Overriding max_epochs: {cfg.optimization.max_epochs} → {args.reset_max_epochs}")
        cfg.optimization.max_epochs = args.reset_max_epochs

    # Handle external weights loading (when --external-prefix is specified with --checkpoint)
    if args.external_prefix is not None and args.checkpoint:
        print(f"🔧 Loading external weights from checkpoint with prefix '{args.external_prefix}'")
        cfg.model.external_weights_path = args.checkpoint
        cfg.model.external_weights_key_prefix = args.external_prefix

    # Override config for fast-dev-run mode
    if args.fast_dev_run:
        print("🔧 Fast-dev-run mode: Overriding config for debugging")
        print(f"   - num_gpus: {cfg.system.training.num_gpus} → 1")
        print(f"   - num_cpus: {cfg.system.training.num_cpus} → 1")
        print(
            f"   - num_workers: {cfg.system.training.num_workers} → 0 (avoid multiprocessing in debug mode)"
        )
        print(
            f"   - batch_size: Controlled by PyTorch Lightning (--fast-dev-run={args.fast_dev_run})"
        )
        print("   - input patch: 64^3 for lightweight debug")
        print("   - MedNeXt size: S for lightweight debug")
        cfg.system.training.num_gpus = 1
        cfg.system.training.num_cpus = 1
        cfg.system.training.num_workers = 0
        cfg.system.inference.num_gpus = 1
        cfg.system.inference.num_cpus = 1
        cfg.system.inference.num_workers = 0
        if hasattr(cfg.model, "input_size"):
            cfg.model.input_size = [64, 64, 64]
        if hasattr(cfg.model, "output_size"):
            cfg.model.output_size = [64, 64, 64]
        if hasattr(cfg.model, "mednext_size"):
            cfg.model.mednext_size = "S"
        # Keep CellMap shapes in sync with the smaller debug patch
        if getattr(cfg.data, "cellmap", None):
            cfg.data.cellmap["input_array_info"]["shape"] = [64, 64, 64]
            cfg.data.cellmap["target_array_info"]["shape"] = [64, 64, 64]

    # CPU-only fallback: avoid multiprocessing workers when no CUDA is available
    if not torch.cuda.is_available():
        if cfg.system.training.num_workers > 0:
            print(
                "🔧 CUDA not available, setting training num_workers=0 to avoid dataloader crashes"
            )
            cfg.system.training.num_workers = 0
        if cfg.system.inference.num_workers > 0:
            print(
                "🔧 CUDA not available, setting inference num_workers=0 to avoid dataloader crashes"
            )
            cfg.system.inference.num_workers = 0

    # Apply inference-specific overrides if in test/tune mode
    if args.mode in ["test", "tune", "tune-test"]:
        if cfg.inference.num_gpus >= 0:
            print(f"🔧 Inference override: num_gpus={cfg.inference.num_gpus}")
            cfg.system.training.num_gpus = cfg.inference.num_gpus
        if cfg.inference.num_cpus >= 0:
            print(f"🔧 Inference override: num_cpus={cfg.inference.num_cpus}")
            cfg.system.training.num_cpus = cfg.inference.num_cpus
        if cfg.inference.batch_size >= 0:
            print(f"🔧 Inference override: batch_size={cfg.inference.batch_size}")
            cfg.system.inference.batch_size = cfg.inference.batch_size
        if cfg.inference.num_workers >= 0:
            print(f"🔧 Inference override: num_workers={cfg.inference.num_workers}")
            cfg.system.inference.num_workers = cfg.inference.num_workers

    # Auto-planning (if enabled)
    if hasattr(cfg.system, "auto_plan") and cfg.system.auto_plan:
        print("🤖 Running automatic configuration planning...")
        from ...config import auto_plan_config

        print_results = (
            cfg.system.print_auto_plan if hasattr(cfg.system, "print_auto_plan") else True
        )
        cfg = auto_plan_config(cfg, print_results=print_results)

    # Validate configuration
    print("✅ Validating configuration...")
    validate_config(cfg)

    # Note: Output directory will be created later in main() with timestamp
    # (see lines around "Create run directory only for training mode")

    return cfg


def expand_file_paths(path_or_pattern) -> List[str]:
    """
    Expand glob patterns to list of file paths.

    Args:
        path_or_pattern: Single file path, glob pattern, or list of paths/patterns

    Returns:
        List of expanded file paths, sorted alphabetically
    """
    # If already a list, return it (may have been expanded by resolve_data_paths)
    if isinstance(path_or_pattern, list):
        return path_or_pattern

    # Check if pattern contains wildcards
    if "*" in path_or_pattern or "?" in path_or_pattern:
        # Expand glob pattern
        paths = sorted(glob(path_or_pattern))
        if not paths:
            raise FileNotFoundError(f"No files found matching pattern: {path_or_pattern}")
        return paths
    else:
        # Single file path
        return [path_or_pattern]


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


__all__ = [
    "parse_args",
    "setup_config",
    "expand_file_paths",
    "extract_best_score_from_checkpoint",
]
