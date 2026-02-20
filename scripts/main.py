#!/usr/bin/env python3
"""
PyTorch Connectomics training script with Hydra configuration and Lightning framework.

This script provides modern deep learning training with:
- Hydra-based configuration management
- Automatic distributed training and mixed precision
- MONAI-based data augmentation
- PyTorch Lightning callbacks and logging

Usage:
    # Basic training
    python scripts/main.py --config tutorials/mito_lucchi++.yaml

    # Testing mode
    python scripts/main.py --config tutorials/mito_lucchi++.yaml --mode test --checkpoint path/to/checkpoint.ckpt

    # Fast dev run (1 batch for debugging, auto-sets num_gpus=1, num_cpus=1, num_workers=1)
    python scripts/main.py --config tutorials/mito_lucchi++.yaml --fast-dev-run
    python scripts/main.py --config tutorials/mito_lucchi++.yaml --fast-dev-run 2  # Run 2 batches

    # Override config parameters
    python scripts/main.py --config tutorials/mito_lucchi++.yaml data.batch_size=8 optimization.max_epochs=200

    # Resume training with different max_epochs
    python scripts/main.py --config tutorials/mito_lucchi++.yaml --checkpoint path/to/ckpt.ckpt --reset-max-epochs 500
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Import Hydra config system
from connectomics.config import Config, save_config
import connectomics.config.hydra_config as hydra_config

# Register safe globals for PyTorch 2.6+ checkpoint loading
# Allowlist all Config dataclasses used inside Lightning checkpoints
try:
    _config_classes = [
        obj
        for obj in hydra_config.__dict__.values()
        if isinstance(obj, type) and obj.__name__.endswith("Config")
    ]
    torch.serialization.add_safe_globals(_config_classes)
except AttributeError:
    # PyTorch < 2.6 doesn't have add_safe_globals
    pass

# Import Lightning components and utilities
from connectomics.training.lit import (
    ConnectomicsModule,
    cleanup_run_directory,
    create_datamodule,
    create_trainer,
    modify_checkpoint_state,
    parse_args,
    setup_config,
    setup_run_directory,
    setup_seed_everything,
)


# Setup seed_everything with version fallback
seed_everything = setup_seed_everything()

_RANK_STDOUT_REDIRECT = None


def suppress_nonzero_rank_stdout() -> None:
    """Reduce duplicate stdout spam from DDP subprocesses.

    In local multi-GPU spawn, each subprocess executes this script and prints
    the same setup logs. Keep rank 0 stdout visible and silence stdout on
    non-zero ranks. stderr is untouched for error visibility.
    """
    global _RANK_STDOUT_REDIRECT
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None or local_rank == "0":
        return
    _RANK_STDOUT_REDIRECT = open(os.devnull, "w")
    sys.stdout = _RANK_STDOUT_REDIRECT


def configure_matmul_precision(cfg: Config) -> None:
    """Enable Tensor Core matmul precision when supported by available CUDA devices."""
    requested_gpus = max(cfg.system.training.num_gpus, cfg.system.inference.num_gpus)
    if requested_gpus <= 0 or not torch.cuda.is_available():
        return

    try:
        visible_gpus = torch.cuda.device_count()
        check_gpus = min(requested_gpus, visible_gpus)

        has_tensor_cores = False
        for idx in range(check_gpus):
            major, _minor = torch.cuda.get_device_capability(idx)
            # Tensor Cores are available on NVIDIA Volta (SM 7.0) and newer.
            if major >= 7:
                has_tensor_cores = True
                break

        if has_tensor_cores:
            torch.set_float32_matmul_precision("medium")
            print("⚙️  Enabled float32 matmul precision='medium' (Tensor Cores detected)")
    except Exception as exc:
        print(f"⚠️  Could not configure float32 matmul precision automatically: {exc}")


def get_output_base_from_checkpoint(checkpoint_path: str) -> Path:
    """
    Determine the output base directory from checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Path to use as base for results/ and tuning/ folders

    Logic:
        1. If checkpoint contains timestamp folder (YYYYMMDD_HHMMSS), use that folder
        2. Otherwise, use checkpoint parent folder / checkpoint_stem

    Examples:
        "outputs/exp/20241124_203930/checkpoints/last.ckpt"
            → "outputs/exp/20241124_203930/"
        "pretrained_models/model.ckpt"
            → "pretrained_models/model/"
    """
    import re

    ckpt_path = Path(checkpoint_path)

    # Look for timestamp pattern (YYYYMMDD_HHMMSS) in path parts
    timestamp_pattern = re.compile(r"^\d{8}_\d{6}$")

    for parent in ckpt_path.parents:
        if timestamp_pattern.match(parent.name):
            # Found timestamp folder, use it as base
            return parent

    # No timestamp found, create folder based on checkpoint filename
    # Use checkpoint's grandparent / checkpoint_stem
    ckpt_stem = ckpt_path.stem  # e.g., "last" or "model"
    return ckpt_path.parent.parent / ckpt_stem


def extract_step_from_checkpoint(checkpoint_path: str) -> str:
    """
    Extract step number from checkpoint filename.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Step string (e.g., "step=195525") or empty string if not found

    Examples:
        "epoch=494-step=195525.ckpt" → "step=195525"
        "last.ckpt" → ""
        "model-epoch=10-step=5000.ckpt" → "step=5000"
    """
    import re

    ckpt_path = Path(checkpoint_path)
    filename = ckpt_path.stem  # Remove .ckpt extension

    # Look for step=XXXXX pattern
    step_pattern = re.compile(r"step=(\d+)")
    match = step_pattern.search(filename)

    if match:
        return f"step={match.group(1)}"
    
    return ""


def main():
    """Main training function."""
    suppress_nonzero_rank_stdout()

    # Parse arguments
    args = parse_args()

    # Handle demo mode
    if args.demo:
        from scripts.demo import run_demo

        run_demo()
        return

    # Validate that config is provided for non-demo modes
    if not args.config:
        print("❌ Error: --config is required (or use --demo for a quick test)")
        print("\nUsage:")
        print("  python scripts/main.py --config tutorials/mito_lucchi++.yaml")
        print("  python scripts/main.py --demo")
        sys.exit(1)

    # Setup config
    print("\n" + "=" * 60)
    print("🚀 PyTorch Connectomics Hydra Training")
    print("=" * 60)
    cfg = setup_config(args)
    configure_matmul_precision(cfg)

    # Run preflight checks for training mode
    if args.mode == "train":
        from connectomics.utils.errors import preflight_check, print_preflight_issues

        issues = preflight_check(cfg)
        if issues:
            print_preflight_issues(issues)

    # Setup run directory (handles DDP coordination and config saving)
    # Determine output base directory from checkpoint for test/tune modes
    if args.mode in ["test", "tune", "tune-test"] and args.checkpoint:
        # Extract base directory from checkpoint path (same logic for all modes)
        output_base = get_output_base_from_checkpoint(args.checkpoint)
        output_base.mkdir(parents=True, exist_ok=True)

        # Extract step number from checkpoint filename (if available)
        step_suffix = extract_step_from_checkpoint(args.checkpoint)
        
        # Create mode-specific subdirectories
        if args.mode in ["tune", "tune-test"]:
            # For tuning modes, append step suffix if available
            if step_suffix:
                results_folder_name = f"results_{step_suffix}"
                tuning_folder_name = f"tuning_{step_suffix}"
            else:
                results_folder_name = "results"
                tuning_folder_name = "tuning"
            
            dirpath = str(output_base / tuning_folder_name)
            results_path = str(output_base / results_folder_name)
            # Override tune output directories in config
            if cfg.tune is not None:
                cfg.tune.output.output_dir = dirpath
                cfg.tune.output.output_pred = results_path
            # For tune-test, also set test output directory and cache suffix
            if args.mode == "tune-test":
                print(f"🔍 Setting test config for tune-test mode")
                print(f"🔍 cfg.test is None: {cfg.test is None}")
                if cfg.test is not None:
                    print(f"🔍 cfg.test.data is None: {cfg.test.data is None}")
                    if cfg.test.data is not None:
                        cfg.test.data.output_path = results_path
                        cfg.test.data.cache_suffix = cfg.tune.output.cache_suffix
                        print(f"📋 Test output: {cfg.test.data.output_path}")
                        print(f"📋 Test cache suffix: {cfg.test.data.cache_suffix}")
                    else:
                        print(f"❌ cfg.test.data is None, cannot set cache_suffix!")
                else:
                    print(f"❌ cfg.test is None, cannot set cache_suffix!")
        else:  # test mode
            # Create results/ folder with step suffix under checkpoint directory
            if step_suffix:
                results_folder_name = f"results_{step_suffix}"
                print(f"📋 Using checkpoint {step_suffix} - output will be saved to: {results_folder_name}")
            else:
                results_folder_name = "results"
            
            results_path = str(output_base / results_folder_name)
            dirpath = results_path
            # Override test output directory in config
            if hasattr(cfg, "test") and hasattr(cfg.test, "data"):
                cfg.test.data.output_path = results_path

        run_dir = setup_run_directory(args.mode, cfg, dirpath)
        print(f"📂 Output base: {output_base}")
    else:
        # Train mode or no checkpoint - use default config paths
        dirpath = cfg.monitor.checkpoint.dirpath
        run_dir = setup_run_directory(args.mode, cfg, dirpath)
        output_base = run_dir.parent

    # Set random seed
    if cfg.system.seed is not None:
        print(f"🎲 Random seed set to: {cfg.system.seed}")
        seed_everything(cfg.system.seed, workers=True)

    # Create model
    print(f"Creating model: {cfg.model.architecture}")
    model = ConnectomicsModule(cfg)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    # Handle checkpoint state resets if requested (function handles early return)
    if args.reset_max_epochs is not None:
        print(f"   - Overriding max_epochs to: {args.reset_max_epochs}")

    # Don't use checkpoint path if external weights were loaded (already in model state)
    # External weights are loaded during config setup via model.external_weights_path
    if args.external_prefix:
        print(
            f"   ⚠️  External weights loaded - checkpoint path will not be used for training/testing"
        )
        ckpt_path = None
    else:
        ckpt_path = modify_checkpoint_state(
            args.checkpoint,
            run_dir,
            reset_optimizer=args.reset_optimizer,
            reset_scheduler=args.reset_scheduler,
            reset_epoch=args.reset_epoch,
            reset_early_stopping=args.reset_early_stopping,
        )

    # Create trainer (pass run_dir for checkpoints and logs, and checkpoint path for resume)
    trainer = create_trainer(
        cfg,
        run_dir=run_dir,
        fast_dev_run=args.fast_dev_run,
        ckpt_path=ckpt_path,
        mode=args.mode,
    )

    # Main training/testing/tuning workflow
    try:
        if args.mode == "train":
            # Create datamodule
            datamodule = create_datamodule(
                cfg, mode=args.mode, fast_dev_run=bool(args.fast_dev_run)
            )
            print("\n" + "=" * 60)
            print("🏃 STARTING TRAINING")
            print("=" * 60)

            trainer.fit(
                model,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
            )
            print("\n✅ Training completed successfully!")

        # Handle tune modes
        if args.mode in ["tune", "tune-test"]:
            # Check if tune config exists and has parameter_space
            if cfg.tune is None or not hasattr(cfg.tune, "parameter_space"):
                raise ValueError("Missing tune or tune.parameter_space configuration")

            from connectomics.decoding import run_tuning

            # Run parameter tuning (automatically skips if best_params.yaml exists)
            run_tuning(model, trainer, cfg, checkpoint_path=ckpt_path)

        # Handle test modes
        if args.mode in ["tune-test", "test"]:
            print("\n" + "=" * 60)
            print("🧪 RUNNING TEST")
            print("=" * 60)

            # Create datamodule
            datamodule = create_datamodule(cfg, mode="test")

            if args.mode == "tune-test":
                from connectomics.decoding import load_and_apply_best_params

                print("\n" + "=" * 80)
                print("LOADING BEST PARAMETERS")
                print("=" * 80)

                # Load and apply best parameters
                cfg = load_and_apply_best_params(cfg)

            trainer.test(
                model,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
            )

    except Exception as e:
        mode_name = args.mode.capitalize() if args.mode else "Operation"
        print(f"\n❌ {mode_name} failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup: Remove timestamp file after training
        if args.mode == "train" and "output_base" in locals():
            cleanup_run_directory(output_base)


if __name__ == "__main__":
    main()
