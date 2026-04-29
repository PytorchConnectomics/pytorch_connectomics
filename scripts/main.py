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
    python scripts/main.py --config tutorials/mito_lucchi++.yaml --mode test \
        --checkpoint path/to/checkpoint.ckpt

    # Fast dev run (1 batch for debugging, auto-sets num_workers=0 and uses
    # GPU only if CUDA is available)
    python scripts/main.py --config tutorials/mito_lucchi++.yaml --fast-dev-run
    python scripts/main.py --config tutorials/mito_lucchi++.yaml --fast-dev-run 2  # Run 2 batches

    # Demo mode (uses tutorials/minimal.yaml + fast-dev-run)
    python scripts/main.py --demo

    # Override config parameters
    python scripts/main.py --config tutorials/mito_lucchi++.yaml \
        data.dataloader.batch_size=8 optimization.max_epochs=200

    # Print fully resolved runtime config
    python scripts/main.py --config tutorials/mito_lucchi++.yaml --debug-config

    # Resume training with different max_epochs
    python scripts/main.py --config tutorials/mito_lucchi++.yaml \
        --checkpoint path/to/ckpt.ckpt --reset-max-epochs 500
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from connectomics.config import Config  # noqa: E402
from connectomics.runtime.cache_resolver import (  # noqa: E402
    create_decode_only_datamodule as _create_decode_only_datamodule,
)
from connectomics.runtime.cache_resolver import handle_test_cache_hit as _handle_test_cache_hit
from connectomics.runtime.cache_resolver import (
    has_cached_predictions_in_output_dir as _has_cached_predictions_in_output_dir,
)
from connectomics.runtime.cache_resolver import has_tta_prediction_file as _has_tta_prediction_file
from connectomics.runtime.cache_resolver import (
    is_test_evaluation_enabled as _is_test_evaluation_enabled,
)
from connectomics.runtime.cache_resolver import (
    preflight_test_cache_hit,
    try_cache_only_test_execution,
)
from connectomics.runtime.checkpoint_dispatch import (  # noqa: E402
    setup_runtime_directories as _setup_runtime_directories,
)
from connectomics.runtime.cli import parse_args, setup_config  # noqa: E402
from connectomics.runtime.output_naming import resolve_prediction_cache_suffix  # noqa: E402
from connectomics.runtime.sharding import (  # noqa: E402
    has_assigned_test_shard,
    maybe_enable_independent_test_sharding,
    maybe_limit_test_devices,
    resolve_test_stage_runtime,
    shard_test_datamodule,
)
from connectomics.runtime.torch_safe_globals import register_torch_safe_globals  # noqa: E402

register_torch_safe_globals()

# Import Lightning components and utilities
from connectomics.training.lightning import (  # noqa: E402
    ConnectomicsModule,
    cleanup_run_directory,
    create_datamodule,
    create_trainer,
    modify_checkpoint_state,
    setup_seed_everything,
)

# Setup seed_everything helper
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
    requested_gpus = cfg.system.num_gpus
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
            print("Enabled float32 matmul precision='medium' (Tensor Cores detected)")
    except Exception as exc:
        print(f"WARNING: Could not configure float32 matmul precision automatically: {exc}")


def main():
    """Main training function."""
    suppress_nonzero_rank_stdout()

    # Parse arguments
    args = parse_args()

    # Handle demo mode: route to canonical minimal config workflow.
    if args.demo:
        minimal_config = Path(__file__).parent.parent / "tutorials" / "minimal.yaml"
        if not minimal_config.exists():
            print(f"Error: Demo config not found: {minimal_config}")
            sys.exit(1)
        if not args.config:
            args.config = str(minimal_config)
        if args.fast_dev_run == 0:
            args.fast_dev_run = 1
        if args.mode != "train":
            args.mode = "train"
        print(f"Demo mode: using minimal config {args.config}")

    # Validate that config is provided for non-demo modes
    if not args.config:
        print("Error: --config is required (or use --demo for a quick test)")
        print("\nUsage:")
        print("  python scripts/main.py --config tutorials/mito_lucchi++.yaml")
        print("  python scripts/main.py --demo")
        sys.exit(1)

    # Setup config
    print("\n" + "=" * 60)
    print("PyTorch Connectomics Hydra Training")
    print("=" * 60)
    cfg = setup_config(args)
    configure_matmul_precision(cfg)

    # Keep cache lookup aligned with the current runtime mode and TTA plan.
    if args.mode in ["test", "tune", "tune-test"]:
        cfg.inference.save_prediction.cache_suffix = resolve_prediction_cache_suffix(cfg, args.mode)

    # Run preflight checks for training mode
    if args.mode == "train":
        from connectomics.runtime import preflight_check, print_preflight_issues

        issues = preflight_check(cfg)
        if issues:
            print_preflight_issues(issues)

    # Setup run directory (handles DDP coordination and config saving)
    run_dir, output_base = _setup_runtime_directories(args, cfg)

    # Set random seed
    if cfg.system.seed is not None:
        print(f"Random seed set to: {cfg.system.seed}")
        seed_everything(cfg.system.seed, workers=True)

    if args.mode == "test":
        maybe_enable_independent_test_sharding(args, cfg)
        if not has_assigned_test_shard(cfg, args):
            return

    # Cache-only preflight path for test mode (can skip model/trainer/dataloader entirely).
    if try_cache_only_test_execution(
        cfg,
        args.mode,
        args.shard_id,
        args.num_shards,
        checkpoint_path=args.checkpoint,
    ):
        return

    # Check for cached/external predictions early so we can skip both the
    # expensive model build and checkpoint restore for test/tune modes.
    _saved_pred = getattr(getattr(cfg, "decoding", None), "input_prediction_path", "")
    has_saved_prediction = bool(
        _saved_pred and isinstance(_saved_pred, str) and _saved_pred.strip()
    )
    tta_cached = args.mode in ("test", "tune", "tune-test") and (
        has_saved_prediction
        or _has_tta_prediction_file(cfg)
        or _has_cached_predictions_in_output_dir(
            cfg,
            mode=args.mode,
            checkpoint_path=args.checkpoint,
        )
    )

    # Create model
    if has_saved_prediction:
        print(f"  Decode-only mode: loading predictions from {_saved_pred}")
        print("  Skipping model build entirely.")
        model = ConnectomicsModule(cfg, model=torch.nn.Identity(), skip_loss=True)
        model._skip_inference = True
        ckpt_path = None
    elif tta_cached:
        print(
            f"  Cached intermediate predictions found; "
            f"creating lightweight module (skipping {cfg.model.arch.type} build)."
        )
        model = ConnectomicsModule(cfg, model=torch.nn.Identity())
        model._skip_inference = True
        ckpt_path = None
    elif args.external_prefix:
        print(f"Creating model: {cfg.model.arch.type}")
        model = ConnectomicsModule(cfg)
        print(
            "   WARNING: External weights loaded - checkpoint path will not "
            "be used for training/testing"
        )
        ckpt_path = None
    else:
        print(f"Creating model: {cfg.model.arch.type}")
        model = ConnectomicsModule(cfg)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {num_params:,}")
        ckpt_path = modify_checkpoint_state(
            args.checkpoint,
            run_dir,
            reset_optimizer=args.reset_optimizer,
            reset_scheduler=args.reset_scheduler,
            reset_epoch=args.reset_epoch,
            reset_early_stopping=args.reset_early_stopping,
        )

    model._prediction_checkpoint_path = args.checkpoint or getattr(
        getattr(cfg, "model", None), "external_weights_path", None
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
            print("STARTING TRAINING")
            print("=" * 60)

            trainer.fit(
                model,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
            )
            print("\n[OK]Training completed successfully!")

        # Handle tune modes
        if args.mode in ["tune", "tune-test"]:
            from connectomics.runtime.tune_runner import run_tuning

            # Run parameter tuning (automatically skips if best_params.yaml exists)
            run_tuning(model, trainer, cfg, checkpoint_path=ckpt_path)

        # Handle test modes
        if args.mode in ["tune-test", "test"]:
            print("\n" + "=" * 60)
            print("RUNNING TEST")
            print("=" * 60)

            # Re-resolve test-stage runtime overrides after tuning, including sentinels.
            cfg = resolve_test_stage_runtime(cfg)
            cfg.inference.save_prediction.cache_suffix = resolve_prediction_cache_suffix(
                cfg,
                args.mode,
                checkpoint_path=args.checkpoint,
            )

            if maybe_enable_independent_test_sharding(args, cfg):
                trainer = create_trainer(
                    cfg,
                    run_dir=run_dir,
                    fast_dev_run=args.fast_dev_run,
                    ckpt_path=ckpt_path,
                    mode="test",
                )
            if not has_assigned_test_shard(cfg, args):
                return

            # Create datamodule (or dummy for decode-only mode)
            if has_saved_prediction:
                datamodule = _create_decode_only_datamodule(cfg, _saved_pred)
            else:
                datamodule = create_datamodule(cfg, mode="test")

            # Apply test volume sharding across machines
            if args.shard_id is not None and args.num_shards is not None:
                datamodule = shard_test_datamodule(datamodule, args.shard_id, args.num_shards)

            if maybe_limit_test_devices(cfg, datamodule):
                trainer = create_trainer(
                    cfg,
                    run_dir=run_dir,
                    fast_dev_run=args.fast_dev_run,
                    ckpt_path=ckpt_path,
                    mode="test",
                )

            if args.mode == "tune-test":
                from connectomics.runtime.tune_runner import load_and_apply_best_params

                print("\n" + "=" * 80)
                print("LOADING BEST PARAMETERS")
                print("=" * 80)

                # Load and apply best parameters
                cfg = load_and_apply_best_params(cfg, checkpoint_path=args.checkpoint)
                cfg.inference.save_prediction.cache_suffix = resolve_prediction_cache_suffix(
                    cfg,
                    args.mode,
                    checkpoint_path=args.checkpoint,
                )

            test_ckpt_path = ckpt_path
            cache_hit, cached_suffix, cache_count = preflight_test_cache_hit(
                cfg,
                datamodule,
                checkpoint_path=args.checkpoint,
            )
            if cache_hit:
                skip_test_loop, test_ckpt_path = _handle_test_cache_hit(
                    args,
                    cfg,
                    cached_suffix,
                    cache_count,
                    ckpt_path,
                )
                if skip_test_loop:
                    return

            trainer.test(
                model,
                datamodule,
                ckpt_path=test_ckpt_path,
            )

    except Exception as e:
        mode_name = args.mode.capitalize() if args.mode else "Operation"
        print(f"\n{mode_name} failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup: Remove timestamp file after training
        if args.mode == "train" and "output_base" in locals():
            cleanup_run_directory(output_base)


if __name__ == "__main__":
    main()
