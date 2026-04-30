"""Runtime mode dispatch for the command-line entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

from ..config import Config
from ..training.lightning import (
    ConnectomicsModule,
    cleanup_run_directory,
    create_datamodule,
    create_trainer,
    modify_checkpoint_state,
    setup_seed_everything,
)
from .cache_resolver import (
    create_decode_only_datamodule,
    handle_test_cache_hit,
    has_cached_predictions_in_output_dir,
    has_tta_prediction_file,
    preflight_test_cache_hit,
    try_cache_only_test_execution,
)
from .checkpoint_dispatch import setup_runtime_directories
from .output_naming import resolve_prediction_cache_suffix
from .sharding import (
    has_assigned_test_shard,
    maybe_enable_independent_test_sharding,
    maybe_limit_test_devices,
    resolve_test_stage_runtime,
    shard_test_datamodule,
)

_RANK_STDOUT_REDIRECT = None
seed_everything = setup_seed_everything()


def suppress_nonzero_rank_stdout() -> None:
    """Silence duplicate stdout from non-zero DDP subprocesses."""
    global _RANK_STDOUT_REDIRECT
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None or local_rank == "0":
        return
    _RANK_STDOUT_REDIRECT = open(os.devnull, "w")
    sys.stdout = _RANK_STDOUT_REDIRECT


def prepare_cli_args(args: Any, repo_root: Path) -> None:
    """Apply CLI-only defaults before config resolution."""
    if args.demo:
        minimal_config = repo_root / "tutorials" / "minimal.yaml"
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

    if not args.config:
        print("Error: --config is required (or use --demo for a quick test)")
        print("\nUsage:")
        print("  python scripts/main.py --config tutorials/mito_lucchi++.yaml")
        print("  python scripts/main.py --demo")
        sys.exit(1)


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
            if major >= 7:
                has_tensor_cores = True
                break

        if has_tensor_cores:
            torch.set_float32_matmul_precision("medium")
            print("Enabled float32 matmul precision='medium' (Tensor Cores detected)")
    except Exception as exc:
        print(f"WARNING: Could not configure float32 matmul precision automatically: {exc}")


def _create_runtime_model(
    args: Any,
    cfg: Config,
    run_dir: Path,
    *,
    has_saved_prediction: bool,
    saved_prediction_path: str,
    tta_cached: bool,
) -> tuple[ConnectomicsModule, str | None]:
    if has_saved_prediction:
        print(f"  Decode-only mode: loading predictions from {saved_prediction_path}")
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
        getattr(cfg, "model", None),
        "external_weights_path",
        None,
    )
    return model, ckpt_path


def _run_training(
    args: Any, cfg: Config, model: ConnectomicsModule, trainer: Any, ckpt_path
) -> None:
    datamodule = create_datamodule(cfg, mode=args.mode, fast_dev_run=bool(args.fast_dev_run))
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )
    print("\n[OK]Training completed successfully!")


def _run_test(
    args: Any,
    cfg: Config,
    model: ConnectomicsModule,
    trainer: Any,
    run_dir: Path,
    ckpt_path,
    *,
    has_saved_prediction: bool,
    saved_prediction_path: str,
) -> None:
    print("\n" + "=" * 60)
    print("RUNNING TEST")
    print("=" * 60)

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

    if has_saved_prediction:
        datamodule = create_decode_only_datamodule(cfg, saved_prediction_path)
    else:
        datamodule = create_datamodule(cfg, mode="test")

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
        from .tune_runner import load_and_apply_best_params

        print("\n" + "=" * 80)
        print("LOADING BEST PARAMETERS")
        print("=" * 80)

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
        skip_test_loop, test_ckpt_path = handle_test_cache_hit(
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


def dispatch_runtime(args: Any, cfg: Config) -> None:
    """Dispatch the configured runtime mode."""
    configure_matmul_precision(cfg)

    if args.mode in ["test", "tune", "tune-test"]:
        cfg.inference.save_prediction.cache_suffix = resolve_prediction_cache_suffix(cfg, args.mode)

    if args.mode == "train":
        from . import preflight_check, print_preflight_issues

        issues = preflight_check(cfg)
        if issues:
            print_preflight_issues(issues)

    run_dir, output_base = setup_runtime_directories(args, cfg)

    if cfg.system.seed is not None:
        print(f"Random seed set to: {cfg.system.seed}")
        seed_everything(cfg.system.seed, workers=True)

    if args.mode == "test":
        maybe_enable_independent_test_sharding(args, cfg)
        if not has_assigned_test_shard(cfg, args):
            return

    if try_cache_only_test_execution(
        cfg,
        args.mode,
        args.shard_id,
        args.num_shards,
        checkpoint_path=args.checkpoint,
    ):
        return

    saved_prediction_path = getattr(getattr(cfg, "decoding", None), "input_prediction_path", "")
    has_saved_prediction = bool(
        saved_prediction_path
        and isinstance(saved_prediction_path, str)
        and saved_prediction_path.strip()
    )
    tta_cached = args.mode in ("test", "tune", "tune-test") and (
        has_saved_prediction
        or has_tta_prediction_file(cfg)
        or has_cached_predictions_in_output_dir(
            cfg,
            mode=args.mode,
            checkpoint_path=args.checkpoint,
        )
    )

    model, ckpt_path = _create_runtime_model(
        args,
        cfg,
        run_dir,
        has_saved_prediction=has_saved_prediction,
        saved_prediction_path=saved_prediction_path,
        tta_cached=tta_cached,
    )

    trainer = create_trainer(
        cfg,
        run_dir=run_dir,
        fast_dev_run=args.fast_dev_run,
        ckpt_path=ckpt_path,
        mode=args.mode,
    )

    try:
        if args.mode == "train":
            _run_training(args, cfg, model, trainer, ckpt_path)

        if args.mode in ["tune", "tune-test"]:
            from .tune_runner import run_tuning

            run_tuning(model, trainer, cfg, checkpoint_path=ckpt_path)

        if args.mode in ["tune-test", "test"]:
            _run_test(
                args,
                cfg,
                model,
                trainer,
                run_dir,
                ckpt_path,
                has_saved_prediction=has_saved_prediction,
                saved_prediction_path=saved_prediction_path,
            )

    except Exception as exc:
        mode_name = args.mode.capitalize() if args.mode else "Operation"
        print(f"\n{mode_name} failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        if args.mode == "train":
            cleanup_run_directory(output_base)


__all__ = [
    "configure_matmul_precision",
    "dispatch_runtime",
    "prepare_cli_args",
    "suppress_nonzero_rank_stdout",
]
