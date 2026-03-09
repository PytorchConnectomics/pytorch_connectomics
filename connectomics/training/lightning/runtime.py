"""Runtime and checkpoint orchestration helpers for Lightning training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _resolve_resume_run_directory(resume_checkpoint_path: str, checkpoint_subdir: str) -> Path:
    """
    Infer the run directory for a resumed training job from the checkpoint path.

    If the checkpoint already lives under a folder named like the configured
    checkpoint subdirectory (usually ``checkpoints``), reuse that run
    directory. Otherwise, treat the checkpoint's parent as the run directory
    and continue saving checkpoints into ``<parent>/<checkpoint_subdir>``.
    """
    checkpoint_path = Path(resume_checkpoint_path).expanduser()
    if checkpoint_path.parent.name == checkpoint_subdir:
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def setup_run_directory(
    mode: str,
    cfg,
    checkpoint_dirpath: str,
    resume_checkpoint_path: Optional[str] = None,
):
    """
    Setup run directory with timestamp for training mode.
    Handles DDP subprocess coordination via timestamp files.

    Args:
        mode: 'train', 'test', 'tune', or 'tune-test'
        cfg: Config object (will be modified in-place for training mode)
        checkpoint_dirpath: Path to checkpoint/output directory from config
        resume_checkpoint_path: Checkpoint path for in-place training resume

    Returns:
        Path: Run directory (timestamped for train, created for tune modes, dummy for test)
    """
    import os
    from datetime import datetime

    from ...config import save_config

    checkpoint_dir = Path(checkpoint_dirpath)
    checkpoint_subdir = checkpoint_dir.name or "checkpoints"
    output_base = checkpoint_dir.parent

    if mode == "train":
        resume_run_dir = None
        if resume_checkpoint_path:
            resume_run_dir = _resolve_resume_run_directory(
                resume_checkpoint_path, checkpoint_subdir
            )
            output_base = resume_run_dir.parent

        # Check if this is a DDP re-launch (LOCAL_RANK is set by PyTorch Lightning)
        is_ddp_subprocess = "LOCAL_RANK" in os.environ
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        timestamp_file = output_base / ".latest_timestamp"

        if not is_ddp_subprocess:
            # First invocation (main process) - create new timestamp unless resuming in place.
            if resume_run_dir is not None:
                run_dir = resume_run_dir
                logger.info(f"Resuming into existing run directory: {run_dir}")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = output_base / timestamp
                logger.info(f"Run directory: {run_dir}")

            checkpoint_path = run_dir / checkpoint_subdir
            cfg.monitor.checkpoint.dirpath = str(checkpoint_path)

            checkpoint_path.mkdir(parents=True, exist_ok=True)

            # Save config to run directory
            config_save_path = run_dir / "config.yaml"
            save_config(cfg, config_save_path)
            logger.info(f"Config saved to: {config_save_path}")

            # Save timestamp for DDP subprocesses to read
            output_base.mkdir(parents=True, exist_ok=True)
            timestamp_file.write_text(run_dir.name)
        else:
            # DDP subprocess - read existing timestamp
            import time

            max_wait = 30  # Maximum 30 seconds
            waited = 0.0
            while not timestamp_file.exists() and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            if timestamp_file.exists():
                timestamp = timestamp_file.read_text().strip()
                run_dir = output_base / timestamp
                checkpoint_path = run_dir / checkpoint_subdir
                cfg.monitor.checkpoint.dirpath = str(checkpoint_path)
                logger.info(f"[DDP Rank {local_rank}] Using run directory: {run_dir}")
            else:
                raise RuntimeError(
                    f"DDP subprocess (LOCAL_RANK={local_rank}) timed out waiting for timestamp file"
                )
    elif mode in ["tune", "tune-test"]:
        # For tune modes, create the directory for Optuna outputs
        run_dir = Path(checkpoint_dirpath)
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Tuning output directory: {run_dir}")
    elif mode == "test":
        # For test mode, create the results directory
        run_dir = Path(checkpoint_dirpath)
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Test output directory: {run_dir}")
    else:
        # Fallback for unknown modes
        run_dir = output_base / f"{mode}_run"
        logger.info(f"Running in {mode} mode")

    return run_dir


def cleanup_run_directory(output_base: Path):
    """
    Clean up timestamp file after training completes.
    Only runs in main process (not DDP subprocesses).

    Args:
        output_base: Base output directory containing timestamp file
    """
    import os

    is_ddp_subprocess = "LOCAL_RANK" in os.environ
    if not is_ddp_subprocess:
        timestamp_file = output_base / ".latest_timestamp"
        if timestamp_file.exists():
            try:
                timestamp_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors


def modify_checkpoint_state(
    checkpoint_path: Optional[str],
    run_dir: Path,
    reset_optimizer: bool = False,
    reset_scheduler: bool = False,
    reset_epoch: bool = False,
    reset_early_stopping: bool = False,
) -> Optional[str]:
    """
    Modify checkpoint state for training resumption with selective resets.

    Args:
        checkpoint_path: Path to checkpoint file to modify (None if no checkpoint)
        run_dir: Run directory for saving modified checkpoint
        reset_optimizer: If True, reset optimizer state
        reset_scheduler: If True, reset scheduler state
        reset_epoch: If True, reset epoch counter
        reset_early_stopping: If True, reset early stopping patience

    Returns:
        Path to modified checkpoint file, or original path if no modifications needed
    """
    import torch

    # Early return if no checkpoint or no resets requested
    if not checkpoint_path:
        return None

    if not (reset_optimizer or reset_scheduler or reset_epoch or reset_early_stopping):
        return checkpoint_path

    logger.info("Modifying checkpoint state:")
    if reset_optimizer:
        logger.info("   - Resetting optimizer state")
    if reset_scheduler:
        logger.info("   - Resetting scheduler state")
    if reset_epoch:
        logger.info("   - Resetting epoch counter")
    if reset_early_stopping:
        logger.info("   - Resetting early stopping patience counter")

    # Load checkpoint (weights_only=False needed for PyTorch 2.6+ to load Lightning
    # checkpoints with custom configs)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Reset optimizer state
    if reset_optimizer and "optimizer_states" in checkpoint:
        del checkpoint["optimizer_states"]

    # Reset scheduler state
    if reset_scheduler and "lr_schedulers" in checkpoint:
        del checkpoint["lr_schedulers"]

    # Reset epoch counter
    if reset_epoch:
        if "epoch" in checkpoint:
            checkpoint["epoch"] = 0
        if "global_step" in checkpoint:
            checkpoint["global_step"] = 0

    # Reset early stopping state
    if reset_early_stopping and "callbacks" in checkpoint:
        for callback_state in checkpoint["callbacks"].values():
            if "wait_count" in callback_state:
                callback_state["wait_count"] = 0
            if "best_score" in callback_state:
                callback_state["best_score"] = None

    # Save modified checkpoint to temporary file
    temp_ckpt_path = run_dir / "temp_modified_checkpoint.ckpt"
    torch.save(checkpoint, temp_ckpt_path)
    logger.info(f"   Modified checkpoint saved to: {temp_ckpt_path}")

    return str(temp_ckpt_path)


__all__ = [
    "setup_run_directory",
    "cleanup_run_directory",
    "modify_checkpoint_state",
]
