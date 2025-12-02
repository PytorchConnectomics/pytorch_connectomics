"""
PyTorch Lightning trainer utilities for PyTorch Connectomics.

This module provides Lightning trainer factory functions with:
- Hydra/OmegaConf configuration
- Modern callbacks (checkpointing, early stopping, logging)
- Distributed training support
- Mixed precision training
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from omegaconf import DictConfig

from ...config import Config
from .callbacks import VisualizationCallback

# Register safe globals for PyTorch 2.6+ checkpoint loading
# This allows our Config class to be unpickled from Lightning checkpoints
try:
    torch.serialization.add_safe_globals([Config])
except AttributeError:
    # PyTorch < 2.6 doesn't have add_safe_globals
    pass


def create_trainer(
    cfg: Config,
    run_dir: Optional[Path] = None,
    fast_dev_run: bool = False,
    ckpt_path: Optional[str] = None,
    mode: str = "train",
) -> pl.Trainer:
    """
    Create PyTorch Lightning Trainer.

    Args:
        cfg: Hydra Config object
        run_dir: Directory for this training run (required for mode='train')
        fast_dev_run: Whether to run quick debug mode
        ckpt_path: Path to checkpoint for resuming (used to extract best_score)
        mode: 'train' or 'test' - determines which system config to use

    Returns:
        Configured Trainer instance
    """
    print(f"Creating Lightning trainer (mode={mode})...")

    # Setup callbacks (only for training mode)
    callbacks = []

    if mode == "train":
        if run_dir is None:
            raise ValueError("run_dir is required when mode='train'")

        # Setup checkpoint directory
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Model checkpoint (in run_dir/checkpoints/)
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=cfg.monitor.checkpoint.checkpoint_filename,
            monitor=cfg.monitor.checkpoint.monitor,
            mode=cfg.monitor.checkpoint.mode,
            save_top_k=cfg.monitor.checkpoint.save_top_k,
            save_last=cfg.monitor.checkpoint.save_last,
            every_n_epochs=cfg.monitor.checkpoint.save_every_n_epochs,
            verbose=True,
            save_on_train_epoch_end=True,  # Save based on training metrics
        )
        callbacks.append(checkpoint_callback)

        # Early stopping (training only)
        if cfg.monitor.early_stopping.enabled:
            # Import here to avoid circular dependency
            from .utils import extract_best_score_from_checkpoint

            # Extract best_score from checkpoint filename if resuming
            best_score = None
            if ckpt_path:
                best_score = extract_best_score_from_checkpoint(
                    ckpt_path, cfg.monitor.early_stopping.monitor
                )
                if best_score is not None:
                    print(
                        f"  Early stopping: Extracted best_score={best_score:.6f} from checkpoint"
                    )

            early_stop_callback = EarlyStopping(
                monitor=cfg.monitor.early_stopping.monitor,
                patience=cfg.monitor.early_stopping.patience,
                mode=cfg.monitor.early_stopping.mode,
                min_delta=cfg.monitor.early_stopping.min_delta,
                verbose=True,
                check_on_train_epoch_end=True,  # Check at end of train epoch (not validation)
                check_finite=cfg.monitor.early_stopping.check_finite,  # Stop on NaN/inf
                stopping_threshold=cfg.monitor.early_stopping.threshold,
                divergence_threshold=cfg.monitor.early_stopping.divergence_threshold,
                strict=False,  # Don't crash if metric not available (wait for it)
            )

            # Manually set best_score if extracted from checkpoint
            if best_score is not None:
                early_stop_callback.best_score = torch.tensor(best_score)

            callbacks.append(early_stop_callback)

        # Learning rate monitor (training only)
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        # Visualization callback (training only, end-of-epoch only)
        if cfg.monitor.logging.images.enabled:
            vis_callback = VisualizationCallback(
                cfg=cfg,
                max_images=cfg.monitor.logging.images.max_images,
                num_slices=cfg.monitor.logging.images.num_slices,
                log_every_n_epochs=cfg.monitor.logging.images.log_every_n_epochs,
            )
            callbacks.append(vis_callback)
            print(
                f"  Visualization: Enabled (every {cfg.monitor.logging.images.log_every_n_epochs} epoch(s))"
            )
        else:
            print(f"  Visualization: Disabled")

    # Progress bar (optional - requires rich package)
    try:
        callbacks.append(RichProgressBar())
    except (ImportError, ModuleNotFoundError):
        pass  # Use default progress bar

    # Setup logger (training only - in run_dir/logs/)
    # Always create a logger for training to avoid warnings about missing logger
    logger = None
    if mode == "train":
        if run_dir is None:
            raise ValueError("run_dir is required when mode='train'")

        logger = TensorBoardLogger(
            save_dir=str(run_dir),
            name="",  # No name subdirectory
            version="logs",  # Logs go directly to run_dir/logs/
        )
        print(f"  Logger: TensorBoard (logs saved to {run_dir}/logs/)")
    else:
        # For test/predict mode, create a minimal logger to avoid warnings
        # if validation metrics are logged
        if run_dir is not None:
            logger = TensorBoardLogger(
                save_dir=str(run_dir),
                name="",
                version="logs",
            )

    # Create trainer
    # Select system config based on mode
    system_cfg = cfg.system.training if mode == "train" else cfg.system.inference

    # Check if GPU is actually available
    use_gpu = system_cfg.num_gpus > 0 and torch.cuda.is_available()

    # Check if anomaly detection is enabled (useful for debugging NaN)
    detect_anomaly = getattr(cfg.monitor, "detect_anomaly", False)
    if detect_anomaly:
        print("  ⚠️  PyTorch anomaly detection ENABLED (training will be slower)")
        print("      This helps pinpoint the exact operation causing NaN in backward pass")

    # Configure DDP strategy for multi-GPU training with deep supervision
    strategy = "auto"  # Default strategy
    if system_cfg.num_gpus > 1:
        # Multi-GPU training: configure DDP
        deep_supervision_enabled = getattr(cfg.model, "deep_supervision", False)
        ddp_find_unused_params = getattr(cfg.model, "ddp_find_unused_parameters", False)
        architecture = getattr(cfg.model, "architecture", "")
        is_mednext = architecture.startswith("mednext")

        # MedNeXt always creates deep supervision layers internally (even when disabled)
        # so it always needs find_unused_parameters=True
        if is_mednext or deep_supervision_enabled or ddp_find_unused_params:
            strategy = DDPStrategy(find_unused_parameters=True)

            # Determine reason for using find_unused_parameters
            if is_mednext and not deep_supervision_enabled:
                reason = "MedNeXt (has unused DS layers)"
            elif deep_supervision_enabled:
                reason = "deep supervision enabled"
            else:
                reason = "explicit config"
            print(f"  Strategy: DDP with find_unused_parameters=True ({reason})")
        else:
            strategy = DDPStrategy(find_unused_parameters=False)
            print("  Strategy: DDP (standard)")

    trainer = pl.Trainer(
        max_epochs=cfg.optimization.max_epochs,
        max_steps=getattr(cfg.optimization, "max_steps", None) or -1,
        accelerator="gpu" if use_gpu else "cpu",
        devices=system_cfg.num_gpus if use_gpu else 1,
        strategy=strategy,
        precision=cfg.optimization.precision,
        gradient_clip_val=cfg.optimization.gradient_clip_val,
        accumulate_grad_batches=cfg.optimization.accumulate_grad_batches,
        val_check_interval=cfg.optimization.val_check_interval,
        log_every_n_steps=cfg.optimization.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        deterministic=cfg.optimization.deterministic,
        benchmark=cfg.optimization.benchmark,
        fast_dev_run=bool(fast_dev_run),
        detect_anomaly=detect_anomaly,
    )

    print(f"  Max epochs: {cfg.optimization.max_epochs}")
    print(f"  Devices: {system_cfg.num_gpus if system_cfg.num_gpus > 0 else 1} ({mode} mode)")
    print(f"  Precision: {cfg.optimization.precision}")

    return trainer


__all__ = [
    'create_trainer',
]
