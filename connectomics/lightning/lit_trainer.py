"""
PyTorch Lightning trainer utilities for PyTorch Connectomics.

This module provides Lightning trainer factory functions with:
- Hydra/OmegaConf configuration
- Modern callbacks (checkpointing, early stopping, logging)
- Distributed training support
- Mixed precision training
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig

from ..config import Config
from .callbacks import NaNDetectionCallback


def create_trainer(
    cfg: Union[Config, DictConfig],
    callbacks: Optional[List] = None,
    logger: Optional[Union[pl.loggers.Logger, bool]] = None,
    **trainer_kwargs
) -> pl.Trainer:
    """
    Create a PyTorch Lightning Trainer from configuration.
    
    Args:
        cfg: Hydra Config object or OmegaConf DictConfig
        callbacks: Optional list of callbacks (if None, creates default callbacks)
        logger: Optional logger (if None, creates TensorBoard logger)
        **trainer_kwargs: Additional arguments to pass to pl.Trainer
        
    Returns:
        pl.Trainer instance
        
    Examples:
        >>> from connectomics.config import load_config
        >>> from connectomics.lightning import create_trainer
        >>> cfg = load_config('config.yaml')
        >>> trainer = create_trainer(cfg, max_epochs=100)
    """
    # Create default callbacks if not provided
    if callbacks is None:
        callbacks = _create_default_callbacks(cfg)
    
    # Create logger if not provided
    if logger is None:
        logger = _create_default_logger(cfg)
    
    # Get training configuration
    training_cfg = cfg.optimization if hasattr(cfg, 'optimization') else None
    
    # Build trainer arguments
    trainer_args = {
        'max_epochs': training_cfg.max_epochs if training_cfg else 100,
        'callbacks': callbacks,
        'logger': logger,
        'accelerator': 'gpu' if (hasattr(cfg.system, 'num_gpus') and cfg.system.num_gpus > 0) else 'cpu',
        'devices': cfg.system.num_gpus if hasattr(cfg.system, 'num_gpus') else 1,
        'precision': training_cfg.precision if training_cfg and hasattr(training_cfg, 'precision') else 32,
        'gradient_clip_val': training_cfg.gradient_clip_val if training_cfg and hasattr(training_cfg, 'gradient_clip_val') else 0.0,
        'accumulate_grad_batches': training_cfg.accumulate_grad_batches if training_cfg and hasattr(training_cfg, 'accumulate_grad_batches') else 1,
        'log_every_n_steps': training_cfg.log_every_n_steps if training_cfg and hasattr(training_cfg, 'log_every_n_steps') else 50,
        'enable_progress_bar': True,
        'enable_model_summary': True,
    }
    
    # Override with any additional kwargs
    trainer_args.update(trainer_kwargs)
    
    return pl.Trainer(**trainer_args)


def _create_default_callbacks(cfg: Union[Config, DictConfig]) -> List:
    """Create default callbacks from configuration."""
    callbacks = []

    # NaN detection callback (enabled by default for debugging)
    nan_detection_cfg = cfg.nan_detection if hasattr(cfg, 'nan_detection') else None
    if nan_detection_cfg is not None:
        # User explicitly configured NaN detection
        if hasattr(nan_detection_cfg, 'enabled') and nan_detection_cfg.enabled:
            nan_callback = NaNDetectionCallback(
                check_grads=getattr(nan_detection_cfg, 'check_grads', True),
                check_inputs=getattr(nan_detection_cfg, 'check_inputs', True),
                debug_on_nan=getattr(nan_detection_cfg, 'debug_on_nan', True),
                terminate_on_nan=getattr(nan_detection_cfg, 'terminate_on_nan', False),
                print_diagnostics=getattr(nan_detection_cfg, 'print_diagnostics', True),
            )
            callbacks.append(nan_callback)
    else:
        # Default: enable NaN detection with debugging
        nan_callback = NaNDetectionCallback(
            check_grads=True,
            check_inputs=True,
            debug_on_nan=True,
            terminate_on_nan=False,
            print_diagnostics=True,
        )
        callbacks.append(nan_callback)

    # Model checkpoint callback
    checkpoint_cfg = cfg.checkpoint if hasattr(cfg, 'checkpoint') else None
    if checkpoint_cfg and hasattr(checkpoint_cfg, 'save_top_k'):
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_cfg.dirpath if hasattr(checkpoint_cfg, 'dirpath') else 'checkpoints/',
            filename=checkpoint_cfg.filename if hasattr(checkpoint_cfg, 'filename') else 'model-{epoch:02d}-{val_loss_total:.4f}',
            monitor=checkpoint_cfg.monitor if hasattr(checkpoint_cfg, 'monitor') else 'val_loss_total',
            mode=checkpoint_cfg.mode if hasattr(checkpoint_cfg, 'mode') else 'min',
            save_top_k=checkpoint_cfg.save_top_k,
            save_last=checkpoint_cfg.save_last if hasattr(checkpoint_cfg, 'save_last') else True,
            every_n_epochs=checkpoint_cfg.save_every_n_epochs if hasattr(checkpoint_cfg, 'save_every_n_epochs') else 1,
        )
        callbacks.append(checkpoint_callback)
    else:
        # Default checkpoint callback
        callbacks.append(ModelCheckpoint(
            dirpath='checkpoints/',
            filename='model-{epoch:02d}-{val_loss_total:.4f}',
            monitor='val_loss_total',
            mode='min',
            save_top_k=3,
            save_last=True,
        ))
    
    # Early stopping callback
    early_stopping_cfg = cfg.early_stopping if hasattr(cfg, 'early_stopping') else None
    if early_stopping_cfg and hasattr(early_stopping_cfg, 'patience'):
        early_stopping_callback = EarlyStopping(
            monitor=early_stopping_cfg.monitor if hasattr(early_stopping_cfg, 'monitor') else 'val_loss_total',
            patience=early_stopping_cfg.patience,
            mode=early_stopping_cfg.mode if hasattr(early_stopping_cfg, 'mode') else 'min',
            min_delta=early_stopping_cfg.min_delta if hasattr(early_stopping_cfg, 'min_delta') else 0.0,
        )
        callbacks.append(early_stopping_callback)
    
    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    
    # Rich progress bar (if available)
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass  # Fall back to default progress bar
    
    return callbacks


def _create_default_logger(cfg: Union[Config, DictConfig]) -> pl.loggers.Logger:
    """Create default logger from configuration."""
    # Check if wandb is configured
    if hasattr(cfg, 'wandb') and hasattr(cfg.wandb, 'project'):
        try:
            return WandbLogger(
                project=cfg.wandb.project,
                name=cfg.wandb.name if hasattr(cfg.wandb, 'name') else None,
                save_dir=cfg.wandb.save_dir if hasattr(cfg.wandb, 'save_dir') else 'logs/',
            )
        except ImportError:
            pass  # Fall back to TensorBoard
    
    # Default to TensorBoard
    return TensorBoardLogger(
        save_dir='logs/',
        name='connectomics',
    )


class ConnectomicsTrainer:
    """
    High-level trainer wrapper for connectomics tasks.
    
    This class provides a convenient interface for training with sensible defaults.
    For more control, use create_trainer() directly.
    
    Args:
        cfg: Hydra Config object or OmegaConf DictConfig
        **trainer_kwargs: Additional arguments to pass to pl.Trainer
        
    Examples:
        >>> from connectomics.config import load_config
        >>> from connectomics.lightning import ConnectomicsTrainer
        >>> cfg = load_config('config.yaml')
        >>> trainer = ConnectomicsTrainer(cfg)
        >>> trainer.fit(model, datamodule)
    """
    
    def __init__(self, cfg: Union[Config, DictConfig], **trainer_kwargs):
        self.cfg = cfg
        self.trainer = create_trainer(cfg, **trainer_kwargs)
    
    def fit(self, model: pl.LightningModule, datamodule: pl.LightningDataModule):
        """Fit the model."""
        return self.trainer.fit(model, datamodule)
    
    def test(self, model: pl.LightningModule, datamodule: pl.LightningDataModule):
        """Test the model."""
        return self.trainer.test(model, datamodule)
    
    def predict(self, model: pl.LightningModule, datamodule: pl.LightningDataModule):
        """Generate predictions."""
        return self.trainer.predict(model, datamodule)


__all__ = [
    'create_trainer',
    'ConnectomicsTrainer',
]