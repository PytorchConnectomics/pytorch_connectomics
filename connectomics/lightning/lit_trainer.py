"""
PyTorch Lightning trainer for PyTorch Connectomics.

This module provides the Lightning trainer interface that replaces the custom
training infrastructure with modern Lightning features.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union
import os
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from ..config import CfgNode
from .lightning_module import ConnectomicsModule


class ConnectomicsTrainer:
    """
    Lightning trainer wrapper for connectomics tasks.

    This class provides a high-level interface that replaces the custom trainer.py
    with Lightning's automatic training features.
    """

    def __init__(self, cfg: CfgNode):
        self.cfg = cfg
        self.trainer = None
        self.module = None
        self.datamodule = None

    def setup(self) -> None:
        """Setup the trainer, module, and datamodule."""
        # Create Lightning module
        self.module = ConnectomicsModule(self.cfg)

        # Create Lightning datamodule
        from .lightning_datamodule import ConnectomicsDataModule
        self.datamodule = ConnectomicsDataModule(self.cfg)

        # Setup callbacks
        callbacks = self._setup_callbacks()

        # Setup logger
        logger = self._setup_logger()

        # Create Lightning trainer
        self.trainer = pl.Trainer(
            max_epochs=self._get_max_epochs(),
            max_steps=self.cfg.SOLVER.ITERATION_TOTAL,
            accelerator='auto',
            devices=self.cfg.SYSTEM.NUM_GPUS if self.cfg.SYSTEM.NUM_GPUS > 0 else 'auto',
            strategy='ddp' if self.cfg.SYSTEM.NUM_GPUS > 1 else 'auto',
            precision=16 if self.cfg.MODEL.get('MIXED_PRECISION', True) else 32,
            gradient_clip_val=self.cfg.SOLVER.get('GRAD_CLIP', 0.0),
            accumulate_grad_batches=self.cfg.SOLVER.get('GRAD_ACCUMULATE', 1),
            val_check_interval=self.cfg.SOLVER.get('VAL_CHECK_INTERVAL', 1.0),
            callbacks=callbacks,
            logger=logger,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            deterministic=False,  # Set to True for reproducibility at cost of performance
        )

    def train(self) -> None:
        """Run training."""
        if self.trainer is None:
            self.setup()

        # Run training
        self.trainer.fit(self.module, self.datamodule)

    def test(self, ckpt_path: Optional[str] = None) -> List[Dict[str, float]]:
        """Run testing."""
        if self.trainer is None:
            self.setup()

        return self.trainer.test(self.module, self.datamodule, ckpt_path=ckpt_path)

    def predict(self, ckpt_path: Optional[str] = None) -> List[Any]:
        """Run prediction."""
        if self.trainer is None:
            self.setup()

        return self.trainer.predict(self.module, self.datamodule, ckpt_path=ckpt_path)

    def _get_max_epochs(self) -> int:
        """Calculate max epochs from iteration settings."""
        # Estimate epochs from total iterations and samples per batch
        total_iterations = self.cfg.SOLVER.ITERATION_TOTAL
        samples_per_batch = self.cfg.SOLVER.SAMPLES_PER_BATCH

        # Rough estimate - may need adjustment based on dataset size
        # This is a fallback; Lightning will use max_steps if provided
        estimated_epochs = max(100, total_iterations // 1000)
        return estimated_epochs

    def _setup_callbacks(self) -> List[pl.Callback]:
        """Setup Lightning callbacks."""
        callbacks = []

        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.cfg.DATASET.OUTPUT_PATH,
            filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            save_on_train_epoch_end=False,
            every_n_train_steps=self.cfg.SOLVER.ITERATION_SAVE,
        )
        callbacks.append(checkpoint_callback)

        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

        # Early stopping (optional)
        if self.cfg.SOLVER.get('EARLY_STOPPING', False):
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=self.cfg.SOLVER.get('EARLY_STOPPING_PATIENCE', 10),
                mode='min',
                verbose=True
            )
            callbacks.append(early_stop_callback)

        return callbacks

    def _setup_logger(self) -> Union[TensorBoardLogger, WandbLogger]:
        """Setup experiment logger."""
        logger_type = self.cfg.get('LOGGER_TYPE', 'tensorboard')

        if logger_type == 'wandb':
            try:
                import wandb
                logger = WandbLogger(
                    name=os.path.basename(self.cfg.DATASET.OUTPUT_PATH),
                    project='pytorch-connectomics',
                    save_dir=self.cfg.DATASET.OUTPUT_PATH,
                )
            except ImportError:
                warnings.warn("Wandb not available, falling back to TensorBoard")
                logger = self._setup_tensorboard_logger()
        else:
            logger = self._setup_tensorboard_logger()

        return logger

    def _setup_tensorboard_logger(self) -> TensorBoardLogger:
        """Setup TensorBoard logger."""
        return TensorBoardLogger(
            save_dir=self.cfg.DATASET.OUTPUT_PATH,
            name='lightning_logs',
            version=None,  # Auto-increment
        )


def create_trainer(cfg: CfgNode) -> ConnectomicsTrainer:
    """
    Factory function to create a Lightning trainer from config.

    Args:
        cfg: Configuration object

    Returns:
        Configured trainer
    """
    return ConnectomicsTrainer(cfg)