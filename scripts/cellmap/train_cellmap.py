#!/usr/bin/env python
"""
Standalone training script for CellMap Segmentation Challenge using PyTC models.

This script uses:
- cellmap-data for data loading (official challenge library)
- PyTC models (MONAI model zoo)
- PyTorch Lightning for training orchestration

NO modifications to PyTC core required.

Usage:
    python scripts/cellmap/train_cellmap.py configs/mednext_cos7.py
    python scripts/cellmap/train_cellmap.py configs/mednext_mito.py

Requirements:
    pip install cellmap-data cellmap-segmentation-challenge
"""

import os
import sys
from pathlib import Path

# Add PyTC to path
PYTC_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PYTC_ROOT))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# CellMap data loading (official)
from cellmap_segmentation_challenge.utils import (
    get_dataloader,                 # Official dataloader factory
    make_datasplit_csv,             # Auto-generate train/val split
    get_tested_classes,             # Official class list
    CellMapLossWrapper,             # NaN-aware loss
)

# PyTC model building (import only, no modification)
from connectomics.models import build_model
from connectomics.models.loss import create_loss

# Import config utilities
from cellmap_segmentation_challenge.utils import load_safe_config


class CellMapLightningModule(pl.LightningModule):
    """
    Minimal Lightning wrapper around PyTC models for CellMap training.

    Uses PyTC models as-is, no modifications needed.
    """

    def __init__(self, model, criterion, optimizer_config, scheduler_config=None, classes=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.classes = classes or []

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model', 'criterion'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch['input']
        labels = batch['output']

        predictions = self(images)
        loss = self.criterion(predictions, labels)

        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['input']
        labels = batch['output']

        predictions = self(images)
        loss = self.criterion(predictions, labels)

        # Compute Dice score per class
        with torch.no_grad():
            pred_binary = (torch.sigmoid(predictions) > 0.5).float()

            # Average Dice across classes
            dice_scores = []
            for c in range(predictions.shape[1]):
                pred_c = pred_binary[:, c]
                label_c = labels[:, c]
                intersection = (pred_c * label_c).sum()
                dice = (2. * intersection) / (pred_c.sum() + label_c.sum() + 1e-7)
                dice_scores.append(dice)

                # Log per-class Dice if we have class names
                if c < len(self.classes):
                    self.log(f'val/dice_{self.classes[c]}', dice, sync_dist=True)

            mean_dice = torch.stack(dice_scores).mean()

        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/dice', mean_dice, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config['lr'],
            weight_decay=self.optimizer_config.get('weight_decay', 1e-5),
        )

        if self.scheduler_config is None or self.scheduler_config.get('name') == 'constant':
            return optimizer

        scheduler_name = self.scheduler_config.get('name', 'cosine')

        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get('T_max', 100),
                eta_min=self.scheduler_config.get('min_lr', 1e-6),
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


def train_cellmap(config_path: str):
    """
    Main training function using CellMap's official tools + PyTC models.

    Args:
        config_path: Path to Python config file (CellMap style)
    """
    # Load config (CellMap's safe config loader)
    print(f"Loading config from: {config_path}")
    config = load_safe_config(config_path)

    # Extract config values
    model_name = getattr(config, 'model_name', 'mednext')
    classes = getattr(config, 'classes', get_tested_classes())
    learning_rate = getattr(config, 'learning_rate', 1e-3)
    batch_size = getattr(config, 'batch_size', 2)
    max_epochs = getattr(config, 'epochs', 1000)
    num_gpus = getattr(config, 'num_gpus', 1)
    precision = getattr(config, 'precision', '16-mixed')

    # Output paths
    output_dir = getattr(config, 'output_dir', 'outputs/cellmap')
    os.makedirs(output_dir, exist_ok=True)

    datasplit_path = getattr(config, 'datasplit_path', f'{output_dir}/datasplit.csv')
    input_array_info = getattr(config, 'input_array_info', {
        'shape': (128, 128, 128),
        'scale': (8, 8, 8),
    })
    target_array_info = getattr(config, 'target_array_info', input_array_info)
    spatial_transforms = getattr(config, 'spatial_transforms', {
        'mirror': {'axes': {'x': 0.5, 'y': 0.5, 'z': 0.5}},
        'transpose': {'axes': ['x', 'y', 'z']},
        'rotate': {'axes': {'x': [-180, 180], 'y': [-180, 180], 'z': [-180, 180]}},
    })

    print(f"Training configuration:")
    print(f"  Model: {model_name}")
    print(f"  Classes: {classes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Precision: {precision}")

    # Generate datasplit CSV if doesn't exist (CellMap's official utility)
    if not os.path.exists(datasplit_path):
        print(f"Generating datasplit CSV: {datasplit_path}")
        make_datasplit_csv(
            classes=classes,
            csv_path=datasplit_path,
            validation_prob=0.15,
            scale=input_array_info.get('scale'),
            force_all_classes='validate',
        )
    else:
        print(f"Using existing datasplit: {datasplit_path}")

    # Get dataloaders (CellMap's official dataloader)
    print("Creating dataloaders...")
    train_loader, val_loader = get_dataloader(
        datasplit_path=datasplit_path,
        classes=classes,
        batch_size=batch_size,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        spatial_transforms=spatial_transforms,
        iterations_per_epoch=1000,
        weighted_sampler=True,
    )

    # Build model using PyTC's model factory (MONAI models)
    print(f"Building model: {model_name}")

    # Create minimal config for PyTC's build_model
    from omegaconf import OmegaConf
    model_config = OmegaConf.create({
        'model': {
            'architecture': model_name,
            'in_channels': 1,
            'out_channels': len(classes),
            'mednext_size': getattr(config, 'mednext_size', 'B'),
            'mednext_kernel_size': getattr(config, 'mednext_kernel_size', 5),
            'deep_supervision': getattr(config, 'deep_supervision', True),
        }
    })

    model = build_model(model_config)
    print(f"Model built successfully")

    # Create loss (CellMap's NaN-aware wrapper + PyTC loss)
    print("Creating loss function...")
    base_loss = torch.nn.BCEWithLogitsLoss
    criterion = CellMapLossWrapper(base_loss, reduction='mean')

    # Create Lightning module
    lit_model = CellMapLightningModule(
        model=model,
        criterion=criterion,
        optimizer_config={'lr': learning_rate, 'weight_decay': 1e-5},
        scheduler_config=getattr(config, 'scheduler_config', {'name': 'constant'}),
        classes=classes,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{output_dir}/checkpoints',
        filename=f'{model_name}-{{epoch:02d}}-{{val/dice:.3f}}',
        monitor='val/dice',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor='val/dice',
        patience=50,
        mode='max',
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Setup loggers
    tb_logger = TensorBoardLogger(
        f'{output_dir}/tensorboard',
        name=model_name,
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=num_gpus,
        precision=precision,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=tb_logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train!
    print("Starting training...")
    print(f"Monitor progress: tensorboard --logdir {output_dir}/tensorboard")
    trainer.fit(lit_model, train_loader, val_loader)

    print(f"\nTraining complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val/dice: {checkpoint_callback.best_model_score:.4f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train PyTC models on CellMap data')
    parser.add_argument('config', type=str, help='Path to config file')
    args = parser.parse_args()

    train_cellmap(args.config)
