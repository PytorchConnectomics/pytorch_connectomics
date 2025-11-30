Lightning Module API
====================

.. currentmodule:: connectomics.training.lit

PyTorch Lightning integration for training orchestration and distributed computing.

Overview
--------

The Lightning module provides three main components:

1. **ConnectomicsModule**: Lightning wrapper for models
2. **ConnectomicsDataModule**: Lightning data handling
3. **create_trainer**: Convenience function for trainer creation

Quick Example
-------------

.. code-block:: python

    from connectomics.config import load_config
    from connectomics.training.lit import (
        ConnectomicsModule,
        ConnectomicsDataModule,
        create_trainer
    )
    from pytorch_lightning import seed_everything

    # Load config
    cfg = load_config("tutorials/lucchi.yaml")

    # Set seed
    seed_everything(cfg.system.seed)

    # Create components
    datamodule = ConnectomicsDataModule(cfg)
    model = ConnectomicsModule(cfg)
    trainer = create_trainer(cfg)

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Test
    trainer.test(model, datamodule=datamodule)

Module Reference
----------------

ConnectomicsModule
^^^^^^^^^^^^^^^^^^

.. autoclass:: connectomics.training.lit.ConnectomicsModule
   :members:
   :undoc-members:
   :show-inheritance:

   Lightning module wrapper for connectomics models.

   This class wraps segmentation models with automatic training features:

   - Distributed training (DDP)
   - Mixed precision (AMP)
   - Gradient accumulation
   - Learning rate scheduling
   - Checkpointing
   - Multi-loss support
   - Deep supervision

   **Example:**

   .. code-block:: python

       from connectomics.config import load_config
       from connectomics.training.lit import ConnectomicsModule

       cfg = load_config("tutorials/lucchi.yaml")
       model = ConnectomicsModule(cfg)

       # Access underlying model
       print(model.model)

       # Get model info
       print(model.get_model_info())

   **With custom model:**

   .. code-block:: python

       import torch.nn as nn
       from connectomics.training.lit import ConnectomicsModule

       class MyModel(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv = nn.Conv3d(1, 2, 3, padding=1)

           def forward(self, x):
               return self.conv(x)

       custom_model = MyModel()
       lit_model = ConnectomicsModule(cfg, model=custom_model)

ConnectomicsDataModule
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: connectomics.training.lit.ConnectomicsDataModule
   :members:
   :undoc-members:
   :show-inheritance:

   Lightning data module for connectomics datasets.

   Handles data loading with MONAI transforms:

   - Train/val/test splits
   - MONAI CacheDataset for fast loading
   - Automatic augmentation pipeline
   - Persistent workers for efficiency

   **Example:**

   .. code-block:: python

       from connectomics.config import load_config
       from connectomics.training.lit import ConnectomicsDataModule

       cfg = load_config("tutorials/lucchi.yaml")
       datamodule = ConnectomicsDataModule(cfg)

       # Setup for training
       datamodule.setup('fit')

       # Access dataloaders
       train_loader = datamodule.train_dataloader()
       val_loader = datamodule.val_dataloader()

       # Get dataset info
       print(f"Train samples: {len(datamodule.train_dataset)}")
       print(f"Val samples: {len(datamodule.val_dataset)}")

create_trainer
^^^^^^^^^^^^^^

.. autofunction:: connectomics.training.lit.create_trainer

   Create PyTorch Lightning Trainer with appropriate callbacks.

   **Example:**

   .. code-block:: python

       from connectomics.config import load_config
       from connectomics.training.lit import create_trainer

       cfg = load_config("tutorials/lucchi.yaml")
       trainer = create_trainer(cfg)

       # Access trainer properties
       print(f"Max epochs: {trainer.max_epochs}")
       print(f"Precision: {trainer.precision}")
       print(f"Devices: {trainer.num_devices}")

   **Custom trainer:**

   .. code-block:: python

       from pytorch_lightning import Trainer
       from pytorch_lightning.callbacks import EarlyStopping

       # Create custom trainer
       trainer = Trainer(
           max_epochs=100,
           accelerator='gpu',
           devices=2,
           callbacks=[EarlyStopping(monitor='val/loss', patience=10)]
       )

Training Features
-----------------

Distributed Training
^^^^^^^^^^^^^^^^^^^^

Automatically uses DistributedDataParallel (DDP) with multiple GPUs:

.. code-block:: yaml

    system:
      num_gpus: 4  # Uses DDP automatically

.. code-block:: python

    trainer = create_trainer(cfg)  # DDP enabled automatically

Mixed Precision
^^^^^^^^^^^^^^^

Enable mixed precision for faster training:

.. code-block:: yaml

    training:
      precision: "16-mixed"  # FP16
      # or
      precision: "bf16-mixed"  # BFloat16 (Ampere+ GPUs)

Gradient Accumulation
^^^^^^^^^^^^^^^^^^^^^

Simulate larger batch sizes:

.. code-block:: yaml

    training:
      accumulate_grad_batches: 4

Gradient Clipping
^^^^^^^^^^^^^^^^^

Prevent exploding gradients:

.. code-block:: yaml

    training:
      gradient_clip_val: 1.0
      gradient_clip_algorithm: "norm"  # or "value"

Learning Rate Scheduling
^^^^^^^^^^^^^^^^^^^^^^^^

Automatic LR scheduling with warmup:

.. code-block:: yaml

    scheduler:
      name: CosineAnnealingLR
      warmup_epochs: 5
      min_lr: 1e-6

Deep Supervision
^^^^^^^^^^^^^^^^

Multi-scale loss computation:

.. code-block:: yaml

    model:
      deep_supervision: true
      loss_functions:
        - DiceLoss
      loss_weights: [1.0]

The module automatically:

- Computes losses at multiple scales
- Resizes ground truth to match each scale
- Averages losses across scales

Callbacks
---------

The trainer includes several useful callbacks:

Model Checkpointing
^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    checkpoint:
      monitor: "val/loss"
      mode: "min"
      save_top_k: 3
      save_last: true
      filename: "epoch{epoch:02d}-loss{val/loss:.2f}"

Early Stopping
^^^^^^^^^^^^^^

.. code-block:: yaml

    early_stopping:
      monitor: "val/loss"
      patience: 10
      mode: "min"
      min_delta: 0.0

Learning Rate Monitoring
^^^^^^^^^^^^^^^^^^^^^^^^

Automatically logs learning rate to TensorBoard/Wandb.

Logging
-------

TensorBoard (Default)
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    logging:
      save_dir: "outputs"
      log_every_n_steps: 10

Logs are saved to ``outputs/lightning_logs/``.

View with:

.. code-block:: bash

    tensorboard --logdir outputs/lightning_logs

Weights & Biases (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    logging:
      use_wandb: true
      wandb_project: "connectomics"
      wandb_entity: "your_team"
      wandb_name: "lucchi_exp"

Advanced Usage
--------------

Custom Callbacks
^^^^^^^^^^^^^^^^

.. code-block:: python

    from pytorch_lightning.callbacks import Callback

    class MyCallback(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            print(f"Epoch {trainer.current_epoch} finished!")

    # Add to trainer
    from pytorch_lightning import Trainer

    trainer = Trainer(
        max_epochs=100,
        callbacks=[MyCallback()]
    )

Custom Training Step
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from connectomics.training.lit import ConnectomicsModule

    class CustomModule(ConnectomicsModule):
        def training_step(self, batch, batch_idx):
            # Custom training logic
            images, labels = batch
            outputs = self.model(images)

            # Custom loss computation
            loss = self.compute_loss(outputs, labels)

            # Log metrics
            self.log('train/loss', loss)

            return loss

Inference
---------

Single Batch Prediction
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Load trained model
    model = ConnectomicsModule.load_from_checkpoint(
        "outputs/epoch=99.ckpt",
        cfg=cfg
    )

    model.eval()
    model.cuda()

    # Predict
    with torch.no_grad():
        output = model(input_batch)

Full Dataset Inference
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Load model
    model = ConnectomicsModule.load_from_checkpoint(
        "outputs/epoch=99.ckpt",
        cfg=cfg
    )

    # Create datamodule
    datamodule = ConnectomicsDataModule(cfg)

    # Create trainer
    trainer = create_trainer(cfg)

    # Run inference
    predictions = trainer.predict(model, datamodule=datamodule)

Resuming Training
-----------------

.. code-block:: python

    # Resume from checkpoint
    trainer = create_trainer(cfg)
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path="outputs/last.ckpt"
    )

Or from command line:

.. code-block:: bash

    python scripts/main.py \
        --config tutorials/lucchi.yaml \
        --resume outputs/last.ckpt

See Also
--------

- :ref:`Configuration Guide <Configuration System>`
- :ref:`Installation Guide <Installation>`
- `PyTorch Lightning Documentation <https://lightning.ai/docs/pytorch/stable/>`_
- `MONAI Documentation <https://docs.monai.io/>`_
