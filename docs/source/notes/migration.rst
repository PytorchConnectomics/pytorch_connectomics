Migration Guide (v1.0 → v2.0)
================================

This guide helps you migrate from PyTorch Connectomics v1.0 to v2.0.

.. note::
   **v2.0 is a major rewrite** with PyTorch Lightning and MONAI integration. The new system provides better performance, features, and ease of use.

Overview of Changes
-------------------

v2.0 introduces significant architectural improvements:

**What's New:**

- ⚡ **PyTorch Lightning** replaces custom trainer
- 🏥 **MONAI** provides models, transforms, and losses
- 🔧 **Hydra/OmegaConf** for modern configuration management
- 📦 **Architecture Registry** for extensible model management
- 🔬 **MedNeXt** state-of-the-art models
- 🧩 **Deep Supervision** support

**What Changed:**

- Custom trainer → PyTorch Lightning (``connectomics/lightning/``)
- Configuration → Hydra/OmegaConf YAML format
- Entry point: ``scripts/main.py``

Migration Checklist
-------------------

.. code-block:: none

    ☐ Update installation (Lightning, MONAI, OmegaConf)
    ☐ Create Hydra YAML configuration files
    ☐ Use scripts/main.py for training
    ☐ Update imports to use Lightning modules
    ☐ Use MONAI models from architecture registry
    ☐ Test training pipeline
    ☐ Update inference scripts
    ☐ Use MONAI transforms for data loading

Installation Updates
--------------------

**v1.0 Installation:**

.. code-block:: bash

    pip install -e .

**v2.0 Installation:**

.. code-block:: bash

    # Install PyTorch first
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # Install with new dependencies
    pip install -e .[full]

See the :ref:`installation guide <Installation>` for details.

Configuration System
--------------------

Hydra/OmegaConf Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**v2.0 uses Hydra/OmegaConf:**

.. code-block:: yaml

    # tutorials/lucchi.yaml
    system:
      num_gpus: 1
      num_cpus: 4
      seed: 42

    model:
      architecture: monai_basic_unet3d
      in_channels: 1
      out_channels: 2
      filters: [32, 64, 128, 256, 512]
      dropout: 0.1
      loss_functions:
        - DiceLoss
        - BCEWithLogitsLoss
      loss_weights: [1.0, 1.0]

    data:
      train_image: "datasets/lucchi/train_image.h5"
      train_label: "datasets/lucchi/train_label.h5"
      val_image: "datasets/lucchi/val_image.h5"
      val_label: "datasets/lucchi/val_label.h5"
      patch_size: [128, 128, 128]
      batch_size: 2
      num_workers: 4

    optimizer:
      name: AdamW
      lr: 1e-4
      weight_decay: 1e-4

    scheduler:
      name: CosineAnnealingLR
      warmup_epochs: 5

    training:
      max_epochs: 100
      precision: "16-mixed"
      gradient_clip_val: 1.0

    checkpoint:
      monitor: "val/loss"
      mode: "min"
      save_top_k: 3
      save_last: true

Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^

Key configuration sections in v2.0:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Section
     - Description
   * - ``system``
     - Hardware setup (GPUs, CPUs, random seed)
   * - ``model``
     - Architecture, input/output channels, loss functions
   * - ``data``
     - Dataset paths, batch size, augmentation settings
   * - ``optimizer``
     - Optimizer type and hyperparameters
   * - ``scheduler``
     - Learning rate scheduling configuration
   * - ``training``
     - Training loop parameters (epochs, precision, etc.)
   * - ``checkpoint``
     - Model checkpointing strategy
   * - ``logging``
     - Logging and monitoring configuration

Configuration Override
^^^^^^^^^^^^^^^^^^^^^^

Override config parameters from CLI:

.. code-block:: bash

    python scripts/main.py --config tutorials/lucchi.yaml \
        data.batch_size=8 \
        training.max_epochs=200 \
        optimizer.lr=2e-4

Training Script Usage
---------------------

**Using main.py:**

.. code-block:: bash

    # Basic training
    python scripts/main.py --config tutorials/lucchi.yaml

    # Override parameters
    python scripts/main.py --config tutorials/lucchi.yaml \
        training.max_epochs=300 \
        data.batch_size=4

    # Fast development run (1 batch)
    python scripts/main.py --config tutorials/lucchi.yaml --fast-dev-run

    # Testing mode
    python scripts/main.py --config tutorials/lucchi.yaml \
        --mode test \
        --checkpoint path/to/checkpoint.ckpt

Python API Usage
----------------

PyTorch Lightning Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**v2.0 Python API:**

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

Model Configuration
-------------------

Using MONAI and MedNeXt Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**MONAI BasicUNet:**

.. code-block:: yaml

    model:
      architecture: monai_basic_unet3d
      in_channels: 1
      out_channels: 3
      filters: [28, 36, 48, 64, 80]

**MedNeXt (State-of-the-Art):**

.. code-block:: yaml

    model:
      architecture: monai_basic_unet3d
      in_channels: 1
      out_channels: 3
      filters: [28, 36, 48, 64, 80]

Using Custom Models
^^^^^^^^^^^^^^^^^^^

You can still use custom models by wrapping them:

.. code-block:: python

    from connectomics.training.lit import ConnectomicsModule
    import torch.nn as nn

    class MyCustomModel(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            # Your model definition

        def forward(self, x):
            # Your forward pass
            return x

    # Create config
    cfg = load_config("my_config.yaml")

    # Use custom model
    custom_model = MyCustomModel(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels
    )

    # Wrap with Lightning
    lit_model = ConnectomicsModule(cfg, model=custom_model)

Data Loading Migration
-----------------------

**v1.0:**

.. code-block:: python

    from connectomics.data import build_dataloader

    # Build dataloaders
    train_loader = build_dataloader(cfg, mode='train')
    val_loader = build_dataloader(cfg, mode='val')

**v2.0:**

.. code-block:: python

    from connectomics.training.lit import ConnectomicsDataModule

    # Create data module (handles all splits)
    datamodule = ConnectomicsDataModule(cfg)

    # Access loaders if needed
    datamodule.setup('fit')
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

Inference Migration
-------------------

**v1.0:**

.. code-block:: bash

    python -u scripts/build.py \
        --config-file configs/Lucchi-Mitochondria.yaml \
        --inference \
        --checkpoint outputs/checkpoint_10000.pth

**v2.0:**

.. code-block:: bash

    python scripts/main.py \
        --config tutorials/lucchi.yaml \
        --mode test \
        --checkpoint outputs/epoch=99-val_loss=0.123.ckpt

Loss Function Migration
------------------------

**v1.0:**

.. code-block:: yaml

    MODEL:
      LOSS_OPTION: [['WeightedBCE', 'DiceLoss']]
      LOSS_WEIGHT: [[1.0, 0.5]]

**v2.0:**

.. code-block:: yaml

    model:
      loss_functions:
        - BCEWithLogitsLoss
        - DiceLoss
      loss_weights: [1.0, 0.5]

**Loss name mappings:**

.. list-table::
   :header-rows: 1

   * - v1.0
     - v2.0
   * - ``WeightedBCE``
     - ``BCEWithLogitsLoss``
   * - ``DiceLoss``
     - ``DiceLoss`` (same)
   * - ``WeightedMSE``
     - ``MSELoss``
   * - ``BCELoss``
     - ``BCEWithLogitsLoss``

Augmentation Migration
-----------------------

**v1.0 (Custom augmentation):**

.. code-block:: python

    from connectomics.data.augmentation import Compose

    augmentor = Compose([...])

**v2.0 (MONAI transforms):**

MONAI transforms are automatically applied through the data module. To customize:

.. code-block:: yaml

    data:
      use_augmentation: true
      augmentation_params:
        rotation_range: 45
        scale_range: [0.9, 1.1]
        elastic_deform: true

Multi-GPU Training Migration
-----------------------------

**v1.0:**

.. code-block:: yaml

    SYSTEM:
      NUM_GPUS: 4

.. code-block:: python

    # Manual DataParallel/DistributedDataParallel setup
    model = nn.DataParallel(model, device_ids=device)

**v2.0:**

.. code-block:: yaml

    system:
      num_gpus: 4  # Automatically uses DDP

Lightning handles distributed training automatically!

Checkpoint Format Migration
----------------------------

**v1.0 checkpoints:**

.. code-block:: python

    checkpoint_10000.pth  # Iteration-based

**v2.0 checkpoints:**

.. code-block:: python

    epoch=99-val_loss=0.123.ckpt  # Epoch-based with metrics

**Loading v1.0 checkpoints in v2.0:**

You may need to manually convert:

.. code-block:: python

    import torch

    # Load old checkpoint
    old_ckpt = torch.load("checkpoint_10000.pth")

    # Extract model weights
    model_weights = old_ckpt['model_state_dict']

    # Load into new model
    model = ConnectomicsModule(cfg)
    model.model.load_state_dict(model_weights)

Logging and Monitoring Migration
---------------------------------

**v1.0:**

.. code-block:: python

    # TensorBoard logging built-in
    # Logs in output directory

**v2.0:**

.. code-block:: yaml

    logging:
      save_dir: "outputs"
      experiment_name: "lucchi_exp"

      # Optional: Weights & Biases
      use_wandb: true
      wandb_project: "connectomics"
      wandb_entity: "your_team"

Lightning provides:

- Automatic TensorBoard logging
- Optional Weights & Biases integration
- Rich console logging with progress bars
- Metric tracking and visualization

Common Migration Issues
-----------------------

Issue: "No module named 'yacs'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Solution:** Install new dependencies:

.. code-block:: bash

    pip install omegaconf>=2.1.0

Issue: Config file not found
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Solution:** Update config path:

.. code-block:: bash

    # Old
    --config-file configs/Lucchi-Mitochondria.yaml

    # New
    --config tutorials/lucchi.yaml

Issue: Iteration vs Epoch
^^^^^^^^^^^^^^^^^^^^^^^^^^

v1.0 used iterations, v2.0 uses epochs.

**Conversion:**

.. code-block:: python

    # iterations = epochs * steps_per_epoch
    # epochs = iterations / steps_per_epoch

    # Example: 10000 iterations, 100 batches per epoch
    epochs = 10000 / 100 = 100

Issue: Model architecture not found
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Solution:** Use MONAI models or architecture registry:

.. code-block:: yaml

    # Old
    MODEL:
      ARCHITECTURE: 'unet_residual_3d'

    # New - MONAI
    model:
      architecture: monai_unet

    # Or use MedNeXt
    model:
      architecture: mednext
      mednext_size: S

Backward Compatibility
----------------------

v2.0 maintains backward compatibility for:

- ✅ YACS configs (still work, but deprecated)
- ✅ Legacy trainer (available in ``engine/trainer.py``)
- ✅ Custom models
- ✅ Data formats (HDF5, TIFF, etc.)
- ✅ Augmentation interface

Not backward compatible:

- ❌ Checkpoint format (need manual conversion)
- ❌ Direct imports from old modules (use new paths)

Running Both Systems Side-by-Side
----------------------------------

You can run both v1.0 and v2.0 systems:

.. code-block:: bash

    # v1.0 style (legacy)
    python scripts/build.py --config-file configs/old_config.yaml

    # v2.0 style (recommended)
    python scripts/main.py --config tutorials/new_config.yaml

Migration Examples
------------------

See the ``tutorials/`` directory for complete v2.0 examples:

- `tutorials/lucchi.yaml <https://github.com/zudi-lin/pytorch_connectomics/blob/master/tutorials/lucchi.yaml>`_
- `tutorials/mednext_lucchi.yaml <https://github.com/zudi-lin/pytorch_connectomics/blob/master/tutorials/mednext_lucchi.yaml>`_

Getting Help
------------

If you encounter issues during migration:

1. Check this migration guide
2. Read the :ref:`configuration guide <Configuration System>`
3. See :ref:`installation guide <Installation>`
4. Search `GitHub Issues <https://github.com/zudi-lin/pytorch_connectomics/issues>`_
5. Ask on `Slack <https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w>`_

Next Steps
----------

After migration:

1. ✅ Test training with new config
2. ✅ Verify metrics match previous results
3. ✅ Update documentation/scripts in your project
4. ✅ Consider using MONAI models for better performance
5. ✅ Explore new features (deep supervision, MedNeXt, etc.)

Welcome to v2.0! 🚀
