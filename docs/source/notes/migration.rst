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

    # tutorials/minimal.yaml
    default:
      system:
        num_gpus: 1
        num_workers: 4
        seed: 42
      model:
        arch:
          type: monai_basic_unet3d
        in_channels: 1
        out_channels: 1
        loss:
          losses:
            - function: DiceLoss
              weight: 1.0
      data:
        dataloader:
          patch_size: [32, 64, 64]
          batch_size: 1

    train:
      optimization:
        max_epochs: 1
        precision: "32"
        optimizer:
          name: AdamW
          lr: 1e-4
      monitor:
        checkpoint:
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
   * - ``optimization``
     - Optimizer, scheduler, precision, and training loop parameters
   * - ``monitor``
     - Checkpointing, logging, early stopping, and experiment tracking

Configuration Override
^^^^^^^^^^^^^^^^^^^^^^

Override config parameters from CLI:

.. code-block:: bash

    python scripts/main.py --config tutorials/minimal.yaml \
        default.data.dataloader.batch_size=8 \
        train.optimization.max_epochs=200 \
        train.optimization.optimizer.lr=2e-4

Training Script Usage
---------------------

**Using main.py:**

.. code-block:: bash

    # Basic training
    python scripts/main.py --config tutorials/minimal.yaml

    # Override parameters
    python scripts/main.py --config tutorials/minimal.yaml \
        train.optimization.max_epochs=300 \
        default.data.dataloader.batch_size=4

    # Fast development run (1 batch)
    python scripts/main.py --config tutorials/minimal.yaml --fast-dev-run

    # Testing mode
    python scripts/main.py --config tutorials/minimal.yaml \
        --mode test \
        --checkpoint path/to/checkpoint.ckpt

Python API Usage
----------------

PyTorch Lightning Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**v2.0 Python API:**

.. code-block:: python

    from connectomics.config import load_config
    from connectomics.training.lightning import (
        ConnectomicsModule,
        create_datamodule,
        create_trainer
    )
    from pytorch_lightning import seed_everything

    # Load config
    cfg = load_config("tutorials/minimal.yaml")

    # Set seed
    seed_everything(cfg.system.seed)

    # Create components
    datamodule = create_datamodule(cfg)
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
      arch:
        type: monai_basic_unet3d
      in_channels: 1
      out_channels: 3
      monai:
        filters: [28, 36, 48, 64, 80]

**MedNeXt (State-of-the-Art):**

.. code-block:: yaml

    model:
      arch:
        type: mednext
      in_channels: 1
      out_channels: 3
      mednext:
        size: S

Using Custom Models
^^^^^^^^^^^^^^^^^^^

You can still use custom models by wrapping them:

.. code-block:: python

    from connectomics.training.lightning import ConnectomicsModule
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

    from connectomics.training.lightning import create_datamodule

    # Create data module (handles all splits)
    datamodule = create_datamodule(cfg)

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
        --config tutorials/minimal.yaml \
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
      loss:
        losses:
          - function: BCEWithLogitsLoss
            weight: 1.0
          - function: DiceLoss
            weight: 0.5

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

    from monai.transforms import Compose
    from connectomics.data.augmentation import RandMisAlignmentd

    augmentor = Compose([...])

**v2.0 (MONAI transforms):**

MONAI transforms are automatically applied through the data module. To customize:

.. code-block:: yaml

    data:
      augmentation:
        profile: aug_standard
        misalignment:
          enabled: true

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

    monitor:
      logging:
        scalar:
          loss_every_n_steps: 100
      wandb:
        use_wandb: true
        project: "connectomics"
        entity: "your_team"

Lightning provides:

- Automatic TensorBoard logging
- Optional Weights & Biases integration
- Rich console logging with progress bars
- Metric tracking and visualization

Common Migration Issues
-----------------------

Issue: "No module named 'yacs'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Solution:** remove the v1 YACS dependency path and convert the config to the v2
Hydra/OmegaConf format:

.. code-block:: bash

    python scripts/main.py --config tutorials/minimal.yaml

Issue: Config file not found
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Solution:** Update config path:

.. code-block:: bash

    # Old
    --config-file configs/Lucchi-Mitochondria.yaml

    # New
    --config tutorials/minimal.yaml

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
      arch:
        type: monai_unet

    # Or use MedNeXt
    model:
      arch:
        type: mednext
      mednext:
        size: S

Backward Compatibility
----------------------

v2.0 is not backward compatible with the v1 runtime:

- ❌ YACS configs are removed; convert configs to Hydra/OmegaConf YAML.
- ❌ Legacy trainer entry points are removed from the supported API.
- ❌ Checkpoint format needs manual conversion.
- ❌ Direct imports from old modules must move to the documented v2 paths.

Still supported:

- ✅ Custom PyTorch models through :class:`connectomics.training.lightning.ConnectomicsModule`.
- ✅ HDF5, TIFF, Zarr, and filename-based data inputs.
- ✅ MONAI-native augmentation pipelines.

Current Entry Point
-------------------

Use the v2 entry point for new runs:

.. code-block:: bash

    python scripts/main.py --config tutorials/minimal.yaml

Migration Examples
------------------

See the ``tutorials/`` directory for complete v2.0 examples:

- `tutorials/minimal.yaml <https://github.com/zudi-lin/pytorch_connectomics/blob/master/tutorials/minimal.yaml>`_
- `tutorials/mito_lucchi++.yaml <https://github.com/zudi-lin/pytorch_connectomics/blob/master/tutorials/mito_lucchi%2B%2B.yaml>`_
- `tutorials/neuron_snemi/neuron_snemi_sdt.yaml <https://github.com/zudi-lin/pytorch_connectomics/blob/master/tutorials/neuron_snemi/neuron_snemi_sdt.yaml>`_

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
