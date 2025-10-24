Configuration System
=====================

.. note::
   **PyTorch Connectomics v2.0** uses **Hydra/OmegaConf** as the configuration system.

PyTorch Connectomics uses a flexible, type-safe configuration system built on
`Hydra <https://hydra.cc/>`_ and `OmegaConf <https://omegaconf.readthedocs.io/>`_.
Configuration files are written in YAML and support CLI overrides, composition, and type checking.

Quick Start
-----------

**Basic training:**

.. code-block:: bash

    # Train with a config file
    python scripts/main.py --config tutorials/lucchi.yaml

    # Override config from CLI
    python scripts/main.py --config tutorials/lucchi.yaml \
        data.batch_size=4 \
        training.max_epochs=200

**Python API:**

.. code-block:: python

    from connectomics.config import load_config, print_config

    # Load config
    cfg = load_config("tutorials/lucchi.yaml")

    # Access values
    print(cfg.model.architecture)  # 'monai_basic_unet3d'
    print(cfg.data.batch_size)     # 2

    # Modify values
    cfg.data.batch_size = 4

    # Print entire config
    print_config(cfg)

Configuration Structure
-----------------------

A typical v2.0 config file has the following sections:

.. code-block:: yaml

    # System configuration
    system:
      num_gpus: 1
      num_cpus: 4
      seed: 42

    # Model configuration
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

    # Data configuration
    data:
      train_image: "datasets/lucchi/train_image.h5"
      train_label: "datasets/lucchi/train_label.h5"
      val_image: "datasets/lucchi/val_image.h5"
      val_label: "datasets/lucchi/val_label.h5"
      patch_size: [128, 128, 128]
      batch_size: 2
      num_workers: 4

    # Optimizer configuration
    optimizer:
      name: AdamW
      lr: 1e-4
      weight_decay: 1e-4

    # Scheduler configuration
    scheduler:
      name: CosineAnnealingLR
      warmup_epochs: 5
      min_lr: 1e-6

    # Training configuration
    training:
      max_epochs: 100
      precision: "16-mixed"
      gradient_clip_val: 1.0
      accumulate_grad_batches: 1

    # Checkpoint configuration
    checkpoint:
      monitor: "val/loss"
      mode: "min"
      save_top_k: 3
      save_last: true

    # Logging configuration
    logging:
      log_every_n_steps: 10
      save_dir: "outputs"

Configuration Sections
----------------------

System Configuration
^^^^^^^^^^^^^^^^^^^^

Controls hardware and reproducibility:

.. code-block:: yaml

    system:
      num_gpus: 1          # Number of GPUs (0 for CPU)
      num_cpus: 4          # Number of CPU workers
      seed: 42             # Random seed for reproducibility
      deterministic: false # Use deterministic algorithms (slower)

Model Configuration
^^^^^^^^^^^^^^^^^^^

Specifies model architecture and loss functions:

.. code-block:: yaml

    model:
      architecture: monai_basic_unet3d  # Model architecture
      in_channels: 1                     # Input channels
      out_channels: 2                    # Output channels
      filters: [32, 64, 128, 256, 512]  # Filter sizes per level
      dropout: 0.1                       # Dropout rate

      # Loss functions
      loss_functions:
        - DiceLoss
        - BCEWithLogitsLoss
      loss_weights: [1.0, 1.0]

      # Optional: Deep supervision
      deep_supervision: true

**Available architectures:**

- ``monai_basic_unet3d``: Simple and fast 3D U-Net
- ``monai_unet``: U-Net with residual units
- ``monai_unetr``: Transformer-based UNETR
- ``monai_swin_unetr``: Swin Transformer U-Net
- ``mednext``: MedNeXt with predefined sizes (S/B/M/L)
- ``mednext_custom``: MedNeXt with custom parameters

**Available loss functions:**

- ``DiceLoss``: Soft Dice loss
- ``FocalLoss``: Focal loss for class imbalance
- ``TverskyLoss``: Tversky loss
- ``DiceCELoss``: Combined Dice + Cross-Entropy
- ``BCEWithLogitsLoss``: Binary cross-entropy
- ``CrossEntropyLoss``: Multi-class cross-entropy

Data Configuration
^^^^^^^^^^^^^^^^^^

Specifies data paths and loading parameters:

.. code-block:: yaml

    data:
      # Data paths
      train_image: "path/to/train_image.h5"
      train_label: "path/to/train_label.h5"
      val_image: "path/to/val_image.h5"
      val_label: "path/to/val_label.h5"
      test_image: "path/to/test_image.h5"  # Optional

      # Patch sampling
      patch_size: [128, 128, 128]

      # Data loader settings
      batch_size: 2
      num_workers: 4
      persistent_workers: true
      pin_memory: true

      # Augmentation
      use_augmentation: true
      augmentation_params:
        rotation_range: 45
        scale_range: [0.9, 1.1]

Optimizer Configuration
^^^^^^^^^^^^^^^^^^^^^^^

Specifies optimizer type and hyperparameters:

.. code-block:: yaml

    optimizer:
      name: AdamW           # Optimizer type
      lr: 1e-4             # Learning rate
      weight_decay: 1e-4   # Weight decay (L2 regularization)

      # Optimizer-specific params
      betas: [0.9, 0.999]  # For Adam/AdamW
      momentum: 0.9        # For SGD

**Supported optimizers:**

- ``Adam``, ``AdamW``, ``SGD``, ``RMSprop``, ``Adagrad``

Scheduler Configuration
^^^^^^^^^^^^^^^^^^^^^^^

Specifies learning rate scheduling:

.. code-block:: yaml

    scheduler:
      name: CosineAnnealingLR
      warmup_epochs: 5
      min_lr: 1e-6

      # Scheduler-specific params
      T_max: 100           # For CosineAnnealingLR
      step_size: 30        # For StepLR
      gamma: 0.1           # For StepLR, ExponentialLR

**Supported schedulers:**

- ``CosineAnnealingLR``, ``StepLR``, ``ExponentialLR``, ``ReduceLROnPlateau``

Training Configuration
^^^^^^^^^^^^^^^^^^^^^^

Controls training loop parameters:

.. code-block:: yaml

    training:
      max_epochs: 100
      precision: "16-mixed"         # "32", "16-mixed", "bf16-mixed"
      gradient_clip_val: 1.0
      gradient_clip_algorithm: "norm"
      accumulate_grad_batches: 1    # Gradient accumulation
      val_check_interval: 1.0       # Validation frequency
      limit_train_batches: 1.0      # For debugging
      limit_val_batches: 1.0

Command Line Overrides
-----------------------

Override any config value from the command line:

.. code-block:: bash

    # Override single values
    python scripts/main.py --config tutorials/lucchi.yaml \
        data.batch_size=4

    # Override multiple values
    python scripts/main.py --config tutorials/lucchi.yaml \
        data.batch_size=4 \
        training.max_epochs=200 \
        optimizer.lr=1e-3

    # Override nested values
    python scripts/main.py --config tutorials/lucchi.yaml \
        model.filters=[64,128,256,512]

    # Add new values
    python scripts/main.py --config tutorials/lucchi.yaml \
        +training.fast_dev_run=true

Multiple Loss Functions
------------------------

Combine multiple loss functions with different weights:

.. code-block:: yaml

    model:
      loss_functions:
        - DiceLoss
        - BCEWithLogitsLoss
        - FocalLoss
      loss_weights: [1.0, 1.0, 0.5]

The total loss is computed as:

.. code-block:: python

    total_loss = (1.0 * dice_loss +
                  1.0 * bce_loss +
                  0.5 * focal_loss)

Deep Supervision
----------------

Enable multi-scale loss computation for improved training:

.. code-block:: yaml

    model:
      architecture: mednext
      deep_supervision: true
      loss_functions:
        - DiceLoss
      loss_weights: [1.0]

Deep supervision automatically:

- Computes losses at multiple scales (5 scales for MedNeXt)
- Resizes ground truth to match each scale
- Averages losses across scales

MedNeXt Configuration
---------------------

**Predefined sizes:**

.. code-block:: yaml

    model:
      architecture: mednext
      mednext_size: S              # S, B, M, or L
      mednext_kernel_size: 3       # 3, 5, or 7
      deep_supervision: true
      in_channels: 1
      out_channels: 2

**Custom configuration:**

.. code-block:: yaml

    model:
      architecture: mednext_custom
      mednext_base_channels: 32
      mednext_exp_r: [2, 3, 4, 4, 4, 4, 4, 3, 2]
      mednext_block_counts: [3, 4, 8, 8, 8, 8, 8, 4, 3]
      mednext_kernel_size: 7
      mednext_grn: true
      deep_supervision: true

See `.claude/MEDNEXT.md <https://github.com/zudi-lin/pytorch_connectomics/blob/master/.claude/MEDNEXT.md>`_ for details.

2D Configuration
----------------

For 2D segmentation tasks:

.. code-block:: yaml

    model:
      architecture: monai_basic_unet2d  # or monai_unet2d
      in_channels: 1
      out_channels: 2
      filters: [32, 64, 128, 256]

    data:
      patch_size: [1, 256, 256]  # [D, H, W] - D=1 for 2D

Mixed Precision Training
------------------------

Use mixed precision for faster training and reduced memory:

.. code-block:: yaml

    training:
      precision: "16-mixed"  # FP16 mixed precision

    # Or for BFloat16 (requires Ampere+ GPUs)
    training:
      precision: "bf16-mixed"

Distributed Training
--------------------

Automatically use distributed training with multiple GPUs:

.. code-block:: yaml

    system:
      num_gpus: 4  # Uses DDP automatically

    data:
      batch_size: 2  # Per-GPU batch size

Effective batch size = ``num_gpus * batch_size = 4 * 2 = 8``

Gradient Accumulation
---------------------

Simulate larger batch sizes:

.. code-block:: yaml

    data:
      batch_size: 2

    training:
      accumulate_grad_batches: 4

Effective batch size = ``batch_size * accumulate_grad_batches = 2 * 4 = 8``

Checkpointing and Logging
--------------------------

**Model checkpointing:**

.. code-block:: yaml

    checkpoint:
      monitor: "val/loss"
      mode: "min"              # "min" or "max"
      save_top_k: 3            # Keep best 3 checkpoints
      save_last: true          # Also save last checkpoint
      filename: "epoch{epoch:02d}-loss{val/loss:.2f}"

**Early stopping:**

.. code-block:: yaml

    early_stopping:
      monitor: "val/loss"
      patience: 10
      mode: "min"
      min_delta: 0.0

**Logging:**

.. code-block:: yaml

    logging:
      log_every_n_steps: 10
      save_dir: "outputs"
      experiment_name: "lucchi_exp"

      # Weights & Biases (optional)
      use_wandb: false
      wandb_project: "connectomics"
      wandb_entity: "your_team"

Configuration in Python
-----------------------

**Load and modify configs:**

.. code-block:: python

    from connectomics.config import load_config, save_config, print_config
    from omegaconf import OmegaConf

    # Load config
    cfg = load_config("tutorials/lucchi.yaml")

    # Access values
    print(cfg.model.architecture)
    print(cfg.data.batch_size)

    # Modify values
    cfg.data.batch_size = 4
    cfg.training.max_epochs = 200

    # Merge configs
    overrides = OmegaConf.create({
        "data": {"batch_size": 8},
        "optimizer": {"lr": 1e-3}
    })
    cfg = OmegaConf.merge(cfg, overrides)

    # Save config
    save_config(cfg, "modified_config.yaml")

    # Print config
    print_config(cfg)

**Create configs programmatically:**

.. code-block:: python

    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "system": {"num_gpus": 1, "seed": 42},
        "model": {
            "architecture": "monai_basic_unet3d",
            "in_channels": 1,
            "out_channels": 2
        },
        "data": {
            "batch_size": 2,
            "patch_size": [128, 128, 128]
        }
    })

Inference Configuration
-----------------------

Many training configs are reused for inference. Key differences:

.. code-block:: yaml

    # inference_config.yaml
    model:
      architecture: monai_basic_unet3d
      # ... same as training

    data:
      test_image: "path/to/test.h5"
      patch_size: [128, 128, 128]
      batch_size: 4  # Can use larger batch size

    inference:
      checkpoint_path: "outputs/best.ckpt"
      output_path: "predictions/"
      overlap: 0.5               # Overlap between patches
      blend_mode: "gaussian"     # "gaussian" or "linear"
      do_tta: false             # Test-time augmentation

**Run inference:**

.. code-block:: bash

    python scripts/main.py \
        --config inference_config.yaml \
        --mode test \
        --checkpoint outputs/best.ckpt

Configuration Examples
----------------------

See the ``tutorials/`` directory for complete examples:

- `tutorials/lucchi.yaml <https://github.com/zudi-lin/pytorch_connectomics/blob/master/tutorials/lucchi.yaml>`_: MONAI BasicUNet
- `tutorials/mednext_lucchi.yaml <https://github.com/zudi-lin/pytorch_connectomics/blob/master/tutorials/mednext_lucchi.yaml>`_: MedNeXt-S
- `tutorials/mednext_custom.yaml <https://github.com/zudi-lin/pytorch_connectomics/blob/master/tutorials/mednext_custom.yaml>`_: Custom MedNeXt

Best Practices
--------------

1. **Use version control** for config files
2. **Document** non-obvious parameter choices
3. **Start simple** with basic configs, then customize
4. **Save configs** with experiment outputs for reproducibility
5. **Use meaningful names** for experiments
6. **Validate configs** before long training runs

For more information:

- `Hydra Documentation <https://hydra.cc/>`_
- `OmegaConf Documentation <https://omegaconf.readthedocs.io/>`_
- `.claude/CLAUDE.md <https://github.com/zudi-lin/pytorch_connectomics/blob/master/.claude/CLAUDE.md>`_
