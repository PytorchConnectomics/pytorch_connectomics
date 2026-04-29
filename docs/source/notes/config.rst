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
    python scripts/main.py --config tutorials/minimal.yaml

    # Override config from CLI
    python scripts/main.py --config tutorials/minimal.yaml \
        default.data.dataloader.batch_size=4 \
        train.optimization.max_epochs=200

**Python API:**

.. code-block:: python

    from connectomics.config import load_config
    from omegaconf import OmegaConf

    # Load config
    cfg = load_config("tutorials/minimal.yaml")

    # Access values
    print(cfg.model.arch.type)             # 'monai_basic_unet3d'
    print(cfg.data.dataloader.batch_size)  # 1

    # Modify values
    cfg.data.dataloader.batch_size = 4

    # Print entire config
    print(OmegaConf.to_yaml(cfg, resolve=True))

Configuration Structure
-----------------------

A typical v2.0 config file has a ``default`` section plus stage-specific
overrides such as ``train`` and ``test``:

.. code-block:: yaml

    experiment_name: example

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
        input_size: [64, 128, 128]
        output_size: [64, 128, 128]
        loss:
          losses:
            - function: DiceLoss
              weight: 1.0
      data:
        dataloader:
          batch_size: 2
          patch_size: [64, 128, 128]

    train:
      data:
        train:
          image: datasets/example/train_image.h5
          label: datasets/example/train_label.h5
        val:
          image: datasets/example/val_image.h5
          label: datasets/example/val_label.h5
      optimization:
        max_epochs: 100
        precision: "16-mixed"
        optimizer:
          name: AdamW
          lr: 1e-4
      monitor:
        checkpoint:
          monitor: train_loss_total_epoch
          save_top_k: 3
          save_last: true

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
      arch:
        type: monai_basic_unet3d         # Model architecture
      in_channels: 1                     # Input channels
      out_channels: 2                    # Output channels
      monai:
        filters: [32, 64, 128, 256]      # Filter sizes per level
        dropout: 0.1                     # Dropout rate

      # Loss functions
      loss:
        deep_supervision: true
        losses:
          - function: DiceLoss
            weight: 1.0
          - function: BCEWithLogitsLoss
            weight: 1.0

      # Optional: architecture-specific nested blocks
      mednext:
        size: S

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
      train:
        image: "path/to/train_image.h5"
        label: "path/to/train_label.h5"
      val:
        image: "path/to/val_image.h5"
        label: "path/to/val_label.h5"
      test:
        image: "path/to/test_image.h5"  # Optional

      dataloader:
        patch_size: [128, 128, 128]
        batch_size: 2
        persistent_workers: true
        pin_memory: true

      # Augmentation
      augmentation:
        profile: aug_standard

Optimizer Configuration
^^^^^^^^^^^^^^^^^^^^^^^

Specifies optimizer type and hyperparameters:

.. code-block:: yaml

    optimization:
      optimizer:
        name: AdamW         # Optimizer type
        lr: 1e-4            # Learning rate
        weight_decay: 1e-4  # Weight decay (L2 regularization)

        # Optimizer-specific params
        betas: [0.9, 0.999]  # For Adam/AdamW
        momentum: 0.9        # For SGD

**Supported optimizers:**

- ``Adam``, ``AdamW``, ``SGD``, ``RMSprop``, ``Adagrad``

Scheduler Configuration
^^^^^^^^^^^^^^^^^^^^^^^

Specifies learning rate scheduling:

.. code-block:: yaml

    optimization:
      scheduler:
        name: CosineAnnealingLR
        warmup_epochs: 5
        min_lr: 1e-6

        # Scheduler-specific params
        params:
          T_max: 100

**Supported schedulers:**

- ``CosineAnnealingLR``, ``StepLR``, ``ExponentialLR``, ``ReduceLROnPlateau``

Training Configuration
^^^^^^^^^^^^^^^^^^^^^^

Controls training loop parameters:

.. code-block:: yaml

    optimization:
      max_epochs: 100
      precision: "16-mixed"         # "32", "16-mixed", "bf16-mixed"
      gradient_clip_val: 1.0
      accumulate_grad_batches: 1    # Gradient accumulation
      val_check_interval: 1.0       # Validation frequency

Command Line Overrides
-----------------------

Override any config value from the command line:

.. code-block:: bash

    # Override single values
    python scripts/main.py --config tutorials/minimal.yaml \
        default.data.dataloader.batch_size=4

    # Override multiple values
    python scripts/main.py --config tutorials/minimal.yaml \
        default.data.dataloader.batch_size=4 \
        train.optimization.max_epochs=200 \
        train.optimization.optimizer.lr=1e-3

    # Override nested values
    python scripts/main.py --config tutorials/minimal.yaml \
        default.model.monai.filters=[64,128,256,512]

    # Add new values
    python scripts/main.py --config tutorials/minimal.yaml \
        +description="debug run"

Multiple Loss Functions
------------------------

Combine multiple loss functions with different weights:

.. code-block:: yaml

    model:
      loss:
        losses:
          - function: DiceLoss
            weight: 1.0
          - function: BCEWithLogitsLoss
            weight: 1.0
          - function: FocalLoss
            weight: 0.5

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
      arch:
        type: mednext
      loss:
        deep_supervision: true
        losses:
          - function: DiceLoss
            weight: 1.0

Deep supervision automatically:

- Computes losses at multiple scales (5 scales for MedNeXt)
- Resizes ground truth to match each scale
- Averages losses across scales

MedNeXt Configuration
---------------------

**Predefined sizes:**

.. code-block:: yaml

    model:
      arch:
        type: mednext
      mednext:
        size: S                    # S, B, M, or L
        kernel_size: 3             # 3, 5, or 7
      in_channels: 1
      out_channels: 2
      loss:
        deep_supervision: true

**Custom configuration:**

.. code-block:: yaml

    model:
      arch:
        type: mednext_custom
      mednext:
        base_channels: 32
        exp_r: [2, 3, 4, 4, 4, 4, 4, 3, 2]
        block_counts: [3, 4, 8, 8, 8, 8, 8, 4, 3]
        kernel_size: 7
        grn: true
      loss:
        deep_supervision: true

See `.claude/MEDNEXT.md <https://github.com/zudi-lin/pytorch_connectomics/blob/master/.claude/MEDNEXT.md>`_ for details.

2D Configuration
----------------

For 2D segmentation tasks:

.. code-block:: yaml

    data:
      train:
        do_2d: true
      dataloader:
        patch_size: [1, 256, 256]  # [D, H, W] - D=1 for 2D

Mixed Precision Training
------------------------

Use mixed precision for faster training and reduced memory:

.. code-block:: yaml

    optimization:
      precision: "16-mixed"  # FP16 mixed precision

    # Or for BFloat16 (requires Ampere+ GPUs)
    optimization:
      precision: "bf16-mixed"

Distributed Training
--------------------

Automatically use distributed training with multiple GPUs:

.. code-block:: yaml

    system:
      num_gpus: 4  # Uses DDP automatically

    data:
      dataloader:
        batch_size: 2  # Per-GPU batch size

Effective batch size = ``num_gpus * batch_size = 4 * 2 = 8``

Gradient Accumulation
---------------------

Simulate larger batch sizes:

.. code-block:: yaml

    data:
      dataloader:
        batch_size: 2

    optimization:
      accumulate_grad_batches: 4

Effective batch size = ``batch_size * accumulate_grad_batches = 2 * 4 = 8``

Checkpointing and Logging
--------------------------

**Model checkpointing:**

.. code-block:: yaml

    monitor:
      checkpoint:
        monitor: "val/loss"
        mode: "min"              # "min" or "max"
        save_top_k: 3            # Keep best 3 checkpoints
        save_last: true          # Also save last checkpoint
        filename: "epoch{epoch:02d}-loss{val/loss:.2f}"

**Early stopping:**

.. code-block:: yaml

    monitor:
      early_stopping:
        enabled: true
        monitor: "val/loss"
        patience: 10
        mode: "min"
        min_delta: 0.0

**Logging:**

.. code-block:: yaml

    monitor:
      logging:
        scalar:
          loss_every_n_steps: 10
      wandb:
        use_wandb: false
        project: "connectomics"
        entity: "your_team"

Configuration in Python
-----------------------

**Load and modify configs:**

.. code-block:: python

    from connectomics.config import load_config, save_config
    from omegaconf import OmegaConf

    # Load config
    cfg = load_config("tutorials/minimal.yaml")

    # Access values
    print(cfg.model.arch.type)
    print(cfg.data.dataloader.batch_size)

    # Modify values
    cfg.data.dataloader.batch_size = 4
    cfg.optimization.max_epochs = 200

    # Merge configs
    overrides = OmegaConf.create({
        "data": {"dataloader": {"batch_size": 8}},
        "optimization": {"optimizer": {"lr": 1e-3}}
    })
    cfg = OmegaConf.merge(cfg, overrides)

    # Save config
    save_config(cfg, "modified_config.yaml")

    # Print config
    print(OmegaConf.to_yaml(cfg, resolve=True))

**Create configs programmatically:**

.. code-block:: python

    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "system": {"num_gpus": 1, "seed": 42},
        "model": {
            "arch": {"type": "monai_unet"},
            "in_channels": 1,
            "out_channels": 2
        },
        "data": {
            "dataloader": {
                "batch_size": 2,
                "patch_size": [128, 128, 128]
            }
        }
    })

Inference Configuration
-----------------------

Many training configs are reused for inference. Key differences:

.. code-block:: yaml

    # inference_config.yaml
    model:
      arch:
        type: monai_unet
      # ... same as training

    data:
      test:
        image: "path/to/test.h5"
      dataloader:
        patch_size: [128, 128, 128]
        batch_size: 4  # Can use larger batch size

    inference:
      output_path: "predictions/"
      sliding_window:
        overlap: 0.5
        blend_mode: gaussian
      test_time_augmentation:
        enabled: false

**Run inference:**

.. code-block:: bash

    python scripts/main.py \
        --config inference_config.yaml \
        --mode test \
        --checkpoint outputs/best.ckpt

Configuration Examples
----------------------

See the ``tutorials/`` directory for complete examples:

- `tutorials/minimal.yaml <https://github.com/zudi-lin/pytorch_connectomics/blob/master/tutorials/minimal.yaml>`_: minimal MONAI smoke config
- `tutorials/mito_lucchi++.yaml <https://github.com/zudi-lin/pytorch_connectomics/blob/master/tutorials/mito_lucchi%2B%2B.yaml>`_: mitochondria segmentation
- `tutorials/neuron_snemi/neuron_snemi_sdt.yaml <https://github.com/zudi-lin/pytorch_connectomics/blob/master/tutorials/neuron_snemi/neuron_snemi_sdt.yaml>`_: MedNeXt SNEMI config

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
