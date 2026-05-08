# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch Connectomics (PyTC) is a modern deep learning framework for automatic and semi-automatic semantic and instance segmentation in connectomics - reconstructing neural connections from electron microscopy (EM) images. The framework integrates PyTorch Lightning for orchestration and MONAI for medical imaging tools, maintained by Harvard's Visual Computing Group.

## Architecture Philosophy

The codebase follows a clean separation of concerns:
- **PyTorch Lightning**: Orchestration layer (training loop, distributed training, mixed precision, callbacks, logging)
- **MONAI**: Domain toolkit (medical image models, transforms, losses, metrics)
- **Hydra/OmegaConf**: Modern configuration management (type-safe, composable configs)

**Key Principle:** Lightning is the outer shell, MONAI is the inner toolbox. No reimplementation of training loops or domain-specific tools.

### V2/V3 Architecture Contract

The codebase enforces an explicit contract from the v2/v3 refactor:

1. **One canonical owner per concept.** No backward-compatibility shims, no facade re-exports, no duplicate import paths.
2. **Strict config.** Unknown top-level keys **raise** at load time. Removed fields raise. `getattr(cfg.x, "y", default)` ghost reads on undeclared fields are forbidden.
3. **Stages are separate.** Pipeline = `train → infer → decode → evaluate → tune`. Each stage has its own package and its own entry function. Combined test-mode is a thin wrapper that calls stage APIs in sequence.
4. **Dependency direction:** `config → utils → data → models → metrics`; `training → {config, data, models, metrics}`; `inference → {config, data, models}`; `decoding → {config, data, utils}`; `evaluation → {config, data, metrics}`; `runtime → {config, training, inference, decoding, evaluation}`. Static AST tests in `tests/unit/test_v3_guardrails.py` enforce this.
5. **Public API is explicit and small.** `tests/unit/test_public_api_snapshot.py` asserts exact `__all__` membership.

## Agent Design Principles

- **Ecosystem-first, no reinvention**: Leverage proven frameworks (PyTorch, Lightning, MONAI, nnU-Net) to keep the codebase modern, minimal, and scalable.
- **Config-first reproducibility**: Use Hydra/OmegaConf YAML composition + CLI overrides so experiments are declarative, reproducible, and easy to customize across datasets/benchmarks.
- **Modular + extensible connectomics workflows**: Separate concerns cleanly (config, data, training, inference, decoding, evaluation, runtime), expose registry-style extension points, and support large-volume EM workloads (tiling, sliding-window, multi-GPU) for both novices and agentic workflows.

## Installation

Requires Python 3.8+, PyTorch 1.8+. Install PyTorch separately for your CUDA version, then:

```bash
pip install -e .              # core
pip install -e .[full]        # +tifffile/wandb/jupyter/gputil
# extras: [optim] [wandb] [tiff] [viz] [metrics] [dev] [docs]
pip install git+https://github.com/PytorchConnectomics/MedNeXt.git   # optional MedNeXt
```

## Development Commands

### Environment Activation
```bash
# Activate your conda/virtual environment
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
```

### Running Training/Inference
```bash
# enable environment
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc

# Lightning-based training (NEW - Primary)
python scripts/main.py --config tutorials/lucchi.yaml

# Override config from CLI
python scripts/main.py --config tutorials/lucchi.yaml data.dataloader.batch_size=8 optimization.max_epochs=200

# Testing mode
python scripts/main.py --config tutorials/lucchi.yaml --mode test --checkpoint path/to/checkpoint.ckpt

# Fast dev run (1 batch for debugging)
python scripts/main.py --config tutorials/lucchi.yaml --fast-dev-run
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_models.py
python -m pytest tests/test_augmentations.py
python -m pytest tests/test_loss_functions.py
```

## Current Package Structure

Post-v3 layout (155 files, ~43K LOC). Many subpackages were renamed in v2/v3:
`models/arch → models/architectures`, `models/loss → models/losses`,
`training/loss → training/losses`, `training/optim → training/optimization`,
`data/dataset → data/datasets`, `data/augment → data/augmentation`,
`data/process → data/processing`. New top-level packages: `runtime/`,
`evaluation/`. Schema split: top-level `decoding`, `evaluation`, `inference`
sections each have their own dataclass module.

```
connectomics/                       # Main Python package (~155 files, ~43K LOC)
├── config/                         # Hydra/OmegaConf configuration system (no domain imports)
│   ├── pipeline/
│   │   ├── config_io.py            # Config loading, saving, merging, strict-key checks
│   │   ├── profile_engine.py       # YAML profile composition engine
│   │   ├── stage_resolver.py       # Multi-stage (train/test/tune) config resolution
│   │   └── dict_utils.py           # Plain-dict + cfg_get accessors
│   ├── hardware/
│   │   ├── auto_config.py          # Auto-configuration planner (GPU-aware)
│   │   ├── gpu_utils.py            # GPU memory estimation and batch-size planning
│   │   └── slurm_utils.py          # SLURM helpers
│   ├── profiles/                   # Section-level profile registries (yaml)
│   ├── templates/                  # Decoding templates (yaml)
│   ├── all_profiles.yaml           # Master registry index used by tutorials
│   └── schema/                     # Dataclass-based config schema definitions
│       ├── root.py                 # Top-level Config dataclass
│       ├── system.py               # System (GPU, CPU, seed)
│       ├── data.py                 # Data, dataloader, augmentation
│       ├── model.py                # Model config (+ model_monai/_mednext/_rsunet/_nnunet)
│       ├── optimization.py         # Optimizer, scheduler, training
│       ├── monitor.py              # Checkpoint, early stopping, logging
│       ├── inference.py            # Inference stage config (raw prediction)
│       ├── decoding.py             # Decoding stage config (split out in PR 8)
│       ├── evaluation.py           # Evaluation stage config (split out in PR 8)
│       └── stages.py               # Multi-stage (test/tune) wrappers
│
├── models/                         # Model architectures and loss functions
│   ├── build.py                    # Model factory (registry-based)
│   ├── architectures/              # Architecture registry + model wrappers
│   │   ├── registry.py             # Architecture registration system
│   │   ├── base.py                 # ConnectomicsModel base interface
│   │   ├── monai_models.py         # MONAI wrappers (4 architectures)
│   │   ├── mednext_models.py       # MedNeXt wrappers (2 architectures)
│   │   ├── nnunet_models.py        # nnU-Net pretrained wrappers (`nnunet`)
│   │   └── rsunet.py               # RSUNet models (2 architectures)
│   └── losses/                     # Loss function implementations
│       ├── build.py                # Loss factory
│       ├── losses.py               # Connectomics-specific losses
│       ├── metadata.py             # Loss metadata (target types, activation info)
│       └── regularization.py       # Regularization losses
│
├── training/                       # Training orchestration (no decoding/evaluation internals)
│   ├── lightning/                  # PyTorch Lightning integration (PRIMARY)
│   │   ├── model.py                # ConnectomicsModule (train/val/test steps, TTA)
│   │   ├── data.py                 # ConnectomicsDataModule
│   │   ├── data_factory.py         # Data dict creation from config
│   │   ├── trainer.py              # Trainer creation utilities
│   │   ├── callbacks.py            # Custom callbacks (NaN, EMA, …)
│   │   ├── runtime.py              # Run directory setup
│   │   ├── path_utils.py           # File path expansion utilities
│   │   ├── prediction_crops.py     # Prediction-crop helpers (extracted in PR 10)
│   │   ├── test_pipeline.py        # Test orchestration; delegates to evaluation/
│   │   ├── visualizer.py           # TensorBoard visualization
│   │   └── utils.py                # Thin remaining glue (was 771; now ~77)
│   ├── losses/                     # Loss orchestration
│   │   ├── orchestrator.py         # Multi-loss + deep supervision orchestrator
│   │   ├── plan.py                 # Loss plan builder from config
│   │   └── balancing.py            # Loss weight balancing
│   ├── optimization/               # Optimizers and schedulers
│   │   ├── build.py                # Optimizer/scheduler factory
│   │   └── lr_scheduler.py         # Custom LR schedulers (WarmupCosine, …)
│   ├── model_weights.py            # Weight loading/conversion utilities
│   └── debugging.py                # NaN detection and debugging utilities
│
├── data/                           # Data loading and preprocessing
│   ├── datasets/                   # Dataset classes
│   │   ├── base.py                 # Base dataset class
│   │   ├── dataset_volume_cached.py
│   │   ├── dataset_volume_h5_lazy.py
│   │   ├── dataset_volume_zarr_lazy.py
│   │   ├── dataset_filename.py     # Filename-based datasets (2D images)
│   │   ├── dataset_multi.py        # Multi-dataset wrapper
│   │   ├── data_dicts.py           # MONAI data dictionary creation
│   │   ├── crop_sampling.py        # Random crop sampling
│   │   ├── sampling.py             # Sampling strategies
│   │   └── split.py                # Train/val/test splitting
│   ├── augmentation/               # MONAI-based augmentations
│   │   ├── build.py                # Transform pipeline builder
│   │   ├── transforms.py           # Custom MONAI transforms
│   │   ├── augment_ops.py          # Augmentation primitive ops
│   │   └── transform_utils.py
│   ├── io/                         # Multi-format I/O
│   │   ├── io.py                   # HDF5, TIFF, PNG, NIfTI, Zarr reading/writing
│   │   ├── transforms.py           # LoadVolumed and related MONAI transforms
│   │   ├── tiles.py                # Tile I/O utilities
│   │   └── utils.py
│   └── processing/                 # Preprocessing and target generation
│       ├── build.py                # Transform pipeline builder
│       ├── target.py               # Label target generation
│       ├── transforms.py           # Processing MONAI transforms
│       ├── distance.py             # Distance transform computation
│       ├── flow.py                 # Optical flow computation
│       ├── weight.py               # Sample weight generation
│       ├── segment.py              # Segmentation utilities
│       ├── bbox.py / bbox_processor.py
│       ├── affinity.py / iou.py
│       ├── nnunet_preprocess.py    # nnU-Net-style preprocessing
│       ├── quantize.py             # Label quantization
│       └── misc.py
│
├── inference/                      # Stage 2: model prediction → raw artifacts
│   ├── stage.py                    # `run_prediction_inference` (canonical entry)
│   ├── manager.py                  # Inference manager
│   ├── sliding.py                  # Sliding window inference
│   ├── lazy.py / lazy_distributed.py  # Lazy-volume sliding window
│   ├── chunked.py                  # Chunked inference for large volumes
│   ├── chunk_grid.py               # Public chunk-grid utilities (per PR-14)
│   ├── tta.py / tta_combinations.py # Test-time augmentation
│   ├── output.py                   # Output saving utilities
│   └── artifact.py                 # `PredictionArtifactMetadata`,
│                                   #   `write_prediction_artifact`
│
├── decoding/                       # Stage 3: raw arrays → segmentation artifacts
│   ├── stage.py                    # `run_decoding_stage` entry
│   ├── pipeline.py                 # Decode-mode normalization + apply pipeline
│   ├── registry.py                 # Decoder registration (lazy registration via _BUILTINS_REGISTERED)
│   ├── base.py                     # Decoder dataclass + protocol
│   ├── postprocess.py              # Binary/instance post-processing
│   ├── streamed_chunked.py         # Chunked decode + CC stitching
│   ├── experiment_log.py           # Decode-experiment logging (extracted from training)
│   ├── decoders/                   # Concrete decoder implementations
│   │   ├── segmentation.py         # CC, distance-watershed, waterz
│   │   ├── segmentation_kernels.py # numba CC kernels
│   │   ├── synapse.py / abiss.py / branch_merge.py / waterz.py
│   ├── tuning/                     # Pure tuner (no `connectomics.training` imports)
│   │   └── optuna_tuner.py
│   └── utils.py
│
├── evaluation/                     # Stage 4: artifacts + GT → metrics (PR 4)
│   ├── stage.py                    # `run_evaluation_stage`
│   ├── metrics.py                  # Test-mode metric instantiation + computation
│   ├── nerl.py                     # Skeleton-based metrics (NERL/ERL)
│   ├── report.py                   # Metrics file writing + epoch logging
│   ├── context.py                  # `EvaluationContext` (decouples from Lightning module)
│   └── curvilinear.py
│
├── runtime/                        # CLI / dispatch / orchestration glue (PR 7)
│   ├── cli.py                      # `parse_args`, `setup_config`
│   ├── dispatch.py                 # Mode dispatch (train/test/tune/decode-only/cache-hit)
│   ├── output_naming.py            # Naming helpers (extracted in PR 2)
│   ├── checkpoint_dispatch.py      # Output-base derivation from checkpoint
│   ├── cache_resolver.py           # Cached prediction file detection / cache-only test path
│   ├── sharding.py                 # Independent test sharding
│   ├── tune_runner.py              # `run_tuning`, `load_and_apply_best_params` (PR 5)
│   ├── preflight.py                # Cross-section validation (moved from config in B5)
│   └── torch_safe_globals.py       # `torch.serialization.add_safe_globals` registry
│
├── metrics/                        # Metric implementations (no orchestration)
│   ├── metrics_seg.py              # TorchMetrics segmentation (Jaccard, Dice, VOI)
│   ├── metrics_skel.py             # Skeleton-based metrics
│   └── segmentation_numpy.py       # NumPy metrics (Adapted Rand, etc.)
│
└── utils/                          # Cross-domain primitives only
    ├── errors.py                   # Preflight config validation
    ├── visualizer.py               # TensorBoard visualization
    ├── download.py                 # Dataset downloading
    ├── debug_utils.py / debug_hooks.py
    └── label_overlap.py            # Vectorized label-overlap helper

scripts/                            # Entry points and utilities
├── main.py                         # Primary entry point — thin: parse → dispatch
├── decode_large.py                 # Large-volume decode workflow (custom config surface)
├── demo.py                         # Demo script for quick testing
├── profile_dataloader.py           # Data loading profiling tool
├── slurm_launcher.py               # SLURM cluster job launcher
├── visualize_neuroglancer.py       # Neuroglancer 3D visualization
├── download_data.py                # Dataset downloader
├── apply_volume_function.py        # Apply functions to volume files
├── images_to_h5.py                 # Convert image stacks to HDF5
├── downsample_nisb.py              # NISB dataset downsampling
├── validate_tutorial_configs.py    # Tutorial config validation (CI)
└── tools/                          # Additional utility scripts
    ├── compare_config.py
    └── eval_curvilinear.py

tutorials/                          # Example configurations (16 canonical YAMLs + custom workflows)
├── mitoEM/, neuron_nisb/, neuron_snemi/  # Multi-config experiment families
├── *.yaml                          # Dataset-specific configs
│                                   #   mito_lucchi++, mito_mitolab, mito_betaseg(_banis_v0/v1/v2),
│                                   #   neuron_liconn_mit(_x2), nuc_nucmm-z, syn_cremi,
│                                   #   vesicle_xm, fiber_linghu26, minimal, waterz_decoding
└── waterz_decoding_large{,_abiss}.yaml  # Custom large-volume workflow YAMLs
                                    #   (top-level `large_decode:`/`abiss_large:` keys;
                                    #   bypass structured Config; consumed by scripts/decode_large.py)

tests/                              # Test suite
├── unit/                           # Unit tests
│   ├── test_v3_guardrails.py       # Boundary AST tests, public API snapshots, strict-config raise
│   └── test_v2_boundaries.py       # V2 boundary contracts
├── integration/                    # Integration tests
├── benchmarks/                     # Smoke benchmarks (chunked write throughput, …)
└── e2e/                            # End-to-end tests (requires data)

docs/                               # Sphinx documentation
notebooks/                          # Jupyter notebooks
docker/                             # Docker containerization
```

### Stage Pipeline (top-level config sections)

| Stage | Top-level config | Entry function | Owns |
|---|---|---|---|
| train | `optimization`, `data`, `model` | `trainer.fit(...)` | model fitting + checkpoints |
| infer | `inference` | `inference.stage.run_prediction_inference` | model → raw prediction artifact |
| decode | `decoding` | `decoding.stage.run_decoding_stage` | raw prediction → segmentation artifact |
| evaluate | `evaluation` | `evaluation.stage.run_evaluation_stage` | artifact + GT → metrics |
| tune | `decoding.tuning` (+ `tune` stage block) | `runtime.tune_runner.run_tuning` | search over decode/postproc params |

## Configuration System

### Hydra Configuration (Primary)
The project uses **Hydra/OmegaConf** with dataclass-based configs for type safety and composability.

Canonical YAML layout:

- `connectomics/config/profiles/*.yaml`: section-level registries selected by `*.profile`
- `connectomics/config/templates/*.yaml`: explicit list-item templates, currently for top-level `decoding`
- `tutorials/*.yaml`: runnable experiments that `_base_` the shared registries

Canonical merge semantics:

- Profile payloads are merged into the target section as the base config.
- Explicit keys override profile keys.
- Explicit lists replace profile lists; list overrides are not additive.
- Canonical decoding expansion is explicit `template:` inside top-level `decoding`.
- Do not introduce `decoding_profile` or `- profile: decoding_*` usages.

**Config File Example** (`tutorials/lucchi.yaml`):
```yaml
system:
  num_gpus: 1
  num_cpus: 4
  seed: 42

model:
  architecture: monai_basic_unet3d
  in_channels: 1
  out_channels: 2
  filters: [32, 64, 128, 256, 512, 1024]
  dropout: 0.1
  loss_functions:
    - DiceLoss
    - BCEWithLogitsLoss
  loss_weights: [1.0, 1.0]

data:
  train_image: "datasets/lucchi/train_image.h5"
  train_label: "datasets/lucchi/train_label.h5"
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
```

**Key Config Sections (top-level):**
- `system`: Hardware (GPUs, CPUs, seed)
- `model`: Architecture, loss functions, model parameters
- `data`: Paths, batch size, augmentation
- `optimization`: Optimizer, scheduler, training-loop parameters
- `monitor`: Checkpoint, early stopping, logging configuration
- `inference`: Raw prediction stage (sliding window, TTA, chunking, output paths)
- `decoding`: Decoding pipeline (decoders, postprocessing, output, tuning)
- `evaluation`: Metric selection + thresholds for `evaluate` stage
- `test` / `tune`: Stage wrappers that pull from the top-level `inference` /
  `decoding` / `evaluation` sections

V3 schema split (PR 8): `inference.postprocessing` → `decoding.postprocessing`,
`inference.decoding_path` → `decoding.output_path`,
`inference.saved_prediction_path` → `decoding.input_prediction_path`. Architecture
`nnunet_pretrained` was renamed to `nnunet`. Strict-config raise: any unknown
top-level key fails at load time.

### Loading and Using Configs
```python
from connectomics.config import load_config, print_config

# Load config
cfg = load_config("tutorials/lucchi.yaml")

# Override from CLI or code
cfg.data.dataloader.batch_size = 8

# Print config
print_config(cfg)
```

## Model Building

### Architecture Registry System
The framework uses an extensible **architecture registry** for managing models:

```python
from connectomics.models.architectures import (
    list_architectures,
    get_architecture_builder,
    register_architecture,
    print_available_architectures,
)

# List all available architectures
archs = list_architectures()  # 8 total architectures

# Get detailed info with counts
print_available_architectures()
```

### Supported Architectures (8 Total)

**MONAI Models (4)** - No deep supervision:
- `monai_basic_unet3d`: Simple and fast 3D U-Net (also supports 2D)
- `monai_unet`: U-Net with residual units and advanced features
- `monai_unetr`: Transformer-based UNETR (Vision Transformer backbone)
- `monai_swin_unetr`: Swin Transformer U-Net (SOTA but memory-intensive)

**MedNeXt Models (2)** - WITH deep supervision:
- `mednext`: MedNeXt with predefined sizes (S/B/M/L) - RECOMMENDED
  - S: 5.6M params, B: 10.5M, M: 17.6M, L: 61.8M
- `mednext_custom`: MedNeXt with full parameter control for research

**RSUNet Models (2)** - Pure PyTorch, WITH deep supervision:
- `rsunet`: Residual symmetric U-Net with anisotropic convolutions (EM-optimized)
- `rsunet_iso`: RSUNet with isotropic convolutions for uniform voxel spacing

#### MedNeXt Integration
MedNeXt (MICCAI 2023) is a ConvNeXt-based architecture optimized for 3D medical image segmentation:

**Predefined Sizes** (`mednext` architecture):
```yaml
model:
  architecture: mednext
  mednext_size: S              # S (5.6M), B (10.5M), M (17.6M), or L (61.8M)
  mednext_kernel_size: 3       # 3, 5, or 7
  deep_supervision: true       # RECOMMENDED for best performance
```

**Custom Configuration** (`mednext_custom` architecture):
```yaml
model:
  architecture: mednext_custom
  mednext_base_channels: 32
  mednext_exp_r: [2, 3, 4, 4, 4, 4, 4, 3, 2]
  mednext_block_counts: [3, 4, 8, 8, 8, 8, 8, 4, 3]
  mednext_kernel_size: 7
  deep_supervision: true
  mednext_grn: true            # Global Response Normalization
```

**Key Features:**
- **Deep Supervision**: Multi-scale outputs (5 scales) for improved training
- **UpKern**: Weight initialization technique for larger kernels
- **Isotropic Spacing**: Prefers 1mm isotropic spacing (unlike nnUNet)
- **Training**: Use AdamW with lr=1e-3, constant LR (no scheduler)

**Note:** MedNeXt is an optional external dependency - see Installation section for setup

### Building Models
```python
from connectomics.models import build_model

# From config
model = build_model(cfg)

# Model info
print(model.get_model_info())  # Shows parameters, architecture details
```

### Model Factory (`models/build.py`)
- Registry-based model building system
- **Hydra/OmegaConf configs only**
- PyTorch Lightning handles parallelization automatically
- Clean error messages with architecture listing

## Loss Functions

### MONAI-Based Losses
```python
from connectomics.models.losses import create_loss

# Available losses
loss = create_loss(loss_name='DiceLoss')
loss = create_loss(loss_name='FocalLoss')
loss = create_loss(loss_name='TverskyLoss')
loss = create_loss(loss_name='DiceCELoss')
```

**Supported Losses:**
- `DiceLoss`: Soft Dice loss for segmentation
- `FocalLoss`: Focal loss for class imbalance
- `TverskyLoss`: Tversky loss for handling FP/FN trade-offs
- `DiceCELoss`: Combined Dice + Cross-Entropy
- `BCEWithLogitsLoss`: Binary cross-entropy with logits
- `CrossEntropyLoss`: Multi-class cross-entropy

Multiple losses can be combined with weights in the config.

## PyTorch Lightning Integration

### LightningModule (`training/lightning/model.py`)
Wraps models with automatic training features:
- Distributed training (DDP)
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate scheduling
- Checkpointing
- Multi-loss support
- **Deep supervision**: Multi-scale loss computation with automatic target resizing

```python
from connectomics.training.lightning import ConnectomicsModule

# Create Lightning module
lit_model = ConnectomicsModule(cfg)

# Or with pre-built model
lit_model = ConnectomicsModule(cfg, model=custom_model)
```

### LightningDataModule (`training/lightning/data.py`)
Handles data loading with MONAI transforms:
- Train/val/test splits
- MONAI CacheDataset for fast loading
- Automatic augmentation pipeline
- Persistent workers for efficiency

```python
from connectomics.training.lightning import ConnectomicsDataModule

datamodule = ConnectomicsDataModule(cfg)
```

### Trainer (`training/lightning/trainer.py`)
Convenience function for creating Lightning Trainer:
```python
from connectomics.training.lightning import create_trainer

trainer = create_trainer(cfg)
trainer.fit(lit_model, datamodule=datamodule)
```

## Data Pipeline

### Dataset Classes (`data/datasets/`)
- Support for HDF5, TIFF stacks, Zarr
- 3D volumetric EM data handling
- Multi-scale and multi-task labels
- Efficient caching and preprocessing

### Augmentation (`data/augmentation/`)
Uses **MONAI transforms** for:
- Intensity transformations
- Spatial transformations (rotation, scaling)
- Elastic deformation
- Random cropping
- Normalization

### Data Format
- **Input**: (batch, channels, depth, height, width)
- **Patch Size**: Typically 128x128x128 for 3D
- **Normalization**: Z-score or min-max per sample

## Training Workflow

### Standard Training Pipeline
```python
# 1. Load config
from connectomics.config import load_config
cfg = load_config("tutorials/lucchi.yaml")

# 2. Set seed
from pytorch_lightning import seed_everything
seed_everything(cfg.system.seed)

# 3. Create data module
from connectomics.training.lightning import ConnectomicsDataModule
datamodule = ConnectomicsDataModule(cfg)

# 4. Create model
from connectomics.training.lightning import ConnectomicsModule
model = ConnectomicsModule(cfg)

# 5. Create trainer
from connectomics.training.lightning import create_trainer
trainer = create_trainer(cfg)

# 6. Train
trainer.fit(model, datamodule=datamodule)

# 7. Test
trainer.test(model, datamodule=datamodule)
```

### CLI Training (Recommended)
```bash
python scripts/main.py --config tutorials/lucchi.yaml
```

## Key Features

### 1. Automatic Mixed Precision
```yaml
training:
  precision: "16-mixed"  # or "32", "bf16-mixed"
```

### 2. Distributed Training
```yaml
system:
  num_gpus: 4  # Automatically uses DDP
```

### 3. Gradient Accumulation
```yaml
training:
  accumulate_grad_batches: 4  # Effective batch size = 4x
```

### 4. Model Checkpointing
```yaml
checkpoint:
  monitor: "val/loss"
  mode: "min"
  save_top_k: 3
  save_last: true
```

### 5. Early Stopping
```yaml
early_stopping:
  monitor: "val/loss"
  patience: 10
  mode: "min"
```

### 6. Learning Rate Scheduling
```yaml
scheduler:
  name: CosineAnnealingLR
  warmup_epochs: 5
  min_lr: 1e-6
```

## Important Files

### Configuration
- `connectomics/config/schema/root.py`: Top-level Config dataclass
- `connectomics/config/schema/`: All dataclass schemas (incl. `decoding.py`, `evaluation.py`)
- `connectomics/config/pipeline/config_io.py`: Config loading, strict-key check, validation
- `connectomics/config/pipeline/profile_engine.py`: YAML profile composition engine
- `connectomics/config/pipeline/stage_resolver.py`: Multi-stage config resolution

### Models
- `connectomics/models/build.py`: Model factory
- `connectomics/models/architectures/registry.py`: Architecture registration system
- `connectomics/models/losses/build.py`: Loss factory
- `connectomics/training/optimization/build.py`: Optimizer/scheduler factory

### Lightning + Stage Entries
- `connectomics/training/lightning/model.py`: ConnectomicsModule (train/val/test)
- `connectomics/training/lightning/data.py`: ConnectomicsDataModule
- `connectomics/training/lightning/trainer.py`: Trainer utilities
- `connectomics/training/lightning/test_pipeline.py`: Test orchestration; delegates to `evaluation/`
- `connectomics/training/losses/orchestrator.py`: Multi-loss + deep supervision
- `connectomics/inference/stage.py`: `run_prediction_inference` (raw artifact)
- `connectomics/decoding/stage.py`: `run_decoding_stage`
- `connectomics/evaluation/stage.py`: `run_evaluation_stage`
- `connectomics/runtime/tune_runner.py`: `run_tuning`, `load_and_apply_best_params`

### Entry Points
- `scripts/main.py`: Primary entry — parse → setup config → `runtime.dispatch`
- `scripts/decode_large.py`: Custom large-volume decode workflow (consumes
  `large_decode:`/`abiss_large:` top-level keys in `tutorials/waterz_decoding_large*.yaml`;
  these YAMLs intentionally bypass the structured `Config` schema)

## Development Guidelines

### Adding New Architectures
1. Add builder function in `connectomics/models/architectures/`
2. Register with `@register_architecture("name")` decorator
3. Add config parameters to appropriate schema file in `config/schema/`
4. Create example config in `tutorials/`
5. Add tests

### Adding New Loss Functions
1. Implement in `connectomics/models/losses/losses.py`
2. Register in `create_loss()` function
3. Update documentation
4. Add unit tests

### Adding New Transforms
1. Use MONAI transforms when possible
2. Add custom transforms to `connectomics/data/augmentation/`
3. Register in transform builder

### Adding New Decoders
1. Implement in `connectomics/decoding/decoders/`
2. Register with `@register_decoder("name")` (lazy registration via `_BUILTINS_REGISTERED`)
3. Wire into `decoding/pipeline.py` if it needs orchestration glue
4. Reference from a tutorial YAML using a `template:` entry under top-level `decoding`

## Best Practices

1. **Use Lightning for training**: Don't reimplement training loops
2. **Use MONAI for domain tools**: Don't reimplement transforms/losses
3. **Use Hydra configs**: Type-safe, composable, CLI-friendly
4. **Modular code**: One responsibility per module
5. **Test everything**: Unit tests for all components
6. **Documentation**: Update docs when adding features

## Code Quality Status

### Migration Status
- ✅ **YACS → Hydra/OmegaConf**: complete
- ✅ **Custom trainer → Lightning**: complete
- ✅ **Custom models → MONAI/MedNeXt/RSUNet/nnUNet**: complete
- ✅ **V2 layout refactor** (Codex): complete — package boundaries, schema dataclasses,
  stage scaffolding
- ✅ **V3 boundary/contract refactor** (PR 0–11): mostly complete

### Codebase Metrics
- **Total Python files**: ~155 (connectomics module)
- **Lines of code**: ~43,000 (connectomics module)
- **Architecture**: enforced via static AST tests (`tests/unit/test_v3_guardrails.py`)
- **Type safety**: dataclass configs with strict-key check; `getattr(cfg.x, "y", default)`
  ghost reads on undeclared fields are forbidden
- **Public API**: snapshot-tested in `tests/unit/test_public_api_snapshot.py`

### V3 Refactor Status (post-PR 11)

The architectural skeleton is correct; behavioral cleanup partial.

**Landed cleanly:**
- Boundary fixes: decoding↛training, inference↛decoding, config↛data execution
- Strict config: unknown top-level keys raise on load
- Schema split: `decoding`, `evaluation` are now top-level config sections
- Stage separation: `inference.stage`, `decoding.stage`, `evaluation.stage`,
  `runtime.tune_runner` each expose a stage entry function
- Runtime extraction: `runtime/` package owns CLI, dispatch, naming,
  cache resolution, sharding, preflight, torch-safe-globals
- Tutorial migration: 38/40 canonical tutorials load through `Config`; the 2
  exceptions (`tutorials/waterz_decoding_large{,_abiss}.yaml`) are custom
  workflow YAMLs consumed by `scripts/decode_large.py` and intentionally bypass
  the structured schema (validator skips via `CUSTOM_WORKFLOW_ROOTS`)
- Architecture rename: `nnunet_pretrained` → `nnunet`
- Lazy decoder registration via `_BUILTINS_REGISTERED` flag
- Public API trim with snapshot tests

**Known follow-ups (post-v3):**
- Evaluation extraction is a file move; `connectomics.evaluation.*` functions
  still take `module` as first arg with `hasattr(module, "_…")` defensive
  fallbacks (PR-13 / `EvaluationContext` rewrite)
- `decoding/streamed_chunked.py` imports public chunk-grid utilities from
  `inference/chunk_grid.py` (PR-14 done) — verify no remaining underscore-prefixed
  cross-package imports
- `DecodingConfig.tuning: Optional[Dict[str, Any]]` should be a typed dataclass
- File splits in PR 10 are partial: `inference/lazy.py` (1295), the two
  `data/{augmentation,processing}/transforms.py` (1346 + 979),
  `training/lightning/data_factory.py` (1168), `callbacks.py` (1001),
  `decoders/segmentation.py` (815) are untouched and remain the highest-value
  splits when adjacent behavior changes are needed
- A3 product items (`RandMixupd`, `auto_plan_config`, `WandbConfig`,
  `GANLoss`, single-task wrappers, `TestConfig.output_path/cache_suffix`)
  deferred pending maintainer sign-off

### Boundary Tests (always run before merging refactors)
```bash
python -m pytest tests/unit/test_v3_guardrails.py tests/unit/test_v2_boundaries.py \
                 tests/unit/test_public_api_snapshot.py -q
python scripts/validate_tutorial_configs.py
```

## Dependencies

Authoritative list lives in `setup.py`/`pyproject.toml`. Highlights:

- **Core (auto-installed)**: torch≥1.8, pytorch-lightning≥2.0, monai≥0.9.1, torchmetrics, omegaconf≥2.1, numpy≥1.23, scipy, scikit-image, h5py, opencv-python, einops, cc3d, kimimaro, mahotas, fastremap, tensorboard, tqdm.
- **Extras**: `[full]` (tifffile, wandb, jupyter, gputil), `[optim]` (optuna), `[wandb]`, `[tiff]`, `[viz]` (neuroglancer), `[metrics]` (funlib.evaluate, manual: `pip install git+https://github.com/funkelab/funlib.evaluate.git`), `[dev]` (pytest, pytest-benchmark), `[docs]` (sphinx).
- **External**: MedNeXt (`pip install git+https://github.com/PytorchConnectomics/MedNeXt.git`, exposes `from nnunet_mednext import create_mednext_v1`); graceful fallback if missing.

## Common Issues

- **Config**: validate YAML, use `print_config(cfg)`; unknown top-level keys raise.
- **GPU OOM**: lower `data.dataloader.batch_size` / `patch_size`, use `precision: "16-mixed"`.
- **Slow data loading**: raise `system.num_workers`, set `data.dataloader.persistent_workers: true`, enable `use_cache` for small datasets.
- **Missing module errors**: reinstall with the matching extra (e.g. `pip install -e .[full]` for tifffile/wandb).

## Further Reading

### Documentation Files
- **README.md**: Project overview and quick start
- **QUICKSTART.md**: 5-minute setup guide
- **TROUBLESHOOTING.md**: Common issues and solutions
- **CONTRIBUTING.md**: Contribution guidelines
- **RELEASE_NOTES.md**: Version history and changes
- **tests/TEST_STATUS.md**: Detailed test coverage status
- **tests/README.md**: Testing guide

### External Resources
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/) - Training orchestration
- [MONAI Docs](https://docs.monai.io/en/stable/) - Medical imaging toolkit
- [Hydra Docs](https://hydra.cc/) - Configuration management
- [Project Documentation](https://zudi-lin.github.io/pytorch_connectomics/build/html/index.html) - Full docs
- [Slack Community](https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w) - Get help
