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

## Installation

### Prerequisites
- **Python**: 3.8 or higher
- **PyTorch**: 1.8.0 or higher (install separately based on your CUDA version)
- **CUDA**: Recommended for GPU acceleration

### Installation Methods

#### 1. Basic Installation (Recommended)
Install core dependencies only:
```bash
# Clone the repository
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics

# Install PyTorch (choose version based on your CUDA version)
# Visit https://pytorch.org/get-started/locally/ for the correct command
# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Connectomics in development mode
pip install -e .
```

#### 2. Full Installation (All Features)
Install with all optional features:
```bash
pip install -e .[full]
```

#### 3. Custom Installation
Install specific feature sets:
```bash
# With hyperparameter optimization
pip install -e .[optim]

# With Weights & Biases experiment tracking
pip install -e .[wandb]

# With TIFF file support
pip install -e .[tiff]

# With 3D visualization
pip install -e .[viz]

# With development and testing tools
pip install -e .[dev]

# With documentation building tools
pip install -e .[docs]

# Multiple extras
pip install -e .[full,dev,docs]
```

#### 4. MedNeXt Integration (Optional)
For MedNeXt model support:
```bash
# Install from external repository
pip install -e /projects/weilab/weidf/lib/MedNeXt

# Or clone and install
git clone https://github.com/MIC-DKFZ/MedNeXt.git
cd MedNeXt
pip install -e .
```
MedNeXt is an optional external package installed separately (see Installation section above).

### Verifying Installation
```bash
# Test import
python -c "import connectomics; print(connectomics.__version__)"

# List available architectures
python -c "from connectomics.models.arch import print_available_architectures; print_available_architectures()"
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
python scripts/main.py --config tutorials/lucchi.yaml data.batch_size=8 training.max_epochs=200

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

```
connectomics/                    # Main Python package (77 files, ~23K lines)
├── config/                      # Hydra/OmegaConf configuration system
│   ├── hydra_config.py          # Dataclass-based config definitions (PRIMARY)
│   ├── hydra_utils.py           # Config utilities (load, save, merge)
│   └── __init__.py
│
├── models/                      # Model architectures and training components
│   ├── build.py                 # Model factory (registry-based)
│   ├── arch/                    # Architecture registry and model wrappers
│   │   ├── __init__.py          # Public API and registration triggers
│   │   ├── registry.py          # Architecture registration system
│   │   ├── base.py              # Base model interface (ConnectomicsModel)
│   │   ├── monai_models.py      # MONAI model wrappers (4 architectures)
│   │   ├── mednext_models.py    # MedNeXt model wrappers (2 architectures)
│   │   └── rsunet.py            # RSUNet models (2 architectures)
│   ├── loss/                    # Loss function implementations
│   │   ├── build.py             # Loss factory (19 loss functions)
│   │   ├── losses.py            # Connectomics-specific losses
│   │   └── regularization.py    # Regularization losses
│   └── solver/                  # Optimizers and learning rate schedulers
│       ├── build.py             # Optimizer/scheduler factory
│       └── lr_scheduler.py      # Custom LR schedulers
│
├── training/                    # Training utilities
│   ├── lit/                     # PyTorch Lightning integration (PRIMARY)
│   │   ├── data.py              # LightningDataModule (Volume/Tile/Cloud datasets)
│   │   ├── model.py             # LightningModule (deep supervision, TTA)
│   │   ├── trainer.py           # Trainer creation utilities
│   │   ├── callbacks.py         # Custom Lightning callbacks
│   │   ├── config.py            # Factory functions for training setup
│   │   └── utils.py             # CLI/config helpers
│   ├── deep_supervision.py      # Deep supervision utilities
│   └── debugging.py             # NaN detection and debugging utilities
│
├── data/                        # Data loading and preprocessing
│   ├── dataset/                 # Dataset classes (HDF5, TIFF, Zarr, Cloud)
│   │   ├── build.py             # Dataset factory
│   │   ├── dataset_base.py      # Base dataset class
│   │   ├── dataset_volume.py    # Volume-based datasets
│   │   ├── dataset_tile.py      # Tile-based datasets
│   │   └── ...                  # Multi-dataset, filename-based, etc.
│   ├── augment/                 # MONAI-based augmentations
│   │   ├── build.py             # Transform pipeline builder (791 lines)
│   │   ├── monai_transforms.py  # Custom MONAI transforms (1.4K lines)
│   │   └── ...                  # EM-specific, geometry, advanced augmentations
│   ├── io/                      # Multi-format I/O (HDF5, TIFF, PNG, Pickle)
│   ├── process/                 # Preprocessing and target generation
│   └── utils/                   # Data utilities
│
├── decoding/                    # Post-processing and instance segmentation
│   └── ...                      # Auto-tuning, instance decoding
│
├── metrics/                     # Evaluation metrics
│   └── metrics_seg.py           # Segmentation metrics (Adapted Rand, VOI, etc.)
│
└── utils/                       # General utilities
    └── ...                      # Visualization, system setup, misc

scripts/                         # Entry points and utilities
├── main.py                      # Primary entry point (53KB, Lightning + Hydra)
├── profile_dataloader.py        # Data loading profiling tool
├── slurm_launcher.py            # SLURM cluster job launcher
├── visualize_neuroglancer.py    # Neuroglancer visualization (29KB)
└── tools/                       # Additional utility scripts

tutorials/                       # Example configurations (11 YAML files)
├── monai_lucchi++.yaml          # Lucchi mitochondria (MONAI)
├── monai_fiber.yaml             # Fiber segmentation
├── monai_bouton-bv.yaml         # Bouton + blood vessel multi-task
├── monai2d_worm.yaml            # 2D C. elegans segmentation
├── mednext_mitoEM.yaml          # MitoEM dataset (MedNeXt)
├── mednext2d_cem-mitolab.yaml   # 2D MedNeXt example
├── rsunet_snemi.yaml            # SNEMI3D neuron segmentation (RSUNet)
├── sweep_example.yaml           # Hyperparameter sweep example
└── ...                          # Additional tutorials

tests/                           # Test suite (organized by type)
├── unit/                        # Unit tests (38/61 passing - 62%)
├── integration/                 # Integration tests (0/6 passing - needs update)
├── e2e/                         # End-to-end tests (requires data setup)
├── test_rsunet.py               # RSUNet model tests
├── test_banis_features.py       # Feature extraction tests
├── TEST_STATUS.md               # Detailed test status report
└── README.md                    # Testing documentation

docs/                            # Sphinx documentation
notebooks/                       # Jupyter notebooks
docker/                          # Docker containerization
conda-recipe/                    # Conda packaging
```

## Configuration System

### Hydra Configuration (Primary)
The project uses **Hydra/OmegaConf** with dataclass-based configs for type safety and composability.

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

**Key Config Sections:**
- `system`: Hardware (GPUs, CPUs, seed)
- `model`: Architecture, loss functions, model parameters
- `data`: Paths, batch size, augmentation
- `optimizer`: Optimizer type and hyperparameters
- `scheduler`: Learning rate scheduling
- `training`: Training loop parameters
- `checkpoint`: Model checkpointing strategy
- `logging`: Logging configuration

### Loading and Using Configs
```python
from connectomics.config import load_config, print_config

# Load config
cfg = load_config("tutorials/lucchi.yaml")

# Override from CLI or code
cfg.data.batch_size = 8

# Print config
print_config(cfg)
```

## Model Building

### Architecture Registry System
The framework uses an extensible **architecture registry** for managing models:

```python
from connectomics.models.arch import (
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
from connectomics.models.loss import create_loss

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

### LightningModule (`training/lit/model.py`)
Wraps models with automatic training features:
- Distributed training (DDP)
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate scheduling
- Checkpointing
- Multi-loss support
- **Deep supervision**: Multi-scale loss computation with automatic target resizing

```python
from connectomics.training.lit import ConnectomicsModule

# Create Lightning module
lit_model = ConnectomicsModule(cfg)

# Or with pre-built model
lit_model = ConnectomicsModule(cfg, model=custom_model)
```

### LightningDataModule (`training/lit/data.py`)
Handles data loading with MONAI transforms:
- Train/val/test splits
- MONAI CacheDataset for fast loading
- Automatic augmentation pipeline
- Persistent workers for efficiency

```python
from connectomics.training.lit import ConnectomicsDataModule

datamodule = ConnectomicsDataModule(cfg)
```

### Trainer (`training/lit/trainer.py`)
Convenience function for creating Lightning Trainer:
```python
from connectomics.training.lit import create_trainer

trainer = create_trainer(cfg)
trainer.fit(lit_model, datamodule=datamodule)
```

## Data Pipeline

### Dataset Classes (`data/dataset/`)
- Support for HDF5, TIFF stacks, Zarr
- 3D volumetric EM data handling
- Multi-scale and multi-task labels
- Efficient caching and preprocessing

### Augmentation (`data/augment/`)
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
from connectomics.training.lit import ConnectomicsDataModule
datamodule = ConnectomicsDataModule(cfg)

# 4. Create model
from connectomics.training.lit import ConnectomicsModule
model = ConnectomicsModule(cfg)

# 5. Create trainer
from connectomics.training.lit import create_trainer
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
- `connectomics/config/hydra_config.py`: Dataclass config definitions
- `connectomics/config/hydra_utils.py`: Config utilities

### Models
- `connectomics/models/build.py`: Model factory
- `connectomics/models/loss/build.py`: Loss factory
- `connectomics/models/solver/build.py`: Optimizer/scheduler factory

### Lightning
- `connectomics/training/lit/model.py`: Lightning module wrapper
- `connectomics/training/lit/data.py`: Data module
- `connectomics/training/lit/trainer.py`: Trainer utilities

### Entry Points
- `scripts/main.py`: Primary training script (Lightning + Hydra)

## Development Guidelines

### Adding New Architectures
1. Add builder function to `connectomics/models/build.py`
2. Register architecture name in supported list
3. Add config parameters to `hydra_config.py`
4. Create example config in `tutorials/`
5. Add tests

### Adding New Loss Functions
1. Implement in `connectomics/models/loss/losses.py`
2. Register in `create_loss()` function
3. Update documentation
4. Add unit tests

### Adding New Transforms
1. Use MONAI transforms when possible
2. Add custom transforms to `connectomics/data/augment/`
3. Register in transform builder

## Best Practices

1. **Use Lightning for training**: Don't reimplement training loops
2. **Use MONAI for domain tools**: Don't reimplement transforms/losses
3. **Use Hydra configs**: Type-safe, composable, CLI-friendly
4. **Modular code**: One responsibility per module
5. **Test everything**: Unit tests for all components
6. **Documentation**: Update docs when adding features

## Code Quality Status

### Migration Status: ✅ Complete (100%)
- ✅ **YACS → Hydra/OmegaConf**: 100% migrated (all YACS code removed)
- ✅ **Custom trainer → Lightning**: 100% migrated
- ✅ **Custom models → MONAI models**: Primary path uses MONAI
- ✅ **Legacy configs**: All YACS config files removed

### Codebase Metrics
- **Total Python files**: 109 (77 in connectomics module)
- **Lines of code**: ~23,000 (connectomics module)
- **Architecture**: Modular, well-organized
- **Type safety**: Good (dataclass configs, type hints in most modules)
- **Test coverage**: 62% unit tests passing (38/61), integration tests need updates

### Known Technical Debt
1. **lit_model.py size**: 1,830 lines (should be split into smaller modules)
2. ~~**Code duplication**: Training/validation steps share deep supervision logic (~140 lines)~~ ✅ **FIXED**
3. ~~**NotImplementedError**: `create_tile_data_dicts_from_json()` not implemented~~ ✅ **FIXED**
4. **Hardcoded values**: Output clamping, deep supervision weights, interpolation bounds
5. **Dummy validation dataset**: Masks configuration errors instead of proper handling

### Overall Assessment: **8.5/10 - Production Ready**
- ✅ Modern architecture (Lightning + MONAI + Hydra)
- ✅ Clean separation of concerns
- ✅ Comprehensive feature set
- ✅ Good documentation
- ✅ No code duplication (refactored)
- ✅ All legacy code removed
- ✅ No NotImplementedError functions (all implemented)
- ⚠️ Integration tests need API v2.0 migration

## Migration Notes

### From Legacy System
The codebase has **fully migrated** from legacy systems:
- ✅ YACS configs → Hydra/OmegaConf configs (100% complete, all legacy removed)
- ✅ Custom trainer → PyTorch Lightning (100% complete)
- ✅ Custom models → MONAI native models (100% complete)
- ✅ `scripts/build.py` → `scripts/main.py` (legacy script removed)
- ✅ All legacy config files removed (`configs/barcode/` deleted)

**Current development stack:**
- Hydra/OmegaConf configs (`tutorials/*.yaml`)
- PyTorch Lightning modules (`connectomics/training/lit/`)
- `scripts/main.py` entry point
- MONAI models and transforms
- Type-safe dataclass configurations

## Dependencies

### Core Dependencies (Always Installed)
The following packages are automatically installed with `pip install -e .`:

| Package | Version | Purpose |
|---------|---------|---------|
| **torch** | >=1.8.0 | Deep learning framework |
| **numpy** | >=1.23.0 | Numerical computing (compatible with mahotas 1.4.18+) |
| **pytorch-lightning** | >=2.0.0 | Training orchestration (PRIMARY) |
| **monai** | >=0.9.1 | Medical imaging toolkit (PRIMARY) |
| **torchmetrics** | >=0.11.0 | Metrics computation |
| **omegaconf** | >=2.1.0 | Hydra configuration (PRIMARY) |
| **scipy** | >=1.5 | Signal processing, optimization |
| **scikit-learn** | >=0.23.1 | Machine learning utilities |
| **scikit-image** | >=0.17.2 | Image processing |
| **opencv-python** | >=4.3.0 | Computer vision (geometric transforms for augmentations) |
| **h5py** | >=2.10.0 | HDF5 file I/O |
| **matplotlib** | >=3.3.0 | Visualization |
| **tensorboard** | >=2.2.2 | Training monitoring |
| **tqdm** | >=4.58.0 | Progress bars |
| **einops** | >=0.3.0 | Tensor operations |
| **psutil** | >=5.8.0 | System monitoring |
| **cc3d** | >=3.0.0 | Connected components (segmentation) |
| **imageio** | >=2.9.0 | Image I/O |
| **kimimaro** | >=1.0.0 | Skeletonization (instance segmentation) |
| **crackle-codec** | >=0.1.0 | Compression codec (required by kimimaro) |
| **mahotas** | >=1.4.18 | Morphological operations & watershed (segmentation postprocessing) |
| **fastremap** | >=1.10.0 | Fast label remapping (segmentation utilities) |
| **Cython** | >=0.29.22 | C extensions |

### Optional Dependencies

Install via `pip install -e .[extra_name]` where `extra_name` is:

#### `[full]` - Recommended Full Installation
- **gputil** (>=1.4.0): GPU utilities
- **jupyter** (>=1.0): Interactive notebooks
- **tifffile** (>=2021.11.2): TIFF file support
- **wandb** (>=0.13.0): Experiment tracking

#### `[optim]` - Hyperparameter Optimization
- **optuna** (>=2.10.0): Automated hyperparameter tuning
- Used in: `connectomics.decoding.auto_tuning`

#### `[wandb]` - Weights & Biases Integration
- **wandb** (>=0.13.0): Experiment tracking and monitoring
- Used in: `connectomics.training.lit.trainer` (optional logger)

#### `[tiff]` - TIFF File Support
- **tifffile** (>=2021.11.2): Advanced TIFF reading/writing
- Used in: `connectomics.data.io`

#### `[viz]` - 3D Visualization
- **neuroglancer** (>=1.0.0): Interactive 3D EM data visualization
- Used in: `scripts/visualize_neuroglancer.py`

#### `[metrics]` - Advanced Metrics
- **funlib.evaluate**: Skeleton-based segmentation metrics (NERL, VOI)
- Manual install: `pip install git+https://github.com/funkelab/funlib.evaluate.git`
- Used in: `connectomics.decoding.auto_tuning`

#### `[dev]` - Development Tools
- **pytest** (>=6.0.0): Testing framework
- **pytest-benchmark** (>=3.4.0): Performance benchmarking

#### `[docs]` - Documentation Building
- **sphinx** (==3.4.3): Documentation generator
- **sphinxcontrib-katex**: Math rendering
- **jinja2** (==3.0.3): Template engine
- Additional sphinx extensions

### External Dependencies

#### MedNeXt Integration (Optional)
MedNeXt models are available from external repository:
- **Location**: `/projects/weilab/weidf/lib/MedNeXt`
- **Import**: `from nnunet_mednext import create_mednext_v1`
- **Installation**: `pip install -e /projects/weilab/weidf/lib/MedNeXt`
- **Documentation**: See `.claude/MEDNEXT.md` for integration guide
- **Note**: Graceful fallback if not installed (try/except import protection)

### Dependency Notes

1. **PyTorch Installation**: Install PyTorch separately based on your CUDA version before installing PyTorch Connectomics. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the correct command.

2. **Optional Features**: All optional dependencies have graceful fallbacks - the package will work without them, but some features will be unavailable.

3. **GPU Utilities**: `gputil` and `psutil` are used for GPU/system monitoring but are not critical for core functionality.

4. **Post-Processing**: `cc3d` is required for connected component analysis in segmentation tasks and is included in core dependencies.

## Common Issues

### Installation Issues

#### Missing PyTorch
```bash
# Error: No module named 'torch'
# Solution: Install PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Missing OmegaConf
```bash
# Error: No module named 'omegaconf'
# Solution: Upgrade to latest version
pip install --upgrade omegaconf
```

#### Missing cc3d
```bash
# Error: No module named 'cc3d'
# Solution: Install connected-components-3d
pip install connected-components-3d
```

#### Import Error: MedNeXt
```bash
# Error: Could not import MedNeXt
# Solution: This is optional - install if needed
pip install -e /projects/weilab/weidf/lib/MedNeXt
# Or see .claude/MEDNEXT.md for installation instructions
```

### Config Loading
- Ensure YAML syntax is correct
- Check paths are absolute or relative to working directory
- Use `print_config(cfg)` to debug
- Verify OmegaConf is installed (`pip install omegaconf>=2.1.0`)

### GPU Memory Issues
- Reduce `data.batch_size`
- Enable gradient checkpointing (model-specific)
- Use mixed precision (`training.precision: "16-mixed"`)
- Try smaller patch sizes in `data.patch_size`

### Data Loading Issues
- Increase `data.num_workers` for faster loading
- Use `data.use_cache` for small datasets
- Check `data.persistent_workers: true` for efficiency
- Verify HDF5 files exist and are accessible (`h5py` installed)

### Version Compatibility
- **Python 3.8+** required (3.10 recommended)
- **PyTorch 1.8+** required (2.0+ recommended)
- **PyTorch Lightning 2.0+** required
- **MONAI 0.9.1+** required (1.0+ recommended)

### Environment Issues
```bash
# Reset environment if needed
pip uninstall connectomics
pip cache purge
pip install -e .[full]

# Verify installation
python -c "import connectomics; print('Version:', connectomics.__version__)"
python -c "from connectomics.models.arch import list_architectures; print(list_architectures())"
```

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
