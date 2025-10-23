<a href="https://github.com/zudi-lin/pytorch_connectomics">
<img src="./.github/logo_fullname.png" width="450"></a>

<p align="left">
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.8+-ff69b4.svg" /></a>
    <a href= "https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-1.8+-2BAF2B.svg" /></a>
    <a href= "https://lightning.ai/">
      <img src="https://img.shields.io/badge/Lightning-2.0+-792EE5.svg" /></a>
    <a href= "https://monai.io/">
      <img src="https://img.shields.io/badge/MONAI-0.9+-00A3E0.svg" /></a>
    <a href= "https://github.com/zudi-lin/pytorch_connectomics/blob/master/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
    <a href= "https://zudi-lin.github.io/pytorch_connectomics/build/html/index.html">
      <img src="https://img.shields.io/badge/Doc-Latest-2BAF2B.svg" /></a>
    <a href= "https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w">
      <img src="https://img.shields.io/badge/Slack-Join-CC8899.svg" /></a>
    <a href= "https://arxiv.org/abs/2112.05754">
      <img src="https://img.shields.io/badge/arXiv-2112.05754-FF7F50.svg" /></a>
</p>

---

## Introduction

The field of **connectomics** aims to reconstruct the wiring diagram of the brain by mapping neural connections at the level of individual synapses. Recent advances in electron microscopy (EM) have enabled the collection of large-scale image stacks at nanometer resolution, but annotation requires expertise and is extremely time-consuming.

**PyTorch Connectomics** (PyTC) is a modern deep learning framework for automatic and semi-automatic **semantic and instance segmentation** in connectomics. Built on [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://lightning.ai/), and [MONAI](https://monai.io/), it provides a scalable, flexible, and easy-to-use platform for EM image analysis.

This repository is maintained by [Dr. Wei's lab](donglaiw.github.io) at Boston College.

🚀 **Version 2.0** features a complete rewrite with PyTorch Lightning orchestration and MONAI medical imaging tools!

---

## Key Features

### Modern Architecture (v2.0)
- ⚡ **PyTorch Lightning** integration for distributed training, mixed-precision, and automatic optimization
- 🏥 **MONAI** integration for medical imaging models, transforms, and losses
- 🔧 **Hydra/OmegaConf** configuration system for type-safe, composable configs
- 📦 **Architecture Registry** for easy model management and extensibility

### Training & Optimization
- 🎯 Multi-task, active, and semi-supervised learning
- 🚄 Distributed training (DDP) with automatic GPU parallelization
- 💾 Mixed-precision training (FP16/BF16) for faster training and reduced memory
- 📊 Gradient accumulation and checkpointing
- 🔄 Advanced learning rate scheduling with warmup

### Models & Architectures
- 🏗️ **MONAI Models**: BasicUNet3D, UNet, UNETR, Swin UNETR
- 🔬 **MedNeXt**: State-of-the-art ConvNeXt-based medical imaging models (MICCAI 2023)
- 🧩 Deep supervision support for multi-scale training
- 🎨 Custom architecture support through registry system

### Data Processing
- 📦 Support for HDF5, TIFF, Zarr formats
- 🔄 Comprehensive MONAI-based augmentations for volumetric data
- 💾 Efficient caching and preprocessing
- 🎲 Multi-scale and multi-task label handling

### Monitoring & Logging
- 📈 TensorBoard integration (default)
- 🔮 Weights & Biases (wandb) support
- ✅ Early stopping and model checkpointing
- 📉 Rich metrics tracking with TorchMetrics

---

## Installation

### TL;DR - Quick Install

```bash
# Automated (recommended)
python install.py

# Manual
conda create -n pytc python=3.10 -y  # Python 3.10 required for cc3d
conda activate pytc
# CRITICAL: Install these together to avoid build-time conflicts
conda install -c conda-forge numpy h5py cython connected-components-3d
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e . --no-build-isolation
```

### Prerequisites
- **Python**: 3.8 to 3.12 (**3.10 strongly recommended**)
  - ⚠️ **Python 3.10 required for cc3d** (connected-components-3d)
  - ⚠️ **Python 3.13 not yet supported** due to limited pre-built wheel availability
- **CUDA**: Recommended for GPU acceleration (11.8+ recommended)
- **GCC**: Version 4.8+ (for systems building from source)

### Quick Start

#### 0. Automated Installation (Recommended)

Run the Python installation script:

```bash
cd /path/to/pytorch_connectomics
python install.py
```

**Features:**
- 🎨 **Colored output** with progress indicators
- 🔍 **Auto-detects CUDA** version (nvidia-smi, nvcc, module system, /usr/local)
- 🌟 **Smart environment detection** - uses current conda env if active
- 🔧 **Command-line arguments** for customization
- ✅ **Better error handling** and user feedback
- 📊 **Installation verification** with detailed output
- 🚀 **CI/CD support** with `--yes` flag

**Smart features:**
```bash
# If you're already in a conda environment, it will detect and offer to use it:
conda activate my_env
python install.py  # Prompts: "Install in current environment 'my_env'? [Y/n]"
```

**Advanced usage:**
```bash
python install.py --env-name my_env --python 3.10  # Custom environment
python install.py --cuda 12.4                       # Specify CUDA version
python install.py --cpu-only                        # CPU-only installation
python install.py --yes                             # Skip prompts (CI mode)
python install.py --help                            # See all options
```

**What it installs:**
- ✅ Conda environment with Python 3.11 (or specified version)
- ✅ Pre-built scientific packages via conda-forge (NumPy, SciPy, h5py, Cython, etc.)
- ✅ PyTorch with matching CUDA support (auto-detected)
- ✅ PyTorch Connectomics and all dependencies

**Why this approach?**
- Uses pre-built conda binaries → No compilation required
- Avoids GCC version issues → Works with old compilers (GCC 4.8.5+)
- Faster installation → No building from source

**CUDA Support:**
- CUDA 11.x → PyTorch cu118
- CUDA 12.1-12.3 → PyTorch cu121
- CUDA 12.4+ → PyTorch cu124
- SLURM/HPC module systems
- CPU-only fallback

#### Manual Installation

##### 0. System Setup (Optional)

**Create Conda Environment:**

```bash
# Create a new conda environment with Python 3.10 (recommended)
conda create -n pytc python=3.10

# Activate the environment
conda activate pytc

# Note: Python 3.11 also works well. Avoid Python 3.13 for now.
```

**On SLURM systems**, load CUDA/cuDNN modules:

```bash
# Search for available CUDA versions
module avail cuda

# Search for available cuDNN versions
module avail cudnn

# Load the latest versions (example)
module load cuda/12.1
module load cudnn/8.9.0

# Verify CUDA version
nvcc --version
```

#### 1. Install PyTorch
Install PyTorch based on your CUDA version. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the correct command.

```bash
# Example for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only (not recommended)
pip install torch torchvision
```

#### 2. Install PyTorch Connectomics

```bash
# Clone the repository
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics

# checkout v2.0 branch
git checkout v2.0

# Basic installation (core dependencies only)
pip install -e .

# Full installation (recommended - includes all features)
pip install -e .[full]

# Development installation (includes testing tools)
pip install -e .[full,dev]
```

#### 3. Optional: Install MedNeXt Models

```bash
# Install MedNeXt for state-of-the-art segmentation
git clone https://github.com/MIC-DKFZ/MedNeXt.git
cd MedNeXt
pip install -e .
```

### Installation Options

Install with specific features:

```bash
# Hyperparameter optimization (Optuna)
pip install -e .[optim]

# Weights & Biases tracking
pip install -e .[wandb]

# TIFF file support
pip install -e .[tiff]

# 3D visualization (Neuroglancer)
pip install -e .[viz]

# Documentation building
pip install -e .[docs]

# Multiple features
pip install -e .[full,dev,docs]
```

### Verify Installation

```bash
# Check version
python -c "import connectomics; print('PyTC Version:', connectomics.__version__)"

# List available architectures
python -c "from connectomics.models.architectures import list_architectures; print('Available models:', list_architectures())"
```

### Docker Installation

We provide a Docker image for easy deployment:

```bash
# Pull the image
docker pull pytorchconnectomics/pytc:latest

# Or build from Dockerfile
cd docker
docker build -t pytc .
```

See [docker/README.md](docker/README.md) for detailed Docker instructions.

---

## Quick Start

This tutorial demonstrates the complete workflow: data download → visualization → training → testing → results visualization.

**Prerequisites:**
- Installed PyTorch Connectomics (see [Installation](#installation))
- [Just](https://github.com/casey/just) command runner (optional, for convenience)
  ```bash
  # Install just (optional)
  cargo install just  # Or: brew install just
  ```

### Lucchi++ Tutorial (Recommended)

#### 1. Download Tutorial Data

```bash
# Download Lucchi++ dataset from HuggingFace
wget https://huggingface.co/datasets/pytc/tutorial/resolve/main/Lucchi%2B%2B.zip
unzip Lucchi++.zip
```

#### 2. Visualize Data (Optional)

```bash
# Visualize training data
just visualize tutorials/monai_lucchi++.yaml train
```

#### 3. Train a Model

**On SLURM cluster:**
```bash
just slurm weilab 8 4 "train monai lucchi++"
# Arguments: partition, #gpus, #cpus, command
```

**On local machine:**
```bash
# Adjust #gpus and #cpus in tutorials/monai_lucchi++.yaml if needed
just train monai lucchi++
```

**Or directly with Python:**
```bash
python scripts/main.py --config tutorials/monai_lucchi++.yaml
```

#### 4. Monitor Training Progress

```bash
# Launch TensorBoard
just tensorboard monai_lucchi++

# Or directly:
tensorboard --logdir outputs/lucchi++_monai_unet
```

#### 5. Test the Model

```bash
# Test with best checkpoint
just test monai lucchi++ outputs/lucchi++_monai_unet/20251012_011259/checkpoints/epoch=869-step=8700.ckpt

# Or directly:
python scripts/main.py --config tutorials/monai_lucchi++.yaml \
    --mode test \
    --checkpoint outputs/lucchi++_monai_unet/20251012_011259/checkpoints/epoch=869-step=8700.ckpt
```

#### 6. Visualize Results

```bash
# Visualize predictions with Neuroglancer
just visualize tutorials/monai_lucchi++.yaml test \
    --port 5005 \
    --volumes pred:image:outputs/lucchi++_monai_unet/results/test_im_prediction.h5:5-5-5
```

### Custom Training Examples

```bash
# Train with example configuration
python scripts/main.py --config tutorials/lucchi.yaml

# Override config parameters from CLI
python scripts/main.py --config tutorials/lucchi.yaml \
    data.batch_size=4 \
    training.max_epochs=200 \
    model.architecture=monai_basic_unet3d

# Fast development run (1 batch for debugging)
python scripts/main.py --config tutorials/lucchi.yaml --fast-dev-run
```

### Example Configuration File

Create a config file (e.g., `my_config.yaml`):

```yaml
system:
  num_gpus: 1
  num_cpus: 4
  seed: 42

model:
  architecture: monai_basic_unet3d  # or 'mednext' for state-of-the-art
  in_channels: 1
  out_channels: 2
  filters: [32, 64, 128, 256, 512]
  dropout: 0.1
  loss_functions:
    - DiceLoss
    - BCEWithLogitsLoss
  loss_weights: [1.0, 1.0]

data:
  train_image: "path/to/train_image.h5"
  train_label: "path/to/train_label.h5"
  val_image: "path/to/val_image.h5"
  val_label: "path/to/val_label.h5"
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
  precision: "16-mixed"  # Mixed precision training
  gradient_clip_val: 1.0

checkpoint:
  monitor: "val/loss"
  mode: "min"
  save_top_k: 3
  save_last: true

logging:
  log_every_n_steps: 10
  save_dir: "outputs"
```

Then train:

```bash
python scripts/main.py --config my_config.yaml
```

### Using MedNeXt (State-of-the-Art)

```yaml
model:
  architecture: mednext
  mednext_size: S  # S (5.6M), B (10.5M), M (17.6M), or L (61.8M)
  mednext_kernel_size: 3  # 3, 5, or 7
  deep_supervision: true  # Recommended for best performance
  in_channels: 1
  out_channels: 2
  loss_functions:
    - DiceLoss
  loss_weights: [1.0]
```

### Python API

```python
from connectomics.config import load_config
from connectomics.lightning import ConnectomicsModule, ConnectomicsDataModule, create_trainer
from pytorch_lightning import seed_everything

# Load configuration
cfg = load_config("tutorials/lucchi.yaml")

# Set seed for reproducibility
seed_everything(cfg.system.seed)

# Create data module
datamodule = ConnectomicsDataModule(cfg)

# Create model
model = ConnectomicsModule(cfg)

# Create trainer
trainer = create_trainer(cfg)

# Train
trainer.fit(model, datamodule=datamodule)

# Test
trainer.test(model, datamodule=datamodule)
```

---

## Architecture Overview

PyTorch Connectomics v2.0 follows a clean separation of concerns:

```
┌─────────────────────────────────────────┐
│     PyTorch Lightning (Orchestration)    │
│  - Training loop                         │
│  - Distributed training (DDP)            │
│  - Mixed precision (AMP)                 │
│  - Callbacks & Logging                   │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│     MONAI (Medical Imaging Toolkit)      │
│  - Models (UNet, UNETR, etc.)           │
│  - Transforms & Augmentations            │
│  - Loss Functions                        │
│  - Metrics                               │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│     Hydra/OmegaConf (Configuration)      │
│  - Type-safe configs                     │
│  - CLI overrides                         │
│  - Composable configs                    │
└─────────────────────────────────────────┘
```

**Key Principle**: Lightning is the outer shell, MONAI is the inner toolbox. No reimplementation of training loops or domain-specific tools.

---

## Supported Models

### MONAI Models
- **BasicUNet3D**: Fast and simple 3D U-Net
- **UNet**: U-Net with residual units
- **UNETR**: Transformer-based architecture
- **Swin UNETR**: Swin Transformer U-Net

### MedNeXt Models (MICCAI 2023)
- **MedNeXt-S**: 5.6M parameters
- **MedNeXt-B**: 10.5M parameters
- **MedNeXt-M**: 17.6M parameters
- **MedNeXt-L**: 61.8M parameters

MedNeXt features:
- ConvNeXt-based architecture optimized for 3D medical imaging
- Deep supervision support (5 scales)
- UpKern weight initialization
- Global Response Normalization (GRN)

### Custom Models
Easily add your own models through the architecture registry system. See [.claude/CLAUDE.md](.claude/CLAUDE.md) for details.

---

## Loss Functions

Supports multiple MONAI-based loss functions:
- **DiceLoss**: Soft Dice for segmentation
- **FocalLoss**: Handles class imbalance
- **TverskyLoss**: FP/FN trade-off control
- **DiceCELoss**: Combined Dice + Cross-Entropy
- **BCEWithLogitsLoss**: Binary cross-entropy
- **CrossEntropyLoss**: Multi-class classification

Multiple losses can be combined with custom weights.

---

## Data Formats

- **HDF5** (.h5): Primary format (recommended)
- **TIFF** (.tif, .tiff): Stack support with optional tifffile
- **Zarr**: For large-scale datasets
- **NumPy** arrays: Direct loading

**Data Shape**: `(batch, channels, depth, height, width)`

**Typical Patch Size**: 128×128×128 for 3D volumes

---

## Advanced Features

### Distributed Training
```yaml
system:
  num_gpus: 4  # Automatically uses DDP
```

### Mixed Precision Training
```yaml
training:
  precision: "16-mixed"  # or "bf16-mixed", "32"
```

### Gradient Accumulation
```yaml
training:
  accumulate_grad_batches: 4  # Effective batch size = 4x
```

### Early Stopping
```yaml
early_stopping:
  monitor: "val/loss"
  patience: 10
  mode: "min"
```

### Deep Supervision
```yaml
model:
  deep_supervision: true  # Multi-scale loss computation
```

### Learning Rate Warmup
```yaml
scheduler:
  name: CosineAnnealingLR
  warmup_epochs: 5
  min_lr: 1e-6
```

---

## Documentation

- 📖 **Full Documentation**: [connectomics.readthedocs.io](https://connectomics.readthedocs.io)
- 📝 **Developer Guide**: [.claude/CLAUDE.md](.claude/CLAUDE.md)
- 🏗️ **Architecture Design**: [.claude/DESIGN.md](.claude/DESIGN.md)
- 🔬 **MedNeXt Integration**: [.claude/MEDNEXT.md](.claude/MEDNEXT.md)
- 🎯 **Tutorials**: See `tutorials/` directory for example configs

---

## Tutorials & Examples

Example configurations are provided in the `tutorials/` directory:

- `lucchi.yaml`: Basic MONAI UNet training
- `mednext_lucchi.yaml`: MedNeXt-S training
- `mednext_custom.yaml`: Custom MedNeXt configuration
- `monai_nucmm-z.yaml`: Nuclear membrane segmentation
- `monai2d_*.yaml`: 2D segmentation examples

Run any tutorial:
```bash
python scripts/main.py --config tutorials/lucchi.yaml
```

---

## Troubleshooting

### Installation Issues

**Problem: Python 3.13 build errors (`NumPy requires GCC >= 9.3`)**

```bash
# Solution 1: Use Python 3.10 (strongly recommended for cc3d compatibility)
conda create -n pytc python=3.10
conda activate pytc
pip install -e .

# Solution 2: Install NumPy via conda (pre-built binaries)
conda install "numpy<2.0" -c conda-forge
pip install -e . --no-build-isolation
```

**Problem: GCC version too old on system**

The installation script automatically handles this by using conda's pre-built binaries.

If you're installing manually:
```bash
# Install scientific packages via conda (pre-built, no GCC needed)
conda activate pytc
# CRITICAL: Include cc3d to avoid building it with wrong numpy version
conda install -c conda-forge numpy scipy scikit-learn scikit-image h5py cython connected-components-3d opencv
pip install -e . --no-build-isolation
```

**Problem: h5py, NumPy, or cc3d build errors (`NumPy requires GCC >= 9.3`)**

**Solution:** Install via conda FIRST, then pip:

```bash
conda activate pytc

# Install core packages via conda (pre-built binaries, no compilation)
# CRITICAL: Install cc3d with numpy to avoid version conflicts
conda install -c conda-forge numpy h5py cython connected-components-3d

# Then install PyTorch Connectomics
pip install -e . --no-build-isolation
```

**Why?** Conda provides pre-built binaries → no GCC 9.3+ needed. If pip installs these packages, it builds from source → fails on old GCC.

---

## Community & Support

- 💬 **Slack**: [Join our community](https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w)
- 📧 **Issues**: [GitHub Issues](https://github.com/zudi-lin/pytorch_connectomics/issues)
- 📚 **Documentation**: [ReadTheDocs](https://connectomics.readthedocs.io)
- 📄 **Paper**: [arXiv:2112.05754](https://arxiv.org/abs/2112.05754)

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where contributions are especially welcome:
- New model architectures
- Additional loss functions
- Data augmentation techniques
- Documentation improvements
- Bug fixes and performance optimizations

---

## Acknowledgements

This project is built upon numerous previous projects. We'd like to thank:

- [PyTorch Lightning](https://lightning.ai/): Lightning AI Team
- [MONAI](https://monai.io/): MONAI Consortium
- [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt): DKFZ Medical Image Computing

We gratefully acknowledge the support from:
- NSF awards IIS-1835231, IIS-2124179 and IIS-2239688

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

Copyright belongs to all PyTorch Connectomics contributors.

---

## Citation

If you find PyTorch Connectomics useful in your research, please cite:

```bibtex
@article{lin2021pytorch,
  title={PyTorch Connectomics: A Scalable and Flexible Segmentation Framework for EM Connectomics},
  author={Lin, Zudi and Wei, Donglai and Lichtman, Jeff and Pfister, Hanspeter},
  journal={arXiv preprint arXiv:2112.05754},
  year={2021}
}
```

---

## Version History

- **v2.0.0** (2024): Complete rewrite with PyTorch Lightning + MONAI
  - Lightning integration for distributed training
  - MONAI models and transforms
  - Hydra/OmegaConf configuration system
  - Architecture registry
  - MedNeXt integration

- **v1.0** (2021): Initial release
  - Custom trainer implementation
  - Custom augmentation pipeline

See [RELEASE_NOTES.md](RELEASE_NOTES.md) for detailed release notes.