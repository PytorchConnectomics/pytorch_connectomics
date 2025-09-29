# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch Connectomics (PyTC) is a deep learning framework for automatic and semi-automatic semantic and instance segmentation in connectomics - reconstructing neural connections from electron microscopy (EM) images. The framework is built on PyTorch and maintained by Harvard's Visual Computing Group.

## Development Commands

### Installation and Setup
```bash
# Install package in development mode
pip install --editable .

# Install from setup.py requirements
python setup.py install
```

### Running Training/Inference
```bash
# Basic training
python scripts/main.py --config-file configs/[dataset]/[config].yaml

# Inference mode
python scripts/main.py --config-file configs/[dataset]/[config].yaml --inference

# Distributed training
python -m torch.distributed.launch --nproc_per_node=4 scripts/main.py --config-file configs/[dataset]/[config].yaml
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_models.py
python -m pytest tests/test_augmentations.py
python -m pytest tests/test_loss_functions.py
python -m pytest tests/test_model_blocks.py
```

## Architecture Overview

### Core Package Structure
- **connectomics/config/**: YACS-based configuration system with defaults in `defaults.py`
- **connectomics/models/**: Neural network architectures and components
  - `arch/`: Complete model architectures (UNet, FPN variants)
  - `backbone/`: Backbone networks (ResNet, etc.)
  - `block/`: Building blocks for networks
  - `loss/`: Loss function implementations
- **connectomics/data/**: Data loading and preprocessing
  - `dataset/`: Dataset classes for different data formats
  - `augmentation/`: Volume-specific data augmentation
- **connectomics/engine/**: Training and inference engine
  - `trainer.py`: Main training loop and orchestration
  - `solver/`: Optimizers and learning rate scheduling
- **connectomics/utils/**: Utilities for system setup, visualization, etc.

### Configuration System
The project uses YACS for configuration management. All options are defined in `connectomics/config/defaults.py`. Configuration files in `configs/` override defaults for specific datasets and experiments.

Key config sections:
- `SYSTEM`: Hardware setup (GPUs, CPUs, distributed training)
- `MODEL`: Architecture, blocks, filters, loss functions
- `DATASET`: Data paths, preprocessing, augmentation
- `SOLVER`: Optimizer, learning rate, training schedule

### Model Building
Models are constructed through the factory pattern in `connectomics/models/build.py`. The system supports:
- Multiple architectures: UNet variants, FPN, custom architectures
- Flexible backbone networks: ResNet, EfficientNet, etc.
- Modular building blocks: residual, dense, squeeze-excitation
- Multi-task learning with different loss functions per task

### Data Pipeline
The data system handles 3D volumetric EM data:
- Custom dataset classes for different data formats (HDF5, TIFF stacks)
- Volume-specific augmentations (rotation, elastic deformation, etc.)
- Support for both isotropic and anisotropic data
- Multi-scale and multi-task label handling

## Common Workflow

1. **Dataset Setup**: Place data according to config file paths, typically in `IMAGE_NAME` and `LABEL_NAME` fields
2. **Configuration**: Create or modify YAML config files in `configs/` directory
3. **Training**: Run `python scripts/main.py --config-file path/to/config.yaml`
4. **Inference**: Add `--inference` flag and set appropriate inference options in config
5. **Evaluation**: Use built-in metrics or external tools depending on task

## Important Files

- `connectomics/config/defaults.py`: All configuration options with documentation
- `scripts/main.py`: Main entry point for training/inference
- `connectomics/engine/trainer.py`: Core training loop implementation
- `connectomics/models/build.py`: Model factory and construction logic

## Development Notes

- The codebase uses mixed precision training and distributed training support
- All models expect 5D tensors (batch, channel, depth, height, width)
- Configuration validation happens at runtime - check defaults.py for valid options
- The project supports both semantic segmentation and instance segmentation tasks
- Custom loss functions and metrics can be added through the modular architecture