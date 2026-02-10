# CellMap Segmentation Challenge - Summary

## Overview

The **CellMap Segmentation Challenge** is a comprehensive benchmark for multi-organelle segmentation in electron microscopy (EM) images. Organized by Janelia's CellMap Project Team, it provides a large-scale dataset with detailed annotations across diverse biological samples.

**Repository**: https://github.com/janelia-cellmap/cellmap-segmentation-challenge
**Local Path**: `/projects/weilab/weidf/lib/seg/cellmap-segmentation-challenge`
**Data Path**: `/projects/weilab/dataset/cellmap`

---

## Dataset Structure

### Organization
The dataset contains **23 datasets** from different biological samples:
- Cell lines: `jrc_cos7-1a/1b`, `jrc_hela-2/3`, `jrc_jurkat-1`, `jrc_sum159-1/4`, `jrc_macrophage-2`
- Tissue samples: `jrc_mus-kidney`, `jrc_mus-liver`, `jrc_mus-heart-1`, `jrc_mus-nacc-1`
- Model organisms: `jrc_fly-mb-1a`, `jrc_fly-vnc-1`, `jrc_zf-cardiac-1`
- Other: `jrc_ctl-id8-1`, `jrc_ut21-1413-003`

### Data Format
Each dataset is stored in **Zarr v2** format with the following structure:

```
/projects/weilab/dataset/cellmap/{dataset}/
└── {dataset}.zarr/
    └── recon-1/
        ├── em/
        │   └── fibsem-uint8/          # Raw EM data
        │       ├── s0/                 # Highest resolution (typically 2nm isotropic)
        │       ├── s1/                 # 2x downsampled
        │       ├── s2/                 # 4x downsampled
        │       └── ...                 # Multi-scale pyramid (s0-s10)
        └── labels/
            └── groundtruth/
                ├── crop234/            # Annotated crop/ROI
                │   ├── all/            # All annotations combined
                │   ├── nuc/            # Nucleus (semantic)
                │   ├── mito/           # Mitochondria (instance)
                │   ├── er/             # Endoplasmic reticulum (instance)
                │   ├── golgi/          # Golgi apparatus (instance)
                │   ├── ves/            # Vesicles (instance)
                │   ├── endo/           # Endosomes (instance)
                │   ├── lyso/           # Lysosomes (instance)
                │   ├── ld/             # Lipid droplets (instance)
                │   ├── perox/          # Peroxisomes (instance)
                │   └── ...             # 60+ organelle classes
                ├── crop236/
                └── ...
```

### Annotation Details

**Total Classes**: 60+ organelle/structure classes including:
- **Semantic classes**: nucleus (nuc), nucleoplasm (nucpl), cytoplasm (cyto), extracellular space (ecs)
- **Instance classes**: mitochondria (mito), ER (er), Golgi (golgi), vesicles (ves), endosomes (endo), lysosomes (lyso), lipid droplets (ld), peroxisomes (perox)
- **Hierarchical**: Many classes have substructures (e.g., `mito_mem`, `mito_lum`, `er_mem`, `er_lum`)

**Resolution**: Multi-scale annotations matching EM data
- Highest resolution: 2nm isotropic (s0) for most datasets
- Some datasets: 4nm, 8nm, or 16nm depending on label class
- Multi-scale pyramid with consistent metadata

**Crops**: Each dataset contains multiple annotated 3D crops (ROIs)
- Crop sizes vary (typically 128³ to 512³ voxels at annotation resolution)
- Not all crops contain all label classes
- Background (ID=0) indicates unannotated regions

---

## CellMap Toolbox Architecture

### Package Structure

```
cellmap-segmentation-challenge/
├── src/cellmap_segmentation_challenge/
│   ├── config.py                      # Global configuration (paths, constants)
│   ├── train.py                       # Training pipeline (530 lines)
│   ├── predict.py                     # Inference pipeline (12KB)
│   ├── process.py                     # Post-processing pipeline
│   ├── evaluate.py                    # Evaluation metrics (41KB)
│   ├── visualize.py                   # Neuroglancer visualization
│   │
│   ├── models/
│   │   ├── unet_model_2D.py           # 2D U-Net (4.3KB)
│   │   ├── unet_model_3D.py           # 3D U-Net (4.4KB)
│   │   ├── resnet.py                  # 2D/3D ResNet (16KB)
│   │   ├── vitnet.py                  # Vision Transformer VNet (15KB)
│   │   └── model_load.py              # Model loading utilities
│   │
│   ├── utils/
│   │   ├── datasplit.py               # Dataset splitting
│   │   ├── dataloader.py              # CellMap dataloader
│   │   └── ...
│   │
│   └── cli/
│       ├── fetch_data.py              # Data download CLI
│       ├── train.py                   # Training CLI
│       ├── process.py                 # Processing CLI
│       └── ...
│
├── examples/
│   ├── train_2D.py                    # 2D training example
│   ├── train_3D.py                    # 3D training example
│   ├── predict_2D.py                  # 2D prediction example
│   ├── predict_3D.py                  # 3D prediction example
│   ├── process_2D.py                  # 2D post-processing
│   └── process_3D.py                  # 3D post-processing
│
└── data/                              # Default data directory (symlink to /projects/weilab/dataset/cellmap)
```

### Key Features

**1. Training Pipeline** (`train.py`)
- Python config files (not YAML) - executable scripts with parameters
- Automatic datasplit generation from discovered crops
- Built on `cellmap-data` package for Zarr I/O
- PyTorch-based with TensorBoard logging
- Supports 2D and 3D models
- Gradient accumulation, mixed precision ready
- Validation time/batch limits
- S3 streaming support (optional)

**2. Data Loading** (`cellmap-data` package)
- Lazy loading from Zarr stores
- Multi-scale support (automatically selects resolution)
- Spatial augmentations (mirror, transpose, rotate)
- Intensity transforms (normalize, binarize)
- Class weighting and balanced sampling
- Mutual exclusion for unlabeled pixels

**3. Models Included**
- **2D U-Net**: Simple encoder-decoder (4 levels)
- **3D U-Net**: 3D encoder-decoder (4 levels)
- **2D/3D ResNet**: ResNet-based encoder (up to 152 layers)
- **ViTVNet**: Vision Transformer with V-Net decoder (MONAI-based)

**4. Configuration System**
- **Python files** as configs (not YAML)
- Variables define hyperparameters directly
- String formatting with `{model_name}`, `{epoch}` placeholders
- Getattr pattern with defaults for robustness

### Example Training Config

```python
# From examples/train_3D.py
learning_rate = 0.0001
batch_size = 8
input_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),  # voxel size in nm
}
target_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}
epochs = 1000
iterations_per_epoch = 1000
classes = ["nuc", "er", "mito"]  # List of classes to train

# Model definition
from cellmap_segmentation_challenge.models import UNet_3D
model = UNet_3D(1, len(classes))

# Spatial augmentations
spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.1}},
    "transpose": {"axes": ["x", "y", "z"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180], "z": [-180, 180]}},
}

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)
```

---

## Dependencies

### Core Dependencies
```python
torch                    # PyTorch
tensorboard              # Logging
tensorboardX             # Extended TensorBoard
cellmap-data             # Zarr I/O and dataset utilities
zarr < 3.0.0             # Zarr v2 format
numcodecs < 0.16.0       # Compression codecs
scipy                    # Scientific computing
numpy                    # Numerical arrays
tensorstore              # Multi-dimensional array I/O
neuroglancer             # 3D visualization
scikit-learn             # ML utilities
scikit-image             # Image processing
boto3                    # S3 access (optional)
cellmap-flow             # Processing workflows
```

### Key Differences from PyTC
1. **No PyTorch Lightning** - Custom training loop
2. **No Hydra/OmegaConf** - Python config files
3. **No MONAI models** - Custom U-Net, ResNet, ViT implementations
4. **Zarr-focused** - No HDF5 support
5. **cellmap-data package** - Specialized Zarr dataloader

---

## CLI Commands

```bash
# Download data
csc fetch-data                                    # Download default data
csc fetch-data --raw-padding 128                  # With padding
csc fetch-data --fetch-all-em-resolutions         # All EM scales

# Training
python examples/train_3D.py                       # Direct execution
csc train examples/train_3D.py                    # Via CLI

# Prediction
csc predict examples/train_3D.py                  # Use training config
csc predict examples/train_3D.py --crops 234,236  # Specific crops

# Post-processing
csc process examples/process_3D.py

# Visualization
csc visualize                                     # All data
csc visualize -d jrc_cos7-1a -c 234,236          # Specific dataset/crops

# Submission
csc pack-results                                  # Package predictions
```

---

## Challenge Details

### Task Types
1. **Semantic Segmentation**: Binary masks per class (e.g., nucleus, cytoplasm)
2. **Instance Segmentation**: Unique IDs per object (e.g., individual mitochondria)

### Evaluation Metrics
- **Semantic**: Dice score, IoU, precision, recall
- **Instance**: Adapted Rand Error, VOI split/merge, panoptic quality

### Submission Format
- Zarr v2 format with specific directory structure
- Predictions at groundtruth resolution and ROI
- Separate arrays per class
- Connected components applied during evaluation (for instance classes)

---

## Key Observations

### Strengths
1. **Large-scale dataset**: 23 diverse datasets, 60+ classes
2. **Production-ready toolbox**: Complete train/predict/evaluate pipeline
3. **Flexible**: Python configs, custom models supported
4. **Multi-scale**: Pyramid structure for efficient training
5. **Cloud-ready**: S3 streaming support

### Limitations
1. **Non-standard config**: Python files instead of YAML/Hydra
2. **Custom implementations**: Not leveraging MONAI/Lightning ecosystems
3. **Limited model zoo**: Only 4 baseline architectures
4. **Documentation**: Some aspects under-documented (class mappings, evaluation details)

### Compatibility with PyTC
- **Data format**: Zarr vs HDF5 (both supported by PyTC via MONAI)
- **Config system**: Python files vs Hydra YAML
- **Training**: Custom loop vs Lightning
- **Models**: Custom vs MONAI registry
- **Augmentations**: cellmap-data transforms vs MONAI transforms

---

## Related Documentation

- [Challenge Homepage](https://janelia.figshare.com/articles/online_resource/CellMap_Segmentation_Challenge/28034561)
- [Documentation](https://janelia-cellmap.github.io/cellmap-segmentation-challenge/)
- [Submission Portal](https://cellmapchallenge.janelia.org/submissions/)
- [GitHub Discussions](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/discussions)
- [FAQ](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/blob/main/FAQ.md)

---

## Next Steps

See [CELLMAP_INTEGRATION_DESIGN.md](./CELLMAP_INTEGRATION_DESIGN.md) for integration plan with PyTorch Connectomics.
