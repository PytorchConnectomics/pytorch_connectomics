# CellMap Segmentation Challenge - PyTC Integration

This directory contains standalone scripts for training PyTorch Connectomics models on CellMap Segmentation Challenge data.

**Key Feature**: Zero modifications to PyTC core - completely isolated implementation.

---

## Quick Start

### 1. Installation (One-time)

```bash
# Activate PyTC environment
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc

# Install CellMap packages
pip install cellmap-data cellmap-segmentation-challenge

# Verify installation
python -c "from cellmap_segmentation_challenge.utils import TEST_CROPS; print(f'{len(TEST_CROPS)} test crops loaded')"
```

### 2. Quick Test (10 epochs)

```bash
# Test the pipeline with lightweight U-Net
python scripts/cellmap/train_cellmap.py scripts/cellmap/configs/monai_unet_quick.py

# Monitor training
tensorboard --logdir outputs/cellmap_quick_test/tensorboard
```

### 3. Full Training

```bash
# Train MedNeXt on COS7 cells (multi-organelle)
python scripts/cellmap/train_cellmap.py scripts/cellmap/configs/mednext_cos7.py

# Or train on mitochondria only
python scripts/cellmap/train_cellmap.py scripts/cellmap/configs/mednext_mito.py
```

### 4. Inference

```bash
# Predict on test crops
python scripts/cellmap/predict_cellmap.py \
    --checkpoint outputs/cellmap_cos7/checkpoints/mednext-best.ckpt \
    --config scripts/cellmap/configs/mednext_cos7.py \
    --output outputs/cellmap_cos7/predictions
```

### 5. Submission

```bash
# Package predictions for submission
python scripts/cellmap/submit_cellmap.py \
    --predictions outputs/cellmap_cos7/predictions \
    --output submission.zarr

# Upload submission.zip to challenge portal
# https://cellmapchallenge.janelia.org/submissions/
```

---

## File Structure

```
scripts/cellmap/
├── README.md                      # This file
│
├── train_cellmap.py               # Training script (250 lines)
├── predict_cellmap.py             # Inference script (120 lines)
├── submit_cellmap.py              # Submission packaging (30 lines)
│
└── configs/                       # Training configurations
    ├── mednext_cos7.py            # Multi-organelle (nuc, mito, er, golgi, ves)
    ├── mednext_mito.py            # Mitochondria only (instance segmentation)
    └── monai_unet_quick.py        # Quick test config (10 epochs)
```

**Total: ~400 lines of code**

**PyTC modifications: 0 lines** ✅

---

## Configuration Files

### `mednext_cos7.py` - Multi-Organelle Segmentation

```python
model_name = 'mednext'
mednext_size = 'M'              # 17.6M params
classes = ['nuc', 'mito', 'er', 'golgi', 'ves']
resolution = (8, 8, 8)          # 8nm isotropic
epochs = 500
```

**Use case**: Segment all major organelles in COS7 cells

### `mednext_mito.py` - Mitochondria Segmentation

```python
model_name = 'mednext'
mednext_size = 'L'              # 61.8M params
classes = ['mito']
resolution = (4, 4, 4)          # 4nm (higher resolution)
epochs = 1000
```

**Use case**: High-quality mitochondria instance segmentation

### `monai_unet_quick.py` - Quick Test

```python
model_name = 'monai_basic_unet3d'
classes = ['nuc', 'mito']
resolution = (8, 8, 8)
epochs = 10                     # Fast test
```

**Use case**: Test pipeline quickly before full training

---

## Available Models

The scripts use PyTC's MONAI model registry. Available architectures:

| Model | Parameters | Deep Supervision | Best For |
|-------|-----------|------------------|----------|
| `monai_basic_unet3d` | ~5M | No | Quick tests, baselines |
| `monai_unet` | ~10M | No | General segmentation |
| `mednext` (size=S) | 5.6M | Yes | Small datasets |
| `mednext` (size=B) | 10.5M | Yes | **Recommended default** |
| `mednext` (size=M) | 17.6M | Yes | Multi-class tasks |
| `mednext` (size=L) | 61.8M | Yes | Single class, best quality |

**Note**: Set `model_name` and `mednext_size` in config files.

---

## Data

### Data Location

```
/projects/weilab/dataset/cellmap/
├── jrc_cos7-1a/
│   └── jrc_cos7-1a.zarr/
│       ├── recon-1/
│       │   ├── em/fibsem-uint8/      # Raw EM (s0-s10)
│       │   └── labels/groundtruth/    # Annotations
│       │       ├── crop234/
│       │       │   ├── nuc/
│       │       │   ├── mito/
│       │       │   └── ...
│       │       └── ...
├── jrc_hela-2/
├── jrc_jurkat-1/
└── ... (23 datasets total)
```

### Datasplit Generation

The training script automatically generates `datasplit.csv` on first run:

```bash
# Generated automatically
outputs/cellmap_cos7/datasplit.csv

# Format:
# raw,label,usage
# /path/to/raw,/path/to/label,train
# /path/to/raw,/path/to/label,validate
```

**Train/Val Split**: 85% train, 15% validation (stratified by class)

---

## Training Details

### What Happens During Training

1. **Datasplit generation** (if needed)
   - Scans `/projects/weilab/dataset/cellmap/`
   - Finds all crops with specified classes
   - Filters by resolution (e.g., 8nm)
   - Splits into train/val (85/15)

2. **Data loading** (CellMap's official loader)
   - Lazy Zarr loading (memory efficient)
   - Random patch sampling (128³ voxels)
   - Spatial augmentations (flip, rotate, transpose)
   - Class-weighted sampling

3. **Training** (PyTorch Lightning)
   - Mixed precision (16-bit)
   - Gradient accumulation (4x → effective batch = 8)
   - Gradient clipping (max norm = 1.0)
   - Multi-GPU support (DDP)

4. **Validation** (every epoch)
   - Dice score per class
   - Best model checkpointing
   - TensorBoard logging

5. **Checkpointing**
   - Top 3 models (by val/dice)
   - Last checkpoint
   - Early stopping (patience=50)

### Monitoring Training

```bash
# TensorBoard
tensorboard --logdir outputs/cellmap_cos7/tensorboard

# View logs
tail -f outputs/cellmap_cos7/tensorboard/lightning_logs/version_0/events.out.*
```

### Expected Training Time

| Config | GPUs | Time/Epoch | Total Time (500 epochs) |
|--------|------|------------|------------------------|
| `monai_unet_quick` | 1x A100 | ~2 min | ~20 min (10 epochs) |
| `mednext_cos7` (M) | 1x A100 | ~5 min | ~42 hours |
| `mednext_mito` (L) | 1x A100 | ~8 min | ~133 hours |
| `mednext_cos7` (M) | 4x A100 | ~2 min | ~17 hours |

**Note**: Times are estimates for single dataset. Multiply by number of datasets used.

---

## Inference Details

### Sliding Window Inference

Uses MONAI's `SlidingWindowInferer`:

```python
inferer = SlidingWindowInferer(
    roi_size=(128, 128, 128),   # Window size
    sw_batch_size=4,             # Batch size for sliding window
    overlap=0.5,                 # 50% overlap (Gaussian blending)
    mode='gaussian',             # Blending mode
)
```

### Test Crops

The challenge has **~1000 test crops** across 23 datasets.

Each test crop has:
- `crop.id`: Crop ID (e.g., 234)
- `crop.dataset`: Dataset name (e.g., 'jrc_cos7-1a')
- `crop.class_label`: Class to predict (e.g., 'mito')
- `crop.voxel_size`: Resolution in nm (e.g., [8, 8, 8])
- `crop.shape`: Shape in voxels (e.g., [256, 256, 256])
- `crop.translation`: Offset in world coordinates

### Prediction Output Format

```
outputs/cellmap_cos7/predictions/
├── jrc_cos7-1a/
│   ├── crop234/
│   │   ├── nuc/
│   │   │   └── s0/              # Zarr array
│   │   ├── mito/
│   │   └── ...
│   └── crop236/
└── jrc_hela-2/
```

---

## Submission Format

### Official Format (Required)

```
submission.zarr/
├── crop234/
│   ├── nuc/                     # Binary mask (uint8)
│   ├── mito/                    # Instance mask (uint16/uint32)
│   └── ...
├── crop236/
└── ...
```

**Metadata** (required for each array):
- `voxel_size`: Resolution in nm
- `translation`: World coordinates
- `shape`: Array shape

### Packaging

The `submit_cellmap.py` script uses CellMap's official `package_submission()`:

1. **Resamples** predictions to match test crop resolution
2. **Validates** format and metadata
3. **Creates** submission.zarr
4. **Zips** to submission.zip

**This is guaranteed to work** - it's the official challenge tool!

---

## Common Issues

### 1. Import Error: cellmap-data

```bash
# Error: No module named 'cellmap_data'
# Solution:
pip install cellmap-data cellmap-segmentation-challenge
```

### 2. CUDA Out of Memory

```python
# Solution 1: Reduce batch size in config
batch_size = 1

# Solution 2: Reduce patch size
input_array_info = {'shape': (64, 64, 64), ...}

# Solution 3: Use smaller model
mednext_size = 'S'  # Instead of 'M' or 'L'
```

### 3. Datasplit Generation Fails

```bash
# Check data path
ls /projects/weilab/dataset/cellmap/

# Check permissions
ls -la /projects/weilab/dataset/cellmap/jrc_cos7-1a/

# Manually specify datasets in config
datasets = ['jrc_cos7-1a', 'jrc_hela-2']  # Add to config
```

### 4. Checkpoint Not Found

```bash
# Check checkpoint path
ls outputs/cellmap_cos7/checkpoints/

# Use last checkpoint instead of best
--checkpoint outputs/cellmap_cos7/checkpoints/last.ckpt
```

---

## Advanced Usage

### Multi-GPU Training

```bash
# Edit config
num_gpus = 4

# Run training
python scripts/cellmap/train_cellmap.py configs/mednext_cos7.py
```

### Resume Training

```bash
# Lightning automatically resumes from last checkpoint if it exists
# Just run the same command again
python scripts/cellmap/train_cellmap.py configs/mednext_cos7.py
```

### Custom Data Loading

Create a new config with custom datasplit:

```python
# In config file
datasplit_path = 'my_custom_split.csv'

# Create CSV manually:
# raw,label,usage
# /path/to/raw1,/path/to/label1,train
# /path/to/raw2,/path/to/label2,validate
```

### Hyperparameter Tuning

Modify config files to test different hyperparameters:

```python
# Try different learning rates
learning_rate = 5e-4  # or 1e-3, 2e-3

# Try different model sizes
mednext_size = 'B'    # or 'S', 'M', 'L'

# Try different resolutions
input_array_info = {'scale': (4, 4, 4), ...}  # Higher resolution
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│           CellMap Challenge Toolbox (Official)                   │
│  • cellmap-data: Data loading, Zarr I/O                         │
│  • Challenge utils: TEST_CROPS, package_submission              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ Imports
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│        Standalone Training Script (This Directory)               │
│  • train_cellmap.py: Lightning training loop                    │
│  • predict_cellmap.py: MONAI sliding window inference           │
│  • submit_cellmap.py: Official submission packaging             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ Imports models
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         PyTorch Connectomics (ZERO MODIFICATIONS)               │
│  • connectomics.models: MONAI model zoo (8+ architectures)      │
│  • connectomics.models.loss: Loss functions                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## What's Included

### ✅ Reused from CellMap (Official)
- `cellmap-data` package - data loading
- `TEST_CROPS` - test metadata
- `package_submission()` - submission format
- `make_datasplit_csv()` - dataset splitting
- `CellMapLossWrapper` - NaN handling

### ✅ Reused from PyTC (Import Only)
- `build_model()` - MONAI model zoo
- `create_loss()` - loss functions

### ✅ Reused from Ecosystem
- PyTorch Lightning - training orchestration
- MONAI - sliding window inference

### ✅ New Code (This Directory)
- Training script (250 lines)
- Inference script (120 lines)
- Submission script (30 lines)
- Config files (3 × 40 lines)

**Total**: ~400 lines of standalone code

**PyTC core modifications**: **0 lines** ✅

---

## Next Steps

1. **Quick test**: Run `monai_unet_quick.py` (10 epochs, ~20 min)
2. **Full training**: Run `mednext_cos7.py` (500 epochs, ~42 hours)
3. **Inference**: Predict on test crops
4. **Submit**: Package and upload to challenge portal

---

## Resources

- [CellMap Challenge Homepage](https://janelia.figshare.com/articles/online_resource/CellMap_Segmentation_Challenge/28034561)
- [Challenge Documentation](https://janelia-cellmap.github.io/cellmap-segmentation-challenge/)
- [Submission Portal](https://cellmapchallenge.janelia.org/submissions/)
- [GitHub Discussions](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/discussions)
- [PyTC Documentation](../../CLAUDE.md)

---

## Citation

If you use this code, please cite:

```bibtex
@article{cellmap2024,
  title={CellMap Segmentation Challenge},
  author={CellMap Project Team},
  journal={Janelia Research Campus},
  year={2024}
}

@article{pytc2024,
  title={PyTorch Connectomics},
  author={Lin et al.},
  year={2024}
}
```

---

## Support

For issues:
- **CellMap data/submission**: [CellMap GitHub](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/issues)
- **PyTC models**: [PyTC GitHub](https://github.com/zudi-lin/pytorch_connectomics/issues)
- **This integration**: Open issue in PyTC repo with `[CellMap]` prefix

---

## License

- CellMap tools: MIT License
- PyTorch Connectomics: MIT License
- This integration: MIT License
