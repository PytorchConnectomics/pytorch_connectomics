# CellMap Segmentation Challenge - PyTC Integration

This directory provides a lightweight integration between the [CellMap Segmentation Challenge](https://www.cellmapchallenge.janelia.org/) and PyTorch Connectomics (PyTC).

## 🎯 Key Features

- **Zero PyTC modifications** - All code isolated in `scripts/cellmap/`
- **Official CellMap tools** - Uses `cellmap-data` package for data loading
- **Hydra YAML configs** - Standard PyTC config format (no Python configs needed)
- **Full PyTC features** - Lightning callbacks, checkpointing, logging, TTA, etc.
- **419 test predictions** - Complete coverage for challenge submission

## 📦 Installation

```bash
# 1. Activate PyTC environment
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc

# 2. Install CellMap packages (official challenge tools)
pip install cellmap-data cellmap-segmentation-challenge

# 3. Verify installation
python -c "from cellmap_segmentation_challenge.utils import get_dataloader; print('✅ CellMap installed')"
```

## 🚀 Quick Start (5 minutes)

```bash
# Train on COS7 multi-organelle (5 classes)
python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_cos7.yaml --fast-dev-run

# Full training (8-12 hours on 1 GPU)
python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_cos7.yaml

# Mitochondria-specific (optimized for instance segmentation)
python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_mito.yaml
```

## 📋 Configuration Files

### Hydra YAML Configs (Recommended)

All configs use PyTC's standard Hydra format:

**[tutorials/cellmap_cos7.yaml](../../tutorials/cellmap_cos7.yaml)** - Multi-organelle segmentation
- **Classes**: nuc, mito, er, golgi, ves (5 organelles)
- **Model**: MedNeXt-M (17.6M params)
- **Resolution**: 8nm isotropic
- **Training**: 500 epochs, ~12 hours on 1 GPU
- **Use case**: General-purpose baseline

**[tutorials/cellmap_mito.yaml](../../tutorials/cellmap_mito.yaml)** - Mitochondria-specific
- **Classes**: mito (single class)
- **Model**: MedNeXt-L (61.8M params) - best for single class
- **Resolution**: 4nm isotropic - higher res for boundaries
- **Training**: 1000 epochs, ~24 hours on 1 GPU
- **Use case**: Best instance segmentation quality

### Key Config Sections

```yaml
# CellMap-specific data configuration
data:
  dataset_type: cellmap              # Special marker for CellMap

  cellmap:
    data_root: /projects/weilab/dataset/cellmap
    datasplit_path: outputs/cellmap_cos7/datasplit.csv  # Auto-generated
    classes: [nuc, mito, er, golgi, ves]
    force_all_classes: both

    # Patch configuration
    input_array_info:
      shape: [128, 128, 128]
      scale: [8, 8, 8]                # 8nm isotropic

    # CellMap-style augmentation
    spatial_transforms:
      mirror: {axes: {x: 0.5, y: 0.5, z: 0.5}}
      transpose: {axes: [x, y, z]}
      rotate: {axes: {x: [-180, 180], y: [-180, 180], z: [-180, 180]}}
```

## 📊 Available Datasets

CellMap challenge provides 23 datasets with 60+ organelle classes:

```bash
# List all available datasets
ls /projects/weilab/dataset/cellmap/

# Example datasets:
# - jrc_cos7-1a, jrc_cos7-1b     : COS7 cells
# - jrc_hela-2, jrc_hela-3       : HeLa cells
# - jrc_jurkat-1                 : Jurkat cells
# - jrc_macrophage-2             : Macrophages
# - jrc_mus-liver, jrc_mus-kidney: Mouse organs
```

## 🎓 Training Workflow

### 1. Data Preparation (Automatic)

The datasplit is automatically generated on first run:

```bash
python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_cos7.yaml
# Generates: outputs/cellmap_cos7/datasplit.csv
```

The datasplit includes:
- Train/validation split
- Crop coordinates
- Class availability per crop
- Uses CellMap's official `make_datasplit_csv()`

### 2. Training

```bash
# Standard training
python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_cos7.yaml

# Override config from CLI
python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_cos7.yaml \
    system.training.num_gpus=4 \
    optimization.max_epochs=1000

# Multi-GPU training (automatic DDP)
python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_cos7.yaml \
    system.training.num_gpus=4
```

### 3. Inference (Challenge Submission)

For challenge submission, use the official CellMap prediction scripts:

```bash
# 1. Run inference on all test crops (uses sliding window + TTA)
python scripts/cellmap/predict_cellmap.py \
    --checkpoint outputs/cellmap_cos7/checkpoints/last.ckpt \
    --config tutorials/cellmap_cos7.yaml \
    --output predictions/

# 2. Package predictions for submission
python scripts/cellmap/submit_cellmap.py \
    --predictions predictions/ \
    --output submission.zarr

# 3. Upload submission.zarr to challenge platform
```

## 📁 File Structure

```
scripts/cellmap/
├── train_cellmap.py          # Training script (258 lines)
├── predict_cellmap.py         # Inference script (for challenge submission)
├── submit_cellmap.py          # Submission packaging (uses official tool)
├── README.md                  # This file
└── QUICKSTART.md              # 5-minute quickstart guide

tutorials/
├── cellmap_cos7.yaml          # Multi-organelle config (Hydra format)
└── cellmap_mito.yaml          # Mitochondria-specific config (Hydra format)
```

## 🔧 How It Works

### Simplified Architecture

The new design is **much simpler** than the original:

**New approach** (258 lines, Hydra configs):
- Reuses `ConnectomicsModule` from PyTC
- Reuses `create_trainer()` from PyTC
- Reuses all callbacks, logging, checkpointing
- Only custom component: `CellMapDataModule` (60 lines)

### Code Reuse from scripts/main.py

`train_cellmap.py` reuses almost everything from `scripts/main.py`:

```python
from connectomics.training.lit import (
    ConnectomicsModule,      # Model wrapper
    create_trainer,          # Trainer setup
    setup_config,            # Config loading
    setup_run_directory,     # Directory management
    modify_checkpoint_state, # Checkpoint handling
    # ... and more
)
```

**Only custom component**: `CellMapDataModule` (60 lines)
- Wraps CellMap's `get_dataloader()` in Lightning interface
- Auto-generates datasplit if missing
- Handles train/val/test splits

## 🎯 Challenge Details

### Task Statistics

- **Total predictions**: 419 predictions
- **Test crops**: 16 crops across 6 datasets
- **Classes**: 47 classes (11 instance + 36 semantic)

### Instance vs Semantic Segmentation

**Instance Segmentation** (11 classes - harder):
- nuc, mito, ves, endo, lyso, ld, perox, np, mt, cell, vim
- Requires unique IDs per object
- Evaluated with Adapted Rand Error, VOI
- **Server auto-runs connected components** on submission

**Semantic Segmentation** (36 classes - easier):
- All membrane/lumen subclasses, cytoplasm, etc.
- Binary masks (0/1)
- Evaluated with Dice, IoU

See [.claude/CELLMAP_SUBMISSION.md](../../.claude/CELLMAP_SUBMISSION.md) for detailed guide.

## 💡 Tips & Best Practices

### Model Selection

**Multi-class segmentation** (5+ classes):
- Use MedNeXt-M or MedNeXt-B (good balance)
- 8nm resolution is sufficient
- 500 epochs usually enough

**Single-class segmentation** (e.g., mitochondria):
- Use MedNeXt-L (61.8M params) for best quality
- Higher resolution (4nm) for better boundaries
- Extended training (1000 epochs)
- Critical for instance segmentation

### Training Strategy

1. **Quick baseline** (1-2 hours):
   ```bash
   # Test with MONAI BasicUNet on small patch size
   python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_cos7.yaml \
       model.architecture=monai_basic_unet3d \
       model.input_size="[64, 64, 64]" \
       optimization.max_epochs=100
   ```

2. **Production training** (8-12 hours):
   ```bash
   # Full MedNeXt training
   python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_cos7.yaml
   ```

3. **Mitochondria optimization** (24 hours):
   ```bash
   # Mito-specific config for best instance segmentation
   python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_mito.yaml
   ```

### Instance Segmentation Quality

For best instance segmentation results:

1. **Binary masks are sufficient** for initial submission
   - Server automatically runs connected components
   - Focus on clean boundaries

2. **Optional: Watershed post-processing** for better quality
   - Implement in `predict_cellmap.py`
   - See `.claude/CELLMAP_SUBMISSION.md` for code examples

3. **Mitochondria is the hardest task**
   - Use dedicated config (`tutorials/cellmap_mito.yaml`)
   - Higher resolution (4nm)
   - Larger model (MedNeXt-L)
   - Extended training (1000 epochs)

## 📚 Additional Resources

**Documentation:**
- [QUICKSTART.md](QUICKSTART.md) - 5-minute quickstart
- [.claude/CELLMAP_SUBMISSION.md](../../.claude/CELLMAP_SUBMISSION.md) - Instance segmentation guide
- [.claude/CELLMAP_CHALLENGE_SUMMARY.md](../../.claude/CELLMAP_CHALLENGE_SUMMARY.md) - Challenge overview
- [.claude/CELLMAP_INTEGRATION_DESIGN_V2.md](../../.claude/CELLMAP_INTEGRATION_DESIGN_V2.md) - Design decisions

**Challenge Links:**
- [Challenge Website](https://www.cellmapchallenge.janelia.org/)
- [CellMap Data Package](https://github.com/janelia-cellmap/cellmap-data)
- [Challenge Utils](https://github.com/janelia-cellmap/cellmap-segmentation-challenge)

**PyTC Documentation:**
- [Main README](../../README.md)
- [CLAUDE.md](../../CLAUDE.md) - PyTC architecture guide
- [PyTC Models](../../connectomics/models/arch/) - Available architectures

## 🐛 Troubleshooting

### Import Error: cellmap-data not found

```bash
pip install cellmap-data cellmap-segmentation-challenge
```

### Datasplit generation fails

```bash
# Check data exists
ls /projects/weilab/dataset/cellmap/jrc_cos7-1a/

# Manually generate datasplit
python -c "
from cellmap_segmentation_challenge.utils import make_datasplit_csv
make_datasplit_csv(
    csv_path='outputs/cellmap_cos7/datasplit.csv',
    raw_path='/projects/weilab/dataset/cellmap',
    classes=['nuc', 'mito', 'er', 'golgi', 'ves'],
)
"
```

### GPU out of memory

Reduce batch size or patch size:
```bash
python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_cos7.yaml \
    system.training.batch_size=1 \
    model.input_size="[96, 96, 96]"
```

### Training too slow

Increase workers or use multiple GPUs:
```bash
python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_cos7.yaml \
    system.training.num_workers=8 \
    system.training.num_gpus=4
```

## 🙋 Getting Help

1. Check [TROUBLESHOOTING.md](../../TROUBLESHOOTING.md)
2. Review [.claude/CELLMAP_*.md](../../.claude/) documentation
3. Open an issue on [PyTC GitHub](https://github.com/zudi-lin/pytorch_connectomics/issues)
4. Join [PyTC Slack](https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w)
