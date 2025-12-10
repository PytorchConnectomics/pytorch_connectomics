# CellMap Challenge - 5-Minute Quickstart

Get started with CellMap challenge in 5 minutes!

---

## 1. Install (30 seconds)

```bash
# Activate PyTC environment
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc

# Install CellMap packages
pip install cellmap-data cellmap-segmentation-challenge
```

---

## 2. Quick Test (2 minutes)

Test the full pipeline with fast-dev-run:

```bash
# Test with 1 batch (very fast, just to verify setup)
python scripts/cellmap/train_cellmap.py \
    --config tutorials/cellmap_cos7.yaml \
    --fast-dev-run

# Monitor training (in another terminal)
tensorboard --logdir outputs/cellmap_cos7/tensorboard
```

---

## 3. Full Training (8-12 hours)

Train production model on COS7 multi-organelle segmentation:

```bash
# Train MedNeXt-M for 500 epochs
python scripts/cellmap/train_cellmap.py \
    --config tutorials/cellmap_cos7.yaml

# Expected time on 1x A100: ~12 hours
# Expected time on 4x A100: ~3-4 hours
```

---

## 4. Phase 2 Option A: Simple Unified (47 Semantic)

For fast submission - all 47 classes as semantic:

```bash
# Train MedNeXt-L for all 47 classes as semantic (8nm resolution)
python scripts/cellmap/train_cellmap.py \
    --config tutorials/cellmap_semantic47.yaml

# Expected time on 4x A100: ~30 hours
# Output: Multi-class binary mask (47 channels)
# No SDT, no instance separation
# Simple unified approach - faster but lower accuracy for instance classes
```

---

## 5. Phase 2 Option B: Combination (36 Semantic + 11 Instance)

For best accuracy - specialized models with SDT:

```bash
# Part 1: Train semantic model (36 classes)
python scripts/cellmap/train_cellmap.py \
    --config tutorials/cellmap_semantic_full.yaml

# Expected time on 4x A100: ~24 hours
# Output: Multi-class binary mask (softmax)
# No post-processing needed

# Part 2: Train instance model (11 classes with SDT)
python scripts/cellmap/train_cellmap.py \
    --config tutorials/cellmap_instance_full.yaml

# Expected time on 4x A100: ~36 hours
# Output: Multi-class binary mask (11 channels, sigmoid)
# Post-processing: Binary → SDT → Watershed → Instance IDs
# Total time: ~60 hours for both models
```

---

## 6. Inference (1-2 hours)

```bash
# Predict on test crops
python scripts/cellmap/predict_cellmap.py \
    --checkpoint outputs/cellmap_cos7/checkpoints/last.ckpt \
    --config tutorials/cellmap_cos7.yaml \
    --output predictions/
```

---

## 7. Submit (10 minutes)

```bash
# Package predictions
python scripts/cellmap/submit_cellmap.py \
    --predictions predictions/ \
    --output submission.zarr

# Upload submission.zarr to:
# https://cellmapchallenge.janelia.org/submissions/
```

---

## File Tree

```
scripts/cellmap/
├── README.md                  # Full documentation
├── QUICKSTART.md             # This file
│
├── train_cellmap.py          # Training script (258 lines, Hydra-based)
├── predict_cellmap.py        # Inference script
├── submit_cellmap.py         # Submission script
│
tutorials/
├── cellmap_cos7.yaml         # Multi-organelle config (Hydra YAML)
└── cellmap_mito.yaml         # Mitochondria config (Hydra YAML)
```

---

## Development Plan

### Phase 1: Make Sure Simple One Works ✅
| Config | Classes | Type | Time (4x A100) |
|--------|---------|------|---------------|
| `cellmap_mito.yaml` | 1 (mito) | Instance + SDT | ~6 hours |
| `cellmap_cos7.yaml` | 5 (nuc, mito, er, golgi, ves) | Semantic | ~3 hours |

### Phase 2: Full Submission 🚀

**Option A: Simple Unified (47 Semantic)**
| Config | Classes | Type | Time (4x A100) |
|--------|---------|------|---------------|
| `cellmap_semantic47.yaml` | **All 47 as semantic** | Multi-class binary mask | ~30 hours |

**Option B: Combination Approach (Best Accuracy)**
| Config | Classes | Type | Time (4x A100) |
|--------|---------|------|---------------|
| `cellmap_semantic_full.yaml` | **36 semantic** | Multi-class binary mask | ~24 hours |
| `cellmap_instance_full.yaml` | **11 instance + SDT** | Binary mask + SDT | ~36 hours |
| **Total** | **47 classes** | **Combined** | **~60 hours** |

### Key Differences

**Phase 1 Configs (Validation):**
- **`cellmap_mito.yaml`**: 1 instance class (mito)
  - Binary mask output + SDT post-processing
  - Validates instance pipeline: Binary → SDT → Watershed → Instance IDs

- **`cellmap_cos7.yaml`**: 5 semantic classes
  - Multi-class binary mask (softmax activation)
  - Validates semantic pipeline: Direct output, no post-processing

**Phase 2 Option A (Simple Unified):**
- **`cellmap_semantic47.yaml`**: All 47 classes as semantic
  - Treats everything as multi-class binary mask
  - No SDT, no instance separation
  - Simpler pipeline, faster training (30h vs 60h)
  - Lower accuracy for instance classes

**Phase 2 Option B (Combination - Best Accuracy):**
- **`cellmap_semantic_full.yaml`**: 36 semantic classes
  - Multi-class binary mask (softmax)
  - Direct output, no post-processing
  - 8nm resolution, 1000 epochs, ~24h

- **`cellmap_instance_full.yaml`**: 11 instance classes
  - Multi-class binary mask (11 channels, sigmoid)
  - SDT post-processing: Binary → SDT → Watershed → Instance IDs
  - 4nm resolution, 1500 epochs, ~36h
  - Critical: `mito` appears in 14/16 test crops

**Recommendation:**
- **Start with Phase 1** to validate pipelines
- **Phase 2 Option A** for quick submission (30h)
- **Phase 2 Option B** for best leaderboard score (60h)

---

## Config Override Examples

```bash
# Quick test with smaller model
python scripts/cellmap/train_cellmap.py \
    --config tutorials/cellmap_cos7.yaml \
    model.architecture=monai_basic_unet3d \
    model.input_size="[64, 64, 64]" \
    optimization.max_epochs=100

# Multi-GPU training
python scripts/cellmap/train_cellmap.py \
    --config tutorials/cellmap_cos7.yaml \
    system.training.num_gpus=4

# Lower batch size for GPU memory
python scripts/cellmap/train_cellmap.py \
    --config tutorials/cellmap_cos7.yaml \
    system.training.batch_size=1
```

---

## What You Get

✅ **Zero PyTC modifications** - Completely isolated
✅ **Official CellMap tools** - Guaranteed compatibility
✅ **PyTC model zoo** - 8+ MONAI architectures
✅ **Hydra configs** - Standard PyTC config format
✅ **Production ready** - Lightning + callbacks + logging
✅ **Easy to use** - Just run 3 commands

---

## Architecture Comparison

The new design is **much simpler**:

**Before (Python configs)**:
- 273 lines, custom LightningModule
- Custom loss wrappers, optimizer setup
- Python-based configuration files

**After (Hydra configs)**:
- 258 lines, reuses PyTC's `ConnectomicsModule`
- Reuses all `scripts/main.py` infrastructure
- Standard Hydra YAML configs
- Only custom: `CellMapDataModule` (60 lines)

**Code reuse**:
```python
from connectomics.training.lit import (
    ConnectomicsModule,      # Model wrapper
    create_trainer,          # Trainer setup
    setup_config,            # Config loading
    # ... everything from main.py
)
```

---

## Troubleshooting

### Import Error

```bash
# Error: No module named 'cellmap_data'
pip install cellmap-data cellmap-segmentation-challenge
```

### CUDA Out of Memory

```bash
# Reduce batch size or patch size
python scripts/cellmap/train_cellmap.py \
    --config tutorials/cellmap_cos7.yaml \
    system.training.batch_size=1 \
    model.input_size="[96, 96, 96]"
```

### Data Not Found

```bash
# Check data location
ls /projects/weilab/dataset/cellmap/

# Should see: jrc_cos7-1a, jrc_hela-2, etc.
```

---

## Next Steps

1. ✅ Run quick test (`--fast-dev-run`)
2. ✅ Run full training (`cellmap_cos7.yaml`)
3. ✅ Optional: Train mito-specific model (`cellmap_mito.yaml`)
4. ✅ Predict on test set
5. ✅ Submit to challenge
6. 📊 Check leaderboard!

---

## Help

- **Full documentation**: [README.md](README.md)
- **Instance segmentation guide**: [.claude/CELLMAP_SUBMISSION.md](../../.claude/CELLMAP_SUBMISSION.md)
- **CellMap challenge**: https://www.cellmapchallenge.janelia.org/
- **PyTC docs**: [../../CLAUDE.md](../../CLAUDE.md)

---

**Time to first results**: 2 minutes (fast-dev-run)
**Time to submission**: ~12 hours (full training + inference)

Let's go! 🚀
