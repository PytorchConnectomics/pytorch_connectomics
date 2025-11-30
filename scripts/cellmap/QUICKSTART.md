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

## 2. Quick Test (20 minutes)

Test the full pipeline with a lightweight model:

```bash
# Train for 10 epochs (~20 minutes)
python scripts/cellmap/train_cellmap.py \
    scripts/cellmap/configs/monai_unet_quick.py

# Monitor training (in another terminal)
tensorboard --logdir outputs/cellmap_quick_test/tensorboard
```

---

## 3. Full Training (2-3 days)

Train production model on COS7 multi-organelle segmentation:

```bash
# Train MedNeXt (Medium) for 500 epochs
python scripts/cellmap/train_cellmap.py \
    scripts/cellmap/configs/mednext_cos7.py

# Expected time on 1x A100: ~42 hours
# Expected time on 4x A100: ~17 hours
```

---

## 4. Inference (1-2 hours)

```bash
# Predict on test crops
python scripts/cellmap/predict_cellmap.py \
    --checkpoint outputs/cellmap_cos7/checkpoints/mednext-epoch=XX-val_dice=0.XXX.ckpt \
    --config scripts/cellmap/configs/mednext_cos7.py \
    --output outputs/cellmap_cos7/predictions
```

---

## 5. Submit (10 minutes)

```bash
# Package predictions
python scripts/cellmap/submit_cellmap.py \
    --predictions outputs/cellmap_cos7/predictions \
    --output submission.zarr

# Upload submission.zip to:
# https://cellmapchallenge.janelia.org/submissions/
```

---

## File Tree

```
scripts/cellmap/
├── README.md                  # Full documentation
├── QUICKSTART.md             # This file
│
├── train_cellmap.py          # Training script
├── predict_cellmap.py        # Inference script
├── submit_cellmap.py         # Submission script
│
└── configs/
    ├── mednext_cos7.py       # Multi-organelle (RECOMMENDED)
    ├── mednext_mito.py       # Mitochondria only
    └── monai_unet_quick.py   # Quick test (10 epochs)
```

---

## Available Configs

| Config | Classes | Resolution | Epochs | Time (1x A100) | Best For |
|--------|---------|-----------|--------|---------------|----------|
| `monai_unet_quick.py` | nuc, mito | 8nm | 10 | ~20 min | **Quick test** |
| `mednext_cos7.py` | nuc, mito, er, golgi, ves | 8nm | 500 | ~42 hours | **Multi-organelle** |
| `mednext_mito.py` | mito | 4nm | 1000 | ~133 hours | **Best quality** |

---

## What You Get

✅ **Zero PyTC modifications** - Completely isolated
✅ **Official CellMap tools** - Guaranteed compatibility
✅ **PyTC model zoo** - 8+ MONAI architectures
✅ **Production ready** - Lightning + callbacks + logging
✅ **Easy to use** - Just run 3 commands

---

## Troubleshooting

### Import Error

```bash
# Error: No module named 'cellmap_data'
pip install cellmap-data cellmap-segmentation-challenge
```

### CUDA Out of Memory

```python
# Edit config file:
batch_size = 1              # Reduce from 2 to 1
mednext_size = 'S'          # Use smaller model (S instead of M)
```

### Data Not Found

```bash
# Check data location
ls /projects/weilab/dataset/cellmap/

# Should see: jrc_cos7-1a, jrc_hela-2, etc.
```

---

## Next Steps

1. ✅ Run quick test (`monai_unet_quick.py`)
2. ✅ Run full training (`mednext_cos7.py`)
3. ✅ Predict on test set
4. ✅ Submit to challenge
5. 📊 Check leaderboard!

---

## Help

- **Full documentation**: [README.md](README.md)
- **CellMap challenge**: https://github.com/janelia-cellmap/cellmap-segmentation-challenge
- **PyTC docs**: [../../CLAUDE.md](../../CLAUDE.md)

---

**Time to first results**: 20 minutes (quick test)
**Time to submission**: ~3 days (full training + inference)

Let's go! 🚀
