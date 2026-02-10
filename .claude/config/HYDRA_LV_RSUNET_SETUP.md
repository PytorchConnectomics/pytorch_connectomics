# Hydra Large Vesicle - RSUNet Configuration (Option 2)

## Configuration Summary

✅ **ACTIVATED: Option 2 - RSUNet Architecture**

### What Changed

1. **Architecture**: `monai_unet` → `rsunet`
2. **Experiment Name**: `hydra_lv_monai_unet` → `hydra_lv_rsunet`
3. **Output Paths**: Updated to `outputs/hydra_lv_rsunet/`

### Why RSUNet for Large Vesicles?

RSUNet is specifically optimized for electron microscopy (EM) data:

✅ **Anisotropic Convolutions**: Perfect for Hydra's 30nm (Z) × 8nm (XY) resolution
✅ **No Checkerboard Artifacts**: Uses upsample+conv instead of transposed convolutions
✅ **Better Gradient Flow**: Residual connections improve training stability
✅ **EM-Optimized**: Designed for connectomics segmentation tasks
✅ **Cleaner Boundaries**: Produces sharper instance segmentation results

### Expected Improvements

| Metric | Expected Gain |
|--------|---------------|
| **Segmentation Accuracy** | +2-3% |
| **Boundary Quality** | Significantly cleaner |
| **Training Stability** | Improved convergence |
| **Instance Separation** | Better for watershed |

---

## Quick Start Commands

### 1. Activate Environment
```bash
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
cd /projects/weilab/weidf/lib/pytorch_connectomics
```

### 2. Training

**Standard Training (4 GPUs):**
```bash
python scripts/main.py --config tutorials/hydra-lv.yaml
```

**Single GPU (for testing):**
```bash
python scripts/main.py --config tutorials/hydra-lv.yaml \
  system.training.num_gpus=1 \
  system.training.batch_size=8
```

**Fast Development Run (1 batch):**
```bash
python scripts/main.py --config tutorials/hydra-lv.yaml --fast-dev-run
```

### 3. Monitor Training
```bash
# In a separate terminal
tensorboard --logdir outputs/hydra_lv_rsunet/
```

### 4. Inference
```bash
python scripts/main.py --config tutorials/hydra-lv.yaml --mode test \
  --checkpoint outputs/hydra_lv_rsunet/checkpoints/last.ckpt
```

---

## Configuration Details

### Model Architecture (RSUNet)
```yaml
model:
  architecture: rsunet
  in_channels: 1
  out_channels: 3                      # Binary, boundary, distance
  filters: [32, 64, 128, 256]          # 4-level encoder
  dropout: 0.1
  rsunet_norm: batch                   # Batch normalization
```

### Multi-Task Learning
```yaml
multi_task_config:
  - [0, 1, "label", [0, 1]]           # Binary: Dice + BCE
  - [1, 2, "boundary", [0, 1]]        # Boundary: Dice + BCE
  - [2, 3, "edt", [2]]                # Distance: WeightedMSE
```

### Data Configuration
```yaml
data:
  train_image: "datasets/hydra-lv/vol*_im.h5"
  train_label: "datasets/hydra-lv/vol*_vesicle_ins.h5"
  train_mask: "datasets/hydra-lv/vol*_mask.h5"
  patch_size: [24, 96, 96]             # Anisotropic patches
  batch_size: 32
  use_preloaded_cache: true            # Fast loading
```

### Optimization
```yaml
optimizer:
  name: AdamW
  lr: 0.001
  weight_decay: 0.01

scheduler:
  name: ReduceLROnPlateau              # Adaptive LR reduction
  patience: 50
  factor: 0.5
```

### Augmentation
- **Geometric**: flip, rotate, affine
- **Intensity**: noise, shift, contrast
- **EM-specific**: misalignment, missing sections, motion blur

---

## What to Watch During Training

### Good Signs ✅
- Training loss steadily decreasing
- Loss values: < 0.5 (good), < 0.2 (excellent)
- No NaN/Inf values
- Smooth convergence curve
- Checkpoints being saved regularly

### Warning Signs ⚠️
- Loss oscillating wildly → reduce learning rate
- Loss not decreasing after 100 epochs → check data paths
- OOM errors → reduce batch_size or patch_size
- NaN/Inf loss → reduce learning rate, increase gradient clipping

---

## Expected Training Timeline

| Phase | Epochs | Expected Loss | Notes |
|-------|--------|---------------|-------|
| **Warmup** | 0-50 | 0.8 → 0.4 | Fast initial descent |
| **Main Training** | 50-200 | 0.4 → 0.15 | Steady improvement |
| **Fine-tuning** | 200-500 | 0.15 → 0.08 | Slow convergence |
| **Convergence** | 500+ | < 0.08 | Minimal improvement |

**Early stopping** will trigger when:
- Loss < 0.02 (excellent convergence), or
- No improvement for 100 epochs

---

## Comparing with MONAI UNet

To compare RSUNet vs MONAI UNet, you can run both:

```bash
# RSUNet (current config)
python scripts/main.py --config tutorials/hydra-lv.yaml

# MONAI UNet (for comparison)
python scripts/main.py --config tutorials/hydra-lv.yaml \
  model.architecture=monai_unet \
  experiment_name=hydra_lv_monai_unet_comparison
```

Compare results in TensorBoard:
```bash
tensorboard --logdir outputs/
```

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python scripts/main.py --config tutorials/hydra-lv.yaml \
  system.training.batch_size=16

# Or reduce patch size
python scripts/main.py --config tutorials/hydra-lv.yaml \
  data.patch_size=[16,64,64]
```

### Slow Data Loading
```bash
# Check if cache is working
python scripts/profile_dataloader.py --config tutorials/hydra-lv.yaml

# If /dev/shm issues, disable workers
python scripts/main.py --config tutorials/hydra-lv.yaml \
  system.training.num_workers=0
```

### Training Not Converging
```bash
# Reduce learning rate
python scripts/main.py --config tutorials/hydra-lv.yaml \
  optimization.optimizer.lr=0.0005

# Increase gradient clipping
python scripts/main.py --config tutorials/hydra-lv.yaml \
  optimization.gradient_clip_val=0.3
```

---

## Next Steps After Training

1. **Evaluate on test set**: Check adapted_rand score
2. **Visualize predictions**: Compare with ground truth
3. **Try Option 5**: Optimize loss weights `[1.0, 1.0, 3.0]`
4. **Try Option 1**: Switch to MedNeXt if GPU memory allows

---

## Performance Tracking

Track these metrics during training:

### Loss Metrics
- `train_loss_total_epoch`: Overall loss (should decrease)
- `train_loss_0`: Binary segmentation loss
- `train_loss_1`: Boundary detection loss
- `train_loss_2`: Distance transform loss

### Checkpoint Metrics
Best models saved based on minimum training loss.

---

## Questions?

- Check [HYDRA_LV_IMPROVEMENTS.md](HYDRA_LV_IMPROVEMENTS.md) for all optimization options
- See [CLAUDE.md](../CLAUDE.md) for full documentation
- Ask for help with specific issues!

---

**Configuration Status**: ✅ Ready to train with RSUNet!
