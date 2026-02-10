# Hydra-LV Fine-Tuning Improvement Plan

## Problem Analysis

### Current Performance (UNACCEPTABLE)
```
test_adapted_rand:          0.471  (target: <0.1, ideal: <0.05)
test_adapted_rand_precision: 0.828  (good - few false positives)
test_adapted_rand_recall:    0.389  (TERRIBLE - 61% of instances missed!)
```

**Root Cause:** Severe **under-segmentation** - the model is missing 61% of vesicles (low recall).

### Dataset Characteristics
- **Volume:** 51 × 516 × 516 = 13.6M voxels
- **Instances:** 51 vesicles (VERY SMALL dataset!)
- **Instance sizes:** 224-1396 voxels (mean: 596, median: 555)
- **Mask coverage:** Only 2.1% of volume
- **Challenge:** Small, broken vesicles that need to be detected

### Training Status Analysis
- **Final loss:** 0.555 (stagnated for 100+ epochs)
- **Loss breakdown:**
  - Label loss: 0.660 (binary + BCE) - NOT converged
  - Boundary loss: 0.449 - NOT converged
  - EDT loss: 0.002 - Well converged
- **Training:** 435 epochs, loss plateaued around epoch 200
- **Data:** Only 1 volume with batch_size=8, 40 batches/epoch = 320 patches/epoch

## Critical Issues

### 1. **MASSIVE OVERFITTING RISK** ⚠️
- **Problem:** Training on 1 tiny volume (51 instances) while testing on SAME volume
- **Current setup:** Should EASILY overfit, but loss=0.555 is too high
- **Conclusion:** Model is NOT overfitting → indicates fundamental training problem

### 2. **Insufficient Training Data**
- **Problem:** 320 patches/epoch from 1 volume is NOT enough
- **Batch size:** 8 across 4 GPUs → only 2 samples/GPU (inefficient)
- **Iterations per epoch:** Only 40 (way too few)

### 3. **Weak Augmentation**
- **Current:** preset="some" (moderate augmentations)
- **Problem:** Not enough diversity for small dataset

### 4. **Sub-optimal Learning Rate**
- **Current:** lr=1e-4 (fine-tuning LR)
- **Problem:** May be too conservative for this dataset
- **LR at epoch 435:** 5e-6 (cosine decay → almost zero)

### 5. **Loss Function Mismatch**
- **Problem:** Using same losses as pretrained (designed for large vesicles)
- **Observation:** Loss weights are balanced (uncertainty weighting), but losses not converging
- **Boundary loss:** 0.449 is high → model struggles with boundaries

### 6. **Architecture Mismatch**
- **Model:** RSUNet (42M params) - HUGE for 51 instances!
- **Deep supervision:** Disabled (should be enabled for small objects)

## Improvement Plan

### 🎯 **PRIMARY GOAL: Overfit the Training Data**

Since testing on training data, we MUST achieve near-perfect performance (ARE < 0.05).

---

## Phase 1: Emergency Fixes (Highest Priority)

### 1.1 Increase Training Iterations ⭐⭐⭐
**Problem:** Only 40 batches/epoch = severe undertraining

```yaml
data:
  iter_num_per_epoch: 5000  # UP FROM: 1280 (increase 4x)
```

**Impact:** Model sees 5000 patches/epoch instead of 320 → better convergence

### 1.2 Enable Deep Supervision ⭐⭐⭐
**Problem:** Small vesicles need multi-scale supervision

Add to model config:
```yaml
model:
  architecture: rsunet
  deep_supervision: true  # ADD THIS - critical for small objects
```

**Impact:** Forces model to learn at multiple scales → better small object detection

### 1.3 Aggressive Augmentation ⭐⭐⭐
**Problem:** Weak augmentation = insufficient diversity

```yaml
data:
  augmentation:
    preset: "all"  # UP FROM: "some"

    # MORE aggressive augmentations
    flip:
      enabled: true
      prob: 0.8  # UP FROM: 0.5

    rotate:
      enabled: true
      prob: 0.8  # UP FROM: 0.5

    affine:
      enabled: true
      prob: 0.5  # UP FROM: 0.3
      scale_range: [0.1, 0.1, 0.1]  # UP FROM: 0.05

    intensity:
      enabled: true
      gaussian_noise_prob: 0.5  # UP FROM: 0.2
      shift_intensity_prob: 0.6  # UP FROM: 0.4
      contrast_prob: 0.6  # UP FROM: 0.4

    # Cutout for regularization
    cutout:
      enabled: true
      prob: 0.3
      num_holes: 3
      hole_size: [8, 16, 16]
```

**Impact:** More diverse training samples → prevents memorization

---

## Phase 2: Optimization Fixes

### 2.1 Increase Learning Rate ⭐⭐
**Problem:** lr=1e-4 too conservative, already decayed to 5e-6

```yaml
optimization:
  max_epochs: 1000  # UP FROM: 500 (need more time)

  optimizer:
    lr: 3e-4  # UP FROM: 1e-4 (3x higher)
    weight_decay: 0.001  # DOWN FROM: 0.01 (less regularization)

  scheduler:
    warmup_epochs: 100  # UP FROM: 50
    min_lr: 1e-5  # UP FROM: 1e-6
```

**Impact:** Faster convergence + longer useful training time

### 2.2 Adjust Batch Size ⭐
**Problem:** batch_size=8 across 4 GPUs = 2/GPU (too small)

```yaml
system:
  training:
    batch_size: 4  # DOWN FROM: 8
```

**Rationale:** With 4 GPUs, effective batch = 16. Smaller per-GPU batch = more gradient updates

### 2.3 Reduce Early Stopping Patience ⭐
**Problem:** patience=50 is too long

```yaml
monitor:
  early_stopping:
    patience: 100  # UP FROM: 50 (give more time since we increased iter_num)
    min_delta: 1e-5  # DOWN FROM: 1e-4 (more sensitive)
```

---

## Phase 3: Loss Function Tuning

### 3.1 Adjust Loss Weights ⭐⭐
**Problem:** All losses weighted equally, but boundary/binary not converging

```yaml
model:
  # Rebalance losses - emphasize binary and boundary
  loss_weights: [2.0, 2.0, 1.0, 1.0, 0.5]  # FROM: [1.0, 1.0, 1.0, 1.0, 1.0]
  # Breakdown:
  # - Binary Dice: 2.0 (emphasize object detection)
  # - Binary BCE: 2.0 (emphasize object detection)
  # - Boundary Dice: 1.0
  # - Boundary Tversky: 1.0
  # - EDT SmoothL1: 0.5 (reduce, it's already converged)

  loss_balancing:
    strategy: none  # DISABLE uncertainty weighting initially to debug
```

### 3.2 Alternative: Use Focal Loss ⭐
**Problem:** BCE may not handle class imbalance well (2.1% mask coverage)

```yaml
model:
  loss_functions: [DiceLoss, FocalLoss, DiceLoss, TverskyLoss, SmoothL1Loss]
  # Change index 1 from WeightedBCEWithLogitsLoss → FocalLoss

  loss_kwargs:
    - {sigmoid: true, smooth_nr: 1e-5, smooth_dr: 1e-5}
    - {alpha: 0.25, gamma: 2.0, reduction: mean}  # Focal loss for hard examples
    - {sigmoid: true, smooth_nr: 1e-5, smooth_dr: 1e-5}
    - {sigmoid: true, alpha: 0.6, beta: 0.4, smooth_nr: 1e-5, smooth_dr: 1e-5}  # More recall-focused
    - {beta: 0.1, reduction: mean, tanh: true}
```

---

## Phase 4: Architecture Alternatives

### 4.1 Try Smaller Architecture ⭐
**Problem:** RSUNet (42M params) massive overkill for 51 instances

**Option A: Smaller RSUNet**
```yaml
model:
  filters: [16, 32, 64, 128, 256]  # DOWN FROM: [32, 64, 128, 256, 512]
  dropout: 0.2  # UP FROM: 0.1 (more regularization)
```

**Option B: Switch to MONAI UNet**
```yaml
model:
  architecture: monai_basic_unet3d
  filters: [16, 32, 64, 128, 256]
  dropout: 0.2
  deep_supervision: true
```

### 4.2 Enable Test-Time Augmentation (TTA) ⭐
**Problem:** Not using TTA for evaluation

```yaml
inference:
  test_time_augmentation:
    flip_axes: [[0], [1], [2], [0, 1], [0, 2], [1, 2]]  # FROM: null
    rotation90_axes: [[1, 2]]  # FROM: null
```

---

## Phase 5: Decoding/Post-processing Fixes

### 5.1 Tune Decoding Thresholds ⭐⭐
**Problem:** Current thresholds may be sub-optimal

**Strategy:** Run hyperparameter sweep on thresholds

```yaml
test:
  decoding:
    - name: decode_instance_binary_contour_distance
      kwargs:
        # Try LOWER binary threshold (more sensitive)
        binary_threshold: [0.3, 0.7]  # FROM: [0.5, 0.7]

        # Try LOWER contour threshold (allow weaker boundaries)
        contour_threshold: [0.5, 13.1]  # FROM: [0.8, 13.1]

        # Try LARGER min_seed_size (filter noise)
        min_seed_size: 10  # FROM: 4
```

**Impact:** Better recall by being more sensitive to detections

### 5.2 Try Watershed Decoding ⭐
**Alternative decoding** that may work better for broken vesicles:

```yaml
test:
  decoding:
    - name: decode_instance_watershed_distance
      kwargs:
        distance_threshold: 0.3
        min_seed_size: 8
```

---

## Recommended Implementation Strategy

### **Quick Win Strategy (Try First)** 🚀

1. **Immediate changes** (restart training):
   ```yaml
   # In hydra-lv-finetune.yaml

   data:
     iter_num_per_epoch: 5000  # 4x more iterations
     augmentation:
       preset: "all"  # Maximum augmentation

   model:
     deep_supervision: true  # CRITICAL
     loss_weights: [2.0, 2.0, 1.0, 1.0, 0.5]

   optimization:
     max_epochs: 1000
     optimizer:
       lr: 3e-4  # 3x higher
       weight_decay: 0.001
   ```

2. **Train overnight** (1000 epochs with 5000 iter/epoch)

3. **Expected result:**
   - Loss should drop to < 0.2
   - Adapted Rand Error < 0.1
   - Recall > 0.9

### **If Still Not Working** 🔧

1. **Disable external weights** (train from scratch):
   ```yaml
   model:
     external_weights_path: null  # Remove pretrained weights
   ```

2. **Try MedNeXt** (better for small objects):
   ```yaml
   model:
     architecture: mednext
     mednext_size: S
     deep_supervision: true
     # MedNeXt requires AdamW with lr=1e-3

   optimization:
     optimizer:
       lr: 1e-3
     scheduler:
       name: constant  # MedNeXt uses constant LR
   ```

3. **Increase model capacity with deep supervision scales:**
   - Deep supervision provides 5 intermediate losses
   - Each scale helps detect vesicles at different sizes

---

## Debugging Checklist

### Before Training
- [ ] Check data loading: `python scripts/profile_dataloader.py --config tutorials/hydra-lv-finetune.yaml`
- [ ] Visualize augmentations in TensorBoard
- [ ] Verify label transform (binary, boundary, EDT) is correct

### During Training
- [ ] Monitor individual loss components (label, boundary, EDT)
- [ ] Check if losses are decreasing (all should trend down)
- [ ] Watch learning rate decay (should stay > 1e-5 for longer)
- [ ] Inspect training visualizations (every epoch)

### After Training
- [ ] Test on training data (should overfit completely)
- [ ] Visualize predictions vs ground truth
- [ ] Check missed instances (low recall regions)
- [ ] Try different decoding thresholds

---

## Expected Results

### Target Metrics (Testing on Training Data)
```
Adapted Rand Error:    < 0.05  (currently: 0.471)
Precision:             > 0.95  (currently: 0.828)
Recall:                > 0.95  (currently: 0.389)
```

### Loss Targets
```
Total loss:       < 0.2   (currently: 0.555)
Label loss:       < 0.1   (currently: 0.660)
Boundary loss:    < 0.1   (currently: 0.449)
EDT loss:         < 0.01  (currently: 0.002) ✓
```

---

## Root Cause Summary

The poor performance (0.471 ARE, 0.389 recall) is caused by:

1. **Insufficient training** (only 320 patches/epoch)
2. **No deep supervision** (critical for small objects)
3. **Weak augmentation** (not enough diversity)
4. **Conservative LR** (already decayed to 5e-6)
5. **Sub-optimal loss weights** (boundary loss not converging)
6. **Potentially wrong thresholds** (decoding too conservative)

The model is **undertrained**, not overfitted. Loss=0.555 on 51 instances is unacceptably high - should be near zero.

---

## Priority Ranking

**MUST DO (Phase 1):**
1. ⭐⭐⭐ Increase `iter_num_per_epoch` to 5000
2. ⭐⭐⭐ Enable `deep_supervision: true`
3. ⭐⭐⭐ Change augmentation to `preset: "all"`

**SHOULD DO (Phase 2-3):**
4. ⭐⭐ Increase LR to 3e-4
5. ⭐⭐ Adjust loss weights [2.0, 2.0, 1.0, 1.0, 0.5]
6. ⭐ Reduce batch_size to 4 (per GPU)

**NICE TO HAVE (Phase 4-5):**
7. ⭐ Try smaller model (reduce params)
8. ⭐ Enable TTA
9. ⭐ Tune decoding thresholds

---

## Next Steps

1. **Update config file** with Phase 1 changes
2. **Start fresh training** (don't resume from checkpoint)
3. **Monitor closely** for first 50 epochs
4. **Check visualizations** to ensure augmentations working
5. **If loss < 0.3 by epoch 200** → on right track
6. **If loss still > 0.4 by epoch 200** → try Phase 4 (architecture change)

Good luck! 🚀
