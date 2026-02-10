# EM-Specific Augmentation Guide

**PyTorch Connectomics Comprehensive EM Augmentation Reference**

This guide documents the built-in EM-specific augmentations in PyTorch Connectomics, which are **more sophisticated than BANIS** and comparable to nnUNet/MedNeXt preprocessing.

---

## Overview

PyTorch Connectomics provides **8 EM-specific augmentations** designed to simulate real artifacts in electron microscopy data:

1. **RandMisAlignmentd** - Section misalignment (translation + rotation)
2. **RandMissingSectiond** - Missing/damaged sections
3. **RandMissingPartsd** - Rectangular missing regions
4. **RandMotionBlurd** - Directional motion blur
5. **RandCutNoised** - Cuboid noise regions
6. **RandCutBlurd** - Super-resolution degradation
7. **RandMixupd** - Sample mixing (regularization)
8. **RandCopyPasted** - Object copy-paste augmentation

**All transforms:**
- ✅ MONAI-compatible (`MapTransform` interface)
- ✅ Lightning-compatible (work in training pipeline)
- ✅ Properly randomized (`RandomizableTransform`)
- ✅ Support multiple keys (image, label, etc.)

---

## Quick Start

### Basic EM Augmentation
```yaml
# tutorials/lucchi.yaml
data:
  augmentation:
    transforms:
      - RandMissingSectiond:
          keys: ["image"]
          prob: 0.5
          num_sections: 2

      - RandMisAlignmentd:
          keys: ["image"]
          prob: 0.5
          displacement: 16
```

### Use Presets (Recommended)
```bash
# Light augmentation (fast training)
python scripts/main.py --config tutorials/presets/aug_light.yaml

# Heavy augmentation (robust model)
python scripts/main.py --config tutorials/presets/aug_heavy.yaml

# Realistic EM artifacts
python scripts/main.py --config tutorials/presets/aug_realistic.yaml
```

---

## Transform Reference

### 1. RandMisAlignmentd

**Purpose:** Simulate section misalignment artifacts from EM imaging/alignment.

**How it works:**
- Randomly applies translation or rotation to entire sections
- Two modes: "slip" (single section) or "translation" (all sections from point)
- Uses proper geometric transforms (cv2.warpAffine)

**Parameters:**
```python
RandMisAlignmentd(
    keys: ["image"],              # Keys to transform
    prob: 0.5,                    # Probability to apply (per sample)
    displacement: 16,             # Maximum pixel displacement
    rotate_ratio: 0.5,            # 0.0-1.0, fraction using rotation vs translation
    allow_missing_keys: False
)
```

**Example:**
```yaml
transforms:
  - RandMisAlignmentd:
      keys: ["image"]
      prob: 0.5
      displacement: 16
      rotate_ratio: 0.5  # 50% rotation, 50% translation
```

**Comparison to BANIS:**
- ✅ **Better:** Uses proper geometric transforms (not circular shifts)
- ✅ **More features:** Supports rotation + translation
- ✅ **More realistic:** Two modes (slip/translation)

---

### 2. RandMissingSectiond

**Purpose:** Simulate missing or damaged sections in EM volumes.

**How it works:**
- Randomly selects and **removes** entire sections from volume
- Avoids first and last sections (preserve boundaries)
- Actually deletes sections (not just zero-filling)

**Parameters:**
```python
RandMissingSectiond(
    keys: ["image"],              # Keys to transform
    prob: 0.5,                    # Probability to apply (per sample)
    num_sections: 2,              # Number of sections to remove
    allow_missing_keys: False
)
```

**Example:**
```yaml
transforms:
  - RandMissingSectiond:
      keys: ["image"]
      prob: 0.5
      num_sections: 2  # Remove 2 random sections
```

**Comparison to BANIS:**
- ✅ **Better:** Actually removes sections (not just zero-filling)
- ✅ **More realistic:** Simulates true missing data
- ⚠️ **Note:** Changes volume shape (z-dimension reduced)

**When to use:**
- Training on data with known missing sections
- Improving robustness to incomplete volumes

---

### 3. RandMissingPartsd

**Purpose:** Create rectangular missing regions (holes) in random sections.

**How it works:**
- Selects random section
- Creates rectangular hole with random size and position
- Sets region to zero (simulates damaged areas)

**Parameters:**
```python
RandMissingPartsd(
    keys: ["image"],              # Keys to transform
    prob: 0.5,                    # Probability to apply (per sample)
    hole_range: (0.1, 0.3),      # Min/max hole size (fraction of section)
    allow_missing_keys: False
)
```

**Example:**
```yaml
transforms:
  - RandMissingPartsd:
      keys: ["image"]
      prob: 0.5
      hole_range: [0.1, 0.3]  # 10%-30% of section size
```

**When to use:**
- Simulating damaged regions in EM data
- Improving robustness to local artifacts

**Not in BANIS** ✅

---

### 4. RandMotionBlurd

**Purpose:** Apply directional motion blur to simulate scan artifacts.

**How it works:**
- Creates horizontal or vertical blur kernel
- Applies to random sections using convolution
- Simulates motion during EM acquisition

**Parameters:**
```python
RandMotionBlurd(
    keys: ["image"],              # Keys to transform
    prob: 0.5,                    # Probability to apply (per sample)
    sections: 2,                  # Number of sections (or tuple for range)
    kernel_size: 11,              # Blur kernel size (pixels)
    allow_missing_keys: False
)
```

**Example:**
```yaml
transforms:
  - RandMotionBlurd:
      keys: ["image"]
      prob: 0.5
      sections: [1, 3]  # 1-3 random sections
      kernel_size: 11   # 11x11 kernel
```

**When to use:**
- Training on data with motion artifacts
- Improving robustness to blur

**Not in BANIS** ✅

---

### 5. RandCutNoised

**Purpose:** Add noise to random cuboid regions for robustness.

**How it works:**
- Selects random cuboid region
- Adds uniform noise
- Clips to valid range [0, 1]

**Parameters:**
```python
RandCutNoised(
    keys: ["image"],              # Keys to transform
    prob: 0.5,                    # Probability to apply (per sample)
    length_ratio: 0.25,           # Size of cuboid (fraction of volume)
    noise_scale: 0.2,             # Noise magnitude
    allow_missing_keys: False
)
```

**Example:**
```yaml
transforms:
  - RandCutNoised:
      keys: ["image"]
      prob: 0.5
      length_ratio: 0.25  # 25% of volume size
      noise_scale: 0.2    # ±0.2 noise
```

**When to use:**
- Regularization (like Cutout/CutMix)
- Improving robustness to local noise

**Not in BANIS** ✅

---

### 6. RandCutBlurd

**Purpose:** Downsample cuboid regions to force super-resolution learning.

**How it works:**
- Selects random cuboid region
- Downsamples by random ratio (2-8x)
- Upsamples back to original size (creates blur)
- Forces model to learn super-resolution

**Parameters:**
```python
RandCutBlurd(
    keys: ["image"],              # Keys to transform
    prob: 0.5,                    # Probability to apply (per sample)
    length_ratio: 0.25,           # Size of cuboid (fraction of volume)
    down_ratio_range: (2.0, 8.0), # Downsampling factor range
    downsample_z: False,          # Whether to downsample z-axis
    allow_missing_keys: False
)
```

**Example:**
```yaml
transforms:
  - RandCutBlurd:
      keys: ["image"]
      prob: 0.5
      length_ratio: 0.25
      down_ratio_range: [2.0, 8.0]  # 2x-8x downsampling
      downsample_z: false           # Keep z-resolution
```

**When to use:**
- Multi-resolution training
- Improving robustness to resolution variations
- Super-resolution tasks

**Not in BANIS** ✅ **Very clever augmentation!**

---

### 7. RandMixupd

**Purpose:** Mix samples within batch for regularization.

**How it works:**
- Linearly interpolates between two samples
- `mixed = alpha * sample1 + (1 - alpha) * sample2`
- Batch-level augmentation

**Parameters:**
```python
RandMixupd(
    keys: ["image"],              # Keys to transform
    prob: 0.5,                    # Probability to apply (per batch)
    alpha_range: (0.7, 0.9),     # Mixing ratio range
    allow_missing_keys: False
)
```

**Example:**
```yaml
transforms:
  - RandMixupd:
      keys: ["image"]
      prob: 0.5
      alpha_range: [0.7, 0.9]  # 70%-90% original sample
```

**When to use:**
- Regularization (prevents overfitting)
- Small datasets
- Improving generalization

**Note:** Requires batch size > 1

**Not in BANIS** ✅

---

### 8. RandCopyPasted

**Purpose:** Copy objects, transform them, and paste in non-overlapping regions.

**How it works:**
- Extracts objects based on label mask
- Applies rotation/flipping
- Pastes in non-overlapping location
- Increases object diversity

**Parameters:**
```python
RandCopyPasted(
    keys: ["image"],              # Keys to transform
    label_key: "label",           # Segmentation mask key
    prob: 0.5,                    # Probability to apply (per sample)
    max_obj_ratio: 0.7,           # Skip if object too large
    rotation_angles: list(range(30, 360, 30)),  # Rotation angles to try
    border: 3,                    # Border for overlap checking
    allow_missing_keys: False
)
```

**Example:**
```yaml
transforms:
  - RandCopyPasted:
      keys: ["image"]
      label_key: "label"
      prob: 0.5
      max_obj_ratio: 0.7
      rotation_angles: [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
```

**When to use:**
- Instance segmentation
- Small datasets (data augmentation)
- Improving object detection

**Not in BANIS** ✅ **Advanced augmentation!**

---

## Augmentation Strategies

### Strategy 1: Light (Fast Training)
**Use case:** Quick experiments, large datasets, fast iteration

```yaml
# tutorials/presets/aug_light.yaml
data:
  augmentation:
    transforms:
      - RandMissingSectiond:
          keys: ["image"]
          prob: 0.3
          num_sections: 1

      - RandMisAlignmentd:
          keys: ["image"]
          prob: 0.3
          displacement: 8
          rotate_ratio: 0.0  # Translation only
```

**Augmentation probability:** ~30%
**Training speed:** Fast (minimal overhead)
**Model robustness:** Basic

---

### Strategy 2: Realistic (BANIS-Style)
**Use case:** Replicating BANIS augmentation strategy

```yaml
# tutorials/presets/aug_realistic.yaml
data:
  augmentation:
    transforms:
      # Missing sections (like BANIS DropSliced but better)
      - RandMissingSectiond:
          keys: ["image"]
          prob: 0.5
          num_sections: 2

      # Misalignment (like BANIS ShiftSliced but better)
      - RandMisAlignmentd:
          keys: ["image"]
          prob: 0.5
          displacement: 10  # BANIS uses max_shift=10
          rotate_ratio: 0.0  # Pure translation like BANIS

      # Additional features BANIS doesn't have
      - RandMotionBlurd:
          keys: ["image"]
          prob: 0.5
          sections: 2
          kernel_size: 11
```

**Augmentation probability:** ~50%
**Training speed:** Medium
**Model robustness:** Good (realistic EM artifacts)

---

### Strategy 3: Heavy (Maximum Robustness)
**Use case:** Small datasets, maximum robustness, production models

```yaml
# tutorials/presets/aug_heavy.yaml
data:
  augmentation:
    transforms:
      - RandMissingSectiond:
          keys: ["image"]
          prob: 0.7
          num_sections: 3

      - RandMisAlignmentd:
          keys: ["image"]
          prob: 0.7
          displacement: 20
          rotate_ratio: 0.5

      - RandMissingPartsd:
          keys: ["image"]
          prob: 0.5
          hole_range: [0.1, 0.3]

      - RandMotionBlurd:
          keys: ["image"]
          prob: 0.5
          sections: [1, 3]
          kernel_size: 11

      - RandCutBlurd:
          keys: ["image"]
          prob: 0.5
          length_ratio: 0.25
          down_ratio_range: [2.0, 8.0]

      - RandCutNoised:
          keys: ["image"]
          prob: 0.3
          length_ratio: 0.25
          noise_scale: 0.2
```

**Augmentation probability:** ~70%
**Training speed:** Slower (more transforms)
**Model robustness:** Excellent (maximum robustness)

---

### Strategy 4: Super-Resolution
**Use case:** Training models for super-resolution or multi-scale learning

```yaml
# tutorials/presets/aug_superres.yaml
data:
  augmentation:
    transforms:
      - RandCutBlurd:
          keys: ["image"]
          prob: 0.8  # High probability
          length_ratio: 0.5  # Larger regions
          down_ratio_range: [2.0, 8.0]
          downsample_z: false

      - RandMisAlignmentd:
          keys: ["image"]
          prob: 0.3
          displacement: 10
```

---

### Strategy 5: Instance Segmentation
**Use case:** Neuron instance segmentation, object detection

```yaml
# tutorials/presets/aug_instance.yaml
data:
  augmentation:
    transforms:
      - RandCopyPasted:
          keys: ["image"]
          label_key: "label"
          prob: 0.6
          max_obj_ratio: 0.7

      - RandMisAlignmentd:
          keys: ["image", "label"]  # Apply to both
          prob: 0.5
          displacement: 16
          rotate_ratio: 0.5
```

---

## Comparison: PyTC vs BANIS vs nnUNet

| Feature | PyTC | BANIS | nnUNet | Notes |
|---------|------|-------|--------|-------|
| **Slice dropout** | `RandMissingSectiond` | `DropSliced` | ❌ | PyTC best (deletes vs zeros) |
| **Slice shifting** | `RandMisAlignmentd` | `ShiftSliced` | ❌ | PyTC best (rotation + translation) |
| **Missing parts** | `RandMissingPartsd` | ❌ | ❌ | PyTC unique |
| **Motion blur** | `RandMotionBlurd` | ❌ | ❌ | PyTC unique |
| **Cut noise** | `RandCutNoised` | ❌ | ✅ (CutMix) | Similar |
| **Cut blur** | `RandCutBlurd` | ❌ | ❌ | PyTC unique (clever!) |
| **Mixup** | `RandMixupd` | ❌ | ❌ | PyTC unique |
| **Copy-paste** | `RandCopyPasted` | ❌ | ❌ | PyTC unique |
| **Elastic deform** | MONAI `Rand3DElasticd` | ✅ | ✅ | Use MONAI |
| **Intensity** | MONAI `RandShiftIntensityd` | ✅ | ✅ | Use MONAI |
| **Affine** | MONAI `RandAffined` | ✅ | ✅ | Use MONAI |

**Score: PyTC 8 unique, BANIS 2 (inferior), nnUNet 1**

---

## Best Practices

### 1. Start Light, Go Heavy
```python
# Start with light augmentation for fast iteration
python scripts/main.py --config tutorials/presets/aug_light.yaml

# Once model works, switch to heavy for final training
python scripts/main.py --config tutorials/presets/aug_heavy.yaml
```

### 2. Match Augmentation to Data Artifacts
```yaml
# If your data has misalignment issues
- RandMisAlignmentd:
    prob: 0.7  # High probability

# If your data is clean
- RandMisAlignmentd:
    prob: 0.2  # Low probability
```

### 3. Apply to Both Image and Label
```yaml
# Geometric transforms: apply to both
- RandMisAlignmentd:
    keys: ["image", "label"]  # Both

# Intensity transforms: image only
- RandCutNoised:
    keys: ["image"]  # Image only
```

### 4. Combine with MONAI Standard Transforms
```yaml
transforms:
  # EM-specific (PyTC)
  - RandMissingSectiond:
      keys: ["image"]
      prob: 0.5

  # Standard (MONAI)
  - RandShiftIntensityd:
      keys: ["image"]
      prob: 0.5
      offsets: 0.1

  - Rand3DElasticd:
      keys: ["image", "label"]
      prob: 0.5
      sigma_range: [5, 7]
      magnitude_range: [50, 150]
```

### 5. Monitor Training Impact
```python
# Track augmentation effect on metrics
# More augmentation = slower convergence but better generalization
# Less augmentation = faster convergence but may overfit
```

---

## Common Issues

### Issue 1: Training Too Slow
**Symptom:** Training takes too long with heavy augmentation

**Solution:**
```yaml
# Reduce augmentation probability
- RandMissingSectiond:
    prob: 0.3  # Reduce from 0.7

# Or use fewer transforms
# Remove expensive transforms like RandCopyPasted
```

### Issue 2: Model Not Improving
**Symptom:** Validation loss not decreasing

**Solutions:**
1. **Too much augmentation:** Reduce probability
2. **Wrong augmentation:** Check if augmentation matches data distribution
3. **Need more epochs:** Heavy augmentation needs longer training

### Issue 3: Shape Mismatch
**Symptom:** `RuntimeError: shape mismatch`

**Cause:** `RandMissingSectiond` changes volume shape

**Solution:**
```yaml
# Option 1: Don't use RandMissingSectiond
# Option 2: Ensure consistent patch sampling after augmentation
# Option 3: Use padding to restore shape
```

---

## References

1. **PyTC Implementation:** `connectomics/data/augment/monai_transforms.py`
2. **MONAI Docs:** https://docs.monai.io/en/stable/transforms.html
3. **BANIS:** `.claude/BANIS_SUMMARY.md`
4. **nnUNet:** https://github.com/MIC-DKFZ/nnUNet

---

## See Also

- [Augmentation Presets](../tutorials/presets/) - Ready-to-use configs
- [BANIS Comparison](.claude/PHASE6_COMPARISON.md) - Detailed comparison
- [MONAI Transforms](https://docs.monai.io/en/stable/transforms.html) - Standard augmentations
