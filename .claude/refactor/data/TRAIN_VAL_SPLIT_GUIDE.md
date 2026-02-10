# Train/Val Split Guide - DeepEM-Style Volumetric Splitting

**Inspired by:** DeepEM's spatial train/val splitting approach

## Overview

PyTorch Connectomics now supports automatic train/val splitting from a single volume, following DeepEM's proven approach of spatial splitting along the Z-axis. This eliminates the need for pre-split datasets and provides a consistent, reproducible splitting strategy.

## Key Concept

Instead of loading separate train/val volumes:
- **Use first 80% of volume** for training (slices 0-80 if 100 total)
- **Use last 20% of volume** for validation (slices 80-100)
- **Automatically pad validation** to model input size if needed

This approach:
✅ Ensures spatial separation (no data leakage)
✅ Works with single-volume datasets
✅ Automatically handles size mismatches
✅ Compatible with DeepEM training strategies

## Configuration

### Basic Setup (80/20 Split)

```yaml
data:
  # Single volume - will be split automatically
  train_image: datasets/volume.h5
  train_label: datasets/label.h5

  # No separate validation files needed
  val_image: null
  val_label: null

  # Enable automatic splitting
  split_enabled: true
  split_train_range: [0.0, 0.8]    # First 80%
  split_val_range: [0.8, 1.0]      # Last 20%
  split_axis: 0                    # Z-axis
  split_pad_val: true              # Pad validation
  split_pad_mode: reflect          # Padding mode
```

### Custom Split Ratios

```yaml
# 70/30 split
split_train_range: [0.0, 0.7]
split_val_range: [0.7, 1.0]

# 90/10 split (smaller validation)
split_train_range: [0.0, 0.9]
split_val_range: [0.9, 1.0]

# 85/15 split
split_train_range: [0.0, 0.85]
split_val_range: [0.85, 1.0]
```

### Different Split Axes

```yaml
# Split along Z-axis (default, most common)
split_axis: 0

# Split along Y-axis (height)
split_axis: 1

# Split along X-axis (width)
split_axis: 2
```

### Padding Options

```yaml
# Reflection padding (mirrors edges) - RECOMMENDED
split_pad_mode: reflect

# Replication padding (repeats edge values)
split_pad_mode: replicate

# Constant padding (fills with zeros)
split_pad_mode: constant

# Circular padding (wraps around)
split_pad_mode: circular

# Disable padding (validation must match patch_size)
split_pad_val: false
```

## Use Cases

### Case 1: Standard EM Volume

```yaml
# Volume: 100 slices, 256x256
# Model: needs 32x256x256 input

data:
  train_image: datasets/em_volume.h5
  train_label: datasets/em_label.h5
  patch_size: [32, 256, 256]

  split_enabled: true
  split_train_range: [0.0, 0.8]    # Slices 0-80 for training
  split_val_range: [0.8, 1.0]      # Slices 80-100 for validation (20 slices)
  split_pad_val: true              # Pad 20→32 slices
```

**Result:**
- Training: 80 slices (use random crops of 32 slices)
- Validation: 20 slices → padded to 32 slices for inference

### Case 2: Large Volume, Small Validation

```yaml
# Volume: 500 slices
# Want small validation set

data:
  split_train_range: [0.0, 0.96]   # 480 slices for training
  split_val_range: [0.96, 1.0]     # 20 slices for validation
```

### Case 3: Multiple Volumes

```yaml
# When you have multiple volumes, disable split
# Use traditional separate train/val files

data:
  train_image: [vol1.h5, vol2.h5, vol3.h5]
  val_image: [vol4.h5]
  split_enabled: false  # Use separate files
```

### Case 4: Non-Standard Axis Split

```yaml
# Split along Y-axis instead of Z
# Useful for anisotropic volumes

data:
  split_axis: 1  # Y-axis
  split_train_range: [0.0, 0.8]
  split_val_range: [0.8, 1.0]
```

## Python API

### Direct Usage

```python
from connectomics.data.utils.split import (
    split_volume_train_val,
    split_and_pad_volume
)

# Load volume
volume = load_volume('path/to/volume.h5')  # Shape: (100, 256, 256)

# Method 1: Get split slices
train_slices, val_slices = split_volume_train_val(
    volume.shape, train_ratio=0.8, axis=0
)
train_data = volume[train_slices]  # (80, 256, 256)
val_data = volume[val_slices]      # (20, 256, 256)

# Method 2: Split and pad in one step
train_data, val_data = split_and_pad_volume(
    volume,
    train_ratio=0.8,
    target_size=(32, 256, 256),  # Pad val to this size
    pad_mode='reflect'
)
# train_data: (80, 256, 256)
# val_data: (32, 256, 256) - padded from 20 to 32
```

### Creating Masks (DeepEM Style)

```python
from connectomics.data.utils.split import create_split_masks, save_split_masks_h5

# Create binary masks
train_mask, val_mask = create_split_masks(
    volume_shape=(100, 256, 256),
    train_ratio=0.8
)
# train_mask: 1 for first 80 slices, 0 elsewhere
# val_mask: 1 for last 20 slices, 0 elsewhere

# Save to HDF5 (compatible with DeepEM)
save_split_masks_h5(
    output_dir='path/to/output',
    volume_shape=(100, 256, 256),
    train_ratio=0.8,
    train_filename='msk_train.h5',
    val_filename='msk_val.h5'
)
```

### Advanced: Custom Split Transform

```python
from connectomics.data.utils.split import split_volume_train_val
from monai.transforms import MapTransform

class SplitVolumeTransform(MapTransform):
    """Custom transform for train/val splitting."""

    def __init__(self, keys, train_ratio=0.8, axis=0, mode='train'):
        super().__init__(keys)
        self.train_ratio = train_ratio
        self.axis = axis
        self.mode = mode

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            if key in d:
                volume = d[key]
                train_slices, val_slices = split_volume_train_val(
                    volume.shape, self.train_ratio, self.axis
                )

                # Apply appropriate split
                if self.mode == 'train':
                    d[key] = volume[train_slices]
                else:  # validation
                    d[key] = volume[val_slices]

        return d
```

## Comparison with DeepEM

| Feature | DeepEM | PyTorch Connectomics |
|---------|--------|----------------------|
| **Split Method** | Pre-computed masks (`msk_train.h5`, `msk_val.h5`) | Config-driven automatic split |
| **Split Ratio** | Fixed in dataset code | Configurable (any ratio) |
| **Padding** | Manual preprocessing | Automatic with multiple modes |
| **Flexibility** | One split per dataset | Multiple splits from config |
| **Axis** | Z-axis only | Any axis (Z, Y, X) |
| **Compatibility** | HDF5 masks | HDF5, TIFF, Zarr, N5 |

## Best Practices

### 1. **Choose Split Ratio Carefully**

```yaml
# For small datasets (< 200 slices)
split_train_range: [0.0, 0.85]  # 85/15 split
split_val_range: [0.85, 1.0]

# For medium datasets (200-500 slices)
split_train_range: [0.0, 0.8]   # 80/20 split (recommended)
split_val_range: [0.8, 1.0]

# For large datasets (> 500 slices)
split_train_range: [0.0, 0.9]   # 90/10 split
split_val_range: [0.9, 1.0]
```

### 2. **Ensure Sufficient Validation Size**

```python
# Bad: Validation too small
volume_shape = (50, 256, 256)
split = 0.9  # Only 5 slices for validation - might not be enough!

# Good: Ensure minimum validation size
from connectomics.data.utils.split import split_volume_train_val

train_slices, val_slices = split_volume_train_val(
    volume_shape=(50, 256, 256),
    train_ratio=0.9,
    min_val_size=10  # Force at least 10 slices for validation
)
# Will adjust ratio to give 40 train / 10 val
```

### 3. **Match Model Requirements**

```yaml
# Model needs 32x256x256 input
# Validation has 20 slices

data:
  patch_size: [32, 256, 256]
  split_pad_val: true           # REQUIRED: pad 20→32
  split_pad_mode: reflect       # Use reflection (better than constant)
```

### 4. **Spatial Considerations**

```yaml
# For anisotropic data (different Z/Y/X resolution)
# Split along axis with best continuity

# If Z is low-res, split along Z (default)
split_axis: 0

# If Y/X are low-res, consider splitting along them
split_axis: 1  # or 2
```

### 5. **Debugging Split**

```python
from connectomics.data.utils.split import split_volume_train_val

# Verify split before training
volume_shape = (100, 256, 256)
train_slices, val_slices = split_volume_train_val(
    volume_shape, train_ratio=0.8, axis=0
)

print(f"Original shape: {volume_shape}")
print(f"Train slices: {train_slices}")
print(f"Val slices: {val_slices}")

# Output:
# Original shape: (100, 256, 256)
# Train slices: (slice(0, 80, None), slice(None, None, None), slice(None, None, None))
# Val slices: (slice(80, 100, None), slice(None, None, None), slice(None, None, None))
```

## Troubleshooting

### Issue 1: Validation Too Small

**Problem:** Validation set has fewer slices than model input size

```
ValueError: Validation volume (15, 256, 256) smaller than patch_size (32, 256, 256)
```

**Solution:** Enable padding

```yaml
split_pad_val: true
split_pad_mode: reflect
```

### Issue 2: Incorrect Split Axis

**Problem:** Split creates non-contiguous regions

```yaml
# Wrong: Splitting along X creates fragmented regions
split_axis: 2

# Correct: Split along Z for volumetric data
split_axis: 0
```

### Issue 3: Overlapping Train/Val

**Problem:** Train and val ranges overlap

```yaml
# WRONG: Overlapping ranges
split_train_range: [0.0, 0.85]
split_val_range: [0.8, 1.0]  # Overlaps at 0.8-0.85!

# CORRECT: Non-overlapping ranges
split_train_range: [0.0, 0.8]
split_val_range: [0.8, 1.0]
```

### Issue 4: Config Not Applied

**Problem:** Split not working

**Check:**
1. `split_enabled: true` is set
2. `val_image` and `val_label` are `null` or not specified
3. Training uses single volume, not multiple

## Examples

### Example 1: Lucchi Dataset

```yaml
# File: tutorials/lucchi_split.yaml

data:
  train_image: datasets/Lucchi/img/train_im.tif
  train_label: datasets/Lucchi/label/train_label.tif

  patch_size: [18, 160, 160]

  # 80/20 split with padding
  split_enabled: true
  split_train_range: [0.0, 0.8]
  split_val_range: [0.8, 1.0]
  split_axis: 0
  split_pad_val: true
  split_pad_mode: reflect
```

### Example 2: CREMI Dataset

```yaml
data:
  train_image: datasets/CREMI/sampleA.h5
  train_label: datasets/CREMI/sampleA_labels.h5

  patch_size: [32, 256, 256]

  # 85/15 split (more training data)
  split_enabled: true
  split_train_range: [0.0, 0.85]
  split_val_range: [0.85, 1.0]
  split_axis: 0
  split_pad_val: true
```

### Example 3: Custom Ratio

```yaml
data:
  # 75/25 split for extra validation
  split_train_range: [0.0, 0.75]
  split_val_range: [0.75, 1.0]
```

## Migration from DeepEM

If you have DeepEM-style mask files (`msk_train.h5`, `msk_val.h5`):

### Option 1: Use Config-Based Split (Recommended)

```yaml
# Replace mask files with automatic split
data:
  split_enabled: true
  split_train_range: [0.0, 0.8]  # Same as mask boundaries
  split_val_range: [0.8, 1.0]
```

### Option 2: Convert Masks to Config

```python
import h5py
import numpy as np

# Load masks
with h5py.File('msk_train.h5') as f:
    train_mask = f['main'][:]

with h5py.File('msk_val.h5') as f:
    val_mask = f['main'][:]

# Find split boundaries
train_indices = np.where(train_mask > 0)
val_indices = np.where(val_mask > 0)

train_start = train_indices[0].min() / train_mask.shape[0]
train_end = train_indices[0].max() / train_mask.shape[0]
val_start = val_indices[0].min() / val_mask.shape[0]
val_end = val_indices[0].max() / val_mask.shape[0]

print(f"split_train_range: [{train_start}, {train_end}]")
print(f"split_val_range: [{val_start}, {val_end}]")
```

## Performance Considerations

**Memory:**
- Split is done at load time, not in advance
- No duplication of data in memory
- Padding only affects validation set

**Speed:**
- Minimal overhead (simple slicing)
- Padding uses efficient numpy/torch operations
- Cache-friendly (MONAI caching works with split data)

**Disk:**
- No need to save split volumes separately
- Original single volume can be reused
- Optional: save masks for visualization

## Advanced: Custom Split Strategies

```python
from connectomics.data.utils.split import split_volume_train_val

# Strategy 1: Leave gap between train/val (avoid boundary effects)
def split_with_gap(volume_shape, train_ratio=0.8, gap_ratio=0.05):
    total_size = volume_shape[0]
    train_size = int(total_size * train_ratio)
    gap_size = int(total_size * gap_ratio)
    val_start = train_size + gap_size

    train_slices = (slice(0, train_size), slice(None), slice(None))
    val_slices = (slice(val_start, None), slice(None), slice(None))

    return train_slices, val_slices

# Strategy 2: Multiple validation regions
def split_multiple_val(volume_shape, train_ratio=0.7, num_val_regions=2):
    # Split into train (70%) and 2 validation regions (15% each)
    total = volume_shape[0]
    train_size = int(total * train_ratio)
    val_size = (total - train_size) // num_val_regions

    train_slices = (slice(0, train_size), slice(None), slice(None))
    val_slices = []

    for i in range(num_val_regions):
        start = train_size + i * val_size
        end = start + val_size
        val_slices.append((slice(start, end), slice(None), slice(None)))

    return train_slices, val_slices
```

## Summary

**Key Takeaways:**
1. ✅ Use `split_enabled: true` for automatic train/val splitting
2. ✅ Configure ranges as percentages: `[0.0, 0.8]` = first 80%
3. ✅ Enable padding for validation: `split_pad_val: true`
4. ✅ Choose appropriate axis (usually 0 for Z-axis)
5. ✅ Use reflection padding for best results
6. ✅ Validate split before training with utility functions

**When to Use:**
- ✅ Single-volume datasets
- ✅ Want reproducible splits
- ✅ Need automatic size handling
- ✅ Following DeepEM training strategies

**When NOT to Use:**
- ❌ Already have separate train/val files
- ❌ Need random/stratified splitting
- ❌ Multiple independent volumes

---

**See Also:**
- [DEEPEM_SUMMARY.md](DEEPEM_SUMMARY.md) - DeepEM codebase analysis
- [EM_AUGMENTATION_GUIDE.md](EM_AUGMENTATION_GUIDE.md) - EM augmentation guide
- [tutorials/examples/train_val_split.py](../tutorials/examples/train_val_split.py) - Example code
- [tutorials/lucchi_split.yaml](../tutorials/lucchi_split.yaml) - Example config
