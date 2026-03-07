# Train/Val Split Guide

Automatic train/val splitting from a single volume, following DeepEM's spatial splitting approach along the Z-axis.

## Configuration

```yaml
data:
  train_image: datasets/volume.h5
  train_label: datasets/label.h5
  val_image: null    # No separate validation files needed
  val_label: null

  split_enabled: true
  split_train_range: [0.0, 0.8]    # First 80% for training
  split_val_range: [0.8, 1.0]      # Last 20% for validation
  split_axis: 0                    # Z-axis (default)
  split_pad_val: true              # Pad validation to patch_size
  split_pad_mode: reflect          # Reflection padding (recommended)
```

## Split Ratios

| Dataset Size | Recommended |
|-------------|-------------|
| < 200 slices | 85/15 |
| 200-500 slices | 80/20 |
| > 500 slices | 90/10 |

## Python API

```python
from connectomics.data.dataset.split import split_volume_train_val, split_and_pad_volume

train_slices, val_slices = split_volume_train_val(volume.shape, train_ratio=0.8, axis=0)
train_data, val_data = split_and_pad_volume(volume, train_ratio=0.8, target_size=(32, 256, 256))
```

## When to Use

- Single-volume datasets
- Want reproducible, config-driven splits
- Need automatic size handling (padding)

## When NOT to Use

- Already have separate train/val files
- Need random/stratified splitting
- Multiple independent volumes
