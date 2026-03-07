# EM-Specific Augmentation Guide

PyTorch Connectomics provides **8 EM-specific augmentations** designed to simulate real artifacts in electron microscopy data. All are MONAI-compatible (`MapTransform`), Lightning-compatible, properly randomized, and support multiple keys.

## Transforms

### 1. RandMisAlignmentd
Simulates section misalignment (translation + rotation). Uses proper geometric transforms (cv2.warpAffine).
- `displacement`: max pixel displacement
- `rotate_ratio`: fraction using rotation vs translation (0.0-1.0)

### 2. RandMissingSectiond
Removes entire sections from volume (not just zero-filling). Avoids first/last sections.
- `num_sections`: number of sections to remove
- Note: Changes volume shape (z-dimension reduced)

### 3. RandMissingPartsd
Creates rectangular missing regions (holes) in random sections.
- `hole_range`: min/max hole size as fraction of section (e.g., `[0.1, 0.3]`)

### 4. RandMotionBlurd
Applies directional motion blur to simulate scan artifacts.
- `sections`: number of sections (or range tuple)
- `kernel_size`: blur kernel size in pixels

### 5. RandCutNoised
Adds noise to random cuboid regions (like Cutout/CutMix).
- `length_ratio`: size of cuboid as fraction of volume
- `noise_scale`: noise magnitude

### 6. RandCutBlurd
Downsamples cuboid regions to force super-resolution learning.
- `length_ratio`: size of cuboid as fraction of volume
- `down_ratio_range`: downsampling factor range (e.g., `[2.0, 8.0]`)
- `downsample_z`: whether to downsample z-axis

### 7. RandMixupd
Linearly interpolates between two samples for regularization. Requires batch size > 1.
- `alpha_range`: mixing ratio range (e.g., `[0.7, 0.9]`)

### 8. RandCopyPasted
Copies objects, transforms them, and pastes in non-overlapping regions. For instance segmentation.
- `label_key`: segmentation mask key
- `max_obj_ratio`: skip if object too large
- `rotation_angles`: list of rotation angles to try

## Augmentation Profiles

Use profiles in `tutorials/bases/augmentation_profiles.yaml`:

| Profile | Use Case |
|---------|----------|
| `aug_light` | Quick experiments, clean data, fine-tuning |
| `aug_standard` | Default for most 3D EM tasks (RECOMMENDED) |
| `aug_strong` | Small datasets, overfitting prevention |
| `aug_instance` | Instance segmentation (neuron, mito, synapse) |
| `aug_superres` | Super-resolution / multi-scale learning |

## Best Practices

1. **Start light, go heavy** -- use light augmentation for fast iteration, switch to heavy for final training
2. **Match augmentation to data artifacts** -- increase probability for transforms matching your data's noise characteristics
3. **Apply geometric transforms to both image and label**, intensity transforms to image only
4. **Combine with MONAI standard transforms** (RandShiftIntensityd, Rand3DElasticd, RandAffined)

## Implementation

- Pure functions: `connectomics/data/augment/augment_ops.py`
- MONAI wrappers: `connectomics/data/augment/transforms.py`
- Build pipeline: `connectomics/data/augment/build.py`
