# CellMap Challenge - Submission Guide: Instance vs Semantic Segmentation

## Overview

The CellMap Segmentation Challenge has **two types of tasks** with different evaluation criteria:
- **Semantic Segmentation**: Binary masks (0 or 1)
- **Instance Segmentation**: Unique IDs per object (0, 1, 2, 3, ...)

Understanding the difference is crucial for optimal submissions.

---

## Task Breakdown

### 📊 Challenge Statistics

- **Total predictions required**: **419 predictions**
- **Unique test crops**: **16 crops**
- **Total classes**: **47 classes**
- **Instance segmentation classes**: **11 classes** (harder, more important)
- **Semantic segmentation classes**: **36 classes** (easier)

---

## Instance vs Semantic Segmentation

### Semantic Segmentation (36 classes)

**Definition**: Predict **presence/absence** of structures

**Output**: Binary mask (0 = absent, 1 = present)

**Examples**:
- `ecs` - Extracellular space
- `pm` - Plasma membrane
- `cyto` - Cytoplasm
- `nucpl` - Nucleoplasm
- `er` - Endoplasmic reticulum (parent class)
- `golgi` - Golgi apparatus (parent class)
- All `_mem` and `_lum` subclasses (membranes, lumens)

**Evaluation Metrics**:
- Dice score
- IoU (Intersection over Union)
- Precision
- Recall
- F1 score

**Difficulty**: ✅ Easier - just detect presence

### Instance Segmentation (11 classes)

**Definition**: Predict **individual objects** with unique IDs

**Output**: Instance mask (0 = background, 1 = object #1, 2 = object #2, ...)

**Classes Requiring Instance Segmentation**:
1. **`nuc`** - Nucleus (nuclei in multi-nucleated cells)
2. **`vim`** - Vimentin filaments
3. **`ves`** - Vesicles (many small vesicles)
4. **`endo`** - Endosomes (multiple per cell)
5. **`lyso`** - Lysosomes (multiple per cell)
6. **`ld`** - Lipid droplets (multiple per cell)
7. **`perox`** - Peroxisomes (multiple per cell)
8. **`mito`** - Mitochondria ⭐ **Most important**
9. **`np`** - Nuclear pores (many per nucleus)
10. **`mt`** - Microtubules (complex network)
11. **`cell`** - Cell boundaries (multiple cells per volume)

**Evaluation Metrics**:
- Adapted Rand Error (ARE)
- Variation of Information (VOI) split/merge
- Panoptic quality

**Difficulty**: ⚠️ Harder - must separate individual objects

---

## Why `mednext_mito.py` is Special

### Mitochondria Instance Segmentation Challenges

**Why mitochondria are the hardest instance segmentation task:**

1. **High Density**
   - Many mitochondria in small volumes
   - Often 10-50+ per cell
   - Tightly packed in metabolically active regions

2. **Complex Morphology**
   - Tubular and branched structures
   - Dynamic shape changes
   - Network-like organization

3. **Touching/Overlapping**
   - Adjacent mitochondria often touch
   - Hard to determine boundaries
   - Easy to merge incorrectly

4. **Critical for Evaluation**
   - Mitochondria appear in 14/16 test crops
   - Instance metrics heavily weighted
   - Poor separation → poor scores

### Optimized Configuration

```python
# configs/mednext_mito.py
model_name = 'mednext'
mednext_size = 'L'              # 61.8M params - largest model
classes = ['mito']              # Single class - all model capacity
resolution = (4, 4, 4)          # 4nm isotropic - 2× higher than default
epochs = 1000                   # Extended training
batch_size = 1                  # Smaller due to high resolution + large model
```

**Why these choices?**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `mednext_size = 'L'` | 61.8M params | Best quality for single class |
| `resolution = (4, 4, 4)` | 4nm vs 8nm | See mitochondrial boundaries better |
| `epochs = 1000` | 2× longer | Instance seg needs more training |
| `batch_size = 1` | Reduced | Memory constraint at 4nm + large model |

---

## Submission Strategies

### 🎯 Three Approaches (Easy → Advanced)

#### Option 1: Semantic Masks Only (Easiest)

**What you do**:
```python
# Train model for binary segmentation
predictions = model(image)  # (C, D, H, W) float32 [0-1]
binary_masks = (predictions > 0.5).astype(np.uint8)

# Submit binary masks for ALL classes (semantic + instance)
save_predictions(binary_masks)
```

**What the server does**:
- For semantic classes: Evaluates as-is (Dice, IoU)
- For instance classes: **Automatically runs connected components** → creates instance IDs
- Computes instance metrics (ARE, VOI)

**Pros**:
- ✅ Simplest approach
- ✅ No post-processing needed
- ✅ Works out-of-box with our scripts

**Cons**:
- ⚠️ Touching objects may merge (lower instance scores)
- ⚠️ No control over instance separation

**When to use**: Quick baseline, first submission

---

#### Option 2: Connected Components Post-Processing (Better)

**What you do**:
```python
import cc3d

instance_classes = ['nuc', 'mito', 'ves', 'endo', 'lyso', 'ld', 'perox', 'np', 'mt', 'cell']

for cls_idx, cls_name in enumerate(classes):
    pred_mask = (predictions[cls_idx] > 0.5).astype(np.uint8)

    if cls_name in instance_classes:
        # Run connected components
        instance_mask = cc3d.connected_components(
            pred_mask,
            connectivity=26,  # 26-connected (3D)
        )
        # Output: 0, 1, 2, 3, ... (unique ID per object)

        # Optional: Filter small objects
        instance_mask = filter_by_size(instance_mask, min_size=100)

        save_instance(instance_mask, cls_name)
    else:
        # Semantic class - binary mask
        save_binary(pred_mask, cls_name)
```

**Pros**:
- ✅ Better than server's automatic connected components
- ✅ Can filter noise/small objects
- ✅ Still simple to implement

**Cons**:
- ⚠️ Still merges touching objects
- ⚠️ No intelligent separation

**When to use**: Second iteration, improved baseline

---

#### Option 3: Watershed Instance Segmentation (Best)

**What you do**:
```python
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import cc3d

def watershed_instance_segmentation(binary_mask, min_distance=10):
    """
    Advanced instance segmentation using watershed.

    Args:
        binary_mask: (D, H, W) binary mask
        min_distance: Minimum distance between object centers (in voxels)

    Returns:
        instance_mask: (D, H, W) with unique IDs per object
    """
    # 1. Compute distance transform
    distance = distance_transform_edt(binary_mask)

    # 2. Find local maxima (object centers)
    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary_mask,
        exclude_border=False,
    )

    # 3. Create markers from peaks
    markers = np.zeros(distance.shape, dtype=np.uint32)
    for i, coord in enumerate(coords):
        markers[tuple(coord)] = i + 1

    # Dilate markers slightly for stability
    markers = cc3d.connected_components(markers > 0)

    # 4. Run watershed
    instance_mask = watershed(
        -distance,           # Inverted distance (valleys become peaks)
        markers,             # Seeds
        mask=binary_mask,    # Constrain to foreground
    )

    return instance_mask


# Apply to instance classes
instance_classes = ['nuc', 'mito', 'ves', 'endo', 'lyso', 'ld', 'perox', 'np', 'mt', 'cell']

for cls_idx, cls_name in enumerate(classes):
    pred_mask = (predictions[cls_idx] > 0.5).astype(np.uint8)

    if cls_name in instance_classes:
        # Class-specific tuning
        if cls_name == 'mito':
            min_distance = 5   # Small mitochondria, close together
        elif cls_name == 'nuc':
            min_distance = 50  # Large nuclei, far apart
        else:
            min_distance = 10  # Default

        # Watershed segmentation
        instance_mask = watershed_instance_segmentation(pred_mask, min_distance)

        # Post-processing: remove small objects
        instance_mask = filter_by_size(instance_mask, min_size=100)

        save_instance(instance_mask, cls_name)
    else:
        save_binary(pred_mask, cls_name)
```

**Pros**:
- ✅ **Best quality** - separates touching objects
- ✅ **Better metrics** - Improved ARE, VOI scores
- ✅ **Tunable** - Adjust `min_distance` per class

**Cons**:
- ⚠️ More complex implementation
- ⚠️ Requires parameter tuning
- ⚠️ Slower post-processing

**When to use**: Final submission, maximum scores

---

## Implementation in Scripts

### Current Scripts (Option 1 - Easiest)

Our `predict_cellmap.py` currently implements **Option 1**:

```python
# In predict_cellmap.py
predictions = inferer(raw_tensor, model)
predictions = torch.sigmoid(predictions).cpu().numpy()[0]  # (C, D, H, W)

# Save binary predictions
for i, cls in enumerate(classes):
    pred_array = (predictions[i] > 0.5).astype(np.uint8)
    save_zarr(pred_array, cls)
```

**This works fine** because:
- Server automatically runs connected components on instance classes
- Gets you started quickly
- Valid submission

### Upgrade to Option 2 or 3

**Create**: `scripts/cellmap/postprocess_instances.py`

```python
#!/usr/bin/env python
"""
Post-process predictions for better instance segmentation.

Upgrades binary masks to instance masks using connected components or watershed.

Usage:
    python scripts/cellmap/postprocess_instances.py \
        --predictions outputs/cellmap_cos7/predictions \
        --output outputs/cellmap_cos7/predictions_processed \
        --method watershed
"""

import os
import zarr
import numpy as np
import cc3d
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from tqdm import tqdm

# Instance classes
INSTANCE_CLASSES = ['nuc', 'vim', 'ves', 'endo', 'lyso', 'ld', 'perox', 'mito', 'np', 'mt', 'cell']

# Class-specific parameters
MIN_DISTANCE = {
    'nuc': 50,     # Large, far apart
    'mito': 5,     # Small, close together
    'ves': 8,      # Small vesicles
    'ld': 10,      # Lipid droplets
    'perox': 10,   # Peroxisomes
    'endo': 10,    # Endosomes
    'lyso': 10,    # Lysosomes
    'np': 5,       # Nuclear pores
    'mt': 15,      # Microtubules
    'cell': 100,   # Whole cells
    'vim': 20,     # Vimentin
}

MIN_SIZE = {
    'nuc': 1000,   # Large
    'mito': 50,    # Small
    'ves': 20,     # Tiny
    'ld': 50,      # Small
    'perox': 50,   # Small
    'endo': 50,    # Small
    'lyso': 50,    # Small
    'np': 10,      # Tiny
    'mt': 100,     # Filaments
    'cell': 5000,  # Very large
    'vim': 100,    # Filaments
}


def filter_by_size(instance_mask, min_size):
    """Remove objects smaller than min_size voxels."""
    unique_ids, counts = np.unique(instance_mask, return_counts=True)

    for uid, count in zip(unique_ids, counts):
        if uid == 0:  # Skip background
            continue
        if count < min_size:
            instance_mask[instance_mask == uid] = 0

    # Relabel to consecutive IDs
    instance_mask = cc3d.connected_components(instance_mask > 0)
    return instance_mask


def connected_components_instance(binary_mask, min_size):
    """Simple connected components."""
    instance_mask = cc3d.connected_components(binary_mask, connectivity=26)
    instance_mask = filter_by_size(instance_mask, min_size)
    return instance_mask


def watershed_instance(binary_mask, min_distance, min_size):
    """Watershed instance segmentation."""
    if binary_mask.sum() == 0:
        return binary_mask.astype(np.uint32)

    # Distance transform
    distance = distance_transform_edt(binary_mask)

    # Find peaks
    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary_mask,
        exclude_border=False,
    )

    if len(coords) == 0:
        # No peaks found - use connected components
        return connected_components_instance(binary_mask, min_size)

    # Create markers
    markers = np.zeros(distance.shape, dtype=np.uint32)
    for i, coord in enumerate(coords):
        markers[tuple(coord)] = i + 1

    # Watershed
    instance_mask = watershed(-distance, markers, mask=binary_mask)

    # Filter small objects
    instance_mask = filter_by_size(instance_mask, min_size)

    return instance_mask


def process_predictions(predictions_dir, output_dir, method='watershed'):
    """
    Post-process predictions to create instance masks.

    Args:
        predictions_dir: Input directory with binary predictions
        output_dir: Output directory for instance predictions
        method: 'cc' (connected components) or 'watershed'
    """

    os.makedirs(output_dir, exist_ok=True)

    # Find all prediction files
    from pathlib import Path
    pred_paths = list(Path(predictions_dir).rglob('*.zarr/*/s0'))

    print(f"Found {len(pred_paths)} predictions")

    for pred_path in tqdm(pred_paths, desc="Post-processing"):
        # Parse path: .../dataset/crop/class/s0
        parts = str(pred_path).split('/')
        class_name = parts[-2]
        crop_name = parts[-3]
        dataset_name = parts[-4]

        # Load prediction
        pred_array = zarr.open(str(pred_path), mode='r')[:]

        # Check if instance class
        if class_name in INSTANCE_CLASSES:
            min_dist = MIN_DISTANCE.get(class_name, 10)
            min_sz = MIN_SIZE.get(class_name, 100)

            if method == 'watershed':
                instance_mask = watershed_instance(pred_array, min_dist, min_sz)
            else:  # connected components
                instance_mask = connected_components_instance(pred_array, min_sz)

            output_array = instance_mask
        else:
            # Semantic class - keep binary
            output_array = pred_array

        # Save
        output_path = f"{output_dir}/{dataset_name}/{crop_name}/{class_name}/s0"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        zarr_out = zarr.open(
            output_path,
            mode='w',
            shape=output_array.shape,
            dtype=output_array.dtype,
            chunks=(64, 64, 64),
        )
        zarr_out[:] = output_array

        # Copy metadata
        src_attrs = zarr.open(str(pred_path), mode='r').attrs
        for k, v in src_attrs.items():
            zarr_out.attrs[k] = v

    print(f"Post-processing complete: {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', choices=['cc', 'watershed'], default='watershed')
    args = parser.parse_args()

    process_predictions(args.predictions, args.output, args.method)
```

---

## Recommended Workflow

### Phase 1: Quick Baseline (Option 1)

```bash
# 1. Train model
python scripts/cellmap/train_cellmap.py scripts/cellmap/configs/mednext_cos7.py

# 2. Predict (binary masks)
python scripts/cellmap/predict_cellmap.py \
    --checkpoint outputs/cellmap_cos7/checkpoints/best.ckpt \
    --config scripts/cellmap/configs/mednext_cos7.py

# 3. Submit (server runs connected components)
python scripts/cellmap/submit_cellmap.py \
    --predictions outputs/cellmap_cos7/predictions
```

**Expected scores**: Decent baseline (60-70% Dice for semantic, 40-60% ARE for instance)

---

### Phase 2: Improved Instance Segmentation (Option 2/3)

```bash
# 1. Post-process predictions
python scripts/cellmap/postprocess_instances.py \
    --predictions outputs/cellmap_cos7/predictions \
    --output outputs/cellmap_cos7/predictions_watershed \
    --method watershed

# 2. Submit improved predictions
python scripts/cellmap/submit_cellmap.py \
    --predictions outputs/cellmap_cos7/predictions_watershed
```

**Expected scores**: +5-15% improvement on instance metrics

---

### Phase 3: Dedicated Instance Models (Best)

```bash
# 1. Train high-resolution mitochondria model
python scripts/cellmap/train_cellmap.py scripts/cellmap/configs/mednext_mito.py

# 2. Predict at 4nm resolution
python scripts/cellmap/predict_cellmap.py \
    --checkpoint outputs/cellmap_mito/checkpoints/best.ckpt \
    --config scripts/cellmap/configs/mednext_mito.py

# 3. Post-process with watershed
python scripts/cellmap/postprocess_instances.py \
    --predictions outputs/cellmap_mito/predictions \
    --method watershed

# 4. Combine with multi-class predictions
# (mito from dedicated model + other classes from multi-class model)

# 5. Submit combined predictions
python scripts/cellmap/submit_cellmap.py \
    --predictions outputs/cellmap_combined
```

**Expected scores**: Best possible (70-85% ARE for mitochondria)

---

## Key Insights

### 1. Server Handles Instance ID Assignment

**Important**: You can submit binary masks for all classes!

The evaluation server will:
- Run connected components on instance classes
- Assign unique IDs automatically
- Compute instance metrics

**This means**:
- ✅ Quick start without post-processing
- ✅ Valid baseline submission
- ⚠️ Suboptimal instance separation

### 2. Instance Classes Are More Important

Instance segmentation classes appear more frequently in test set:
- `mito`: 14/16 crops
- `nuc`: 16/16 crops
- `ld`: 13/16 crops
- `perox`: 12/16 crops

**These classes dominate your final score!**

### 3. Resolution Matters for Instance Segmentation

Higher resolution helps see boundaries:
- **8nm**: OK for semantic, marginal for instance
- **4nm**: Better for instance, can see mitochondrial cristae
- **2nm**: Best, but very expensive (memory/compute)

**Recommendation**: Train instance-critical classes at 4nm

### 4. Model Capacity Matters

Larger models → better instance separation:
- **MedNeXt-S (5.6M)**: OK for semantic
- **MedNeXt-B (10.5M)**: Good for multi-class
- **MedNeXt-M (17.6M)**: Better for multi-class with instances
- **MedNeXt-L (61.8M)**: Best for single-class instance

**Trade-off**: Memory vs. quality

---

## Summary Table

| Approach | Effort | Quality | When to Use |
|----------|--------|---------|-------------|
| **Option 1: Binary masks** | ✅ Low | ⭐⭐ | First submission, baseline |
| **Option 2: Connected components** | ✅ Medium | ⭐⭐⭐ | Second iteration |
| **Option 3: Watershed** | ⚠️ High | ⭐⭐⭐⭐ | Final submission |
| **Dedicated instance models** | ⚠️ Very High | ⭐⭐⭐⭐⭐ | Maximum scores |

---

## FAQ

### Q: Do I need to submit instance IDs?

**A**: No! You can submit binary masks for all classes. The server will run connected components on instance classes automatically.

### Q: Why is mitochondria so hard?

**A**: High density, complex shapes, touching objects, critical for evaluation (14/16 test crops).

### Q: Should I use watershed or connected components?

**A**:
- Start with **server's automatic connected components** (easiest)
- Upgrade to **your own connected components** (better filtering)
- Final submission: **watershed** (best separation)

### Q: Can I mix binary and instance predictions?

**A**: Yes! Submit binary for semantic classes, instance IDs for instance classes. The evaluation handles both.

### Q: What's the best strategy?

**A**:
1. Quick baseline: Binary masks, server handles instances
2. Improved: Post-process with watershed
3. Best: Dedicated high-res models for critical classes (mito, nuc)

---

## References

- [CellMap Challenge FAQ](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/blob/main/FAQ.md#is-the-downloaded-data-intended-for-semantic-segmentation-and-how-can-i-obtain-labels-for-instance-segmentation)
- [Submission Format](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/blob/main/docs/source/submission_data_format.rst)
- [Test Crop Manifest](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/blob/main/src/cellmap_segmentation_challenge/utils/test_crop_manifest.csv)

---

## Next Steps

1. ✅ Train multi-class model (`mednext_cos7.py`)
2. ✅ Submit baseline with binary masks
3. ⚡ Implement watershed post-processing
4. ⚡ Train dedicated instance model (`mednext_mito.py`)
5. 🏆 Submit optimized predictions

Good luck! 🚀
