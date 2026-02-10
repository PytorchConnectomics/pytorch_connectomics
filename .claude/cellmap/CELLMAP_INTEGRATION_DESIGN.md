# CellMap Challenge Integration with PyTorch Connectomics

## Design Plan for Training/Testing on CellMap Data

This document outlines the strategy for integrating CellMap Segmentation Challenge data with the PyTorch Connectomics (PyTC) framework while leveraging PyTC's modern architecture (PyTorch Lightning + MONAI + Hydra).

---

## Executive Summary

**Goal**: Train and evaluate PyTC models on CellMap challenge data using PyTC's Lightning/MONAI/Hydra stack.

**Strategy**: Build a bridge layer that adapts CellMap's Zarr data to PyTC's data pipeline while maintaining PyTC's architecture philosophy.

**Key Principle**: Use PyTC's existing infrastructure (Lightning orchestration, MONAI transforms/models, Hydra configs) rather than reimplementing CellMap's custom components.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PyTorch Connectomics Stack                    │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │   Hydra    │  │   Lightning  │  │        MONAI           │  │
│  │  Configs   │  │ Orchestration│  │  Models & Transforms   │  │
│  └────────────┘  └──────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │
                    ┌───────┴────────┐
                    │  Bridge Layer  │
                    │  (NEW)         │
                    └───────┬────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CellMap Data Layer                          │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │    Zarr    │  │ cellmap-data │  │   Multi-scale Crops    │  │
│  │   Stores   │  │   Package    │  │   (23 datasets)        │  │
│  └────────────┘  └──────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Data Integration (Priority: HIGH)

#### 1.1 Create CellMap Dataset Adapter

**File**: `connectomics/data/dataset/dataset_cellmap.py`

**Purpose**: Adapt CellMap Zarr data to PyTC's dataset interface

**Design**:
```python
from connectomics.data.dataset.dataset_base import BaseDataset
import zarr
from typing import List, Dict, Optional, Tuple
import numpy as np

class CellMapDataset(BaseDataset):
    """
    Dataset for CellMap Segmentation Challenge data.

    Loads multi-scale Zarr arrays with sparse annotations across multiple crops.
    Compatible with MONAI transforms and PyTorch Lightning DataModule.
    """

    def __init__(
        self,
        data_root: str,                          # /projects/weilab/dataset/cellmap
        datasets: List[str],                      # ['jrc_cos7-1a', 'jrc_hela-2']
        crops: Optional[List[str]] = None,        # ['crop234', 'crop236'] or None (auto-discover)
        classes: List[str] = ['nuc', 'mito'],     # Label classes to load
        mode: str = 'train',                      # 'train', 'val', 'test'
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        resolution: Tuple[float, float, float] = (8.0, 8.0, 8.0),  # nm voxel size
        augmentor=None,                           # MONAI transforms
        mode_kwargs: Dict = {},
    ):
        """
        Args:
            data_root: Root directory containing dataset folders
            datasets: List of dataset names to include
            crops: List of crop names (e.g., 'crop234'). If None, auto-discover all crops.
            classes: List of organelle classes to segment
            mode: Dataset mode (train/val/test)
            patch_size: Size of random crops in voxels
            resolution: Target voxel size in nm (will select appropriate scale level)
            augmentor: MONAI transform pipeline
            mode_kwargs: Additional mode-specific arguments
        """
        super().__init__(augmentor, mode)

        self.data_root = data_root
        self.datasets = datasets
        self.classes = classes
        self.patch_size = patch_size
        self.resolution = resolution

        # Discover data
        self.data_samples = self._discover_data(datasets, crops, classes)

        # Build Zarr store handles (lazy loading)
        self.zarr_stores = {}

    def _discover_data(self, datasets, crops, classes):
        """
        Scan datasets and build list of (dataset, crop, class) tuples.

        Returns:
            List of dicts with keys: 'dataset', 'crop', 'class', 'raw_path', 'label_path', 'scale_level'
        """
        samples = []
        for dataset in datasets:
            zarr_path = f"{self.data_root}/{dataset}/{dataset}.zarr"

            # Auto-discover crops if not specified
            if crops is None:
                gt_path = f"{zarr_path}/recon-1/labels/groundtruth"
                crop_list = [d for d in os.listdir(gt_path) if d.startswith('crop')]
            else:
                crop_list = crops

            for crop in crop_list:
                # Check which classes exist in this crop
                crop_path = f"{zarr_path}/recon-1/labels/groundtruth/{crop}"
                available_classes = [c for c in classes if os.path.exists(f"{crop_path}/{c}")]

                for cls in available_classes:
                    # Find appropriate scale level for target resolution
                    scale_level = self._find_scale_level(zarr_path, crop, cls, self.resolution)

                    samples.append({
                        'dataset': dataset,
                        'crop': crop,
                        'class': cls,
                        'raw_path': f"{zarr_path}/recon-1/em/fibsem-uint8/s{scale_level}",
                        'label_path': f"{crop_path}/{cls}/s{scale_level}",
                        'scale_level': scale_level,
                    })

        return samples

    def _find_scale_level(self, zarr_path, crop, cls, target_resolution):
        """
        Find the scale level (s0, s1, ...) that matches target resolution.

        CellMap uses OME-NGFF multiscale metadata.
        """
        label_path = f"{zarr_path}/recon-1/labels/groundtruth/{crop}/{cls}"
        store = zarr.open(label_path, mode='r')

        # Read OME-NGFF multiscale metadata
        multiscale_meta = store.attrs.get('multiscales', [{}])[0]
        datasets_meta = multiscale_meta.get('datasets', [])

        # Find closest scale to target resolution
        best_scale = 0
        min_diff = float('inf')

        for i, ds_meta in enumerate(datasets_meta):
            scale = ds_meta.get('coordinateTransformations', [{}])[0].get('scale', [1, 1, 1])
            # scale is [z, y, x] in nm
            avg_resolution = np.mean(scale)
            diff = abs(avg_resolution - np.mean(target_resolution))

            if diff < min_diff:
                min_diff = diff
                best_scale = i

        return best_scale

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        """
        Returns a random patch from the dataset.

        Returns:
            dict: {
                'image': (1, D, H, W) float32 tensor,
                'label': (C, D, H, W) float32 tensor (C = num classes),
            }
        """
        sample_info = self.data_samples[idx]

        # Load Zarr arrays (lazy)
        raw_array = zarr.open(sample_info['raw_path'], mode='r')
        label_array = zarr.open(sample_info['label_path'], mode='r')

        # Get array shapes
        raw_shape = raw_array.shape  # (Z, Y, X)
        label_shape = label_array.shape

        # Random crop location (ensure within bounds)
        z_max = max(1, raw_shape[0] - self.patch_size[0])
        y_max = max(1, raw_shape[1] - self.patch_size[1])
        x_max = max(1, raw_shape[2] - self.patch_size[2])

        z_start = np.random.randint(0, z_max)
        y_start = np.random.randint(0, y_max)
        x_start = np.random.randint(0, x_max)

        z_end = min(z_start + self.patch_size[0], raw_shape[0])
        y_end = min(y_start + self.patch_size[1], raw_shape[1])
        x_end = min(x_start + self.patch_size[2], raw_shape[2])

        # Load patches (Zarr loads only requested chunks - efficient!)
        raw_patch = raw_array[z_start:z_end, y_start:y_end, x_start:x_end]
        label_patch = label_array[z_start:z_end, y_start:y_end, x_start:x_end]

        # Convert to float32 and add channel dimension
        raw_patch = raw_patch.astype(np.float32) / 255.0  # Normalize uint8 to [0, 1]
        raw_patch = raw_patch[np.newaxis, ...]  # (1, D, H, W)

        # Binarize labels (CellMap has instance IDs, convert to binary masks)
        label_patch = (label_patch > 0).astype(np.float32)
        label_patch = label_patch[np.newaxis, ...]  # (1, D, H, W)

        # Apply MONAI transforms
        if self.augmentor is not None:
            data_dict = {'image': raw_patch, 'label': label_patch}
            data_dict = self.augmentor(data_dict)
            return data_dict

        return {'image': raw_patch, 'label': label_patch}
```

**Key Features**:
- ✅ Lazy Zarr loading (memory efficient)
- ✅ Multi-scale resolution matching
- ✅ Auto-discovery of crops and classes
- ✅ Compatible with MONAI transforms
- ✅ Random patch sampling for training
- ✅ Handles sparse annotations (not all crops have all classes)

---

#### 1.2 Register Dataset in Factory

**File**: `connectomics/data/dataset/build.py`

**Modification**:
```python
def get_dataset(cfg, augmentor, mode='train'):
    """Dataset factory with CellMap support."""

    # ... existing code ...

    # Add CellMap dataset
    elif cfg.dataset.name == 'cellmap':
        from .dataset_cellmap import CellMapDataset
        return CellMapDataset(
            data_root=cfg.dataset.data_root,
            datasets=cfg.dataset.datasets,
            crops=cfg.dataset.get('crops', None),
            classes=cfg.dataset.classes,
            mode=mode,
            patch_size=tuple(cfg.data.patch_size),
            resolution=tuple(cfg.dataset.resolution),
            augmentor=augmentor,
        )

    # ... existing code ...
```

---

#### 1.3 Create Multi-Class Dataset Wrapper

**Challenge**: CellMap dataset returns single class per sample, but we need multi-class predictions.

**Solution**: Wrapper that loads multiple classes per patch

**File**: `connectomics/data/dataset/dataset_cellmap.py` (add class)

```python
class CellMapMultiClassDataset(BaseDataset):
    """
    Multi-class variant that loads all specified classes for each crop.

    Returns patches with C-channel labels (C = number of classes).
    """

    def __init__(self, data_root, datasets, crops, classes, mode, patch_size, resolution, augmentor=None):
        super().__init__(augmentor, mode)

        self.data_root = data_root
        self.datasets = datasets
        self.classes = classes
        self.patch_size = patch_size
        self.resolution = resolution

        # Group samples by (dataset, crop)
        self.crop_groups = self._group_by_crop(datasets, crops, classes)

    def _group_by_crop(self, datasets, crops, classes):
        """
        Returns: List of dicts with keys 'dataset', 'crop', 'raw_path', 'label_paths' (dict of class->path)
        """
        groups = {}

        for dataset in datasets:
            zarr_path = f"{self.data_root}/{dataset}/{dataset}.zarr"

            if crops is None:
                gt_path = f"{zarr_path}/recon-1/labels/groundtruth"
                crop_list = [d for d in os.listdir(gt_path) if d.startswith('crop')]
            else:
                crop_list = crops

            for crop in crop_list:
                key = (dataset, crop)
                crop_path = f"{zarr_path}/recon-1/labels/groundtruth/{crop}"

                # Find scale level (use first available class as reference)
                scale_level = None
                label_paths = {}

                for cls in classes:
                    cls_path = f"{crop_path}/{cls}"
                    if os.path.exists(cls_path):
                        if scale_level is None:
                            scale_level = self._find_scale_level(zarr_path, crop, cls, self.resolution)
                        label_paths[cls] = f"{cls_path}/s{scale_level}"

                if label_paths:  # Only add if at least one class exists
                    groups[key] = {
                        'dataset': dataset,
                        'crop': crop,
                        'raw_path': f"{zarr_path}/recon-1/em/fibsem-uint8/s{scale_level}",
                        'label_paths': label_paths,
                        'scale_level': scale_level,
                    }

        return list(groups.values())

    def __len__(self):
        return len(self.crop_groups)

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'image': (1, D, H, W) float32,
                'label': (C, D, H, W) float32,  # C = len(classes)
            }
        """
        group = self.crop_groups[idx]

        # Load raw
        raw_array = zarr.open(group['raw_path'], mode='r')
        raw_shape = raw_array.shape

        # Random crop location
        z_start, y_start, x_start = self._random_crop_location(raw_shape)
        z_end = z_start + self.patch_size[0]
        y_end = y_start + self.patch_size[1]
        x_end = x_start + self.patch_size[2]

        # Load raw patch
        raw_patch = raw_array[z_start:z_end, y_start:y_end, x_start:x_end]
        raw_patch = raw_patch.astype(np.float32) / 255.0
        raw_patch = raw_patch[np.newaxis, ...]  # (1, D, H, W)

        # Load all label patches
        label_patches = []
        for cls in self.classes:
            if cls in group['label_paths']:
                label_array = zarr.open(group['label_paths'][cls], mode='r')
                label_patch = label_array[z_start:z_end, y_start:y_end, x_start:x_end]
                label_patch = (label_patch > 0).astype(np.float32)  # Binarize
            else:
                # Class not annotated in this crop - use zeros
                label_patch = np.zeros(self.patch_size, dtype=np.float32)

            label_patches.append(label_patch)

        # Stack into (C, D, H, W)
        label_patches = np.stack(label_patches, axis=0)

        # Apply transforms
        if self.augmentor is not None:
            data_dict = {'image': raw_patch, 'label': label_patches}
            data_dict = self.augmentor(data_dict)
            return data_dict

        return {'image': raw_patch, 'label': label_patches}
```

---

### Phase 2: Configuration (Priority: HIGH)

#### 2.1 Create Hydra Config Template

**File**: `tutorials/cellmap_base.yaml`

```yaml
# CellMap Segmentation Challenge - Base Configuration
# PyTorch Connectomics training on CellMap data

system:
  num_gpus: 1
  num_cpus: 8
  seed: 42

dataset:
  name: cellmap                                    # Use CellMapMultiClassDataset
  data_root: /projects/weilab/dataset/cellmap
  datasets:                                         # List of dataset names to include
    - jrc_cos7-1a
    - jrc_hela-2
  crops: null                                       # null = auto-discover all crops
  classes:                                          # Organelle classes to segment
    - nuc
    - mito
    - er
  resolution: [8.0, 8.0, 8.0]                      # Target voxel size in nm
  train_val_split: 0.85                            # 85% train, 15% val

model:
  architecture: mednext                             # Use MedNeXt (SOTA for medical imaging)
  in_channels: 1                                    # Grayscale EM
  out_channels: 3                                   # 3 classes (nuc, mito, er)
  mednext_size: B                                   # Base size (10.5M params)
  mednext_kernel_size: 5                            # 5x5x5 kernels
  deep_supervision: true                            # Multi-scale loss

  loss_functions:
    - DiceLoss                                      # Soft Dice
    - FocalLoss                                     # Handle class imbalance
  loss_weights: [1.0, 1.0]

data:
  patch_size: [128, 128, 128]                      # Training patch size (voxels)
  batch_size: 2                                     # Small for 3D
  num_workers: 8
  persistent_workers: true
  use_cache: false                                  # Too large for caching

  # MONAI augmentations
  augmentations:
    RandFlipd:
      keys: [image, label]
      prob: 0.5
      spatial_axis: [0, 1, 2]

    RandRotate90d:
      keys: [image, label]
      prob: 0.5
      spatial_axes: [[0, 1], [0, 2], [1, 2]]

    RandAffined:
      keys: [image, label]
      prob: 0.5
      rotate_range: [0.1, 0.1, 0.1]               # Small rotations
      scale_range: [0.1, 0.1, 0.1]                # Small scaling
      mode: [bilinear, nearest]

    RandGaussianNoised:
      keys: [image]
      prob: 0.2
      mean: 0.0
      std: 0.1

    RandGaussianSmoothd:
      keys: [image]
      prob: 0.2
      sigma_x: [0.5, 1.5]
      sigma_y: [0.5, 1.5]
      sigma_z: [0.5, 1.5]

optimizer:
  name: AdamW
  lr: 1e-3                                          # MedNeXt recommended
  weight_decay: 1e-5

scheduler:
  name: constant                                    # MedNeXt uses constant LR
  warmup_epochs: 0

training:
  max_epochs: 1000
  precision: "16-mixed"                             # Mixed precision
  gradient_clip_val: 1.0
  accumulate_grad_batches: 4                        # Effective batch_size = 8

checkpoint:
  monitor: "val/dice"
  mode: "max"
  save_top_k: 3
  save_last: true

early_stopping:
  monitor: "val/dice"
  patience: 50
  mode: "max"

logging:
  use_tensorboard: true
  use_wandb: false
  log_every_n_steps: 50
```

---

#### 2.2 Create Dataset-Specific Configs

**File**: `tutorials/cellmap_cos7.yaml`

```yaml
# CellMap COS7 cells - Multi-organelle segmentation

defaults:
  - cellmap_base

dataset:
  datasets:
    - jrc_cos7-1a
    - jrc_cos7-1b
  classes:
    - nuc       # Nucleus
    - mito      # Mitochondria
    - er        # Endoplasmic reticulum
    - golgi     # Golgi apparatus
    - ves       # Vesicles
  resolution: [8.0, 8.0, 8.0]

model:
  out_channels: 5                                   # 5 classes
  mednext_size: M                                   # Medium (17.6M params)

training:
  max_epochs: 500
```

**File**: `tutorials/cellmap_mito.yaml`

```yaml
# CellMap Mitochondria - Instance segmentation

defaults:
  - cellmap_base

dataset:
  datasets:
    - jrc_cos7-1a
    - jrc_hela-2
    - jrc_jurkat-1
    - jrc_mus-liver
  classes:
    - mito
  resolution: [4.0, 4.0, 4.0]                      # Higher resolution for small structures

model:
  out_channels: 1
  mednext_size: L                                   # Large model (61.8M params)

  # For instance segmentation, predict semantic mask + post-process with watershed
  loss_functions:
    - DiceLoss
    - BCEWithLogitsLoss
  loss_weights: [1.0, 1.0]

data:
  patch_size: [96, 96, 96]                         # Smaller patches for higher resolution

optimizer:
  lr: 5e-4                                          # Lower LR for larger model

training:
  max_epochs: 1000
```

---

### Phase 3: Training Pipeline (Priority: MEDIUM)

#### 3.1 Training Script

**File**: `scripts/train_cellmap.py` (optional convenience wrapper)

```python
#!/usr/bin/env python
"""
Training script for CellMap Segmentation Challenge.

Usage:
    python scripts/train_cellmap.py --config tutorials/cellmap_cos7.yaml

Or use main.py directly:
    python scripts/main.py --config tutorials/cellmap_cos7.yaml
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.main import main

if __name__ == "__main__":
    main()
```

**Usage**:
```bash
# Activate environment
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc

# Train with base config
python scripts/main.py --config tutorials/cellmap_cos7.yaml

# Override parameters
python scripts/main.py --config tutorials/cellmap_cos7.yaml \
    data.batch_size=4 \
    training.max_epochs=200 \
    model.mednext_size=S

# Multi-GPU training
python scripts/main.py --config tutorials/cellmap_cos7.yaml \
    system.num_gpus=4

# Resume from checkpoint
python scripts/main.py --config tutorials/cellmap_cos7.yaml \
    --checkpoint outputs/cellmap_cos7/version_0/checkpoints/last.ckpt
```

---

### Phase 4: Inference & Evaluation (Priority: MEDIUM)

#### 4.1 Inference Pipeline

**Approach**: Use PyTC's existing Lightning inference module with CellMap data

**File**: `connectomics/lightning/inference.py` (existing, may need minor updates)

**Usage**:
```bash
# Predict on test crops
python scripts/main.py --config tutorials/cellmap_cos7.yaml \
    --mode test \
    --checkpoint outputs/cellmap_cos7/version_0/checkpoints/best.ckpt \
    dataset.crops=['crop234', 'crop236']
```

**Modifications Needed**:
```python
# In ConnectomicsModule.predict_step() or separate inference script

def predict_cellmap_crops(model, config, checkpoint_path, output_dir):
    """
    Run inference on CellMap crops and save predictions in challenge format.

    Args:
        model: Trained Lightning module
        config: Hydra config
        checkpoint_path: Path to model checkpoint
        output_dir: Where to save predictions (Zarr format)
    """

    # Load checkpoint
    model = ConnectomicsModule.load_from_checkpoint(checkpoint_path, cfg=config)
    model.eval()
    model.cuda()

    # Iterate over test crops
    for dataset_name in config.dataset.datasets:
        for crop_name in config.dataset.crops:
            # Load full crop volume
            zarr_path = f"{config.dataset.data_root}/{dataset_name}/{dataset_name}.zarr"
            raw_array = zarr.open(f"{zarr_path}/recon-1/em/fibsem-uint8/s{scale_level}", mode='r')

            # Sliding window inference with overlap (MONAI)
            from monai.inferers import SlidingWindowInferer

            inferer = SlidingWindowInferer(
                roi_size=config.data.patch_size,
                sw_batch_size=4,
                overlap=0.5,
                mode='gaussian',
            )

            # Run inference
            with torch.no_grad():
                predictions = inferer(
                    inputs=torch.from_numpy(raw_array[:]).cuda(),
                    network=model,
                )

            # Save predictions in CellMap format (Zarr)
            save_cellmap_predictions(
                predictions=predictions.cpu().numpy(),
                output_path=f"{output_dir}/{dataset_name}.zarr/{crop_name}",
                classes=config.dataset.classes,
            )
```

---

#### 4.2 Post-Processing for Instance Segmentation

**File**: `connectomics/decoding/cellmap_postprocess.py` (NEW)

```python
"""
Post-processing for CellMap instance segmentation.

Converts semantic predictions to instance IDs using connected components or watershed.
"""

import numpy as np
import cc3d
from scipy import ndimage


def semantic_to_instance(semantic_mask, min_size=100, method='cc3d'):
    """
    Convert binary semantic mask to instance segmentation.

    Args:
        semantic_mask: (D, H, W) binary mask
        min_size: Minimum object size in voxels
        method: 'cc3d' (connected components) or 'watershed'

    Returns:
        instance_mask: (D, H, W) with unique IDs per object
    """

    if method == 'cc3d':
        # Simple connected components
        instance_mask = cc3d.connected_components(semantic_mask.astype(np.uint8))

        # Filter by size
        instance_mask = filter_by_size(instance_mask, min_size)

    elif method == 'watershed':
        # Distance transform + watershed for better separation
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max

        # Compute distance transform
        distance = ndimage.distance_transform_edt(semantic_mask)

        # Find peaks (object centers)
        coords = peak_local_max(distance, min_distance=10, labels=semantic_mask)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = ndimage.label(mask)[0]

        # Watershed
        instance_mask = watershed(-distance, markers, mask=semantic_mask)

        # Filter by size
        instance_mask = filter_by_size(instance_mask, min_size)

    return instance_mask


def filter_by_size(instance_mask, min_size):
    """Remove objects smaller than min_size voxels."""
    unique_ids, counts = np.unique(instance_mask, return_counts=True)

    for uid, count in zip(unique_ids, counts):
        if uid == 0:  # Skip background
            continue
        if count < min_size:
            instance_mask[instance_mask == uid] = 0

    return instance_mask
```

---

#### 4.3 Evaluation Metrics

**File**: `connectomics/metrics/cellmap_metrics.py` (NEW)

```python
"""
CellMap challenge evaluation metrics.

Implements official challenge metrics:
- Semantic: Dice, IoU, precision, recall
- Instance: Adapted Rand Error, VOI split/merge
"""

from connectomics.metrics.metrics_seg import get_dice, get_iou
import numpy as np


def evaluate_semantic(pred, gt):
    """
    Evaluate semantic segmentation.

    Args:
        pred: (D, H, W) binary prediction
        gt: (D, H, W) binary groundtruth

    Returns:
        dict: {'dice': float, 'iou': float, 'precision': float, 'recall': float}
    """
    dice = get_dice(pred, gt)
    iou = get_iou(pred, gt)

    tp = np.sum((pred > 0) & (gt > 0))
    fp = np.sum((pred > 0) & (gt == 0))
    fn = np.sum((pred == 0) & (gt > 0))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
    }


def evaluate_instance(pred_instances, gt_instances):
    """
    Evaluate instance segmentation using Adapted Rand Error.

    Requires funlib.evaluate package (optional dependency).
    """
    try:
        from funlib.evaluate import rand_voi

        metrics = rand_voi(
            gt_instances,
            pred_instances,
            return_results=True,
        )

        return {
            'rand_error': metrics['rand_error'],
            'voi_split': metrics['voi_split'],
            'voi_merge': metrics['voi_merge'],
        }

    except ImportError:
        print("Warning: funlib.evaluate not installed. Skipping instance metrics.")
        return {}
```

---

### Phase 5: Submission Preparation (Priority: LOW)

#### 5.1 Export to Challenge Format

**File**: `scripts/export_cellmap_submission.py` (NEW)

```python
"""
Export PyTC predictions to CellMap challenge submission format.

Converts predictions from PyTC output to required Zarr structure.
"""

import zarr
import numpy as np
from pathlib import Path


def export_submission(
    predictions_dir: str,
    output_path: str,
    classes: list,
    datasets: list,
):
    """
    Package predictions for CellMap submission.

    Expected input structure:
        predictions_dir/
        ├── jrc_cos7-1a/
        │   ├── crop234_nuc.npy
        │   ├── crop234_mito.npy
        │   └── ...
        └── ...

    Output structure (CellMap format):
        output_path/
        ├── jrc_cos7-1a.zarr/
        │   └── crop234/
        │       ├── nuc/
        │       │   └── s0/  (Zarr array)
        │       ├── mito/
        │       └── ...
        └── ...
    """

    for dataset in datasets:
        dataset_dir = Path(predictions_dir) / dataset

        # Find all prediction files
        pred_files = list(dataset_dir.glob("crop*_*.npy"))

        # Group by crop
        crops = {}
        for pred_file in pred_files:
            # Parse filename: crop234_mito.npy
            parts = pred_file.stem.split('_')
            crop_name = parts[0]  # 'crop234'
            class_name = '_'.join(parts[1:])  # 'mito' or 'mito_mem'

            if crop_name not in crops:
                crops[crop_name] = {}
            crops[crop_name][class_name] = pred_file

        # Save in Zarr format
        for crop_name, class_files in crops.items():
            for class_name, pred_file in class_files.items():
                # Load prediction
                pred = np.load(pred_file)

                # Create Zarr array
                zarr_path = f"{output_path}/{dataset}.zarr/{crop_name}/{class_name}/s0"
                zarr_array = zarr.open(
                    zarr_path,
                    mode='w',
                    shape=pred.shape,
                    dtype=pred.dtype,
                    chunks=(64, 64, 64),
                    compressor=zarr.Blosc(cname='zstd', clevel=5),
                )

                zarr_array[:] = pred

                print(f"Saved: {zarr_path}")

    print(f"\nSubmission package saved to: {output_path}")
    print("Create zip: zip -r submission.zip {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_dir', required=True)
    parser.add_argument('--output_path', default='submission.zarr')
    parser.add_argument('--classes', nargs='+', required=True)
    parser.add_argument('--datasets', nargs='+', required=True)

    args = parser.parse_args()

    export_submission(
        predictions_dir=args.predictions_dir,
        output_path=args.output_path,
        classes=args.classes,
        datasets=args.datasets,
    )
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/test_cellmap_dataset.py`

```python
"""Unit tests for CellMap dataset."""

import pytest
import numpy as np
from connectomics.data.dataset.dataset_cellmap import CellMapDataset, CellMapMultiClassDataset


def test_cellmap_dataset_discovery():
    """Test dataset discovery and initialization."""

    dataset = CellMapDataset(
        data_root='/projects/weilab/dataset/cellmap',
        datasets=['jrc_cos7-1a'],
        crops=None,  # Auto-discover
        classes=['nuc', 'mito'],
        mode='train',
        patch_size=(64, 64, 64),
        resolution=(8.0, 8.0, 8.0),
    )

    assert len(dataset) > 0
    assert all('jrc_cos7-1a' in sample['dataset'] for sample in dataset.data_samples)


def test_cellmap_dataset_getitem():
    """Test data loading."""

    dataset = CellMapDataset(
        data_root='/projects/weilab/dataset/cellmap',
        datasets=['jrc_cos7-1a'],
        crops=['crop234'],
        classes=['nuc'],
        mode='train',
        patch_size=(64, 64, 64),
        resolution=(8.0, 8.0, 8.0),
    )

    sample = dataset[0]

    assert 'image' in sample
    assert 'label' in sample
    assert sample['image'].shape == (1, 64, 64, 64)
    assert sample['label'].shape == (1, 64, 64, 64)
    assert sample['image'].dtype == np.float32


def test_cellmap_multiclass_dataset():
    """Test multi-class dataset."""

    dataset = CellMapMultiClassDataset(
        data_root='/projects/weilab/dataset/cellmap',
        datasets=['jrc_cos7-1a'],
        crops=['crop234'],
        classes=['nuc', 'mito', 'er'],
        mode='train',
        patch_size=(64, 64, 64),
        resolution=(8.0, 8.0, 8.0),
    )

    sample = dataset[0]

    assert sample['image'].shape == (1, 64, 64, 64)
    assert sample['label'].shape == (3, 64, 64, 64)  # 3 classes
```

### Integration Tests

```bash
# Test full training pipeline (1 epoch)
python scripts/main.py --config tutorials/cellmap_cos7.yaml \
    training.max_epochs=1 \
    data.batch_size=1 \
    --fast-dev-run

# Test inference
python scripts/main.py --config tutorials/cellmap_cos7.yaml \
    --mode test \
    --checkpoint outputs/test.ckpt
```

---

## Implementation Roadmap

### Week 1: Core Dataset Implementation
- [ ] Implement `CellMapDataset` class
- [ ] Implement `CellMapMultiClassDataset` class
- [ ] Register in dataset factory
- [ ] Write unit tests
- [ ] Test with single dataset/crop

### Week 2: Configuration & Training
- [ ] Create base Hydra config (`cellmap_base.yaml`)
- [ ] Create dataset-specific configs
- [ ] Test training with small model (1 class, 1 dataset)
- [ ] Verify MONAI transforms work correctly
- [ ] Test multi-GPU training

### Week 3: Inference & Evaluation
- [ ] Implement inference pipeline
- [ ] Test sliding window inference
- [ ] Implement post-processing (instance segmentation)
- [ ] Implement evaluation metrics
- [ ] Validate against CellMap baseline

### Week 4: Optimization & Documentation
- [ ] Performance tuning (batch size, workers, caching)
- [ ] Add more augmentations
- [ ] Write usage documentation
- [ ] Create example notebooks
- [ ] Prepare submission tools

---

## Expected Performance

### Computational Requirements

**Single GPU (A100 80GB)**:
- Patch size: 128³ voxels
- Batch size: 2-4
- Model: MedNeXt-B (10.5M params)
- Memory: ~40GB
- Training time: ~24 hours for 100 epochs (single dataset)

**Multi-GPU (4x A100)**:
- Batch size: 8-16 (total)
- Training time: ~6 hours for 100 epochs

### Data Loading Performance

**Zarr Lazy Loading**:
- No preloading required (unlike HDF5 caching)
- Fast random access to chunks
- Efficient for large datasets (1TB+)
- Network latency minimal (local filesystem)

**Optimization Tips**:
- Use persistent workers (`persistent_workers=true`)
- Increase `num_workers` (8-16 for large datasets)
- Enable pin memory (`pin_memory=true`)
- Consider MONAI CacheDataset for small datasets

---

## Alternative Approaches Considered

### ❌ Option 1: Use CellMap Toolbox Directly
**Why Not**: Duplicates PyTC functionality, abandons Lightning/MONAI/Hydra stack

### ❌ Option 2: Convert Zarr to HDF5
**Why Not**: Storage overhead (2x disk space), slow conversion, loses multi-scale pyramid

### ❌ Option 3: Fork CellMap and Modify
**Why Not**: Hard to maintain, diverges from upstream

### ✅ Option 4: Bridge Layer (Chosen)
**Why**: Minimal changes, leverages both ecosystems, maintainable

---

## Benefits of This Design

1. **Leverages PyTC Strengths**
   - Lightning orchestration (DDP, mixed precision, callbacks)
   - MONAI models (8 architectures + MedNeXt)
   - Hydra configs (composable, type-safe)

2. **Preserves CellMap Data Format**
   - No conversion needed
   - Multi-scale pyramid intact
   - Cloud-ready (S3 support via zarr)

3. **Minimal Code Duplication**
   - Reuses PyTC training loop
   - Reuses MONAI transforms
   - Only dataset adapter is new

4. **Easy to Extend**
   - Add more architectures (MONAI registry)
   - Add more datasets (just update config)
   - Swap loss functions (Hydra override)

5. **Production Ready**
   - Proven stack (Lightning + MONAI)
   - Well-tested components
   - Community support

---

## Open Questions & Future Work

### Questions
1. **Train/Val Split**: How to split crops? (by dataset? by crop? stratified by class?)
2. **Class Weighting**: How to handle imbalanced classes? (focal loss? weighted sampler?)
3. **Instance Segmentation**: Watershed vs learned embeddings? (compare both)
4. **Multi-Task Learning**: Joint semantic + instance? (requires different losses)

### Future Enhancements
1. **Self-Supervised Pretraining**: Use unlabeled EM data
2. **Test-Time Augmentation**: Flip/rotate ensembling
3. **Uncertainty Estimation**: Monte Carlo dropout
4. **Active Learning**: Iterative annotation
5. **Cross-Dataset Transfer**: Pretrain on all → finetune on one

---

## Conclusion

This design provides a **clean integration** between CellMap data and PyTC's modern architecture. By implementing a thin adapter layer, we gain access to CellMap's rich dataset while leveraging PyTC's production-ready training infrastructure.

**Key Takeaway**: Don't rebuild what works. Adapt data format, keep training stack.

---

## References

- [CellMap Challenge](https://janelia-cellmap.github.io/cellmap-segmentation-challenge/)
- [cellmap-data Package](https://github.com/janelia-cellmap/cellmap-data)
- [MONAI Documentation](https://docs.monai.io/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/)
- [MedNeXt Paper](https://arxiv.org/abs/2303.09975)
