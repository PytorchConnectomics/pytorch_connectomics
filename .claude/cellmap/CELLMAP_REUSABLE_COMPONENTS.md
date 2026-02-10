# CellMap Repository - Reusable Components Analysis

## Overview

This document identifies reusable libraries and utilities from the CellMap Segmentation Challenge repository that should be leveraged rather than reimplemented in PyTorch Connectomics.

**Key Principle**: Reuse challenge-specific infrastructure (data loading, metadata, submission) but replace training components with PyTC's Lightning/MONAI stack.

---

## ✅ Highly Reusable Components

### 1. **cellmap-data Package** (CRITICAL - MUST USE)

**Location**: External dependency (`pip install cellmap-data`)

**Purpose**: Official data loading library for CellMap challenge

**Key Classes**:
- `CellMapDataLoader`: Custom PyTorch DataLoader for CellMap Zarr data
- `CellMapDataSplit`: Train/val split management from CSV
- Built-in transforms: `Normalize()`, `Binarize()`, `NaNtoNum()`

**Why Reuse**:
- ✅ Official challenge library - guaranteed compatibility
- ✅ Handles multi-scale Zarr pyramids automatically
- ✅ Efficient lazy loading with caching
- ✅ Spatial transforms (mirror, transpose, rotate)
- ✅ Class weighting and mutual exclusion logic
- ✅ Already tested and optimized

**Integration Strategy**:
```python
# In PyTC's CellMapDataset, wrap cellmap-data components
from cellmap_data import CellMapDataLoader, CellMapDataSplit
from cellmap_data.transforms.augment import Normalize, Binarize

class CellMapDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping cellmap-data."""

    def setup(self, stage=None):
        # Use cellmap-data's CSV-based datasplit
        datasplit = CellMapDataSplit(
            input_arrays=self.input_arrays,
            target_arrays=self.target_arrays,
            classes=self.classes,
            csv_path=self.datasplit_path,
            spatial_transforms=self.spatial_transforms,
            # ... cellmap-data handles all Zarr I/O
        )

        self.train_dataset = datasplit.train_datasets_combined
        self.val_dataset = datasplit.validation_blocks

    def train_dataloader(self):
        return CellMapDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            iterations_per_epoch=self.iterations_per_epoch,
            # ... cellmap-data handles sampling
        )
```

**Status**: ✅ **MUST USE** - Don't reimplement this!

---

### 2. **Crop Metadata & Manifests**

**Location**: `src/cellmap_segmentation_challenge/utils/crops.py`

**Purpose**: Official crop metadata for test set evaluation

**Key Functions**:
```python
from cellmap_segmentation_challenge.utils import (
    fetch_test_crop_manifest,  # Get test crop metadata
    TEST_CROPS,                 # Preloaded test crops
    TEST_CROPS_DICT,            # Lookup by (id, class)
    get_test_crops,             # Get test crops with EM URLs
)
```

**Key Data Structures**:
```python
@dataclass
class TestCropRow:
    id: int                              # Crop ID (e.g., 234)
    dataset: str                         # Dataset name (e.g., 'jrc_cos7-1a')
    class_label: str                     # Class name (e.g., 'mito')
    voxel_size: tuple[float, ...]       # Resolution in nm (z, y, x)
    translation: tuple[float, ...]       # Offset in world coordinates
    shape: tuple[int, ...]               # Crop shape in voxels
```

**Manifest Files**:
- `test_crop_manifest.csv`: Test crops for evaluation (940KB)
- `train_crop_manifest.csv`: Training crops metadata (62KB)
- `manifest.csv`: Full dataset manifest

**Why Reuse**:
- ✅ Official challenge metadata - required for submission
- ✅ Includes voxel sizes, translations, shapes
- ✅ Auto-updates from GitHub (fallback to local)
- ✅ Used by evaluation server

**Integration Strategy**:
```python
# Use for test set inference
from cellmap_segmentation_challenge.utils import TEST_CROPS

for test_crop in TEST_CROPS:
    print(f"Crop {test_crop.id}: {test_crop.dataset}/{test_crop.class_label}")
    print(f"  Resolution: {test_crop.voxel_size} nm")
    print(f"  Shape: {test_crop.shape}")

    # Run inference at correct resolution and ROI
    predict_on_crop(
        dataset=test_crop.dataset,
        crop_id=test_crop.id,
        class_label=test_crop.class_label,
        target_resolution=test_crop.voxel_size,
        target_shape=test_crop.shape,
        translation=test_crop.translation,
    )
```

**Status**: ✅ **HIGHLY RECOMMENDED** - Official challenge metadata

---

### 3. **Class Definitions & Hierarchies**

**Location**: `src/cellmap_segmentation_challenge/utils/classes.csv`

**Purpose**: Official class IDs and parent-child relationships

**Structure**:
```csv
class_name,class_id,parent_classes
ecs,1,""
pm,2,""
mito_mem,3,""
mito_lum,4,""
mito_ribo,5,""
mito,50,"3,4,5"        # Parent class composed of membrane, lumen, ribosomes
nuc,37,"20,21,22,..."  # Nucleus includes many substructures
```

**Key Functions**:
```python
from cellmap_segmentation_challenge.utils import (
    get_class_relations,    # Get class hierarchy dict
    get_tested_classes,     # Get classes for evaluation
)
```

**Why Reuse**:
- ✅ Official challenge class definitions
- ✅ Hierarchical relationships (e.g., mito = mito_mem + mito_lum + mito_ribo)
- ✅ Required for mutual exclusion logic
- ✅ Maps instance IDs to semantic classes

**Integration Strategy**:
```python
# Use in config
from cellmap_segmentation_challenge.utils import get_tested_classes

# Get official challenge classes
classes = get_tested_classes()  # Returns tested class names
print(classes)  # ['nuc', 'mito', 'er', 'golgi', ...]

# Get class hierarchy for mutual exclusion
class_relations = get_class_relations(named_classes=classes)
# Returns dict: {'mito': ['mito_mem', 'mito_lum', 'mito_ribo'], ...}
```

**Status**: ✅ **HIGHLY RECOMMENDED** - Ensures consistency with challenge

---

### 4. **Submission Packaging**

**Location**: `src/cellmap_segmentation_challenge/utils/submission.py`

**Purpose**: Package predictions in official submission format

**Key Functions**:
```python
from cellmap_segmentation_challenge.utils import (
    save_numpy_class_arrays_to_zarr,  # Save predictions to Zarr
    package_submission,                 # Full submission pipeline
)
from cellmap_segmentation_challenge.evaluate import match_crop_space  # Resample to target resolution
```

**Key Features**:
1. **Zarr Saving**: Correct compression, chunking, metadata
2. **Resolution Matching**: Resample predictions to match test crop resolution
3. **Parallel Processing**: ThreadPoolExecutor for speed
4. **Validation**: Checks shape, dtype, metadata

**Why Reuse**:
- ✅ Official submission format - guaranteed acceptance
- ✅ Handles resolution resampling (critical!)
- ✅ Parallel packaging for efficiency
- ✅ Auto-zips for upload

**Integration Strategy**:
```python
from cellmap_segmentation_challenge.utils import package_submission
from cellmap_segmentation_challenge import PROCESSED_PATH, SUBMISSION_PATH

# After inference, package predictions
package_submission(
    input_search_path=PROCESSED_PATH,   # Where PyTC saved predictions
    output_path=SUBMISSION_PATH,         # submission.zarr
    overwrite=False,
    max_workers=16,
)
# Output: submission.zip ready for upload
```

**Example - Save Single Prediction**:
```python
from cellmap_segmentation_challenge.utils import save_numpy_class_arrays_to_zarr

# After PyTC inference
predictions = model.predict(image)  # (C, D, H, W)

# Convert to list of binary arrays
label_arrays = [predictions[i] > 0.5 for i in range(len(classes))]

# Save in challenge format
save_numpy_class_arrays_to_zarr(
    save_path='submission.zarr',
    test_volume_name='crop234',
    label_names=['nuc', 'mito', 'er'],
    labels=label_arrays,
    mode='append',
    attrs={'voxel_size': (8, 8, 8), 'translation': (0, 0, 0)},
)
```

**Status**: ✅ **MUST USE** - Required for valid submissions

---

### 5. **Datasplit Generation**

**Location**: `src/cellmap_segmentation_challenge/utils/datasplit.py`

**Purpose**: Auto-generate train/val split CSV from discovered data

**Key Functions**:
```python
from cellmap_segmentation_challenge.utils import (
    make_datasplit_csv,        # Local Zarr stores
    make_s3_datasplit_csv,     # S3 streaming
    get_formatted_fields,      # Parse paths
)
```

**Features**:
- Auto-discovers datasets, crops, classes
- Filters by resolution (e.g., only 8nm data)
- Stratified splitting by class
- Force all classes in validation set
- S3 bucket support

**Why Reuse**:
- ✅ Handles CellMap's complex directory structure
- ✅ Resolution filtering (important for multi-scale data)
- ✅ Class balancing logic
- ✅ Outputs CSV compatible with `cellmap-data` package

**Integration Strategy**:
```python
from cellmap_segmentation_challenge.utils import make_datasplit_csv

# Generate datasplit CSV (one-time)
make_datasplit_csv(
    classes=['nuc', 'mito', 'er'],
    csv_path='datasplit.csv',
    validation_prob=0.15,
    scale=(8, 8, 8),              # Filter for 8nm resolution
    force_all_classes='validate',  # Ensure all classes in validation
)

# Then use with cellmap-data
from cellmap_data import CellMapDataSplit
datasplit = CellMapDataSplit(csv_path='datasplit.csv', ...)
```

**Status**: ✅ **HIGHLY RECOMMENDED** - Saves time, ensures correctness

---

### 6. **Loss Wrapper**

**Location**: `src/cellmap_segmentation_challenge/utils/loss.py`

**Purpose**: Handle NaN values in sparse annotations

**Code**:
```python
class CellMapLossWrapper(torch.nn.Module):
    """
    Wrapper for loss functions to handle NaN values in targets.

    CellMap data has NaN/0 for unannotated regions - this wrapper
    masks them out before computing loss.
    """

    def __init__(self, loss_fn, **loss_kwargs):
        super().__init__()
        self.loss_fn = loss_fn(**loss_kwargs)

    def forward(self, predictions, targets):
        # Create mask for valid (non-NaN, non-zero) targets
        mask = ~torch.isnan(targets) & (targets != 0)

        # Apply mask
        masked_preds = predictions[mask]
        masked_targets = targets[mask]

        # Compute loss only on annotated regions
        return self.loss_fn(masked_preds, masked_targets)
```

**Why Reuse**:
- ✅ Essential for sparse annotations
- ✅ Simple, tested implementation
- ✅ Works with any PyTorch loss

**Integration Strategy**:
```python
# In PyTC's LightningModule
from cellmap_segmentation_challenge.utils.loss import CellMapLossWrapper
import torch.nn as nn

class CellMapLitModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # Wrap PyTC's loss with CellMap's NaN handler
        base_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.criterion = CellMapLossWrapper(
            loss_fn=type(base_loss),  # Pass class, not instance
            reduction='mean',
        )
```

**Status**: ✅ **RECOMMENDED** - Simple and effective

---

### 7. **Data Fetching**

**Location**: `src/cellmap_segmentation_challenge/utils/fetch_data.py`

**Purpose**: Download CellMap data from S3

**Key Function**:
```python
from cellmap_segmentation_challenge.cli.fetch_data import fetch_data

# Download data programmatically
fetch_data(
    datasets=['jrc_cos7-1a', 'jrc_hela-2'],
    crops=['crop234', 'crop236'],
    raw_padding=128,
    fetch_all_em_resolutions=False,
)
```

**Why Reuse**:
- ✅ Handles S3 authentication
- ✅ Supports padding, resolution filtering
- ✅ Progress bars, resume capability
- ✅ Validates downloads

**Integration Strategy**:
```python
# Use CLI for initial download
# csc fetch-data --raw-padding 128

# Or programmatically in setup script
from cellmap_segmentation_challenge.utils import download_file

# Download specific files
download_file(
    url='s3://janelia-cosem-datasets/...',
    output_path='/projects/weilab/dataset/cellmap/...',
)
```

**Status**: ✅ **RECOMMENDED** - Convenient for data management

---

## ⚠️ Use with Caution

### 8. **Dataloader (`get_dataloader`)**

**Location**: `src/cellmap_segmentation_challenge/utils/dataloader.py`

**Purpose**: Factory function for creating CellMap dataloaders

**Why Use Caution**:
- ⚠️ Returns `CellMapDataLoader` (not standard PyTorch DataLoader)
- ⚠️ Custom iteration logic (not compatible with Lightning out-of-box)
- ⚠️ Hardcoded transforms (limited flexibility)

**Recommendation**:
- ✅ Use underlying `cellmap-data` package directly
- ✅ Wrap in Lightning DataModule
- ❌ Don't use `get_dataloader()` function directly

**Better Approach**:
```python
# Instead of:
# train_loader, val_loader = get_dataloader(config)  # ❌

# Do this:
from cellmap_data import CellMapDataSplit, CellMapDataLoader

class CellMapDataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        datasplit = CellMapDataSplit(...)
        self.train_ds = datasplit.train_datasets_combined
        self.val_ds = datasplit.validation_blocks

    def train_dataloader(self):
        return CellMapDataLoader(self.train_ds, ...)  # ✅
```

---

## ❌ Do NOT Reuse - Replace with PyTC

### 9. **Training Loop (`train.py`)**

**Location**: `src/cellmap_segmentation_challenge/train.py` (530 lines)

**Why NOT Reuse**:
- ❌ Custom training loop (no Lightning)
- ❌ Manual GPU handling, mixed precision
- ❌ TensorBoard only (no WandB, other loggers)
- ❌ No DDP support
- ❌ Limited callbacks

**Replacement**: Use PyTC's `ConnectomicsModule` (Lightning)

---

### 10. **Models (`models/`)**

**Location**: `src/cellmap_segmentation_challenge/models/`

**Models Available**:
- `UNet_2D` / `UNet_3D`: Simple U-Nets (4 levels)
- `ResNet`: ResNet encoder (2D/3D)
- `ViTVNet`: Vision Transformer V-Net

**Why NOT Reuse**:
- ❌ Limited architectures (only 4 models)
- ❌ No deep supervision
- ❌ No modern models (MedNeXt, nnUNet, etc.)
- ❌ Not optimized for medical imaging

**Replacement**: Use PyTC's MONAI model registry (8+ architectures)

---

### 11. **Prediction Pipeline (`predict.py`)**

**Location**: `src/cellmap_segmentation_challenge/predict.py`

**Why NOT Reuse**:
- ❌ Tiled inference (MONAI has better sliding window)
- ❌ No test-time augmentation
- ❌ Manual stitching logic

**Replacement**: Use MONAI's `SlidingWindowInferer`

---

### 12. **Augmentations**

**Location**: Via `cellmap-data` package (spatial transforms dict)

**Why NOT Reuse**:
- ❌ Limited augmentations (mirror, transpose, rotate only)
- ❌ No intensity augmentations (blur, noise, contrast)
- ❌ Custom format (not torchvision/MONAI compatible)

**Replacement**: Use MONAI transforms (more comprehensive)

**Note**: Basic spatial transforms from `cellmap-data` are OK for compatibility

---

## 🔄 Hybrid Approach - Adapt and Extend

### 13. **Evaluation Metrics (`evaluate.py`)**

**Location**: `src/cellmap_segmentation_challenge/evaluate.py` (41KB)

**Features**:
- Official challenge metrics (Dice, IoU, Rand Error, VOI)
- Resolution resampling (`match_crop_space`)
- Parallel evaluation

**Strategy**:
- ✅ Reuse `match_crop_space()` for resampling
- ✅ Reuse metric computation for validation
- ⚠️ Integrate with PyTC's metrics module
- ✅ Expose via Lightning callbacks

**Integration**:
```python
from cellmap_segmentation_challenge.evaluate import match_crop_space, evaluate_predictions

class CellMapEvaluationCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Use official challenge metrics
        metrics = evaluate_predictions(
            predictions=pl_module.predictions,
            ground_truth=pl_module.ground_truth,
        )
        pl_module.log_dict(metrics)
```

**Status**: ✅ **RECOMMENDED** - Official challenge evaluation

---

## Summary Table

| Component | Location | Status | Integration Method |
|-----------|----------|--------|-------------------|
| **cellmap-data package** | External | ✅ MUST USE | Wrap in Lightning DataModule |
| **Crop manifests** | `utils/crops.py` | ✅ HIGHLY RECOMMENDED | Import directly |
| **Class definitions** | `utils/classes.csv` | ✅ HIGHLY RECOMMENDED | Import `get_tested_classes()` |
| **Submission packaging** | `utils/submission.py` | ✅ MUST USE | Call `package_submission()` |
| **Datasplit generation** | `utils/datasplit.py` | ✅ HIGHLY RECOMMENDED | Call `make_datasplit_csv()` |
| **Loss wrapper** | `utils/loss.py` | ✅ RECOMMENDED | Wrap PyTC losses |
| **Data fetching** | `utils/fetch_data.py` | ✅ RECOMMENDED | Use CLI or import |
| **Dataloader factory** | `utils/dataloader.py` | ⚠️ CAUTION | Use underlying `cellmap-data` instead |
| **Training loop** | `train.py` | ❌ REPLACE | Use PyTC Lightning |
| **Models** | `models/` | ❌ REPLACE | Use PyTC MONAI models |
| **Prediction** | `predict.py` | ❌ REPLACE | Use MONAI SlidingWindowInferer |
| **Augmentations** | `cellmap-data` | ⚠️ BASIC ONLY | Use MONAI for advanced |
| **Evaluation** | `evaluate.py` | 🔄 ADAPT | Reuse metrics, wrap in callbacks |

---

## Recommended Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                PyTorch Connectomics (PyTC)                       │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Lightning  │  │ MONAI Models │  │  Hydra Configs       │  │
│  │ (Training)   │  │ (8+ archs)   │  │  (YAML)              │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │        CellMapDataModule (NEW - Lightning wrapper)       │  │
│  │  - Wraps cellmap-data package                            │  │
│  │  - Implements Lightning DataModule interface             │  │
│  │  - Adds MONAI transforms on top of cellmap-data spatial  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │ Uses
                            │
┌─────────────────────────────────────────────────────────────────┐
│              CellMap Challenge Libraries (Reuse)                 │
│                                                                  │
│  ✅ cellmap-data          - Data loading (CRITICAL)             │
│  ✅ TEST_CROPS            - Test metadata (CRITICAL)             │
│  ✅ get_tested_classes    - Class definitions                   │
│  ✅ package_submission    - Submission format (CRITICAL)         │
│  ✅ make_datasplit_csv    - Train/val splitting                 │
│  ✅ CellMapLossWrapper    - NaN handling                         │
│  ✅ match_crop_space      - Resolution resampling               │
│  ✅ evaluate_predictions  - Challenge metrics                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation Requirements

To use CellMap components in PyTC:

```bash
# Activate PyTC environment
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc

# Install cellmap packages
pip install cellmap-data                           # Core data loading
pip install cellmap-segmentation-challenge         # Challenge utilities

# Or install from local clone
cd /projects/weilab/weidf/lib/seg/cellmap-segmentation-challenge
pip install -e .
```

**Dependencies Added**:
- `cellmap-data`: ~500KB, brings Zarr utilities
- `zarr < 3.0.0`: Zarr v2 format support
- `numcodecs`: Compression codecs
- `tensorstore`: Multi-dimensional arrays
- `upath`: Universal path handling

---

## Example Usage in PyTC

```python
# tutorial/cellmap_cos7.yaml

dataset:
  name: cellmap
  data_root: /projects/weilab/dataset/cellmap

  # Use CellMap's class definitions (official challenge)
  classes: !python/object/apply:cellmap_segmentation_challenge.utils.get_tested_classes []

  # Use CellMap's datasplit generation
  datasplit_csv: datasplit.csv

  # Use cellmap-data's resolution filtering
  resolution: [8.0, 8.0, 8.0]

# Training uses PyTC Lightning stack
model:
  architecture: mednext  # PyTC's MONAI model
  # ...

training:
  # PyTC Lightning training
  # ...
```

```python
# In PyTC code
from cellmap_data import CellMapDataSplit
from cellmap_segmentation_challenge.utils import (
    make_datasplit_csv,
    TEST_CROPS,
    package_submission,
)

# 1. Generate datasplit (one-time)
make_datasplit_csv(
    classes=cfg.dataset.classes,
    csv_path=cfg.dataset.datasplit_csv,
    validation_prob=0.15,
    scale=cfg.dataset.resolution,
)

# 2. Train with PyTC Lightning (uses cellmap-data for loading)
trainer = create_trainer(cfg)
model = ConnectomicsModule(cfg)
datamodule = CellMapDataModule(cfg)  # Wraps cellmap-data
trainer.fit(model, datamodule)

# 3. Predict on test crops (using official metadata)
for test_crop in TEST_CROPS:
    predictions = model.predict(
        dataset=test_crop.dataset,
        crop_id=test_crop.id,
        resolution=test_crop.voxel_size,
    )

# 4. Package submission (using official format)
package_submission(
    input_search_path='outputs/predictions',
    output_path='submission.zarr',
)
```

---

## Key Takeaways

### ✅ MUST Reuse:
1. **cellmap-data package** - Core data loading
2. **TEST_CROPS** - Official test metadata
3. **package_submission()** - Submission formatting

### ✅ SHOULD Reuse:
4. **get_tested_classes()** - Class definitions
5. **make_datasplit_csv()** - Dataset splitting
6. **CellMapLossWrapper** - NaN handling
7. **match_crop_space()** - Resolution resampling

### ❌ REPLACE with PyTC:
8. Training loop → **Lightning**
9. Models → **MONAI model registry**
10. Inference → **MONAI SlidingWindowInferer**
11. Advanced augmentations → **MONAI transforms**

### 🎯 Result:
- **Official challenge compatibility** (data format, evaluation, submission)
- **Modern training stack** (Lightning + MONAI + Hydra)
- **Best of both worlds**

---

## Next Steps

Update [CELLMAP_INTEGRATION_DESIGN.md](.claude/CELLMAP_INTEGRATION_DESIGN.md) to:
1. Remove custom dataset implementation
2. Use `cellmap-data` package directly
3. Add proper imports from CellMap utils
4. Reference official crop manifests
5. Use official submission packaging

This will **significantly reduce implementation complexity** while ensuring **100% challenge compatibility**.
