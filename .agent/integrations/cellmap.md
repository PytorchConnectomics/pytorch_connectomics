# CellMap Challenge Integration

The CellMap Segmentation Challenge (Janelia) provides multi-organelle segmentation benchmarks for EM data.

## Design: Lightweight PyTC Integration

Zero modifications to PyTC core. Integration is done via:
1. Standard PyTC YAML configs pointing to CellMap data
2. External data conversion scripts (CellMap Zarr -> HDF5)
3. Standard PyTC training/inference pipeline

## Task Types

### Semantic Segmentation
- Single-label per voxel
- Standard Dice + CE loss
- Direct PyTC support

### Instance Segmentation
- Requires post-processing (connected components, watershed)
- Use affinity prediction + decoding pipeline
- Multi-task output (binary + boundary + EDT)

## Data Format

CellMap uses Zarr format. Two options:
1. Convert to HDF5 (recommended for small datasets)
2. Use `LazyZarrVolumeDataset` for direct Zarr loading

## Key Datasets

- Multiple EM volumes from different organisms/tissues
- Multi-class annotations (mito, ER, nucleus, etc.)
- Variable resolutions (4-8nm isotropic)

## Configuration Example

```yaml
model:
  architecture: mednext
  mednext_size: B
  out_channels: 2
  deep_supervision: true

data:
  train_image: "datasets/cellmap/train_*.h5"
  train_label: "datasets/cellmap/train_*_label.h5"

inference:
  decoding:
    method: connected_components
    threshold: 0.5
```

## Reusable Components from CellMap Repo

- `cellmap-segmentation-challenge` pip package for data loading
- Evaluation scripts compatible with challenge submission format
- Pre-computed dataset splits
