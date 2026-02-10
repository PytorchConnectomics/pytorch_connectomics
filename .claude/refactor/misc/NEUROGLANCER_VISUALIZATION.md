# Neuroglancer Visualization Guide

## Overview

PyTorch Connectomics includes a Neuroglancer-based visualization tool for exploring 3D EM volumes directly in your web browser.

**Features:**
- ✅ Load volumes from config files or direct paths
- ✅ Visualize image and segmentation volumes side-by-side
- ✅ Support for HDF5, TIFF, and other formats
- ✅ Automatic type detection (image vs segmentation)
- ✅ Configurable resolution and server settings
- ✅ Remote access support for cluster visualization

## Installation

```bash
pip install neuroglancer
```

## Quick Start

### From Config File

```bash
# Visualize training data from config
just visualize tutorials/monai_lucchi.yaml

# Visualize test data
just visualize tutorials/monai_lucchi.yaml -- --mode test

# Visualize both train and test
just visualize tutorials/monai_lucchi.yaml -- --mode both
```

### From File Paths

```bash
# Visualize image and label
just visualize-files datasets/Lucchi/img/train_im.tif datasets/Lucchi/label/train_label.tif

# Or use the script directly
python scripts/visualize_neuroglancer.py \
  --image datasets/Lucchi/img/train_im.tif \
  --label datasets/Lucchi/label/train_label.tif
```

### Multiple Volumes

```bash
# Visualize image, label, and prediction with custom names
just visualize-volumes image:datasets/img.tif label:datasets/label.h5 pred:outputs/prediction.h5
```

## Usage Examples

### Example 1: Visualize Training Data

```bash
just visualize tutorials/monai_lucchi.yaml
```

**Output:**
```
Loading train image: datasets/Lucchi/img/train_im.tif
Loading train label: datasets/Lucchi/label/train_label.tif

Starting Neuroglancer server on localhost:9999

Adding layer: train_image
  Shape: (165, 1024, 768), Type: image, Resolution: (30, 6, 6)

Adding layer: train_label
  Shape: (165, 1024, 768), Type: segmentation, Resolution: (30, 6, 6)

======================================================================
Neuroglancer viewer ready!
======================================================================

Open this URL in your browser:
  http://localhost:9999/v/...

Server: localhost:9999
Volumes: ['train_image', 'train_label']

Press Ctrl+C to stop the server
======================================================================
```

### Example 2: Custom Port

```bash
# Use port 8080 instead of default 9999
just visualize-port 8080 tutorials/monai_lucchi.yaml
```

### Example 3: Remote Access (Cluster)

```bash
# Allow remote access (e.g., from your laptop to cluster)
just visualize-remote 8080 tutorials/monai_lucchi.yaml
```

Then on your laptop, SSH tunnel:
```bash
ssh -L 8080:localhost:8080 user@cluster.address
```

Open in browser: `http://localhost:8080`

### Example 4: Custom Resolution

```bash
# CREMI dataset with different resolution (40nm x 4nm x 4nm)
python scripts/visualize_neuroglancer.py \
  --image datasets/cremi/img.h5 \
  --label datasets/cremi/label.h5 \
  --resolution 40 4 4
```

### Example 5: Compare Prediction with Ground Truth

```bash
just visualize-volumes \
  image:datasets/test_im.tif \
  ground_truth:datasets/test_label.h5 \
  prediction:outputs/lucchi_monai_unet/results/test_pred.h5
```

## Command Reference

### Just Commands

| Command | Description | Example |
|---------|-------------|---------|
| `just visualize CONFIG` | Load from config file | `just visualize tutorials/monai_lucchi.yaml` |
| `just visualize-files IMG LBL` | Load specific files | `just visualize-files img.tif label.h5` |
| `just visualize-volumes VOLS...` | Load multiple named volumes | `just visualize-volumes img:a.tif lbl:b.h5` |
| `just visualize-port PORT CONFIG` | Use custom port | `just visualize-port 8080 tutorials/monai_lucchi.yaml` |
| `just visualize-remote PORT CONFIG` | Remote access mode | `just visualize-remote 8080 tutorials/monai_lucchi.yaml` |

### Script Arguments

```bash
python scripts/visualize_neuroglancer.py [OPTIONS]
```

**Input Sources (choose one):**
- `--config PATH` - Load from config YAML
- `--image PATH --label PATH` - Load two volumes
- `--volumes NAME:PATH [NAME:PATH ...]` - Load multiple volumes

**Server Settings:**
- `--ip IP` - Server IP (default: localhost, use 0.0.0.0 for remote)
- `--port PORT` - Server port (default: 9999)

**Volume Metadata:**
- `--resolution Z Y X` - Voxel size in nm (default: 30 6 6)
- `--offset Z Y X` - Volume offset (default: 0 0 0)

**Config-Specific:**
- `--mode MODE` - Which data to load: train/test/both (default: train)

## Volume Type Detection

The script automatically detects volume types:

**Segmentation** (displayed with color map):
- Files with "label", "seg", "gt", "pred", "output" in name
- Integer dtype arrays

**Image** (displayed as grayscale):
- All other files
- Float or uint8 arrays

## Supported Formats

Via `connectomics.data.io.read_volume`:
- HDF5 (`.h5`, `.hdf5`)
- TIFF stacks (`.tif`, `.tiff`)
- PNG/JPG stacks
- Zarr arrays (`.zarr`)
- NumPy arrays (`.npy`)

## Neuroglancer Controls

Once the viewer is open:

**Navigation:**
- **Left click + drag**: Rotate 3D view
- **Right click + drag**: Pan
- **Scroll**: Zoom in/out
- **Middle click**: Reset view

**Layer Controls (right sidebar):**
- Toggle layer visibility (eye icon)
- Adjust opacity (slider)
- Change color map (for segmentations)
- Adjust contrast/brightness (for images)

**Keyboard Shortcuts:**
- `1`, `2`, `3`: Switch to Z/Y/X slice views
- `4`: 3D view
- `[` / `]`: Previous/next slice
- `Shift + Click`: Select segment (for segmentations)

## Troubleshooting

### Port Already in Use

```bash
# Use a different port
just visualize-port 8888 tutorials/monai_lucchi.yaml
```

### Cannot Access from Remote Machine

```bash
# Enable remote access
just visualize-remote 9999 tutorials/monai_lucchi.yaml

# Then SSH tunnel from your laptop:
ssh -L 9999:localhost:9999 user@cluster
```

### Volumes Not Loading

```bash
# Check file paths
python scripts/visualize_neuroglancer.py \
  --image path/to/image.tif \
  --label path/to/label.h5

# Check if files exist
ls -lh datasets/Lucchi/img/train_im.tif
```

### Memory Issues with Large Volumes

The script loads full volumes into memory. For very large volumes (>10GB):
- Use a machine with sufficient RAM
- Or visualize smaller crops/downsampled versions

## Integration with Training

### Visualize Training Data Before Training

```bash
# Check your data looks correct
just visualize tutorials/monai_lucchi.yaml --mode train
```

### Visualize Predictions After Testing

```bash
# After running inference
just test monai lucchi outputs/checkpoints/best.ckpt

# Visualize results
just visualize-volumes \
  image:datasets/Lucchi/img/test_im.tif \
  ground_truth:datasets/Lucchi/label/test_label.h5 \
  prediction:outputs/lucchi_monai_unet/results/test_prediction.h5
```

## Advanced Usage

### Custom Server Settings

```python
# In your own script
from scripts.visualize_neuroglancer import visualize_volumes
import numpy as np

volumes = {
    'my_image': (np.random.rand(100, 512, 512), 'image'),
    'my_segmentation': (np.random.randint(0, 10, (100, 512, 512)), 'segmentation'),
}

visualize_volumes(
    volumes=volumes,
    ip='0.0.0.0',
    port=8080,
    resolution=(30, 6, 6),  # z, y, x in nm
    offset=(0, 0, 0)
)
```

### Load From Custom Config

```python
from scripts.visualize_neuroglancer import load_volumes_from_config

volumes = load_volumes_from_config('tutorials/monai_lucchi.yaml', mode='both')
# Returns: {'train_image': (array, 'image'), 'train_label': (array, 'segmentation'), ...}
```

## Comparison with Other Tools

| Feature | Neuroglancer | napari | ImageJ/Fiji |
|---------|--------------|--------|-------------|
| **3D rendering** | ✅ Excellent | ✅ Good | ⚠️ Limited |
| **Large volumes** | ✅ Streaming | ⚠️ RAM limited | ⚠️ RAM limited |
| **Browser-based** | ✅ Yes | ❌ Desktop | ❌ Desktop |
| **Remote access** | ✅ Easy (HTTP) | ⚠️ X11/VNC | ⚠️ X11/VNC |
| **Segmentation overlay** | ✅ Built-in | ✅ Built-in | ✅ Via plugins |
| **Python integration** | ✅ Native | ✅ Native | ⚠️ Via Jython |

**When to use Neuroglancer:**
- ✅ Visualizing on remote clusters
- ✅ Very large volumes (>10GB)
- ✅ Sharing visualizations via URL
- ✅ Need 3D navigation

**When to use napari instead:**
- Detailed annotation/editing
- Advanced visualization plugins
- Prefer desktop GUI

## References

- **Neuroglancer**: https://github.com/google/neuroglancer
- **PyTC I/O**: `connectomics/data/io/`
- **Script**: `scripts/visualize_neuroglancer.py`
- **Just Commands**: `justfile`

---

**Quick Reference:**

```bash
# Most common usage
just visualize tutorials/monai_lucchi.yaml

# Custom files
just visualize-files datasets/img.tif datasets/label.h5

# Remote access
just visualize-remote 8080 tutorials/monai_lucchi.yaml
```
