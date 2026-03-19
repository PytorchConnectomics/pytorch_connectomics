# fill_voids

**GitHub:** https://github.com/seung-lab/fill_voids
**Language:** C++ | **Stars:** 29

High-performance hole filling in binary 2D and 3D images. Significantly faster than `scipy.ndimage.binary_fill_holes` with lower memory usage.

## Key Features
- 2D and 3D binary void filling
- In-place editing option for reduced memory
- Scan line flood fill algorithm (much faster than SciPy's serial dilations)
- Returns fill count for diagnostics
- C++ and Python APIs

## API
```python
import fill_voids
filled = fill_voids.fill(img, in_place=False)
filled, N = fill_voids.fill(img, return_fill_count=True)
```

## Relevance to Connectomics
Fills holes in binary segmentation masks from EM volumes, essential for cleaning up neuron/organelle segmentations.
