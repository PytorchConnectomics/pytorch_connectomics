# euclidean-distance-transform-3d

**GitHub:** https://github.com/seung-lab/euclidean-distance-transform-3d
**Language:** C++ | **Stars:** 261

Multi-label anisotropic 3D Euclidean distance transform (MLAEDT-3D) using marching parabolas. Computes EDT and signed distance fields for 1D/2D/3D labeled images with support for anisotropic voxel spacing and parallel execution.

## Key Features
- Single-pass multi-label distance transform (no per-label masking needed)
- Anisotropic voxel spacing support (critical for EM data)
- Signed distance function (SDF) computation
- Parallel multi-threaded execution
- Per-label iteration via `edt.each()`
- Voxel connectivity graph for self-touching labels

## API
```python
import edt
import numpy as np

labels = np.ones((512, 512, 512), dtype=np.uint32, order='F')
dt = edt.edt(labels, anisotropy=(6, 6, 30), black_border=True, parallel=4)
sdf = edt.sdf(labels, anisotropy=(6, 6, 30))

for label, image in edt.each(labels, dt, in_place=True):
    process(image)
```

## Relevance to Connectomics
Computes distance transforms for EM segmentation post-processing, used in TEASAR skeletonization (kimimaro) and boundary-based loss functions. PyTC uses distance transforms in `connectomics/data/process/distance.py`.
