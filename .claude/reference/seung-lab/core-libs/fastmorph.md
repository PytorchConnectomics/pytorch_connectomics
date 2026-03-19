# fastmorph

**GitHub:** https://github.com/seung-lab/fastmorph
**Language:** C++ | **Stars:** 19

Multithreaded multilabel and grayscale 3D morphological image processing: dilation, erosion, opening, closing, and hole filling tuned for dense segmentation volumes.

## Key Features
- Multi-label aware dilation (mode of surrounding labels), erosion, opening, closing
- Grayscale stenciled and spherical morphological operations
- Multithreaded with low memory usage
- Fast multilabel hole filling (v1: binary image analysis, v2: contact graph)
- Supports anisotropic voxel spacing for spherical operations

## API
```python
import fastmorph
morphed = fastmorph.dilate(labels, parallel=2)
morphed = fastmorph.erode(labels)
morphed = fastmorph.opening(labels, parallel=2)
morphed = fastmorph.closing(labels, parallel=2)
morphed = fastmorph.spherical_dilate(labels, radius=1, anisotropy=(1,1,1))
filled, holes = fastmorph.fill_holes_v2(labels)
```

## Relevance to Connectomics
Post-processing of 3D segmentation labels -- morphological cleanup, hole filling, and boundary smoothing of neuron/organelle segmentations.
