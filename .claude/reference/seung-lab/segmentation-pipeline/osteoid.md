# osteoid

**GitHub:** https://github.com/seung-lab/osteoid
**Language:** C++ | **Stars:** 3

Skeleton data structure for representing neurons, adjacent cells, and organelles. Refactored from CloudVolume's Skeleton code into a standalone library.

## Key Features
- Load/save skeletons in SWC, Neuroglancer Precomputed, and OSTD formats
- Cable length, connected components, path decomposition
- Downsample, smooth, crop, and consolidate skeletons
- Physical/voxel space transforms with anisotropy support
- Visualization via matplotlib or VTK
- Interop with Navis and NetworkX

## API
```python
from osteoid import Skeleton, Bbox
skel = osteoid.load("skeleton.swc")
skel = Skeleton(vertices, edges, radii=radii, transform=matrix)
length = skel.cable_length()
comps = skel.components()
paths = skel.paths()
skel2 = skel.crop(bbox)
skel.viewer(color_by='radius')
```

## Relevance to Connectomics
Core data structure for neuron skeleton representation, used by kimimaro, microviewer, and cloud-volume for morphological analysis and visualization.
