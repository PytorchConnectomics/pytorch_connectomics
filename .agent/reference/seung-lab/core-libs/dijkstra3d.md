# dijkstra3d

**GitHub:** https://github.com/seung-lab/dijkstra3d
**Language:** C++ | **Stars:** 84

Dijkstra's shortest path variants for 6, 18, and 26-connected 3D image volumes (and 4/8-connected 2D). Designed for voxel-based pathfinding without explicit graph construction.

## Key Features
- Dijkstra, bidirectional Dijkstra, and A* (compass) search on 3D volumes
- Binary Dijkstra for foreground/background images
- Euclidean distance field computation with anisotropy support
- Parental field for efficient multi-target path extraction
- Voxel connectivity graph support for custom traversal constraints
- No explicit graph construction needed (implicit edges from image grid)

## API
```python
import dijkstra3d
import numpy as np

field = np.ones((512, 512, 512), dtype=np.int32)
path = dijkstra3d.dijkstra(field, source=(0,0,0), target=(511,511,511), connectivity=26)
path = dijkstra3d.binary_dijkstra(field, source, target, background_color=0)
dist = dijkstra3d.euclidean_distance_field(field, source=(0,0,0), anisotropy=(4,4,40))
parents = dijkstra3d.parental_field(field, source=(0,0,0))
path = dijkstra3d.path_from_parents(parents, target=(511,511,511))
```

## Relevance to Connectomics
Core dependency of kimimaro (TEASAR skeletonization); provides shortest-path computation through 3D segmentation volumes for skeleton extraction and distance-based analysis.
