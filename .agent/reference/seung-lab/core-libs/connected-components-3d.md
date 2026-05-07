# connected-components-3d

**GitHub:** https://github.com/seung-lab/connected-components-3d
**Language:** C++ | **Stars:** 450

Fast connected components labeling on multilabel 2D and 3D images. Supports 4/8-connected (2D) and 6/18/26-connected (3D) neighborhoods, continuous value CCL, and periodic boundaries. Uses Union-Find with decision trees.

## Key Features
- Single-pass multilabel CCL (no per-label masking needed)
- Continuous value CCL for grayscale images (delta-based grouping)
- Statistics: centroids, bounding boxes, voxel counts
- Dust removal (small/large object filtering), k-largest extraction
- Contact surface area and contact network computation
- Per-voxel connectivity graph extraction
- Periodic boundary support for simulations

## API
```python
import cc3d
import numpy as np

labels_in = np.ones((512, 512, 512), dtype=np.int32)
labels_out = cc3d.connected_components(labels_in, connectivity=26)
labels_out = cc3d.dust(labels_out, threshold=100, connectivity=26)
labels_out = cc3d.largest_k(labels_out, k=10)
stats = cc3d.statistics(labels_out)  # centroids, bboxes, voxel_counts
```

## Relevance to Connectomics
Core dependency of PyTC (`cc3d` in requirements). Used in segmentation post-processing to split disconnected components, remove dust, and compute instance statistics after watershed/agglomeration.
