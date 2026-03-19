# fastremap

**GitHub:** https://github.com/seung-lab/fastremap
**Language:** C++ | **Stars:** 63

High-performance label manipulation for numpy arrays at C++ speed: remap, mask, renumber, unique, in-place transposition, and point cloud extraction for 3D labeled images.

## Key Features
- Fast `unique` (often faster than `np.unique`), `renumber`, `remap`, `mask`
- In-place array transposition (C <-> Fortran order) without copying
- `refit` dtype to smallest sufficient type
- `component_map` / `inverse_component_map` for label correspondence
- `point_cloud` extraction by label
- `foreground` counting, `minmax`, `pixel_pairs`, `tobytes` utilities

## API
```python
import fastremap

labels, mapping = fastremap.renumber(labels, in_place=True)
labels = fastremap.remap(labels, {1: 2}, preserve_missing_labels=True, in_place=True)
labels = fastremap.mask(labels, [1, 5, 13])
uniq, cts = fastremap.unique(labels, return_counts=True)
fastremap.transpose(labels)  # in-place
ptc = fastremap.point_cloud(labels)
```

## Relevance to Connectomics
Core dependency of PyTC (listed in requirements). Used throughout segmentation post-processing for efficient label remapping, renumbering, and manipulation of large 3D segmentation volumes.
