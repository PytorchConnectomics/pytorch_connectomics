# crackle

**GitHub:** https://github.com/seung-lab/crackle
**Language:** C++ | **Stars:** 15

Next-generation 3D segmentation compression codec based on crack codes. Provides high compression ratios for dense label volumes with fast random access and label queries without full decompression.

## Key Features
- Compress/decompress 2D and 3D dense segmentation arrays
- Extract binary images, labels, voxel counts, centroids, bounding boxes without decompressing
- Array slicing via CrackleArray with read/write support
- Connected components, contact surface analysis, voxel connectivity graph
- Remap, refit, and renumber labels in compressed form
- CLI tool for file conversion and integrity checking

## API
```python
import crackle
binary = crackle.compress(labels, allow_pins=False, markov_model_order=0)
labels = crackle.decompress(binary)
uniq = crackle.labels(binary)
arr = crackle.CrackleArray(binary)
res = arr[:10,:10,:10]
crackle.save(labels, "output.ckl")
```

## Relevance to Connectomics
Primary compression format for large-scale EM segmentation volumes; used as a dependency by kimimaro, fastmorph, and cloud-volume. Listed as a core dependency of PyTorch Connectomics.
