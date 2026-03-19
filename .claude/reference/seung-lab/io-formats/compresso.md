# compresso

**GitHub:** https://github.com/seung-lab/compresso
**Language:** C++ | **Stars:** 4

Efficient compression of segmentation data for connectomics (MICCAI 2017). Achieves 600-2200x compression for label volumes by exploiting redundancy in boundary regions.

## Key Features
- Compress/decompress 3D numpy segmentation arrays
- Random access to individual Z slices (format version 1)
- Label extraction without full decompression
- In-stream label remapping (e.g., for proofreading)
- CompressoArray for array-like access to compressed data
- CLI tool for file-based compression

## API
```python
import compresso
compressed = compresso.compress(labels)                    # 3D array -> bytes
labels = compresso.decompress(compressed)                  # bytes -> 3D array
labels = compresso.decompress(compressed, z=3)             # single slice
uniq = compresso.labels(compressed)                        # unique labels
remapped = compresso.remap(compressed, {1: 2, 2: 3})      # remap labels
arr = compresso.CompressoArray(compressed)                 # array-like access
```

## Relevance to Connectomics
Specialized compression for dense segmentation volumes that are common outputs of EM reconstruction pipelines.
