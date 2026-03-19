# compressedseg

**GitHub:** https://github.com/seung-lab/compressedseg
**Language:** C++ | **Stars:** 4

Python library for compressing and decompressing Neuroglancer compressed_segmentation format. Adapted from the Neuroglancer project (Google/Janelia).

## Key Features
- Compress/decompress segmentation labels in Neuroglancer format
- Random access to single voxels without full decompression
- Extract unique labels without decompression
- Remap labels in compressed form
- CLI for numpy file conversion
- C++, Python, and Go interfaces

## API
```python
import compressed_segmentation as cseg
import numpy as np

labels = np.arange(0, 128**3, dtype=np.uint64).reshape((128,128,128))
compressed = cseg.compress(labels, order='C')
recovered = cseg.decompress(compressed, (128,128,128), dtype=np.uint64, order='C')
arr = cseg.CompressedSegmentationArray(compressed, shape=(128,128,128), dtype=np.uint64)
label = arr[54, 32, 103]  # random access without decompression
```

## Relevance to Connectomics
Provides the compression format used by Neuroglancer/CloudVolume for storing segmentation labels, enabling efficient storage and transfer of large EM segmentation volumes.
