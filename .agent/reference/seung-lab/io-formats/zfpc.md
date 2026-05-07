# zfpc

**GitHub:** https://github.com/seung-lab/zfpc
**Language:** Python | **Stars:** 1

Experimental container format for zfp-encoded vector fields. Splits multi-dimensional arrays along uncorrelated dimensions, compresses each slice as a separate zfp stream, and packs them into a single file for seamless decompression.

## Key Features
- Compress 1-4D arrays with per-dimension correlation control
- Supports tolerance, rate, precision, and lossless modes
- Single-file output compatible with data viewers like Neuroglancer
- 15-byte header with index for potential random access

## API
```python
import zfpc
binary = zfpc.compress(array, tolerance=0.01, correlated_dims=[True, True, False, False])
recovered = zfpc.decompress(binary)
header = zfpc.header(binary)
```

## Relevance to Connectomics
Efficient compression of vector fields (e.g., optical flow, affinity maps) used in EM alignment and segmentation pipelines.
