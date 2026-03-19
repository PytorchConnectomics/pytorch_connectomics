# DracoPy

**GitHub:** https://github.com/seung-lab/DracoPy
**Language:** C++ | **Stars:** 117

Python wrapper for Google's Draco mesh compression library. Encode and decode 3D meshes and point clouds with configurable quantization and compression levels.

## Key Features
- Encode/decode 3D meshes (vertices, faces, normals, colors)
- Point cloud encoding (faces omitted)
- Configurable quantization bits, compression level, and order preservation
- Cross-platform: Linux, macOS, Windows
- Numpy integration

## API
```python
import DracoPy

mesh = DracoPy.decode(draco_bytes)
print(len(mesh.points), len(mesh.faces))

binary = DracoPy.encode(mesh.points, mesh.faces,
    quantization_bits=14, compression_level=1)
```

## Relevance to Connectomics
Compresses 3D neuron meshes generated from EM segmentations for efficient storage and streaming in visualization tools like Neuroglancer.
