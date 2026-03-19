# pyfqmr-Fast-Quadric-Mesh-Reduction

**GitHub:** https://github.com/seung-lab/pyfqmr-Fast-Quadric-Mesh-Reduction
**Language:** Python | **Stars:** 0

Cython wrapper around sp4cerat's Fast Quadric Mesh Simplification algorithm. Fork with Seung Lab modifications for mesh triangle reduction using quadric error metrics.

## Key Features
- Fast quadric mesh decimation via Cython
- Configurable target triangle count and aggressiveness
- Border preservation option
- Lossless simplification mode

## API
```python
import pyfqmr
mesh_simplifier = pyfqmr.Simplify()
mesh_simplifier.setMesh(vertices, faces)
mesh_simplifier.simplify_mesh(target_count=1000, aggressiveness=7, preserve_border=True)
vertices, faces, normals = mesh_simplifier.getMesh()
```

## Relevance to Connectomics
Used to reduce mesh complexity for neuron meshes generated from EM segmentation, enabling efficient visualization in Neuroglancer.
