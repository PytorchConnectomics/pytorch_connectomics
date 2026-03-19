# zmesh

**GitHub:** https://github.com/seung-lab/zmesh
**Language:** C++ | **Stars:** 72

Multi-label marching cubes and mesh simplification for 3D volumetric segmentation data. Generates 3D surface meshes from labeled volumes with built-in simplification and export to OBJ/PLY/Neuroglancer formats.

## Key Features
- Marching cubes on multi-label 3D images (all labels in one pass)
- Mesh simplification with configurable reduction factor and max error
- Topology-preserving and non-topology-preserving (FQMR) simplification
- Connected component analysis (face and vertex based)
- Dust removal and largest-k component retention
- Export to OBJ, PLY, and Neuroglancer Precomputed formats

## API
```python
from zmesh import Mesher

mesher = Mesher((4, 4, 40))  # voxel anisotropy
mesher.mesh(labels, close=False)
mesh = mesher.get(obj_id, reduction_factor=100, max_error=8)
mesh = zmesh.simplify_fqmr(mesh, triangle_count=1000)
mesh.to_obj()  # or .to_ply(), .to_precomputed()
```

## Relevance to Connectomics
Converts dense voxel segmentation volumes into 3D surface meshes for visualization and analysis of reconstructed neurons.
