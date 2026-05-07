# em_coregistration

**GitHub:** https://github.com/seung-lab/em_coregistration
**Language:** Python | **Stars:** 0

Tools for aligning a 3D EM dataset to another modality (e.g., 2-photon optical data). Developed for the IARPA Allen EM / Baylor 2P coregistration project.

## Key Features
- 3D alignment solve between landmark points across modalities
- Thin Plate Spline (TPS) transform model
- Bidirectional transforms (EM-to-optical, optical-to-EM)
- Neuroglancer link generation for visualization
- Transform serialization to JSON

## API
```python
from alignment.solve_3d import Solve3D
s = Solve3D(input_data=data, args=[])
s.run()  # computes transform
s.transform.transform(source_points)  # apply transform
```

## Relevance to Connectomics
Enables registration of EM volumes with light microscopy data for correlative analysis of neural structure and function.
