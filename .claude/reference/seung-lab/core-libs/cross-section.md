# cross-section

**GitHub:** https://github.com/seung-lab/cross-section
**Language:** C++ | **Stars:** 6

Compute cross-sectional area and arbitrary 2D slice projections of 3D volumetric image objects. Published as the `xs3d` PyPI package.

## Key Features
- Cross-sectional area measurement at any point/orientation in a 3D binary image
- Arbitrary-angle 2D slicing of 3D volumes
- Anisotropy-aware with physical unit support
- Edge contact detection for underestimate warnings
- Per-voxel area contribution maps

## API
```python
import xs3d
area = xs3d.cross_sectional_area(binary_image, vertex, normal, resolution)
area, contact = xs3d.cross_sectional_area(binary_image, vertex, normal, resolution, return_contact=True)
image2d = xs3d.slice(labels, vertex, normal, anisotropy)
section_map = xs3d.cross_section(binary_image, vertex, normal, resolution)
```

## Relevance to Connectomics
Measures neurite caliber (cross-sectional area) along skeletons for compartment simulations and morphological analysis.
