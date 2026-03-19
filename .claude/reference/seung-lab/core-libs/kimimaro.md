# kimimaro

**GitHub:** https://github.com/seung-lab/kimimaro
**Language:** C++ | **Stars:** 193

Rapidly skeletonize all non-zero labels in 2D/3D numpy arrays using a TEASAR-derived method. Produces medial axis transforms (skeletons with boundary distance) in cloud-volume Skeleton format.

## Key Features
- Single-pass multi-label skeletonization (no per-label masking)
- TEASAR algorithm with soma detection and invalidation
- Cross-sectional area computation along skeletons
- Join close disconnected components within radius
- Synapse-to-skeleton target mapping
- CLI for SWC file generation and viewing
- Parallel multi-process execution
- Handles anisotropic voxel spacing

## API
```python
import kimimaro

skels = kimimaro.skeletonize(
    labels,
    teasar_params={"scale": 1.5, "const": 300, "pdrf_scale": 100000, "pdrf_exponent": 4},
    dust_threshold=1000, anisotropy=(16, 16, 40),
    fix_branching=True, fix_borders=True, progress=True, parallel=1,
)
skel = kimimaro.postprocess(skel, dust_threshold=1000, tick_threshold=3500)
skels = kimimaro.cross_sectional_area(labels, skels, anisotropy=(16, 16, 40))
skel = kimimaro.join_close_components([skel1, skel2], radius=1500)
```

## Relevance to Connectomics
Core dependency of PyTC (listed in requirements). Used for skeletonizing neuron instance segmentations to extract morphological structure, enabling skeleton-based metrics and analysis.
