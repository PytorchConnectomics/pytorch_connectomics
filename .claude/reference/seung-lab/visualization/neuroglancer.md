# neuroglancer

**GitHub:** https://github.com/seung-lab/neuroglancer
**Language:** TypeScript | **Stars:** 24

Seung Lab fork of Google's Neuroglancer -- a WebGL-based viewer for volumetric data. Capable of displaying arbitrary cross-sectional views, 3D meshes, and skeleton models from EM datasets.

## Key Features
- WebGL 2.0 based rendering of volumetric image data
- Arbitrary (non axis-aligned) cross-sectional views
- 3D mesh and skeleton visualization
- Multi-threaded architecture (frontend UI + backend WebWorker)
- Supports BOSS, DVID, Render, Precomputed, NIfTI, N5, and Python in-memory formats
- Python bindings for in-memory volume display

## Data Sources
- Precomputed chunks over HTTP (primary format for seung-lab pipelines)
- BOSS, DVID, Render, NIfTI, N5
- Python in-memory volumes with auto mesh generation

## Relevance to Connectomics
The primary interactive visualization tool for EM connectomics data; used to inspect images, segmentations, meshes, and skeletons throughout the reconstruction pipeline.
