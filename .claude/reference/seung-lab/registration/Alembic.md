# Alembic

**GitHub:** https://github.com/seung-lab/Alembic
**Language:** Jupyter Notebook / Julia | **Stars:** 10

ALignment of Electron Microscopy By Image Correlograms. A Julia toolbox for elastic image registration of serial EM sections using block matching, triangle mesh deformation, and piecewise affine rendering.

## Key Features
- Block matching between coarsely-aligned EM sections
- Automatic and manual match filtering
- Triangle mesh-based elastic deformation solver
- Piecewise affine rendering to CloudVolume
- Distributed task scheduling via AWS SQS + Kubernetes

## API
```julia
ms = make_stack()        # create meshset from params
match!(ms)               # blockmatch between meshes
elastic_solve!(ms)       # relax the spring system
render(ms)               # render and save to CloudVolume
```

## Relevance to Connectomics
Elastic registration pipeline for aligning serial EM sections -- a critical preprocessing step before segmentation.
