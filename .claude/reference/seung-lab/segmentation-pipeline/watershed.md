# watershed

**GitHub:** https://github.com/seung-lab/watershed
**Language:** C++ | **Stars:** 6

C++ implementation of watershed segmentation and single linkage clustering on affinity graphs. Given affinity predictions from a ConvNet, produces watershed basins and a merge hierarchy (dendrogram) for subsequent agglomeration.

## Key Features
- Watershed segmentation on 3D affinity graphs
- Single linkage clustering / hierarchical merging
- Size-dependent subtree collapsing to reduce oversegmentation
- High/low threshold controls for basin merging

## Relevance to Connectomics
Core watershed algorithm for converting affinity map predictions into initial oversegmentation -- the first step in instance segmentation pipelines. The Julia port is Watershed.jl.
