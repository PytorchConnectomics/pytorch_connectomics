# Watershed.jl

**GitHub:** https://github.com/seung-lab/Watershed.jl
**Language:** Julia | **Stars:** 6

Julia implementation of 3D hierarchical watershed segmentation on affinity graphs. Translation of Zlateski's C++ watershed code. Produces watershed basins and a single-linkage clustering hierarchy (dendrogram) from affinity maps.

## Key Features
- Watershed segmentation on 3D affinity graphs (6-connectivity)
- Hierarchical single linkage clustering with dendrogram output
- High threshold collapsing to reduce oversegmentation
- Size-dependent subtree collapsing for small basin removal
- Low threshold background detection (singletons labeled as 0)

## Citation
Zlateski & Seung (2015) "Image segmentation by size-dependent single linkage clustering of a watershed basin graph" (arXiv:1505.00249)

## Relevance to Connectomics
Core watershed algorithm for converting affinity predictions into initial oversegmentation; essential first step before agglomeration in EM neuron reconstruction.
