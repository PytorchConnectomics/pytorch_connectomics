# pcg_skel

**GitHub:** https://github.com/seung-lab/pcg_skel
**Language:** None | **Stars:** 0

Robust skeletonization of PyChunkedGraph-backed objects. Generates neuronal skeletons directly from dynamic segmentations stored in the ChunkedGraph, using level-2 chunk topology.

## Key Features
- Skeleton generation from ChunkedGraph (PCG) dynamic segmentations
- Level-2 chunk-based topology derivation
- Mesh-based vertex refinement (chunk index to Euclidean space)
- Root point specification for skeleton orientation
- Integration with CAVEclient/FrameworkClient
- Annotation support via MaterializationEngine

## API
```python
from pcg_skel import chunk_index_skeleton, refine_chunk_index_skeleton

# Generate skeleton in chunk index space
sk_ch, l2dict, l2dict_reversed = chunk_index_skeleton(
    root_id, client=client, return_l2dict=True
)

# Refine to Euclidean (nanometer) space
sk = refine_chunk_index_skeleton(sk_ch, l2dict_reversed, cv)
```

## Relevance to Connectomics
Enables rapid skeletonization of neurons from dynamic proofreading segmentations, bridging the gap between chunked graph storage and morphological analysis.
