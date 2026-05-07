# Agglomeration

**GitHub:** https://github.com/seung-lab/Agglomeration
**Language:** Julia | **Stars:** 1

Julia package for supervoxel agglomeration -- merging oversegmented supervoxels into neuron segments. Processes affinity graphs and initial segmentations to produce dendrograms.

## API
```julia
using Agglomeration, Process
dend, dendValues = Process.forward(aff, segm)
```

## Relevance to Connectomics
Implements supervoxel agglomeration, a key post-processing step that merges watershed oversegmentation into neuron instances using affinity predictions.
