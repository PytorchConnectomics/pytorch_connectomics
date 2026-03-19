# ca3_paper

**GitHub:** https://github.com/seung-lab/ca3_paper
**Language:** Jupyter Notebook | **Stars:** 1

Code repository for the paper "Connectomic reconstruction from hippocampal CA3 reveals spatially graded mossy fiber inputs and selective feedforward inhibition to pyramidal cells."

## Key Features
- CloudVolume access to mouse hippocampal CA3 EM and segmentation data
- CAVE-based synapse and annotation queries
- Mossy fiber bouton extraction pipeline
- Cellpose-based vesicle segmentation
- Data hosted at pyr.ai with Codex cell type explorer

## API
```python
from cloudvolume import CloudVolume as cv
vol = cv('gs://zheng_mouse_hippocampus_production/v2/seg_m195', use_https=True)
mesh = vol.mesh.get(segment_id)
skeleton = vol.skeleton.get(segment_id)
```

## Relevance to Connectomics
Demonstrates a complete connectomics analysis workflow from EM reconstruction to circuit-level neuroscience findings in mammalian brain.
