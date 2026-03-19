# FlyConnectome

**GitHub:** https://github.com/seung-lab/FlyConnectome
**Language:** Jupyter Notebook | **Stars:** 17

Tutorials and documentation for programmatic access to the FlyWire connectome data, including volumetric EM data, segmentation, meshes, and synaptic annotations.

## Key Features
- CloudVolume-based access to EM and segmentation volumes
- CAVE integration for synaptic connection and annotation queries
- Per-segment mesh downloads
- Bulk download from Codex portal
- Compatible with Navis, natverse (R), and fafbseg

## API
```python
from caveclient import CAVEclient
client = CAVEclient("flywire_fafb_production")
from cloudvolume import CloudVolume
vol = CloudVolume("precomputed://gs://...")
```

## Relevance to Connectomics
Primary data access portal for the FlyWire whole-brain Drosophila connectome, a major connectomics dataset.
