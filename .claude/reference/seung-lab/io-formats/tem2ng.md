# tem2ng

**GitHub:** https://github.com/seung-lab/tem2ng
**Language:** Python | **Stars:** 1

Convert raw TEM microscope images into a Neuroglancer-compatible volume. Handles ingestion of transmission electron microscopy data into the cloud-based Precomputed format.

## Key Features
- TEM image ingestion to Neuroglancer Precomputed format
- Configurable dataset size, resolution, and chunk size
- Subtile upload support

## Usage
```bash
tem2ng info matrix://BUCKET/LAYER --dataset-size 10000,10000,1 --resolution 4,4,40 --chunk-size 1000,1000,1
tem2ng upload subtiles matrix://BUCKET/LAYER
```

## Relevance to Connectomics
Handles the first step of the EM pipeline: converting raw microscope images into a format suitable for visualization and downstream processing.
