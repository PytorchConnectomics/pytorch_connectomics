# chunkflow

**GitHub:** https://github.com/seung-lab/chunkflow
**Language:** Python | **Stars:** 55

Composable chunk operator pipeline for local or distributed petabyte-scale 3D image processing. Proven at 18+ petabytes across 3600 GPU nodes on Google Cloud.

## Key Features
- Composable CLI operators (inference, segmentation, meshing, I/O, etc.)
- Hybrid cloud distributed computation via AWS SQS task scheduling
- Supports HDF5, TIFF, Zarr, PNG, Neuroglancer Precomputed formats
- ConvNet inference with PyTorch, connected components, agglomeration
- Neuroglancer visualization integration

## API
```bash
chunkflow \
    load-tif --file-name image.tif -o image \
    inference --convnet-model model.py --convnet-weight-path weight.pt \
        --input-patch-size 20 256 256 -i image -o affs \
    plugin -f agglomerate --threshold 0.7 -i affs -o seg \
    neuroglancer -i image,affs,seg -p 33333
```

## Relevance to Connectomics
Production-scale pipeline for distributed EM image segmentation, from inference through agglomeration and mesh generation.
