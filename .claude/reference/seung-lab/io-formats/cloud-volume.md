# cloud-volume

**GitHub:** https://github.com/seung-lab/cloud-volume
**Language:** Python | **Stars:** 170

Serverless Python client for random access reading and writing of Neuroglancer Precomputed volumes, meshes, and skeletons. The primary I/O library for petavoxel-scale connectomics datasets on cloud storage (S3, GCS, local filesystem).

## Key Features
- Random access read/write of petavoxel Neuroglancer images, meshes, and skeletons
- Supports Precomputed, Graphene (proofreading), Zarr, N5 (read-only), and BOSS formats
- Multi-threaded, multi-process, and green thread support
- Lossless connectomics codecs (compressed_segmentation, compresso, crackle, fpzip, zfpc, png, brotli)
- Image hierarchy and anisotropic resolution support
- Sharded format support for petascale dataset cost savings
- Nearly all output immediately visualizable in Neuroglancer
- Shared memory support for memory optimization

## API
```python
from cloudvolume import CloudVolume, Bbox

vol = CloudVolume('gs://mylab/mouse/image', parallel=True, progress=True)
image = vol[:,:,:]               # download whole stack as numpy array
vol[:,:,:] = image               # upload numpy array to cloud

mesh = vol.mesh.get(label)       # download mesh for segment
skel = vol.skeleton.get(label)   # download skeleton for segment
```

## Installation
```bash
pip install cloud-volume
pip install cloud-volume[all_codecs,all_viewers]  # with all optional features
```

## Relevance to Connectomics
The central I/O library for the seung-lab ecosystem; used by Igneous, python-task-queue, and neuroglancer to read/write EM image volumes, segmentations, meshes, and skeletons at petascale.
