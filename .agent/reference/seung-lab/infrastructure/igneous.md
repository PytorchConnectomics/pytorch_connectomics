# igneous

**GitHub:** https://github.com/seung-lab/igneous
**Language:** Python | **Stars:** 66

Scalable pipeline for producing and managing Neuroglancer Precomputed volumes. Handles downsampling, meshing, skeletonization, contrast normalization, and data transfers at petavoxel scale using CloudVolume and python-task-queue.

## Key Features
- Image downsampling (multi-resolution pyramid generation)
- Mesh generation from segmentation volumes
- Skeleton extraction (via Kimimaro TEASAR)
- Contrast normalization (CLAHE)
- Volume transfers between cloud storage providers
- Connected component labeling
- CLI interface and Python scripting API
- SQS cloud queues or filesystem-based local queues
- Docker container for horizontal scaling on Kubernetes
- FileQueue for cluster use on shared filesystems

## CLI Usage
```bash
igneous image downsample file://./my-data --mip 0 --queue ./ds-queue
igneous mesh forge s3://my-data/seg --mip 2 --queue sqs://mesh-queue
igneous skeleton forge s3://my-data/seg --mip 2 --queue sqs://skel-queue
igneous skeleton merge s3://my-data/seg --queue sqs://skel-queue
igneous execute -x ./ds-queue  # execute queue, exit when done
```

## Python API
```python
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc

tq = LocalTaskQueue(parallel=8)
tasks = tc.create_meshing_tasks('gs://bucket/labels', mip=3, shape=(256,256,256))
tq.insert(tasks)
tq.execute()
```

## Installation
```bash
pip install igneous-pipeline
```

## Relevance to Connectomics
The standard post-processing pipeline for seung-lab: generates multi-resolution image pyramids, meshes, and skeletons from EM segmentations for visualization in Neuroglancer.
