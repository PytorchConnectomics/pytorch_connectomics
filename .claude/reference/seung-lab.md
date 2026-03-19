# Seung Lab Repository Index

**GitHub:** https://github.com/seung-lab (186 repos)

Princeton Seung Lab — tools for connectomics: large-scale EM reconstruction, visualization, and analysis. Reference docs for each repo are in `.claude/reference/seung-lab/`.

## Directory Structure

```
seung-lab/
├── core-libs/          (11)  Image processing libraries (PyPI packages)
├── segmentation-pipeline/ (17)  Watershed, agglomeration, meshing, skeletonization
├── io-formats/         (21)  I/O, compression, cloud storage
├── visualization/       (8)  Neuroglancer, viewers, rendering
├── registration/       (13)  Stitching, alignment, elastic registration
├── deep-learning/      (20)  CNN architectures, training, inference
├── infrastructure/     (22)  Task queues, pipelines, deployment
├── julia/              (10)  Julia-language packages
├── datasets-papers/    (17)  Paper code, connectome datasets
└── misc/               (48)  Forks, utilities, archived
```

## Core Libraries (used by PyTC)

| Package | Repo | Stars | PyTC Usage |
|---------|------|-------|------------|
| `cc3d` | connected-components-3d | 450 | Connected component labeling in decoding |
| `edt` | euclidean-distance-transform-3d | 261 | Distance transforms for SDT targets |
| `kimimaro` | kimimaro | 193 | TEASAR skeletonization for skeleton-aware EDT |
| `fastremap` | fastremap | 63 | Fast label remapping in branch_merge |
| `crackle-codec` | crackle | 15 | Transitive dep via kimimaro |
| `fill-voids` | fill_voids | 29 | Hole filling in morphological ops |
| `xs3d` | cross-section | 6 | Cross-sectional area computation |
| `dijkstra3d` | dijkstra3d | 84 | Shortest path (used by kimimaro) |

## Segmentation Pipeline

| Repo | Stars | Description |
|------|-------|-------------|
| **abiss** | 6 | Affinity-based instance segmentation (C++ CLI) |
| **zmesh** | 72 | Marching cubes + mesh simplification |
| **watershed** | 6 | C++ watershed on affinity graphs |
| **segascorus** | 6 | Rand/VOI segmentation error metrics |
| **pcg_skel** | 0 | ChunkedGraph skeletonization |
| **Synaptor** | 0 | Synapse detection pipeline |
| **MMAAPP** | 2 | Mean affinity agglomeration |

## I/O & Cloud

| Repo | Stars | Description |
|------|-------|-------------|
| **cloud-volume** | 170 | Read/write Neuroglancer Precomputed volumes |
| **cloud-files** | 44 | Threaded GCS/S3/local file client |
| **fpzip** | 36 | Floating-point compression |
| **compresso** | 4 | Segmentation compression (600-2200x) |
| **DracoPy** | 117 | Google Draco mesh compression |
| **tinybrain** | 11 | Image pyramid generation |
| **mapbuffer** | 10 | Fast serialized int-to-bytes dict |

## Visualization

| Repo | Stars | Description |
|------|-------|-------------|
| **neuroglancer** | 24 | WebGL volumetric data viewer (Seung fork) |
| **microviewer** | 16 | Browser-based 3D numpy viewer |
| **NeuroBlender** | 8 | Blender neuron visualization |

## Registration & Alignment

| Repo | Stars | Description |
|------|-------|-------------|
| **SEAMLeSS** | 9 | ML-based EM section alignment |
| **corgie** | 16 | Petascale volume registration CLI |
| **metroem** | 9 | EM alignment model training |
| **feabas** | 0 | Finite-element EM stitching |
| **Alembic** | 10 | Julia elastic registration |

## Deep Learning

| Repo | Stars | Description |
|------|-------|-------------|
| **znn-release** | 94 | Multi-core 3D ConvNet (historical, archived) |
| **NCCNet** | 40 | Normalized cross-correlation template matching |
| **DeepEM** | 16 | Deep learning for EM connectomics |
| **chunkflow** | 55 | Distributed petabyte-scale processing |
| **torchfields** | 51 | PyTorch displacement field / spatial transformers |

## Infrastructure

| Repo | Stars | Description |
|------|-------|-------------|
| **igneous** | 66 | Scalable downsampling, meshing, skeletonizing |
| **python-task-queue** | 39 | SQS/filesystem async task queue |
| **seuron** | 7 | Distributed neuron reconstruction pipeline |
| **CAVEpipelines** | 5 | ChunkedGraph/meshing/L2cache deployment |

## Datasets & Papers

| Repo | Stars | Description |
|------|-------|-------------|
| **FlyConnectome** | 17 | FlyWire connectome data access |
| **FlyWirePaper** | 3 | FlyWire paper figure reproduction |
| **MicronsBinder** | 3 | MICrONS dataset notebooks |
| **zebrafish** | 1 | Zebrafish hindbrain connectome |
| **e2198-gc-analysis** | 8 | Retinal ganglion cell connectomics |
