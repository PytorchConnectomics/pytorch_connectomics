# em_pipeline Reference

**Location**: `/projects/weilab/weidf/lib/em_pipeline`

Multi-GPU/multi-core pipeline engine for large-scale 3D EM neuron segmentation and reconstruction. Implements a two-stage approach: voxel→supervoxel (waterz watershed) then supervoxel→instance (region graph + branch resolution). Designed for distributed, chunked processing of datasets like zebrafish brain (5700x10913x10664 voxels).

## Directory Structure

```
em_pipeline/
├── main.py                      # CLI entry point (106 lines), SLURM integration
├── test.py                      # Dev/visualization script (983 lines)
├── conf/
│   ├── j0126.yml                # Project config (zebrafish example)
│   └── cluster.yml              # Cluster/environment config
├── db/                          # Local database/cache (HDF5 files)
├── em_pipeline/
│   ├── tasks/
│   │   ├── __init__.py          # Task factory
│   │   ├── task.py              # Base Task class (34 lines)
│   │   ├── waterz.py            # Waterz segmentation tasks (217 lines)
│   │   ├── branch.py            # Branch resolution tasks (363 lines)
│   │   ├── region_graph.py      # Region graph / soma BFS (104 lines)
│   │   └── eval.py              # ERL evaluation (25 lines)
│   └── lib/
│       └── rpca.py              # Robust PCA (68 lines)
├── setup.py
├── environment.yml
└── README.md
```

~1,933 lines total.

## Architecture

### Two-Stage Pipeline

```
Affinity Predictions (HDF5/Zarr)
  ↓  Stage 1: Voxel → Supervoxel
Waterz Segmentation (per-chunk supervoxels)
  ↓  Chunk merging (global segment IDs)
  ↓  Soma constraint (BFS-based soma assignment)
  ↓  Stage 2: Supervoxel → Instance
Branch Resolution (S1/S2/S3 IOU+affinity checks)
  ↓
Skeleton Generation (multi-scale morphology)
```

### Task Class Hierarchy

```
Task (base)
├── WaterzTask           # Chunk-wise waterz segmentation
├── WaterzSoma2DTask     # 2D soma-aware waterz (per z-slice)
├── WaterzStatsTask      # Consolidate stats across chunks
├── BranchChunkTask      # Per-chunk branch resolution
│   ├── BranchBorderTask # Cross-chunk boundary handling
│   └── BranchAllTask    # Global aggregation + skeletonization
├── RegionGraphChunkTask # Soma constraints on region graphs
├── RegionGraphBorderTask
└── ERLTask              # Evaluation against ground truth
```

## Usage

```bash
# CLI entry point
python main.py -c conf/j0126.yaml -t [task] -i [job_id] -n [job_num] -nc [num_cpu]

# Stage 1: Voxel → Supervoxel
python main.py -c conf/j0126.yaml -t waterz
python main.py -c conf/j0126.yaml -t waterz-stats
python main.py -c conf/j0126.yaml -t rg-border
python main.py -c conf/j0126.yaml -t rg-all

# Stage 2: Supervoxel → Instance
python main.py -c conf/j0126.yaml -t branch-border -o relabel
python main.py -c conf/j0126.yaml -t branch-all -o s2-4-8-8
```

CLI arguments: `-c` config file, `-t` task name, `-i` job index, `-n` total jobs, `-nc` CPUs per job, `-p` SLURM partition.

## Configuration (YAML)

### Project Config (`conf/j0126.yml`)

| Section | Keys | Purpose |
|---------|------|---------|
| `im` | path, shape, tile_shape, res | Input image volume |
| `mask` | blood_vessel, soma, border, soma_ratio, soma_id0/id1 | Segmentation constraints |
| `aff` | path, aff_shape, low, high | Affinity predictions + thresholds |
| `waterz` | mf, thres, num_z, nb, opt_frag, small_size, small_aff, small_dust, bg_thres | Watershed parameters |
| `branch` | s1_iou, s1_sz, s1_rg, s3_iou, s3_sz, skel_dust | Branch resolution thresholds |
| `rg` | thres_z | Region graph parameters |
| `output` | path | Output directory |
| `eval` | val, test | Evaluation datasets |

### Cluster Config (`conf/cluster.yml`)

Keys: `folder`, `env` (setup commands), `python` (executable), `num_gpu`, `memory`.

## Key Algorithms

### Waterz Task
- Applies affinity masks (blood vessel, border)
- Runs waterz agglomeration with configurable merge function and threshold
- Generates per-chunk region graphs
- Output: HDF5 with `seg`, `id`, `score` datasets

### Soma-Aware Waterz (2D)
- Processes each z-slice independently
- Integrates soma mask constraints via seeded watershed
- Resolves soma-based false splits/merges

### Branch Resolution (3 stages)
- **S1**: IOU-based merge scoring within chunks
- **S2**: IOU best-buddy pairing (bidirectional agreement)
- **S3**: One-sided IOU + affinity validation for remaining candidates

### Soma BFS
- Breadth-first search to grow soma regions outward
- Assigns non-soma segments to nearest soma
- Handles ambiguous segments (multiple soma connections)

## Data Formats

- **HDF5** (.h5): Primary I/O for volumes and results
- **Zarr** (.zarr): Large-scale chunked arrays
- **Pickle**: Dask arrays and serialized objects
- **PNG/TIFF**: 2D mask/image files

## Dependencies

### Core (from environment.yml)
numpy 1.24.3, scipy, h5py, pyyaml 6.0.2, pillow 10.4.0, cc3d 3.18.0, fastremap 1.15.0, mahotas 1.4.18, zarr 2.16.1, dask 2023.5.0, networkx 3.1, cloudpickle 3.0.0

### External (installed separately)
- **em_util**: `git clone git@github.com:PytorchConnectomics/em_util.git && pip install -e .`
- **waterz**: `git clone -b affuint8 git@github.com:donglaiw/waterz.git && pip install -e .`
- **zwatershed**: `git clone git@github.com:donglaiw/zwatershed.git && pip install -e .`
- **em_erl**: Evaluation utilities (referenced in code)

## Design Principles

1. **Chunked processing**: Overlapping chunks for parallel execution on large volumes
2. **Hierarchical resolution**: Sequential refinement from coarse to fine
3. **Constraint integration**: Soma masks and boundary constraints guide segmentation
4. **IOU-based merging**: Intersection-over-union drives segment agglomeration
5. **Skeleton output**: Multi-scale neuron skeletons for morphological analysis
