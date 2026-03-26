# pytc-deploy

**Location:** `/projects/weilab/weidf/lib/pytc-deploy`
**License:** MIT (2024, donglai)
**Purpose:** Deployment/workflow management for EM connectomics data processing pipelines using PyTorch Connectomics. Orchestrates large-scale segmentation, instance merging, and visualization on SLURM clusters.

## Repository Structure

```
pytc-deploy/
├── util/                    # Shared utility modules
│   ├── __init__.py
│   ├── args.py              # CLI argument parsing
│   └── task.py              # Core segmentation algorithms
├── mito-h01/                # H01 dataset mitochondria processing
│   ├── main.py              # Pipeline orchestration (305 lines)
│   ├── const.py             # Dataset constants
│   └── param.yml            # SLURM/path configuration
├── nuc-worm/                # C. elegans nucleus/worm processing
│   ├── main.py              # Pipeline orchestration (302 lines, mirrors mito-h01)
│   ├── const.py             # Dataset constants
│   └── param.yml            # SLURM/path configuration
└── syn-alzhemier/           # Alzheimer's synapse analysis
    └── main.py              # Multi-step pipeline (858 lines)
```

## CLI Entry Point

All projects use:
```bash
python main.py -t <task> [flags]
```

## util/args.py — `get_parser()`

Returns an `ArgumentParser` with these flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `-t, --task` | `""` | Task name to execute |
| `-s, --cmd` | `""` | SLURM command |
| `-e, --env` | `"imu"` | Conda environment name |
| `-ji, --job-id` | `0` | Job ID for parallel processing |
| `-jn, --job-num` | `1` | Total number of jobs |
| `-cn, --chunk-num` | `1` | Number of chunks |
| `-n, --neuron` | `""` | Neuron IDs (comma-separated) |
| `-r, --ratio` | `"1,1,1"` | Downsample ratio (Z,Y,X) |
| `-cp, --partition` | `"lichtman"` | SLURM partition |
| `-cm, --memory` | `"50GB"` | Memory allocation |
| `-ct, --run-time` | `"0-12:00"` | Job runtime |
| `-cg, --num_gpu` | `-1` | Number of GPUs |

## util/task.py — Core Algorithms

### `generate_jobs_dl(conf, neuron, job_num=1, mem='50GB', run_time='1-00:00', job_order=1)`
Generates SLURM batch scripts for deep learning inference using PyTorch Connectomics.

### `neuron_to_tile(neuron, zid, zran, f_box, f_seg)`
Maps neuron IDs to tile coordinates. Returns bounding box and tile bounding boxes for a neuron.

### `seg_zran_merge(f_zran_p, job_num)`
Merges Z-range (min/max Z) data from parallel jobs. Returns merged ID array and Z-range array.

### `seg_zran_p(f_box, job_id, job_num)`
Computes Z-range for each segmentation ID in parallel. Returns array of `[ID, min_z, max_z]`.

### `seg_bbox_p(f_seg, f_box, job_id, job_num)`
Computes bounding boxes for all segmented objects in parallel, slice-by-slice. Writes to HDF5.

### `remove_small_instances(segm, thres_small=128, mode='background')`
Removes spurious small instances from segmentation.
- **Modes:** `none`, `background` (3D), `background_2d`, `neighbor` (merge with nearest, 3D), `neighbor_2d`

### `bc_watershed(volume, thres1=0.9, thres2=0.8, thres3=0.85, thres_small=128, scale_factors=(1.0,1.0,1.0), remove_small_mode='background', seed_thres=32, precomputed_seed=None)`
Converts binary foreground probability + instance contour maps to instance masks using watershed.
- `volume`: Shape `(C, Z, Y, X)` with 2 channels (foreground, boundary)
- `thres1`: Seed threshold (0.9)
- `thres2`: Contour threshold (0.8)
- `thres3`: Foreground threshold (0.85)

### `mito_watershed_iou(f_mito_ws_func, arr_mito)`
Computes IoU between adjacent tiles for instance matching across X, Y, Z directions.

### `mito_neuron_sid(f_mito_ws, arr_mito, ratio=0.6)`
Finds mitochondrial instance IDs within a neuron mask, filtered by overlap ratio (default 60%).

## mito-h01/main.py — Tasks

| Task | Description |
|------|-------------|
| `seg-bbox` | Compute bounding boxes per segmentation slice |
| `seg-zran_p` | Compute Z-ranges in parallel |
| `seg-zran` | Merge Z-range data from parallel jobs |
| `neuron-tile` | Map neuron IDs to tile coordinates |
| `mito-folder` | Create output directory structure |
| `mito-ts` | Write TensorStore config pickle |
| `mito-neuron-watershed` | Decode U-Net predictions to instances via watershed |
| `mito-neuron-watershed-iou` | Compute IoU between adjacent tiles |
| `mito-neuron-check` | Verify file completeness |
| `mito-neuron-sid` | Extract mito instance IDs within neuron mask |
| `mito-neuron-sid-count` | Cumulative count of instance IDs |
| `mito-neuron-sid-iou` | Merge instances across tiles using IoU + UnionFind |
| `mito-neuron-export` | Generate final HDF5 with instance relabeling |
| `mito-neuron-export-ds` | Downsample exported segmentation |
| `mito-neuron-ng` | Create Neuroglancer-compatible tiles |
| `mito-neuron-mesh` | Generate 3D mesh from segmentation |
| `mito-neuron-test` | Debugging/testing |
| `slurm` | Generate and submit SLURM batch jobs |

## mito-h01/const.py — Dataset Constants

- `neuron_volume_size = [1324, 15552, 27072]` (Z, Y, X voxels)
- `neuron_volume_offset = [0, 2560, 3520]`
- `neuron_tile_size = [25, 128, 128]`
- `mito_volume_ratio = [4, 16, 16]` (mito resolution vs neuron)
- `mito_tile_size = [100, 2048, 2048]`
- `neuron_id = [590612150, 36750893213]`

## nuc-worm/ — C. elegans Nucleus Processing

Code-identical to `mito-h01/` (same pipeline structure, different dataset parameters).

## syn-alzhemier/main.py — Alzheimer's Synapse Pipeline

Multi-step pipeline driven by numeric option codes:

| Option | Description |
|--------|-------------|
| `0.x` | Image preprocessing: extract frames, VAST-to-HDF5 conversion, downsampling |
| `2.x` | Vesicle processing: extraction, annotation processing, mito mask application |
| `3.x` | Data export & validation: range checks, consistency, bbox fixes |
| `4.x` | Vesicle classification: patch extraction, Laplacian quality scores, sorting |
| `5.0-5.2` | Load TIF stacks, convert to HDF5 |
| `5.3-5.4` | Tissue sample preparation and decoding |
| `5.5` | Generate test file list (72x9x8 = 5184 tiles) + SLURM jobs |
| `5.6x` | Instance merging: extract IDs, merge across tiles (IoU + UnionFind), relabel |
| `5.63` | TensorStore upload to Google Cloud (multi-scale pyramid: 1x, 4x, 8x) |
| `6.x` | Cell segmentation visualization, Neuroglancer setup |

### Key Functions in syn-alzhemier:
- **`merge_syn_ins()`**: Merges pre/post-synaptic instances across tile boundaries using UnionFind

## Data Flow (Mito-h01 Pipeline)

```
Raw segmentation → [seg-bbox] → [seg-zran_p] → [seg-zran]
    → [neuron-tile] → U-Net inference → [mito-neuron-watershed]
    → [mito-neuron-sid] → [mito-neuron-sid-iou]
    → [mito-neuron-export] → [mito-neuron-export-ds]
    → [mito-neuron-ng/mesh]
```

## Key Algorithms

1. **Watershed Segmentation**: Seed detection + watershed flooding for pixel-to-instance conversion
2. **UnionFind**: Disjoint set union for tracking/merging connected instances across tiles
3. **IoU-Based Merging**: Matches instances across tile boundaries by overlap threshold
4. **Tile-Based Parallelism**: SLURM job arrays for memory-efficient large-volume processing
5. **Multi-Scale Pyramids**: Downsampled representations for TensorStore/Neuroglancer visualization

## Dependencies

- **Core:** numpy, scipy, h5py, cv2, scikit-image, imageio
- **EM Utilities:** em_util (I/O, clustering, Neuroglancer helpers)
- **Segmentation:** cc3d, fastremap
- **Cloud Storage:** tensorstore (Google Cloud)
- **Custom:** T_util, T_util_seg
