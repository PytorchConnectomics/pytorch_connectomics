# em_util Reference

**Location:** `/projects/weilab/weidf/lib/em_util`
**GitHub:** https://github.com/PytorchConnectomics/em_util
**License:** MIT

Utility library for EM connectomics: volume I/O, segmentation operations, evaluation metrics, neuroglancer visualization, WebKnossos integration, and SLURM job management.

## Module Overview

| Module | Purpose |
|--------|---------|
| `em_util.io` | Universal I/O (HDF5, TIFF, PNG, Zarr, CloudVolume), bbox, skeleton, chunked ops, UnionFind |
| `em_util.seg` | Segmentation ops: relabel, remove small, morphology, connected components, IoU |
| `em_util.eval` | Metrics: adapted_rand, VOI, confusion matrix |
| `em_util.ng` | Neuroglancer 3D visualization |
| `em_util.wk` | WebKnossos remote dataset management |
| `em_util.vast` | VAST annotation format support |
| `em_util.cluster` | SLURM job submission utilities |

## Conventions

- **Volume axis order:** ZYX (depth, height, width)
- **Bounding boxes:** 2D: `[seg_id, y0, y1, x0, x1, count]`, 3D: `[seg_id, z0, z1, y0, y1, x0, x1, count]`
- **Segmentation:** Integer arrays, 0 = background

## Key Functions

### I/O (`em_util.io`)

```python
read_vol(filename, dataset=None)          # Universal reader (.h5, .tif, .npy, .zarr, etc.)
read_h5(filename, dataset=None)           # HDF5 reader
write_h5(filename, data, dataset="main")  # HDF5 writer (gzip compressed)
read_image(filename, image_type="image")  # Single image reader
read_image_folder(filename, ...)          # Image stack reader (glob patterns)
compute_bbox_all(seg, do_count=False)     # Bounding boxes for all segments
vol_to_skel(labels, res=(32,32,30))       # Kimimaro skeletonization
```

### Segmentation (`em_util.seg`)

```python
seg_to_count(seg, do_sort=True)           # Segment sizes (sorted)
seg_relabel(seg, do_sort=True)            # Relabel by size (largest=1)
seg_remove_small(seg, threshold=100)      # Remove small segments
seg_to_cc(seg)                            # Connected component relabeling
seg_biggest_cc(seg)                       # Keep only largest CC per label
```

### IoU (`em_util.seg.iou`) — Key for branch merge

```python
seg_to_iou(seg0, seg1, uid0=None, bb0=None, uid1=None, uc1=None, th_iou=0)
```
Compute IoU between two segmentations (2D or 3D). Uses bounding-box-accelerated overlap counting — only scans the bbox region of each segment in seg0, masking against seg1.

Returns `(N, 5)` array: `[seg_id, best_match_id, count0, count1, overlap_count]`

```python
segs_to_iou(get_seg, index, th_iou=0)
```
Track segments across z-slices. `get_seg(i)` returns 2D segmentation at slice `i`. Iterates consecutive pairs, computing IoU with bbox acceleration. Returns list of overlap matrices (one per boundary).

**Performance:** Bbox-accelerated — only scans pixels within each segment's bounding box, making it fast for sparse segmentations where segments occupy a small fraction of the image.

### Evaluation (`em_util.eval`)

```python
adapted_rand(seg, gt, all_stats=False)    # SNEMI3D adapted rand error
voi(seg, gt)                              # VOI (split, merge)
```

### Chunked Processing (large volumes)

```python
vol_func_chunk(input_file, vol_func, ...)        # Apply function chunk-by-chunk
vol_downsample_chunk(input_file, ratio=[1,2,2])  # Downsample large HDF5
compute_bbox_all_chunk(seg_file, chunk_num=1)     # Bbox from large file
```

### UnionFind (`em_util.io`)

```python
uf = UnionFind(elements)
uf.union(a, b)                 # Merge two sets
uf.find(a)                     # Find root
uf.components()                # List all components
uf.component_relabel_arr()     # Numpy relabel array
```

## Dependencies

**Core:** numpy, scipy, h5py, imageio, scikit-image, tqdm, cc3d, pyyaml, networkx
**Optional:** cloudvolume, zarr, kimimaro, neuroglancer, webknossos, igneous
