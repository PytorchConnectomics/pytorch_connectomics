# waterz - Watershed and Region Agglomeration Library

**Location:** `/projects/weilab/weidf/lib/waterz/`
**Version:** 0.8
**License:** MIT
**Origin:** Fork of [funkey/waterz](https://github.com/funkey/waterz) by donglaiw, with CREMI scoring functions from Mala_v2.zip
**Language:** Python + Cython + C++11 (Boost multi_array)
**Dependencies:** cython, numpy, scipy, mahotas, boost (C++ headers)

## Purpose

Waterz ("water-zed") performs watershed segmentation and hierarchical region agglomeration on 3D affinity graphs. It is the core post-processing library for converting voxel-level affinity predictions (from neural networks) into instance segmentations of neurons/organelles in connectomics EM volumes.

## Installation

```bash
conda create -n zw python==3.8 cython numpy
pip install --editable /projects/weilab/weidf/lib/waterz
# or: cd /projects/weilab/weidf/lib/waterz && python setup.py build_ext --inplace
```

## Architecture Overview

```
waterz/
  __init__.py               # Public API: agglomerate(), waterz(), watershed(), etc.
  seg_waterz.py             # High-level waterz() and getRegionGraph() wrappers
  seg_watershed.py          # Python 2D watershed (mahotas-based, slice-by-slice)
  seg_region_graph.py       # Soma-aware BFS merging, branch IoU utilities
  seg_util.py               # Helpers: scoring function string builder, HDF5 I/O, border masks
  agglomerate.pyx           # Cython bridge: agglomerate() -> C++ initialize/mergeUntil/free
  region_graph.pyx          # Cython bridge: merge_id() variants -> C++ union-find merging
  evaluate.pyx              # Cython bridge: Rand/VOI metrics -> C++ compare_volumes
  frontend_agglomerate.h/cpp  # C++ agglomeration pipeline (WaterzContext state machine)
  frontend_region_graph.h/cpp # C++ union-find merge with optional affinity/count filtering
  frontend_evaluate.h/cpp     # C++ evaluation (compare_arrays, chunked statistics)
  frontend_basic.h            # Type definitions: SegID=uint32, AffValue=uint8, ScoreValue=uint8
  backend/                    # C++ template library (header-only)
    types.hpp                 # boost::multi_array typedefs, watershed_traits
    basic_watershed.hpp       # C++ watershed on affinity graph (BFS plateau division)
    RegionGraph.hpp           # Region Adjacency Graph with node/edge maps
    region_graph.hpp          # Extract RAG from segmentation + affinities
    IterativeRegionMerging.hpp  # Priority-queue agglomeration engine
    PriorityQueue.hpp         # Min-heap priority queue wrapper
    BinQueue.hpp              # Discretized bin queue (approximate priority queue)
    StatisticsProvider.hpp    # Base class with merge/edge callbacks
    MergeProviders.hpp        # Template meta-programming to combine providers
    CompoundProvider.hpp      # Multiple inheritance provider combiner
    MergeFunctions.hpp        # Scoring functions: MinSize, MaxSize, MinAffinity, MeanAffinity, etc.
    Operators.hpp             # Composable operators: OneMinus, One255Minus, Multiply, Add, etc.
    MeanAffinityProvider.hpp  # Running mean of edge affinities
    MinAffinityProvider.hpp   # Min affinity per edge
    MaxAffinityProvider.hpp   # Max affinity per edge
    HistogramQuantileProvider.hpp  # Histogram-based quantile (approximate, 256 bins)
    VectorQuantileProvider.hpp    # Exact quantile via nth_element
    MaxKAffinityProvider.hpp  # Top-K affinities per edge
    RegionSizeProvider.hpp    # Voxel count per region (node statistic)
    ContactAreaProvider.hpp   # Contact area per edge (edge statistic)
    RandomNumberProvider.hpp  # Random scoring (baseline)
    ConstantProvider.hpp      # Constant scoring
    Histogram.hpp             # Fixed-bin histogram data structure
    MaxKValues.hpp            # Sorted top-K value tracker
    discretize.hpp            # [0,1] <-> integer bin conversion
    evaluate.hpp              # Rand index and VOI computation
```

## Key Data Types

| Type | C++ | Notes |
|------|-----|-------|
| **Affinities** | `uint8_t[3][Z][Y][X]` | 3-channel (z/y/x neighbor) affinity predictions, range [0, 255] |
| **Segmentation** | `uint32_t[Z][Y][X]` | Fragment/segment IDs, 0 = background |
| **Ground truth** | `uint32_t[Z][Y][X]` | For evaluation |
| **Score** | `uint8_t` | Edge merge score (lower = more similar) |
| **Region graph** | `(uint32_t u, uint32_t v, uint8_t score)[]` | Weighted edge list |

**Important:** This fork operates on **uint8 affinities** (0-255 range), not float32. The Python `waterz()` wrapper and scoring functions are designed for this integer representation.

## Public Python API

### `waterz.waterz(affs, thresholds, ...)` - Main entry point

```python
import waterz
seg_list = waterz.waterz(
    affs,                        # [3,Z,Y,X] uint8 or float32 affinities
    thresholds=[0.1, 0.3, 0.6],  # agglomeration thresholds
    merge_function='aff50_his256',  # scoring function (see below)
    aff_threshold=[1, 254],      # low/high for initial watershed
    gt=None,                     # optional ground truth for metrics
    gt_border=25/4.0,            # border mask distance for GT
    fragments=None,              # pre-computed fragments (skip watershed)
    fragments_opt=0,             # 0: use C++ watershed; !=0: use mahotas watershed
    return_rg=False,             # also return region graph
    return_seg=True,             # return segmentation arrays
)
# Returns: list of uint32 segmentation arrays (one per threshold)
```

### `waterz.agglomerate(affs, thresholds, ...)` - Low-level generator

Returns a generator yielding `(segmentation, [metrics], [merge_history])` tuples. The segmentation array is modified in-place between yields (copy if needed).

### `waterz.watershed(affs, ...)` - 2D slice-by-slice watershed

```python
fragments = waterz.watershed(
    affs,                          # [3,Z,Y,X] affinities
    seed_method='maxima_distance', # 'grid', 'minima', 'maxima_distance', 'maxima_distance2'
    label_nb=np.ones([5,5]),       # structuring element for seed labeling
    bg_thres=1,                    # background threshold (<1 to assign background)
)
```

Uses mahotas `cwatershed` per 2D slice. Converts affinities to boundary map: `boundary = 1 - 0.5*(aff_y + aff_x) / 255`.

### `waterz.getRegionGraph(affs, fragments, ...)` - Extract region graph

```python
rg_ids, rg_scores = waterz.getRegionGraph(
    affs, fragments,
    rg_opt=1,                      # 1: all slices, 2: skip first, 3: z-border only
    merge_function='aff50_his256',
)
# rg_ids: [N,2] uint32 - edge endpoints
# rg_scores: [N] uint8 - edge scores (sorted ascending)
```

### `waterz.merge_id(id1, id2, ...)` - Union-find merging

```python
mapping = waterz.merge_id(
    id1, id2,              # [N] uint32 edge endpoint arrays
    score=None,            # [N] uint8 affinity scores (optional)
    count=None,            # [M] uint32 segment sizes (optional)
    id_thres=0,            # relabel threshold
    aff_thres=1,           # affinity threshold for filtering
    count_thres=50,        # size threshold (don't merge if both sides >= this)
    dust_thres=50,         # remove segments smaller than this
)
# Returns: [M] uint32 mapping array (old_id -> new_id)
```

Four merge modes based on which optional args are provided:
1. `score=None, count=None`: merge by ID only
2. `score!=None, count=None`: merge by ID + affinity threshold
3. `score=None, count!=None`: merge by ID + size constraint
4. `score!=None, count!=None`: merge by ID + affinity + size

### `waterz.evaluate_total_volume(seg, gt)` - Evaluation metrics

```python
metrics = waterz.evaluate_total_volume(seg_uint64, gt_uint64)
# Returns dict with: V_Rand_split, V_Rand_merge, V_Info_split, V_Info_merge
```

### Chunked evaluation

```python
stat = waterz.initialize_stats()
for chunk_seg, chunk_gt in chunks:
    stat = waterz.update_statistics_using_volume(stat, seg_uint16, gt_uint16)
metrics = waterz.compute_final_metrics(stat)
```

## Scoring Functions (Merge Functions)

Scoring functions are specified as C++ template type strings. The `getScoreFunc()` helper in `seg_util.py` translates shorthand notation:

### Shorthand notation

Format: `aff{Q}_his{B}[_ran255]` or `max{K}[_ran255]`

| Shorthand | C++ Type | Description |
|-----------|----------|-------------|
| `aff50_his256` | `OneMinus<HistogramQuantileAffinity<RG, 50, SV, 256>>` | Median affinity via 256-bin histogram |
| `aff50_his0` | `OneMinus<QuantileAffinity<RG, 50, SV>>` | Exact median affinity (vector-based) |
| `aff85_his256` | `OneMinus<HistogramQuantileAffinity<RG, 85, SV, 256>>` | 85th percentile via histogram |
| `aff50_his256_ran255` | `One255Minus<HistogramQuantileAffinity<RG, 50, SV, 256>>` | Same but score = 255 - quantile |
| `max10` | `OneMinus<MeanMaxKAffinity<RG, 10, SV>>` | Mean of top-10 affinities |

### Available C++ scoring primitives

**Edge statistics (from providers):**
- `MinAffinity` - minimum affinity across edge voxels
- `MaxAffinity` - maximum affinity
- `MeanAffinity` - running mean affinity
- `HistogramQuantileAffinity<RG, Q, Prec, Bins>` - Q-th percentile via histogram
- `QuantileAffinity<RG, Q, Prec>` - exact Q-th percentile
- `MeanMaxKAffinity<RG, K, Prec>` - mean of top-K affinities
- `ContactArea` - number of adjacent voxel pairs

**Node statistics:**
- `MinSize` / `MaxSize` - min/max region size of edge endpoints

**Operators (composable):**
- `OneMinus<F>` - `1 - f(e)` (converts affinity to distance)
- `One255Minus<F>` - `255 - f(e)` (for uint8 range)
- `Multiply<F1, F2>` - `f1(e) * f2(e)`
- `Add<F1, F2>` / `Subtract<F1, F2>`
- `Divide<F1, F2>` (safe division)
- `Invert<F>` - `1/f(e)`
- `Square<F>` - `f(e)^2`
- `Step<F1, F2>` - `f1(e) < f2(e) ? 0 : 1`

**Special:**
- `Random` - random score (baseline)
- `Constant<C>` - constant integer score

## Agglomeration Pipeline

1. **Watershed** (C++ `basic_watershed.hpp` or Python `seg_watershed.py`):
   - Finds local maxima in the affinity graph
   - BFS to divide plateaus
   - Assigns fragment IDs to each basin
   - Background (affinity below `aff_threshold_low`) gets ID 0

2. **Region Graph Extraction** (`region_graph.hpp`):
   - Scans all voxel pairs across the 3 affinity channels
   - Collects affinities between adjacent fragments
   - Builds RAG with edge statistics (via StatisticsProvider callbacks)

3. **Iterative Region Merging** (`IterativeRegionMerging.hpp`):
   - Scores all edges using the scoring function
   - Pushes edges into priority queue (min-score first)
   - Pops cheapest edge, merges regions if score < threshold
   - Updates incident edges (marks stale for lazy re-scoring)
   - Uses union-find with path compression for fast root lookup
   - Supports incremental merging across multiple thresholds

4. **Evaluation** (optional, `evaluate.hpp`):
   - Computes Rand index (split/merge) and VOI (split/merge)
   - Uses co-occurrence matrix between segmentation and ground truth

## Queue Types

- **PriorityQueue** (default): Standard min-heap, exact ordering
- **BinQueue**: Discretized into N bins, approximate but faster for large graphs

Selected at compile time via `discretize_queue` parameter (0 = PriorityQueue, N>0 = BinQueue with N bins).

## JIT Compilation

The `agglomerate()` function uses **just-in-time Cython compilation**. The scoring function is specified as a C++ type string, which gets written into a generated header file (`ScoringFunction.h`). The module is compiled once and cached in `~/.cython/inline/` with a hash-based filename for reuse.

## Connectomics-Specific Utilities

### `seg_region_graph.py`

- **`somaBFS(edges, soma_ids)`**: BFS-based soma-aware merging that prevents false merges between different soma bodies. Iteratively removes edges that would merge two somas into one segment.
- **`branchIoU(bbs, sid, ...)`**: Tracks neurite branches across z-slices using IoU overlap between consecutive 2D segmentations. Used for reconstructing neuron morphology from 2D segments.
- **`branchIoUBFS(bbs, sid, ...)`**: BFS extension of branchIoU that recursively finds all branches belonging to a neuron by following IoU connections.

### `seg_util.py`

- **`create_border_mask(gt, max_dist, bg_label)`**: Creates border masks on ground truth by labeling boundary pixels (within `max_dist` of a label boundary) as background. Used for fair evaluation by ignoring ambiguous boundary regions.
- **`mappingToList(mapping)`**: Converts a dense mapping array to a sparse `[N,2]` list of `(old_id, new_id)` pairs for efficient I/O.

## Usage in PyTorch Connectomics

Waterz is used in the decoding/post-processing pipeline of PyTorch Connectomics for:
1. Converting predicted affinity maps to initial over-segmentation (watershed)
2. Agglomerating fragments into neuron instances at various thresholds
3. Extracting region graphs for downstream processing
4. Evaluating segmentation quality (Rand/VOI metrics)

Typical workflow:
```python
import waterz
import numpy as np

# affs: [3, Z, Y, X] uint8 affinity predictions from model
seg = waterz.waterz(
    affs,
    thresholds=[0.3],
    merge_function='aff50_his256',
    aff_threshold=[1, 254],
)[0]  # single threshold -> single segmentation
```
