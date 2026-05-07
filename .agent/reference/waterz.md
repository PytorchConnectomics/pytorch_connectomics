# waterz - Watershed and Region Agglomeration Library

**Location:** `lib/waterz/` (inside pytorch_connectomics repo)
**Origin:** Fork of [funkey/waterz](https://github.com/funkey/waterz) at [PytorchConnectomics/waterz](https://github.com/PytorchConnectomics/waterz)
**License:** MIT
**Language:** Python + Cython + C++11 (Boost multi_array)
**Dependencies:** numpy>=1.20, witty[cython]>=0.3.1, Cython>=0.29

## Purpose

Waterz performs watershed segmentation and hierarchical region agglomeration on 3D affinity graphs. It is the core post-processing library for converting voxel-level affinity predictions (from neural networks) into instance segmentations of neurons/organelles in connectomics EM volumes.

## Installation

```bash
pip install -e lib/waterz
# Requires: witty[cython]>=0.3.1 (JIT Cython compilation)
```

## Package Structure

```
lib/waterz/
├── pyproject.toml              # Modern packaging (setuptools + setuptools-scm)
├── setup.py                    # Legacy setup (Cython extensions)
├── src/waterz/
│   ├── __init__.py             # Public API: agglomerate(), evaluate()
│   ├── _agglomerate.py         # High-level agglomerate() wrapper (JIT via witty)
│   ├── agglomerate.pyx         # Cython bridge: -> C++ initialize/mergeUntil/free
│   ├── evaluate.pyx            # Cython bridge: Rand/VOI metrics -> C++ compare_volumes
│   ├── frontend_agglomerate.h/cpp  # C++ agglomeration pipeline
│   ├── frontend_evaluate.h/cpp     # C++ evaluation
│   └── backend/                # C++ template library (header-only)
│       ├── types.hpp           # boost::multi_array typedefs, watershed_traits
│       ├── basic_watershed.hpp # C++ watershed on affinity graph (BFS plateau division)
│       ├── RegionGraph.hpp     # Region Adjacency Graph with node/edge maps
│       ├── region_graph.hpp    # Extract RAG from segmentation + affinities
│       ├── IterativeRegionMerging.hpp  # Priority-queue agglomeration engine
│       ├── PriorityQueue.hpp   # Min-heap priority queue wrapper
│       ├── BinQueue.hpp        # Discretized bin queue (approximate priority queue)
│       ├── StatisticsProvider.hpp    # Base class with merge/edge callbacks
│       ├── MergeProviders.hpp  # Template meta-programming to combine providers
│       ├── CompoundProvider.hpp      # Multiple inheritance provider combiner
│       ├── MergeFunctions.hpp  # Scoring functions: MinSize, MaxSize, MinAffinity, MeanAffinity, etc.
│       ├── Operators.hpp       # Composable operators: OneMinus, One255Minus, Multiply, Add, etc.
│       ├── MeanAffinityProvider.hpp  # Running mean of edge affinities
│       ├── MinAffinityProvider.hpp   # Min affinity per edge
│       ├── MaxAffinityProvider.hpp   # Max affinity per edge
│       ├── HistogramQuantileProvider.hpp  # Histogram-based quantile (approximate, 256 bins)
│       ├── VectorQuantileProvider.hpp    # Exact quantile via nth_element
│       ├── MaxKAffinityProvider.hpp  # Top-K affinities per edge
│       ├── RegionSizeProvider.hpp    # Voxel count per region (node statistic)
│       ├── ContactAreaProvider.hpp   # Contact area per edge (edge statistic)
│       ├── RandomNumberProvider.hpp  # Random scoring (baseline)
│       ├── ConstantProvider.hpp      # Constant scoring
│       ├── Histogram.hpp       # Fixed-bin histogram data structure
│       ├── MaxKValues.hpp      # Sorted top-K value tracker
│       ├── discretize.hpp      # [0,1] <-> integer bin conversion
│       └── evaluate.hpp        # Rand index and VOI computation
└── tests/                      # Test suite
```

## Public Python API

### `waterz.agglomerate()` — Main entry point

```python
import waterz

for segmentation in waterz.agglomerate(
    affs,                        # [3,Z,Y,X] float32 affinities in [0,1]
    thresholds=[0.1, 0.3, 0.6], # agglomeration thresholds
    gt=None,                     # optional uint32 ground truth for metrics
    fragments=None,              # optional uint64 pre-computed fragments
    aff_threshold_low=0.0001,    # low threshold for initial watershed
    aff_threshold_high=0.9999,   # high threshold for initial watershed
    return_merge_history=False,  # include merge history in output
    return_region_graph=False,   # include region graph in output
    scoring_function='OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
    discretize_queue=0,          # 0=PriorityQueue, N>0=BinQueue with N bins
    force_rebuild=False,         # force recompile of Cython module
):
    # segmentation: uint64 [Z,Y,X] - modified in-place between yields, copy if needed!
    seg = segmentation.copy()
```

**Key points:**
- Takes **float32 affinities** in [0, 1] range (NOT uint8)
- Returns a **generator** — segmentation array is modified in-place between yields
- Uses **witty** for JIT Cython compilation (compiled modules cached automatically)
- `scoring_function` is a **C++ type string** (see below for shorthand conversion)

### `waterz.evaluate()` — Evaluation metrics

```python
metrics = waterz.evaluate(seg_uint64, gt_uint64)
# Returns dict with: V_Rand_split, V_Rand_merge, V_Info_split, V_Info_merge
```

## Key Data Types

| Type | NumPy | Notes |
|------|-------|-------|
| **Affinities** | `float32[3,Z,Y,X]` | 3-channel (z/y/x neighbor) affinity predictions, range [0, 1] |
| **Segmentation** | `uint64[Z,Y,X]` | Fragment/segment IDs, 0 = background |
| **Ground truth** | `uint32[Z,Y,X]` | For evaluation |
| **Fragments** | `uint64[Z,Y,X]` | Pre-computed over-segmentation |

## Scoring Functions

The `scoring_function` parameter takes C++ type strings. Common shorthand-to-C++ mappings:

| Shorthand | C++ Type | Description |
|-----------|----------|-------------|
| `aff50_his256` | `OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>` | Median affinity via 256-bin histogram |
| `aff75_his256` | `OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256>>` | 75th percentile via histogram |
| `aff85_his256` | `OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>>` | 85th percentile via histogram |
| `aff50_his0` | `OneMinus<QuantileAffinity<RegionGraphType, 50, ScoreValue>>` | Exact median affinity (vector-based) |
| `max10` | `OneMinus<MeanMaxKAffinity<RegionGraphType, 10, ScoreValue>>` | Mean of top-10 affinities |

**Note:** This version does NOT include `getScoreFunc()` — shorthand conversion is handled by the `decode_waterz` decoder in PyTorch Connectomics.

### Available C++ scoring primitives

**Edge statistics (from providers):**
- `MinAffinity` — minimum affinity across edge voxels
- `MaxAffinity` — maximum affinity
- `MeanAffinity` — running mean affinity
- `HistogramQuantileAffinity<RG, Q, Prec, Bins>` — Q-th percentile via histogram
- `QuantileAffinity<RG, Q, Prec>` — exact Q-th percentile
- `MeanMaxKAffinity<RG, K, Prec>` — mean of top-K affinities
- `ContactArea` — number of adjacent voxel pairs

**Node statistics:**
- `MinSize` / `MaxSize` — min/max region size of edge endpoints

**Operators (composable):**
- `OneMinus<F>` — `1 - f(e)` (converts affinity to distance)
- `One255Minus<F>` — `255 - f(e)` (for uint8 range)
- `Multiply<F1, F2>` — `f1(e) * f2(e)`
- `Add<F1, F2>` / `Subtract<F1, F2>`
- `Divide<F1, F2>` (safe division)
- `Invert<F>` — `1/f(e)`
- `Square<F>` — `f(e)^2`
- `Step<F1, F2>` — `f1(e) < f2(e) ? 0 : 1`

**Special:**
- `Random` — random score (baseline)
- `Constant<C>` — constant integer score

## Agglomeration Pipeline

1. **Watershed** (C++ `basic_watershed.hpp`):
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

## JIT Compilation

Uses **witty** library for JIT Cython compilation. The scoring function is specified as a C++ type string, written into a temporary `ScoringFunction.h` header. Compiled modules are cached by witty for reuse.

## Usage in PyTorch Connectomics

Waterz is used via the `decode_waterz` decoder in `connectomics/decoding/decoders/waterz.py`:

```python
from connectomics.decoding import decode_waterz

# affs: [C, Z, Y, X] float32 affinity predictions from model
seg = decode_waterz(
    affs,
    thresholds=0.3,
    merge_function='aff85_his256',  # shorthand, auto-converted to C++ type
    aff_threshold=(0.001, 0.999),
)
```

Or in YAML config:
```yaml
inference:
  decoding:
    - name: decode_waterz
      kwargs:
        thresholds: 0.3
        merge_function: aff85_his256
        aff_threshold: [0.001, 0.999]
```

## Differences from donglaiw/waterz (old fork)

| Feature | Old (donglaiw/waterz) | New (PytorchConnectomics/waterz) |
|---------|----------------------|----------------------------------|
| Location | `/projects/weilab/weidf/lib/waterz/` | `lib/waterz/` (in repo) |
| Affinities | uint8 [0, 255] | float32 [0, 1] |
| API | `waterz.waterz()` + `waterz.agglomerate()` | `waterz.agglomerate()` only |
| Shorthand | `getScoreFunc()` in `seg_util.py` | Not included (handled by decoder) |
| JIT compiler | Manual distutils + Cython | witty library |
| Extra utils | `watershed()`, `merge_id()`, `getRegionGraph()`, `somaBFS()` | None (agglomerate + evaluate only) |
| Packaging | `setup.py` only | `pyproject.toml` + `setup.py` |
