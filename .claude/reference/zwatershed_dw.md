# Zwatershed Reference (donglaiw fork)

**Repository:** /projects/weilab/weidf/lib/zwatershed
**GitHub:** https://github.com/PytorchConnectomics/zwatershed

Cython-compiled Python package implementing steepest-ascent watershed + region-graph agglomeration for connectomics EM segmentation. ABISS was forked/derived from this codebase.

## Repository Layout

```
zwatershed/
├── zwatershed/
│   ├── __init__.py                     # Public Python API (8 exports)
│   ├── zwatershed.pyx                  # Cython bindings to C++ (~410 lines)
│   ├── zwatershed.h                    # C++ function declarations
│   ├── zwatershed_main.cpp             # C++ implementations (~305 lines)
│   ├── mean_aff.py                     # Mean-affinity agglomeration (pure Python)
│   └── zwatershed_util/                # C++ template headers
│       ├── types.hpp                   # Type aliases, watershed_traits, bit masks
│       ├── basic_watershed.hpp         # Steepest ascent, plateau division, basin finding
│       ├── region_graph.hpp            # get_region_graph (max) & get_region_graph_average
│       ├── agglomeration.hpp           # merge_segments_with_function (two-pass merge)
│       ├── mst.hpp                     # Maximum spanning tree (Kruskal-like)
│       ├── mergerg.hpp                 # Second-pass MST-based merging
│       ├── limit_functions.hpp         # Utility functors
│       └── main_helper.hpp             # Metrics and helpers
│   └── zi/disjoint_sets/              # Union-find (path compression + union by rank)
├── test/
│   ├── do_zwatershed.py                # End-to-end example with HDF5 I/O
│   └── do_mean_agglomeration.py        # Mean-affinity agglomeration example
└── setup.py                            # Cython/C++11 build configuration
```

## Algorithm Pipeline

### 1. Watershed (`basic_watershed.hpp`)

Steepest-ascent watershed on 3-channel affinity input (x, y, z directions):

1. **Steepest ascent**: For each voxel, find max-affinity neighbor among 6-connected. Set direction bits if affinity exceeds thresholds:
   ```cpp
   F negx = (x>0) ? aff[x][y][z][0] : low;  // x-direction (toward x-1)
   F negy = (y>0) ? aff[x][y][z][1] : low;  // y-direction (toward y-1)
   F negz = (z>0) ? aff[x][y][z][2] : low;  // z-direction (toward z-1)
   F posx = (x<(xdim-1)) ? aff[x+1][y][z][0] : low;
   F posy = (y<(ydim-1)) ? aff[x][y+1][z][1] : low;
   F posz = (z<(zdim-1)) ? aff[x][y][z+1][2] : low;
   F m = std::max({negx, negy, negz, posx, posy, posz});
   // Set direction bits (0x01–0x20) if affinity is max OR exceeds high threshold
   ```
2. **Plateau division**: BFS from plateau corners inward to break ties in flat regions.
3. **Basin finding**: BFS to assign connected components unique IDs. Tracks voxel counts per region.

Bit-encoded seg volume: `high_bit` = processed, `dir_mask` = direction, `visited` = temp BFS mark.

### 2. Region Graph (`region_graph.hpp`)

Two variants:

#### `get_region_graph()` — Max affinity per edge
```cpp
// Scan all voxels, check 3 neighbors (x-1, y-1, z-1)
if ((x > 0) && seg[x-1][y][z] && seg[x][y][z] != seg[x-1][y][z]) {
    auto mm = std::minmax(seg[x][y][z], seg[x-1][y][z]);
    F& curr = edges[mm.first][mm.second];
    curr = std::max(curr, aff[x][y][z][0]);  // store MAX affinity
}
// Similar for y and z neighbors
// Flatten to vector<tuple<F, ID, ID>>, sort descending by affinity
```

#### `get_region_graph_average()` — Mean affinity per edge
Same scan, but accumulates `(sum, count)` per edge, computes average at flatten:
```cpp
F avg = p.second.first / ((float) p.second.second);
```

### 3. Agglomeration (`agglomeration.hpp`)

Two-pass merge:

**Pass 1 — Size+affinity merge:**
```cpp
for each edge (affinity, s1, s2) in descending affinity order:
    if (s1 != s2 && s1 && s2 && affinity > weight_th):
        if (counts[s1] < size_th || counts[s2] < size_th):
            merge(s1, s2)   // union-find + combine counts
```
An edge must pass **both** checks: affinity > `weight_th` AND at least one region < `size_th`.
Then dust removal: discard regions with count < `dust_th`.

**Pass 2 — MST merge (optional, when merge_th > 0):**
- Build MST from remaining region graph (`mst.hpp`)
- Merge edges in MST with affinity > `merge_th` (`mergerg.hpp`)
- Simple parent→child relabeling without size constraints

### 4. MST (`mst.hpp`)

Kruskal-like: process edges descending by affinity, add edge only if it connects two components. Output: tree with |V|-1 edges.

### 5. MST Merge (`mergerg.hpp`)

For MST edges with affinity > T_merge: build child→parent mapping, chase chains to root, relabel volume.

## Data Types (`types.hpp`)

```cpp
template<typename T> using volume = boost::multi_array<T, 3>;           // Fortran order
template<typename T> using affinity_graph = boost::multi_array<T, 4>;   // (X, Y, Z, 3)
template<typename ID, typename F> using region_graph = vector<tuple<F, ID, ID>>;  // sorted descending
```

Segment IDs: `uint64_t`. Affinities: `float32`.

**Array ordering:**
- Python API: `(Z, Y, X, 3)` affinities
- C++ internal: `(X, Y, Z, 3)` Fortran order (Cython auto-transposes via stride wrapping)

## Threshold Parameters

| Parameter | Description |
|-----------|-------------|
| `T_aff[0]` / `affs_low` | Min affinity for watershed direction bits |
| `T_aff[1]` / `affs_high` | Upper affinity for watershed (always set direction if ≥) |
| `T_aff[2]` / `weight_th` | Min affinity for agglomeration edges (pass 1) |
| `T_size` / `size_th` | Max voxel count for merge eligibility — merge if **either** region < this |
| `T_dust` / `lowt` | Min voxel count to keep a region (dust removal) |
| `T_merge` | Second-pass MST merge threshold (0 = skip) |
| `T_threshes` | List of size thresholds for batch evaluation |

### T_aff relative mode

When `T_aff_relative=True`, the 3 values in `T_aff` are **percentiles** of the non-zero affinity distribution:
- `T_aff[0]` → percentile for `affs_low`
- `T_aff[1]` → percentile for `affs_high`
- `T_aff[2]` → percentile for `affs_merge`

## Python API

### End-to-end

```python
from zwatershed import zwatershed

segs, rgs, counts = zwatershed(
    affs,                          # (Z, Y, X, 3) float32
    T_threshes=[400, 600, 800],    # size thresholds to sweep
    T_aff=[1, 50, 30],             # percentiles: low, high, merge
    T_aff_relative=True,
    T_dust=600,                    # dust removal
    T_merge=0.5,                   # MST merge threshold
)
# segs: list of uint64 segmentations (one per T_threshes)
# rgs: list of region graphs, shape (num_edges, 3) each
# counts: list of region counts
```

### Step-by-step

```python
from zwatershed import (
    zw_initial,
    zw_get_region_graph,
    zw_get_region_graph_average,
    zw_merge_segments_with_function,
    zw_mst,
    zw_do_mapping,
    zw_do_mapping_id,
)

# 1. Initial watershed
init = zw_initial(affs, affs_low=0.1, affs_high=0.9)
seg, rg, counts = init['seg'], init['rg'], init['counts']

# 2. Region graph (max or average)
rg_affs, id1, id2 = zw_get_region_graph(affs, seg)
# or: rg_affs, id1, id2 = zw_get_region_graph_average(affs, seg)
# rg_affs: (num_edges,) float32 — affinity per edge
# id1, id2: (num_edges,) uint64 — region IDs

# 3. Merge
counts_out, (rg_affs_out, id1_out, id2_out) = \
    zw_merge_segments_with_function(
        seg, rg_affs, id1, id2, counts,
        T_size=600, T_weight=0.3, T_dust=600, T_merge=0.5
    )

# 4. Mapping utilities
mapping = zw_do_mapping_id(id1, id2)        # no size constraints
mapping = zw_do_mapping(id1, id2, counts, max_count)  # with size cap
```

### Mean-affinity agglomeration (pure Python alternative)

```python
from zwatershed import mean_agglomeration, ma_get_region_graph, ma_merge_region

# Simple: merge all edges with mean boundary affinity ≥ threshold
seg_merged = mean_agglomeration(seg, aff, threshold=0.5)

# Step-by-step:
rg = ma_get_region_graph(seg, aff)  # shape (num_edges, 3): [id1, id2, mean_aff]
seg_merged = ma_merge_region(seg, rg, threshold=0.5)
```

Uses scipy sparse matrices. No size constraints — purely boundary affinity based.

## Key Differences: Zwatershed vs ABISS

| Aspect | Zwatershed | ABISS |
|--------|-----------|-------|
| Language | C++ with Cython/Python bindings | Pure C++ CLI |
| Merge eligibility | **Either** region < size_threshold | **Both** regions < size_threshold → reject |
| Second pass | MST-based merge (T_merge) | Not present in atomic_chunk |
| Region graph scoring | Max or Average (two functions) | Configurable: max, mean, percentile |
| Multi-threshold | Sweeps `T_threshes` (size thresholds) | Sweeps merge thresholds (affinity thresholds) |
| Python API | Rich step-by-step functions | None (CLI only) |

### Critical merge logic difference

**Zwatershed** (`agglomeration.hpp`):
```cpp
if ((counts[s1] < size_th) || (counts[s2] < size_th))  // merge if EITHER is small
```

**ABISS** (`agglomeration.hpp`):
```cpp
if ((real_size_s1 >= size_threshold) && (real_size_s2 >= size_threshold))
    return false;  // reject if BOTH are large
```

Logically equivalent for basic case, but ABISS adds border-aware logic.

## Key Differences: Zwatershed vs waterz (funkey)

| Aspect | Zwatershed | waterz |
|--------|-----------|--------|
| Affinities | float32, (Z,Y,X,3) | float32, (3,Z,Y,X) |
| Channel order | x=ch0, y=ch1, z=ch2 | z=ch0, y=ch1, x=ch2 |
| Watershed | Steepest-ascent (direction bits) | BFS plateau division |
| Agglomeration | Size+affinity merge + MST | Configurable scoring functions (C++ templates) |
| Region graph | Max or average per edge | Full affinity distribution per edge (histogram, quantile, etc.) |
| Scoring | Fixed: max or mean | Composable: `OneMinus<HistogramQuantileAffinity<...>>` |
| Threshold sweep | Size thresholds (`T_threshes`) | Agglomeration thresholds |

## Build

```bash
cd /projects/weilab/weidf/lib/zwatershed
pip install -e .
```

Requires: Cython, NumPy, Boost headers. Compiles with `-std=c++11`.
