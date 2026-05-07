# Zwatershed Reference

Repo: https://github.com/PytorchConnectomics/zwatershed

Cython-compiled Python package implementing steepest-ascent watershed + region-graph agglomeration for connectomics EM segmentation. ABISS was forked/derived from this codebase.

## Repository Layout

```
zwatershed/
├── zwatershed/
│   ├── __init__.py                  # Public API
│   ├── zwatershed.pyx               # Cython bindings
│   ├── zwatershed.h                 # C++ declarations
│   ├── zwatershed_main.cpp          # C++ implementation
│   ├── mean_aff.py                  # Mean-affinity agglomeration (pure Python)
│   └── zwatershed_util/             # C++ template headers
│       ├── types.hpp                # Type defs, traits, bit masks
│       ├── basic_watershed.hpp      # Core watershed (steepest ascent)
│       ├── region_graph.hpp         # Region graph (max & average variants)
│       ├── agglomeration.hpp        # Size+affinity merge logic
│       ├── mst.hpp                  # Maximum spanning tree
│       ├── mergerg.hpp              # Second-pass MST-based merging
│       ├── limit_functions.hpp      # Utility functions
│       └── main_helper.hpp          # Helper functions
│   └── zi/disjoint_sets/            # Union-find (path compression + union by rank)
├── test/
│   ├── do_zwatershed.py
│   └── do_mean_agglomeration.py
└── setup.py                         # Cython + C++11 build
```

## Algorithm Pipeline

### 1. Watershed (`basic_watershed.hpp`)

Steepest-ascent watershed on 3-channel affinity input (x, y, z directions):

1. **Steepest ascent**: For each voxel, find max-affinity neighbor among 6-connected. Set direction bits if affinity exceeds thresholds.
2. **Plateau division**: BFS from plateau corners inward to break ties in flat regions.
3. **Basin finding**: BFS to assign connected components unique IDs. Tracks voxel counts per region.

Bit-encoded seg volume: `high_bit` = processed, `dir_mask` = direction, `visited` = temp BFS mark.

### 2. Region Graph (`region_graph.hpp`)

Two variants:
- **`get_region_graph`**: Stores **max** affinity per edge pair (same as original ABISS).
- **`get_region_graph_average`**: Stores **mean** affinity per edge pair (more accurate for multi-contact boundaries).

Both iterate all voxels, check 3 neighbors (x-1, y-1, z-1), collect edges as `(affinity, id1, id2)`, sorted descending.

### 3. Agglomeration (`agglomeration.hpp`)

Two-pass merge:

**Two merge thresholds work together** — size and affinity:
```
merge_segments_with_function(seg, rg, counts, size_th, weight_th, dust_th, merge_th)
```

**Pass 1 — Size+affinity merge:**
```
for each edge (affinity, s1, s2) in descending affinity order:
    if affinity < weight_th: stop
    if counts[s1] < size_th OR counts[s2] < size_th:
        merge(s1, s2)   # union-find + combine counts
```
An edge must pass **both** checks: affinity ≥ `weight_th` AND at least one region < `size_th`.
Then dust removal: discard regions with count < `dust_th`.

**Pass 2 — MST merge (optional, when merge_th > 0):**
- Build MST from remaining region graph (`mst.hpp`)
- Merge edges in MST with affinity > `merge_th` (`mergerg.hpp`)
- Simple parent→child relabeling without size constraints

In the top-level `zwatershed()` API, `T_threshes` sweeps the **size** threshold while `T_aff[2]` provides the fixed **affinity** threshold — so multiple size-based agglomeration levels are evaluated in one run.

### 4. MST (`mst.hpp`)

Kruskal-like: process edges descending by affinity, add edge only if it connects two components. Output: tree with |V|-1 edges.

### 5. MST Merge (`mergerg.hpp`)

For MST edges with affinity > T_merge: build child→parent mapping, chase chains to root, relabel volume.

## Threshold Parameters

| Parameter | Description |
|-----------|-------------|
| `T_aff_low` / `affs_low` | Min affinity for watershed direction bits |
| `T_aff_high` / `affs_high` | Upper affinity for watershed (always set direction if ≥) |
| `T_size` / `size_th` | Max voxel count for merge eligibility — merge if **either** region < this |
| `T_weight` / `weight_th` | Min affinity for agglomeration edges |
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
    affs,                          # (batch, H, W, D, 3) float32
    T_threshes=[400, 600, 800],    # size thresholds to sweep
    T_aff=[1, 50, 30],             # percentiles: low, high, merge
    T_aff_relative=True,
    T_dust=600,                    # dust removal
    T_merge=0.5,                   # MST merge threshold
)
# Returns lists indexed by T_threshes
```

### Step-by-step (for parameter search / debugging)

```python
from zwatershed import (
    zw_initial,
    zw_get_region_graph,
    zw_get_region_graph_average,
    zw_merge_segments_with_function,
    zw_mst,
)

# 1. Initial watershed
init = zw_initial(affs, affs_low=0.1, affs_high=0.9)
seg, rg, counts = init['seg'], init['rg'], init['counts']

# 2. Region graph (max or average)
rg_affs, id1, id2 = zw_get_region_graph(affs, seg)
# or: rg_affs, id1, id2 = zw_get_region_graph_average(affs, seg)

# 3. Merge
counts_out, (rg_affs_out, id1_out, id2_out) = \
    zw_merge_segments_with_function(
        seg, rg_affs, id1, id2, counts,
        T_size=600, T_weight=0.3, T_dust=600, T_merge=0.5
    )
```

### Mean-affinity agglomeration (pure Python alternative)

```python
from zwatershed import mean_agglomeration

seg_merged = mean_agglomeration(seg, aff, threshold=0.5)
```

Uses scipy sparse matrices. Simpler: merge all edges with mean boundary affinity ≥ threshold, no size constraints.

## Key Differences: Zwatershed vs ABISS

| Aspect | Zwatershed | ABISS |
|--------|-----------|-------|
| Language | C++ with Cython/Python bindings | Pure C++ CLI |
| Merge eligibility | **Either** region < size_threshold | **Both** regions < size_threshold → reject |
| Second pass | MST-based merge (T_merge) | Not present in atomic_chunk |
| Region graph scoring | Max or Average (two functions) | Configurable: max, mean, percentile (new) |
| Edge score storage | Scalar per edge (max or avg computed inline) | Vector of all affinities per edge, reduced at end |
| Multi-threshold | Sweeps `T_threshes` (size thresholds) | Sweeps merge thresholds (affinity thresholds) |
| Boundary handling | No chunk boundary logic | Border-aware merging (`on_border` bit) |
| Plateau handling | Full plateau division algorithm | Same (inherited) |
| Python API | Rich step-by-step functions | None (CLI only) |
| Array library | Boost multi_array | Boost multi_array |

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

These are logically equivalent for the basic case, but ABISS adds border-aware logic that can reject merges even when one segment is small (if the small one touches a chunk boundary and the large one doesn't meet size threshold).

## Data Types

```cpp
volume<T>           = boost::multi_array<T, 3>          // Fortran order
affinity_graph<T>   = boost::multi_array<T, 4>          // (x, y, z, 3)
region_graph<ID, F> = vector<tuple<F, ID, ID>>          // sorted descending
```

Segment IDs: `uint64_t`. Affinities: `float32`.

## Build

```bash
cd /projects/weilab/weidf/lib/zwatershed
pip install -e .
```

Requires: Cython, NumPy, Boost headers. Compiles with `-std=c++11`.
