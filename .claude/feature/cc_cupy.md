# GPU connected components for affinity decode (`backend="cupy"`)

## Goal

`decode_affinity_cc` (the BANIS-semantics affinity → instance segmentation step
used by `tutorials/neuron_nisb/base_banis*.yaml`) was bottlenecked by the serial
numba DFS in `segmentation_kernels.connected_components_affinity_3d_numba`. For
a 3000³ NISB volume that's a single-threaded scan over ~2.7e10 voxels — minutes
to hours. The aim was a GPU-resident kernel preserving BANIS directed-affinity
semantics (so `cc3d`, which collapses to undirected foreground, was not
acceptable).

## What landed

### Backends offered by `decode_affinity_cc`

| backend | semantics | implementation | notes |
|---|---|---|---|
| `cc3d` | undirected foreground | `cc3d.connected_components(hard_aff.any(axis=0))` | fastest, but ignores directed edge gating — different result from BANIS |
| `numba_serial` | BANIS directed graph | original `@jit(nopython=True)` DFS flood-fill | reference implementation; bit-exact baseline |
| `numba` | BANIS directed graph | parallel min-label sweeps via `@jit(parallel=True) + prange` | bit-exact with `numba_serial` for `edge_offset=0`; same partition for `edge_offset=1`; ~1× CPU due to high iteration count |
| **`cupy`** | BANIS directed graph | per-axis CUDA `RawKernel` doing forward+backward column sweeps | new; same partition as `numba_serial`, bit-exact for `edge_offset=0` |

### Files touched

- `connectomics/decoding/decoders/segmentation_kernels.py`
  - `CUPY_AVAILABLE` flag (optional dep, graceful fallback).
  - `_build_affinity_cc_sweep_kernels(label_key)` — caches three CUDA RawKernels
    (`sweep_axis0/1/2`) per label dtype. Source string templates `LABEL_T` →
    `unsigned int` (uint32) or `unsigned long long` (uint64).
  - `connected_components_affinity_3d_cupy(hard_aff, edge_offset, max_iters,
    use_uint32)` — driver.
- `connectomics/decoding/decoders/segmentation.py`
  - Wires `backend="cupy"` and updates the docstring + error messages.
- `tests/integration/test_affinity_cc3d.py`
  - `test_cupy_matches_serial` (5 seeds × 2 edge_offsets × 3 densities = 30
    cases) and `test_cupy_empty_and_full`. Skipped automatically when
    cupy isn't installed.

### Algorithm

Same propagation strategy as `..._numba_parallel`:

1. **Foreground mask** — voxel is "foreground" iff it is a serial-DFS seed
   (`hard_aff[:3].any(axis=0)`, regardless of in-bounds neighbors so phantom-edge
   singletons are preserved) ∪ endpoint of any in-bounds edge.
2. **Initial labels** — `cp.cumsum(fg_flat, dtype=label_dtype)` then in-place
   multiply by `fg_flat` to zero background. Foreground voxels get rank
   1..n_fg in scan order; lex-smallest voxel of each component gets the
   smallest label → matches serial DFS for `edge_offset=0`. No
   boolean-scatter scratch.
3. **Iterative sweeps** — each outer iteration launches three RawKernels
   (one per axis). Each thread owns one axis-orthogonal column and does
   forward then backward serial scan along the axis, race-free because
   different threads write disjoint columns. One outer iter fully
   propagates labels along each axis line.
4. **Convergence** — each kernel sets a 1-element device-side `changed`
   flag if any modification happened (racy write of constant 1, no atomic
   needed). After the three axis kernels, host reads the flag (1-byte
   sync). No full-volume snapshot or `array_equal` pass.
5. **Compaction** — `cp.asnumpy(seg)` then `fastremap.renumber(in_place=True,
   preserve_zero=True)` (C++ hash, O(V), low memory). Skips `cp.unique`'s
   O(V log V) sort and V-sized int64 inverse buffer.

### `use_uint32` flag

`connected_components_affinity_3d_cupy(..., use_uint32=True)` (default) picks
`uint32` for the seg buffer when the foreground voxel count fits in 32 bits
(always true below ~1600³). Falls back to `uint64` automatically if not. The
RawKernels are templated on the label C-type and cached per dtype, so the same
sweep code serves both.

## Correctness

`tests/integration/test_affinity_cc3d.py::test_cupy_matches_serial` —
30 randomized parity cases:

- All match `numba_serial` as a partition (`_assert_same_partition`, which
  asserts a bijective label remap exists).
- For `edge_offset=0` we additionally assert bit-exact `np.testing.assert_array_equal`.

A direct `uint32 == uint64` parity check on a 128³ smooth-affinity volume also
passes. Total: 31 cupy parity tests green.

## Benchmarks (NVIDIA A10, 23 GB) — after P2-1 + P2-2 + P3 optimizations

Smooth synthetic affinities (`gaussian_filter(rand, sigma)` then thresholded):

### Fragmented case — many small components (BANIS at high threshold)

| volume | numba_serial | cupy | cupy peak | cc3d |
|---|---|---|---|---|
| 128³ (~3K comps) | 99 ms | **54 ms (1.84×)** | 19 MB | 46 ms |
| 256³ (~22K comps) | 1052 ms | **706 ms (1.49×)** | 151 MB | 582 ms |
| 384³ (~72K comps) | 3865 ms | **2748 ms (1.41×)** | 510 MB | 1912 ms |
| 512³ (~168K comps) | 10560 ms | **6946 ms (1.52×)** | 1208 MB | 4643 ms |

### Large-component case — closer to BANIS smooth-volume neuron path

| volume | numba_serial | cupy | cupy peak | cc3d |
|---|---|---|---|---|
| 256³ (1 comp) | 651 ms | **214 ms (3.05×)** | 151 MB | 171 ms |
| 512³ (1 comp) | 5885 ms | **1976 ms (2.98×)** | 1208 MB | 1783 ms |
| 768³ (1 comp) | 20478 ms | **7205 ms (2.84×)** | 4077 MB | 5862 ms |
| 1024³ (1 comp) | 51000 ms | **17303 ms (2.95×)** | 9664 MB | 14199 ms |

### Speed/memory delta vs the pre-optimization implementation

| volume / scenario | time before → after | peak before → after |
|---|---|---|
| 256³ fragmented | 930 → 706 ms (1.32×) | 871 → **151 MB (5.8×)** |
| 512³ fragmented | 9676 → 6946 ms (1.39×) | 6964 → **1208 MB (5.8×)** |
| 256³ large-comp | 256 → 214 ms (1.20×) | 828 → **151 MB (5.5×)** |
| 512³ large-comp | 2269 → 1976 ms (1.15×) | 6621 → **1208 MB (5.5×)** |

Headroom: 1024³ now fits comfortably (9.7 GB peak) on the 23 GB A10.

### Why not bigger?

- Algorithm is `O(turns × V)` (where `turns` is the number of axis-direction
  changes a component takes); shrinking that needs hierarchical/pointer-jumping
  union-find — separate piece of work.
- Per-thread stride access along the propagation axis is cache-unfriendly on
  A10 for the fragmented case.
- `cc3d` is still ~1.2–1.5× faster but uses different (undirected) semantics.

## Memory footprint

Threshold happens on CPU first (`segmentation.py:534`), so the GPU never sees
floats — only the post-threshold bool, ~3 B/voxel transferred. Per-voxel GPU
working set during sweeps (post-optimization):

| buffer | uint32 | uint64 |
|---|---|---|
| hard_aff_gpu (3 ch bool) | 3 | 3 |
| fg (bool) | 1 | 1 |
| seg | 4 | 8 |
| device-side `changed` flag | trivial | trivial |
| compaction (`cp.asnumpy` + fastremap.renumber) | host-side | host-side |

Empirical pool peak (cupy `MemoryPool.total_bytes()`):

| volume | use_uint32=True | use_uint32=False | savings |
|---|---|---|---|
| 256³ fragmented | 151 MB | 218 MB | −31% |
| 512³ fragmented | 1208 MB | 1745 MB | −31% |
| 256³ large-comp | 151 MB | 218 MB | −31% |
| 512³ large-comp | 1208 MB | 1745 MB | −31% |
| 768³ large-comp | 4077 MB | 5889 MB | −31% |

A10 (23 GB) headroom with `use_uint32=True`:

- 1024³ fits comfortably (~9.7 GB) — was infeasible before the optimizations.
- ~1500³ should fit (~30 GB extrapolated) — close to A10's limit; safer with
  a beefier card.
- 3000³ NISB still needs the chunked decode pipeline (`tutorials/neuron_nisb/base_banis_chunk.yaml`)
  which decodes per ~1008³ chunk and stitches CC across chunk boundaries via
  `connectomics.decoding.streamed_chunked.run_chunked_affinity_cc_inference`.

## When to use which backend

- **BANIS production at scale (3000³ neurons)**: don't pick a backend in
  isolation — switch to `tutorials/neuron_nisb/base_banis_chunk.yaml` which
  runs `streamed_chunked` decode. Within each chunk, `backend="cupy"` is the
  fastest BANIS-correct option.
- **Single-volume test on a workstation, ≤700³, BANIS semantics**:
  `backend="cupy"` if a GPU is available; otherwise `backend="numba_serial"`.
- **`backend="numba"` (parallel min-prop on CPU)**: kept as a CPU
  reference — it's correct but rarely faster than `numba_serial` because the
  algorithm is `O(turns × V)` and 7 CPU threads can't overcome that. Use
  `numba_serial` for everyday CPU runs.
- **Speed > exact BANIS semantics**: `backend="cc3d"` (collapses to
  undirected foreground; check the `adapted_rand` delta before committing).

## Config example

```yaml
decoding:
  steps:
    - name: decode_affinity_cc
      kwargs:
        threshold: 0.65
        backend: cupy           # uses A10 / single GPU
        edge_offset: 0          # BANIS stores edges at lex-lower voxel
```

`backend="cupy"` raises `ImportError` with an install hint if `cupy` isn't
present (e.g., `pip install cupy-cuda12x`).

## Caveats and follow-ups

- Convergence loop iterates until the device-side `changed` flag stays 0 over
  one outer iter. `max_iters=4096` cap with a `RuntimeWarning` on overflow;
  haven't seen it trip on real data.
- A faster algorithm on GPU would be hierarchical union-find with pointer
  jumping (`O(V log diameter)` instead of `O(V × turns)`). That would close the
  remaining gap to `cc3d` on the BANIS-correct path. Significant new code; not
  warranted unless someone has a workload where the current `cupy` path is the
  bottleneck.
- `cupy` install bumps `numpy` to 2.x, which surfaces a separate import error
  in unrelated parts of `connectomics.config.schema` (`OutputArrayConfig`)
  that exists in the current dirty tree. Doesn't affect the parity tests
  (which pass under pytest's path setup) but watch for it in inline `python
  -c` invocations.

## Optimization history

- **v1 (initial):** `cp.minimum`/`cp.where` slice-shift sweeps + `cp.unique`
  compaction. Correct but slow because each step propagated 1 voxel and
  `cp.unique` allocated multi-V-sized buffers.
- **v2 (RawKernel sweep):** custom CUDA kernels per axis doing
  forward+backward column scans → 1 outer iter propagates fully along each
  axis line.
- **v3 (current — P2-1 + P2-2 + P3 from review):**
  - Device-side `changed` flag replaces `seg_prev` snapshot + `cp.array_equal`
    (drops a label-sized buffer and a per-iter full-volume read).
  - `cp.cumsum` + in-place `*= fg_flat` replaces boolean-scatter init.
  - `fastremap.renumber` (CPU, O(V) hash) replaces `cp.unique` (O(V log V)
    sort + V-sized inverse).
  - Net: ~30% peak memory drop, ~40% time drop on fragmented case. 1024³
    now fits on a 23 GB A10.
