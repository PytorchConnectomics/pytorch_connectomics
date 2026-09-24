# Plan v0

## Summary

Cut the peak RSS of the single-invocation ABISS watershed (`lib/abiss/src/ws/atomic_chunk.cpp`,
targets `ws` and `ws64`) and of its Python driver `scripts/run_abiss_volume.py`, and add the
requested **opt-in uint32 segmentation output**. Everything except the uint32 option must be
**bit-exact**: every output file byte-identical to the stock binaries.

Where the memory goes today (bytes per padded voxel; `S = sizeof(internal_seg_t)` = 4 for `ws`,
8 for `ws64`; the affinity is a file-backed mmap, 12 B/voxel of page cache):

| Phase | Live anonymous memory | Source |
|---|---|---|
| watershed | `seg` S + **BFS queue 8** (`std::vector<std::ptrdiff_t> bfs(size+1)`) | `basic_watershed.hpp` |
| region graph | `seg` S + `edges`: a `MapContainer<ID, std::vector<F>>` per supervoxel storing **every** boundary affinity (4 B/face + vector/map overhead) | `region_graph.hpp` |
| write, 1 threshold | `seg` S + **`relabeled_seg` 8** (uint64 copy made only to add `offset`) | `atomic_chunk.cpp::relabel_segments` |
| write, K thresholds | `seg` S + **`seg_copy` S** + **`relabeled_seg` 8** + `rg_copy` per threshold | `atomic_chunk.cpp` multi-threshold loop |

S + 8 + 12 = 24 B/voxel for `ws` matches the measured 92 GB at 3.82 Gvox (Moritz L4), which
says the watershed/relabel phase is the peak there. The ws64 multi-threshold composite (~58 B/voxel)
is not fully explained by the table, so the first step is instrumentation, not guessing.

Expected savings (to be measured, not claimed): −4 B/voxel in the watershed phase (BFS index),
−8 B/voxel in the write phase (streamed relabel), −S B/voxel more in multi-threshold mode (no
`seg_copy`), a region-graph reduction proportional to boundary area for `max`/`mean`, and −4
B/voxel on disk and in the Python process with uint32 output.

## Scope

In scope:

* `lib/abiss/src/ws/{atomic_chunk.cpp, basic_watershed.hpp, region_graph.hpp, agglomeration.hpp, utils.hpp}` — the code compiled into `ws`/`ws64` only.
* `scripts/run_abiss_volume.py` (`_run_abiss_ws`, `_read_segmentation_xyz`, CLI) — additive, default behaviour unchanged.
* Tests: `tests/unit/` for the Python side; a bit-exact gate script for the binaries.

Out of scope (deliberately):

* The hierarchical pipeline binaries (`ws2`/`ws3`/`agg*`/`src/seg/*`). They consume the `seg_o_*/seg_i_*`, `counts_*`, `dend_*` files, so those files stay **uint64 and byte-identical in every mode**, including uint32 mode.
* Changing global `seg_t` to uint32 — chunked-pipeline offsets need the 64-bit space.
* Lower-precision affinity (float16/uint8) — changes results; not a memory-only change.
* Shrinking `CHUNK_SIZE` — adds stitching; see memory note on j0126-lineage configs.
* `connectomics/decoding/decoders/abiss.py::_load_output` (in-pipeline decoder casts to uint64) — separate consumer, leave.
* Rebuilding or overwriting `lib/abiss/build/ws` (md5 `bf4c7343…`) or `build64/ws64` — they are used by live runs. New binaries go to a scratch build tree; swapping them in is the user's call after the gate.

## Proposed Changes

**C0. Phase memory logging (ws/ws64).** After watershed, region graph, each merge, and each write,
print one line `[mem] <phase>: rss_gb=<VmRSS> hwm_gb=<VmHWM>` parsed from `/proc/self/status`.
Stdout only; no behaviour change. This attributes the ws64 58 B/voxel and is how every later
claim is checked.

**C1. BFS queue index width (bit-exact).** In `watershed()`, the queue stores linear voxel indices
as `ptrdiff_t` (8 B × (size+1)). Make the queue element type a template parameter and dispatch at
runtime: `uint32_t` when `size + 1 <= UINT32_MAX`, else `ptrdiff_t`. `ws` always qualifies (its
assert caps size at 2^31); `ws64` qualifies below 4.29 Gvox. Arithmetic on neighbours
(`y + dir[d]`) must stay in signed `index`; only storage narrows.

**C2. Streamed relabel + write (bit-exact).** Delete the `relabel_segments` full-volume copy.
Add a writer that walks the interior of `seg` (same `range(1, n-1)` crop and Fortran order as
`write_volume`) and emits `lut(seg[i])` → `out_t` through a fixed-size buffer (e.g. one z-plane)
into the output file. `lut` is `id==0 ? 0 : id + offset` in single-threshold mode (exactly
`relabel_segments`), composed with C3's remap in merge mode. `write_chunk_boundaries` gets the same
per-element transform for the `seg_o_*`/`seg_i_*` faces (always written as uint64 `seg_t`); the
`aff_i_*` faces are unchanged. `relabel_region_graph` stays (edge list, small).

**C3. Merge without mutating/copying `seg` (bit-exact).** Split `merge_segments` into:
(a) the union-find + `remaps` + `counts` + new region graph part, unchanged in logic, taking the
input `rg` by const reference and returning the new graph (so the multi-threshold loop no longer
copies `rg`), and producing a per-supervoxel LUT `lut[id] = remaps[find_set(id)]` (size = number of
supervoxels, not voxels); (b) application of that LUT, which now happens inside the C2 writer.
`seg` keeps the raw watershed ids for the whole run, so the K-threshold loop needs neither
`seg_copy` nor the in-place remap pass. The old in-place loop computed exactly
`remaps[find_set(seg[idx])]` per voxel, so precomputing it per id is identical.
`counts_copy` stays (per supervoxel).

**C4. Streaming edge accumulators in `get_region_graph` (bit-exact for max/mean).** For `MAX`
store a running max per edge; for `MEAN` store `(F sum, uint32 count)` updated as `sum = sum + a`
in first-seen order starting from `F(0)` — the same operation sequence as
`std::accumulate(begin, end, F(0))` over the push order, so the float result is identical. `pairs`
(first-seen order) is kept, so the pre-sort order and the `stable_sort` result are identical.
`PERCENTILE` keeps the per-edge vectors (needs all values). Implement as a small accumulator type
selected by mode; do not change `compute_edge_score` semantics.

**C5. Opt-in uint32 segmentation output (the user's ask).**
* `ws`/`ws64`: accept `--seg-dtype=uint32|uint64` as an optional token anywhere after argv[7];
  strip it before the existing merge-function/threshold positional parsing (which keys off
  `isalpha(argv[8][0])`, so a `-` token must be removed first, not misparsed). Default `uint64`.
* In uint32 mode only `seg_<tag>.data` / `seg_<tag>_<i>.data` are written as uint32. Before
  writing, check `offset + max_output_id <= UINT32_MAX`; on failure print a clear error and exit
  non-zero **without leaving a partial seg file**. Never wrap silently (cf. the uint32 index
  overflow that is silent under `-DNDEBUG`).
* `scripts/run_abiss_volume.py`: `--seg-dtype {uint64,uint32,auto}` (default `uint64`, so the
  `tutorials/neuron_liconn_*` scripts that memmap `np.uint64` keep working). `_run_abiss_ws(...,
  seg_dtype="uint64")` passes the token and `_read_segmentation_xyz(..., dtype=...)` reads with it
  (the file-size halo detection uses that itemsize). `auto` picks uint32 iff
  `offset + (number of interior voxels) <= 2**32 - 1` — a static upper bound on ids, so it can
  never trip the binary's check; otherwise uint64. The output array/h5 keeps the chosen dtype.

**Rejected:** read-only affinity mmap (hygiene, not memory; would need const-propagation through
templates — churn); float16 affinity; global uint32 `seg_t`.

## Files and Areas

| File | Change |
|---|---|
| `lib/abiss/src/ws/basic_watershed.hpp` | C1 queue element type |
| `lib/abiss/src/ws/region_graph.hpp` | C4 accumulators |
| `lib/abiss/src/ws/agglomeration.hpp` | C3 split `merge_segments` (keep a thin in-place wrapper only if another TU includes it — check `grep -r merge_segments src/`) |
| `lib/abiss/src/ws/utils.hpp` | C2 streamed volume writer; transform-aware `write_chunk_boundaries` |
| `lib/abiss/src/ws/atomic_chunk.cpp` | C0 logging, C5 flag, wire C2/C3, drop `relabel_segments`/`seg_copy`/`rg_copy` |
| `scripts/run_abiss_volume.py` | C5 Python side |
| `tests/unit/test_abiss_seg_dtype.py` (new) | `_read_segmentation_xyz` dtype + halo detection; `auto` resolution incl. boundary case |
| `tutorials/neuron_liconn_ist/slurm/ws_gate_memory.py` or `lib/abiss/tests/` (new) | bit-exact + peak-RSS gate script (coder picks location, states why) |

`lib/abiss` is a separate repo in the main checkout; work on a branch there (not `main`), do not
commit (CCC rule), and include `git -C <abiss> diff` in `code_vN.md`.

## Verification Plan

1. **Build** both `ws` and `ws64` from the modified source into a scratch tree
   (`lib/abiss/build_mem/`, Release, same CMake options as `build/`). Confirm `build/ws` md5 is
   still `bf4c7343dccf86e60d78f772e583492d` and `build64/ws64` is untouched.
2. **Bit-exact gate**, stock vs new, for both `ws` (vs `build/ws`) and `ws64` (vs `build64/ws64`):
   inputs = crops of the real 18 nm LICONN val affinity used by `tutorials/neuron_liconn_ist/slurm/ws_sizing.py`
   written through `run_abiss_volume` helpers, at 64×512×512 and 128×1024×1024;
   cases = single threshold `max`; 3 thresholds × {`max`, `mean`, `p75`}; boundary flags
   `1 1 1 1 1 1` **and** `1 1 1 0 0 0` (so face files are exercised); nonzero `offset`.
   Pass = every output file (`seg_*`, `counts_*`, `dend_*`, `meta_*`, `seg_o_*`, `seg_i_*`, `aff_i_*`)
   byte-identical (`cmp`). Report the file count compared per case.
3. **uint32 mode:** new `seg_*.data` == stock uint64 file `.astype(np.uint32)` exactly; all other
   files byte-identical to stock; overflow case (`offset = 2**32 - 10`, `--seg-dtype=uint32`) exits
   non-zero with the message and leaves no `seg_*` file.
4. **Peak RSS** via `/usr/bin/time -v` plus the C0 `[mem]` lines, stock vs new, on 128×1024×1024
   and one ≥1 Gvox crop, single and 5-threshold, for `ws` and `ws64`. Report a table in B/voxel
   (padded). Run on a compute node via SLURM, not the login node; scripts and logs on /projects.
5. **Python:** `pytest tests/unit/test_abiss_seg_dtype.py tests/unit/test_decode_abiss_wrapper.py
   tests/unit/test_abiss_edge_storage.py tests/unit/test_abiss_s3_chunked.py -q`; plus an
   end-to-end `run_abiss_volume.py` on a small crop with `--seg-dtype uint64` vs `uint32` vs `auto`
   giving equal label values.
6. `ctest` in the scratch build (existing nuc tests) still passes.

## Risks and Questions

* **C4 mean bit-exactness** depends on the accumulation order matching `std::accumulate`; the gate
  (step 2, `mean`) is the proof. If it does not match, drop C4-mean and keep C4-max only.
* **C1** must not narrow the neighbour arithmetic; a wrong cast silently corrupts. Gate step 2 plus
  a ws64 run with a chunk ≥ 2^31 would be ideal but is expensive — at minimum the dispatch branch
  for `ptrdiff_t` must be exercised once (e.g. temporarily force it on a small crop and compare).
* **Streamed writes** go through page cache; under a cgroup memory limit, dirty pages count until
  flushed. Use a bounded buffer and plain `write`, not a full-size `mapped_file_sink` (which is
  what `write_multi_array` does today and would reintroduce an 8 B/voxel mapping).
* **uint32 consumers:** anything that memmaps `seg_*.data` as uint64 (`mip0_wholeval_sweep.py`,
  `upload_seg_precomputed.py`) must not be pointed at a uint32 file. Default stays uint64; these
  scripts are not changed. Question for the user (non-blocking): should the LICONN sweep scripts
  adopt `auto` in a follow-up?
* The 58 B/voxel ws64 composite may be dominated by C4's region-graph term rather than the voxel
  arrays; C0 settles it. If the region graph is the peak for `PERCENTILE`, a further step (e.g.
  streaming quantile sketch) would not be bit-exact and is not proposed here.

## Changes Since Previous Plan Version

Initial plan.
