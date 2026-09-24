# Plan v1

## Summary

Cut the peak memory of the single-invocation ABISS watershed (`lib/abiss/src/ws/atomic_chunk.cpp`,
targets `ws` and `ws64`) and of its Python driver `scripts/run_abiss_volume.py`, and add the
requested **opt-in uint32 segmentation output**. Every change except the uint32 option must
preserve outputs under the explicit contract in Verification step 2.

Two facts that reshape v0:

* Both stock builds are `Release` with `-O3 -DNDEBUG` (conda `x86_64-conda-linux-gnu-c++`, conda
  CXXFLAGS incl. `-march=nocona -mtune=haswell`, system allocator, no absl). The 2^31 chunk
  `assert` is therefore **compiled out**, so the "92 GB at 3.82 Gvox with `ws`" number was almost
  certainly a silently overflowed run. It is **not** used as a baseline; every baseline below is
  re-measured.
* The write path is heavier than v0 said: `write_volume` builds an **owning cropped copy** of the
  relabeled volume and `write_multi_array` copies that into a **full-size `mapped_file_sink`**.

Memory model, stock (P = padded voxels, I = interior voxels, S = `sizeof(internal_seg_t)`: 4 for
`ws`, 8 for `ws64`, E = region-graph edges, N = supervoxels). "anon" = anonymous heap; "file" =
resident file-backed pages (charged to the cgroup; clean pages reclaimable, dirty pages not until
written back):

| Phase | anon | file |
|---|---|---|
| watershed | `seg` S·P + BFS queue 8·(P+1) + `counts` 8·N | affinity up to 12·P (clean) |
| region graph | `seg` S·P + `edges` (per-supervoxel hash map of `vector<float>` holding **every** boundary face) + `pairs` + `rg` + stable_sort buffer | affinity |
| merge (per threshold) | `seg` S·P (+ `seg_copy` S·P in multi mode) + `rg_copy` + rank/parent/remaps S·N each + `in_rg` N `std::set`s + `new_rg` | affinity |
| write (per threshold) | `relabeled_seg` 8·P + **cropped copy 8·I** (+ `seg` S·P in multi mode) | **output mapping 8·I (dirty)** + affinity |

Target after this plan: anon ≈ S·P + O(N + E) throughout, with no per-voxel output staging.

## Scope

In scope:

* `lib/abiss/src/ws/{atomic_chunk.cpp, basic_watershed.hpp, region_graph.hpp, agglomeration.hpp, utils.hpp}` — compiled only into `ws`/`ws64` (verified: `merge_segments`, `relabel_segments`, `write_chunk_boundaries`, `basic_watershed.hpp`, `region_graph.hpp` have no other users under `lib/abiss/src`).
* `lib/abiss/CMakeLists.txt` — one new test executable under `BUILD_TESTING`.
* `scripts/run_abiss_volume.py` — additive; default behaviour and default `ws` argv unchanged.
* Tests: `tests/unit/test_abiss_seg_dtype.py` (new); `lib/abiss/tests/test_ws_bfs.cpp` (new);
  `lib/abiss/tests/ws_gate.py` (new, bit-exact + memory gate driver).

Out of scope (unchanged from v0, reasons kept): hierarchical binaries (`ws2`/`ws3`/`agg*`/`src/seg/*`)
— so `seg_o_*`/`seg_i_*`/`counts_*`/`dend_*`/`meta_*` stay uint64 and value-identical in **every**
mode; global `seg_t` → uint32; lower-precision affinity; shrinking `CHUNK_SIZE`;
`connectomics/decoding/decoders/abiss.py`; the tutorial scripts that memmap `seg_*.data` as uint64.
**Not overwriting** `lib/abiss/build/ws` (md5 `bf4c7343dccf86e60d78f772e583492d`) or
`build64/ws64`; new binaries go to scratch build trees and swapping them in is the user's decision.
`lib/abiss` is a separate repo in the main checkout: work on a new branch there
(`feature/ws-memory`), no commits, report `git -C <abiss> diff` in `code_vN.md`.

## Proposed Changes

**C0. In-phase memory instrumentation.** A helper that prints
`[mem] <label>: anon_gb=<RssAnon> file_gb=<RssFile> hwm_gb=<VmHWM>` from `/proc/self/status`.
Call points are chosen to catch in-phase peaks, not phase ends: BFS loop end *before* the queue
leaves scope; region graph after the edge fill and before scoring; after scoring before freeing
`edges`; after `stable_sort`; inside merge after the union-find pass and after `new_rg`; around each
write. `RssAnon` vs `RssFile` separates heap from mapped pages. Stdout only.

**C1. Runtime size check (replaces the compiled-out assert).** Compute `P = xdim*ydim*zdim` with
`__builtin_mul_overflow`; if it overflows `size_t`, or `P >= watershed_traits<internal_seg_t>::high_bit`,
or `P > PTRDIFF_MAX`, print a clear error naming the binary width and exit 2 before touching the
affinity. This is a deliberate behaviour change: over-cap chunks now fail loudly instead of
producing a silently wrong segmentation.

**C2. Bounded BFS queue (exact).** In `watershed()`, `bfs_start`, `bfs_index`, `bfs_end` are all equal
after every seed iteration (a search either copies the hit label to `[bfs_start, bfs_end)` or
assigns a new id to it; the zero-seed branch sets `bfs_start = bfs_index = bfs_end`). So resetting
all three to 0 at each seed and indexing relatively performs the identical sequence of reads and
writes on `seg_raw`. Store the queue in a reused `std::vector` grown by `push_back` (never shrunk),
so its size is ~2× the **largest single search**, not P. Element type is a template parameter
`Q`: `uint32_t` when `bfs_fits_u32(P)` (all indices `< P` fit, i.e. `P - 1 <= UINT32_MAX`), else
`std::ptrdiff_t`; neighbour arithmetic `y + dir[d]` stays in signed `index`, only storage narrows.
`bfs_fits_u32` is a standalone function so its cutoff is unit-testable.

**C3. Merge without mutating or copying `seg` (exact).** Split `merge_segments` into
`compute_merge(const rg&, counts&, tholds, lowt) -> {lut, new_rg}` and output-time application.
The union-find, `try_merge`, `remaps`, `counts` update and new-graph construction are unchanged in
logic and order. After `remaps` is built, form `lut[id] = remaps[find_set(id)]` for all `id < N`
(exactly what the old per-voxel loop computed), then **free rank/parent/remaps before building the
LUT's consumers** except what `new_rg` needs (the new-graph loop uses `remaps[find_set(.)]`; build
`new_rg` from `lut` instead, which is the same value). `seg` keeps raw watershed ids for the whole
run, so the K-threshold loop needs no `seg_copy`, no `rg_copy` (input taken by const ref), and no
in-place remap pass. LUT lifetime: one threshold, freed after that threshold's write. It is O(N)
and N can approach P in pathological inputs — reported by C0, not assumed small.

**C4. Drop redundant `in_rg` (exact).** In the new-graph loop an edge is emitted only if
`a1 = mst.find(s1) != a2 = mst.find(s2)`, after which `mst.link(a1, a2)` makes them equal forever;
any later edge with the same `(s1, s2)` therefore fails the component test before reaching the
`in_rg` check, so `in_rg[mm.first].count(mm.second) == 0` is always true when evaluated. Remove
`in_rg` (N `std::set` headers plus nodes). The gate's `dend_*` comparison is the empirical check.

**C5. Streamed output (exact).** Replace `relabel_segments` + `write_volume` +
`write_multi_array` for the main volume with a writer that walks the interior in the same Fortran
order and `range(1, n-1)` crop, maps each voxel through the threshold's LUT then
`v == 0 ? 0 : v + offset` (exactly `relabel_segments`), and writes through a fixed buffer
(one x-y plane) with `write(2)`/`std::ofstream::write` to `<name>.tmp`, renamed on success. No
full-size copy, no full-size mapping. `write_chunk_boundaries` takes the same per-element transform
and extracts each face into a face-sized buffer (still uint64 `seg_t`); `aff_i_*` untouched.
`relabel_region_graph` takes its input by const ref (drops the by-value copy).

**C6. Streaming edge accumulators (exact for max/mean).** In `get_region_graph`, keyed map values
become a small accumulator chosen by mode, keeping `pairs` in first-seen order so pre-sort order and
the `stable_sort` result are unchanged:
* `MAX`: initialise from the first observed value, update with `if (cur < a) cur = a;` — this is
  `std::max_element`'s comparison (first maximal element under `operator<`), so NaN and ±0 behave
  identically.
* `MEAN`: `F sum` starting at `F(0)`, `sum = sum + a` in push order, plus `std::size_t count`;
  score `sum / static_cast<F>(count)` — the same expression sequence as `std::accumulate(..., F(0))`
  followed by the existing division. Exactness additionally relies on identical compiler/FP flags;
  the scratch builds reuse the stock `CMAKE_CXX_FLAGS` verbatim.
* `PERCENTILE`: unchanged (keeps per-edge vectors; `nth_element` needs all values).
If the gate shows any MEAN mismatch, drop C6-MEAN and keep C6-MAX.

**C7. Opt-in uint32 segmentation output.**
* Binary: optional token `--seg-dtype=uint32|uint64`, accepted anywhere after argv[7], removed
  before the existing positional parsing (which keys off `isalpha(argv[8][0])`). Unknown value or
  duplicate token → error, exit 2. Default uint64.
* Only `seg_<tag>.data` / `seg_<tag>_<i>.data` change dtype; every other file is unchanged.
* Per threshold, before writing: require `offset <= UINT32_MAX` and
  `max_id <= UINT32_MAX - offset` (no unchecked `offset + max_id`), where `max_id = lut size - 1`.
  On failure: error message, exit 3, no file for that threshold (the `.tmp` is never created or is
  removed). In batch mode earlier thresholds' complete outputs may remain; the non-zero exit makes
  `subprocess.run(check=True)` raise, so the Python driver never reads them. Documented.
* Python: `--seg-dtype {uint64,uint32,auto}` (default `uint64`). The token is passed to the binary
  **only for uint32**, so default argv is byte-identical to today and works with the stock
  binaries. `auto` resolves to uint32 iff `offset + I <= 2**32 - 1` (I interior voxels bounds every
  output id), else uint64. `_read_segmentation_xyz(..., dtype=np.uint64)` uses the dtype's itemsize
  for both the interior and the halo file-size layouts. Output arrays/h5 keep the chosen dtype.

**C8. Python driver holds no input copy across the subprocess.** In `main()`, hand `predictions`
to `_run_abiss_ws` without keeping a caller reference (e.g. `holder = [predictions]; del predictions;
...predictions_czyx=holder.pop()`), and inside `_run_abiss_ws` `del predictions_czyx` right after
`_to_abiss_affinity`. Tutorial callers keep their own references and are unaffected.

**Considered, not pursued:** releasing the affinity mapping after graph construction (the pages
are clean and reclaimable, and `aff_i_*` faces are needed at write time; C0 reports `RssFile` so
this is visible if it matters); read-only affinity mmap (hygiene, template const churn); returning
a memmap-backed segmentation from `_read_segmentation_xyz` (temp-dir lifetime contract — follow-up;
uint32 already halves this copy); a percentile quantile sketch (not exact).

## Files and Areas

| File | Change |
|---|---|
| `lib/abiss/src/ws/basic_watershed.hpp` | C2 bounded, width-templated queue; `bfs_fits_u32` |
| `lib/abiss/src/ws/region_graph.hpp` | C6 accumulators; C0 call points |
| `lib/abiss/src/ws/agglomeration.hpp` | C3 `compute_merge` + LUT; C4 drop `in_rg` |
| `lib/abiss/src/ws/utils.hpp` | C5 streamed volume writer, transform-aware face writer, const-ref graph relabel; C0 helper |
| `lib/abiss/src/ws/atomic_chunk.cpp` | C1 check, C7 token parse + overflow check, wire C3/C5, remove `relabel_segments`/`seg_copy`/`rg_copy` |
| `lib/abiss/CMakeLists.txt` | `test_ws_bfs` under `BUILD_TESTING` |
| `lib/abiss/tests/test_ws_bfs.cpp` | `watershed<ID,uint32_t>` vs `watershed<ID,ptrdiff_t>` identical on synthetic affinities; `bfs_fits_u32` at P = 2^32−1, 2^32, 2^32+1 |
| `lib/abiss/tests/ws_gate.py` | synthetic + real-crop bit-exact gate, uint32/overflow checks, memory table |
| `scripts/run_abiss_volume.py` | C7 Python side, C8 |
| `tests/unit/test_abiss_seg_dtype.py` | reader dtype × {interior, halo} layouts; `auto` at exact fit / one past; default argv has no token; uint32 argv has it; batch callback receives uint32 arrays |

## Verification Plan

0. **Reference build + determinism.** Build unmodified `92abc91` source into `lib/abiss/build_ref/`
   with the stock `CMAKE_CXX_FLAGS`/compiler; build the modified source into `build_mem/` (both
   `ws` and `ws64`). Run stock `build/ws` twice and `build_ref/ws` once on the same small input and
   compare all files. This establishes (a) whether `dend_*` padding bytes are deterministic and
   (b) whether `build/ws` (Aug 13) matches current source. Confirm `build/ws` md5 unchanged.
1. **Unit tests.** `ctest` in `build_mem` (existing nuc tests + `test_ws_bfs`). `pytest
   tests/unit/test_abiss_seg_dtype.py tests/unit/test_decode_abiss_wrapper.py
   tests/unit/test_abiss_edge_storage.py tests/unit/test_abiss_s3_chunked.py -q`.
2. **Output contract** (new vs `build_ref` and vs stock `build/ws`, `build64/ws64`): all files
   byte-identical, **except** `dend_*` if step 0 shows stock-vs-stock padding nondeterminism, in
   which case `dend_*` must have identical size and identical `(score, id1, id2)` fields parsed per
   record. The record layout (`std::tuple<float, uint64, uint64>`, 24 B) must not change.
   Cases, each for `ws` and `ws64`, nonzero `offset`:
   * Synthetic (generated by `ws_gate.py`, ~20×24×16, fixed seeds): all-zero affinity
     (background-only); zero planes separating blocks (isolated components); affinities quantized
     to a 0.1 grid (ties, repeated edge observations); small components under `size`/`dust`
     thresholds.
   * Real: crops of the 18 nm LICONN val affinity used by
     `tutorials/neuron_liconn_ist/slurm/ws_sizing.py`, written through `run_abiss_volume` helpers,
     at 64×512×512 and 128×1024×1024.
   * Modes: single threshold `max`; 3 thresholds × {`max`, `mean`, `p75`}.
   * Boundary flags: `1 1 1 1 1 1`, `0 0 0 0 0 0` (every seg and `aff_i` face), `0 1 0 1 0 1`.
   Report files compared per case and any mismatch verbatim.
3. **uint32 mode.** New `seg_*.data` equals the stock uint64 file `.astype(np.uint32)`; all other
   files satisfy step 2. Deterministic overflow on a synthetic input with a known label count M
   (read from `meta_*`): `offset = 2**32 - 1 - M` succeeds and round-trips; `offset + 1` exits 3 and
   leaves no `seg_*` file for that threshold. Token placement before/after the merge function and
   thresholds; invalid value → exit 2.
4. **C1.** A param file whose product exceeds `high_bit` for `ws` exits 2 immediately with the
   message (no affinity read needed — use a tiny/sparse file).
5. **Memory and runtime** on a compute node via SLURM (scripts and logs on /projects), stock vs new,
   `ws` and `ws64`, single and 5-threshold, on 128×1024×1024 and one ≥1 Gvox crop (≤ 2^31 padded
   so `ws` is valid): binary peak via `/usr/bin/time -v` (max RSS, wall) and the C0 `[mem]` lines;
   and the whole `run_abiss_volume.py` job peak via the job cgroup `memory.peak` if readable, else
   `sacct MaxRSS`, with and without `--seg-dtype uint32`. Report a B/voxel table.
   **Acceptance:** new ≤ stock peak RSS in every case; ≥ 30% lower binary peak RSS for 5-threshold
   `ws64` on the ≥1 Gvox crop; wall time ≤ 1.15× stock. If a target is missed, report the number
   and the C0 attribution rather than tuning around it.
6. **End-to-end Python:** `run_abiss_volume.py` on a small crop with `--seg-dtype uint64`, `uint32`,
   `auto`, single and batch (`on_batch_result`) paths → identical label values; uint64 output
   identical to today's.

## Risks and Questions

* **C2 order equivalence** rests on the cursor invariant stated above; `test_ws_bfs` (both queue
  widths) and gate step 2 are the checks. The wide-queue branch is exercised at small scale by the
  unit test, not by a > 2^32 run.
* **C6 MEAN** depends on identical FP flags; fallback defined.
* **C5 page cache:** streamed writes still create dirty file pages; with a bounded buffer they are
  written back progressively instead of materialising an 8·I mapping. C0 `RssFile` shows the effect.
* **Stock baseline validity:** if step 0 shows `build/ws` ≠ `build_ref/ws`, the gate compares against
  `build_ref` for exactness and reports the stock difference separately (it would mean the deployed
  binary predates the current source).
* **uint32 consumers:** unchanged default; tutorial scripts that memmap uint64 must not be pointed at
  uint32 outputs. Non-blocking question for the user: adopt `auto` in the LICONN sweep scripts later?
* Acceptance metric (answering review Q3): both — binary RSS decides node class; job peak is what
  SLURM enforces.

## Changes Since Previous Plan Version

Addresses every plan_v0_review finding:

1. [major] default uint64 token → C7: token passed only for uint32; unit test asserts default argv unchanged and stock-compatible.
2. [major] C4-mean overflow / max semantics → C6: `size_t` count; MAX uses first-value init and `std::max_element`'s `<` comparison; FP-flag dependency stated, scratch builds reuse stock flags; fallback if MEAN mismatches.
3. [major] incomplete memory model → Summary table rewritten with P vs I, anon vs file, including the cropped copy, output mapping, union-find/remap/counts, `in_rg`, graph copies and sort buffer; C5 now removes both the copy and the mapping.
4. [major] instrumentation/acceptance → C0 call points inside phases, `RssAnon`/`RssFile`/`VmHWM`; step 5 measures binary and whole-job (cgroup/sacct) peaks, adds acceptance thresholds and runtime bound; C8 removes the driver's retained input.
5. [major] assert under `-DNDEBUG` → verified stock builds use `-DNDEBUG`; C1 replaces the assert with checked multiplication and a runtime exit; 3.82 Gvox number declared invalid; `bfs_fits_u32` cutoff unit-tested at 2^32±1 and both queue widths compared on identical inputs.
6. [major] verification gaps → step 2 adds synthetic edge cases and flag sets `0 0 0 0 0 0` / `0 1 0 1 0 1`; step 3 deterministic exact-fit/one-past overflow via known label count and overflow-safe check; token placement/invalid values; step 6 batch callback; reader layouts in unit tests.
7. [major] `dend_*` padding → step 0 determinism check and an explicit contract (byte-exact, or field-exact with unchanged record layout if stock itself is nondeterministic).
8. [minor] LUT lifetime → C3 states O(N), per-threshold lifetime, early release of union-find storage; `relabel_region_graph` by const ref (C5).
9. [minor] extra reductions → adopted bounded BFS queue (C2) and `in_rg` removal with proof (C4); memmap-backed reader and affinity release considered and deferred with reasons.

Review questions answered: 3.82 Gvox (invalid, `-DNDEBUG`); compiler flags (listed in Summary); metric (both); padding contract (step 2); batch overflow (earlier outputs may remain, non-zero exit, documented).
