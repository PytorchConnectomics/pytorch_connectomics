# Plan v2

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

Target after this plan: anon ≈ S·P + Qcap·sizeof(Q) + O(N + E), with no per-voxel output staging,
where Qcap is the BFS vector capacity (≤ ~2× the largest single search; O(P) in the worst case of
one search spanning the volume, reported by C0 rather than assumed small).

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
`lib/abiss` is a separate repo in the main checkout (branch `main`, HEAD `92abc91`, clean
apart from untracked build dirs). Edit its `src/ws/`, `tests/` and `CMakeLists.txt` in place on the
existing branch — no new branch, no commits, no `stash`/`checkout`. The stock binaries are prebuilt
and are not rebuilt by this work, so in-place source edits do not change live runs. Pristine
reference source for `build_ref/` comes from `git -C <abiss> archive 92abc91 | tar -x -C <scratch>`.
Report `git -C <abiss> diff` and new untracked files in `code_vN.md`.

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
with `clear()` (logical size 0, capacity retained) at each seed, so its capacity is ~2× the
**largest single search**, not P (still O(P) if one search spans the volume). Element type is a template parameter
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
`in_rg` (N `std::set` headers plus nodes). The gate's `dend_*` field comparison is the empirical check.

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
  `max_id <= UINT32_MAX - offset` (no unchecked `offset + max_id`), where `max_id` is the final
  **compact** label count `counts.size() - 1` after `compute_merge` (the LUT spans original watershed
  ids, so its length is NOT the max output id).
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
| `lib/abiss/tests/test_ws_bfs.cpp` | `watershed<ID,uint32_t>` vs `watershed<ID,ptrdiff_t>` identical on synthetic affinities; `bfs_fits_u32` at P = 2^32−1, 2^32, 2^32+1; C1 check function at `high_bit`±1 and product overflow; C6 accumulator vs `compute_edge_score` on rounding-sensitive vectors (MAX incl. NaN/±0, MEAN); `--dend-layout` prints ABI field offsets |
| `lib/abiss/tests/ws_gate.py` | synthetic + real-crop bit-exact gate, uint32/overflow checks, memory table |
| `scripts/run_abiss_volume.py` | C7 Python side, C8 |
| `tests/unit/test_abiss_seg_dtype.py` | reader dtype × {interior, halo} layouts; `auto` at exact fit / one past; default argv has no token; uint32 argv has it; batch callback receives uint32 arrays; callback never called when the subprocess fails (fake binary exiting 3) |

## Verification Plan

0. **Reference build + determinism.** Build unmodified `92abc91` source (from `git archive`) into a
   scratch `build_ref/`
   with the stock `CMAKE_CXX_FLAGS`/compiler; build the modified source into `build_mem/` (both
   `ws` and `ws64`). Run stock `build/ws` twice and `build_ref/ws` once on the same small input and
   compare all files. This reports (for information only) whether `dend_*` padding bytes happen to
   match, and establishes whether `build/ws` (Aug 13) matches current source. Confirm `build/ws` md5
   unchanged. Record the stock FP environment: the full stock flag set has no `-ffast-math`,
   `-fassociative-math` or `-Ofast`, and `-march=nocona` has no FMA, so GCC cannot contract or
   reassociate the float additions in C6-MEAN; the gate also greps the scratch builds'
   `compile_commands.json`/flags to confirm the same holds for `build_mem`.
1. **Unit tests.** `ctest` in `build_mem` (existing nuc tests + `test_ws_bfs`). `pytest
   tests/unit/test_abiss_seg_dtype.py tests/unit/test_decode_abiss_wrapper.py
   tests/unit/test_abiss_edge_storage.py tests/unit/test_abiss_s3_chunked.py -q`.
2. **Output contract** (new vs `build_ref` and vs stock `build/ws`, `build64/ws64`): every file
   except `dend_*` byte-identical. `dend_*` (unconditionally): identical file size, identical record
   count and record size (24 B), and identical `(score, id1, id2)` field values parsed at the
   **actual ABI offsets**, which `test_ws_bfs --dend-layout` prints from
   `region_graph<seg_t, aff_t>` element addresses (`&std::get<k>(t) - &t`), not inferred from the
   template argument order. Padding-byte differences are reported separately and do not fail.
   Cases, each for `ws` and `ws64`, nonzero `offset`:
   * Synthetic (generated by `ws_gate.py`, ~20×24×16, fixed seeds): all-zero affinity
     (background-only); zero planes separating blocks (isolated components); affinities quantized
     to a 0.1 grid (ties, repeated edge observations); small components under `size`/`dust`
     thresholds; a heavy-merge case (low merge threshold, many watershed basins → few outputs) so
     the compact label count is far below the watershed id count; rounding-sensitive MEAN inputs
     (values like 0.1, 0.2, 0.3, 1e-7 mixed in varying order along one edge).
   * Real: crops of the 18 nm LICONN val affinity used by
     `tutorials/neuron_liconn_ist/slurm/ws_sizing.py`, written through `run_abiss_volume` helpers,
     at 64×512×512 and 128×1024×1024.
   * Modes: single threshold `max`; 3 thresholds × {`max`, `mean`, `p75`}.
   * Boundary flags: `1 1 1 1 1 1`, `0 0 0 0 0 0` (every seg and `aff_i` face), `0 1 0 1 0 1`.
   Report files compared per case and any mismatch verbatim.
3. **uint32 mode.** New `seg_*.data` equals the stock uint64 file `.astype(np.uint32)`; all other
   files satisfy step 2. Deterministic bound on the heavy-merge synthetic input with known final
   compact count M (from `meta_*`, which is `counts.size()-1`): `offset = 2**32 - 1 - M` succeeds
   and round-trips; `offset + 1` exits 3 and leaves no `seg_*`/`.tmp` for that threshold.
   `offset = 2**32` (> UINT32_MAX) with uint32 exits 3. Token placement before/after the merge
   function and thresholds; invalid value → exit 2; duplicate token → exit 2. **Batch failure after
   success:** 3 thresholds ordered so threshold 0 fits and threshold 1 overflows (a high merge
   threshold leaves more labels, so choose `offset` between the two compact counts) → exit 3,
   `seg_*_0.data` complete, no `seg_*_1*` or `seg_*_2*` artifacts; and the Python driver
   (`_run_abiss_ws` with an `on_batch_result` spy) raises `CalledProcessError` with the spy never
   called.
4. **C1.** Param files (tiny, no affinity read needed): P exactly `high_bit − 1` for `ws` passes the
   check (verified by the check's own log line; the run may then be aborted before allocation via a
   missing affinity file), P = `high_bit` exits 2, a larger P exits 2, and dimensions whose product
   overflows `size_t` exit 2. The pure check function is also unit-tested in `test_ws_bfs`.
5. **Memory and runtime** on a compute node via SLURM (scripts and logs on /projects), stock vs new,
   `ws` and `ws64`, single and 5-threshold, on 128×1024×1024 and one ≥1 Gvox crop (≤ 2^31 padded
   so `ws` is valid).
   * **Binary peak + attribution:** `ws_gate.py` samples `/proc/<pid>/status` (`RssAnon`,
     `RssFile`, `VmRSS`) every 20 ms and timestamps each C0 `[mem]` marker line from the child's
     stdout, so every sample is assigned to the phase bracketed by consecutive markers (the
     `stable_sort` temp buffer falls in the "graph: sort" bracket; C0 therefore logs a marker
     *before* and *after* each allocation-heavy step, not only after). Per-phase max anon/file is
     reported; `VmHWM` only as the overall cross-check. Also `/usr/bin/time -v` max RSS and wall.
   * **Aggregate job memory:** each measured run executes as its own `srun` step; the gate reads
     that step's cgroup v2 `memory.peak` if present, else samples its `memory.current` every 20 ms
     and reports the max, plus `memory.stat` `file`/`file_dirty` at the write phase (the
     streamed-write page cache appears here, not in the writer's `RssFile`). If neither file is
     readable in the step cgroup, the aggregate metric is reported as **unverified**; `sacct MaxRSS`
     is shown only as a labelled per-process number, not as a substitute. Done for both the bare
     binary and the whole `run_abiss_volume.py` job (with and without `--seg-dtype uint32`).
   * Report a B/voxel table (padded).
   **Acceptance:** new ≤ stock peak binary RSS in every case; ≥ 30% lower binary peak RSS for
   5-threshold `ws64` on the ≥1 Gvox crop; wall time ≤ 1.15× stock. If a target is missed, report
   the number and the per-phase attribution rather than tuning around it.
6. **End-to-end Python:** `run_abiss_volume.py` on a small crop with `--seg-dtype uint64`, `uint32`,
   `auto`, single and batch (`on_batch_result`) paths → identical label values; uint64 output
   identical to today's.

## Risks and Questions

* **C2 order equivalence** rests on the cursor invariant stated above; `test_ws_bfs` (both queue
  widths) and gate step 2 are the checks. The wide-queue branch is exercised at small scale by the
  unit test, not by a > 2^32 run.
* **C6 MEAN** depends on identical FP flags; fallback defined.
* **C5 page cache:** streamed writes still create dirty page cache charged to the cgroup (not visible
  in the writer's `RssFile`); with a bounded buffer and `write(2)` the kernel can write back and
  reclaim progressively, whereas today's 8·I shared mapping is resident in the process. Step 5's
  cgroup `memory.stat` is where this is measured.
* **Stock baseline validity:** if step 0 shows `build/ws` ≠ `build_ref/ws`, the gate compares against
  `build_ref` for exactness and reports the stock difference separately (it would mean the deployed
  binary predates the current source).
* **uint32 consumers:** unchanged default; tutorial scripts that memmap uint64 must not be pointed at
  uint32 outputs. Non-blocking question for the user: adopt `auto` in the LICONN sweep scripts later?
* Acceptance metric (answering review Q3): both — binary RSS decides node class; job peak is what
  SLURM enforces.

## Changes Since Previous Plan Version

Addresses every plan_v1_review finding (plan_v0_review resolutions from v1 are retained):

1. [major] wrong uint32 bound → C7 now uses the final compact count `counts.size() - 1`; step 3 adds a heavy-merge fixture whose compact count is far below the watershed id count and tests exact fit / one past on it.
2. [major] memory attribution → step 5: 20 ms `/proc/<pid>/status` sampling bracketed by timestamped C0 markers placed before *and* after each allocation-heavy step (incl. `stable_sort`); per-step cgroup v2 `memory.peak` or sampled `memory.current` + `memory.stat` for aggregate and page cache; explicit "unverified" if unreadable; `sacct MaxRSS` demoted to a labelled per-process figure. C5 risk text corrected (page cache is cgroup-charged, not writer `RssFile`).
3. [major] `dend_*` contract → unconditional: identical size, record count, record size and field values at ABI offsets printed by `test_ws_bfs --dend-layout`; padding diffs reported separately. Step 0 determinism is informational only.
4. [major] failure paths → step 3 adds duplicate token, `offset > UINT32_MAX`, batch failure after an earlier success (artifact check + Python callback spy); step 4 adds exact `high_bit − 1`/`high_bit`/larger/product-overflow cases; unit tests for the check function.
5. [minor] queue term → target bound now includes `Qcap·sizeof(Q)` (O(P) worst case); C2 specifies `clear()` with retained capacity at each seed.
6. [minor] branch creation → removed; edit lib/abiss in place on the existing branch, no commits; reference source via `git archive 92abc91`.

Review questions answered: compact count (Q1); field equality is unconditional (Q2); sampled cgroup `memory.current`, else "unverified" (Q3); stock flags contain no `-ffast-math`/`-fassociative-math`/`-Ofast` and `-march=nocona` has no FMA, recorded and re-checked in step 0 (Q4).
