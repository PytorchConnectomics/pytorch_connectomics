# Code v0

## Overview

Implemented C0–C8 for ABISS `ws`/`ws64` and `scripts/run_abiss_volume.py`, including all four binding review notes. Main segmentation output optionally uses uint32; default output and argv remain uint64-compatible. Stock builds were never rebuilt or overwritten. No commits, branches, stashes, checkouts, or SLURM submissions were made.

The memory/runtime acceptance benchmark is deliberately deferred to the coordinator. Its driver and submission script are ready; no memory-reduction percentage or runtime target is claimed from the local correctness tests.

## What Changed

- Checked dimension multiplication and watershed index bounds before accessing affinity data; invalid sizes exit 2 independently of assertions.
- Replaced the full-volume BFS allocation with a reused, width-selected vector retaining only peak search capacity.
- Replaced segmentation/region-graph copies and per-voxel remapping with a per-threshold merge LUT; removed redundant `in_rg` sets and freed union-find storage before constructing the new graph.
- Streamed the main cropped segmentation through one x-y plane to an atomically renamed temporary file. Boundary segmentation remains uint64 and uses face-sized buffers.
- Used ordered MAX/MEAN accumulators instead of retaining all boundary observations; kept exact PERCENTILE vectors. Preserved first-seen pair ordering and stable sorting.
- Added flushed in-phase memory markers, dtype-aware Python reads, conservative `auto`, overflow/token failure handling, and input ownership transfer before the subprocess.
- Added C++ and Python contract tests, an exhaustive synthetic/real-crop comparison driver, and a deferred SLURM benchmark driver.

## Implementation Details

**C0 and review note 2:** `/proc/self/status` markers report RssAnon, RssFile, and VmHWM in GiB, with `std::endl` flushing. Allocation-heavy phases are bracketed, including watershed, BFS (before destruction), edge fill/scoring, sort, counts copies, union-find/LUT/new graph, and output. The benchmark samples every 20 ms, timestamps markers, stores per-process samples and cgroup `memory.stat` file/file_dirty, prefers each step's `memory.peak`, falls back to sampled `memory.current`, and explicitly reports unverified aggregate memory if neither is readable. Stock binaries have no C0 markers: their phase attribution is explicitly unavailable; only overall peaks are reported. `/usr/bin/time -v` and labelled per-process `sacct` output are retained. Each measured command gets its own `srun` step, including whole Python jobs.

**C1:** The pure size predicate checks both multiplications for overflow, `P < high_bit`, and `P <= PTRDIFF_MAX`. The CLI logs a successful size check before the separate minimum-halo-dimension check. Therefore the exact `high_bit-1` probes using `(high_bit-1,1,1)` prove the size predicate passes, then safely exit 2 for invalid halo dimensions without mapping or allocating an enormous affinity. At/high/overflow probes fail the size check itself.

**C2:** Queue storage is uint32 when `P-1 <= UINT32_MAX`; otherwise it is ptrdiff_t. Neighbour arithmetic remains signed. Queue capacity and element width are logged. The forced-width unit tests compare all segmentation bytes and counts for both internal ID widths.

**C3–C5:** The LUT spans original watershed IDs, but output bounds use the final compact `counts.size()-1`. Graph construction consumes the same LUT; raw watershed segmentation stays immutable across thresholds. The writer uses x-fastest Fortran order and the original one-voxel interior crop. Segmentation faces apply the identical LUT/offset transform and remain uint64; affinity faces are unchanged. Only main `seg_<tag>[_<i>].data` can be uint32. A failed uint32 threshold exits 3 before writing any of that threshold's artifacts; previous completed thresholds may remain. Python raises before reading or invoking batch callbacks.

**C6 and review note 1:** Let P be padded voxels, N original watershed IDs, E graph edges, B retained boundary observations, A an interior x-y plane, and F the largest face. For MAX/MEAN the live anonymous-storage bound is `S*P + Qcap*sizeof(Q) + O(N+E) + O(sizeof(output)*A + 8*F)` (buffers belong to different phases; this is an upper bound, not a measured peak). PERCENTILE additionally retains `O(sizeof(aff_t)*B)` during graph scoring. Queue capacity can be O(P), and N can approach P. Affinity mappings, sidecar/face mappings, and kernel write page cache are separate file-backed/cgroup terms. The main volume no longer has a full-size output mapping or owning cropped copy. Allocator retention may keep RSS above live allocation size; step 5 measures that.

**C7:** The binary strips exactly one optional dtype token anywhere after argv[7] before positional parsing; invalid values or duplicates exit 2. The default Python call adds no token. `auto` uses Python integer arithmetic and selects uint32 only when `offset + interior_voxels <= UINT32_MAX`. Reader byte-layout checks use the selected dtype's itemsize for both cropped and halo layouts, and arrays/HDF5 outputs preserve dtype.

**C8 correction to the suggested implementation:** A regression test demonstrated that `predictions_czyx=holder.pop()` at the call site retains the array on CPython's call stack throughout the call. `main()` instead passes the one-element holder, and `_run_abiss_ws` pops it internally, then deletes the ndarray after conversion and drops affinity storage before the subprocess. Existing ndarray callers keep their existing ownership semantics. The weakref test verifies actual reclamation before `subprocess.run`.

**Review notes 3–4:** Dendrogram ABI layout is obtained using byte-pointer differences, not template ordering: record size 24, score offset 16 (4 bytes), id1 offset 8 (8 bytes), id2 offset 0 (8 bytes). Comparison checks exact field bytes, including signed zero/NaN representation, and reports padding separately. `test_ws_bfs` is explicitly compiled with NDEBUG and uses throwing checks/nonzero exit status, never assertions. Accumulator tests compare float bytes, including NaNs and signed zeros.

**Correction to the plan's stock-build assumption:** The cached Release value is `-O3 -DNDEBUG`, but the generated stock `flags.make` actually uses `-O3 -fopenmp` and no NDEBUG. `strings build/ws` contains the original chunk-size assertion expression. Thus the plan's claim that this deployed binary compiled out that assertion is not supported, nor is its inference that the old 3.82 Gvox run silently overflowed. Both scratch builds match the actual stock CXX_FLAGS, compiler path, Release mode, system allocator, and absence of absl. `readelf -p .comment` reports the same conda GCC 15.2.0 in stock/new. No fast-math, associative-math, Ofast, or FMA flag is present; architecture remains nocona. The test target alone explicitly adds NDEBUG.

## Files Changed

Paths beginning `lib/abiss/` refer to the separate repository at `/projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss`, not a directory inside this Python worktree. All source/review changes are listed below. Generated verification trees are listed separately as directories rather than enumerating CMake internals and per-case binary outputs.

| File | Purpose |
|---|---|
| `lib/abiss/CMakeLists.txt` | Register the active-under-NDEBUG C++ watershed contract test. |
| `lib/abiss/src/ws/atomic_chunk.cpp` | Size guard, dtype parsing/bounds, queue dispatch, immutable merge/write orchestration, markers. |
| `lib/abiss/src/ws/basic_watershed.hpp` | Reusable width-templated BFS queue and capacity instrumentation. |
| `lib/abiss/src/ws/region_graph.hpp` | MAX/MEAN accumulators, unchanged exact percentile scoring, phase markers and earlier release. |
| `lib/abiss/src/ws/agglomeration.hpp` | `compute_merge`, per-threshold LUT, no segmentation mutation or redundant edge sets. |
| `lib/abiss/src/ws/utils.hpp` | Checked-size/queue-cutoff helpers, memory markers, plane streaming and transformed uint64 faces. |
| `lib/abiss/tests/test_ws_bfs.cpp` | Queue-width equivalence, size boundaries, bytewise accumulator checks, actual dend ABI layout. |
| `lib/abiss/tests/ws_gate.py` | Steps 0/2/3/4/6 correctness gates; deferred step-5 SLURM measurement/acceptance driver. |
| `lib/abiss/tests/ws_benchmark.sbatch` | Ready-to-submit compute-node job, modest threads, separate srun measurements. |
| `scripts/run_abiss_volume.py` | Output dtype/auto, unchanged default argv, dtype-aware reads and input ownership transfer. |
| `tests/unit/test_abiss_seg_dtype.py` | Layouts, auto bounds, argv, callbacks/failures, actual input lifetime. |
| `.agent/decoding/abiss_memory/artifacts/code_v0.md` | This implementation and verification artifact. |
| `lib/abiss/build_ref/` (generated, untracked) | Pristine archived reference source, CMake outputs, reference ws/ws64 and build logs. |
| `lib/abiss/build_mem/` (generated, untracked) | Modified binaries/tests, build/test logs, gate_scratch data, results and per-case logs. |

## Git Baseline

run_start_ref: 3cbdf39c06bf42c4b7cf739e1208751403a18bc7
current_head: 3cbdf39c06bf42c4b7cf739e1208751403a18bc7
abiss_start_ref: 92abc91f496304ecfe3d5c593e7713e8abd98011
abiss_current_head: 92abc91f496304ecfe3d5c593e7713e8abd98011

Python initially had only the pre-existing untracked CCC run directory. ABISS initially had untracked `build/`, `build64/`, `scripts/__pycache__/`, and `tests/__pycache__/`; these were not cleaned or staged. Review ABISS with `git -C /projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss diff` and its three new files under `tests/`. Scratch builds remain untracked. No HEAD or index changes were made.

## Verification

All local validation used `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc` and `OMP_NUM_THREADS=4`. Let `A=/projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss` and `W=/projects/weilab/weidf/lib/pytorch_connectomics/.claude/worktrees/bridge-cse_01Y1t9gvwzh3sWG6hJ5p7qXF` in the commands below.

**Build and baseline commands actually run:** A Python subprocess wrapper ran `git -C "$A" archive 92abc91` piped to `tar -x -C "$A/build_ref/source"`; it read the compiler and CXX_FLAGS verbatim from `build/CMakeCache.txt`, then ran CMake with these arguments for each scratch tree:

```bash
cmake -S "$A/build_ref/source" -B "$A/build_ref" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/projects/weilab/weidf/lib/miniconda3/envs/pytc/bin/x86_64-conda-linux-gnu-c++ \
  '-DCMAKE_CXX_FLAGS=-fvisibility-inlines-hidden -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /projects/weilab/weidf/lib/miniconda3/envs/pytc/include' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_DISABLE_FIND_PACKAGE_absl=ON \
  -DABISS_ALLOCATOR=system -DBUILD_TESTING=ON -DEXTRACT_SIZE=ON
# Same arguments, with -S "$A" -B "$A/build_mem", for the modified build.
cmake --build "$A/build_mem" --target ws ws64 test_ws_bfs test_nuc_algebra test_nuc_extractor agg match_chunks -j 2
cmake --build "$A/build_ref" --target ws ws64 -j 2
cmake --build "$A/build_mem" --target ws ws64 -j 2
ctest --test-dir "$A/build_mem" --output-on-failure
```

All build commands completed successfully. CTest passed all 5 tests (ws_bfs and the four existing nucleus tests), both initially and after the last counts-copy marker was added. See `build_mem/{configure,build,build_final,ctest,ctest_final}.log` and `build_ref/{configure,build}.log`. Stock ws ran twice and reference ws once on the same fixture; their file counts and padding-only differences are included below. Actual compiler flags were compared across all four trees; `readelf -p .comment` and `strings build/ws` were also inspected, as described above.

**Focused Python command actually run (final rerun):**

```bash
pytest tests/unit/test_abiss_seg_dtype.py tests/unit/test_decode_abiss_wrapper.py \
  tests/unit/test_abiss_edge_storage.py tests/unit/test_abiss_s3_chunked.py -q \
  --basetemp="$A/build_mem/gate_scratch/pytest"
```

Result: **34 passed, 1 skipped, 3 existing SWIG deprecation warnings**. The new file contributes 13 passing test instances. The skipped relative-script wrapper test requires a binary under this worktree's nonexistent `lib/abiss/build/ws`. See `build_mem/pytest_verified.log`. Earlier runs were 33 passed/1 skipped before adding the lifetime test, then 1 failed/33 passed/1 skipped when that test exposed the call-site-pop retention; the correction and exact failure are documented below.

**Correctness command actually run:**

```bash
python "$A/tests/ws_gate.py" --mode local --python-repo "$W"
```

Outputs are in `build_mem/local_gate.log`, `build_mem/gate_scratch/local_results.json`, and `build_mem/gate_scratch/logs/<case>/{ref,stock,new,u32}.log`. Large per-case output files were deleted only after successful comparison; logs and summaries remain. The one real crop was read with `f['main'][:3,:64,:512,:512]` from the specified EB2 source and converted using the driver helpers. No full-source read or larger real crop was performed locally. Main non-overflow cases use offset 17, and test each mode/flag combination in both binary widths against both stock and archived reference, in both output dtypes.

**Gate result: PASS.** All 168 matrix cases passed (144 synthetic + 24 real). Each case passed all four comparisons: reference/new-uint64, reference/new-uint32, stock/new-uint64, stock/new-uint32. The matrix performed **19,600 file-pair comparisons**, including exact non-dend bytes (or uint32 label casting) and unconditional dend field-byte/record-size checks. No stock/reference disagreement and no MEAN mismatch occurred; the MEAN accumulator is retained. Padding-only dend differences occurred in 213 dend files for reference/new-uint64 and 213 for stock/new-uint64 (the uint32 comparison counts are 213 and 213). These are informational, excluded from the field-byte contract.

Both determinism comparisons passed 19 files each with no padding differences. The final ABI probe returned `{"size":24,"score":16,"id1":8,"id2":0}`. Closing stock `build/ws` MD5: `bf4c7343dccf86e60d78f772e583492d`. `build64/ws64` MD5 was also recorded: `ab83e053e14c570a1903a59a9236b797`.

Per-case results follow. `max1` is one threshold; `max3`, `mean3`, `p753` are three thresholds. “Files” is the count **per comparison** (four comparisons per row). Padding columns count differing dend files, not differing payload fields. All four comparisons passed every row.

| Fixture | Binary | Mode | Flags | Files | Ref/new padding files | Stock/new padding files | Result |
|---|---|---|---|---|---|---|---|
| zero | ws | max1 | 111111 | 4 | 0 | 0 | PASS (all 4) |
| zero | ws | max1 | 000000 | 19 | 0 | 0 | PASS (all 4) |
| zero | ws | max1 | 010101 | 12 | 0 | 0 | PASS (all 4) |
| zero | ws | max3 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| zero | ws | max3 | 000000 | 57 | 0 | 0 | PASS (all 4) |
| zero | ws | max3 | 010101 | 36 | 0 | 0 | PASS (all 4) |
| zero | ws | mean3 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| zero | ws | mean3 | 000000 | 57 | 0 | 0 | PASS (all 4) |
| zero | ws | mean3 | 010101 | 36 | 0 | 0 | PASS (all 4) |
| zero | ws | p753 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| zero | ws | p753 | 000000 | 57 | 0 | 0 | PASS (all 4) |
| zero | ws | p753 | 010101 | 36 | 0 | 0 | PASS (all 4) |
| zero | ws64 | max1 | 111111 | 4 | 0 | 0 | PASS (all 4) |
| zero | ws64 | max1 | 000000 | 19 | 0 | 0 | PASS (all 4) |
| zero | ws64 | max1 | 010101 | 12 | 0 | 0 | PASS (all 4) |
| zero | ws64 | max3 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| zero | ws64 | max3 | 000000 | 57 | 0 | 0 | PASS (all 4) |
| zero | ws64 | max3 | 010101 | 36 | 0 | 0 | PASS (all 4) |
| zero | ws64 | mean3 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| zero | ws64 | mean3 | 000000 | 57 | 0 | 0 | PASS (all 4) |
| zero | ws64 | mean3 | 010101 | 36 | 0 | 0 | PASS (all 4) |
| zero | ws64 | p753 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| zero | ws64 | p753 | 000000 | 57 | 0 | 0 | PASS (all 4) |
| zero | ws64 | p753 | 010101 | 36 | 0 | 0 | PASS (all 4) |
| blocks | ws | max1 | 111111 | 4 | 0 | 0 | PASS (all 4) |
| blocks | ws | max1 | 000000 | 19 | 1 | 1 | PASS (all 4) |
| blocks | ws | max1 | 010101 | 12 | 1 | 1 | PASS (all 4) |
| blocks | ws | max3 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| blocks | ws | max3 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| blocks | ws | max3 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| blocks | ws | mean3 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| blocks | ws | mean3 | 000000 | 57 | 3 | 3 | PASS (all 4) |
| blocks | ws | mean3 | 010101 | 36 | 3 | 3 | PASS (all 4) |
| blocks | ws | p753 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| blocks | ws | p753 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| blocks | ws | p753 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| blocks | ws64 | max1 | 111111 | 4 | 0 | 0 | PASS (all 4) |
| blocks | ws64 | max1 | 000000 | 19 | 1 | 1 | PASS (all 4) |
| blocks | ws64 | max1 | 010101 | 12 | 1 | 1 | PASS (all 4) |
| blocks | ws64 | max3 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| blocks | ws64 | max3 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| blocks | ws64 | max3 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| blocks | ws64 | mean3 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| blocks | ws64 | mean3 | 000000 | 57 | 3 | 3 | PASS (all 4) |
| blocks | ws64 | mean3 | 010101 | 36 | 3 | 3 | PASS (all 4) |
| blocks | ws64 | p753 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| blocks | ws64 | p753 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| blocks | ws64 | p753 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| ties | ws | max1 | 111111 | 4 | 1 | 1 | PASS (all 4) |
| ties | ws | max1 | 000000 | 19 | 1 | 1 | PASS (all 4) |
| ties | ws | max1 | 010101 | 12 | 1 | 1 | PASS (all 4) |
| ties | ws | max3 | 111111 | 12 | 3 | 3 | PASS (all 4) |
| ties | ws | max3 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| ties | ws | max3 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| ties | ws | mean3 | 111111 | 12 | 2 | 2 | PASS (all 4) |
| ties | ws | mean3 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| ties | ws | mean3 | 010101 | 36 | 3 | 3 | PASS (all 4) |
| ties | ws | p753 | 111111 | 12 | 3 | 3 | PASS (all 4) |
| ties | ws | p753 | 000000 | 57 | 3 | 3 | PASS (all 4) |
| ties | ws | p753 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| ties | ws64 | max1 | 111111 | 4 | 1 | 1 | PASS (all 4) |
| ties | ws64 | max1 | 000000 | 19 | 1 | 1 | PASS (all 4) |
| ties | ws64 | max1 | 010101 | 12 | 1 | 1 | PASS (all 4) |
| ties | ws64 | max3 | 111111 | 12 | 3 | 3 | PASS (all 4) |
| ties | ws64 | max3 | 000000 | 57 | 3 | 3 | PASS (all 4) |
| ties | ws64 | max3 | 010101 | 36 | 3 | 3 | PASS (all 4) |
| ties | ws64 | mean3 | 111111 | 12 | 2 | 2 | PASS (all 4) |
| ties | ws64 | mean3 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| ties | ws64 | mean3 | 010101 | 36 | 3 | 3 | PASS (all 4) |
| ties | ws64 | p753 | 111111 | 12 | 3 | 3 | PASS (all 4) |
| ties | ws64 | p753 | 000000 | 57 | 3 | 3 | PASS (all 4) |
| ties | ws64 | p753 | 010101 | 36 | 3 | 3 | PASS (all 4) |
| dust | ws | max1 | 111111 | 4 | 1 | 1 | PASS (all 4) |
| dust | ws | max1 | 000000 | 19 | 1 | 1 | PASS (all 4) |
| dust | ws | max1 | 010101 | 12 | 1 | 1 | PASS (all 4) |
| dust | ws | max3 | 111111 | 12 | 2 | 2 | PASS (all 4) |
| dust | ws | max3 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| dust | ws | max3 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| dust | ws | mean3 | 111111 | 12 | 2 | 2 | PASS (all 4) |
| dust | ws | mean3 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| dust | ws | mean3 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| dust | ws | p753 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| dust | ws | p753 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| dust | ws | p753 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| dust | ws64 | max1 | 111111 | 4 | 1 | 1 | PASS (all 4) |
| dust | ws64 | max1 | 000000 | 19 | 1 | 1 | PASS (all 4) |
| dust | ws64 | max1 | 010101 | 12 | 1 | 1 | PASS (all 4) |
| dust | ws64 | max3 | 111111 | 12 | 2 | 2 | PASS (all 4) |
| dust | ws64 | max3 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| dust | ws64 | max3 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| dust | ws64 | mean3 | 111111 | 12 | 2 | 2 | PASS (all 4) |
| dust | ws64 | mean3 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| dust | ws64 | mean3 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| dust | ws64 | p753 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| dust | ws64 | p753 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| dust | ws64 | p753 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| heavy | ws | max1 | 111111 | 4 | 0 | 0 | PASS (all 4) |
| heavy | ws | max1 | 000000 | 19 | 1 | 1 | PASS (all 4) |
| heavy | ws | max1 | 010101 | 12 | 1 | 1 | PASS (all 4) |
| heavy | ws | max3 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| heavy | ws | max3 | 000000 | 57 | 3 | 3 | PASS (all 4) |
| heavy | ws | max3 | 010101 | 36 | 3 | 3 | PASS (all 4) |
| heavy | ws | mean3 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| heavy | ws | mean3 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| heavy | ws | mean3 | 010101 | 36 | 3 | 3 | PASS (all 4) |
| heavy | ws | p753 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| heavy | ws | p753 | 000000 | 57 | 0 | 0 | PASS (all 4) |
| heavy | ws | p753 | 010101 | 36 | 1 | 1 | PASS (all 4) |
| heavy | ws64 | max1 | 111111 | 4 | 0 | 0 | PASS (all 4) |
| heavy | ws64 | max1 | 000000 | 19 | 1 | 1 | PASS (all 4) |
| heavy | ws64 | max1 | 010101 | 12 | 1 | 1 | PASS (all 4) |
| heavy | ws64 | max3 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| heavy | ws64 | max3 | 000000 | 57 | 3 | 3 | PASS (all 4) |
| heavy | ws64 | max3 | 010101 | 36 | 3 | 3 | PASS (all 4) |
| heavy | ws64 | mean3 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| heavy | ws64 | mean3 | 000000 | 57 | 3 | 3 | PASS (all 4) |
| heavy | ws64 | mean3 | 010101 | 36 | 3 | 3 | PASS (all 4) |
| heavy | ws64 | p753 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| heavy | ws64 | p753 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| heavy | ws64 | p753 | 010101 | 36 | 3 | 3 | PASS (all 4) |
| rounding | ws | max1 | 111111 | 4 | 1 | 1 | PASS (all 4) |
| rounding | ws | max1 | 000000 | 19 | 1 | 1 | PASS (all 4) |
| rounding | ws | max1 | 010101 | 12 | 1 | 1 | PASS (all 4) |
| rounding | ws | max3 | 111111 | 12 | 1 | 1 | PASS (all 4) |
| rounding | ws | max3 | 000000 | 57 | 1 | 1 | PASS (all 4) |
| rounding | ws | max3 | 010101 | 36 | 1 | 1 | PASS (all 4) |
| rounding | ws | mean3 | 111111 | 12 | 1 | 1 | PASS (all 4) |
| rounding | ws | mean3 | 000000 | 57 | 1 | 1 | PASS (all 4) |
| rounding | ws | mean3 | 010101 | 36 | 1 | 1 | PASS (all 4) |
| rounding | ws | p753 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| rounding | ws | p753 | 000000 | 57 | 1 | 1 | PASS (all 4) |
| rounding | ws | p753 | 010101 | 36 | 0 | 0 | PASS (all 4) |
| rounding | ws64 | max1 | 111111 | 4 | 1 | 1 | PASS (all 4) |
| rounding | ws64 | max1 | 000000 | 19 | 1 | 1 | PASS (all 4) |
| rounding | ws64 | max1 | 010101 | 12 | 1 | 1 | PASS (all 4) |
| rounding | ws64 | max3 | 111111 | 12 | 1 | 1 | PASS (all 4) |
| rounding | ws64 | max3 | 000000 | 57 | 1 | 1 | PASS (all 4) |
| rounding | ws64 | max3 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| rounding | ws64 | mean3 | 111111 | 12 | 1 | 1 | PASS (all 4) |
| rounding | ws64 | mean3 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| rounding | ws64 | mean3 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| rounding | ws64 | p753 | 111111 | 12 | 0 | 0 | PASS (all 4) |
| rounding | ws64 | p753 | 000000 | 57 | 1 | 1 | PASS (all 4) |
| rounding | ws64 | p753 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| real_64x512x512 | ws | max1 | 111111 | 4 | 1 | 1 | PASS (all 4) |
| real_64x512x512 | ws | max1 | 000000 | 19 | 1 | 1 | PASS (all 4) |
| real_64x512x512 | ws | max1 | 010101 | 12 | 1 | 1 | PASS (all 4) |
| real_64x512x512 | ws | max3 | 111111 | 12 | 2 | 2 | PASS (all 4) |
| real_64x512x512 | ws | max3 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| real_64x512x512 | ws | max3 | 010101 | 36 | 1 | 1 | PASS (all 4) |
| real_64x512x512 | ws | mean3 | 111111 | 12 | 2 | 2 | PASS (all 4) |
| real_64x512x512 | ws | mean3 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| real_64x512x512 | ws | mean3 | 010101 | 36 | 1 | 1 | PASS (all 4) |
| real_64x512x512 | ws | p753 | 111111 | 12 | 2 | 2 | PASS (all 4) |
| real_64x512x512 | ws | p753 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| real_64x512x512 | ws | p753 | 010101 | 36 | 2 | 2 | PASS (all 4) |
| real_64x512x512 | ws64 | max1 | 111111 | 4 | 1 | 1 | PASS (all 4) |
| real_64x512x512 | ws64 | max1 | 000000 | 19 | 0 | 0 | PASS (all 4) |
| real_64x512x512 | ws64 | max1 | 010101 | 12 | 0 | 0 | PASS (all 4) |
| real_64x512x512 | ws64 | max3 | 111111 | 12 | 2 | 2 | PASS (all 4) |
| real_64x512x512 | ws64 | max3 | 000000 | 57 | 0 | 0 | PASS (all 4) |
| real_64x512x512 | ws64 | max3 | 010101 | 36 | 1 | 1 | PASS (all 4) |
| real_64x512x512 | ws64 | mean3 | 111111 | 12 | 2 | 2 | PASS (all 4) |
| real_64x512x512 | ws64 | mean3 | 000000 | 57 | 0 | 0 | PASS (all 4) |
| real_64x512x512 | ws64 | mean3 | 010101 | 36 | 1 | 1 | PASS (all 4) |
| real_64x512x512 | ws64 | p753 | 111111 | 12 | 2 | 2 | PASS (all 4) |
| real_64x512x512 | ws64 | p753 | 000000 | 57 | 2 | 2 | PASS (all 4) |
| real_64x512x512 | ws64 | p753 | 010101 | 36 | 2 | 2 | PASS (all 4) |

**Bounds, token, size and initial end-to-end results (verbatim JSON):**

```json
{"case": "determinism/stock_repeat", "files": 19, "dend_padding_differences": []}
{"case": "determinism/stock_reference", "files": 19, "dend_padding_differences": []}
{"case": "ws/exact_fit", "compact_counts": [1, 2115, 33], "files": 4, "dend_padding_differences": []}
{"case": "ws/one_past", "exit": 3, "files": [], "message": "uint32 segmentation overflow at threshold 0: offset=4294967295 max_id=1"}
{"case": "ws/offset_past", "exit": 3, "files": [], "message": "uint32 segmentation overflow at threshold 0: offset=4294967296 max_id=1"}
{"case": "ws/batch_failure", "exit": 3, "files": ["counts_gate_0.data", "dend_gate_0.data", "meta_gate_0.data", "seg_gate_0.data"], "message": "uint32 segmentation overflow at threshold 1: offset=4294967294 max_id=2115"}
{"case": "ws/invalid_token", "tokens": ["--seg-dtype=invalid"], "exit": 2, "message": "Invalid or duplicate --seg-dtype token: --seg-dtype=invalid\n"}
{"case": "ws/invalid_token", "tokens": ["--seg-dtype=uint32", "--seg-dtype=uint64"], "exit": 2, "message": "Invalid or duplicate --seg-dtype token: --seg-dtype=uint64\n"}
{"case": "ws/token_position/8", "files": 4, "dend_padding_differences": []}
{"case": "ws/token_position/9", "files": 4, "dend_padding_differences": []}
{"case": "ws/token_position/10", "files": 4, "dend_padding_differences": []}
{"case": "ws/python_batch_failure", "exit": 3, "callbacks": 0}
{"case": "ws64/exact_fit", "compact_counts": [1, 2115, 33], "files": 4, "dend_padding_differences": []}
{"case": "ws64/one_past", "exit": 3, "files": [], "message": "uint32 segmentation overflow at threshold 0: offset=4294967295 max_id=1"}
{"case": "ws64/offset_past", "exit": 3, "files": [], "message": "uint32 segmentation overflow at threshold 0: offset=4294967296 max_id=1"}
{"case": "ws64/batch_failure", "exit": 3, "files": ["counts_gate_0.data", "dend_gate_0.data", "meta_gate_0.data", "seg_gate_0.data"], "message": "uint32 segmentation overflow at threshold 1: offset=4294967294 max_id=2115"}
{"case": "ws64/invalid_token", "tokens": ["--seg-dtype=invalid"], "exit": 2, "message": "Invalid or duplicate --seg-dtype token: --seg-dtype=invalid\n"}
{"case": "ws64/invalid_token", "tokens": ["--seg-dtype=uint32", "--seg-dtype=uint64"], "exit": 2, "message": "Invalid or duplicate --seg-dtype token: --seg-dtype=uint64\n"}
{"case": "ws64/token_position/8", "files": 4, "dend_padding_differences": []}
{"case": "ws64/token_position/9", "files": 4, "dend_padding_differences": []}
{"case": "ws64/token_position/10", "files": 4, "dend_padding_differences": []}
{"case": "ws64/python_batch_failure", "exit": 3, "callbacks": 0}
{"case": "ws/size/below", "exit": 2, "message": "merge function: max\nthresholds: 0.95 0.05 20 0 merge=[0.2]\nChunk size check passed: 2147483647\nEach dimension must include an interior and two halo voxels\n"}
{"case": "ws/size/at", "exit": 2, "message": "merge function: max\nthresholds: 0.95 0.05 20 0 merge=[0.2]\nInvalid chunk size for ws32: dimension product overflows or exceeds watershed index limit\n"}
{"case": "ws/size/above", "exit": 2, "message": "merge function: max\nthresholds: 0.95 0.05 20 0 merge=[0.2]\nInvalid chunk size for ws32: dimension product overflows or exceeds watershed index limit\n"}
{"case": "ws/size/overflow", "exit": 2, "message": "merge function: max\nthresholds: 0.95 0.05 20 0 merge=[0.2]\nInvalid chunk size for ws32: dimension product overflows or exceeds watershed index limit\n"}
{"case": "ws64/size/below", "exit": 2, "message": "merge function: max\nthresholds: 0.95 0.05 20 0 merge=[0.2]\nChunk size check passed: 9223372036854775807\nEach dimension must include an interior and two halo voxels\n"}
{"case": "ws64/size/at", "exit": 2, "message": "merge function: max\nthresholds: 0.95 0.05 20 0 merge=[0.2]\nInvalid chunk size for ws64: dimension product overflows or exceeds watershed index limit\n"}
{"case": "ws64/size/above", "exit": 2, "message": "merge function: max\nthresholds: 0.95 0.05 20 0 merge=[0.2]\nInvalid chunk size for ws64: dimension product overflows or exceeds watershed index limit\n"}
{"case": "ws64/size/overflow", "exit": 2, "message": "merge function: max\nthresholds: 0.95 0.05 20 0 merge=[0.2]\nInvalid chunk size for ws64: dimension product overflows or exceeds watershed index limit\n"}
{"case": "e2e/stock", "files": 1, "dtype": "uint64", "equal": true}
{"case": "e2e/uint64", "files": 1, "dtype": "uint64", "equal": true}
{"case": "e2e/uint32", "files": 1, "dtype": "uint32", "equal": true}
{"case": "e2e/auto", "files": 1, "dtype": "uint32", "equal": true}
{"case": "e2e/stock_batch", "files": 3, "dtype": "uint64", "equal": true}
{"case": "e2e/uint64_batch", "files": 3, "dtype": "uint64", "equal": true}
{"case": "e2e/uint32_batch", "files": 3, "dtype": "uint32", "equal": true}
{"case": "e2e/auto_batch", "files": 3, "dtype": "uint32", "equal": true}
{"case": "stock_md5", "md5": "bf4c7343dccf86e60d78f772e583492d"}
```

The heavy fixture compact counts are `[1, 2115, 33]` for thresholds `[0.0, 0.99, 0.5]` in both widths. Exact fit uses offset 4294967294 and one output label; one-past/offset-too-large exit 3 with no output artifacts. Batch overflow leaves exactly four complete threshold-0 files, no threshold-1/2 data or `.tmp`, and invokes zero Python callbacks. All six token-placement comparisons passed 4 files each; invalid/duplicate tokens exit 2. Both widths pass the pure high-bit boundary tests; CLI high-bit-minus-one logs its successful predicate before the minimum-dimension rejection. Size, bound and token failures above are expected successful negative tests.


Step 3 also ran explicit `--seg-dtype=uint64` before `max` in both binaries against the exact-fit uint64 reference: 4 files matched per width, no padding differences (`build_mem/explicit_uint64.log`). Step 6 was repeated after the Python ownership correction by importing `ws_gate.Gate` with scratch `build_mem/gate_scratch/final_checks` and calling `end_to_end()`. All eight CLI invocations (stock/uint64/uint32/auto, single/batch) passed, checking 16 HDF5 arrays for identical labels and expected dtype (`build_mem/e2e_final.log`). CLI batch mode uses the callback path.

**Style/static commands actually run:**

```bash
python -m flake8 --max-line-length=100 scripts/run_abiss_volume.py tests/unit/test_abiss_seg_dtype.py
python -m py_compile scripts/run_abiss_volume.py tests/unit/test_abiss_seg_dtype.py
python -m mypy --config-file .github/mypy_changed.ini scripts/run_abiss_volume.py tests/unit/test_abiss_seg_dtype.py
# From the ABISS repository:
python -m black --check --workers 1 --target-version py311 tests/ws_gate.py
python -m isort --check-only --profile black tests/ws_gate.py
python -m flake8 --max-line-length=100 tests/ws_gate.py
python -m py_compile tests/ws_gate.py
bash -n tests/ws_benchmark.sbatch
python tests/ws_gate.py --help
# Both repositories:
git diff --check
```

Flake8, syntax checks, isort, the ABISS Black check, and both diff checks passed. Python-worktree Black CLI runs stalled and was interrupted (exit 130); equivalent `black.format_str` equality checks (line_length=100, target PY311) and `isort.code` equality checks using `pyproject.toml` passed for both changed Python files. The initial cross-repo Black invocation also emitted: `Warning: Python 3.11 cannot parse code formatted for Python 3.12.` Subsequent formatting/checks explicitly selected PY311. Final benchmark-sampler edits were formatted again and passed isort, flake8, py_compile and bash syntax checks. No benchmark or SLURM execution was performed for these checks.

Mypy remains unsuccessful on six pre-existing paths. The baseline extracted by `git show HEAD:scripts/run_abiss_volume.py` was separately checked with the same mypy configuration and had eight errors, including those six. The new output-shape tuple is explicitly three-dimensional, removing two baseline diagnostics and the initially introduced dtype-resolver diagnostic. Verbatim test/type/style failures are recorded here; raw pytest/mypy logs are retained under `build_mem/`:

Initial lifetime regression failure (subsequently corrected and passed):

```text
=================================== FAILURES ===================================
____________________ test_input_released_before_subprocess _____________________

script = <module 'run_abiss_volume' from '/projects/weilab/weidf/lib/pytorch_connectomics/.claude/worktrees/bridge-cse_01Y1t9gvwzh3sWG6hJ5p7qXF/scripts/run_abiss_volume.py'>
tmp_path = PosixPath('/tmp/pytest-of-weidf/pytest-911/test_input_released_before_sub0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x1555162c6d90>

    def test_input_released_before_subprocess(script, tmp_path, monkeypatch):
        kwargs = run_kwargs(tmp_path)
        predictions = kwargs.pop("predictions_czyx")
        reference = weakref.ref(predictions)
        holder = [predictions]
        del predictions
    
        def fake_run(cmd, cwd, check):
            assert reference() is None
            np.zeros(60, dtype=np.uint64).tofile(Path(cwd) / f"seg_{script._ABISS_TAG}.data")
    
        monkeypatch.setattr(script.subprocess, "run", fake_run)
>       script._run_abiss_ws(predictions_czyx=holder.pop(), **kwargs)

tests/unit/test_abiss_seg_dtype.py:135: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
scripts/run_abiss_volume.py:389: in _run_abiss_ws
    subprocess.run(cmd, cwd=str(ws_dir), check=True)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

cmd = ['/fake/ws', '/tmp/pytest-of-weidf/pytest-911/test_input_released_before_sub0/param.txt', '/tmp/pytest-of-weidf/pytest-911/test_input_released_before_sub0/aff.raw', '0.9', '0.1', '10', ...]
cwd = '/tmp/pytest-of-weidf/pytest-911/test_input_released_before_sub0'
check = True

    def fake_run(cmd, cwd, check):
>       assert reference() is None
E       AssertionError: assert array([[[[1., 1., 1.],\n         [1., 1., 1.],\n         [1., 1., 1.],\n         [1., 1., 1.]],\n\n        [[1., 1., 1.],\n ...., 1.]],\n\n        [[1., 1., 1.],\n         [1., 1., 1.],\n         [1., 1., 1.],\n         [1., 1., 1.]]]], dtype=float32) is None
E        +  where array([[[[1., 1., 1.],\n         [1., 1., 1.],\n         [1., 1., 1.],\n         [1., 1., 1.]],\n\n        [[1., 1., 1.],\n ...., 1.]],\n\n        [[1., 1., 1.],\n         [1., 1., 1.],\n         [1., 1., 1.],\n         [1., 1., 1.]]]], dtype=float32) = <weakref at 0x155442ceb240; to 'numpy.ndarray' at 0x155516491bf0>()

tests/unit/test_abiss_seg_dtype.py:131: AssertionError
```

Initial mypy output (new dtype-shape diagnostic subsequently removed):

```text
scripts/run_abiss_volume.py:165: error: No overload variant of "__setitem__" of "list" matches argument types "int", "int"  [call-overload]
scripts/run_abiss_volume.py:165: note: Possible overload variants:
scripts/run_abiss_volume.py:165: note:     def __setitem__(self, SupportsIndex, slice[Any, Any, Any], /) -> None
scripts/run_abiss_volume.py:165: note:     def __setitem__(self, slice[Any, Any, Any], Iterable[slice[Any, Any, Any]], /) -> None
scripts/run_abiss_volume.py:225: error: Need type annotation for "mm"  [var-annotated]
scripts/run_abiss_volume.py:256: error: Incompatible types in assignment (expression has type "tuple[int, ...]", variable has type "tuple[int, int, int] | None")  [assignment]
scripts/run_abiss_volume.py:336: error: Argument 3 to "_resolve_seg_dtype" has incompatible type "tuple[int, ...]"; expected "tuple[int, int, int]"  [arg-type]
scripts/run_abiss_volume.py:382: error: Item "None" of "list[float] | None" has no attribute "__iter__" (not iterable)  [union-attr]
scripts/run_abiss_volume.py:401: error: Argument 1 to "enumerate" has incompatible type "list[float] | None"; expected "Iterable[float]"  [arg-type]
scripts/run_abiss_volume.py:409: error: Argument 2 to "_read_segmentation_xyz" has incompatible type "tuple[int, ...]"; expected "tuple[int, int, int]"  [arg-type]
scripts/run_abiss_volume.py:424: error: Argument 2 to "_read_segmentation_xyz" has incompatible type "tuple[int, ...]"; expected "tuple[int, int, int]"  [arg-type]
scripts/run_abiss_volume.py:664: error: Argument 2 to "_write_array" has incompatible type "ndarray[Any, Any] | dict[float, ndarray[Any, Any]]"; expected "ndarray[Any, Any]"  [arg-type]
Found 9 errors in 1 file (checked 2 source files)
```

Final mypy output (six baseline diagnostics remain; this check did NOT pass):

```text
scripts/run_abiss_volume.py:165: error: No overload variant of "__setitem__" of "list" matches argument types "int", "int"  [call-overload]
scripts/run_abiss_volume.py:165: note: Possible overload variants:
scripts/run_abiss_volume.py:165: note:     def __setitem__(self, SupportsIndex, slice[Any, Any, Any], /) -> None
scripts/run_abiss_volume.py:165: note:     def __setitem__(self, slice[Any, Any, Any], Iterable[slice[Any, Any, Any]], /) -> None
scripts/run_abiss_volume.py:225: error: Need type annotation for "mm"  [var-annotated]
scripts/run_abiss_volume.py:256: error: Incompatible types in assignment (expression has type "tuple[int, ...]", variable has type "tuple[int, int, int] | None")  [assignment]
scripts/run_abiss_volume.py:388: error: Item "None" of "list[float] | None" has no attribute "__iter__" (not iterable)  [union-attr]
scripts/run_abiss_volume.py:407: error: Argument 1 to "enumerate" has incompatible type "list[float] | None"; expected "Iterable[float]"  [arg-type]
scripts/run_abiss_volume.py:670: error: Argument 2 to "_write_array" has incompatible type "ndarray[Any, Any] | dict[float, ndarray[Any, Any]]"; expected "ndarray[Any, Any]"  [arg-type]
Found 6 errors in 1 file (checked 2 source files)
```

Baseline mypy output: `lib/abiss/build_mem/mypy_baseline.log` (8 errors).

Initial flake8 output, corrected before final checks:

```text
tests/ws_gate.py:181:101: E501 line too long (104 > 100 characters)
tests/ws_gate.py:900:101: E501 line too long (108 > 100 characters)
tests/ws_gate.py:901:101: E501 line too long (129 > 100 characters)
```

The first artifact-generation section check incorrectly matched shell comments inside fenced code as headings; it was corrected to ignore fenced blocks. Its initial diagnostic was:

```text
Traceback (most recent call last):
  File "<stdin>", line 53, in <module>
AssertionError
```


**Step 5, prepared but NOT run:** The driver measures stock/new-uint64/new-uint32 for ws/ws64, single/five-threshold, bare binary and whole Python job. It uses 128x1024x1024 and 128x3072x3072 centered crops, the latter 1,207,959,552 interior and 1,228,431,880 padded voxels (below 2^31). The supplied source's shape is `(3,145,4290,3345)`, so a 256-slice crop would not fit. Samples stream to JSONL to bound monitor memory. Each measured step emits command spec, stdout, time-v, markers, process/cgroup samples, and JSON result; the job emits a B/padded-voxel table, acceptance ratios and labelled per-process sacct output. The whole-Python stock baseline is extracted from the requested Python run-start commit, preserving its pre-change memory behavior. Aggregate scope is explicitly the measured srun step, including the lightweight measurement helper.

Exact submission command for the coordinator (not executed):

```bash
sbatch /projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss/tests/ws_benchmark.sbatch \
  /projects/weilab/weidf/lib/pytorch_connectomics/.claude/worktrees/bridge-cse_01Y1t9gvwzh3sWG6hJ5p7qXF
```

Results will be under `lib/abiss/build_mem/gate_scratch/benchmark_<jobid>/`; submission stdout goes to `gate_scratch/benchmark_<jobid>.log`. Missing cgroup access yields an explicit unverified metric. RSS/runtime acceptance results remain pending this run.

## Review Focus

- Queue cursor reset equivalence and signed neighbour arithmetic.
- LUT equivalence to the former voxel remap, especially graph reconstruction and compact-ID overflow checks.
- Fortran-order streamed interior and transformed face ordering; only main segmentation may change dtype.
- MAX first-observation/NaN/signed-zero semantics and ordered MEAN accumulation under the matched compiler flags.
- The corrected Python holder transfer, verified by weakref rather than inferred from `del`.
- Deferred benchmark cgroup visibility and per-phase attribution on the actual SLURM installation.

## Risks and Unknowns

- Step 5 has not run, as explicitly requested. Binary RSS reduction, >=30% improvement for five-threshold ws64 at >=1 Gvox, <=1.15x runtime, and aggregate job memory remain unverified. No SLURM jobs were submitted.
- SLURM resource availability, partition limits, actual cgroup mount/read permissions, and end-to-end benchmark execution have not been tested locally. The submission script requests short/240G/8 CPUs/12 hours; the coordinator can override site resource options at submission.
- Only the permitted one real 64x512x512 crop was checked locally. Larger correctness crops and >2^32-voxel execution were not run. Wide queue storage was forced on small inputs in C++ tests.
- PERCENTILE still retains every boundary observation; worst-case queue/LUT sizes remain O(P), and allocator retention can affect measured RSS.
- uint32 consumers must opt in explicitly; existing tutorial scripts reading uint64 were intentionally untouched.
- Existing mypy errors remain on pre-existing code paths; exact diagnostics are recorded above. The relative-script wrapper pytest skip is due to this worktree having no `lib/abiss/build/ws`; explicit-path CLI tests cover the driver with the main checkout's binaries.

## Changes Since Previous Code Version

Initial implementation.
