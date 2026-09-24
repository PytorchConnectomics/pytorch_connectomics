# Code v1

## Overview

Accepted and fixed all four review_v0 findings. The C0–C8 memory implementation from code_v0 remains in place; this revision fixes benchmark threshold preparation, restores stdout contracts, handles runtime exceptions, and restores the internal-width rationale. Step 5 remains for the coordinator; no SLURM jobs or benchmark runs were performed in this stage.

## What Changed

- Compute benchmark percentiles in float32 and reject nonfinite or unordered thresholds before launching measurements.
- Restore single-threshold counts, agglomeration/writing timing, multi-threshold banner, and indexed summaries with threshold values and elapsed seconds.
- Catch standard exceptions at the binary entry point, print `ws runtime error: ...`, and exit 4.
- Explain watershed flag bits, high_bit ceilings, internal-width memory cost, and independent output width.
- Extend the local gate with threshold regression/rejection tests, stdout contract checks, and output rename failure tests for both binary widths.

## Implementation Details

`benchmark_thresholds(crop)` converts to float32 before computing the 20th and 94th percentiles together, then requires finite `low < high`. The benchmark uses this helper before any measured invocation. Local regression coverage uses 70,000 float16 values and rejects constant, NaN, and infinite inputs. The actual centered 128×1024×1024 crop was checked separately through this same helper without running a binary in that check.

The common threshold loop retains streamed LUT output. Its stdout again contains the original parser-compatible strings. CPU timings use `clock()` as before; agglomeration measures counts copying and merge computation, writing measures main segmentation, faces, and metadata, and multi-threshold elapsed time covers the whole iteration. Counts/timing lines are also emitted in multi-threshold mode.

The function-level `std::exception` handler documents exit 4 for runtime failures, including output I/O. The writer's existing temporary-file cleanup remains active during stack unwinding. Local tests force rename failure by making the destination a directory, then require exit 4, the diagnostic, and no temporary segmentation. Earlier sidecar files can remain after an output failure; this is not an all-artifacts transaction.

## Files Changed

`lib/abiss/` below refers to the separate repository at `/projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss`. Only its `atomic_chunk.cpp` and `ws_gate.py` changed in this revision; other listed implementation files are cumulative from code_v0.

| File | Purpose |
|---|---|
| `lib/abiss/CMakeLists.txt` | Register C++ watershed contract tests (v0). |
| `lib/abiss/src/ws/atomic_chunk.cpp` | Restore stdout/timing and width rationale; catch runtime exceptions (v1); size/dtype/LUT orchestration (v0). |
| `lib/abiss/src/ws/basic_watershed.hpp` | Width-selected reusable BFS storage (v0). |
| `lib/abiss/src/ws/region_graph.hpp` | Ordered MAX/MEAN accumulators and memory markers (v0). |
| `lib/abiss/src/ws/agglomeration.hpp` | Immutable segmentation with per-threshold merge LUT (v0). |
| `lib/abiss/src/ws/utils.hpp` | Size guards, markers, streamed output and transformed faces (v0). |
| `lib/abiss/tests/test_ws_bfs.cpp` | Width, size, accumulator, ABI contract tests (v0). |
| `lib/abiss/tests/ws_gate.py` | Float32 benchmark threshold helper and regression tests for thresholds, stdout, and I/O failures (v1); correctness/benchmark driver (v0). |
| `lib/abiss/tests/ws_benchmark.sbatch` | Deferred compute-node benchmark submission script (v0). |
| `scripts/run_abiss_volume.py` | Dtype-aware output and input ownership transfer (v0). |
| `tests/unit/test_abiss_seg_dtype.py` | Python dtype and ownership contracts (v0). |
| `.agent/decoding/abiss_memory/artifacts/code_v1.md` | This revision's implementation and validation report. |
| `lib/abiss/build_mem/` | Generated binaries, logs, and gate results; only build destination used in v1. |
| `lib/abiss/build_ref/` | Existing archived reference build; not rebuilt in v1. |

## Git Baseline

run_start_ref: 3cbdf39c06bf42c4b7cf739e1208751403a18bc7
current_head: 3cbdf39c06bf42c4b7cf739e1208751403a18bc7
abiss_start_ref: 92abc91f496304ecfe3d5c593e7713e8abd98011
abiss_current_head: 92abc91f496304ecfe3d5c593e7713e8abd98011

Both HEADs were read and match the run baseline. No commits, branches, stashes, checkouts, staging, or stock builds were performed. Existing dirty implementation files and untracked build/cache/run directories were preserved.

## Verification

All execution used `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc`, `OMP_NUM_THREADS=4`, and `TOKIO_WORKER_THREADS=4`. Define `A=/projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss` and `W=/projects/weilab/weidf/lib/pytorch_connectomics/.claude/worktrees/bridge-cse_01Y1t9gvwzh3sWG6hJ5p7qXF`.

Commands actually run:

```bash
cmake --build "$A/build_mem" --target ws ws64 test_ws_bfs test_nuc_algebra test_nuc_extractor agg match_chunks -j 2
ctest --test-dir "$A/build_mem" --output-on-failure
pytest tests/unit/test_abiss_seg_dtype.py tests/unit/test_decode_abiss_wrapper.py tests/unit/test_abiss_edge_storage.py tests/unit/test_abiss_s3_chunked.py -q --basetemp="$A/build_mem/pytest_v1"
python "$A/tests/ws_gate.py" --mode local --python-repo "$W"
```

Build succeeded; CTest **5 passed, 0 failed**. Focused pytest **34 passed, 1 skipped**, with 3 existing SWIG deprecation warnings. The skip requires the nonexistent worktree-relative stock binary; the gate uses explicit binary paths. Logs: `build_mem/build_v1.log`, `ctest_v1.log`, `pytest_v1.log`, `local_gate_v1.log`.

Full local gate **PASS (exit 0): 168 matrix cases passed, 0 failed** (144 synthetic + 24 real), each checked against stock and archived reference in uint64 and uint32 output, totaling **19,600 file-pair comparisons**. No stock/reference mismatch was recorded. Both determinism checks, size/token/overflow negative tests, all eight end-to-end CLI cases, and the added threshold and two I/O failure checks completed successfully. `gate_scratch/local_results.json` contains 214 result records, including build metadata and checksum records, rather than 214 independent matrix cases. Per-case stdout remains under `gate_scratch/logs/`; the complete rerun log is `build_mem/local_gate_v1.log`.

Closing stock checksums: `build/ws` **bf4c7343dccf86e60d78f772e583492d**, `build64/ws64` **ab83e053e14c570a1903a59a9236b797**. The required stock ws checksum also passed both the gate's opening and closing assertions.

F1 crop-only command body (executed via Python with `$A/tests` on `sys.path`; no binaries or measurement driver invoked):

```python
from ws_gate import SOURCE, benchmark_thresholds, h5py, np
shape = (128, 1024, 1024)
with h5py.File(SOURCE) as f:
    dataset = f['main']
    starts = [(a-b)//2 for a,b in zip(dataset.shape[1:], shape)]
    crop = dataset[:3, *(slice(start,start+size) for start,size in zip(starts,shape))]
high, low = benchmark_thresholds(crop)
assert np.isfinite(high) and np.isfinite(low) and low < high
print(f'crop={crop.shape} dtype={crop.dtype} high={high} low={low}')
```

Output, retained in `build_mem/thresholds_v1.log`:

```text
crop=(3, 128, 1024, 1024) dtype=float16 high=0.7255859375 low=0.1888427734375
```

F2 actual new `ws` stdout excerpts from `gate_scratch/logs/zero/ws/max1/111111/new.log`:

```text
finished agglomeration in 0.000237 seconds
num of sv:0
size of rg:0
finished writing in 0.000381 seconds
```

Multi-threshold excerpt from `gate_scratch/logs/zero/ws/max3/111111/new.log`:

```text
Multi-threshold mode: 3 merge thresholds
finished agglomeration in 0.000247 seconds
num of sv:0
size of rg:0
finished writing in 0.000495 seconds
merge threshold 0 (0.1): sv=0 rg=0 in 0.000888 seconds
merge threshold 1 (0.4): sv=0 rg=0 in 0.001164 seconds
merge threshold 2 (0.8): sv=0 rg=0 in 0.000924 seconds
```

The local gate checks counts and timing regexes on every successful modified-binary run and checks every indexed threshold summary in multi mode. Both injected I/O failures returned 4 with `ws runtime error: Cannot rename segmentation: seg_gate.data` and removed the temporary file.

Additional checks actually run: Black formatting with `--workers 1 --target-version py311`; flake8 `--max-line-length=100`, isort `--check-only --profile black`, and py_compile for `tests/ws_gate.py`; `git diff --check` in both repositories. All completed successfully. The invalid-infinite threshold rejection test emits NumPy's expected interpolation RuntimeWarning. The gate also reports an unwritable default Matplotlib cache and uses a temporary cache successfully. No mypy rerun was made: the unchanged Python driver retains the six baseline diagnostics documented in code_v0.

## Review Focus

- Benchmark threshold helper and its pre-invocation validation, including the real crop evidence.
- Restored stdout compatibility for `ws_sizing.py` and multi-threshold consumers.
- Defined runtime failure exit and temporary segmentation cleanup.
- Full local gate equivalence against unchanged stock/reference builds.

## Risks and Unknowns

- Step 5 memory/runtime acceptance remains unmeasured. The coordinator's earlier job 3039644 was cancelled after F1 and produced no results; this stage submits no jobs and runs no benchmark.
- Float32 percentile preparation allocates a converted crop and percentile working storage. This happens before measured runs; the larger benchmark crop and actual SLURM/cgroup behavior remain untested.
- Output failures can leave completed sidecars; exit 4 signals failure to callers. No disk-full simulation was performed; deterministic rename failure exercises the writer exception/cleanup path.
- Existing code_v0 limitations remain: PERCENTILE retains boundary observations, worst-case queue/LUT storage is O(P), allocator retention and aggregate memory require measurement, and >2^32-voxel execution was not tested.

## Changes Since Previous Code Version

1. **F1 accepted/fixed:** float32 percentile helper; finite ordered-threshold guard before benchmark invocation; local regression/rejection checks; real 128×1024×1024 crop yields finite high/low values above. No benchmark executed.
2. **F2 accepted/fixed:** original count strings, timing strings, multi banner, and threshold-value/elapsed summary restored. Required single/multi stdout excerpts above came from new runs; local gate checks the contract automatically.
3. **F3 accepted/fixed:** standard exceptions now produce a clear diagnostic and documented exit 4. Both widths pass deterministic output rename failure tests, including temporary cleanup.
4. **F4 accepted/fixed:** rationale explains packed flag bits, high_bit ceilings, internal-width memory cost, independent main output dtype, and unchanged uint64 boundary/graph IDs.
