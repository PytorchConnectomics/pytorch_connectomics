# Code v0

## Overview

Implemented the terminal plan_v2 correctness fix in the isolated `work2/abiss` clone. A new
pre-pass resolves nucleus identity once per chunkmap-remapped supervoxel over the complete volume,
and atomic extraction uses that global state/id while retaining local additive counts. The
no-`NUC_TABLE` path remains byte-identical to baseline. No commit was created.

V0 confirms the defect in the correct agglomeration ID space: the representative
`72198606672811349` resolves globally to CONFLICT while archived atomic records resolve it as
PROPER(275), PROPER(319), PROPER(373), CONFLICT, or NONE depending on the chunk. V3 reproduces the
70/30 adversarial case and demonstrates that the table path removes that partition dependence.

## What Changed

- Added `build_nucleus_table.py`, which reads watershed and nucleus volumes on the configured
  atomic grid, applies the watershed chunkmap, accumulates `(sid, nucleus, count)` partials,
  globally resolves dominance with the existing threshold/ratio rules, and emits sorted 29-byte
  `nuc_wire_t` records.
- Extracted the existing nucleus cutout/validation helpers so the production cut path and table
  pre-pass use one implementation.
- Added optional table loading to `NucExtractor`. PROPER/CONFLICT state and identity come from the
  table; counts remain local so `nuc_join` remains additive. Globally sub-threshold NONE records
  retain the existing zero-count/zero-total representation.
- Plumbed `NUC_TABLE` from param JSON only when `NUC_PATH` is configured. A stale `NUC_TABLE`
  without `NUC_PATH` is inert, including when the named file does not exist.
- Copied the required untracked test harness, corrected its v2 baseline/current-only assertions,
  and replaced the hierarchy fixture with the adversarial 70/30 split-supervoxel case.

## Implementation Details

`cut_chunk_agg.py` reads the post-watershed volume into `seg.raw`. At runtime the sampled
watershed IDs were already chunk-prefixed (`72057594037927937..72057594037971691`).
`atomic_chunk_ME.cpp` then loads `chunkmap.data`, and `NucExtractor::output` applies exactly one
lookup before emitting records. The pre-pass mirrors this by loading all atomic
`chunkmap_0_x_y_z.data.zst` files, rejecting disagreeing duplicate keys, and applying one lookup
to every nonzero watershed voxel before histogram reduction. The crop map contained 98,437 keys
in the same high-ID domain; its target IDs are the IDs consumed by agglomeration. A focused test
also proves raw SID 100 is emitted under target SID 72057594037928036.

The table file uses the existing packed wire ABI: `(uint64 sid, uint8 state, uint32 id,
uint64 count, uint64 total)`. `NucExtractor` validates record states, strict SID ordering,
truncation, and lookup completeness. Missing table entries abort rather than silently restoring
partition-dependent local resolution.

For table-selected PROPER records, each child emits its local count for the selected global ID and
its local tagged total. CONFLICT records emit local total. NONE preserves the established
`count=total=0` algebra, which is required for single-chunk table/local byte equivalence. The
unchanged reducers therefore sum each voxel once; global counts are never repeated per child.

## Files Changed

| File | Purpose |
|---|---|
| `scripts/build_nucleus_table.py` | New remapped-ID accumulation, global reduction, and wire-file emitter |
| `scripts/nucleus_utils.py` | Canonical nucleus transform and input-validation helpers |
| `scripts/cut_chunk_agg.py` | Uses the shared nucleus helpers |
| `scripts/set_env.py` | Exports `NUC_TABLE` only inside the existing `NUC_PATH` configuration block |
| `src/seg/NucExtractor.hpp` | Loads/validates the optional table and emits globally selected state/id with local counts |
| `src/seg/atomic_chunk_ME.cpp` | Passes the inert-without-mask `NUC_TABLE` path into the extractor |
| `work/test/` | Required copied untracked harness and its existing fixtures |
| `work/test/run_v2_invariance.sh` | Uses baseline 312bf54, requires zero current-only files, and compares both table-inert modes |
| `work/test/make_hierarchy_fixture.py` | Constructs local 100/0 and 40/60 children plus the 70/30 monolith |
| `work/test/run_hierarchy.sh` | Asserts local conflict and table-enabled child/parent/monolith equivalence |
| `work/test/test_nuc_extractor.cpp` | Adds single-chunk table/local byte-equivalence coverage |
| `work/test/test_input_contracts.py` | Covers conditional `NUC_TABLE` export and no-mask inertness |
| `work/test/test_build_nucleus_table.py` | Guards chunkmap target-ID emission and wire contents |
| `work/v0/` | Generated worst3 table, build log, and extracted atomic V0 records |

## Git Baseline

run_start_ref: 312bf54183faa94be1aff8ae246fcc211808f0bb
current_head: 312bf54183faa94be1aff8ae246fcc211808f0bb

## Verification

**V0 -- PASS.** Ran:

```text
HDF5_USE_FILE_LOCKING=FALSE python scripts/build_nucleus_table.py \
  /projects/weilab/weidf/lib/pytorch_connectomics/dev/zebrafinch/nuc_z1_y7_x6/worst3/param \
  work/v0/worst3_nucleus_table.data | tee work/v0/build_nucleus_table.log
```

Actual result: 100/100 chunks processed; 139,294 records and 4,039,526 bytes emitted;
NONE=5,401, PROPER=133,892, CONFLICT=1. The sole CONFLICT is SID
`72198606672811349`, total 28,576,381. Then extracted the 100 archived
`ongoing_nuclei_labels_0_*.data` files and ran an inline NumPy comparison against the table.
Among 116,484 table SIDs present in those ongoing records, 10,827 have at least one local
state/id different from the global result; one flips from at least one local PROPER record to
global CONFLICT. That one is the representative above, whose local states/ids are NONE,
PROPER(275), PROPER(319), PROPER(373), and CONFLICT.

**V1 -- PASS.** Ran the unchanged algebra unit:

```text
$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++ -std=gnu++20 \
  -I$CONDA_PREFIX/include work/test/test_nuc_algebra.cpp -o build/test_nuc_algebra && \
  build/test_nuc_algebra
```

Actual result: `test_nuc_algebra: PASS`; the overflow diagnostic is the test's expected child
SIGABRT. Ran the extractor equivalence unit with the same compiler command shape; actual result:
`test_nuc_extractor: PASS (single-chunk table/local identical)`. Ran
`HDF5_USE_FILE_LOCKING=FALSE python work/test/test_build_nucleus_table.py`; actual result:
`PASS (raw sid 100 -> agglomeration sid 72057594037928036)`. Ran
`HDF5_USE_FILE_LOCKING=FALSE python work/test/test_input_contracts.py`; actual result:
`test_input_contracts: PASS`, including conditional table export.

Two preliminary standalone C++ compile attempts did not run tests: the first omitted the conda
Boost include path, and the first extractor revision also hit a vexing-parse linker error. Both
were corrected before the passing commands above.

**V2 -- PASS.** Final-source command:

```text
HDF5_USE_FILE_LOCKING=FALSE ABISS_BUILD_JOBS=4 bash work/test/run_v2_invariance.sh
```

Actual result: baseline/current builds PASS. With `NUC_TABLE` unset:
`identical=38 differing=0 missing=0`, current-only files 0. With `NUC_TABLE` naming a nonexistent
file but no `NUC_PATH`: `identical=38 differing=0 missing=0`, current-only files 0. Both emitted
zero nucleus log lines; stale invalid nucleus knobs were also inert. Final line:
`run_v2_invariance: PASS`. CMake printed only its existing optional-absl discovery warning.

**V3 -- PASS.** Final-source command:

```text
cmake --build build --parallel 4 --target acme && \
HDF5_USE_FILE_LOCKING=FALSE bash work/test/run_hierarchy.sh
```

Actual result: local child A PROPER(1), child B PROPER(2), parent CONFLICT; table-enabled children
both PROPER(1), and child+parent exactly equals the monolithic PROPER(1) record with
count/total 14,000/20,000. Final line: `run_hierarchy: PASS`.

**V4 -- NOT RUN.** The crop control/treatment A/B requires the external 12-hour SLURM arm setup
and writes configuration/results outside the isolated clone. This code stage did not submit it.

**Additional checks.** `cmake --build build --parallel 4 --target acme agg meme split_remap assort
reduce_chunk` passed. `python -m py_compile` for changed Python files and `bash -n` for changed
shell scripts passed. `git diff --check` passed. Focused flake8 over new Python/test files passed.
A broader flake8 command including touched legacy scripts failed on their pre-existing E302/E501,
bare-except, and comment-spacing findings; no unrelated formatting cleanup was applied.

## Review Focus

- Confirm the pre-pass and `NucExtractor::output` both perform the same one-level chunkmap lookup,
  so table keys are target IDs rather than the wrong raw watershed namespace.
- Confirm per-child PROPER/CONFLICT counts stay local and additive, while state/id are global; no
  global totals are duplicated across chunks.
- Confirm the no-table branch is structurally the old resolution logic and V2 covers both unset
  and stale-table/no-mask cases.
- Inspect strict table validation and the decision to fail closed on a missing SID.

## Risks and Unknowns

- V4 is unmeasured. Per the task-owner decision and V0 evidence, this correctness fix is not
  expected by itself to dissolve the reported 291M-voxel blob; ABISS parameter tuning follows.
- A global pre-pass adds one complete watershed+nucleus read and serializes agglomeration startup.
  The crop emitted a 4.0 MB table, but whole-volume runtime and memory remain unmeasured.
- The table cannot split a watershed supervoxel that intrinsically spans multiple nuclei. The crop
  representative is globally CONFLICT, but other globally dominant mixed supervoxels still retain
  their minority identity loss by design.
- `NUC_TABLE` must name a worker-visible filesystem path. This change provides the builder and
  configuration plumbing; deployment must stage the completed file before atomic agg workers run.
- V0's 10,827 count compares the table to archived post-atomic ongoing records at matching SIDs,
  exactly as planned, but those records can already represent in-chunk merges rather than pristine
  extractor-only supervoxels.

## Changes Since Previous Code Version

Initial implementation.
