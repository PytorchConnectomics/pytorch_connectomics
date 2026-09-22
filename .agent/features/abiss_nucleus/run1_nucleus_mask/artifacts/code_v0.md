# Code v0
## Overview

Implemented plan_v6 in the isolated `work/abiss` clone. ABISS now accepts an optional
`NUC_PATH` uint32 instance volume, extracts fixed-width dominant-nucleus records, carries
them through agglomeration and the chunk hierarchy, and applies the affinity-independent
Invariant D veto. No dependency was added and no commit was created.

## What Changed

- Added the packed nucleus record/wire ABI, exact rational dominance, checked addition,
  associative record join, and the complete three-clause merge predicate.
- Added nullable `NucExtractor` collection with deterministic dominant-id resolution,
  `min_tagged`, and the four planned counters.
- Added `nuc.raw` mapping/validation, agglomeration loading/veto/propagation/logging, and
  unconditional nucleus-file lifecycle.
- Added collision-safe reducer/matcher propagation, OVERLAP=2 veto feedback, and all
  Python/shell driver plumbing.
- Documented the exact guarantee, limitations, counters, tripwires, and perinuclear-shell
  caveat.
- Added executable algebra, extraction, input-contract, hierarchy, cut-chain, and
  two-child fixtures.

## Implementation Details

`nuc_wire_t` is packed as 29 bytes:
`(uint64 sid, uint8 state, uint32 id, uint64 count, uint64 total)`, with static assertions
for every offset. `nuc_is_dominant` uses `__uint128_t`; `nuc_add` aborts before uint64
overflow. NONE is the `(count,total) == (0,0)` identity, equal PROPER ids add, and
different ids or conflict-containing records join to CONFLICT.

Extraction builds a sparse id histogram per supervoxel. Sub-floor evidence becomes NONE;
otherwise the deterministic argmax is PROPER when it meets the parsed rational dominance
threshold, or CONFLICT. The extractor always creates its output file.

Agglomeration evaluates `nuc_can_merge` immediately after the frozen-edge block, without
an affinity gate. Refused edges go to `nuc_cuts.data`. Permitted merges recheck the same
predicate as a tripwire before `nuc_join` propagation. Reducer and matcher collisions
legitimately join to CONFLICT and are counted rather than aborted.

Fixture layouts were confirmed against production definitions: nucleus wire 29 bytes,
region graph 28 bytes, remap/size records 16 bytes, matching entries 32 bytes, boundary
ids 8 bytes, and veto pairs 16 bytes. The hierarchy fixture asserts the packed NumPy
sizes for the first five layouts.

## Files Changed

| File | Purpose |
|---|---|
| `src/seg/Types.h` | Nucleus ABI, exact algebra, parsing, and merge predicate |
| `src/seg/NucExtractor.hpp` | Sparse nullable extractor, resolution, output, and counters |
| `src/seg/atomic_chunk_ME.cpp` | Map/validate `nuc.raw`, traverse, and serialize |
| `src/agg/mean_aggl.cpp` | Load, veto, log, propagate, and manage nucleus files |
| `src/seg/reduce_chunk.cpp` | Remap/join nucleus records in reduction |
| `src/seg/match_chunks.cpp` | Canonicalize/join nucleus records in matching |
| `scripts/set_env.py` | Export optional `NUC_PATH` |
| `scripts/cut_chunk_agg.py` | Load and validate nucleus cutouts |
| `scripts/merge_chunks_me.py` | Merge child nucleus streams |
| `scripts/merge_chunks_overlap.py` | Merge neighbour nucleus streams |
| `scripts/atomic_chunk_me.sh` | Route atomic nucleus artifacts |
| `scripts/composite_chunk_me.sh` | Route composite nucleus artifacts |
| `scripts/overlap_chunk_me.sh` | Propagate payloads and nucleus cuts |
| `README.md` | Contract, tuning, diagnostics, limitations, and caveat |
| `work/test/make_fixture.py` | Fixed-seed atomic fixtures |
| `work/test/run_atomic_fixture.sh` | Atomic `acme`/`agg` sequence |
| `work/test/test_nuc_algebra.cpp` | Algebra, Closure, overflow, parser, and predicate tests |
| `work/test/test_nuc_extractor.cpp` | PROPER/CONFLICT/NONE resolution fixture |
| `work/test/test_input_contracts.py` | Dtype/range/shape tests |
| `work/test/test_hierarchy_binaries.py` | Reducer/matcher collision fixtures |
| `work/test/t3_overlap_fallback.sh` | Exact overlap-driver fallback block |
| `work/test/test_cut_chain.py` | Real-cut merge/match veto chain |
| `work/test/make_hierarchy_fixture.py` | Shared-face child fixture |
| `work/test/run_hierarchy.sh` | Mandatory two-child hierarchy test |

## Git Baseline

run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f
current_head: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f

## Verification

All V1-V10 checks ran. V1-V8 and V10 passed. V9 passed using the plan-authorized T3
fallback; its literal real-driver attempt failed before the extracted block. T4 passed.
The B4 abort was not executed because B3 rejects the same pairs first; V8 executed the
shared predicate and inspected both call sites.

**V1 — PASS.** The first unconfigured attempt failed before compilation:

```text
CMake Error at /usr/share/cmake/Modules/FindPackageHandleStandardArgs.cmake:230 (message):
  Could NOT find Boost (missing: Boost_INCLUDE_DIR iostreams system)
```

After selecting the repository's existing `pytc` conda toolchain, the clean build and the
final literal command completed:

```text
[ 94%] Built target agg_nonoverlap
[ 94%] Built target agg_extra
[ 94%] Built target agg
[100%] Built target agg_overlap
[100%] Built target mecs
```

Only the existing optional `abslConfig.cmake` configure warning appeared. Operational
fixtures used `-DEXTRACT_SIZE=ON`, matching the shell drivers' size payloads.

**V2 — PASS.** The exact-SHA baseline export and modified binaries reported:

```text
V2 cmp: 35 baseline files byte-identical
V2 modified-only empty files: done_nuc.data, nuc_cuts.data, ongoing_nuc.data
V2 set_env: no NUC_ variable exported
V2 cut_chunk_agg: success; nuc.raw absent without NUC_PATH
```

An earlier invalid setup omitted shell-provided `WS_PATH` and raised `KeyError: 'WS_PATH'`;
it was not counted. V2 is a synthetic fixture, not a Seuron provenance rerun.

**V3 — PASS.**

```text
V3/V4 none: remaps=1 nuc_cuts=0
V3/V4 different: remaps=0 nuc_cuts=1
```

The cut record contained `{100, 200}` at high affinity.

**V4 — PASS.**

```text
V3/V4 same: remaps=1 nuc_cuts=0
V3/V4 one: remaps=1 nuc_cuts=0
V3/V4 max: remaps=0 nuc_cuts=1
```

The max case used nucleus id `0xFFFFFFFF` against id 1.

**V5 — PASS.**

```text
nuc: voxel count overflow: 18446744073709551615 + 1
test_nuc_algebra: PASS
```

The SIGABRT was the expected child-process death test. The same binary checked full-record
pair/triple algebra, Closure, J1, the `2^53 + 1` comparison, and invalid ratio values.

**V6 — PASS.**

```text
nuc: conflict_sv 1
nuc: minority_sv 1
nuc: subfloor_sv 1
nuc: subfloor_voxels 10
test_nuc_extractor: PASS
```

It asserted PROPER `(5000,100)`, CONFLICT `(500,400)`, and ten voxels -> NONE with
`total == 0`.

**V7 — PASS.**

```text
V7 uint16: accepted; nuc.raw=2048 bytes uint32
V7 too_large: rejected; greater than 0xFFFFFFFF
V7 float32: rejected; must have an integer dtype
V7 negative: rejected; negative instance id
V7 shape: rejected; does not match segmentation shape
test_input_contracts: PASS
nuc: nuc.raw size mismatch: expected 1048576 bytes, got 1048575 bytes
V7 truncated nuc.raw: exit=134
```

**V8 — PASS.** The algebra binary exercised all allowed/refused predicate classes.
Production call-site inspection reported:

```text
737:            if (!nuc_ids.empty() && !nuc_can_merge(nuc_ids[v0], nuc_ids[v1])) {
807:                if (!nuc_can_merge(nucleus0, nucleus1)) {
```

The line-807 abort was verified by inspection, not forced through `agg`.

**V9 — PASS with T3 fallback.** T1/T2:

```text
V9 T1 preserve: remapped sid 100->300; payloads preserved
V9 T1 conflict: sid=300 state=CONFLICT collisions=1
V9 T2 conflict: canonical sid=200 state=CONFLICT collisions=1
test_hierarchy_binaries: PASS
```

The literal real-driver attempt failed exactly as follows:

```text
V9 T3 real driver exit=1
--- stderr ---
scripts/init.sh: line 64: AIRFLOW_TMP_DIR: unbound variable
V9 T3 real driver path unavailable; fallback required
```

The permitted fallback then reported:

```text
V9 T3 fallback block: exact match to overlap_chunk_me.sh lines 44-70
V9 T3 fallback: real agg nuc_cuts appended and archived
V9 T3 match: nucleus-vetoed edge absent from region graph
test_cut_chain: PASS
```

Mandatory T4 promoted and vetoed one shared-face composite edge:

```text
new_edges.data 1 [(200, 100, 407.4004211425781, 420)]
nuc_cuts.data 1 [(100, 200)]
V9 T4 mapping composition: localmap and remap are identity for ids 100/200
V9 T4 final representatives differ: 100 != 200
run_hierarchy: PASS
```

**V10 — PASS.**

```text
V10 bash -n: PASS (6 modified shell scripts)
git diff --check: PASS
```

## Review Focus

- Check that the no-affinity nucleus veto implements exactly Invariant D and remains
  outside the frozen-edge condition.
- Review NONE closure and legitimate hierarchy CONFLICT creation versus the B4 tripwire.
- Check the 29-byte ABI, deterministic tie break, and unconditional empty-file lifecycle.
- Scrutinize the Python lossless-cast checks and V2 default-path invariance.
- Review T4's promoted shared-face edge to ensure it tests composite vetoing, not merely
  disconnected child graphs.

## Risks and Unknowns

- The user-approved contract is Invariant D, not identity-set cannot-link. The documented
  99/1 counterexample remains possible; Bound C covers recorded usable evidence only.
- Defaults 50/0.6 are unmeasured on real data. Raw nucleus interiors may remain detached
  from the soma; perinuclear-shell tagging remains external preprocessing.
- Plan_v6's scratch-worktree command failed because `.git` is read-only:
  `fatal: could not create leading directories of '.git/worktrees/baseline_src': Read-only file system`.
  The baseline was exported with `git archive` from the exact SHA instead.
- Plan_v6 was actually wrong in T4 where it passed `""` to `merge_chunks_me.py`.
  With two children that requested an empty-named metadata stream and failed with
  `FileNotFoundError: ongoing__0_0_0_0.data`. The test uses no extra argument, matching
  the real driver when `$META` is empty.
- Plan_v6's T3 matcher named tag `0_0_0_0`, but a mip-1 merge creates
  `o_incomplete_edges_1_0_0_0.tmp`; the fallback correctly uses `1_0_0_0`.
- Plan_v6 omitted operational `EXTRACT_SIZE` from V1. The default targets built, then
  pipeline/hierarchy binaries were rebuilt with `-DEXTRACT_SIZE=ON`.
- The requested artifact directory was mounted read-only in this coder sandbox. Writes to
  `/projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_nucleus/artifacts/code_v0.md`
  failed with `Read-only file system`; this complete fallback artifact is therefore at the
  repository root pending coordinator relocation.
- `build/` remains untracked. All source changes are uncommitted and HEAD is unchanged.

## Changes Since Previous Code Version

Initial implementation.
