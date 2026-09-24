# Code v2
## Overview

Final code revision for the nucleus-instance guidance feature. G1 and G2 are
fixed, and G3 is documented without adding must-link behavior. Lower-resolution
nucleus masks now support an explicit nearest-neighbour coordinate transform,
and nucleus identity removed with boundary counts during overlap reduction is
carried through matching and restored beside those counts before composite
agglomeration.

No dependency or commit was added. All changes and builds remained in the
isolated `work/abiss` checkout.

## What Changed

- Added `NUC_RATIO` and `NUC_OFFSET` as `[z,y,x]` configuration values, with
  identity defaults `[1,1,1]` and `[0,0,0]`.
- Added nearest-neighbour low-resolution mask sampling with the reference
  coordinate rule `(q + ratio // 2) // ratio + offset` and retained the shape
  assertion after upsampling.
- Kept the identity transform on the previous `cut_data` path and added a
  byte-for-byte regression between implicit and explicit identity settings.
- Preserved boundary nucleus records in a deferred overlap sidecar, remapped
  them in `match_chunks`, wrote `extra_nuc.data`, and appended it beside
  `extra_sv_counts.data` before composite aggregation.
- Added a two-child reduce-to-match-to-aggregate regression that proves two
  restored nucleus identities still veto a high-affinity merge.
- Documented that the feature observes identity but supplies no must-link snap,
  so soma fragmentation and the large-cluster size-veto interaction remain.

## Implementation Details

`cut_chunk_agg.py` validates both transform vectors as three integer values and
requires positive ratios. The public order is `[z,y,x]`; the loader reverses it
to the repository's `[x,y,z]` volume coordinates. It computes the low-resolution
index for each requested high-resolution coordinate, reads the minimal source
box, gathers repeated indices without interpolation, applies the existing chunk
padding, then runs the existing dtype/range/shape validation. The exact identity
transform delegates to the original `cut_data` call. `set_env.py` exports the
transform and its defaults only when `NUC_PATH` exists, leaving the default path
free of nucleus variables.

For overlap reduction, `reduce_nuc` still excludes the same boundary SIDs as
`reduce_counts`, preserving the F5 count/payload subset guarantee. It now writes
those excluded fixed-width records to
`reduced_boundary_nuclei_labels_<chunk>.data`. The overlap driver archives that
file, and the `OVERLAP=2` child merge concatenates it independently of the main
nucleus stream. `match_chunks` uses the same already-established restored
supervoxel mapping as `extra_sv_counts.data`, joins duplicate deferred evidence
with the nucleus algebra, maps it to the selected representative, and writes
`extra_nuc.data`. The composite driver appends that file to
`ongoing_nuclei_labels.data` immediately after appending restored counts.

The existing count-restoration iteration was deliberately retained, including
its sentinel convention, so the nucleus fix does not alter default overlap
count output. The new regression creates two reduced children whose boundary
records disappear from the main streams, confirms `match_chunks` restores ids
1 and 2, appends both restoration sidecars as the composite driver does, and
runs `agg` on a high-affinity 100/200 edge. The edge appears in
`nuc_cuts.data` and not in `remap.data`.

## Files Changed
| File | Purpose |
|---|---|
| `README.md` | Nucleus configuration, identity contract, caveats, upgrade warning, and G3 must-link limitation |
| `src/seg/Types.h` | Nucleus wire ABI, exact record algebra, environment parsing, and merge predicate |
| `src/seg/NucExtractor.hpp` | Optional per-supervoxel uint32 nucleus extraction |
| `src/seg/atomic_chunk_ME.cpp` | Optional `nuc.raw` mapping and extraction hookup |
| `src/agg/mean_aggl.cpp` | Nucleus loading, veto, propagation, diagnostics, and cut logging |
| `src/seg/reduce_chunk.cpp` | Nucleus remap/join, boundary filtering, and deferred boundary record output |
| `src/seg/match_chunks.cpp` | Nucleus canonicalization plus `extra_nuc.data` restoration |
| `scripts/set_env.py` | Conditional `NUC_PATH`, `NUC_RATIO`, and `NUC_OFFSET` export |
| `scripts/cut_chunk_agg.py` | Nearest-neighbour mask alignment, validation, and uint32 serialization |
| `scripts/merge_chunks_me.py` | Main and deferred child nucleus stream merging |
| `scripts/merge_chunks_overlap.py` | Neighbour nucleus stream merging |
| `scripts/atomic_chunk_me.sh` | Atomic nucleus artifact lifecycle |
| `scripts/composite_chunk_me.sh` | Composite nucleus lifecycle and restored-record append |
| `scripts/overlap_chunk_me.sh` | Overlap propagation, deferred-record archive, and cut archive |
| `work/test/make_fixture.py` | Fixed-seed atomic fixture |
| `work/test/run_atomic_fixture.sh` | Atomic `acme`/`agg` fixture sequence |
| `work/test/run_v2_invariance.sh` | Exact-baseline default-path bit-invariance gate |
| `work/test/test_nuc_algebra.cpp` | Nucleus algebra, predicate, parser, and overflow tests |
| `work/test/test_nuc_extractor.cpp` | PROPER/CONFLICT/NONE extraction fixture |
| `work/test/test_input_contracts.py` | Dtype/range/shape plus G1 transform and `set_env` tests |
| `work/test/test_hierarchy_binaries.py` | Reduce/match/load tests and G2 boundary-identity survival proof |
| `work/test/t3_overlap_fallback.sh` | Overlap-driver fixture including deferred boundary records |
| `work/test/test_cut_chain.py` | Nucleus-cut and deferred-sidecar propagation fixture |
| `work/test/make_hierarchy_fixture.py` | Shared-face two-child fixture |
| `work/test/run_hierarchy.sh` | Composite shared-face veto test |

## Git Baseline

run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f
current_head: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f

## Verification

The final focused build completed for all affected C++ targets:

```text
[100%] Built target acme
[100%] Built target agg
[100%] Built target reduce_chunk
[100%] Built target match_chunks
```

The real HDF5/CloudVolume input-contract test passed with the configured
`pytc` Python environment:

```text
V7 uint16: accepted; nuc.raw=2048 bytes uint32
V7 identity_transform: accepted; nuc.raw=2048 bytes uint32
V10 G1 identity defaults: implicit/explicit nuc.raw bytes identical
V7 too_large: rejected; greater than 0xFFFFFFFF
V7 float32: rejected; must have an integer dtype
V7 negative: rejected; negative instance id
V7 shape: rejected; does not match segmentation shape
V10 G1 resolution: ratio [4,16,16] and z-offset 1 are voxel-aligned
V10 G1 set_env: nucleus transform defaults exported only with NUC_PATH
test_input_contracts: PASS
```

An initial invocation with `/usr/bin/python3` stopped before test execution
because that interpreter lacks `h5py`; the passing run above used the
repository build environment, which provides `h5py`, `cloudfiles`, and
`cloudvolume`.

The hierarchy binary suite, including the new G2 survival assertion, passed:

```text
V9 T1 preserve: remapped sid 100->300; payloads preserved
V9 T1 conflict transition: sid=100 state=CONFLICT collisions=1
V9 T1 boundary filter: nucleus/count sid sets remain aligned
V9 T2 conflict transition: canonical sid=200 collisions=1
V10 G2 boundary survival: reduce->match restored ids 1/2; composite agg vetoed 100/200
V9 T5 composite load: count/nucleus sid subset accepted; collisions=1
test_hierarchy_binaries: PASS
```

Cut propagation and the existing two-child hierarchy test passed:

```text
V9 T3 fallback: real agg nuc_cuts appended and archived
V9 T3 match: nucleus-vetoed edge absent from region graph
test_cut_chain: PASS
V9 T4 mapping composition: localmap and remap are identity for ids 100/200
V9 T4 final representatives differ: 100 != 200
run_hierarchy: PASS
```

The standalone C++ units passed. The overflow line is the expected child
process abort case:

```text
nuc: voxel count overflow: 18446744073709551615 + 1
test_nuc_algebra: PASS
nuc: conflict_sv 1
nuc: minority_sv 1
nuc: subfloor_sv 1
nuc: subfloor_voxels 10
test_nuc_extractor: PASS
```

The mandatory final `work/test/run_v2_invariance.sh` run passed:

```text
V2 build baseline: PASS
V2 build current: PASS
V2 baseline comparison: identical=35 differing=0 missing=0
V2 current-only empty sidecars: done_nuc.data nuc_cuts.data ongoing_nuc.data
V2 default-path nucleus log lines: 0
V2 stale nucleus environment without nuc.raw: PASS
run_v2_invariance: PASS
```

Both V2 configurations emitted only the repository's existing optional
`abslConfig.cmake` warning. Final shell syntax and whitespace checks passed:

```text
bash -n: PASS
git diff --check: PASS
```

## Review Focus

- Confirm the `[z,y,x]` public transform is reversed once for `[x,y,z]` access
  and follows the documented nearest-low-resolution coordinate formula.
- Confirm the identity branch invokes the former `cut_data` behavior directly
  and that transform parsing remains unreachable without `NUC_PATH`.
- Trace deferred boundary records from `reduce_chunk` through the overlap tar,
  child merge, `match_chunks`, `extra_nuc.data`, and the composite append.
- Compare the `extra_nuc.data` representative selection with the unchanged
  `extra_sv_counts.data` sentinel/iteration behavior.
- Re-run the G2 test and mandatory V2 invariance gate.
- Confirm the README states observation-only behavior and names the must-link
  snap as a follow-up rather than implying it was implemented.

## Risks and Unknowns

- Defaults `50` and `0.6` remain uncalibrated on production masks, and raw
  nucleus interiors can remain disconnected from the soma without the
  documented perinuclear-shell preprocessing.
- The transform requires every mapped low-resolution coordinate to exist in
  the nucleus volume. It raises rather than silently clipping an invalid
  ratio/offset mapping.
- The deferred boundary file is another mixed-version hierarchy boundary;
  every overlap child must be regenerated with nucleus-aware binaries, as the
  README upgrade warning requires.
- Must-link behavior remains deliberately absent. The size veto can still
  block soma/proximal-dendrite absorption, and fragmentation is not fixed.
- The fixed-width dominant-id record still implements Invariant D rather than
  tracking all minority identities.
- The known unrelated `match_chunks.py` semantic `nlabels = 5` inconsistency
  remains untouched.
- The G1 and G2 tests are deterministic synthetic fixtures, not a whole-volume
  production replay. The repository's `build/` and `work/` trees remain
  untracked, no commit was created, and HEAD is unchanged.

## Changes Since Previous Code Version

- **G1 — Fixed.** Added defaulted `NUC_RATIO`/`NUC_OFFSET` plumbing, validated
  `[z,y,x]` transforms, nearest-neighbour low-resolution sampling with z-offset
  support, the post-upsample shape check, a real `[4,16,16]` alignment test,
  and an implicit/explicit identity byte comparison.
- **G2 — Fixed.** Investigation confirmed there was no other survival route.
  Boundary records removed by `reduce_nuc` are now retained in a deferred
  sidecar, canonicalized by `match_chunks` into `extra_nuc.data`, and appended
  beside restored counts. A two-child reduce-to-match-to-composite-`agg` test
  proves ids 1 and 2 still veto the high-affinity 100/200 edge.
- **G3 — Documented only, as directed.** The README now says nucleus guidance
  never forces a merge, does not solve soma fragmentation, can still be limited
  by the large-cluster size veto, and names the `lib/em_seg`-style must-link snap
  as the follow-up.
