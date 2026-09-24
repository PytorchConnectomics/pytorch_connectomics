# Code v1
## Overview

Revised the code_v0 nucleus implementation in response to review_v0. All six
findings F1-F6 were accepted and addressed. The default path no longer parses
nucleus settings or emits nucleus diagnostics without an input, conflict
counters count transitions rather than every subsequent join, overlap
reduction preserves the nucleus/count SID relationship, mixed-version
pipelines are documented as unsupported, and the V2 bit-invariance gate is
now a single re-runnable script.

No dependency or commit was added. All work remained in the isolated
`work/abiss` checkout.

## What Changed

- Moved `ABISS_NUC_DOMINANCE` and `ABISS_NUC_MIN_TAGGED` parsing inside the
  `nuc.raw` existence branch.
- Suppressed nucleus extraction/loading/collision diagnostics when no nucleus
  source or payload exists while retaining the required empty file lifecycle.
- Changed load, reduce, and match collision counters to increment only when an
  accumulator transitions from a non-CONFLICT state to CONFLICT.
- Made overlap reduction filter boundary nucleus records exactly where it
  filters the corresponding ongoing supervoxel counts.
- Added the mixed-binary upgrade warning to the README.
- Added `work/test/run_v2_invariance.sh`, a complete exact-SHA baseline/current
  build and output comparison with empty-sidecar, default-log, and stale-env
  assertions.
- Extended the hierarchy binary fixture with three-record counter-transition,
  overlap boundary-alignment, and composite `load_nuc` cases.

## Implementation Details

`atomic_chunk_ME.cpp` now initializes the default ratio and threshold without
reading the environment. It parses the two environment variables only after
`nuc.raw` is found and validated. The nullable extractor still participates in
the default traversal and creates the expected empty payload, but its counter
printing is conditional on having a source.

Each collision site records whether the accumulator was already CONFLICT
before calling `nuc_join`; only a newly produced CONFLICT increments the
observable counter. Empty reduce and match inputs no longer emit zero-valued
nucleus counter lines.

The ordinary composite path preserves the required subset definitively:
agglomeration partitions counts and nucleus records from the same
`seg_indices` loop, child merging concatenates both streams, and OVERLAP=2
matching applies the same representative mapping to both while only adding
extra count records. Review also exposed a separate overlap-reduction gap:
`reduce_counts` discarded boundary SIDs but `reduce_nuc` did not. Passing the
same `boundary_sv` set into `reduce_nuc` and applying the same pre-remap filter
closes that gap.

The V2 harness exports the baseline source from the exact run-start commit,
builds baseline and current `acme`/`agg` with Release and `EXTRACT_SIZE=ON`,
runs the fixed-seed fixture under both, and compares exactly the files produced
by the baseline. It accepts `ABISS_CXX_COMPILER`,
`ABISS_TOOLCHAIN_PREFIX`, and `ABISS_BUILD_JOBS` overrides and otherwise
discovers the existing configured compiler.

## Files Changed
| File | Purpose |
|---|---|
| `README.md` | Nucleus contract, limitations, diagnostics, and mixed-binary upgrade warning |
| `src/seg/Types.h` | Nucleus wire ABI, exact algebra, parsing, and merge predicate |
| `src/seg/NucExtractor.hpp` | Nullable extractor and source-gated diagnostics |
| `src/seg/atomic_chunk_ME.cpp` | Optional input mapping and source-gated environment parsing |
| `src/agg/mean_aggl.cpp` | Nucleus loading, veto, propagation, logging, and transition-only collision count |
| `src/seg/reduce_chunk.cpp` | Remap/join payloads, transition count, and boundary SID filtering |
| `src/seg/match_chunks.cpp` | Canonicalize/join payloads and transition-only collision count |
| `scripts/set_env.py` | Optional `NUC_PATH` export |
| `scripts/cut_chunk_agg.py` | Nucleus cutout validation and uint32 serialization |
| `scripts/merge_chunks_me.py` | Child nucleus stream merging |
| `scripts/merge_chunks_overlap.py` | Neighbour nucleus stream merging |
| `scripts/atomic_chunk_me.sh` | Atomic nucleus artifact lifecycle |
| `scripts/composite_chunk_me.sh` | Composite nucleus artifact lifecycle |
| `scripts/overlap_chunk_me.sh` | Overlap nucleus propagation and cut archival |
| `work/test/make_fixture.py` | Fixed-seed atomic fixture |
| `work/test/run_atomic_fixture.sh` | Atomic `acme`/`agg` fixture sequence |
| `work/test/run_v2_invariance.sh` | Re-runnable exact-baseline default-path bit-invariance gate |
| `work/test/test_nuc_algebra.cpp` | Algebra, predicate, parser, and overflow tests |
| `work/test/test_nuc_extractor.cpp` | PROPER/CONFLICT/NONE extraction fixture |
| `work/test/test_input_contracts.py` | Nucleus dtype, range, and shape validation |
| `work/test/test_hierarchy_binaries.py` | Reduce/match counters, boundary alignment, and composite load tests |
| `work/test/t3_overlap_fallback.sh` | Overlap-driver fallback block |
| `work/test/test_cut_chain.py` | Nucleus-cut propagation fixture |
| `work/test/make_hierarchy_fixture.py` | Shared-face two-child fixture |
| `work/test/run_hierarchy.sh` | Composite shared-face veto test |

## Git Baseline

run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f
current_head: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f

## Verification

The focused rebuild completed for all C++ targets changed in code_v1:

```text
[100%] Built target acme
[100%] Built target agg
[100%] Built target reduce_chunk
[100%] Built target match_chunks
```

`work/test/run_v2_invariance.sh` built both versions from source and passed:

```text
V2 build baseline: PASS
V2 build current: PASS
V2 baseline comparison: identical=35 differing=0 missing=0
V2 current-only empty sidecars: done_nuc.data nuc_cuts.data ongoing_nuc.data
V2 default-path nucleus log lines: 0
V2 stale nucleus environment without nuc.raw: PASS
run_v2_invariance: PASS
```

The counter-transition, overlap-alignment, and composite loader fixture passed:

```text
V9 T1 preserve: remapped sid 100->300; payloads preserved
V9 T1 conflict transition: sid=100 state=CONFLICT collisions=1
V9 T1 boundary filter: nucleus/count sid sets remain aligned
V9 T2 conflict transition: canonical sid=200 collisions=1
V9 T5 composite load: count/nucleus sid subset accepted; collisions=1
test_hierarchy_binaries: PASS
```

The two-child composite veto remained intact:

```text
V9 T4 mapping composition: localmap and remap are identity for ids 100/200
V9 T4 final representatives differ: 100 != 200
run_hierarchy: PASS
```

The standalone units passed after compiling with the configured toolchain's
Boost include path:

```text
nuc: voxel count overflow: 18446744073709551615 + 1
test_nuc_algebra: PASS
nuc: conflict_sv 1
nuc: minority_sv 1
nuc: subfloor_sv 1
nuc: subfloor_voxels 10
test_nuc_extractor: PASS
```

The expected overflow line is from the child-process abort test. An initial
standalone compile command omitted the Boost include path and failed before
creating either test binary; the corrected command above compiled and ran
both.

Final hygiene checks:

```text
bash -n: PASS
git diff --check: PASS
```

The V2 configurations emitted only the repository's existing optional
`abslConfig.cmake` warning.

## Review Focus

- Confirm that environment parsing is unreachable when `nuc.raw` is absent
  and that the nullable extractor remains behaviorally inert.
- Check that all three collision counters use the same non-CONFLICT to
  CONFLICT transition definition.
- Review the overlap reducer's boundary filter against `reduce_counts`.
- Re-run `work/test/run_v2_invariance.sh` as the default-path gate.
- Confirm the README upgrade note is sufficiently explicit that all child
  chunks must be regenerated together.

## Risks and Unknowns

- The user-approved guarantee remains Invariant D, not identity-set
  cannot-link; the documented minority-identity counterexample remains.
- Defaults `50` and `0.6` remain uncalibrated on real production nucleus
  masks, and perinuclear-shell preprocessing remains outside this repository.
- Mixed pre-change and nucleus-aware chunk artifacts remain incompatible.
  Code_v1 documents this rather than adding a compatibility layer.
- V2 is a deterministic baseline/current pipeline fixture, not a replay of a
  full Seuron volume.
- The repository's `build/` and `work/` trees remain untracked. No commit was
  created, and HEAD is unchanged.

## Changes Since Previous Code Version

- **F1 — Fixed.** Both nucleus environment parsers now execute only inside the
  `exists("nuc.raw")` branch. The stale-environment default-path test passed
  with both values set to `garbage`.
- **F2 — Fixed.** Added and ran `work/test/run_v2_invariance.sh`; it builds the
  exact run-start baseline and current source, then reports 35 identical, 0
  differing, and 0 missing baseline files.
- **F3 — Fixed.** `load_nuc`, `reduce_nuc`, and `process_nucs` increment only
  on entry into CONFLICT. Three-record tests report exactly one collision in
  load, reduce, and match.
- **F4 — Fixed.** Extraction counters require a real source, the empty loader
  is silent, and empty hierarchy inputs do not print nucleus collision lines.
  V2 observed zero default-path nucleus log lines.
- **F5 — Fixed.** Code reading established the ordinary composite subset
  guarantee and found an overlap-reducer exception. Boundary nucleus records
  are now filtered with boundary counts; both a direct alignment regression
  and composite-level `agg` load passed without abort.
- **F6 — Fixed as documentation.** The README now requires regenerating all
  child chunks after upgrade and explicitly disallows mixed pre-change and
  nucleus-aware hierarchies. Runtime compatibility was intentionally not
  added because the review marked it out of scope.
