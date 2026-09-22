You are the CODER for a CCC run, at stage code_v2. This is the FINAL code revision available.
Your code_v1 came back NEEDS_CHANGES with two new majors (G1, G2) and one minor (G3).

## Hard constraints (unchanged)

1. WORK ONLY IN:
       /projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_nucleus/work/abiss
   NEVER touch /projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss.
2. No git commits. No new dependencies.
3. V2 default-path bit-invariance must still hold — re-run work/test/run_v2_invariance.sh and
   quote its output.
4. Do not claim a command passed unless you ran it and saw the output.

## Artifact

Write to  work/abiss/code_v2.md  (NOT artifacts/ — outside your sandbox). Exact sections:

    # Code v2
    ## Overview
    ## What Changed
    ## Implementation Details
    ## Files Changed
    | File | Purpose |
    |---|---|
    ## Git Baseline
    ## Verification
    ## Review Focus
    ## Risks and Unknowns
    ## Changes Since Previous Code Version

"## Git Baseline" must contain run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f and
current_head: <git rev-parse HEAD>. "## Changes Since Previous Code Version" must address G1, G2,
G3 by name.

## What the reviewer verified (already trusted, do not redo)

The reviewer ran work/test/run_v2_invariance.sh cold themselves and it passed 35/0/0 with zero
default-path nucleus log lines and the stale-env test passing. F1-F6 are all confirmed fixed. Your
F5 boundary-filter fix was confirmed correct against reduce_counts (reduce_chunk.cpp:137-139).

## The work

G2 [major] — FIX THIS FIRST, it is the correctness one.
  Your F5 fix (reduce_nuc skipping boundary_sv) mirrors reduce_counts correctly, but it is only
  half the pattern. Sizes dropped by reduce_counts are RESTORED:
      match_chunks.cpp:329       writes extra_sv_counts.data
      composite_chunk_me.sh:45   cat extra_sv_counts.data >> ongoing_supervoxel_counts.data
  There is no nucleus analogue. So a supervoxel on a chunk boundary keeps its size across an
  overlap reduction but LOSES its nucleus record — a soma straddling an overlap boundary silently
  loses its cannot-link constraint, and in a whole-volume run boundaries are everywhere.
  Fix preserving the existing symmetry: write extra_nuc.data in match_chunks and append it in
  composite_chunk_me.sh beside line 45.
  Your evidence for this area was "V9 T1 boundary filter: nucleus/count sid sets remain aligned",
  which shows absence of abort, NOT survival of identity. ADD THE TEST THAT SHOWS SURVIVAL:
  two chunks, a tagged pair straddling the overlap boundary, run reduce_chunk -> match_chunks ->
  composite agg, assert the pair is STILL VETOED.
  NOTE: the reviewer reasoned this from file flow and did not execute a proof of loss. If on
  investigation the record DOES survive by some route they missed, say so with the evidence and
  do not add dead machinery — that is a valid outcome.

G1 [major] — nucleus mask resolution.
  cut_chunk_agg.py:24-27 hard-raises unless the nucleus cutout shape equals seg.raw's, and there
  is no NUC_RATIO/NUC_OFFSET. Nucleus/cell-body masks are normally produced at a lower mip. The
  reference pattern, from lib/em_seg (same class of data, a real nucleus volume):
      em_seg/demo/r0.yaml:20-23     SOMA_EROSION: 0, SOMA_RATIO: [4,16,16], SOMA_OFFSET: [14,0,0]
      em_seg/em_seg/dataloader.py:46-51
          zz = (z + self.ratio[0]/2) // self.ratio[0]   # EM z -> nearest low-res z
          zz = int(zz + self.st[0])                     # plus offset
  Add NUC_RATIO (z,y,x downsample factor of the mask relative to AFF_RESOLUTION, default [1,1,1])
  and NUC_OFFSET (default [0,0,0]) to cut_chunk_agg.py and set_env.py, using nearest-neighbour
  upsampling so instance ids are never interpolated. Keep the existing shape assertion as the
  post-upsample check. Test: a [4,16,16]-downsampled mask with a z-offset is accepted and lands
  voxel-aligned with seg.raw; default [1,1,1]/[0,0,0] must be byte-identical to today's behavior.

G3 [minor] — documentation only, do NOT implement.
  Must-link is deliberately out of scope for this run. Add a README limitation stating that the
  feature only OBSERVES nucleus identity and never forces a merge, so soma fragmentation is not
  addressed and mean_aggl.cpp:709-718 can still block a soma from absorbing proximal dendrites.
  Name the follow-up: a must-link snap, as lib/em_seg does at seg_pipeline.py:366-380 by snapping
  every fragment overlapping a soma onto one reserved id before the region graph is built.

=========================== BEGIN task.md ===========================
# Task

Add a first-class **nucleus instance mask** input to `lib/abiss` (`NUC_PATH`, dtype
`uint32`) and use it to guide agglomeration, so that supervoxel clusters carrying
different nucleus ids are never merged into one segment.

The literal user request:

> "given the context above, add a feature in abiss to make use of nucleus instance mask uint32"

## Repository and Working Copy

The code lives in the **`lib/abiss` repository** (`https://github.com/PytorchConnectomics/abiss`),
which is a standalone git repo. `lib/` is gitignored by the parent `pytorch_connectomics`
repo, so abiss changes never appear in a parent diff. All CCC git baselines and diffs for
this run therefore target abiss, not pytorch_connectomics.

**Standing constraint carried over from `.agent/features/abiss_speedup/`:** build
experiments must NOT touch `lib/abiss` in place. A previous run broke a live SLURM chain by
rebuilding `lib/abiss/build/` while jobs were executing those binaries. SLURM job 2787863
(`v3mim_verify`, from the abiss_speedup bench) was running against abiss at the start of
this run.

Accordingly, this run works on an isolated clone:

```text
.agent/features/abiss_nucleus/work/abiss   (git clone of lib/abiss, detached at 3c4f562, clean, no build/)
```

The live `lib/abiss` checkout is read-only for this run. All edits, builds, and diffs happen
inside `work/abiss`.

## Background: what abiss already has

ABISS already carries a *semantic class* guidance path end-to-end, at three injection
points. The nucleus feature should parallel it, not overload it.

**P1 - pre-watershed affinity cut.** `scripts/cut_chunk_ws.py:31` calls
`augment_affinity.py:147` `mask_affinity_with_semantic_labels`, which zeroes the affinity
edge wherever the two voxels carry *different nonzero* sem labels. This test is already
identity-based (`sem_offset != sem_aligned`), not class-based. Gated by `SEMANTIC_WS`.

**P2 - per-supervoxel payload.** `src/seg/SemExtractor.hpp` accumulates a fixed 3-bin
histogram per supervoxel through a 6->3 class LUT (`sem_map`, line 61). Serialized as
`ongoing_semantic_labels.data` and carried through the chunk hierarchy by
`match_chunks.py:126`, `reduce_chunk.py:111`, `merge_chunks_me.py:59`,
`merge_chunks_overlap.py:65`.

**P3 - agglomeration veto.** `src/agg/mean_aggl.cpp:625` `sem_can_merge` requires both
clusters to have a dominant label (>= `dominant_signal_ratio` 0.6 of counts, >=
`total_signal_threshold` 100k voxels); if the dominants differ the edge is refused and
logged to `sem_cuts.data`. Applied at line 699 **only when the edge weight is <= 0.5**
(`sem_params.aff_threshold`, line 130). Cluster payload is combined by summation at lines
754-758.

**P4 - size veto.** `src/agg/mean_aggl.cpp:709-718` refuses a merge when both sides are
large. This heuristic exists precisely because the code cannot otherwise tell a real cell
body from a runaway merge.

## Why a separate `NUC_PATH` rather than reusing `SEM_PATH`

Decided with the user before this run started:

1. `semantic_t = uint8_t` (`src/seg/Types.h:25`) caps instances at 255. Widening it in
   place would change the on-disk meaning of every existing sem file.
2. `SemExtractor` is lossy by design (6 classes -> 3 bins); an instance id must survive
   verbatim.
3. The predicates genuinely differ. Sem is "different classes should not merge, and only
   for weak edges." Nucleus is "different cells must never merge, at any affinity." One
   shared `aff_threshold` forces one of them to be wrong.
4. They compose: class veto for axon/dendrite *and* instance veto for cell bodies should be
   usable simultaneously. Overloading one channel makes them mutually exclusive.
5. The sem path is already inconsistent: `match_chunks.py:122` reads `nlabels = 5` while
   `reduce_chunk.py:108` and `sem_array_t` are both 3. That OVERLAP=2 branch is broken or
   dead; building on it inherits the defect. Do not copy `process_sems` blindly.

## Design intent

Carry a fixed-width per-cluster nucleus record so every plumbing script stays a one-line
mirror of the existing sem lines:

```text
(seg_t sid, nuc_t id, size_t count, size_t total)
```

`count` = voxels of the dominant id, `total` = all nucleus-tagged voxels in the cluster.
This is exact *given* the veto, because a cluster can never legitimately end up holding two
different ids: any such merge is refused before it happens. Do not serialize a
variable-length id->count map; that is only needed for a soft veto and it breaks the
fixed-record assumption in five scripts.

The veto should be **ungated by affinity**, unlike the sem veto: an instance nucleus mask is
far more trustworthy than a semantic net prediction, and a soma-soma fusion is far more
expensive than a stray cut. Keep a minimum tagged-voxel threshold so a handful of
misassigned voxels cannot hard-split a real cell.

Optional companion, if the plan judges it in scope: a `NUC_WS` flag mirroring `SEMANTIC_WS`
at `cut_chunk_ws.py:31`, reusing the `mask_affinity_with_semantic_labels` mechanism. It
makes "one supervoxel carries at most one nucleus id" true by construction, which is what
makes the fixed-width record exact.

`traverseSegments` (`src/seg/Utils.hpp:112`) is fully variadic, so adding an extractor is
mechanical. Note that `src/seg/atomic_chunk_ME.cpp:78-104` currently duplicates the whole
call for the sem/no-sem case; a second optional payload turns that into four branches, so
prefer making the new extractor a no-op when its input file is absent, or hoist the pack.

Also note: agglomeration parameters are compile-time constants
(`src/agg/mean_aggl.cpp:121-153`); `main` only parses `argv[1]` as the agglomeration
threshold (lines 1145-1149). Any new tunable is a recompile unless the plan adds env or
argument parsing.

## Scope

**In scope:** the `lib/abiss` repository only - C++ (`src/seg/`, `src/agg/`) and the
Python/bash chunk drivers under `scripts/`.

**Out of scope:** the pytorch_connectomics wrapper. `connectomics/decoding/decoders/abiss.py`
has no `SEM_PATH` plumbing today and gains no `NUC_PATH` plumbing in this run. Also out of
scope: generating the nucleus instance mask itself, and the perinuclear-shell tagging
preprocessing step described below.

## Known caveat the implementation cannot fix

The nuclear envelope is a membrane; affinity networks put a boundary on it, so the nucleus
interior often becomes its own cluster that never joins the soma. An id parked there
constrains nothing. Correct usage tags *perinuclear cytoplasm* (dilate each instance, take
shell = dilated minus nucleus, assign id k to supervoxels dominated by shell k). That is a
preprocessing step on the nucleus volume, outside abiss and outside this run's scope, but
the feature should be documented so a user does not assume raw nucleus interiors work.

## Constraints

* Do not modify the live `lib/abiss` checkout; work only in `work/abiss`.
* Do not create git commits during this CCC run.
* Do not add runtime or build dependencies without explicit user approval
  (`AGENTS.md` Environment section, parent repo).
* Existing behavior with no `NUC_PATH` configured must be bit-identical to the current
  pipeline. This repo reproduces a Seuron provenance record; a silent change to default
  segmentation output is a hard failure.
* Keep the diff scoped. Do not opportunistically fix the `nlabels = 5` sem bug in the same
  change unless the plan argues it is required; if it is left alone, say so.

## Success criteria

1. `NUC_PATH` (plus any needed `NUC_MIP`) is accepted end-to-end: config JSON -> `set_env.py`
   -> `cut_chunk_agg.py` writes `nuc.raw` -> C++ extractor -> agglomeration veto ->
   per-chunk serialization -> chunk hierarchy reduce/merge/match.
2. Two clusters carrying different nucleus ids are never merged, at any affinity, subject to
   the minimum tagged-voxel threshold.
3. Refused nucleus edges are logged to their own cut file, mirroring `sem_cuts.data`.
4. Nucleus ids are `uint32`, not `uint8`.
5. With `NUC_PATH` absent, output is unchanged from baseline.
6. The build succeeds inside `work/abiss` and the verification evidence is real command
   output, not asserted.
=========================== END task.md ===========================

=========================== BEGIN artifacts/code_v1.md ===========================
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
=========================== END artifacts/code_v1.md ===========================

=========================== BEGIN artifacts/review_v1.md (THE REVIEW TO ADDRESS) ===========================
# Review v1

## Summary

Reviewer: claude (planner, in-session). Raw notes: `state/review_v1.review.raw.md`.

**All six review_v0 findings are fixed, and I verified each one against the code rather than
against `code_v1.md`.** The headline check: I ran `work/test/run_v2_invariance.sh` cold, as an
outsider would, and it passed end to end —

```text
V2 build current: PASS
V2 baseline comparison: identical=35 differing=0 missing=0
V2 current-only empty sidecars: done_nuc.data nuc_cuts.data ongoing_nuc.data
V2 default-path nucleus log lines: 0
V2 stale nucleus environment without nuc.raw: PASS
run_v2_invariance: PASS
```

That single run confirms F2 (the gate is now reproducible by someone who did not write it), F1
(stale nucleus environment with no `nuc.raw` no longer aborts), and F4 (zero default-path nucleus
log lines, asserted rather than assumed). F3 is fixed at all three sites by gating on the
*transition* into CONFLICT (`!was_conflict && result == CONFLICT`). F6 is documented.

**F5 deserves specific credit:** investigating my open question surfaced a real bug rather than
just answering it. `reduce_nuc` kept boundary supervoxel ids while `reduce_counts`
(`reduce_chunk.cpp:137-139`) drops them, so `load_nuc` could legitimately meet a sid absent from
`seg_indices` and abort. The fix mirrors `reduce_counts` exactly and is correct.

Two majors below. **G1 is the scope item the user directed be folded in** from the `lib/em_seg`
read. **G2 is new, and it is the other half of F5's fix** — the half that was missed. Neither is a
regression; `V2` still passes and nothing from `code_v0` broke.

## Diff Baseline

run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f

Reviewed: `git diff 3c4f5621` — 13 tracked files, 511 insertions (vs 484 in `code_v0`), plus the
untracked `src/seg/NucExtractor.hpp` and `work/test/`. `HEAD` equals `run_start_ref`; no commits
created; the review itself made no edits.

## Findings

**G1 [major] — the mask-resolution contract makes the feature unusable on a realistically
produced mask.**
`scripts/cut_chunk_agg.py:24-27` hard-raises when the nucleus cutout's shape differs from
`seg.raw`'s, and there is no `NUC_RATIO` or `NUC_OFFSET` anywhere in the tree. But nucleus and
cell-body masks are normally produced at a lower mip, and `lib/em_seg` shows exactly that
production pattern for the same class of data:

```yaml
# em_seg/demo/r0.yaml:20-23   (source: .../jwr15/nucleus/cell_yl_cb_cc_fix.h5)
SOMA_EROSION : 0
SOMA_RATIO   : [4,16,16]
SOMA_OFFSET  : [14,0,0]
```

```python
# em_seg/em_seg/dataloader.py:46-51
zz = (z + self.ratio[0]/2) // self.ratio[0]   # EM z -> nearest low-res z
zz = int(zz + self.st[0])                     # plus offset
```

The mask is stored downsampled with a z-offset and the loader reconciles resolution on read
(in-plane zoom and erosion in the subclass — the demos import `scipy.ndimage.zoom` and
`binary_erosion`). Our contract rejects such a mask outright, and nothing documents that the caller
must pre-upsample and re-align first. Either add `NUC_RATIO`/`NUC_OFFSET` mirroring the `SOMA_*`
params, or state the pre-processing requirement explicitly in the README with the exact expected
alignment.

**G2 [major] — boundary nucleus identity is dropped in the OVERLAP=2 path and never restored.**
F5's fix is correct but is only half of the pattern it mirrors. Sizes dropped by `reduce_counts`
come back:

```text
match_chunks.cpp:329        writes extra_sv_counts.data
composite_chunk_me.sh:45    cat extra_sv_counts.data >> ongoing_supervoxel_counts.data
```

There is no nucleus analogue — `grep` for `extra_nuc` / `extra_*nuclei` finds nothing. So a
supervoxel on a chunk boundary retains its **size** across an overlap reduction but loses its
**nucleus record**. A soma straddling an overlap boundary therefore silently loses its cannot-link
constraint at that level, and in a whole-volume run boundaries are everywhere — which is precisely
where the constraint matters most.

`code_v1.md`'s evidence for this area is `V9 T1 boundary filter: nucleus/count sid sets remain
aligned`. That demonstrates the *absence of an abort*, not the *survival of identity*; they are
different claims and only the first is tested.

Stated honestly: I reasoned this from the file flow and the missing analogue and did **not** execute
a case proving loss. The test that settles it is named under Tests to Add. The fix that preserves
the existing symmetry is an `extra_nuc.data` written by `match_chunks` and appended in
`composite_chunk_me.sh` beside line 45.

**G3 [minor] — must-link is absent by design; record it as a limitation.**
Per the user's decision this stays out of the run, so it is not a defect against plan_v6, which
never promised it. But `lib/em_seg` shows what we are giving up: `seg_pipeline.py:366-380` snaps
every watershed fragment overlapping a soma onto a single reserved id **before the region graph
exists**, so the soma becomes one node with one aggregated affinity per neighbour. That is what
prevents soma fragmentation and makes the size veto irrelevant for cell bodies. Our feature only
*observes* identity — it never forces a merge — so `mean_aggl.cpp:709-718` can still block a real
soma from absorbing proximal dendrites. The README should say this plainly and name the follow-up.

## Tests to Add

1. **Boundary-spanning soma across an overlap reduction** (settles G2, highest value): two chunks,
   a tagged pair straddling the overlap boundary, run `reduce_chunk` → `match_chunks` → composite
   `agg`, and assert the pair is *still vetoed*. Today's test asserts only that sid sets stay
   aligned.
2. **Mask-resolution acceptance** (G1): if `NUC_RATIO`/`NUC_OFFSET` are added, a test that a
   `[4,16,16]`-downsampled mask with a z-offset is accepted and lands voxel-aligned with `seg.raw`;
   if instead documented, a test asserting the error message names the required pre-processing.

## Questions

1. G2 is the one thing I could not settle by reading alone. Is there any route by which a boundary
   supervoxel's nucleus record re-enters the ongoing payload at the next hierarchy level that I
   missed? If there is, G2 collapses to a documentation note.
2. For G1, which way do you want it — a real `NUC_RATIO`/`NUC_OFFSET` upsample path, or a documented
   caller requirement? The former is more code in a file we have already touched; the latter is free
   but pushes an easy-to-get-wrong alignment step onto the user.

## Verdict

VERDICT: NEEDS_CHANGES
=========================== END artifacts/review_v1.md ===========================

Now fix G1 and G2, document G3, verify, and write work/abiss/code_v2.md.
