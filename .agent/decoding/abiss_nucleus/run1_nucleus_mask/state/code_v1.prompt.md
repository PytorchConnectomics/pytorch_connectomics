You are the CODER for a CCC run, at stage code_v1. Your code_v0 implementation was reviewed and
came back NEEDS_CHANGES. Triage the review, fix the accepted findings, verify, and summarize.

## Hard constraints (unchanged)

1. WORK ONLY IN:
       /projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_nucleus/work/abiss
   NEVER touch /projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss (live checkout; SLURM
   jobs execute binaries from its build/).
2. DO NOT create git commits.
3. DO NOT add runtime or build dependencies.
4. Default-path bit-invariance (V2) must still hold.
5. Do not claim a command passed unless you ran it and saw the output.

## Artifact location - IMPORTANT

Last time you could not write to the artifacts/ directory because it is outside your sandbox.
Do not retry that path. Write your artifact to:

       work/abiss/code_v1.md      (i.e. code_v1.md at the root of your working directory)

The coordinator will relocate it. Use EXACTLY these level-2 sections, each exactly once:

    # Code v1
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

"## Git Baseline" must contain:
    run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f
    current_head: <git rev-parse HEAD>

"## Changes Since Previous Code Version" must address EACH finding F1-F6 by name and say
explicitly whether it was fixed, and if not, why not. Do not silently drop any.

## What the reviewer verified independently (so you know what is already trusted)

The reviewer did NOT take code_v0.md on trust. They rebuilt baseline binaries at run_start_ref in
a throwaway git worktree and reproduced V2 themselves (35 identical / 0 differing / 0 missing),
recompiled and re-ran both unit binaries, and re-ran all five nucleus modes confirming the veto
fires at ~0.97 affinity and that 0xFFFFFFFF works as an ordinary id. So the FEATURE is confirmed
working. The findings are about the default path, verification reproducibility, and hygiene.

## Priorities

F1 and F2 are the majors and must be fixed:
  F1 - move the nucleus env parsing inside the `exists("nuc.raw")` guard so a stale
       ABISS_NUC_DOMINANCE cannot abort acme on a run with no nucleus input.
  F2 - add work/test/run_v2_invariance.sh that builds the baseline at run_start_ref and performs
       the whole comparison, printing identical/differing/missing counts. Then RUN it and quote
       its output. This is the Seuron bit-identity gate and it must be re-runnable by someone
       else.
Then F3 (counter over-count), F4 (default-path log noise), F5 (composite-path load_nuc question -
if you can settle it by reading the code, say so definitively; if not, say you could not), and F6
(README upgrade note about mixed-binary pipelines).

Also add the four tests listed in the review's "Tests to Add" section where they are cheap.

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

=========================== BEGIN run.md ===========================
# CCC Run

## Description

Add a first-class nucleus instance mask input (`NUC_PATH`, `uint32`) to `lib/abiss` and use
it to guide agglomeration: clusters carrying different nucleus ids must never merge. The
feature parallels the existing semantic-class channel rather than overloading it, because
`semantic_t` is `uint8_t`, `SemExtractor`'s class LUT is lossy, the two vetoes need
different affinity gating, and they should be usable together.

Git baselines target the **`lib/abiss` repository**, not pytorch_connectomics, because
`lib/` is gitignored by the parent and abiss changes appear in no parent diff.

Per the standing constraint recorded in `.agent/features/abiss_speedup/run.md`, this run
does not touch the live `lib/abiss` checkout. All edits, builds, and diffs happen in an
isolated clone:

```text
work/abiss   (git clone of lib/abiss, detached at 3c4f562, clean, no build/)
```

That constraint exists because a previous run broke a live SLURM chain by rebuilding
`lib/abiss/build/` underneath running jobs. SLURM 2787863 (`v3mim_verify`, abiss_speedup
bench) was running against abiss when this run started.

## Runtime

planner: claude
coder: codex
plan_code: claude-codex
session_detected: claude
plan_code_source: default

## Rounds

plan_rounds: 6
revision_rounds: 2

## Superseded Blocking Reason

plan_v3_review changed the character of the run. Rounds 1-3 found defects a better plan could fix
(11 -> 6 -> 6 findings). Round 4 found a **representational impossibility** the planner cannot
resolve without a decision about what the feature promises.

Finding I1, with the floor disabled and dominance 0.6:

    A: 60 voxels id1, 40 voxels id2  -> PROPER id1
    B: 60 voxels id1, 40 voxels id3  -> PROPER id1

B3 allows A+B because both records name id1, yet the merge joins nucleus 2 and nucleus 3 material.
Bound C still holds exactly (80 == 0.4 * 200), so Bound C is mathematically compatible with
violating cannot-link. Any single-dominant-id record has this property.

The reviewer states the trap directly: "The current representation offers no setting that provides
both noise tolerance and the claimed guarantee." Floor enabled -> evidence is discarded and the
accumulation leak returns; floor disabled -> one stray tagged voxel creates an absolute veto that
hard-splits a cell.

This is a scope/contract decision, not a planning defect. The three viable features are materially
different in cost and in what they deliver. Recorded for the user; the run is parked pending that
decision rather than guessing.

Findings I3 (abort sites must mirror the whole `nuc_can_merge` predicate), I4 (the `match_chunks`
abort is data-reachable and would be a DoS on a long run), and I5 (V8 still not executable) are
ordinary defects with clear fixes and are not blocking on their own.

## Superseded Blocking Reason

plan_v5 is the terminal plan version under the extended `p5` rounds and is unreviewed.

Five review rounds, every finding [major]: 11 -> 6 -> 6 -> 5 -> 5.

Round 5 (plan_v4_review) confirmed I3 and I4 fixed, and found that plan_v4's own fix for the
evidence floor broke merge closure: a NONE record carrying nonzero `total` poisons the dominance
property of any PROPER record it joins. It also showed plan_v4's "honest limitations" text was
itself inaccurate in two places, and that the B4 abort test targeted a condition unreachable
through `agg`.

plan_v5 answers with `NONE => total == 0`, which restores closure strongly enough to state as a
proved theorem with Bound C as a corollary; amends Invariant D to include the CONFLICT+CONFLICT
clause the code already enforced; withdraws both false claims; replaces the unreachable abort test
with a standalone unit binary over `Types.h`; and resolves every remaining fixture format from
source into a layout table.

The decision: proceed to `code_v0` on the unreviewed plan_v5, or extend to `p6` for a sixth review.

## Superseded Blocking Reason

plan_v6 is the terminal plan version under the extended `p6` rounds and is unreviewed.

Six review rounds, every finding [major]: 11 -> 6 -> 6 -> 5 -> 5 -> 5.

Round 6 (plan_v5_review) confirmed J2 fixed and B3's truth table matching Invariant D exactly. Its
five findings were: an exact-arithmetic hole in the Closure proof reachable only at 2^53 / 2^64
voxels (a whole brain is ~10^15); Bound C describing recorded evidence but claiming actual voxel
mass; a third false-precision claim in the honesty section; `nuc_can_merge` assigned to two
different files by different sections of the same document; and remaining fixture-format gaps.

plan_v6 fixes all five. It changes NO design semantics - the record, lattice, veto, hierarchy
policy, and contract are exactly plan_v5's. The delta is exact integer arithmetic, three corrected
statements, one file-ownership fix, and a completed fixture table.

Observed trend: the marginal finding rate is flat at 5, and the content has shifted from design
defects (rounds 1-4) to statement precision, physically-unreachable numeric edges, and fixture
minutiae (rounds 5-6) - the last of which is the kind of thing resolved in minutes while writing
the test rather than by specifying it in prose.

The decision: proceed to `code_v0`, or extend to `p7` for a seventh review.

## Coordinator Verification of code_v0

The coordinator independently reproduced the load-bearing checks rather than accepting the
coder's report:

* V2 (default-path bit-invariance, the hard gate): built baseline binaries from `run_start_ref`
  in a throwaway `git worktree`, ran the same fixture under both builds, compared every file the
  baseline produced. **35 identical, 0 differing, 0 missing**; the three modified-only sidecars
  (`ongoing_nuc.data`, `done_nuc.data`, `nuc_cuts.data`) exist and are empty. Matches the coder's
  reported numbers exactly.
* V5/V6 unit binaries: recompiled and re-ran both. `test_nuc_algebra: PASS` (including the
  overflow abort) and `test_nuc_extractor: PASS` with counters `conflict_sv 1 / minority_sv 1 /
  subfloor_sv 1 / subfloor_voxels 10`.
* V3/V4: re-ran all five nucleus modes. `none/same/one` merge; `different` and `max` are vetoed
  with `nuc_cuts` naming `(100, 200)`. The veto fires at affinity 0.97, confirming it is not
  affinity-gated, and `0xFFFFFFFF` behaves as an ordinary id.

Note for review_v0: the coder left **no re-runnable V2 harness** and no baseline build in the
tree, so V2 was reproducible only because the coordinator rebuilt it by hand. That is a gap worth
a finding.

Separately: the live `lib/abiss` advanced during this run to `fb149622` (two commits from the
abiss_speedup line: `ABISS_ARCH`/`ABISS_ALLOCATOR` knobs, and a `volume_backends` URI fix). Our
baseline `3c4f562` is an ancestor, and those commits touch `.gitignore`, `CMakeLists.txt`,
`scripts/volume_backends.py`, `scripts/test_volume_backends.py` - none of which overlap the 13
files changed here, so this work rebases cleanly.

## Approvals

2026-07-31, user (weidf): "do coding". Directed the run to proceed to `code_v0` on plan_v6 without
a seventh plan review. Plan rounds stay at `p6` (plan_v6 is terminal and unreviewed); revision
rounds `c2` remain, so code_v0 -> review_v0 -> code_v1 -> review_v1 -> code_v2 is available.

2026-07-30, user (weidf): chose "Extend to p6, review plan_v5 first". Plan rounds raised from `p5`
to `p6`, adding `plan_v5_review` and permitting `plan_v6`. Revision rounds unchanged at `c2`.

2026-07-30, user (weidf): chose "Extend to p5, review plan_v4 first" at the terminal-plan decision
point. Plan rounds raised from `p4` to `p5`, adding `plan_v4_review` and permitting `plan_v5`.
Revision rounds unchanged at `c2`.

2026-07-30, user (weidf), contract decision at the plan_v3_review impossibility result: chose
"Ship the weaker guarantee". The feature promises **Invariant D** (no merge joins clusters whose
recorded dominant nucleus ids differ), NOT exact cannot-link. Identity sets and NUC_WS are
declined for this run and recorded as the follow-up if the extraction counters show mixed
supervoxels are common on real data. plan_v4 implements that decision.

2026-07-30, user (weidf): chose "Extend to p4, review plan_v3 first" at the second terminal-plan
decision point. Plan rounds raised from `p3` to `p4`, adding `plan_v3_review` and permitting
`plan_v4`. Revision rounds unchanged at `c2`.

2026-07-30, user (weidf): chose "Extend to p3, review plan_v2 first" at the terminal-plan
decision point. Plan rounds raised from `p2` to `p3`, which adds `plan_v2_review` and permits
`plan_v3`. Revision rounds unchanged at `c2`.

## Superseded Blocking Reason

plan_v3 is the terminal plan version under the extended `p3` rounds, and it is unreviewed.

Three review rounds, every finding [major], each round finding real defects:

* plan_v0_review - 11 findings. Root cause: a merge-time voxel threshold legalized clusters
  holding two different nucleus ids, destroying record exactness and associativity.
* plan_v1_review - 6 findings, with a reachable counterexample proving "Invariant N" false, plus
  the catch that `std::pair<seg_t, nuc_record_t>` is not a wire-format contract.
* plan_v2_review - 6 findings: a false idempotence claim, a second route to the two-nucleus merge
  via sub-floor evidence accumulating across supervoxels, and the observation that resolving a
  conflicting remap to TOP detects an invalid coalescence rather than preventing it.

plan_v3 replaces the absolute evidence floor with a dominance ratio (yielding Bound C, a provable
contamination bound), aborts at the three sites where a conflict can only mean a broken veto,
allows CONFLICT to absorb NONE, restores the full uint32 id domain via a separate state field, and
rewrites the hierarchy tests with concrete commands.

Findings per round: 11 -> 6 -> 6. Narrowing, but not converged.

The decision: proceed to `code_v0` on the unreviewed plan_v3, or extend to `p4` for one more
review round.

## Superseded Blocking Reason

plan_v2 is the terminal plan version under `p2`, and it has not been reviewed.

Both completed review rounds found real, invariant-level defects rather than polish:

* plan_v0_review: 11 findings, all [major]. Root cause: a merge-time `min_voxel_threshold`
  legalized clusters holding two different nucleus ids, destroying both record exactness and
  associativity.
* plan_v1_review: 6 findings, all [major], with a concrete reachable counterexample proving
  "Invariant N" did not hold, plus a correct catch that `std::pair<seg_t, nuc_record_t>` is not
  a wire-format contract.

plan_v2 responds with a substantial redesign — a three-state flat lattice (BOTTOM / proper id /
TOP) with TOP acting as a merge barrier, a single shared `nuc_join`, an explicit `nuc_wire_t`
with static_asserts, and a rewritten hierarchy test plan. Per the normal-mode table, an
unreviewed terminal plan version is a human decision point.

The decision: proceed to `code_v0` on the unreviewed plan_v2, or extend the plan rounds
(`p3-c2`) so codex reviews plan_v2 first.


## Task Summary

Accept `NUC_PATH` (uint32 nucleus instance segmentation) alongside the existing `AFF_PATH`
and `SEM_PATH` inputs; accumulate a fixed-width per-supervoxel nucleus record
`(sid, id, count, total)`; propagate it through the distributed chunk hierarchy exactly as
`ongoing_semantic_labels.data` is propagated; and add an affinity-ungated cannot-link veto
in `mean_aggl.cpp` that refuses any merge between two clusters holding different nucleus
ids, logging refusals to a dedicated cut file. With `NUC_PATH` absent, behavior must be
bit-identical to baseline. Scope is the abiss repo only; the pytorch_connectomics wrapper is
not touched.

## Git Baseline

run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

## Workflow State

current_stage: review_v0
latest_artifact: artifacts/review_v0.md
latest_verdict: NEEDS_CHANGES
next_action: code_v1

## Status

active
=========================== END run.md ===========================

=========================== BEGIN artifacts/code_v0.md (your previous implementation) ===========================
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
=========================== END artifacts/code_v0.md ===========================

=========================== BEGIN artifacts/review_v0.md (THE REVIEW TO ADDRESS) ===========================
# Review v0

## Summary

Reviewer: claude (planner, in-session). Raw notes: `state/review_v0.review.raw.md`.

The implementation is substantively correct. I read the full tracked diff against
`run_start_ref` file by file plus the untracked `src/seg/NucExtractor.hpp`, and independently
re-ran the load-bearing verification rather than accepting `code_v0.md`:

* **V2, the Seuron bit-identity gate — reproduced.** Built baseline binaries at `3c4f5621` in a
  throwaway `git worktree` with the same flags, ran the same fixture under both builds:
  **35 identical, 0 differing, 0 missing**, and exactly the three modified-only sidecars
  (`ongoing_nuc.data`, `done_nuc.data`, `nuc_cuts.data`) present and size 0. Matches the coder's
  reported numbers exactly.
* **V5/V6 — recompiled and re-ran both unit binaries.** `test_nuc_algebra: PASS` including the
  checked-overflow abort; `test_nuc_extractor: PASS` with `conflict_sv 1 / minority_sv 1 /
  subfloor_sv 1 / subfloor_voxels 10`.
* **V3/V4 — re-ran all five nucleus modes.** `none`/`same`/`one` merge; `different` and `max` are
  vetoed with `nuc_cuts` naming `(100, 200)`. The fixture's affinity is ~0.97, so the veto is
  demonstrably **not** affinity-gated, and `0xFFFFFFFF` behaves as an ordinary id.

The six review rounds paid off in the code: `nuc_is_dominant` uses exact `__uint128_t` arithmetic
(K1a), `nuc_add` is checked (K1b), `nuc_can_merge` lives in `Types.h` where the unit binary can see
it (K4), the sub-floor branch leaves `total == 0` (J1), the frozen-edge condition at
`mean_aggl.cpp:686-687` was correctly left alone, and the README carries Invariant D's three
clauses, the `99/1` counterexample, and Bound C's corrected scope.

Two majors and four minors below. **None is a correctness defect in the nucleus feature itself** —
the majors are a new default-path failure mode and a verification-reproducibility gap.

## Diff Baseline

run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f

Reviewed: `git diff 3c4f5621` (13 tracked files, 484 insertions, 4 deletions) plus the untracked
`src/seg/NucExtractor.hpp`. `HEAD` verified equal to `run_start_ref`; no commits were created.

## Findings

**F1 [major] — nucleus env vars are parsed on the default path.**
`src/seg/atomic_chunk_ME.cpp:104-107` evaluates `nuc_ratio_from_env()` and
`nuc_min_tagged_from_env()` when constructing the extractor, *outside* the
`std::filesystem::exists("nuc.raw")` guard. Both abort on a malformed value. So a stale or
mistyped `ABISS_NUC_DOMINANCE` left in the environment now kills `acme` on a run that has **no
nucleus input at all** — a failure mode that did not exist before this change. V2 cannot catch it
because V2 runs with a clean environment. Move both calls inside the `exists` branch, or default
them silently when no nucleus source is present.

**F2 [major] — no re-runnable V2 harness.**
`work/test/` contains `make_fixture.py`, `run_atomic_fixture.sh`, and the per-feature tests, but
nothing that builds the baseline at `run_start_ref` and performs the comparison. V2 is the gate
protecting Seuron provenance bit-identity; I could only confirm it by reconstructing the baseline
worktree, the build, and the file-by-file comparison by hand. That is precisely the step a future
change will skip. Add `work/test/run_v2_invariance.sh` that does the whole thing and prints the
identical/differing/missing counts.

**F3 [minor] — collision counters over-count.**
`load_nuc` (`mean_aggl.cpp`), `reduce_nuc` (`reduce_chunk.cpp`), and `process_nucs`
(`match_chunks.cpp`) each increment whenever the *result* of `nuc_join` is CONFLICT, so a third
record joining an already-CONFLICT accumulator increments again. These counters are the designated
observable for detecting a veto failure (plan_v6 B6 and Q2); inflating them makes that signal
harder to read. Increment only on the transition into CONFLICT.

**F4 [minor] — default-path log noise.**
`NucExtractor::output` unconditionally prints four `nuc:` counter lines, and `load_nuc` prints
`nuc: no nucleus labels`, on every chunk even with no nucleus input — roughly 600 extra lines on a
143-shard whole-volume run. Gate the prints on the extractor having a source.

**F5 [minor] — `load_nuc` aborts on a nucleus sid missing from `seg_indices`.**
This mirrors `load_sem` (`mean_aggl.cpp:255-266`), so it is consistent with existing behavior and
not a regression. But it turns a data/config mismatch into a hard abort mid-run. Worth confirming
that the ongoing/done partition keeps the two sets aligned in the **composite** path, where
`seg_indices` is built from `ongoing_supervoxel_counts.data` rather than `ns.data`.

**F6 [minor] — mixed-binary pipelines break at the merge.**
`merge_chunks_me.py:60` now merges `ongoing_nuclei_labels` unconditionally, so a run whose earlier
chunks were produced by a pre-change binary has no such child files and the merge fails. Out of
scope to fix; it belongs in the README as an upgrade note.

## Tests to Add

1. **`work/test/run_v2_invariance.sh`** (addresses F2) — baseline worktree at `run_start_ref`,
   matching cmake flags, fixture under both builds, comparison over exactly the baseline-produced
   file set, plus the empty-sidecar assertion. This is the single most valuable missing artifact.
2. **A default-path environment test** (addresses F1) — run `acme` with no `nuc.raw` and
   `ABISS_NUC_DOMINANCE=garbage` set; it must succeed, not abort.
3. **A counter-transition test** (addresses F3) — three records for one sid, two of which already
   force CONFLICT, asserting the collision counter reports 1 rather than 2.
4. **A composite-path `load_nuc` test** (addresses F5) — exercise `agg` at the composite level with
   `ongoing_supervoxel_counts.data` as the ns file and a nucleus payload, confirming no abort.

## Questions

1. F1 and F4 are both "the feature changes behavior when it is switched off". Is there any other
   place the nucleus path is reachable with no `NUC_PATH` configured? The three readers all use
   `filesize() == 0` and degrade correctly, which is right, but `acme`'s env parsing shows the
   pattern is not applied uniformly.
2. F5's composite-path question is the one thing I could not settle by reading. Does the
   ongoing/done partition guarantee every sid in the merged `ongoing_nuclei_labels.data` appears in
   the merged `ongoing_supervoxel_counts.data`?

## Verdict

VERDICT: NEEDS_CHANGES
=========================== END artifacts/review_v0.md ===========================

Now triage the review, fix the accepted findings, verify, and write work/abiss/code_v1.md.
