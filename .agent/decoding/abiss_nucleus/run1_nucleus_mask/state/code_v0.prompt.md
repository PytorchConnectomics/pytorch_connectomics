You are the CODER for a CCC run. Implement plan_v6 (included below in full).

You planned nothing here; a planner wrote plan_v0..v6 and you reviewed v0..v5 across six rounds.
plan_v6 is the terminal plan and is authoritative. Follow it. If you find something in it that is
actually wrong while implementing, implement the correct thing and say so explicitly in the
"Risks and Unknowns" section of your artifact - do not silently deviate.

## Hard constraints

1. WORK ONLY IN THIS DIRECTORY:
       /projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_nucleus/work/abiss
   This is an isolated git clone of lib/abiss detached at 3c4f562.
   NEVER touch /projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss - it is the live checkout
   and SLURM jobs execute binaries from its build/ directory. A previous run broke a running job by
   rebuilding it in place.
2. DO NOT create git commits. Leave all changes in the working tree.
3. DO NOT add runtime or build dependencies.
4. Behavior with no NUC_PATH configured must be bit-identical to baseline. This repo reproduces a
   Seuron provenance record. V2 in the plan is the check; it is a hard failure if it does not pass.
5. Do not claim a command passed unless you actually ran it and saw the output. A step you did not
   run is reported as NOT RUN. This is more important than appearing complete.

## What to produce

Implement Phases A-D, then run the verification plan V1-V10 as far as it goes.

Then write this file (exact path):

    /projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_nucleus/artifacts/code_v0.md

with EXACTLY these level-2 sections, in this order, each present exactly once:

    # Code v0
    ## Overview
    ## What Changed
    ## Implementation Details
    ## Files Changed
    ## Git Baseline
    ## Verification
    ## Review Focus
    ## Risks and Unknowns
    ## Changes Since Previous Code Version

Requirements on that file:
- "## Files Changed" must contain a markdown table with header exactly:
      | File | Purpose |
      |---|---|
- "## Git Baseline" must contain these two lines:
      run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f
      current_head: <output of `git rev-parse HEAD` in work/abiss>
- "## Changes Since Previous Code Version" must contain exactly the text:
      Initial implementation.
- "## Verification" must quote REAL command output for every check, and state plainly which of
  V1-V10 ran, which passed, which failed, and which did not run and why.
- Summarize the ACTUAL diff, not your intent.

## Priorities if you run short on time

Land in this order, and report honestly where you stopped:
  1. Phase A + B (types, nuc_join/nuc_can_merge/nuc_is_dominant/nuc_add, NucExtractor,
     atomic_chunk_ME, mean_aggl veto+propagation+file lifecycle)
  2. V1 (build) and V2 (default-path bit-invariance) - V2 is the one that must not be skipped
  3. V5/V8 (the standalone test_nuc_algebra unit binary - cheap and covers Closure + Invariant D)
  4. Phase C + C5 plumbing, then V3/V4/V6/V7
  5. V9 T1-T4 (hierarchy binaries)
  6. Phase D (README)

If T4 genuinely cannot be run, say so with the exact failing command and its stderr. Do not describe
it as passing.

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

current_stage: plan_v6
latest_artifact: artifacts/plan_v6.md
latest_verdict: NEEDS_CHANGES
next_action: code_v0

## Status

active
=========================== END run.md ===========================

=========================== BEGIN artifacts/plan_v6.md (AUTHORITATIVE) ===========================
# Plan v6

## Summary

Add a `NUC_PATH` input to ABISS carrying a **uint32 nucleus instance segmentation**, used as a
dominant-id cannot-link constraint during agglomeration, under the contract the user fixed on
2026-07-30.

Plan_v6 changes five things relative to plan_v5, all from `plan_v5_review`. Two are arithmetic
hygiene in the Closure proof (K1), two are statements that were stronger than the truth (K2, K3),
one is an internal contradiction about where a function lives (K4), and one completes the fixture
specification (K5). **No design semantics change.** The record, the lattice, the veto, the
hierarchy policy, and the contract are all as in plan_v5.

The properties this plan rests on, stated correctly this time:

> **Invariant D.** No merge performed by agglomeration (1) joins two clusters whose recorded
> dominant nucleus ids differ, (2) joins a CONFLICT cluster to a cluster carrying a recorded
> dominant id, or (3) joins two CONFLICT clusters.
>
> **Closure.** Every reachable record satisfies `state == PROPER => count*den >= num*total` (exact
> integer form of the dominance ratio) and `state != PROPER => count == 0`.
>
> **Bound C — scope corrected (K2).** In a PROPER cluster, the *recorded usable* minority evidence
> is `total - count <= (1 - dominance_ratio) * total`. **This does not bound the cluster's actual
> minority voxels.** Sub-floor supervoxels are absorbed as NONE and contribute real voxels that no
> record accounts for; that mass is reported only in aggregate by `nuc: subfloor_voxels` and is not
> bounded per cluster.
>
> **What Invariant D does not promise.** Minority identities are not tracked at all, at any
> contamination level. `A: 99 id1 + 1 id2 -> PROPER id1` and `B: 99 id1 + 1 id3 -> PROPER id1` may
> merge, joining identities 2 and 3. A mixed supervoxel does not necessarily become CONFLICT:
> `60 id1 + 40 id2` is PROPER id1 at the default ratio, and any supervoxel below `min_tagged` is
> NONE however mixed.

## Scope

**In scope** (`lib/abiss`, in the isolated clone `work/abiss`): record and wire types; `nuc_join`
and `nuc_can_merge`; `NucExtractor`; propagation through the distributed hierarchy including
OVERLAP=2 veto feedback; the veto in `mean_aggl.cpp`; `NUC_PATH` plumbing; documentation carrying
the contract verbatim; executable tests.

**Out of scope, decided by the user:** identity sets and `NUC_WS` — the follow-up if the counters
show mixed or sub-floor supervoxels are common on real data. Also out: `NUC_MIP` (replaced by the
C5 shape assertion), the pytorch_connectomics wrapper, mask generation, perinuclear-shell tagging,
and `scripts/reduce_chunk.py` / `scripts/match_chunks.py`, which are **dead legacy**
(`overlap_chunk_me.sh:44` and `composite_chunk_me.sh:42` invoke the C++ binaries).

## Proposed Changes

### Phase A — types, algebra, extraction

**A1. `src/seg/Types.h`** — beside `using semantic_t = uint8_t;` (line 25):

```cpp
using nuc_t = uint32_t;                       // voxel dtype of nuc.raw; 0 == background

enum : uint8_t {
    NUC_STATE_NONE     = 0,   // no usable evidence; count == 0 AND total == 0
    NUC_STATE_PROPER   = 1,   // count*den >= num*total
    NUC_STATE_CONFLICT = 2,   // tagged, no dominant id; count == 0
};

struct __attribute__((packed)) nuc_record_t {
    uint8_t  state = NUC_STATE_NONE;
    uint32_t id    = 0;   // meaningful only when state == PROPER
    uint64_t count = 0;   // voxels backing `id`; 0 unless PROPER
    uint64_t total = 0;   // usable tagged voxels; 0 when state == NONE
};
```

Full uint32 id domain `[1, 0xFFFFFFFF]` usable; only `0` is background. `NONE => total == 0` is
what makes Closure hold; document the property above the struct.

**A2. Wire format** — unchanged from plan_v5:

```cpp
struct __attribute__((packed)) nuc_wire_t {
    seg_t sid; uint8_t state; uint32_t id; uint64_t count; uint64_t total;
};
static_assert(sizeof(nuc_wire_t) == 29);
static_assert(offsetof(nuc_wire_t, sid) == 0 && offsetof(nuc_wire_t, state) == 8);
static_assert(offsetof(nuc_wire_t, id) == 9 && offsetof(nuc_wire_t, count) == 13);
static_assert(offsetof(nuc_wire_t, total) == 21);
```

**A3. The dominance ratio as an exact rational, and checked addition (K1).** Both live in
`Types.h`:

```cpp
struct nuc_ratio_t { uint64_t num = 3, den = 5; };   // default 0.6 == 3/5

// exact: no floating point in the comparison path
inline bool nuc_is_dominant(uint64_t count, uint64_t total, const nuc_ratio_t & r) {
    return static_cast<__uint128_t>(count) * r.den >= static_cast<__uint128_t>(r.num) * total;
}

inline uint64_t nuc_add(uint64_t a, uint64_t b) {      // checked
    if (a > UINT64_MAX - b) {
        std::cerr << "nuc: voxel count overflow: " << a << " + " << b << std::endl;
        std::abort();
    }
    return a + b;
}
```

Plan_v5 compared `count >= dominance_ratio * total` in `double`, which the reviewer showed silently
passes at `total == 2^53 + 1`, and summed `uint64` unchecked, which wraps at `~10^19` voxels. Both
counterexamples are **physically unreachable** — a whole-brain EM dataset is on the order of `10^15`
voxels — so this is proof hygiene, not a data path; but the exact form costs nothing and removes the
hole from the Closure proof rather than arguing around it. `__uint128_t` is a GCC/Clang extension
and this project already requires C++20 with those compilers (`CMakeLists.txt:10`).

`ABISS_NUC_DOMINANCE` is parsed to a rational: reject non-finite values (`NaN`, `inf`) and anything
outside `(0.5, 1.0]`, then set `num = llround(value * 1000)`, `den = 1000`. Reject `NaN` explicitly
rather than relying on comparison behavior, since every comparison with `NaN` is false and a bare
range check would silently accept it.

**A4. `nuc_join`** — in `Types.h`, using checked addition:

```cpp
inline nuc_record_t nuc_join(const nuc_record_t & a, const nuc_record_t & b)
{
    nuc_record_t r;
    r.total = nuc_add(a.total, b.total);
    if (a.state == NUC_STATE_CONFLICT || b.state == NUC_STATE_CONFLICT) {
        r.state = NUC_STATE_CONFLICT;
    } else if (a.state == NUC_STATE_NONE) {
        r.state = b.state; r.id = b.id; r.count = b.count;
    } else if (b.state == NUC_STATE_NONE) {
        r.state = a.state; r.id = a.id; r.count = a.count;
    } else if (a.id == b.id) {
        r.state = NUC_STATE_PROPER; r.id = a.id; r.count = nuc_add(a.count, b.count);
    } else {
        r.state = NUC_STATE_CONFLICT;
    }
    return r;
}
```

**Closure proof, now sound.** Extraction establishes it: NONE gets `(0,0)`; PROPER by
`nuc_is_dominant`; CONFLICT has `count == 0`. `nuc_join` preserves it: a NONE operand contributes
`(0,0)` so the other side's exact inequality carries unchanged; two same-id PROPER operands add both
sides of `count*den >= num*total` (valid because addition is now checked, so no wraparound can
break the inequality); every other combination yields CONFLICT with `count == 0`. ∎

Algebra: the state/id projection is a flat-lattice join — associative, commutative, idempotent on
that projection. `count`/`total` are an additive commutative monoid — associative and commutative,
**not** idempotent. So `nuc_join` is associative and commutative on the full record, which suffices
because every combination site joins records over **disjoint voxel sets**.

**A5. `nuc_can_merge` — declared in `Types.h` (K4).**

```cpp
inline bool nuc_can_merge(const nuc_record_t & a, const nuc_record_t & b)
{
    const bool ca = (a.state == NUC_STATE_CONFLICT), cb = (b.state == NUC_STATE_CONFLICT);
    if (ca && cb) return false;                          // Invariant D clause 3
    if (ca)       return b.state == NUC_STATE_NONE;      // clause 2
    if (cb)       return a.state == NUC_STATE_NONE;      // clause 2
    if (a.state == NUC_STATE_NONE || b.state == NUC_STATE_NONE) return true;
    return a.id == b.id;                                 // clause 1
}
```

Plan_v5 was internally inconsistent here: V5 compiled a unit binary that included only `Types.h` and
called this predicate, while B3 and the Files table placed it in `mean_aggl.cpp`. `Types.h` is the
correct owner — it is a pure inline predicate over the record type, with no agglomeration state —
and `mean_aggl.cpp` calls it. Each branch is annotated with the Invariant D clause it implements so
code and contract cannot drift.

**A6. `src/seg/NucExtractor.hpp` (new).** Modeled on `SemExtractor.hpp`; no class LUT; nullable
source so A7 does not branch.

* Constructor takes `const Chunk *` (may be `nullptr`), `nuc_ratio_t`, `min_tagged`.
* `collectVoxel(Coord c, Tseg segid)`: return if null; else read
  `id = (*m_nuc)[c[0]][c[1]][c[2]]` and, if `id != 0`, increment `m_counts[segid][id]`
  (`MapContainer<Tseg, MapContainer<uint32_t, uint64_t>>`).
* `collectBoundary`, `collectContactingSurface`: empty.

`output(chunkMap, filename)` remaps supervoxel ids through `chunkMap` and merges the id→count maps
of supervoxels sharing a target, as `SemExtractor::output` does (`SemExtractor.hpp:31-51`). Then per
target supervoxel:

```text
tagged := sum of all counts (via nuc_add)
if tagged == 0 or tagged < min_tagged:              (NONE,     -,      0,         0)
(max_id, max_count) := largest count, ties -> smaller id
if nuc_is_dominant(max_count, tagged, ratio):       (PROPER,   max_id, max_count, tagged)
else:                                               (CONFLICT, -,      0,         tagged)
```

`min_tagged` (`ABISS_NUC_MIN_TAGGED`, default **50**) stops a single bleed voxel in an otherwise
untagged supervoxel from becoming PROPER and casting a veto that hard-splits a real cell.
`dominance_ratio` (`ABISS_NUC_DOMINANCE`, default **0.6**, matching
`agglomeration_semantic_heuristic_t::dominant_signal_ratio`, `mean_aggl.cpp:132`) handles bleed
inside an already-tagged supervoxel.

Tie-break by smaller id: `MapContainer` may be `absl::flat_hash_map` (`CMakeLists.txt:42-49`), whose
iteration order is unstable. This argmax runs **once**, over complete evidence; no downstream stage
selects a winner.

Counters, `nuc:`-prefixed: `conflict_sv`, `minority_sv` (PROPER with `total > count`), `subfloor_sv`,
`subfloor_voxels`. The last two are the only record of sub-floor mass — see Bound C's corrected
scope — and are the measurement that decides whether identity sets or `NUC_WS` are worth a follow-up.

`output()` **always creates the file**, writing zero records when the source was null.

**A7. `src/seg/atomic_chunk_ME.cpp`.** The file duplicates the `traverseSegments<1>(...)` call for
the sem-present and sem-absent cases (lines 78-104); a nullable extractor keeps it at two branches.
Declare the `bio::mapped_file_source` for `nuc.raw` in the same scope as `seg_file`/`aff_file` so it
outlives the traversal; if `std::filesystem::exists("nuc.raw")`, open it and build
`ConstChunkRef<nuc_t, 3>` with the same extents and `fortran_storage_order()` as `sem.raw` (lines
79-84). **Unconditional** size check, never `assert`: on mismatch with
`sizeof(nuc_t)*dim[0]*dim[1]*dim[2]`, print expected and actual bytes and `std::abort()`. Add
`nuc_extractor` to both packs; call `nuc_extractor.output(map, "ongoing_nuclei_labels.data")`
unconditionally after the branch. Parse `ABISS_NUC_DOMINANCE` and `ABISS_NUC_MIN_TAGGED` here.

### Phase B — agglomeration

**B1. No new agglomeration parameters.** `agglomeration_param_t` and `heuristics_aff_threshold`
(line 148) untouched; all tuning is at extraction.

**B2. Loading.** Add `std::vector<nuc_record_t> nuc_ids;` to `agglomeration_data_t` (line 156) and
`load_nuc(...)` beside `load_sem` (line 245), called next to line 545 on
`ongoing_nuclei_labels.data`, reading `nuc_wire_t`. Mirror `load_sem`'s index mapping; empty or
missing yields an empty vector (lines 248-251). Duplicate sids combine with `nuc_join`; CONFLICT is
legitimate, because the file is concatenated across chunks and one boundary-spanning supervoxel
contributes one record per chunk over disjoint voxels.

| Site | Operation | Conflict policy |
|---|---|---|
| `NucExtractor::output` | aggregate, resolve (A6) | CONFLICT valid |
| `load_nuc` duplicates | `nuc_join` | CONFLICT valid |
| `reduce_chunk` collision | `nuc_join` | CONFLICT valid; **counted** |
| `match_chunks` collision | `nuc_join` | CONFLICT valid; **counted** |
| merge propagation (B4) | `nuc_join` | **abort** if `!nuc_can_merge(a,b)` |

**B3. The veto.** Call `nuc_can_merge` (A5) in `agglomerate_cc`'s main loop immediately after the
frozen-edge block ending at line 697 and before the semantic check at line 699, guarded by
`if (!nuc_ids.empty())`, with **no affinity gate**. On refusal push the edge to a new
`nuc_rg_vector` in `agglomeration_output_t` (line 167), set `e.edge->w = Limits::min()`, `continue`
— the shape of the semantic refusal at lines 700-706. A nucleus refusal is logged in preference to
a semantic one.

**Clause 3's cost, stated honestly (K3).** Refusing `CONFLICT + CONFLICT` **does** cost
over-segmentation: two adjacent CONFLICT clusters with a mergeable edge stay two segments instead of
one. Plan_v5 claimed there was "no additional over-segmentation" and "no benefit"; both claims are
withdrawn. The clause is kept because merging two clusters that each already mix nuclei compounds
contamination, and CONFLICT clusters are expected to be rare — `nuc: conflict_sv` measures whether
that expectation holds.

**Do not add the nucleus payload to the frozen-edge condition at lines 686-687**, which already
reads `!sem_counts.empty()`; replicating it would make enabling nuclei silently alter frozen-edge
handling.

**B4. Propagation on merge.** Beside the `sem_counts` transform (lines 754-758), same `v0`/`v1`/`s`
swap discipline, guarded by `!nuc_ids.empty()`. Evaluate `nuc_can_merge(a, b)`; if **false**, print
both records and both `seg_indices` values and `std::abort()`. Then combine with `nuc_join`.

This abort is **unreachable through `agg` by construction** — B3 rejects the same pairs earlier. It
is a tripwire against a future edit that breaks that correspondence. V8 tests the predicate directly
and verifies the call site by inspection; it does not claim `agg` can be made to abort on data.

**B5. File lifecycle.** Creation unconditional — `agg` opens and closes `ongoing_nuc.data`,
`done_nuc.data`, `nuc_cuts.data` every run regardless of `nuc_ids.empty()`, and
`NucExtractor::output` always creates `ongoing_nuclei_labels.data`. Required, because the drivers
`mv` these under `set -euo pipefail` and `reduce_chunk.cpp`'s `read_array` aborts on a file it
cannot open. Content conditional on `!nuc_ids.empty()`. Mirror lines 848-857, 880-891, 960-961,
1050-1052.

**B6. Diagnostics.** All `nuc:`-prefixed: the four A6 counters, `load_nuc` duplicate combinations
producing CONFLICT, `reduce_chunk` / `match_chunks` collisions producing CONFLICT. Only B4's
violation aborts, plus `nuc_add`'s overflow guard.

### Phase C — hierarchy and drivers

Unchanged from plan_v5.

**C1. `src/seg/reduce_chunk.cpp`** — nucleus reducer reading `nuc_wire_t`, applying the `sid` remap
from `remap.data`, combining collisions with `nuc_join` and counting those yielding CONFLICT; one
record per sid. Call from line 228 with `ongoing_nuclei_labels_%1%.data` /
`reduced_ongoing_nuclei_labels_%1%.data`. Absent or empty input still produces the output.

**C2. `src/seg/match_chunks.cpp`** — mirror `process_sems` (line 382, called at line 440) with
`process_nucs`, reading `o_ongoing_nuclei_labels.data`, writing `ongoing_nuclei_labels.data`, with
C1's collision handling. Per I4, this site must **not** abort.

**C3. OVERLAP=2 veto feedback** — after the existing `cp` at `overlap_chunk_me.sh:50`, add
`cat nuc_cuts.data >> vetoed_edges_"$output_chunk".data`. `merge_chunks_me.py:62` merges the stream
and `match_chunks.cpp:190-238` removes those edges; the record is the same `(seg_t, seg_t)` pair and
the consumer already sorts and dedups (`match_chunks.cpp:212-214`), so no C++ change is needed.

**C4. Shell drivers.**

* `scripts/atomic_chunk_me.sh` — `touch ongoing_nuclei_labels.data` (line 36 group);
  `mv done_nuc.data ${output_path}/info/nuclei_labels_"$output_chunk".data` (line 50 group);
  `mv nuc_cuts.data ${output_path}/info/nuc_rejected_edges_"$output_chunk".log` (line 53 group);
  `mv ongoing_nuc.data ongoing_nuclei_labels_"$output_chunk".data` (line 60 group).
* `scripts/composite_chunk_me.sh` — the OVERLAP=2 rename at line 40 plus the same four moves at the
  line 88/97 groups.
* `scripts/overlap_chunk_me.sh` — `mv reduced_ongoing_nuclei_labels_...` (line 60), `done_nuc.data`
  (line 64), `nuc_cuts.data` (line 66), and the C3 `cat`.

**C5. Python drivers.**

* `scripts/set_env.py:30` — add `"NUC_PATH"`; not `NUC_MIP`. The loop is `for e in env: if e in
  data:`, so a JSON without `NUC_PATH` exports nothing new; comment it.
* `scripts/cut_chunk_agg.py` — after the `SEM_PATH` block (lines 46-49): `load_data` at
  `AFF_RESOLUTION` with `NUC_FILL_MISSING`, `cut_data` with the same `start_coord`/`end_coord` as
  `seg.raw`, validate, `save_raw_data("nuc.raw", ...)`. Assert the nucleus cutout's spatial shape
  equals `seg.raw`'s. Raise on non-integer dtype, on a negative value in a signed dtype, or on any
  value `> 0xFFFFFFFF`; otherwise `astype(numpy.uint32)` losslessly. The cast is mandatory because
  `save_raw_data` writes `data.dtype` verbatim (`cut_chunk_common.py:38-50`) while the binary mmaps
  `nuc.raw` as `nuc_t`.
* `scripts/merge_chunks_me.py:59`, `scripts/merge_chunks_overlap.py:65` — add
  `"ongoing_nuclei_labels"`.

### Phase D — documentation

`README.md`: the `NUC_PATH` key with dtype, range, alignment contracts; `ABISS_NUC_DOMINANCE` and
`ABISS_NUC_MIN_TAGGED`; the three-state record and Closure; the `nuc:` counters and the follow-up
decision they inform; the `nuc_cuts` / `nuc_rejected_edges` outputs; the B4 abort and the `nuc_add`
overflow guard as tripwires; **Invariant D verbatim with all three clauses, clause 3's real
over-segmentation cost, the `99/1` counterexample, and Bound C with its corrected scope — including
that it does not bound actual minority voxels**; and the caveat that the mask should tag
**perinuclear cytoplasm**, not raw nucleus interiors.

## Files and Areas

| File | Change |
|---|---|
| `src/seg/Types.h` | `nuc_t`, state enum, `nuc_record_t`, `nuc_wire_t` + static_asserts, `nuc_ratio_t`, `nuc_is_dominant`, `nuc_add`, `nuc_join`, **`nuc_can_merge`**, Closure comment |
| `src/seg/NucExtractor.hpp` | **new**; nullable source, exact dominance + `min_tagged`, four counters |
| `src/seg/atomic_chunk_ME.cpp` | mmap `nuc.raw` + unconditional size check, extractor in both branches, env parsing |
| `src/agg/mean_aggl.cpp` | `load_nuc`, veto calling `nuc_can_merge`, `nuc_join` propagation + predicate abort, counters, unconditional file creation |
| `src/seg/reduce_chunk.cpp` | nucleus reducer; `nuc_join` + counter |
| `src/seg/match_chunks.cpp` | `process_nucs`; `nuc_join` + counter; never aborts |
| `scripts/set_env.py`, `cut_chunk_agg.py` | `NUC_PATH`; `nuc.raw` with shape/dtype/range validation |
| `scripts/atomic_chunk_me.sh`, `composite_chunk_me.sh`, `overlap_chunk_me.sh` | nucleus artifacts; `cat nuc_cuts.data >> vetoed_edges_*` |
| `scripts/merge_chunks_me.py`, `merge_chunks_overlap.py` | merge `ongoing_nuclei_labels` |
| `README.md` | Invariant D, Closure, Bound C scope, counters, caveats |
| `work/test/` | fixtures, `test_nuc_algebra.cpp`, hierarchy scripts |

Untouched: `scripts/reduce_chunk.py`, `scripts/match_chunks.py` (dead legacy); `src/ws/*`;
`CMakeLists.txt` (header-only addition needs no target change — confirm, edit only if the build
proves otherwise).

## Verification Plan

**All paths are relative to `work/abiss`, the working directory for every command.**

**Fixture layouts — complete (K5).** Little-endian, `align=False`, all packed:

| File | numpy dtype | size | Source |
|---|---|---|---|
| `ongoing_nuclei_labels*.data`, `ongoing_nuc.data`, `done_nuc.data` | `[('sid','<u8'),('state','u1'),('id','<u4'),('count','<u8'),('total','<u8')]` | 29 | A2 |
| `input_rg.data`, `residual_rg_<tag>.data`, `edges_<tag>.data`, `new_edges.data`, `o_residual_rg.data`, **`o_incomplete_edges_<tag>.tmp`** | `[('s1','<u8'),('s2','<u8'),('aff','<f4'),('area','<u8')]` | 28 | `rg_entry<seg_t,aff_t>`, `Types.h:37-51`; `match_chunks.cpp:144-145` |
| **`remap.data`, `localmap.data`, `ongoing.data`, `ongoing_<tag>.data`, `extra_remaps.data`** | `[('os','<u8'),('ns','<u8')]` | 16 | `std::pair<T,T>`, `split_remap.cpp:62`; `reduce_chunk.cpp:24` |
| `ongoing_segments.data`, `done_segments.data`, `ns.data`, `ongoing_seg_size.data`, **`o_ongoing_supervoxel_counts.data`** | `[('sid','<u8'),('size','<u8')]` | 16 | `size_data_t`; `match_chunks.cpp:342` |
| `boundary_<face>_<tag>.data`, `frozen.data` | `'<u8'` flat | 8 | `BoundaryExtractor.hpp:34-44` |
| `matching_faces.data`, `o_boundary_<i>_<tag>.tmp` | `[('oid','<u8'),('boundary_size','<u8'),('nid','<u8'),('agg_size','<u8')]` | 32 | `matching_entry_t<seg_t>`, `Types.h:28-34` |
| `vetoed_edges.data`, `nuc_cuts.data`, `sem_cuts.data` | `[('v0','<u8'),('v1','<u8')]` | 16 | `mean_aggl.cpp:1050-1052` |
| `chunkmap.data` | `[('k','<u8'),('v','<u8')]` | 16 | `Utils.hpp:53-66` |

**`chunk_offset.txt`** is a text file holding the decimal `ac_offset` and nothing else, as written by
`cut_chunk_agg.py`. **`param.txt`** is three lines: `offset[0..2]`, `dim[0..2]`, `ac_offset`
(`atomic_chunk_ME.cpp:31-34`). **Any ancillary input a test does not exercise must be created empty
rather than omitted**, because `read_array` aborts on a file it cannot open. The coder must confirm
each layout against the cited source before building a fixture and record the confirmation in
`code_v0.md`.

**V1 — build.** `mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8 &&
cd ..`. Clean compile including A2's `static_assert`s. Report any new warning.

**V2 — default-path invariance.** Baseline binaries from `run_start_ref` in a scratch worktree into
a separate build directory; modified binaries alongside. `work/test/make_fixture.py` (fixed seed),
chunk `(64,64,64)`, tag `0_0_0_0`, no `nuc.raw`. Stage and run exactly as `atomic_chunk_me.sh:34-41`
under both builds in separate directories. **The comparison set is exactly the files the baseline
produced**; each byte-identical (`cmp`). Files produced only by the modified build must exist and be
**empty**. Plus: `set_env.py` on a JSON without `NUC_PATH` exports no `NUC_` variable, and
`cut_chunk_agg.py` against a small local HDF5 volume writes no `nuc.raw`.

**V3 — the veto fires.** Two groups tagged 1 and 2, joined by a high-affinity path, ≥2000 tagged
voxels each. Without `nuc.raw`: one segment. With: two segments and `nuc_cuts.data` naming the pair.

**V4 — pass-through.** One group tagged (allowed); both same id (allowed); one tagged `0xFFFFFFFF`
and one `1` (vetoed — proves the full uint32 domain).

**V5 — algebra, Closure, and the K1 edges.** Standalone binary
`work/test/test_nuc_algebra.cpp`, `#include "../../src/seg/Types.h"`:

```bash
g++ -std=c++20 -O1 -I src work/test/test_nuc_algebra.cpp -o work/test/test_nuc_algebra && \
  work/test/test_nuc_algebra
```

Over `{ NONE(0,0,0), PROPER(id1,c,t), PROPER(id2,c,t), CONFLICT(0,0,t>0) }` with several `(c,t)`
satisfying the exact inequality, for all pairs and triples: commutativity and associativity on the
full record including `(id1,id2,id2)` in both groupings; idempotence on the state/id projection only,
with `nuc_join(a,a).count == 2*a.count` asserted; **Closure** on every result. Plus the specific
cases:

* J1: `NONE(0,0,0) + PROPER(id1,50,50) == PROPER(id1,50,50)` — `total` stays 50.
* K1a: `nuc_is_dominant(2^53, 2^53 + 1, {1,1})` is **false** — the exact integer comparison the
  `double` form got wrong.
* K1b: `nuc_add(UINT64_MAX, 1)` aborts (checked via a death test or a separate expected-failure
  invocation).
* `ABISS_NUC_DOMINANCE` parsing rejects `NaN`, `inf`, `0.5`, `1.1`.

**V6 — resolution.** `5000 id1 + 100 id2` → PROPER id1, `total=5100`, `minority_sv` 1;
`500 id1 + 400 id2` → CONFLICT, `total=900`; 10 tagged voxels → NONE with **`total == 0`**,
`subfloor_sv` 1, `subfloor_voxels` 10. Barrier: CONFLICT fails against PROPER id3 and against
another CONFLICT, succeeds with NONE.

**V7 — input contracts.** uint16 accepted at uint32 width; `> 0xFFFFFFFF` raises; float32 raises;
negative in a signed dtype raises; shape mismatch raises. A **truncated `nuc.raw`** makes `acme`
exit nonzero with the A7 message, against the Release build.

**V8 — the B4 tripwire.** `test_nuc_algebra` asserts `nuc_can_merge` false for
`PROPER id1 + PROPER id2`, `CONFLICT + PROPER`, `CONFLICT + CONFLICT`, and true for the permitted
pairs — Invariant D's three clauses exactly. `code_v0.md` states plainly that the B4 abort is
unreachable through `agg` because B3 rejects the same pairs first, and that its call site is
verified by inspection, not execution.

**V9 — hierarchy binaries.** Scripts under `work/test/`, each command commented with the driver line
it replicates.

* **T1 `../build/reduce_chunk 0_0_0_0`** — build `ongoing_nuclei_labels_0_0_0_0.data`, `remap.data`,
  `ongoing_segments.data`, `done_segments.data`, `residual_rg_0_0_0_0.data`,
  `boundary_{0..5}_0_0_0_0.data` per the table; all other consumed files created empty. Assert sids
  remapped and payloads preserved. Then the conflicting case: records `(sid=100, PROPER id1)` and
  `(sid=200, PROPER id2)` with `remap.data` containing `(100,300)` and `(200,300)` must produce one
  record `sid=300, state=CONFLICT` and a collision count of 1 — not a nonzero exit.
* **T2 `../build/match_chunks 0_0_0_0`** — consumes `matching_faces.data`, `o_residual_rg.data`,
  `o_incomplete_edges_0_0_0_0.tmp`, `vetoed_edges.data`, `o_boundary_{0..5}_0_0_0_0.tmp`,
  `o_ongoing_supervoxel_counts.data`, `o_ongoing_seg_size.data`, `o_ongoing_semantic_labels.data`,
  `o_ongoing_nuclei_labels.data` (`match_chunks.cpp:33,144,145,190,250,342,363,384`); every file the
  test does not exercise is created **empty**. **The concrete canonicalization (K5):**
  `matching_faces.data` carries `matching_entry_t` records `(oid=100, boundary_size=10, nid=200,
  agg_size=10)` so face matching maps sid 100 onto 200, and `o_ongoing_nuclei_labels.data` carries
  `(100, PROPER id1)` and `(200, PROPER id2)`. Assert the output holds one record for sid 200 with
  `state == CONFLICT` and the collision counter at 1 — the legitimate boundary-reconciliation case,
  which must not abort.
* **T3 the real cut chain.** Run `agg` on a nucleus-conflicting fixture so it produces
  `nuc_cuts.data`. Attempt the real driver:
  `OVERLAP=2 UPLOAD_CMD="cp -r" DOWNLOAD_CMD="cp -r" FILE_PATH=<local> IO_SCRATCH_PATH=<local>
  BIN_PATH=<abs build> SCRIPT_PATH=<abs scripts> bash scripts/overlap_chunk_me.sh
  work/test/run/chunk_a.json`, where the JSON carries the keys `chunk_utils.read_inputs` reads:
  `bbox` (6 ints), `ac_offset`, `boundary_flags` (6), `mip_level`, `indices`, `neighbours`. If it
  cannot run, record the exact failing command and its stderr in `code_v0.md`, then execute the
  command sequence extracted from lines 44-66 and `diff` the extracted block against those lines to
  prove no drift. Report which path ran. Then `python3 scripts/merge_chunks_me.py
  work/test/run/composite.json ""` and `../build/match_chunks 0_0_0_0`, asserting the vetoed edge is
  absent from the region graph.
* **T4 two-chunk end-to-end, mandatory.** `work/test/run_hierarchy.sh`, two `(64,64,64)` chunks with
  different nucleus ids under `work/test/run/{chunk_a,chunk_b}`, tags `0_0_0_0` and `0_1_0_0`. Per
  chunk, mirroring `atomic_chunk_me.sh:34-60`: `acme param.txt <tag>`;
  `mv edges_<tag>.data input_rg.data`; six `cat boundary_*` lines;
  `touch ns.data ongoing_semantic_labels.data ongoing_nuclei_labels.data ongoing_seg_size.data`;
  `agg 0.25 input_rg.data frozen.data ns.data`; `cat remap.data >> localmap.data`;
  `split_remap chunk_offset.txt <tag>`; `assort <tag> ""`;
  `mv residual_rg.data residual_rg_<tag>.data`;
  `mv ongoing_segments.data ongoing_supervoxel_counts_<tag>.data`;
  `mv ongoing_nuc.data ongoing_nuclei_labels_<tag>.data`.

  Composite under `work/test/run/composite`, mirroring `composite_chunk_me.sh:37-70`:
  `python3 scripts/merge_chunks_me.py work/test/run/composite.json ""`;
  `mv ongoing.data localmap.data`; `mv residual_rg.data input_rg.data`; `meme 0_0_0_1 ""`;
  `cat new_edges.data >> input_rg.data`; six `cat boundary_*` lines;
  `agg 0.25 input_rg.data frozen.data ongoing_supervoxel_counts.data`. `composite.json` carries
  `mip_level: 1`, `children` naming the two atomic tags, and the union `bbox`.

  **Final assertion, with the id-resolution step spelled out (K5, reviewer question 2).** Child
  supervoxel ids are **not** assumed directly addressable in the composite `remap.data`. Build the
  transitive map: load `localmap.data` (`[('os','<u8'),('ns','<u8')]`) and the composite
  `remap.data` (same layout), compose them in that order, and follow each mapping to a fixed point.
  Resolve every supervoxel of each tagged group through that composed map and assert the two
  groups' representatives **differ**. If composing turns out to be unnecessary because
  `localmap.data` is the identity for these ids, assert that explicitly rather than skipping the
  step. **If T4 cannot be run, the run is `Status: blocked` — not a passing run with a caveat.**

**V10 — driver syntax.** `bash -n` every modified shell script.

Every claim in `code_v0.md` quotes real command output; a step not run is reported as not run.

## Risks and Questions

**R1 — the shipped guarantee is Invariant D**, with the `99/1` counterexample verbatim in the README.
User-approved contract, not an oversight.

**R2 — Bound C does not bound actual minority voxels (K2).** It bounds recorded usable evidence
only. Sub-floor supervoxels absorbed as NONE contribute unaccounted real voxels; that mass appears
only in aggregate as `nuc: subfloor_voxels`. Stated in the README.

**R3 — `min_tagged = 50` and `dominance = 0.6` are unmeasured.** The counters make both visible on
the first real run.

**R4 — three tripwires now abort:** B4's predicate violation and `nuc_add`'s overflow guard. Both are
unreachable on real data — the overflow needs `~10^19` voxels against `~10^15` in a whole brain — so
they convert a future silent corruption into a loud failure rather than adding a runtime risk.

**R5 — V2 proves invariance on a synthetic fixture, not real EM data.** A Seuron provenance re-run
is out of scope and must be stated in `code_v0.md`.

**R6 — T3's real-driver path may still be unrunnable**; the plan requires attempting it, reporting
the exact blocker, and proving the fallback matches by `diff`.

**R7 — memory.** `MapContainer<Tseg, MapContainer<uint32,uint64>>` allocates an inner map per tagged
supervoxel; negligible when nuclei cover a small voxel fraction. Worth a code comment.

**Q1 — is `min_tagged = 50` right?** Both failure directions are counted, so the first real run
answers it.

**Q2 — should the hierarchy collision counters carry per-occurrence detail?** Counts alone cannot
distinguish legitimate boundary reconciliation from a hierarchy defect. Deferred; an env-gated id
list is the natural follow-up.

## Changes Since Previous Plan Version

Responds to `plan_v5_review.md`. All five findings accepted. The reviewer confirmed J2 fixed and
I3/I4 intact; those designs carry forward unchanged. **No design semantics changed in this
revision** — the record, lattice, veto, hierarchy policy, and contract are as in plan_v5.

* **K1 (Closure fails at numeric limits) — fixed with exact arithmetic.** A3 replaces the `double`
  comparison with `nuc_is_dominant`, an exact `__uint128_t` integer form `count*den >= num*total`
  over a rational ratio, and adds `nuc_add`, a checked addition that aborts on `uint64` overflow.
  The reviewer's `total = 2^53 + 1` case is now decided exactly, and the wrapping-addition case
  cannot occur. `ABISS_NUC_DOMINANCE` parsing explicitly rejects `NaN` and `inf`, which a bare range
  check would have silently accepted. V5 tests both counterexamples directly. Recorded honestly:
  these thresholds are physically unreachable — a whole-brain EM dataset is on the order of `10^15`
  voxels against `2^53 ≈ 9·10^15` and `2^64 ≈ 1.8·10^19` — so this closes a hole in the *proof*
  rather than a hole in the pipeline. Answers reviewer question 1: overflow aborts.
* **K2 (Bound C misdescribes actual mass) — scope corrected.** Bound C is now stated explicitly as
  bounding **recorded usable evidence**, with an explicit sentence that it does **not** bound the
  cluster's actual minority voxels, because sub-floor supervoxels absorbed as NONE contribute real
  voxels no record accounts for. The reviewer's `NONE(49 id2) + PROPER(id1,50,50)` case is exactly
  this, and `nuc: subfloor_voxels` is named as the only place that mass is visible. R2 and the
  README carry the correction.
* **K3 (clause-3 justification contains another false claim) — withdrawn.** "No additional
  over-segmentation" and "no benefit" are removed. B3 now states the real cost — two adjacent
  CONFLICT clusters with a mergeable edge stay two segments — and keeps the clause on the narrower
  grounds that merging two already-mixed clusters compounds contamination, with `nuc: conflict_sv`
  measuring how often it bites. This is the third false-precision claim the reviewer has caught in
  this document's own honesty section; each has been withdrawn rather than defended.
* **K4 (the unit-test seam contradicts the file layout) — fixed.** `nuc_can_merge` is now declared
  in `src/seg/Types.h` (A5) beside `nuc_join`, which is where a pure inline predicate over the
  record type belongs, and `mean_aggl.cpp` calls it. B3, the Files table, and V5/V8 all now agree,
  so the unit binary can see the production predicate.
* **K5 (fixtures still not reproducible) — completed.** The layout table adds
  `o_incomplete_edges_<tag>.tmp` (`rg_entry`, `match_chunks.cpp:145`),
  `o_ongoing_supervoxel_counts.data` (`size_data_t`, `match_chunks.cpp:342`), and
  `localmap.data` / `ongoing.data` / `ongoing_<tag>.data` (`std::pair<seg_t,seg_t>`,
  `split_remap.cpp:62`), plus `chunk_offset.txt`'s content and a blanket rule that unexercised
  ancillary inputs are created **empty** rather than omitted. T1 and T2 now give concrete records —
  T2 supplies the `matching_entry_t` that actually canonicalizes sid 100 onto 200 and the two
  nucleus records that collide there. T4 gains the missing `cat remap.data >> localmap.data` step
  and an explicit transitive resolution through `localmap.data` then the composite `remap.data`,
  answering reviewer question 2 without assuming child ids are directly addressable.
=========================== END artifacts/plan_v6.md ===========================

=========================== BEGIN artifacts/plan_v5_review.md (your last review; plan_v6 answers it) ===========================
# Plan v5 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, reasoning effort `xhigh`, sandbox read-only). Raw transcript:
`state/plan_v5_review.review.raw.md`.

`READY: no`, five findings, all `[major]`. Reviewer verdict: "J2 is fixed: B3's truth table matches
Invariant D's three clauses exactly, and I3/I4 remain intact. J1 is fixed only for ordinary,
non-overflowing arithmetic. J3, J4, and J5 still have concrete gaps."

Confirmed fixed: **J2** — B3 now implements exactly Invariant D's three clauses, no more and no
less. I3 and I4 remain intact.

The character of the findings has shifted. K1 is a genuine hole in the Closure proof, but its two
counterexamples require `2^53` and `~10^19` tagged voxels — respectively about 9 quadrillion and 10
quintillion, against roughly `10^15` voxels in a whole-brain EM dataset. It is a proof-hygiene
defect with a cheap fix, not a reachable failure. K2 and K3 are statement-precision defects: two
more places where a claim is stronger than the truth. K4 is a real internal contradiction with a
one-line fix. K5 is fixture minutiae.

All findings accepted.

## Findings

**K1 [major] — Closure fails at the declared numeric limits.** Two counterexamples:

* Extraction with `dominance_ratio = 1.0`, `total = 2^53 + 1`, `count = 2^53`: converting `total`
  to `double` rounds it to `2^53`, so the comparison passes even though the exact inequality
  `count >= total` is false.
* Two operands each `PROPER(id1, count = 17·2^59, total = 9·2^60)`, ratio `17/18`. Their `uint64`
  additions wrap, producing `PROPER(id1, count = 2^60, total = 2^61)`, ratio `0.5`. Reachable in
  `load_nuc`, in reduce/match collisions, or in merge propagation.

The proof needs checked addition and an exact comparison strategy, or explicit enforced bounds.
Validation should also reject non-finite dominance values such as `NaN`.

**K2 [major] — Bound C still falsely describes actual voxel mass.** For separately extracted
records `A: 49 voxels id2 -> NONE(0,0)` and `B: 50 voxels id1 -> PROPER(id1,50,50)`, their permitted
merge records `PROPER(id1,50,50)`, so `total - count == 0` — while the resulting cluster actually
contains 49 minority voxels. Closure holds over the deliberately retained *usable* evidence, but
Bound C does not bound all minority voxels in the cluster. It must be limited explicitly to
recorded usable evidence.

**K3 [major] — the new clause-3 justification contains another false honesty claim.** Refusing
`CONFLICT + CONFLICT` *does* have an over-segmentation cost: two adjacent CONFLICT clusters with a
mergeable edge become one segment if allowed and remain two under B3. Clause 3 may stand, but "no
additional over-segmentation" and "no benefit" must be removed.

**K4 [major] — J4's standalone seam is internally contradictory.** V5 compiles a binary that
includes only `Types.h` and calls `nuc_can_merge`, and Changes Since asserts the predicate lives
there — but B3 and the Files table assign `nuc_can_merge` to `mean_aggl.cpp`, and A1-A3 never add it
to `Types.h`. Following the proposed file changes leaves V8 unable to see the production predicate.
Header ownership must be stated consistently.

**K5 [major] — J5 remains incomplete for an unfamiliar implementer.** Specific gaps:

* No layout or explicit empty-file instruction for `o_incomplete_edges_<tag>.tmp` and
  `o_ongoing_supervoxel_counts.data`.
* No format for `ongoing.data` / `localmap.data`, although T4 creates that mapping and the final
  child-to-composite resolution may require it.
* No content or format for `chunk_offset.txt`.
* T2 lists files but gives no concrete matching/remap records that actually cause its two nucleus
  sids to canonicalize onto one sid, nor says which ancillary inputs must be empty.
* T4 does not state how original child ids pass through `localmap.data` before the composite
  `remap.data` is applied, nor establish that this step is unnecessary.

"The table improves J5 substantially, but it still does not define a reproducible fixture."

## Questions

1. Should numeric overflow abort, or will the implementation enforce and document a smaller maximum
   aggregate?
2. Are child segment ids guaranteed to remain directly addressable in the composite `remap.data`?
   If not, T4 must specify the `localmap.data` step.

## Verdict

VERDICT: NEEDS_CHANGES
=========================== END artifacts/plan_v5_review.md ===========================

Now implement plan_v6 and write code_v0.md as specified.
