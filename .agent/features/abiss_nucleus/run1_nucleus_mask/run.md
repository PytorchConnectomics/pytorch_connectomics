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

## Coordinator Verification of code_v2 (final)

code_v2 is the terminal code version under `c2` and the round structure provides no `review_v2`.
Rather than close on the coder's report, the coordinator re-verified both majors independently:

* **G2 (correctness).** `extra_nuc.data` is written by `generate_extra_nucs`
  (`match_chunks.cpp:347,385`, called at 547) and appended at `composite_chunk_me.sh:47`, beside
  the pre-existing `extra_sv_counts.data` restoration — the symmetry the finding identified as
  missing. Ran `work/test/test_hierarchy_binaries.py` directly:
  `V10 G2 boundary survival: reduce->match restored ids 1/2; composite agg vetoed 100/200`.
  That asserts survival of the veto across the reduction, not merely sid-set alignment, which was
  the exact gap in code_v1's evidence. The coder also reports that investigation confirmed no
  other survival route existed, so the finding was real rather than a false alarm.
* **G1 (usability).** `NUC_RATIO`/`NUC_OFFSET` wired through `set_env.py:38-40` and
  `cut_chunk_agg.py:113-115`. The upsample reproduces em_seg's
  `(z + ratio/2) // ratio` formula on all three axes with an explicit xyz/zyx conversion, and
  selects integer indices via `numpy.ix_`, so instance ids are never interpolated. Out-of-range
  mappings raise with the computed span.
* **V2 gate** re-run cold by the coordinator: `identical=35 differing=0 missing=0`,
  empty sidecars, `default-path nucleus log lines: 0`, stale-env PASS.

No findings remain open. G3 was documentation-only by user direction and is in the README.

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

2026-07-31, user (weidf): "do it" — directed review_v1 to fold in the lib/em_seg mask-resolution
gap as a major finding (G1), and to keep the must-link soma snap OUT of this run as a documented
limitation plus a follow-up (G3). code_v2 is the last revision available under `c2`.

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

current_stage: code_v2
latest_artifact: artifacts/code_v2.md
latest_verdict: NEEDS_CHANGES
next_action: complete

## Status

complete
