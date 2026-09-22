# Plan v0 Review

## Summary

Codex (`gpt-5.6-sol`, `--sandbox read-only`) reviewed `plan_v0.md` against `task.md` and returned
**READY: no** with **11 major and 1 minor** findings. Raw transcript:
`state/plan_v0_review.review.raw.md` (6,099 bytes). The repository mutation guard (`git diff` and
`git diff --cached` captured immediately before and after the call) showed no change, and the CLI
exited 0.

The verdict is well-founded and this coordinator does not soften it. Three findings identify real
design defects in the plan rather than presentation problems:

- **Territory-ID preallocation is self-contradictory.** The plan moved id allocation into
  `nuccomp_scan` so flood tasks would consume rather than allocate ids — but the specification
  says the *largest* territory retains the parent watershed id, and scan cannot know which
  territory is largest before flooding. The determinism mitigation therefore does not close.
- **The Slurm chain is not submittable as described.** `submit_wholevol_sharded.sh` cannot know
  the flood array's bounds until `nuccomp_scan` has written `units.json`, and no dynamic
  submission mechanism was specified. The zero-unit case (a real outcome — the r10 arm) cannot be
  expressed as an 8–9 element array at all.
- **The bit-identity gate is both inconsistent and insufficient.** It cannot be "bit-identical"
  while also adding instrumentation keys, `zero_repairs`, and newly sorted serialization; and
  manifest equality says nothing about whether `terr_<unit>.npz` holds identical voxel labels,
  which is the thing that actually reaches the segmentation.

Two further findings are about the plan's relationship to its own task rather than its mechanics,
and the coordinator accepts both: the plan omitted deliverables `task.md` §7 explicitly requested
(the watershed-reuse recommendation, explicit R1–R7 answers, a ship-readiness verdict), and it
deferred *all* acceptance work when at least the unchanged-material and rejected-edge checks are
proper regression gates for a refactor that claims to preserve behaviour.

The reviewer's answer to Q-3 is a material correction to the verification plan: per-run bit
identity is achievable against each run's own frozen parameters, but only if the existing
serializer is preserved — otherwise the gate must honestly be renamed *normalized semantic
equality*. Calling it bit-identity while changing serialization would be the same category of
self-deception this feature has already been bitten by twice.

## Findings

All 11 major and 1 minor findings are preserved verbatim in
`state/plan_v0_review.review.raw.md`. Grouped by what plan_v1 must do:

**Design defects to resolve before implementation**

1. [major] Territory-ID preallocation versus the largest-territory-keeps-parent-id rule.
2. [major] Dynamic Slurm array bounds unknown at submit time; zero-unit case unrepresentable.
3. [major] Bit-identity gate internally inconsistent and blind to `terr_*.npz` contents.
4. [major] Stale-artifact hazards on retry: an old per-unit record could satisfy fan-in, and a
   stale manifest could remain visible when a merge fails. Needs run/config identity, plan
   digests, atomic writes, and an exact expected artifact set.
5. [major] The proposed B2 test lacks an oracle — both paths agreeing on an *unsplit* result would
   pass it. It must assert the expected competitive labels survive the real remap path.

**Scope and sequencing**

6. [major] Section B is conditional on measurement with no decision threshold or revision
   checkpoint; the "shard by nucleus range" alternative is not concrete enough to implement.
7. [major] Scope omits `task.md` §7 deliverables and converts a review/design task into
   implementation without reconciling the conflict.
8. [major] Deferring all acceptance work is too broad.
9. [major] R3, R5, R6 unaddressed; R4's non-equivalence must constrain any ship verdict.
10. [major] R7 accounting is wrong — five abort conditions were claimed but only four tests
    proposed, omitting invalid nucleus coordinates, and the existing missing-manifest test is not
    a substitute for one of the five.

**Verification**

11. [major] Efficiency acceptance is incomplete: no expected speedup per change, an uninstrumented
    prior run cannot supply the "before" phase table, and the comparison must be end-to-end
    critical path rather than summed array-task time.
12. [minor] `ru_maxrss` units are platform-dependent and per-process; after sharding the report
    must distinguish scan, per-flood-task, merge, max-task and concurrent aggregate.

## Questions

The reviewer answered its own Q-1..Q-3, and the coordinator adopts all three for plan_v1:

- **Q-1** — keep `nuccomp_scan` single for the instrumentation milestone; do not commit to either
  scan sharding or the flood array until the phase table justifies one. The open branch is
  precisely what makes plan_v0 unexecutable.
- **Q-2** — `units.json` should be a separate immutable artifact with a schema version,
  input/config fingerprints, a plan digest, canonical unit ordering and the exact expected output
  set; publish the manifest only after validated completion.
- **Q-3** — per-run bit identity is valid against each run's own original parameters and frozen
  inputs; the two runs must never be compared to each other. Exact byte identity additionally
  requires preserving the existing serializer.

Open for the planner, not the reviewer: whether the CCC task string ("refactor and improve")
overrides `task.md` §7's review-shaped deliverables, or whether plan_v1 must deliver both. The
coordinator's reading is that the task string authorizes implementation and `task.md` §7 remains
owed — so plan_v1 should carry the R1–R7 answers and the ship verdict as written deliverables
rather than dropping them.

## Verdict

VERDICT: NEEDS_CHANGES
