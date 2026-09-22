# Plan v1 Review

## Summary

Codex reviewed `plan_v1.md` against `task.md`, `plan_v0.md` and its own prior review, and returned
**READY: no** with **12 major and 2 minor** findings. Raw transcript:
`state/plan_v1_review.review.raw.md` (6,865 bytes). Mutation guard clean, CLI exit 0.

The reviewer's own accounting of its previous findings: **#2, #3, #5 and #12 genuinely resolved**,
**#8 substantially addressed**, and **#1, #4, #6, #7, #9, #10, #11 partial or unresolved**. It
added two new findings (#13 launcher behaviour, #14 wording).

The central objection is not mechanical, and the coordinator accepts it: plan_v1 narrowed the
increment to instrumentation plus tests, so **it measures efficiency without improving it**, while
"more efficient" is central to the task string. The reviewer offered two acceptable exits —
explicit user approval to narrow the task to a measurement milestone, **or** a measurement
checkpoint followed by a concrete, selected optimization inside this run.

The second exit is available without spending the user's time, because the reviewer's answer to
Q-2 makes the measurement cheap: re-running `nuccomp` alone against an already-completed
watershed is a valid baseline (the watershed is nucleus-blind, so rebuilding it would add cost and
unrelated variance), provided the rerun uses frozen fingerprinted inputs, a fresh output
namespace, read-only reference artifacts, and records critical-path elapsed time. `plan_v2` will
therefore take that route rather than block.

Three findings are worth flagging as substantive rather than editorial:

- **#1 (partial)** — moving the parent-id assignment to an overlay-time mapping fixed the
  contradiction but created a new artifact-representation contract in `nucleus_overlay.py`, a file
  plan_v1 excluded from scope, and no test requires sharded output to equal serial output
  voxel-for-voxel.
- **#4 (unresolved)** — "removes *or* refuses to overwrite" is two incompatible policies, and
  refusing leaves a stale manifest visible after a failed merge. Path and size are also too weak
  as input fingerprints.
- **#10 (incomplete)** — the plan promised tests for abort conditions without auditing whether the
  production checks exist at all; tests alone cannot create a guarantee.

## Findings

All 12 major and 2 minor findings are preserved verbatim in
`state/plan_v1_review.review.raw.md`. Carried into `plan_v2`:

**Resolved, no further action:** #2 (fixed-capacity array), #3 (gate covers territory artifacts,
serializer preserved, per-run comparison — with the added requirement of a separate output
namespace so the rerun cannot overwrite its own oracle), #5 (B2 oracle adequate), #12 (RSS
semantics).

**Still open, must be closed by plan_v2:**

1. [major] #1 — overlay translation contract undesigned; `nucleus_overlay.py` excluded from scope;
   no voxel-level sharded-equals-serial test.
2. [major] #4 — single publication policy required (quarantine then publish after validation);
   fingerprints must use content or authoritative artifact identities covering watershed manifest,
   nucleus input, affinity/cost input, configuration and code version; `stage_report.json` needs
   atomic publication and defined failure semantics.
3. [major] #6 — the scan-dominant branch is not an executable design; the 40%/60% boundary can
   select both branches; deferring all optimization fails the task's efficiency requirement.
4. [major] #7 — §7 deliverable 2 (ordered efficiency plan with expected speedup and determinism
   risk per change) is missing; the watershed-reuse deliverable must answer *how* to repair
   `CHUNKMAP_INPUT` precedence, not merely promise a recommendation.
5. [major] #8 — omits the "already-clean nuclei unchanged" half of acceptance item 4; does not say
   how reference rejected-edge counts are established; gates need named inputs, outputs and
   failure criteria.
6. [major] #9 — R3/R5/R6 need planned work, not a promised "position": R3 emit `local_only` unless
   containment is demonstrable; R5 trace ordering inputs and test invariance; R6 test or document
   whether local scoring excludes cross-chunk merges, else retain the null as a ship blocker; R4
   must qualify every cross-arm conclusion.
7. [major] #10 — reachability unstated for an incompatible watershed manifest; must audit whether
   each abort check exists in production and add validation where it does not.
8. [major] #11 — no efficiency acceptance criterion for an increment containing no optimization;
   scan branch lacks expected speedup and determinism risk.
9. [major] #13 — `submit_wholevol_sharded.sh` runs before the result exists and cannot surface a
   future zero-repair outcome; the stage should emit the durable flag and a separate
   acceptance/reporting step should judge it.
10. [minor] #14 — "committed as" contradicts the no-commits-during-a-run rule; say "written" or
    "produced".

## Questions

The reviewer answered both of plan_v1's open questions; the coordinator adopts both:

- **Q-1** — keep `docs/nucleus_competition_review.md` in the run folder as the reviewed source of
  truth, explicitly *because* `dev/` has no recoverable git history. The lesson file may later
  summarize or link to it, but must not be the sole copy.
- **Q-2** — re-running `nuccomp` alone against an existing completed watershed is the appropriate
  baseline. Requirements: frozen fingerprinted inputs and parameters, a fresh output namespace,
  reference artifacts kept read-only, critical-path elapsed time recorded, and a separate baseline
  for **each** of the two reference parameterizations (`ABISS_NUC_MIN_TAGGED` 50 and 1024).

Open for the planner: `plan_v2` is the final plan version under `p2-c2`. If it does not clear
review, the protocol requires this run to block for a human decision in `normal` mode.

## Verdict

VERDICT: NEEDS_CHANGES
