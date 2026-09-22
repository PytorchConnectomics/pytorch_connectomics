# Plan v1 Review

## Summary

Codex (coder) reviewed `plan_v1.md`. Verdict: **READY: no**. It confirmed 4 prior major
findings + both minors resolved, but judged 8 prior majors still partly open plus several
new gate-definition problems, so the plan was not yet implementable with trustworthy
G0/G1 pass/fail. Full reviewer output: `state/plan_v1_review.review.raw.md`. (Coordinator
note: this artifact was reconstructed after the stage was initially committed out of order;
its content faithfully summarizes the recorded raw transcript.)

## Findings

Faithful summary of the raw transcript (no material finding softened):

Resolved by plan_v1: harness anchors (+0.545); predicted-SDT 0.985 removed; one G1
starting map bound; runtime/verdict semantics; read-only prior art; second chunk deferred.

Open [major] after plan_v1:
- Canonical G0 pipeline still under-specified: medial-band construction, coordinate
  conversion, edge states, exact offset banks.
- Skip rasterization was removed rather than made collision-safe (foreign-band traversal
  audit then impossible).
- Perturbation protocol not reproducible: C1/C2 side/section definitions, C3 distribution
  + threshold, RNG reset; fragments/GT + retained length must use physical length (I2).
- G1 bridge/event definition, endpoint neighborhoods, hyperedges, dedup, matching undefined.
- Proposal cap logic: raw max 9–12 wrongly allowed to pass; two-endpoint arbitration undefined.
- Realizer trivially passable (seeds = supervoxels = BASE_SEG); coverage + repaired-seed gates absent.
- Exact affinity asset not bound (basename only; key/axes/halo/channel mapping missing).
- Indexed 43-channel design only promised.
- Verification not resource-grounded for a cheap gate (450 M-voxel watersheds, per-bridge
  full-score calls) — needs a bounded/incremental strategy.
- Absolute Phase-1 gates (0.752/0.742) meaningless on a crop scoring 0.836 — declare a
  scale-appropriate local gate or label G1 a recall-only surrogate.

## Questions

- Is G1 meant to certify full Phase 1, or only a center-crop recall/realizer surrogate?
- What exact affinity file/key/axes/channel mapping produced the 0.836 anchor?

## Verdict

VERDICT: NEEDS_CHANGES
