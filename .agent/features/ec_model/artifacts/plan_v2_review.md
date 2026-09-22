# Plan v2 Review

## Summary

Codex (coder role, `codex exec --sandbox read-only`, `TOKIO_WORKER_THREADS=4`) reviewed plan_v2, the final allowed plan version under p2, against the plan_v1 review, using only the prompt contents. Raw transcript: `state/plan_v2_review.review.raw.md` (READY: no).

Prior-finding status:
- Resolved: 7 of 8 (classification completeness, eligibility-based support, B-guard bypass, alignment exclusions, mining/sampler edge cases, per-type audit counts, overfit pair coverage).
- Partially resolved: #4 (evaluation choices). The bootstrap clustering does not keep dependent observations together.

That leaves one major finding and four minor ones. Mode is normal and no plan version remains, so the unresolved major finding blocks the run for a human decision; the verdict below is not softened.

## Findings

- [major] Bootstrap clusters do not preserve shared-neuron dependence. With pair → min(k, j), pairs (1,100) and (2,100) land in different clusters even though they share neuron 100, and ordinary sites seeded by 100 land in a third. Repeated candidate fragments also add correlated observations under different seeded-neuron clusters. The stated 95% intervals are therefore unjustified. Reviewer fix: freeze dependency groups that keep shared neurons/fragments and both pair members together, then bootstrap those groups. Report the number of independent groups; with too few groups, report descriptive metrics without a CI. Add tests with overlapping pairs and a candidate repeated across seeded neurons.
- [minor] B access should be reserved before inference: an exclusive `STARTED` record is created before loading B data and finalized with the verdict, so a crash still consumes access. Test concurrent invocations and failure after reservation.
- [minor] Seed-swap examples need an explicit exception to owner-label consistency (M=0). Separate each member's seed-fragment label from the conditioning-owner field, or mark swaps owner-free, and validate seed-fragment/GT consistency per member.
- [minor] Overfit construction assumes optional populations exist (contamination/controls, 8 thin-site pairs). Use available examples with deterministic replacements, and report a missing prerequisite (fewer than 8 eligible pairs) directly rather than as a model-learning failure.
- [minor] Two mining statements need reconciliation. "Edge midpoint node (u)" is ambiguous: choose the midpoint or endpoint u and test it. The contamination test must require the whole-crop fallback to succeed, with the site unseedable only when the dominant part is absent from the entire crop.

## Questions

- None from the reviewer.
- Coordinator note for the human decision: the major finding concerns confidence intervals only. All gate thresholds in plan_v2 are point estimates, so a narrow resolution is available. Drop CIs from gate logic; report CIs only as descriptive spatial-block bootstrap (blocks aligned to non-overlapping crop-size cells within A or B, which keeps co-located pairs, repeated candidates and shared neurons in one block); report the block count and omit CIs below 20 blocks. Adopting this, plus the four minor fixes, needs your direction because no plan version remains.

## Verdict

VERDICT: NEEDS_CHANGES
