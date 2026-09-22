# Plan v0 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, `xhigh`), read-only sandbox. Raw transcript:
`state/plan_v0_review.review.raw.md`.

Not executable safely as written. Nine findings, all tagged major by the reviewer, none softened
here. Two are structural rather than fixable-in-place: the reducibility argument has a concrete
counterexample, and the proposed ordering operates within a single threshold-ladder rung while the
bug's competing edges can sit on different rungs. The rest are bookkeeping, harness and
gate-specification defects that would each have produced a silently wrong result.

## Findings

* **[major] The reducibility argument is false.** With `A` PROPER, `B`/`C` NONE, equal-area links
  `w(A,C)=0.2` and `w(B,C)=0.9`: before merging, the lexicographic maximum is `(true, 0.2)`; after
  the permitted `A+B` merge, `A u B` is PROPER with mean linkage `0.55`, giving `(true, 0.55)` --
  greater than both prior keys. PROPER monotonicity therefore does not establish Ran reducibility.
  The distributed-correctness constraint in `task.md` is unmet.
* **[major] Deferred membership is never consumed.** After promotion an edge stays in the deferred
  set, so a later sweep of the same PROPER survivor can emplace it a second time; the wrappers keep
  only the newest handle and the orphaned heap entry then has a key that `plus` mutates without an
  `update()`, corrupting fibonacci-heap ordering.
* **[major] The proposed liveness test is not exact.** `incident[e->v0].at(e->v1).edge == e`
  neither confirms the reverse wrapper nor that both agree on edge and handle, and `at()` throws
  when the forward slot is absent.
* **[major] The plan implements `(threshold bucket, phase, affinity)`, not the claimed global
  `(has_proper_id, affinity)`.** `agglomerate_cc` is invoked separately at every ladder step, so a
  shell-shell edge eligible at 0.8 merges in phase 2 before a shell-to-own-nucleus edge becomes
  eligible at 0.7. V0's comparison against the final 0.25 threshold cannot detect this.
* **[major] The frozen-boundary claim is unsupported and contradicted by the supplied code.** When
  an edge touches a frozen endpoint both endpoints are marked frozen, so reordering earlier merges
  can change which composite segment becomes frozen and is handed upward. "Only the interior order"
  is invalid.
* **[major] No distributed hierarchy verification.** V1 tests algebra and V3 a single atomic
  fixture; neither compares monolithic execution against child chunks plus parent aggregation, nor
  validates `ongoing_nuc.data` propagation.
* **[major] V3 does not meet the stated determinism criterion.** Two identical runs prove
  repeatability for one insertion order only, and it relies on a phase-tagged log no proposed
  change creates.
* **[major] Retargeting `run_v2_invariance.sh` alone makes its sidecar assertion fail.** At
  `312bf54` the nucleus sidecars exist in the baseline too, so they can no longer be the three
  expected `current_only` files.
* **[major] The empirical gates are not objectively specified.** "Small tolerance", "mostly below
  threshold", dominance "does not regress", size-distribution "collapse" and control reproduction
  have no formulas or pass/fail values; nucleus 275's control dominance is already 0.897, so a
  literal ~0.90 floor is ambiguous.

## Questions

1. Must nucleus priority dominate affinity across the complete threshold ladder, or only within
   each `agglomerate_cc` invocation?
2. Is exact chunk independence mandatory? If so the lexicographic policy needs redesign, because
   the stated reducibility premise has a counterexample.
3. What exact contamination tolerance and permitted per-nucleus dominance delta define success?

## Verdict

VERDICT: NEEDS_CHANGES
