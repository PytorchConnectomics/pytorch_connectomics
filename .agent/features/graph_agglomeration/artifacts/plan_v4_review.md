# Plan v4 Review

## Summary

Reviewer: Codex (gpt-5.6-sol, ultra), read-only. Attested summary of
`state/plan_v4_review.review.raw.md`. Codex confirms the best-other-partner
ambiguity formula and the one-union-per-undirected-pair canonicalization are
correct and that **no other blocker is apparent**. `READY: no` on two
internal-consistency items only (introduced by the plan_v3→v4 edits, not design
problems).

## Findings

- [major] **Verification 3(a) not propagated.** The S1 namespace fix scoped
  node-wise distinct-id injectivity to the zero-union control, but Verification
  Plan step 3(a) still states "each enumerated node → one distinct nonzero id"
  unconditionally — as a hard check it would fail after any successful union.
  Scope 3(a) to the zero-union remap and state the root-wise assertions
  (`R(u)==R(v) ⇔ find(u)==find(v)`, distinct ids across roots) for linked output.
- [minor] **Certificate enum missing `same_root_noop`.** S6's decision enum cannot
  represent a `find(u)==find(v)` no-op; add `same_root_noop` (or define how it is
  recorded) so it is not misclassified as accepted/rejected.

## Questions

None — both fixes are exact and internal-consistency-only. plan_v5 (rounds p4→p5)
will apply them; the approach and executability are approved.

## Verdict

VERDICT: NEEDS_CHANGES
