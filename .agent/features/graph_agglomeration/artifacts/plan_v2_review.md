# Plan v2 Review

## Summary

Reviewer: Codex (gpt-5.6-sol, ultra), read-only. Attested summary of
`state/plan_v2_review.review.raw.md`. Codex confirms **5 of the 6 prior [major]
findings are resolved** and the design coherent (total namespace, exactly-once
remap, centralized marker stage, quarantine/transitive firewall, `tau_iou_min=0`,
single `eps=1e-4` oracle gate). It returns `READY: no` on **one remaining
[major]** — the operating-point scoring is still under-specified (a circular
`ambiguity` term and an unpinned caliber floor) — plus four [minor] precision
items. plan_v2 is the last allowed plan version (p2), so this is a normal-mode
decision point rather than another plan revision.

## Findings

- [major] **Scoring under-specified / `ambiguity` circular.** The declared `score`
  subtracts `ambiguity`, but `ambiguity` is defined from competing *pair scores*
  which already include the subtraction — circular. Must define a separate **raw**
  score, and specify competitor exclusion, cross-endpoint aggregation,
  no-competitor behavior, clip/division guards, and whether mutual-best ranks raw
  or final. Also (same finding) S5 requires a "caliber floor" with no threshold or
  flag — pin `tau_caliber_min` or remove the floor.
- [minor] The no-edge control validates sampled-evaluator equivalence but cannot
  *prove* collision-freedom (sparse GT, aggregate values). Add **structural**
  zero-union asserts: exactly one distinct nonzero id per enumerated node,
  background fixed at 0, complete remap-lookup coverage.
- [minor] S4 drops non-mutual candidates while S5/S6 promise rejected/ambiguous
  edges in the certificate — retain **all generated candidates** for
  certification; allow only mutual-best into union processing.
- [minor] The marker gate is qualitative ("close"/"large shortfall") and the
  fallback wording weakens the pinned source. Define a **numeric** pass criterion
  and forbid automatic fallback for this run.
- [minor] Make crossing-schema validation an explicit hard step: assert all 600
  crossing files have the required keys and consistent row lengths before linking.

## Questions

- All findings are precise, non-controversial specification fixes (Codex stated
  exactly what to specify). A concrete non-circular resolution is proposed to the
  coordinator/user (raw score for ranking + mutual-best + `tau_score`; ambiguity
  as a separate *defer* gate `ambiguity > tau_amb`; caliber floor removed —
  caliber enters only via the weighted raw score). The decision is whether to
  carry these into `code_v0` as pinned implementation requirements or hold.

## Verdict

VERDICT: NEEDS_CHANGES
