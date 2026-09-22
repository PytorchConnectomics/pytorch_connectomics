# Plan v3 Review

## Summary

Reviewer: Codex (gpt-5.6-sol, ultra), read-only. Attested summary of
`state/plan_v3_review.review.raw.md`. Codex states the plan is "close" and confirms
the prior scoring/marker/schema/candidate revisions are resolved and the pipeline,
paths, dtype, and `--remap` eval coherent. It returns `READY: no` on **two [major]**
precise corrections plus one [minor]. All three are exact, non-controversial fixes
Codex dictated; the approach is fully approved.

## Findings

- [major] **Ambiguity competitor formula.** "Second-highest raw among other
  partners excluding `b`" can skip `a`'s strongest alternative. Pin it as
  `comp_a = max({raw(a,x) | x ≠ b}, default=0)` (best *other* partner; the runner-up
  overall when `e` is `a`'s top), symmetrically `comp_b`. Changes defer decisions.
- [major] **Structural distinct-ID assertion contradicts unions.** Requiring a
  distinct id per node breaks after the first accepted union (unions intentionally
  share an id). Scope node-wise injectivity to the **zero-union control** only. For
  linked output assert: total nonzero mapping, `R(u) == R(v) ⇔ find(u) == find(v)`,
  and distinct ids across distinct roots.
- [minor] **Directed vs undirected candidate.** "All candidates (both directions)"
  leaves ambiguous whether a physical edge is processed once or twice. Canonicalize
  each undirected pair once for the single union attempt (define same-root no-op);
  directed certificate records are fine but there is one union attempt per pair.

Codex confirms resolved: raw ranking non-circular, endpoint aggregation/guard/clip
specified, caliber floor removed, numeric marker gate with no fallback, 600-crossing
schema hard-checked, all candidates retained.

## Questions

None outstanding — the three fixes are exact. plan_v4 (plan rounds bumped p3→p4)
will pin them, then one final plan review.

## Verdict

VERDICT: NEEDS_CHANGES
