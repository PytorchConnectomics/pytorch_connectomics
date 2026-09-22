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
