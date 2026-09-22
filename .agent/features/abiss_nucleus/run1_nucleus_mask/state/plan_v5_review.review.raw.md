## Summary

Not ready. J2 is fixed: B3’s truth table matches Invariant D’s three clauses exactly, and I3/I4 remain intact. J1 is fixed only for ordinary, non-overflowing arithmetic. J3, J4, and J5 still have concrete gaps.

## Findings

[major] **J1 is not fully fixed: Closure fails at the declared numeric limits.** Two counterexamples:

- Extraction with `dominance_ratio = 1.0`, `total = 2^53 + 1`, and `count = 2^53`: conversion of `total` to `double` rounds it to `2^53`, so the comparison passes even though the exact inequality `count >= total` is false.
- Let both operands be `PROPER(id1, count=17·2^59, total=9·2^60)`. Each has ratio `17/18`. Their uint64 additions wrap, producing `PROPER(id1, count=2^60, total=2^61)`, whose ratio is `0.5`. This can occur in `load_nuc`, reduce/match collisions, or merge propagation.

The proof needs checked addition and an exact comparison strategy, or explicit enforced bounds. Validation should also explicitly reject non-finite dominance values such as `NaN`.

[major] **Bound C still falsely describes actual voxel mass.** Consider separately extracted records:

```text
A: 49 voxels id2 -> NONE(0,0)
B: 50 voxels id1 -> PROPER(id1,50,50)
```

Their permitted merge records `PROPER(id1,50,50)`, so `total-count == 0`, while the resulting cluster actually contains 49 minority voxels. Closure may hold over the deliberately retained “usable” evidence, but Bound C does not bound all minority voxels in the cluster. It must be limited explicitly to recorded usable evidence.

[major] **The new clause-3 justification contains another false honesty claim.** Refusing `CONFLICT + CONFLICT` does have an over-segmentation cost. Two adjacent CONFLICT clusters with a mergeable edge become one segment if allowed and remain two under B3. Clause 3 may remain, but “no additional over-segmentation” and “no benefit” must be removed.

[major] **J4’s standalone seam is internally contradictory.** V5 compiles a binary that includes only `Types.h` and calls `nuc_can_merge`, and Changes Since says the predicate lives there. But B3 and the Files table assign `nuc_can_merge` to `mean_aggl.cpp`, while A1–A3 never add it to `Types.h`. Following the proposed file changes makes V8 unable to see the production predicate. Its header ownership must be stated consistently.

[major] **J5 remains incomplete for an unfamiliar implementer.** Specific gaps include:

- No specified layout or explicit empty-file instruction for `o_incomplete_edges_<tag>.tmp` and `o_ongoing_supervoxel_counts.data`.
- No format for `ongoing.data`/`localmap.data`, although T4 creates that mapping and the final child-to-composite resolution may require it.
- No content or format for `chunk_offset.txt`.
- T2 lists files but gives no concrete matching/remap records that cause its two nucleus sids to canonicalize onto one sid, nor identifies which ancillary inputs must be empty.
- T4 does not state how original child ids pass through `localmap.data` before applying the composite `remap.data`, or explicitly establish that this mapping is unnecessary.

The table improves J5 substantially, but it still does not define a reproducible fixture.

## Questions

1. Should numeric overflow abort, or will the implementation enforce and document a smaller maximum aggregate?
2. Are child segment ids guaranteed to remain directly addressable in the composite `remap.data`? If not, T4 must specify the `localmap.data` step.

READY: no