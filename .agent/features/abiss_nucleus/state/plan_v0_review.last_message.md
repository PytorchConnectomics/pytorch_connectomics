## Summary

The plan is not executable safely as written. The heap bookkeeping can create duplicate live handles, the proposed reducibility argument has a concrete counterexample, and the verification plan does not prove distributed correctness or input-order independence.

## Findings

- [major] The reducibility argument is false. Let `A` be PROPER and `B,C` be NONE, with equal-area links `w(A,C)=0.2` and `w(B,C)=0.9`. Before merging `A+B`, the lexicographic maximum is `(true, 0.2)`. After the permitted merge, `A∪B` is PROPER and mean linkage to `C` is `0.55`, producing `(true, 0.55)`, which is greater than both prior keys under the lexicographic order. Thus PROPER monotonicity does not establish Ran reducibility; promoting a previously non-priority affinity contribution can increase the linkage key. The hard distributed-correctness constraint is therefore unmet.

- [major] Deferred membership is never consumed. After promotion, the edge remains in the deferred set. A later sweep of the same PROPER survivor can emplace it again because the three eligibility checks do not test whether it already has a valid handle. The wrappers then retain only the newest handle while an orphaned heap entry remains. A later `plus` updates only the recorded handle, silently leaving the orphan with a changed key and corrupting Fibonacci-heap ordering. The plan needs a single explicit outstanding-deferred state transition—normally erase on promotion/admission—and must prove one live handle per edge.

- [major] The proposed liveness test is not exact. `incident[e->v0].at(e->v1).edge == e` neither confirms the reverse wrapper nor checks that both wrappers agree on edge and handle. It can also throw if the forward slot is absent. Promotion/admission needs non-throwing lookups of both directions, matching edge pointers, consistent invalid handles before insertion, and atomic write-back of the new handle to both wrappers.

- [major] The plan implements `(threshold bucket, phase, affinity)`, not the claimed global lexicographic `(has_proper_id, affinity)` order. `agglomerate_cc` is invoked separately at every threshold-ladder step. A shell-shell edge eligible at `0.8` can therefore merge in phase 2 before a shell-to-own-nucleus edge becomes eligible at `0.7`. V0 comparing links only against the final `0.25` threshold does not detect this failure mode. The intended relationship between phases and the outer threshold ladder must be settled before implementation.

- [major] The frozen-boundary claim is unsupported and contradicted by the supplied code. When an edge touches a frozen endpoint, both endpoints are marked frozen. Reordering earlier merges can change the cluster that later encounters that boundary, so it can change which composite segment becomes frozen and is handed upward. The statement that phase ordering changes “only the interior order” is therefore invalid.

- [major] There is no distributed hierarchy verification. V1 tests nucleus algebra, while V3 exercises a single atomic fixture. Neither compares monolithic execution with child-chunk output plus parent aggregation, validates `ongoing_nuc.data` propagation, nor covers frozen-boundary interactions. Given the reducibility counterexample, a real hierarchical comparison is mandatory and may expose a fundamental design blocker.

- [major] V3 does not meet the stated determinism criterion. Two identical runs prove repeatability for one insertion order; comparing only phase membership after permutation does not prove merge ordering or output independence. The plan expressly declines output equality for affinity ties. It also relies on a “phase-tagged log” that no proposed code change creates. The opt-in path needs a stable tie-breaker or the success criterion must be explicitly narrowed.

- [major] Retargeting `run_v2_invariance.sh` alone will make its sidecar assertion fail. At baseline `312bf54`, the nucleus feature already creates the nucleus sidecars, so they should occur in both baseline and current outputs rather than as the three `current_only` files expected by the carried-over script. The harness must be updated to compare/assert empty sidecars appropriately for this new baseline.

- [major] The empirical gates are not objectively specified. “Small tolerance,” “mostly below threshold,” dominance “does not regress,” size-distribution “collapse,” and control reproduction lack exact formulas and pass/fail values. In particular, nucleus 275 already has dominance `0.897`, so a literal `~0.90` floor is ambiguous. These values must be fixed before examining treatment results to prevent a silent or post-hoc pass.

## Questions

1. Must nucleus priority dominate affinity across the complete threshold ladder, or only within each `agglomerate_cc` invocation?
2. Is exact chunk independence mandatory? If so, the lexicographic policy needs redesign because the stated reducibility premise has a counterexample.
3. What exact contamination tolerance and permitted per-nucleus dominance delta define success?

READY: no