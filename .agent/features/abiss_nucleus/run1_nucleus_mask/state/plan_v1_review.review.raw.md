## Summary

Plan v1 is still not executable safely. F1, F5–F7, F10, and F11 are fixed. F8 fixes the id width but leaves the enclosing wire layout underspecified. F2–F4 remain unresolved, and F9 is only partially addressed. Most importantly, Invariant N does not hold.

## Findings

- [major] **F2 remains unresolved: abstention does not establish Invariant N.** A supervoxel containing above-floor ids 1 and 2 emits `id=0`, but the underlying cluster still contains both ids. It can then merge with a cluster carrying id 3 because `nuc_can_merge` treats zero as unconstrained; B4 changes its record to id 3. The resulting segment physically contains ids 1, 2, and 3 while recording only 3. This is a reachable counterexample to the hard cannot-link guarantee. Abstention avoids choosing the wrong dominant id, but it does not make the fixed record exact or remove conflicting evidence. `NUC_WS`, an exact set representation, or a distinct conflict state with appropriately restrictive merge behavior is needed.

- [major] **F3 remains unresolved: B2’s combination rule is not associative.** The plan calls abstention absorbing, but implements zero as an identity:
  `0 ⊕ x = x`. For records `a=id1`, `b=id2`, and `c=id2`,
  `(a ⊕ b) ⊕ c = 0 ⊕ c = id2`, while
  `a ⊕ (b ⊕ c) = a ⊕ id2 = 0`.
  Thus remap grouping or input order changes the recorded id and later veto behavior. The representation conflates “no evidence” with “conflicting evidence,” so the required associative operation cannot be implemented as written.

- [major] **F4 remains unresolved: the “pure sid remaps” can be semantic many-to-one cluster merges.** The plan explicitly admits that duplicate sids and conflicting ids can arise from remapping. C1/C2 would then emit two records such as `(sid=K,id=1)` and `(sid=K,id=2)`: one logical cluster already carries two nonzero ids without B3 ever seeing a merge edge. `load_nuc` abstaining afterward cannot undo that coalescence. The induction proof must cover every many-to-one remap, either by proving such conflicts unreachable or vetoing them before the mapping is accepted.

- [major] **F9 is only partially fixed.** T3 starts with a hand-built final `vetoed_edges.data`, bypassing the actual C3 path that appends `nuc_cuts.data`, merges per-chunk streams, and produces that input. T1/T2 only test byte-preserving remaps and do not exercise conflicting many-to-one mappings. T4—the only proposed final hierarchy test—is allowed to fail and is not specified through the complete hierarchy workflow. A successful executable test must cover the actual cut producer/append/merge/consumer chain and a multi-level case with different nucleus ids.

- [major] **A4’s size “assert” is ineffective in the specified Release build.** Release builds commonly define `NDEBUG`, removing C/C++ `assert` checks. A malformed `nuc.raw` could therefore still be mmaped and silently misread. This needs an unconditional runtime size check and a direct malformed-file test, not only validation through `cut_chunk_agg.py`.

- [major] **The stable on-disk layout remains underspecified.** Although `nuc_record_t` is declared packed and 20 bytes, the hierarchy uses raw `std::pair<seg_t,nuc_record_t>` records. That pair has implementation-defined layout and likely trailing padding; the packed record also contains unaligned `uint64_t` members. The plan does not state the exact pair size/offsets or how the hand-built V7 fixtures encode it. Define an explicit wire record with static size/offset assertions and use it consistently at every producer and consumer.

## Questions

1. Should an above-floor mixed supervoxel require `NUC_WS`, become a permanent conflict/barrier state, or force an exact multi-id payload?
2. Can the planner prove that every hierarchy remap is incapable of mapping different recorded ids onto one sid? If not, where is that remap vetoed before the ids are coalesced?

READY: no