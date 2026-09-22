## Summary

Not ready. G5 and G6 are fixed. G2’s original associativity failure is fixed for canonical records, and G1’s original above-threshold counterexample is blocked. However, G1 remains leaky below the extraction floor, G3 is not fixed at all, and G4 remains insufficiently executable. The redesign also falsely claims full idempotence, unnecessarily isolates TOP from BOTTOM, and removes one valid `uint32` input value.

## Findings

1. **[major] `nuc_join` is not idempotent as claimed, including both `count` and `total`.** For:

   ```text
   a = (id=1, count=5, total=5)
   ```

   `nuc_join(a, a)` returns `(1, 10, 10)`, not `a`. The ID-state projection is a lattice join, but the complete record is an additive aggregation, not an idempotent lattice. It is associative and commutative for canonical records, but not idempotent. Consequently V5’s required idempotence test either fails on realistic records or accidentally tests only zero-count records and misses the defect. Duplicate copies of the same logical evidence would also double the diagnostics.

2. **[major] G1 is only partially fixed: discarded sub-floor identities permit an actual agglomeration merge containing two nuclei.** With a floor of 100:

   ```text
   A: 60 voxels of id 1 -> NONE, total=60
   B: 60 voxels of id 1 -> NONE, total=60
   C: 100 voxels of id 2 -> id 2
   ```

   Agglomerating A and B produces `NONE, total=120`; the lost id cannot be promoted because decisions ignore `total`. That cluster may then merge with C, producing a cluster containing 120 tagged voxels of nucleus 1 and 100 of nucleus 2. Thus material from distinct nuclei enters one cluster through an agglomeration merge. TOP fixes only conflicts whose IDs individually survive extraction. V4 actually normalizes part of this behavior, and no test covers accumulation of discarded evidence.

3. **[major] G3 is not fixed: applying `nuc_join` after a many-to-one remap detects a forbidden coalescence but does not prevent it.** If records `(sid=A,id=1)` and `(sid=B,id=2)` both remap to `K`, C1/C2 emit one `(sid=K,TOP)` record. At that point A and B are already one logical cluster. A later TOP barrier cannot undo the fusion. T1/T2 explicitly expect this result, so they validate detection of the G3 failure rather than preservation of the cannot-link constraint. The remap must be rejected or prevented before acceptance, or conflicting remaps must be proven impossible.

   The same post-hoc problem appears in B4: if different proper IDs reach propagation, warning and converting the already-merged cluster to TOP commits an invalid result. Plan v1’s abort was correct for a supposedly unreachable software-invariant violation.

4. **[major] Refusing TOP–BOTTOM merges creates avoidable over-segmentation and is not required by the lattice.** Consider a conflicted atomic supervoxel adjacent only to ordinary untagged cytoplasm. Every incident merge is refused, leaving it as a singleton even though no new nucleus identity would be introduced. Allowing TOP to merge with BOTTOM would still yield TOP and would continue to reject every later proper-ID merge. Full isolation is therefore an additional containment policy, not something required to close G1. It needs explicit acceptance because it can orphan many supervoxels and generate large cut streams.

5. **[major] The sentinel redesign breaks the previously valid full-`uint32` input contract.** Plan v1 accepted nucleus ID `4294967295`; plan v2 rejects it because it is now `NUC_CONFLICT`. A legitimate `uint32` instance mask using that ID fails C5 despite the task requesting a `uint32` mask. Preserving the full domain requires a separate state/tag field rather than consuming an ID value, unless reserving the maximum value is explicitly approved.

6. **[major] G4 and the verification plan remain insufficiently executable.** T1/T2 assert the unsafe post-remap TOP result. T3 does not give exact commands or require execution of `overlap_chunk_me.sh`; “apply the actual append” can still mean manually reproducing its `cat`, bypassing shell integration. T4 provides neither concrete binary arguments, complete fixture files, local upload/download substitutions, file-move sequence, nor a defined second-level expected artifact. The coder must reconstruct the hierarchy protocol to write it. V5 also demands an algebraic property the specified record operation cannot satisfy, and no verification covers accumulated sub-floor evidence merging into a different proper ID.

## Questions

1. Is the minimum evidence floor intended to apply independently to each original supervoxel, even when discarded evidence later accumulates above the floor?
2. Is permanently reserving `0xFFFFFFFF` acceptable, or must every nonzero `uint32` nucleus ID remain usable?
3. Is full TOP isolation required, or should TOP merge with BOTTOM while continuing to veto proper IDs?

READY: no