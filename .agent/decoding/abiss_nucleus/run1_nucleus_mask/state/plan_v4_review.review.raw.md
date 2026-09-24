# Plan v4 Review

## Summary

Not ready. I3 and I4 are fixed in the implementation design. I2 remains broken at cluster propagation, and I5 remains incomplete. Invariant D’s principal prohibitions are enforced, but B3 is stricter than the stated contract.

## Findings

1. [major] The aggregate floor keeps `total` exact only at extraction; `nuc_join` does not preserve the record semantics or Bound C. With defaults:

   ```text
   A: 49 voxels id2 -> NONE, count=0, total=49
   B: 50 voxels id1 -> PROPER id1, count=50, total=50
   ```

   B3 permits `A+B`, and `nuc_join` produces `PROPER id1, count=50, total=99`. Id1 is only 50.5% of the tagged voxels, below the required 60%, and the 49 minority voxels exceed Bound C’s `0.4 * 99`. If A instead contains id1, `count=50` also ceases to mean all voxels backing id1. Two 49-voxel NONE records similarly remain NONE with `total=98`, even though their aggregate exceeds the floor. Thus I2 is not fixed, and v4 breaks the merge-closed state semantics that v3 had when its floor was disabled.

2. [major] B3/B4 do not implement exactly Invariant D. `CONFLICT + CONFLICT` has no pair of differing recorded dominant ids and neither cluster carries a recorded dominant id, so Invariant D permits it. B3 refuses it, and B4 mirrors that refusal. No Invariant-D-forbidden pair is permitted, but this extra refusal is an undocumented stronger containment policy with an over-segmentation cost. Either add it explicitly to Invariant D or allow the merge.

3. [major] The explanation of Invariant D is not fully honest. The statement that contamination within a supervoxel results in CONFLICT is contradicted by A4 and by the plan’s own example: `60 id1 + 40 id2` is PROPER, while a mixed supervoxel below 50 tagged voxels is NONE. The claim that the limitation is pathological only near 40% contamination is also false: `99 id1 + 1 id2` and `99 id1 + 1 id3` are both PROPER id1 and may merge, joining recorded minority mask identities 2 and 3. These limitations should be stated without the “~40%” qualification.

4. [major] I5 remains unfixed because V8 cannot reach the condition it claims to test. Through normal `agg` execution, B3 rejects every pair for which `nuc_can_merge` is false, so such a pair cannot reach B4. “Force them into merge propagation via a unit harness” specifies neither an injectable hook nor an extracted propagation function, while simultaneously requiring that `agg` itself abort. A concrete test seam or harness build/invocation is required.

5. [major] The hierarchy verification remains under-specified:

   - V1 says all commands start inside `work/abiss` but then runs `cd work/abiss`, which targets a nonexistent nested directory.
   - T1 does not define the binary layouts or required contents of `residual_rg_<tag>.data` or the six boundary files.
   - T2 says “same construction” without enumerating all files consumed by `match_chunks`.
   - T3 gives no positional invocation for `overlap_chunk_me.sh`, no exact fallback extraction boundaries, and leaves its tag, JSON path/values, and staging topology unspecified.
   - T4 still leaves the composite `<json>` and `<tag>` unresolved and does not name the final remap file or its format for the concluding assertion.

   The instruction to derive formats from source is useful implementation work, but it does not make these tests runnable as specified by someone unfamiliar with the protocol.

I4’s hierarchy denial-of-service is fixed: the hierarchy sites now join and count rather than abort. This does not make Invariant D itself unobservable because forbidden agglomeration pairs are checked immediately in B3/B4. The counters are only coarse diagnostics and cannot distinguish legitimate boundary reconciliation from a hierarchy defect.

## Questions

1. Is `CONFLICT + CONFLICT` intentionally forbidden? If so, should that clause be added to Invariant D?
2. Should Bound C be dropped, or should joins involving `NONE(total>0)` transition conservatively so PROPER records continue satisfying their stated dominance semantics?

READY: no