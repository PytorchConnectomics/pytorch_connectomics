# Plan v4 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, reasoning effort `xhigh`, sandbox read-only). Raw transcript:
`state/plan_v4_review.review.raw.md`.

`READY: no`, five findings, all `[major]`. Reviewer verdict: "I3 and I4 are fixed in the
implementation design. I2 remains broken at cluster propagation, and I5 remains incomplete.
Invariant D's principal prohibitions are enforced, but B3 is stricter than the stated contract."

Confirmed fixed: **I3** (the merge-time check now reuses the whole predicate) and **I4** (both
hierarchy aborts removed; the reviewer explicitly confirms the denial-of-service is gone and that
removing them does not make Invariant D unobservable, since forbidden pairs are caught immediately
in B3/B4).

Three findings land on plan_v4's own new material. J1 is the substantive one: moving the floor to
aggregate `total` fixed extraction but broke **merge closure** — a `NONE` record carrying nonzero
`total` poisons the dominance property of any `PROPER` record it joins. J2 and J3 are both accuracy
failures in how the contract is stated, which matters more than usual here because the contract
being honestly stated is the entire point of the user's decision.

## Findings

**J1 [major] — the aggregate floor keeps `total` exact only at extraction; `nuc_join` preserves
neither the record semantics nor Bound C.** With defaults:

```text
A: 49 voxels id2  -> NONE,      count=0,  total=49
B: 50 voxels id1  -> PROPER id1, count=50, total=50
```

B3 permits `A + B`, and `nuc_join` produces `PROPER id1, count=50, total=99`. Id1 is 50.5% of the
tagged voxels, below the required 60%, and the 49 minority voxels exceed Bound C's `0.4 * 99`. If A
had instead contained id1, `count=50` would also stop meaning "all voxels backing id1". Two
49-voxel NONE records likewise stay NONE with `total=98` even though their aggregate clears the
floor.

The reviewer's conclusion: I2 is not fixed, and **v4 broke the merge-closed state semantics that v3
had when its floor was disabled**.

**J2 [major] — B3/B4 do not implement exactly Invariant D.** `CONFLICT + CONFLICT` involves no pair
of differing recorded dominant ids and neither side carries a recorded dominant id, so Invariant D
as written *permits* it — but B3 refuses it and B4 mirrors that refusal. No Invariant-D-forbidden
pair is permitted, so the code is strictly stronger than the contract, but the extra refusal is an
undocumented containment policy with an over-segmentation cost. Either add the clause to Invariant D
or allow the merge.

**J3 [major] — the statement of Invariant D's limits is itself not honest.** Two specific errors:

* The claim that contamination within a supervoxel yields CONFLICT is contradicted by A4 and by the
  plan's own example: `60 id1 + 40 id2` is PROPER, and a mixed supervoxel below 50 tagged voxels is
  NONE.
* The claim that the limitation is pathological only near 40% contamination is false.
  `99 id1 + 1 id2` and `99 id1 + 1 id3` are both PROPER id1 and may merge, joining mask identities
  2 and 3. The limitation applies at *any* contamination level.

The "~40%" qualification must go.

**J4 [major] — V8 cannot reach the condition it claims to test.** Through normal `agg` execution
B3 rejects every pair for which `nuc_can_merge` is false, so such a pair cannot reach B4. "Force
them into merge propagation via a unit harness" specifies neither an injectable hook nor an
extracted propagation function, while simultaneously requiring that `agg` itself abort. A concrete
test seam, or a harness build and invocation, is required.

**J5 [major] — the hierarchy verification remains under-specified.** Concretely:

* V1 says all commands start inside `work/abiss` and then runs `cd work/abiss`, targeting a
  nonexistent nested directory.
* T1 does not define the binary layout or required contents of `residual_rg_<tag>.data` or the six
  boundary files.
* T2 says "same construction" without enumerating every file `match_chunks` consumes.
* T3 gives no positional invocation for `overlap_chunk_me.sh`, no exact fallback extraction
  boundaries, and leaves its tag, JSON path and values, and staging topology unspecified.
* T4 leaves the composite `<json>` and `<tag>` unresolved and never names the final remap file or
  its format for the concluding assertion.

The instruction to derive formats from source is useful implementation work but does not make these
tests runnable as specified by someone unfamiliar with the protocol.

## Questions

1. Is `CONFLICT + CONFLICT` intentionally forbidden? If so, should that clause be added to
   Invariant D?
2. Should Bound C be dropped, or should joins involving `NONE(total > 0)` transition conservatively
   so PROPER records continue satisfying their stated dominance semantics?

## Verdict

VERDICT: NEEDS_CHANGES
