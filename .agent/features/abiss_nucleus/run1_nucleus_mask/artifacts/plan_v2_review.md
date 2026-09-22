# Plan v2 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, reasoning effort `xhigh`, sandbox read-only). Raw transcript:
`state/plan_v2_review.review.raw.md`.

`READY: no`, six findings, all `[major]`. The reviewer's verdict: "G5 and G6 are fixed. G2's
original associativity failure is fixed for canonical records, and G1's original above-threshold
counterexample is blocked. However, G1 remains leaky below the extraction floor, G3 is not fixed
at all, and G4 remains insufficiently executable. The redesign also falsely claims full
idempotence, unnecessarily isolates TOP from BOTTOM, and removes one valid `uint32` input value."

Confirmed fixed: **G5** (unconditional size check) and **G6** (explicit wire layout). G2's
associativity defect is fixed for canonical records, and G1's above-floor counterexample is
closed by TOP-as-barrier.

Three results change the design again:

* The idempotence claim is simply false — the id field is a lattice join but `count`/`total` are
  additive, so `nuc_join(a,a) != a` (H1).
* The absolute extraction floor discards evidence that can *accumulate* across supervoxels back
  above the floor, reopening G1 by a different route (H2).
* Converting a conflicting many-to-one remap to TOP **detects** an invalid coalescence after it
  has already happened rather than preventing it; plan_v1's `abort()` was the correct response to
  a supposedly unreachable invariant violation (H3).

The reviewer also answers plan_v2's own Q1 (TOP-BOTTOM isolation is an unnecessary policy) and
raises a contract regression plan_v2 introduced (the reserved sentinel removes a valid uint32 id).
All findings accepted.

## Findings

**H1 [major] — `nuc_join` is not idempotent as claimed, in either `count` or `total`.** For
`a = (id=1, count=5, total=5)`, `nuc_join(a, a)` returns `(1, 10, 10)`, not `a`. The id-state
projection is a lattice join, but the complete record is an additive aggregation. It is
associative and commutative for canonical records but not idempotent. Consequently V5's required
idempotence test either fails on realistic records or accidentally tests only zero-count records
and misses the defect. Duplicate copies of the same logical evidence would also double the
diagnostic counters.

**H2 [major] — G1 only partially fixed: discarded sub-floor identities permit a real
two-nucleus merge.** With `ABISS_NUC_MIN_VOXELS = 100`:

```text
A: 60 voxels of id 1   -> NONE, total=60
B: 60 voxels of id 1   -> NONE, total=60
C: 100 voxels of id 2  -> id 2
```

Agglomerating A and B yields `NONE, total=120`; the discarded id cannot be promoted because
decisions ignore `total`. That cluster may then merge with C, producing a cluster holding 120
tagged voxels of nucleus 1 and 100 of nucleus 2 — material from two distinct nuclei joined by an
agglomeration merge, which is exactly what Invariant N' promises cannot happen. TOP fixes only
conflicts whose ids individually survive extraction. V4 normalizes part of this behavior as
expected, and no test covers accumulation of discarded evidence.

**H3 [major] — G3 is not fixed: `nuc_join` after a many-to-one remap detects a forbidden
coalescence but does not prevent it.** If `(sid=A, id=1)` and `(sid=B, id=2)` both remap to `K`,
C1/C2 emit one `(sid=K, TOP)` record — but A and B are already one logical cluster at that point,
and a TOP barrier cannot undo the fusion. T1/T2 as written therefore validate *detection of the
G3 failure* rather than *preservation of the cannot-link constraint*. The remap must be rejected
or prevented before acceptance, or conflicting remaps must be proven impossible.

The same post-hoc problem appears in B4: if two different proper ids reach merge propagation,
warning and converting the already-merged cluster to TOP commits an invalid result. **Plan_v1's
`abort()` was correct** for a supposedly unreachable software-invariant violation.

**H4 [major] — refusing TOP-BOTTOM merges causes avoidable over-segmentation and is not
required by the lattice.** A conflicted supervoxel adjacent only to ordinary untagged cytoplasm
has every incident merge refused and is left a singleton, even though merging would introduce no
new nucleus identity. Allowing TOP to merge with BOTTOM still yields TOP and still rejects every
later proper-id merge. Full isolation is an additional containment *policy*, not a requirement for
closing G1, and it needs explicit acceptance because it can orphan many supervoxels and generate
large cut streams.

**H5 [major] — the sentinel redesign breaks the full-`uint32` input contract.** Plan_v1 accepted
nucleus id `4294967295`; plan_v2 rejects it because that value is now `NUC_CONFLICT`. A legitimate
uint32 instance mask using that id fails C5, despite the task specifying a uint32 mask. Preserving
the full domain requires a separate state/tag field rather than consuming an id value, unless
reserving the maximum value is explicitly approved.

**H6 [major] — G4 and the verification plan remain insufficiently executable.** T1/T2 assert the
unsafe post-remap TOP result (see H3). T3 gives no exact commands and does not require executing
`overlap_chunk_me.sh`; "apply the actual append" can still mean hand-reproducing its `cat`,
bypassing shell integration. T4 supplies neither concrete binary arguments, complete fixture
files, local upload/download substitutions, file-move sequence, nor a defined second-level
expected artifact, so the coder must reconstruct the hierarchy protocol to write it. V5 demands an
algebraic property the record operation cannot satisfy (H1), and nothing verifies accumulated
sub-floor evidence merging into a different proper id (H2).

## Questions

1. Is the minimum evidence floor intended to apply independently to each original supervoxel, even
   when discarded evidence later accumulates above the floor?
2. Is permanently reserving `0xFFFFFFFF` acceptable, or must every nonzero `uint32` nucleus id
   remain usable?
3. Is full TOP isolation required, or should TOP merge with BOTTOM while continuing to veto proper
   ids?

## Verdict

VERDICT: NEEDS_CHANGES
