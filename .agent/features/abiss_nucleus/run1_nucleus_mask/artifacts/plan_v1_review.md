# Plan v1 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, reasoning effort `xhigh`, sandbox read-only). Raw transcript:
`state/plan_v1_review.review.raw.md`.

`READY: no`, with six findings, all tagged `[major]`. The reviewer's verdict: "Plan v1 is still
not executable safely. F1, F5-F7, F10, and F11 are fixed. F8 fixes the id width but leaves the
enclosing wire layout underspecified. F2-F4 remain unresolved, and F9 is only partially
addressed. Most importantly, Invariant N does not hold."

Six of the eleven plan_v0 findings are confirmed fixed: F1, F5, F6, F7, F10, F11.

The central result is that **Invariant N does not hold**, and the reviewer supplied a concrete
reachable counterexample rather than a general doubt. The root cause it identifies is precise
and correct: plan_v1's record conflates "no evidence" with "conflicting evidence", using a
single `id = 0` for both. That conflation is simultaneously the hole in the invariant (G1), the
associativity break (G2), and the reason remap coalescence cannot be handled (G3). All findings
are accepted; none are softened or discarded.

One finding (G5) rests on a premise that is factually wrong for this repository. It is recorded
in full below with the correction, and its remedy is adopted anyway because the remedy is
strictly better and free.

## Findings

**G1 [major] — F2 remains unresolved: abstention does not establish Invariant N.** A supervoxel
containing above-floor ids 1 and 2 emits `id=0`, but the underlying cluster still physically
contains both. It can then merge with a cluster carrying id 3, because `nuc_can_merge` treats
zero as unconstrained, and B4 rewrites the record to id 3. The resulting segment physically
contains material from nuclei 1, 2, and 3 while recording only 3. That is a reachable
counterexample to the hard cannot-link guarantee. Abstention avoids choosing a wrong dominant
id, but it neither makes the record exact nor removes the conflicting evidence. The reviewer
names the acceptable remedies: `NUC_WS`, an exact set representation, or **a distinct conflict
state with appropriately restrictive merge behavior**.

**G2 [major] — F3 remains unresolved: B2's combination rule is not associative.** Plan_v1 calls
abstention absorbing but implements zero as an *identity*, `0 + x = x`. The reviewer's
counterexample, for `a = id1`, `b = id2`, `c = id2`:

```text
(a + b) + c = 0 + c = id2
a + (b + c) = a + id2 = 0
```

Remap grouping or input order therefore changes the recorded id and the resulting veto behavior.
The representation conflates "no evidence" with "conflicting evidence", so the required
associative operation cannot be implemented as written.

**G3 [major] — F4 remains unresolved: the "pure sid remaps" can be semantic many-to-one cluster
merges.** Plan_v1 itself admits duplicate sids with conflicting ids can arise from remapping.
`reduce_chunk` / `match_chunks` would then emit `(sid=K, id=1)` and `(sid=K, id=2)` — one logical
cluster already carrying two nonzero ids, with `B3` never seeing a merge edge. A later
`load_nuc` abstention cannot undo that coalescence. The induction must cover every many-to-one
remap: either prove such conflicts unreachable, or veto them before the mapping is accepted.

**G4 [major] — F9 only partially fixed.** T3 starts from a hand-built final `vetoed_edges.data`,
which bypasses the actual C3 path that appends `nuc_cuts.data`, merges the per-chunk streams, and
produces that input. T1/T2 test only byte-preserving remaps and never exercise a conflicting
many-to-one mapping. T4, the sole proposed end-to-end hierarchy test, is permitted to fail and is
not specified through the complete workflow. A sufficient test must cover the real
producer/append/merge/consumer chain plus a multi-level case with different nucleus ids.

**G5 [major] — A4's size check uses `assert`, which the reviewer notes is removed under
`NDEBUG` in Release builds**, so a malformed `nuc.raw` could still be mmaped and silently
misread. An unconditional runtime size check and a direct malformed-file test are needed.

*Coordinator correction, recorded rather than used to soften the finding:* the premise does not
hold for this repository. `CMakeLists.txt:11` overrides the Release flags outright —
`set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fopenmp")` — dropping CMake's default
`-DNDEBUG`, so `assert` remains live in this project's Release build. The finding's *remedy* is
nonetheless adopted in full: relying on a non-standard flag override for a data-integrity check
is fragile, an unconditional check costs nothing, and the surrounding code already depends on
`assert` for the same purpose. The direct malformed-file test the reviewer asks for is also
added.

**G6 [major] — the stable on-disk layout remains underspecified.** Although `nuc_record_t` is
declared packed at 20 bytes, the hierarchy reads and writes raw `std::pair<seg_t, nuc_record_t>`
records. That pair has implementation-defined layout and likely trailing padding, and the packed
record also contains unaligned `uint64_t` members. The plan states neither the exact pair
size/offsets nor how the hand-built V7 fixtures encode it. An explicit wire record with static
size and offset assertions, used consistently at every producer and consumer, is required.

## Questions

1. Should an above-floor mixed supervoxel require `NUC_WS`, become a permanent conflict/barrier
   state, or force an exact multi-id payload?
2. Can the planner prove that every hierarchy remap is incapable of mapping different recorded
   ids onto one sid? If not, where is that remap vetoed before the ids are coalesced?

## Verdict

VERDICT: NEEDS_CHANGES
