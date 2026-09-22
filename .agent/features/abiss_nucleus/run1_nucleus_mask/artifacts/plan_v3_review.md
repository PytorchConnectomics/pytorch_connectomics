# Plan v3 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, reasoning effort `xhigh`, sandbox read-only). Raw transcript:
`state/plan_v3_review.review.raw.md`.

`READY: no`, five findings, all `[major]`. Reviewer verdict: "H1 and H5 are fixed. H4 is fixed only
when the optional floor is disabled. H2, H3, and H6 are not fully fixed. At the default settings,
Bound C's arithmetic induction is sound, but it is too weak to establish the requested cannot-link
guarantee."

This round is different in kind from the previous three. Rounds 1-3 found defects that a better
plan could fix. Round 4 identifies a **representational impossibility**: with a fixed-width
record that stores one dominant id, no parameter setting delivers both noise tolerance and exact
cannot-link. The reviewer states it directly — "The current representation offers no setting that
provides both noise tolerance and the claimed guarantee" — and demonstrates both horns.

I1 is the load-bearing finding and is not a planning error that plan_v4 can absorb; it is a
question about what the feature promises. I3, I4, and I5 are ordinary defects with clear fixes.

## Findings

**I1 [major] — Invariant N'' permits an agglomeration merge joining distinct nuclei, even with the
floor disabled.** With `dominance_ratio = 0.6`:

```text
A: 60 voxels id1, 40 voxels id2  -> PROPER id1
B: 60 voxels id1, 40 voxels id3  -> PROPER id1
```

B3 allows `A + B` because both records name id1, yet the merge newly joins material from nuclei 2
and 3. The plan's disclaimer about contamination within a single watershed supervoxel does not
cover this cross-supervoxel fusion.

Bound C still holds exactly after the merge — 80 minority voxels equals `(1 - 0.6) * 200` — so
**Bound C is mathematically compatible with violating the cannot-link requirement.** Invariant N''
protects only recorded *dominant* identities, not all nucleus identities the clusters carry.

**I2 [major] — H2 is fixed only at the default; the supported positive floor invalidates Bound C
and reopens the original leak.** With `ABISS_NUC_MIN_VOXELS = 100`, a supervoxel of `100 id1 +
99 id2` drops id2 before `total` is computed, producing `PROPER id1, count=100, total=100`. The
actual minority count of 99 exceeds `(1 - 0.6) * 100 = 40`, and the record contradicts A1's own
definition of `total` as all nonzero-tagged voxels. The original H2 case remains reachable exactly
as V9 concedes. *"A warning that an advertised option voids the guarantees is not a fix."*

The same option makes the new CONFLICT-NONE allowance unsafe:

```text
X: 100 id1 + 100 id2  -> CONFLICT
Y: 99 id3             -> NONE (after the floor)
```

B3 allows `X + Y`, joining a CONFLICT cluster to real nucleus evidence. Plan_v2's full barrier
would have rejected this, so H4's fix is a **regression** in this configuration.

Conversely with the floor disabled, a single erroneous tagged voxel in an otherwise untagged
supervoxel makes it PROPER and creates an absolute veto — failing the plan's own stated
requirement that a handful of misassigned voxels must not hard-split a cell.

**I3 [major] — H3 is incomplete: the abort sites cover only differing PROPER ids, not every merge
B3 forbids.** `CONFLICT + PROPER id7` and `CONFLICT + CONFLICT` are equally forbidden by B3, and a
collision of either kind is equally evidence the veto was bypassed. C1/C2 instead apply `nuc_join`
and silently emit CONFLICT, and B4 aborts only on two differing PROPER records. Each invariant
check should reject **every pair for which `nuc_can_merge` is false**, rather than duplicating one
branch of that predicate.

**I4 [major] — the `match_chunks` abort is reachable on legitimate data.** B2 itself recognizes
that disjoint chunk portions of one boundary-spanning supervoxel can resolve to different PROPER
ids, and that coalescing them should yield CONFLICT. The same data reaches C2 when boundary
matching canonicalizes two chunk-local sids to one. C2 would abort merely because the identity
reconciliation happens in `process_nucs` rather than as duplicate input to `load_nuc`. No proof or
provenance rule distinguishes the two cases. As specified this is a **data-reachable
denial-of-service** on a long distributed run, not an invariant check. C1 may be justified for a
purely agg-derived remap; C2 is not established.

**I5 [major] — H6 remains unfixed; V8 is still not executable without reconstructing the hierarchy
protocol.** Specifically:

* V1 gives `cmake ..` but never creates or enters a build directory.
* T1/T2 list required files but not their binary layouts, contents, boundary filenames, or
  complete fixture construction; the coder must derive them from the binaries.
* T3 deliberately does not execute `overlap_chunk_me.sh`; evaluating a grepped `cat` line cannot
  test its working directory, surrounding file creation, variable setup, or control flow. The
  subsequent `merge_chunks_me.py` invocation, arguments, JSON, and staging remain unspecified.
* T4 stops the atomic sequence around line 53 and **omits C4's essential later move**
  `ongoing_nuc.data -> ongoing_nuclei_labels_<chunk>.data`, so the merge step it is meant to test
  lacks the per-chunk nucleus payload entirely.
* T4 leaves `<tag>`, `<META>`, `<json>`, directory topology, chunk offsets, and fixture formats
  undefined.
* V5 asks for records drawn from a set including NONE and CONFLICT "with non-zero counts and
  totals", contradicting A1's canonical NONE record and CONFLICT's required zero count.

## Questions

1. Is the intended guarantee being reduced from "all distinct mask ids cannot merge" to "different
   recorded dominant ids cannot merge"? The dominance representation can only provide the latter.
2. Should weak evidence remain represented in a separate state or exact identity set, so it can
   avoid single-voxel hard vetoes without becoming NONE?
3. What proves that C2's many-to-one mappings are exclusively B3-vetted agglomeration remaps rather
   than legitimate cross-chunk identity reconciliation?

## Verdict

VERDICT: NEEDS_CHANGES
