## Summary

H1 and H5 are fixed. H4 is fixed only when the optional floor is disabled. H2, H3, and H6 are not fully fixed. At the default settings, Bound C’s arithmetic induction is sound, but it is too weak to establish the requested cannot-link guarantee.

## Findings

**[major] Invariant N'' permits an agglomeration merge that joins distinct nuclei, even with the default floor of zero.** Consider `dominance_ratio = 0.6`:

```text
A: 60 voxels id1, 40 voxels id2 -> PROPER id1
B: 60 voxels id1, 40 voxels id3 -> PROPER id1
```

B3 allows `A+B` because both records name id1. That merge newly joins material from nuclei 2 and 3. The disclaimer about contamination within one watershed supervoxel does not cover this cross-supervoxel fusion.

Bound C still holds exactly after the merge: 80 minority voxels equals `(1 - 0.6) * 200`. Thus Bound C is mathematically compatible with violating the task’s cannot-link requirement. N'' only protects recorded dominant identities, not all nucleus identities carried by the clusters.

**[major] H2 is fixed only at the default setting; the supported positive floor immediately invalidates Bound C and reopens the original merge leak.** With `ABISS_NUC_MIN_VOXELS=100`:

```text
A: 100 voxels id1, 99 voxels id2
```

The id2 count is dropped before `total` is computed, producing `PROPER id1, count=100, total=100`. The actual minority count is 99, exceeding `(1 - 0.6) * 100 = 40`. This also contradicts A1’s definition of `total` as all nonzero-tagged voxels.

The original H2 case remains reachable exactly as V9 acknowledges: two `60 × id1` supervoxels become NONE and can merge with `100 × id2`. A warning that an advertised option voids the guarantees is not a fix.

Allowing CONFLICT–NONE also becomes unsafe under this option:

```text
X: 100 id1 + 100 id2 -> CONFLICT
Y: 99 id3            -> NONE after the floor
```

B3 allows `X+Y`, joining a CONFLICT cluster to real nucleus evidence. Plan v2’s full barrier would at least have rejected this particular merge, so this is also a regression relative to v2.

Conversely, with the floor disabled, a single erroneous tagged voxel makes a large supervoxel PROPER and can create an absolute veto. That fails the stated requirement that a handful of misassigned voxels not hard-split a cell. The current representation offers no setting that provides both noise tolerance and the claimed guarantee.

**[major] H3 is incomplete: the three checks cover only different PROPER ids, not every merge B3 declares forbidden.** These collisions are also evidence that the veto was bypassed:

```text
CONFLICT + PROPER id7 -> forbidden by B3
CONFLICT + CONFLICT   -> forbidden by B3
```

C1 and C2 instead apply `nuc_join` and silently emit CONFLICT. B4 likewise aborts only for two differing PROPER records. Therefore the same detect-after-the-fact defect remains for two classes of forbidden merge. Each invariant-check site should reject every pair for which `nuc_can_merge` is false, not duplicate only one branch of that predicate.

**[major] The `match_chunks` abort may be reachable on legitimate data and is therefore not justified as an invariant check.** B2 explicitly recognizes this legitimate situation: disjoint chunk portions of one boundary-spanning supervoxel can resolve to different PROPER ids, and coalescing them should yield CONFLICT because the watershed object genuinely straddles nuclei.

The same data can reach C2 with two chunk-local sids that boundary matching canonicalizes to one sid. C2 would abort merely because the identity reconciliation occurs in `process_nucs` rather than as duplicate input to `load_nuc`. The plan provides no proof or provenance rule showing that every `match_chunks` many-to-one mapping was produced by a B3-vetted agglomeration merge. Without such a distinction, this is a data-reachable denial-of-service path. C1 may be justified for a purely agg-derived remap; C2 is not established.

**[major] H6 remains unfixed; V8 is still not executable without reconstructing the hierarchy protocol.**

- V1 gives `cmake ..` but never creates or enters a build directory.
- T1/T2 list required files but do not specify their binary layouts, contents, boundary filenames, or complete fixture construction. The coder must derive those from the binaries.
- T3 deliberately does not execute `overlap_chunk_me.sh`; evaluating a grepped `cat` line cannot test its working directory, surrounding file creation, variable setup, or control flow. The subsequent `merge_chunks_me.py` invocation, arguments, JSON, and staging are still unspecified.
- T4 stops the atomic sequence at the line-53 area. It omits C4’s essential later move:
  `ongoing_nuc.data -> ongoing_nuclei_labels_<chunk>.data`. The following merge step therefore lacks the per-chunk nucleus payload it is supposed to test.
- T4 still leaves `<tag>`, `<META>`, `<json>`, directory topology, chunk offsets, and fixture formats undefined.
- V5 asks for records from a set including NONE and CONFLICT “with non-zero counts and totals,” contradicting A1’s canonical NONE record and CONFLICT’s required zero count.

## Questions

1. Is the intended guarantee being reduced from “all distinct mask ids cannot merge” to “different recorded dominant ids cannot merge”? The dominance representation can only provide the latter.
2. Should weak evidence remain represented in a separate state or exact identity set, so it can avoid single-voxel hard vetoes without becoming NONE?
3. What proves that C2’s many-to-one mappings are exclusively B3-vetted agglomeration remaps rather than legitimate cross-chunk identity reconciliation?

READY: no