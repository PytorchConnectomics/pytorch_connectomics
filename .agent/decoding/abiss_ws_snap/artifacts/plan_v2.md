# Plan v2

## Summary

v1's must-link cannot exist as written, and that is the finding that reshapes this plan.
`merge_segments` only ever visits region-graph edges above `tholds.second`, so a rule of the form
"same tag -> union" never fires for two components that have no qualifying edge -- which is the
common case for a nucleus fragmented across many basins. The MST phase cannot do it either: it runs
after `remaps`, dusting and compaction, and only suppresses dendrogram cycles.

So the snap becomes an explicit **tag-group closure**: a separate union pass over all components
sharing a nonzero tag, independent of the region graph, run at a named point in the existing
sequence. Everything else in the plan follows from placing it correctly.

`merge_segments` (`src/ws/agglomeration.hpp`) is:

```
 66  try_merge(counts, sets, s1, s2, size)          RG-edge loop, DSU `sets`
 73  std::vector<ID> remaps(counts.size())
 80  dust loop: counts[s] >= low ? new id : remaps[s] = 0
111  seg_raw[idx] = remaps[sets.find_set(seg_raw[idx])]     relabel + compaction
129  MST over rg, via remaps[sets.find_set(...)], mst.link
```

The closure goes **between line 66's loop and line 73**. Placed there it needs no remap emission of
its own: `sets` is the same DSU that lines 111 and 129 read through, so the unions are carried into
the relabel, the compaction and the dendrogram automatically. That answers reviewer question 1 for
the atomic level; the hierarchy level gets the same closure before its own `remap_pairs` emission.

## Scope

Unchanged from v1: `src/ws/` (atomic chunk, agglomeration, chunk merging), `cut_chunk_ws.py`, the
per-supervoxel nucleus-tag side file and its hierarchy transport, and the crop A/B.

Out of scope: the two `abiss_nucleus` review_v0 findings (that run's `code_v1`); whole-volume
re-run; NERL; `scripts/nucleus_snap.py`, which stays disabled.

## Proposed Changes

### A. Mask into the ws stage

Unchanged: `cut_chunk_ws.py` writes `nuc.raw` via `nucleus_utils.cut_nucleus_data`, guarded by
`NUC_PATH`.

### B. Tag every basin, not only split ones (v1-F3)

After `watershed()`, for each basin accumulate the set of nucleus ids its voxels carry:

* **exactly one** -> tag the basin with it. v1 omitted this and it is the *main* snap case: one
  nucleus spread over many basins;
* **two or more** -> seeded watershed inside the basin's bounding box, seeded by the overlapping
  nucleus regions; each piece takes a fresh label and its seed's tag; unreached voxels form one
  untagged piece;
* **none** -> tag 0.

Then re-densify labels and rebuild `counts` from scratch, real size plus the `on_border` bit
derived from contact with the applicable non-real chunk faces, and emit `basin_tag` in that same
dense namespace.

### C. The tag-group closure (v1-F1, v1-F2, v1-F8)

Inserted after the `try_merge` loop, before `remaps` is built:

```
for each nonzero tag t:
    pick the group's representative r = the surviving root of the first member
    for every other member m with sets.find_set(m) != sets.find_set(r):
        nuc_union(counts, basin_tag, sets, r, m)
```

`nuc_union` is a dedicated primitive, modelled on `try_merge` lines 24-29 but with the size and
border tests removed:

```
counts[s1] += counts[s2] & ~on_border;      // real sizes add
counts[s1] |= counts[s2] &  on_border;      // border bits OR
counts[s2]  = 0;
sets.link(s1, s2);
s = sets.find_set(s1);
if (s != s1) { swap(counts[s], counts[s1]); swap(basin_tag[s], basin_tag[s1]); }
```

The tag must follow the representative `sets.link` actually chose -- v1 left this unspecified
(v1-F4). Cross-tag pairs are never passed to `nuc_union`; tagged+untagged unions arising from
`try_merge` keep the tag on the surviving root, which requires the same swap there.

Unconditional is correct and is now settled rather than left open (v1-F14): the task's contract is
that the mask is trusted ground truth. An affinity floor would contradict the imposed constraint
and would recreate the fragmentation this is meant to fix. Region-graph adjacency may discover
*ordinary* merges; it must not gate the nucleus must-link.

Multi-threshold mode needs a fresh `basin_tag` copy beside each `seg_copy` / `rg_copy` /
`counts_copy` (v1-F4), not merely one output file per `out_tag`.

### D. Cross-tag veto at every union site

Atomic: `try_merge`, and the MST loop must also skip cross-tag edges so no such edge reaches the
dendrogram.

`merge_chunks.cpp` has **four** union sites, not one (v1-F5): plateau unions in `same`, descent
unions, `try_merge`, and the final MST `sets.link`. All four check tags before uniting; the first
three update tag and count ownership after. The `face_size == 0` early return must pass tags
through or finalise them rather than dropping the side data.

### E. Dust protection at every level (v1-F9)

Atomic dust loop (agglomeration.hpp line 80-94) and `merge_chunks.cpp`'s independent
`dust_threshold` mapping both exempt components with a nonzero tag.

### F. Hierarchy transport, with an explicit contract (v1-F6, v1-F7)

Per level: consume the children's tag files keyed by their emitted ids; match them through the
level's `ongoing` / `done_pre` / `done_post` and updated boundary ids; run the same tag-group
closure (C) over this level's components **before** `remap_pairs` is produced, so same-tag children
become real remaps rather than merely agreeing labels; emit one tag record per surviving
representative. Conflicting records (two different nonzero tags for one id) abort rather than pick.

This is the step that converts tags into segmentation identity. v1 left it implicit, which is
exactly how "tags survive perfectly while the ids stay fragmented" happens.

## Files and Areas

| File | Purpose |
|---|---|
| `work2/abiss/scripts/cut_chunk_ws.py` | write `nuc.raw` when `NUC_PATH` is set, via `nucleus_utils` |
| `work2/abiss/src/ws/atomic_chunk.cpp` | load `nuc.raw`; tag every basin; split multi-nucleus basins; re-densify labels; rebuild `counts` with `on_border`; emit a tag file per `out_tag` |
| `work2/abiss/src/ws/agglomeration.hpp` | `nuc_union` primitive; tag-group closure between the `try_merge` loop and `remaps`; cross-tag veto in `try_merge` and the MST loop; tag-exempt dusting |
| `work2/abiss/src/ws/merge_chunks.cpp` | tag checks at all four union sites; closure before `remap_pairs`; tag-exempt dusting; `face_size == 0` passthrough |
| `work2/abiss/CMakeLists.txt` | `ABISS_WS_NUC_GUARD` option, default ON |
| `work2/abiss/scripts/{atomic_chunk_ws,remap_chunk_ws}.sh` | carry the tag file alongside `dend`/`seg` |
| `work2/abiss/work/test/` | V1-V6 fixtures |
| `dev/zebrafinch/nuc_z1_y7_x6/` | `wssnap` arm |

Untouched: live `lib/abiss`, `lib/abiss/build/`, `scripts/nucleus_snap.py`.

## Verification Plan

**V1 -- bit-invariance (hard gate).** `run_v2_invariance.sh`, no `NUC_PATH`,
`identical=N differing=0 missing=0`.

**V2 -- split, snap, counts, dust.** Fixtures:
(a) one nucleus overlapping **several** original basins -> after C they are one component (this is
the snap; v1's two-nucleus fixture did not test it);
(b) one basin overlapping two nuclei -> divided, each piece single-tagged, all of each nucleus's
mask voxels in one piece;
(c) an original **border** basin split into a border-touching child and an interior child, with an
independent oracle that rescans the post-split labels and derives `on_border` from contact with the
applicable non-real faces -- comparing complete raw count words, so inheriting the parent bit fails
(v1-F11);
(d) a sub-`lowt` tagged piece survives dusting, atomic **and** at a hierarchy level.

**V3 -- guards, with a named seam (v1-F10).** CMake option `ABISS_WS_NUC_GUARD`, default `ON`;
tests build a second binary with it `OFF`, which bypasses only the tag checks while keeping the
same split and tag inputs. Two fixtures: cross-tag components that merge with the guard off and do
not with it on (and no cross-tag edge in `dend_<tag>.data`); and **same-tag components that both
exceed `size_threshold` with an edge below the ordinary merge threshold, which must merge anyway**
-- that is the only test that proves C rather than assuming it. Also report, from the crop run,
whether the real split pieces were eligible for ordinary refusion at all; if not, say C is inert on
this data rather than claiming it as the fix.

**V4 -- hierarchy, two tags plus an untagged bridge.** Child+parent equals monolithic; tags survive
representative changes; no cross-tag stitch; parent tag records and remaps inspected, not just
segmentation equality.

**V5 -- end-to-end id trace (v1-F7).** A small fixture tracing known atomic ids through boundary
files, parent remaps, final `ws3`, `chunkmap.data`, and the agglomeration nucleus records: the
existing global nucleus table must show each nucleus mapping to exactly one supervoxel. Without
this the tags can remain an unused side channel.

**V6 -- crop A/B.** Arms `w2ctl` and `wssnap`. The watershed artifact checked is the **final
remapped `WS_PATH` precomputed layer** -- the one `upload_chunk.py` writes in `remap_chunk_ws.sh`
after `ws3`, which is also what feeds `chunkmap.data` -- not an atomic `seg_<tag>.data`
(answering reviewer question 2). Report for both arms:

1. all eight nuclei present, and each nucleus's mask voxel count assigned to a **nonzero** label
   exactly preserved versus control;
2. cannot-link invariant: ws supervoxels overlapping >= 2 nuclei == **0** (control >= 1);
3. must-link invariant: each of the eight masks overlaps **exactly one** nonzero final ws id
   (v1-F12 -- without this a split-only implementation passes);
4. shared mask mass (`--tol 0.0`) 110,244 -> **0**;
5. segments covering 95% of each nucleus's mask == 1 for all eight;
6. dominance >= 0.99 for all eight;
7. largest segment <= 450,000,000; root segment count within 0.9x-1.25x of 3,613;
8. ws wall-clock, both arms.

## Risks and Questions

* **[major] The fourth no-op (v1-F13).** v1 could only *detect* it after the crop run. C-in-the-DSU
  and F-before-`remap_pairs` are the structural blocks: at the atomic level the closure is upstream
  of the only relabel path, and at each hierarchy level it is upstream of the only remap emission,
  so a tag that exists cannot fail to become identity. V6.3 is the end-to-end check, V5 the
  fixture-level one.
* **[major] The cut has no image support.** Bottleneck ~0.999 between all three pairs against a 0.3
  agglomeration threshold; the seeded watershed cuts at an equal-resistance surface. Imposed
  structure, defensible only because the mask is confirmed correct and conservative. Say so in the
  code.
* **[major] Unconditional must-link merges across anything.** If the mask ever bleeds past a real
  membrane, C will union across it with no recourse. That is the accepted cost of treating the mask
  as ground truth; criteria 5-7 price it and V1 keeps the no-mask path exact.
* **[minor] Cost.** Atomic work is bounded by the ~252^3 chunk; the 243M object exists only after
  stitching. Bounding the seeded flood to the basin bbox is still worth doing.

## Changes Since Previous Plan Version

All fourteen findings addressed; the must-link mechanism changed as a result.

* **v1-F1, v1-F2, v1-F8 (the must-link is undefined / the MST cannot implement it / it needs a
  dedicated primitive) -- accepted.** Section C is a real tag-group closure at a named line, with
  `nuc_union` specified including the `sets.link` root swap, placed upstream of `remaps` so no
  separate remap emission is needed atomically.
* **v1-F3 (one-nucleus basins never tagged) -- accepted.** B tags every basin; V2(a) tests it.
* **v1-F4 (tag mutation under DSU, multi-threshold copies) -- accepted**, specified in C.
* **v1-F5 (four union sites in `merge_chunks`) -- accepted**, all four named in D along with the
  `face_size == 0` early return.
* **v1-F6, v1-F7 (hierarchy contract; ids stay fragmented) -- accepted.** F states consume/emit and
  puts the closure before `remap_pairs`; V5 traces ids end to end.
* **v1-F9 (dust at every level) -- accepted**, E plus V2(d).
* **v1-F10 (V3 seam) -- accepted.** `ABISS_WS_NUC_GUARD`, default ON, plus the same-tag
  merge-anyway fixture that actually proves the must-link.
* **v1-F11 (`on_border` oracle) -- accepted**, V2(c) with an independent rescan and full raw count
  word comparison.
* **v1-F12 (V5.6 permits split-only) -- accepted.** V6 names the final `WS_PATH` artifact and adds
  the must-link invariant.
* **v1-F13 (no-op still possible) -- accepted** and answered structurally rather than by detection.
* **v1-F14 (close the open question) -- accepted.** Unconditional, for the stated reason; the
  question is removed rather than left open.
* **Reviewer questions 1-3** are answered in C (closure and primitive), V6 (final `ws3` artifact and
  where the mapping is verified), and V3 (`ABISS_WS_NUC_GUARD` and what its OFF build bypasses).
