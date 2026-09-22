# Plan v1

## Summary

Same goal as v0 -- make "no supervoxel overlaps two nuclei" true at the watershed -- but the
mechanism changes. v0's central idea, giving snapped basins a **nucleus-derived id** so ABISS's
stitching would join a nucleus for free, does not survive `relabel_segments(seg, offset)`: that
function maps compact local labels into the chunk-prefixed namespace, so reserving an id range
does not make nucleus `N` receive the same id in every chunk, and forcing it to would drag the
counts, dendrogram, boundary files, remaps and `chunkmap` along with it. v0-F2 is correct and the
idea is withdrawn.

Answering the reviewer's question 1 explicitly: **compact local ids, plus hierarchy-aware tag
transport.** ABISS's id scheme is untouched. A per-supervoxel `nucleus tag` travels beside the
segmentation in exactly the pattern the agglomeration level already uses for nucleus records
(`nuc_wire_t`, reduced by `reduce_chunk.cpp` and matched by `match_chunks.cpp` from run 1), and
every merge decision consults it.

That makes the snap a **must-link** rather than an id assignment, which is simpler and testable:

* **split** a basin overlapping >= 2 nuclei (seeded watershed within the basin);
* **must-link** basins carrying the same nonzero tag -- this is the snap, and goal (b);
* **cannot-link** basins carrying different nonzero tags, in *every* place a union can happen;
* **never dust** a tagged component;
* **transport** tags across the ws hierarchy, reduced at each level.

## Scope

Unchanged from v0 except that the id-space work (v0 step E) is replaced by tag transport.

## Proposed Changes

### A. Deliver the mask to the ws stage

Unchanged from v0: `cut_chunk_ws.py` writes `nuc.raw` via `nucleus_utils.cut_nucleus_data`,
guarded by `NUC_PATH`, reusing the helper (and its far-edge clamp) rather than re-deriving it.

### B. Split, then rebuild `counts` (v0-F1)

Between `watershed()` and `get_region_graph()`:

1. accumulate, per basin, the set of nucleus ids its voxels carry;
2. for a basin with >= 2, run a seeded watershed restricted to that basin's bounding box, seeded by
   the overlapping nucleus regions; each piece gets a fresh label and its seed's tag; unreached
   voxels form an untagged piece;
3. **re-densify labels and rebuild `counts` from scratch**, preserving the `on_border` bit per
   component. v0 assumed `counts` survived; it does not. `counts` carries sizes *and* border flags
   that `try_merge` reads, and `get_region_graph(..., counts.size()-1, ...)` is indexed by it, so a
   stale `counts` is silent corruption rather than a crash.
4. emit `basin_tag` in the same dense namespace as the rebuilt `counts`.

### C. Guard every union, not just `try_merge` (v0-F5, v0-F7)

`merge_segments` has two phases and v0 only guarded one. Both need it:

* the `try_merge` phase -- note it already rejects when both components are >= `size_threshold`,
  so **high affinity alone does not cause refusion**; v0's claim that C is obviously load-bearing
  was unfounded, and V3 below is rewritten to establish it rather than assert it;
* the `new_rg` / MST loop, which can emit a high-score edge between differently-tagged components
  into the **dendrogram**, where the ws hierarchy or the agglomeration will later consume it.

Rules, applied in both: different nonzero tags -> never union and never emit the edge; same nonzero
tag -> **union unconditionally**, bypassing the size and affinity tests (this is the snap); tagged
+ untagged -> allowed, result inherits the tag.

If the global nucleus-table veto at the agglomeration level would have caught a leaked cross-tag
edge anyway, that makes it load-bearing here too. The plan does not rely on it: V3 asserts the
dendrogram itself contains no cross-tag edge.

### D. Dust removal must preserve tags (v0-F6)

The final remap loop drops components below `lowt` to label 0. A small seeded piece of a nucleus
-- exactly what step B produces -- is eligible. Tagged components are exempt from dusting, and V2
asserts a deliberately tiny tagged piece survives.

### E. Tag transport across the ws hierarchy (v0-F3, v0-F4)

* The side file is written **after** `merge_segments` has compacted labels, keyed by the same ids
  as `seg_<tag>.data`, and emitted once per `out_tag` in the multi-threshold path. v0 keyed it to
  pre-merge basin ids, which would have made every lookup miss and silently turned D into a no-op.
* At each parent level, tags are matched to the level's remaps, propagated through accepted
  tagged+untagged unions, reduced, and re-emitted for the next level -- the same treatment the
  nucleus records get at the agglomeration levels. Reuse that code path's shape; do not invent a
  second one.
* Wire format: reuse `nuc_wire_t` so the existing reducers and their tests apply.

## Files and Areas

| Path | Change |
|---|---|
| `work2/abiss/scripts/cut_chunk_ws.py` | write `nuc.raw` when `NUC_PATH` is set |
| `work2/abiss/src/ws/atomic_chunk.cpp` | load `nuc.raw`; split; rebuild `counts` + dense labels; emit tags per `out_tag` |
| `work2/abiss/src/ws/agglomeration.hpp` | tag-aware `try_merge` **and** MST/dendrogram phase; tag-aware dusting |
| `work2/abiss/src/ws/merge_chunks.cpp` | tag-aware stitching; tag reduce/emit per level |
| `work2/abiss/scripts/{atomic_chunk_ws,remap_chunk_ws}.sh` | carry the tag file alongside `dend`/`seg` |
| `work2/abiss/work/test/` | V1-V5 fixtures |
| `dev/zebrafinch/nuc_z1_y7_x6/` | `wssnap` arm |

Untouched: live `lib/abiss`, `lib/abiss/build/`, `scripts/nucleus_snap.py`.

## Verification Plan

**V1 -- bit-invariance (hard gate).** `run_v2_invariance.sh`, no `NUC_PATH`,
`identical=N differing=0 missing=0`.

**V2 -- split and survival.** Synthetic basin spanning two nucleus seeds with uniform high
affinity between them (mirroring the measured ~0.999 bottleneck). Assert: the basin divides; every
piece carries at most one tag; all of each nucleus's mask voxels land in one piece; **a
deliberately sub-`lowt` tagged piece is still nonzero after dusting** (D); `counts` after the
rebuild matches a recomputed reference including `on_border` bits (B3).

**V3 -- the guards, with a real seam (v0-F7, v0-F8).** Build the fixture so an in-chunk refusion is
genuinely *eligible*: one split piece below `size_threshold`, border conditions satisfied, edge
score above threshold -- verified by asserting the merge happens with the guard compiled out. Then
with the guard on, assert no union **and** no cross-tag edge in `dend_<tag>.data`. The guard is
toggled by an explicit build flag so "disabled" is a real configuration, not a thought experiment.
Additionally report, from the crop run, whether the real split pieces were eligible -- if they were
not, C is inert on this data and the plan says so instead of claiming a fix.

**V4 -- hierarchy with two tags (v0-F9).** Two nuclei meeting across the chunk boundary with an
**untagged bridge** between them. Assert: child+parent equals monolithic; tags survive the parent's
representative changes; no cross-tag stitch. A single-nucleus fixture cannot detect tag loss and is
not sufficient.

**V5 -- crop A/B, with anti-gaming checks (v0-F10).** Arms `w2ctl` and `wssnap`. Report for both:

1. all eight nuclei present, and **each nucleus's mask voxel count assigned to a nonzero label is
   exactly preserved** versus control -- this closes the "improve the metric by dropping mask
   voxels to background" hole;
2. shared mask mass (`--tol 0.0`) 110,244 -> **0**;
3. segments covering 95% of each nucleus's mask == 1 for all eight;
4. dominance >= 0.99 for all eight;
5. largest segment <= 450,000,000; root segment count within 0.9x-1.25x of 3,613;
6. the watershed artifacts themselves, not only final statistics: the number of ws supervoxels
   overlapping >= 2 nuclei must be **0** (it is >= 1 in the control), read from the ws volume and
   the mask directly;
7. ws wall-clock, both arms.

## Risks and Questions

* **[major] The fourth no-op is the thing to design against (v0-F11).** The reviewer's exact
  scenario -- split locally, tags left in the pre-merge namespace, hierarchy reads them as zero,
  cross-tag dendrogram edges emitted, ordinary chunk-offset ids applied, treatment reconstructs the
  control -- is now explicitly blocked at each step: E1 fixes the namespace, E2 the hierarchy, C
  the dendrogram, and V5.6 checks the watershed artifact rather than trusting the end metric.
* **[major] The cut has no image support.** Bottleneck ~0.999 between all three pairs against a 0.3
  agglomeration threshold, so the seeded watershed cuts at an equal-resistance surface, not a
  membrane. Imposed structure, defensible only because the mask is confirmed correct and
  conservative. This belongs in the code comments, not only here.
* **[major] The must-link is a new failure mode.** Forcing same-tag basins to union regardless of
  size and affinity is what buys goal (b), and it can also merge across a genuine membrane if the
  mask ever bleeds. Criteria 3-5 price it and V1 keeps the no-mask path exact.
* **[minor] v0's cost claim was wrong (v0-F12).** Atomic processing is bounded by the ~252^3 chunk;
  the 243M object exists only after stitching across 88 chunks. The per-basin seeded watershed is
  therefore cheap; bounding it to the basin's bbox is still worth doing.
* **Question:** step C's must-link makes the snap unconditional. If a nucleus's mask overlaps two
  basins separated by a real membrane, we merge them. Should the must-link be gated on the pair
  also being adjacent in the region graph above some floor, or is unconditional correct given the
  mask is ground truth? I lean unconditional, matching em_seg, but it is a judgement call.

## Changes Since Previous Plan Version

All twelve findings addressed; the central mechanism changed as a result.

* **v0-F2 (id space) -- accepted, and the design changed.** Nucleus-derived ids are withdrawn;
  reviewer question 1 is answered as "compact local ids + hierarchy-aware tag transport", and the
  snap becomes a must-link (C) instead of an id assignment.
* **v0-F1 (`counts`) -- accepted.** B3 rebuilds dense labels and `counts` including `on_border`.
* **v0-F3 (tag namespace) -- accepted.** E1: keyed post-`merge_segments`, emitted per `out_tag`.
* **v0-F4 (hierarchy transport) -- accepted.** E2 reuses the nucleus-record reduce/match shape.
* **v0-F5 (both phases) -- accepted.** C guards `try_merge` and the MST/dendrogram loop, and V3
  asserts the dendrogram directly rather than relying on the downstream veto.
* **v0-F6 (dusting) -- accepted.** D exempts tagged components; V2 asserts it.
* **v0-F7 (C not shown load-bearing) -- accepted.** v0 asserted it; V3 now establishes eligibility
  first, and reports if C turns out inert on the real crop.
* **v0-F8 (V3 could not fail) -- accepted.** The guard is a build flag so "disabled" is real.
* **v0-F9 (V4 too weak) -- accepted.** Two tags plus an untagged bridge across the boundary.
* **v0-F10 (V5 gameable) -- accepted.** Mask-voxel preservation, the expected nucleus mapping, and
  a direct watershed-artifact check added.
* **v0-F11 (the likely no-op) -- accepted** and turned into the first Risk, with the specific block
  at each step named.
* **v0-F12 (cost claim) -- accepted** and corrected.
