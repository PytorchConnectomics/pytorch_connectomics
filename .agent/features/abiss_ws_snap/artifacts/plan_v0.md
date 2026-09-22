# Plan v0

## Summary

Make "no supervoxel overlaps two nuclei" true at the watershed, so that ABISS's existing nucleus
veto -- which has always assumed it -- finally has it.

The measured defect is that one watershed supervoxel holds 100.0% of the contamination
(`sv 72198606672811349`, 28,576,381 tagged voxels, 243M voxels across 88 atomic chunks). The veto
operates on supervoxels and so cannot see inside one; that is why merge ordering, the CONFLICT
clause, the global table and the frozen rule all measured out at zero effect.

Four pieces, in `src/ws/`:

1. **Split** a chunk-local basin that overlaps two or more nuclei, by a seeded watershed restricted
   to that basin. This is the case `em_seg/seg_pipeline.py` marks `# need to split sometimes` and
   does not handle.
2. **Snap** every basin overlapping exactly one nucleus onto a single per-nucleus label, as
   `_affinityToSeg2D` does (`ii = np.unique(seg[0][mask_soma==soma_id]); rl[ii] = seg_m + i`).
3. **Cannot-link inside the chunk**: `merge_segments` must not merge two basins carrying different
   nucleus tags. Without this the watershed's own agglomeration (which runs down to
   `WS_LOW_THRESHOLD=0.00001`) immediately re-fuses what step 1 split.
4. **Cannot-link across chunks**: the same rule in `merge_chunks.cpp`, because basins are
   chunk-local and stitching is identity-based on boundary faces. A 252^3 chunk is ~2.3 x 2.3 x 5
   um, smaller than a soma, so a nucleus is always split across chunks and steps 1-3 alone would be
   undone at the seam.

Steps 3 and 4 are together the `waterz.somaBFS` guarantee. Step 1 is the new work.

## Scope

In scope: `src/ws/atomic_chunk.cpp`, `src/ws/agglomeration.hpp`, `src/ws/merge_chunks.cpp`,
`scripts/cut_chunk_ws.py` (to deliver the mask into the ws stage as `cut_chunk_agg.py` already does
for agglomeration), the per-chunk nucleus-tag side file, and the crop A/B.

Out of scope: the two `abiss_nucleus` review_v0 findings (they belong to that run's `code_v1`);
whole-volume re-run; NERL; `scripts/nucleus_snap.py`, which stays disabled.

## Proposed Changes

### A. Deliver the nucleus mask to the watershed stage

`cut_chunk_agg.py` already cuts the mask for agglomeration via `nucleus_utils.cut_nucleus_data`
(including the far-edge clamp). `cut_chunk_ws.py` must do the same and write `nuc.raw` alongside
`aff.raw`, guarded by `NUC_PATH` exactly as the agglomeration path is. Reuse the helper; do not
re-derive the transform.

### B. Split and snap, in `atomic_chunk.cpp`

The current order is

```
watershed()  ->  get_region_graph()  ->  merge_segments()  ->  relabel_segments()  ->  write {counts, dend, seg, boundaries, meta}
```

Insert between `watershed()` and `get_region_graph()`, so that the region graph, the dendrogram,
the counts and the boundary faces are all derived from the corrected labels and stay consistent
with each other:

* For each basin, accumulate the set of nucleus ids its voxels carry.
* **Overlap == 1:** record the basin's tag.
* **Overlap >= 2:** run a seeded watershed restricted to that basin's voxels, seeded by the
  overlapping nucleus regions, using the same affinity. Assign each resulting piece a fresh basin
  id and the tag of its seed. Voxels the flood does not reach keep an untagged piece.
* Produce `std::vector<nuc_t> basin_tag` indexed by basin id, 0 == untagged.

Doing this before `get_region_graph` is what makes the split stick: the RG is then built over the
split basins, so there is an explicit edge between the pieces rather than no edge at all.

The snap onto a single per-nucleus label is a relabel of the basins sharing a tag. Note it is
**not** required for correctness -- the tag plus steps C/D already prevent fusion -- but it is
required for goal (b), collapsing the whole-volume median of 34 segments per nucleus to 1.

### C. Cannot-link in the chunk's own agglomeration

`merge_segments` (`src/ws/agglomeration.hpp`) merges RG edges above `merge_threshold`
(`WS_LOW_THRESHOLD` = 1e-5), so without a guard it re-fuses the split immediately. Add: skip an
edge whose two endpoints have different nonzero tags, propagating the tag on every accepted merge
(a merge of tagged + untagged yields the tag). Same shape as `nuc_can_merge`, one level down.

### D. Cannot-link across chunk boundaries

`merge_chunks.cpp` stitches basins by boundary-face identity. Carry the per-chunk `basin_tag` as a
side file written next to `dend_<tag>.data`, load it there, and refuse a stitch between different
nonzero tags. Without D, nucleus 275's basin in one chunk and 319's in the next are stitched into
one supervoxel and steps A-C are wasted.

### E. Id space

Snapped labels must not collide with watershed ids. `relabel_segments(seg, offset)` assigns from
the chunk's `offset`; reserve an explicit sub-range for nucleus labels and assert on overflow
rather than trusting that nucleus ids are small.

## Files and Areas

| Path | Change |
|---|---|
| `work2/abiss/scripts/cut_chunk_ws.py` | cut and write `nuc.raw` when `NUC_PATH` is set, via `nucleus_utils` |
| `work2/abiss/src/ws/atomic_chunk.cpp` | load `nuc.raw`; split multi-nucleus basins; snap; build `basin_tag`; write the tag side file |
| `work2/abiss/src/ws/agglomeration.hpp` | tag-aware `merge_segments` |
| `work2/abiss/src/ws/merge_chunks.cpp` | tag-aware stitching |
| `work2/abiss/work/test/` | fixtures below |
| `dev/zebrafinch/nuc_z1_y7_x6/` | `wssnap` arm |

Untouched: live `lib/abiss`, `lib/abiss/build/`, `scripts/nucleus_snap.py`.

## Verification Plan

Environment `set +u; source .../activate pytc; set -u`; build only in `work2/abiss/build`.

**V1 -- bit-invariance (hard gate).** `work/test/run_v2_invariance.sh`, already retargeted to
`312bf54`, with no `NUC_PATH`. Must report `identical=N differing=0 missing=0`. The pipeline
reproduces a Seuron provenance record.

**V2 -- the split, on a synthetic basin.** A fixture where one basin provably spans two nucleus
seeds (uniform high affinity between them, mirroring the measured bottleneck of ~0.999): assert the
basin is divided, that every resulting piece carries at most one tag, and that each nucleus's mask
voxels all land in one piece.

**V3 -- the two cannot-links.** With C disabled, assert the split basin re-fuses (this is the test
that proves C is load-bearing rather than decoration). With D disabled, assert two chunks' tagged
basins stitch. With both enabled, neither happens.

**V4 -- hierarchy.** `run_hierarchy.sh` with a nucleus spanning the chunk boundary: child+parent
must equal monolithic.

**V5 -- the crop A/B.** Arms `w2ctl` (control, already run) and `wssnap`, via `run_arm_table.sh`
adapted. Criteria fixed here, before the treatment is seen:

1. **primary** -- shared mask mass (`nucleus_shell_contamination.py --tol 0.0`) **110,244 -> 0**.
   With one supervoxel per nucleus this is exact, not approximate; anything above 0 means the snap
   leaked and should be explained, not tuned away.
2. **fragmentation** -- segments covering 95% of each nucleus's mask == 1 for all eight
   (control: 2 for 275/319/373).
3. **dominance** -- >= 0.99 for all eight (control 0.8957 / 0.9104 / 0.8988 and 1.000 x5).
4. **no runaway** -- largest segment <= 450,000,000 (control 291,235,662; `nonuc` 640,562,148).
5. **no shatter** -- root segment count within 0.9x-1.25x of 3,613.
6. **cost** -- ws wall-clock vs control; the per-basin seeded watershed is the new expense.

## Risks and Questions

* **[major] The cut location has no image support.** The min-pooled bottleneck between all three
  nucleus pairs is ~0.999, so any separating surface severs affinity >= 0.999 against a 0.3
  agglomeration threshold. The seeded watershed will cut where its fronts meet, which is an
  equal-resistance surface, not a membrane. This is imposed structure and is only defensible
  because the task owner has confirmed the mask is correct and conservative. It should be stated
  in the code, not just here.
* **[major] Step 1 changes the watershed for every run with a mask, not only where nuclei
  collide.** Snapping merges basins that the affinity kept apart; criteria 3-5 exist to price that,
  and V1 guarantees the no-mask path is untouched.
* **[major] Steps C and D are where this most likely fails silently.** If either is missed the
  split is undone and the crop result looks exactly like the control -- which is precisely the
  failure mode of the last three attempts. V3 is written to fail loudly instead.
* **[minor] Cost.** A seeded watershed per multi-nucleus basin, on a basin that can be 243M voxels.
  Restricting the flood to the basin's bounding box is necessary, not an optimisation.
* **Question:** should the snap (goal b) ship in the same change as the split (goal a)? They are
  separable, and the split alone is what the contamination needs. Shipping both at once makes a
  regression in criteria 3-5 ambiguous between the two. I lean to both, since the user asked for
  both and V3/V5 can attribute, but flag it if you disagree.

## Changes Since Previous Plan Version

Initial plan.
