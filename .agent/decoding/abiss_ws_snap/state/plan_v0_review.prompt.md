You are reviewing plan v0 for a CCC run. You are the CODER: review it for executability.

Context you should carry in: you reviewed two earlier plans in the sibling run
`.agent/features/abiss_nucleus/` and returned READY: no on both, and both were withdrawn on the
strength of your findings. Three agglomeration-level attempts at this bug measured out at exactly
zero effect. Apply the same scrutiny here; a plan that looks plausible and moves nothing is the
established failure mode.

RULES:
- Do not edit files.
- Review only the artifacts and text included in this prompt. Do not inspect other repository files.
- Tag each finding as [minor] or [major].
- Ambiguous severity is major.
- End your response with a line of the form: READY: yes   or   READY: no

Judge plan v0 on:
1. Is the insertion point right? The plan puts split+snap between `watershed()` and
   `get_region_graph()` in atomic_chunk.cpp so the RG, dend, counts and boundary faces all derive
   from corrected labels. Check that against the code below. What breaks downstream?
2. Steps C and D (the two cannot-links) are claimed to be load-bearing -- without them the split is
   undone by the chunk's own agglomeration or by cross-chunk stitching. Is that right, and are
   there OTHER places that would undo it that the plan misses?
3. Is the id-space reservation (step E) adequate, given `relabel_segments(seg, offset)` and the
   chunk-prefixed id scheme?
4. Does the per-chunk tag side file survive the ws hierarchy, or does it need the same
   reduce/match treatment the nucleus records needed at the agglomeration level?
5. Are the V5 criteria gameable, and is V3 actually capable of failing?
6. Anything in the plan that would produce a result identical to the control -- i.e. the fourth
   consecutive no-op.

=========================== BEGIN task.md ===========================
# Task

Snap watershed supervoxels onto the nucleus instance mask, and split the ones that overlap more
than one nucleus, so that ABISS's nucleus constraint operates on supervoxels that each belong to
exactly one nucleus.

The literal user request:

> "for the abiss run, just make sure, to merge all ws small supervoxel to overlapped nucleus mask,
> if overlapped with multiple, do a simple seeded watershed with the overlapped region
> (e.g. lib/em_seg)"

asked in answer to "is it better to do it during abiss or better to do another run with
graph-based split?".

## Why the watershed stage, and not a region-graph split

Measured in `.agent/features/abiss_nucleus/state/evidence_conflict.md`: the entire contamination on
the `worst3` crop lives inside **one** watershed supervoxel.

```
globally resolved nucleus table (nuctable arm):
  139,294 records    NONE 5,401   PROPER 133,892   CONFLICT 1
  CONFLICT sv 72198606672811349   tagged 28,576,381
  final CONFLICT segment tagged   28,576,381        -> ratio 1.000
  the object is 243,481,814 vox across 88 atomic chunks before agglomeration starts
```

Region-graph nodes *are* supervoxels, so no graph-based split can separate voxels that share a
node. Only the watershed stage can. Four agglomeration-level levers were tried first and all
measured out at zero effect on this: merge ordering, the `nuc_can_merge` CONFLICT clause, the
global nucleus table, and the frozen-boundary rule (already strict -- `agg` is built with `FINAL`
only, `EXTRA` is commented out in `CMakeLists.txt:131-135`).

The affinity offers no help in placing the cut: min-pooled bottleneck between all three nucleus
pairs is ~0.999, i.e. **any** separating surface must sever affinity >= 0.999, versus a 0.3
agglomeration threshold. So the split is imposed by the mask, not discovered in the image. That is
acceptable here only because the task owner has confirmed the mask is correct and conservative
(smaller than the true nucleus), which makes it ground truth for "these voxels are one neuron".

## Repository and baseline

`lib/abiss`, worked in `work2/abiss` (symlinked here), detached at **312bf54**. The working tree
already carries the completed, unreviewed-for-merge global-nucleus-table change from the
`abiss_nucleus` run 2 (`state/run_start.diff`, 257 lines). That is the baseline for this run --
do not revert it.

Never touch the live `lib/abiss` checkout or `lib/abiss/build/`; SLURM jobs execute those binaries.

## Required behaviour

1. **Snap.** After the atomic watershed, every supervoxel overlapping nucleus `N`'s mask is
   relabelled to a single id derived deterministically from `N` -- **not** a chunk-local id. This
   is `em_seg/seg_pipeline.py:_affinityToSeg2D` ("snap to soma id",
   `ii = np.unique(seg[0][mask_soma==soma_id]); rl[ii] = seg_m + i`).
   Because the id is derived from `N`, ABISS's ws remap (`merge_remaps.py` -> `ws3` ->
   `chunkmap.data`) stitches `N`'s pieces across chunks for free, and a merge between two different
   nuclei becomes inexpressible rather than something that has to be filtered out. That is the
   `waterz.somaBFS` guarantee obtained by construction.
2. **Split.** A supervoxel overlapping two or more nuclei is divided by a seeded watershed
   restricted to that supervoxel's own voxels, seeded by the overlapping nucleus regions. em_seg
   marks this case `# need to split sometimes` and does not handle it; this run must.
3. The nucleus id space must not collide with watershed ids. Reserve or offset explicitly.

## Hard constraints

* **Default-path bit-invariance.** With no `NUC_PATH`, the watershed output must be byte-identical
  to baseline. `work/test/run_v2_invariance.sh` (already retargeted to `312bf54` by run 2) is the
  gate. Non-negotiable: the pipeline reproduces a Seuron provenance record.
* **Do the relabel before `chunkmap` is generated.** `chunkmap.data` records a remap already
  applied to the watershed volume; injecting later strands the snapped-away supervoxels. This is
  exactly how `scripts/nucleus_snap.py` shattered nucleus 173 into 39,439 fragments in run 1. That
  script stays disabled and is not a starting point.
* Chunk-locality: an atomic chunk is 252^3 (~2.3 x 2.3 x 5 um), smaller than a soma, so a chunk
  often sees only part of one nucleus. The snap must still be correct under that, which is why the
  id is derived from the nucleus rather than assigned locally.
* No git commits. No new dependencies.

## Success criteria

Measured on the `worst3` crop (BBOX `[2772,9324,1260,3780,10584,2520]`), against the `w2ctl`
control, with `dev/zebrafinch/nucleus_shell_contamination.py --tol 0.0`:

1. **Primary** -- shared mask mass (mask voxels in any segment holding mask voxels of >= 2 nuclei)
   drops from **110,244** to **0**. With one supervoxel per nucleus this should be exact, not
   approximate; anything above 0 means the snap leaked.
2. **Fragmentation** -- segments needed to cover 95% of each nucleus's mask goes to **1** for all
   eight nuclei (control: 2 for 275/319/373, 1 for the rest), and the whole-volume median of 34
   segments per nucleus is the standing target for the follow-up run.
3. **Dominance** -- >= 0.99 for all eight nuclei (control 0.8957 / 0.9104 / 0.8988 for the three,
   1.000 for the clean five).
4. **No runaway, no shatter** -- largest segment <= 450,000,000 voxels (control 291,235,662,
   `nonuc` 640,562,148); root segment count within 0.9x-1.25x of the control's 3,613.
5. `run_v2_invariance.sh` passes.

Report all five for control and treatment side by side. A treatment that buys criterion 1 by
shattering the somata fails 2-4.

## Out of scope

The two `abiss_nucleus` review_v0 findings (per-worker table memory, stale-table provenance) --
they belong to that run's `code_v1`. Whole-volume re-run. NERL.
=========================== END task.md ===========================

=========================== BEGIN artifacts/plan_v0.md ===========================
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
=========================== END plan_v0.md ===========================

Sources. NOTE the working tree already carries the sibling run's global-nucleus-table change.

=========================== BEGIN src/ws/atomic_chunk.cpp lines 100-235 ===========================
        merge_thresholds.push_back(low_threshold);
    }

    std::cout << "thresholds: " << ht << " " << lt << " " << st << " " << dt
              << " merge=[";
    for (size_t i = 0; i < merge_thresholds.size(); i++) {
        if (i > 0) std::cout << ",";
        std::cout << merge_thresholds[i];
    }
    std::cout << "]" << std::endl;

    param_file >> xdim >> ydim >> zdim;
    std::cout << xdim << " " << ydim << " " << zdim << std::endl;

#ifdef USE_MIMALLOC
    size_t huge_pages = xdim * ydim * zdim * 4 * 3 * 4 / 1024 / 1024 / 1024 + 1;
    auto mi_ret = mi_reserve_huge_os_pages_interleave(huge_pages, 0, 0);
    if (mi_ret == ENOMEM) {
       std::cout << "failed to reserve 1GB huge pages" << std::endl;
    }
#endif

    std::array<bool,6> flags({true,true,true,true,true,true});
    for (size_t i = 0; i != 6; i++) {
        param_file >> flag;
        flags[i] = (flag > 0);
        if (flags[i]) {
            std::cout << "real boundary: " << i << std::endl;
        }
    }
    param_file >> offset;
    std::cout << "supervoxel id offset:" << offset << std::endl;

    size_t chunk_size = xdim * ydim * zdim;

    assert(chunk_size < static_cast<size_t>(watershed_traits<internal_seg_t>::high_bit));

    clock_t begin = clock();
    std::array<size_t, 4> aff_dim({xdim,ydim,zdim,3});
    MMArray<aff_t, 4> aff_data(argv[2], aff_dim);
    affinity_graph_ptr<aff_t> aff = aff_data.data_ptr();
    //    read_affinity_graph<float>(argv[2],
    //                               xdim, ydim, zdim);
    //                               //2050, 2050, 258);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "loaded affinity map in " << elapsed_secs << " seconds" << std::endl;

    volume_ptr<internal_seg_t> seg;
    std::vector<std::size_t> counts;

    begin = clock();
    std::tie(seg , counts) = watershed<internal_seg_t>(aff, low_threshold, high_threshold, flags);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "finished watershed in " << elapsed_secs << " seconds" << std::endl;
    begin = clock();
    auto rg = get_region_graph(aff, seg , counts.size()-1, low_threshold, flags, score_cfg);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "finished region graph in " << elapsed_secs << " seconds" << std::endl;

    if (merge_thresholds.size() == 1) {
        // ------ Single merge threshold: original behaviour ------
        begin = clock();
        merge_segments(seg, rg, counts, std::make_pair(size_threshold, merge_thresholds[0]), dust_threshold);
        end = clock();
        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

        auto relabeled_seg = relabel_segments(seg, offset);
        free_container(seg);
        auto relabeled_rg = relabel_region_graph(rg, offset);
        free_container(rg);

        std::cout << "finished agglomeration in " << elapsed_secs << " seconds" << std::endl;
        auto c = write_counts(counts, offset, tag);
        free_container(counts);
        auto d = write_vector(str(boost::format("dend_%1%.data") % tag), relabeled_rg);
        free_container(relabeled_rg);
        begin = clock();
        write_volume(str(boost::format("seg_%1%.data") % tag), relabeled_seg);
        write_chunk_boundaries(relabeled_seg, aff, flags, tag);
        std::vector<size_t> meta({xdim,ydim,zdim,c,d,0});
        write_vector(str(boost::format("meta_%1%.data") % tag), meta);
        std::cout << "num of sv:" << c << std::endl;
        std::cout << "size of rg:" << d << std::endl;
        end = clock();
        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "finished writing in " << elapsed_secs << " seconds" << std::endl;
    } else {
        // ------ Multiple merge thresholds: reuse watershed + RG ------
        std::cout << "Multi-threshold mode: " << merge_thresholds.size()
                  << " merge thresholds" << std::endl;

        for (size_t mi = 0; mi < merge_thresholds.size(); mi++) {
            clock_t mt_begin = clock();

            // Deep-copy seg, rg, counts so merge_segments can modify them
            auto seg_copy = volume_ptr<internal_seg_t>(
                new volume<internal_seg_t>(
                    boost::extents[xdim][ydim][zdim],
                    boost::fortran_storage_order()));
            std::copy(seg->data(), seg->data() + chunk_size, seg_copy->data());
            auto rg_copy = rg;
            auto counts_copy = counts;

            std::string out_tag = str(boost::format("%1%_%2%") % tag % mi);

            merge_segments(seg_copy, rg_copy, counts_copy,
                           std::make_pair(size_threshold, merge_thresholds[mi]),
                           dust_threshold);

            auto relabeled_seg = relabel_segments(seg_copy, offset);
            free_container(seg_copy);
            auto relabeled_rg = relabel_region_graph(rg_copy, offset);
            free_container(rg_copy);

            auto c = write_counts(counts_copy, offset, out_tag.c_str());
            free_container(counts_copy);
            auto d = write_vector(str(boost::format("dend_%1%.data") % out_tag), relabeled_rg);
            free_container(relabeled_rg);

            write_volume(str(boost::format("seg_%1%.data") % out_tag), relabeled_seg);
            write_chunk_boundaries(relabeled_seg, aff, flags, out_tag.c_str());
            std::vector<size_t> meta({xdim,ydim,zdim,c,d,0});
            write_vector(str(boost::format("meta_%1%.data") % out_tag), meta);

            clock_t mt_end = clock();
            double mt_secs = double(mt_end - mt_begin) / CLOCKS_PER_SEC;
            std::cout << "merge threshold " << mi << " (" << merge_thresholds[mi]
                      << "): sv=" << c << " rg=" << d
                      << " in " << mt_secs << " seconds" << std::endl;
        }

        free_container(seg);
        free_container(rg);
=========================== END ===========================

=========================== BEGIN src/ws/agglomeration.hpp ===========================
#pragma once

#include "types.hpp"

#include <boost/pending/disjoint_sets.hpp>
#include <map>
#include <vector>
#include <set>
#include <iostream>

template<typename C, typename S, typename T>
inline bool try_merge(C & counts, S & sets, T s1, T s2, size_t size_threshold)
{
    using traits = watershed_traits<T>;
    auto real_size_s1 = counts[s1]&(~traits::on_border);
    auto real_size_s2 = counts[s2]&(~traits::on_border);
    if ( (real_size_s1 >= size_threshold) && (real_size_s2 >= size_threshold) ) {
        return false;
    }
    if (((traits::on_border&(counts[s1]|counts[s2]))==0)
          || (real_size_s1 >= size_threshold && counts[s2] < size_threshold)
          || (real_size_s2 >= size_threshold && counts[s1] < size_threshold)) {
    //if ((traits::on_border&(counts[s1]|counts[s2]))==0) {
        counts[s1] += counts[s2]&(~traits::on_border);
        counts[s1] |= counts[s2]&traits::on_border;
        counts[s2]  = 0;
        sets.link(s1, s2);
        T s = sets.find_set(s1);
        std::swap(counts[s], counts[s1]);
        return true;
    }
    else {
        counts[s1] |= counts[s2]&traits::on_border;
        counts[s2] |= counts[s1]&traits::on_border;
        return false;
    }
}

template< typename ID, typename F, typename L, typename M >
inline void merge_segments( const volume_ptr<ID>& seg_ptr,
                            region_graph<ID,F>& rg,
                            std::vector<std::size_t>& counts,
                            const L& tholds,
                            const M& lowt)
{
    using traits = watershed_traits<id_t>;
    std::vector<ID> rank(counts.size());
    std::vector<ID> parent(counts.size());
    boost::disjoint_sets<ID*, ID*> sets(&rank[0], &parent[0]);
    for (ID i = 0; i < counts.size(); i++) {
        sets.make_set(i);
    }

    typename region_graph<ID,F>::iterator rit = rg.begin();

    std::size_t size = static_cast<std::size_t>(tholds.first);
    //F           thld = static_cast<F>(it.second);

    while ( (rit != rg.end()) && ( std::get<0>(*rit) > tholds.second) )
    {
        ID s1 = sets.find_set(std::get<1>(*rit));
        ID s2 = sets.find_set(std::get<2>(*rit));

        if ( s1 != s2 && s1 && s2 )
        {
            try_merge(counts, sets, s1, s2, size);
        }
        ++rit;
    }

    std::cout << "Done with merging" << std::endl;

    std::vector<ID> remaps(counts.size());

    counts[0] &= ~traits::on_border;
    remaps[0] = 0;

    ID next_id = 1;

    std::size_t low = static_cast<std::size_t>(lowt);

    for ( ID id = 0; id < counts.size(); ++id )
    {
        ID s = sets.find_set(id);
        if ( counts[id]&(~traits::on_border) ) {
            if ( s && (counts[s] >= low) )
            {
                if (remaps[s] == 0) {
                    remaps[s] = next_id;
                    counts[next_id] = counts[s]&(~traits::on_border);
                    ++next_id;
                }
            } else {
                counts[s] = remaps[s] = 0;
            }
        }
    }

    counts.resize(next_id);

    std::ptrdiff_t xdim = seg_ptr->shape()[0];
    std::ptrdiff_t ydim = seg_ptr->shape()[1];
    std::ptrdiff_t zdim = seg_ptr->shape()[2];

    std::ptrdiff_t total = xdim * ydim * zdim;

    ID* seg_raw = seg_ptr->data();

    for ( std::ptrdiff_t idx = 0; idx < total; ++idx )
    {
        seg_raw[idx] = remaps[sets.find_set(seg_raw[idx])];
    }

    std::cout << "Done with remapping, total: " << (next_id-1) << std::endl;

    region_graph<ID,F> new_rg;

    std::vector<std::set<ID>> in_rg(next_id);

    std::vector<ID> rank_mst(next_id);
    std::vector<ID> parent_mst(next_id);
    boost::disjoint_sets<ID*, ID*> mst(&rank_mst[0], &parent_mst[0]);
    for (ID i = 0; i < next_id; i++) {
        mst.make_set(i);
    }

    for ( auto& it: rg )
    {
        ID s1 = remaps[sets.find_set(std::get<1>(it))];
        ID s2 = remaps[sets.find_set(std::get<2>(it))];
        ID a1 = mst.find_set(s1);
        ID a2 = mst.find_set(s2);

        if ( a1 != a2 && a1 && a2 && std::get<0>(it) > tholds.second)
        {
            mst.link(a1, a2);
            auto mm = std::minmax(s1,s2);
            if ( in_rg[mm.first].count(mm.second) == 0 )
            {
                new_rg.emplace_back(std::get<0>(it), mm.first, mm.second);
                in_rg[mm.first].insert(mm.second);
            }
        }
    }

    rg.swap(new_rg);

    std::cout << "Done with updating the region graph, size: "
              << rg.size() << std::endl;
}
=========================== END ===========================

=========================== BEGIN scripts/atomic_chunk_ws.sh + remap_chunk_ws.sh + merge_remaps.py ===========================
#!/bin/bash
set -euo pipefail
INIT_PATH="$(dirname "$0")"
. ${INIT_PATH}/init.sh $1
output_chunk=`basename $1 .json`
output_path=out/ws
try acquire_cpu_slot
try mkdir remap
try mkdir -p ${output_path}/{seg,dend}
try taskset -c $cpuid python3 $SCRIPT_PATH/cut_chunk_ws.py $1
try taskset -c $cpuid $BIN_PATH/ws param.txt aff.raw $WS_HIGH_THRESHOLD $WS_LOW_THRESHOLD $WS_SIZE_THRESHOLD $WS_DUST_THRESHOLD $output_chunk
try touch ongoing_"${output_chunk}".data
try $COMPRESS_CMD seg_"${output_chunk}".data
try mv seg_"${output_chunk}".data."${COMPRESSED_EXT}" ${output_path}/seg/
try mv remap ${output_path}/
if [ "$PARANOID" = "1" ]; then
    try md5sum *_"${output_chunk}".data > ${output_path}/dend/"${output_chunk}".data.md5sum
fi
try tar -cvf - *_"${output_chunk}".data | $COMPRESS_CMD > ${output_path}/dend/"${output_chunk}".tar."${COMPRESSED_EXT}"
retry 10 $UPLOAD_CMD "${output_path}" $IO_SCRATCH_PATH/
try rm -rf ${output_path}
try release_cpu_slot
#!/bin/bash
set -euo pipefail
INIT_PATH="$(dirname "$0")"
. ${INIT_PATH}/init.sh $1
output_chunk=`basename $1 .json`

try acquire_cpu_slot
try touch chunkmap.data
retry 10 $DOWNLOAD_CMD $FILE_PATH/seg/seg_"${output_chunk}".data."${COMPRESSED_EXT}" seg_"${output_chunk}".data."${COMPRESSED_EXT}"
try python3 $SCRIPT_PATH/merge_remaps.py $1
try $COMPRESS_CMD -d seg_"${output_chunk}".data."${COMPRESSED_EXT}"
try taskset -c $cpuid $BIN_PATH/ws3 param.txt seg_"${output_chunk}".data
try taskset -c $cpuid python3 $SCRIPT_PATH/upload_chunk.py $1 $WS_PATH $WS_MIP
try mv chunkmap.data chunkmap_"${output_chunk}".data
retry 10 $COMPRESS_CMD chunkmap_"${output_chunk}".data
retry 10 $UPLOAD_CMD chunkmap_"${output_chunk}".data."${COMPRESSED_EXT}" "$CHUNKMAP_OUTPUT"/chunkmap_"${output_chunk}".data."${COMPRESSED_EXT}"

try release_cpu_slot
import sys
import os
import chunk_utils as cu

def merge_remaps(prefixes, ancestor_tags, offset):
    content = b''
    for a in ancestor_tags:
        for p in prefixes:
            payload = cu.download_slice(p, a, offset)
            if payload:
                content += payload

    with open('remap.data','wb') as out:
        out.write(content)

    return len(content)

param = cu.read_inputs(sys.argv[1])
global_param = cu.read_inputs(os.environ['PARAM_JSON'])

ancestors = cu.generate_ancestors(sys.argv[1])
ancestors = list(ancestors)
if os.environ["STAGE"] == "ws":
    prefixes = ['remap/done_pre', 'remap/done_post']
    ancestors = ancestors[1:]
elif os.environ["STAGE"] == "agg":
    prefixes = ['remap/done']

offset = param["offset"]

chunked_agg_output = False
if len(sys.argv) > 2:
    chunked_agg_output = bool(sys.argv[2])

#print(ancestors)

if param["mip_level"] == 0:
    bbox = param["bbox"]
    sizes = [bbox[i+3]-bbox[i] for i in range(3)]
    actual_size = merge_remaps(prefixes, ancestors, offset)//16
    print(actual_size)
    with open("param.txt","w") as f:
        f.write(" ".join([str(i) for i in sizes]))
        f.write("\n")
        f.write(str(actual_size))
        f.write("\n")
        if os.environ["STAGE"] == "ws" or chunked_agg_output:
            f.write("1")
        else:
            f.write("0")
else:
    print("only atomic chunks need remapping")
=========================== END ===========================

=========================== BEGIN em_seg reference: _affinityToSeg2D (the snap + somaBFS this plan follows) ===========================
    def _affinityToSeg2D(self, aff, mask_bv=None, mask_soma=None, mask_border=None):
        # aff: 3x1xHxW
        param = self.param_s['SEG2D']
        ws_low, ws_nb, ws_dust = param['AFF_LOW_THRES'], param['WS_NB_SIZE'], param['DUST_SIZE']
        rg_m1_func, rg_m1_aff = param['RG_MERGE_FUNC'], param['RG_MERGE_AFF']
        rg_m2_size, rg_m2_aff = param['RG_MERGE2_SIZE'], param['RG_MERGE2_AFF']

        # 1. set low-aff region to background
        aff[aff < ws_low] = 0
        # get external mask
        if mask_bv is not None:
            aff[:,0] = aff[:,0] * (1 - mask_bv)
        if mask_border is not None:
            aff[:,0] = aff[:,0] * (1 - mask_border)

        # initial watershed
        seg = waterz.watershed(aff, label_nb = np.ones([ws_nb,ws_nb]), bg_thres = 1 - ws_low/255.)
        seg_m = seg.max() + 1

        # snap to soma id
        soma_rl = np.zeros([1,2])
        if mask_soma is not None and mask_soma.any():
            soma_ids = np.unique(mask_soma[mask_soma>0])
            rl = np.arange(seg_m).astype(self.dtype)
            # need to split sometimes
            soma_rl = np.zeros([len(soma_ids),2],int)
            for i,soma_id in enumerate(soma_ids):
                ii = np.unique(seg[0][mask_soma==soma_id])
                # make sure no overlap among somas 
                ii = ii[(ii>0)*(ii<seg_m)]
                # remove small olap
                if len(ii) > 0:
                    rl[ii] = seg_m + i
                    soma_rl[i] = [soma_id, seg_m + i]
            # merge soma regions
            seg = rl[seg]
            
        # compute region graph
        rg_id, rg_score = waterz.getRegionGraph(aff, seg, 1, rg_m1_func, rebuild=False)

        # merge 1: conservative
        jj = rg_id[rg_score <= rg_m1_aff]
        if soma_rl[:,1].any():
            # remove pairs that lead to soma seg to merge
            jj = waterz.somaBFS(jj, soma_rl[soma_rl[:,1]>0, 1])

        out = waterz.merge_id(jj[:,0], jj[:,1], id_thres = seg_m)
        out_l = len(out)
        seg[seg < out_l] = out[seg[seg < out_l]]
        
        # merge 2: small size with higher thres [sorted aff]
=========================== END ===========================

Now produce your review: a short summary, then findings each tagged [minor] or [major], then
any questions, then the final READY: line.
