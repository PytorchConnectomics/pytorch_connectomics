You are reviewing plan v1 for a CCC run. You are the CODER: review it for executability.

You reviewed plan v0 and returned READY: no with twelve findings. Plan v1 accepts all twelve and
CHANGES THE CENTRAL MECHANISM as a result: your F2 killed the nucleus-derived-id design, and your
question 1 is answered as "compact local ids + hierarchy-aware tag transport", which turns the snap
into a must-link instead of an id assignment.

Three prior attempts at this bug (in the sibling run) each measured out at exactly zero effect on
the target metric. Your F11 described precisely how this one could become the fourth. Judge
whether plan v1 actually blocks that path or merely says it does.

RULES:
- Do not edit files.
- Review only the artifacts and text included in this prompt. Do not inspect other repository files.
- Tag each finding as [minor] or [major].
- Ambiguous severity is major.
- End your response with a line of the form: READY: yes   or   READY: no

Judge plan v1 on:
1. Does the compact-ids + tag-transport design actually work end to end, or does it have the same
   class of hole the id design had? Trace a nucleus through: split -> counts rebuild -> tag emit ->
   merge_segments -> relabel -> boundary files -> ws remap -> chunkmap -> agglomeration.
2. The snap is now an UNCONDITIONAL must-link (same nonzero tag -> union, bypassing size and
   affinity). Is that safe in `merge_segments`, given what it does to `counts`, `on_border` and the
   MST? Could it produce a cycle, a size-invariant violation, or a component that dusting or the
   hierarchy then mishandles?
3. Is V3's build-flag seam a real test, and does V2's `counts` reference check actually verify
   `on_border`?
4. V5.6 reads the ws volume and mask directly to assert zero multi-nucleus supervoxels. Is that the
   right invariant, and is it checkable from the artifacts that exist?
5. Anything still capable of producing a result identical to the control.
6. The open question at the end of Risks: unconditional must-link vs gating on region-graph
   adjacency. Which is correct?

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

=========================== BEGIN artifacts/plan_v0_review.md ===========================
# Plan v0 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, `xhigh`), read-only sandbox. Raw transcript:
`state/plan_v0_review.review.raw.md`. Twelve findings, eleven major, none softened here.

The insertion point (between `watershed()` and `get_region_graph()`) is judged conceptually
correct, but the plan is not executable: it omits the `counts` rebuild, the canonical-id encoding
is unworkable as described, and tag propagation across the ws hierarchy is absent. The reviewer's
closing finding is the one that matters most -- it describes, step by step, how every individual
change could run and still reconstruct the control segmentation, which is the failure mode of the
three preceding attempts.

## Findings

* **[major] `counts` must be rebuilt.** It describes the original basins and carries `on_border`
  bits that `try_merge` reads; `get_region_graph(..., counts.size()-1, ...)` is indexed by it.
  Splitting, coalescing and adding labels invalidates sizes, indexing and flags.
* **[major] Step E is inadequate and conflicts with the implementation.**
  `relabel_segments(seg, offset)` converts compact local labels into the chunk-prefixed namespace,
  so reserving a nucleus range does not give nucleus `N` the same id in every chunk. An exact
  formula and a path through segmentation, counts, dendrogram endpoints, boundary files and remaps
  is required; canonical ids applied before `get_region_graph` may also exceed `internal_seg_t` or
  make `basin_tag[id]` impractically sparse.
* **[major] The tag side file must align to post-`merge_segments` labels**, not pre-merge basin
  ids, and must be emitted per `out_tag` in the multi-threshold path -- otherwise `merge_chunks.cpp`
  looks up absent tags, treats everything as untagged, and D becomes a silent no-op.
* **[major] An atomic per-chunk side file does not survive the hierarchy.** Parent merges change
  representatives and emit remaps; tags must be matched, propagated through tagged+untagged unions,
  reduced at every parent and re-emitted -- the same treatment the nucleus records needed.
* **[major] C must guard both phases of `merge_segments`.** The `new_rg`/MST loop can still emit a
  cross-tag edge into the dendrogram for the hierarchy or agglomeration to consume. If the global
  nucleus-table veto saves it, that veto is load-bearing and the claim that C+D alone give the
  `somaBFS` guarantee is false.
* **[major] Dust removal is a missed destructive path.** The final remap loop drops components
  below `lowt` without regard to tags, so a small seeded nucleus piece can become label 0.
* **[major] C is not shown to be load-bearing.** `try_merge` rejects when both components are
  >= `size_threshold`, so high affinity alone does not cause refusion. V3 must force eligibility
  and report whether the real crop's split pieces were eligible at all.
* **[major] V3 cannot reliably exercise either guard as written.** "With C/D disabled" needs a
  concrete seam, and the fixtures must force an eligible in-chunk merge and an eligible stitch.
* **[major] V4 uses one nucleus spanning a boundary**, so it cannot detect tag loss or a broken
  cross-tag veto. It needs two tags meeting through a boundary, preferably with an untagged bridge,
  and should inspect parent tags/remaps as well as segmentation equality.
* **[major] V5 is gameable.** Shared mass, fragmentation and dominance all improve if mask voxels
  are dropped to background, painted with final ids without fixing the watershed, or omitted from
  the report. Require all eight nuclei present, exact preservation of their mask voxel counts as
  nonzero labels, the expected nucleus-to-id mapping, and inspection of the watershed/chunkmap
  artifacts.
* **[major] The most likely fourth no-op:** split locally, leave tags in the pre-merge namespace,
  let the hierarchy read them as zero, emit cross-tag dendrogram edges, apply ordinary chunk-offset
  ids. Every change runs; the treatment reconstructs the control.
* **[minor] The 243M-voxel per-basin cost claim is misleading** at this insertion point: atomic
  processing is bounded by the ~252^3 chunk, and the 243M object exists only after stitching across
  88 chunks.

## Questions

1. Will canonical nucleus identity be encoded directly in the output id, or represented by
   hierarchy-aware remaps from compact local ids? Step E must choose one design.
2. What exact artifact does each hierarchy level consume and emit to preserve tags after
   representative changes?

## Verdict

VERDICT: NEEDS_CHANGES
=========================== END artifacts/plan_v0_review.md ===========================

=========================== BEGIN artifacts/plan_v1.md ===========================
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
=========================== END artifacts/plan_v1.md ===========================

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

=========================== BEGIN src/ws/merge_chunks.cpp ===========================
#include "types.hpp"
#include "utils.hpp"
#include "agglomeration.hpp"
#include "mmap_array.hpp"
#include "../seg/SlicedOutput.hpp"
#include <vector>
#include <tuple>
#include <boost/pending/disjoint_sets.hpp>
#include <ctime>
#include <filesystem>
#include <cstdlib>
#include <boost/format.hpp>
#include <execution>

template<typename T>
std::vector<std::pair<T, T> > load_remaps(size_t data_size)
{
    std::vector<std::pair<T, T> > remap_vector;
    if (data_size > 0) {
        MMArray<std::pair<T, T>, 1> remap_data("ongoing.data", std::array<size_t, 1>({data_size}));
        auto data = remap_data.data();
        std::copy(std::begin(data), std::end(data), std::back_inserter(remap_vector));
        std::filesystem::remove("ongoing.data");
        std::stable_sort(std::execution::par, std::begin(remap_vector), std::end(remap_vector), [](auto & a, auto & b) { return std::get<0>(a) < std::get<0>(b); });
    }
    return remap_vector;
}

template<typename T>
std::vector<std::pair< T, size_t> >  load_sizes(size_t data_size)
{
    if (data_size > 0) {
        MMArray<std::pair<T, size_t>, 1> count_data("counts.data",std::array<size_t, 1>({data_size}));
        auto counts = count_data.data();
        std::vector<std::pair<T, size_t> > sizes(counts.begin(), counts.end());
        std::filesystem::remove("counts.data");
        return sizes;
    } else {
        return std::vector<std::pair<T, size_t> >();
    }
}

template<typename ID, typename F>
region_graph<ID,F> load_dend(size_t data_size)
{
    if (data_size > 0) {
        MMArray<std::tuple<F, ID, ID>, 1> dend_data("dend.data",std::array<size_t, 1>({data_size}));
        auto dend_tuple = dend_data.data();
        region_graph<ID, F> rg(dend_tuple.begin(), dend_tuple.end());
        std::filesystem::remove("dend.data");
        return rg;
    } else {
        return region_graph<ID,F>();
    }
}

template<typename ID, typename F>
std::tuple<std::vector<std::pair<ID, ID>>, size_t, size_t>
process_chunk_borders(size_t face_size, std::vector<std::pair<ID, size_t> > & size_pairs, size_t dend_size, auto high_threshold, auto low_threshold, auto size_threshold, auto dust_threshold, const std::string & tag, size_t remap_size, size_t ac_offset)
{
    std::vector<size_t> sizes;
    std::vector<ID> segids;
    sizes.reserve(size_pairs.size());
    segids.reserve(size_pairs.size());
    segids.push_back(0);
    sizes.push_back(0);
    std::stable_sort(std::execution::par, std::begin(size_pairs), std::end(size_pairs), [](auto & a, auto & b) { return a.first < b.first; });
    clock_t begin = clock();
    std::cout << size_pairs.size() << " supervoxels to populate" << std::endl;
    for (auto & kv : size_pairs) {
        if (kv.first == 0 || kv.second == 0) {
            std::cerr << "Impossible segid: " << kv.first << " or size: " << kv.second << std::endl;
            std::abort();
        }
        segids.push_back(kv.first);
        sizes.push_back(kv.second);
    }
    free_container(size_pairs);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "populate maps in " << elapsed_secs << " seconds" << std::endl;

    using traits = watershed_traits<ID>;
    using rank_t = MapContainer<ID,std::size_t>;
    using parent_t = MapContainer<ID,ID>;
    std::vector<ID> rank(sizes.size());
    std::vector<ID> parent(sizes.size());
    boost::disjoint_sets<ID*, ID*> sets(&rank[0], &parent[0]);

    std::vector<F> descent(sizes.size(), high_threshold);

    sets.make_set(0);

    for (size_t i = 0; i != segids.size(); i++) {
        sets.make_set(i);
    }

    std::vector<id_pair<size_t> > same;
    MapContainer<id_pair<ID>, F, HashFunction<id_pair<ID> > > edges;

    begin = clock();
    std::vector<ID> vfi;
    std::vector<ID> vfo;
    std::vector<ID> vbi;
    std::vector<ID> vbo;
    auto conn_data = MMArray<F, 1>("aff_b.data", std::array<size_t, 1>({face_size}));
    auto conn = conn_data.data();
{
    auto fi_data = MMArray<ID, 1>("seg_fi.data", std::array<size_t, 1>({face_size}));
    auto fo_data = MMArray<ID, 1>("seg_fo.data", std::array<size_t, 1>({face_size}));
    auto bi_data = MMArray<ID, 1>("seg_bi.data", std::array<size_t, 1>({face_size}));
    auto bo_data = MMArray<ID, 1>("seg_bo.data", std::array<size_t, 1>({face_size}));

    auto fi = fi_data.data();
    auto fo = fo_data.data();
    auto bi = bi_data.data();
    auto bo = bo_data.data();
    vfi.resize(fi.size(), 0);
    vfo.resize(fo.size(), 0);
    vbi.resize(bi.size(), 0);
    vbo.resize(bo.size(), 0);

    std::transform(std::execution::par, fi.begin(), fi.end(), vfi.begin(), [&segids](ID a){
            auto it = std::lower_bound(segids.begin(), segids.end(), a);
            if (it == segids.end()) {
                std::cerr << "Should not happen, face element does not exist: " << a << std::endl;
                std::abort();
            }
            if (a == *it) {
                return std::distance(segids.begin(), it);
            } else {
                std::cerr << "Should not happen, face element does not exist: " << a << std::endl;
                std::abort();
            }
            });

    std::transform(std::execution::par, fo.begin(), fo.end(), vfo.begin(), [&segids](ID a){
            auto it = std::lower_bound(segids.begin(), segids.end(), a);
            if (it == segids.end()) {
                std::cerr << "Should not happen, face element does not exist: " << a << std::endl;
                std::abort();
            }
            if (a == *it) {
                return std::distance(segids.begin(), it);
            } else {
                std::cerr << "Should not happen, face element does not exist: " << a << std::endl;
                std::abort();
            }
            });

    std::transform(std::execution::par, bi.begin(), bi.end(), vbi.begin(), [&segids](ID a){
            auto it = std::lower_bound(segids.begin(), segids.end(), a);
            if (it == segids.end()) {
                std::cerr << "Should not happen, face element does not exist: " << a << std::endl;
                std::abort();
            }
            if (a == *it) {
                return std::distance(segids.begin(), it);
            } else {
                std::cerr << "Should not happen, face element does not exist: " << a << std::endl;
                std::abort();
            }
            });

    std::transform(std::execution::par, bo.begin(), bo.end(), vbo.begin(), [&segids](ID a){
            auto it = std::lower_bound(segids.begin(), segids.end(), a);
            if (it == segids.end()) {
                std::cerr << "Should not happen, face element does not exist: " << a << std::endl;
                std::abort();
            }
            if (a == *it) {
                return std::distance(segids.begin(), it);
            } else {
                std::cerr << "Should not happen, face element does not exist: " << a << std::endl;
                std::abort();
            }
            });

}
    for (size_t idx = 0; idx != face_size; idx++) {
        if ( vfi[idx] && vbi[idx] ) {
            bool needs_an_edge = false;
            //std::cout << "id: " << fi[idx] << ", id: " << bi[idx] << std::endl;
            id_pair<size_t> xp = std::minmax(vfi[idx], vbi[idx]);
            if ( conn[idx] >= low_threshold ) {
                if ( vfo[idx] ) {
                    if (vfi[idx] != vfo[idx]) {
                        std::cerr << "something is wrong in fo" << std::endl;
                        std::abort();
                    }
                    if ( conn[idx] >= high_threshold ) {
                        if (vbi[idx] != vbo[idx]) {
                            std::cerr << "something is wrong in merge" << std::endl;
                            std::abort();
                        }
                        same.push_back(xp);
                    } else {
                        needs_an_edge = true;
                        if (descent[vfi[idx]] != high_threshold && descent[vfi[idx]] != conn[idx]) {
                            std::cerr << "This should not happen in a" << std::endl;
                            std::cerr << idx << " " << segids[vfi[idx]] << " " << segids[vbi[idx]] << std::endl;
                            std::cerr << descent[vfi[idx]] << " " << conn[idx] << std::endl;
                            std::abort();
                        }
                        descent[vfi[idx]] = conn[idx];
                    }
                } else if ( vbo[idx] ) {
                    if (vbi[idx] != vbo[idx]) {
                        std::cerr << "something is wrong in bo" << std::endl;
                        std::abort();
                    }
                    if ( conn[idx] >= high_threshold ) {
                        if (vfi[idx] != vfo[idx]) {
                            std::cerr << "something is wrong in merge" << std::endl;
                            std::abort();
                        }
                        same.push_back(xp);
                    } else {
                        needs_an_edge = true;
                        if (descent[vbi[idx]] != high_threshold && descent[vbi[idx]] != conn[idx]) {
                            std::cerr << "This should not happen in b" << std::endl;
                            std::cerr << segids[vfi[idx]] << " " << segids[vbi[idx]] << std::endl;
                            std::cerr << descent[vbi[idx]] << " " << conn[idx] << std::endl;
                            std::abort();
                        }
                        descent[vbi[idx]] = conn[idx];
                    }
                } else {
                    if (conn[idx] >= high_threshold) {
                        std::cerr << "something is wrong in edge" << std::endl;
                        std::abort();
                    }
                    needs_an_edge = true;
                }
                if (needs_an_edge) {
                    F & f = edges[xp];
                    if (f < conn[idx]) {
                        f = conn[idx];
                    }
                }
            }
        }
    }
    free_container(vfi);
    free_container(vfo);
    free_container(vbi);
    free_container(vbo);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "merge faces in " << elapsed_secs << " seconds" << std::endl;

    std::cout << edges.size() << " edges and " << same.size() << " mergers" << std::endl;
    begin = clock();
    auto rg = load_dend<seg_t, aff_t>(dend_size);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "load dend in " << elapsed_secs << " seconds" << std::endl;

    std::for_each(std::execution::par, std::begin(rg), std::end(rg), [&segids](auto & a) {
            auto it = std::lower_bound(segids.begin(), segids.end(), std::get<1>(a));
            if (it == segids.end()) {
                std::cerr << "Should not happen, rg element does not exist: " << std::get<1>(a) << std::endl;
                std::abort();
            }
            if (std::get<1>(a) == *it) {
                std::get<1>(a) = std::distance(segids.begin(), it);
            } else {
                std::abort();
            }
            it = std::lower_bound(segids.begin(), segids.end(), std::get<2>(a));
            if (it == segids.end()) {
                std::cerr << "Should not happen, rg element does not exist: " << std::get<2>(a) << std::endl;
                std::abort();
            }
            if (std::get<2>(a) == *it) {
                std::get<2>(a) = std::distance(segids.begin(), it);
            } else {
                std::abort();
            }
        });

    begin = clock();
    for (auto & kv : edges) {
        auto & p = kv.first;
        rg.emplace_back(kv.second, p.first, p.second);
    }
    free_container(edges);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "populate region graph in " << elapsed_secs << " seconds" << std::endl;

    begin = clock();
    std::stable_sort(std::execution::par, std::begin(rg), std::end(rg), [](auto & a, auto & b) { return std::get<0>(a) > std::get<0>(b); });
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "sort region graph in " << elapsed_secs << " seconds" << std::endl;
    begin = clock();
    for (auto & p : same) {
        const ID v1 = sets.find_set( p.first );
        const ID v2 = sets.find_set( p.second );
        if (v1 != v2) {
            sets.link(v1,v2);
            const ID vr = sets.find_set(v1);
            sizes[v1] += sizes[v2]&(~traits::on_border);
            sizes[v1] |= sizes[v2]&traits::on_border;
            sizes[v2]  = 0;
            std::swap( sizes[vr], sizes[v1] );
            descent[vr] = high_threshold;
        }
    }
    free_container(same);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "merge plateau in " << elapsed_secs << " seconds" << std::endl;

    begin = clock();
    size_t n_merger = 0;
    for (auto & t : rg) {
        const F val = std::get<0>(t);
        const ID v1 = sets.find_set( std::get<1>(t) );
        const ID v2 = sets.find_set( std::get<2>(t) );

        if (val < low_threshold) {
            break;
        }

        if (v1 == v2) {
            continue;
        }

        if ( descent[v1] == val || descent[v2] == val ) {
            sets.link(v1,v2);
            const ID vr = sets.find_set(v1);
            sizes[v1] += sizes[v2]&(~traits::on_border);
            sizes[v1] |= sizes[v2]&traits::on_border;
            sizes[v2]  = 0;
            std::swap( sizes[vr], sizes[v1] );
            descent[vr] = std::max( descent[v1], descent[v2] );
            n_merger += 1;
            if ( descent[vr] != val )
            {
                descent[vr] = high_threshold;
            }
        }
    }
    free_container(descent);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "merge descent in " << elapsed_secs << " seconds" << std::endl;
    std::cout << n_merger << " mergers" << std::endl;

    std::cout << "merge" << std::endl;
    begin = clock();
    n_merger = 0;
    region_graph<ID,F> res_rg;
    for (auto & t : rg) {
        const F val = std::get<0>(t);
        const ID v1 = sets.find_set( std::get<1>(t) );
        const ID v2 = sets.find_set( std::get<2>(t) );

        if (val < low_threshold) {
            break;
        }

        if ( v1 != v2 && v1 && v2 ) {
            if (try_merge(sizes, sets, v1, v2, size_threshold)) {
                n_merger += 1;
            }
            else {
                res_rg.push_back(t);
            }
        }
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "rg size: " << rg.size() << std::endl;
    std::cout << "res rg size: " << res_rg.size() << std::endl;
    free_container(rg);
    std::cout << "merge region graph in " << elapsed_secs << " seconds" << std::endl;
    std::cout << n_merger << " mergers" << std::endl;

    std::vector<size_t> remaps(sizes.size(), 0);

    ID next_id = 0;

    begin = clock();

    for (size_t v = 0; v != sizes.size(); v++) {
        size_t size = sizes[v];
        const ID s = sets.find_set(v);
        if (sizes[s] >= dust_threshold) {
            remaps[v] = s;
        }

        if ( (size & (~traits::on_border)) && size >= dust_threshold  ) {
            if (s != v) {
                std::cout << "s("<<s<<") != v("<<v<<")" << std::endl;
            }
            ++next_id;
        }
    }

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "generate remap in " << elapsed_secs << " seconds" << std::endl;

    MapContainer<size_t, std::set<size_t> > in_rg;

    region_graph<ID,F> unique_rg;
    region_graph<ID,F> new_rg;

    begin = clock();

    std::for_each(std::execution::par, std::begin(res_rg), std::end(res_rg), [&remaps](auto & a) {
            auto mm = std::minmax(remaps[std::get<1>(a)], remaps[std::get<2>(a)]);
            std::get<1>(a) = mm.first;
            std::get<2>(a) = mm.second;
            });

    std::stable_sort(std::execution::par, std::begin(res_rg), std::end(res_rg), [](auto & a, auto & b) {
            return (std::get<1>(a) < std::get<1>(b)) \
            || ((std::get<1>(a) == std::get<1>(b)) && (std::get<2>(a) < std::get<2>(b))) \
            || ((std::get<1>(a) == std::get<1>(b)) && (std::get<2>(a) == std::get<2>(b)) && (std::get<0>(a) > std::get<0>(b))); });

    std::unique_copy(std::execution::par, std::begin(res_rg), std::end(res_rg), std::back_inserter(unique_rg), [](auto & a, auto & b) {return (std::get<1>(a) == std::get<1>(b) && std::get<2>(a) == std::get<2>(b));});

    std::stable_sort(std::execution::par, std::begin(unique_rg), std::end(unique_rg), [](auto & a, auto & b) {return std::get<0>(a) > std::get<0>(b);});
    //rank_t rank_mst_map;
    //parent_t parent_mst_map;

    //boost::associative_property_map<rank_t> rank_mst_pmap(rank_mst_map);
    //boost::associative_property_map<parent_t> parent_mst_pmap(parent_mst_map);

    //boost::disjoint_sets<boost::associative_property_map<rank_t>, boost::associative_property_map<parent_t> > mst(rank_mst_pmap, parent_mst_pmap);

    for ( auto& it: unique_rg )
    {
        ID s1 = std::get<1>(it);
        ID s2 = std::get<2>(it);
        ID a1 = sets.find_set(s1);
        ID a2 = sets.find_set(s2);

        if ( a1 != a2 && a1 && a2 )
        {
            sets.link(a1, a2);
            if (((sizes[s1] & traits::on_border) && (sizes[s2] & traits::on_border)))
            {
                new_rg.emplace_back(std::get<0>(it), segids[s1], segids[s2]);
            }
        }
    }

    auto d = write_vector(str(boost::format("dend_%1%.data") % tag), new_rg);
    free_container(res_rg);
    free_container(in_rg);
    free_container(new_rg);
    free_container(rank);
    free_container(parent);

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "generate MST in " << elapsed_secs << " seconds" << std::endl;

    begin = clock();
    std::vector<std::pair<ID, size_t> > counts;
    for (size_t v = 0; v != sizes.size(); v++) {
        if (sizes[v] & traits::on_border) {
            counts.emplace_back(segids[v],sizes[v]&(~traits::on_border));
        }
    }

    auto c = write_vector(str(boost::format("counts_%1%.data") % tag), counts);
    free_container(counts);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "write supervoxel sizes in " << elapsed_secs << " seconds" << std::endl;

    size_t current_ac = std::numeric_limits<std::size_t>::max();
    std::ofstream of_ongoing;
    of_ongoing.open(str(boost::format("ongoing_%1%.data") % tag));
    if (!of_ongoing.is_open()) {
        std::cerr << "Failed to open ongoing remap file for " << tag << std::endl;
        std::abort();
    }

    auto remap_vector = load_remaps<ID>(remap_size);
    std::for_each(std::execution::par, std::begin(remap_vector), std::end(remap_vector), [&segids](auto & a) {
            auto it = std::lower_bound(segids.begin(), segids.end(), a.second);
            if (a.second == *it) {
                a.second = std::distance(segids.begin(), it);
            } else {
                std::abort();
            }
            });

    MapContainer<ID, ID> reps;

    begin = clock();

    SlicedOutput<std::pair<ID, ID>, ID> remap_output(str(boost::format("done_pre_%1%.data") % tag));

    for (size_t i = 0; i != remap_vector.size(); i++) {
        auto & s = remap_vector[i].first;
        if (current_ac != (s - (s % ac_offset))) {
            remap_output.flushChunk(current_ac);
            reps.clear();
            current_ac = s - (s % ac_offset);
        }
        const auto seg = remaps[remap_vector[i].second];
        const auto size = sizes[seg];
        if (size & traits::on_border) {
            if (reps.count(seg) == 0) {
                of_ongoing.write(reinterpret_cast<const char *>(&(s)), sizeof(ID));
                of_ongoing.write(reinterpret_cast<const char *>(&(segids[seg])), sizeof(ID));
                reps[seg] = s;
            } else {
                remap_output.addPayload(std::make_pair(s, reps.at(seg)));
            }
        } else {
            remap_output.addPayload(std::make_pair(s, segids[seg]));
        }
        if (of_ongoing.bad()) {
            std::cerr << "Error occurred when writing ongoing remap file for " << tag << " " << current_ac << std::endl;
            std::abort();
        }
    }

    remap_output.flushChunk(current_ac);
    remap_output.flushIndex();

    free_container(remap_vector);

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "update remaps in " << elapsed_secs << " seconds" << std::endl;

    current_ac = std::numeric_limits<std::size_t>::max();

    SlicedOutput<std::pair<ID, ID>, ID> remap2_output(str(boost::format("done_post_%1%.data") % tag));

    begin = clock();
    for (size_t i = 1; i != segids.size(); i++) {
        auto s = segids[i];
        if (current_ac != (s - (s % ac_offset))) {
            remap2_output.flushChunk(current_ac);
            reps.clear();
            current_ac = s - (s % ac_offset);
        }
        if (s == 0) {
            std::cerr << "svid = 0, should not happen" << std::endl;
            std::abort();
        }
        const auto seg = remaps[i];
        const auto size = sizes[seg];
        if (size & traits::on_border) {
            if (reps.count(seg) == 0) {
                of_ongoing.write(reinterpret_cast<const char *>(&(s)), sizeof(ID));
                of_ongoing.write(reinterpret_cast<const char *>(&(segids[seg])), sizeof(ID));
                reps[seg] = s;
            } else {
                remap2_output.addPayload(std::make_pair(s, reps.at(seg)));
            }
        } else {
            remap2_output.addPayload(std::make_pair(s, segids[seg]));
        }
        if (of_ongoing.bad()) {
            std::cerr << "Error occurred when writing ongoing remap file for " << tag << " " << current_ac << std::endl;
            std::abort();
        }
    }

    remap2_output.flushChunk(current_ac);
    remap2_output.flushIndex();

    of_ongoing.close();

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "generate new remap in " << elapsed_secs << " seconds" << std::endl;

    std::cout << "number of supervoxels:" << remaps.size() << "," << next_id << std::endl;
    std::vector<std::pair<ID, ID> > remap_pairs;
    for (size_t i = 0; i != segids.size(); i++) {
        remap_pairs.push_back(std::make_pair(segids[i], segids[remaps[i]]));
    }
    return std::make_tuple(std::move(remap_pairs), c, d);
}

template<typename T>
void mark_border_supervoxels(std::vector<std::pair<T, size_t> > & sizes, const std::array<bool,6> & boundary_flags, const std::array<size_t, 6> & face_dims, const std::string & tag)
{
    using traits = watershed_traits<T>;
    for (size_t i = 0; i != 6; i++) {
        if (!boundary_flags[i]) {
            auto fn = str(boost::format("seg_i_%1%_%2%.data") % i % tag);
            std::cout << "loading: " << fn << std::endl;
            MMArray<T, 1> face_array(fn,std::array<size_t, 1>({face_dims[i]}));
            auto face_data = face_array.data();
            std::vector<T> boundary_segs(face_data.begin(), face_data.end());
            if (!face_array.close()) {
                std::cerr << "Failed to close the file" << std::endl;
                std::abort();
            }
            std::stable_sort(std::execution::par, boundary_segs.begin(), boundary_segs.end());
            std::for_each(std::execution::par, sizes.begin(), sizes.end(), [&boundary_segs](auto & p) {
                    if (p.first != 0) {
                        auto newid = std::lower_bound(boundary_segs.begin(), boundary_segs.end(), p.first);
                        if (*newid == p.first) {
                            p.second |= traits::on_border;
                        }
                    }
                }
            );
            //for (size_t j = 0; j != face_dims[i]; j++) {
            //    if (sizes.count(face_data[j]) != 0 && face_data[j]!=0) {
            //        sizes[face_data[j]] |= traits::on_border;
            //    } else {
            //        if (face_data[j] != 0) {
            //            std::cout << "supervoxels does not exist" << std::endl;
            //        }
            //    }
            //}
        }
    }
    //std::cout << "check supervoxel: 240854 " << sizes[240854] << std::endl;
    //std::cout << "check supervoxel: 240855 " << sizes[240855] << std::endl;
}

template<typename T>
void update_border_supervoxels(const std::vector<std::pair<T, T> > & remaps, const std::array<bool,6> & boundary_flags, const std::array<size_t, 6> & face_dims, const std::string & tag)
{
    for (size_t i = 0; i != 6; i++) {
        if (!boundary_flags[i]) {
            auto fi = str(boost::format("seg_i_%1%_%2%.data") % i % tag);
            std::cout << "update: " << fi << ",size:" << face_dims[i] << std::endl;
            MMArray<T, 1> face_i_array(fi,std::array<size_t, 1>({face_dims[i]}));
            auto face_i_data = face_i_array.data();
            std::vector<T> face_i_vector(face_i_data.begin(), face_i_data.end());
            if (!face_i_array.close()) {
                std::cerr << "Failed to close the file" << std::endl;
                std::abort();
            }
            std::for_each(std::execution::par, face_i_vector.begin(), face_i_vector.end(), [&remaps](size_t & a) {
                    auto newid = std::lower_bound(remaps.begin(), remaps.end(), a, [](auto & x, auto & y){
                            return x.first < y;
                            });
                    if (newid != remaps.end() && a == newid->first) {
                        a = newid->second;
                    }
            });
            write_vector(fi, face_i_vector);

            auto fo = str(boost::format("seg_o_%1%_%2%.data") % i % tag);
            std::cout << "update: " << fo << ",size:" << face_dims[i] << std::endl;
            MMArray<T, 1> face_o_array(fo,std::array<size_t, 1>({face_dims[i]}));
            auto face_o_data = face_o_array.data();
            std::vector<T> face_o_vector(face_o_data.begin(), face_o_data.end());
            if (!face_o_array.close()) {
                std::cerr << "Failed to close the file" << std::endl;
                std::abort();
            }
            std::for_each(std::execution::par, face_o_vector.begin(), face_o_vector.end(), [&remaps](size_t & a) {
                    auto newid = std::lower_bound(remaps.begin(), remaps.end(), a, [](auto & x, auto & y){
                            return x.first < y;
                            });
                    if (newid != remaps.end() && a == newid->first) {
                        a = newid->second;
                    }
            });
            write_vector(fo, face_o_vector);
        }
    }
}

int main(int argc, char* argv[])
{
    size_t xdim,ydim,zdim;
    int flag;
    size_t face_size, counts, dend_size, remap_size, ac_offset;
    std::ifstream param_file(argv[1]);
    std::string ht(argv[2]);
    std::string lt(argv[3]);
    std::string st(argv[4]);
    std::string dt(argv[5]);
    std::cout << "thresholds: "<< ht << " " << lt << " " << st << " " << dt << std::endl;
    const char * tag = argv[6];
    auto high_threshold = read_float<aff_t>(ht);
    auto low_threshold = read_float<aff_t>(lt);
    auto size_threshold = read_int(st);
    auto dust_threshold = read_int(dt);
    param_file >> xdim >> ydim >> zdim;
    std::cout << xdim << " " << ydim << " " << zdim << std::endl;
    std::array<bool,6> flags({true,true,true,true,true,true});
    for (size_t i = 0; i != 6; i++) {
        param_file >> flag;
        flags[i] = (flag > 0);
        if (flags[i]) {
            std::cout << "real boundary: " << i << std::endl;
        }
    }
    param_file >> face_size >> counts >> dend_size >> remap_size >> ac_offset;

#ifdef USE_MIMALLOC
    size_t huge_pages = (face_size + dend_size + counts) * (8 + 8 + 8) * 4 / 1024 / 1024 / 1024 + 1;
    auto mi_ret = mi_reserve_huge_os_pages_interleave(huge_pages, 0, 0);
    if (mi_ret == ENOMEM) {
       std::cout << "failed to reserve 1GB huge pages" << std::endl;
    }
#endif

    if (face_size == 0) {
        std::cout << "Nothing to merge, exit!" << std::endl;
        SlicedOutput<std::pair<seg_t, seg_t>, seg_t> remap_output(str(boost::format("done_pre_%1%.data") % tag));
        SlicedOutput<std::pair<seg_t, seg_t>, seg_t> remap2_output(str(boost::format("done_post_%1%.data") % tag));
        remap_output.flushIndex();
        remap2_output.flushIndex();
        return 0;
    }

    std::cout << "supervoxel id offset:" << face_size << " " << counts << " " << dend_size << std::endl;
    clock_t begin = clock();
    auto sizes = load_sizes<seg_t>(counts);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "load size in " << elapsed_secs << " seconds" << std::endl;
    begin = clock();
    mark_border_supervoxels(sizes, flags, std::array<size_t, 6>({ydim*zdim, xdim*zdim, xdim*ydim, ydim*zdim, xdim*zdim, xdim*ydim}), tag);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "mark boundary supervoxels in " << elapsed_secs << " seconds" << std::endl;

    std::vector<std::pair<seg_t, seg_t> > remaps;
    size_t c = 0;
    size_t d = 0;

    std::tie(remaps, c, d) = process_chunk_borders<seg_t, aff_t>(face_size, sizes, dend_size, high_threshold, low_threshold, size_threshold, dust_threshold, tag, remap_size, ac_offset);
    update_border_supervoxels(remaps, flags, std::array<size_t, 6>({ydim*zdim, xdim*zdim, xdim*ydim, ydim*zdim, xdim*zdim, xdim*ydim}), tag);
    //auto m = write_remap(remaps, tag);
    std::vector<size_t> meta({xdim,ydim,zdim,c,d,0});
    write_vector(str(boost::format("meta_%1%.data") % tag), meta);
    std::cout << "num of sv:" << c << std::endl;
    std::cout << "size of rg:" << d << std::endl;
    //std::cout << "num of remaps:" << m << std::endl;
}
=========================== END ===========================

Now produce your review: a short summary, then findings each tagged [minor] or [major], then
any questions, then the final READY: line.
