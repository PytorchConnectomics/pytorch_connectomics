You are the CODER for a CCC run. Implement plan_v2 (included below in full). It is the terminal
plan and is authoritative. If something in it is actually wrong while implementing, implement the
correct thing and say so explicitly in "Risks and Unknowns" -- do not silently deviate.

You reviewed plan_v0 and plan_v1 and returned READY: no on both, with 12 and 14 findings. plan_v2
accepts all of them; your finding that `merge_segments` only visits region-graph edges above
`tholds.second` -- so "same tag -> union" can never fire for components with no qualifying edge --
is why the must-link is now an explicit tag-group closure at a named insertion point.

READ THIS FIRST: three previous attempts at this bug each produced a result IDENTICAL to the
control. Your plan_v1 review finding F13 spelled out how this one could become the fourth: split
locally, write correct tags, fail to materialise tag-equivalence remaps, lose tags in a
plateau/descent/early-return path, and let ordinary remapping reconstruct the control. The two
structural blocks are (i) the closure runs inside the same DSU `sets` upstream of `remaps`, and
(ii) at each hierarchy level the closure runs before `remap_pairs` is emitted. If you find you
cannot honour either, stop and say so rather than shipping something that runs.

## Hard constraints

1. WORK ONLY IN:
       /projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_nucleus/work2/abiss
   (reachable as work2/abiss from this run folder). Detached at 312bf54, already built.
   The working tree ALREADY CARRIES the sibling run's global-nucleus-table change -- that is this
   run's baseline, recorded in state/run_start.diff. DO NOT revert it.
   NEVER touch /projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss (live; SLURM runs its
   binaries) or .agent/features/abiss_nucleus/work/abiss (pre-rebase 8afcd87, broken URI handling).
2. No git commits. No new dependencies.
3. Behaviour with no NUC_PATH must be bit-identical to baseline. V1 is the gate; failing it is a
   hard failure. The pipeline reproduces a Seuron provenance record.
4. Do not claim a command passed unless you ran it and saw the output. A step not run is reported
   as NOT RUN. That matters more than appearing complete.
5. Environment: `set +u; source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc; set -u`.
   Build only into work2/abiss/build. HDF5 on /projects needs HDF5_USE_FILE_LOCKING=FALSE.
6. The seeded split imposes a cut with NO image support (measured bottleneck ~0.999 between all
   three nucleus pairs, against a 0.3 agglomeration threshold). Say so in a comment where the split
   happens; it is defensible only because the mask is confirmed correct and conservative.

## What to produce

Implement A-F, then run V1-V6 as far as they go. V1, V2, V3 and V5 are the ones that matter most.
V6 needs SLURM and may be left to the coordinator -- say so rather than skipping it silently.

Then write, at exactly this path:

    /projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_ws_snap/artifacts/code_v0.md

with EXACTLY these level-2 sections, in order, each exactly once:

    # Code v0
    ## Overview
    ## What Changed
    ## Implementation Details
    ## Files Changed
    ## Git Baseline
    ## Verification
    ## Review Focus
    ## Risks and Unknowns
    ## Changes Since Previous Code Version

- "## Files Changed" must contain a table with header exactly:
    | File | Purpose |
    |---|---|
- "## Git Baseline" must contain:
    run_start_ref: 312bf54183faa94be1aff8ae246fcc211808f0bb
    current_head: <git -C work2/abiss rev-parse HEAD>
- "## Changes Since Previous Code Version" must contain exactly: Initial implementation.
- "## Verification" must give, per check, the command run and its actual result, or NOT RUN.

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

=========================== BEGIN artifacts/plan_v2.md ===========================
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
=========================== END artifacts/plan_v2.md ===========================

=========================== BEGIN artifacts/plan_v1_review.md ===========================
# Plan v1 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, `xhigh`), read-only sandbox. Raw transcript:
`state/plan_v1_review.review.raw.md`. Fourteen findings, all major, none softened here.

The architecture (compact ids + tag transport) is judged viable, but the plan does not specify an
executable must-link. The decisive finding: `merge_segments` only visits region-graph edges above
`tholds.second`, so "same tag -> union" never fires for two components with no qualifying edge --
the common case for a nucleus spread over many basins -- and the MST phase cannot do it either,
since it runs after remapping and compaction and only suppresses dendrogram cycles. Several
hierarchy union sites also remain unguarded, and the plan's tags could survive perfectly while the
segmentation ids stay fragmented.

## Findings

- [major] The unconditional must-link is not actually defined. `merge_segments` only visits region-graph edges above `tholds.second`; the MST loop also only sees existing eligible edges. Two components with the same tag but no qualifying edge will never be joined. This includes same-nucleus pieces separated by a low-affinity edge, disconnected mask pieces, and pieces that do not meet until a later hierarchy level. The implementation needs a tag-group closure that unions every nonzero-tag equivalence class independently of region-graph eligibility and emits remaps for every absorbed ID.

- [major] The MST phase cannot implement the snap. Its `mst.link` only suppresses dendrogram cycles after segmentation remapping and `counts` compaction have already happened; it does not merge segmentation components. A same-tag decision made there is too late. Must-links must be materialized before dusting, compaction, boundary writing, and remap generation. The MST should then operate on the resulting representatives, omit self-edges, and reject cross-tag edges.

- [major] Step B does not explicitly tag basins that overlap exactly one nucleus. It assigns tags while splitting a basin with two or more seeds, but the ordinary one-nucleus/many-basin case is the main snap case. Every basin with exactly one nonzero overlap must receive that tag before same-tag closure. V2 also needs a fixture where one nucleus overlaps multiple original basins; its current two-nucleus split fixture does not prove the snap.

- [major] Tag mutation inside `merge_segments` is unspecified. For every union, the tag must follow the DSU representative even when `sets.link` chooses the opposite root; tagged+untagged must retain the tag, different tags must fail closed, and counts and tags must be swapped together after representative selection. Tags must also be compacted using the exact `remaps` used for `seg`. In multi-threshold mode, `basin_tag` needs a fresh copy beside each `seg_copy`, `rg_copy`, and `counts_copy`; merely writing one file per `out_tag` does not ensure that.

- [major] `merge_chunks.cpp` contains more union sites than the plan identifies: plateau unions in `same`, descent unions, `try_merge`, and the final MST `sets.link`. Saying “tag-aware stitching” is insufficient. Different tags must be checked before all four sites, and tag/count ownership must be updated after the first three. The `face_size == 0` early return must also pass through or finalize tags rather than silently dropping the side data.

- [major] Hierarchy transport still lacks the required consume/emit contract. The plan does not say how tag records are matched through `ongoing`, `done_pre`, `done_post`, updated boundary IDs, and returned `remap_pairs`; which representative becomes the emitted key; or how conflicting records fail. Reusing `nuc_wire_t` gives a serialization shape, not these semantics. Most importantly, it does not say how same-tag child IDs are converted into actual hierarchy remaps. Tags can therefore survive perfectly while the segmentation IDs remain fragmented.

- [major] The trace stops before the downstream contract is established. The plan must show that the final hierarchy remaps are applied to `ws3`, those exact IDs populate `chunkmap.data`, and the existing agglomeration nucleus table sees the expected one-nucleus mapping. A final small-fixture assertion should trace known atomic IDs through boundary files, parent remaps, final `ws3`, `chunkmap`, and the agglomeration input. Without that, the tags can remain an unused side channel.

- [major] Unconditional must-link is safe only with a dedicated union primitive. DSU prevents cycles, and the correct count operation is the sum of real sizes plus the OR of `on_border`, with the losing count cleared and the tag moved to the selected root. The plan does not specify that primitive or require all must-links before remapping. Applying “must-link” separately in the MST or after compaction could leave stale counts, duplicate representatives, or dendrogram edges referring to pre-union nodes.

- [major] Dust protection is not covered at every hierarchy level. V2 exercises the atomic `merge_segments` dust loop, but `merge_chunks.cpp` independently maps components below `dust_threshold` to zero. Tagged components must be exempt there and at every later reduction, with a hierarchy-level tiny-tagged-component test.

- [major] V3’s build flag is not yet a concrete seam. It needs a named build option with a defined default, two isolated test binaries/configurations, and a guarantee that the disabled build bypasses only the tested tag guard while retaining the same split and tag inputs. V3 presently tests the atomic cross-tag veto, but not same-tag unconditional merging or the plateau/descent/stitch guards in `merge_chunks`. A second fixture must prove that two same-tag components merge even when both exceed `size_threshold` and their edge is below the ordinary merge threshold.

- [major] V2 mentions `on_border` but does not yet define an independent oracle. The reference should scan the post-split labels, count voxels, and derive the bit from contact with the applicable non-real chunk faces. The fixture must split one original border basin into at least one border-touching child and one interior child, then compare the complete raw count words. Inheriting the parent bit for every child would otherwise satisfy a weak “preservation” interpretation while remaining wrong.

- [major] V5.6 is the right cannot-link invariant only if “ws volume” means the final remapped `ws3` used to construct `chunkmap`, not an atomic `seg_<tag>.data` written before hierarchy remaps. That final artifact appears to exist from the stated pipeline, so the check is feasible, but the plan must name it and its crop/mask alignment. It must also assert the complementary must-link invariant: each of the eight nucleus masks overlaps exactly one nonzero final watershed ID. Zero multi-nucleus supervoxels alone permits a split-only implementation whose missing snap is repaired later—or not at all—by agglomeration.

- [major] The fourth no-op remains possible: split locally, write correct atomic tags, fail to materialize tag-equivalence remaps, lose or ignore tags in a plateau/descent/early-return path, and let ordinary ws remapping and agglomeration reconstruct the control. V5.2 would eventually report failure, and final-`ws3` V5.6 would also catch it, but E1/E2 do not themselves block it. This is detection after the crop run, not an end-to-end mechanism proving the path impossible.

- [major] The open semantic question must be closed in the plan. Unconditional same-tag equivalence is correct under the task’s explicit contract that the nucleus mask is trusted ground truth. An affinity floor contradicts the requested imposed constraint and can recreate the known fragmentation. Region-graph adjacency may be used to discover ordinary merges, but it must not gate the nucleus must-link; otherwise compact IDs plus tags do not reproduce the withdrawn nucleus-derived-ID guarantee.

## Questions

1. What exact operation groups all IDs with tag `N`, selects their representative, and writes the corresponding atomic and hierarchy remaps when no qualifying region-graph edge exists?
2. Which named final `ws3` artifact feeds `chunkmap.data`, and where will the test verify its nucleus-to-watershed mapping and the resulting agglomeration nucleus records?
3. What is the named V3 build option, and which precise union/edge checks does its disabled mode bypass?

## Verdict

VERDICT: NEEDS_CHANGES
=========================== END artifacts/plan_v1_review.md ===========================

You have full read access to work2/abiss. Start at src/ws/{atomic_chunk.cpp,agglomeration.hpp,
merge_chunks.cpp}, scripts/{cut_chunk_ws,cut_chunk_agg,nucleus_utils}.py, CMakeLists.txt,
and work/test/. The sibling run's evidence is at
../abiss_nucleus/state/evidence_conflict.md if you want the measurements.
