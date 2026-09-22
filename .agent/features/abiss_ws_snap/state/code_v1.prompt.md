You are the CODER for a CCC run. This is code_v1: fix the findings in review_v0 (included below).

Your code_v0 is complete against plan_v2 and its fixtures pass. The coordinator then ran V6, which
you correctly reported NOT RUN, and it ABORTED on real data. The cause is a defect in plan_v2
section D, written by the planner, not a slip by you.

## The bug, in one paragraph

`merge_chunks.cpp` has four union sites and plan_v2 treated them as interchangeable. They are not.
`try_merge` and the MST are POLICY -- they decide whether two well-formed components should join.
The plateau (`same`) loop and the descent loop are STRUCTURE -- they build the watershed and
maintain `descent[]` as they go. Your cross-tag veto at the structural sites skips a union that the
surrounding code assumes happened, leaving `descent[v1]`/`descent[v2]` unreconciled, and the next
boundary pair touching that supervoxel trips the pre-existing invariant at merge_chunks.cpp:262-267:

    This should not happen in b
    72057869184326525 72128237928471919
    0.999918 0.999898
    ws2 ... Aborted (core dumped)      <- composite chunk 1_0_2_2

Those affinities are exactly the band measured for this data (pairwise bottleneck ~0.999 between
the nuclei), so this is the intended case reaching the wrong mechanism, not a rare edge.

## What to change

1. **Restore the unconditional union at BOTH structural sites** -- the `same`/plateau loop and the
   `descent[v1] == val || descent[v2] == val` loop. Keep the cross-tag veto at `try_merge` and the
   MST only.

2. **Represent the resulting conflict; never silently drop it.** `ws_nuc_join_tags` currently
   returns 0 when two different nonzero tags meet, which erases the identity -- that is precisely
   the failure class this whole run exists to fix, and after change 1 it becomes reachable in the
   guard-ON build. Add an explicit conflict state to `ws_nuc_tag_t` (a reserved sentinel such as
   0xFFFFFFFF, asserted not to collide with any real nucleus id) with:
   - join(a,b) for differing nonzero tags -> CONFLICT, and CONFLICT absorbs anything;
   - CONFLICT never decays to 0, at any level, including through the hierarchy transport and the
     emitted tag records;
   - a `ws nuc:` counter for how many components become CONFLICT this way, printed per chunk so
     its real-data frequency is visible in the log.
   Choose the policy-site behaviour for CONFLICT deliberately and justify it in the artifact; note
   that at the agglomeration level `nuc_can_merge(CONFLICT, NONE) == true` is the permissive clause
   that let a 291M-voxel blob grow, so think about whether the ws level should be stricter.

3. **Add the fixture that would have caught this**: two tagged basins meeting across a chunk face
   at `>= high_threshold` (0.99999) AND at a descent-equal value, asserting `ws2` completes and the
   result matches the policy chosen in 2. Your existing hierarchy fixture reaches `try_merge` and
   the MST but never produces a cross-tag pair at plateau or descent strength, which is why this
   reached a cluster run.

4. **Do not delete `nuc_tag.data` on read** (`load_nucleus_tags` currently calls
   `std::filesystem::remove`). It made the failed run non-rerunnable in place and hid the input
   from post-mortem. Let the caller clean up.

5. **Print the tags in the abort messages** in `merge_chunks.cpp`. The code has them now; the
   existing message prints supervoxel ids and affinities only, which cost diagnosis time.

Also answer, in the artifact: does the structural-versus-policy distinction apply inside the atomic
`merge_segments` too? The tag-group closure sits between the `try_merge` loop and `remaps`, which
is policy-side, so it should not -- but confirm no plateau-equivalent bookkeeping exists there.

## Hard constraints (unchanged)

Work only in .agent/features/abiss_nucleus/work2/abiss (detached 312bf54, baseline includes the
sibling run's global-table change -- do not revert). Never touch lib/abiss or
.agent/features/abiss_nucleus/work/abiss. No commits, no new dependencies. No-NUC_PATH behaviour
must stay bit-identical; V1 is the gate. Do not claim a command passed unless you ran it and saw
the output; NOT RUN is the correct report for anything you did not run. V6 needs SLURM and stays
with the coordinator.

## What to produce

Re-run V1-V5 plus the new fixture, then write, at exactly this path:

    /projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_ws_snap/artifacts/code_v1.md

with EXACTLY these level-2 sections, in order, each exactly once:

    # Code v1
    ## Overview
    ## What Changed
    ## Implementation Details
    ## Files Changed
    ## Git Baseline
    ## Verification
    ## Review Focus
    ## Risks and Unknowns
    ## Changes Since Previous Code Version

"## Files Changed" needs a table headed exactly `| File | Purpose |` then `|---|---|`.
"## Git Baseline" needs `run_start_ref: 312bf54183faa94be1aff8ae246fcc211808f0bb` and
`current_head: <git -C work2/abiss rev-parse HEAD>`.
"## Changes Since Previous Code Version" must state what changed versus code_v0 and address each
review_v0 finding explicitly.
"## Verification" must give, per check, the command run and its actual result, or NOT RUN.

=========================== BEGIN artifacts/review_v0.md ===========================
# Review v0

## Summary

The implementation is complete against plan v2 and its fixture tests are real: V1 holds
bit-invariance at `identical=38 differing=0 missing=0`, V3 demonstrates the cannot-link is
load-bearing with an actual `ABISS_WS_NUC_GUARD=OFF` build that merges the plateau bridge, V2
includes an independent `on_border` oracle over full raw count words, and V5 traces ids end to end
into the agglomeration nucleus records. The coder correctly reported V6 NOT RUN and enumerated
every number it was therefore not claiming.

**V6, run by me, aborts.** `ws2` core-dumps at composite chunk `1_0_2_2`, so the crop A/B produced
no result at all:

```
This should not happen in b
72057869184326525 72128237928471919
0.999918 0.999898
scripts/init.sh: line 5: 471130 Aborted (core dumped)
```

That message is a pre-existing ABISS invariant check in `merge_chunks.cpp:262-267`: when a boundary
pair does not merge, `descent[vbi[idx]]` must be either `high_threshold` or exactly `conn[idx]`.
The cause is the cross-tag veto having been placed at the **plateau and descent union sites**, and
this is a design error in plan v2 (mine), not a coding slip.

Those two sites are not discretionary merges. They construct the watershed: the plateau pass
resolves equal-affinity flats, and the descent pass performs steepest-descent linking, and both
maintain `descent[]` as they go. Vetoing a union there leaves `descent[v1]` and `descent[v2]`
unmerged while the surrounding code continues to assume they were reconciled, so the next boundary
pair that touches the same supervoxel trips the invariant. The printed affinities, 0.999918 and
0.999898, sit exactly in the band my own bottleneck measurement identified -- two nuclei connected
at ~0.999 -- so this is the intended case reaching the wrong mechanism, not an unlucky edge case.

Plan v2 section D said "all four check tags before uniting; the first three update tag and count
ownership after", treating the four union sites in `merge_chunks.cpp` as interchangeable. They are
not. `try_merge` and the MST are policy; `same`/plateau and descent are structure.

## Diff Baseline

run_start_ref: 312bf54183faa94be1aff8ae246fcc211808f0bb

## Findings

* **[major] The cross-tag veto must not be applied at the plateau and descent union sites.**
  `merge_chunks.cpp`, the `same` loop and the `descent[v1] == val || descent[v2] == val` loop.
  Skipping the union breaks the `descent[]` invariant asserted at lines 242-246 and 262-267 and
  aborts `ws2` on real data. Restore the unconditional union at both sites. The veto belongs at
  `try_merge` and the MST only, which are the sites that decide whether two *already well-formed*
  components should join.
* **[major] That leaves the cross-tag plateau case unhandled, and it must not be left silent.**
  Two differently-tagged basins genuinely connected by a plateau or a steepest-descent link will
  now merge. Options, in my order of preference: (a) let the union happen and mark the result
  CONFLICT in the tag, so the existing agglomeration-level nucleus veto sees it and the run reports
  it via a counter -- the tag lattice already has this state at the agglomeration level; (b) split
  the plateau at the atomic level so composite chunks never see a cross-tag plateau, which is
  strictly better but needs the atomic split to consider faces; (c) abort with a clear message
  naming the two nuclei. Do **not** silently drop the tag: that reproduces the class of failure
  this whole run exists to fix. Whichever is chosen, add a counter so its frequency on real data is
  visible in the log.
* **[major] The fixtures passed while the real pipeline aborted, so V4/V5 do not cover the plateau
  path.** The hierarchy fixture reaches `try_merge` and the MST but evidently never produces a
  cross-tag pair at the plateau or descent stage. Add a fixture whose two tagged basins meet across
  a chunk face at `>= high_threshold` (0.99999) and at a descent-equal value, and assert `ws2`
  completes and the result matches whichever policy is chosen above.
* **[minor] `load_nucleus_tags` deletes its input** (`std::filesystem::remove("nuc_tag.data")`).
  That makes a failed run non-rerunnable in place and hides the input from post-mortem, which cost
  time here. Leave it and let the caller clean up.
* **[minor] The abort message does not name the nuclei.** It prints supervoxel ids and affinities.
  Since this code now has tags, printing them would have identified the cause immediately.

## Tests to Add

* The plateau/descent cross-tag fixture described above, which is the gap that let this reach a
  cluster run.
* A composite-level assertion that `descent[]` is reconciled for every pair the tag logic touches,
  so a future veto in the wrong place fails in a unit test rather than at chunk `1_0_2_2`.
* Re-run V6 after the fix; none of its numbers exist yet.

## Questions

1. Which policy for the cross-tag plateau -- (a) merge and mark CONFLICT, (b) split earlier at the
   atomic level, or (c) hard abort? I lean (a) for this round because it composes with the
   nucleus veto that already exists downstream and cannot deadlock the pipeline, with (b) as the
   follow-up if the counter shows it is common.
2. Does the same structural-versus-policy distinction apply inside the atomic `merge_segments`?
   The tag-group closure sits between the `try_merge` loop and `remaps`, which is policy-side, so I
   believe not -- but it is worth confirming that no plateau-equivalent bookkeeping exists there.

## Verdict

VERDICT: NEEDS_CHANGES
=========================== END artifacts/review_v0.md ===========================

=========================== BEGIN artifacts/code_v0.md ===========================
# Code v0
## Overview

Implemented nucleus-mask snapping in the atomic watershed and transported the resulting nucleus
equivalence through every watershed hierarchy level. Multi-nucleus basins are split by a
deterministic affinity-priority seeded watershed; every tagged basin is then materialized as a
must-link in the same DSU used for atomic compaction and, separately, before hierarchy remap
emission. The no-`NUC_PATH` invariance gate passed.

No commit was created. The sibling run's global-nucleus-table working-tree changes were preserved
as the run baseline.

## What Changed

- `cut_chunk_ws.py` now creates a shape-checked `uint32` `nuc.raw` using the existing
  `nucleus_utils` coordinate transform, only when `NUC_PATH` is configured.
- Atomic watershed output tags every single-nucleus basin, splits every multi-nucleus basin inside
  its own bounding box, re-densifies labels, and independently reconstructs complete count words
  and boundary bits from the post-split labels.
- Nucleus IDs live in a packed side-channel record rather than the watershed label namespace, so
  the two ID spaces cannot collide.
- Atomic agglomeration performs the tag-group closure after ordinary `try_merge` processing and
  before `remaps`; tagged components are dust-exempt and cross-tag edges cannot enter the emitted
  dendrogram.
- Hierarchy processing loads child tag records, guards plateau/descent/ordinary/MST union sites,
  performs the same closure before building remaps, and aborts on conflicting records for one ID.
- Tagged interior components remain transportable below the top hierarchy level, including the
  zero-face path, and are finalized into `done_pre`/`done_post` at the top level. This prevents a
  correct tag from remaining permanently `ongoing` and missing final `ws3` remapping.
- Added `ABISS_WS_NUC_GUARD`, default `ON`. The `OFF` build disables only cross-tag compatibility
  checks; split inputs, tags, closures, and dust behavior remain enabled.

## Implementation Details

The seeded split uses every mask voxel as a fixed marker and floods only voxels belonging to the
original basin. Candidates are ordered by maximum bottleneck affinity, then nucleus ID and voxel
index for deterministic ties. Unreached basin voxels receive one untagged label. The code comments
state that this cut has no image support: the measured bottleneck is approximately 0.999 against a
0.3 merge threshold, so imposing it is defensible only because the mask is confirmed correct and
conservative.

`nuc_union` adds real sizes, ORs `on_border`, clears the losing count/tag, links the DSU, and moves
both count and tag to the representative actually chosen by Boost. Ordinary tagged+untagged unions
use the same ownership rule. Multi-threshold execution takes a fresh tag vector copy beside each
segmentation, region graph, and count copy.

At hierarchy levels, non-top tagged components are emitted in the count/tag/ongoing streams even
without physical boundary contact, allowing disconnected same-tag pieces to meet at a later
parent. The top-level flag added to `param.txt` changes only their final classification: after the
top closure they become real done remaps instead of an unused terminal side channel. A single-child
top level is processed rather than lifted so it receives the same finalization.

The existing atomic and composite shell tar globs already include `nuc_tag_<tag>.data`; no shell
change was needed to carry the new file. `remap_chunk_ws.sh` consumes the materialized hierarchy
remaps through its existing `merge_remaps.py -> ws3 -> upload_chunk.py` path, so the tag side file
is not required after hierarchy completion.

## Files Changed

| File | Purpose |
|---|---|
| `CMakeLists.txt` | Add the default-ON guard seam and focused ON/OFF test targets. |
| `scripts/cut_chunk_ws.py` | Cut and validate the nucleus mask into the padded atomic watershed coordinate frame. |
| `scripts/merge_chunks_ws.py` | Merge/lift tag files, process a single-child top level, and pass top-level finalization state to `ws2`. |
| `src/ws/nucleus_tags.hpp` | Define the collision-free packed tag wire format and guard compatibility algebra. |
| `src/ws/nucleus_snap.hpp` | Implement basin tagging, seeded splitting, deterministic re-densification, and complete count/border reconstruction. |
| `src/ws/agglomeration.hpp` | Add representative-safe tagged unions, atomic tag closure, guard checks, tag compaction, dust protection, and MST vetoes. |
| `src/ws/atomic_chunk.cpp` | Load `nuc.raw`, invoke snapping before region-graph construction, copy tags per threshold, and emit tag records. |
| `src/ws/merge_chunks.cpp` | Consume/emit hierarchy tags, guard all four union sites, close tags before remaps, protect tagged dust, handle zero-face work, and finalize at the top. |
| `work/test/test_ws_nucleus.cpp` | Check split count words, independent border oracle, same-tag unconditional closure, dust survival, and atomic guard ON/OFF. |
| `work/test/test_ws_nucleus_atomic.py` | Exercise HDF5 mask ingress, the real `ws` binary, split/tag output, and multi-threshold tag copies. |
| `work/test/test_ws_nucleus_hierarchy.py` | Exercise zero-face closure, child/parent versus monolithic identity, parent tag/remap inspection, bridge guards, `ws3`, chunkmap, global-table, and `acme` tracing. |
| `scripts/cut_chunk_agg.py`, `scripts/set_env.py`, `src/seg/NucExtractor.hpp`, `src/seg/atomic_chunk_ME.cpp`, `scripts/nucleus_utils.py`, `scripts/build_nucleus_table.py`, and sibling tests | Pre-existing global-nucleus-table baseline consumed or preserved by this run; not reverted or re-owned here. |

## Git Baseline

run_start_ref: 312bf54183faa94be1aff8ae246fcc211808f0bb
current_head: 312bf54183faa94be1aff8ae246fcc211808f0bb

## Verification

- **V1 — PASS.** Command:
  `set +u; source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc; set -u; export HDF5_USE_FILE_LOCKING=FALSE; work/test/run_v2_invariance.sh`
  Result: `NUC_TABLE unset` and `NUC_TABLE set without NUC_PATH` each reported
  `identical=38 differing=0 missing=0`; both reported zero extra files and zero nucleus log lines;
  stale nucleus environment passed; final line `run_v2_invariance: PASS`.
- **V2 — PASS.** Command:
  `python work/test/test_ws_nucleus_atomic.py --repo . --ws build/ws`
  Result: HDF5 mask ingress produced the padded `nuc.raw`; one real watershed basin split into two
  single-tag pieces; mask seeds mapped to different IDs; full counts summed to the output volume;
  both multi-threshold outputs retained tags `{11,22}`. Command `build/test_ws_nucleus` also passed
  the independent post-split count-word/border oracle, same-tag closure below the ordinary edge
  threshold, and tagged dust survival.
- **V3 — PASS.** Commands:
  `build/test_ws_nucleus && build/test_ws_nucleus_guard_off` and
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DABISS_WS_NUC_GUARD=OFF && cmake --build build --parallel 4 --target ws2 && python work/test/test_ws_nucleus_hierarchy.py --repo . --ws2 build/ws2 --guard 0`.
  Result: atomic tests passed with `guard=1` and `guard=0`; the real OFF `ws2` build merged the
  tag-to-untagged-to-other-tag plateau bridge. The tree was then reconfigured and rebuilt with
  `ABISS_WS_NUC_GUARD=ON`; ON kept two tag components. Same-tag closure merged components whose
  sizes exceeded the ordinary size threshold and whose edge was below the merge threshold.
- **V4 — PASS.** Command:
  `python work/test/test_ws_nucleus_hierarchy.py --repo . --ws2 build/ws2 --guard 1 --trace`.
  Result: zero-face same-tag IDs became remaps before emission; hierarchy-level tagged dust
  survived; child+parent identity equaled monolithic identity; parent tag records and
  `done_pre`/`done_post` remaps were inspected; a tag-to-untagged bridge did not cross-stitch the
  second tag.
- **V5 — PASS.** Same command as V4 with `--trace`.
  Result: known IDs were followed through parent remaps, `ws3`, `chunkmap.data`, the existing
  global nucleus-table reducer, and `acme`; the final records were two `PROPER` records mapping
  nuclei `{7,8}` to one supervoxel each. This fixture's chunkmap was the valid empty identity map
  because the chosen final representatives were already raw watershed IDs.
- **V6 — NOT RUN.** `/usr/bin/sbatch` is available, but this allowed worktree has no
  `dev/zebrafinch/nuc_z1_y7_x6` crop arm or `w2ctl`/`wssnap` inputs. Per the task, the coordinator
  must run the SLURM crop A/B. Therefore the eight-nucleus mass preservation, cannot-link and
  must-link invariants, 110,244-to-0 shared mass, 95% coverage, dominance, largest segment/root
  count, wall-clock, and whether real split pieces were eligible for ordinary refusion are all
  **NOT RUN / unknown**, not claimed.
- **Input contracts — PASS.** Command:
  `python work/test/test_input_contracts.py` with `HDF5_USE_FILE_LOCKING=FALSE`.
  Result: accepted integer/identity/resolution-aligned nucleus inputs, rejected oversized/float/
  negative/wrong-shape inputs, verified set-env gating, and ended `test_input_contracts: PASS`.
- **Build/source checks — PASS.** Command:
  `cmake --build build --parallel 4 --target ws ws2 ws3 acme test_ws_nucleus test_ws_nucleus_guard_off`
  completed, followed by both C++ test binaries and `git diff --check`. Focused Python checks ran
  Black, black-profile isort, and flake8 on the new test files, plus `py_compile` on the two changed
  pipeline scripts. Final output: `final build/status gate: PASS`; HEAD remained the run start SHA.

## Review Focus

- Confirm the atomic closure remains between the ordinary region-graph merge loop and `remaps`,
  and the hierarchy closure remains before its `remaps` vector and `done_pre`/`done_post` output.
- Inspect tag/count ownership after Boost chooses a representative, especially ordinary
  tagged+untagged unions and hierarchy plateau/descent unions.
- Check the below-top transport versus top finalization contract, including single-child and
  `face_size == 0` paths, for consistency with production hierarchy JSONs' `top_mip_level`.
- Check the seeded flood's affinity indexing, deterministic tie order, bbox restriction, and
  post-split boundary oracle against the watershed's Fortran-order coordinate convention.
- Verify that OFF only bypasses `ws_nuc_tags_compatible` and does not disable split/tag inputs,
  closure, or dust protection.

## Risks and Unknowns

- The seeded cut is deliberately unsupported by image evidence (approximately 0.999 bottleneck
  against 0.3); correctness depends on the accepted premise that the mask is correct and
  conservative. A bleeding mask can impose an unconditional false must-link.
- V6 was not run, so crop-level quality, runtime, runaway/shatter limits, and ordinary-refusion
  eligibility remain unknown. No claim is made that the feature fixes `worst3` until that A/B runs.
- The per-conflict flood allocates arrays over the original basin bounding box. Atomic chunk size
  bounds this, but a large sparse bbox can cost more memory than the basin voxel count alone.
- No terminal-plan requirement was dropped. One necessary hierarchy detail was made explicit while
  implementing: tagged interior components must be carried below the top but finalized at the top.
  Omitting that distinction would satisfy local tag tests while stranding final remaps—the exact
  fourth no-op risk identified in review.

## Changes Since Previous Code Version

Initial implementation.
=========================== END artifacts/code_v0.md ===========================

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

Full read access to work2/abiss. The failing run's log is at
/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/nucdec_nuc_wssnap_2821533.out
