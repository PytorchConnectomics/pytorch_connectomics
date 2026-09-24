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
