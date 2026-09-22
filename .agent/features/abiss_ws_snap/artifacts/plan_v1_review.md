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
