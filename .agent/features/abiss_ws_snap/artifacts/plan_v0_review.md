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
