## Summary

The insertion point is conceptually correct, but plan v0 is not executable as written. It omits the label/count remapping, canonical-ID encoding, and hierarchy-level tag propagation needed to keep the correction alive. Those omissions can yield inconsistent artifacts or another result identical to control.

## Findings

- [major] Splitting between `watershed()` and `get_region_graph()` is the right insertion point only if `counts` is rebuilt. The current `counts` describes the original basins and carries `on_border` bits used by `try_merge`. Splitting, coalescing same-nucleus basins, and adding fresh labels invalidates its sizes, indexing, and border flags. The plan must require dense local labels and reconstruct `counts` before `get_region_graph(..., counts.size()-1, ...)`.

- [major] Step E is inadequate and conflicts with the proposed implementation. `relabel_segments(seg, offset)` converts compact local labels into the chunk-prefixed namespace, so merely reserving a nucleus range does not make nucleus `N` receive the same ID in every chunk. The plan needs an exact ID formula and a concrete path through segmentation, counts, dendrogram endpoints, boundary files, and remaps. If canonical IDs are applied before `get_region_graph`, they may also exceed `internal_seg_t` or make `basin_tag[id]` impractically sparse. Without this design, fragmentation will not improve across chunks.

- [major] The tag side file must be aligned to labels after `merge_segments` compacts them, not to the pre-merge basin IDs. It must also be emitted separately for every multi-threshold `out_tag`. Otherwise `merge_chunks.cpp` will look up unrelated or absent tags, silently treating everything as untagged and making D a no-op.

- [major] An atomic per-chunk side file does not survive the watershed hierarchy by itself. Parent merges change representatives and emit remaps; tags must be matched to those remaps, propagated through accepted tagged-plus-untagged unions, reduced at every parent, and written for the next level. This needs the same kind of hierarchy-aware treatment as the nucleus records. The current scripts only show hierarchy remap collection, not tag transport.

- [major] C must guard both phases of `merge_segments`. Skipping `try_merge` is insufficient: the `new_rg`/MST loop can still emit a high-score edge between different tagged components into the dendrogram. That edge can subsequently be consumed by hierarchy or agglomeration. The existing global nucleus-table veto may save the final agglomeration, but that makes it another load-bearing component and disproves the claim that C+D alone provide the `somaBFS` guarantee.

- [major] Dust removal is another missed destructive path. The final remapping loop deletes components below `lowt` without considering nucleus tags. A small seeded nucleus piece can therefore become label zero. Tagged components need explicit preservation or a stated, tested alternative.

- [major] C is not demonstrated to be load-bearing by the supplied code. `try_merge` immediately rejects an edge when both components are at least `size_threshold`; a high affinity alone does not cause refusion. V3 must deliberately make one split piece small enough, satisfy the border conditions, and confirm its edge score exceeds the threshold. It should also report whether the real crop’s split pieces were actually eligible for this merge.

- [major] V3 is therefore not reliably capable of exercising either guard as written. “With C/D disabled” needs a concrete test seam, and the fixtures must force an eligible in-chunk merge and an eligible cross-chunk stitch. Otherwise the enabled and disabled cases can behave identically for reasons unrelated to the cannot-links.

- [major] V4 uses only one nucleus spanning a boundary, so it cannot detect loss of different-nucleus tags or a broken cross-tag veto. It needs at least two tags meeting through a boundary, preferably with an untagged bridge, and should inspect parent tags/remaps as well as segmentation equality.

- [major] V5 is gameable. Shared mass, fragmentation, and dominance can improve if mask voxels are deleted to background, painted with final IDs without correcting watershed artifacts, or omitted from the report. Require all eight nuclei to be present, exact preservation of their mask voxel counts as nonzero labels, the expected nucleus-to-canonical-ID mapping, and inspection of the watershed/chunkmap artifacts—not only final segmentation statistics. All required criteria should be reported for both arms explicitly.

- [major] The most likely fourth no-op is: split locally, leave tags indexed in the pre-merge namespace, let hierarchy treat them as zero, emit cross-tag dendrogram edges, and finally apply ordinary chunk-offset IDs. Every individual change would run, but the treatment could reconstruct the control segmentation.

- [minor] The stated 243M-voxel per-basin watershed cost is misleading at this insertion point: atomic processing is bounded by the roughly 252³ chunk. The 243M object exists only after stitching across 88 chunks.

## Questions

- Will canonical nucleus identity be encoded directly in the output ID, or represented by hierarchy-aware remaps from compact local IDs? Step E must choose one design.

- What exact artifact does each hierarchy level consume and emit to preserve tags after representative changes?

READY: no