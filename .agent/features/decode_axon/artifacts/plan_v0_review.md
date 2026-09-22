# Plan v0 Review

## Summary
Codex (coder) reviewed plan_v0 for executability under a read-only sandbox. Verdict: NOT executable as
written — the plan is a coherent design but under-specifies the exact call contracts of the reused
functions, the measured default parameters, and the safety/exclusivity logic of the new stages
(weak_bridge, crumb, prob2 area-match). All findings are fixable in a revision; none are hard failures.
Raw transcript: `state/plan_v0_review.review.raw.md`.

## Findings
Preserved from the raw review (all tagged [major] unless noted):

1. Tier 1 data flow incomplete: `_raw_iou_table` needs `z_of`; `detect_fusions` needs `area/fuse_iou/
   area_tol`; `eligible_fusions` needs `min_len`; `constrained_relabel` needs `runs` + `candidate_links`;
   `_candidate_links(...)` is omitted; how force-split yields `runs` is unstated. → extract ONE canonical
   pure pipeline function from `decode_v2._run_pipeline` or specify every call/return/default.
2. `force_split` and `section_index` sources/signatures not shown (they live in `force_split_decode`);
   the force-split return structure consumed by `constrained_relabel` must be identified.
3. `build_strong_sections` cannot call `_load_inputs(args)` (argparse object). Must map
   `waterz_2d_spacefill` args directly; for nonzero `--zslice`, slice `[z0:z1]` from the full `z0-800`
   cache — a `z0-{depth}` exact-cache name is wrong for a sub-slice.
4. `decode_3d` composition not callable: `prob1_carve`/`prob1_extend` omit `affxy` and keyword-only
   values; `prob2_merge` omits `aff_raw`; "measured defaults" are not stated.
5. Prob-1 carve does not enforce a bump-safe proposal/commit rule and does not restrict to incomplete
   tubes explicitly (only extension is described as gated).
6. `prob1_extend` does not define `extend_end` params (`area_tol/host_margin/mem_thr/max_steps`), how
   it builds `zr/area/carved/bp`, bidirectional ordering, or recompute-after-mutation; copy-vs-extract
   from `tube_extend.main` unresolved.
7. Prob-2 omits the exact area-match statistic/threshold and cites `compute_bbox_all_3d` (available as
   `z_ranges`/`z_overlap_frac`); `region_graph(seg)` takes no affinity, so the extraction must preserve
   channel-correct raw boundary-affinity aggregation and state whether checks apply to original labels
   or evolving union-find components.
8. Weak sectioning leaves topology-defining choices open: binary vs multilabel opening, exact fastmorph
   op, structuring element/radius, border behavior, `min_cc` default.
9. `weak_bridge` under-specified: no numeric limits for z-gap, centroid distance, area match, overlap/
   contact, skipped slices; helper reuse unstated; "never weak-weak" conflicts with the multi-slice weak
   chain (correct contract: weak-to-weak allowed only when anchored by exactly two accepted strong ends).
10. Weak bridging lacks global exclusivity (a branched weak component could link ≥3 strong ends). Require
    two-anchor-only paths + weak-node/strong-endpoint mutexes + deterministic conflict resolution +
    rejection of ambiguous chains.
11. `gate_bumpsafe` is orphan-carve-specific, not a generic merge checker; a concrete adapter or a
    separate bridge/absorption safety test is required for strong–weak–strong merges and crumb absorb.
12. Crumb absorption non-deterministic ("adjacent/larger/unambiguous" undefined; bump-safety alone
    doesn't stop parallel merges). Make drop-only the default; absorption behind its own flag with
    explicit affinity/z-overlap/contact/area/exclusivity rules; state fate of unmatched orphans.
13. Metric calls invalid: `valid_tube_metric.score` takes a PATH + nine metric args, not an ndarray.
    Save each ablation tier as HDF5 then score with the corrected-baseline params (or add an array API).
14. Full-run NERL unspecified: identify the `liconn_nerl` callable/CLI, pred/GT args, keys, params,
    execution point.
15. Default-output validation incomplete: crumb is default-on but the gate checks only strong+3D and
    optional weak; the FINAL saved artifact must satisfy all three metric limits; state whether weak
    becomes default on success or stays disabled on failure.
16. CLI/output semantics: require exactly one mode, validate slice bounds, `--self-test` must exit
    before data I/O, prevent an untagged z-slice run from overwriting `decode_axon.h5`, define whether
    slab boundaries count as faces (label z-slice metrics smoke-only).
17. Self-tests omit load-bearing contracts (Prob-1 carve rejection, Prob-2 sequential-accept/concurrent-
    reject, weak-only-chain removal, ambiguous-chain rejection, accepted-chain exclusivity, safe crumb
    fallback); the bump-safe fixture must be constructed explicitly.
18. Behavior-preserving extraction should compare saved `main` arrays (normalized partitions), not HDF5
    bytes; and because `decode_v3_merge.py`/`decode_v4_split.py` are also changed and gitignored, their
    full contents/diffs must accompany the later code review.
19. [minor] "strong-only reproduces decode_p1p2" is wrong terminology: Tier 1 = decode_v2; strong+3D =
    decode_p1p2.
20. [minor] Verification commands should use `conda run -n pytc ...` per repo instructions.

## Questions
None blocking; the primary-metric assumption (valid-tube volume, NERL secondary) was accepted implicitly
by the reviewer.

## Verdict
VERDICT: NEEDS_CHANGES
