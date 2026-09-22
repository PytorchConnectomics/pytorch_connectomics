# Plan v1 Review

## Summary
Codex reviewed plan_v1 (read-only). It confirms v1 resolved the Tier-1 call graph, force-split
ownership, cache slicing, the full-volume metric API, NERL invocation, extraction verification, and most
CLI protections. Verdict remains NEEDS_CHANGES: several contract details are still missing, concentrated
in the weak-recovery and crumb-absorption tiers, plus a few real gaps in the shipped core (prob2
raw-affinity slab offset, orchestrator affinity loading, prob1_extend `bp` defaults / recompute scope,
prob1_carve explicit bump gate). Raw transcript: `state/plan_v1_review.review.raw.md`.

## Findings
Preserved from the raw review ([major] unless noted):
1. prob1_carve still lacks an explicit bump-safe proposal/commit gate; the solid-tube test only exercises
   "no candidate," not rejection of an unsafe candidate. (prior #5/#17)
2. `prob1_extend` not callable: `bp=(tau,min_excess,max_len,med_w)` values are neither args nor fixed
   constants; after an accepted mutation only `area` is updated, not `zr/nf`/eligibility, so the 2nd
   direction can use stale state (contradicts recompute-on-accept).
3. Prob-2 area-match: default `area_tol=None` disables it on the shipped path while the task requires
   area-matched merges; the optional comparison uses whole-component voxel counts, not a cross-sectional/
   endpoint statistic. (prior #7)
4. Prob-2 raw-affinity transport undefined: `region_graph(seg)` takes no affinity arg; `prob2_merge(...,
   aff_path=AFF)` doesn't specify the signature change or loading; no `z0` offset, so a nonzero slab
   selects the wrong raw-affinity slices.
5. `weak_bridge` incomplete callable: undeclared `max_zoverlap`/bump params, centroid recorded without a
   bound, weak-chain area statistic undefined, strong-endpoint↔weak-section overlap calc undefined;
   `section_index` alone doesn't recover two endpoints+centroids from cross-slice-repeated tube labels.
6. Weak exclusivity incomplete: a connected weak component can branch/loop while touching exactly two
   strong ends and be wholly accepted; no rejection of same-tube anchors, already-unified roots,
   non-facing ends, unsafe transitive unions; checks not defined against evolving UF component
   ranges/profiles; no deterministic tie-break on equal min-IoU.
7. `bump_safe_merge` under-specified: weak/strong label namespaces collide so `area,ids` isn't a
   collision-free profile; `bp` defaults and the `zs` input to `bumps` absent; "largest member" undefined;
   count-only inequality can let one bump replace another.
8. Crumb absorption not implementable from its signature (min_contact/max_zoverlap/bump params not
   accepted/defaulted; "dominant neighbor" no numeric rule; no order/recompute/exclusivity). Drop-only
   default leaves orphans in [min_vox, orphan_min_vox), so it doesn't perform the requested orphan cleanup.
9. Slab-face semantics unimplemented: Tier-2/weak call local `face_counts`; `score_array` gets no global
   bounds/enabled-face mask, so artificial slab z-boundaries affect incomplete detection + scoring. (prior
   #16 partial)
10. Orchestrator raw-input data flow incomplete: `run_pipeline` uses `aff`/`affxy` without loading;
    `build_strong_sections` loads affinity only on cache miss and returns only sections; pseudocode uses
    undefined `shape`; cache-hit path can't reach Tiers with the needed affinity arrays.
11. Weak morphology still partly deferred (spherical_open vs opening keyword contract/border not
    established; binary substituted for the task's multi-label opening). (prior #8)
12. Self-tests omit: unsafe-but-eligible carve rejection, weak-only-chain discard, opt-in crumb-absorb
    accept/ambiguity/bump fallback, and a concrete Prob-2 raw-affinity injection contract. (prior #17)
13. [minor] Default-policy prose inconsistent (weak opt-in vs "default-on after validation"; "reproduces
    decode_p1p2" while adding extension/expecting decode_p1p2eg — distinct pipelines).

## Questions
None blocking; corrections follow from the task + prior review.

## Verdict
VERDICT: NEEDS_CHANGES
