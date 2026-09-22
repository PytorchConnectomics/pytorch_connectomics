# Plan v1 Review

## Summary

Plan v1 is not ready. It genuinely resolves the Tier-1 call graph, force-split ownership, cache slicing, full-volume metric API, NERL invocation, extraction verification, and most CLI protections. However, several load-bearing contracts remain missing or contradict the task.

## Findings

1. [major] Prob-1 carve still lacks the mandatory bump-safe proposal/commit gate. Through-intruder geometry, orphan status, and area matching do not prove that the carve removes a host bump without adding one to either result. The proposed solid-tube test only exercises “no candidate,” not rejection of an unsafe candidate. Prior findings #5 and part of #17 remain unresolved.

2. [major] `prob1_extend` is not callable as specified. It constructs `bp=(tau,min_excess,max_len,med_w)`, but none of those values is an argument or declared fixed constant. After an accepted mutation, the plan updates only `area`, not `zr`, `nf`, open ends, or orphan eligibility. Thus the second direction can run using stale completeness and endpoint state, contradicting the claimed recompute-on-accept behavior.

3. [major] Prob-2 violates the mandatory area-match rule. Its default `area_tol=None` disables area matching on the shipped path, while the task says a pair merges iff it is area-matched. The optional comparison uses whole-component voxel counts, not a defined cross-sectional or endpoint-area statistic, so it does not resolve prior finding #7.

4. [major] Prob-2 raw-affinity transport remains undefined. The supplied `region_graph(seg)` accepts no affinity argument, yet the proposed `prob2_merge(..., aff_path=AFF)` says that helper reads the parameter without specifying a signature change or loading path. It also has no `z0` offset, so a nonzero slab cannot select the corresponding raw-affinity slices.

5. [major] `weak_bridge` is not a complete callable contract. It uses undeclared `max_zoverlap` and bump parameters, records centroids without a centroid bound, and leaves the weak-chain area statistic undefined. It also does not define the cross-array overlap calculation between a strong endpoint mask and a weak section. `section_index(seg2d)` alone does not specify how two endpoints and centroids are recovered from tube labels repeated across slices.

6. [major] Weak exclusivity is incomplete. A connected weak component can branch or loop while still touching exactly two strong ends, yet the entire component would be accepted. The plan does not explicitly reject same-tube anchors, already-unified roots, non-facing ends, or unsafe transitive unions. Candidate checks are not defined against evolving union-find component ranges/profiles, and equal min-IoU candidates lack a deterministic tie-break.

7. [major] `bump_safe_merge` remains under-specified. Weak labels and strong labels occupy separate label namespaces, so `area, ids` does not define a collision-free combined profile. `bp` defaults and the `zs` input required by `bumps(zs, ar, ...)` are absent. “No new bump vs the largest member” does not define “largest” or the actual comparison, and count-only inequalities can allow one bump to replace another.

8. [major] Crumb absorption is not implementable from its signature. It references raw region-graph data, `min_contact`, `max_zoverlap`, and bump parameters without accepting or defaulting them. “Dominant neighbor” has no numeric rule, and no affinity threshold, area criterion, processing order, graph recomputation, or cumulative exclusivity rule is given. The drop-only default also leaves orphan segments between `min_vox=500` and `orphan_min_vox=2000`, so the default does not perform the requested orphan cleanup.

9. [major] The promised slab-face semantics cannot be implemented. Tier-2 and weak stages call local `face_counts(seg)`, and `score_array` receives no global bounds or enabled-face mask. Consequently, artificial slab z-boundaries will affect incomplete-tube selection and scoring. This leaves prior finding #16 only partially resolved.

10. [major] The orchestrator still has an incomplete raw-input data flow. `run_pipeline` uses `aff` and `affxy` without loading them. `build_strong_sections` loads affinity only locally on a cache miss and returns only sections. Its direct waterz pseudocode also uses an undefined `shape`. As written, the cache-hit path cannot reach Tier 1, Tier 2, or weak sectioning with the required affinity arrays.

11. [major] Weak morphology is still partly deferred. The plan specifies `spherical_open(..., radius=...)` and then says the coder will choose between `spherical_open` and `opening`; the supplied API list does not establish that keyword contract or border behavior. It also substitutes binary opening for the task’s requested multi-label opening. Prior finding #8 is not fully resolved.

12. [major] The self-tests still omit load-bearing cases from prior finding #17: an unsafe-but-otherwise-eligible carve rejection, explicit weak-only-chain discard, and opt-in crumb absorption acceptance/ambiguity/bump fallback. A synthetic Prob-2 test also lacks a concrete raw-affinity injection contract.

13. [minor] Default-policy prose is inconsistent. The summary and CLI make weak recovery always opt-in, while “Changes Since” says it becomes default-on after successful validation. The summary also says the default reproduces `decode_p1p2` while adding extension and expecting `decode_p1p2eg`; those are distinct pipelines.

## Questions

None blocking; the necessary corrections follow directly from the task and prior review.

## Verdict

VERDICT: NEEDS_CHANGES

READY: no