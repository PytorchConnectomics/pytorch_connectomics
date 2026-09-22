# Plan v0 Review

## Summary

The outline is broadly right, but the plan is not ready: it carries several stale concepts from `tracklet.md` into what should be a clean description of the current pipeline. The strongest framing is the asymmetric repair strategy—remove merge risk, then conservatively recover connectivity—not “one merge operator reused at growing scale.”

## Findings

- [major] The definition of `om` is conceptually inverted. Oracle merging removes false splits, so `om` is the ceiling after ideal split recovery, still limited by false merges and coverage. Calling it a “false-merge-free ceiling” contradicts both the old operational definition and the fact that splitting raises `om`. Define base and `om` operationally.

- [major] “One merge operator reused at growing scale, differing only in the split signal” is stale `v3_ab` framing. The current stages perform materially different operations: seam cutting or marker-based carving in v1, completion plus adjacent mutual-IoU union in v2, and trajectory-based weak-gap bridging in v3. Reframe them as one risk policy with distinct operators.

- [major] The proposed signal set mixes old and current methods. Current v1 explicitly uses consecutive-slice real IoU, while shifted/centroid-aligned IoU is foregrounded by the older design. The document must distinguish spatial IoU from shifted shape IoU and use each only where the current pipeline supports it. Likewise, not every split is propagated 2D watershed: link cuts sever a seam; watershed is used when carving fused cross-sections.

- [major] The signals table should distinguish evidence from operations and decision rules. Cross-section completion is preprocessing, mutual-best is an ambiguity rule, affinity is an eligibility floor, and IoU/trajectory/caliber are evidence. The table should prioritize current cues: local IoU discontinuity, anchored trajectory/caliber consistency, adjacent cross-section overlap, weak-path continuity, and affinity’s supporting role. One-sided containment belongs only as an optional residual cue, not a core signal unless its current status is established.

- [major] v3 is optional and has no fair `thr=10`/943-GT result in the supplied table. It must not appear as part of the quantitatively validated v2 result. Also, collinearity is positive evidence for v3, not a reliable veto; the supplied history says collinear wrong-partner bridges remain a failure mode. Label v2 as the validated operating point and v3 as a risk-bearing extension.

- [major] The residual branch-split material needs a sharper boundary. The durable lesson is observability: two-sided anchors can make a fused interval separable, while a one-ended disappearing tube is ambiguous. Do not promote “2-1-2,” one-sided/two-sided gates, or a single-example success into the canonical pipeline.

- [major] Add an evidence-scope limit: the thresholds and reported gains come from one volume and one corrected yardstick. The transferable contribution is the representation, signal hierarchy, and asymmetric error policy; new volumes require calibration. Also retain weak/missing-foreground failure and v3’s false-merge cost, not only the `sep=0` case.

- [minor] A target of 150–250 lines risks manufacturing bloat. Prefer a one-read cap, such as roughly 1,000–1,500 words, with no minimum.

- [minor] Do not include the proposed code-map line. It is implementation routing, can become stale, and does not help explain the method.

- [minor] Broaden verification beyond four seg-ids. Check for all example-specific identifiers, old sub-method names, M1–M8 labels, 911-yardstick numbers, bug history, and named failed refinements.

## Recommended outline

1. Objective, tube representation, and asymmetric error cost.
2. Current pipeline, clearly marking v2 as validated and v3 as optional.
3. Evidence hierarchy and repair operators.
4. Evaluation definition and fair-yardstick results.
5. Observability limits, bridge risk, and transfer scope.

KEEP the core idea, current signals with their roles, reasons for 2D cross-sections and shape-primary linking, fair metric definition, validated evidence, and structural limits. CUT the taxonomy, seg-ids, named historical sub-methods, per-case gates, bug chronology, failed-refinement log, stale metrics, and code map.

READY: no