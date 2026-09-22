# Plan v1 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, `xhigh`), read-only sandbox. Raw transcript:
`state/plan_v1_review.review.raw.md`. Eighteen major findings, none softened.

v1 fixed v0's circular oracle, ROI-local anchor fractions, bridge over-repair and invented
threshold, but the tightened plan exposed deeper contract holes: an invariant that the design
contradicts by construction, a measured gate with a `NOT RUN` escape and a vacuous-abstention
loophole, a constraint lifecycle that installs cannot-links against ids consolidation later
removes, and a bounded ROI that can leave the anchors joined through the untouched exterior.

## Findings

- [major] Certified-but-unrepaired bridge conflicts contradict invariant 1. Those components will still contain multiple qualifying anchors, so verification must either fail or silently exempt them. The plan must define partial-success semantics: repaired scopes satisfy the invariant, unrepaired certificates remain explicitly unresolved, and the checkpoint result cannot report them as verified repairs.

- [major] V4 is not actually a gate if missing data yields `NOT RUN`. That permits completion with only synthetic evidence. The required `worst3` input must be checked in or V4 must be a mandatory external acceptance step whose absence prevents declaring the implementation complete.

- [major] V4 still permits vacuous “repair.” Conservation allows arbitrary abstention, so an implementation could remove or abstain most contaminated voxels and improve sharing/dominance. The gate needs pinned assigned/abstained counts or an exact expected output partition/hash. The misplaced-voxel tolerance must be fixed before implementation, not documented after observing results.

- [major] The arm0_96 gate asserts only `16/16` counts. An implementation could repair the wrong 16 conflicts. It must assert the exact conflict identities/scopes classified as contact versus bridge, plus that no bridge-class scope was mutated.

- [major] Contact eligibility remains an unresolved design choice: caller-supplied scopes or a gap selector. If using supplied scopes, their immutable format and provenance must be specified. If using a selector, its exact distance definition, threshold, units, tie handling, and tests must be fixed. This is a decision required before coding, particularly because V4 depends on it.

- [major] The golden ordering is procedural, not enforceable or auditable as written. A genuine oracle requires a separately reviewable baseline-capture commit or manifest containing untouched script hashes, repository revision, exact commands/configuration, environment/version information, input hashes, and output arrays/hashes. Merely saying “captured before the lift” cannot prove that it was.

- [major] The fixture story is internally inconsistent. Section B says a real `worst3` sub-ROI is checked in, while V4 says the required crop may be unavailable. It must identify whether the checked-in real fixture is sufficient to reproduce all V4 numbers; otherwise the golden regression and measured gate are different datasets and both contracts need explicit boundaries.

- [major] Rewiring `dev/zebrafinch/*.py` “last” cannot satisfy the existing-CLI regression contract in a clean checkout because those files are gitignored. The plan must either leave those scripts untouched and test only the lifted behavior, or define a tracked mechanism that preserves and tests their CLI behavior.

- [major] Test #3 has a strong numerical oracle, but its integration boundary is ambiguous. It must exercise serialized describe → plan → apply through the real executor, explicitly provide one atomic fragment/supervoxel ID, assert that it becomes multiple anchor-driven territories, and compare exact assigned and abstained masks. A kernel-only test would still allow the production executor to perform no repair.

- [major] “Meaningfully assigned” is not executable. Replace it with exact expected territory and abstention masks, including a fixed nonzero count of non-anchor component voxels assigned to each expected territory.

- [major] The numerical lift remains underspecified despite being called fully specified. Connectivity has no selected value, tie-breaking has no rule, channel/store conventions are conditional, pooling origin and boundary padding are unspecified, and upsampling alignment/cropping is undefined. These choices affect exact territories and must be pinned before golden capture and coding.

- [major] Symbolic constraint binding has a lifecycle hole. One `(component, anchor)` may produce multiple pieces; `forbid_merge` is installed before consolidation; consolidation and RAG rebuilding may then change IDs. The plan must define stable territory identity, pair expansion, and rebinding so the final decoder-consumable constraints still reference every final distinct-anchor territory.

- [major] Invariant 7 is only nominally addressed. Comparing canonical plan JSON proves deterministic planning, not deterministic execution. Repeated application from the same frozen inputs must compare output partition, abstention mask, territory bindings, constraint manifest, action outcomes, and graph invalidation/rebuild result, excluding only timing and timestamps.

- [major] The real cannot-link format remains explicitly deferred to implementation. That format determines constraint identity, serialization, and replay semantics, so it must be discovered and selected before finalizing the schema. Similarly, `rebuild_local_rag` still has no chosen backend, executable postcondition, or focused test.

- [major] Scalable I/O is still mostly a file-list promise. There is no test requiring bounded reads or rejecting a whole-volume read, no concrete corrected-volume/delta representation that avoids a full copy, and no atomic output/append-log contract. A fake blockwise store that fails on unbounded access should cover this.

- [major] The anchor-total command contract is inconsistent. Section D treats totals as a required input, while sections D/K say `describe` computes and writes them even though descriptors need them during that same operation. Define a separate streaming precompute artifact or make `describe` explicitly consume an existing totals artifact, with the nucleus-volume hash and reference extent included.

- [major] v0-F14 is only nominally addressed. “Add `annotate`” does not define its typed parameters, execution behavior, or serialization. The representation of per-anchor IDs, local counts, authoritative totals, overlap fractions, and `certified_unrepaired` status also remains unspecified.

- [major] Bounded-ROI repair can leave an exterior bridge through the unchanged portion of the original component. The plan must require full containment of the conflicting parent component or define interface relabeling and global connectivity verification; outside-scope partition equivalence alone does not prevent the repaired territories from remaining connected outside the ROI.

## Questions

1. Will eligibility use a checked-in authoritative contact-scope artifact, or a deterministic gap selector? If the latter, what exact threshold and distance definition?
2. Which exact checked-in data and predeclared tolerance make V4 mandatory and reproducible, including expected abstention?
3. What consumer format and stable identity model will keep cannot-links valid through consolidation and local RAG rebuilding?

## Verdict

VERDICT: NEEDS_CHANGES
