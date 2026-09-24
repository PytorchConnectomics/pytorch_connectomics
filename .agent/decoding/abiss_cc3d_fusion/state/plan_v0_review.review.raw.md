# Plan v0 Review

## Summary

The plan is directionally sound and correctly resolves the affinity source as `arm0_win144`. However, it is not executable as written: Phase 0 crosses the GT firewall, Phase 1 cannot construct the claimed directional candidates from skeleton-node samples, and the policy thresholds and global conflict resolver are insufficiently specified.

## Findings

1. [major] **Phase 0 violates the GT-free/evaluator-only separation.** `phase0_input_audit.py` is described as writing `gt_free/input_manifest.json` while also reproducing ABISS NERL from skeleton LUTs and re-aggregating human-GT tier10 results. A GT-free stage may not read either source. Split this into:

   - a GT-free identity manifest written before any decoding;
   - evaluator-only fidelity results under `evaluation_gt/`.

   The known expected NERL may be recorded as a constant in the input manifest, but the measured reproduction and human-GT tier10 result must remain evaluator-only.

2. [major] **The proposed Phase 1 data is insufficient for its directional oracles and Gate A.** Sampling CC3D only at skeleton nodes cannot determine all ABISS overlaps within a component, whether an unseen established anchor is present, global fragment size/extent, neighboring face evidence, or CC3D boundaries throughout a suspect ABISS object. Consequently, a true `multi_anchor` component can be misreported as `one_anchor`, producing wrong candidate precision and oracle headroom. The plan also departs from the explicit instruction to stream all 726 chunks. Candidate membership must be generated from GT-free dense-population evidence before evaluator ownership is overlaid; skeleton samples alone may support partition controls, not the claimed deployable candidate families.

3. [major] **Phase 1 artifacts cannot be reused in Phase 2.** Node-bearing chunk selection and extraction are informed by test skeletons. Processing-order invariance does not remove that provenance. Reusing those artifacts under `gt_free/` violates the firewall even if their array contents happen not to contain node identities. Only code and deterministic algorithms may be shared; Phase 2 artifacts must be regenerated solely from GT-free inputs.

4. [major] **The tier10 fidelity gate is not a reproduction.** Re-aggregating existing CSVs and independently recomputing only two chunks does not reproduce the recorded ten-chunk score. The new implementation must decode and evaluate all ten chunks end-to-end and reproduce `0.775832` within the established tolerance.

   The two-mask division is defensible—tier10 tissue-only for historical reproduction and the ABISS tissue+border keep-mask for fusion—but it leaves the main fusion CC3D partition unvalidated. Add a paired ten-chunk bridge control using the fusion mask and report its score, coverage, and candidate-topology differences. Otherwise any change may be caused by the mask rather than the intended fusion evidence.

5. [major] **The deployable policies are not actually predeclared.** No numerical definitions are supplied for dust/mid/anchor bands, overlap support, affinity floor, persistence, size ratio, winner margin, nearby robustness settings, or morphology/nuclei/glia vetoes. Their data sources are also unspecified. Because Phase 3 occurs after evaluator-only Phase 1, selecting these values then would be test-informed. All numerical policy variants, anchor definitions, suspect-detector inputs, and veto rules must be frozen before the first evaluator result is viewed.

6. [major] **The one-anchor resolver is asserted but not completely specified or tested.** A sequential union-find can accept the first of two conflicting anchor claims and reject only the second, despite the plan saying conflicting claims abstain. Define batch-global resolution over immutable original anchors, including what happens to every edge in a connected conflict subgraph. Add resolver-specific permutation tests covering:

   - one fragment claimed by two anchors;
   - direct and transitive anchor bridges;
   - cycles and duplicated edges;
   - inconsistent hosts across chunks;
   - confirmation that assigned fragments never become anchors or relays.

   Existing shard-order and face-order tests do not establish policy-resolution order independence.

7. [major] **Several Phase 0 identity checks remain incomplete or ambiguous.** The manifest must explicitly enumerate actual per-file channel count, dtype, and shape for all 726 chunks, not only sample probes. It must also state ABISS shape, dtype, coordinate origin, parameter/log identity, and expected NERL explicitly. “Displaced controls must fail” is not a testable criterion unless the plan identifies discriminative probe values or mask transitions; adjacent voxels can legitimately have identical labels. Exact observable assertions are needed to prevent an axis or origin error from passing.

8. [major] **Required conclusions and deliverables are not fully routed.** The plan does not explicitly create the required representative visual-probe manifests, and `evaluation_gt/results.json`/`results.md` are only implied. The final report should answer primary questions 1–6 in that order, including on a Gate-A negative path. It must specifically cross-tab CC3D disagreement against the three ERL-visible false merges and attribute residual cases to morphology, endpoint linking, nuclei/glia quarantine, or later gap completion.

9. [major] **The prior correction-map composition lacks an identity and safety gate.** Before scoring `frozen_endpoint_merges.npz`, the plan must prove it uses the identical raw ABISS label namespace and substrate, define conflict semantics for both composition orders, and recheck the one-anchor invariant after composition. Otherwise the composed score can silently represent incompatible labels or an anchor-anchor merge.

10. [minor] **Byte-identical NPZ output is not a portable determinism test.** NPZ container metadata can vary even when arrays are identical. Compare canonical decoded arrays plus stable content hashes, or specify a deterministic writer. Shard writes and `.done` publication should also use temporary files followed by atomic rename.

## Questions

No external decision is required. The planner can resolve these findings by separating provenance domains, adding a GT-free candidate-generation path, and freezing exact policy definitions before evaluation.

## Verdict

VERDICT: NEEDS_CHANGES

## Summary of findings

The affinity-tag resolution is evidence-based, and the overall staged architecture is appropriate. Revision is required because the current Phase 0 and reuse path breach the GT firewall, the node-only audit cannot support Gate A or one-anchor counts, the actual fusion-mask CC3D path lacks a fidelity bridge, and the policy/invariant/evaluation contracts are not yet precise enough to prevent test selection or incorrect numbers.

READY: no