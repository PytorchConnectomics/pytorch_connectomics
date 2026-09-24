# Plan v0 Review

## Summary

Reviewer: codex (plan-code=claude-codex, coder owns plan review). Raw transcript:
`state/plan_v0_review.review.raw.md`.

The reviewer accepts the staged architecture and confirms the `arm0_win144` resolution is
argued from evidence rather than convenience. It rejects the plan as not executable as
written, on four structural grounds: Phase 0 crosses the GT firewall, the node-sampled
Phase 1 cannot construct the directional candidate families or Gate A, the fusion-mask CC3D
path has no fidelity bridge, and the policy/resolver/evaluation contracts are too loose to
prevent test-informed selection or a silently wrong number.

Nine findings are tagged major and one minor. The coordinator judges all ten material and
none of them softened here.

## Findings

1. [major] Phase 0 violates the GT-free/evaluator-only separation. `phase0_input_audit.py`
   writes `gt_free/input_manifest.json` while also reproducing ABISS NERL from skeleton LUTs
   and re-aggregating human-GT tier10 results; a GT-free stage may read neither. Split into a
   GT-free identity manifest written before any decoding, and evaluator-only fidelity results
   under `evaluation_gt/`. The expected NERL may appear in the manifest as a constant, but the
   measured reproduction and the human-GT tier10 result must stay evaluator-only.

2. [major] The proposed Phase 1 data is insufficient for its own directional oracles and for
   Gate A. Sampling CC3D only at skeleton nodes cannot determine all ABISS overlaps inside a
   component, whether an unseen established anchor is present, global fragment size/extent,
   neighbouring face evidence, or CC3D boundaries throughout a suspect ABISS object. A true
   `multi_anchor` component can therefore be misreported as `one_anchor`, producing wrong
   candidate precision and wrong oracle headroom. It also departs from the task's explicit
   instruction to stream all 726 chunks. Candidate membership must be generated from GT-free
   dense-population evidence before evaluator ownership is overlaid; skeleton samples support
   the partition controls only.

3. [major] Phase 1 artifacts cannot be reused in Phase 2. Node-bearing chunk selection is
   informed by test skeletons, and processing-order invariance does not remove that
   provenance. Reusing those artifacts under `gt_free/` breaches the firewall even if the
   array contents contain no node identities. Only code and deterministic algorithms may be
   shared; Phase 2 artifacts must be regenerated from GT-free inputs alone.

4. [major] The tier10 fidelity gate is not a reproduction. Re-aggregating existing CSVs and
   recomputing only two chunks does not reproduce the recorded ten-chunk score; the new
   implementation must decode and evaluate all ten chunks end-to-end and reproduce 0.775832
   within the established tolerance. Separately, the two-mask division is defensible
   (tier10 tissue-only for historical reproduction, ABISS tissue+border for fusion) but it
   leaves the actual fusion partition unvalidated: add a paired ten-chunk bridge control under
   the fusion mask and report its score, coverage, and candidate-topology differences, or a
   later change cannot be attributed to fusion rather than to the mask.

5. [major] The deployable policies are not actually predeclared. No numerical definitions are
   given for dust/mid/anchor bands, overlap support, affinity floor, persistence, size ratio,
   winner margin, nearby-robustness settings, or morphology/nuclei/glia vetoes, and their data
   sources are unspecified. Because Phase 3 runs after the evaluator-only Phase 1, choosing
   these values then would be test-informed. Every numerical variant, anchor definition,
   suspect-detector input, and veto rule must be frozen before the first evaluator result is
   viewed.

6. [major] The one-anchor resolver is asserted but neither fully specified nor tested. A
   sequential union-find can accept the first of two conflicting anchor claims and reject only
   the second, contradicting the plan's "conflicting claims abstain". Define batch-global
   resolution over immutable original anchors, including the fate of every edge in a connected
   conflict subgraph, and add resolver-specific permutation tests for: one fragment claimed by
   two anchors; direct and transitive anchor bridges; cycles and duplicated edges; inconsistent
   hosts across chunks; and confirmation that assigned fragments never become anchors or
   relays. Shard-order and face-order tests do not establish policy-resolution order
   independence.

7. [major] Several Phase 0 identity checks are incomplete or ambiguous. The manifest must
   enumerate actual per-file channel count, dtype, and shape for all 726 chunks, not sample
   probes, and must state ABISS shape, dtype, coordinate origin, parameter/log identity, and
   expected NERL explicitly. "Displaced controls must fail" is not a testable criterion unless
   discriminative probe values or mask transitions are identified — adjacent voxels can
   legitimately carry identical labels. Exact observable assertions are needed so an axis or
   origin error cannot pass.

8. [major] Required conclusions and deliverables are not fully routed. The plan does not
   create the required representative visual-probe manifests, and `evaluation_gt/results.json`
   and `results.md` are only implied. The final report must answer primary questions 1–6 in
   order, including on a Gate-A negative path, and must cross-tab CC3D disagreement against the
   three ERL-visible false merges and attribute residual cases to morphology, endpoint linking,
   nuclei/glia quarantine, or later gap completion.

9. [major] The prior correction-map composition lacks an identity and safety gate. Before
   scoring `frozen_endpoint_merges.npz` the plan must prove it uses the identical raw ABISS
   label namespace and substrate, define conflict semantics for both composition orders, and
   recheck the one-anchor invariant after composition; otherwise the composed score can
   silently represent incompatible labels or an anchor-anchor merge.

10. [minor] Byte-identical NPZ output is not a portable determinism test — container metadata
    can vary while arrays are identical. Compare canonical decoded arrays plus stable content
    hashes, or specify a deterministic writer. Shard writes and `.done` publication should also
    use temporary files followed by atomic rename.

## Questions

The reviewer raised no question requiring an external decision. It states the planner can
resolve all findings by separating provenance domains, adding a GT-free candidate-generation
path, and freezing exact policy definitions before evaluation.

Coordinator note, recorded for the next plan version rather than as a reviewer question: since
this review was produced, a whole-volume dense CC3D@0.70 store for this exact affinity has
been launched (SLURM array 2855306, `…grid1008_halo72.cc3d_t070_tissue.chunks`, 726 chunks,
tier10 tissue-mask recipe). That removes the I/O argument that motivated the node-sampled
Phase 1 and makes the reviewer's demand for full-population GT-free candidate generation
cheap, so finding 2 and finding 3 can be resolved by construction rather than by compromise.

## Verdict

VERDICT: NEEDS_CHANGES
