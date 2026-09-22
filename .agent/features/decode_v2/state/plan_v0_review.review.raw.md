The overall scope is appropriate, but the load-bearing relink design is not yet sound or sufficiently verifiable.

- [major] Direct forbidden-edge deletion does not guarantee separation. Union-find can reconnect opposing sides through `A–X–B`, another split run, or an unsplit section. Merely detecting that `(c) ≈ (b)` afterward does not satisfy the required relink invariant. V0 should use component-level cannot-link constraints, with split pieces first anchored to their incoming seed identities and unions rejected whenever components contain opposing anchors.

- [major] The same-axon recovery claim contradicts the proposed constraint. Every detected fusion receives A/B sides regardless of GT identity. If both sides can reconnect through the “same tube,” that is exactly the indirect path that also defeats true-axon separation. Conversely, a correctly enforced constrained union-find will preserve false-positive splits. Given roughly 539 candidates but only about 48 true fusions, the plan needs an explicit eligibility rule—such as independently traced upstream/downstream tubes—or another hard-versus-soft constraint policy.

- [major] Side identity is underspecified. `A_ids`/`B_ids` must use globally unique graph-node identities such as `(z, label)`, not potentially slice-local label values. They must include the original incoming seed anchors, propagated pieces, and any terminal anchors, and remain valid through relabeling and `merge_id`.

- [major] Fusion-run ownership is ambiguous. Processing “each fusion entry” while propagating can revisit and overwrite sections already handled by an earlier run, leaving stale side sets. The plan needs deterministic run-head selection, processed-section tracking, overlap/conflict handling, and ordering.

- [major] N-way handling is inconsistent. Area matching sums all incoming sections, but only the two largest are split and constrained. Either restrict candidates to exactly two incoming sections and recompute the area test for that pair, or define genuine N-way watershed and pairwise incompatibility behavior.

- [major] The acceptance rule contradicts the task. The plan calls `(c) < 0.593` acceptable when documented, while the task explicitly requires no regression. Such a result must trigger another iteration or a safe default/fallback; it cannot be declared successful merely because `(c)` beats `(b)`.

- [major] The evaluation deliverable is incomplete. The task requires `decode_v2.py` itself to print real NERL, oracle-merge NERL, and instance metrics, but the plan delegates NERL to the SLURM wrapper. It also does not explain how all three ablations receive real NERL without retaining their segmentations or invoking evaluation directly. Exact default/tagged output names and `--zslice` output isolation should be defined.

- [major] Verification lacks a focused test of the new invariant. Add deterministic graph-level cases covering direct and indirect A/B paths, seed anchoring, multiple interacting runs, union ordering, repeated slice-local labels, and allowed same-side unions. A 96-slice smoke test with few fusions cannot validate the central relink behavior.

- [minor] Validation commands should explicitly activate or use the `pytc` environment, and the SLURM wrapper should identify the exact generated path passed to both evaluation commands.

- [minor] “Not much worse,” “closer to 62,” and “instances stay ~2625” are not executable thresholds. Either make them reporting-only or assign concrete bounds.

For the reviewer’s question: constrained union-find should ship in v0; direct forbidden-edge deletion is suitable only as an ablation. It must be seed-anchored and paired with fusion eligibility, because a naive hard constraint across all 539 candidates would likely convert same-axon detections into persistent false splits.

Findings tagged [minor]/[major] above.
READY: no