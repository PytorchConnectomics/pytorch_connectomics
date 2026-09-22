## Summary

V1 substantially improves the plan: oracle framing is explicit, seed-only comparisons remove the owner-mask confound, and preservation controls reject empty masks. However, fragment classification and gate-verdict rules still contain correctness gaps, and several verification requirements remain contradictory or incomplete. The plan is not ready for implementation. This review evaluates only the supplied artifacts; referenced source claims remain unverified.

## Prior Findings Status

- **Finding 1: partially resolved** — Spatial separation, A-only development, and documented neuron overlap address the main concern. The B guard protects a frozen-file identity rather than the held-out evaluation itself.
- **Finding 2: resolved** — Paired inputs hold EM/M/A fixed, and own-target recall prevents empty predictions from passing.
- **Finding 3: partially resolved** — Whole-ROI histograms and mixed-fragment penalties improve truth assignment, but the purity classes are neither exhaustive nor mutually exclusive as written.
- **Finding 4: partially resolved** — Frozen crops, zero-denominator rules, and support requirements are added. Prediction-dependent support, unspecified candidate radii, and incomplete threshold-selection rules still make verdicts ambiguous.
- **Finding 5: resolved** — Preservation and expansion are measured explicitly, and empty control predictions fail.
- **Finding 6: partially resolved** — Graph traversal and resampling are clearer, but degenerate graph components and sampler exhaustion lack executable rules.
- **Finding 7: partially resolved** — Distribution bands are now diagnostics, but fixed per-type audit counts still fail when legitimate types are absent or scarce. Accepted node/GT mismatches also lack a handling policy.
- **Finding 8: resolved** — Complete source patches, hashes, command results, provenance, and an explicit prompt-size limit provide a concrete review-delivery contract.

## Findings

- [major] **Fragment purity is not a complete classification.** In “Fragment purity,” two GT ids cannot each exceed 99% of nonzero voxels. A 60/40 fragment without other-GT skeleton evidence can therefore fail both `pure_k` and the stated `mixed` rule. Mostly-background fragments can also satisfy the skeleton-based mixed rule, leaving precedence unclear. Define classification in order: handle `nz=0`/insufficient annotation, identify a unique dominant GT, apply purity requirements, then classify every remaining annotated fragment as mixed. Specify precedence and test 60/40, 99/1, all-background, and overlapping-condition cases.

- [major] **Support depends on model success, preventing meaningful failure verdicts.** “Gate verdict” requires ≥30 correct joins before any verdict other than `INSUFFICIENT_SUPPORT`. A model making zero joins on abundant eligible data consequently cannot fail or be killed. Base dataset support on eligible sites, target nodes, positive/negative candidate opportunities, and pairs; separately specify how insufficient predictions affect precision and verdicts. Define verdict precedence and the kill outcome when no τ achieves recall ≥0.10. Add tests for a zero-output model on adequately supported data.

- [major] **The B guard permits repeated held-out development.** It refuses only a second run for the same frozen file; creating another file or choosing another checkpoint bypasses the restriction. Key the guard to the experiment and B dataset manifest, and freeze checkpoint content hash, τ, preprocessing/configuration, and evaluation-site hashes before B access. Any override should explicitly invalidate the original confirmatory verdict. Test attempted reruns using renamed files and different checkpoints.

- [major] **The evaluation contract still leaves consequential choices unspecified.** “Gate evaluator” bins joins by *candidate radius* without defining that radius for multi-node or mixed fragments. It also does not define the τ-selection objective/tie-breaker, or whether repeated fragments/nodes across overlapping crops count as distinct proposal opportunities or unique biological entities. Specify these choices before training, including bootstrap treatment of repeated observations and two-neuron seed-swap pairs. Support counts must use the same declared units.

- [major] **The alignment check permits inconsistent supervision without defining exclusions.** “Real-data correctness” accepts up to 2% disagreement between skeleton GT ids and voxel GT, yet mining and node metrics subsequently use those identities as truth. Those mismatches can concentrate in the thin structures being measured. Require agreement for all participating nodes, or explicitly exclude and count mismatches before mining and evaluation, with defined graph handling. Report exclusions by radius and split.

- [major] **Mining and sampling remain undefined for valid edge cases.** A singleton zero-label component has no internal edge length, so its length-weighted centroid needs an explicit fallback. Boundary-free components and tied choices also need deterministic rules. Additionally, removing invalid sites during sampling can empty a previously nonempty bucket—or the entire sampler—after initial redistribution. Define these cases, recompute active weights after removal, and terminate with a readiness failure when no eligible samples remain. Add focused synthetic tests.

- [major] **The real-data audit reintroduces mandatory per-type abundance.** Requiring 50 random sites and ≥12 montages per type cannot pass when a legitimate type is absent or has fewer examples, despite per-type counts now being diagnostic. Audit `min(available, requested)` examples, report absent types explicitly, and retain smoke-readiness minima only for the required training/evaluation populations. Clarify that listing montage paths alone does not establish completion of a human visual audit.

- [major] **The overfit seed-dependence test does not guarantee both targets were trained.** The 32 fixed examples need not include paired swaps, while the subsequent test creates eight swaps from their crops and demands successful alternate-neuron predictions. That tests generalization as well as fitting and can incorrectly block smoke training. Include both members of the designated eight pairs in the fixed overfit set, reconcile their count with the ≥16 thin-example requirement, and distinguish any unseen-pair diagnostic from the overfit pass criterion.

## Questions

- None.

READY: no