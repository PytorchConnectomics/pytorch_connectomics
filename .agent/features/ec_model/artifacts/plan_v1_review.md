# Plan v1 Review

## Summary

Codex (coder role, `codex exec --sandbox read-only`) reviewed plan_v1 against plan_v0 and the plan_v0 review, using only the prompt contents. The first attempt crashed at startup (tokio `failed to spawn thread`, EAGAIN at the 400-thread user limit on the login node; log kept at `state/plan_v1_review.codex.attempt1_panic.log`). The retry with `TOKIO_WORKER_THREADS=4` succeeded and is the raw transcript `state/plan_v1_review.review.raw.md` (READY: no).

The reviewer says v1 substantially improves the plan: oracle framing is explicit, seed-only swaps remove the owner-mask confound, and preservation rejects empty masks. Prior-finding status: #2, #5, #8 resolved; #1, #3, #4, #6, #7 partially resolved. There are 8 major findings, on fragment-classification completeness, success-dependent support, B-guard bypass, unspecified evaluation choices, alignment exclusions, mining/sampler edge cases, per-type audit counts, and overfit pair coverage.

## Findings

- [major] Fragment purity is not a complete classification. Two GT ids cannot each exceed 99%, so a 60/40 fragment with no other-GT skeleton node fails both `pure_k` and `mixed`. Precedence for mostly-background fragments is unclear. Fix: order the rules (nz=0/insufficient annotation → unique dominant GT → purity → everything else mixed), specify precedence, and test 60/40, 99/1, all-background and overlapping cases.
- [major] Support depends on model success. Requiring ≥30 correct joins means a zero-join model on abundant data can never FAIL or KILL. Fix: base support on eligible sites, nodes, positive/negative candidate opportunities and pairs. Define how insufficient predictions affect precision, the verdict precedence, and the kill outcome when no τ reaches recall ≥0.10. Test a zero-output model on supported data.
- [major] The B guard permits repeated held-out development: a renamed frozen file or another checkpoint bypasses it. Fix: key the guard to the experiment and B manifest; freeze checkpoint content hash, τ, preprocessing/config and eval-site hashes before B access. An override must invalidate the original confirmatory verdict. Test renamed files and different checkpoints.
- [major] The evaluation contract leaves consequential choices unspecified: candidate radius for multi-node/mixed fragments, the τ-selection objective and tie-breaker, whether repeated fragments/nodes across overlapping crops are distinct opportunities or unique entities, and bootstrap treatment of repeats and two-neuron swap pairs. Fix: specify all of these before training, with support counts in the same units.
- [major] The alignment check accepts ≤2% node/voxel GT disagreement without defining exclusions, although mining and metrics use those identities as truth, and mismatches may concentrate in thin structures. Fix: require agreement for participating nodes, or exclude and count mismatches before mining and evaluation (by radius and split) with defined graph handling.
- [major] Mining and sampling are still undefined for valid edge cases: singleton zero-label components (no internal length for a centroid), boundary-free components, tied choices, and sampler buckets (or the whole sampler) emptying after invalid-site removal. Fix: deterministic rules, recompute active weights after removal, readiness failure when no eligible samples remain, focused synthetic tests.
- [major] The real-data audit reintroduces mandatory per-type abundance (50 re-derivations and ≥12 montages per type). Fix: use `min(available, requested)`, report absent types, keep readiness minima only for the training and evaluation populations, and do not treat listed montage paths as a completed human audit.
- [major] The overfit seed-dependence test does not guarantee both pair members were trained, so it tests generalization and could wrongly block smoke training. Fix: include both members of the 8 designated pairs in the fixed overfit set, reconcile with the ≥16-thin requirement, and keep any unseen-pair check as a diagnostic.

## Questions

- None.

## Verdict

VERDICT: NEEDS_CHANGES
