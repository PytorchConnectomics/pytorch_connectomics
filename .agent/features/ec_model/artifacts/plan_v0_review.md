# Plan v0 Review

## Summary

Codex (coder role, `codex exec --sandbox read-only`) reviewed plan_v0 using only the prompt contents. It judged the structure clear and liked that mask evaluation is kept separate from NERL claims. It is not ready to implement, with 8 major findings: val-half-B isolation, seed-swap confounded with owner-mask changes, majority-GT fragment truth, evaluation sampling and zero-denominator rules, no-edit preservation, mining edge-case rules, over-specified real-data pass criteria, and the code-review delivery contract for ignored files. Raw reviewer output: `state/plan_v0_review.review.raw.md` (READY: no). The reviewer did not verify referenced sources.

## Findings

- [major] Val half B is not isolated from training decisions. Periodic validation uses an unspecified val subset. Fix: define the neuron-disjoint A/B partition before training; restrict monitoring, checkpoint selection and τ calibration to A; evaluate a frozen checkpoint once on B. Keep neurons in targets or candidate fragments from crossing the partition, or document the remaining dependence.
- [major] Seed-swap can measure owner-mask dependence instead of seed dependence, because moving the seed also changes M. Fix: paired tests that hold EM, M and A fixed and change only P and the target. Require adequate target Dice/recall together with low pair IoU; empty predictions must not pass.
- [major] Majority-GT fragment truth can count contaminated joins as correct, including contamination outside the write region. Fix: define fragment purity and how mixed or under-annotated fragments are handled. Report mixed-fragment proposals separately, and do not read majority-based precision as meeting the join-economics bar.
- [major] Evaluation sampling and undefined metrics are unspecified. Jitter can push sites to the write-region boundary, and crops or bins may have empty denominators. Fix: freeze eval crops and seeds; require the site to be measurable inside the write region; define aggregation and zero-denominator handling; set minimum support for a gate verdict.
- [major] No-edit accuracy does not check preservation: an empty prediction passes. Fix: require retained target/owner coverage, define the empty-mask case, and report preservation, unwanted expansion and unwanted joins separately.
- [major] Mining and sampling need executable edge-case rules. "Maximal run" is ambiguous on branching skeletons, contamination sites may lack a majority seed within the window, and sampling buckets may be empty. Fix: define graph traversal, branch/boundary handling, seed eligibility, duplicate handling, and bounded resampling with a documented fallback mixture, plus synthetic tests.
- [major] Real-data pass criteria assume an unestablished distribution: the 4–8% zero-node band and "every site type in every ROI". Fix: treat these as diagnostics. Use artifact alignment, label consistency, independently checked examples and sufficient eligible samples as the correctness and readiness criteria.
- [major] Ignored implementation files lack a concrete review-delivery contract. Fix: the code-stage submission must carry complete source contents or an explicit baseline-relative patch for every new source file, plus exact commands, results and artifact provenance.

## Questions

- Is GT-assisted seed construction intentionally an oracle feasibility experiment? If so, label the limitation and keep those results separate from performance with deployment-available seeds.

## Verdict

VERDICT: NEEDS_CHANGES
