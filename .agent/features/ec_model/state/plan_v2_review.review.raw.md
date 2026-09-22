## Summary

V2 resolves most prior blockers, including exhaustive purity classes, prediction-independent support, alignment exclusions, and trained-pair coverage. The remaining scientific issue is the bootstrap: assigning pairs to the smaller GT id does not preserve dependence between pairs sharing a neuron. Several smaller contradictions can be resolved during implementation.

## Prior Findings Status

- **1. Fragment-classification completeness: resolved** — Ordered rules cover all compositions, define background precedence, and include boundary-case tests.
- **2. Success-dependent support: resolved** — Eligibility determines support; zero-output models receive `KILL` on supported data.
- **3. B-guard bypass: resolved** — Renamed frozen files and different checkpoints cannot reset access for the same manifest; overrides invalidate the original verdict.
- **4. Unspecified evaluation choices: partially resolved** — Radius, threshold selection, repeated-opportunity units, and support units are explicit. The specified bootstrap still splits dependent observations across clusters.
- **5. Alignment exclusions: resolved** — Misaligned nodes and incident edges are removed before downstream use, with counts and assertions.
- **6. Mining/sampler edge cases: resolved** — Singleton and boundary-free components, deterministic choices, redistribution, and exhaustion have executable rules and tests.
- **7. Per-type audit abundance: resolved** — Audits use available counts, absent types are reported, and generated montages are distinguished from human review.
- **8. Overfit pair coverage: resolved** — Both members of eight designated pairs are trained, and the thin-example minimum is reconciled. Unseen pairs are diagnostic only.

## Findings

- **[major] The bootstrap clusters do not preserve shared-neuron dependence.** In **Gate evaluator → Aggregation**, pairs `(1, 100)` and `(2, 100)` enter different clusters despite sharing neuron 100; ordinary sites seeded by neuron 100 enter a third cluster. Repeated candidate fragments can likewise contribute correlated observations under different seeded-neuron clusters. Consequently, the reported 95% intervals are not justified by the stated clustering scheme. **Fix:** freeze dependency groups that keep shared contributing neurons/fragments and both pair members together, then bootstrap those groups. Report the number of independent groups; when too few remain, report descriptive metrics without a confidence interval. Add a test with overlapping pairs and a repeated candidate across different seeded neurons.

- **[minor] Reserve B access before inference, including failed attempts.** **Confirmatory B guard** says the first evaluation atomically creates a record containing the verdict, but does not distinguish reservation from completion. **Fix:** exclusively create a `STARTED` record before loading evaluation data or running inference, then finalize it with the verdict. A crash must retain consumed access. Test concurrent invocations and failure after reservation.

- **[minor] Seed-swap examples need an explicit exception to owner-label consistency.** **Real-data correctness → Label consistency** requires every seed to belong to the recorded owner, while seed swaps change neurons with `M=0`. **Fix:** distinguish each member’s seed-fragment label from the conditioning owner field, or explicitly mark swap examples as owner-free. Validate seed-fragment/GT consistency for each member.

- **[minor] Overfit construction still assumes optional populations exist.** **Verification Plan → Overfit test** requires contamination/control coverage and eight thin-site swap pairs, while readiness guarantees only thin gap/split counts. **Fix:** use available contamination/control examples and deterministic replacements when absent. Separately check availability of eight eligible training pairs; report that prerequisite failure directly rather than calling it failed model learning.

- **[minor] Two mining statements need reconciliation.** “The edge midpoint node (u)” does not uniquely specify whether a split anchor is the geometric midpoint or endpoint `u`. The contamination test also calls a seed outside the initial window unseedable despite the whole-crop fallback. **Fix:** choose one split-anchor convention and test it; require successful fallback for a dominant seed elsewhere inside the crop, and unseedability only when absent from the entire crop.

## Questions

- None.

READY: no