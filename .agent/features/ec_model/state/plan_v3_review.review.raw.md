## Summary
Plan v3 resolves four prior findings and substantially improves dependency grouping, but it still separates some observations sharing the same neuron. Calling the resulting intervals descriptive does not establish their statistical validity. The implementation scope and verification plan are otherwise concrete; two evaluation contracts need clarification.

## Prior Findings Status
- Finding 1 — Bootstrap dependence: partially resolved — overlapping seed pairs and repeated fragment IDs are grouped correctly, but different candidate fragments belonging to the same non-seeded neuron can remain in separate groups.
- Finding 2 — B access reservation: resolved — exclusive reservation precedes inference, crashes consume access, and concurrency/failure tests are specified.
- Finding 3 — Seed-swap owner consistency: resolved — separate seed-fragment and conditioning-owner fields support per-member validation.
- Finding 4 — Optional overfit populations: resolved — deterministic replacements and explicit prerequisite outcomes handle missing populations.
- Finding 5 — Mining inconsistencies: resolved — split anchors use the lower-index endpoint, and contamination tests require the whole-crop fallback.

## Findings
- [major] **Dependency groups still omit shared candidate-neuron dependence.** In “Gate evaluator → Aggregation,” a site seeded by neuron 1 with candidate fragment 101 belonging to neuron 100 and a site seeded by neuron 2 with candidate fragment 102 belonging to neuron 100 need not be grouped. They share neither a seeded neuron nor a fragment ID, yet share neuron 100. This affects join metrics, beyond the documented wrong-neuron voxel limitation. **Fix:** include all GT neuron IDs represented in evaluated candidate fragments when forming groups, and include relevant wrong-inclusion neurons for intervals on that metric. Alternatively, omit confidence intervals entirely for this experiment. Add a test using distinct candidate fragments of the same non-seeded neuron.

- [minor] **The metric meaning of `owner` is ambiguous after introducing `conditioning_owner`.** Seed-swap members have `conditioning_owner=0`, but join candidates exclude `owner` and missed-node recovery compares labels against `owner`. Using zero would count the seed fragment itself as a join candidate. **Fix:** explicitly reserve swaps for the seed-dependence metric, or define metric ownership as `seed_fragment_label` and specify their inclusion in pooled metrics and support counts.

- [minor] **Deployment-like seeds conflict with the universal label-consistency assertion.** The separate diagnostic deliberately removes GT intersection, so a contamination example’s seed can belong to another neuron. “100% of manifest examples” nevertheless requires every seed voxel to equal the target GT. **Fix:** scope that assertion to trusted-seed examples; validate fragment membership for deployment-like duplicates and report seed/target mismatches without rejecting them.

## Questions
- None.

READY: no