## Summary
The plan has a clear implementation structure and appropriately separates mask evaluation from whole-volume NERL claims. It is not ready to implement: validation isolation, seed-dependence testing, fragment truth, and several acceptance criteria need correction. This review covers only the supplied artifacts; referenced source code, data, and reports were not verified.

## Findings

- [major] **Val half B is not isolated from training decisions.** Periodic validation uses an unspecified val subset, while half B is described as held out. Define the neuron-disjoint partition before training, restrict monitoring, checkpoint selection, and τ calibration to half A, and evaluate a frozen checkpoint once on B. Prevent neurons occurring in either crop’s targets or candidate fragments from crossing the partition, or explicitly document the remaining dependence.

- [major] **Seed-swap testing can measure owner-mask dependence instead of seed dependence.** Moving the seed to another neuron changes M under the channel contract; predictions could differ while ignoring P entirely. Add paired tests holding EM, M, and A fixed while changing only P and the corresponding target. Require adequate target Dice or recall alongside low prediction-pair IoU; low IoU alone can reward failed or nearly empty predictions.

- [major] **Majority-GT fragment truth can count contaminated joins as correct.** A fragment whose majority is neuron k can still contain another neuron, including outside the write region. Define fragment purity and how mixed or insufficiently annotated fragments are treated. Report mixed-fragment proposals separately and avoid interpreting majority-based local precision as satisfying the merge-safe join economics threshold.

- [major] **Evaluation sampling and undefined metrics are unspecified.** Jitter can move sites to the write-region boundary, and individual crops or radius bins may have no missed nodes, candidates, positive candidates, or predicted foreground. Freeze evaluation crops and seeds, require the intended site to be measurable within the write region, define aggregation and zero-denominator handling, and set minimum support for a gate verdict. Otherwise an empty or tiny eligible sample can produce a misleading pass.

- [major] **No-edit accuracy does not check preservation.** An empty prediction fires no joins and could satisfy the wrong-neuron criterion, despite deleting the seeded neuron. Require retained target/owner coverage and explicitly define the empty-mask case. Separate preservation, unwanted expansion, and unwanted joins in the report.

- [major] **Mining and sampling need executable edge-case rules.** “Maximal run” is ambiguous on branching skeletons; contamination sites may have no majority-neuron seed within the permitted search window; requested sampling buckets may be empty. Define graph traversal, branch/boundary handling, seed eligibility, duplicate-site handling, and bounded resampling with a documented fallback mixture. Add synthetic tests for these cases.

- [major] **The real-data pass criteria assume an unestablished distribution.** Requiring 4–8% zero-labelled nodes based on a different whole-volume reference, and every site type in every ROI, can reject correct curation or incentivize changing it to match expectations. Treat these as diagnostics. Use artifact alignment, label consistency, independently checked examples, and sufficient eligible samples as correctness and readiness criteria.

- [major] **Ignored implementation files lack a concrete review-delivery contract.** An unchanged tracked worktree and a list of ignored paths do not make the implementation reviewable, especially under artifact-only review. Require complete source contents or an explicit baseline-relative patch covering every new source file, plus exact commands, results, and generated-artifact provenance in the code-stage submission.

## Questions
- Is the GT-assisted seed construction intentionally an oracle feasibility experiment? If so, label that limitation explicitly and distinguish its results from performance with seeds available during deployment.

READY: no