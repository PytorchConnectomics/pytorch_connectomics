# Plan v3 Review

## Summary

Codex (coder role, `codex exec --sandbox read-only`, `TOKIO_WORKER_THREADS=4`) reviewed plan_v3, the extra planning round the user authorized, against the plan_v2 review, using only the prompt contents.

The first attempt hung: one line of output, no tool calls, killed by the 1800 s timeout, no transcript. Its log is kept at `state/plan_v3_review.codex.attempt1_hang.log`. The single retry succeeded, and its output is the raw transcript `state/plan_v3_review.review.raw.md` (READY: no).

Prior findings: #2 (B reservation), #3 (seed-swap owner fields), #4 (overfit populations) and #5 (mining statements) are resolved. #1 (bootstrap dependence) is partially resolved: overlapping pairs and repeated fragment ids now group correctly, but distinct candidate fragments of the same non-seeded neuron can still fall in different groups.

This leaves one major and two minor findings. All plan rounds (3) are used and mode is normal, so the unresolved major finding blocks the run for a human decision.

## Findings

- [major] Dependency groups still omit shared candidate-neuron dependence. A site seeded by neuron 1 with candidate fragment 101 (neuron 100) and a site seeded by neuron 2 with candidate fragment 102 (neuron 100) share neither seed nor fragment id, yet both depend on neuron 100. This affects join metrics as well as the documented voxel wrong-inclusion limitation. Reviewer fix: include every GT neuron id represented in evaluated candidate fragments (and relevant wrong-inclusion neurons for that metric) when forming groups, **or omit confidence intervals entirely for this experiment**. Add a test with distinct candidate fragments of the same non-seeded neuron.
- [minor] Metric meaning of `owner` is ambiguous after introducing `conditioning_owner`. For swap members `conditioning_owner=0`, so the seed fragment itself could count as a join candidate and missed-node recovery compares against 0. Fix: reserve swaps for the seed-dependence metric only, or define metric ownership as `seed_fragment_label` and state inclusion in pooled metrics and support counts.
- [minor] Deployment-like seeds conflict with the universal label-consistency assertion. Those seeds deliberately skip GT intersection, so a contamination example's seed can lie on another neuron. Fix: scope the 100% assertion to trusted-seed examples; for deployment-like duplicates, validate fragment membership and report seed/target mismatches without rejecting.

## Questions

- None from the reviewer.
- Coordinator notes for the human decision:
  1. **Recommended resolution of the major finding: omit confidence intervals entirely** (the reviewer's stated alternative). No gate or verdict uses CIs; the eligibility-based support minima already bound sample size; and the dependence structure in dense tissue has now defeated two grouping schemes. Report point estimates, opportunity counts and unique-entity counts only.
  2. Minor fixes, both mechanical:
     - swap members appear only in the seed-dependence metric (excluded from pooled join/recovery/inclusion metrics and their support counts), and metric ownership elsewhere is `seed_fragment_label`;
     - the label-consistency assertion is scoped to trusted-seed examples, with deployment-like duplicates validated for fragment membership and mismatches reported.
  3. **Environment:** another session's CCC run (`.agent/features/embedding_model`) is running Codex `code_v0` with workspace-write in this same checkout. Tracked files now differ from this run's clean baseline (`M connectomics/models/losses/build.py`, `M connectomics/models/losses/metadata.py`, and new `connectomics/models/losses/embedding.py` and unit tests). That will break plan_v3 Verification step 8 (`git status` equals run-start status) and pollute this run's code-review diff and mutation guard. It needs either a separate worktree for this run's code stage or a baseline rule scoped to `dev/ec_model/`.

## Verdict

VERDICT: NEEDS_CHANGES
