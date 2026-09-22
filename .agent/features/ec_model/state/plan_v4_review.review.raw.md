## Summary
Plan v4 resolves all three prior review findings through explicit metric rules and corresponding tests. The detached worktree and package import checks address the coordinator’s shared-checkout concern. The plan is implementable and verifiable; one minor clarification remains in the filesystem isolation contract.

## Prior Findings Status
- Finding 1 — shared candidate-neuron dependence: resolved — confidence intervals and bootstrap grouping are removed entirely, adopting the reviewer’s explicit alternative. Reports retain point estimates and opportunity and unique-entity counts.
- Finding 2 — ambiguous metric ownership: resolved — swaps enter only metric 5; metrics 1–4 consistently use `seed_fragment_label`, with explicit exclusions from calibration and support counts.
- Finding 3 — deployment-like seed assertions: resolved — trusted seeds require target consistency; deployment-like seeds require fragment membership and report target mismatches without rejection.
- Coordinator shared-checkout note: resolved — implementation, imports, jobs, and git guards move to a detached worktree at the baseline commit. Other sessions’ tracked changes no longer enter this run’s review baseline.

## Findings
- [minor] **The “no file under the main checkout” rule contradicts the selected output locations.** The worktree itself resides beneath the main checkout’s `.claude/worktrees/`, and coordinator records and delivery patches reside in the run folder. Verification step 8 therefore cannot satisfy that statement literally; checking that `<main>/dev/ec_model` is absent also does not establish it. Define the protected scope as the main working tree excluding the designated worktree and CCC run-artifact directory, and apply the command-log audit to that scope. This preserves the intended isolation without changing the implementation location.

## Questions
- None.

READY: yes