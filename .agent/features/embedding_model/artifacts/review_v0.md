# Review v0

## Summary

**Reviewer.** Planner (claude), in-session: the reviewer role is the current session, so no CLI subprocess was used. Raw evidence log: `state/review_v0.review.raw.md`. GPU jobs 3010373, 3011548, 3011625 and 3011635 on L40S; logs in `state/review_v0/`.

**Checks.**
- Read the full diff and all four git-excluded deliverables.
- Re-ran CPU tests: 81 passed / 2 skipped, and regression 92 passed / 1 skipped.
- Ran the GPU gates from plan_v2 Verification steps 4 (probe rerun), 6, 7 and 8, plus an overhead profile.

**Correct as written.**
- The loss math matches DeepEM `MeanLoss`; the oracle is faithful.
- Shape, precision, mask and CC semantics match plan_v2.
- The label pipeline preserves un-eroded `gt_seg`, with the affinity target and mask identical to banis+.
- The watchdog implements the amendment (NONFINITE → verified cancel, exit 8).

**Real-data probe.** The step that stalled in the coder sandbox runs on a compute node: max N = 73, `gt_seg` float32, mask fraction 1.0.

**Failures.** At weight 1.0, three of the six matched-smoke criteria fail:

| Gate | embed12 | banis+ | Ratio / result |
|---|---|---|---|
| (d) affinity term, steps 400–500 | 2.391 | 1.583 | 1.51× |
| (e) seconds per step | 1.200 | 0.916 | 1.31× |
| (f) AMP skipped updates | 4 (all in the first 5 steps) | 0 | exceeds base+2 |

**Weight rule.** The pre-registered trunk-gradient rule, computed with a corrected probe, gives r_median = 5.35 at step 500, so **w = 0.3**.

**Unit test issue.** The CUDA mixed-precision test has an unsatisfiable `atol=0` spec, which was my error in plan_v2.

These need code_v1, followed by a fresh smoke at w = 0.3.

## Diff Baseline

run_start_ref: 1546f47ece4777e20bcddd1028fb1bb8908ae02d

- `HEAD` = run_start_ref; no commits.
- `git diff` and `git diff --cached` were identical before and after all review commands.
- Changed tracked files: `connectomics/models/losses/build.py`, `connectomics/models/losses/metadata.py`.
- New untracked files: `connectomics/models/losses/embedding.py`, `tests/unit/test_embedding_mean_loss.py`, `tests/unit/test_embedding_label_pipeline.py`.
- Git-excluded files, read from disk: `tutorials/neuron_nisb/base_banis+_embed12.yaml`, `slurm_jobs/nisb_banis_plus_embed12_train.sbatch`, `dev/nisb/scripts/train_gate_watch.py`, `dev/nisb/scripts/test_train_gate_watch.py`.

## Findings

- **[major] Weight rule selects w = 0.3; the config still has 1.0.**
  - Measured on the step-500 checkpoint of the w=1.0 smoke, with unweighted per-term trunk-gradient norms over 4 fixed batches (final `conv_out` weight and bias excluded).
  - Per-batch r = 3.65, 4.95, 5.74, 24.65; median 5.35.
  - Largest w in {1.0, 0.3, 0.1, 0.03} with w·r ≤ 5 is 0.3. At init r_median was 147 (context only).
  - **Required:** set `EmbeddingMeanLoss` `weight: 0.3` in `tutorials/neuron_nisb/base_banis+_embed12.yaml`, and update any test that asserts the configured weight.
  - Per plan_v2 the rule is not re-applied. The planner will run a fresh embed12 smoke at w=0.3, and it must pass every step-8 criterion.
  - The decision r is only 7% above the w=1.0 boundary; recorded.
- **[major] Speed gate (e) fails: 1.200 vs 0.916 s/step (1.31× > 1.15×).**
  - The profile on real 2×128³ batches (L40S, job 3011625) shows the dataloader is not the cause (0.0727 vs 0.0678 s/batch).
  - The loss is: forward 0.049 s (CC CPU round-trip 0.033 s, `unique` 0.001 s), **backward 0.174 s**; that covers ≈0.22 of the 0.28 s/step overhead.
  - Likely cost: autograd backward of advanced indexing over ≈2M voxels. `means[inv]` backprops through `index_put_(accumulate=True)`, and the boolean-mask gather `pred[b].movedim(0,-1)[valid]` is also expensive. This is inferred, not op-profiled.
  - **Required:** rewrite the voxel gathers without advanced-index autograd paths, keeping exactly the same values and gradients:
    - gather via flat indices, e.g. `idx = valid.flatten().nonzero().squeeze(1)`; `e = pred[b].reshape(D, -1).index_select(1, idx).T`;
    - use `means.index_select(0, inv)` in place of `means[inv]`;
    - keep `index_add_` for the sums.
  - All parity, gradient and chunking tests must still pass.
  - Speed target, measured by the planner on L40S: loss forward+backward ≤ 0.10 s on a real batch, and smoke s/step ≤ 1.15× banis+. If the coder cannot reach GPU, report CPU before/after timings for the same random real-size input and state that GPU is unmeasured.
- **[major] `test_cuda_mixed_precision` uses exact equality (`rtol=0, atol=0`) and fails on CUDA.**
  - Expected 1085.2596, got 1085.2600 (relative 3.4e-7) from non-bitwise-deterministic CUDA kernels. This spec came from plan_v2 test 6.7 (planner error).
  - The failure at iteration 0 also skipped the test's AMP assertions.
  - **Required:** change that `assert_close` to `rtol=1e-5, atol=1e-5` and keep the rest.
  - Verified: with only this change the test passes on L40S (`state/review_v0/cuda_amp_tolpatched.log`), including finite grads and no scale decrease for iterations 11–30.
- **[major] Affinity interference at w=1.0, gate (d).**
  - Training affinity term 1.51× banis+ over steps 400–500; val affinity 1.967 vs 1.417 at step 499.
  - No separate code change beyond the w=0.3 finding. It is re-gated in the fresh smoke; if (d) still fails at w=0.3, the next review returns BLOCKER for a human decision.
- **[minor] AMP gate (f) at w=1.0.**
  - embed12 scale 4096 (skipped = log2(65536/4096) = 4) with `_growth_tracker` 495, so all skips are within the first 5 steps (calibration at the initial embedding loss ≈25). banis+ had 0 skips.
  - Fails the pre-registered `skipped ≤ base+2` numerically, though the "no skip after step 250" criterion passes.
  - No code change requested; re-gated at w=0.3.
- **[minor] Watchdog `job_state` uses `subprocess.run(..., check=True)`.**
  - An `squeue` failure (e.g. a purged job id or a transient slurmctld error) raises out of `watch()` as a traceback instead of following the exit-code contract.
  - **Required:** in `watch()`, treat an `squeue` failure as unknown state and keep polling; after 3 consecutive failures print `JOB STATE UNAVAILABLE last_step=…` and exit 4 (report-only, no cancel). Add a test.
- **[minor] Recorded for transparency, no action.** The reviewer's first gradient probe excluded only `conv_out.bias`, because MedNeXt `OutBlock` is `ConvTranspose3d` with weight `[32,18,1,1,1]`. It was re-run with name-based exclusion; the numbers above are corrected, and the selection did not change (0.3).
- **[minor] Recorded, no action.** The smoke needed `train.optimization.scheduler.params.t_max=500` to satisfy preflight `_validate_cosine_horizon`. It was applied identically to both runs, so the matched comparison is intact. The first GPU attempt failed on this (orchestration error).

## Tests to Add

- Watchdog: `job_state` raising `CalledProcessError` once → polling continues; three consecutive failures → exit 4, cancel not called.
- Loss: keep the existing oracle/gradient/chunk parity suite as the regression net for the index_select refactor.
- Add one CPU test that gathers on a non-contiguous `pred` (e.g. `pred.transpose` → permuted back) to guard the flat-index gather against stride assumptions.
- Config: assert the `EmbeddingMeanLoss` weight is 0.3, if the tests reference the configured weight.

## Questions

- None blocking. Scientific note for the human, not for code_v1: at init the embedding term's trunk gradient is about 150× the affinity term's. A 0.3 weight still leaves the embedding dominant for the first few hundred steps. The fresh w=0.3 smoke's gate (d) is the decisive check; if it fails, the options (lower w, warm-up ramp, or stop-grad into a separate head) are a human decision, not a code_v1 choice.

## Verdict

VERDICT: NEEDS_CHANGES
