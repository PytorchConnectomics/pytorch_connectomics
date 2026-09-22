# Review v1

## Summary

**Reviewer.** Planner (claude), in-session. Raw evidence: `state/review_v1.review.raw.md`. GPU job 3011760 (g011, L40S); logs in `state/review_v1/`.

**code_v1 fixed every finding it was asked to fix.**
- Speed: loss backward 0.174 → **0.0097 s** (18×), forward+backward 0.223 → **0.101 s**, meeting the ≤0.10 s target. Smoke s/step is now **1.063×** banis+ (was 1.311×), so gate (e) passes.
- Unit tests with CUDA present: **94 passed, 0 skipped**, including both previously skipped CUDA tests.
- Weight set to 0.3, the CUDA tolerance corrected, and the watchdog `squeue` contract implemented with 9 new tests.

**One gate still fails, and the one-shot weight rule is spent.** At w=0.3 the affinity term over steps 400–500 is 1.9237 vs banis+ 1.5353, a ratio of **1.253 > 1.10** (gate d). It improved from 1.510 at w=1.0 but does not clear the bar. Gate (f) also misses numerically: 3 skipped AMP updates versus 0, all inside the first 4 steps.

Per plan_v2 the weight rule is applied once and not re-applied, and review_v0 stated that a further (d) failure escalates. The remaining choice is a science and allocation decision, so this is a BLOCKER rather than NEEDS_CHANGES.

## Diff Baseline

run_start_ref: 1546f47ece4777e20bcddd1028fb1bb8908ae02d

`HEAD` equals run_start_ref; no commits. Changed tracked files are `connectomics/models/losses/{build,metadata}.py`; new untracked files are `connectomics/models/losses/embedding.py`, `tests/unit/test_embedding_mean_loss.py`, `tests/unit/test_embedding_label_pipeline.py`. The git-excluded config, sbatch, watchdog and watchdog test were read from disk. GPU artifacts stayed in ignored paths.

## Findings

- **[major] Gate (d) affinity interference persists at w=0.3: 1.253× (bar 1.10).**
  - Training affinity term, steps 400–500: 1.9237 (embed12) vs 1.5353 (banis+).
  - Validation affinity: 1.8606 vs 1.5303 at step 249 (1.216×), 1.5955 vs 1.4713 at step 499 (**1.084×**).
  - The gradient-ratio rule already moved the weight 1.0 → 0.3 and is spent, so no further automatic action exists.
  - Decision context (not an override): the smoke is 500 of 200,000 steps; the validation gap narrows 1.216 → 1.084 and is already inside the 1.10 in-run threshold at the second validation; the watchdog re-tests exactly this ratio at steps 4999 and 9999 and auto-cancels on a persistent regression.
- **[minor] Gate (f): 3 skipped AMP updates vs 0 for banis+, exceeding "≤ base+2".**
  - Scale 8192 with `_growth_tracker` 496 at step 500, so every skip is in the first 4 steps: GradScaler calibration against the large initial embedding loss.
  - The "no skip after step 250" criterion passes. No code change recommended.
- **[resolved] Speed gate (e).** 0.9716 vs 0.9137 s/step (1.063×). The `index_select` rewrite removed the advanced-index autograd cost as predicted.
- **[resolved] CUDA mixed-precision test.** Passes with the corrected tolerance, including finite gradients and no scale decrease over iterations 11–30.
- **[resolved] Memory.** N=4000: 1.2505 GB (pair_checkpoint False, configured) and 0.6071 GB (True); under the 1.5 GB bound.
- **[resolved] Watchdog `squeue` failures.** Consecutive-failure counter with reset, `JOB STATE UNAVAILABLE` and exit 4 without cancelling, covered by tests.
- **[resolved] Weight 0.3** is in the config and asserted by the resolved-config test.

## Tests to Add

None required for this verdict. If the user chooses a warm-up ramp on the embedding weight, that needs its own unit test for the schedule and a fresh matched smoke.

## Questions

- How should the run proceed given gate (d) at 1.253×? Options put to the user:
  1. launch at w=0.3 and let the 5k/10k watchdog decide (it uses the validation ratio, currently 1.084 at step 499, against a 1.10 threshold with auto-cancel);
  2. lower the weight further (e.g. 0.1), which deviates from the pre-registered rule and weakens the signal being tested, then re-smoke;
  3. code_v2 adds a warm-up ramp (weight 0 → 0.3 over the first few thousand steps), then re-smoke;
  4. stop and do not launch.

## Verdict

VERDICT: BLOCKER
