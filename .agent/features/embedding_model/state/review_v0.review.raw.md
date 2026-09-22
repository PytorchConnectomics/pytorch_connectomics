# review_v0 raw transcript (planner = claude, in-session)

The reviewer is the current session (plan-code claude-codex, planner owns review). There is no CLI subprocess; this file is the raw evidence log the review artifact attests to.

## Baseline / mutation guard
- `git rev-parse HEAD` = 1546f47ece4777e20bcddd1028fb1bb8908ae02d = run_start_ref.
- `git diff` and `git diff --cached` were captured at review start (/tmp/rv0_pre*.diff) and compared after all review commands: identical.
- GPU artifacts were written only under git-ignored `outputs/_smoke/embed12_review_v0/` and this run's `state/review_v0/`.
- `git status --short`: M build.py, M metadata.py, ?? embedding.py, ?? test_embedding_label_pipeline.py, ?? test_embedding_mean_loss.py. The excluded files (config, sbatch, watchdog and its test) were read from disk.

## Code read
- `connectomics/models/losses/embedding.py` (116 lines), read in full.
  - Math matches DeepEM MeanLoss:
    - pull is the mean over objects of the per-object mean of `max(0, L1 − δv)^2`;
    - push is `Σ_{parents differ} max(0, 2δd − L1)^2 / n(n−1)`;
    - nrm is `mean(L1(means))`.
  - Shapes are normalized by rank. Autocast is disabled; fp16 is promoted to fp32 and fp64 is preserved. CC runs on valid foreground. The empty case returns a graph-connected zero.
- `build.py` / `metadata.py` diffs: one registry line and one metadata entry (`spatial_weight_arg=mask`, `gt_seg_arg=gt_seg`).
- `dev/nisb/scripts/train_gate_watch.py`, read in full. It matches plan §5 plus the amendment: NONFINITE is checked first, verified cancel, exit 8.
  - Gap: `job_state` uses `check=True`, so `squeue` rc≠0 (e.g. a purged job id) raises from `watch()` instead of returning exit 4.
- `slurm_jobs/nisb_banis_plus_embed12_train.sbatch`: diffed against `nisb_banis_seed43_train.sbatch`. Only job-name, logs, MASTER_PORT, config and names differ.
- `tests/unit/test_embedding_mean_loss.py`: the oracle (lines 1-140) is upstream DeepEM code with only accumulator dtype changed (documented). `test_embedding_label_pipeline.py` uses the real `build_train_transforms`.

## CPU tests (login node, rerun independently)
- `pytest -q tests/unit/test_embedding_mean_loss.py tests/unit/test_embedding_label_pipeline.py dev/nisb/scripts/test_train_gate_watch.py` → 81 passed, 2 skipped (CUDA), 22.51 s.
- `pytest -q tests/unit/test_loss_orchestrator.py tests/unit/test_loss_functions.py tests/unit/test_malis_loss.py tests/unit/test_banis_reproduction_transforms.py` → 92 passed, 1 skipped. Same as the pre-edit baseline.

## GPU gates
Jobs: 3010373 (g018 L40S), 3011548 (g018 L40S), 3011625 (g011 L40S). Logs are in `state/review_v0/`.

### Step 4 rerun — real-data dataloader probe (`data_probe.log`)
- Codex's sandbox stalled in `zarr.open`; on the compute node it ran: 20/20 batches in 24.3 s, setup 0.04 s.
- `label` [2,6,128³] float32; `label_mask` [2,6,128³] bool, mask fraction 1.0; `gt_seg` [2,1,128³] **float32**.
- Per sample: GT ids 14–22; 26-connected components N = 34–54; **max N over 20 batches = 73**.
- CC 0.065 s per batch on CPU. Loss forward on CPU 0.22–0.24 s per batch with random pred; loss ≈ 7.6–8.1.

### Step 6 — CUDA tests (`cuda_tests.log`, `cuda_amp_tolpatched.log`)
- `test_cuda_memory_stress` with N = max(4000, 2·73) = 4000: peak above input 1,252,564,992 B (pair_checkpoint False) and 609,205,248 B (True). The configured False value is < 1.5 GB. PASS.
- `test_cuda_mixed_precision` FAILED at line 461, `assert_close(loss_fp16path, loss(pred.float()), rtol=0, atol=0)`:
  - expected 1085.2596435546875, got 1085.260009765625;
  - absolute difference 3.66e-4, relative 3.37e-7.
  - Cause: CUDA reduction/scatter kernels are not bitwise deterministic across calls. The `atol=0` spec came from plan_v2 test 6.7; that is a planner spec error.
  - Because the failure is at iteration 0, the AMP skip and finite-grad assertions never ran.
  - Re-run with only that line changed to `rtol=1e-5, atol=1e-5` (`state/review_v0/test_cuda_amp_tolpatched.py`, `-k mixed_precision`): **1 passed**. Finite grads after unscale and no scale decrease for iterations 11–30.

### Step 7 — trunk-gradient probe (`grad_probe_init.log`, `grad_probe_step500.log`)
- Excluded final conv: `model.out_0.conv_out.bias`. The weight tensor has shape[0]=18 too, but `named_parameters` listed only the bias under that shape filter; see the note below. 516 trunk tensors.
- Terms: term_0 BCE pred 0:6 target 0:6; term_1 embedding pred 6:18 target 0:3.
- **Init:**
  - raw losses (aff, emb): (3.67, 27.9), (2.75, 25.3), (3.14, 29.7), (1.19, 22.6);
  - trunk ||g|| (aff, emb): (1.12, 124.2), (0.82, 99.5), (0.86, 123.2), (0.43, 99.3);
  - r = 111.2, 120.6, 143.2, 233.5 → median 131.9. Context only.
- **Step 500 of the w=1.0 smoke (decision r):**
  - raw (aff, emb): (3.23, 0.95), (2.38, 1.10), (2.76, 0.89), (1.01, 2.09);
  - ||g|| (aff, emb): (0.504, 1.93), (0.360, 1.92), (0.258, 1.62), (0.351, 9.38);
  - r = 3.82, 5.34, 6.30, 26.76 → **median 5.82**;
  - all finite, no zero denominator.
  - Rule: largest w in {1.0, 0.3, 0.1, 0.03} with w·r ≤ 5 → **w = 0.3** (0.3·5.82 = 1.75; 1.0·5.82 > 5).
- **CORRECTION: probe exclusion defect found and fixed by the reviewer.** The `p.shape[0] == out_ch` filter excluded only `model.out_0.conv_out.bias`. MedNeXt `OutBlock` is `nn.ConvTranspose3d(in, n_classes, 1)` (`lib/MedNeXt/nnunet_mednext/network_architecture/mednextv1/blocks.py:200-212`), whose weight is `[32, 18, 1, 1, 1]`, so the output-conv weight gradient had been counted as trunk. The first-run numbers above are **superseded**.
- Rerun with exclusion by name `.conv_out.` (job 3011635, g011 L40S; `grad_probe_init_r3.log`, `grad_probe_step500_r3.log`; the old script is kept as `grad_probe_v1_shapefilter.py`). Excluded: `model.out_0.conv_out.weight` and `model.out_0.conv_out.bias`; 515 trunk tensors.
  - **Init:** trunk ||g|| (aff, emb) = (0.987, 120.4), (0.723, 96.0), (0.738, 119.2), (0.377, 96.1) → r = 121.9, 132.9, 161.5, 255.2, median **147.2**. Context only.
  - **Step 500 (decision):** raw losses are unchanged from the first run. ||g|| (aff, emb) = (0.455, 1.661), (0.313, 1.549), (0.233, 1.335), (0.281, 6.916) → r = 3.65, 4.95, 5.74, 24.65, **median 5.35**. All finite; no zero denominator.
  - **Rule:** w = largest of {1.0, 0.3, 0.1, 0.03} with w·r ≤ 5 → **w = 0.3** (1.0·5.35 = 5.35 > 5; 0.3·5.35 = 1.60). This is the same selection as the defective probe. The decision r sits only 7% above the w = 1.0 boundary; that closeness is recorded.

### Step 8 — matched 500-step smoke
- 1× L40S (g018), seed 43. Overrides: max_steps=500, **scheduler t_max=500** (required by the preflight `_validate_cosine_horizon`; identical for both runs), val every 250 steps with 5 batches, log every 10 steps.
- The first attempt (job 3010373) failed both runs at preflight because t_max was not overridden. That was a coordinator orchestration error, fixed in 3011548.
- Run dirs: `outputs/_smoke/embed12_review_v0/banis/20260915_175514`, `.../embed12/20260915_180326`. `smoke_analyze.log`:
  - (a) both runs completed training, validation at steps 249 and 499 (including image logging with 18 pred vs 6 label channels) and checkpoint save: PASS.
  - (b) all logged loss scalars finite: PASS.
  - (c) embedding term mean, steps ≤50 → 450–500: 6.254 → 1.173. PASS.
  - (d) mean `train_loss_term_0_weighted_step` over 400–500 = **2.391** vs banis+ `train_loss_total_step` **1.583** → ratio **1.51 > 1.10**. FAIL. Val affinity: embed12 term_0 2.216/1.967 vs banis+ 1.497/1.417 at 249/499.
  - (e) median s/step over steps 100–500: embed12 **1.200** vs banis+ **0.916** → **1.31 > 1.15**. FAIL.
  - (f) from `last.ckpt` MixedPrecision:
    - banis+: scale 65536, _growth_tracker 500, skipped 0;
    - embed12: scale 4096, _growth_tracker 495, **skipped 4**, all within the first 5 steps;
    - `_growth_tracker ≥ 250` PASS; `skipped ≤ base+2` (4 > 2) **FAIL**; growth_interval 2000 confirmed.

### Overhead profile (job 3011625, g011 L40S, `profile_3011625.out`)
- Dataloader, 16 workers, 40 batches: banis+ 0.0678 s/batch, embed12 0.0727 s/batch. Not the bottleneck.
- EmbeddingMeanLoss on real 2×128³ batches, random pred, medians of 12:
  - CC CPU round-trip 0.033 s;
  - `torch.unique` 0.001 s;
  - **loss forward 0.049 s**;
  - **loss backward 0.174 s**;
  - forward+backward 0.223 s.
- This accounts for about 0.22 of the 0.28 s/step overhead. Backward dominates. The likely cost centers are the autograd backward of advanced indexing over about 2M voxels: `means[inv]` backward is `index_put_(accumulate=True)`, a sort-based CUDA path, and boolean-mask gathers `pred[b].movedim(0,-1)[valid]`. Their `index_select`/flat-index equivalents have scatter-add backward. This is inference, not a measured op-level profile.

## Verdict reasoning
Several items require code changes that code_v1 is allowed to make:
- the pre-registered weight rule requires w = 0.3;
- the speed gate fails, with an actionable profile;
- the AMP unit test has an unsatisfiable exact-equality spec.

Gates (d) and (f) must be re-evaluated in the fresh smoke at w = 0.3 after code_v1. No BLOCKER: every failure has a concrete fix path. → NEEDS_CHANGES.
