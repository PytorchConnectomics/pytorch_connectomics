CONTINUATION NOTICE (read first): A previous coder session for this exact stage already implemented most of plan_v2 in the working tree, then the session died (its multi-file `black` deadlocked in the sandbox and the API session became unrecoverable). The files currently present are that session's work:
- modified: connectomics/models/losses/build.py, connectomics/models/losses/metadata.py
- new: connectomics/models/losses/embedding.py, tests/unit/test_embedding_mean_loss.py, tests/unit/test_embedding_label_pipeline.py
- new (git-excluded/ignored): tutorials/neuron_nisb/base_banis+_embed12.yaml, slurm_jobs/nisb_banis_plus_embed12_train.sbatch, dev/nisb/scripts/train_gate_watch.py, dev/nisb/scripts/test_train_gate_watch.py
Pre-edit regression baseline (recorded by the coordinator from the previous session's transcript, run BEFORE any edit): `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && pytest -q tests/unit/test_loss_orchestrator.py tests/unit/test_loss_functions.py tests/unit/test_malis_loss.py tests/unit/test_banis_reproduction_transforms.py` -> 92 passed, 1 skipped. (`conda run` is not on PATH in the sandbox; use the `source ... activate pytc` form.) The previous session wrote all files in ~7 minutes, so audit carefully.

Your job: audit these files line-by-line against plan_v2 and the binding amendment below, fix every gap or deviation (e.g., rank-based shape normalization, fp64 work dtype rule, chunked/checkpointed pair term, denominator semantics, NONFINITE watchdog exit 8, all listed tests), then run the full Verification (coder steps 1-5) and write the artifact. Treat the existing code as a draft, not as correct.

Formatting: do NOT run black/isort on multiple files (multi-file black deadlocks in this sandbox). If needed, run `black --workers 1 <single file>` on new files only. Never reformat pre-existing files.

You are the CCC coder for stage code_v0 in the PyTorch Connectomics (pytc) repository at /projects/weilab/weidf/lib/pytorch_connectomics (your working directory).

Implement the approved plan_v2 exactly, including the binding human amendment below. Then verify and write the code summary artifact.

Hard rules:
- Do not create git commits, branches, stashes, or change HEAD. HEAD must stay at run_start_ref 1546f47ece4777e20bcddd1028fb1bb8908ae02d.
- Keep changes scoped to the files listed in plan_v2 "Files and Areas" (code-stage files only). Do not reformat or touch unrelated files. Do NOT write the prereg or the dw-research log (coordinator-only).
- Do not launch training: no sbatch, no srun, no scripts/main.py training/test runs. This login node has no GPU: CUDA-only tests must skip cleanly; report them as skipped (the planner runs them on GPU).
- Python env: use `conda run -n pytc ...` or `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc`. Run the regression test command once BEFORE editing to record pre-existing failures.
- Read the real repo code you touch (loss registry/metadata, orchestrator gt_seg/mask dispatch, MalisLoss, label transforms / build_train_transforms, existing tests) and follow existing patterns and style.
- Do not claim a test passed unless you ran it; report the actual command and summarized output.
- The dataloader probe (Verification step 4) reads real NISB zarr data; bound it (20 batches, num_workers 0 or small) and report timings; if it cannot run in your sandbox, say exactly why.

When done, write the CCC artifact to /projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/embedding_model/artifacts/code_v0.md.tmp (the coordinator validates and renames it). It MUST use exactly this structure (headings verbatim, each exactly once):

# Code v0
## Overview
## What Changed
## Implementation Details
## Files Changed
| File | Purpose |
|---|---|
## Git Baseline
run_start_ref: <sha>
current_head: <sha>
## Verification
## Review Focus
## Risks and Unknowns
## Changes Since Previous Code Version
Initial implementation.

Artifact requirements: summarize the ACTUAL diff (not intent); include every command run with pass/fail/skip counts; include the dataloader probe numbers (max N, CC/loss timings, mask fraction); list any deviation from plan_v2 with the reason; paste the FULL contents of the git-excluded/ignored files (tutorials/neuron_nisb/base_banis+_embed12.yaml, slurm_jobs/nisb_banis_plus_embed12_train.sbatch, dev/nisb/scripts/train_gate_watch.py, dev/nisb/scripts/test_train_gate_watch.py) inside ## Implementation Details, because they do not appear in git diff.

===== task.md =====
# Task

Output folder: /projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/embedding_model/
Repository: /projects/weilab/weidf/lib/pytorch_connectomics

a) read /projects/weilab/weidf/lib/dw-kb/papers/wiki/connectomics/kisuk_embedding*.md and https://github.com/ZettaAI/DeepEM/blob/feature/sr-zero-pad/deepem/loss/mean.py.
b) build a model to add embedding (e.g. 12-channel) in addition to 6-affinity channel.
c) read previous lessons: /projects/weilab/weidf/lib/dw-research/projects/2026_nisb_base and code /projects/weilab/weidf/lib/pytorch_connectomics/dev/nisb to see if there is any failed attempt.
d) work on pytc repo and kick off the training similar to banis+ on nisb-base data after the code is complete

===== run.md =====
# CCC Run

## Description
Add a mean-shift-style embedding head (e.g. 12 channels, Lee et al. / DeepEM mean loss) alongside the 6-channel affinity output in pytorch_connectomics, check prior NISB lessons for failed attempts, then launch a BANIS+-like training on nisb-base.

## Runtime
planner: claude
coder: codex
plan_code: claude-codex
session_detected: claude
plan_code_source: default

## Rounds
plan_rounds: 2
revision_rounds: 2

## Task Summary
See task.md. (a) read Kisuk embedding wiki + DeepEM mean.py loss; (b) 6-aff + N-ch embedding model; (c) review nisb_base lessons and dev/nisb for failed attempts; (d) kick off BANIS+-like training on nisb-base.

## Git Baseline
run_start_ref: 1546f47ece4777e20bcddd1028fb1bb8908ae02d
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

## Workflow State
current_stage: plan_v2_review
latest_artifact: artifacts/plan_v2_review.md
latest_verdict: NEEDS_CHANGES
next_action: code_v0

## Status
active

## Human Decisions
- 2026-09-15: plan_v2_review unresolved [major] (watchdog non-finite handling) resolved by user choice "Cancel + report": non-finite gate value => immediate verified scancel, exit 8, never PASS. Binding amendment: state/human_decision_plan_v2.md. Run resumed to code_v0 on the approved plan_v2 + amendment.

===== artifacts/plan_v2.md (APPROVED PLAN) =====
# Plan v2

## Summary

**Goal.** Add a 12-channel dense voxel-embedding output to pytorch_connectomics (pytc), trained with a DeepEM-style mean (discriminative) loss, next to the unchanged 6-channel BANIS+ affinity output. Then launch a BANIS+-equivalent 200k-step training on nisb-base.

**Background.**

- **Paper/wiki** (`dw-kb/papers/wiki/connectomics/kisuk_embedding*.md`; Lee, Lu, Luther & Seung, TMI 2021, devoEM):
  - dense voxel embeddings from metric learning;
  - metric graph (short- and long-range affinities computed from the embeddings);
  - Mutex Watershed;
  - mean-embedding agglomeration (MEA) for self-contact splits;
  - supp: MEA threshold θd ∈ [1.0, 1.5).
- **DeepEM `deepem/loss/mean.py` (`feature/sr-zero-pad`), `MeanLoss`.** Per object k with voxel embeddings e_i and mean μ_k:
  - `L_int = mean_k mean_{i∈k} max(0, ||e_i − μ_k||_1 − δv)^2`
  - `L_ext = Σ_{k≠l, mext[k,l]} max(0, 2δd − ||μ_k − μ_l||_1)^2 / (N(N−1))`
  - `L_nrm = mean_k ||μ_k||_1`
  - `loss = α L_int + β L_ext + γ L_nrm`
  - Defaults: α=1, β=1, γ=0.001, δv=0, δd=1.5; id 0 excluded.
  - `recompute_ext`: objects are in-patch connected components of the GT ids; components sharing a GT id are not pushed apart.
  - Only batch element 0 is used.
  - `embed_dim` default 12.
  - Decode (`vec2aff`): `aff = max(0, (2δd − ||e1−e2||_1)/(2δd))^2`.
- **Prior NISB attempts** (`dw-research/projects/2026_nisb_base`, `pytc/dev/nisb`):
  - **No embedding model was ever trained on NISB.** `research_plan/plan_contrastive_embeddings.md` proposed it; status "Proposed", and its Phase 0 frozen-feature gate never ran.
  - **Failures that constrain the design:**
    - (i) ABISS (long-range affinity + mutex watershed, the embedding lineage) over-merges on NISB (`lesson_abiss`). So the devoEM dense segmenter is **not** adopted; the embedding is an auxiliary identity signal for a later heal decider.
    - (ii) MALIS, SDT/clDice, 2x-XY and banis2-occupancy did not heal thin splits (lessons 9, 10, 13, 16, 18, 19).
    - (iii) Removing the affinity erosion margin collapses cc3d NERL (`lesson_affinity_vs_occupancy`). **The affinity target keeps erosion=2.**
    - (iv) Erosion=2 erases thin processes (`lesson_seg_filled_thin`).
    - (v) Invariant I5: thin-split fixes need connected per-instance identity. An instance embedding is that signal.
  - **Process:** `dev/nisb/scripts/prereg_gate.sh` blocks `sbatch` without `PREREG=`.
- **Facts verified in the repo for this plan:**
  - `seg_to_affinity` (`connectomics/data/processing/affinity.py:424-507`) sets `mask[c]` from the offset storage slice and `seg != -1` only. It never depends on foreground.
  - `seg_erosion_instance` preserves negative ids. So voxels erased by erosion keep a valid affinity mask.
  - The seed-43 banis+ run (`outputs/nisb_base_banis_v3_erosion2_seed43/20260905_212323`, job 2968810, 200k steps in 2d05h) logs `val_loss_total`, which is affinity-only: 0.5034 @4999, 0.4522 @9999, 0.3989 @14999.
  - A multi-term run logs `val_loss_term_{i}_weighted` and `train_loss_term_{i}_weighted_step`.

## Scope

**In scope.**

1. **`EmbeddingMeanLoss`:** a faithful, vectorized, batch-aware port of DeepEM `MeanLoss` with `recompute_ext`. Registered in the pytc loss registry and metadata, and fed the raw GT via the existing `emit_gt_seg` → `gt_seg` path (the MALIS mechanism).
2. **Unit and integration tests:**
   - DeepEM parity (values and gradients);
   - batching and edge cases;
   - mixed-precision gradients;
   - memory stress;
   - orchestrator integration with an 18-ch output;
   - the full label pipeline (`gt_seg` exact, affinity identical to banis+, thin voxels eligible).
3. **Config** `tutorials/neuron_nisb/base_banis+_embed12.yaml`.
4. **Sbatch** `slurm_jobs/nisb_banis_plus_embed12_train.sbatch`.
5. **Watchdog** `dev/nisb/scripts/train_gate_watch.py`: tfevents early-stop gate plus liveness.
6. **Planner, during review_v0:** GPU gates (smoke, matched baseline smoke, trunk-gradient probe, grad-scaler check), with a pre-registered embedding-weight rule.
7. **Coordinator, after approval:** prereg, launch, watchdog, dw-research log.

**Out of scope** (follow-ups, recorded in the prereg):
- embedding→affinity decode, Mutex Watershed, MEA;
- fragment-pair AUC evaluation;
- test-time export of embeddings;
- separate head branches;
- warm-start from the banis+ checkpoint;
- moving CC to the dataloader.

## Proposed Changes

### 1. `connectomics/models/losses/embedding.py` (new): `EmbeddingMeanLoss(nn.Module)`

**Kwargs**: `alpha=1.0, beta=1.0, gamma=0.001, delta_v=0.0, delta_d=1.5, recompute_ext=True, connectivity=26, pair_chunk_size=1024, pair_checkpoint=False`.
- There is **no `mask_background` option**. Background (id 0) is always excluded, and negative ids (the pytc ignore sentinel) are always ignored.
- Validate `connectivity in (6, 18, 26)`, `delta_d > 0`, `delta_v >= 0`, `pair_chunk_size >= 1`.

**Signature**: `forward(pred, target, mask=None, gt_seg=None)`.
- `target` is ignored. The orchestrator's `pred_target` dispatch passes it.
- `pred` must be 5-D `[B, D, S0, S1, S2]`.
- **Shape normalization is decided by rank, never by `squeeze`.** This covers singleton spatial dims.
  - `gt_seg`:
    - `ndim == 5` → require `shape[1] == 1` and use `gt_seg[:, 0]`;
    - `ndim == 4` → use as-is;
    - else `ValueError`.
  - `mask`:
    - `ndim == 5` → `[B, C, S…]`, reduced per voxel with `.bool().all(dim=1)`;
    - `ndim == 4` → `[B, S…]`;
    - else `ValueError`.
  - The normalized `gt_seg` and `mask` spatial shapes must equal `pred.shape[2:]`, and B must match; otherwise `ValueError`.
- Missing `gt_seg` → `ValueError` naming `data.label_transform.emit_gt_seg: true`.

**Precision**: the body runs under `torch.autocast(device_type=pred.device.type, enabled=False)`.
- Work dtype is `wd = torch.float64 if pred.dtype == torch.float64 else torch.float32`, and `pred = pred.to(wd)`.
- So fp16/bf16 run in fp32, and fp64 stays fp64. This allows exact fp64 parity tests; training never uses fp64.

**Per sample b.**
1. `ids = gt_seg_n[b]`. If floating-point, `torch.round`. Then `.long()`.
2. `valid = ids > 0`, then `valid &= mask_n[b]` if a mask was given.
3. **Object labels.**
   - If `recompute_ext`:
     - `comp = cc3d.connected_components(np.where(valid_np, ids_np, 0).astype(np.uint64 or int64-compatible), connectivity=connectivity)` on CPU.
     - CC runs only on valid foreground, so masked or ignored voxels never bridge components.
     - Object label = comp. Parent id per component = the GT id at its voxels (take it from `ids[valid]` via the component map).
   - Otherwise, object label = `ids`.
4. **Gather**: flatten valid voxels, `e ∈ [M, D]`, `lab ∈ [M]`, then `obj, inv = torch.unique(lab, return_inverse=True)` and `N = len(obj)`. If N == 0, the sample is empty.
5. **Means**: `counts = torch.bincount(inv, minlength=N)` and `means = zeros(N, D, dtype=wd).index_add_(0, inv, e) / counts[:, None]`.
6. **`L_int`**:
   - `h = clamp((e − means[inv]).abs().sum(1) − δv, min=0)^2`;
   - `per_obj = zeros(N).index_add_(0, inv, h) / counts`;
   - `L_int = per_obj.sum() / max(1, N)`, i.e. the mean over objects of the per-object mean.
7. **`L_ext`**, exact and memory-bounded:
   - Loop over row chunks `i0:i1` of size `pair_chunk_size`. For each chunk compute `pd = (means[i0:i1, None, :] − means[None, :, :]).abs().sum(-1)` (`[c, N]`) and `m = clamp(2δd − pd, min=0)^2`.
   - `keep` = off-diagonal AND (not `recompute_ext` OR parents differ).
   - `chunk_sum = (m * keep).sum()`.
   - If `pair_checkpoint`, compute each chunk inside `torch.utils.checkpoint.checkpoint(fn, means, …, use_reentrant=False)`.
   - `total = Σ chunk_sum`. If `keep` is empty everywhere (N ≤ 1 or all one parent), `L_ext = 0`. Otherwise `L_ext = total / (N(N−1))`.
   - The denominator counts all ordered pairs, including excluded ones, exactly as DeepEM.
8. **`L_nrm`** `= means.abs().sum(1).mean()`.
9. **Sample loss** `= α L_int + β L_ext + γ L_nrm`.

**Batch reduction**:
- The loss is the mean of the sample losses over non-empty samples.
- If every sample is empty, return `(pred * 0).sum()`. That is a graph-connected zero.
- Policy: `[non-empty, empty]` equals the non-empty sample's loss.
- The returned dtype is `wd`. The orchestrator's weighted sum under autocast accepts fp32.

**Documented deviations from DeepEM**:
- (a) the loss is averaged over the batch rather than using element 0 only;
- (b) object voxels are always intersected with the valid mask (DeepEM's `generate_vecs` ignores the mask when `mask_background=True`); this is identical for an all-true mask;
- (c) CC is computed inside the loss, with default connectivity 26 (DeepEM computes it in a dataloader `recompute` augmentation);
- (d) there is no background-as-object mode.
### 2. Registration

- `connectomics/models/losses/build.py`: import and register `"EmbeddingMeanLoss"`.
- `connectomics/models/losses/metadata.py`: `"EmbeddingMeanLoss": LossMetadata("EmbeddingMeanLoss", spatial_weight_arg="mask", gt_seg_arg="gt_seg")`.
- `__init__.py` is untouched unless an import requires it (`MalisLoss` is not exported there).

**Why `spatial_weight_arg="mask"`.** With `None`, the orchestrator `masked_fill`s pred with −20 through a mask whose channel count (3) mismatches the 12 embedding channels. With `"mask"`, pred passes through unchanged except the global ±20 clamp, and the loss receives the routed validity mask.

### 3. `tutorials/neuron_nisb/base_banis+_embed12.yaml` (new; excluded via `.git/info/exclude`)

`_base_: [base_banis+.yaml]`.

- `experiment_name: nisb_base_banis+_embed12`, `save_path: outputs/nisb_base_banis+_embed12`.
- **`default.model.out_channels: 18`.** Single head: one 1×1 output conv, equivalent to DeepEM per-task 1×1 convs off a shared trunk. The affinity path is architecturally identical to banis+.
- **`default.model.loss.losses`** (the list replaces the base list, so BCE is restated):
  - `PerChannelBCEWithLogitsLoss`: weight 1.0, `pred_slice "0:6"`, `target_slice "0:6"`, `auto_pos_weight: true`, `max_pos_weight: 10.0`.
  - `EmbeddingMeanLoss`: weight `1.0`, `pred_slice "6:18"`, `target_slice "0:3"`, kwargs `alpha 1.0, beta 1.0, gamma 0.001, delta_v 0.0, delta_d 1.5, recompute_ext true, connectivity 26`.
    - The weight is subject to the pre-registered rule in Verification step 7; any change goes back through code_v1.
    - A YAML comment must state that `target_slice "0:3"` only routes the short-range affinity validity mask. The mask is geometry plus ignore, not foreground, and target values are unused.
- **`default.data.label_transform`:**
  - `erosion: 0`;
  - `emit_gt_seg: true`;
  - `targets: [{name: affinity, kwargs: {offsets: ["1-0-0","0-1-0","0-0-1","10-0-0","0-10-0","0-0-10"], affinity_mode: banis, erosion: 2}}]`.
- **Logging:** add `train_loss_term_0_weighted`, `train_loss_term_1_weighted` to `train.monitor.logging.scalar.loss`, as in `base_banis+_sdt_1head.yaml`.
- **Inherited unchanged:**
  - MedNeXt-L/k3, 128³, batch 2/GPU;
  - AdamW 1e-3, cosine 200k, EMA 0.999, 16-mixed, clip 1.0;
  - `inference.model.select_channel: [0,1,2]`, so test and decoding only see the short-range affinity.

### 4. `slurm_jobs/nisb_banis_plus_embed12_train.sbatch` (new; git-ignored)

- Clone of `slurm_jobs/nisb_banis_seed43_train.sbatch`: 4 GPUs, `long`, 32 CPUs, 160G, 5 days, same excludes and env.
- `--job-name=bemb12_train`, `MASTER_PORT=29712`, logs `slurm_jobs/logs/nisb_banis_plus_embed12_train_%j.{out,err}`.
- Command: `python scripts/main.py --config tutorials/neuron_nisb/base_banis+_embed12.yaml system.seed=43 experiment_name=nisb_base_banis+_embed12_seed43 save_path=outputs/nisb_base_banis+_embed12_seed43`.
- Header comment: pairs with the seed-43 banis+ run; see the Summary.

### 5. `dev/nisb/scripts/train_gate_watch.py` (new; `dev/` is git-ignored) + `dev/nisb/scripts/test_train_gate_watch.py`

Stdlib plus `tensorboard.backend.event_processing.event_accumulator`. The logic is a pure `watch(...)` function with injectable `read_scalars(run_dir) -> {tag: [(step, value, wall_time)]}`, `job_state(job_id) -> str|None`, `cancel(job_id) -> bool`, `now() -> float` and `sleep(s)`. A thin `main()` wires the real implementations: `squeue -h -j <id> -o %T`, `scancel <id>`, and verification by polling `job_state` for up to 120 s until it is no longer RUNNING/PENDING.

**Args**:
- `--run-dir`: explicit, pinned to the submitted job's run directory. The coordinator passes the directory created by that job; there is no glob and no newest-dir selection.
- `--job-id`
- `--baseline-logdir`
- `--pair RUN_TAG=BASE_TAG` (default `val_loss_term_0_weighted=val_loss_total`)
- `--train-tag` (default `train_loss_total_step`, used for progress)
- `--steps 4999,9999`
- `--max-ratio 1.10`
- `--gate-grace-steps 1500`
- `--stale-minutes 45`
- `--poll-seconds 300`
- `--dry-run`

**Loop semantics**. Each poll re-reads the run's scalars.
1. **Baseline**: every gate step must have a finite baseline value at startup. Otherwise exit 2 (`BAD BASELINE`).
2. **Per gate step s**:
   - If the run has `RUN_TAG` at step s: `ratio = run/base`. A non-finite run value counts as `ratio = inf`. Print `GATE s run=… base=… ratio=…`.
   - Else, if the max `--train-tag` step ≥ s + `gate-grace-steps` with no `RUN_TAG` at s: print `GATE DATA MISSING at s` and exit 6. This does not cancel.
3. **All gates evaluated**:
   - If every `ratio > max_ratio`, the gate fails:
     - with `--dry-run`: print `GATE FAIL — WOULD CANCEL <id>` and exit 3;
     - otherwise call `cancel`. If it returns True (scancel rc 0 and the job left RUNNING/PENDING within 120 s), print `GATE FAIL — CANCELLED <id>` to stdout and stderr, exit 3. If it returns False, print `GATE FAIL — CANCEL FAILED <id>` and exit 7.
   - Otherwise print `GATE PASS` and exit 0.
4. **Before all gates are evaluated**:
   - If `job_state` is None or a terminal state: print `JOB ENDED EARLY last_step=…` and exit 4.
   - If the job is RUNNING and the newest scalar wall_time is older than `stale-minutes` (by `now()`): print `STALE last_step=…` and exit 5. This does not cancel.
   - A job that is PENDING never goes stale.

**Tests** (`dev/nisb/scripts/test_train_gate_watch.py`, pytest; fake readers, fake clock, fake job functions; no tfevents or Slurm needed):
- pass (both ≤ 1.10);
- mixed (fail at 5k, pass at 10k) → PASS;
- both fail → cancel called once, and CANCELLED plus exit 3;
- both fail with cancel False → exit 7;
- both fail with dry-run → WOULD CANCEL, exit 3, cancel not called;
- non-finite run value at both gates → treated as fail;
- missing gate tag past the grace window → exit 6, cancel not called;
- stale while RUNNING → exit 5;
- no stale while PENDING;
- job disappears before the gates → exit 4;
- bad baseline → exit 2.

A separate test covers the real `read_scalars` on a tiny tfevents file written with `torch.utils.tensorboard.SummaryWriter` in `tmp_path`.
### 6. Tests: `tests/unit/test_embedding_mean_loss.py` (new)

The file contains a trimmed verbatim copy of DeepEM `MeanLoss` + `create_mapping` + `compute_ext_matrix` as the oracle, with the source URL. The oracle is fed the same `splt` (from `cc3d` at the same connectivity) when `recompute_ext`.

1. **Parity (fp64, both sides).**
   - Fixture: B=1, D=12, an 8×16×16 label with several ids, background, and one id split into two disconnected pieces; the mask is all-true.
   - Run the implementation and the oracle on the *same* float64 `pred` leaf (the implementation keeps fp64 by the precision rule).
   - Compare loss and `pred.grad` with `rtol=1e-9, atol=1e-12`.
   - Cases:
     - `recompute_ext` ∈ {False, True};
     - `delta_v` ∈ {0, 0.5};
     - `connectivity` ∈ {6, 26};
     - one random 20-object label.
   - **fp32 sanity:** the implementation on `pred.float()` vs itself on fp64 matches within `rtol=1e-5, atol=1e-6`.
2. **Hand-computed denominator.**
   - D=1, `alpha=gamma=0`, `beta=1`, `delta_d=1.5`, `recompute_ext=True`.
   - Three components A, B, C, each filled with a constant embedding: A=0.0, B=0.5, C=2.0. A and B share a parent id (two disconnected pieces of the same GT id); C is a different id.
   - Kept ordered pairs are (A,C), (C,A), (B,C), (C,B), contributing 2·(3−2)^2 + 2·(3−1.5)^2 = 2 + 4.5 = 6.5.
   - Assert `L_ext == 6.5 / 6` (not /4).
   - With `recompute_ext=False` and three distinct ids at the same values, the (A,B) pairs add 2·(3−0.5)^2 = 12.5, so assert `L_ext == 19 / 6`.
3. **Connectivity fixture.** A same-id object made of two voxel runs that touch only diagonally, with distinct constant-but-noisy embeddings.
   - `connectivity=6` gives 2 components, so `L_int` equals the **mean** of the two components' per-object pulls computed independently, and the pair is excluded from push.
   - `connectivity=26` gives 1 component.
4. **Batching and shapes.**
   - B=2 equals the mean of per-sample losses.
   - `[non-empty, empty]` equals the non-empty sample's loss.
   - An all-empty batch returns 0, `requires_grad`, and backward yields zero grads.
   - `gt_seg` as `[B,1,S…]` and as `[B,S…]` gives identical results, including singleton spatial dims (e.g. spatial `(1, 8, 8)` and `(8, 1, 8)`).
   - `mask` as `[B,C,S…]` and `[B,S…]` works likewise.
   - A wrong rank or shape raises `ValueError`.
5. **Edge cases.**
   - A single object gives `L_ext = 0`.
   - All components under one parent give `L_ext = 0`.
   - `gt_seg=None` raises `ValueError`.
6. **Background, ignore and mask** (both recompute settings): perturbing embeddings at voxels with `gt_seg == 0`, `gt_seg < 0` or `mask == False` changes neither the loss nor the grads elsewhere. An ignore or masked voxel sitting between two same-id pieces must not connect them under `recompute_ext` (they remain two components).
7. **Mixed-precision gradients** (CUDA; skip otherwise).
   - A tiny `Conv3d(1→12)` + `AdamW` under `torch.autocast("cuda", dtype=torch.float16)`, with `torch.amp.GradScaler()` (defaults).
   - The input is scaled so outputs reach about ±20. The label has 1–3-voxel objects and one about 2000-voxel object.
   - Run 30 optimizer iterations. **Calibration:** the first 10 iterations may skip (scale backoff is normal AMP behavior).
   - Assert over iterations 11–30:
     - no skipped step (`scaler.get_scale()` never decreases);
     - all parameter grads are finite after `unscale_`.
   - Loss accuracy on the *same* quantized prediction: the loss on the fp16 pred tensor (internally cast) equals the loss on that same tensor explicitly `.float()`-ed, bitwise within `atol=0`.
8. **Chunking, checkpointing and memory.**
   - (a) On CPU with a small N≈300 label, loss **and grads** are identical (`rtol=1e-6`) across `pair_chunk_size ∈ {1, 64, 1024}` × `pair_checkpoint ∈ {False, True}`.
   - (b) CUDA stress (skip otherwise): a fragmented 128³ label with N = max(4000, 2 × max N observed in Verification step 4) components, D=12. Report `torch.cuda.max_memory_allocated()` for forward+backward above the input baseline, for `pair_checkpoint` False and True.
   - The training config's value must be one whose measured peak is < 1.5 GB. If False meets it, keep False.
9. **Orchestrator integration** (modeled on the `gt_seg` pass-through tests in `tests/unit/test_loss_orchestrator.py`).
   - Setup: 18-ch pred, 6-ch labels + bool `label_mask`, `gt_seg`, and the two-term config.
   - The embedding term receives `pred[:, 6:18]` unmodified (no −20 fill) and `gt_seg`.
   - Backward from the embedding term alone gives zero grad on channels 0:6 and non-zero grad on 6:18.
   - The total loss is finite.

### 7. Label-pipeline test: `tests/unit/test_embedding_label_pipeline.py` (new)

This uses the real config-built training transform chain, built by `build_train_transforms` from the resolved `base_banis+.yaml` and `base_banis+_embed12.yaml`.
- Geometric and intensity augmentations are disabled or set to identity, so the post-augmentation label equals the input.
- The leading spatial crop is deterministic (`LeadingSpatialCropd`), so both chains crop identically.
- If a small patch size is needed for test speed, override `patch_size` and `target_context` identically in both configs.

**Fixture**: a synthetic label with `target_context` trailing padding. It contains:
- two touching instances;
- a 1–2-voxel-thick process adjacent to another instance, which erosion=2 removes;
- background;
- a `-1` ignore slab.

The thin process lies at least 11 voxels from every trailing border of the read region and at least 3 voxels from every leading border. That places it inside the valid storage region of all 6 banis offsets after the crop, and away from reflect-mode erosion borders.

**Assertions**:
- (a) The embed12 chain's `gt_seg` equals the input label cropped by the same leading crop, exactly.
- (b) `label` and `label_mask` from the embed12 chain are bitwise identical to the banis+ chain (global erosion=2).
- (c) On every thin-process voxel (zero in the eroded label):
  - `gt_seg > 0`;
  - `label_mask[0:3]` is true;
  - `EmbeddingMeanLoss` on a `requires_grad` pred gives non-zero grad at those voxels.
- (d) Voxels in the ignore slab have `label_mask` false for every channel, and zero embedding grad.
## Files and Areas

Code diff:
- `connectomics/models/losses/embedding.py` (new)
- `connectomics/models/losses/build.py` (import + 1 registry line)
- `connectomics/models/losses/metadata.py` (+1 entry)
- `tests/unit/test_embedding_mean_loss.py` (new)
- `tests/unit/test_embedding_label_pipeline.py` (new)

Outside the git diff (paste full content into `code_vN.md`):
- `tutorials/neuron_nisb/base_banis+_embed12.yaml` (new; git-excluded)
- `slurm_jobs/nisb_banis_plus_embed12_train.sbatch` (new; git-ignored)
- `dev/nisb/scripts/train_gate_watch.py` (new; git-ignored)
- `dev/nisb/scripts/test_train_gate_watch.py` (new; git-ignored)

Coordinator only, after approval:
- `dev/nisb/research_plan/prereg_nisb_embed12.md`
- `dw-research/projects/2026_nisb_base/logs/2026-09-15_embedding_model.md`

No changes to MedNeXt, the orchestrator, data transforms, inference, or decoding.
## Verification Plan

### Coder (code stage)

1. **Unit tests.** Run `conda run -n pytc pytest -q tests/unit/test_embedding_mean_loss.py tests/unit/test_embedding_label_pipeline.py dev/nisb/scripts/test_train_gate_watch.py`. All must pass. Report which CUDA-only tests ran and which were skipped; the planner runs any skipped ones in step 6.
2. **Regression.** Run `conda run -n pytc pytest -q tests/unit/test_loss_orchestrator.py tests/unit/test_loss_functions.py tests/unit/test_malis_loss.py tests/unit/test_banis_reproduction_transforms.py`. Run it once BEFORE editing to record pre-existing failures. No new failures are allowed.
3. **Config.** Run `scripts/validate_tutorial_configs.py` (per-file mode if available) and dump the resolved config. Confirm:
   - `out_channels` = 18;
   - the two loss terms with their slices and weight;
   - erosion = 0, and the affinity target has erosion 2;
   - `emit_gt_seg` = true.
4. **Dataloader probe.** CPU only, using the real lazy zarr train datamodule from the config, over 20 batches. Report:
   - shapes and dtypes of `label`, `label_mask` and `gt_seg`;
   - the fraction of `label_mask[0:3]` that is true;
   - per sample: number of GT ids, number of 26-connected components, and max N;
   - per batch: CC time and loss wall time on CPU.
5. **Watchdog.** Unit tests are part of step 1. Also run a real dry-run read, `train_gate_watch.py --dry-run --run-dir outputs/nisb_base_banis_v3_erosion2_seed43/20260905_212323 --baseline-logdir outputs/nisb_base_banis_v3_erosion2_seed43/20260905_212323/logs --pair val_loss_total=val_loss_total --job-id 0`. Expected: ratios 1.0 and `GATE PASS`.

### Planner (review_v0)

Runs on 1 L40S/A100 via `srun`; every `main.py` command is `PREREG=`-prefixed.

6. **CUDA tests** the coder skipped: 6.7 mixed-precision gradients and 6.8(b) memory stress.
7. **Trunk-gradient probe and weight rule** (defined once; not re-applied).
   - **Probe setup.** Load the model state, take 4 fixed training batches (datamodule seed 43, first 4 draws), fp32, no autocast, train mode.
   - **Per batch j.** Compute the *unweighted* affinity loss (the BCE term's raw value) and the *unweighted* embedding loss (the raw `EmbeddingMeanLoss` value, ignoring the config weight). Backward each separately and take the L2 norm of the gradients over all parameters except the final 1×1 output conv, giving `g_aff,j` and `g_emb,j`.
   - `r = median_j(||g_emb,j|| / ||g_aff,j||)`.
   - **Blocker.** If any `||g_aff,j|| == 0` or any value is non-finite, return BLOCKER with the numbers.
   - **When to measure.** Report `r` at init for context. The **decision `r`** is measured on the step-500 checkpoint of the **weight-1.0** embed12 smoke from step 8.
   - **Rule.** `w` = the largest of {1.0, 0.3, 0.1, 0.03} such that `w·r ≤ 5`.
     - If `w < 1.0`, return NEEDS_CHANGES so code_v1 sets that weight. The planner then runs a **fresh** embed12 smoke at `w` (the banis+ smoke is reused), and it must pass every step-8 criterion. The rule is not re-applied to the new smoke.
     - If `0.03·r > 5`, return BLOCKER for a human.
8. **Matched smoke.** Same node, `system.seed=43`, 1 GPU. Overrides:
   - `train.optimization.max_steps=500`, `train.optimization.val_check_interval=250`, `train.optimization.val_steps_per_epoch=5`;
   - `train.optimization.log_every_n_steps=10`, `train.monitor.logging.scalar.loss_every_n_steps=10`;
   - `save_path` under `outputs/_smoke/`.

   Run `base_banis+.yaml` and `base_banis+_embed12.yaml`. Pass criteria:
   - **(a) Completes cleanly.** No crash through training, both validations (including image logging with 18 predicted vs 6 label channels) and checkpoint save.
   - **(b) Losses finite.** Every logged loss scalar is finite in both runs.
   - **(c) Embedding loss decreases.** Mean embedding term over steps 450–500 is below its mean over steps ≤ 50.
   - **(d) Affinity loss not hurt.** Mean `train_loss_term_0_weighted_step` over steps 400–500 must be ≤ 1.10 × banis+'s mean `train_loss_total_step` over the same steps.
   - **(e) Speed.** Seconds per step over steps 100–500, from tfevents wall_time with validation intervals excluded, must be ≤ 1.15 × banis+.
   - **(f) AMP.** Read `ckpt["MixedPrecision"]` from each run's `last.ckpt`.
     - GradScaler defaults are `init_scale=65536` and `growth_interval=2000`, and the smoke has 500 < 2000 steps, so no growth can occur. Therefore `skipped = log2(65536 / scale)` exactly, and `_growth_tracker` = consecutive non-skipped updates at the end.
     - Calibration allowance is the first 250 steps. Require embed12 `_growth_tracker ≥ 250` (no skipped update in steps 251–500) and embed12 `skipped ≤ banis+ skipped + 2`.
     - If the key is absent or `growth_interval ≠ 2000`, return BLOCKER.

   Any failure returns NEEDS_CHANGES or BLOCKER with the numbers.

### Coordinator (after approval)

9. **Prereg and launch.** Write the prereg at `dev/nisb/research_plan/prereg_nisb_embed12.md` (header below). Then run `PREREG=dev/nisb/research_plan/prereg_nisb_embed12.md sbatch slurm_jobs/nisb_banis_plus_embed12_train.sbatch`.
10. **Liveness.** Confirm the job is RUNNING. Identify the run directory it created under `outputs/nisb_base_banis+_embed12_seed43/`: the only timestamp dir created after submission, whose `config.yaml` has the matching `experiment_name`. Confirm the first scalar events exist. Record job id, log paths and run dir in `run.md` and the dw-research log.
11. **Watchdog.** Run it as a background task so its exit notifies the session:
    `train_gate_watch.py --job-id <id> --run-dir <pinned run dir> --baseline-logdir outputs/nisb_base_banis_v3_erosion2_seed43/20260905_212323/logs --pair val_loss_term_0_weighted=val_loss_total --steps 4999,9999 --max-ratio 1.10`
    - Baselines are 0.5034 and 0.4522.
    - Exit 3: report CANCELLED loudly.
    - Exit 7: CANCEL FAILED; report immediately and ask the user.
    - Exits 4, 5, 6: report immediately. No cancel.
    - Exit 0: report `GATE PASS` along with the ratios.
### Prereg header (loop.md §2)

- **hypothesis:** a 12-ch mean-loss embedding head on the shared MedNeXt-L trunk learns per-instance identity that persists across thin affinity gaps, without degrading the 6-ch affinity.
- **track:** false_split.
- **targets:** the 0.772 branch-merge oracle, via a later mean-embedding heal decider.
- **predicted_dNERL:** +0.00 ± 0.01 off seed-43 cc3d@0.66 from the affinity channels alone; any gain needs the follow-up decider.
- **cheap_gate:** user-directed launch. The loop's Phase 0 frozen-feature gate did not run (recorded honestly). In-run gates: steps 7–8 and the 5k/10k watchdog.
- **kill_criterion:**
  - (a) watchdog affinity val-loss ratio > 1.10 at 5k and 10k → cancelled;
  - (b) at 200k, center-chunk (tile_1_1_1) cc3d@0.66 NERL of the affinity more than 0.02 below seed-43 banis+ → the embedding hurts the trunk;
  - (c) mean-embedding fragment-pair AUC on heal/veto candidates < 0.77 (the LSD ceiling) → kill the embedding-decider branch.
- **cost:** retrain, 4 GPUs × about 2.5 days.

## Risks and Questions

1. **Trunk interference.**
   - Equal weights follow DeepEM's per-task default. The initial push term is ≈ (2δd)^2 = 9 vs a BCE below 1, but scalar magnitudes do not measure gradient share.
   - Mitigations: the gradient probe and weight rule (step 7), the matched smoke (step 8), and the 5k/10k auto-cancel watchdog.
   - The `w·r ≤ 5` threshold is a judgment call, not a derived number.
2. **CPU `cc3d` in the loss.** It syncs `gt_seg` (2×128³) to host every step on every rank. Step 8's s/step gate bounds the cost; exceeding 15% blocks and routes to a follow-up that moves CC to the dataloader.
3. **Un-eroded `gt_seg`.** It includes boundary voxels between touching neurons, the hardest voxels. This was chosen because erosion=2 deletes thin processes, which is where the NISB prize is. It is an assumption, not a measured result.
4. **`recompute_ext=True` with 26-connectivity.**
   - The output FOV equals the input FOV (128³), so pulling in-patch-disconnected pieces of one neuron together is an unlearnable target.
   - Limitation: components that share a GT id are neither pulled together nor pushed apart. Identity supervision is lost between pieces whose connection lies outside the patch.
   - 26-connectivity avoids also losing it for diagonal-only in-patch contacts (the fixture in test 6.3 pins both behaviors).
   - Thin processes that are GT-connected **within valid foreground** stay one component, so identity across *predicted* affinity gaps is still supervised.
5. **Checkpoint top-k monitors `val_loss_total`**, which is now BCE plus the embedding loss. Step checkpoints (every 50k) and `last.ckpt` are unaffected; evaluation uses step 200000, as for banis+.
6. **The global ±20 clamp** zeroes gradients beyond ±20. The norm term and δd=1.5 should keep embeddings far inside that range.
7. **Validation noise in the watchdog.** Different output channels change the init RNG draw, so seed 43 is not a fully matched run. The 10% tolerance and the requirement to fail at both gates are meant to absorb that.
8. **Git exclusions.** Config, sbatch, watchdog and its test are git-excluded, so their full contents must be pasted into `code_vN.md` for review.
9. **Later, not blocking:** whether affinity should be decoded from the embedding (DeepEM `rsunet_embed_iso`) rather than predicted in parallel. Parallel is the minimal, non-regressing choice.

## Changes Since Previous Plan Version

Addresses every plan_v1_review finding:

- **[major] fp16 gradients (finding 4), partially resolved before.**
  - Test 6.7 now allows a 10-iteration AMP calibration period, then requires no skipped step and finite parameter grads for iterations 11–30.
  - The loss-accuracy check compares the same quantized fp16 tensor with and without an explicit cast, instead of separate fp16 and fp32 conv outputs.
  - The smoke AMP gate (8f) is now an exact skipped-update count. It reads `MixedPrecision.scale` and `_growth_tracker` from the checkpoint; with default `growth_interval=2000` > 500 steps, skipped = log2(65536/scale) exactly. Checkpoint key presence and defaults were verified on a seed-43 banis+ checkpoint (`{'scale': 2048.0, ..., 'growth_interval': 2000, '_growth_tracker': 1798}`). Thresholds: no skip after step 250; skips ≤ banis+ + 2.
- **[major] Weight rule (finding 5), partially resolved before.** Step 7 now pins down:
  - the loss: `r` uses *unweighted* per-term losses, so the weight is not double-counted;
  - aggregation: median over 4 fixed batches;
  - zero or non-finite values: BLOCKER;
  - when it applies: only to the weight-1.0 smoke at step 500, applied once;
  - after a change: a fresh embed12 smoke at the selected weight must pass every step-8 criterion.
- **[major] Denominator fixture.**
  - Test 6.2 now uses means 0, 0.5, 2.0 with `alpha=gamma=0`, expecting `L_ext = 6.5/6`, not /4.
  - The `recompute_ext=False` control expects 19/6.
- **[major] `squeeze` shape bug.**
  - §1 now normalizes `gt_seg` and `mask` by rank only (5-D requires channel 1 for `gt_seg`; 4-D is used as-is).
  - Test 6.4 covers both layouts with singleton spatial dims, plus wrong-rank errors.
- **[major] Parity precision.**
  - The work dtype is fp64 when pred is fp64, and fp32 otherwise.
  - Parity (6.1) runs both implementations on the same fp64 leaf (`rtol=1e-9`), plus a separate fp32-vs-fp64 sanity check (`rtol=1e-5`).
- **[major] Watchdog contracts.**
  - The run dir is pinned explicitly (`--run-dir`); no globbing.
  - A missing gate tag past `gate-grace-steps` exits 6 without cancelling.
  - A non-finite run value counts as a fail. A bad baseline exits 2.
  - `scancel` is verified by rc and by the job leaving RUNNING/PENDING before CANCELLED is printed; otherwise it exits 7. Dry-run prints "WOULD CANCEL".
  - Logic is a pure function with injected readers, job state, cancel and clock.
  - Deterministic tests cover pass, mixed, fail-cancel, fail-cancel-failure, dry-run, non-finite, missing tag, stale-running, no-stale-pending, early end and bad baseline, plus a real tfevents read.
- **[minor] Chunk grad parity.** Test 6.8(a) checks loss and grads across chunk sizes × checkpoint on/off. `pair_checkpoint` is an explicit kwarg.
- **[minor] Label-pipeline determinism.**
  - §7 states the crop is deterministic.
  - The thin process sits ≥11 voxels from trailing and ≥3 from leading borders (valid for all 6 offsets, away from reflect-mode erosion edges).
  - The ignore-slab assertion now covers all mask channels.
- **[minor] Wording.** Test 6.3 says **mean** of the two pulls. Risk 4 now says "connected within valid foreground".

===== artifacts/plan_v2_review.md =====
# Plan v2 Review

## Summary

**Reviewer:** `codex exec --sandbox read-only` (coder role).
- Raw transcript: `state/plan_v2_review.review.raw.md`.
- Prompt: `state/plan_v2_review.prompt.md` (68,463 bytes).
- No repository mutation: `git diff` identical before and after; `git status` empty.

**Result:** `READY: no`.
- Resolved: 9 of the 10 plan_v1_review findings (background, label pipeline, AMP, weight rule, chunking/memory, denominator fixture, connectivity wording, shape normalization, parity precision). The reviewer confirms the rest of plan_v2 is implementable without further design decisions.
- One major finding remains in the watchdog's non-finite handling.

plan_v2 was the last allowed plan version (p2), so in normal mode the run is blocked for a human decision.

## Findings

- [minor] **Background:** resolved.
- [minor] **Label-pipeline findings 2 and 3:** resolved.
- [minor] **AMP (finding 4):** resolved at plan level.
- [minor] **Weight rule (finding 5):** resolved.
- [minor] **Memory/chunking (finding 6):** resolved.
- [minor] **Denominator fixture:** resolved.
- [minor] **Connectivity wording:** resolved.
- [minor] **Shape normalization:** resolved.
- [minor] **Precision parity:** resolved.
- [major] **Watchdog non-finite handling (partially resolved).**
  - Problem: §5 maps a non-finite gate value to ratio=inf, and `GATE PASS` is allowed when any gate passes. So NaN at step 4999 plus ratio 1.0 at 9999 gives PASS, which certifies invalid training.
  - Required: a non-finite gate value must produce a distinct failure outcome that can never become PASS, plus a mixed finite/non-finite test.

## Questions

- Should a non-finite validation gate value trigger an immediate cancel, or an immediate failure report without cancelling? The reviewer defers this policy choice to a human.

## Verdict

VERDICT: NEEDS_CHANGES

===== state/human_decision_plan_v2.md (BINDING AMENDMENT) =====
# Human decision: plan_v2 amendment (binding for code_v0)

**Date:** 2026-09-15. **Decided by:** the user, via an AskUserQuestion answer: "Cancel + report (Recommended)".

**Resolves:** the plan_v2_review [major] finding on the watchdog's non-finite gate handling.

## Amendment to plan_v2 §5 (`dev/nisb/scripts/train_gate_watch.py`)

1. **Non-finite value is a hard fail.** A non-finite run value (NaN or ±inf) for RUN_TAG at *any* gate step is a distinct hard failure, `GATE NONFINITE`. It is not converted to ratio=inf and can never become `GATE PASS`.
2. **Evaluated immediately.** As soon as a non-finite value is observed at any gate step, even if other gate steps are not yet evaluated or would pass:
   - **Without `--dry-run`:** call `cancel(job_id)`.
     - If `cancel` returns True (scancel rc 0 and the job left RUNNING/PENDING within 120 s): print `GATE NONFINITE at step s — CANCELLED <id>` to stdout and stderr, exit 8.
     - If `cancel` returns False: print `GATE NONFINITE at step s — CANCEL FAILED <id>`, exit 7.
   - **With `--dry-run`:** print `GATE NONFINITE at step s — WOULD CANCEL <id>`, exit 8, and do not call cancel.
3. **Everything else unchanged.** The finite-ratio rule stays: fail only if every gate ratio > max_ratio, which cancels and exits 3. The missing / stale / early-end / bad-baseline behaviors are also unchanged.
4. **Tests to add.** In `dev/nisb/scripts/test_train_gate_watch.py`:
   - (a) NaN at 4999 and ratio 1.0 at 9999 → NONFINITE, cancel called once, exit 8, never PASS.
   - (b) Ratio 1.0 at 4999 and inf at 9999 → NONFINITE, exit 8.
   - (c) NaN with cancel failing → exit 7.
   - (d) NaN with dry-run → WOULD CANCEL, exit 8, cancel not called.
   - This replaces the plan_v2 test "non-finite run value at both gates → treated as fail".
5. **Coordinator reporting.** In plan_v2 Verification step 11, exit 8 is reported loudly as a diverged-and-cancelled run.
