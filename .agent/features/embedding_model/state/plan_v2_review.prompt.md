You are the CCC plan reviewer (coder role) for a single incremental change in the PyTorch Connectomics (pytc) repository.

Do not edit files.
Review only the artifacts and diffs included in this prompt. Do not inspect other repository files.
Tag each finding as [minor] or [major].
Finish with exactly one line: READY: yes|no

This is the third and final plan review (plan_v2 is the last allowed plan version; unresolved major findings will require a human decision). Included: task.md, the previous plan (plan_v1), your previous review (plan_v1_review), and the revised plan (plan_v2). Focus on:
- whether each plan_v1_review finding is resolved by plan_v2 (state per finding);
- genuinely new correctness problems introduced by plan_v2;
- whether you can implement plan_v2 without further design decisions.
Severity: [major] = affects correctness, training validity, data integrity, or verification; [minor] = nits/follow-ups that can be handled during implementation. Ambiguous severity is major. Do not re-raise resolved issues, and do not escalate implementation details that plan_v2 already leaves to measured choice.

Output format:
## Findings
- [major|minor] ...
## Questions
- ...
READY: yes|no

===== task.md =====
# Task

Output folder: /projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/embedding_model/
Repository: /projects/weilab/weidf/lib/pytorch_connectomics

a) read /projects/weilab/weidf/lib/dw-kb/papers/wiki/connectomics/kisuk_embedding*.md and https://github.com/ZettaAI/DeepEM/blob/feature/sr-zero-pad/deepem/loss/mean.py.
b) build a model to add embedding (e.g. 12-channel) in addition to 6-affinity channel.
c) read previous lessons: /projects/weilab/weidf/lib/dw-research/projects/2026_nisb_base and code /projects/weilab/weidf/lib/pytorch_connectomics/dev/nisb to see if there is any failed attempt.
d) work on pytc repo and kick off the training similar to banis+ on nisb-base data after the code is complete

===== artifacts/plan_v1.md =====
# Plan v1

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

**Kwargs:** `alpha=1.0, beta=1.0, gamma=0.001, delta_v=0.0, delta_d=1.5, recompute_ext=True, connectivity=26, pair_chunk_size=1024`.
- There is **no `mask_background` option.** Background (id 0) is always excluded and negative ids (the pytc ignore sentinel) are always ignored. This removes the ill-defined background-component case (review finding 1).
- Validate `connectivity in (6, 18, 26)` and `delta_d > 0`.

**Signature:** `forward(pred, target, mask=None, gt_seg=None)`.
- `target` is ignored; the orchestrator's `pred_target` dispatch passes it.
- `pred` is [B, D, *spatial] with 3 spatial dims.
- `gt_seg` is [B, 1, *spatial] or [B, *spatial], integer or integer-valued float.
- `mask` is optional: [B, C, *spatial] or [B, *spatial].
- Missing `gt_seg` → `ValueError` naming `data.label_transform.emit_gt_seg: true`.
- Spatial shape mismatch → `ValueError`.

**Precision.** The body runs under `torch.autocast(device_type=pred.device.type, enabled=False)` with `pred.float()`.

**Per sample b.**

1. `ids = gt_seg[b].squeeze(0).long()` (after `round()` if the tensor is float).
2. `valid = ids > 0`. If `mask` is given, `valid &= mask[b].bool().all(dim=0)` (or the 3D mask itself).
3. **Object labels.**
   - If `recompute_ext`:
     - `comp = cc3d.connected_components(np.where(valid_np, ids_np, 0), connectivity=connectivity)`, computed on CPU;
     - object label = comp; parent id per component = the GT id at any of its voxels.
     - CC runs on the *valid* foreground, so masked or ignored voxels never bridge components.
   - Otherwise object label = `ids`.
4. **Gather.** `e = pred[b].reshape(D, -1).T[valid.flatten()]`, `lab = labels.flatten()[valid.flatten()]`, then `obj, inv = torch.unique(lab, return_inverse=True)`, `N = len(obj)`. If `N == 0` the sample is empty.
5. **Means.** `counts = torch.bincount(inv, minlength=N)`, `means = zeros(N, D).index_add_(0, inv, e) / counts[:, None]`.
6. **`L_int`:**
   - `h = clamp((e − means[inv]).abs().sum(1) − δv, min=0)^2`;
   - `per_obj = zeros(N).index_add_(0, inv, h) / counts`;
   - `L_int = per_obj.sum() / max(1, N)`.
7. **`L_ext`**, exact and memory-bounded:
   - Loop over row chunks `i0:i1` of size `pair_chunk_size`: `pd = (means[i0:i1, None, :] − means[None, :, :]).abs().sum(-1)` ([c, N]), `m = clamp(2δd − pd, min=0)^2`.
   - Keep entries that are off-diagonal and, when `recompute_ext`, have a different parent.
   - Accumulate `m[keep].sum()`.
   - If no ordered pair is kept anywhere, `L_ext = 0`; otherwise `L_ext = total / (N(N−1))`. The denominator includes excluded pairs, exactly as DeepEM.
   - Chunking bounds the forward peak to c×N×D, but autograd saves each chunk's diff tensor. If the memory stress test shows backward peak above budget, wrap each chunk in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. The coder measures and decides; parity tests must still pass.
8. **`L_nrm = means.abs().sum(1).mean()`.**
9. **Sample loss** = `α L_int + β L_ext + γ L_nrm`.

**Batch reduction.** Mean of the sample losses over non-empty samples. If every sample is empty, return `(pred.float() * 0).sum()` (a graph-connected zero). Policy: `[non-empty, empty]` equals the non-empty sample's loss.

**Documented deviations from DeepEM:**
- (a) averaged over the batch instead of element 0 only;
- (b) object voxels are always intersected with the valid mask (DeepEM's `generate_vecs` ignores the mask when `mask_background=True`); identical for an all-true mask;
- (c) CC is computed inside the loss, with default connectivity 26 (DeepEM computes it in a dataloader `recompute` augmentation, connectivity not stated in `mean.py`);
- (d) no background-as-object mode.

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

### 5. `dev/nisb/scripts/train_gate_watch.py` (new; `dev/` is git-ignored)

A small stdlib + tensorboard script, runnable as `python dev/nisb/scripts/train_gate_watch.py`.

**Args:**
- `--run-glob` (resolves the newest run dir under `save_path`);
- `--job-id`;
- `--baseline-logdir`;
- `--pairs "val_loss_term_0_weighted=val_loss_total"`;
- `--steps 4999,9999`;
- `--max-ratio 1.10`;
- `--stale-minutes 45`;
- `--poll-seconds 300`;
- `--dry-run` (never scancel).

**Behavior:**
- Poll the run's tfevents.
- At each gate step, compute `ratio = run_tag / baseline_tag` at the same step and print it.
- **If `ratio > max_ratio` at every gate step:**
  - `scancel <job-id>` (unless `--dry-run`);
  - print `GATE FAIL ... CANCELLED` to stdout and stderr;
  - exit code 3.
- **If all gate steps have been evaluated and at least one ratio passes:** print `GATE PASS` and exit 0.
- **If the job leaves RUNNING/PENDING** (`squeue` empty) before the gates finish: print `JOB ENDED EARLY` with the last step, exit 4.
- **If no new scalar arrives for `stale-minutes` while RUNNING:** print `STALE`, exit 5. It does not cancel.

### 6. Tests: `tests/unit/test_embedding_mean_loss.py` (new)

The file contains a trimmed verbatim copy of DeepEM `MeanLoss`, `create_mapping` and `compute_ext_matrix` as the oracle, with the source URL.

1. **Parity.**
   - Fixture: B=1, D=12, an 8×16×16 label with several ids, background, and one id split into two disconnected pieces; all-true mask; float64.
   - Compare loss and `pred.grad` against the oracle (`atol` 1e-6) for:
     - `recompute_ext` False;
     - `recompute_ext` True (oracle `splt` from `cc3d` with the same connectivity);
     - `delta_v` ∈ {0, 0.5};
     - `connectivity` ∈ {6, 26}.
   - Also a random 20-object label.
2. **Hand-computed denominator.**
   - D=1, three components A, B, C where A and B share a parent id; means set by construction (e.g. 0, 0.5, 5.0); `recompute_ext=True`.
   - Assert `L_ext = 2·max(0, 3 − |μ_A − μ_C|)^2 + 2·max(0, 3 − |μ_B − μ_C|)^2 over 6`, i.e. the A–B pair is excluded but still counted in the denominator N(N−1)=6.
   - With `recompute_ext=False` and 3 distinct ids, all 6 ordered pairs are included.
3. **Connectivity fixture.** A same-id object whose two voxel runs touch only diagonally:
   - `connectivity=6` → 2 components, excluded from push, not pulled together (assert that `L_int` equals the sum of two independent pulls);
   - `connectivity=26` → 1 component.
4. **Batching.**
   - B=2 equals the mean of the per-sample losses.
   - `[non-empty, empty]` equals the non-empty sample's loss.
   - An all-empty batch returns 0, requires grad, and backward gives zero grads.
5. **Edge cases.**
   - A single object → `L_ext = 0`.
   - All components under one parent → `L_ext = 0`.
   - `gt_seg=None` → `ValueError`.
6. **Background, ignore and mask** (both recompute settings): perturbing embeddings at voxels with `gt_seg == 0`, `gt_seg < 0`, or `mask == 0` changes neither the loss nor the grads elsewhere.
   - An ignore/masked voxel sitting between two same-id pieces must not connect them under `recompute_ext`: they stay two components.
7. **Mixed-precision gradients** (CUDA; skip otherwise).
   - A tiny `Conv3d(1→12)` under `torch.autocast("cuda", dtype=torch.float16)` produces pred scaled to about ±20.
   - Include objects of 1–3 voxels and one of about 2000 voxels.
   - Scale the loss by `torch.amp.GradScaler(init_scale=65536)`, backward, `scaler.unscale_(opt)`.
   - Assert all parameter grads are finite and `scaler.step` is not skipped (`scaler.get_scale()` unchanged after `update()`).
   - Assert the fp16-path loss equals the fp32 loss within 1e-4 relative.
8. **Memory stress** (CUDA; skip otherwise).
   - A fragmented label with N = max(4000, 2 × the max N observed in Verification step 4) components on a 128³ grid, D=12.
   - Report `torch.cuda.max_memory_allocated()` for forward+backward minus the input baseline.
   - Assert < 1.5 GB and that the loss matches a small-chunk run (`pair_chunk_size=64`) within 1e-5.
9. **Orchestrator integration** (modeled on the `gt_seg` pass-through tests in `tests/unit/test_loss_orchestrator.py`).
   - Setup: 18-ch pred, 6-ch labels with a bool `label_mask`, `gt_seg`, and the two-term config.
   - The embedding term receives `pred[:, 6:18]` unmodified (no −20 fill) plus `gt_seg`.
   - Backward from the embedding term alone yields zero grad on 0:6 and non-zero grad on 6:18.
   - The total loss is finite.

### 7. Label-pipeline test: `tests/unit/test_embedding_label_pipeline.py` (new)

Uses the real config-built training transform chain, built by `build_train_transforms` from the resolved `base_banis+.yaml` and `base_banis+_embed12.yaml` configs. Augmentations are disabled/identity so the post-augmentation label equals the input.

**Fixture:** a synthetic 1×138³ (or smaller-context equivalent) label with:
- two touching instances;
- a 1–2-voxel-thick process adjacent to another instance, which erosion=2 removes;
- background;
- a `-1` ignore slab.

**Assertions:**
- (a) The embed12 chain's emitted `gt_seg` equals the input label after the same leading spatial crop, exactly (ids and dtype values).
- (b) `label` and `label_mask` from the embed12 chain are identical (bitwise) to the banis+ chain, which uses global erosion=2.
- (c) On every voxel of the thin process (zero in the eroded label):
  - `gt_seg > 0`;
  - `label_mask[0:3]` is all true;
  - `EmbeddingMeanLoss` on a `requires_grad` pred yields non-zero grad at those voxels.
- (d) `label_mask` is false where the ignore slab touches, and those voxels get zero embedding grad.

## Files and Areas

**Code diff:**
- `connectomics/models/losses/embedding.py` (new)
- `connectomics/models/losses/build.py` (import + 1 registry line)
- `connectomics/models/losses/metadata.py` (+1 entry)
- `tests/unit/test_embedding_mean_loss.py` (new)
- `tests/unit/test_embedding_label_pipeline.py` (new)

**Outside the git diff; content must be pasted into `code_vN.md`:**
- `tutorials/neuron_nisb/base_banis+_embed12.yaml` (new; git-excluded)
- `slurm_jobs/nisb_banis_plus_embed12_train.sbatch` (new; git-ignored)
- `dev/nisb/scripts/train_gate_watch.py` (new; git-ignored)

**Coordinator only, after approval:**
- `dev/nisb/research_plan/prereg_nisb_embed12.md`
- `dw-research/projects/2026_nisb_base/logs/2026-09-15_embedding_model.md`

There are no changes to MedNeXt, the orchestrator, data transforms, inference, or decoding.

## Verification Plan

### Coder (code stage)

1. **Tests:** `conda run -n pytc pytest -q tests/unit/test_embedding_mean_loss.py tests/unit/test_embedding_label_pipeline.py` must all pass. CUDA-only tests are reported as run or skipped; if skipped, the planner runs them in step 6.
2. **Regression:** `conda run -n pytc pytest -q tests/unit/test_loss_orchestrator.py tests/unit/test_loss_functions.py tests/unit/test_malis_loss.py tests/unit/test_banis_reproduction_transforms.py`. Run once BEFORE editing to record pre-existing failures; require no new failures.
3. **Config:** `scripts/validate_tutorial_configs.py` (per-file mode if available) plus a resolved-config dump confirming:
   - `out_channels` 18;
   - the two loss terms and slices;
   - erosion 0;
   - affinity target erosion 2;
   - `emit_gt_seg` true.
4. **Dataloader probe** (CPU; the real lazy zarr train datamodule from the config; 20 batches). Report:
   - shapes and dtypes of `label` / `label_mask` / `gt_seg`;
   - fraction of `label_mask[0:3]` true;
   - per sample: #GT ids, #CC components (26-conn), max N;
   - CC and loss wall time per batch on CPU.
5. **Watchdog dry-run:** `train_gate_watch.py --dry-run --run-glob <seed43 run> --baseline-logdir <seed43 logs> --pairs val_loss_total=val_loss_total --job-id 0` → ratio 1.0 → `GATE PASS`. Also a fixture tfevents written with values ×1.2 → `GATE FAIL` (dry-run: no scancel).

### Planner (review_v0; GPU, 1 L40S/A100 via `srun`; all `main.py` commands `PREREG=`-prefixed)

6. **CUDA tests,** if the coder skipped them: mixed-precision gradients and memory stress.
7. **Trunk-gradient probe at init and after smoke.** Build the embed12 model and 4 real training batches. For each loss term separately:
   - backward (fp32);
   - `||g||` over all parameters except the final output conv.
   - Report `r = ||g_emb|| / ||g_aff||` at init and at the step-500 smoke checkpoint.

   **Pre-registered weight rule** (applied to r at step 500):
   - w = the largest of {1.0, 0.3, 0.1, 0.03} with `w·r ≤ 5`;
   - if w < 1.0, review returns NEEDS_CHANGES asking code_v1 to set that weight;
   - if even `0.03·r > 5`, block for a human.
8. **Matched smoke** (same node, `system.seed=43`, 1 GPU, `max_steps=500`, `val_check_interval=250`, `val_steps_per_epoch=5`, `save_path` under `outputs/_smoke/`) for `base_banis+.yaml` and `base_banis+_embed12.yaml`. Pass criteria:
   - no crash through train, val (including image logging with 18 pred vs 6 label channels) and checkpoint save;
   - both terms finite;
   - the embedding term at step 500 below its step-≤50 mean;
   - mean `train_loss_term_0_weighted_step` over steps 400–500 ≤ 1.10 × banis+'s `train_loss_total_step` over the same steps;
   - s/step ≤ 1.15 × banis+;
   - the checkpoint's AMP grad-scaler scale (Lightning precision-plugin state in the ckpt) ≥ 1024 and within 4× of banis+'s. A collapsing scale means repeated non-finite gradients or skipped steps.

   Any failure → NEEDS_CHANGES or BLOCKER with the numbers.

### Coordinator (after approval)

9. **Prereg** at `dev/nisb/research_plan/prereg_nisb_embed12.md` (header below), then `PREREG=dev/nisb/research_plan/prereg_nisb_embed12.md sbatch slurm_jobs/nisb_banis_plus_embed12_train.sbatch`.
10. **Liveness:** job RUNNING and the first scalar events present. Record job id, log paths and run dir in `run.md` and the dw-research log.
11. **Watchdog** as a background task, so its exit notifies the session: `train_gate_watch.py --job-id <id> --run-glob 'outputs/nisb_base_banis+_embed12_seed43/*' --baseline-logdir outputs/nisb_base_banis_v3_erosion2_seed43/20260905_212323/logs --pairs val_loss_term_0_weighted=val_loss_total --steps 4999,9999 --max-ratio 1.10`.
    - Baseline values: 0.5034, 0.4522.
    - Failure at both gates → auto-`scancel` and a loud report to the user.
    - `STALE` / `JOB ENDED EARLY` → report immediately.

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
   - The r ≤ 5 threshold is a judgment call, not a derived number.
2. **CPU `cc3d` in the loss.** It syncs `gt_seg` (2×128³) to host every step on every rank. Step 8's s/step gate bounds the cost; exceeding 15% blocks and routes to a follow-up that moves CC to the dataloader.
3. **Un-eroded `gt_seg`.** It includes boundary voxels between touching neurons, the hardest voxels. This was chosen because erosion=2 deletes thin processes, which is where the NISB prize is. It is an assumption, not a measured result.
4. **`recompute_ext=True` with 26-connectivity.**
   - The output FOV equals the input FOV (128³), so pulling in-patch-disconnected pieces of one neuron together is an unlearnable target.
   - Limitation: components that share a GT id are neither pulled together nor pushed apart. Identity supervision is lost between pieces whose connection lies outside the patch.
   - 26-connectivity avoids also losing it for diagonal-only in-patch contacts (the fixture in test 6.3 pins both behaviors).
   - GT-connected thin processes stay one component, so identity across *predicted* affinity gaps is still supervised.
5. **Checkpoint top-k monitors `val_loss_total`**, which is now BCE plus the embedding loss. Step checkpoints (every 50k) and `last.ckpt` are unaffected; evaluation uses step 200000, as for banis+.
6. **The global ±20 clamp** zeroes gradients beyond ±20. The norm term and δd=1.5 should keep embeddings far inside that range.
7. **Validation noise in the watchdog.** Different output channels change the init RNG draw, so seed 43 is not a fully matched run. The 10% tolerance and the requirement to fail at both gates are meant to absorb that.
8. **Git exclusions.** Config, sbatch and watchdog are git-excluded, so their full contents must be pasted into `code_vN.md` for review.
9. **Later, not blocking:** whether affinity should be decoded from the embedding (DeepEM `rsunet_embed_iso`) rather than predicted in parallel. Parallel is the minimal, non-regressing choice.

## Changes Since Previous Plan Version

Addresses every plan_v0_review finding:

- **[major] `mask_background=False` with `recompute_ext`:** the option is removed. Background is always excluded and negative ids are always ignored. CC runs on valid foreground only, so masked or ignored voxels cannot bridge components. Tests 6.6 cover background, ignore, masks and both recompute settings.
- **[major] Weak un-eroded `gt_seg` check:** replaced by test 7, which runs the real config-built transform chain and asserts:
  - `gt_seg` equals the post-augmentation label exactly;
  - `label` and `label_mask` are bitwise identical to the banis+ global-erosion chain;
  - the fixture includes thin processes, touching instances and an ignore slab.
- **[major] Affinity-mask routing vs thin supervision:** verified from code that the affinity mask depends only on geometry and `seg != -1`, never on foreground, and erosion preserves negative ids. Test 7(c) asserts thin voxels erased by erosion stay eligible and receive gradient.
- **[major] fp16 gradients:** test 6.7 checks finite parameter grads through fp16 autocast plus GradScaler (skip detection via the scale). The smoke (step 8) checks the checkpoint's grad-scaler scale against banis+.
- **[major] Interference gate:**
  - added a trunk-gradient probe with a pre-registered weight rule (step 7);
  - a matched banis+ smoke comparing the affinity term (step 8);
  - the watchdog script (§5) that auto-`scancel`s when affinity-only val loss (`val_loss_term_0_weighted` vs baseline `val_loss_total`: 0.5034 / 0.4522) is >1.10× at both 5k and 10k;
  - the monitor is a coordinator background task.
- **[major] N×N×D memory:** push term chunked by rows (`pair_chunk_size`), with optional per-chunk checkpointing; memory stress test 6.8 at N = max(4000, 2× the observed max) with a 1.5 GB bound and chunk-size parity; probe step 4 reports the observed N.
- **[minor] Recompute justification:** Risk 4 now states the limitation. Default connectivity changed 6 → 26; deterministic fixture test 6.3.
- **[minor] Batch reduction:** tests 6.4 cover mixed-empty and all-empty batches; test 6.2 is the hand-computed three-component case with a shared parent, checking the denominator.
- **Also:** the GPU gates now run in the planner's review_v0, so a weight change flows back through code_v1 instead of a post-review edit.

===== artifacts/plan_v1_review.md =====
# Plan v1 Review

## Summary

**Reviewer:** `codex exec --sandbox read-only` (coder role).
- Raw transcript: `state/plan_v1_review.review.raw.md`.
- Prompt: `state/plan_v1_review.prompt.md` (53,058 bytes).
- The repo was not mutated: `git diff` was identical before and after, and `git status` was empty.

**Result:** `READY: no`.
- Resolved: plan_v0 finding 1. Plan-level resolution of findings 2, 3 and 6, and of the minor connectivity finding.
- Partially resolved: finding 4 (fp16 gradients), finding 5 (gradient-ratio / weight rule), and the minor denominator fixture.
- New majors: a `squeeze` shape bug, fp32-vs-fp64 parity tolerance, and incomplete watchdog contracts.

All findings are reproduced below without softening.

## Findings

- **[minor] Finding 1 (background) resolved.** Removing `mask_background` makes background handling well defined, and computing CC on valid foreground stops masked/ignored voxels from bridging.
- **[minor] Findings 2 and 3 resolved at plan level.** Two refinements:
  - make the crop deterministic;
  - put the tested thin process inside the valid region of all three offsets, since boundary voxels cannot satisfy a universal mask assertion.
- **[major] Finding 4 only partially resolved.** Three problems:
  - Requiring no overflow on the first backward at scale 65536 with outputs near ±20 can reject normal AMP scale calibration.
  - A final checkpoint scale ≥1024 and within 4× of baseline does not prove updates were not repeatedly skipped.
  - Comparing separate fp16 and fp32 conv outputs to within 1e-4 is unreliable.
  - Needed: a bounded calibration period, then actual counts of skipped updates and non-finite gradients during the smoke; compare loss accuracy on the same quantized predictions.
- **[major] Finding 5 partially resolved.** The weight rule does not say:
  - how to aggregate over the 4 batches;
  - whether `g_emb` includes the configured weight (risk of double-counting `w`);
  - how to handle a zero or non-finite denominator.
  - It also needs to require a fresh matched smoke at the selected weight. `r` should be defined from unweighted losses.
- **[minor] Finding 6 resolved at plan level.** Add gradient parity across chunk sizes and with and without checkpointing; the stress comparison currently checks loss values only.
- **[major] Denominator fixture still broken.** Means 0, 0.5, 5.0 make every allowed push term zero, so dividing by 6 and dividing by 4 give the same result. Use e.g. 0, 0.5, 2.0 (numerator 6.5, `L_ext = 6.5/6`) and isolate the term with alpha=gamma=0.
- **[minor] Connectivity finding resolved.** Two wording fixes:
  - the fixture's "sum of two independent pulls" should be their **mean**;
  - "GT-connected thin processes stay one component" should say "connected within valid foreground".
- **[major] New shape bug in §1.** Unconditionally calling `gt_seg[b].squeeze(0)` drops Z for `[B,Z,Y,X]` input when Z=1. Normalize the channel dimension based on rank, and test singleton spatial dimensions in both layouts.
- **[major] New precision mismatch in test 6.1.** The implementation always casts to fp32 while the oracle runs in fp64 with atol 1e-6, which is not a reliable criterion. Either define absolute and relative tolerances suitable for fp32, or run both on identical fp32 inputs.
- **[major] Watchdog verification incomplete.** Gaps:
  - Training scalars can keep the watchdog fresh even if the gate validation tag or step never appears. It needs a deadline for missing gate data.
  - Reject non-finite values.
  - Pin the directory to the submitted job instead of picking the newest run.
  - Confirm `scancel` succeeded before reporting CANCELLED; dry-run must say "would cancel".
  - Add deterministic tests for cancellation, cancellation failure, missing tags, stale data, early termination, and pass/fail combinations.

## Questions

- How is the four-batch gradient ratio aggregated, and does each selected weight need a fresh smoke?
- What skipped-update threshold after calibration blocks the launch?
- What deadline and action apply when training progresses but a required validation gate never appears?

## Verdict

VERDICT: NEEDS_CHANGES

===== artifacts/plan_v2.md =====
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
