You are the CCC plan reviewer (coder role) for a single incremental change in the PyTorch Connectomics (pytc) repository.

Do not edit files.
Review only the artifacts and diffs included in this prompt. Do not inspect other repository files.
Tag each finding as [minor] or [major].
Finish with exactly one line: READY: yes|no

Your job: judge whether plan_v0 below is correct, executable, and adequately verified, as the engineer who will implement it. Focus on:
- correctness of the proposed vectorized port of DeepEM MeanLoss (math, normalizers, recompute_ext semantics, gradients, edge cases) against the DeepEM description in the plan;
- whether the loss/config plumbing described (orchestrator pred_slice/target_slice routing, spatial_weight_arg="mask", gt_seg_arg, per-target affinity erosion vs global erosion, emit_gt_seg) is internally consistent as described;
- scientific/design risks that would waste a multi-day 4-GPU training run (loss scaling vs affinity BCE, un-eroded gt_seg, recompute_ext, fp16);
- whether the verification plan would catch the likely failure modes before launch;
- scope creep or missing pieces.
Severity: [major] = affects correctness, training validity, data integrity, or verification; [minor] = nits/follow-ups. Ambiguous severity is major.

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

===== artifacts/plan_v0.md =====
# Plan v0

## Summary

Add a 12-channel dense voxel-embedding output, trained with a DeepEM-style mean
(discriminative) loss, alongside the unchanged 6-channel BANIS+ affinity output in
pytorch_connectomics (pytc). Then launch a BANIS+-equivalent 200k-step training on
nisb-base with that model.

Background read for this plan:

- **Paper/wiki** (`dw-kb/papers/wiki/connectomics/kisuk_embedding*.md`, Lee, Lu, Luther & Seung, TMI 2021 / devoEM):
  - dense voxel embeddings learned by metric learning;
  - a metric graph (short + long-range affinities computed from embeddings);
  - Mutex Watershed;
  - mean-embedding agglomeration (MEA) to fix self-contact splits;
  - supp: MEA distance threshold θd in [1.0, 1.5).
- **DeepEM `deepem/loss/mean.py` (branch `feature/sr-zero-pad`), `MeanLoss`.** Per object `k` with voxel embeddings `e_i` and mean `μ_k`:
  - `L_int = mean_k mean_{i∈k} max(0, ||e_i − μ_k||_1 − δv)^2`
  - `L_ext = Σ_{k≠l, mext[k,l]} max(0, 2δd − ||μ_k − μ_l||_1)^2 / (N(N−1))`
  - `L_nrm = mean_k ||μ_k||_1`
  - `loss = α L_int + β L_ext + γ L_nrm`
  - Defaults: α=1, β=1, γ=0.001, δv=0, δd=1.5, `mask_background=True` (id 0 excluded).
  - `recompute_ext`: object = in-patch connected component of the GT id; components that share a GT id are *not* pushed apart.
  - It only reads batch element 0 (DeepEM trains with batch 1).
  - DeepEM's `embed_dim` default = 12.
  - Embedding→affinity for decode: `vec2aff`, `aff = max(0, (2δd − ||e1−e2||_1)/(2δd))^2`.
- **Prior NISB attempts** (`dw-research/projects/2026_nisb_base`, `pytc/dev/nisb`):
  - **No embedding model has ever been trained on NISB.** `research_plan/plan_contrastive_embeddings.md` proposed it (Phase 0 frozen-feature separability gate → Phase 1 embedding head → Phase 2 mean-embedding heal decider), status "Proposed", Phase 0 never run.
  - **Adjacent failures that constrain this design:**
    - (i) ABISS (embedding-lineage long-range affinity + mutex watershed) over-merges on NISB, NERL 0.0017 standalone (`lesson_abiss`). So this plan does **not** adopt the devoEM dense segmenter; the embedding is an auxiliary identity signal for a later heal decider.
    - (ii) MALIS, SDT/clDice, 2x-XY and banis2-occupancy training-side knobs did not heal thin splits (lessons 9, 10, 13, 16, 18, 19).
    - (iii) Removing the affinity erosion margin collapses cc3d NERL (`lesson_affinity_vs_occupancy`). **The affinity target must keep erosion=2.**
    - (iv) Erosion=2 erases thin processes from targets (`lesson_seg_filled_thin`).
    - (v) Invariant I5: thin-split fixes need connected per-instance identity. A plain scalar field does not clear it; an instance embedding is exactly per-instance identity.
  - **Process:** training launches require a prereg header (`dev/nisb/scripts/prereg_gate.sh` blocks `sbatch` without `PREREG=`).

## Scope

In scope:

1. **New loss `EmbeddingMeanLoss`.** A faithful, vectorized, batch-aware port of DeepEM `MeanLoss`, including `recompute_ext`. Registered in the pytc loss registry and metadata so it receives the raw GT segmentation via the existing `emit_gt_seg` → `gt_seg` path (same mechanism as `MalisLoss`).
2. **Unit tests:**
   - numerical and gradient parity with DeepEM's reference implementation;
   - batching, edge cases and mixed precision;
   - orchestrator integration on an 18-channel single-head output;
   - equivalence of per-target affinity erosion vs global erosion.
3. **Config `tutorials/neuron_nisb/base_banis+_embed12.yaml`:**
   - BANIS+ unchanged, except `out_channels: 18`;
   - loss routing (BCE on 0:6, embedding loss on 6:18);
   - erosion moved from global to the affinity target so `gt_seg` is un-eroded;
   - `emit_gt_seg: true`.
4. **Sbatch `slurm_jobs/nisb_banis_plus_embed12_train.sbatch`**, cloned from `slurm_jobs/nisb_banis_seed43_train.sbatch`.
5. **After code review approves (coordinator):**
   - GPU smoke run;
   - prereg file;
   - `sbatch` launch;
   - liveness check and early affinity-loss comparison vs the clean banis+ seed-43 run;
   - project log in dw-research.

Out of scope (follow-ups, recorded in the prereg, not built now):

- embedding→affinity decode (`vec2aff`), Mutex Watershed, mean-embedding agglomeration;
- the fragment-pair AUC evaluation;
- whole-volume test/decoding of the embedding channels;
- separate per-task head branches;
- warm-starting from the banis+ checkpoint.

## Proposed Changes

### 1. `connectomics/models/losses/embedding.py` (new): `EmbeddingMeanLoss(nn.Module)`

Constructor kwargs (DeepEM defaults): `alpha=1.0, beta=1.0, gamma=0.001, delta_v=0.0, delta_d=1.5, recompute_ext=True, connectivity=6, mask_background=True`.

Signature: `forward(pred, target, mask=None, gt_seg=None)`.

- `target` is ignored; it exists only because the orchestrator's `pred_target` call passes it.
- **Input shapes:**
  - `pred` [B, D, Z, Y, X]: embedding channels (routed by `pred_slice`);
  - `gt_seg` [B, 1, Z, Y, X] or [B, Z, Y, X]: any integer or integer-valued float dtype;
  - `mask` optional [B, C, Z, Y, X] or [B, 1, …].
- **Missing `gt_seg`:** raise `ValueError` with a message pointing at `data.label_transform.emit_gt_seg: true`.
- **Precision:** run under `torch.autocast(device_type=pred.device.type, enabled=False)` with `pred = pred.float()`. Squared L1 norms of 12 channels clamped to ±20 reach ~2e5, which overflows fp16.
- **Per sample b:**
  1. `ids = gt_seg[b].long()`.
  2. `valid = (ids > 0)` if `mask_background` else `(ids >= 0)`. Negative ids are always ignored (pytc's ignore sentinel).
  3. If `mask` is given: `valid &= mask[b].bool().all(dim=0)`, i.e. a voxel is valid only if every supervised channel of the routed mask is valid.
  4. If `recompute_ext`:
     - `comp = cc3d.connected_components(ids_np * (ids_np>0), connectivity=6)` on CPU;
     - object label = `comp`; parent id per component = the GT id;
     - components sharing a parent are excluded from the push matrix.
  5. Otherwise object label = `ids`.
  6. **Vectorize** over valid voxels:
     - `obj, inv = torch.unique(labels[valid], return_inverse=True)`, `N = len(obj)`;
     - `counts = bincount(inv)`, `sums = zeros(N, D).index_add_(0, inv, e)`, `means = sums / counts[:, None]`;
     - `L_int`: `d = (e − means[inv]).abs().sum(1)`, `h = clamp(d − δv, min=0)^2`, per-object mean via `index_add_` / counts, summed over objects, divided by `max(1, N)`;
     - `L_ext` (N > 1): `pd = (means[:, None] − means[None]).abs().sum(-1)`, `m = clamp(2δd − pd, min=0)^2`, keep entries where off-diagonal and (not `recompute_ext`, or parents differ), sum, divide by `max(1, N(N−1))`. The normalizer counts all ordered pairs, excluded pairs included, exactly as DeepEM. If no pair survives the mask, `L_ext = 0`.
     - `L_nrm = means.abs().sum(1).mean()`.
  7. Sample loss = `α L_int + β L_ext + γ L_nrm`.
- **Batch reduction:** mean of sample losses over samples with N ≥ 1. If no sample has an object, return `(pred * 0).sum()` so the graph stays connected under DDP.
- **Documented deviations from DeepEM:**
  - (a) batch > 1 is averaged instead of reading only element 0;
  - (b) object voxels are intersected with `valid` even when `mask_background=True`. DeepEM's `generate_vecs` ignores the mask in that branch; identical when the mask is all-true, which is the NISB case;
  - (c) CC is computed inside the loss on CPU instead of by a dataloader `recompute` augmentation.

### 2. Registration

- `connectomics/models/losses/build.py`: add `"EmbeddingMeanLoss": EmbeddingMeanLoss` to `_get_loss_registry`.
- `connectomics/models/losses/metadata.py`: add `"EmbeddingMeanLoss": LossMetadata("EmbeddingMeanLoss", spatial_weight_arg="mask", gt_seg_arg="gt_seg")`. `call_kind="pred_target"` is the default.
- `connectomics/models/losses/__init__.py`: `MalisLoss` is not exported there; touch only if needed.

**Why `spatial_weight_arg="mask"`:** with `None`, the orchestrator `masked_fill`s pred with −20 using a mask whose channel count (3) does not match the 12 embedding channels. With `"mask"`, pred passes through unmodified (only the global ±20 clamp) and the loss receives the routed validity mask.

### 3. `tutorials/neuron_nisb/base_banis+_embed12.yaml` (new)

`_base_: [base_banis+.yaml]`. Fields:

- `experiment_name: nisb_base_banis+_embed12`, `save_path: outputs/nisb_base_banis+_embed12`.
- `default.model.out_channels: 18`. Single head, no `heads:` dict. One 1×1 output conv emitting 6 + 12 channels, equivalent to DeepEM's per-task 1×1 output convs off a shared trunk. The affinity path is architecturally identical to banis+.
- `default.model.loss.losses`. The list is replace-not-merge, so BCE is restated:
  - `PerChannelBCEWithLogitsLoss`, weight 1.0, `pred_slice "0:6"`, `target_slice "0:6"`, kwargs `auto_pos_weight: true, max_pos_weight: 10.0`;
  - `EmbeddingMeanLoss`, weight 1.0, `pred_slice "6:18"`, `target_slice "0:3"`, kwargs `alpha 1.0, beta 1.0, gamma 0.001, delta_v 0.0, delta_d 1.5, recompute_ext true`.
  - `target_slice "0:3"` is used only to route the short-range affinity validity mask. Target values are ignored; comment this in the YAML.
- `default.data.label_transform`:
  - `erosion: 0`, `emit_gt_seg: true`;
  - `targets: [{name: affinity, kwargs: {offsets: ["1-0-0","0-1-0","0-0-1","10-0-0","0-10-0","0-0-10"], affinity_mode: banis, erosion: 2}}]`.
  - Per-target erosion calls the same `seg_erosion_instance(label, 2)` on the same post-augmentation label as the global `SegErosionInstanced`, so the affinity target should be bit-identical to banis+ (unit test below). `gt_seg` becomes the un-eroded label, so thin processes stay supervised in the embedding.
- `train.monitor.logging.scalar.loss`: add `train_loss_term_0_weighted`, `train_loss_term_1_weighted` (as in `base_banis+_sdt_1head.yaml`).
- **Everything else inherited:**
  - MedNeXt-L/k3, 128³, batch 2/GPU, AdamW 1e-3, cosine 200k, EMA 0.999, 16-mixed, clip 1.0;
  - `inference.model.select_channel: [0,1,2]`, so test/decoding sees only short-range affinity and is unaffected by the extra channels.
- **Note:** `tutorials/neuron_nisb/*.yaml` is excluded via `.git/info/exclude`, so this file will not appear in `git status`. Code-stage summaries and review must include its full content.

### 4. `slurm_jobs/nisb_banis_plus_embed12_train.sbatch` (new)

- Clone of `nisb_banis_seed43_train.sbatch`: 4 GPUs, `long`, 32 CPUs, 160G, 5-day limit, same excludes and env.
- Unique `--job-name=bemb12_train` and `MASTER_PORT=29712`.
- Logs: `slurm_jobs/logs/nisb_banis_plus_embed12_train_%j.{out,err}`.
- Command: `python scripts/main.py --config tutorials/neuron_nisb/base_banis+_embed12.yaml system.seed=43 experiment_name=nisb_base_banis+_embed12_seed43 save_path=outputs/nisb_base_banis+_embed12_seed43`.
- **Why seed 43:** it pairs with the clean banis+ seed-43 run `outputs/nisb_base_banis_v3_erosion2_seed43/20260905_212323` (job 2968810, same current code/config lineage, completed 200k in 2d05h), whose tfevents give the matched-step affinity BCE curve.
- **Git:** `slurm_jobs/` is git-ignored (`.gitignore:165`); same note as the config.

### 5. Tests: `tests/unit/test_embedding_mean_loss.py` (new)

The test file contains a verbatim (trimmed) copy of DeepEM `MeanLoss` + `create_mapping` + `compute_ext_matrix` as the reference oracle, with the source URL.

- **Parity:**
  - B=1, D=12, random 16×16×8 label with several ids, background, and one id split into two disconnected in-patch pieces; mask all-true.
  - Compare loss value and `pred.grad` against the reference (fp64 or fp32, `atol` ~1e-5).
  - Cases: `recompute_ext` False and True (reference gets `splt` from `cc3d` 6-connectivity); `delta_v` 0 and 0.5.
- **Batch:** B=2 loss == mean of the two single-sample losses.
- **Edge cases:**
  - all background → loss is exactly 0, finite, `requires_grad` and backward works;
  - one object → `L_ext = 0`;
  - all objects share one parent under `recompute_ext` → `L_ext = 0`;
  - `gt_seg=None` → `ValueError`.
- **Mask/ignore:** voxels with `mask=0` or `gt_seg<0` do not affect the loss (perturbing their embeddings leaves loss and other grads unchanged).
- **Mixed precision:** `pred.half()` on CUDA if available (skip otherwise), and large-magnitude inputs (±20) → finite, equal to the fp32 result within tolerance.
- **Orchestrator integration** (modeled on the existing `gt_seg` pass-through tests in `tests/unit/test_loss_orchestrator.py`):
  - 18-ch pred, 6-ch labels with a bool `label_mask`, `gt_seg`, the two-term config above;
  - total loss is finite;
  - the embedding term receives `pred[:, 6:18]` unmodified (no −20 fill) and `gt_seg`;
  - backward from the embedding term alone yields zero grad on channels 0:6.
- **Erosion equivalence** (add to `tests/unit/test_banis_reproduction_transforms.py` or the new file): on a random label with touching instances,
  - `MultiTaskLabelTransformd` affinity (banis mode, 6 offsets) with target kwarg `erosion: 2` on the raw label
  - equals `SegErosionInstanced(tsz_h=2)` followed by the same transform without the kwarg;
  - compare both `label` and `label_mask`.

## Files and Areas

- `connectomics/models/losses/embedding.py` (new)
- `connectomics/models/losses/build.py` (registry, +1 line and import)
- `connectomics/models/losses/metadata.py` (+1 entry)
- `connectomics/models/losses/__init__.py` (MalisLoss is not exported there; touch only if
  needed for imports/tests)
- `tests/unit/test_embedding_mean_loss.py` (new)
- `tests/unit/test_banis_reproduction_transforms.py` (erosion-equivalence test, or folded into the new file)
- `tutorials/neuron_nisb/base_banis+_embed12.yaml` (new; git-excluded locally)
- `slurm_jobs/nisb_banis_plus_embed12_train.sbatch` (new; git-ignored)
- Post-approval, coordinator only, outside the code diff:
  - `dev/nisb/research_plan/prereg_nisb_embed12.md`;
  - `dw-research/projects/2026_nisb_base/logs/2026-09-15_embedding_model.md`.

No changes to MedNeXt, the orchestrator, data transforms, inference, or decoding code.

## Verification Plan

**Coder (code stage):**

1. `conda run -n pytc pytest -q tests/unit/test_embedding_mean_loss.py` plus the erosion-equivalence test: all pass.
2. Regression: `conda run -n pytc pytest -q tests/unit/test_loss_orchestrator.py tests/unit/test_loss_functions.py tests/unit/test_malis_loss.py tests/unit/test_banis_reproduction_transforms.py`: no new failures vs baseline. Run the same command once BEFORE editing to record pre-existing failures.
3. Config validation: `conda run -n pytc python scripts/validate_tutorial_configs.py` (or its per-file mode) on `base_banis+_embed12.yaml`, plus a config-load/preflight check that resolves the merged config. Confirm:
   - `out_channels 18`;
   - 2 loss terms with the stated slices;
   - `erosion 0`, affinity target `erosion 2`;
   - `emit_gt_seg true`.
4. **Dataloader probe (CPU, no training):** build the train datamodule from the config, draw 2 batches, and report:
   - `label` shape [2,6,128,128,128], `label_mask` all-true fraction, `gt_seg` shape/dtype;
   - number of GT ids and in-patch CCs per sample;
   - that `gt_seg` has more foreground voxels than `seg_erosion_instance(gt_seg, 2)` (un-eroded);
   - `EmbeddingMeanLoss` on random 12-ch pred for that batch: value and wall time (CPU and GPU if available).
5. `--fast-dev-run` of the config if a GPU is reachable from the coder environment. If not, report "not run" explicitly; the coordinator runs it.

**Coordinator (after review approves, before `sbatch`):**

6. **GPU smoke on a single L40S/A100 via `srun`**, with real 128³ MedNeXt-L. `PREREG=` prefixed, 1 GPU:
   - `train.optimization.max_steps=300`, `val_check_interval=100`, `val_steps_per_epoch=5`, separate `save_path` under `outputs/_smoke/`.
   - Pass criteria:
     - no crash through training, validation (including image logging with 18 pred vs 6 label channels) and checkpoint save;
     - both loss terms finite; the embedding term decreasing from its ~9 initial value (all means ≈0 → push term ≈ (2δd)^2);
     - s/step within ~15% of the same smoke with `base_banis+.yaml` on the same node (100 steps).
7. Write the prereg (below), then `PREREG=dev/nisb/research_plan/prereg_nisb_embed12.md sbatch slurm_jobs/nisb_banis_plus_embed12_train.sbatch`.
8. **Liveness:** job RUNNING, first `train_loss_term_*` lines by step ~200, record job id / log / run dir.
9. **At the first validation (step 5k), and again at 10k:**
   - compare `train_loss_term_0` (affinity BCE) and val loss with the seed-43 banis+ tfevents at the same steps;
   - flag if affinity BCE is >10% worse. That means the embedding term is starving the affinity; the fix is to lower the embedding loss weight, which needs a human decision.

**Prereg header (loop.md §2):**

- hypothesis: a 12-ch mean-loss embedding head on the shared trunk learns per-instance identity that survives thin affinity gaps, without degrading the 6-ch affinity.
- track: `false_split`.
- targets: 0.772 branch-merge oracle via a later mean-embedding heal decider.
- predicted_dNERL: +0.00 ± 0.01 off 0.601 (seed-43 cc3d@0.66) from the affinity channels alone; any gain requires the follow-up decider.
- cheap_gate: user-directed launch. The loop's Phase 0 frozen-feature gate was not run; recorded honestly. In-run gates: smoke (6) and affinity BCE parity at 5k/10k (9).
- kill_criterion:
  - (a) affinity cc3d NERL on the center chunk (tile_1_1_1) at 200k more than 0.02 below seed-43 banis+ → embedding hurts the trunk;
  - (b) mean-embedding fragment-pair AUC on the heal/veto candidates below the 0.77 LSD ceiling → no identity signal; kill the embedding-decider branch.
- cost: retrain, 4 GPUs × ~2.5 days.

## Risks and Questions

1. **Trunk interference.** Initial embedding loss ≈ 9 vs affinity BCE < 1 at equal weight 1.0 (DeepEM's per-task default, but DeepEM's relative scaling vs this BCE is unverified). The embedding gradient may dominate early and slow or degrade affinity. Mitigation: step 9 comparison. Alternative: an embedding weight of 0.1, not chosen now because it deviates from DeepEM without evidence.
2. **CPU `cc3d` inside the loss** forces a GPU→CPU sync of `gt_seg` (2×128³) each step on each rank. Expected ~tens of ms vs ~1 s/step; step 6 measures it. If it is >15%, move CC to the dataloader (a follow-up, not now).
3. **Un-eroded `gt_seg` puts boundary voxels between touching neurons in the pull/push**, the hardest voxels, whereas eroded would drop them. Chosen because erosion=2 deletes thin processes, the NISB prize (lessons T and 18). This is an assumption about what helps the heal track, not a measured result.
4. **`recompute_ext=True`** (not DeepEM's CLI default).
   - Chosen because the output FOV equals the input FOV (128³), so pulling in-patch-disconnected pieces of one neuron together is an unlearnable target.
   - It does not remove the identity signal across *predicted* affinity gaps: GT-connected thin processes remain one component.
   - 6-connectivity may split diagonal-only contacts into separate components; these are only exempted from push, which is harmless.
5. **Checkpoint top-k monitors `val_loss_total`,** which now sums BCE and the embedding loss. Step checkpoints (every 50k) and `last.ckpt` are unaffected; evaluation uses step=200000 as for banis+.
6. **Global ±20 clamp in the orchestrator** zeroes gradients beyond ±20. The norm regularizer and δd=1.5 keep embeddings well inside that range, but it is a silent ceiling.
7. **Validation image logging** (`channel_mode: all`) with 18 pred vs 6 label channels is untested. Step 6 covers it; if it breaks, restrict image channels in the config (config-only fix).
8. **Tutorial configs and possibly `slurm_jobs/` are git-excluded.** The review must be given their contents explicitly.
9. **Open question for later, not blocking:** whether the embedding should feed the affinity (DeepEM `rsunet_embed_iso` decodes task heads from the embedding) instead of being a parallel output. Parallel is chosen as the minimal, non-regressing design.

## Changes Since Previous Plan Version

Initial plan.
