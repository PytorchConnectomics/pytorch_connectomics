You are the CCC plan reviewer (coder role) for a single incremental change in the PyTorch Connectomics (pytc) repository.

Do not edit files.
Review only the artifacts and diffs included in this prompt. Do not inspect other repository files.
Tag each finding as [minor] or [major].
Finish with exactly one line: READY: yes|no

This is the second plan review. Included: task.md, the previous plan (plan_v0), your previous review (plan_v0_review), and the revised plan (plan_v1). Focus on:
- whether each plan_v0_review finding is actually resolved by plan_v1 (say so per finding);
- new correctness issues introduced by plan_v1 (loss math/denominator/chunking/checkpointing, recompute_ext on valid foreground, batch policy, label-pipeline test feasibility, watchdog logic incl. auto-scancel semantics, GPU gate thresholds and weight rule);
- whether the plan is now executable by you as the implementer without further design decisions.
Severity: [major] = affects correctness, training validity, data integrity, or verification; [minor] = nits/follow-ups. Ambiguous severity is major. Do not re-raise resolved issues as major.

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

===== artifacts/plan_v0_review.md =====
# Plan v0 Review

## Summary

Reviewer: `codex exec --sandbox read-only` (coder role). Raw transcript: `state/plan_v0_review.review.raw.md`. Prompt: `state/plan_v0_review.prompt.md`, 22,218 bytes.

The reviewer returned `READY: no`: 6 major and 2 minor findings. All findings are reproduced below, not softened. After the review, `git status --short` in the repo was empty, so there was no mutation.

## Findings

- **[major] `mask_background=False` with `recompute_ext=True` is ill-defined.**
  - CC treats 0 as background and collapses negative ids to 0.
  - Admitting `ids >= 0` then makes all background voxels one object labelled 0.
  - Required: define the behavior or reject the combination, and test background, negative ids, and both recompute settings.
- **[major] The un-eroded `gt_seg` check is too weak.**
  - "More foreground than after another erosion" also holds for an already-eroded segmentation.
  - Required: test the full label-transform pipeline so that the emitted `gt_seg` equals the post-augmentation label exactly, while the affinity targets and masks match the global-erosion pipeline. Include thin processes and touching instances.
- **[major] Routing the affinity validity mask may remove thin-process supervision.**
  - `all()` over 3 affinity-mask channels ties embedding eligibility to the affinity target.
  - The plan does not show those channels stay valid on foreground that erosion removed.
  - Required: an integration assertion that thin-process voxels stay eligible and receive gradient, or a segmentation-based validity mask.
- **[major] The mixed-precision test covers the forward pass only.**
  - Required: finite gradients for predictions and parameters under fp16 plus gradient scaling, including large inputs and small objects.
  - The GPU smoke should detect skipped optimizer updates and non-finite gradients.
- **[major] The equal-weight launch has no actionable interference gate.**
  - Loss magnitudes say nothing about relative gradient strength.
  - Required: measure each term's gradient contribution to the shared trunk and compare affinity learning against a matched baseline during the smoke.
  - The 5k/10k check needs a concrete stop action and should compare affinity-only validation loss, because the new total is not comparable to the baseline.
- **[major] The N×N×12 push computation has no memory bound.**
  - Required: a deliberately fragmented stress case with peak GPU memory for forward and backward at realistic upper component counts.
  - Use chunked exact accumulation if needed, keeping the denominator and gradients.
- **[minor] The recompute justification overstates what is preserved.**
  - Components of the same GT id are neither pulled together nor pushed apart.
  - With 6-connectivity, GT connections that are diagonal only lose identity supervision, so "harmless" is unsupported.
  - Required: record the limitation and add a deterministic connectivity fixture.
- **[minor] Batch reduction needs more tests:** a batch with one empty and one non-empty sample, a fully empty batch, and a hand-computed case with three components where two share a parent (checks the excluded-pair denominator).

## Questions

- Is the 3-channel affinity mask guaranteed to keep supervision on un-eroded thin foreground, including crop boundaries and ignored regions?
- What concrete result stops the run if affinity learning degrades, and who monitors that gate?
- Will including background with recomputation be supported, or rejected explicitly?

## Verdict

VERDICT: NEEDS_CHANGES

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
