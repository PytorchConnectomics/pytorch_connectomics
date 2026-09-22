You are the CCC plan reviewer (coder role) for a single incremental code change in the repository /projects/weilab/weidf/lib/pytorch_connectomics (PyTorch Connectomics). You will later implement this plan, so judge whether it is correct, executable, scoped, and verifiable.

Do not edit files.
Review only the artifacts and diffs included in this prompt. Do not inspect other repository files.
Tag each finding as [minor] or [major].
Major = affects correctness, scientific validity (leakage, wrong targets, invalid metrics), safety, data integrity, verification, or makes the result unsafe to judge. Ambiguous severity is major.

Output format (markdown):
## Summary
<2-5 sentences>
## Findings
- [major|minor] <finding with concrete fix>
## Questions
- <questions, or "None">
READY: yes|no

=== task.md ===
# CCC Task

a) read /projects/weilab/weidf/lib/dw-kb/reports/mednext/seed_conditioned_local_resegmentation_design.md. b) build a model to do the second pass. like the deep learning dynamics it learns big stuff quite fast, but the small things are hard to learn. c) use inference result on training data to curate training data for the second model. d) read previous lessons: /projects/weilab/weidf/lib/dw-research/projects/2026_nisb_base and code /projects/weilab/weidf/lib/pytorch_connectomics/dev/nisb as we had a failed attempt to learn the error correction model

Invocation: /ccc /projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/ec_model/ "<task above>" (defaults: p2-c2, normal, plan-code from default)
Repository root: /projects/weilab/weidf/lib/pytorch_connectomics

=== artifacts/plan_v0.md ===
# Plan v0

## Summary

Build a **second-pass, seed-conditioned local re-segmentation model** (design report
Milestone 1, arm B3: raw EM + positive seed + first-pass owner mask + first-pass
affinity advice → binary membership mask of the seeded neuron). Train it only on
**sites where the first-pass BANIS+ model failed on training data**. The premise is the
user's learning-dynamics observation: the first pass has already learned large
objects, and the residual (thin necks, gaps, crumbs, contamination) is a distinct and
harder distribution. The second model is trained on that residual, not on generic
GT-centred crops.

The run delivers four things:
1. A curation pipeline: first-pass decode on existing train/val ROIs, error-site mining against GT, and a typed site manifest.
2. A crop dataset with an explicit channel contract.
3. A warm-started MedNeXt trainer.
4. A preregistered mask-level gate evaluator on the held-out val seed100 ROI.

Verification covers synthetic unit tests, curation statistics on real ROIs, visual
montages, an overfit test, a seed-dependence test, and a bounded smoke training and
evaluation run. Full-scale training, ROI expansion and whole-volume NERL integration are
prepared as launch scripts but are **not** executed in this run.

It is designed against the recorded failure of SENSE v1 (seed101 NERL 0.604428 →
0.408730; `dev/nisb/x2diag/rescue.log`). Each v1 flaw has a specific correction:

| SENSE v1 flaw (evidence) | Correction in this plan |
|---|---|
| Seed fixed at crop centre; model never had to read it (`sense_train_offcenter.py:3-6`) | Seed placed inside a first-pass fragment near the error site; crop centre jittered; seed-swap examples where the same crop has a different target; seed-dependence gate |
| Training crops centred on GT skeleton voxels, not on real errors (`sense_prep_data.py:61-70`) | Sites mined from first-pass decode errors on train ROIs (omission gaps, split edges, contamination), plus no-edit controls |
| Target selection and paste used GT nodes; paste relabelled every node inside the mask (`sense_rescue.py:169-181`) | This run scores masks only (no paste). Join decisions are measured as precision/recall per fragment against the join break-even bar (`lesson_ladder_split` §1, p* ≥ 0.95) |
| All evaluation and tuning on seed101 test | Train on train seeds, calibrate/gate on val seed100, seed101 untouched |
| EDT on foreground union; could seed background (`sense_train_offcenter.py:69`) | Seed = voxels of (first-pass fragment ∩ GT neuron k), per-instance EDT, never background; assertion + unit test |
| No augmentation, no validation | XY/flip augmentation with symmetric advice channel; periodic val metrics by radius bin |

## Scope

In scope:

1. **First-pass decode on existing ROIs.**
   - Inputs: canonical seed42 affinities already saved for `train_seed0_center`, `train_seed1_center` and `val_seed100_center` (`dev/false_split/runs/20260905_seed_ensemble/<roi>/predictions/seed42.h5`, (3,1000,1000,450) float16 CXYZ, sha-verified canonical checkpoint).
   - Decode: affinity-graph CC with `decode_affinity_cc(backend="numba", threshold=0.66, edge_offset=0)`, no mask. This is the project baseline decoder (`dev/false_split/REGENERATED_SEED_INFERENCE.md`).
   - Save the voxel segmentation per ROI.
   - Sample per-node first-pass labels at the existing `anatomy.npz` nodes.
2. **Error-site mining** using the existing GT preparation per ROI (`gt.h5`, `anatomy.npz`: `node_coords` int XYZ local voxels, `edge_u/v`, `edge_len`, `radius_nm`, `global_gt_id`). Site types:
   - `gap`: maximal run of GT-skeleton nodes with first-pass label 0, bounded on at least one side by a nonzero fragment of the same GT neuron. Records gap length (nm) and minimum radius.
   - `split`: GT edge whose two endpoints carry different nonzero first-pass labels.
   - `contamination`: first-pass label containing nodes of ≥2 GT neurons; the site is placed at the minority nodes and seeded in the majority neuron.
   - `endpoint_control`: GT skeleton degree-1 node correctly owned by its fragment (true ending, no growth expected).
   - `correct_control`: random correctly owned edge, stratified by radius.
3. A **site manifest** per ROI (`sites.npz` + `sites_summary.json`): type, centre, seed fragment id, GT id, radius, gap length. The summary includes a **train-vs-val error-rate comparison by radius bin**, because the base model trained on these train seeds (see Risks).
4. A **crop dataset** implementing the channel contract below, plus seed-swap examples.
5. **Model builder:** MedNeXt from `tutorials/neuron_nisb/base_banis+.yaml` with `in_channels=4`, `out_channels=1`, warm-started from the canonical 200k checkpoint. Stem extra channels are zero-initialised, the head is new, and all other weights load strictly.
6. **Trainer** (plain PyTorch, following existing `dev/nisb` loops): `--overfit N` mode, periodic val metrics by radius bin (logged curves test "thin learned later" directly on the second model), checkpoints, JSON log.
7. **Gate evaluator** on val seed100 sites, including the B0 baseline (first-pass owner mask as the prediction).
8. **Preregistration file**, sbatch launchers, README, and unit tests.

Out of scope, documented in the README with follow-up commands only:
- Whole-volume inference on train seeds.
- New ROI inference and GT preparation beyond launch scripts.
- Full-length training.
- Expand-then-correct (B4/B5).
- Converting proposals into voxel edits and whole-volume seed101 NERL (design Milestone 3).
- Automatic triggers.
- Changes to core `connectomics/` package code.

## Proposed Changes

### Channel contract (all arrays stored XYZ, crops (128,128,128) voxels = 1.15×1.15×2.56 µm)

| ch | name | construction |
|---|---|---|
| 0 | EM | source `data.zarr/img`, /255 (same as base model) |
| 1 | positive seed P | binary ball, radius (2,2,1) voxels (≈18×18×20 nm), intersected with (owner fragment ∩ GT neuron k); placed at the max per-instance-EDT voxel of that intersection within 24 voxels (XY) / 12 (Z) of the site; single-voxel fallback for thin fragments; never touches background or another GT |
| 2 | owner mask M | first-pass CC label containing the seed, binary (may be contaminated or incomplete) |
| 3 | advice A | first-pass affinity, per-voxel mean of the 6 incident short-range edges (symmetric under flips and XY transpose, so augmentation does not need edge re-indexing) |

Training-time channel dropout: M and A are each zeroed independently with p=0.15, so the
model cannot just copy them. There is no image-validity channel: every crop lies fully inside
the ROI with real image context, and a constant-1 channel carries no information. This is
recorded as a deliberate deviation from design §4.

Target `Y = (gt == k)` over the whole crop, including parts not connected inside the crop
(design §3). GT `seg` is dense, so label validity is all voxels.

### Sampling (hard-residual emphasis; configurable)

Per-batch mixture defaults:
- 40% `gap` at radius ≤45 nm
- 15% `gap`/`split` at 45–90 nm
- 10% `split` at ≤45 nm
- 10% `contamination`
- 10% `endpoint_control`
- 10% `correct_control`
- 5% seed-swap

Seed-swap examples take any site crop containing ≥2 GT neurons with first-pass fragments, move the seed to a different neuron j, and set target `gt == j`. Radius comes from `anatomy.radius_nm`.

Crop centre is the site centre plus uniform jitter of ±32 (XY) and ±16 (Z) voxels, then clamped so the crop stays inside the ROI. The seed must remain inside the crop; if not, the example is re-drawn.

Augmentation: random flips on X, Y, Z and random XY transpose. EM intensity jitter (±10% contrast/brightness) is off by default in the overfit test.

### Model and training

- Build via the `dev/nisb/scripts/sense_model.py:build_sense_model` pattern (config through `setup_config(--mode test)` so the MedNeXt-L is actually built, not the schema-default BasicUNet), with a new strict loader:
  - Copy stem weight channel 0 from the checkpoint and set channels 1..3 to zero.
  - Load every other key strictly except the output head.
  - Fail on any other missing/unexpected/shape-mismatched key. This replaces `load_state_tolerant`, which silently drops mismatches.
- Loss: BCE + soft Dice averaged per crop. Optional `--thin-weight` (off by default) multiplies BCE by `1 + w·1[target ∧ gt-radius≤45nm]`, for a later ablation only.
- Optimiser: AdamW, lr 1e-4, wd 0.01, cosine schedule with 500-step warmup, bf16 autocast, batch 2, grad clip 1.0.
- Checkpoints every N steps; val metrics every M steps on a fixed val subset (≤200 sites).

### Gate evaluator (val seed100 sites only; prediction threshold 0.5; no TTA)

Metrics are computed inside the central write region (inner 64×64×64 voxels):

1. **Missed-node recovery by radius bin.** Among GT-k skeleton nodes not correctly owned by the first pass (label 0 or ≠ owner), the fraction covered by the mask. Also report total GT-k node recall, where B0 is the owner mask.
2. **Wrong-neuron inclusion.** Other-GT nodes covered ÷ GT-k nodes in region, and voxel fraction `|mask ∩ gt∉{0,k}| / |mask|`; compared with B0.
3. **Fragment join precision/recall.** Candidates are the other first-pass fragments (≠ owner, ≠ 0) with ≥1 skeleton node in the region. Join fires when mask covers ≥ τ of the fragment's in-region voxels; truth is fragment majority GT == k.
   - PR over τ ∈ {0.3, 0.5, 0.7, 0.9}, split by fragment radius (≤45 / >45 nm).
   - τ is selected on GT-neuron-disjoint half A of val sites; the metric is reported on half B.
4. **No-edit accuracy** on `endpoint_control` / `correct_control`: fraction of crops with no join fired and wrong-neuron voxel fraction < 0.01.
5. **Seed dependence** on seed-swap pairs: IoU(pred_k, pred_j) and Dice of each prediction to its own target.

Output: `eval_report.json` + `eval_report.md`, each metric with site counts.

### Preregistration (`dev/ec_model/PREREG.md`, written before any cluster job; project loop header)

- **hypothesis:** A seed-conditioned object-mask model trained on first-pass failure sites on train seeds recovers thin missed skeleton nodes with join precision high enough to be NERL-positive, which the first pass and SENSE v1 could not.
- **track:** false_split (+fill).
- **targets:** mask-level gate on val100 half B.
- **predicted_dNERL:** not claimed at this stage. The mask gate precedes whole-volume integration, and targets are deltas off realized 0.604428 only once edits are applied.
- **cheap_gate** (full training run, not the smoke):
  - join precision ≥ 0.95 at join recall ≥ 0.30 on ≤45 nm fragments;
  - missed-node recovery at ≤45 nm ≥ +0.20 absolute over B0 with wrong-neuron voxel fraction ≤ B0 + 0.01;
  - seed-swap median IoU ≤ 0.30.
- **kill_criterion:** join precision < 0.90 at every τ with recall ≥ 0.10 (joins uneconomic, `lesson_ladder_split`), or seed-swap median IoU ≥ 0.70 (SENSE v1 failure repeated).
- **cost:**
  - This run: ≤ 3 L40S GPU-hours (overfit + smoke) plus CPU decode/mining on 3 ROIs.
  - Full training later: ≤ 24 GPU-hours, gated by the prereg.

## Files and Areas

All new files are under `dev/ec_model/` (gitignored research harness, matching `dev/false_split/`, `dev/nisb/`). No tracked files change.

| File | Purpose |
|---|---|
| `dev/ec_model/README.md` | Goal, design-report and lesson links, SENSE v1 correction table, pipeline commands, outputs, follow-up launch commands, dw-research path note |
| `dev/ec_model/PREREG.md` | Preregistration header above |
| `dev/ec_model/common.py` | Paths, ROI list read from `dev/false_split/manifest.json` (+ origin offsets), resolution (9,9,20) nm, radius bins, crop/region sizes, small IO helpers (h5/zarr readers) |
| `dev/ec_model/decode_rois.py` | Decode `seed42.h5` → `runs/<roi>/seg_cc066.h5` (uint32 XYZ); node LUT at `anatomy.node_coords` → `runs/<roi>/node_labels.npz`; report zero-node fraction |
| `dev/ec_model/mine_sites.py` | Typed site mining and `sites.npz`/`sites_summary.json`, including train-vs-val radius-bin error rates |
| `dev/ec_model/dataset.py` | Site sampler, crop extraction, seed placement, owner/advice channels, target, seed-swap, augmentation, channel dropout |
| `dev/ec_model/model.py` | Warm-started 4-channel MedNeXt builder with strict stem-inflation loader |
| `dev/ec_model/train.py` | Training loop, `--overfit`, val curves by radius bin, checkpoints, JSON log |
| `dev/ec_model/evaluate.py` | Gate metrics, B0 baseline, τ selection on half A, report on half B |
| `dev/ec_model/montage.py` | PNG montages of sampled crops (EM with seed/owner/target contours, mid-Z slices) and of predictions |
| `dev/ec_model/test_ec_model.py` | Synthetic unit tests (below) |
| `dev/ec_model/run_ec.sh` | Stage launcher (`decode`, `mine`, `overfit`, `smoke`, `eval`, `train`) with pytc python, thread caps, `HDF5_USE_FILE_LOCKING=FALSE` |
| `dev/ec_model/runs/` | Generated artifacts (not source) |

Reused without modification:
- `connectomics.decoding.decoders.segmentation.decode_affinity_cc`
- `connectomics.models.build_model` / `connectomics.runtime.cli.setup_config`
- `dev/false_split/manifest.json` and ROI GT preparations
- `dev/false_split/infer.py` / `prepare.py` (only referenced by follow-up commands for new ROIs)

## Verification Plan

1. **Unit tests** (`pytest dev/ec_model/test_ec_model.py`, CPU; threads capped at 2), on synthetic XYZ volumes with known GT and first-pass labels:
   - Mining finds exactly the planted gap (length, min radius), split edge and contamination, and no errors on a clean control.
   - Seed voxels ⊂ owner fragment ∩ GT k, never background, and seed not at crop centre for jittered crops (over 200 draws, centre-coincidence fraction < 5%).
   - Target equals full `gt == k` including a planted crop-disconnected piece.
   - Seed-swap changes both seed and target to neuron j.
   - Advice channel is invariant under flip/transpose (augment-then-compute == compute-then-augment).
   - Stem inflation: channel-0 stem weights equal the checkpoint's, extra channels are zero, and non-head keys match exactly. Tested on a tiny MedNeXt built from the same code path, or on the real checkpoint's state dict loaded CPU-only if the tiny build is impractical.
   - Evaluator metrics on hand-built masks give known recall, precision, inclusion and IoU values.
2. **Real curation** on the 3 existing ROIs: `run_ec.sh decode` and `run_ec.sh mine`, via Slurm CPU jobs if the login node is inadequate. Pass criteria:
   - Zero-node fraction is 4–8% of anatomy nodes (seed101 whole-volume reference 5.8%).
   - Nonzero counts for every site type in every ROI.
   - `sites_summary.json` prints train-vs-val omission rates at ≤45 nm. Flag if a train ROI rate is < 0.8× val.
3. **Visual check:** `montage.py` writes ≥12 PNGs per site type from train and val for human audit; paths are listed in code_v0.md.
4. **Overfit test** (1 L40S, ≤45 min): 32 fixed examples, including ≥16 thin gap/split sites, no augmentation or dropout, 2k steps. Pass: mean Dice ≥ 0.90 on those examples and ≥ 0.85 on the ≤45 nm subset. Failure indicates a data/label/model bug and blocks smoke training.
5. **Seed dependence after overfit:** on ≥8 overfit seed-swap pairs, median IoU(pred_k, pred_j) ≤ 0.3.
6. **Smoke training + eval** (1 L40S, ≤2 h): train on train-ROI sites for a bounded step count, then `evaluate.py` on val100. The report and val curves are produced and attached to code_v0.md, explicitly labelled **smoke, not a gate verdict**. Check: no NaN, loss decreasing, B0 numbers are sane (e.g. B0 join recall = 0 by definition).
7. **Launch readiness:** `run_ec.sh train` passes `bash -n`; full-training sbatch and new-ROI infer/prepare commands are written in README but not submitted. Every sbatch in this run is prefixed `PREREG=dev/ec_model/PREREG.md`.
8. **Baseline integrity:** `git status --short` unchanged relative to run start (tracked files untouched). List every new ignored file for review.

## Risks and Questions

1. **In-sample first pass (not out-of-fold)** [design §6.1].
   - The canonical model trained on train seeds 0–4, so train-ROI errors are in-sample. Out-of-fold retraining costs ~53 L40S-hours per fold and is not justified before a gate.
   - Evidence that in-sample errors are representative: the canonical model's thin (≤45 nm) positive-affinity recall is 29.9% / 33.2% on train seed0/1 ROIs vs 31.2% on val100 (`dev/false_split/audit_prior_run.md`). Thin structures are underfit in-sample, not memorised.
   - The mining summary re-checks this on decoded errors; val100 remains fully held out for both models.
2. **Only 3 ROIs** (2 train, 1 val) exist now; site diversity is limited to ~1,600 local skeletons. Full training should wait for ≥10 train ROIs (infer ~9 min GPU + prepare 1–2 h CPU each). Commands prepared, not run.
3. **Evidence may be absent at gaps (I3).** If EM carries no signal at confident-low gaps, the model will fail missed-node recovery; the kill criterion measures this directly.
4. **Mask gate ≠ NERL.** Whole-volume split healing must be judged whole-volume (`lesson_mesa_gates`: crop gating is structurally invalid for NERL). The mask gate only licenses building the edit layer. Join economics set the precision bar (p* ≥ 0.95 carries 76% of the join prize).
5. **Advice copying.** The model may reproduce first-pass gaps from the owner/affinity channels; channel dropout mitigates, and an EM+seed-only ablation flag (`--no-advice`) is provided but not run here.
6. **Crop FOV** of 128³ (1.15×1.15×2.56 µm) matches the base model; gaps longer than the crop are out of reach. Gap-length distribution is reported in the mining summary to bound this.
7. **Kimimaro radius** from `anatomy.npz` (p10 = 40 nm on val100) differs from EDT radius used elsewhere; bins are comparable only within this pipeline.
8. **Repository/paths.** `dev/` is gitignored, so review needs explicit file contents. The user-named `/projects/weilab/weidf/lib/dw-research` copy lags `/projects/weilab/dw-research`, whose newer lessons (`lesson_ladder_split`, `lesson_mesa_gates`, `lesson_wholevol_decomposition`, `lesson_thin_branch_coverage`, `lesson_erl_resolution`) were used here.
9. **Question for later (not blocking):** after the mask gate, should the edit layer target the cc0.66 substrate (0.915 merge-safe ceiling) or cc0.75 (0.984 ceiling, `lesson_wholevol_decomposition` Finding 3)? This run trains on cc0.66 errors only.

## Changes Since Previous Plan Version

Initial plan.

=== end of artifacts ===
