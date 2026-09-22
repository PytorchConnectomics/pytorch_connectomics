# Plan v1

## Summary

Build a **second-pass, seed-conditioned local re-segmentation model**: design report
Milestone 1, arm B3. Inputs are raw EM, a positive seed, the first-pass owner mask and
first-pass affinity advice; the output is a binary membership mask of the seeded neuron.

It is trained only on **sites where the first-pass BANIS+ model failed on training
data**. This follows the user's learning-dynamics premise: the first pass has learned large
objects, and the residual (thin gaps, necks, crumbs, contamination) is a harder distribution
that deserves its own model. We train on that residual, not on generic GT-centred crops.

This run delivers:
- a curation pipeline (first-pass decode on existing ROIs, fragment purity, typed error-site mining);
- a crop dataset with an explicit channel contract;
- a warm-started MedNeXt trainer;
- a preregistered mask-level gate evaluator on a **spatially disjoint A/B split** of held-out val seed100. A is used for all development decisions; B is evaluated once, only for the final full-training verdict.

Verification covers synthetic unit tests, correctness/readiness checks on real ROIs, montages,
an overfit test, a seed-only dependence test, and a bounded smoke training + evaluation **on A only**.

Out of scope, prepared but not executed: full training, ROI expansion, whole-volume NERL integration.

**Framing: this is a capability experiment with oracle error locations and trusted seeds**
(design §9: begin with reviewed/oracle locations). Site locations come from GT comparison, and
seeds are drawn from (first-pass fragment ∩ GT neuron). Results do not measure trigger recall,
and they do not measure performance with seeds available at deployment. A deployment-like variant
(seed from the owner fragment's interior without GT intersection) is reported separately.

SENSE v1 failed on seed101: NERL 0.604428 → 0.408730 (`dev/nisb/x2diag/rescue.log`). This design
corrects each recorded flaw:

| SENSE v1 flaw (evidence) | Correction in this plan |
|---|---|
| Seed fixed at crop centre, so the model never had to read it (`sense_train_offcenter.py:3-6`) | Seed inside a first-pass fragment near the site; crop centre jittered; seed-swap training examples; paired seed-only dependence gate (EM, M, A held fixed) |
| Training crops centred on GT skeleton voxels, not on real errors (`sense_prep_data.py:61-70`) | Sites mined from first-pass decode errors on train ROIs, plus preservation controls |
| Target selection and paste used GT nodes; paste relabelled every node inside the mask (`sense_rescue.py:169-181`) | Masks are scored only. Joins are scored per fragment using **purity**, not majority. Mixed fragments count as wrong joins, judged against the join break-even bar (`lesson_ladder_split` §1) |
| All evaluation and tuning on seed101 test | Train on train seeds; develop on val100-A; final verdict once on val100-B; seed101 untouched |
| EDT on foreground union, so a seed could land on background (`sense_train_offcenter.py:69`) | Seed ⊂ fragment ∩ GT k (per-instance EDT), never background; asserted and unit-tested |
| No augmentation, no validation | Flip/transpose augmentation with a symmetric advice channel; val-A curves by radius bin |

## Scope

In scope:

1. **First-pass decode and fragment purity on existing ROIs** (`dev/false_split/runs/20260905_seed_ensemble/{train_seed0_center,train_seed1_center,val_seed100_center}`).
   - Source: canonical `predictions/seed42.h5`, (3,1000,1000,450) float16 CXYZ.
   - Decode: `decode_affinity_cc(backend="numba", threshold=0.66, edge_offset=0)`, no mask.
   - Save the voxel segmentation, sample node labels at `anatomy.node_coords`, and compute a per-fragment GT voxel histogram over the whole ROI from `gt.h5`.
2. **Error-site mining** with explicit graph rules (Proposed Changes). Output: per-ROI `sites.npz` and `sites_summary.json`, containing diagnostics plus unseedable/excluded counts.
3. **Val A/B spatial partition** of `val_seed100_center`, frozen before any training. Frozen eval sites and seeds are written to `eval_sites_A.npz` / `eval_sites_B.npz`.
4. **Crop dataset:** channel contract, seed placement, seed-swap examples, bucket sampling with redistribution and bounded resampling, augmentation, channel dropout.
5. **Model builder:** MedNeXt-L from `tutorials/neuron_nisb/base_banis+.yaml`, `in_channels=4`, `out_channels=1`, strict warm start from the canonical 200k checkpoint.
6. **Trainer:** plain PyTorch; `--overfit N`; val-A curves by radius bin; checkpoints; JSON log.
7. **Gate evaluator** with B0 baseline, frozen sites, defined aggregation, zero-denominator handling, minimum support, and a guarded one-time B evaluation.
8. **Supporting files:** preregistration, sbatch launchers, README, unit tests, a review-delivery patch.

Out of scope (README follow-up commands only):
- whole-volume train inference;
- new ROI inference/preparation;
- full-length training;
- expand-then-correct (B4/B5);
- proposal→voxel edits and seed101 NERL (design Milestone 3);
- automatic triggers;
- changes to `connectomics/` package code.

## Proposed Changes

### Geometry and channel contract

All arrays are XYZ. Crops are (128,128,128) voxels = 1.15×1.15×2.56 µm. The **write region** is
the inner 64³ (offset 32).

| ch | name | construction |
|---|---|---|
| 0 | EM | source `data.zarr/img` (ROI origin offset applied), /255 |
| 1 | positive seed P | binary ball, radius (2,2,1) voxels, intersected with (owner fragment ∩ GT k); centred on the max per-instance-EDT voxel of that intersection within 24 XY / 12 Z voxels of the site anchor; single-voxel fallback |
| 2 | owner mask M | binary first-pass CC label containing the seed (may be contaminated or incomplete) |
| 3 | advice A | first-pass affinity, per-voxel mean of the 6 incident short-range edges (flip/XY-transpose symmetric) |

- **Training-time channel dropout:** M and A are each zeroed with p=0.15.
- **No image-validity channel.** Every crop lies inside the ROI with real context, so the channel would be constant 1. This is a deliberate, documented deviation from design §4.
- **Target:** `Y = (gt == k)` over the whole crop, including crop-disconnected parts. GT `seg` is dense, so every voxel is label-valid.

### Fragment purity (computed once per ROI over the whole ROI)

For each first-pass label f ≠ 0, over f's voxels:
- `h_f[g]`: voxel count per GT id g;
- `nodes_f[g]`: skeleton-node count per GT id g.

Let `nz = Σ_{g≠0} h_f[g]`. Classes:

| class | rule |
|---|---|
| `pure_k` | `h_f[k] ≥ 0.99·nz`, no skeleton node of any other GT, and `h_f[0] ≤ 0.5·|f|` |
| `mixed` | ≥2 GT ids exceed the 0.99 rule, or any other-GT skeleton node |
| `unannotated` | `h_f[0] > 0.5·|f|` (mostly extracellular) |

- Composition is known only inside the ROI; fragments touching an ROI face are flagged `roi_truncated`.
- Owners for gap/split/control sites must be `pure_k`. Contamination sites have `mixed` owners by definition.

### Error-site mining (per ROI; anchors must allow a full crop inside the ROI with anchor ≥ 8 voxels inside the write region)

Build node adjacency from `anatomy.edge_u/v` within each `local_skeleton_id`. Every node has a first-pass
label `L`, a GT id `k = global_gt_id`, and `radius_nm`.

- **Gap components.** Take connected components of the subgraph induced by nodes with `L=0`, restricted to edges joining two such nodes of the same local skeleton. This handles branching; a component can be a subtree.
  - Boundary labels are the distinct nonzero labels on nodes adjacent to the component.
  - Recorded per component: node count, length (internal edge length plus half of each boundary edge), minimum and median radius, boundary label set with purity.
  - Types:
    - `gap_bridge`: ≥2 distinct `pure_k` boundary labels.
    - `gap_terminal`: exactly 1 `pure_k` boundary label (missing tail or neck to an end).
    - `gap_unseedable`: no `pure_k` boundary label. Counted, never sampled.
  - Anchor: the component node nearest the component's length-weighted centroid.
  - Seed: from the boundary fragment adjacent to the anchor's nearest boundary node. For bridges this is randomised per draw between the two nearest fragments, so both sides are seen.
  - A component whose anchor cannot satisfy crop feasibility is recorded as `excluded_border`.
- **Split sites.** GT edges (u,v) of the same local skeleton with nonzero `L[u] ≠ L[v]`, both labels `pure_k`. The anchor is the edge midpoint node (u). Seed from `L[u]`.
- **Contamination sites.** For `mixed` labels: majority GT = most skeleton nodes; minority nodes are connected components of minority-GT nodes carrying that label. The anchor is the minority component's first node.
  - Seed is eligible only if the majority part (label ∩ GT majority) has ≥1 voxel inside the crop. The search window extends from 24/12 voxels to the whole crop; otherwise the site is `contamination_unseedable` (counted).
  - Target is the majority neuron.
- **Endpoint controls.** Degree-1 nodes of a local skeleton that are ≥64 voxels from every ROI face (not an ROI truncation), whose label is `pure_k` and equal to the neighbour's label.
- **Correct controls.** Random edges with both endpoints on the same `pure_k` label, stratified by radius bin, capped at 2× the error-site count per bin.
- **Deduplication.** Same type, same k, anchors within 16 XY / 8 Z voxels: keep the one with the smallest radius.
- **Radius bins** (kimimaro `radius_nm`): ≤45, 45–90, >90 nm. The site radius is the minimum over the component/edge.

### Val A/B spatial partition (frozen before training)

- Split `val_seed100_center` along X: A = crops whose x-extent ⊂ [0, 436), B = crops whose x-extent ⊂ [564, 1000). No A crop shares a voxel with a B crop; a 128-voxel buffer is excluded.
- The same GT neuron can appear in both halves in different voxels. This remaining dependence is documented; neuron-disjointness is infeasible in dense tissue.
- Train sites come only from train ROIs.
- Monitoring, checkpoint selection, τ calibration and all smoke evaluation use A.
- `evaluate.py --split B` requires `--final --frozen <json>` naming a checkpoint and τ. It appends a record to `runs/val_B_evaluations.jsonl` and refuses a second B run for the same frozen file unless `--allow-rerun` is set, which is recorded.

### Frozen evaluation sites

- `eval_sites_{A,B}.npz` stores, per site: crop origin (deterministic jitter from a fixed RNG seed), seed voxels, owner label, k, type, radius, and anchor node ids.
- A site is **measurable** only if its anchor nodes (gap component nodes, split edge endpoints, minority nodes, endpoint node) fall inside the write region with ≥8-voxel margin. Others are dropped and counted.
- A deployment-like duplicate of each site uses a seed from the owner fragment interior (max EDT of the fragment alone, no GT intersection). It is reported as a separate column.

### Sampling (training)

Default mixture:

| bucket | weight |
|---|---:|
| `gap_*` ≤45 nm | 40% |
| `gap_*` / `split` 45–90 nm | 15% |
| `split` ≤45 nm | 10% |
| `contamination` | 10% |
| `endpoint_control` | 10% |
| `correct_control` | 10% |
| paired seed-swap | 5% |

- **Empty buckets:** weight is redistributed proportionally to nonempty buckets and logged at init.
- **Per draw:** up to 20 attempts to find a jitter where the anchor stays in the write region and the seed stays in the crop. After that, zero jitter. If still invalid, the site is removed from the sampler and counted.
- **Crop jitter:** ±32 XY, ±16 Z voxels.
- **Seed-swap examples** (training and evaluation) hold EM, M and A fixed and change only P and the target. Given a crop containing neurons k and j, each with a `pure` fragment voxel in the write region:
  - input 1: P on k, target `gt==k`;
  - input 2: P on j, target `gt==j`;
  - both inputs use M = 0 (the M-dropout state, in distribution) and the same A.
- **Augmentation:** random X/Y/Z flips and XY transpose. EM intensity jitter is off in the overfit test.

### Model and training

- Build via the `dev/nisb/scripts/sense_model.py:build_sense_model` pattern (`setup_config(--mode test)`, so the real MedNeXt-L is built).
- Strict loader: stem channel 0 is copied from the checkpoint and channels 1–3 are zero. All non-head keys must match exactly; any other missing/unexpected/shape-mismatched key raises (replacing `load_state_tolerant`).
- Loss: BCE + soft Dice per crop. The empty-target convention is Dice = 1 when target and prediction are both empty. Seeded crops have nonempty targets by construction, so an empty target raises. Optional `--thin-weight` is off by default.
- Optimisation: AdamW lr 1e-4, wd 0.01, cosine with 500-step warmup, bf16, batch 2, grad clip 1.0.
- Checkpoints every N steps. Val-A metrics every M steps on the frozen A sites, capped at 200 sites for monitoring.

### Gate evaluator

Prediction threshold 0.5, no TTA, write region only. The primary aggregation is pooled
(micro) over nodes, fragments or pairs. Per-site macro means are secondary. 95% intervals come
from bootstrap over GT neurons (1000 draws, seeded).

1. **Missed-node recovery** by radius bin. Denominator: GT-k skeleton nodes in the write region not correctly owned by the first pass (`L=0`, or `L ≠ owner`). Numerator: those covered by the mask. B0 = 0 by definition. Also reported: total GT-k node recall with B0 = owner mask.
2. **Wrong-neuron inclusion:** `|mask ∩ gt∉{0,k}| / |mask|` (voxels) and other-GT nodes covered ÷ GT-k nodes in the region. B0 is computed from the owner mask.
3. **Fragment joins.**
   - Candidates are first-pass labels ≠ owner, ≠ 0, with ≥1 skeleton node in the write region. A join fires when the mask covers ≥ τ of the fragment's write-region voxels.
   - Truth:
     - `pure_k`: correct;
     - `pure_j` (j≠k) or `mixed` containing any non-k GT: wrong, and **counted as wrong even if its majority is k**;
     - `unannotated`: reported separately, excluded from precision;
     - `roi_truncated` candidates: included, flagged in the report.
   - PR over τ ∈ {0.3, 0.5, 0.7, 0.9}, split by candidate radius (≤45 / >45 nm). τ is selected on A only.
4. **Preservation controls** (`endpoint_control`, `correct_control`), reported as three separate components:
   - preservation = fraction of the owner's GT-k nodes in the write region covered by the mask;
   - unwanted expansion = `|mask \ gt==k| / |mask ∩ gt==k|`;
   - unwanted joins = number of fired joins.
   - Control success requires preservation ≥ 0.9, unwanted expansion ≤ 0.05 and 0 wrong joins. An empty mask has preservation 0 and fails.
5. **Seed-only dependence** (paired, M = 0): pair IoU(pred_k, pred_j), plus each prediction's own-target node recall in the write region. A pair is **valid** only if both predictions have own-target recall ≥ 0.5. Reported: fraction of valid pairs, and median IoU over valid pairs.

**Zero denominators and support.**
- A per-site metric with a zero denominator is excluded from macro averages and counted.
- A pooled metric with a zero denominator is reported as `NA`.
- Every metric reports n sites, n nodes/fragments/pairs, and the count of empty predictions.

**Gate verdict** is one of `PASS` / `FAIL` / `KILL` / `INSUFFICIENT_SUPPORT`, computed only for `--split B --final`. It is `INSUFFICIENT_SUPPORT` unless all of the following hold:
- ≥500 thin (≤45 nm) missed nodes over ≥50 sites;
- ≥100 thin candidate fragments with ≥30 correct joins at the selected τ;
- ≥50 control crops;
- ≥50 seed-swap pairs.

### Preregistration (`dev/ec_model/PREREG.md`, written before any cluster job)

- **hypothesis:** A seed-conditioned object-mask model trained on first-pass failure sites from train seeds recovers thin missed skeleton nodes with join precision high enough to be NERL-positive, which the first pass and SENSE v1 could not.
- **track:** false_split (+fill).
- **targets:** mask-level gate on val100-B, once, with the frozen checkpoint and τ chosen on A.
- **predicted_dNERL:** not claimed. The mask gate precedes edit integration; NERL targets will be deltas off 0.604428 only after edits exist.
- **cheap_gate** (full training verdict; the smoke run cannot pass or kill):
  - thin join precision ≥ 0.95 at thin join recall ≥ 0.30, with mixed fragments counted wrong;
  - thin missed-node recovery ≥ 0.20 with wrong-neuron voxel fraction ≤ B0 + 0.01;
  - control success ≥ 0.90;
  - ≥50% of seed-swap pairs valid, with median valid-pair IoU ≤ 0.30;
  - all support minima met.
- **kill_criterion:** thin join precision < 0.90 at every τ with recall ≥ 0.10, **or** < 25% of seed-swap pairs valid / median valid-pair IoU ≥ 0.70.
- **limitation:** oracle site locations and trusted seeds; the deployment-like seed column is informative, not gating.
- **cost:**
  - this run: ≤ 3 L40S GPU-hours plus CPU decode/mining on 3 ROIs;
  - full training later: ≤ 24 GPU-hours.

### Real-data correctness and readiness criteria (replacing v0's distribution-based pass bands)

Correctness (must pass):
1. **Alignment.** Decoded seg, affinity spatial shape and `gt.h5` shape are identical. At anatomy nodes, `gt.h5[node] == global_gt_id` for ≥ 98% of nodes, with the exact value reported.
2. **Label consistency.** For 100% of manifest sites, the first-pass label at the seed voxels equals the recorded owner, and GT at the seed voxels equals k. The decode is deterministic: a re-run gives the same sha256 of the label array.
3. **Independent re-derivation.** For 50 random sites per type, a separate voxel-array code path re-checks the type:
   - gap anchors: seg==0 and gt==k at the anchor voxel, with the recorded boundary labels;
   - split: labels differ at the endpoints;
   - contamination: the label's voxel histogram contains ≥2 GT ids;
   - controls: correct ownership.
4. **Visual audit.** ≥12 montage PNGs per type (train and val A) listed for the human.

Readiness for smoke training: ≥200 thin (≤45 nm) gap/split train sites and ≥50 thin val-A sites.

Diagnostics only (reported, not pass/fail): zero-node fraction, per-type counts per ROI, train-vs-val omission/split rates by radius bin, gap length distribution, unseedable/excluded counts, purity class counts.

### Review delivery contract (for dev/ files, which are gitignored)

Code submissions include `state/code_vN.patch` in the run folder. It concatenates `git diff --no-index /dev/null <file>` for every new or changed source file under `dev/ec_model/` (source and docs only, excluding `runs/`), so the coordinator can embed it in the code-review prompt.

`code_vN.md` lists, for every file: path, line count and sha256. It also gives every command run with exit code and key output, and artifact provenance (paths, sizes, Slurm job IDs).

Source is kept ≤ ~120 KB so patch plus artifacts fit the 200 KB review prompt. If it would not fit, the coordinator blocks instead of truncating.

## Files and Areas

All new files are under `dev/ec_model/` (gitignored research harness, like `dev/false_split/` and `dev/nisb/`). No tracked files change.

| File | Purpose |
|---|---|
| `dev/ec_model/README.md` | Goal, oracle-location framing, design and lesson links, SENSE v1 correction table, pipeline commands, outputs, follow-ups, dw-research path note |
| `dev/ec_model/PREREG.md` | Preregistration above |
| `dev/ec_model/common.py` | Paths, ROI list from `dev/false_split/manifest.json` with origins, (9,9,20) nm, bins, crop/write-region sizes, A/B bounds, IO helpers |
| `dev/ec_model/decode_rois.py` | Decode → `runs/<roi>/seg_cc066.h5`; node LUT → `node_labels.npz`; purity table → `fragments.npz`; sha256 + alignment report |
| `dev/ec_model/mine_sites.py` | Graph rules above → `sites.npz`, `sites_summary.json`; A/B freezing → `eval_sites_{A,B}.npz`; independent re-derivation check |
| `dev/ec_model/dataset.py` | Sampler (buckets, redistribution, bounded resampling), crops, seeds, channels, paired seed-swap, augmentation, dropout |
| `dev/ec_model/model.py` | Strict warm-started 4-channel MedNeXt |
| `dev/ec_model/train.py` | Training loop, `--overfit`, val-A curves by radius bin, checkpoints, JSON log |
| `dev/ec_model/evaluate.py` | Metrics, B0, deployment-like seed column, τ on A, guarded B run, support minima, verdict |
| `dev/ec_model/montage.py` | PNG montages of crops and predictions |
| `dev/ec_model/test_ec_model.py` | Synthetic unit tests |
| `dev/ec_model/run_ec.sh` | Stage launcher (`decode`, `mine`, `overfit`, `smoke`, `eval`, `train`) |
| `dev/ec_model/runs/` | Generated artifacts |

Reused unmodified: `decode_affinity_cc`, `build_model` / `setup_config`, `dev/false_split/manifest.json` and ROI GT preparations. `dev/false_split/infer.py` / `prepare.py` are referenced only in follow-up commands.

## Verification Plan

1. **Unit tests** (`pytest dev/ec_model/test_ec_model.py`, CPU, threads ≤2). Synthetic XYZ volumes with known GT, first-pass labels and skeleton graphs. Tests cover:
   - a straight gap (length and min radius);
   - a **Y-branch** with one missing branch (`gap_terminal`) and a missing junction segment (`gap_bridge`);
   - a gap touching the ROI face (`excluded_border`);
   - a split edge;
   - contamination with the majority seed outside the window (`contamination_unseedable`) and inside it;
   - dedupe of adjacent sites;
   - a clean control with no errors;
   - purity classes (pure, mixed with majority k, unannotated);
   - seed ⊂ fragment ∩ GT k and never background;
   - jittered crops with centre-coincidence < 5% over 200 draws;
   - target includes a crop-disconnected piece;
   - paired seed-swap keeps EM/M/A byte-identical and changes only P and target;
   - empty-bucket redistribution and bounded-resampling fallback;
   - advice invariance under flip/transpose;
   - strict stem inflation (channel 0 equals checkpoint, 1–3 zero, other keys exact) on a tiny model built from the same code path, or on the real state dict loaded on CPU;
   - evaluator metrics on hand-built masks (recall, inclusion, purity-based join precision with a mixed-majority-k fragment counted wrong, preservation fails on an empty mask, `NA` on zero denominators, `INSUFFICIENT_SUPPORT`);
   - the B-split guard refusing a second run.
2. **Real curation** on the 3 ROIs (`run_ec.sh decode`, `run_ec.sh mine`; Slurm CPU if needed): all four correctness criteria and the readiness counts above. Diagnostics printed in code_v0.md.
3. **Visual audit** PNG paths listed.
4. **Overfit test** (1 L40S, ≤45 min): 32 fixed train examples, ≥16 thin gap/split, no augmentation or dropout, 2k steps. Pass: mean Dice ≥ 0.90 overall and ≥ 0.85 on the ≤45 nm subset.
5. **Seed-only dependence after overfit:** 8 paired swaps built from the overfit crops. Pass: ≥6 valid pairs (own-target recall ≥ 0.5) and median valid-pair IoU ≤ 0.3.
6. **Smoke training + evaluation on A only** (1 L40S, ≤2 h). Report and val-A curves are attached and labelled **smoke, not a gate verdict**. Checks: no NaN, decreasing loss, B0 join recall = 0, `val_B_evaluations.jsonl` absent (B untouched).
7. **Launch readiness:** `bash -n run_ec.sh`; full-training and new-ROI commands in README, not submitted. Every sbatch is prefixed `PREREG=dev/ec_model/PREREG.md`.
8. **Baseline integrity and delivery:** `git status --short` equals the run-start status (empty). `state/code_v0.patch` is present and covers every source file in the table, with per-file sha256 in code_v0.md.

## Risks and Questions

1. **In-sample first pass, not out-of-fold** (design §6.1).
   - The canonical model trained on train seeds 0–4. Out-of-fold retraining costs ~53 L40S-hours per fold and is not justified before a gate.
   - Evidence in-sample errors are representative: thin (≤45 nm) positive-affinity recall is 29.9% / 33.2% on train seed0/1 ROIs vs 31.2% on val100 (`dev/false_split/audit_prior_run.md`). Thin structures are underfit, not memorised.
   - Mining diagnostics re-check this on decoded errors; val100 A/B stays held out from both models.
2. **Only 3 ROIs** (2 train, 1 val split into A/B). Support minima may not be met on B until more val ROIs exist; the verdict is then `INSUFFICIENT_SUPPORT`, not pass. Adding ROIs (infer ~9 min GPU + prepare 1–2 h CPU each) is a prepared follow-up.
3. **Evidence may be absent at gaps (I3).** The missed-node recovery gate and the kill criterion measure it.
4. **Mask gate ≠ NERL.** Split healing must be judged on the whole volume (`lesson_mesa_gates`). This gate only licenses building the edit layer; the join bar follows `lesson_ladder_split` (p* ≥ 0.95 carries 76% of the join prize).
5. **Oracle locations and trusted seeds** make results a capability ceiling. The deployment-like seed column and trigger recall (later milestone) bound the realisable fraction.
6. **Advice copying.** Channel dropout mitigates; an EM+seed-only `--no-advice` flag is provided but not run.
7. **FOV** of 128³ (1.15×1.15×2.56 µm) caps reachable gap length. The gap-length distribution is reported.
8. **Radius.** Kimimaro radius (val100 p10 = 40 nm) is comparable only within this pipeline.
9. **Purity inside the ROI only.** Fragments truncated by the ROI can hide contamination outside it; they are flagged in join reports.
10. **Paths.** `dev/` is gitignored (delivery contract above). The user-named `/projects/weilab/weidf/lib/dw-research` lags `/projects/weilab/dw-research`, whose newer lessons (`lesson_ladder_split`, `lesson_mesa_gates`, `lesson_wholevol_decomposition`, `lesson_thin_branch_coverage`, `lesson_erl_resolution`) were used.
11. **Later question (not blocking):** should the edit layer use the cc0.66 substrate (0.915 merge-safe ceiling) or cc0.75 (0.984; `lesson_wholevol_decomposition` Finding 3)? This run uses cc0.66 errors.

## Changes Since Previous Plan Version

Each plan_v0_review finding and question, with the change made:

1. **[major] Val half B not isolated.**
   - A/B is now a frozen **spatial** partition of val100 with a 128-voxel buffer, so no voxel is shared.
   - Monitoring, checkpoint selection, τ and all smoke evaluation use A only.
   - B runs once via `--split B --final --frozen`, logged in `val_B_evaluations.jsonl` with a rerun guard.
   - Remaining neuron-identity dependence across halves is documented.
2. **[major] Seed-swap confounded by M.** Paired seed-only swaps now hold EM, M (= 0) and A byte-identical and change only P and the target. Low IoU counts only on **valid** pairs (both own-target recall ≥ 0.5), so empty or failed predictions cannot pass.
3. **[major] Majority-GT truth.** Fragment purity classes (`pure_k` / `mixed` / `unannotated`, `roi_truncated` flag) are computed over the whole ROI. Mixed fragments are wrong joins even when their majority is k; unannotated are reported separately.
4. **[major] Eval sampling and undefined metrics.**
   - Eval sites and seeds are frozen, with a measurable-anchor requirement.
   - Pooled primary aggregation with neuron-level bootstrap CIs.
   - Zero-denominator rules (excluded/counted, pooled `NA`).
   - Explicit support minima with an `INSUFFICIENT_SUPPORT` verdict.
5. **[major] No-edit ignores preservation.** Controls report preservation, unwanted expansion and unwanted joins separately; success needs all three, and an empty mask fails.
6. **[major] Mining edge cases.**
   - Defined: gap components on branching graphs, boundary labels and bridge/terminal/unseedable types, ROI-border exclusion, contamination seed eligibility with an extended window, dedupe rule, endpoint-control truncation margin.
   - Sampler: empty-bucket redistribution and bounded resampling with a zero-jitter fallback.
   - Synthetic tests added for each.
7. **[major] Distribution-based pass criteria.** The 4–8% zero-node band and "every type in every ROI" are now diagnostics only. Correctness is alignment, label consistency, independent voxel-path re-derivation and visual audit; readiness is minimum eligible sample counts.
8. **[major] Review delivery for ignored files.** `state/code_vN.patch` (`git diff --no-index` per source file) plus per-file sha256, commands with exit codes and artifact provenance in code_vN.md. Source is budgeted to fit the review prompt, and the coordinator blocks rather than truncates.
9. **Question (GT-assisted seeds).** Yes, intentionally an oracle-location, trusted-seed capability experiment. This is stated in the Summary, PREREG and Risks, with a separate deployment-like seed column that is reported but not gating.
