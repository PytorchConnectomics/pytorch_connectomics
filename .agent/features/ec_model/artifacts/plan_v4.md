# Plan v4

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

### Node alignment exclusion (before purity and mining)

A skeleton node is `aligned` iff `gt.h5[node] == global_gt_id[node]`. Misaligned nodes are
removed from all graphs before mining, and edges touching them are dropped, so components
are computed on the aligned subgraph.
- Misaligned nodes are never anchors, seeds, metric nodes or purity evidence.
- They are counted by radius bin and by split (train / val-A / val-B).
- Hard stop: the overall misaligned fraction per ROI must be ≤ 2%; above that, stop and report as an alignment bug.
- Every node that participates in mining, purity or metrics is aligned by construction.

### Fragment purity (computed once per ROI over the whole ROI; rules applied in order, first match wins)

For each first-pass label f ≠ 0, over f's voxels: `h_f[g]` is the voxel count per GT id g;
`nodes_f[g]` is the aligned skeleton-node count per GT id g. Let `nz = Σ_{g≠0} h_f[g]`.

1. `nz == 0` → **`unannotated`**.
2. `d = argmax_{g≠0} h_f[g]`, ties broken by the lowest GT id; `share = h_f[d] / nz`.
3. If `share ≥ 0.99` and `nodes_f[g] == 0` for every `g ≠ d`:
   - `h_f[0] > 0.5·|f|` → **`bgdom_pure_d`** (dominant neuron d, mostly extracellular voxels);
   - otherwise → **`pure_d`**.
4. Otherwise → **`mixed`**. This covers 60/40 splits, any other-GT node, and background-heavy fragments carrying two neurons.

The classes are exhaustive and mutually exclusive. Additional flag: `roi_truncated` if the fragment touches an ROI face (composition outside the ROI is unknown).

Owners:
- gap/split/control sites must be `pure_k`;
- contamination owners are `mixed` with dominant d = k, where the target is k.

Unit tests: 60/40 → mixed; 99/1 with no other nodes → pure; 98.9/1.1 → mixed; all-background → unannotated; background-heavy 99.5% d with no other nodes → bgdom_pure; background-heavy with 2 GT nodes → mixed; tie in dominant id → lowest id.

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
  - Anchor: the component node nearest the component's length-weighted centroid, using internal edge midpoints weighted by length.
    - Singleton component (no internal edge): the anchor is the node itself and length = half the sum of its boundary edge lengths.
    - Isolated node with no edges at all: `gap_unseedable`.
    - Distance ties: lowest node index.
  - Seed: from the boundary fragment whose boundary node is nearest the anchor by graph hop count (ties: lowest node index, then lowest label id).
    - For `gap_bridge`, the two nearest distinct `pure_k` boundary labels (same tie rules) are both stored; the training sampler picks one per draw.
    - Frozen eval sites fix side 0.
  - A component whose anchor cannot satisfy crop feasibility is recorded as `excluded_border`.
- **Split sites.** GT edges (u,v) of the same local skeleton with nonzero `L[u] ≠ L[v]`, both labels `pure_k`.
  - Orientation: u is the endpoint with the lower node index.
  - **The anchor is node u itself** (a skeleton node, not a geometric midpoint).
  - Seed from `L[u]`; `L[v]` is stored as the far-side label.
  - Unit test: the anchor equals the lower-index endpoint when edge direction is reversed in the input.
- **Contamination sites.** For `mixed` labels with dominant d (voxel rule above; k = d): minority nodes are aligned nodes with GT ≠ d carrying that label. Their connected components (same graph rules) form sites. The anchor is the component's lowest-index node.
  - `mixed` labels with no minority node (voxel-only mixing) produce no contamination site; they are counted.
  - Seed is eligible only if the dominant part (label ∩ GT d) has ≥1 voxel inside the crop. The search window extends from 24/12 voxels to the whole crop; otherwise the site is `contamination_unseedable` (counted).
  - Target is the dominant neuron d = k.
- **Endpoint controls.** Degree-1 nodes of a local skeleton that are ≥64 voxels from every ROI face (not an ROI truncation), whose label is `pure_k` and equal to the neighbour's label.
- **Correct controls.** Random edges with both endpoints on the same `pure_k` label, stratified by radius bin, capped at 2× the error-site count per bin.
- **Deduplication.** Same type, same k, anchors within 16 XY / 8 Z voxels: keep the one with the smallest radius.
- **Radius bins** (kimimaro `radius_nm`): ≤45, 45–90, >90 nm. The site radius is the minimum over the component/edge.

### Val A/B spatial partition (frozen before training)

- Split `val_seed100_center` along X: A = crops whose x-extent ⊂ [0, 436), B = crops whose x-extent ⊂ [564, 1000). No A crop shares a voxel with a B crop; a 128-voxel buffer is excluded.
- The same GT neuron can appear in both halves in different voxels. This remaining dependence is documented; neuron-disjointness is infeasible in dense tissue.
- Train sites come only from train ROIs.
- Monitoring, checkpoint selection, τ calibration and all smoke evaluation use A.

**Confirmatory B guard**, keyed to the experiment and the B manifest, not to a file name:

- `evaluate.py --split B` requires `--final --frozen <json>`. The JSON must contain:
  - checkpoint path and its **content sha256** (verified against the file);
  - τ;
  - sha256 of the dataset/channel/preprocessing config dict;
  - sha256 of `eval_sites_B.npz`;
  - sha256 of `evaluate.py`.
  All are verified before any B data is loaded.
- **Reservation before any B data or inference.** After hash verification and before loading B sites or running the model, the evaluator exclusively creates `runs/val_B_access.json` (`os.open` with `O_CREAT|O_EXCL`), keyed by the `eval_sites_B.npz` sha256.
  - The record holds the full frozen tuple, `status: STARTED`, timestamp and host.
  - On completion it is finalized atomically (temp file + `os.replace`) with `status: COMPLETED` and the verdict.
  - A crash or exception leaves `STARTED`, and a `STARTED` record counts as consumed access.
- Any later B evaluation for the same B manifest is refused, whatever the frozen file name, checkpoint or τ.
- The only exception is `--override-invalidates-confirmatory "<reason>"`. It appends the new record and rewrites the original verdict as `INVALIDATED` in `val_B_access.json` and in the original report.
- Unit tests:
  - a renamed copy of the frozen file → refused;
  - a different checkpoint → refused;
  - override → original marked `INVALIDATED`;
  - two concurrent invocations: exactly one reserves, the other fails at exclusive create before loading data;
  - an exception injected after reservation leaves `STARTED`, and a subsequent run is refused.

### Frozen evaluation sites

- `eval_sites_{A,B}.npz` stores, per example:
  - crop origin (deterministic jitter from a fixed RNG seed);
  - seed voxels;
  - **`seed_fragment_label`** (first-pass label at the seed voxels);
  - **`conditioning_owner`** (label rendered into channel M; 0 = owner-free, used for both seed-swap members);
  - target neuron k;
  - type, radius, anchor node ids, and `pair_id` / member index for swaps.
- A site is **measurable** only if its anchor nodes (gap component nodes, split edge endpoints, minority nodes, endpoint node) fall inside the write region with ≥8-voxel margin. Others are dropped and counted.
- **Deployment-like duplicates** are stored separately in `eval_sites_{A,B}_deploy.npz` with `seed_trust: deployment`, alongside `seed_trust: trusted` for the main sites.
  - Each duplicate re-seeds the same site from its trusted example's `seed_fragment_label` interior: max EDT of the fragment alone, no GT intersection, same ball rule.
  - They are scored only in separate report columns and never enter gate metrics, τ selection or support counts.

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
- **After any removal:** active weights are recomputed by renormalising the original weights over currently nonempty buckets.
- **Sampler exhaustion:** if all buckets become empty, the sampler raises `SamplerExhausted`. `train.py` then exits non-zero with a readiness-failure message (no silent fallback).
- **At init:** empty buckets are logged; startup fails if the required training populations are below the readiness minima.
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

Settings: prediction threshold 0.5, no TTA, write region only.

**Units (declared before training).** The unit of every pooled metric is a **proposal opportunity**:
- a (site, node) pair for node metrics;
- a (site, candidate fragment) pair for joins;
- a site for controls;
- a pair for seed swaps.

The same fragment or node seen from two frozen sites counts twice, because at deployment each site
issues its own proposal. Frozen sites are deduplicated (16 XY / 8 Z voxels) to limit this.
Unique-entity counts (unique seeded neurons, unique candidate fragments, unique candidate-dominant neurons) are reported next to every opportunity count.
Support minima use opportunity counts.

**Which examples enter which metric; ownership.**
- Seed-swap examples enter **only** metric 5 (seed-only dependence) and its pair support count. They are excluded from metrics 1–4, from τ selection, and from those metrics' support counts.
- For every other example, the metric **owner** is `seed_fragment_label` (the first-pass label at the seed voxels), which equals `conditioning_owner` for these examples.
- Every use of "owner" in metrics 1–4 (missed-node recovery `L ≠ owner`, B0 owner mask, join candidates `≠ owner`, preservation) means `seed_fragment_label`.
- Deployment-like duplicates (below) are scored in separate report columns and never enter gate metrics, τ selection or support counts.

**Aggregation.**
- Primary: pooled (micro) over opportunities. Per-site macro means are secondary.
- **No confidence intervals and no bootstrap are computed in this experiment.** In dense tissue, opportunities share seeded neurons, swap partners, candidate fragments and candidate neurons through overlapping crops, and two frozen grouping schemes still left dependent observations in separate clusters. No gate or verdict uses intervals, and the eligibility-based support minima bound sample size.
- **Reports give** pooled point estimates, opportunity counts, unique-entity counts (unique seeded neurons, unique candidate fragments, unique candidate-dominant neurons) and empty-prediction counts. The reports state that no intervals are given and why.

**Candidate radius.** Median `radius_nm` over the candidate fragment's aligned skeleton nodes inside the write region, over all its nodes regardless of GT (including mixed fragments). Thin means ≤45 nm.

**Metrics.**

1. **Missed-node recovery** by radius bin.
   - Denominator: aligned GT-k nodes in the write region not correctly owned by the first pass (`L=0`, or `L ≠ owner`).
   - Numerator: those covered by the mask.
   - B0 = 0. Also reported: total GT-k node recall, with B0 = the owner mask.
2. **Wrong-neuron inclusion:** `|mask ∩ gt∉{0,k}| / |mask|` (voxels), and other-GT aligned nodes covered ÷ GT-k nodes in the region. B0 is computed from the owner mask. An empty mask gives voxel fraction 0 and is counted in `empty_predictions`.
3. **Fragment joins.**
   - Candidates: first-pass labels ≠ owner, ≠ 0, with ≥1 aligned skeleton node in the write region. A join fires when the mask covers ≥ τ of the fragment's write-region voxels.
   - Truth by purity class:

     | class | truth |
     |---|---|
     | `pure_k` | **positive** (correct if joined) |
     | `pure_j` (j≠k), `bgdom_pure_j`, `mixed` | **negative**; wrong if joined, even when a mixed fragment's dominant GT is k |
     | `bgdom_pure_k` | reported separately, excluded from PR |
     | `unannotated` | reported separately, excluded from PR |

   - `roi_truncated` candidates are included and flagged.
   - precision = correct / fired; recall = correct / positives, at τ ∈ {0.3, 0.5, 0.7, 0.9}, split thin / non-thin.
   - If no join fires at a τ, precision at that τ is `NA` and is treated as **not meeting** any precision requirement; recall is 0.
4. **Preservation controls** (`endpoint_control`, `correct_control`), reported as three separate components:
   - preservation = fraction of the owner's aligned GT-k nodes in the write region covered by the mask;
   - unwanted expansion = `|mask \ gt==k| / max(|mask ∩ gt==k|, 1)`;
   - unwanted joins = number of negatives fired.
   - Success requires preservation ≥ 0.9, expansion ≤ 0.05 and 0 negative joins. An empty mask has preservation 0 and fails.
5. **Seed-only dependence** (paired, M = 0):
   - pair IoU(pred_k, pred_j), where IoU of two empty masks is defined as 1;
   - own-target aligned-node recall of each prediction in the write region.
   - A pair is valid iff both recalls are ≥ 0.5.
   - Reported: valid fraction, and median IoU over valid pairs (`NA` if none are valid).

**τ selection (A only, frozen before B).**
- Choose the τ maximising thin-candidate recall subject to thin precision ≥ 0.95 on A; ties go to the larger τ.
- If no τ meets 0.95 on A, choose the τ with the highest thin precision among τ with ≥1 fired join (ties → larger τ).
- If no τ fires any join on A, choose τ = 0.9.
- The chosen τ and the rule branch used are written into the frozen JSON.

**Zero denominators.** A per-site metric with a zero denominator is excluded from macro averages and counted. A pooled metric with a zero denominator is `NA`. Every metric reports opportunity count, unique-entity count and empty-prediction count.

**Support (eligibility only, independent of predictions)**, computed from frozen B sites and first-pass labels:
- ≥500 thin missed-node opportunities over ≥50 sites;
- ≥100 thin candidate opportunities, including ≥30 positives and ≥30 negatives;
- ≥50 control sites;
- ≥50 seed-swap pairs.

**Verdict** (`--split B --final` only). Evaluated in order; the first match wins:

1. `INSUFFICIENT_SUPPORT` if any support minimum fails.
2. `KILL` if any of:
   - no τ in the grid has thin recall ≥ 0.10 (no join capability, including the zero-output model);
   - every τ with thin recall ≥ 0.10 has thin precision < 0.90;
   - valid seed-swap fraction < 0.25;
   - median valid-pair IoU ≥ 0.70.
3. `PASS` if all cheap_gate criteria hold at the frozen τ.
4. `FAIL` otherwise.

Unit test: a model that outputs zeros on a synthetic set meeting every support minimum → `KILL`, not `INSUFFICIENT_SUPPORT`.

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
  - support minima met (eligibility-based);
  - verdict precedence as defined in the gate evaluator.
- **kill_criterion:** exactly the `KILL` rule of the gate verdict: no τ reaches thin recall ≥ 0.10; or every τ with thin recall ≥ 0.10 has thin precision < 0.90; or valid seed-swap fraction < 0.25; or median valid-pair IoU ≥ 0.70.
- **limitation:** oracle site locations and trusted seeds; the deployment-like seed column is informative, not gating.
- **cost:**
  - this run: ≤ 3 L40S GPU-hours plus CPU decode/mining on 3 ROIs;
  - full training later: ≤ 24 GPU-hours.

### Real-data correctness and readiness criteria (replacing v0's distribution-based pass bands)

Correctness (must pass):
1. **Alignment.**
   - Decoded seg, affinity spatial shape and `gt.h5` shape are identical.
   - The misaligned-node fraction per ROI is ≤ 2%, with exact counts by radius bin and split reported.
   - Misaligned nodes are excluded before mining (see Node alignment exclusion); an assertion checks that no misaligned node appears in any manifest or metric.
2. **Label consistency.**
   - **Trusted-seed examples** (`seed_trust: trusted`; training manifests, frozen eval sites, **each member of every seed-swap pair**), 100%:
     - the first-pass label at the seed voxels equals `seed_fragment_label`;
     - that fragment is `pure` for the target neuron (or `mixed` with dominant target for contamination);
     - GT at the seed voxels equals the target neuron.
   - **Deployment-like duplicates** (`seed_trust: deployment`):
     - asserted 100%: the first-pass label at every seed voxel equals `seed_fragment_label` (fragment membership);
     - GT at seed voxels ≠ target neuron is **reported, not rejected**, as `deploy_seed_target_mismatch` counts by site type and radius bin.
   - For non-swap examples, `conditioning_owner == seed_fragment_label`.
   - For swap members, `conditioning_owner == 0`, and the channel M input is verified all-zero.
   - Decode is deterministic: a re-run gives the same sha256 of the label array.
3. **Independent re-derivation.** For `min(available, 50)` random sites per type, a separate voxel-array code path re-checks the type:
   - gap anchors: seg==0 and gt==k at the anchor voxel, plus the boundary labels;
   - split: labels differ at the endpoints;
   - contamination: the label's voxel histogram has ≥2 GT ids, and the minority node's GT ≠ dominant;
   - controls: correct ownership.
   - Absent types are reported explicitly as `absent` (not a failure).
4. **Montages.** `min(available, 12)` PNGs per type (train and val A) are generated and their paths listed. Listing paths is **not** a completed human audit: code_vN.md marks the visual audit `pending human review`.

Readiness (only these populations gate the next stage):
- smoke training needs ≥200 thin (≤45 nm) gap/split train sites;
- evaluation development needs ≥50 thin gap/split val-A sites;
- `train.py` / `evaluate.py` refuse to start below these.

Diagnostics only (reported, not pass/fail): zero-node fraction, per-type counts per ROI, train-vs-val omission/split rates by radius bin, gap length distribution, unseedable/excluded counts, purity class counts.

### Code-stage isolation (shared checkout)

The main checkout `/projects/weilab/weidf/lib/pytorch_connectomics` is shared with other sessions. At plan_v3 review time, another CCC run was editing tracked files there (`connectomics/models/losses/build.py`, `metadata.py`, new `embedding.py`). This run's code stage is therefore isolated:

1. **Worktree.** Before code_v0, the coordinator (not the coder) creates a detached worktree at the run's `run_start_ref`:
   `git -C /projects/weilab/weidf/lib/pytorch_connectomics worktree add --detach /projects/weilab/weidf/lib/pytorch_connectomics/.claude/worktrees/ccc-ec_model 1546f47ece4777e20bcddd1028fb1bb8908ae02d`.
   It records the path in `run.md` under a `## Code Stage Location` section. If the path exists, it is reused only if its HEAD equals `run_start_ref` and `git status --short` is empty; otherwise the run blocks.
2. **Where code and outputs go.** All new source files are created under `<worktree>/dev/ec_model/`, and generated artifacts under `<worktree>/dev/ec_model/runs/`. Slurm launchers `cd` into the worktree.
3. **Inputs are absolute and read-only:**
   - `/projects/weilab/weidf/lib/pytorch_connectomics/dev/false_split/manifest.json` and `.../runs/20260905_seed_ensemble/<roi>/{predictions/seed42.h5,gt.h5,anatomy.npz}`;
   - the canonical checkpoint `.../outputs/nisb_base_banis_v3_erosion2/20260508_224029/checkpoints/step=00200000.ckpt`;
   - `/projects/weilab/dataset/nisb/...`.
   `common.py` defines these; no code writes under the main checkout.
4. **Package import pinning.** Every entry point (`decode_rois.py`, `mine_sites.py`, `train.py`, `evaluate.py`, `montage.py`, tests) inserts the worktree root at `sys.path[0]`. It then asserts that `Path(connectomics.__file__).resolve()` lies inside the worktree and fails otherwise. This prevents the pytc editable install from resolving to the main checkout's uncommitted edits. Config paths such as `tutorials/neuron_nisb/base_banis+.yaml` resolve from the worktree root.
5. **Git guards run in the worktree:**
   - `git rev-parse HEAD == run_start_ref`;
   - `git status --short` empty (tracked);
   - `git diff` / `git diff --cached` captured before and after any code-review command.
   The main checkout's tracked state is not part of this run's baseline.
6. **Promotion.** After the run completes, copying `dev/ec_model/` into the main checkout (both locations gitignored, so no merge) is a separate step that needs user approval, not part of this run. `runs/` artifacts stay in the worktree unless the user asks to move them.

### Review delivery contract (for dev/ files, which are gitignored)

Code submissions include `state/code_vN.patch` in the run folder. It concatenates `git diff --no-index /dev/null <file>` for every new or changed source file under `<worktree>/dev/ec_model/` (source and docs only, excluding `runs/`), so the coordinator can embed it in the code-review prompt.

`code_vN.md` lists, for every file: path, line count and sha256. It also gives every command run with exit code and key output, and artifact provenance (paths, sizes, Slurm job IDs).

Source is kept ≤ ~120 KB so patch plus artifacts fit the 200 KB review prompt. If it would not fit, the coordinator blocks instead of truncating.

## Files and Areas

All new files are under `<worktree>/dev/ec_model/`, where `<worktree>` = `/projects/weilab/weidf/lib/pytorch_connectomics/.claude/worktrees/ccc-ec_model` (detached at `run_start_ref`; see Code-stage isolation). This is a gitignored research harness like `dev/false_split/` and `dev/nisb/`. No tracked files change in the worktree or the main checkout.

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

1. **Unit tests** (`pytest dev/ec_model/test_ec_model.py`, CPU, threads ≤2). Synthetic XYZ volumes with known GT, first-pass labels and skeleton graphs.
   - **Mining:**
     - straight gap: length and min radius;
     - Y-branch with a missing branch (`gap_terminal`) and a missing junction segment (`gap_bridge`);
     - singleton zero-label component (anchor = node, half-boundary length);
     - isolated edgeless node (`gap_unseedable`);
     - boundary-free component (`gap_unseedable`);
     - tie cases (lowest node index / label id);
     - gap touching the ROI face (`excluded_border`);
     - split edge;
     - contamination: the dominant part inside the 24/12 window → seeded there; outside the window but elsewhere in the crop → **seeded via the whole-crop fallback**; absent from the entire crop → `contamination_unseedable`;
     - split anchor = lower-index endpoint node;
     - voxel-only mixed label with no minority node (no site, counted);
     - dedupe of adjacent sites;
     - clean control with no errors;
     - misaligned node excluded before mining (breaks a gap component; never appears in manifests).
   - **Purity:** 60/40 → mixed; 99/1 → pure; 98.9/1.1 → mixed; all-background → unannotated; background-heavy 99.5% → bgdom_pure; background-heavy with 2 GT nodes → mixed; dominant tie → lowest id.
   - **Dataset / sampler:**
     - seed ⊂ fragment ∩ GT k, never background;
     - jittered centre-coincidence < 5% over 200 draws;
     - target includes a crop-disconnected piece;
     - paired seed-swap keeps EM/M/A byte-identical and changes only P and target;
     - empty-bucket redistribution at init;
     - weight recomputation after a removal empties a bucket;
     - `SamplerExhausted` when all buckets empty;
     - bounded-resampling zero-jitter fallback;
     - advice invariance under flip/transpose.
   - **Model:** strict stem inflation (channel 0 equals checkpoint, channels 1–3 zero, other keys exact) on a tiny model from the same code path, or on the real state dict loaded on CPU.
   - **Import pinning:** `connectomics.__file__` resolves inside the worktree; the guard raises when the root is not first on `sys.path` (simulated).
   - **Label consistency scoping:** a deployment-like contamination seed on a non-target neuron is reported as `deploy_seed_target_mismatch` and not rejected; the same condition on a trusted seed raises.
   - **Evaluator:**
     - hand-built recall and inclusion;
     - purity-based join precision with a mixed-dominant-k fragment counted wrong;
     - `bgdom_pure_k` excluded from PR;
     - candidate radius = median over in-region nodes;
     - τ selection branches and ties;
     - an empty mask fails preservation;
     - `NA` on zero denominators;
     - eligibility-based `INSUFFICIENT_SUPPORT`;
     - **a zero-output model on supported data → `KILL`**;
     - verdict precedence order;
     - no interval or bootstrap fields in reports; the report states intervals are not computed;
     - seed-swap examples absent from metrics 1–4, τ selection and their support counts, and present in metric 5;
     - for a non-swap example the seed fragment is never a join candidate (owner = `seed_fragment_label`);
     - deployment-like duplicates absent from gate metrics and support counts, present in the separate columns;
     - B guard: renamed frozen file refused, different checkpoint refused, hash mismatch refused before data load, override marks the original `INVALIDATED`.
2. **Real curation** on the 3 ROIs (`run_ec.sh decode`, `run_ec.sh mine`; Slurm CPU if needed): the four correctness criteria and the readiness counts. Diagnostics printed in code_v0.md.
3. **Montages** generated (`min(available, 12)` per type); paths listed; human visual audit marked `pending human review`, not claimed.
4. **Overfit test** (1 L40S, ≤45 min; no augmentation or dropout; 2k steps). The fixed 32-example set is built deterministically (sites sorted by id, fixed RNG seed) from train sites:
   - **Pair prerequisite check first.** An eligible pair is two neurons in the same train crop, each with a `pure` fragment voxel in the write region, where the seed-k member is a thin (≤45 nm) gap/split site. Up to 8 are selected, contributing both members (≤16 examples).
     - If fewer than 8 eligible pairs exist, the seed-dependence check reports `PREREQUISITE_UNAVAILABLE (n_pairs=<n>)`. That is a data-availability outcome, not a model-learning failure. The available pairs are still included.
   - **Remaining slots** (32 minus 2·n_pairs), filled in this order:
     - contamination examples, up to 4 if available;
     - controls, up to 4 if available;
     - thin gap/split sites for everything left, so replacements are deterministic whenever contamination or controls are scarce.
   - If fewer than 16 thin gap/split examples can be assembled in total, the overfit test reports `PREREQUISITE_UNAVAILABLE` instead of running the Dice criterion. Readiness (≥200 thin train sites) makes this unexpected.
   - The per-category composition actually used is reported.
   - Pass: mean Dice ≥ 0.90 over all examples, and ≥ 0.85 on the thin gap/split subset.
5. **Seed-only dependence after overfit**, on the **trained pairs**, when n_pairs = 8: ≥6 valid pairs (both own-target recalls ≥ 0.5) and median valid-pair IoU ≤ 0.3 → pass. With 1–7 trained pairs, the same metric is reported under `PREREQUISITE_UNAVAILABLE` and does not block smoke training. Up to 8 **unseen** pairs from other train crops are a diagnostic only.
6. **Smoke training + evaluation on A only** (1 L40S, ≤2 h). Report and val-A curves are attached and labelled **smoke, not a gate verdict**. Checks: no NaN; decreasing loss; B0 join recall = 0; `runs/val_B_access.json` absent (B untouched).
7. **Launch readiness:** `bash -n run_ec.sh`. Full-training and new-ROI commands are in README, not submitted. Every sbatch is prefixed `PREREG=dev/ec_model/PREREG.md`.
8. **Baseline integrity and delivery:**
   - In the worktree: `git rev-parse HEAD` equals `run_start_ref` and `git status --short` is empty.
   - Every entry point prints and asserts a worktree-local `connectomics.__file__`; the import-pinning unit test passes.
   - No file under the main checkout is created or modified by this run (checked with `find <main>/dev/ec_model -maxdepth 0` absent, plus the command log).
   - `state/code_v0.patch` covers every source file in the table, with per-file sha256 in code_v0.md.

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

plan_v4 was authorized by the user ("do another round of planing") after the plan_v3_review block; `plan_rounds` was raised from 3 to 4. It addresses every plan_v3_review finding and the environment issue the coordinator raised there. Earlier review resolutions are unchanged.

1. **[major] Dependency groups omitted shared candidate-neuron dependence.**
   - Adopted the reviewer's stated alternative: **no confidence intervals and no bootstrap in this experiment**. Dependency groups, `eval_groups_{A,B}.npz` and all CI text are removed.
   - Rationale: two grouping schemes still separated dependent observations; no gate or verdict used intervals; eligibility-based support minima bound sample size.
   - Reports give point estimates, opportunity counts, unique-entity counts (including unique candidate-dominant neurons) and empty-prediction counts, and say why intervals are absent.
   - Tests assert no interval fields.
2. **[minor] `owner` ambiguous after `conditioning_owner`.**
   - Seed-swap examples enter only metric 5 and its pair support; they are excluded from metrics 1–4, τ selection and those support counts.
   - For all other examples the metric owner is `seed_fragment_label`, so the seed fragment is never a join candidate.
   - Tests added.
3. **[minor] Deployment-like seeds conflicted with the universal label-consistency assertion.**
   - The 100% seed-on-target assertion is scoped to trusted-seed examples, including each swap member.
   - Deployment-like duplicates (`eval_sites_{A,B}_deploy.npz`, `seed_trust: deployment`) assert only fragment membership; GT mismatches are reported as `deploy_seed_target_mismatch` by type and radius.
   - Duplicates are excluded from gate metrics and support. Tests added.
4. **Coordinator-raised environment issue (shared checkout).** New Code-stage isolation section:
   - a coordinator-created detached worktree at `run_start_ref` (`.claude/worktrees/ccc-ec_model`), with all source and artifacts there;
   - absolute read-only inputs from the main checkout and dataset;
   - `connectomics.__file__` pinned to the worktree with a test, so another session's uncommitted `connectomics/models/losses/` edits cannot leak in;
   - git guards run in the worktree;
   - promotion into the main checkout is a separate user-approved step.
   - Files and Areas, the delivery contract and Verification step 8 are updated accordingly.
