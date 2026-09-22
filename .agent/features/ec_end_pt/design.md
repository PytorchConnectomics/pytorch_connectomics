# EC end-point growth — design for the second-pass error-correction model

Status: design agreed 2026-09-17, **not yet implemented**. Supersedes the crop/seed
design in `dev/ec_model` (feature `ec_model`), which stays on disk as the baseline.
Everything below that is a number was measured this session; provenance is given inline
so a fresh session can re-derive rather than trust.

Repository root: `/projects/weilab/weidf/lib/pytorch_connectomics`
Working worktree: `.claude/worktrees/ccc-ec_model` (detached; `common.py` pins a
worktree-local `connectomics`). Artifacts under `dev/ec_model/runs/`.
Conda env: `/projects/weilab/weidf/lib/miniconda3/envs/pytc/bin/python`. Slurm required.

---

## 1. The task, stated as deployment does it

Second-round inference has **only the segmentation**. It enumerates the tips of the
predicted fragments' skeletons and tries to grow each one. There is no oracle telling it
which tips are real breaks. The training population must therefore be that same
population — every trunk tip — not a GT-selected subset of "known false splits".

The crop is **centred on the tip**, and the centre *is* the prompt: the object to
segment is whatever fragment owns the centre voxel. No movable seed, no learned
promptable segmentation.

## 2. Why the previous design could not deploy

`dev/ec_model` anchors gap sites on the GT skeleton node inside the *uncovered* stretch.
Measured on the mined sites:

| type | where the crop centre lands | n |
|---|---|---|
| gap_terminal | **background (`seg == 0`)** | 2650 |
| gap_bridge | **background (`seg == 0`)** | 305 |
| correct_control | inside the owner fragment | 2203 |
| endpoint_control | inside the owner fragment | 182 |
| split | inside the owner fragment | 1 |

100% of gap sites centre on `seg == 0`, at zero jitter, by construction. That location is
findable only from the GT that defined it, so the task as posed has no inference-time
counterpart. It also makes `gap_bridge` ambiguous: ≥2 owner fragments flank the gap and
the old `freeze()` picked one at random while the origin did not depend on the owner, so
the *identical crop* carried two different correct answers, resolvable only by the prompt
channels. That ambiguity is the reason the architecture needed a prompt at all.

Provenance of the old choice: `dev/ec_model/README.md:29` records it as a correction to
SENSE v1's "fixed central seed", whose flaw is stated in
`dw-research/projects/2026_nisb_base/lessons/lesson_sdt_decode_ceiling.md:233` — the model
never had to *read* the seed and degenerated to "segment the central/dominant object,"
grabbing thick neighbours over thin tubes. **Re-centring reverses that decision
deliberately.** It is safe to do so because SENSE v1's input was `[EM, seed Gaussian]`
with no membership signal, whereas this model has `m = (seg == owner)`, a dense fragment
mask, and the centre is *constrained* to lie inside it — so "the object at the centre"
and "the fragment in `m`" are the same object by construction.

## 3. Candidate centres come from predicted skeleton tips

`dev/ec_model/skeletonize_pred.py` (written, run on all six ROIs) skeletonises
`seg_cc066.h5` with kimimaro using the project's established teasar parameters
(`em_erl/skel.py`: scale 1.5, const 500, max_paths 50, anisotropy 9/9/20) and writes
`runs/<roi>/pred_tips.npz` + `pred_tips_report.json`.

| roi | frags | skeletonised | tips | on ROI face | interior | interior @1 µm cut |
|---|---|---|---|---|---|---|
| train_seed0_center | 4159 | 1208 | 3674 | 1749 | 1925 | 1096 |
| train_seed1_center | 4418 | 1256 | 3820 | 1914 | 1906 | 1160 |
| train_seed2_center | 4483 | 1266 | 4070 | 1899 | 2171 | 1346 |
| train_seed3_center | 3981 | 1050 | 3310 | 1634 | 1676 | 1103 |
| train_seed4_center | 4360 | 1338 | 4243 | 1936 | 2307 | 1442 |
| val_seed100_center | 3793 | 1159 | 3725 | 1653 | 2072 | 1263 |

**5 train ROIs: 9,985 interior tips; 6,147 at a 1 µm trunk cut.** ~13–23 min per ROI at
8 threads.

**Tips on an ROI face are excluded.** 45–50% of all tips are crop-boundary artefacts —
a fragment clipped by a face gets a tip there that is not a neurite terminal and must not
be trained as one. `skeletonize_pred.py` flags them as `on_roi_face` (within 2 voxels).

## 4. Grow from trunks, not crumbs

Seeds come only from fragments with enough centreline to define a trajectory. Crumbs stay
in the *target* (things to absorb) but are never seeded from. Cable-length percentiles on
seed0: p10 159, p25 319, **p50 830**, p75 2837, p90 6327, p99 90853 nm.

Cost of the restriction, measured over seed0's 2,955 gap sites via each gap's
`boundary_labels`:

| trunk cut | ≥1 flank is a trunk | both flanks | unreachable |
|---|---|---|---|
| 0 nm | 100.0% | 10.3% | 0.0% |
| 500 nm | 97.1% | 4.0% | 2.9% |
| **1000 nm** | **93.8%** | **1.6%** | **6.2%** |
| 2000 nm | 89.6% | 0.7% | 10.4% |
| 5000 nm | 81.7% | 0.1% | 18.3% |

**Recommended cut: 1000 nm** — forfeits 6.2% of gaps, keeps 550 fragments and ~1,100
interior tips per ROI, and gives ~111 voxels of xy centreline to establish direction.
500 nm buys 3.3 points of coverage for half the shape context.

The middle column is the important one: **only 1.6% of gaps have trunks on both flanks.**
The typical false split is trunk → crumb. So the merge-dangerous case — a model welding
two backbones — is 1.6% of sites. This is the structural merge-safety that
`lesson_pathfinder_thin_weakness` claims for trunk-seeded growth, now measured here.

## 5. Augmentation: precomputed centres, no on-the-fly jitter

For each trunk tip, precompute every admissible centre and treat each as its **own
training example**. No random jitter at training time; the population is deterministic
and reproducible.

- A centre must lie **inside the owner fragment**. (Verified on the interim
  endpoint-centred implementation: 0 of 1,095 freezes put the centre off its fragment.)
- Candidates are accepted greedily walking back from the tip, keeping a candidate only if
  its **write-region overlap with every already-kept centre is < 0.8**. Use the exact
  per-pair overlap `Π_axis max(0, 1 − |d_i| / W)`, not a scalar distance — the
  displacement between tips is not axis-aligned. Axis-aligned this is ≈13 voxels
  (≈115 nm xy, 260 nm z) at `W = 64`.
- Purpose is to stop the model memorising near-duplicate crops, not to make the crops
  disjoint. An earlier "separation ≥ write extent" rule was rejected as too aggressive.

**Open parameter:** a cap on centres per tip. At 576 nm separation on GT skeletons the
multiplier was ~13.7×; at <80% overlap expect ~5× more again. Uncapped this produces a
population far larger than the step budget can consume — size it before committing.

## 6. Crop size: keep 128³

Do **not** enlarge the crop. Measured against gap lengths (n=16,891; median 18 nm, p90
164, p99 646, max 1709; voxel 9/9/20 nm):

- A 128³ crop centred on the tip reaches 576 nm in xy. Gaps exceeding that: **1.5%**.
- The finished baseline used **11.7 GB peak CUDA at batch 2** (`peak_cuda_bytes`).
  192³ scales to ≈40 GB — unsafe on a 48 GB L40S; 192×192×96 to ≈20 GB.

Growing the crop buys 1.5% of gaps for 1.7–3.4× compute. Keep 128³ for the first run: it
is measured-sufficient, preserves comparability with the baseline, and changes one thing
at a time. Revisit only if thin recovery stalls.

## 7. Target stays the whole neuron in the box

`y = (gt == target)`. Do not narrow it to a reachability-masked variant. Measured
decomposition of the target mass lying outside the owner fragment:

| type | y outside m | of that, contiguous with the seed | disconnected |
|---|---|---|---|
| correct_control | 25.9% | 97.6% | 1.7% |
| endpoint_control | 16.1% | 100.0% | 0.0% |
| gap_bridge | 79.1% | 87.5% | 12.5% |
| gap_terminal | 23.1% | 97.8% | 2.2% |
| **ALL** | **26.4%** | **97.3%** | **2.4%** |

**97.3% of the missing mass is contiguous with the seed**, so the target already *is*
"grow outward from the seed". GT is dense, so the neuron is one object and the
segmentation gap is not a gap in the label; disconnected pieces only arise when a neuron
leaves and re-enters the crop. Masking for reachability would buy 2% and cost a
connected-components pass per example.

This also sets the floor for Dice. With `m ⊆ y`, copy-Dice ≈ `2(1−f)/(2−f)` for missing
fraction `f`: **f = 0.25 → copy-Dice ≈ 0.86** overall, and ≈0.46–0.56 for `gap_bridge`.
**Quote Dice against 0.86, never against 0.** The Phase A overfit result (mean Dice 0.976)
closed ~83% of the real headroom — a genuine result, but flattered by an uncomputed floor.

## 8. Sampling

One stratum. Inference cannot tell a false split from a true terminal, so neither does the
sampler: every tip type, every calibre, nothing excluded and nothing reweighted.

Classes that disappear under this design:
- **`correct_control`** — an *interior* GT skeleton edge inside one pure fragment
  (`mine_sites.py:155`), i.e. not a tip at all. It has no deployment analogue; the "don't
  over-grow" job is carried by tips that are genuine terminals.
- **`seed_swap`** — degenerate once the centre is the prompt; both members carry the
  identical centre seed. Keep mining them only if a held-out seed-dependence report is
  still wanted.
- **`contamination`** — 11 sites drew 10% of every batch in the old sampler (**545 draws
  per site**). Whatever is done, do not restore a stratum that small at that weight.

The GT-mined population (30,502 sites) does **not** carry over; 12,446 of it was
`correct_control`. The tip population is ~6,100 trunk tips × centres per tip.

## 9. Metrics and gates

Mask metrics cannot establish a segmentation gain (`dev/ec_model/README.md`). Gate on:

1. **The count of fragments spanning >1 GT neuron must not rise.** Gate on the count,
   not the max: cc3d on val_seed100_center is *already* max-span 4 with 4 spanning
   fragments, so "max span must stay 1" is unreachable and the max is insensitive — it
   stayed at 4 in the run below while the count tripled. Baseline to beat: **4**.
2. **frags/GT must drop** meaningfully versus cc3d.
3. **Beat `b0 = (seg == owner)`** — the copy-the-input baseline. `evaluate.py:site_score`
   already emits `b0_recall`, `b0_wrong_voxels`, `b0_wrong_nodes`, `b0_control`. With `m`
   as an input channel and the centre inside it, copying `m` is the cheap degenerate
   solution and is the single most likely failure of this design. Read `b0` as the primary
   gate, not a side column.

### 9.1 First end-to-end result — the decoder is the bottleneck, not the mask

`runs/decode_merge_val_seed100_center.json`, job 3017117: baseline checkpoint
(`full_radius_stratified/last.pt`) over 2,366 deployment sites on val A.

```
baseline (cc3d):  218 GT neurons, 1244 fragments, frags/GT 5.734 (median 5)
                  max_gt_span 4,  fragments spanning >1 GT: 4

tau=0.3   frags/GT 5.298 (-0.436)  span 4   spanning: 14   joins 602  precision 0.922
tau=0.5   frags/GT 5.298 (-0.436)  span 4   spanning: 14   joins 599  precision 0.925
tau=0.7   frags/GT 5.307 (-0.427)  span 4   spanning: 14   joins 596  precision 0.925
tau=0.9   frags/GT 5.321 (-0.413)  span 4   spanning: 13   joins 592  precision 0.926
```

**Verdict: rejected.** It heals splits (frags/GT −7.6%, median 5 → 4) but triples false
merges (4 → 14). ~45 wrong joins out of 599 at 92.5% precision. NERL is merger-asymmetric
— one false merge truncates a whole skeleton's ERL contribution — so 10 new merges across
218 GT neurons will very likely swamp the split gain.

Two consequences for the redesign:

- **Coverage τ is not a usable lever.** 0.3 → 0.9 moves joins 602 → 592 and precision
  0.922 → 0.926. The masks are confident; a coverage threshold cannot separate good joins
  from bad. A different discriminator is required — the non-transitive one-to-one endpoint
  matching of `lesson_ec_endpoint_bridge` (the one merge-safe win on record), not a knob
  on this one.
- **Mask quality was not the binding constraint.** 0.766 thin recovery already surfaced
  real joins; what failed was deciding which to trust. Budget effort accordingly: a better
  mask model on tip-centred crops does not by itself fix this.

Scope: old checkpoint in its own regime (gap-centred, promptable). A floor for the
redesign, not a prediction of it.

`dev/ec_model/decode_merge.py` (written this session) does 1 and 2: it applies a
checkpoint at deployment sites, unions every fragment a mask claims at coverage ≥ τ into
the seed's fragment, and scores frags/GT and GT span from `node_labels` + `anatomy`
without rewriting a volume. Union-find is transitive, which is exactly the flood that gave
span 152 in `lesson_sdt_decode_ceiling`; the raw pairwise join count is reported alongside
so a chain reaction is distinguishable from many bad joins.

## 10. Budget note — do not repeat my error

I repeatedly asserted the step budget was the binding constraint. **It is not, on the old
design.** From `runs/full_radius_stratified/val_A_curves.json`, thin-bin pooled recovery:

```
step   500 -> 0.494    7500 -> 0.746   15000 -> 0.751   30000 -> 0.766
```

It saturated by step 7,500 and gained 0.02 over the remaining 22,500 steps. 30k steps took
11h14m at 1.28 s/step. Re-measure saturation on the new population before assuming more
steps are needed; the epoch arithmetic was measuring the wrong thing.

Also note the 45–90 nm and >90 nm bins swing 0.15↔0.53 between checkpoints — 266 and 230
opportunities with 184 and 137 zero-denominator sites. Only the thin bin (1,890
opportunities) is stable enough to read.

## 11. Held-out protocol — read before touching val

`val_seed100_center` is split A/B; **seed101 is the untouched test volume** and
"test-seed development" is listed in `README.md` as a SENSE v1 failure. Do not evaluate on
seed101.

Eval sites are **frozen crops** (`runs/val_seed100_center/eval_sites_A.npz`), baked with
origins at mine time under the old gap-centred anchoring. Training tip-centred while
scoring gap-centred measures neither design, so **val A must be re-mined** for this
redesign. The mine stage also writes `eval_sites_B*`, which sits behind a one-time-access
reservation (`reserve_B`, `runs/val_B_access.json`, and the `B untouched invariant` assert
at `train.py:229`).

`--split A` is safe: `reserve_B` is only reached under `if args.split == "B"`
(`evaluate.py:503`) and the A path *asserts* `val_B_access.json` does not exist
(`:547`). A does rewrite `frozen_A.json`, the (checkpoint, tau, B-manifest-hash) tuple a
later B run must match.

**Decision still needed:** re-mine val A only (needs the mine stage to accept a split
argument, which it currently does not), or re-mine the val ROI wholesale and re-reserve B,
which costs the "B was never looked at" property. Recommended: the former.

## 12. Build order

1. **Centre enumeration** — extend `skeletonize_pred.py` (or a new `mine_tips.py`) to walk
   back from each interior tip on a trunk (cable ≥ cut), emit centres under the <0.80
   overlap rule with a cap, and write per-ROI site records carrying `origin`, `owner`
   (= `seg[centre]`), `target` (= `gt[centre]`), and a post-hoc label (does GT continue
   past the fragment). Six ROIs.
2. **Dataset** — `freeze()` becomes a lookup of a precomputed origin; drop jitter, drop
   `seed_voxels`' GT masking (`deployment=True` semantics), keep the 4 channels
   `[em, seed, owner mask, affinity advice]` and the 128³ / 64³ geometry.
3. **Sampler** — one stratum over all centres.
4. **Re-mine val A** per §11 and retrain.
5. **Gate** with `decode_merge.py` on val A: frags/GT down, span 1, beats `b0`.

## 13. Current state on disk

- `dev/ec_model/skeletonize_pred.py` — written, run on all six ROIs; `pred_tips.npz` and
  `pred_tips_report.json` present for each.
- `dev/ec_model/decode_merge.py` — written and run (job 3017117); result in
  `runs/decode_merge_val_seed100_center.json` and summarised in §9.1. Negative: splits
  down 7.6%, false merges 4 → 14.
- `dev/ec_model/runs/full_radius_stratified/` — the **baseline**: old design, 30k steps,
  11h14m, final loss 0.065, thin recovery 0.766. Preserved under this name; the train
  stage writes to `runs/full`, so a new run will not collide.
- `dataset.py` / `train.py` / `test_ec_model.py` carry an **interim** endpoint-centred
  implementation (re-anchor to owner boundary node, centre-inside-fragment jitter
  constraint, single stratum, swaps dropped). 53 tests pass. It is a stepping stone —
  §3's predicted-skeleton tips supersede its GT-derived anchors.
- `evaluate.py` — `pair_score` recall now uses dense GT, not skeleton nodes. Verified over
  all 30,594 mined swap members: 0 zero-denominators under the fix, 5,816 (19.0%) under
  the old one. Evidence: `runs/pair_denominator_check.json`.
