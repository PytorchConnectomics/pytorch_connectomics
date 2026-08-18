# j0126 / zebrafinch — a ground-truth-free reconstruction

Three steps from raw EM to a merged segmentation, scored against the 50 held-out skeletons
that FFN was scored on. **No step uses ground truth**, including the merge: the agglomerator
is never told which segments are neurons or which belong together, so it grows *every*
fragment and decides each one's host from step 1's own affinity prediction.

| step | file | what | GT used |
|---|---|---|---|
| 1 | `1_affinity_zeroshot.yaml` | EM → 3-channel affinity, **zero-shot** from a NISB-trained model | none |
| 2 | `2_abiss.yaml` | affinity → over-segmentation (ABISS watershed + agglomeration) | none |
| 3 | `3_merge.yaml` | grow every fragment, then link | none |

The 50 test skeletons appear exactly once, in step 3's `evaluation:` block. Delete that block
and the `test:` block under it and the pipeline runs unchanged and writes the same
segmentation — that is the check that the reconstruction never saw them.

One **optional, non-pipeline** variant ships alongside: `1_affinity_supervised.yaml` replaces
step 1 with a model trained from scratch on the 33 zebrafinch dense GT cubes. Those cubes
are j0126 ground truth, so it is **not** GT-free and is deliberately not a fourth row above —
any run starting from it forfeits the claim in this section. It is here to measure what the
zero-shot path gives up. Note it must be inferred at its own `[48,96,96]` window: MedNeXt's
normalization has no running statistics, so the forward pass is window-size dependent and
the native-window affinity ranked better on the fixed local chunks. The historical
whole-volume 0.444376 result is the same checkpoint at `[144,144,144]` (`arm0_win144`), not
the native-window artifact; the native full-volume decode is tracked separately.

Unlike the three pipeline steps, that variant is **trainable from this repository** — it
carries the full from-scratch recipe, so `--mode train` reproduces the reference checkpoint
rather than only consuming it (see [Training the supervised variant](#training-the-supervised-variant)).

## Run

```bash
# 1. affinity (GPU; shard with --shard-id/--num-shards, one GPU each)
python scripts/main.py --config tutorials/neuron_j0126/1_affinity_zeroshot.yaml --mode test \
    --checkpoint outputs/nisb_base_banis_v3_erosion2/20260508_224029/checkpoints/step=00200000.ckpt

# 2. ABISS decode (CPU; ABISS runs its own chunk hierarchy)
python scripts/run_abiss_chunk.py --config tutorials/neuron_j0126/2_abiss.yaml

# 3. merge + score (CPU, decode-only: no checkpoint, no GPU)
python scripts/main.py --config tutorials/neuron_j0126/3_merge.yaml --mode test
```

For a whole-volume run, step 3's contact graph does not fit in memory. Build it in Z-slabs
first and hand it to the decoder via `contact_path`:

```bash
sbatch --array=0-22 --wrap "python scripts/build_contact_graph.py \
    --seg outputs/neuron_j0126/abiss/seg.zarr --slab \$SLURM_ARRAY_TASK_ID --out contacts/"
python scripts/build_contact_graph.py --seg outputs/neuron_j0126/abiss/seg.zarr --merge --out contacts/
```

## Training the supervised variant

Step 1 of the pipeline needs no training: it runs a released NISB checkpoint zero-shot, and
that model's own recipe is `tutorials/neuron_nisb/base_banis+.yaml`.

The supervised reference is different — `1_affinity_supervised.yaml` carries its own training
block and reproduces the exact recipe the `arm0_96` results came from
(`nisb_base_banis_plus_zebrafinch_heavy`, checkpoint `20260726_114349/step=00200000.ckpt`):

```bash
# train from scratch on the 33 dense GT cubes (4 GPUs, 200k steps)
python scripts/main.py --config tutorials/neuron_j0126/1_affinity_supervised.yaml --mode train

# then predict affinity with the resulting checkpoint
python scripts/main.py --config tutorials/neuron_j0126/1_affinity_supervised.yaml --mode test \
    --checkpoint outputs/1_affinity_supervised/<timestamp>/checkpoints/step=00200000.ckpt
```

What that recipe pins, all of it inherited from `../banis+.yaml` except where noted:

- **Model** MedNeXt-L / kernel 3 (61.8 M params), 6-channel affinity output, no deep
  supervision, `external_weights_path: null` — from scratch, *not* initialized from NISB.
- **Patch** `[48, 96, 96]`, which is ~864 × 864 × 960 nm and so near-isotropic for
  9 × 9 × 20 nm voxels, versus the inherited 128³ (Z-heavy, 5× more voxels). Both sides
  divide MedNeXt-L's stride of 16. This patch is why inference uses a `[48, 96, 96]` window.
- **Batch** 8 per GPU across 4 GPUs, cached in memory (the cubes are small h5 volumes).
- **Augmentation** `aug_em_neuron` — the DeepEM-matched heavy profile: elastic, ±50%
  contrast, slice shift/drop, lost sections, motion blur, missing parts, defect mutex. With
  only 33 cubes this carries the regularization instead of shrinking the model.
- **Schedule** AdamW at `lr=1e-3` (the from-scratch peak, not a finetune's 1e-4), cosine to
  200k steps, `16-mixed`, EMA at 0.999.
- **Data** all 33 padded cubes (`im_*[0-9].h5` / `gt_*[0-9].h5` — the numeric glob avoids
  pairing the `gt_*_skeleton.h5` caches against the images). Validation is a 3-cube subset
  that is *also in train*: it drives EMA validation and checkpoint selection only and is not
  a generalization estimate. Real evaluation is step 3 against the 50 test skeletons.

## The idea: grow everything, then link

An ABISS segmentation of neural tissue is dominated by **splits** — one neurite arrives as a
backbone plus a cloud of small fragments. Repairing that needs no ground truth, because two
facts the pipeline already carries are enough:

- a fragment's **voxel count** says whether it can stand alone, and
- the **mean affinity across a shared surface** says who it belongs to.

Note the second one is *not* contact area. Area looks like the obvious choice and is in fact
below chance on the hard cases — see the next section, which is the main result here.

**Round 1 — grow.** Every segment below `anchor_size` (40,000 vox) is absorbed into whichever
surviving component it is most strongly connected to by affinity, repeated for 8 shells so
chains of fragments come in one layer at a time. Note what this round does *not* need: it never asks
"is this fragment part of a neuron I care about?" It grows all of them. And it is
**merge-safe by construction** — it assigns a label to a fragment, it never unions two
anchors, so it cannot fuse two objects that were separate.

**Round 2 — link.** A fragment touching exactly *two* anchor components is evidence that
those two are one neurite interrupted by a thin break. Unioning them **is** a merge, so this
is the only round that can go wrong, and every parameter on it is a safety gate: the fragment
must be substantial (`link_min_size`), it must pass *through* rather than graze
(`link_balance`), and `max_hub_size` refuses any union that would put two backbone-scale
segments in one component.

That last guard is load-bearing, not decoration. Without it two neurons weld through a single
bad link and **both** score ≈0; removing it cost −0.008 aggregate NERL and regressed two
neurons in the reference run. The general rule, which this pipeline is built around: *as
joining gets stronger, the cost of every pre-existing false merge rises with it.*

Step 2 is tuned the same way — deliberately under-agglomerating (`AGG_THRESHOLD: 0.20`) so
that ambiguous decisions are deferred to step 3's merge-safe round rather than taken by a
watershed that cannot undo them.

## The merge decides hosts by AFFINITY, not contact area — and that is the whole result

An earlier version of this file reported round 1 at **0.5452** from a bound that used the **GT
skeleton graph** as a stand-in for voxel contact. That bound does not survive the substitution, and
chasing why produced the actual finding.

The GT-edge graph contains only the 19,072 segments carrying a GT node. On the real graph **a
fragment's contact area is on average only 27 % to those segments, and 84 % of fragments have most
of their contact to segments the bound could not see** — so the bound picked a host from a quarter
of the real neighbourhood. Rebuilt on the true 128 M-contact graph the same rule scores **0.4405,
four neurons regressed**. No gating rescued it: every drop-free cell of a full sweep (fragment-size
floors, contact floors, ambiguity margin, dominance, hops 1–8) landed in **+0.001…+0.003**.

The reason is that contact area is not weak evidence, it is *anti*-correlated on the hard cases.
Scoring both rules against the true host on 300 fragments that have ≥2 candidate anchors:

| candidate hosts | n | contact area | mean affinity |
|---|--:|--:|--:|
| 2 | 31 | 64.5 % | 80.6 % |
| 3–4 | 121 | 48.8 % | 86.8 % |
| **5+** | 148 | **16.2 %** | **87.2 %** |
| all | 300 | 34.3 % | **86.3 %** |

Random guessing scores 24.2 %. On the hardest half, **contact area is below chance** — the biggest
neighbour is systematically the fattest passing object, not the one the fragment continues into —
while affinity is ~87 % and flat in difficulty. Affinity is right where area is wrong 162 times;
the reverse happens 6 times.

Swapping the feature converts directly into score:

| host rule | fragments absorbed | NERL | Δ vs 0.4629 | ≥0.6 | regressed |
|---|--:|--:|--:|--:|--:|
| contact area | 5,168 | 0.4510 | −0.0119 | 14 | 3 |
| **mean affinity** | 8,108 | **0.5036** | **+0.0408** | 17 | **0** |

**+0.0408 at zero regressions, twenty times the best contact-area gate, with no gating at all.**
Affinity absorbs *more* while being safer — it is not trading recall for precision, it is simply
the right feature. This is an exact measurement rather than a bound: NERL reads only GT-carrying
segments' labels, and at hop 1 each such fragment's label depends only on its own host choice.

The uncomfortable part is worth stating: ABISS already computes mean affinity per region-graph edge
as its own agglomeration weight. The first version of this step operated on labels alone, discarding
exactly that signal, then spent a full parameter sweep trying to recover it by gating a
near-random substitute. **When re-deriving a step an upstream tool already performs, check which
features that tool used before inventing new ones.**

**What is settled independently:**

- The anchor set needs no ground truth: `≥50 GT nodes` scores 0.5432 against `≥40,000 vox` at
  0.5418 — 0.0014 apart. Nothing has to know which segments matter, which is what makes "grow every
  segment" implementable at all.
- Round 1 cannot weld two objects whatever its feature: it assigns a fragment a label and never
  unions two anchors.
- `max_hub_size` is load-bearing in round 2. Removing it cost −0.008 and zeroed two neurons.
- Channel order matters and will not announce itself: a reversed ZYX↔XYZ mapping still scored 84 %
  against 86 % for the correct one. Verify it; do not assume it.

## Choosing `anchor_size`

The one parameter worth tuning, and the trade is recall against safety. **Swept on the GT-edge
stand-in, so treat the ordering as indicative and the absolute values as superseded** by the note
above (aggregate NERL over the 50 skeletons; parenthesis = neurons regressed below the starting
point):

| `anchor_size` | anchors | fragments absorbed | NERL | regressed |
|---:|---:|---:|---:|---:|
| 5,000 | 6,664 | 12,258 | 0.4838 | 0 |
| 10,000 | 5,430 | 13,396 | 0.4937 | 0 |
| 20,000 | 3,909 | 14,565 | 0.5149 | 0 |
| **40,000** | **2,425** | **15,407** | **0.5452** | **0** |
| 80,000 | 1,631 | 15,073 | 0.5522 | 1 |
| 160,000 | 1,115 | 13,952 | 0.5631 | 2 |
| 400,000 | 629 | 12,765 | 0.5774 | 3 |

40,000 is the largest drop-free value. Above it the gain keeps coming, but it is bought by
absorbing real pieces into a neighbouring neuron — which is exactly the failure this pipeline
refuses to trade for run length.

`hops` behaves completely differently on the two graphs, and the real one is the surprise. On the
GT-edge stand-in each shell adds something (1 hop 0.5130 → 2 hops 0.5306 → 4 hops 0.5418 → 8 hops
0.5452), which is what makes the "absorb in shells" story sound right. On the **real** contact graph
hops 1, 2, 4 and 8 give **0.4405 to four decimals** — depth changes nothing, because the error is in
the single-step host choice, not in how far it propagates. Do not tune `hops` expecting the
stand-in's behaviour.

## Results

Scored with `nerl_merge_threshold: 50` against `test_50_skeletons.h5`, the same threshold and
skeleton set as the published FFN number. A NERL quoted without its threshold is not
comparable.

| segmentation | NERL | notes |
|---|---:|---|
| FFN (Januszewski et al. 2018, published segmentation) | 0.5390 | the reference to beat |
| ABISS, as decoded | 0.4135 | |
| ABISS on arm0 checkpoint at `[144,144,144]` (`arm0_win144`, supervised — **NOT GT-free**) | 0.444376 | historical full-volume result; not the native-window config |
| + published skeleton join (starting point of the 0.5118 row; its BOX PLACEMENT uses GT) | 0.4629 | not GT-free |
| step 3 round 1, **GT-edge stand-in** for contact | 0.545–0.571 | **superseded** — see Status; the range is tie-breaking |
| step 3 round 1, contact area, ungated | 0.4405 | −0.022, 4 regressed — refuted |
| step 3 rounds 1+2, contact area, ungated | 0.4289 | −0.034, 5 regressed — refuted |
| step 3 round 1, contact area, best drop-free gate | 0.4649 | +0.0020, 0 regressed |
| step 3, greedy affinity assignment, raw segmentation | 0.4578 | +0.0444, 0 regressed |
| **step 3, max-bottleneck assignment, raw segmentation (best GT-free)** | **0.4691** | **+0.0556, 0 regressed, 14/50 ≥0.6** |
| step 3 on top of the GT-scoped skeleton join | 0.5118 | +0.0489, 0 regressed, 19/50 ≥0.6 — **not end-to-end GT-free** |
| step 3 + round 2 link (any threshold) | ≤0.5024 | worse, 2–3 regressed — **refuted** |
| step 3 + anchor–anchor join (any threshold) | ≤0.4970 | worse, 4–27 regressed — **refuted** |

The honest current state: **end-to-end GT-free this pipeline scores 0.4691, below FFN's 0.5390.**
The merge contributes +0.0444 at zero regressions. A 0.5118 figure appears in the lessons and in
earlier drafts of this file; it is measured on top of a skeleton join whose box placement uses GT
node positions, so it is NOT an end-to-end GT-free number and must not be compared to FFN. The 0.5036 is round 1 at a single hop; round 2
and multi-hop have only been measured on the refuted contact-area feature, so both need re-deriving
on affinity before being trusted. The shipped `link: true` default has not been re-measured with
affinity — set `link: false` if you want only the number quoted above.

The recall ceiling of this architecture measured on the GT-edge stand-in is 0.5900 for grow+link
and 0.6456 with a real-real join added. **Do not treat those as headroom**: the same 27 % blind
spot that inflated the round-1 bound inflates them too. They are upper bounds on an easier
problem, not promises about this one.

**Three caveats, stated plainly.**

1. Every number in the "GT-edge stand-in" rows was computed with segment adjacency taken from the
   GT skeleton graph, and the substitution to real voxel contact does not preserve it. Do not
   quote them as results.
2. The starting point (0.4629) comes from a skeleton join whose *box placement* uses GT node
   positions. It is a published intermediate, not one of this tutorial's three steps; step 3 is
   measured as a delta on top of it. A from-ABISS-only number requires re-running step 3 on the
   raw 0.4135 segmentation.
3. No claim of beating FFN is made or supported here.

## Adapting to another dataset

Point `3_merge.yaml`'s `load_prediction_path` at your segmentation. The two shape parameters
are physical, not learned: `anchor_size` should sit near the voxel count of the smallest
object you would accept as standing on its own, and `link_min_size` near the smallest piece
you would trust as evidence of continuity. Re-run the sweep above on your data — the
drop-free point moves with voxel size and object caliber.

`max_hub_size` should be set to roughly the voxel count of a full backbone in your volume; its
job is only to notice that a union would join two things that are each already object-sized.

## Provenance

Algorithm, measurements and the negative results behind each gate:
`dev/zebrafinch/lessons.md` (L80–L84). The decoder is
`connectomics/decoding/decoders/segmentation_merge.py`; its module docstring carries what each
round can and cannot break.
