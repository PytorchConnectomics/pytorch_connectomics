# IST LICONN: affinity + conservative ABISS decode

This tutorial turns the IST LICONN `ExPID82_1` volume into a neuron segmentation:

1. Predict voxel affinities.
2. Decode them with ABISS (conservative watershed + max-affinity agglomeration),
   scored against the proofread FFN segmentation.

## Which LICONN this is

**The real IST LICONN volume**, FFN-proofread — *not* the simulated NISB liconn
under `/projects/weilab/dataset/nisb/liconn` that `tutorials/neuron_nisb/liconn_*.yaml`
uses. None of that volume's numbers (the 0.211 NERL, the erosion ablations, the
decode-lever retraction) were measured here. Details:
`pytc-agent/projects/2026_ist_liconn/lessons/lesson_mip0_resolution_gate.md` §0.

| | value |
|---|---|
| source | `liconn/ffn/ExPID82_1/image_230130b` + `segmentation/231030_agg_240123` |
| resolution | **18×18×24 nm XYZ** = `[24, 18, 18]` nm ZYX, image **and** GT |
| crop ZYX | `[140,240,240] → [555,4530,3585]`, split at z=270 |
| train / val | `270×4290×3345` / `145×4290×3345` |

The source image has a finer 9×9×12 nm mip0, but the proofread segmentation's
finest scale *is* 18×18×24, so 18 nm is the ceiling for paired image+GT. A
higher-resolution variant exists (`tutorials/neuron_nisb/liconn_final_banis+_mip0.yaml`,
image from mip0 + nearest-upsampled GT) but is not part of this pipeline and has
not been trained.

## Before running

Edit **only** [params.yaml](params.yaml): repository checkout, dataset root,
output root, and the existing checkpoint/affinity artifacts. Both step YAMLs
inherit it, so paths are never duplicated.

`params.paths.repository` must point at the **main checkout**, not a worktree —
worktrees do not carry `lib/`, and step 2 needs `lib/abiss/build/ws`.

## Step 1 — affinity

```bash
python scripts/main.py --config tutorials/neuron_liconn_ist/1_affinity.yaml --mode train

python scripts/main.py --config tutorials/neuron_liconn_ist/1_affinity.yaml \
  --mode test --checkpoint <run>/checkpoints/step=00200000.ckpt
```

**Already done.** `outputs/liconn_final_banis_plus_tube/20260728_032436` is a
clean 200k run (val_loss 1.2863 → 1.0601, 2026-07-30) and its val affinity —
`(3, 145, 4290, 3345)` float16, verified non-degenerate — is what step 2 reads.
You do not need to re-run step 1 to reproduce the decode.

Two output conventions step 2 depends on:

- Arrays are ZYX, so **channel `c` is the edge along array axis `c`**: ch0=Z, ch1=Y, ch2=X.
- `channel_activations: scale_sigmoid`, so the stored value is `sigmoid(0.2·logit)`, **not** a probability.

## Step 2 — ABISS decode

```bash
python scripts/main.py --config tutorials/neuron_liconn_ist/2_abiss.yaml \
  --mode test --checkpoint <ckpt>
```

ABISS rather than affinity-CC because CC's errors are false merges baked into
its fragments, which no agglomeration can undo. On MICrONS Pinky, each decoder
at its own leave-one-out optimum: CC 2.03 VOI (oracle-merge ceiling 1.94 — i.e.
nothing recoverable) vs ABISS 1.02 (ceiling 0.90).

### Thresholds do not transfer from the other ABISS tutorials

Because of `scale_sigmoid`, this affinity spans about **[0.03, 0.81]** and never
reaches 0.88. Copying `neuron_microns_pinky_abiss.yaml`'s `ws_high_threshold: 0.88`
puts the seeding threshold **above the data maximum** and seeds nothing.

So `ws_high`/`ws_low` are given as **percentiles** (invariant to the compression),
and `ws_merge_threshold` is absolute in the compressed space and **must be swept
per volume**. `ws_merge_function: max` is monotone-invariant, so the sweep covers
exactly the family of segmentations it would on uncompressed probabilities —
`mean` would not, so do not switch it without uncompressing first.

### Sweeping the merge threshold

```bash
python tutorials/neuron_liconn_ist/sweep_merge_threshold.py --slabs 3
```

One watershed per slab, then the merge step repeated per threshold (ABISS batch
mode), so the sweep costs little more than a single decode. Slabs are **full-Z**
so GT is truncated only in XY, and GT ids are used as-is — deliberately avoiding
the re-cc3d-inside-a-small-crop harness that inverted merge-vs-coverage
comparisons on the NISB liconn volume.

Then set the winning value as `ws_merge_threshold` in
[2_abiss.yaml](2_abiss.yaml). **Until that sweep has been run the value in the
YAML is a placeholder, and any number produced with it should not be quoted as
tuned.**

### Memory

Single-volume ABISS holds the whole affinity in memory: `3 × 145 × 4290 × 3345`
float32 ≈ 25 GB before `ws` internals. Validate on a slab first; for the full
val volume use a big-memory node, or the chunked runner
`scripts/run_abiss_chunk.py` (see `tutorials/neuron_j0126/2_abiss.yaml` for a
chunked-hierarchy example).

## Results

### Whole val (the real number)

Job 2947912 (ABISS decode, 43 min, peak RSS 147 GB) + 2947953 (scoring, 7 min),
all 145×4290×3345 = 2.08 G voxels, mt=0.47:

| | VOI ↓ | split | merge | Adapted-Rand err ↓ | pred segs | GT segs |
|---|---:|---:|---:|---:|---:|---:|
| **ABISS max, mt 0.47** | **0.9129** | 0.6732 | 0.2397 | 0.3195 | 79,056 | 35,815 |

**The threshold above was tuned on slabs and is probably biased high.** Slabs
truncate GT in XY, so cross-slab splits go unpenalised: going slab → volume costs
+0.176 VOI, and **0.149 of that is the split term** (0.524 → 0.673) while merge
barely moves (0.213 → 0.240). A slab objective therefore under-prices splitting
and prefers a higher merge threshold than the volume would. Expect the true
volume optimum at or below 0.45; confirm with a whole-volume batch sweep
(one watershed, several merge thresholds) before treating 0.47 as final.

### Slab sweep (threshold selection only)

Mean over **3 disjoint full-Z 1024² val slabs**, ABISS max-affinity, `ws_high`/`ws_low`
at the 94th/20th percentile (resolving to ≈0.72 / ≈0.20, stable across slabs):

| merge thr | VOI ↓ | split | merge | Adapted-Rand err ↓ |
|---:|---:|---:|---:|---:|
| 0.38 | 1.5563 | 0.3692 | 1.1872 | 0.7128 |
| 0.41 | 0.9036 | 0.4298 | 0.4739 | 0.3424 |
| 0.43 | 0.7842 | 0.4579 | 0.3263 | 0.2582 |
| 0.45 | 0.7388 | 0.4879 | 0.2508 | 0.2003 |
| **0.47** | **0.7367** | 0.5238 | 0.2129 | 0.1799 |
| 0.49 | 0.7484 | 0.5628 | 0.1856 | **0.1731** |
| 0.52 | 0.8368 | 0.6721 | 0.1647 | 0.2176 |

**Read this before quoting 0.7367 — it is not the volume score; 0.9129 above is.**

- **The optimum is flat, the cliff is not.** 0.45–0.49 spans 0.012 of VOI, so the
  exact value barely matters; but 0.41 is already 0.90 and 0.38 collapses to 1.56
  with merge-VOI 1.19. Err high, never low.
- **Slab VOI ≠ volume VOI.** GT objects are truncated in XY, so cross-slab splits
  go unpenalised, and the absolute number is slab-size dependent (one 512² slab
  gave 0.620 at mt=0.45). This is a valid basis for choosing a threshold and for
  comparing decoders at matched settings — it is *not* the volume score. The
  full-val number still needs a big-memory node or the chunked runner.
- **`ws_high`/`ws_low` were not swept**, only mapped through the `scale_sigmoid`
  compression from Pinky's values. Pinky found that band insensitive on its own
  data; that has not been verified here.
- ABISS is **split-dominated at its optimum** (split 0.524 vs merge 0.213), i.e.
  behaving as the conservative decoder it is configured to be. That is the regime
  a downstream error-correction step would target.

Neither of the two 200k runs on this volume had ever been decoded or scored
before this tutorial, so there is no prior number to compare against.
