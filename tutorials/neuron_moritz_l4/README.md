# Moritz L4: masked ABISS decode of a 158-Gvoxel mouse cortex volume

Turns a whole-volume affinity into a neuron segmentation in three steps: mask the
affinity to tissue, decode it with ABISS under competitive nucleus growth, then guard and
score the result. The masks are predictions, not annotations, so the decode stays
ground-truth-free; the 96 manual skeletons are only ever read by the scorer.

The volume is **11.24 x 11.24 x 28 nm (x, y, z)** = `[28, 11.24, 11.24]` ZYX, 3306 x 8534
x 5599 voxels, 63 x 96 x 93 um.

**This tutorial starts at the decode.** Unlike [neuron_j0126](../neuron_j0126/), it does
not train or predict: the affinity already exists as a 645 GB precomputed layer written
by a separate run (`outputs/segem_banis_plus/20260824_165054/`, config preserved in that
directory). It is read in place and never copied — `2_abiss.yaml` deliberately omits
`source_affinity_h5`.

## Run it

Edit **[params.yaml](params.yaml)** — three paths, the two mask paths, and the Slurm
resources for each step. Then:

```bash
python scripts/run_moritz_l4.py
```

That is the whole tutorial. It builds the keep mask, runs the pre-flight box, decodes the
whole volume, guards the output and scores it. Each step is one `sbatch`, chained with
`--dependency=afterok`, so the command queues the pipeline and returns.

```bash
python scripts/run_moritz_l4.py --check              # what exists, what is missing
python scripts/run_moritz_l4.py --dry-run            # print the commands only
python scripts/run_moritz_l4.py --steps abiss,guard  # run part of it
python scripts/run_moritz_l4.py --force guard        # rerun a step that looks complete
```

Before committing a node for a day, run the pre-flight alone — 1.2% of the volume, about
13 minutes, every stage exercised including nucleus growth:

```bash
python scripts/run_moritz_l4.py --steps smoke
```

Each step declares the artifact that proves it finished and is skipped when that artifact
is there, so the same command resumes a partial pipeline. ABISS additionally resumes
**per chunk** from `scratch/done/<task>.txt`, so an interrupted decode continues rather
than restarting.

The one prerequisite the driver will not install is ABISS itself; build it as
[neuron_j0126](../neuron_j0126/README.md) describes, with `-DEXTRACT_SIZE=ON`.

## What each step does

| # | step | config | complete when |
|---|---|---|---|
| 0 | keep mask | `params.yaml` | `keep_mask_z4y8x8.h5` exists |
| 1 | pre-flight | `2_abiss_smoke.yaml` | the smoke `seg` layer is fully written |
| 2 | decode | `2_abiss.yaml` | every storage chunk of `abiss/precomputed/seg` is on disk |
| 3 | guard | — | `reports/guard_seg.json` |
| 4 | score | — | `reports/nerl.json` |

### Step 0 — the keep mask

`keep = NOT(blood vessel) AND tissue`, from a TriSAM vessel prediction (no ground truth
used) plus the volume's own `y < 103 / x < 103` zero-padding border. It drops **5.05%**
of the volume: 2.0% vessel, 3.05% border.

It stays on the 4x8x8 grid the vessel mask was predicted on (4.9 MB) and ABISS upsamples
it per read via `AFF_KEEP_MASK_RATIO`, so no full-resolution copy of a 158-Gvoxel mask is
ever written. `AFF_KEEP_MASK` is honoured for a **precomputed** `AFF_PATH` only since the
keep-mask fix in `lib/abiss/scripts/volume_backends.py`; before that a precomputed
affinity was read unmasked no matter what the config said.

Why it matters here: in the padding walls and inside vessel lumen this affinity reads
~0.90 with 96-99% of voxels above 0.8. Those are solid high-affinity slabs that touch
everything, and the watershed grows straight through them.

### Step 2 — the decode

Affinity → watershed → competitive nucleus growth → mean-edge agglomeration, chunked.
Three parameters carry all the hard-won knowledge, and each comment in `2_abiss.yaml`
gives the measurement behind it:

- **`WS_HIGH_THRESHOLD: 0.95`.** This is the parameter that decides whether the volume
  segments at all. At the previously fitted `0.5`, one watershed object owns ~70% of every
  chunk and agglomeration merely stitches those giants together — the finished volume had
  a single segment holding 67.5% of it. `0.5` won 26/26 val cubes on VOI; a 100^3 cube
  simply has no room for a basin to flood.
- **`AGG_THRESHOLD: 0.80` is inert.** 0.80, 0.95, 0.98, 1.5 and 1e6 produce
  byte-identical segmentations on this volume. It is left at j0126's 0.20-family value
  because nothing measured here justifies any other; do not tune it before the watershed
  is right.
- **`CHUNK_SIZE: [2816, 2944, 32]`.** Sized to fit in RAM, not to minimise stitching. See
  [RESOURCE.md](RESOURCE.md) — this is the difference between a 5-hour decode and a
  20-hour one.

Setting `NUC_PATH` inserts competitive nucleus growth between the watershed remap and
agglomeration: watershed objects holding two nucleus identities are split between them,
and the manifest it writes is the firewall a later error-correction pass reads. Whether
it has anything to correct is a property of the watershed, not of the nucleus mask.

### Steps 3 and 4 — is it any good

The guard is **ground-truth-free** and is the gate: sample the finished segmentation and
report the largest label's share of the volume. A percolated decode is obvious (67.5%);
a healthy one is a few percent. VOI on small cubes cannot see this failure at all, which
is exactly how it survived a week of threshold sweeps.

The scorer reads segment ids at the 263,938 in-bounds skeleton nodes and reports NERL,
skeleton VOI, and a **merge oracle** — every false merge cut, splits untouched. The
oracle is the diagnostic that matters: at 0.92 against a base of 0.0007 it says the
fragments were always good and only the merges were fatal.

**Frame.** The skeletons are in the GLOBAL frame and the segmentation is in the VOLUME
frame, whose origin is global `(118, 0, 0)` ZYX, so a node at nml `z` maps to volume
`z - 118`. Under that mapping 263,938 of 264,926 nodes land in bounds; under the identity
258,764 do. Getting it wrong scores a good segmentation at ~0.

## Results

Whole-volume decodes, scored against the 96 manual reconstructions:

| affinity | decode | largest segment | NERL mt0 | merge oracle | VOI split | VOI merge |
|---|---|---:|---:|---:|---:|---:|
| segem_banis_plus | ABISS, WS_HIGH 0.5, AGG 0.001 | 67.1% | 0.0007 | 0.9221 | 0.2885 | 5.8794 |
| segem_banis_plus | ABISS, WS_HIGH 0.5, AGG 0.80 | 67.5% | 0.0004 | 0.9210 | 0.3264 | 5.8730 |
| segem_banis_plus | + keep mask + nuclei, WS_HIGH 0.95 | **0.73%** | 0.0047 | 0.0052 | 9.1105 | 0.0144 |

"Largest segment" is the guard: one label's share of all voxels. The first two rows are
the same failure at three orders of magnitude of `AGG_THRESHOLD` apart, which is what
finally pointed at the watershed.

**Read the third row honestly.** The percolation is gone -- the guard passes by 7x and
VOI merge falls from 5.87 to 0.014 -- and the segmentation is now shattered instead:
VOI split 0.33 -> 9.11, and 79,645 predicted segments cover the skeletons where 2,100 did.
The diagnostic is the merge oracle: at 0.0052 it equals the base NERL of 0.0047, meaning
there are no false merges left to cut and **splits are the entire remaining ceiling**.

That is the expected consequence of fixing the watershed without re-fitting what comes
after it. `AGG_THRESHOLD` was inert while every chunk contained a watershed giant; with
the watershed healthy at 0.21% it should finally bite, and 0.80 only merges very
confident edges on an affinity whose boundaries sit high. Re-sweeping it -- downward --
is the next step, and it is cheap: the watershed is already computed, so only the last
two stages need to re-run.

Planning figures: [RESOURCE.md](RESOURCE.md). Disk hygiene: [CLEANUP.md](CLEANUP.md).
