# LICONN ExPID96 ("moe") — cross-sample application of the IST-LICONN model

Applies the 200k `banis+` checkpoint trained on the **IST LICONN
final_proofread** volume to all eight **ExPID96 "moe"** expansion-LM volumes, and
decodes each to an instance segmentation with ABISS, published as neuroglancer
precomputed layers published to `gs://donglai_public/liconn/moe/`.

There is **no ground truth for any moe volume.** This workflow is inference +
decode only — no training, no NERL, no VOI. Every parameter below that was fitted
elsewhere is flagged as such, and the one parameter that had to be re-fitted per
volume is the subject of its own section.

## Which volumes, and how each is put on the training grid

The checkpoint is a 3D conv net trained at **[24, 18, 18] nm ZYX**. Voxel size is
not a free parameter for it: neurite caliber measured in voxels is part of what it
learned. moe spacings are *biological* nm — the ND2 records the pre-expansion
pixel size and the expansion factor is divided out — so matching nm really does
match caliber. The eight volumes sit at four expansions, none a native match:

| volume | native ZYX (nm) | factor to [24,18,18] | recipe | prepared |
|---|---|---|---|---|
| `ExPID96_2ndgel_S1_40XW001_18x` | 22.22 × 9.03 × 9.03 | 1.08, 1.99, 1.99 | `--factor 1 2 2` | (585, 1152, 1152) |
| `ExPID96_S1_40XW002_18x` | 22.22 × 9.03 × 9.03 | 1.08, 1.99, 1.99 | `--factor 1 2 2` | (316, 1152, 1152) |
| `ExPID96_2ndgel_S2_40XW002_22x` | 18.18 × 7.39 × 7.39 | 1.32, 2.44, 2.44 | `--target-spacing 24 18 18` | (621, 945, 945) |
| `ExPID96_2ndgel_S2_40XW_22x` | 18.18 × 7.39 × 7.39 | 1.32, 2.44, 2.44 | `--target-spacing 24 18 18` | (643, 945, 945) |
| `ExPID96_2ndgel_S3_40XW004_28xx` | 14.29 × 5.80 × 5.80 | 1.68, 3.10, 3.10 | `--target-spacing 24 18 18` | (507, 743, 743) |
| `ExPID96_2ndgel_S3_40XW_28x` | 14.29 × 5.80 × 5.80 | 1.68, 3.10, 3.10 | `--target-spacing 24 18 18` | (452, 743, 743) |
| `ExPID96_2ndgel_S4_40XW002_32x` | 12.50 × 5.08 × 5.08 | 1.92, 3.54, 3.54 | `--target-spacing 24 18 18` | (415, 650, 650) |
| `ExPID96_2ndgel_S4_40XW003_32x` | 12.50 × 5.08 × 5.08 | 1.92, 3.54, 3.54 | `--target-spacing 24 18 18` | (602, 650, 650) |

**The 18× pair takes an exact block average.** `(1, 2, 2)` lands on
[22.22, 18.06, 18.06] — within 0.3 % of the training XY and 7.4 % of its Z, with
no interpolation at all. That is the recipe validated on the first volume
(affinity QC, threshold sweep, published layer), so the second 18× volume
inherits it unchanged. The NGFF pyramid already in the source cannot supply it:
its level 1 halves Z as well, giving [44.44, 18.06, 18.06].

**The other six take an area-average resample to [24, 18, 18].** `cv2.INTER_AREA`
is the exact area average of the source footprint: for an integer factor it
reproduces the block average to within one gray level, and for a fractional one
it is the anti-aliased generalisation. Interpolating Z by 1.32–1.92 is the price.
The alternative — rounding to the nearest integer factor — would leave XY 18–26 %
off the training scale, and the whole reason this workflow resamples at all is
that voxel size is not a free parameter for the model.

**Their spacing is not exactly the nominal target, and the difference matters
downstream.** `out = round(n / f)` cannot in general tile the source extent, and
INTER_AREA maps the whole source range onto the whole output range, so the
effective spacing is `n · s / out` — within 0.1 % of [24, 18, 18] here. That is
what `prepare_volume.py` records and what the neuroglancer layer must use;
publishing at the nominal value instead would drift the segmentation against the
native image group.

`volumes.py` is the single source of truth for this table and prints it:

```bash
python tutorials/neuron_liconn_moe/volumes.py
```

## Prepared volumes are `.h5`, and that is not a convenience

`inference/output.py::resolve_output_filenames` names the per-volume output
folder from the **last component of the image path**. For an OME-NGFF volume read
as `<vol>.zarr/0` that component is `"0"` for *every* volume — `"0"` is not in
`_UNINFORMATIVE_STEMS`, so the walk-up-to-parent rule that turns `data.zarr/img`
into `seed101` never fires — and the second volume through the checkpoint
silently overwrites the first. Writing `<vol>.h5` gives the folder the volume's
own name.

Note this is a *different* helper from the one that reads `data.test.name`:
`runtime/output_naming.py::resolve_dataset_volume_stems` honours that field (its
documented resolution order 1, and what `config/schema/data.py` advertises it
for), but it is not what writes the file. `data.test.name` is still set, because
it does steer the resolver side (cache detection, tuning).

The first volume predates this fix and its artifacts still live under the `0/`
leaf. `volumes.py::LEGACY` records that rather than rerunning it.

## Steps

All of these are submitted **from the repository root** — the sbatch scripts take
`REPO` from `$SLURM_SUBMIT_DIR` and write logs to `slurm_logs/neuron_liconn_moe/`
relative to it, so they carry no checkout-specific paths.

```bash
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
mkdir -p slurm_logs/neuron_liconn_moe

# 0. Resample onto the checkpoint's grid            (CPU, ~20 s/volume)
sbatch --array=0-6%2 tutorials/neuron_liconn_moe/slurm/0_prepare.sbatch

# 1. image -> 3-channel affinity                    (1 L40S, 4-10 min/volume)
sbatch --array=0-6%4 tutorials/neuron_liconn_moe/slurm/1_affinity.sbatch

# 2. affinity -> ABISS, swept over merge thresholds (CPU, ~10 min/volume)
sbatch --array=0-6%2 tutorials/neuron_liconn_moe/slurm/2_sweep.sbatch

# 3. precomputed layer + meshes                     (CPU 16 cores)
sbatch --array=0-6%3 tutorials/neuron_liconn_moe/slurm/3_precomputed.sbatch

# 4. upload -- needs an interactive `gcloud auth login` first, see below
bash tutorials/neuron_liconn_moe/publish_all.sh

# status of the whole batch at any point
python tutorials/neuron_liconn_moe/summarize_batch.py
```

Array index = position in `volumes.py::PENDING`. Each step writes a per-volume
override config with `make_volume_config.py`, whose `_base_` is the absolute path
to `1_affinity.yaml`, so the recipe lives in one place and the batch adds four
lines per volume.

## Where the outputs land

In `--mode test`, `runtime/checkpoint_dispatch.py` overwrites
`inference.save_path` with `<checkpoint run dir>/test_<ckpt stem>/`, so **the
`save_path:` in these YAMLs is ignored** and results are written next to the IST
training run:

```
outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/
├── val/                          <- the IST tutorial's held-out val (pre-existing)
├── 0/                            <- first moe volume (legacy leaf)
└── <volume name>/                <- every other moe volume
    └── raw_x1_ch0-1-2.h5         <- (3, Z, Y, X) float16

outputs/neuron_liconn_moe/<volume name>/
├── mt_sweep/seg_mt{0..8}.h5      <- one per swept merge threshold
├── mt_sweep.json                 <- the table + the chosen row
└── <volume>_seg_abiss_mt###.h5   <- hard link to the chosen threshold
```

## Does the transfer work?

At the affinity level, yes on all eight, checked with `qc_affinity.py` — the only
readout available without ground truth. Mid-plane percentiles against the
in-domain IST val reference:

| volume | p25 | p50 | p75 |
|---|---|---|---|
| IST final_proofread val (in-domain) | 0.26 | **0.53** | 0.68 |
| `..._S1_40XW001_18x` (published) | 0.26 | 0.47 | 0.67 |
| `..._S1_40XW002_18x` | 0.26 | 0.49 | 0.65 |
| `..._S2_40XW002_22x` | 0.25 | 0.47 | 0.65 |
| `..._S3_40XW004_28xx` | 0.33 | 0.48 | 0.64 |
| `..._S3_40XW_28x` | 0.29 | 0.54 | 0.67 |
| `..._S4_40XW002_32x` | 0.31 | **0.41** | 0.58 |

The systematic difference is a median that drifts *down* with expansion factor,
which tracks image contrast after resampling — mid-plane intensity std falls
59.5 → 55.5 → 45.8 → 40.0 across 18× / 22× / 28× / 32× against 53.7 for the
training volume. A boundary-leaning affinity errs toward over-segmentation at a
fixed absolute threshold, which is why **the merge threshold cannot be shared
across the batch** (next section).

Inference cost: 4–10 min per volume on one L40S, 48 min for all seven.

## What is and is not validated

**Matched:** voxel size (above), and the intensity histogram, which lands close by
coincidence after downsampling (train mean 128.2 / std 53.7 / median 120 vs
127.8–129.1 / 40.0–59.5 / ~120 for the prepared moe volumes). That is why
`normalize: divide-255` is left alone.

**Not matched, and the reason to treat all of this as provisional:** different
sample (ExPID96 vs ExPID82_1), different expansion protocol, and moe went through
a per-plane 1–99 percentile clip + CLAHE that the IST export did not.

## Choosing the merge threshold

This is the whole difficulty of the batch. Everything else — resample, infer,
mesh, upload — is mechanical.

### What transfers is the percentile, not the value

The merge threshold is compared against affinities, and the affinity distribution
is not constant across these volumes (see the table above): its median drifts
down with expansion factor, because image contrast falls as more of the field
resolves into boundary. A fixed absolute threshold therefore means a *different*
operating point on every volume. Two measurements make this concrete:

* **IST's 0.47 fails on moe.** On the first moe volume it produced a largest
  segment of 975 µm³ — 17.35 % of the whole 5624 µm³ field — with a second at
  6.58 % and the third down at 0.63 %. Two runaway merge chains.
* **moe's 0.60 fails on the rest of the batch.** On the 28× and 32× volumes,
  ABISS agglomeration **stops entirely** at 0.68–0.72: every threshold at or
  above that returns the raw watershed unchanged, byte-identical label count and
  coverage, because no region-graph edge is that strong.

So the anchor is carried as a percentile of each volume's *own* affinity. The
published operating point (0.60 on `ExPID96_2ndgel_S1_40XW001_18x`) is the
**62.89th percentile** of that volume's affinity. This is the same reasoning that
already makes `ws_high`/`ws_low` percentiles rather than absolutes — the merge
threshold was simply left behind.

### Why the original "knee" rule had to go

The first run picked 0.60 by hand as the lowest threshold whose largest segment
stayed under 2 % of the volume. Applied to the batch that rule picks the
**saturation plateau on four of seven volumes** — i.e. it publishes the
un-agglomerated watershed at coverage 0.29–0.44 instead of ~0.71. Two independent
reasons:

* **Share is not comparable across fields.** These volumes run 1362–5624 µm³, so
  a fixed 2 % cap is three times stricter on a 32× field than on the 18× one it
  was calibrated on.
* **Size cannot tell a merge chain from a soma.** A 12 µm box can be legitimately
  dominated by a single cell body. Volumes that contain one show a "runaway"
  segment at every threshold below saturation, and the rule walks off the cliff.

### Shape can tell them apart

A merge chain threads the whole field through thin bridges, so its bounding box
covers the field while its fill fraction is a few percent. A soma is compact.
`inspect_top_segments.py` prints volume / bbox / fill per segment, and on the
published volume the separation is unambiguous:

| mt | rank | µm³ | %vol | bbox/field | fill | |
|---|---|---|---|---|---|---|
| 0.47 | 1 | 975.7 | 17.35 % | **1.000** | 0.173 | chain |
| 0.47 | 2 | 370.4 | 6.58 % | **1.000** | 0.066 | chain |
| 0.47 | 3 | 35.6 | 0.63 % | 0.527 | 0.012 | |
| 0.55 | 1 | 128.8 | 2.29 % | 0.775 | 0.030 | |
| 0.60 | 1 | 89.3 | 1.59 % | 0.537 | 0.030 | published pick |

At the known-bad 0.47 the top two segments span **every axis of the field in
full**; at 0.55 and above nothing spans more than 0.78. Both conditions are
needed: a thin process can touch all six faces at 21 µm³ (seen on the 22× volume
at 0.55) and is not a chain. Hence `share ≥ 2 % AND bbox span ≥ 0.90`.

### The rule, and its calibration

**Percentile-matched threshold, vetoed by the chain test and the plateau.** On a
veto the pick steps *up* the grid — a higher threshold merges less — to the first
chain-free threshold that is still doing agglomeration.

Run against the published volume, whose answer is known:

```
    mt    labels  largest %vol  largest um^3  2nd %vol  covered  chains
 0.470     40444        17.35%         975.7     6.58%   0.759       2   <- known bad
 0.550     71068         2.29%         128.8     1.05%   0.738       0
 0.600    105975         1.59%          89.2     0.43%   0.714       0   <- chosen
 0.650    163978         1.22%          68.8     0.36%   0.666       0
 0.700    285900         0.25%          14.3     0.24%   0.475       0
 0.750    333961         0.07%           3.8     0.04%   0.367       0
selected by percentile: target mt 0.6000
```

The chain test fires on 0.47 and only 0.47, and the percentile target lands on
0.60 with no veto needed — reproducing the hand-picked operating point.

### What this does NOT establish

There is still no ground truth. The percentile anchor inherits the first volume's
GT-free pick, and the chain test only rejects one failure mode — it is blind to
false splits, which are the dominant error above the knee. Read these as
defensible operating points for GT-free volumes, not as validated ones. The
18×/22× layers are the closest to the training domain; the 32× pair is the most
provisional, with the lowest image contrast and an affinity median ~0.1 below
in-domain.

## Results, and the published layers

Per-volume operating points, chosen by the rule above. `covered` is the fraction
of the field carrying a label after dust removal; the published volume's 0.714 is
the reference.

| volume | mt | how | largest µm³ | covered | labels | meshed |
|---|---|---|---|---|---|---|
| `ExPID96_2ndgel_S1_40XW001_18x` | 0.600 | percentile | 89.2 | 0.714 | 105,975 | 15,986 |
| `ExPID96_S1_40XW002_18x` | 0.589 | percentile | 185.3 | 0.732 | 67,970 | 15,633 |
| `ExPID96_2ndgel_S2_40XW002_22x` | 0.600 | chain-veto +1 | 87.6 | 0.708 | 108,749 | 20,115 |
| `ExPID96_2ndgel_S2_40XW_22x` | 0.587 | percentile | 60.7 | 0.713 | 89,018 | 14,762 |
| `ExPID96_2ndgel_S3_40XW004_28xx` | 0.537 | percentile | 185.0 | 0.652 | 39,826 | 9,303 |
| `ExPID96_2ndgel_S3_40XW_28x` | 0.611 | percentile | 127.4 | 0.666 | 37,303 | 3,589 |
| `ExPID96_2ndgel_S4_40XW002_32x` | 0.550 | chain-veto +1 | 55.9 | 0.556 | 30,561 | 5,177 |
| `ExPID96_2ndgel_S4_40XW003_32x` | 0.550 | chain-veto +1 | 12.6 | 0.684 | 50,986 | 9,393 |

Two independent 22× volumes landing on 87.6 and 60.7 µm³ at coverage 0.708/0.713
— against the published volume's 89.2 µm³ at 0.714 — is the closest thing to a
cross-check available without ground truth.

**Read the 28× and 32× rows with more caution than the rest.** `S3_40XW004_28xx`
keeps two ~8 %-of-field segments at every threshold below its plateau; the shape
test clears them (neither spans the field), so they are large real objects rather
than chains, but its coverage is the second lowest in the batch.
`S4_40XW002_32x` has both the lowest coverage (0.556) and the lowest affinity
median (0.417 against 0.53 in-domain), which is consistent with its image
contrast — mid-plane std 40.0 against 53.7 for the training volume.

Meshes cover 92–95 % of foreground per volume (objects ≥ 1000 voxels); the median
ABISS object is ~320 voxels of dust and is deliberately not meshed.

### Publishing

`upload_seg_precomputed.py` builds under `file://` and ships with
`gcloud storage rsync`; `publish_all.sh` loops it over the batch. Target:

```
gs://donglai_public/liconn/moe/<layer>
precomputed://gs://donglai_public/liconn/moe/<layer>
```

`python tutorials/neuron_liconn_moe/summarize_batch.py --urls` prints them all.
The bucket and prefix live in `volumes.py::GCS_BUCKET` / `GCS_PREFIX`; nothing
else hard-codes them.

**Three things follow from this bucket being a different one from the images.**

1. **Image and segmentation now live in separate buckets.** The eight OME-Zarr
   image groups stay at `gs://donglai/liconn/moe/clip_percentile_1_99/`, which is
   private and served through ngauth. A neuroglancer view that overlays the two
   spans both buckets and may need two credentials. Alignment is unaffected —
   both carry true physical resolution and the resample preserves the corners, so
   there is still no offset.
2. **`donglai_public` is not (yet) anonymously readable**, despite the name. As of
   2026-09-04 an unauthenticated
   `https://storage.googleapis.com/storage/v1/b/donglai_public/o` returns **401**
   — identical to the known-private `donglai` bucket, where a genuinely public
   bucket answers 200. So the plain `precomputed://gs://...` URL above will not
   load for an anonymous viewer until `allUsers:objectViewer` is granted on the
   bucket. The alternative is the ngauth form (`volumes.py::NGAUTH`), but it has
   not been established that the deployed ngauth server is authorised for this
   bucket — it was set up for `donglai`.
3. **The `clip_percentile_1_99` component is gone from the path.** In the source
   tree that component is load-bearing: a second clip variant with *identical*
   dataset names exists at `preprocessed/zarr/`. These segmentations all derive
   from the percentile variant, so there is no collision today, but publishing the
   fixed-window variant's segmentations later would need a prefix again.

**Uploading needs an interactive login and cannot be done from a compute node.**
`gcloud`'s refresh token for `donglai@mindspan.org` expires and can only be
renewed by a browser flow — and that is the only identity available here, since
there is no ADC on this host (so CloudVolume / igneous / tensorstore cannot
authenticate at all). Write access to `donglai_public` for that account has not
been verified from this session. So the build runs unattended and the upload is a
separate manual step:

```bash
~/google-cloud-sdk/bin/gcloud auth login
bash tutorials/neuron_liconn_moe/publish_all.sh          # uploads + verifies all seven
```

Everything is written with **gzip off**, and there is a pre-upload guard that
scans for the gzip magic number and refuses to proceed: `gcloud storage rsync`
uploads bytes verbatim and does not set `Content-Encoding`, so a gzipped chunk
arrives as gzip bytes that neuroglancer reads as raw and fails on. Both
CloudVolume's `compress=True` and igneous meshing's `compress='gzip'` — which is
the **default** — would do this. The size is recovered with
`encoding="compressed_segmentation"` instead, which neuroglancer decodes natively.

## Always check the decoded volume is non-empty

`sweep_merge_threshold.py::check_nonempty` now enforces this, because it keeps
happening. ABISS can log correct supervoxel counts and still have its
`seg_*.data` payload come back all zeros from `/projects`:

* once on a single decode (SLURM job 2952245), which still reported
  `[OK]Test completed successfully` — `decoding/utils.py::cast2dtype` read
  `max() == 0`, chose `uint8`, and wrote a 1.39 MB volume of nothing;
* and on **three of seven** sweeps run concurrently with eight diagnostic jobs,
  where 6 of 9 outputs per job were empty. It surfaced only as an opaque
  `IndexError` on an empty percentile.

The tell is **identical file sizes across thresholds that must differ** (3.4 MB
of gzipped zeros against 48 MB). Nothing else downstream notices: the h5 exists,
it is merely empty, and it would be meshed and published as such. The cache
preflight will then *reuse* it — a bad artifact has to be moved aside before a
retry re-decodes.

**Do not co-schedule these.** The re-run that fixed it was throttled to two
concurrent jobs.

## Related

- `tutorials/neuron_liconn_ist/` — the volume this checkpoint was trained on, and
  where the IST thresholds were fitted.
- `/projects/weilab/dataset/liconn/moe/preprocessed/README.md` — provenance of the
  eight ND2 → uint8 volumes, the two clip variants, and the voxel-size caveat
  (spacing is *inferred* from the expansion factor in the filename, not recorded
  in the ND2 metadata).
