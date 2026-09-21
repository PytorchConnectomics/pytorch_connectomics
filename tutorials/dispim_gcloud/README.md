# Paired diSPIM SNEMI: YAML → four L4s → CPU ABISS → arand

`dispim_snemi_20epoch.yaml` trains from scratch for 20 epochs / 4,000 optimizer
steps on the 20 nm simulations in `gs://donglai/dispim/snemi/snemi-train.zarr`.
It tests on the separate `snemi-test.zarr`. Both contain registered ZYX
`image_blur_A` and `image_blur_B` arrays, shape `150×307×307`. Inputs become one
`[2,Z,Y,X]` HDF5 `main` dataset, ordered A then B; a YAML list would mean
separate samples and is not a paired-view representation.

## Architecture and research basis

The local research notes are `kb/reports/dispim_dual_encoder.md` (August 5,
2026) and `kb/initiative/wiki/human_udc/diSPIM_deep_research.md` (July 22, 2026)
in the surrounding `dw-os` workspace. The former recommends separate shallow
encoders, shared deeper features, reliability fusion at every scale, and one
decoder. This recipe uses the existing `mednext_dual_view / partial_gated`
implementation: separate stems/stage-0/down-0; shared deeper MedNeXt-S blocks;
fixed-angle FiLM; equal-initialized spatial/channel gates; one decoder with
nine affinity outputs and deep supervision. Nearest-neighbor channels are
**XYZ**, stored at the destination voxel; radii 3 and 9 are auxiliary targets.

The report is a design hypothesis with unresolved citations, not a measured
comparison. The implementation shares whole deeper blocks including GroupNorm
and uses angle FiLM; it does not implement the report's separate per-view
GroupNorm affine parameters. Its gate also follows the existing repository
variant. This run measures that implemented variant with independent per-view
normalization and 15% whole-view dropout. It makes no claim about real diSPIM
accuracy or superiority to alternative encoders.

The simulator source is `dispim_sim/tutorials/snemi` in the adjacent repository.
The cloud metadata describes uniform 20× expansion, 0.4 µm detector spacing
(20 nm effective), orthogonal PSFs, whole-volume rendering, and no spatial
motion, sectioning, tiling, or added intensity noise. A and B therefore already
share a grid. Finer sampling does not add anatomical information to SNEMI.

## Prepare data without changing the source Zarr groups

Download both cloud groups with `gcloud storage rsync --recursive`. The paired
preparer requires the simulator's original `snemi3d_viewer.zarr` containing
`segments` and `test_segments`, each `100×1024×1024`. Run with an environment
that already includes Zarr v3, NumPy, and h5py (the simulator's `.venv` does):

```bash
../dispim_sim/.venv/bin/python tutorials/dispim_gcloud/prepare_data.py \
  --input-dir /absolute/path/to/downloaded-groups \
  --labels-root ../dispim_sim/data/original/snemi3d_viewer.zarr \
  --output /absolute/path/to/new-prepared-directory
```

The script validates simulator metadata and array geometry, preserves integer
instance IDs, and maps labels with voxel-center nearest-neighbor sampling:
`floor((output_index + 0.5) * input_size / output_size)`. These are the same
coordinates used by the simulator's `align_corners=False` foreground renderer;
instance IDs themselves are never linearly interpolated. Training uses rendered
Z `[0:120]`, validation `[120:150]`, and testing uses the separate full test
volume. Provenance records source metadata, label/view hashes, split bounds,
effective spacing, and output file hashes. Do not regenerate test images from
the training labels or tune decoder settings against the test labels.

## Docker stages

Build the tested base as described in `tutorials/neuron_snemi_gcloud`, or load
its saved image archive after verifying its checksum. Then build this checkout:

```bash
docker build -f tutorials/dispim_gcloud/Dockerfile -t pytc:dispim-abiss .
```

Build-time smoke checks run real dual-view forward/backward and real compiled
ABISS on perfect affinities, requiring full-volume arand 0. The GPU stage also
performs an actual-data training/validation step before the full run.

Mount prepared files at `/workspace/datasets/diSPIM_SNEMI` and a writable run
directory at `/workspace/outputs`. Each stage requires a new output directory:

```bash
# Four visible L4s, batch size 2 per device, 64³ patches, BF16, checkpointing.
docker run --rm --gpus all --ipc=host \
  -v "$PWD/prepared:/workspace/datasets/diSPIM_SNEMI:ro" \
  -v "$PWD/results:/workspace/outputs" pytc:dispim-abiss \
  python -m connectomics.runtime.dispim_benchmark --stage train \
  --config tutorials/dispim_gcloud/dispim_snemi_20epoch.yaml \
  --output /workspace/outputs/train

# Fresh container, x1 inference using one GPU and verified final weights.
docker run --rm --gpus all --ipc=host \
  -v "$PWD/prepared:/workspace/datasets/diSPIM_SNEMI:ro" \
  -v "$PWD/results:/workspace/outputs" pytc:dispim-abiss \
  python -m connectomics.runtime.dispim_benchmark --stage infer \
  --config tutorials/dispim_gcloud/dispim_snemi_20epoch.yaml \
  --output /workspace/outputs/infer --checkpoint /workspace/outputs/train/final.ckpt

# Run on a separate CPU VM, after copying the affinity artifact from GCS.
docker run --rm --ipc=host \
  -v "$PWD/prepared:/workspace/datasets/diSPIM_SNEMI:ro" \
  -v "$PWD/results:/workspace/outputs" pytc:dispim-abiss \
  python -m connectomics.runtime.dispim_benchmark --stage score \
  --config tutorials/dispim_gcloud/dispim_snemi_20epoch.yaml \
  --output /workspace/outputs/score --prediction /workspace/outputs/infer/affinities.h5
cat results/score/metric.txt
```

The final metric is adapted Rand **error**, lower is better, on the full
`150×307×307` rendered test grid, ignoring ground-truth background according to
the canonical evaluator. It is not the historical vEM challenge crop and is
not directly comparable to the previous vEM score. ABISS thresholds are fixed
at the YAML values; no test-label search is performed. `metrics.json` records
support, precision, and recall; `manifest.json` records hashes and commands.

## Google Cloud lifecycle

Create a unique GCS run prefix with `prepared/`, `vm_stage.sh`, the recipe
uploaded as `config.yaml`, and the built
`build/image.tar.gz`, `build/image.json`, `build/image.sha256`. The checksum
file must name `image.tar.gz` with no directory prefix. The VM binds the
uploaded `config.yaml` as a sibling `run.yaml` and saves that exact file beside
the logs; the runtime manifests hash it. `launch_stage.py`
checks prerequisites before allocating a VM:

```bash
python tutorials/dispim_gcloud/launch_stage.py gpu \
  --run-prefix gs://donglai/dispim/snemi/cloud-runs/UNIQUE-RUN \
  --project sunny-catalyst-506019-a2 --zone us-west1-a --name UNIQUE-GPU-VM \
  --service-account pytc-trainer@sunny-catalyst-506019-a2.iam.gserviceaccount.com
```

`vm_stage.sh` uses fresh Docker containers for smoke, training, and inference.
It stores unique checkpoint snapshots every 15 minutes and uploads the final
checkpoint before inference. Verify `gpu/COMPLETE`, the manifest, final epoch,
and uploaded affinities; then delete the GPU VM and verify its boot disk is
gone. Launch `cpu` with a new VM name using the same command (for example,
`--zone us-central1-a`). CPU scoring runs on `e2-standard-16` without a GPU.
Verify `cpu/COMPLETE`, `cpu/score/metric.txt`, and artifact hashes, download the
results, and delete that VM/disk. Both stages have auto-delete boot disks and
maximum run durations (12 hours GPU, 6 hours CPU) as a backstop. Failure logs
are uploaded before exit; never treat a failed or missing marker as success.

The prior cloud run exposed two pitfalls handled here: a copied resume input
can make Lightning write `last-v1.ckpt`, and a long-lived container can lose
CUDA visibility. This run starts fresh, verifies checkpoint epoch/step, and
opens a new GPU container for inference. It never silently substitutes a CPU
training run.

Lightning 2.6.5 refreshes `last.ckpt` at epoch end only when its ranked
checkpoint is saved. This recipe sets `save_top_k: -1` so every epoch is
durable. The stage runner verifies both epoch and optimizer step;
`train --checkpoint /path/to/checkpoint.ckpt` can resume completed epochs
without resetting optimizer/scheduler state.

## Validation and cloud run

All 50 focused tests pass: paired data/model, checkpoint guards and resume,
integer label mapping, GPU requirements, CLI argument parsing, stage overrides,
and full-grid/challenge-crop evaluation. The YAML passes the canonical tutorial
validator; shell syntax and diff whitespace checks pass.

Run prefix: `gs://donglai/dispim/snemi/cloud-runs/dispim-snemi20-20260910/`.
The image passed real dual-view gradients and real ABISS perfect-affinity arand 0.
Actual-data GPU smoke passed. Four L4s trained the model; the final checkpoint
has epoch index 19 / global step 4,000. The final three epochs were resumed
from the epoch-17 checkpoint after the guard detected stale `last.ckpt` weights.
Inference produced finite `[3,150,307,307]` XYZ affinities. The checkpoint,
config, and prediction hashes were independently verified after download.
The separate CPU-only ABISS run returned **adapted Rand error 0.254113117313**
(lower is better), precision **0.698403599152**, recall **0.800297779821**, on
the full `150×307×307` rendered test grid. The report, segmentation and raw
input hash passed download verification. See `result_20nm.json`.
The builder, GPU and CPU VMs and their disks are deleted; final cloud listings
confirm no resources for this run remain. Reports, logs, provenance, the
checkpoint and predictions are retained locally under
`.gcp-run/dispim-snemi20-20260910/results/` and in the GCS run prefix.

Recovery records are preserved under `failed-smoke-accelerator/`,
`failed-image-identity/`, `failed-checkpoint/`, `failed-infer-path/`, and
`failed-infer-cli/`. Corrections use PyTC accelerator `cuda`, compare image
layers/runtime settings across Docker storage backends, save every epoch,
anchor the data root, and place positional YAML overrides after CLI flags.
The GPU test initially also decoded because runtime dispatch reapplied YAML
stage defaults over CLI flags. Plain test mode now retains the CLI-resolved
config; only tune-test transitions stages. CPU ABISS independently scores the
saved raw predictions. Recovery source and commands are in `recovery-source/`.

The 10 nm and 5 nm simulations use separate upload destinations,
`snemi-{train,test}-{10,5}nm.zarr`. All four uploads passed checksum-only rsync
dry-run verification (515 files per 10 nm group, 2,051 per 5 nm group).
This run uses only the original 20 nm pair.

## 10 nm and 5 nm runs

Use `dispim_snemi_10nm_20epoch.yaml` or `dispim_snemi_5nm_20epoch.yaml`.
Both inherit the 20-epoch recipe, changing voxel spacing and run names.
They start from scratch with the same 4,000-step budget, 64³ voxel patches,
nine affinity targets and fixed ABISS voxel thresholds. Their physical patch
widths are 640 nm and 320 nm, versus 1,280 nm at 20 nm; these experiments
compare fixed voxel settings, not matched physical context.

The preparer accepts `--spacing-nm 10` or `--spacing-nm 5` and reads the
corresponding `snemi-{train,test}-{10,5}nm.zarr` groups. It streams Z slabs,
preserves integer IDs, and hashes logical arrays without allocating the full
paired volume. The 10 nm grid is `300×614×614`, split at training Z240;
the 5 nm grid is `600×1229×1229`, split at training Z480. Test volumes remain
separate and use their full grid for adapted Rand error.

For these inherited recipes, `vm_stage.sh` binds external `config.yaml` to
`/workspace/tutorials/dispim_gcloud/run.yaml`, preserving the baked shared
recipe. Use a newly built image containing the 20 nm recovery fixes.
`launch_stage.py` selects 200 GB disks and an `e2-standard-16` CPU scorer for
10 nm; 500 GB disks and `n2-highmem-32` (256 GB RAM) for 5 nm. Both train on
four L4s. CPU runs have a six-hour deletion backstop.

The runs are `dispim-snemi10-20260910` and `dispim-snemi5-20260910`
under `gs://donglai/dispim/snemi/cloud-runs/`. Both prepared datasets and the
rebuilt image are uploaded and checksum verified. The builder and its disk
are deleted.

The 10 nm run completed 20 fresh epochs / 4,000 optimizer steps on four L4s.
Inference produced finite `[3,300,614,614]` XYZ affinities. Separate CPU ABISS
returned **adapted Rand error 0.895048196298**, precision **0.0557092680451**,
recall **0.904133310809**, over the full test grid. Download verification
checked checkpoint, config, prediction and segmentation hashes, shapes, and
metric consistency. See `result_10nm.json`. Its GPU and CPU VMs and disks are
deleted. A transient CUDA/NVML smoke failure before training was preserved
under `failed-cuda-smoke/`; the fresh retry completed all stages.

The 5 nm four-L4 run completed 20 fresh epochs / 4,000 steps in `us-east1-c`.
Inference took about 28 minutes and produced finite `[3,600,1229,1229]`
XYZ affinities. Independent CPU ABISS on `n2-highmem-32` (256 GB RAM)
returned full-grid **adapted Rand error 0.917821129022**, precision
**0.042899286873**, recall **0.973951505753**. These statistics are consistent
with severe merging. Download verification checked the checkpoint epoch and
step, probability range, full shapes, config/prediction/segmentation hashes,
and metric consistency. See `result_5nm.json`. Local state is recorded in
`.gcp-run/dispim-fine-20260910.json` and each run's `state.json`.

| Spacing | Full test grid | Epochs / steps | Fixed ABISS arand ↓ |
|---|---|---|---|
| 20 nm | 150×307×307 | 20 / 4,000 | 0.254113117313 |
| 10 nm | 300×614×614 | 20 / 4,000 | 0.895048196298 |
| 5 nm | 600×1229×1229 | 20 / 4,000 | 0.917821129022 |

These experiments retain voxel-scale settings, so physical patch context,
erosion radius and ABISS size thresholds differ across resolutions.

Each run retains `gpu/infer/affinities.h5` for inference and
`cpu/score/segmentation.h5`, `cpu/score/metric.txt`, and `cpu/score/metrics.json`
for ABISS and evaluation in Cloud Storage. Checkpoints, resolved configs,
logs and provenance are also retained. A stage's `COMPLETE` marker is written
only after successful computation and upload; VMs are deleted after the
uploaded artifacts pass independent download verification.

### Erosion and the 10 nm ABISS follow-up

The inherited target preprocessing uses `label_transform.erosion: 1` at all
three resolutions. This scalar is an XY-only neighborhood half-size:
within each Z slice, a 3×3 window containing multiple positive instance IDs
marks its center as background. There is no Z erosion. Changing the YAML
resolution does not rescale this operation. Thus its physical XY radius is
20 nm, 10 nm and 5 nm respectively. Radii 2 and 4 at 10 nm and 5 nm would
match the 20 nm radius; that would change training targets and require a
separate training experiment. Evaluation reads the original prepared labels.

On the corresponding held-out validation slabs, the fraction of positive
label voxels removed by the current radius 1 is 8.76% at 20 nm, 2.97% at
10 nm and 1.16% at 5 nm. Using radii 2 and 4 instead gives 8.79% and 8.78%
at 10 nm and 5 nm. These are target preprocessing measurements, not scores
from retrained models.

The 10 nm result is dominated by merging: its largest predicted segment
contains 77.6% of all ground-truth foreground voxels. The requested decoder
follow-up keeps the checkpoint and targets fixed. Infer the independent
`60×614×614` validation slab using
`dispim_snemi_10nm_abiss_validation.yaml`, then run ABISS on a separate CPU VM.
The 60-slice validation input is symmetrically minimum-padded to 64 before
context padding. Its YAML removes two extra slices at each Z end to recover
exactly the original 60 slices; the initial padded artifact is archived under
`failed-validation-padding/` and is not used for scoring.
The prespecified search uses high thresholds 0.9/0.95/0.99/1.0, low thresholds
0.1/0.5, size/dust pairs 0/0, 800/600 and 6400/4800, and mean merge thresholds
0.5/0.8/0.95 (only 0.5 when size is zero): 56 candidates. The largest size
pair matches the original 20 nm physical volumes. Select by validation arand,
freeze the selected settings, then evaluate the independent full test volume
once. The baseline test result remains unchanged.

This follow-up is saved under the 10 nm run's
`abiss-validation-tune-20260910/` prefix. Its local state and orchestration
are in `.gcp-run/dispim-snemi10-abiss-tune-20260910/`.

All 56 validation candidates completed. With low=0.1 and mean merge=0.5:

| High threshold | Size / dust | Validation arand |
|---|---|---|
| 0.9 | 800 / 600 (baseline) | 0.960835420294 |
| 1.0 | 800 / 600 | 0.291216122092 |
| 1.0 | 6400 / 4800 (selected) | 0.209649394321 |

The validation-selected configuration returned full-test arand
**0.770851423851**, versus baseline **0.895048196298**. Precision rose to
**0.980713989508**, but recall fell to **0.129730350620**: the output is now
strongly fragmented. Its largest segment contains 2.53% of labeled test
voxels, versus 77.6% before; 2.88% of labeled voxels are predicted background.
The large validation improvement did not transfer well to the independent
test volume. This experiment does not establish convergence or identify
erosion as the sole cause of poor segmentation.

The original 20-epoch weights and targets are unchanged. See
`dispim_snemi_10nm_abiss_selected.yaml` and `result_10nm_abiss_tuned.json`.
The selected settings remain a separate experiment, rather than replacing
the original recipe. Cloud outputs include validation affinities under
`gpu/infer/` and all trial scores, the frozen selection, selected YAML,
validation/test segmentations, and metrics under `cpu/sweep/`.
Independent download verification checked the selection against all 56
trials, hashes and shapes, and recomputed the full-test metric locally.

Both tuning VMs and their boot disks are deleted. All three baseline runs
have completed their training, inference and CPU scoring stages.

The 5 nm GPU and CPU VMs and their disks are deleted. Final project listings
show no remaining diSPIM run VMs or disks. All result artifacts remain in GCS.

### EM-intensity test volumes (existing checkpoints)

`dispim_snemi_em_test_20nm.yaml` and `dispim_snemi_em_test_10nm.yaml` apply the
existing mask-simulation-trained 20-epoch checkpoints to
`gs://donglai/dispim/snemi/snemi-test-em-{20nm,10nm}.zarr/`. These paired A/B
views were rendered from the SNEMI test EM intensities rather than foreground
masks. This is an inference-only transfer experiment: no retraining, target
changes, or parameter selection on the new test volumes. The image and label
grids match the previous tests, and the original mapped test labels are reused.

| Test input | ABISS settings | Full-volume arand error ↓ |
| --- | --- | ---: |
| EM-intensity, 20 nm | Fixed baseline | 0.574498206948 |
| EM-intensity, 10 nm | Fixed baseline | 0.894734413457 |
| EM-intensity, 10 nm | Previously selected on mask-simulation validation | 0.737825236834 |

One L4 performed inference; a separate `e2-standard-16` CPU VM ran ABISS.
Downloaded prediction and segmentation hashes and shapes passed verification,
and all three full-volume metrics and precision/recall were independently
recomputed. See `result_em_test_20260910.json` for checkpoint identities and
artifact pointers. Cloud outputs, exact YAMLs and provenance are under
`gs://donglai/dispim/snemi/cloud-runs/dispim-em-test-20260910T201241Z/`.
Both run VMs and their boot disks were deleted after verification. The uploaded
result summary and provenance pointer were downloaded again and matched locally.
