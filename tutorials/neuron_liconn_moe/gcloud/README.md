# LICONN moe on Google Cloud

Runs the [parent tutorial's](../README.md) pipeline — resample → affinity →
ABISS → precomputed — on a Google Cloud GPU VM instead of BC SLURM, reading the
source volume from GCS and publishing the results back beside it.

First target: **`ExPID108_32x_Cortex_L1_01`**, which arrived already published at
`gs://donglai_public/liconn/moe/expid108/image/`. It is not on `/projects`, so
BC cannot run it without staging Mindspan collaborator data onto BC scratch.

**There is no ground truth for this volume.** This is inference and decode only:
no NERL, no VOI, and the merge threshold is picked GT-free. Everything in
[the parent README's](../README.md) "What this does NOT establish" applies here.

## There is one recipe, not two

The container runs the **same scripts** as the SLURM batch —
`run_prepare.py`, `make_volume_config.py`, `scripts/main.py`,
`sweep_merge_threshold.py`, `upload_seg_precomputed.py`. All of them resolve
paths through [`volumes.py`](../volumes.py), which now reads its roots from the
environment and defaults to the BC paths:

| variable | BC (default) | container |
|---|---|---|
| `LICONN_MOE_REPO` | `/projects/weilab/weidf/lib/pytorch_connectomics` | `/workspace` |
| `LICONN_MOE_SRC_ZARR` | `…/preprocessed/clip_percentile_1_99/zarr` | `/work/src` |
| `LICONN_MOE_PREPARED` | `…/prepared_train_grid` | `/work/prepared` |
| `LICONN_MOE_OUT_ROOT` | `…/outputs/neuron_liconn_moe` | `/work/out` |
| `LICONN_MOE_CKPT` | the BC 200k checkpoint | the HuggingFace copy |
| `LICONN_MOE_GCS_PREFIX` | `liconn/moe/clip_percentile_1_99` | `liconn/moe/expid108` |

So there is no second copy of the recipe table, the resample, the threshold rule
or the uploader to drift out of sync, and a SLURM submission with a clean
environment behaves exactly as before. `run_volume.sh` sets all six.

## The checkpoint is the same weights, from a different place

BC reads `outputs/liconn_final_banis_plus_tube/20260728_032436/checkpoints/step=00200000.ckpt`.
The cloud run pulls [`pytc/liconn`](https://huggingface.co/pytc/liconn)
`affinity_expid82_18nm_128x128x128.ckpt` — same specimen (ExPID82_1), same
`[24, 18, 18]` nm grid, same 200 k steps, **0.9129 VOI** on the held-out IST
validation volume. Its model card is the published description of what this
pipeline is applying, including the fact that the stored affinity is
`sigmoid(0.2 · logit)` and spans about `[0.01, 0.80]` — which is why every
threshold here is a percentile.

`LICONN_MOE_CKPT` must sit under a `YYYYmmdd_HHMMSS` directory.
`runtime/checkpoint_dispatch.py::get_output_base_from_checkpoint` walks the
checkpoint's parents looking for exactly that pattern to decide where test
outputs land; with no such ancestor it falls back to `<ckpt>/../../<stem>`,
which for a checkpoint in a top-level directory is a path at the filesystem
root. `run_volume.sh` stages it as
`/work/ckpt/20260921_000000/checkpoints/<file>.ckpt`. The timestamp is arbitrary
and fixed; it is not a claim about when anything was trained.

## The grid, and why one VM is enough

| | |
|---|---|
| source | `(965, 2304, 2304)` at `[12.5, 5.078125, 5.078125]` nm ZYX — 5.1 Gvoxel, 32× |
| recipe | `--target-spacing 24 18 18` (factors 1.92, 3.545, 3.545) |
| prepared | `(503, 650, 650)` at `[23.98, 18.0, 18.0]` nm ZYX — **213 Mvoxel** |
| affinity | `(3, 503, 650, 650)` float16 — 1.28 GB |
| field | 12.1 × 11.7 × 11.7 µm = 1651 µm³ |

Two consequences of that 24× reduction, both of which contradict guidance
written for native-resolution volumes:

* **Whole-volume ABISS fits.** `run_abiss_volume.py` is capped by ABISS's
  `uint32` internal watershed index at 2,147,483,648 voxels; 213 Mvoxel is
  **9.9 %** of it, at roughly 15 GB peak RSS against the measured ~71 GB per
  Gvoxel. The chunked path is not needed. A decode at *native* resolution would
  be 5.1 Gvoxel — over the cap and ~360 GB — which is where the "chunked decode
  is mandatory" rule comes from.
* **One VM does the whole job.** `g2-standard-16` (1× L4, 16 vCPU, 64 GB) covers
  GPU inference, the decode and igneous meshing. Releasing the GPU before the
  decode is worth ~28 % of a large bill; here the decode is minutes and moving
  1.3 GB of affinity between machines costs more than it saves. Revisit for a
  batch.

Only one GPU is used. The model's `GroupNorm` has no running statistics, so the
forward pass is window-size dependent and the inference ROI is **fixed at
`[128, 128, 128]`** to match training — extra devices help a batch, not a
volume.

## Run it

### The image is a GCS object, not a registry image

This follows the pattern that already worked for
[`tutorials/dispim_gcloud`](../../dispim_gcloud/) from this same Mac: a disposable
builder VM builds the image, `docker save | gzip`s it, and publishes
`image.tar.gz` + `image.json` + `image.sha256` to a stable GCS prefix. Each run VM
verifies the checksum, loads it, and then re-verifies the loaded image's filesystem
layers and runtime config against the published `image.json`.

Three reasons, in order of weight:

1. **Cloud Build can build in this project but cannot publish.** It now runs as the
   Compute Engine default service account, which holds no Artifact Registry binding,
   and only the project *owner* can grant one — `roles/editor` excludes every
   `setIamPolicy` permission except the bucket-level one. Measured, not assumed: the
   build step succeeds and the overall build fails at the push, leaving 0 images in
   the repo. Naming the account that *does* hold `roles/cloudbuild.builds.builder` is
   refused, because Cloud Build requires a user-managed service account.
2. This workstation is an arm64 Mac with **no Docker installed**, so a local build
   would be an emulated cross-build of PyTorch plus a C++ watershed.
3. A GCS archive needs only **bucket** permissions, which the operating account
   already has — so nothing on this path waits on an owner.

The archive's sha256 is this run's image identity, recorded in the run manifest as a
registry digest would be. Cost: a ~4-5 GB object and a ~1-2 minute load on each run
VM, skipped entirely when the disk already holds a matching image.

### One-time setup

```bash
# Disposable builder VM: docker/Dockerfile, then this tutorial's layer, saved and
# published. ~25-35 min. The VM deletes itself either way and uploads its log
# even on failure.
bash tutorials/neuron_liconn_moe/gcloud/launch.sh --build
bash tutorials/neuron_liconn_moe/gcloud/launch.sh --build-status
```

The image lands at `gs://donglai/liconn/moe/images/$IMAGE_ID/build/` and is
**reused across runs** — a second volume does not rebuild it. Bump `IMAGE_ID` to
publish a new one beside it rather than overwriting.

The VM service account also needs bucket access. `--preflight` prints these and
never runs them:

```bash
SA=pytc-trainer@sunny-catalyst-506019-a2.iam.gserviceaccount.com
gcloud storage buckets add-iam-policy-binding gs://donglai_public \
  --member=serviceAccount:$SA --role=roles/storage.objectAdmin
gcloud storage buckets add-iam-policy-binding gs://donglai \
  --member=serviceAccount:$SA --role=roles/storage.objectAdmin
```

The VM runs unattended, so **the service account is the only identity available** —
there is no interactive `gcloud auth login` on it, and `donglai@mindspan.org` being
able to read and write proves nothing about whether the run can. `objectAdmin` on
`donglai_public` covers both reading the source group and writing results, which is
why one grant satisfies two preflight rows.

### Each run

```bash
bash tutorials/neuron_liconn_moe/gcloud/launch.sh --preflight   # changes nothing
bash tutorials/neuron_liconn_moe/gcloud/launch.sh --run
RUN_ID=<printed> bash tutorials/neuron_liconn_moe/gcloud/launch.sh --status
```

`--run` re-runs `--preflight` first and refuses to launch if anything is missing.
Another volume:

```bash
VOLUME=<name> SRC_ZARR=gs://.../<name>.zarr PUBLISH_PREFIX=gs://.../<prefix> \
  bash tutorials/neuron_liconn_moe/gcloud/launch.sh --run
```

Both stage scripts are uploaded to GCS and fetched by a three-line bootstrap
startup script, so a stage script can be corrected and retried without recreating
the VM.

### What lands where

| artifact | destination |
|---|---|
| affinity, `(3,Z,Y,X)` float16 | `gs://donglai_public/liconn/moe/expid108/affinity/<vol>_affinity_x1_ch0-1-2.h5` |
| segmentation h5, uint32 | `…/expid108/seg/<vol>_seg_abiss_mt###.h5` |
| threshold sweep table | `…/expid108/seg/<vol>_mt_sweep.json` |
| precomputed layer + meshes | `…/expid108/<vol>_seg_abiss_mt###/` |
| log, run manifest, image identity | `gs://donglai/liconn/moe/runs/<RUN_ID>/` — **private** |
| the image archive (shared across runs) | `gs://donglai/liconn/moe/images/<IMAGE_ID>/build/` — **private** |

The run manifest records the image archive's sha256, the source commit and whether
the tree was dirty, the checkpoint's origin and training set and its fixed inference
ROI, and the zero-shot caveat. `STATUS` holds the exit code; `COMPLETE` is written
only after the work *and* the upload succeed, so a missing marker never reads as
success. Intermediates and logs go to the private bucket; only the results are
published.

**`donglai_public` is not anonymously readable**, despite the name. Its IAM
policy has no `allUsers` binding — verified 2026-09-04, re-confirmed
2026-09-21 — and an unauthenticated request returns 401. So publishing here
does not expose the volume, and the plain `precomputed://gs://…` URL will not
load for an anonymous viewer. Use an authenticated session.

## Failure modes this is built around

Each of these has cost a run somewhere in this project's history, and each is
now checked rather than hoped for.

* **A silent all-zero decode.** ABISS can log correct supervoxel counts and
  write a volume of nothing, exiting `[OK]`; nothing downstream notices, and a
  cache preflight will then reuse it. `sweep_merge_threshold.py::check_nonempty`
  fails on it. The tell is identical file sizes across thresholds that must
  differ.
* **A constant affinity.** `run_volume.sh` reads the mid-plane percentiles and
  fails if the standard deviation is ~0. In-domain IST reference is
  p25/p50/p75 = 0.26/0.53/0.68.
* **gzip.** Everything is written with gzip off and `upload_seg_precomputed.py`
  scans for the magic number before uploading. `gcloud storage rsync` uploads
  bytes verbatim without setting `Content-Encoding`, so a gzipped chunk arrives
  as gzip bytes that neuroglancer reads as raw. Both CloudVolume's
  `compress=True` and igneous meshing's `compress='gzip'` — the **default** —
  would do this.
* **A threshold above the data maximum.** `scale_sigmoid` affinities never reach
  0.88, so `ws_high_threshold: 0.88` copied from another tutorial seeds nothing
  and fails quietly. Percentiles throughout.
* **An 18 GB source archive.** The working tree carries prior run artifacts under
  `.gcp-run/` alone worth ~18 GB. `launch.sh::source_archive` tars only the
  directories the build needs, with the same exclusions as `.dockerignore`; the
  archive is ~1 MB. (The repo-root `.gcloudignore` does the same job for any
  `gcloud builds`/`gcloud storage` command that walks the tree, since gcloud reads
  neither `.dockerignore` nor, when `.gcloudignore` exists, `.gitignore`.)
* **A silently different image.** `docker`'s classic and containerd stores report
  different `.Id` values for the same saved archive, so the run VM compares the
  loaded image's `RootFS.Layers` and its `User`/`Env`/`Entrypoint`/`Cmd`/`WorkingDir`
  against the published `image.json`, after a `sha256sum -c` on the archive. A
  mismatch aborts before the GPU is used.
* **A forgotten GPU VM.** The startup script deletes the instance on success
  *and* on failure, and the VM carries `--max-run-duration` with termination
  action `DELETE` as a backstop. Deleting rather than stopping matters: a
  stopped VM keeps billing for its disk and loses its termination timestamp.
* **A build that fails on the GPU.** `smoke_image.py` runs as a `RUN` step: it
  imports zarr/cv2/h5py, cloud-volume and igneous, and huggingface_hub; executes the
  ABISS binary; checks the eight scripts the container invokes are present; and
  asserts the checkpoint-to-output-path rule below. A missing library or a moved
  script fails in the builder, not at hour one of a GPU run.
* **BuildKit-only Dockerfile syntax.** The builder VM installs Debian's
  `docker.io`, which ships no BuildKit, so `build.sh` sets `DOCKER_BUILDKIT=0`.
  Heredoc `RUN <<EOF` is BuildKit-only and would not parse — which is why the smoke
  test is a file. `docker/Dockerfile` is already heredoc-free;
  `tutorials/neuron_snemi_gcloud/Dockerfile` is not, and would need the same
  treatment to build this way.

## Files

| file | |
|---|---|
| `Dockerfile` | ABISS `ws` + zarr/cloud-volume/igneous/huggingface_hub on top of `docker/Dockerfile`. Classic-builder compatible. No `gcloud` — the container never touches GCS. |
| `smoke_image.py` | the build-time smoke test, as a file rather than a heredoc `RUN`. |
| `build.sh` | on the builder VM: build both stages, save, publish the archive triple. |
| `run_volume.sh` | in-container: prepare → affinity → sweep → precomputed. |
| `vm_startup.sh` | on the GPU VM: load and verify the image, stage in, run, publish, self-delete. Streams its log to GCS every 60 s, because a VM that dies takes its console with it. |
| `launch.sh` | host: `--preflight` / `--build` / `--build-status` / `--run` / `--status`. |
