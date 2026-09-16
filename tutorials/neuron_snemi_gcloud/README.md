# SNEMI vEM on Google Cloud: YAML → ABISS → metric

This walkthrough takes the SNEMI v1 MedNeXt affinity recipe through **ABISS
single-volume watershed and size-based merging** to a numeric adapted Rand
error. It uses the existing PyTC training, inference, decoding, and SNEMI
evaluation implementations. It does not run the distributed ABISS chunk
hierarchy. The ABISS parameters are an untuned starting point.

The final number is **adapted Rand error on the historical SNEMI3D Grand
Challenge crop**, lower is better. The complete TIFF/HDF5 volume is inferred
and decoded before the evaluator applies its centered crop. This differs from
the full-volume metric printed by ordinary `scripts/main.py --mode test`.
Never compare scores with different evaluation support as if they were equal.

The input is [neuron_snemi_gcloud.yaml](neuron_snemi_gcloud.yaml). It inherits
the canonical SNEMI v1 recipe and shared profiles, replaces the decoder list
with ABISS, removes Waterz tuning, limits fresh training to three epochs, and
uses x1 inference for screening. Training retains batch size 12 per GPU,
200 steps per epoch, the z `[0:80]` / `[80:100]` training/validation split,
and the 100-epoch warmup/cosine horizon. Four L4s match the prior cloud smoke
run; adjust the YAML batch size for a smaller GPU before building.

## 1. Prepare the host and stage data

Use a Linux x86-64 GPU host with Docker, NVIDIA Container Toolkit, host drivers,
and the Google Cloud CLI. Work from this repository's root. Cloud authentication
stays on the host, using the VM's attached service account. GCS objects must be
copied to disk; `/workspace/datasets` is a Docker bind mount, not a GCS mount.

The project and bucket below come from the last verified
[cloud handoff](../../dev/gcloud_vem/STATUS.md). These commands do not create a
VM. Check current VM, disk, quota, and image state before a billable launch;
use the runtime/deletion backstop described in that handoff.

```bash
export CLOUDSDK_CORE_PROJECT=sunny-catalyst-506019-a2
export SNEMI_GCS=gs://donglai/em/snemi
export DATA_ROOT=/work/datasets
export OUTPUT_ROOT="/work/outputs/snemi-job-$(date -u +%Y%m%d_%H%M%S)"
sudo install -d -o "$(id -u)" -g "$(id -g)" \
  "$DATA_ROOT" "$OUTPUT_ROOT" /work/checkpoints /work/predictions
mkdir -p "$DATA_ROOT/SNEMI"

gcloud storage cp "$SNEMI_GCS/image/train-input.tif" "$DATA_ROOT/SNEMI/train-input.tif"
gcloud storage cp "$SNEMI_GCS/seg/train-labels.tif" "$DATA_ROOT/SNEMI/train-labels.tif"
gcloud storage cp "$SNEMI_GCS/image/test-input.tif" "$DATA_ROOT/SNEMI/test-input.tif"
gcloud storage cp "$SNEMI_GCS/seg/test-labels.h5" "$DATA_ROOT/SNEMI/test-labels.h5"
nvidia-smi
```

The test labels must be a 3D ZYX integer HDF5 dataset named `main`. Raw affinity
input must be float probabilities in `[0,1]`, shape `(3,Z,Y,X)`, dataset `main`,
with nearest-neighbor channels ordered **X, Y, Z**, stored at the destination
voxel. These are the SNEMI v1 `deepem` conventions. The runner refuses shape
mismatches, logits, and NaNs; it does not silently crop or transpose inputs.

For CPU-only decoding of a saved prediction, only the raw HDF5 and test labels
are required; skip the TIFF downloads and GPU check.

## 2. Build the two image layers

Build both images from the **same current repository checkout**, including this
tutorial. The first layer is the repository's existing GPU image. The second
adds the pinned [ABISS source](https://github.com/PytorchConnectomics/ABISS/tree/452efa5f87f9d3cb241891ee44010d966a33b316)
and compiles its `ws` target with a portable x86-64 CPU target. The single-volume
runner requires no CloudVolume workers or additional Python packages.
The ABISS layer also runs the complete CPU benchmark on two synthetic objects
with perfect affinities and requires adapted Rand error zero during the build.

```bash
docker build -f docker/Dockerfile -t pytc:gpu \
  --build-arg PYTC_UID="$(id -u)" --build-arg PYTC_GID="$(id -g)" .
docker build -f tutorials/neuron_snemi_gcloud/Dockerfile -t pytc:snemi-abiss \
  --build-arg PYTC_UID="$(id -u)" --build-arg PYTC_GID="$(id -g)" .

docker run --rm --gpus all pytc:snemi-abiss \
  python -c 'import torch; print(torch.__version__, torch.cuda.device_count()); assert torch.cuda.is_available()'
docker run --rm pytc:snemi-abiss \
  python scripts/validate_tutorial_configs.py --glob 'tutorials/neuron_snemi_gcloud/*.yaml'
```

Build and validate before reserving repeated GPU jobs. For reuse, publish the
tested image to your existing Artifact Registry repository and run by digest.
Do not assume a locally edited image is available remotely.

## 3. Run from YAML to a number

Pick a unique run name. The runner requires its output directory not to exist;
this prevents stale checkpoints, affinities, or scores from a prior attempt
being treated as a new result. Mount the output **parent** directory.

```bash
export RUN_ID="snemi_abiss_$(date -u +%Y%m%d_%H%M%S)"
set -o pipefail
docker image inspect pytc:snemi-abiss > "$OUTPUT_ROOT/$RUN_ID.image.json"
git rev-parse HEAD > "$OUTPUT_ROOT/$RUN_ID.source.txt"
git status --short >> "$OUTPUT_ROOT/$RUN_ID.source.txt"

docker run --rm --gpus all --ipc=host \
  --volume "$DATA_ROOT:/workspace/datasets:ro" \
  --volume "$OUTPUT_ROOT:/workspace/outputs" \
  --env TMPDIR=/workspace/outputs \
  pytc:snemi-abiss \
  python -m connectomics.runtime.snemi_benchmark \
    --config tutorials/neuron_snemi_gcloud/neuron_snemi_gcloud.yaml \
    --output "/workspace/outputs/$RUN_ID" \
  2>&1 | tee "$OUTPUT_ROOT/$RUN_ID.console.log"

cat "$OUTPUT_ROOT/$RUN_ID/metric.txt"
```

This runs training, saves raw affinities with decoding disabled, releases the
model process, runs the standalone ABISS decoding stage on CPU, and invokes
`scripts/evaluate_snemi3d.py --crop challenge`. ABISS temporary HDF5/NumPy/mmap
files live on the output disk rather than the container writable layer. Keep
tens of GiB free for raw affinities and ABISS scratch, and use a host with
enough RAM for whole-volume decoding; this is not a streaming tutorial.

Training and inference logs are in the run directory (`train.log`, `infer.log`).
For a live view, use `tail -f` on the relevant log. The outer console log captures
ABISS output. The runner records failures in `manifest.json` and exits nonzero;
`metric.txt` is written only after successful evaluation with a finite score.

The fresh three-epoch run is a pipeline smoke check. No accuracy target or
runtime is claimed for it. The earlier approximately `0.8788` seeded-watershed
score in the handoff is **not an ABISS result**.

## 4. Continue the saved cloud checkpoint

**September 9 audit:** this particular checkpoint saved `max_iters=3`,
`warmup_iters=1`, epoch 2/global step 600, and LR `1e-6`. Contrary to the earlier
handoff, its scheduler was already complete. `--resume-to` changes the training
limit only; it does not repair or extend saved scheduler state. For the live
20-epoch continuation, use the audited derived checkpoint and matching YAML:

```bash
gcloud storage cp \
  gs://donglai/em/snemi/cloud-runs/snemi-abiss20-20260909/resume20.ckpt \
  /work/checkpoints/snemi_resume20.ckpt
gcloud storage cp \
  gs://donglai/em/snemi/cloud-runs/snemi-abiss20-20260909/resume-provenance.json \
  /work/checkpoints/resume-provenance.json
```

That copy retains the model, optimizer moments, and counters, extends the
cosine horizon to 20, and updates the current LR to approximately `4.73e-4`.
Use `neuron_snemi_gcloud_20epoch.yaml` with it. It is an explicit schedule
extension, not a claim that the original three epochs used a 20-epoch schedule.

Stage the existing three-epoch checkpoint once:

```bash
export SMOKE_GCS="$SNEMI_GCS/cloud-runs/snemi_4l4_20260903b"
gcloud storage cp \
  "$SMOKE_GCS/training/20260903_151723/checkpoints/last.ckpt" \
  /work/checkpoints/snemi_epoch2.ckpt
```

Run the same container with an additional read-only checkpoint mount and
`--checkpoint`. Without `--resume-to`, it performs inference and scoring only.
With `--resume-to 20`, it first resumes to **20 total epochs**:

```bash
export RUN_ID="snemi_abiss_resume20_$(date -u +%Y%m%d_%H%M%S)"
set -o pipefail
docker run --rm --gpus all --ipc=host \
  --volume "$DATA_ROOT:/workspace/datasets:ro" \
  --volume "$OUTPUT_ROOT:/workspace/outputs" \
  --volume /work/checkpoints:/checkpoints:ro \
  --env TMPDIR=/workspace/outputs \
  pytc:snemi-abiss \
  python -m connectomics.runtime.snemi_benchmark \
    --config tutorials/neuron_snemi_gcloud/neuron_snemi_gcloud_20epoch.yaml \
    --output "/workspace/outputs/$RUN_ID" \
    --checkpoint /checkpoints/snemi_resume20.ckpt \
    --resume-to 20 \
  2>&1 | tee "$OUTPUT_ROOT/$RUN_ID.console.log"
```

The runner copies the source checkpoint into the new run before training
because Lightning resumes into the checkpoint's directory. The source stays
untouched. The only reset flag passed is `--reset-max-epochs 20`; optimizer,
scheduler, global step, and epoch state are restored from the audited derived
checkpoint. Its cosine horizon is 20 epochs. Check the restored learning rate and GPU use in the first resumed
epoch. Final inference uses the new `last.ckpt`; it does not select a checkpoint
using test-label scores.

## 5. Decode the existing raw prediction on CPU

This is the shortest continuation of the prior cloud smoke run. Locate its
exact raw artifact and stage it; the handoff records the filename but not the
complete inference subdirectory:

```bash
gcloud storage ls "$SMOKE_GCS/inference/**/raw_x16_ch0-1-2.h5"
# Set this to the exact object printed above, then copy it.
export RAW_GCS='gs://donglai/em/snemi/cloud-runs/snemi_4l4_20260903b/inference/REPLACE_WITH_SUBDIRECTORY/raw_x16_ch0-1-2.h5'
gcloud storage cp "$RAW_GCS" /work/predictions/raw_x16_ch0-1-2.h5

export RUN_ID="snemi_abiss_saved_raw_$(date -u +%Y%m%d_%H%M%S)"
set -o pipefail
docker run --rm --ipc=host \
  --volume "$DATA_ROOT:/workspace/datasets:ro" \
  --volume "$OUTPUT_ROOT:/workspace/outputs" \
  --volume /work/predictions:/predictions:ro \
  --env TMPDIR=/workspace/outputs \
  pytc:snemi-abiss \
  python -m connectomics.runtime.snemi_benchmark \
    --config tutorials/neuron_snemi_gcloud/neuron_snemi_gcloud.yaml \
    --output "/workspace/outputs/$RUN_ID" \
    --prediction /predictions/raw_x16_ch0-1-2.h5 \
  2>&1 | tee "$OUTPUT_ROOT/$RUN_ID.console.log"
cat "$OUTPUT_ROOT/$RUN_ID/metric.txt"
```

This command needs no GPU, checkpoint, or model installation at execution time
beyond the supplied image. `--prediction` uses the affinities exactly as saved;
the YAML's x1 setting does not change an existing x16 artifact. The manifest
records its source path and hash. Keep the source inference config/checkpoint
manifest alongside it to establish its TTA and model provenance.

## 6. Inspect and upload the result

| Artifact | Meaning |
| --- | --- |
| `metric.txt` | One scalar adapted Rand error |
| `metrics.json` | Metric name, value, direction, precision/recall, crop support |
| `metrics.tsv` | Existing evaluator's auditable row with shape and dataset |
| `segmentation.h5` | Full-volume ABISS ZYX labels, dataset `main` |
| `resolved_test.yaml` | Expanded inference/decoder configuration |
| `manifest.json` | Status, commands, timestamps, source and artifact hashes |
| `training/<timestamp>/` | Checkpoints, training config, and raw test affinities when generated |

```bash
gcloud storage rsync "$OUTPUT_ROOT" \
  "$SNEMI_GCS/cloud-runs/$RUN_ID" --recursive
```

Use a dedicated `OUTPUT_ROOT` for each job so this uploads only that job's run
and companion image/source/console files. During a long training run, repeat
this sync from a second host terminal every 15 minutes. The attached identity
must be allowed to update changing logs and `last.ckpt`; a final upload alone
does not protect against VM loss. Verify the uploaded metric, manifest, and
checkpoint before deleting the GPU VM and its disk after GPU work is done.

Choose ABISS thresholds and checkpoint candidates using the held-out training
slab z `[80:100]`. Freeze them before evaluating test labels. For a final x16
prediction, enable `default.inference.test_time_augmentation.enabled` in the
YAML, rebuild the image, and use a new run directory. The runner intentionally
reports one fixed-decoder result rather than searching test-label scores.

## Validation status

Cloud execution started on September 9, 2026 under run
`snemi-abiss20-20260909`. The Docker build passed the real ABISS synthetic
end-to-end test with adapted Rand error 0. CUDA passthrough showed four L4s;
training resumed at epoch index 3 and logged finite loss 2.9455762 at global
step 624. The 20-epoch model score is still pending. The CPU builder was
deleted; the training VM has a twelve-hour auto-delete backstop and an active
completion/cleanup monitor. See the cloud handoff for current execution state.

Local validation covers structured config/profile resolution, the benchmark's
artifact/failure/resume contracts, ABISS wrapper and edge-storage tests, and
the real SNEMI challenge evaluator on synthetic labels. These checks establish
workflow correctness, not the pending model's segmentation accuracy.

Verified locally in conda environment `pytc`:

```bash
conda run -n pytc pytest -q tests/unit/test_snemi_benchmark.py \
  tests/unit/test_abiss_edge_storage.py tests/unit/test_decode_abiss_wrapper.py \
  tests/unit/test_evaluate_snemi3d.py
# 24 passed, 1 skipped (compiled ABISS ws unavailable locally)
conda run -n pytc python scripts/validate_tutorial_configs.py \
  --glob 'tutorials/neuron_snemi_gcloud/*.yaml'
# 20 canonical configs passed, 3 custom workflow YAMLs skipped
```

The validator also includes its default `tutorials/*.yaml` glob. Changed-file
Black, isort, Flake8, and mypy checks could not run because those tools are not
installed in the local `pytc` environment. Python compilation checks passed.
