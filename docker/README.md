# PyTC GPU container

This container packages the checked-out PyTorch Connectomics source on the
official PyTorch 2.13 image with CUDA 12.6 and cuDNN 9. The default CUDA line
retains support for older NVIDIA architectures that are not supported by CUDA
13 builds. Select a CUDA 13 base explicitly when targeting a newer GPU that
requires it.

The image contains code and dependencies only. Keep datasets, checkpoints, and
credentials outside the image and attach them at runtime.

## Host requirements

- Linux x86-64 with an NVIDIA GPU and a compatible NVIDIA driver.
- Docker 23 or newer.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  configured for Docker.

Verify GPU passthrough before building PyTC:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

## Build

Run the build from the repository root, not from `docker/`, so the checked-out
source is included in the image:

```bash
docker build \
  --file docker/Dockerfile \
  --tag pytc:gpu \
  --build-arg PYTC_UID="$(id -u)" \
  --build-arg PYTC_GID="$(id -g)" \
  .
```

The PyTorch base is pinned by version and digest for reproducibility. Override
it only with an official `pytorch/pytorch` image whose Python version satisfies
`pyproject.toml` and whose CUDA version is supported by the host driver:

```bash
docker build \
  --file docker/Dockerfile \
  --tag pytc:gpu \
  --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime \
  .
```

## Verify

```bash
docker run --rm --gpus all pytc:gpu \
  python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'

docker run --rm --gpus all --ipc=host pytc:gpu \
  python scripts/main.py --demo
```

The first command must print `True` for CUDA availability.

## Train with host-mounted data

Mount input data read-only and mount the output directory read-write. Using
`--ipc=host` prevents PyTorch DataLoader workers from exhausting Docker's small
default shared-memory allocation.

```bash
mkdir -p data outputs

docker run --rm \
  --gpus all \
  --ipc=host \
  --volume "$PWD/data:/data:ro" \
  --volume "$PWD/outputs:/workspace/outputs" \
  pytc:gpu \
  python scripts/main.py \
    --config tutorials/mito_lucchi++/mito_lucchi++.yaml \
    --mode train \
    data.train.image=/data/train.h5 \
    data.train.label=/data/label.h5 \
    save_path=/workspace/outputs/run-001 \
    system.num_gpus=1
```

Change the tutorial and data overrides for the target dataset. Use
`system.num_gpus=-1` to make all GPUs passed through by `--gpus` available to
Lightning.

For an interactive shell:

```bash
docker run --rm -it --gpus all --ipc=host \
  --volume "$PWD/data:/data:ro" \
  --volume "$PWD/outputs:/workspace/outputs" \
  pytc:gpu bash
```

## Google Cloud Storage workflow

Attach a user-managed service account to the GPU VM and grant it only the
bucket-level permissions it needs. Do not copy a service-account JSON key into
the image.

For HDF5 and other randomly accessed volumes, stage data on Persistent Disk or
Local SSD before starting the container:

```bash
mkdir -p data outputs
gcloud storage rsync gs://INPUT_BUCKET/TRAINING_DATA ./data --recursive
```

Run training with the bind mounts shown above, then upload the durable outputs:

```bash
gcloud storage rsync ./outputs gs://OUTPUT_BUCKET/TRAINING_RUNS/run-001 --recursive
```

For data that is too large to stage, mount the bucket on the host with Cloud
Storage FUSE and enable file caching, then bind-mount that directory into the
container. Keep the FUSE mount and Google Cloud authentication on the host.

## Publishing for repeated jobs

Tag and push the tested image to Artifact Registry when it will be reused by
multiple VMs or a Vertex AI Custom Job. Do not use a floating `latest` base tag;
update the pinned PyTorch image deliberately and rerun the verification steps.
