#!/bin/bash
# Builds the LICONN moe cloud image on a disposable builder VM and publishes it
# to the run prefix as a GCS archive. Runs as root, unattended.
#
# WHY A BUILDER VM AND NOT CLOUD BUILD OR A LOCAL DOCKER. This is the pattern
# that already worked for `tutorials/dispim_gcloud` from this same workstation
# (see gs://donglai/dispim/.../cloud-runs/*/build.sh):
#   * the workstation is an arm64 Mac with no Docker installed, so a local build
#     would be an emulated cross-build of PyTorch plus a C++ watershed;
#   * Cloud Build can build in this project but cannot publish. It now runs as
#     the Compute Engine default service account, which holds no Artifact
#     Registry binding, and only the project OWNER can grant one. Measured:
#     the build step succeeds and the push fails.
# Shipping the image as a GCS object needs only bucket permissions, which the
# operating account already has. It also removes Artifact Registry from the
# critical path entirely.
#
# Metadata in: source-archive, image-tag, run-prefix, base-image-archive (opt).
set -Eeuo pipefail

md() { curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }

SOURCE_ARCHIVE=$(md source-archive)
RUN_PREFIX=$(md run-prefix)
IMAGE_TAG=$(md image-tag)
BASE_ARCHIVE=$(md base-image-archive || true)
SELF_DELETE=$(md self-delete || echo yes)
ZONE=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
NAME=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/name)

mkdir -p /work/build /work/src
exec > >(tee -a /work/build/build.log) 2>&1

# Push the log every 60 s. Without this the log only appears in the exit trap,
# so a HUNG build is invisible until the max-run-duration backstop fires. The
# VM's serial console is the fallback and needs no cooperation from this script.
( while sleep 60; do
    gcloud storage cp /work/build/build.log "$RUN_PREFIX/build/build.log" -q 2>/dev/null
  done ) &
LOGGER_PID=$!

# SELF-DELETE IS BEST-EFFORT AND USUALLY FAILS. The VM's service account holds
# only bucket-level bindings -- it has no project role, so no
# `compute.instances.delete`. Measured: three builder VMs "self-deleted"
# successfully in their own logs and were still RUNNING. The proven diSPIM
# workflow does not self-delete for exactly this reason; its launcher docstring
# says so outright, and the operator deletes from the workstation.
#
# THE GUARANTEE IS `--max-run-duration` WITH TERMINATION ACTION DELETE, which
# the Compute Engine control plane enforces and which needs no permission from
# the VM at all. `launch.sh --cleanup` is the prompt path; the backstop is the
# floor. A failure here must be loud, not silent.
finish() {
    status=$?
    trap - EXIT
    kill $LOGGER_PID 2>/dev/null || true
    echo "$status" > /work/build/exit-code.txt
    # Upload the log and the exit code even on failure -- a deleted VM takes
    # its console with it, and a missing marker must never read as success.
    gcloud storage cp /work/build/build.log /work/build/exit-code.txt "$RUN_PREFIX/build/" || true
    echo "BUILD FINISHED exit=$status"
    if [[ "$SELF_DELETE" == yes ]]; then
        if ! gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet 2>&1; then
            echo "SELF-DELETE FAILED for $NAME in $ZONE -- the service account has no"
            echo "compute.instances.delete. Run: launch.sh --cleanup, or wait for the"
            echo "max-run-duration backstop. THIS VM IS STILL BILLING."
        fi
    fi
    exit "$status"
}
trap finish EXIT

echo "=== start $(date -Is) tag=$IMAGE_TAG ==="
# Wait for the dpkg lock rather than fail on it (see vm_startup.sh).
apt-get -o DPkg::Lock::Timeout=600 update
apt-get -o DPkg::Lock::Timeout=600 install -y docker.io
systemctl enable --now docker

cd /work/build
# WARM START. The base image compiles waterz and MedNeXt and installs the whole
# PyTorch stack -- roughly 25 of the build's 30 minutes -- and it changes far
# less often than this tutorial's layer does. So the base is cached as its own
# GCS archive: load it when one exists, publish it when one had to be built.
# Iterating on stage 2 then costs minutes instead of half an hour.
BASE_CACHED=no
if [[ -n "$BASE_ARCHIVE" ]] && gcloud storage ls "$BASE_ARCHIVE/image.tar.gz" >/dev/null 2>&1; then
    echo "=== loading cached base image from $BASE_ARCHIVE ==="
    gcloud storage cp "$BASE_ARCHIVE/image.tar.gz" base-image.tar.gz
    gcloud storage cp "$BASE_ARCHIVE/image.sha256" base-image.sha256
    # The published checksum names `image.tar.gz`; rewrite it for the local name.
    sed 's/[^ ]*$/base-image.tar.gz/' base-image.sha256 > base-check.sha256
    sha256sum -c base-check.sha256
    gzip -dc base-image.tar.gz | docker load
    BASE_CACHED=yes
else
    echo "=== no cached base image (BASE_ARCHIVE='${BASE_ARCHIVE:-unset}') ==="
fi

echo "=== source $SOURCE_ARCHIVE ==="
gcloud storage cp "$SOURCE_ARCHIVE" source.tar.gz
tar -xzf source.tar.gz -C /work/src
cd /work/src

# Debian's docker.io has no BuildKit; the Dockerfiles are written for the
# classic builder (no heredoc RUN, no `# syntax=` reliance) for this reason.
export DOCKER_BUILDKIT=0

if ! docker image inspect pytc:gpu >/dev/null 2>&1; then
    echo "=== stage 1/2: building base image pytc:gpu ==="
    docker build -f docker/Dockerfile -t pytc:gpu .
    if [[ -n "$BASE_ARCHIVE" ]]; then
        echo "=== publishing the base image so the next build skips this stage ==="
        cd /work/build
        docker image inspect pytc:gpu > new-base.json
        docker save pytc:gpu | gzip -1 > new-base.tar.gz
        sha256sum new-base.tar.gz | sed 's/new-base.tar.gz/image.tar.gz/' > new-base.sha256
        gcloud storage cp new-base.tar.gz "$BASE_ARCHIVE/image.tar.gz"
        gcloud storage cp new-base.json   "$BASE_ARCHIVE/image.json"
        gcloud storage cp new-base.sha256 "$BASE_ARCHIVE/image.sha256"
        rm -f new-base.tar.gz
        cd /work/src
    fi
else
    echo "=== stage 1/2: pytc:gpu already present (cached=$BASE_CACHED), skipping ==="
fi

echo "=== stage 2/2: $IMAGE_TAG ==="
# Its smoke test imports zarr/cloud-volume/igneous/huggingface_hub, executes the
# ABISS binary and asserts the checkpoint->output path rule, so a broken image
# fails HERE rather than on a running GPU VM.
docker build --build-arg PYTC_IMAGE=pytc:gpu \
    -f tutorials/neuron_liconn_moe/gcloud/Dockerfile -t "$IMAGE_TAG" .

echo "=== save and publish ==="
docker image inspect "$IMAGE_TAG" > /work/build/new-image.json
docker save "$IMAGE_TAG" | gzip -1 > /work/build/new-image.tar.gz
cd /work/build
# The checksum file must name `image.tar.gz` with no directory prefix, because
# the run VM verifies it from its own working directory.
sha256sum new-image.tar.gz | sed 's/new-image.tar.gz/image.tar.gz/' > new-image.sha256
gcloud storage cp new-image.tar.gz "$RUN_PREFIX/build/image.tar.gz"
gcloud storage cp new-image.json   "$RUN_PREFIX/build/image.json"
gcloud storage cp new-image.sha256 "$RUN_PREFIX/build/image.sha256"
ls -l new-image.tar.gz
echo 'BUILD COMPLETE'
