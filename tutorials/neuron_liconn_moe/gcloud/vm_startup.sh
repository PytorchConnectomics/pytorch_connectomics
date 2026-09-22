#!/bin/bash
# GPU VM stage: load the image archive, stage the volume in, run the container,
# publish, self-delete. Runs UNATTENDED as root.
#
# Every parameter arrives through instance metadata (see launch.sh); nothing is
# baked in. The log is streamed to the run prefix throughout rather than at
# exit, because a VM that dies takes its serial console with it.
#
# The VM's service account is the only identity here -- there is no interactive
# `gcloud auth login` on an unattended VM. launch.sh refuses to start until that
# account can actually read the source and write the destination.
#
# The image arrives as a GCS archive rather than from a registry: see build.sh
# for why (Cloud Build cannot publish in this project without the owner). This
# is the same mechanism the proven diSPIM cloud runs used.
set -uo pipefail

md() { curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }

VOLUME=$(md volume)
IMAGE_TAG=$(md image-tag)
SRC_ZARR=$(md src-zarr)
RUN_PREFIX=$(md run-prefix)
BUILD_PREFIX=$(md build-prefix)
PUBLISH_PREFIX=$(md publish-prefix)
HF_REPO=$(md hf-repo)
HF_CKPT=$(md hf-ckpt)
SELF_DELETE=$(md self-delete || echo yes)
ZONE=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
NAME=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/name)

WORK=/work
LOG=$WORK/run.log
CKPT_RUN=20260921_000000

mkdir -p "$WORK"/{src,prepared,out,build,hf_cache,"ckpt/$CKPT_RUN/checkpoints"}
exec > >(tee -a "$LOG") 2>&1

( while sleep 60; do gcloud storage cp "$LOG" "$RUN_PREFIX/run.log" -q 2>/dev/null; done ) &
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
    local rc=$1
    echo "=== finish rc=$rc $(date -Is) ==="
    kill $LOGGER_PID ${WIP_PID:-} 2>/dev/null
    # A preempted spot VM is not a failure worth debugging; it is a relaunch.
    if [[ "$(curl -sf -H 'Metadata-Flavor: Google' \
        http://metadata.google.internal/computeMetadata/v1/instance/preempted 2>/dev/null)" == TRUE ]]; then
        echo "THIS VM WAS PREEMPTED (spot). Relaunch with the same RUN_ID to resume:"
        echo "  RUN_ID=\${RUN_PREFIX##*/} bash launch.sh --run"
    fi
    gcloud storage cp "$LOG" "$RUN_PREFIX/run.log" -q 2>/dev/null
    echo "$rc" > /tmp/status
    gcloud storage cp /tmp/status "$RUN_PREFIX/STATUS" -q 2>/dev/null
    # A COMPLETE marker is written only after the work AND the upload succeeded.
    # Never treat a missing or failed marker as success.
    if [[ "$rc" -eq 0 ]]; then
        echo complete > /tmp/COMPLETE
        gcloud storage cp /tmp/COMPLETE "$RUN_PREFIX/COMPLETE" -q 2>/dev/null
    fi
    # Release the GPU. A stopped VM keeps billing for its disk and loses the
    # max-run-duration termination timestamp -- delete, do not stop.
    if [[ "$SELF_DELETE" == yes ]]; then
        if ! gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet 2>&1; then
            echo "SELF-DELETE FAILED for $NAME in $ZONE -- the service account has no"
            echo "compute.instances.delete. Run: launch.sh --cleanup, or wait for the"
            echo "max-run-duration backstop. THIS VM IS STILL BILLING."
            gcloud storage cp "$LOG" "$RUN_PREFIX/run.log" -q 2>/dev/null
        fi
    fi
    exit "$rc"
}
die() { echo "FAILED: $*"; finish 1; }

echo "=== start $(date -Is) volume=$VOLUME image=$IMAGE_TAG ==="
nvidia-smi || die "no GPU driver"

# --- docker + the image archive ---------------------------------------------
command -v docker >/dev/null || { apt-get update && apt-get install -y docker.io; }
nvidia-ctk runtime configure --runtime=docker || die "nvidia-ctk configure"
systemctl enable --now docker
systemctl restart docker

cd "$WORK/build"
gcloud storage cp "$BUILD_PREFIX/image.json" expected-image.json || die "fetch image.json"

# Idempotence: a re-run on a reused disk skips the 4-5 GB download. Docker's
# classic and containerd stores report different `.Id` for the same saved
# image, so compare the filesystem layers and the runtime config instead.
image_matches() {
    docker image inspect "$IMAGE_TAG" > loaded-image.json 2>/dev/null || return 1
    python3 - <<'PY'
import json
expected = json.load(open('expected-image.json'))[0]
actual = json.load(open('loaded-image.json'))[0]
assert actual['RootFS']['Layers'] == expected['RootFS']['Layers'], 'layer mismatch'
for key in ('User', 'Env', 'Entrypoint', 'Cmd', 'WorkingDir'):
    assert actual['Config'].get(key) == expected['Config'].get(key), key
PY
}

if ! image_matches; then
    echo "=== loading image from $BUILD_PREFIX ==="
    gcloud storage cp "$BUILD_PREFIX/image.tar.gz" image.tar.gz || die "fetch image.tar.gz"
    gcloud storage cp "$BUILD_PREFIX/image.sha256" image.sha256 || die "fetch image.sha256"
    sha256sum -c image.sha256 || die "image archive checksum mismatch"
    gzip -dc image.tar.gz | docker load || die "docker load"
fi
image_matches || die "loaded image does not match its published identity"
docker image inspect "$IMAGE_TAG" > "$WORK/out/image.json"

# --- resume from a preempted run --------------------------------------------
# Spot VMs are preempted and deleted with their disk, so the two expensive
# intermediates are mirrored to the run prefix while the pipeline runs and
# restored here. Relaunching with the same RUN_ID therefore resumes at whichever
# stage completed instead of starting over. This costs nothing: the buckets and
# the VM are in the same region, so GCS transfer is free -- which is exactly why
# it was not worth doing while the VM sat in the wrong region.
WIP="$RUN_PREFIX/wip"
TEST_DIR="$WORK/ckpt/$CKPT_RUN/test_${HF_CKPT%.ckpt}"
echo "=== resume check $WIP ==="
if gcloud storage ls "$WIP/prepared/$VOLUME.h5" >/dev/null 2>&1; then
    gcloud storage cp "$WIP/prepared/$VOLUME.h5" "$WORK/prepared/$VOLUME.h5" \
        && echo "restored prepared volume -- step 0 will be skipped"
fi
if gcloud storage ls "$WIP/affinity/$VOLUME/raw_x1_ch0-1-2.h5" >/dev/null 2>&1; then
    mkdir -p "$TEST_DIR/$VOLUME"
    gcloud storage cp "$WIP/affinity/$VOLUME/raw_x1_ch0-1-2.h5" \
        "$TEST_DIR/$VOLUME/raw_x1_ch0-1-2.h5" \
        && echo "restored affinity -- step 1 will be skipped"
fi

# Mirror both back every 2 minutes for the benefit of the NEXT attempt.
( while sleep 120; do
    [[ -f "$WORK/prepared/$VOLUME.h5" ]] && \
        gcloud storage rsync "$WORK/prepared" "$WIP/prepared" -q 2>/dev/null
    [[ -f "$TEST_DIR/$VOLUME/raw_x1_ch0-1-2.h5" ]] && \
        gcloud storage rsync -r "$TEST_DIR" "$WIP/affinity" -q 2>/dev/null
  done ) &
WIP_PID=$!

# --- stage in ---------------------------------------------------------------
echo "=== staging $SRC_ZARR ==="
# Local disk, not GCS FUSE: the resample reads level 0 densely, and random
# access through FUSE is the slower and less reliable of the two. The
# destination name must be <volume>.zarr -- volumes.py resolves the source as
# $LICONN_MOE_SRC_ZARR/<volume>.zarr.
gcloud storage rsync -r "$SRC_ZARR" "$WORK/src/$VOLUME.zarr" -q || die "stage source"
du -sh "$WORK/src/$VOLUME.zarr"

chown -R 1000:1000 "$WORK/src" "$WORK/prepared" "$WORK/out" "$WORK/ckpt" "$WORK/hf_cache"

echo "=== checkpoint $HF_REPO/$HF_CKPT ==="
# DOWNLOADED INSIDE THE CONTAINER, which already carries a huggingface_hub that
# the image's build-time smoke test imported. The Deep Learning VM host does not
# have it, and installing one there is a second, unverified mechanism: the first
# attempt did exactly that, both `pip3 install` fallbacks failed, their output
# was swallowed by `>/dev/null`, and the run died on
# `ModuleNotFoundError: No module named 'huggingface_hub'` after paying for the
# image load and the 5.5 GB source stage. Nothing here is silenced any more.
#
# Public repo, no token. `HF_HOME` is on the bind mount so the cache is not lost
# inside a `--rm` container. Placed under a YYYYmmdd_HHMMSS directory on purpose
# -- see run_volume.sh for why that path shape decides where the affinity lands.
docker run --rm -v "$WORK":/work -e HF_HOME=/work/hf_cache "$IMAGE_TAG" python -c "
from huggingface_hub import hf_hub_download
import shutil
src = hf_hub_download('$HF_REPO', '$HF_CKPT')
dst = '/work/ckpt/$CKPT_RUN/checkpoints/$HF_CKPT'
shutil.copyfile(src, dst)
print('checkpoint ->', dst)
" || die "checkpoint download"

CKPT_FILE="$WORK/ckpt/$CKPT_RUN/checkpoints/$HF_CKPT"
[[ -f "$CKPT_FILE" ]] || die "checkpoint not on disk at $CKPT_FILE"
# ~247 MB per the model card. A truncated or LFS-pointer download would pass a
# bare existence check and then fail inside Lightning.
CKPT_BYTES=$(stat -c %s "$CKPT_FILE")
echo "checkpoint $CKPT_FILE  $((CKPT_BYTES / 1000000)) MB"
(( CKPT_BYTES > 200000000 )) || die "checkpoint is only $CKPT_BYTES bytes -- truncated?"
chown -R 1000:1000 "$WORK/ckpt"

# --- run --------------------------------------------------------------------
# --ipc=host: DataLoader workers exhaust Docker's default 64 MB /dev/shm.
echo "=== pipeline ==="
docker run --rm --gpus all --ipc=host \
    -v "$WORK":/work \
    -e LICONN_MOE_GCS_BUCKET="$(echo "$PUBLISH_PREFIX" | sed -E 's|gs://([^/]+)/.*|\1|')" \
    -e LICONN_MOE_GCS_PREFIX="${PUBLISH_PREFIX#gs://*/}" \
    "$IMAGE_TAG" \
    bash tutorials/neuron_liconn_moe/gcloud/run_volume.sh "$VOLUME" || die "pipeline"

# --- publish ----------------------------------------------------------------
# Results go beside the image group they overlay; the run's own bookkeeping
# (log, manifest, sweep table, image identity) goes to the private run prefix.
echo "=== publish ==="
TEST_OUT="$WORK/ckpt/$CKPT_RUN/test_${HF_CKPT%.ckpt}/$VOLUME"
AFF="$TEST_OUT/raw_x1_ch0-1-2.h5"
[[ -f "$AFF" ]] || die "affinity missing at $AFF"
gcloud storage cp "$AFF" "$PUBLISH_PREFIX/affinity/${VOLUME}_affinity_x1_ch0-1-2.h5" -q \
    || die "publish affinity"

SEG=$(ls "$WORK/out/$VOLUME"/*_seg_abiss_mt*.h5 2>/dev/null | head -1)
[[ -n "$SEG" ]] || die "segmentation missing under $WORK/out/$VOLUME"
gcloud storage cp "$SEG" "$PUBLISH_PREFIX/seg/$(basename "$SEG")" -q || die "publish seg"

# The precomputed layer, built by the container under out/precomputed/<layer>.
# Everything in it is written with gzip off; `gcloud storage rsync` uploads
# bytes verbatim without setting Content-Encoding, so a gzipped chunk would
# arrive as gzip bytes that neuroglancer reads as raw and fails on.
for layer in "$WORK/out/precomputed"/*; do
    [[ -d "$layer" ]] || continue
    gcloud storage rsync -r "$layer" "$PUBLISH_PREFIX/$(basename "$layer")" -q \
        || die "publish layer $(basename "$layer")"
    echo "layer -> $PUBLISH_PREFIX/$(basename "$layer")"
done

gcloud storage cp "$WORK/out/$VOLUME/mt_sweep.json" "$RUN_PREFIX/mt_sweep.json" -q 2>/dev/null
gcloud storage cp "$WORK/out/$VOLUME/mt_sweep.json" \
    "$PUBLISH_PREFIX/seg/${VOLUME}_mt_sweep.json" -q 2>/dev/null
gcloud storage cp "$WORK/out/image.json" "$RUN_PREFIX/image.json" -q 2>/dev/null

# The published artifacts supersede the resume copies; keeping them would leave
# ~1.2 GB of duplicate per run in the private bucket.
gcloud storage rm -r "$WIP" -q 2>/dev/null && echo "cleared resume checkpoints"

echo "=== published under $PUBLISH_PREFIX ==="
gcloud storage ls -r "$PUBLISH_PREFIX/**" | tail -20
finish 0
