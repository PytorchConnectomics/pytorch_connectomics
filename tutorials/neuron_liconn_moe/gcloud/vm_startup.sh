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
# all | gpu | cpu -- see run_volume.sh. `gpu` stops after the affinity and
# leaves it in the run prefix; `cpu` restores that affinity and decodes. The
# handoff reuses the preemption-resume mirror, so it needed no new mechanism.
STAGES=$(md stages || echo all)
# Which model grid and which GCS kind folder this run is. Defaults reproduce the
# original mip1_eb2 runs exactly; a mip0 run sets moe-grid=mip0, gcs-kind=mip0_eb8.
MOE_GRID=$(md moe-grid || echo train)
GCS_KIND=$(md gcs-kind || echo mip1_eb2)
# mip1_eb2 keeps its original affinity/ and seg/ folders; any other kind nests
# them under its own folder so two models never write the same object.
if [[ "$GCS_KIND" == mip1_eb2 ]]; then OUT_PREFIX="$PUBLISH_PREFIX"
else OUT_PREFIX="$PUBLISH_PREFIX/$GCS_KIND"; fi
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

# HOST MEMORY, once a minute, into the same log. On 2026-09-22 all four ExPID107
# 14.5x GPU stages (g2-standard-16, 64 GB) stopped printing at sliding-window
# batch 815-824 -- the same batch count on volumes of 1080-1368 batches -- and
# then sat silent until the 3 h max-run backstop deleted them. No trace said why.
# The mem line and any kernel OOM-killer message make the next stall
# diagnosable from run.log alone.
( while sleep 60; do
    echo "[mem $(date -u +%H:%M)] $(free -m | awk '/^Mem:/ {printf "used %d MB / %d MB, avail %d MB", $3, $2, $7}')"
    dmesg 2>/dev/null | grep -iE "out of memory|oom-kill|killed process" | tail -2
  done ) &

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

echo "=== start $(date -Is) volume=$VOLUME image=$IMAGE_TAG stages=$STAGES grid=$MOE_GRID kind=$GCS_KIND ckpt=$HF_CKPT ==="
if [[ "$STAGES" != cpu ]]; then
    nvidia-smi || die "no GPU driver"
fi

# --- docker + the image archive ---------------------------------------------
# WAIT FOR THE DPKG LOCK. The NVIDIA DLVM image ships no docker, and on first
# boot unattended-upgrades can hold /var/lib/dpkg/lock-frontend for minutes. A
# plain apt-get then fails at once, docker never installs, and the run dies at
# `docker load` after paying for the image download -- measured 2026-09-22 on
# ExPID107_14.5x_05 (us-east1-b). DPkg::Lock::Timeout makes apt wait instead.
APT=(apt-get -o DPkg::Lock::Timeout=600)
command -v docker >/dev/null || { "${APT[@]}" update && "${APT[@]}" install -y docker.io; } \
    || die "install docker.io"
if [[ "$STAGES" != cpu ]]; then
    nvidia-ctk runtime configure --runtime=docker || die "nvidia-ctk configure"
fi
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

# --- current tutorial source over the baked one -----------------------------
# The image is a snapshot from build time; this is the working tree as of the
# launch. Mounting it keeps the recipe table and the tutorial scripts current
# without a rebuild, and leaves the framework (connectomics/, scripts/) exactly
# as built and smoke-tested. Read-only: the pipeline writes to /work, never here.
TUTORIAL_MOUNT=()
if gcloud storage ls "$RUN_PREFIX/tutorial.tar.gz" >/dev/null 2>&1; then
    mkdir -p "$WORK/code"
    gcloud storage cp "$RUN_PREFIX/tutorial.tar.gz" "$WORK/code/tutorial.tar.gz" \
        || die "fetch tutorial overlay"
    tar -xzf "$WORK/code/tutorial.tar.gz" -C "$WORK/code" || die "extract tutorial overlay"
    chown -R 1000:1000 "$WORK/code"
    TUTORIAL_MOUNT=(-v "$WORK/code/neuron_liconn_moe":/workspace/tutorials/neuron_liconn_moe:ro)
    echo "tutorial overlay mounted from $RUN_PREFIX/tutorial.tar.gz"
    # Fail here rather than inside the pipeline if the volume is not in the
    # table the container will actually read.
    grep -q "\"$VOLUME\"" "$WORK/code/neuron_liconn_moe/volumes.py" \
        || die "$VOLUME is not in the shipped volumes.py"
else
    echo "no tutorial overlay published; using the image's baked copy"
fi

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
# ONLY COMPLETED INTERMEDIATES ARE RESUMABLE. Each is restored only when its
# COMPLETE marker exists, and the mirror writes a marker only after the step that
# produces the file has finished. Before 2026-09-22 the mirror copied the
# affinity every 2 minutes WHILE inference was still writing it, and a resume
# trusted whatever it found. A preemption mid-inference therefore left a
# truncated affinity that the next attempt skipped step 1 on and decoded
# silently. All four ExPID107 14.5x GPU stages left 60-77%-complete copies.
if gcloud storage ls "$WIP/prepared/PREPARED_COMPLETE" >/dev/null 2>&1; then
    gcloud storage cp "$WIP/prepared/$VOLUME.h5" "$WORK/prepared/$VOLUME.h5" \
        && echo "restored prepared volume -- step 0 will be skipped"
elif gcloud storage ls "$WIP/prepared/$VOLUME.h5" >/dev/null 2>&1; then
    echo "ignoring $WIP/prepared/$VOLUME.h5: no PREPARED_COMPLETE marker"
fi
if gcloud storage ls "$WIP/affinity/AFFINITY_COMPLETE" >/dev/null 2>&1; then
    mkdir -p "$TEST_DIR/$VOLUME"
    gcloud storage cp "$WIP/affinity/$VOLUME/raw_x1_ch0-1-2.h5" \
        "$TEST_DIR/$VOLUME/raw_x1_ch0-1-2.h5" \
        && echo "restored affinity -- step 1 will be skipped"
elif gcloud storage ls "$WIP/affinity/$VOLUME/raw_x1_ch0-1-2.h5" >/dev/null 2>&1; then
    echo "ignoring $WIP/affinity/$VOLUME: no AFFINITY_COMPLETE marker (partial)"
fi

# Mirror each intermediate once its step has FINISHED, then mark it complete.
# run_volume.sh prints "=== <n> <name> ===" at the start of each step, so the
# header of step n+1 in the log means step n is done.
mark() { echo complete | gcloud storage cp - "$1" -q 2>/dev/null; }
( while sleep 120; do
    if [[ -f "$WORK/prepared/$VOLUME.h5" ]] && grep -q "^=== 1 affinity" "$LOG" \
       && ! gcloud storage ls "$WIP/prepared/PREPARED_COMPLETE" >/dev/null 2>&1; then
        gcloud storage rsync "$WORK/prepared" "$WIP/prepared" -q 2>/dev/null \
            && mark "$WIP/prepared/PREPARED_COMPLETE"
    fi
    if [[ -f "$TEST_DIR/$VOLUME/raw_x1_ch0-1-2.h5" ]] && grep -q "^=== 2 " "$LOG" \
       && ! gcloud storage ls "$WIP/affinity/AFFINITY_COMPLETE" >/dev/null 2>&1; then
        gcloud storage rsync -r "$TEST_DIR" "$WIP/affinity" -q 2>/dev/null \
            && mark "$WIP/affinity/AFFINITY_COMPLETE"
    fi
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
GPUFLAG=(--gpus all)
[[ "$STAGES" == cpu ]] && GPUFLAG=()
docker run --rm "${GPUFLAG[@]+"${GPUFLAG[@]}"}" --ipc=host \
    -v "$WORK":/work \
    -e STAGES="$STAGES" \
    "${TUTORIAL_MOUNT[@]+"${TUTORIAL_MOUNT[@]}"}" \
    -e LICONN_MOE_GCS_BUCKET="$(echo "$PUBLISH_PREFIX" | sed -E 's|gs://([^/]+)/.*|\1|')" \
    -e MOE_GCS_KIND="$GCS_KIND" \
    -e MOE_GRID="$MOE_GRID" \
    -e CKPT_FILE="$HF_CKPT" \
    "$IMAGE_TAG" \
    bash tutorials/neuron_liconn_moe/gcloud/run_volume.sh "$VOLUME" || die "pipeline"

# --- publish ----------------------------------------------------------------
# Results go beside the image group they overlay; the run's own bookkeeping
# (log, manifest, sweep table, image identity) goes to the private run prefix.
echo "=== publish ==="
TEST_OUT="$WORK/ckpt/$CKPT_RUN/test_${HF_CKPT%.ckpt}/$VOLUME"

if [[ "$STAGES" == gpu ]]; then
    # Hand the affinity to the CPU phase and stop. The mirror is the handoff;
    # it must NOT be cleared here, and no segmentation exists yet to publish.
    [[ -f "$TEST_DIR/$VOLUME/raw_x1_ch0-1-2.h5" ]] || die "affinity missing after the GPU stage"
    gcloud storage rsync -r "$TEST_DIR" "$WIP/affinity" -q || die "hand off affinity"
    echo complete | gcloud storage cp - "$WIP/affinity/AFFINITY_COMPLETE" -q \
        || die "mark affinity complete"
    gcloud storage rsync "$WORK/prepared" "$WIP/prepared" -q \
        && echo complete | gcloud storage cp - "$WIP/prepared/PREPARED_COMPLETE" -q || true
    echo "affinity handed off to $WIP/affinity"
    echo "next: RUN_ID=${RUN_PREFIX##*/} STAGES=cpu MACHINE=<high-mem> launch.sh --run"
    finish 0
fi

AFF="$TEST_OUT/raw_x1_ch0-1-2.h5"
[[ -f "$AFF" ]] || die "affinity missing at $AFF"
gcloud storage cp "$AFF" "$OUT_PREFIX/affinity/${VOLUME}_affinity_x1_ch0-1-2.h5" -q \
    || die "publish affinity"

SEG=$(ls "$WORK/out/$VOLUME"/*_seg_abiss_mt*.h5 2>/dev/null | head -1)
[[ -n "$SEG" ]] || die "segmentation missing under $WORK/out/$VOLUME"
gcloud storage cp "$SEG" "$OUT_PREFIX/seg/$(basename "$SEG")" -q || die "publish seg"

# The precomputed layer, built by the container under out/precomputed/<layer>.
# Everything in it is written with gzip off; `gcloud storage rsync` uploads
# bytes verbatim without setting Content-Encoding, so a gzipped chunk would
# arrive as gzip bytes that neuroglancer reads as raw and fails on.
for layer in "$WORK/out/precomputed"/*; do
    [[ -d "$layer" ]] || continue
    gcloud storage rsync -r "$layer" "$PUBLISH_PREFIX/$GCS_KIND/$(basename "$layer")" -q \
        || die "publish layer $(basename "$layer")"
    echo "layer -> $PUBLISH_PREFIX/$GCS_KIND/$(basename "$layer")"
done

gcloud storage cp "$WORK/out/$VOLUME/mt_sweep.json" "$RUN_PREFIX/mt_sweep.json" -q 2>/dev/null
gcloud storage cp "$WORK/out/$VOLUME/mt_sweep.json" \
    "$OUT_PREFIX/seg/${VOLUME}_mt_sweep.json" -q 2>/dev/null
gcloud storage cp "$WORK/out/image.json" "$RUN_PREFIX/image.json" -q 2>/dev/null

# The published artifacts supersede the resume copies; keeping them would leave
# ~1.2 GB of duplicate per run in the private bucket.
gcloud storage rm -r "$WIP" -q 2>/dev/null && echo "cleared resume checkpoints"

echo "=== published under $PUBLISH_PREFIX ==="
gcloud storage ls -r "$PUBLISH_PREFIX/**" | tail -20
finish 0
