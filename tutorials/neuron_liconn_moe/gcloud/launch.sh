#!/bin/bash
# Launch one LICONN moe volume on a Google Cloud GPU VM, from this workstation.
#
#   bash tutorials/neuron_liconn_moe/gcloud/launch.sh --preflight   # changes nothing
#   bash tutorials/neuron_liconn_moe/gcloud/launch.sh --build       # disposable builder VM
#   bash tutorials/neuron_liconn_moe/gcloud/launch.sh --run
#   bash tutorials/neuron_liconn_moe/gcloud/launch.sh --status
#   bash tutorials/neuron_liconn_moe/gcloud/launch.sh --cleanup   # delete VMs
#
# --preflight checks, and never changes, IAM / quota / source / image. It prints
# the exact command for anything missing rather than applying it.
#
# THE IMAGE IS A GCS OBJECT, NOT A REGISTRY IMAGE. This follows the pattern that
# already worked for tutorials/dispim_gcloud from this same Mac. Three reasons,
# in order of weight:
#   1. Cloud Build can build in this project but cannot PUBLISH. It runs as the
#      Compute Engine default service account, which holds no Artifact Registry
#      binding, and only the project owner can grant one. Measured, not assumed:
#      the build step succeeds and the push fails with 0 images in the repo.
#      Specifying the account that *does* hold roles/cloudbuild.builds.builder is
#      refused -- Cloud Build requires a user-managed service account.
#   2. This workstation is an arm64 Mac with no Docker, so a local build would be
#      an emulated cross-build of PyTorch plus a C++ watershed.
#   3. A GCS archive needs only bucket permissions, which the operating account
#      already has, so nothing in this path waits on an owner.
#
# WHY ONE GPU VM AND NOT THE GPU-THEN-CPU SPLIT. Releasing the GPU before the
# decode is ~28% of a large bill. This volume is 213 Mvoxel after the resample
# onto the model's [24,18,18] nm grid -- 9.9% of ABISS's uint32 watershed cap,
# ~15 GB peak RSS -- so the decode is minutes, and moving 1.3 GB of affinity
# between machines costs more than the idle GPU. Revisit for a batch, or for any
# volume decoded at native resolution.

set -euo pipefail

PROJECT="${PROJECT:-sunny-catalyst-506019-a2}"
# MUST MATCH THE BUCKETS' REGION. `gs://donglai` and `gs://donglai_public` are
# both US-EAST1 regional. Running the VMs in us-central1 made every byte an
# inter-region transfer at $0.02/GiB: 14.1 GiB per volume (4.83 image archive +
# 5.5 source in, ~3.8 results out) = $0.28, which was 27% of the $1.04 that the
# first ExPID108 run cost. Same-region GCS<->GCE transfer is free, and faster.
# `preflight` fails if this drifts from where the data actually lives.
REGION="${REGION:-us-east1}"
ZONE="${ZONE:-us-east1-c}"
# SPOT CAPACITY IS PER ZONE, the bucket region is not. On 2026-09-22 all six
# ExPID107 inserts (g2-standard-16 spot, us-east1-c) died at create with
# ZONE_RESOURCE_POOL_EXHAUSTED and no VM ever booted. `--run` therefore tries
# ZONE first and then the rest of ZONES -- all inside REGION, so every byte
# stays same-region and free. Only a capacity refusal falls through; any other
# create error stops the launch. Set ZONES="$ZONE" to pin one zone.
ZONES="${ZONES:-us-east1-c us-east1-b us-east1-d}"
BUILD_ZONE="${BUILD_ZONE:-$ZONE}"

VOLUME="${VOLUME:-ExPID108_32x_Cortex_L1_01}"
SRC_ZARR="${SRC_ZARR:-gs://donglai_public/liconn/moe/expid108/image/$VOLUME.zarr}"
PUBLISH_PREFIX="${PUBLISH_PREFIX:-gs://donglai_public/liconn/moe/expid108}"
RUN_BUCKET="${RUN_BUCKET:-gs://donglai}"          # private: image, logs, manifests
HF_REPO="${HF_REPO:-pytc/liconn}"
HF_CKPT="${HF_CKPT:-affinity_expid82_18nm_128x128x128.ckpt}"

IMAGE_TAG="${IMAGE_TAG:-pytc:liconn-moe}"
# The image lives at a STABLE prefix, not under a run, so a second volume reuses
# it instead of spending 25 minutes rebuilding PyTorch and ABISS.
IMAGE_ID="${IMAGE_ID:-liconn-moe-v1}"
BUILD_PREFIX="$RUN_BUCKET/liconn/moe/images/$IMAGE_ID/build"
# The base image (PyTorch + MedNeXt + waterz, ~25 of the build's 30 minutes)
# is cached separately from this tutorial's layer, so iterating on the layer
# does not rebuild it. Keyed by the base Dockerfile, not by IMAGE_ID.
BASE_ARCHIVE="${BASE_ARCHIVE:-$RUN_BUCKET/liconn/moe/images/pytc-gpu-base/build}"

# 1 L4 / 16 vCPU / 64 GB. Only one device is used: the model's GroupNorm has no
# running statistics, so the inference ROI is fixed at [128,128,128] to match
# training and extra devices help a batch, not a volume.
# SPOT BY DEFAULT. Every component is a flat 40% off in us-east1 (Cloud Billing
# Catalog, 2026-09-21): G2 core $0.01499 vs $0.02498821, G2 RAM $0.001756 vs
# $0.00292745, L4 $0.336 vs $0.56004024 -- so g2-standard-16+L4 is $0.6882/h
# against $1.1472/h. Persistent disk is NOT discounted.
#
# The risk is preemption, and the arithmetic says take it: a whole run is ~32
# minutes and ~$0.51 on spot against $0.76 on-demand, so spot wins even if
# EVERY run were preempted and restarted from scratch. It does better than that
# here, because `vm_startup.sh` checkpoints the prepared volume and the affinity
# to the run prefix -- free, now that the buckets and the VM share a region --
# and a relaunch with the same RUN_ID resumes from whichever stage completed.
#
# Set PROVISIONING=STANDARD for on-demand when a run must not be interrupted.
PROVISIONING="${PROVISIONING:-SPOT}"
# ON-DEMAND FALLBACK. When SPOT is refused in EVERY zone of ZONES, retry the same
# zones on-demand rather than leave the volume unlaunched. On-demand capacity
# held while spot did not on 2026-09-22 (g2-standard-48, and g2-standard-16 for
# ExPID108_32x_Hippocampus_01). The price is ~$0.46/h more for g2-standard-16
# ($1.1472 vs $0.6882), ~$0.25 on a 32-minute run -- far cheaper than moving the
# data to reach another region's spot pool. The manifest records which model
# was actually used. Set ONDEMAND_FALLBACK=no to wait for spot instead.
ONDEMAND_FALLBACK="${ONDEMAND_FALLBACK:-yes}"
# all | gpu | cpu. The split exists because the decode peaks at ~71 GB/Gvoxel
# while inference needs one GPU, and the most RAM available with ONE L4 is
# 128 GiB -- so a 2 Gvoxel volume would otherwise need g2-standard-48, four L4s
# bought to rent RAM, which was capacity-exhausted on spot across all us-east1
# zones on 2026-09-22. Run `gpu` then `cpu` with the SAME RUN_ID; the affinity
# hands off through the run prefix.
STAGES="${STAGES:-all}"
case "$STAGES" in all|gpu|cpu) ;; *) echo "STAGES must be all|gpu|cpu" >&2; exit 2 ;; esac
# A CPU stage needs no accelerator, so default it to a high-memory N2 shape.
if [[ "$STAGES" == cpu ]]; then MACHINE="${MACHINE:-n2-highmem-32}"; fi
MACHINE="${MACHINE:-g2-standard-16}"
# EMPTY ON PURPOSE. The accelerator-optimized families (g2, a2, a3) have their
# GPUs predefined by the machine type -- g2-standard-16 is exactly one L4 -- and
# passing --accelerator alongside one is rejected. Set GPU only for a
# general-purpose machine type that attaches a GPU separately.
GPU="${GPU:-}"
DISK_GB="${DISK_GB:-300}"
BUILD_MACHINE="${BUILD_MACHINE:-e2-standard-16}"
BUILD_DISK_GB="${BUILD_DISK_GB:-200}"
SA="${SA:-pytc-trainer@$PROJECT.iam.gserviceaccount.com}"
# Backstop only. Both scripts delete their own VM on success and on failure;
# this catches a hang. Termination action DELETE, not STOP.
# THIS IS THE ONLY GUARANTEED DELETION, so keep it as tight as the work safely
# allows rather than generous. The stage scripts try to delete their own VM and
# cannot: the service account holds bucket bindings only, with no
# compute.instances.delete. Three builder VMs reported a successful self-delete
# and were still RUNNING. `--cleanup` is the prompt path; this is the floor.
# Measured: warm build ~8 min, cold build ~35 min, full pipeline well under 1 h.
MAX_RUN="${MAX_RUN:-3h}"
BUILD_MAX_RUN="${BUILD_MAX_RUN:-90m}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
RUN_ID="${RUN_ID:-$VOLUME-$(date -u +%Y%m%d-%H%M%S)}"
RUN_PREFIX="$RUN_BUCKET/liconn/moe/runs/$RUN_ID"
vmname() { echo "$1" | tr '[:upper:]_.' '[:lower:]--' | cut -c1-62; }

g() { gcloud --project="$PROJECT" "$@"; }
ok()  { printf '  \033[32mok\033[0m    %s\n' "$*"; }
bad() { printf '  \033[31mMISSING\033[0m %s\n' "$*"; FAILED=1; }
warn(){ printf '  \033[33mwarn\033[0m  %s\n' "$*"; }
fix() { printf '        fix: %s\n' "$*"; }

# One tarball of the checkout for the Docker build context. The tree is ~19 GB
# (`.gcp-run/` alone is ~18 GB), so the archive names the directories the build
# needs rather than excluding the ones it does not.
#
# THE EXCLUDE LIST IS DELIBERATELY SHORT. `tar --exclude` matches at EVERY path
# level, not just the top, so `--exclude=data` also drops `connectomics/data`
# -- the core data package -- along with `connectomics/data/datasets` and
# `tests/unit/data`. That is not hypothetical: it is what the first two builds
# did, and `smoke_image.py` is what caught it, at
# `ModuleNotFoundError: No module named 'connectomics.data'`. Because the
# include list below is explicit, top-level `data/`, `datasets/`, `outputs/`,
# `.gcp-run/`, `slurm_logs/` and `.git` are already excluded by omission. Only
# patterns that must be filtered *inside* the included trees belong here, and
# each one must be safe at any depth.
source_archive() {
    local out=$1
    tar czf "$out" -C "$REPO_ROOT" \
        --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' \
        --exclude='.pytest_cache' --exclude='*.egg-info' --exclude='.DS_Store' \
        connectomics scripts tutorials tests docker pyproject.toml setup.py \
        MANIFEST.in README.md 2>/dev/null || true
    [[ -s "$out" ]] || { echo "source archive is empty"; return 1; }
    # The build fails ~20 minutes in if a package is missing, so assert the
    # modules the pipeline imports are actually in the archive.
    local missing=()
    for member in connectomics/__init__.py connectomics/data/__init__.py \
                  connectomics/runtime/checkpoint_dispatch.py \
                  connectomics/config/__init__.py scripts/main.py \
                  scripts/run_abiss_volume.py docker/Dockerfile \
                  tutorials/neuron_liconn_moe/volumes.py \
                  tutorials/neuron_liconn_moe/gcloud/smoke_image.py; do
        tar tzf "$out" "$member" >/dev/null 2>&1 || missing+=("$member")
    done
    if (( ${#missing[@]} )); then
        printf 'source archive is missing required members:\n'
        printf '  %s\n' "${missing[@]}"
        return 1
    fi
    printf 'archive contents verified (%s members)\n' "$(tar tzf "$out" | wc -l | tr -d ' ')"
}

# A three-line bootstrap, so the real stage script can be re-uploaded and
# retried without recreating the VM.
bootstrap() {
    local script_uri=$1 out=$2
    printf '#!/bin/bash\nset -Eeuo pipefail\ngcloud storage cp %q /tmp/stage.sh\nexec bash /tmp/stage.sh\n' \
        "$script_uri" > "$out"
}

preflight() {
    FAILED=0
    local src_bkt_r pub_bkt_r
    src_bkt_r=$(echo "$SRC_ZARR"       | sed -E 's|(gs://[^/]+).*|\1|')
    pub_bkt_r=$(echo "$PUBLISH_PREFIX" | sed -E 's|(gs://[^/]+).*|\1|')
    echo "project $PROJECT   zone $ZONE   volume $VOLUME"
    echo
    echo "identity"
    local who; who=$(gcloud config get-value account 2>/dev/null)
    if [[ -z "$who" ]]; then
        bad "no gcloud account"
    elif gcloud auth print-access-token >/dev/null 2>&1; then
        ok "authenticated as $who, token valid"
    else
        # `gcloud config get-value account` keeps returning the account name
        # long after its refresh token has expired, so checking only that is a
        # green light that then fails on the first real call. Mint a token.
        bad "$who is configured but its refresh token has EXPIRED"
        fix "gcloud auth login   (browser flow; it cannot be renewed any other way)"
        printf '        note: this blocks only THIS workstation. A running VM uses its\n'
        printf '        own service account and is unaffected.\n'
        # Bail here. Every check below reads GCS or the Compute API, so with a
        # dead token they all report MISSING and bury the one real cause under
        # a cascade of false failures.
        echo
        echo "preflight STOPPED at the credential -- later checks would all be"
        echo "false failures. Nothing was launched and nothing was changed."
        return 1
    fi

    # The recipe table is the first thing the container touches and the last
    # thing anyone checks. A volume missing from it fails with a bare KeyError
    # *inside* the pipeline -- after the image load, the source stage and the
    # checkpoint download, on paid GPU time. Measured: that is exactly how
    # ExPID71_Hippocampus_300nm_40XW01 burned a VM on 2026-09-22.
    echo "volume registered in volumes.py"
    if python3 -c "
import sys
sys.path.insert(0, '$HERE/..')
import volumes as V
sys.exit(0 if '$VOLUME' in V.VOLUMES else 1)" 2>/dev/null; then
        ok "$VOLUME is in the recipe table"
    else
        bad "$VOLUME is NOT in tutorials/neuron_liconn_moe/volumes.py::VOLUMES"
        fix "add an entry (usually {\"auto\": True}) before launching"
    fi

    echo "source volume"
    if g storage ls "$SRC_ZARR/.zattrs" >/dev/null 2>&1; then
        ok "$SRC_ZARR"
        local listed present
        listed=$(g storage cat "$SRC_ZARR/.zattrs" \
            | python3 -c "import json,sys;print(len(json.load(sys.stdin)['multiscales'][0]['datasets']))")
        present=$(g storage ls "$SRC_ZARR/" | grep -cE '/[0-9]+/$' || true)
        # The resample reads level 0 only, but a viewer reads whatever
        # `multiscales` advertises, and a level listed but absent is a 404.
        if [[ "$listed" == "$present" ]]; then ok "$present multiscale levels, all present"
        else warn ".zattrs advertises $listed levels, $present exist in the bucket"
             fix "harmless here (level 0 only), but a viewer will 404 on the rest"
        fi
    else
        bad "$SRC_ZARR not readable"
    fi

    if [[ "$STAGES" == cpu ]]; then
        echo "handed-off affinity (STAGES=cpu)"
        # The marker, not the file: an affinity mirrored mid-inference is a
        # truncated h5 that a CPU stage would decode silently (vm_startup.sh).
        if g storage ls "$RUN_PREFIX/wip/affinity/AFFINITY_COMPLETE" >/dev/null 2>&1; then
            ok "completed affinity under $RUN_PREFIX/wip/affinity"
        else
            bad "no COMPLETED affinity at $RUN_PREFIX/wip/affinity (no AFFINITY_COMPLETE marker)"
            fix "run the GPU stage first with the SAME RUN_ID: STAGES=gpu bash $0 --run"
        fi
    fi

    echo "image archive"
    if g storage ls "$BUILD_PREFIX/image.tar.gz" >/dev/null 2>&1; then
        local sz
        sz=$(g storage du -s "$BUILD_PREFIX/image.tar.gz" | awk '{printf "%.2f GB", $1/1e9}')
        ok "$BUILD_PREFIX/image.tar.gz ($sz)"
        g storage ls "$BUILD_PREFIX/image.sha256" >/dev/null 2>&1 \
            && ok "checksum published" || bad "image.sha256 missing -- the VM will refuse to load"
        g storage ls "$BUILD_PREFIX/image.json" >/dev/null 2>&1 \
            && ok "image identity published" || bad "image.json missing -- the VM cannot verify it"
    else
        bad "no image archive at $BUILD_PREFIX"
        fix "bash $0 --build"
    fi

    # The single highest-leverage check here: a region mismatch costs real money
    # on every byte and is invisible in any log. Checked against the buckets'
    # own metadata rather than a hardcoded expectation.
    echo "region match (VM zone vs bucket location)"
    local b loc mismatch=0
    for b in "$src_bkt_r" "$pub_bkt_r" "$RUN_BUCKET"; do
        loc=$(g storage buckets describe "$b" --format="value(location)" 2>/dev/null | tr 'A-Z' 'a-z')
        if [[ -z "$loc" ]]; then warn "$b location unreadable"; continue; fi
        if [[ "$loc" == "$REGION" ]]; then ok "$b is $loc = VM region"
        else bad "$b is $loc but the VM runs in $REGION -- every byte is inter-region"
             mismatch=1
        fi
    done
    (( mismatch )) && fix "re-run with REGION=<bucket region> ZONE=<that region>-c, or move the data"

    echo "gpu quota"
    local lim use
    read -r lim use < <(g compute regions describe "$REGION" --format=json \
        | python3 -c "
import json,sys
q={x['metric']:x for x in json.load(sys.stdin)['quotas']}
n=q.get('NVIDIA_L4_GPUS',{'limit':0,'usage':0})
print(int(n['limit']), int(n['usage']))")
    [[ "$lim" -ge 1 ]] && ok "NVIDIA_L4_GPUS $use/$lim in $REGION" \
                       || bad "no L4 quota in $REGION"
    if [[ "$PROVISIONING" == SPOT ]]; then
        # Spot draws on a SEPARATE quota; on-demand headroom says nothing about it.
        local plim
        plim=$(g compute regions describe "$REGION" --format=json \
            | python3 -c "
import json,sys
q={x['metric']:x for x in json.load(sys.stdin)['quotas']}
print(int(q.get('PREEMPTIBLE_NVIDIA_L4_GPUS',{'limit':0})['limit']))")
        [[ "$plim" -ge 1 ]] \
            && ok "PREEMPTIBLE_NVIDIA_L4_GPUS limit $plim -- spot, ~40% cheaper" \
            || { bad "no preemptible L4 quota in $REGION"
                 fix "PROVISIONING=STANDARD bash $0 --run   (on-demand, ~2x the price)"; }
    else
        warn "PROVISIONING=STANDARD -- on-demand, about 2x the spot price"
    fi

    echo "vm service account $SA"
    g iam service-accounts describe "$SA" >/dev/null 2>&1 \
        && ok "exists" \
        || { bad "does not exist"; fix "gcloud iam service-accounts create pytc-trainer --project=$PROJECT"; }
    # The VM runs unattended, so `donglai@mindspan.org` being able to read and
    # write says nothing. These are what the stage scripts actually need. Note
    # there is no Artifact Registry row: the image is a GCS object, which is
    # exactly why this path needs no grant from the project owner.
    local spec bkt role why
    for spec in "$src_bkt_r|roles/storage.objectViewer|read the source volume" \
                "$pub_bkt_r|roles/storage.objectAdmin|write results" \
                "$RUN_BUCKET|roles/storage.objectAdmin|read the image and write logs"; do
        IFS='|' read -r bkt role why <<<"$spec"
        # Check the ROLE, not just that the account appears in the policy: a
        # reader binding passes a substring match and then fails at the first
        # write, an hour into a GPU run. objectAdmin implies objectViewer.
        if g storage buckets get-iam-policy "$bkt" --format=json 2>/dev/null \
           | python3 -c "
import json,sys
want, sa = sys.argv[1], 'serviceAccount:' + sys.argv[2]
implies = {'roles/storage.objectViewer':
               {'roles/storage.objectViewer','roles/storage.objectAdmin','roles/storage.admin'},
           'roles/storage.objectAdmin':
               {'roles/storage.objectAdmin','roles/storage.admin'}}[want]
held = {b['role'] for b in json.load(sys.stdin).get('bindings',[]) if sa in b.get('members',[])}
sys.exit(0 if held & implies else 1)" "$role" "$SA"; then
            ok "$SA can $why on $bkt"
        else
            bad "$SA cannot $why on $bkt"
            fix "gcloud storage buckets add-iam-policy-binding $bkt --member=serviceAccount:$SA --role=$role"
        fi
    done

    echo
    if [[ "$FAILED" == 1 ]]; then
        echo "preflight FAILED -- nothing was launched and nothing was changed."
        return 1
    fi
    echo "preflight ok"
}

build() {
    local name tmp
    name=$(vmname "liconn-build-$IMAGE_ID-$(date -u +%Y%m%d-%H%M%S)")
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' RETURN

    echo "packing source from $REPO_ROOT"
    source_archive "$tmp/source.tar.gz"
    du -h "$tmp/source.tar.gz"
    g storage cp "$tmp/source.tar.gz" "$BUILD_PREFIX/source.tar.gz"
    g storage cp "$HERE/build.sh" "$BUILD_PREFIX/build.sh"
    bootstrap "$BUILD_PREFIX/build.sh" "$tmp/startup.sh"

    g compute instances create "$name" \
        --zone="$BUILD_ZONE" --machine-type="$BUILD_MACHINE" \
        --image-family=debian-12 --image-project=debian-cloud \
        --boot-disk-size="${BUILD_DISK_GB}GB" --boot-disk-type=pd-balanced \
        --boot-disk-auto-delete \
        --provisioning-model="$PROVISIONING" \
        --service-account="$SA" --scopes=cloud-platform \
        --max-run-duration="$BUILD_MAX_RUN" --instance-termination-action=DELETE \
        --no-restart-on-failure \
        --metadata-from-file=startup-script="$tmp/startup.sh" \
        --metadata="source-archive=$BUILD_PREFIX/source.tar.gz,run-prefix=${BUILD_PREFIX%/build},image-tag=$IMAGE_TAG,base-image-archive=$BASE_ARCHIVE,self-delete=yes" \
        --quiet

    cat <<EOF

builder $name launched in $BUILD_ZONE ($BUILD_MACHINE)
  It builds docker/Dockerfile then this tutorial's layer, saves the image, and
  publishes image.tar.gz / image.json / image.sha256. Expect ~25-35 min. The VM
  deletes itself either way; the log is uploaded even on failure.

  log     $BUILD_PREFIX/build.log
  status  $BUILD_PREFIX/exit-code.txt   (0 = success; absent = still running)

  IMAGE_ID=$IMAGE_ID bash $0 --build-status
EOF
}

build_status() {
    echo "image $BUILD_PREFIX"
    g storage cat "$BUILD_PREFIX/exit-code.txt" 2>/dev/null \
        && echo "(finished)" || echo "(no exit-code yet -- still building, or never started)"
    local vm
    vm=$(g compute instances list --filter="name~liconn-build" --format="value(name,zone)" 2>/dev/null | head -1)
    g compute instances list --filter="name~liconn-build" --format="table(name,status,zone)" 2>/dev/null
    if g storage ls "$BUILD_PREFIX/build.log" >/dev/null 2>&1; then
        g storage cat "$BUILD_PREFIX/build.log" | grep -E "^=== |^Step |^Successfully|error|Error|ERROR" | tail -30
    elif [[ -n "$vm" ]]; then
        # The uploaded log lags by up to 60 s and an older builder only writes
        # it at exit. The serial console needs no cooperation from the script.
        echo "(no uploaded log yet -- reading the serial console)"
        g compute instances get-serial-port-output ${vm%%	*} --zone="${vm##*	}" 2>/dev/null \
            | grep -oE "(^|: )(=== .*|Step [0-9]+/[0-9]+ :.*|Successfully .*)" | tail -20
    else
        echo "no log and no builder VM -- check $BUILD_PREFIX"
    fi
}

run() {
    preflight || exit 1
    local name tmp digest
    name=$(vmname "liconn-$RUN_ID")
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' RETURN

    # The image archive's checksum is this run's image identity -- the GCS
    # equivalent of a registry digest, and what the VM verifies before loading.
    digest=$(g storage cat "$BUILD_PREFIX/image.sha256" | awk '{print $1}')

    g storage cp "$HERE/vm_startup.sh" "$RUN_PREFIX/vm_startup.sh"
    bootstrap "$RUN_PREFIX/vm_startup.sh" "$tmp/startup.sh"

    # SHIP THE CURRENT TUTORIAL DIRECTORY WITH THE RUN. The image carries a
    # build-time snapshot of the repo, so a volume added to volumes.py after the
    # image was built does not exist as far as the container is concerned --
    # which is precisely how a run died on `KeyError: ExPID71_...` in 2026-09-22
    # after paying for the image load, a 3.8 GB stage and the checkpoint.
    # Rebuilding the image for every recipe-table edit is the wrong answer: it
    # is ~8 minutes and trivially forgotten. vm_startup.sh bind-mounts this over
    # /workspace/tutorials/neuron_liconn_moe instead, so the recipe table, the
    # prep, the sweep and the uploader are always the working-tree versions
    # while the framework stays exactly as built and tested.
    tar czf "$tmp/tutorial.tar.gz" -C "$HERE/../.." \
        --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
        neuron_liconn_moe
    g storage cp "$tmp/tutorial.tar.gz" "$RUN_PREFIX/tutorial.tar.gz"


    # `${arr[@]}` on an EMPTY array is an unbound-variable error under `set -u`
    # in bash 3.2, which is what macOS ships -- so the optional flag is carried
    # as a scalar and word-split, not as an array.
    local accel=""
    [[ -n "$GPU" && "$STAGES" != cpu ]] && accel="--accelerator=$GPU"
    # The CPU stage has no GPU, so the NVIDIA Deep Learning image buys nothing
    # and costs boot time; Debian 12 already carries gcloud.
    local IMAGE_FLAGS
    if [[ "$STAGES" == cpu ]]; then
        IMAGE_FLAGS=(--image-family=debian-12 --image-project=debian-cloud)
    else
        IMAGE_FLAGS=(--image-family=common-cu129-ubuntu-2204-nvidia-580
                     --image-project=deeplearning-platform-release)
    fi
    # Try ZONE first, then the rest of ZONES (same region), on spot; if every zone
    # refuses, the same zones on-demand (ONDEMAND_FALLBACK). A capacity refusal
    # falls through; anything else is a real error and stops.
    local z tried created="" err="$tmp/create.err" prov models="$PROVISIONING"
    [[ "$PROVISIONING" == SPOT && "$ONDEMAND_FALLBACK" == yes ]] && models="SPOT STANDARD"
    for prov in $models; do
    tried=""
    for z in $ZONE $ZONES; do
        case " $tried " in *" $z "*) continue ;; esac
        tried="$tried $z"
        [[ "$z" == "$REGION"-* ]] || { echo "zone $z is outside REGION $REGION" >&2; return 1; }
        echo "creating $name in $z ..."
        g compute instances create "$name" \
            --zone="$z" --machine-type="$MACHINE" \
            ${accel} --maintenance-policy=TERMINATE \
            "${IMAGE_FLAGS[@]}" \
            --boot-disk-size="${DISK_GB}GB" --boot-disk-type=pd-balanced \
            --boot-disk-auto-delete \
            --provisioning-model="$prov" \
            --service-account="$SA" --scopes=cloud-platform \
            --max-run-duration="$MAX_RUN" --instance-termination-action=DELETE \
            --no-restart-on-failure \
            --metadata-from-file=startup-script="$tmp/startup.sh" \
            --metadata="volume=$VOLUME,image-tag=$IMAGE_TAG,src-zarr=$SRC_ZARR,run-prefix=$RUN_PREFIX,build-prefix=$BUILD_PREFIX,publish-prefix=$PUBLISH_PREFIX,hf-repo=$HF_REPO,hf-ckpt=$HF_CKPT,stages=$STAGES,self-delete=yes" \
            --quiet 2>"$err" && { created=$z; break 2; }
        if grep -qE 'ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources|STOCKOUT' "$err"; then
            echo "  $z: no $prov capacity for $MACHINE -- trying the next zone"
        else
            cat "$err" >&2; return 1
        fi
    done
    [[ "$prov" == SPOT && "$models" == *STANDARD* ]] \
        && echo "no SPOT capacity in any zone of $REGION -- falling back to on-demand (~1.7x the spot price)"
    done
    if [[ -z "$created" ]]; then
        echo "no capacity ($models) for $MACHINE in any of:$tried" >&2
        echo "nothing is running; retry later" >&2
        return 1
    fi
    ZONE=$created
    PROVISIONING=$prov

    python3 - "$RUN_PREFIX" "$digest" <<PY > "$tmp/manifest.json"
import json, subprocess, sys, datetime
print(json.dumps({
    "run_id": "$RUN_ID", "volume": "$VOLUME",
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_zarr": "$SRC_ZARR", "publish_prefix": "$PUBLISH_PREFIX",
    "run_prefix": sys.argv[1],
    "image": {"tag": "$IMAGE_TAG", "archive": "$BUILD_PREFIX/image.tar.gz",
              "archive_sha256": sys.argv[2],
              "distribution": "GCS archive, not a registry -- see launch.sh"},
    "checkpoint": {"hf_repo": "$HF_REPO", "file": "$HF_CKPT",
                   "train_grid_nm_zyx": [24, 18, 18],
                   "trained_on": "LICONN ExPID82_1 final_proofread (FFN-proofread GT)",
                   "held_out_val_voi": 0.9129,
                   "inference_roi_zyx": [128, 128, 128]},
    "vm": {"name": "$name", "zone": "$ZONE", "machine": "$MACHINE",
           "provisioning": "$PROVISIONING",
           "gpu": "$GPU" or "predefined by machine type"},
    "source_commit": subprocess.run(["git","-C","$REPO_ROOT","rev-parse","HEAD"],
                                    capture_output=True, text=True).stdout.strip(),
    "source_dirty": bool(subprocess.run(["git","-C","$REPO_ROOT","status","--porcelain"],
                                        capture_output=True, text=True).stdout.strip()),
    "ground_truth": None,
    "caveat": ("Zero-shot across samples and expansion folds: the checkpoint was "
               "trained on ExPID82_1 and has never seen this specimen. There is no "
               "ground truth for this volume, so no segmentation-quality number is "
               "reported and the merge threshold is chosen GT-free. Do not read this "
               "segmentation as validated."),
}, indent=2))
PY
    g storage cp "$tmp/manifest.json" "$RUN_PREFIX/manifest.json" -q
    echo "manifest -> $RUN_PREFIX/manifest.json"

    cat <<EOF

launched $name in $ZONE  [$PROVISIONING, stage=$STAGES, $MACHINE]
  log      $RUN_PREFIX/run.log     (refreshed every 60s)
  status   $RUN_PREFIX/STATUS      (0 = success; absent = still running)
  marker   $RUN_PREFIX/COMPLETE    (written only after work AND upload succeed)
  results  $PUBLISH_PREFIX/{affinity,seg}/ and the precomputed layer

  RUN_ID=$RUN_ID bash $0 --status
$( [[ "$PROVISIONING" == SPOT ]] && cat <<SPOTNOTE

  SPOT instance: it can be preempted at any time, and is deleted if it is.
  The prepared volume and the affinity are checkpointed to the run prefix, so
  relaunching with the SAME RUN_ID resumes rather than starting over:

    RUN_ID=$RUN_ID bash $0 --run
SPOTNOTE
)
EOF
}

status() {
    echo "run $RUN_PREFIX"
    local finished=no
    if g storage cat "$RUN_PREFIX/STATUS" 2>/dev/null; then
        echo "(finished)"; finished=yes
    else
        echo "(no STATUS yet -- still running, or never started)"
    fi
    g storage ls "$RUN_PREFIX/COMPLETE" >/dev/null 2>&1 && echo "COMPLETE marker present"
    local up
    up=$(g compute instances list --filter="name~liconn-" --format="value(name)" 2>/dev/null)
    g compute instances list --filter="name~liconn-" --format="table(name,status,zone)" 2>/dev/null
    if [[ "$finished" == yes && -n "$up" ]]; then
        printf '\n\033[33mThe run has finished but a VM is still up and BILLING.\033[0m\n'
        printf 'The stage script cannot delete itself (no compute.instances.delete on\n'
        printf 'the service account). Delete it now:\n\n  bash %s --cleanup\n\n' "$0"
    fi
    g storage cat "$RUN_PREFIX/run.log" 2>/dev/null | tail -40 || echo "no log yet"
}

# Delete this project's VMs from the workstation, which is the identity that
# actually holds the permission. Matches the proven diSPIM workflow, where the
# operator deletes the VM and verifies its boot disk is gone.
cleanup() {
    local rows
    rows=$(g compute instances list --filter="name~liconn-" \
        --format="value(name,zone)" 2>/dev/null)
    if [[ -z "$rows" ]]; then echo "no liconn VMs running"; else
        echo "deleting:"; echo "$rows"
        while IFS=$'\t' read -r n z; do
            [[ -n "$n" ]] && g compute instances delete "$n" --zone="$z" --quiet
        done <<< "$rows"
    fi
    # Boot disks are created with --boot-disk-auto-delete, but verify rather
    # than assume: an orphaned disk bills silently and forever.
    local disks
    disks=$(g compute disks list --filter="name~liconn-" --format="value(name,zone)" 2>/dev/null)
    if [[ -n "$disks" ]]; then
        printf '\n\033[33mORPHANED DISKS -- these bill until removed:\033[0m\n%s\n' "$disks"
        printf 'delete with: gcloud compute disks delete <name> --zone=<zone> --project=%s\n' "$PROJECT"
    else
        echo "no orphaned liconn disks"
    fi
}

case "${1:---preflight}" in
    --preflight)    preflight ;;
    --build)        build ;;
    --build-status) build_status ;;
    --run)          run ;;
    --status)       status ;;
    --cleanup)      cleanup ;;
    *) sed -n '2,10p' "$0"; exit 1 ;;
esac
