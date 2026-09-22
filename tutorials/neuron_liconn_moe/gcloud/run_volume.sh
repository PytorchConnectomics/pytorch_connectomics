#!/bin/bash
# One LICONN moe volume, end to end, INSIDE the container: resample -> affinity
# -> ABISS threshold sweep -> precomputed layer + meshes.
#
# Nothing here is cloud-specific except the paths. Every step is the same script
# the BC SLURM batch runs; `volumes.py` reads the LICONN_MOE_* variables set
# below, so there is no second copy of the recipe table, the prep, the sweep or
# the uploader to drift out of sync. See gcloud/README.md.
#
# The container never touches GCS. `launch.sh` stages the source group in and
# copies the artifacts out; credentials stay on the host.
#
#   docker run ... <image> bash tutorials/neuron_liconn_moe/gcloud/run_volume.sh <volume>

set -euo pipefail

VOL="${1:-}"
if [[ -z "$VOL" || "$VOL" == "--help" || "$VOL" == "-h" ]]; then
    sed -n '2,15p' "$0"
    exit 1
fi

WORK="${WORK:-/work}"
REPO="${REPO:-/workspace}"

# `runtime/checkpoint_dispatch.py::get_output_base_from_checkpoint` walks the
# checkpoint's parents for a `YYYYmmdd_HHMMSS` directory and uses it as the
# output base; with no such ancestor it falls back to `<ckpt>/../../<stem>`,
# which for a checkpoint in a top-level directory resolves under `/`. The value
# is arbitrary -- it just has to match that pattern and stay fixed, because it
# is where the affinity is written and where `volumes.py::TEST_OUT` looks for
# it. It is NOT a claim about when anything was trained.
CKPT_RUN="${CKPT_RUN:-20260921_000000}"
CKPT_FILE="${CKPT_FILE:-affinity_expid82_18nm_128x128x128.ckpt}"

export LICONN_MOE_REPO="$REPO"
export LICONN_MOE_SRC_ZARR="$WORK/src"
export LICONN_MOE_PREPARED="$WORK/prepared"
export LICONN_MOE_OUT_ROOT="$WORK/out"
export LICONN_MOE_CKPT="$WORK/ckpt/$CKPT_RUN/checkpoints/$CKPT_FILE"
export LICONN_MOE_GCS_BUCKET="${LICONN_MOE_GCS_BUCKET:-donglai_public}"
export LICONN_MOE_GCS_PREFIX="${LICONN_MOE_GCS_PREFIX:-liconn/moe/expid108}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
export ABISS_HOME="${ABISS_HOME:-/opt/abiss}"

T="$REPO/tutorials/neuron_liconn_moe"
cd "$REPO"
mkdir -p "$LICONN_MOE_PREPARED" "$LICONN_MOE_OUT_ROOT"

step() { echo; echo "=== $* === $(date -Is)"; }

step "environment"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),
'devices',torch.cuda.device_count())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no nvidia-smi"
python "$T/volumes.py" | sed -n "1p;/$VOL/p"

# --- 0. resample onto the checkpoint's [24,18,18] nm training grid -----------
# The checkpoint is a conv net: voxel size is not a free parameter for it, and
# moe spacings are biological nm (the expansion factor is already divided out),
# so matching nm matches neurite caliber in voxels. ExPID108 is 32x at
# [12.5, 5.078125, 5.078125] nm -> factors [1.92, 3.545, 3.545].
step "0 prepare"
if [[ -f "$LICONN_MOE_PREPARED/$VOL.h5" ]]; then
    echo "prepared volume exists, skipping (delete it to redo)"
else
    python "$T/run_prepare.py" --volume "$VOL"
fi

# --- 1. affinity ------------------------------------------------------------
step "1 affinity"
# Restored by vm_startup.sh when a preempted spot run already produced it. The
# check is on the file this repo writes, not on the framework's own cache logic,
# so it stays true regardless of how `scripts/main.py` resolves cache hits.
AFF_H5=$(python -c "
import os, sys
sys.path.insert(0, os.path.join('$REPO', 'tutorials/neuron_liconn_moe'))
import volumes as V
print(V.affinity_h5('$VOL'))")
if [[ -s "$AFF_H5" ]]; then
    echo "affinity exists, skipping inference: $AFF_H5"
else
    CFG=$(python "$T/make_volume_config.py" --volume "$VOL" | tail -1)
    echo "config $CFG"
    python scripts/main.py --config "$CFG" --mode test --checkpoint "$LICONN_MOE_CKPT"
fi

# The only GT-free readout there is at this stage: if the affinity mid-plane is
# constant the run has silently produced nothing and the decode would still
# "succeed". In-domain IST val reference is p25/p50/p75 = 0.26/0.53/0.68.
python - "$VOL" <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["LICONN_MOE_REPO"], "tutorials/neuron_liconn_moe"))
import h5py, numpy as np, volumes as V
p = V.affinity_h5(sys.argv[1])
with h5py.File(p, "r") as f:
    d = f["main"]
    print(f"affinity {p} {d.shape} {d.dtype} {os.path.getsize(p)/1e9:.2f} GB")
    s = np.asarray(d[:, d.shape[1] // 2]).astype(np.float32)
q = np.percentile(s, [25, 50, 75])
print(f"mid-plane affinity p25/p50/p75 = {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}"
      f"  (IST val in-domain 0.26/0.53/0.68)")
if s.std() <= 0.01:
    raise SystemExit("affinity mid-plane is constant -- inference produced nothing")
PY

# --- 2. ABISS watershed + GT-free merge-threshold sweep ---------------------
# One watershed, several agglomerations. The merge threshold is carried as a
# PERCENTILE of this volume's own affinity, not as an absolute: the affinity
# distribution shifts with expansion factor, so a fixed value is a different
# operating point on every volume. The chain test vetoes field-spanning merge
# chains. Neither is a validation -- there is no ground truth here.
step "2 abiss sweep"
python "$T/sweep_merge_threshold.py" --volume "$VOL"

# --- 3. precomputed layer + meshes ------------------------------------------
# Built here, uploaded by the host: `gcloud` is not in this image.
step "3 precomputed"
python "$T/upload_seg_precomputed.py" --volume "$VOL" \
    --create --downsample --mesh --parallel "$(nproc)"

step "done"
find "$LICONN_MOE_OUT_ROOT" -maxdepth 3 \( -name '*.json' -o -name '*.h5' \) -print | sort
du -sh "$LICONN_MOE_OUT_ROOT"/precomputed/* 2>/dev/null || true
