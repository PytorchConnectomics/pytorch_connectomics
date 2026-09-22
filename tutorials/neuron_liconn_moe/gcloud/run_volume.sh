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
export MOE_OUT_ROOT="$WORK/out"
export MOE_CKPT="$WORK/ckpt/$CKPT_RUN/checkpoints/$CKPT_FILE"
export LICONN_MOE_GCS_BUCKET="${LICONN_MOE_GCS_BUCKET:-donglai_public}"
export MOE_GCS_KIND="${MOE_GCS_KIND:-mip1_eb2}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
export ABISS_HOME="${ABISS_HOME:-/opt/abiss}"
export DECODE="${DECODE:-}"
export MERGE_CRITERION="${MERGE_CRITERION:-mean}"
export RUN_PREFIX="${RUN_PREFIX:-$MOE_OUT_ROOT/s3/$VOL}"
if [[ -z "${S3_ACCEPT+x}" ]]; then
    [[ "$VOL" == ExPID108* ]] && S3_ACCEPT=1 || S3_ACCEPT=0
fi
export S3_ACCEPT

# This is deliberately before the GPU half. The criterion has one source of
# truth and resolution must fail before a VM does any useful work.
if [[ -n "$DECODE" ]]; then
    [[ "$DECODE" == whole || "$DECODE" == chunked ]] || { echo "DECODE must be whole or chunked"; exit 2; }
    python - "$MERGE_CRITERION" <<'PY'
import sys
from tutorials.neuron_liconn_moe.gcloud.resolve_chunked import resolve
resolve({"merge_criterion": sys.argv[1], "CHUNK_SIZE": [256, 256, 128],
         "seg_chunk_size_xyz": [128, 128, 128], "BBOX": [0, 0, 0, 512, 512, 256],
         "AFF_CHANNELS": [0, 1, 2]})
PY
fi

T="$REPO/tutorials/neuron_liconn_moe"
cd "$REPO"
mkdir -p "$LICONN_MOE_PREPARED" "$MOE_OUT_ROOT"

# STAGES selects which half of the pipeline this container runs, so GPU work and
# the memory-hungry decode can sit on different machines.
#
# Why that is not a micro-optimisation: the decode peaks at ~71 GB per Gvoxel,
# and the most RAM obtainable with ONE L4 is 128 GiB (g2-standard-32), because
# L4s attach only to G2 shapes. A 2 Gvoxel volume therefore needs
# g2-standard-48 -- four L4s bought to rent RAM -- and on 2026-09-22 spot
# g2-standard-48 was ZONE_RESOURCE_POOL_EXHAUSTED in all three us-east1 zones
# while g2-standard-16 spot was plentiful. Splitting buys one cheap GPU box and
# one cheap high-memory CPU box instead, and is step one of the 100 um
# architecture (card MSIDEPLOY-SCALE-001) rather than a workaround.
#
#   all   prepare + affinity + decode + precomputed  (default; small volumes)
#   gpu   prepare + affinity, then stop
#   cpu   decode + precomputed, from a restored affinity
STAGES="${STAGES:-all}"
case "$STAGES" in all|gpu|cpu) ;; *) echo "STAGES must be all|gpu|cpu"; exit 2 ;; esac
runs() { case "$STAGES" in all) return 0 ;; gpu) [[ "$1" == gpu ]] ;; cpu) [[ "$1" == cpu ]] ;; esac }

step() { echo; echo "=== $* === $(date -Is)"; }
echo "STAGES=$STAGES"

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
if runs gpu; then
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
    python scripts/main.py --config "$CFG" --mode test --checkpoint "$MOE_CKPT"
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
fi   # end GPU half

# --- 2. Default sweep, or opt-in S3 ABISS decode ----------------------------
# One watershed, several agglomerations. The merge threshold is carried as a
# PERCENTILE of this volume's own affinity, not as an absolute: the affinity
# distribution shifts with expansion factor, so a fixed value is a different
# operating point on every volume. The chain test vetoes field-spanning merge
# chains. Neither is a validation -- there is no ground truth here.
if runs cpu; then
if [[ -z "$DECODE" ]]; then
step "2 abiss sweep"
python "$T/sweep_merge_threshold.py" --volume "$VOL"
else
step "2 abiss decode ($DECODE)"
AFF_H5=$(python -c "
import os, sys
sys.path.insert(0, os.path.join('$REPO', 'tutorials/neuron_liconn_moe'))
import volumes as V
print(V.affinity_h5('$VOL'))")
SOURCE_DATASET=$(python - "$AFF_H5" <<'PY'
import sys, h5py
with h5py.File(sys.argv[1], "r") as handle:
    names = [k for k, v in handle.items() if isinstance(v, h5py.Dataset)]
if len(names) != 1:
    raise SystemExit(f"source affinity must contain exactly one dataset, got {names}")
print(names[0])
PY
)
CANON="$RUN_PREFIX/aff_canon/aff_canon.h5"
mkdir -p "$RUN_PREFIX"
python "$T/make_prob_affinity.py" --source "$AFF_H5" --output "$CANON"
read -r -a BBOX <<< "$(python - "$CANON" <<'PY'
import sys
from pathlib import Path
from tutorials.neuron_liconn_moe.gcloud.resolve_chunked import artifact_bbox
print(*artifact_bbox(Path(sys.argv[1])))
PY
 )"
read -r -a CHUNK_A <<< "$(python - "${BBOX[@]:3:3}" <<'PY'
import sys
from tutorials.neuron_liconn_moe.gcloud.resolve_chunked import variant_specs
print(*variant_specs([0, 0, 0, *map(int, sys.argv[1:])])["A"][0])
PY
 )"
read -r -a CHUNK_B <<< "$(python - "${BBOX[@]:3:3}" <<'PY'
import sys
from tutorials.neuron_liconn_moe.gcloud.resolve_chunked import variant_specs
print(*variant_specs([0, 0, 0, *map(int, sys.argv[1:])])["B"][0])
PY
 )"
python - "$CANON" "$MERGE_CRITERION" "$RUN_PREFIX/affinity_diagnostic.json" "${BBOX[@]}" <<'PY'
import sys
from tutorials.neuron_liconn_moe.gcloud.resolve_chunked import preflight, resolve
cfg = resolve({"merge_criterion": sys.argv[2], "affinity_h5": sys.argv[1],
               "diagnostic_path": sys.argv[3], "BBOX": [int(v) for v in sys.argv[4:10]], "CHUNK_SIZE": [256, 256, 128],
               "seg_chunk_size_xyz": [128, 128, 128], "AFF_CHANNELS": [0, 1, 2]})
preflight(cfg)
PY
THRESHOLDS="$RUN_PREFIX/ws_thresholds.json"
python "$T/resolve_thresholds.py" --source "$CANON" --output "$THRESHOLDS"
WS_HIGH=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["ws_high"])' "$THRESHOLDS")
WS_LOW=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["ws_low"])' "$THRESHOLDS")
python - "$CANON" "$RUN_PREFIX/affinity_diagnostic.json" "$WS_HIGH" "$WS_LOW" <<'PY'
import sys
from pathlib import Path
from tutorials.neuron_liconn_moe.gcloud.resolve_chunked import write_affinity_diagnostic
write_affinity_diagnostic(Path(sys.argv[1]), Path(sys.argv[2]),
                          ws_high=float(sys.argv[3]), ws_low=float(sys.argv[4]))
PY

if [[ "$DECODE" == whole || "$S3_ACCEPT" == 1 ]]; then
python "$REPO/scripts/run_abiss_volume.py" --input "$AFF_H5" --input-dataset "$SOURCE_DATASET" \
      --output "$RUN_PREFIX/ref_max_compressed/seg.h5" --channels 2,1,0 \
      --edge-storage source --abiss-home "$ABISS_HOME" --ws-high-threshold 94% \
      --ws-low-threshold 20% --ws-size-threshold 10000000 --ws-dust-threshold 200 \
      --ws-merge-function max --ws-merge-threshold 0.47
python "$REPO/scripts/run_abiss_volume.py" --input "$CANON" --input-dataset main \
      --output "$RUN_PREFIX/ref_max_canon/seg.h5" --channels 0,1,2 \
      --edge-storage destination --abiss-home "$ABISS_HOME" --ws-high-threshold "$WS_HIGH" \
      --ws-low-threshold "$WS_LOW" --ws-size-threshold 10000000 --ws-dust-threshold 200 \
      --ws-merge-function max --ws-merge-threshold 0.35417863
python "$T/equivalence_test.py" --integrity --diagnostic "$RUN_PREFIX/affinity_diagnostic.json" \
      --reference "$RUN_PREFIX/ref_max_compressed/seg.h5" \
      --chunked "$RUN_PREFIX/ref_max_canon/seg.h5" --chunk-size 256 256 128
fi
if [[ "$DECODE" == whole || "$S3_ACCEPT" == 1 ]]; then
python "$REPO/scripts/run_abiss_volume.py" --input "$CANON" --input-dataset main \
      --output "$RUN_PREFIX/ref_mean/seg.h5" --channels 0,1,2 \
      --edge-storage destination --abiss-home "$ABISS_HOME" --ws-high-threshold "$WS_HIGH" \
      --ws-low-threshold "$WS_LOW" --ws-size-threshold 10000000 --ws-dust-threshold 200 \
      --ws-merge-function "$MERGE_CRITERION" --ws-merge-threshold 0.1394546
fi
if [[ "$DECODE" == chunked ]]; then
    CHUNKED_CONFIG="${CHUNKED_CONFIG:-$T/chunked_abiss.yaml}"
    python - "$CHUNKED_CONFIG" "$RUN_PREFIX" "$CANON" "$WS_HIGH" "$WS_LOW" "$MERGE_CRITERION" "$S3_ACCEPT" "${BBOX[@]}" <<'PY'
import sys
from pathlib import Path
from tutorials.neuron_liconn_moe.gcloud.resolve_chunked import preflight, resolve_variant, write_variant_config
template, prefix, canon, high, low, criterion, accept = sys.argv[1:8]
bbox = [int(v) for v in sys.argv[8:14]]
names = ("one", "A", "B") if accept == "1" else ("A",)
for name in names:
    out = Path(prefix) / f"config_{name}.yaml"
    write_variant_config(Path(template), out, run_prefix=Path(prefix), affinity_h5=Path(canon),
                        variant=name, ws_high=float(high), ws_low=float(low), bbox=bbox,
                        criterion=criterion)
    cfg = {"merge_criterion": criterion, "affinity_h5": canon,
           "BBOX": bbox, "AFF_CHANNELS": [0, 1, 2]}
    resolved = resolve_variant(cfg, name)
    if name != "one":
        preflight(resolved)
PY
    STAGES_FOR_CRITERION=$(python - "$MERGE_CRITERION" <<'PY'
import sys
from tutorials.neuron_liconn_moe.gcloud.resolve_chunked import stages_for_criterion
stages = stages_for_criterion(sys.argv[1])
print(*stages)
PY
)
    if [[ "$S3_ACCEPT" == 1 ]]; then
      python "$REPO/scripts/run_abiss_chunk.py" --config "$RUN_PREFIX/config_one.yaml" --stages $STAGES_FOR_CRITERION
    fi
    python "$REPO/scripts/run_abiss_chunk.py" --config "$RUN_PREFIX/config_A.yaml" --stages $STAGES_FOR_CRITERION
    if [[ "$S3_ACCEPT" == 1 ]]; then
      python "$REPO/scripts/run_abiss_chunk.py" --config "$RUN_PREFIX/config_B.yaml" --stages $STAGES_FOR_CRITERION
      python "$T/equivalence_test.py" --reference "$RUN_PREFIX/ref_mean/seg.h5" \
        --chunked "$RUN_PREFIX/chunked_one/seg" --chunk-size "${BBOX[3]}" "${BBOX[4]}" "${BBOX[5]}" --plumbing
      python "$T/equivalence_test.py" --reference "$RUN_PREFIX/ref_mean/seg.h5" \
        --chunked "$RUN_PREFIX/chunked_A/seg" --chunk-size "${CHUNK_A[@]}" \
        --chunked "$RUN_PREFIX/chunked_B/seg" --chunk-size "${CHUNK_B[@]}"
    fi
    exit 0
fi
exit 0
fi

# --- 3. precomputed layer + meshes ------------------------------------------
# Built here, uploaded by the host: `gcloud` is not in this image.
step "3 precomputed"
python "$T/upload_seg_precomputed.py" --volume "$VOL" \
    --create --downsample --mesh --parallel "$(nproc)"
fi   # end CPU half

step "done"
find "$MOE_OUT_ROOT" -maxdepth 3 \( -name '*.json' -o -name '*.h5' \) -print | sort
du -sh "$MOE_OUT_ROOT"/precomputed/* 2>/dev/null || true
