#!/bin/bash
# mip0 step 2: decode the mip0 val affinity with ABISS (max, compressed space)
# and score every merge threshold ON THE 18 nm GRID against the same GT the
# mip1 baseline used.
#
# NO GPU. The decode is pure CPU, and the node class matters more than usual:
# cb001-cb020 are 2 TB / 96 core CPU nodes in `short`. cluster-bc.yaml claims
# anything over ~187 GB "schedules only on the GPU nodes (l40s ~503 GB,
# h200 ~2 TB)" and so concludes this needs the single shared h200; that misses
# the cb class entirely. gb001 (the h200) is excluded here to leave it for work
# that actually needs a GPU.
#
# WHICH BINARY, and why it is not the stock one. The stock lib/abiss/build/ws
# uses a uint32 internal watershed index, so one invocation cannot exceed
# 2,147,483,648 voxels (watershed_traits<uint32_t>::high_bit). This volume is
# 16,769,777,248 voxels with halo -- 7.8x over -- and job 3027382 duly aborted
# on that assert in 0.04 s after spending 29 minutes writing aff.raw. It is a
# size limit, NOT an allocation failure; no --mem can move it. ws64 is the same
# source built with WS_INTERNAL_SEG64 into a separate build64/ tree, leaving the
# stock binary byte-identical (md5 bf4c7343...). It was gated on reproducing the
# published 18 nm numbers exactly before being used here (job 3028181: 0.934057
# / 0.913016 / 0.968032 and all three segment counts, zero difference).
#
# MEMORY. Measured: ws64 peaks at 57.9 GB per Gvox (120.5 GB on 2.08 Gvox,
# job 3028181), against 52.8 for the uint32 build -- the extra 5.1 B/voxel is
# the wider `seg` and `seg_copy`. 16.65 Gvox -> ~964 GB, so 1500G leaves ~55%
# headroom and still fits the 1.4-1.8 TB free on the cb nodes. The scoring stage
# that follows in the same allocation needs ~130 GB, but only after ws64 exits.
#
# The python side never holds a volume-sized array: the affinity is streamed into
# aff.raw a z-slab at a time and each segmentation is read back through a memmap
# and downsampled on the fly. That is what makes 1200G enough rather than 1.2 TB+.
#
# WHY NOT run_abiss_chunk.py, which the card suggests: it agglomerates with mean
# and cannot do max (every agg* target in lib/abiss/CMakeLists.txt compiles
# src/agg/mean_aggl.cpp). The baseline is max; swapping the criterion would
# change the decoder and the resolution at once.
#
# SEEDING is pinned to ws_high 94% / ws_low 20% -- the exact settings behind
# 0.9130 (eb2) and 0.9188 (eb8). The parallel tuning card has since adopted
# ws_low 0.0, which moves eb2 to 0.9099 and eb8 to 0.9147; that is a 0.003-0.004
# shift, inside the ~0.01 noise floor, and ws_high is inert on this volume
# (bit-identical VOI from 79% to 1.0). Matching the baseline's settings exactly
# keeps the comparison unconfounded; a confirmation at ws_low 0.0 can follow.
#
# Resume: if this times out after aff.raw exists, re-run with --reuse-aff.
#SBATCH --job-name=mip0_abiss
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=1500G
#SBATCH --exclude=gb001
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_mip0_2_abiss_sweep.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_mip0_2_abiss_sweep.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

echo "host=$(hostname) start=$(date -Is)"
free -g | head -2
df -h /projects | tail -1

AFF="$REPO/outputs/liconn_final_banis_plus_mip0/20260908_164220/test_step=00200000/val/raw_x1_ch0-1-2.h5"
THRESHOLDS="${MIP0_THRESHOLDS:-0.38,0.41,0.44,0.47,0.50,0.53}"
LABEL="${MIP0_LABEL:-mip0}"

python tutorials/neuron_liconn_ist/mip0_wholeval_sweep.py \
  --affinity "$AFF" \
  --label "$LABEL" \
  --thresholds "$THRESHOLDS" \
  --ws-high 94% --ws-low 20% \
  --factor 2 --phase 0 \
  --ws-binary "$REPO/lib/abiss/build64/ws64" \
  --workdir "$REPO/outputs/neuron_liconn_ist/abiss_work/${LABEL}" \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/${LABEL}_wholeval_sweep.json" \
  ${MIP0_EXTRA_ARGS:-}

echo "--- mip1 reference, same settings (ws_high 94% / ws_low 20%, max, whole val,"
echo "--- 18 nm grid): eb2 0.9130 and eb8 0.9188, both at mt 0.47."
echo "DONE end=$(date -Is)"
