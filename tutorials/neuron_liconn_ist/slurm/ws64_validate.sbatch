#!/bin/bash
# Gate on ws64 before any mip0 result is believed.
#
# ws64 is lib/abiss/src/ws/atomic_chunk.cpp built with WS_INTERNAL_SEG64, i.e.
# a uint64 internal watershed index instead of uint32. It exists because the
# stock `ws` asserts out above 2,147,483,648 voxels per invocation and mip0 val
# is 7.8x over that. The stock binary is untouched (md5 bf4c7343...).
#
# Widening an index type that the algorithm packs FLAG BITS into
# (watershed_traits<T>::high_bit / visited / dir_mask) is exactly the kind of
# change that can alter results subtly rather than loudly. So this re-runs the
# 18 nm val decode with ws64 at the settings behind the published numbers and
# requires the SAME values:
#   mt 0.44 -> 0.934057 | mt 0.47 -> 0.913016 | mt 0.50 -> 0.968032
# Those were already reproduced bit-for-bit with the stock `ws` through this
# same harness (job 3026421), so any difference here is attributable to the
# index width alone.
#SBATCH --job-name=ws64_val
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=280G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_ws64_validate.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_ws64_validate.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

echo "host=$(hostname) start=$(date -Is)"
md5sum lib/abiss/build/ws lib/abiss/build64/ws64

EB2="$REPO/outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/val/raw_x1_ch0-1-2.h5"

python tutorials/neuron_liconn_ist/mip0_wholeval_sweep.py \
  --affinity "$EB2" \
  --label eb2ws64 \
  --thresholds 0.44,0.47,0.50 \
  --ws-high 94% --ws-low 20% \
  --factor 1 --percentile-sample 1 \
  --ws-binary "$REPO/lib/abiss/build64/ws64" \
  --workdir "$REPO/outputs/neuron_liconn_ist/abiss_work/ws64val" \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/eb2_ws64_validate.json"

echo "--- required: 0.44 -> 0.934057 | 0.47 -> 0.913016 | 0.50 -> 0.968032"
echo "DONE end=$(date -Is)"
