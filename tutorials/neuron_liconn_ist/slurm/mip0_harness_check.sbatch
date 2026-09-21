#!/bin/bash
# Harness check: run the NEW streaming mip0 sweep code against the OLD 18 nm val
# affinity at --factor 1, and confirm it reproduces the published eb2 numbers
# (mt 0.44/0.47/0.50 -> VOI 0.9341/0.9130/0.9680, ws_high 94% / ws_low 20%).
#
# This is the same consistency check the decode-tuning card used on its own
# harness. It isolates the rewrite: same affinity, same GT, same grid, same
# thresholds -- only the code path differs (streamed aff.raw + memmapped seg
# read instead of whole-array numpy). If this does not land on 0.9130 the mip0
# number means nothing, so it runs before the mip0 decode, not after.
#
# --percentile-sample 1 uses the full array for the percentile resolution, so
# the only variable under test is the streaming, not the subsampling.
#SBATCH --job-name=mip0_harness
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=260G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_mip0_harness_check.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_mip0_harness_check.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

echo "host=$(hostname) start=$(date -Is)"
free -g | head -2

EB2="$REPO/outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/val/raw_x1_ch0-1-2.h5"

python tutorials/neuron_liconn_ist/mip0_wholeval_sweep.py \
  --affinity "$EB2" \
  --label eb2harness \
  --thresholds 0.44,0.47,0.50 \
  --ws-high 94% --ws-low 20% \
  --factor 1 --percentile-sample 1 \
  --workdir "$REPO/outputs/neuron_liconn_ist/abiss_work/harness" \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/eb2_harness_check.json"

echo "--- expected (eb2_wholeval_low20.json): 0.44 -> 0.9341 | 0.47 -> 0.9130 | 0.50 -> 0.9680"
echo "DONE end=$(date -Is)"
