#!/bin/bash
# 600 nm gate, CONTROL sweep: decode val_z600rep and score it. Same Z information
# loss as the primary, on the baseline's own 145-plane grid, so its 18 nm-grid VOI
# is directly comparable to 0.9130 with no oracle floor and no re-grid.
# Card MSIDEPLOY-MODEL-002.
#
# Memory: 2.08 Gvox, i.e. exactly the baseline sweep's problem (147 GB measured,
# ~176 GB inside `ws`), and only one 16.6 GB segmentation is held at a time
# thanks to the on_batch_result sink. 480G matches the precedent.
#SBATCH --job-name=z600_sweeprep
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_z600_3b_sweep_rep.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_z600_3b_sweep_rep.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
AFF="$REPO/outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/val_z600rep/raw_x1_ch0-1-2.h5"
echo "host=$(hostname) start=$(date -Is)"; free -g | head -2
python tutorials/neuron_liconn_ist/z600_wholeval_sweep.py \
  --affinity "$AFF" --label z600rep --source-grid 18nm \
  --thresholds 0.41,0.44,0.47,0.50,0.53 \
  --workdir "$REPO/outputs/neuron_liconn_ist/abiss/ws_scratch_z600rep" \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/z600rep_wholeval_sweep.json"
echo "DONE end=$(date -Is)"
