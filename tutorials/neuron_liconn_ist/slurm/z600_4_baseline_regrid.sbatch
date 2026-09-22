#!/bin/bash
# 600 nm gate, the grid control: re-score the EXISTING 18 nm val affinity on both
# grids. Card MSIDEPLOY-MODEL-002, Stage 1 step 4.
#
# Two jobs in one. It must reproduce the published whole-val VOI 0.9130 at
# mt 0.47 on the 18 nm grid -- that is the end-to-end check that
# z600_wholeval_sweep.py scores the same way wholeval_sweep.py did -- and it
# gives the same segmentation's score on the 104-plane grid, which is the offset
# the degraded run has to be read against. Without it the 104-grid number would
# be uninterpretable, because VOI is not comparable across voxel grids.
#
# Memory: this is the 2.08 Gvox volume, i.e. exactly what wholeval_sweep.sbatch
# ran at (147 GB peak with one segmentation held, ~176-201 GB inside `ws`).
# Three thresholds hold three uint64 segmentations (16.6 GB each) plus the GT and
# its 104-plane copy. 480G matches the precedent and keeps it on an l40s/cb node.
#SBATCH --job-name=z600_regrid
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_z600_4_baseline_regrid.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_z600_4_baseline_regrid.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
AFF="$REPO/outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/val/raw_x1_ch0-1-2.h5"
echo "host=$(hostname) start=$(date -Is)"; free -g | head -2
python tutorials/neuron_liconn_ist/z600_wholeval_sweep.py \
  --affinity "$AFF" --label eb2_regrid --source-grid 18nm \
  --thresholds 0.44,0.47,0.50 \
  --workdir "$REPO/outputs/neuron_liconn_ist/abiss/ws_scratch_z600_eb2" \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/z600_eb2_regrid.json"
echo "DONE end=$(date -Is)"
