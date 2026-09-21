#!/bin/bash
# Whole-val merge-threshold sweep, eb2 and eb8, same thresholds.
#
# This is the measurement the slab sweep could not give: eb8's slab optimum
# (0.44) has never been scored on the whole volume, so "each model at its own
# optimum" has only ever been compared on biased numbers.
#
# Memory: the eb2 whole-val decode peaked at 147 GB with one segmentation held.
# Three thresholds hold three uint32 segmentations (~8.3 GB each at 2.08 Gvox)
# on top of that, so ~175 GB. 480 GB gives headroom and keeps it on an l40s node.
#SBATCH --job-name=wval_sweep
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=10:00:00
#SBATCH --output=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_logs/neuron_liconn_ist/%A_%a_wval_sweep.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
case "$SLURM_ARRAY_TASK_ID" in
  0) AFF="$REPO/outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/val/raw_x1_ch0-1-2.h5"; L=eb2 ;;
  1) AFF="$REPO/outputs/liconn_final_banis_plus_tube_4gpu/20260910_060545/test_step=00200000/val/raw_x1_ch0-1-2.h5"; L=eb8 ;;
esac
echo "host=$(hostname) label=$L start=$(date -Is)"; free -g | head -2
python tutorials/neuron_liconn_ist/wholeval_sweep.py --affinity "$AFF" --label "$L" \
  --thresholds 0.47,0.50,0.53,0.56,0.59 \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/${L}_wholeval_sweep.json"
echo "DONE end=$(date -Is)"
