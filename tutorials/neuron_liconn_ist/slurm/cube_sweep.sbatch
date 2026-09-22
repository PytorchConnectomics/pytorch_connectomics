#!/bin/bash
# ABISS seeding sweep (ws_high x ws_low) on interior val cubes, eb2 and eb8.
#
# 6 cubes x 4 ws_high x 3 ws_low = 72 watersheds, each reused across 4 merge
# thresholds. A cube is 96x384x384 = 14 Mvox (val is only 145 deep, so Z caps the
# cube), against 2.08 Gvox for the whole volume -- which is what makes a 2D
# seeding sweep affordable at all, since ws_high/ws_low cannot reuse a watershed.
#SBATCH --job-name=cube_sweep
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_logs/neuron_liconn_ist/%A_%a_cube_sweep.out
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
echo "host=$(hostname) label=$L start=$(date -Is)"
python tutorials/neuron_liconn_ist/cube_sweep.py --affinity "$AFF" --label "$L" \
  --ws-high 79%,82%,85%,88%,94% --ws-low 4%,7%,10%,20% --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/${L}_cube_sweep.json"
echo "DONE end=$(date -Is)"
