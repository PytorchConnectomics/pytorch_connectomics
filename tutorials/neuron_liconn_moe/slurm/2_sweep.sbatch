#!/bin/bash
# Steps 2+3: ABISS watershed + a GT-free merge-threshold sweep, and pick the knee.
# One watershed, six agglomerations; the chosen seg is hard-linked to
# outputs/neuron_liconn_moe/<vol>/<layer>.h5.
#SBATCH --job-name=moe_sweep
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/neuron_liconn_moe/%A_%a_2_sweep.out
#SBATCH --error=slurm_logs/neuron_liconn_moe/%A_%a_2_sweep.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"   # submit from the repo root
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID start=$(date -Is)"
python tutorials/neuron_liconn_moe/sweep_merge_threshold.py --index "$SLURM_ARRAY_TASK_ID"
echo "DONE end=$(date -Is)"
