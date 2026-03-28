#!/bin/bash
#SBATCH --job-name=waterz_worker
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/waterz_worker_%A_%a.out
#SBATCH --error=slurm_outputs/waterz_worker_%A_%a.err

# Usage:
#   sbatch --array=0-7 scripts/decode_large_worker.sh tutorials/waterz_decoding_large.yaml
#
# Each array task is an independent worker that claims and executes
# tasks from the shared workflow directory. Workers coordinate via
# file locks — no central scheduler needed.

CONFIG=${1:-tutorials/waterz_decoding_large.yaml}

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
cd /projects/weilab/weidf/lib/pytorch_connectomics

export CCACHE_DISABLE=1

echo "Worker ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT} on $(hostname)"
echo "Config: ${CONFIG}"
echo "Start: $(date)"

python scripts/decode_large.py \
    --config ${CONFIG} \
    --worker \
    --worker-id "$(hostname)-${SLURM_ARRAY_TASK_ID}" \
    --job-id "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --idle-timeout 120

echo "End: $(date)"
